import asyncio
from langchain_community.vectorstores import FAISS
from langchain_upstage import UpstageEmbeddings
from typing import Callable, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pathlib import Path
import json
from langchain.schema import Document
from langchain_community.vectorstores import FAISS

from collections import defaultdict
from langchain_teddynote import logging
from pathlib import Path

import minwon_maru.tools.llms as llms
import minwon_maru.tools.myPDFparser as myPDFparser
from minwon_maru.tools.json_tool import load_json_basic
'''
### retriever 하이퍼 파라미터
| 설정                                       | 설명                    | 장점                           | 단점                 |
| ---------------------------------------- | --------------------- | ---------------------------- | ------------------ |
| `k=1`                                    | 가장 유사한 문서 1개          | 빠름, 단순                       | 답변이 부실할 수 있음       |
| `k=5`, `mmr`, `lambda_mult=0.25`         | 다양한 문맥 확보, 유사도 적절히 반영 | **유사한 문서 중 다양성 확보**, 정확도+풍부함 | 느릴 수 있음            |
| `fetch_k=50`, `k=5` + `mmr`              | 후보군 확장 → Top 다양성 선택   | 유사한 문서가 많을 때 좋음              | `fetch_k`가 클수록 느려짐 |
| `score_threshold=0.8`                    | 유사도 높은 문서만 사용         | 노이즈 방지, 불필요 문서 제거            | 질의가 불명확하면 빈 결과     |
| `filter={...}`                           | 특정 조건 필터링             | 특정 context 제한 가능             | 일반 RAG엔 부적합        |
| `search_type="similarity"` + `k=5` (기본값) | 단순 유사도 정렬             | 빠르고 안정적                      | 다양성 부족 가능          |

'''

# 하이퍼파라미터 설정
retriever_configs = {
    "balanced": {
        "search_type": "mmr",
        "search_kwargs": {"k": 4, "fetch_k": 20, "lambda_mult": 0.3}
    },
    "strict": {
        "search_type": "similarity_score_threshold",
        "search_kwargs": {"k": 5, "score_threshold": 0.8}
    },
    "fast": {
        "search_kwargs": {"k": 2}
    }
}

context_configs = {
    "gpt": {
        "embeddings": OpenAIEmbeddings(),
        "chunk_size": 500,
        "chunk_overlap": 50
    },
    "upstage": {
        "embeddings": UpstageEmbeddings(model="solar-embedding-1-large"),
        "chunk_size": 500,
        "chunk_overlap": 50
    }
}


class Context:
    def __init__(self, document):
        self.document = document

    
    @classmethod # 인스턴스(self)가 아닌 클래스 자체(cls)를 첫 번째 인자로 받음
    async def create(cls, document_path: str):  # cls는 일반적으로 "클래스 자신"을 가리키는 변수명
        document = await myPDFparser.upstageParser2Document(file_path=document_path)
        return cls(document)

    def set_context(self, embeddings, chunk_size, chunk_overlap):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.split_documents = self.text_splitter.split_documents(self.document)
        self.vectorstore = FAISS.from_documents(documents=self.split_documents, embedding=embeddings)

    def set_retriever(self, retriever_config):
        self.retriever = self.vectorstore.as_retriever(**retriever_config)

    def get_retriever(self):
        return self.retriever

    
metadata_retriever_config = {
    "search_type": "similarity_score_threshold",
    "search_kwargs": {"k": 8, "score_threshold": 0.5}
}

workpage_retriever_config = {
    "search_type": "similarity_score_threshold",
    "search_kwargs": {"k": 3, "score_threshold": 0.6}
}


#########################################################################################################
# meta_data용 리트리버 생성
#########################################################################################################
def create_metadata_retriever_with_map(metadata_path: Path, 
                                       embeddings, 
                                       retriever_config = metadata_retriever_config):
    ''' 
    inputs : meta_data.json 경로
    embeddings : upstage 나 openAI 임베딩 인스턴스

    return : 
        retriever (리트리버 객체)
        summarize_to_doc (summarize → 원본 문서 정보 매핑 딕셔너리)
    '''
    # 1. meta_data.json 로드
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    # 2. Document 리스트와 매핑 dict 생성
    docs = []
    summarize_to_doc = {}
    for d in meta_data["Docs"]:
        summarize = d.get("summarize", "")
        if summarize.strip():
            doc = Document(
                page_content=summarize,
                metadata={
                    "path": d.get("json_path", "")
                }
            )
            docs.append(doc)
            summarize_to_doc[summarize] = {
                "path": d.get("json_path", "")
            }

    # 3. summarize들을 벡터스토어로 변환 후 retriever 생성
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(**retriever_config)

    return retriever, summarize_to_doc

#########################################################################################################
# work_page용 리트리버 생성. 
#########################################################################################################
def create_workpages_retriever(workpages_path: Path, 
                               embeddings, 
                               retriever_config = workpage_retriever_config):
    with open(workpages_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # JSON이 list일 수도 있고 dict일 수도 있으니 통일 처리
    workpage_list = data if isinstance(data, list) else data["workpages"]

    docs = [
        Document(
            page_content=w["page_summarize"],
            metadata={"url": w.get("url") or w.get("page_link"), "desc": w.get("desc", "")}
        )
        for w in workpage_list
    ]

    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    return vectorstore.as_retriever(**retriever_config)


workpage_retriever_config = {
    "search_type": "similarity_score_threshold",
    "search_kwargs": {"k": 8, "score_threshold": 0.5}
}


## minwon_maru.tools.contexts.py
# TODO_1


def _load_department_info(dept_path: Path) -> tuple[list[Dict], Dict[str, Dict], Dict[str, Dict]]:
    """
    department_info.json을 로드해 반환한다.
    - departments: 전체 부서 리스트
    - name_to_info: {부서명 -> 부서 정보 dict}
    - id_to_info:   {부서id -> 부서 정보 dict}

    JSON 형식 예시:
    [
      {"id": "a", "name": "기획예산과", "desc": "...", "phone": "051-000-0001"},
      {"id": "b", "name": "행정지원과", "desc": "...", "phone": "051-000-0002"},
      ...
    ]
    """
    departments: list[Dict] = load_json_basic(dept_path)

    name_to_info: Dict[str, Dict] = {}
    id_to_info: Dict[str, Dict] = {}

    for d in departments:
        name_to_info[d["name"]] = d
        id_to_info[d["id"]] = d

    return departments, name_to_info, id_to_info



def create_department_info_retriever(
    department_info_path: Path,
    embeddings,
    retriever_config = workpage_retriever_config,  # 또는 department_retriever_config
):
    """
    department_info.json을 로드해 부서 설명 기반 벡터 검색기를 만든다.
    return:
      - department_info_retriever: 각 부서(desc+name+tags+examples)로 구성된 리트리버
      - name_to_info: { 부서명 -> {id,name,desc,phone,tags?,examples?} } 매핑 딕셔너리
    """
    departments, name_to_info, id_to_info = _load_department_info(department_info_path)

    # === 변경된 검색용 문서 구성 ===
    docs = []
    for d in departments:
        name = d.get("name", "")
        desc = d.get("desc", "")
        tags = d.get("tags", []) or []
        examples = d.get("examples", []) or []

        # 임베딩용 텍스트: 짧은 요약 + 부서명 + 키워드 + 실제 예시 발화
        parts = [desc, name]
        if tags:
            parts.append(" ".join(tags))          # 동의어/생활어 커버
        if examples:
            parts.extend(examples)                # 문장 단위 매칭 강화

        page_content = "\n".join([p for p in parts if p])  # 빈 문자열 제거
        docs.append(
            Document(
                page_content=page_content,
                metadata=d,   # {"id","name","desc","phone",["tags"],["examples"]}
            )
        )
    # ==============================

    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(**retriever_config)
    return retriever, name_to_info