from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
import asyncio
import json
import sys

from langchain_openai import OpenAIEmbeddings
from minwon_maru.engine.crag_chain import get_chain
from minwon_maru.engine.init.load_docs import load_docs
from minwon_maru.engine.init.check_consistency import check_filesystem_concurrency
from minwon_maru.engine.init.make_metadata import generate_metadata
from minwon_maru.tools.context import create_department_info_retriever

# ---------------------------
# 경로 총괄 (필요 최소치)
# ---------------------------
@dataclass(frozen=True)
class ProjPath:
    data_root_str: str
    metadata_name: str
    raw_docs_dir: str
    parsed_docs_dir: str
    workpages_name: str
    reference_root_str: str
    departemt_info_name : str

    @property
    def data_root(self) -> Path:
        return Path(self.data_root_str)

    @property
    def metadata_path(self) -> Path:
        return self.data_root / self.metadata_name

    @property
    def raw_docs_path(self) -> Path:
        return self.data_root / self.raw_docs_dir

    @property
    def parsed_docs_path(self) -> Path:
        return self.data_root / self.parsed_docs_dir

    @property
    def reference_root_path(self) -> Path:
        return self.data_root / self.reference_root_str

    @property
    def workpages_path(self) -> Path:
        return self.reference_root_path / self.workpages_name
    
    @property
    def departemt_info_path(self) -> Path:
        return self.reference_root_path / self.departemt_info_name


# ---------------------------
# Chat (채팅 1개)
# ---------------------------
class Chat:
    def __init__(self, chat_id: str, paths: ProjPath):
        self.embeddings = OpenAIEmbeddings()
        self.chat_id = chat_id
        self.paths = paths
        self.count = 1
        self.input_history = []  # 유저의 질의 기록
        self.output_history = [] # llm 답변 기록
        self.chain = get_chain(
            metadata_path=self.paths.metadata_path,
            workpages_path=self.paths.workpages_path,
            embeddings=self.embeddings,
        )

    def get_most_relevant_department(self, top_k: int = 3, window : int = 3):
        ''' 
        top_k (int) : 현재 대화와 유사도가 높은 부서 k개 정보
        window (int) : 지금으로부터 몇개까지의 대화를 쓸것인지. (3이면, 3개의 질의,응답 대화를 문맥으로 제공)
        '''
        # 리트리버 준비
        department_info_retriever, name_to_info = create_department_info_retriever(
            department_info_path=self.paths.departemt_info_path,  # ProjPath의 프로퍼티 사용
            embeddings=self.embeddings,
        )

        # 히스토리 기반 질의 텍스트 구성 (최근 대화 위주)
        texts = []
        if isinstance(self.input_history, list):
            texts.extend([t for t in self.input_history if isinstance(t, str) and t.strip()])
        if isinstance(self.output_history, list):
            texts.extend([t for t in self.output_history if isinstance(t, str) and t.strip()])

        if not texts:
            return []

        
        query_text = "\n".join(texts[-window:])

        # 검색
        docs = department_info_retriever.get_relevant_documents(query_text)
        if not docs:
            return []

        # 상위 top_k 반환 (중복 부서명 제거)
        results = []
        seen = set()
        for d in docs:
            name = d.metadata.get("name")
            if not name or name in seen:
                continue
            info = name_to_info.get(name, {})
            results.append({
                "id": info.get("id"),
                "name": info.get("name"),
                "phone": info.get("phone"),
            })
            seen.add(name)
            if len(results) >= top_k:
                break

        return results


    def get_chat_history(self):
        '''
        input : None
        return : tuple(사용자 질의 히스토리 리스트, llm답변 히스토리 리스트)
        '''
        return self.input_history, self.output_history

    def ask(self, text: str) -> str:
        resp = self.chain.invoke(
            {
                "ability": "민원 행정",
                "input": text,
            },
            config={"configurable": {"session_id": self.chat_id}},
        )

        # 히스토리 보관 -> 추후 활용할꺼임.
        self.count += 1
        self.input_history.append(text)
        self.output_history.append(resp["generation"])
        return resp


# ---------------------------
# ChatManager (단일 채팅)
# ---------------------------
class ChatManager:
    def __init__(self, paths: ProjPath):
        self.paths = paths
        self.server_id = None
        self.chat: Chat | None = None
        self._init_store()

    def _init_store(self):
        ok = check_filesystem_concurrency()
        if ok and self.paths.metadata_path.exists():
            try:
                meta = json.loads(self.paths.metadata_path.read_text(encoding="utf-8"))
                self.server_id = str(meta.get("serverID")) if meta else None
            except Exception:
                self.server_id = None

        if not ok or self.server_id is None:
            asyncio.run(
                load_docs(
                    data_root_str=str(self.paths.data_root),
                    rawDocs_name=self.paths.raw_docs_dir,
                    parsed_docs_name=self.paths.parsed_docs_dir,
                )
            )
            self.server_id = datetime.now().strftime("%Y%m%d%H%M%S")
            asyncio.run(
                generate_metadata(
                    server_id=self.server_id,
                    data_root_str=str(self.paths.data_root),
                    parsed_docs_name=self.paths.parsed_docs_dir,
                )
            )

    def start_chat(self) -> str:
        if self.chat is not None:
            return self.chat.chat_id
        chat_id = datetime.now().strftime("%Y%m%d%H%M%S")
        self.chat = Chat(chat_id, self.paths)
        return chat_id

    def end_chat(self) -> bool:
        if self.chat is None:
            return False
        self.chat = None
        return True

    def ask(self, text: str) -> str:
        if self.chat is None:
            raise RuntimeError("채팅이 시작되지 않았습니다. 먼저 '채팅시작'을 해주세요.")
        return self.chat.ask(text)