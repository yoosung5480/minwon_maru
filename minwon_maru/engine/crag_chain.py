# basic_CRAG.py  (chain.py + basic_CRAG.py 통합 / search_document CRAG 내부 완전 통합)
from pathlib import Path
import json
from typing import Annotated, List, Any, Optional
from typing_extensions import TypedDict

# ===== chain.py 에 있던 import =====
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_openai import ChatOpenAI  # (원본 주석 유지)

from minwon_maru.tools.context import create_metadata_retriever_with_map, create_workpages_retriever
from minwon_maru.tools.llms import llm_list
from minwon_maru.tools.personal_info_keeper import personal_info_keeper
import asyncio

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings  # (원본 주석 유지)

# ===== basic_CRAG.py 에 있던 import =====
from langgraph.graph import END, StateGraph, START
from minwon_maru.prompts import prompt
from minwon_maru.tools import llms
from langchain_teddynote.tools.tavily import TavilySearch
from pydantic import BaseModel, Field


#########################################################################################################
# 1) 프롬프트 정의 (chain.py 원본 유지)
#########################################################################################################
prompt_tpl = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 {ability}에 능숙한 친절한 민원 응대 상담원입니다. "
            "아래 문서 맥락(Context)을 우선적으로 참고해 정확하게 답변하세요. "
            "모르면 모른다고 답하세요.\n\n"
            "[Context]\n{context}"
        ),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)


# ===========================
# Grader Definition
# ===========================
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


#########################################################################################################
# basic_CRAG.py 원본 타입
#########################################################################################################
class GraphState(TypedDict):
    input: Annotated[str, "The question text from user"]
    ability: Annotated[str, "The domain ability (예: 민원 행정, 수학 등)"]
    generation: Annotated[str, "The generation from the LLM"]
    web_search: Annotated[str, "Whether to add search (Yes/No)"]
    relavent_workpages : Annotated[str, "relavent workpages link info"]
    context : Annotated[str, "relavent documents info"]
    documents: Annotated[Optional[List[Document]], "Documents already retrieved (may be empty or None)"]

class Workflow:
    def ainvoke(self, question: str) -> str: ...


#########################################################################################################
# CRAG (search_document 내부 통합 + 세션 store 필드 보유)
#########################################################################################################
class CRAG(Workflow):
    def __init__(
        self,
        rag_pipeline,
        *,  # 명시적 키워드 인자
        metadata_retriever,
        summarize_map,
        workpage_retriever,
        llm_list=llms.llm_list,
    ):  
        self.chat_count = 0
        self.raw_inputs = []
        self.masked_inputs = []
        self.llm_outputs = []
        self.llm_list = llm_list
        self.rag_pipeline = rag_pipeline
        self.web_search_prompt = prompt.prompt_to_refine_text

        # 평가자 정의
        self.structured_llm_grader = self.llm_list["gpt-4.1-mini"].with_structured_output(GradeDocuments)

        # 🔽 추가: 문서-질문 관련성 그레이더 프롬프트 + 체인
        grader_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a strict retrieval evaluator. "
             "Given a user question and a document chunk, decide if the document helps answer the question. "
             "Answer 'yes' only if the document contains facts, procedures, definitions, hours, numbers, or rules that are directly relevant. "
             "Otherwise answer 'no'. Output 'binary_score' as 'yes' or 'no'."),
            ("human", "Question:\n{question}\n\nDocument:\n{document}")
        ])
        self.retrieval_grader = grader_prompt | self.structured_llm_grader

        # 세션 히스토리 store를 CRAG 필드로 보유
        self.store: dict[str, ChatMessageHistory] = {}
        self.question_history = []

        # chain.py에서 가져오던 리트리버/맵을 CRAG 필드로 보관
        self.metadata_retriever = metadata_retriever
        self.summarize_map = summarize_map
        self.workpage_retriever = workpage_retriever

        self.web_search_tool = TavilySearch(max_results=3)
        self.work_chain = self.define_workflow()

    # [추가] CRAG 인스턴스의 세션 히스토리 getter
    def get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

    # ==================== 노드 함수 ====================
    def search_document(self, state: GraphState):
        """
        (통합) chain.py의 search_document 로직을 CRAG 내부로 완전 통합.
        state에 documents/context/doc_retriever를 채워서 반환.
        """
        query = state["input"]
        outputs = dict(state)

        # 1. 질의와 유사한 상위 문서 가져오기
        candidate_docs = self.metadata_retriever.get_relevant_documents(query)

        # 2. 질의와 유사한 workpage 검색
        selected_work = self.workpage_retriever.get_relevant_documents(query)

        # 3. 선택된 문서를 chunk로 분할
        split_documents = []
        for d in candidate_docs:
            origin_info = self.summarize_map.get(d.page_content, {})
            json_path = origin_info.get("path")
            if json_path:
                with open(json_path, "r", encoding="utf-8") as f:
                    origin_json = json.load(f)
                if isinstance(origin_json, list) and len(origin_json) > 0:
                    content = origin_json[0].get("page_content", "")
                else:
                    content = origin_json.get("content", "") if isinstance(origin_json, dict) else ""
            else:
                content = d.page_content

            splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                encoding_name="cl100k_base",
                chunk_size=300,   
                chunk_overlap=50,
                separators=["\n### ","\n## ","\n# ","\n\n","\n"," ",""]
            )
            base_doc = Document(page_content=content, metadata={"path": json_path})
            split_documents.extend(splitter.split_documents([base_doc]))

        # 4. 벡터스토어 생성 후, query와 0.5 이상 문서 chunk 검색
        if split_documents:
            vectorstore = FAISS.from_documents(
                split_documents,
                embedding=self.metadata_retriever.vectorstore.embedding_function
            )
            doc_retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": 0.5, "k": 8}
            )
            retrieved_chunks = doc_retriever.get_relevant_documents(query)
        else:
            doc_retriever = None
            retrieved_chunks = []

        # context 조합
        context_parts=[]
        workpath_context_parts = []
        if selected_work:
            workpath_context_parts.append("[행정 Work Pages]")
            for w in selected_work:
                url = w.metadata.get("url") or w.metadata.get("page_link", "")
                desc = w.metadata.get("desc", "")
                summarize = w.page_content
                workpath_context_parts.append(f"- {desc or summarize} | {url}")

        if retrieved_chunks:
            context_parts.append("[관련 문서]")
            for c in retrieved_chunks:
                context_parts.append(c.page_content)

        context_str = "\n".join(context_parts)
        workpath_context_str = "\n".join(workpath_context_parts)
        outputs["relavent_workpages"] = workpath_context_str

        # 결과 주입 
        outputs["documents"] = retrieved_chunks
        # print("===============context===============")
        # print(context_str)
        outputs["context"] = context_str
        outputs["doc_retriever"] = doc_retriever
        return outputs

    # def grade_documents(self, state: GraphState):
    #     """
    #     search_document가 만들어 둔 documents(List[Document])가
    #     존재하고 길이가 >0 이면 바로 generate, 아니면 web_search.
    #     """
    #     print("\n==== [CHECK DOCUMENT RELEVANCE TO QUESTION] ====\n")
    #     docs = state.get("documents")
    #     print(docs)
    #     has_docs = bool(docs) and len(docs) > 0
    #     result = {"web_search": "No" if has_docs else "Yes"}
    #     print(result)
    #     return result
    
    def grade_documents(self, state: GraphState):
        print("\n==== [CHECK DOCUMENT RELEVANCE TO QUESTION] ====\n")
        question = state.get("input", "")
        documents = state.get("documents") or []

        # 문서가 없으면 웹서치로 전환
        if not documents:
            print("==== [GRADE: NO DOCUMENTS] ====")
            return {"documents": [], "web_search": "Yes"}

        filtered_docs = []
        for d in documents:
            doc_text = (d.page_content or "")  # 과도한 토큰 방지로 트렁케이트
            try:
                grade = self.retrieval_grader.invoke({"question": question, "document": doc_text})
                keep = (grade.binary_score or "").strip().lower() == "yes"
            except Exception as e:
                print(f"==== [GRADE: ERROR] ==== {e}")
                # 에러 시 보수적으로 유지(망실 방지)
                keep = True

            if keep:
                print("==== [GRADE: DOCUMENT RELEVANT] ====")
                filtered_docs.append(d)
            else:
                print("==== [GRADE: DOCUMENT NOT RELEVANT] ====")

        # 문서가 없으면 웹서치로 전환 -> filtered_docs가 없을때도 전환.
        # context가 없이 제공되는 상황을 방지한다. 어떤일이 있어서 context는 웹서치 기반이라도 반환하게 해야힘
        if not documents or not filtered_docs:
            print("==== [GRADE: NO DOCUMENTS] ====")
            return {"documents": [], "web_search": "Yes"}
        
        # 기존 context에서 [관련 문서] 섹션만 교체하여 유지(Work Pages는 보존)
        original_context = state.get("context", "")
        before, sep, _after = original_context.partition("[관련 문서]")

        new_context_parts = [before.strip()] if before.strip() else []
        if filtered_docs:
            new_context_parts.append("[관련 문서]")
            for c in filtered_docs:
                new_context_parts.append(c.page_content)

        new_context = ""
        if state["relavent_workpages"] :
            new_context  = "[관련 링크]\n"+(state["relavent_workpages"])
        new_context = "\n".join(new_context_parts)
        print("new context \n", new_context)
        web_search_flag = "No" if filtered_docs else "Yes"
        state["documents"] = filtered_docs
        state["web_search"] = web_search_flag
        state["context"] = new_context
        return state


    def web_search(self, state: GraphState):
        print("\n==== [WEB SEARCH] ====\n")
        query_text, ability = state["input"], state["ability"]
        history = state.get("history", [])
        raw_input = state.get("raw_input", query_text)  # ← 추가

        docs = self.web_search_tool.invoke({"query": query_text})
        context = docs[0] if docs else self.llm_list["gpt-4.1-mini"].invoke(query_text)
        print("web search result : \n", context)
        messages = self.web_search_prompt.format(context=context, question=query_text)

        generation = self.rag_pipeline.invoke({
            "input": messages,
            "raw_input": raw_input,      # ← 추가
            "ability": ability,
            "history": history,
            "context": ""                # 웹서치 브릿지 텍스트를 그대로 던지는 경우
        })
        return {"generation": generation}


    def generate(self, state: GraphState):
        print("\n==== GENERATE ====\n")
        query_text, ability = state["input"], state["ability"]
        print("======query_text======")
        print(query_text)
        history = state.get("history", [])
        context = state.get("context", "")
        print("======context======")
        print(context)
        raw_input = state.get("raw_input", query_text)  # ← 추가

        generation = self.rag_pipeline.invoke({
            "input": query_text,
            "raw_input": raw_input,      # ← 추가: 히스토리 저장용 원본
            "ability": ability,
            "history": history,
            "context": context
        })
        return {"generation": generation}


    def passthrough(self, state: GraphState):
        print("==== [PASS THROUGH] ====")
        print("Final Answer:", state["generation"])
        return {"generation": state["generation"]}

    def decide_to_generate(self, state: GraphState):
        print("==== [ASSESS GRADED DOCUMENTS] ====")
        return "web_search_node" if state["web_search"] == "Yes" else "generate"

    # ==================== 그래프 정의 ====================
    def define_workflow(self):
        workflow = StateGraph(GraphState)

        workflow.add_node("search_document", self.search_document)   # ← 먼저 검색
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("web_search_node", self.web_search)
        workflow.add_node("pass", self.passthrough)

        workflow.add_edge(START, "search_document")
        workflow.add_edge("search_document", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "web_search_node": "web_search_node",
                "generate": "generate",
            },
        )
        workflow.add_edge("web_search_node", "pass")
        workflow.add_edge("generate", "pass")   # 누락됐던 엣지 추가
        workflow.add_edge("pass", END)

        return workflow.compile()


    # ==================== 진입점 ====================
    # CRAG.invoke 내부 수정
    def invoke(self, question: Any, config: Optional[dict] = None) -> str:
        """
        question 예:
        {
        "input": "...",
        "ability": "...",
        "documents": [...],
        "history": [...]
        }
        """
        # 1) 원본/마스킹 분리 (체인 실행 전)
        masked_dict, raw_dict = personal_info_keeper(question)
        raw_text = raw_dict.get("input", "")
        masked_text = masked_dict.get("input", "")

        # 2) work_chain용 초기 state 구성:
        #    - LLM에는 masked input 사용 (state["input"])
        #    - 세션 히스토리에는 raw_input 저장 (RunnableWithMessageHistory 에서 사용)
        state = {
            **question,
            "input": masked_text,                 # LLM에 들어갈 마스킹 입력
            "raw_input": raw_text,                # 히스토리에 저장할 원본 입력
            "ability": question.get("ability", "일반 지식"),
            "history": question.get("history", []),
            "documents": question.get("documents"),
        }
        
        result = self.work_chain.invoke(state, config=config)
        
        self.raw_inputs.append(raw_text)
        self.masked_inputs.append(masked_text)
        self.llm_outputs.append(result)
        self.chat_count += 1
        return result



#########################################################################################################
# 답변 체인 생성. (chain.py 의 get_chain → 통합)
# - 요구사항:
#   1) 채팅 히스토리 넘기기
#   2) 세션에는 "원본(raw) input"만 기록, LLM에는 personal_info_keeper로 마스킹된 input 사용
#########################################################################################################
def get_chain(
    metadata_path: Path,
    workpages_path: Path,
    embeddings,
) -> CRAG:
    # 1) retriever 생성
    metadata_retriever, summarize_map = create_metadata_retriever_with_map(
        metadata_path,
        embeddings,
    )
    workpage_retriever = create_workpages_retriever(
        workpages_path,
        embeddings,
    )

    # 2) RAG 체인 (응답 문자열 생성용 그대로 유지)
    RAGchain = (
    {
        # context: search_document가 만든 것을 그대로 사용 (문자열/리스트 모두 안전 처리)
        "context": RunnableLambda(lambda x: (
            "\n".join(
                [d.page_content if isinstance(d, Document) else str(d)]
                for d in x["context"]
            ) if isinstance(x.get("context"), list)
            else x.get("context", "")
        )),
        "documents": RunnableLambda(lambda x: x.get("documents", [])),
        "ability": RunnableLambda(lambda x: x.get("ability", "일반 지식")),
        # input: 반드시 문자열만 들어가게
        "input": RunnableLambda(lambda x: x.get("input", "")),
        "history": RunnableLambda(lambda x: x.get("history", [])),
    }
    | prompt_tpl
    | llm_list["gpt-4.1"]
    | StrOutputParser()
)

    # 3) CRAG 인스턴스 생성 (store 필드 보유, 리트리버 보관)
    crag_chain = CRAG(
        rag_pipeline=None,  # 아래에서 with_message_history 주입
        metadata_retriever=metadata_retriever,
        summarize_map=summarize_map,
        workpage_retriever=workpage_retriever,
    )

    # 4) personal_info_keeper → RAG
    #    - LLM에는 마스킹된 input만 사용
    #    - 세션 히스토리에는 "원본(raw) input"만 기록
    #    => personal_info_keeper 결과 (masked_dict, raw_str)를 받아
    #       masked_dict에 raw_input 키를 추가해서 반환
    # 수정
    def _mask_only_keep_others(x):
        masked, _raw = personal_info_keeper(x)
        # 나머지 키 보존 + input만 마스킹 값으로 교체
        return {**x, "input": masked.get("input", x.get("input", ""))}

    personal_info_keeper_r = RunnableLambda(_mask_only_keep_others)

    full_chain = personal_info_keeper_r | RAGchain  # LLM에는 마스킹된 input만 전달됨

    # 5) 세션 히스토리까지 붙이기
    #    - input_messages_key="raw_input" 으로 설정하여
    #      세션에는 "원본 입력"만 저장되도록 함
    with_message_history = RunnableWithMessageHistory(
        full_chain,                         # 기존과 동일
        crag_chain.get_session_history,     # CRAG 필드(store) 사용
        input_messages_key="raw_input",     # ★ 원본 input만 세션에 저장
        history_messages_key="history",
    )


    # 6) CRAG에 rag_pipeline 주입
    crag_chain.rag_pipeline = with_message_history

    return crag_chain
