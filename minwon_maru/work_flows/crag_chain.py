# basic_CRAG.py  (chain.py 통합 버전)

from __future__ import annotations
from pathlib import Path
import json
from typing import Annotated, List, Any, Optional, Tuple, Dict

from typing_extensions import TypedDict
from langgraph.graph import END, StateGraph, START

from langchain.schema import Document, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from minwon_maru.tools.context import create_metadata_retriever_with_map, create_workpages_retriever
from minwon_maru.tools.llms import llm_list
from minwon_maru.tools.personal_info_keeper import personal_info_keeper
from minwon_maru.prompts import prompt
from langchain_teddynote.tools.tavily import TavilySearch


# ===========================
# Graph State
# ===========================
class GraphState(TypedDict):
    input: Annotated[str, "Masked user input for LLM (개인정보 마스킹된 입력)"]
    ability: Annotated[str, "The domain ability (예: 민원 행정, 수학 등)"]
    generation: Annotated[str, "The generation from the LLM"]
    web_search: Annotated[str, "Whether to add search (Yes/No)"]
    documents: Annotated[Optional[List[Document]], "Documents retrieved from local docs"]
    history: Annotated[list, "Chat history messages (Human/Ai messages)"]
    context: Annotated[Optional[str], "Composed textual context from search"]
    doc_retriever: Annotated[Optional[Any], "FAISS retriever used inside this turn"]


# ===========================
# CRAG Workflow
# ===========================
class Workflow:
    def ainvoke(self, question: str) -> str: ...


class CRAG(Workflow):
    """
    단일 파일 통합:
    - chain.py의 검색/프롬프트/히스토리 관리 로직을 이 클래스 내부로 흡수
    - 세션 히스토리에는 '원본 입력(raw)'을 저장하고,
      LLM에 전달되는 입력은 personal_info_keeper로 마스킹된 텍스트를 사용
    """

    def __init__(
        self,
        metadata_path: Path,
        workpages_path: Path,
        embeddings,
        *,
        k_doc: int = 5,
        score_threshold: float = 0.5,
        llm_key: str = "gpt-4o-mini",
    ):
        # --- 외부 리소스(리트리버) 준비 ---
        self.embeddings = embeddings
        self.metadata_retriever, self.summarize_map = create_metadata_retriever_with_map(
            metadata_path, embeddings
        )
        self.workpage_retriever = create_workpages_retriever(workpages_path, embeddings)

        self.k_doc = k_doc
        self.score_threshold = score_threshold

        # --- LLM / Prompt ---
        self.llm = llm_list[llm_key]
        self.chat_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 {ability}에 능숙한 민원 응대 상담원입니다. "
                    "아래 문서 맥락(Context)을 우선적으로 참고해 정확하게 답변하세요. "
                    "모르면 모른다고 답하세요.\n\n"
                    "[Context]\n{context}",
                ),
                MessagesPlaceholder("history"),
                ("human", "{input}"),
            ]
        )
        self.web_search_prompt = prompt.prompt_to_refine_text
        self.web_search_tool = TavilySearch(max_results=3)

        # --- 세션 히스토리 저장소 ---
        self._store: Dict[str, ChatMessageHistory] = {}

        # --- RAG 체인 (문자열 생성 전용) ---
        self._rag_generate = (
            {
                # context/documents는 state에 이미 포함. 여기서는 단순히 포맷팅/LLM/파서만.
                "context": RunnableLambda(lambda x: x.get("context", "")),
                "ability": RunnableLambda(lambda x: x.get("ability", "일반 지식")),
                "input": RunnablePassthrough(),
                "history": RunnableLambda(lambda x: x.get("history", [])),
            }
            | self.chat_prompt
            | self.llm
            | StrOutputParser()
        )

        # --- LangGraph 정의 ---
        self.work_chain = self._define_workflow()

    # -----------------------
    # 내부 유틸: 세션 히스토리
    # -----------------------
    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self._store:
            self._store[session_id] = ChatMessageHistory()
        return self._store[session_id]

    # -----------------------
    # 내부 유틸: 문서 검색 (chain.py → 통합)
    # -----------------------
    def _search_document(self, query: str) -> dict:
        # 1) 메타데이터에서 상위 문서 후보
        candidate_docs = self.metadata_retriever.get_relevant_documents(query)

        # 2) work pages
        selected_work = self.workpage_retriever.get_relevant_documents(query)

        # 3) 후보 문서를 chunk로 분할
        split_documents: List[Document] = []
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

            splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
            base_doc = Document(page_content=content, metadata={"path": json_path})
            split_documents.extend(splitter.split_documents([base_doc]))

        # 4) 벡터스토어로 질의와 유사 chunk 검색
        if split_documents:
            vectorstore = FAISS.from_documents(
                split_documents,
                embedding=self.metadata_retriever.vectorstore.embedding_function,
            )
            doc_retriever = vectorstore.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"score_threshold": self.score_threshold, "k": self.k_doc},
            )
            retrieved_chunks = doc_retriever.get_relevant_documents(query)
        else:
            doc_retriever = None
            retrieved_chunks = []

        # 5) context 문자열 구성
        context_parts = []
        if selected_work:
            context_parts.append("[행정 Work Pages]")
            for w in selected_work:
                url = w.metadata.get("url") or w.metadata.get("page_link", "")
                desc = w.metadata.get("desc", "")
                summarize = w.page_content
                context_parts.append(f"- {desc or summarize} | {url}")

        if retrieved_chunks:
            context_parts.append("[관련 문서]")
            for c in retrieved_chunks:
                context_parts.append(c.page_content)

        context_str = "\n".join(context_parts)

        return {
            "documents": retrieved_chunks,
            "doc_retriever": doc_retriever,
            "context": context_str,
        }

    # -----------------------
    # 노드: 검색 (그래프 첫 단계)
    # -----------------------
    def search_node(self, state: GraphState) -> GraphState:
        # state["input"] 은 이미 personal_info_keeper를 거친 "마스킹된 입력"
        query = state["input"]
        enriched = self._search_document(query)
        return {
            **state,
            "documents": enriched.get("documents"),
            "context": enriched.get("context"),
            "doc_retriever": enriched.get("doc_retriever"),
        }

    # -----------------------
    # 노드: 문서 채점/분기
    # -----------------------
    def grade_documents(self, state: GraphState):
        print("\n==== [CHECK DOCUMENT RELEVANCE TO QUESTION] ====\n")
        docs = state.get("documents")
        print(docs)
        has_docs = bool(docs) and len(docs) > 0
        result = {"web_search": "No" if has_docs else "Yes"}
        print(result)
        return result

    # -----------------------
    # 노드: 웹 서치
    # -----------------------
    def web_search(self, state: GraphState):
        print("\n==== [WEB SEARCH] ====\n")
        query_text, ability, history = state["input"], state["ability"], state["history"]
        docs = self.web_search_tool.invoke({"query": query_text})
        context = docs[0] if docs else self.llm.invoke(query_text)
        print("web search result : \n", context)
        # 웹서치 결과를 간략 맥락으로 요약/정제
        messages = self.web_search_prompt.format(context=context, question=query_text)
        generation = self._rag_generate.invoke(
            {"input": messages, "ability": ability, "history": history, "context": ""}
        )
        return {"generation": generation}

    # -----------------------
    # 노드: 생성
    # -----------------------
    def generate(self, state: GraphState):
        print("\n==== GENERATE ====\n")
        query_text, ability, history, context = (
            state["input"],
            state["ability"],
            state["history"],
            state.get("context", ""),
        )
        generation = self._rag_generate.invoke(
            {"input": query_text, "ability": ability, "history": history, "context": context}
        )
        return {"generation": generation}

    # -----------------------
    # 노드: 패스스루
    # -----------------------
    def passthrough(self, state: GraphState):
        print("==== [PASS THROUGH] ====")
        print("Final Answer:", state["generation"])
        return {"generation": state["generation"]}

    def decide_to_generate(self, state: GraphState):
        print("==== [ASSESS GRADED DOCUMENTS] ====")
        return "web_search_node" if state["web_search"] == "Yes" else "generate"

    # -----------------------
    # 그래프 정의
    # -----------------------
    def _define_workflow(self):
        workflow = StateGraph(GraphState)

        workflow.add_node("search_document", self.search_node)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("web_search_node", self.web_search)
        workflow.add_node("pass", self.passthrough)

        workflow.add_edge(START, "search_document")
        workflow.add_edge("search_document", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {"web_search_node": "web_search_node", "generate": "generate"},
        )
        workflow.add_edge("web_search_node", "pass")
        workflow.add_edge("generate", "pass")
        workflow.add_edge("pass", END)

        return workflow.compile()

    # -----------------------
    # 진입점
    # -----------------------
    def invoke(self, question: Any, config: Optional[dict] = None) -> str:
        """
        question 예:
        {
          "input": "<원본 raw 입력>",
          "ability": "민원 행정",
          "history": [...]   # 선택. 없으면 세션에서 로드
        }
        config 예:
        {
          "configurable": {"session_id": "abc123"}
        }
        """
        # 1) 세션 추출
        session_id = None
        if isinstance(config, dict):
            session_id = (
                config.get("configurable", {}).get("session_id")
                if config.get("configurable")
                else None
            )
        session_id = session_id or "default"

        # 2) 히스토리 로드
        history_obj = self._get_session_history(session_id)

        # 3) 원본 입력(raw)과 마스킹 입력(masked) 분리
        raw_input = question["input"]
        masked_input, raw_again = personal_info_keeper({"input": raw_input})  # (masked, raw)

        # 4) 세션 히스토리에 "원본 입력" 기록 (요구사항 2)
        history_obj.add_message(HumanMessage(content=raw_again))

        # 5) 그래프 State 구성: LLM에는 마스킹된 input 사용, history는 세션 전체 메시지
        state: GraphState = {
            "input": masked_input,
            "ability": question.get("ability", "일반 지식"),
            "documents": None,
            "history": history_obj.messages,  # 요구사항 1
            "generation": "",
            "web_search": "No",
            "context": "",
            "doc_retriever": None,
        }

        # 6) 그래프 실행
        result = self.work_chain.invoke(state, config=config)
        answer = result["generation"]

        # 7) 세션 히스토리에 모델 응답 기록
        history_obj.add_message(AIMessage(content=answer))

        return answer


# ===========================
# Factory (필요시 외부에서 간단 생성)
# ===========================
def get_chain(
    metadata_path: Path,
    workpages_path: Path,
    embeddings,
    *,
    k_doc: int = 5,
    score_threshold: float = 0.5,
    llm_key: str = "gpt-4o-mini",
) -> CRAG:
    """
    chain.py의 get_chain과 동일한 역할.
    - 반환: CRAG 인스턴스 (invoke로 바로 사용)
    """
    return CRAG(
        metadata_path=metadata_path,
        workpages_path=workpages_path,
        embeddings=embeddings,
        k_doc=k_doc,
        score_threshold=score_threshold,
        llm_key=llm_key,
    )
