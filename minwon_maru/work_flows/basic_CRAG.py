# basic_CRAG.py
from langgraph.graph import END, StateGraph, START
from typing import Annotated, List, Any, Optional
from typing_extensions import TypedDict
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser

from minwon_maru.prompts import prompt
from minwon_maru.tools import llms
from langchain_teddynote.tools.tavily import TavilySearch

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
    
class CRAG(Workflow):
    def __init__(self, rag_pipeline, search_fn, llm_list=llms.llm_list):
        self.llm_list = llm_list
        self.rag_pipeline = rag_pipeline
        self.web_search_prompt = prompt.prompt_to_refine_text

        # 주입형 search 함수(순환 import 회피)
        self.search_fn = search_fn

        self.web_search_tool = TavilySearch(max_results=3)
        self.work_chain = self.define_workflow()

    # ==================== 노드 함수 ====================
    def search_document(self, state: GraphState):
        """
        외부에서 주입된 search_fn을 호출해
        state에 documents/context/doc_retriever를 채워서 반환.
        """
        enriched = self.search_fn(state)  # dict 반환 가정
        # GraphState에 필요한 키만 merge
        return {
            **state,
            "documents": enriched.get("documents"),
            "input": enriched.get("input", state.get("input")),
            "ability": enriched.get("ability", state.get("ability")),
            # 필요시 context/doc_retriever도 상태로 유지하고 싶으면 여기에 포함
            "context": enriched.get("context"),
            "doc_retriever": enriched.get("doc_retriever"),
        }
    

    def grade_documents(self, state: GraphState):
        """
        search_document가 만들어 둔 documents(List[Document])가
        존재하고 길이가 >0 이면 바로 generate, 아니면 web_search.
        """
        print("\n==== [CHECK DOCUMENT RELEVANCE TO QUESTION] ====\n")
        docs = state.get("documents")
        print(docs)
        has_docs = bool(docs) and len(docs) > 0
        result = {"web_search": "No" if has_docs else "Yes"}
        print(result)
        return result

    def web_search(self, state: GraphState):
        print("\n==== [WEB SEARCH] ====\n")
        query_text, ability = state["input"], state["ability"]

        docs = self.web_search_tool.invoke({"query": query_text})
        context = docs[0] if docs else self.llm_list["gpt-4o-mini"].invoke(query_text)
        print("web search result : \n", context)
        messages = self.web_search_prompt.format(context=context, question=query_text)
        generation = self.rag_pipeline.invoke({"input": messages, "ability": ability})
        return {"generation": generation}

    def generate(self, state: GraphState):
        print("\n==== GENERATE ====\n")
        # rag_pipeline(with_message_history)은 personal_info_keeper → search_document → RAGchain
        # 순으로 이미 context를 만들게 구성되어 있다고 가정.
        # (state["documents"]는 분기 판단에만 사용)
        query_text, ability = state["input"], state["ability"]
        generation = self.rag_pipeline.invoke({"input": query_text, "ability": ability})
        return {"generation": generation}

    def passthrough(self, state: GraphState):
        print("==== [PASS THROUGH] ====")
        print("Final Answer:", state["generation"])
        return {"generation": state["generation"]}

    def decide_to_generate(self, state: GraphState):
        print("==== [ASSESS GRADED DOCUMENTS] ====")
        return "web_search_node" if state["web_search"] == "Yes" else "generate"

    # ==================== 그래프 정의 ====================

    # 문제점. generate에서는 invoke로 완성된 문자열만 반환해준다. 즉, search_document의 그 어떤 
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
        workflow.add_edge("generate", "pass")   # ← 누락됐던 엣지 추가
        workflow.add_edge("pass", END)

        return workflow.compile()


    # ==================== 진입점 ====================
    def invoke(self, question: Any, config: Optional[dict] = None) -> str:
        """
        question 예:
        {
          "input": "...",
          "ability": "...",
          "documents": [...],   # ← search_document가 만든 List[Document]를 그대로 넣으면 분기에 사용됨
          "history": [...]
        }
        """
        state = {**question}
        result = self.work_chain.invoke(state, config=config)
        return result["generation"]

