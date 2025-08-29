from pathlib import Path
import json

# speak_note.engine.chatting # chat.py 구현내용
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

from minwon_maru.tools.context import create_metadata_retriever_with_map, create_workpages_retriever
from minwon_maru.tools.llms import llm_list
from minwon_maru.tools.personal_info_keeper import personal_info_keeper
from minwon_maru.work_flows.basic_CRAG import CRAG
import asyncio


from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings




#########################################################################################################
# 현재 meta_data, work_page 를 기반으로 유사도 검색후 코사인 유사도 0.5 이상만 검출, 
#########################################################################################################
def search_document(inputs: dict,
                    metadata_retriever,       # 외부에서 생성한 retriever
                    summarize_map: dict,      # retriever와 함께 생성된 매핑 dict
                    workpage_retriever        # 외부에서 생성한 retriever
                   ) -> dict:
    '''
    Args:
        inputs : {"input": "...", "ability": "...", "history": ...}
        metadata_retriever : create_metadata_retriever_with_map()로 만든 retriever
        summarize_map : summarize → 원문 정보 매핑 dict
        workpage_retriever : create_workpages_retriever()로 만든 retriever

    Returns:
        inputs + {"context": "..."}
    '''

    query = inputs["input"]
    outputs = dict(inputs)
    # 1. 질의와 유사한 상위 문서 가져오기
    candidate_docs = metadata_retriever.get_relevant_documents(query)

    # 2. 질의와 유사한 workpage 검색
    selected_work = workpage_retriever.get_relevant_documents(query)

    # 3. 선택된 문서를 chunk로 분할
    split_documents = []
    for d in candidate_docs:
        origin_info = summarize_map.get(d.page_content, {})
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

    # 4. 벡터스토어 생성 후, query와 0.5 이상 문서 chunk 검색
    if split_documents:
        vectorstore = FAISS.from_documents(
            split_documents,
            embedding=metadata_retriever.vectorstore.embedding_function
        )
        doc_retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.5, "k": 5}
        )
        retrieved_chunks = doc_retriever.get_relevant_documents(query)
                                                              # 출력이 존재한다.
    else:
        doc_retriever = None
        retrieved_chunks = []

    # context 조합
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
    # print("context : \n", context_str)      # 
    
    outputs["documents"] = retrieved_chunks
    print("===============documents===============")  
    print(outputs["documents"])
    outputs["context"] = context_str
    outputs["doc_retriever"] = doc_retriever   
    return outputs



# 1) 프롬프트 정의
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "당신은 {ability}에 능숙한 민원 응대 상담원입니다. "
            "아래 문서 맥락(Context)을 우선적으로 참고해 정확하게 답변하세요. "
            "모르면 모른다고 답하세요.\n\n"
            "[Context]\n{context}"
        ),
        MessagesPlaceholder("history"),
        ("human", "{input}"),
    ]
)



#########################################################################################################
# 답변 체인 생성.
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
            "context": RunnableLambda(
                lambda x: (
                    x["doc_retriever"].get_relevant_documents(x["input"])
                    if x.get("doc_retriever") else []
                )
            ),
            "documents": RunnableLambda(lambda x: x.get("documents", [])),
            "ability": RunnableLambda(lambda x: x.get("ability", "일반 지식")),
            "input": RunnablePassthrough(),
            "history": RunnableLambda(lambda x: x.get("history", [])),
        }
        | prompt
        | llm_list["gpt-4o-mini"]
        | StrOutputParser()
    )

    # 3) 세션 히스토리
    store = {}
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    # 4) personal_info_keeper → search_document → RAG
    personal_info_keeper_r = RunnableLambda(lambda x: personal_info_keeper(x)[0])   # personal_info_keeper(x)[1] 값도 사용해서  "원본 input값을 세션 히스토리로 남겨야함"

    # 주입용 search_fn (CRAG 그래프의 search_document 노드가 호출)
    def search_fn(payload: dict) -> dict:
        return search_document(
            payload,
            metadata_retriever=metadata_retriever,
            summarize_map=summarize_map,
            workpage_retriever=workpage_retriever,
        )

    # 그래프 밖에선 personal_info_keeper만 적용 (검색은 그래프 첫 노드에서 수행)
    full_chain = personal_info_keeper_r | RAGchain  # llm에 전달할때는 마스킹된 input만 사용한다.

    with_message_history = RunnableWithMessageHistory(
        full_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # CRAG에 rag_pipeline과 search_fn을 주입
    crag_chain = CRAG(
        rag_pipeline=with_message_history,
        search_fn=search_fn, 
    )
    return crag_chain

