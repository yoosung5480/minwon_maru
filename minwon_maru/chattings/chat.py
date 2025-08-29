import minwon_maru.tools.llms as llms
import speak_note.work_flows.RAG as rag
# import speak_note.work_flows.basic_CRAG as crag
from minwon_maru.work_flows.basic_CRAG import CRAG, en_CRAG
import minwon_maru.tools.myPDFparser as myPDFparser
import minwon_maru.tools.context as context


class Chat():
    def __init__(self, chat_id):   
        # chat_id 는 채팅별로 고유해야한다.
        self.chat_id = chat_id      # 
        self.documnet_path_list = []        # 
        self.RAG_chains = None
        self.work_chain = None
        self.input_texts = []       # STT 반환 텍스트
    

    async def set_document(self, document_path: str):
        self.documnet_path_list.append(document_path)
        self.cur_document_path = document_path

        # GPT context (비동기 생성자 사용)
        gpt_ctx = await context.Context.create(document_path)
        gpt_ctx.set_context(**context.context_configs["gpt"])
        gpt_ctx.set_retriever(context.retriever_configs["balanced"])

        # Upstage context
        upstage_ctx = await context.Context.create(document_path)
        upstage_ctx.set_context(**context.context_configs["upstage"])
        upstage_ctx.set_retriever(context.retriever_configs["balanced"])

        # RAG 체인 등록
        RAG_chains = {
            "gpt_RAG": rag.basic_RAG(gpt_ctx.get_retriever()),
            "upstage_RAG": rag.basic_RAG(upstage_ctx.get_retriever())
        }
        self.RAG_chains = RAG_chains
        return self




    def set_work_chain(self, work_chain = None):
        # work_chain : invoke(str) 라는 공통 API를 갖는 객체여야한다.
        if self.RAG_chains == None:
            print("document not ready, you should invoke set_document before set work chain.")
            return
        
        if work_chain == None:
            self.work_chain = CRAG(rag_pipeline=self.RAG_chains["gpt_RAG"])
        else :
            self.work_chain = work_chain
    
    
    async def ainvoke(self, chat_id:str, msg:str):
        if self.work_chain == None:
            print("you should set work chain first.")
            return
        
        if chat_id != self.chat_id :
            print("id not matched")
            return
        else : 
            result = await self.work_chain.ainvoke(msg)
            return result



class en_Chat(Chat):
    async def set_document(self, document_path : str):
        self.documnet_path_list.append(document_path)
        self.cur_document_path = document_path

        # GPT context
        gpt_ctx = await context.en_Context.create(document_path)
        gpt_ctx.set_context(**context.context_configs["gpt"])
        gpt_ctx.set_retriever(context.retriever_configs["balanced"])

        # Upstage context
        upstage_ctx = await context.en_Context.create(document_path)
        upstage_ctx.set_context(**context.context_configs["upstage"])
        upstage_ctx.set_retriever(context.retriever_configs["balanced"])

        # RAG 체인 등록
        RAG_chains = {
            "gpt_RAG": rag.en_RAG(gpt_ctx.get_retriever()),
            "upstage_RAG": rag.en_RAG(upstage_ctx.get_retriever())
        }
        self.RAG_chains = RAG_chains
        return self



    def set_work_chain(self, work_chain = None):
        # work_chain : invoke(str) 라는 공통 API를 갖는 객체여야한다.
        if self.RAG_chains == None:
            print("document not ready, you should invoke set_document before set work chain.")
            return
        
        if work_chain == None:
            self.work_chain = en_CRAG(rag_pipeline=self.RAG_chains["gpt_RAG"])
        else :
            self.work_chain = work_chain







    
