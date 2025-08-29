import asyncio
import json
from pathlib import Path
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from minwon_maru.tools.json_tool import load_json
from minwon_maru.tools.llms import llm_list


prompt_to_summarize = """   
    당신은 행정문서를 한줄 요약해주는 어시스턴트입니다.
    문서의 정보를 반영한 최대한 짧고 간결한 문서 요약내용을 제시하세요.
    이모티콘, 기호 사용금지.

    #User Input (요약 요청 내용):
    {question}

    #요약 내용:
    """


def get_str_from_doc(doc_contents):
    """여러 Document 객체 리스트에서 page_content만 합쳐서 반환"""
    page_content = ""
    for idx in range(len(doc_contents)):
        try:
            contents = doc_contents[idx].to_json()["kwargs"]["page_content"]
        except Exception as e:
            print(f"⚠️  페이지 {idx} 읽기 실패: {e}")
            continue
        page_content = page_content + "\n" + contents
    return page_content.strip()


async def generate_metadata(
    server_id: str,
    data_root_str: str = "/Users/yujin/Desktop/코딩shit/python_projects/대한민국해커톤/민원마루ver1/datas",
    parsed_docs_name: str = "parsedDocs",
    llm=llm_list["solar-pro2"],
    prompt_to_summarize: str = prompt_to_summarize,
) -> dict:
    """
    주어진 파라미터에 따라 문서를 읽고 요약 후 metadata 생성.

    Args:
        server_id: 서버 식별자
        data_root_str: 데이터 루트 경로
        parsed_docs_name: JSON이 저장된 폴더 이름
        llm: 사용할 LLM (예: llm_list["solar-pro2"])
        prompt_to_summarize: 요약용 프롬프트 문자열

    Returns:
        metadata 딕셔너리
    """

    data_root = Path(data_root_str)
    metadata_path = data_root.joinpath("metadata.json")
    parsed_docs_root = data_root.joinpath(parsed_docs_name)

    # JSON 파일 불러오기
    json_paths = [f for f in parsed_docs_root.rglob("*.json") if f.is_file()]
    if not json_paths:
        print("❌ JSON 파일을 찾지 못했습니다.")
        return {"DocNum": 0, "serverID": server_id, "Docs": []}

    print(f"🔍 감지된 JSON 파일 수: {len(json_paths)}")

    docs = []
    for json_path in json_paths:
        try:
            loaded = load_json(json_path)
            docs.append((json_path, loaded))
        except Exception as e:
            print(f"⚠️ {json_path.name} 로드 실패: {e}")

    print(f"📂 성공적으로 읽은 파일 수: {len(docs)}")

    if len(docs) == 0:
        return {"DocNum": 0, "serverID": server_id, "Docs": []}

    # Document → 문자열 변환
    doc_contents = [get_str_from_doc(doc) for _, doc in docs]

    # Chain 준비
    try:
        basic_prompt_template = PromptTemplate.from_template(prompt_to_summarize)
        chain = basic_prompt_template | llm | StrOutputParser()
    except Exception as e:
        print(f"❌ 프롬프트/LLM 체인 생성 실패: {e}")
        return {"DocNum": 0, "serverID": server_id, "Docs": []}

    # 비동기 요약 태스크 실행
    try:
        tasks = [chain.ainvoke(doc_content) for doc_content in doc_contents]
        results = await asyncio.gather(*tasks)
    except Exception as e:
        print(f"❌ 요약 태스크 실행 중 오류 발생: {e}")
        return {"DocNum": 0, "serverID": server_id, "Docs": []}

    # metadata 생성
    docs_list = []
    for summary, (path, _) in zip(results, docs):
        docs_list.append({
            "summarize": summary,
            "json_path": str(path)
        })

    metadata = {
        "DocNum": len(results),
        "serverID": server_id,
        "Docs": docs_list,
    }

    try:
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"✅ metadata.json 저장 완료 → {metadata_path}")
    except Exception as e:
        print(f"❌ metadata 저장 실패: {e}")

    return metadata
