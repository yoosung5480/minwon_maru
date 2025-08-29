import os
from pathlib import Path
import asyncio
import json

from minwon_maru.tools.myPDFparser import upstageParser2Document
from minwon_maru.tools.json_tool import load_json


async def load_docs(
        data_root_str :str = "/Users/yujin/Desktop/코딩shit/python_projects/대한민국해커톤/민원마루ver1/datas",
        rawDocs_name : str = "rawDocs",
        parsed_docs_name : str = "parsedDocs"
        ) -> int:   # 성공 0, 실패 -1

    data_root = Path(data_root_str)
    raw_docs_root = data_root.joinpath(rawDocs_name)
    parsed_docs_root = data_root.joinpath(parsed_docs_name)

    # 원본 파일 리스트 (파일명, Path)
    files = [(f.name, f) for f in raw_docs_root.rglob("*.pdf") if f.is_file()]
    if not files:
        print(" no pdf files found")
        return -1

    # 대응되는 JSON 저장 경로 생성
    def pdf2json_path(pdf_file: str):
        return parsed_docs_root.joinpath(Path(pdf_file).stem + ".json")

    save_names = [pdf2json_path(f[0]) for f in files]

    try:
        # 비동기 병렬 파싱
        tasks = [upstageParser2Document(str(f[1])) for f in files]
        results = await asyncio.gather(*tasks)

        # 파일별로 저장
        for i, save_name in enumerate(save_names):
            docs = results[i]  # 한 pdf의 여러 페이지 Document 리스트
            docs_serialized = [d.dict() for d in docs]

            with save_name.open("w", encoding="utf-8") as f:
                json.dump(docs_serialized, f, ensure_ascii=False, indent=2)

        return 0

    except Exception as e:
        print("error:", e)
        return -1
