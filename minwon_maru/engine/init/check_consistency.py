import os
import json

def check_filesystem_concurrency(
    data_root: str = "/Users/yujin/Desktop/코딩shit/python_projects/대한민국해커톤/민원마루ver1/datas",
    rawDocs_name: str = "rawDocs",
    parsed_docs_name: str = "parsedDocs",
    metadata_name: str = "metadata.json",
) -> bool:
    """
    파일시스템과 metadata.json의 동시성 체크 함수
    1. rawDocs/*.pdf 와 parsedDocs/*.json 이름 일치 확인
    2. metadata.json 안의 json 경로와 parsedDocs/*.json 정확히 일치 확인
    
    Returns:
        bool: True (정상 동기화), False (불일치 발생)
    """

    raw_dir = os.path.join(data_root, rawDocs_name)
    parsed_dir = os.path.join(data_root, parsed_docs_name)
    metadata_path = os.path.join(data_root, metadata_name)

    # --- 1. rawDocs와 parsedDocs 이름 동기화 ---
    pdf_files = [os.path.splitext(f)[0] for f in os.listdir(raw_dir) if f.endswith(".pdf")]
    json_files = [os.path.splitext(f)[0] for f in os.listdir(parsed_dir) if f.endswith(".json")]

    # rawDocs에 없는 json 파일 제거
    for jf in json_files:
        if jf not in pdf_files:
            os.remove(os.path.join(parsed_dir, jf + ".json"))
            print(f"[정리] rawDocs에 없는 {jf}.json 제거")

    # json_files 갱신
    json_files = [os.path.splitext(f)[0] for f in os.listdir(parsed_dir) if f.endswith(".json")]

    # pdf와 json 개수/이름 비교
    if set(pdf_files) != set(json_files):
        print("[경고] rawDocs와 parsedDocs가 일치하지 않습니다.")
        return False

    # --- 2. metadata.json과 parsedDocs 일치 확인 ---
    if not os.path.exists(metadata_path):
        print("[에러] metadata.json 파일이 존재하지 않습니다.")
        return False

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    docnum = metadata.get("DocNum", 0)
    print(docnum)
    docs_list = metadata.get("Docs", [])

    # metadata["Docs"]["json_path] 에서 json 경로 추출해야하는데, 그냥 다 가져오고있다.
    # 그래서 실제론 데이터 갯수가 DocNum=7로 맞지만, 현재  print(len(metadata_json_files)) = 14이다.
    metadata_json_files = []
    for d in docs_list:
        if "json_path" in d:
            fname = os.path.splitext(os.path.basename(d["json_path"]))[0]
            metadata_json_files.append(fname)

    parsed_json_set = set(json_files)
    metadata_set = set(metadata_json_files)
    parsed_json_set = set(json_files)
    metadata_set = set(metadata_json_files)
    print(len(metadata_json_files))
    if len(metadata_json_files) != docnum:
        print("[경고] metadata DocNum과 Docs 리스트 개수가 다릅니다.")
        return False

    if parsed_json_set != metadata_set:
        print("[경고] metadata.json과 parsedDocs 내용이 불일치합니다.")
        return False

    print("[성공] 파일시스템과 metadata.json이 동기화되어 있습니다.")
    return True
