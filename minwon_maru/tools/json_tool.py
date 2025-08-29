import os
from typing import Dict, Any
from pathlib import Path
import json
from langchain.schema import Document

def load_json_basic(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"[WARN] JSONDecodeError: {path} → 빈 dict 반환")
        return {}
    

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        docs = [Document(**d) for d in data]
        return docs
    except json.JSONDecodeError:
        print(f"[WARN] JSONDecodeError: {path} → 빈 리스트 반환")
        return {}
    
def load_metadata(metadata_path: str | Path) -> dict:
    """
    metadata.json 파일을 읽어서 파이썬 dict로 반환

    Args:
        metadata_path: 메타데이터 JSON 파일 경로 (str 또는 Path)

    Returns:
        dict: {
            "DocNum": int,
            "serverID": str,
            "Docs": [ {요약: 경로}, ... ]
        }
    """
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    return metadata


