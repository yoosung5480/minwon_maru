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
from minwon_maru.engine.chat import ProjPath, Chat, ChatManager

# ---------------------------
# 터미널 인터페이스
# ---------------------------
def chat_with_counselor():
    # 필요에 맞게 경로 설정
    paths = ProjPath(
        data_root_str="/Users/yujin/Desktop/코딩shit/python_projects/대한민국해커톤/민원마루ver1/datas",
        metadata_name="metadata.json",
        raw_docs_dir="rawDocs",
        parsed_docs_dir="parsedDocs",
        workpages_name="work_page_info.json",
        reference_root_str="reference",
    )

    manager = ChatManager(paths)

    while True:
        print("\n[메뉴]")
        print("1) 채팅시작")
        print("2) 채팅끝")
        print("0) 종료(프로그램 종료)")
        cmd = input("> ").strip()

        if cmd == "1":
            chat_id = manager.start_chat()
            print(f"채팅이 시작되었습니다. chat_id={chat_id}")

            while True:
                user_q = input("\n질문을 입력하세요:\n> ").strip()
                if not user_q:
                    print("빈 입력입니다. 다시 입력하세요.")
                    continue

                try:
                    answer = manager.ask(user_q)
                except Exception as e:
                    print(f"[에러] {e}")
                    break

                print("\n[답변]")
                print(answer)

                print("\n다음 선택지를 고르세요:")
                print("1) 추가질문")
                print("2) 대화끝")
                follow = input("> ").strip()

                if follow == "1":
                    continue
                elif follow == "2":
                    ok = manager.end_chat()
                    print("대화를 종료했습니다." if ok else "이미 종료된 대화입니다.")
                    break
                else:
                    print("잘못된 입력입니다. 추가질문으로 계속합니다.")
                    continue

        elif cmd == "2":
            ok = manager.end_chat()
            print("대화를 종료했습니다." if ok else "진행 중인 대화가 없습니다.")

        elif cmd == "0":
            print("프로그램을 종료합니다.")
            sys.exit(0)

        else:
            print("올바른 번호를 입력하세요.")

