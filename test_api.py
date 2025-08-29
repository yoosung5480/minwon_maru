#!/usr/bin/env python3
"""
민원 마루 Flask API 테스트 스크립트
"""

import requests
import json
import time

# API 기본 URL
BASE_URL = "http://localhost:5000"

def test_health():
    """헬스 체크 테스트"""
    print("=== 헬스 체크 테스트 ===")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"에러: {e}")
        return False

def test_start_chat():
    """채팅 시작 테스트"""
    print("\n=== 채팅 시작 테스트 ===")
    try:
        response = requests.post(f"{BASE_URL}/api/chat/start")
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Response: {result}")
        
        if result.get('success'):
            return result.get('chatId')
        return None
    except Exception as e:
        print(f"에러: {e}")
        return None

def test_ask_question(chat_id, question):
    """질문하기 테스트"""
    print(f"\n=== 질문하기 테스트: {question} ===")
    try:
        data = {
            "chatId": chat_id,
            "question": question
        }
        
        response = requests.post(
            f"{BASE_URL}/api/chat/ask",
            json=data
        )
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        return result.get('success', False)
    except Exception as e:
        print(f"에러: {e}")
        return False

def test_get_history(chat_id):
    """히스토리 조회 테스트"""
    print(f"\n=== 히스토리 조회 테스트 ===")
    try:
        response = requests.get(f"{BASE_URL}/api/chat/{chat_id}/history")
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        return result.get('success', False)
    except Exception as e:
        print(f"에러: {e}")
        return False

def test_end_chat(chat_id):
    """채팅 종료 테스트"""
    print(f"\n=== 채팅 종료 테스트 ===")
    try:
        data = {
            "chatId": chat_id
        }
        
        response = requests.post(
            f"{BASE_URL}/api/chat/end",
            json=data
        )
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Response: {result}")
        
        return result.get('success', False)
    except Exception as e:
        print(f"에러: {e}")
        return False

def main():
    """메인 테스트 함수"""
    print("민원 마루 Flask API 테스트를 시작합니다...")
    
    # 1. 헬스 체크
    if not test_health():
        print("서버가 실행되지 않았습니다. flask_app.py를 먼저 실행해주세요.")
        return
    
    # 2. 채팅 시작
    chat_id = test_start_chat()
    if not chat_id:
        print("채팅 시작에 실패했습니다.")
        return
    
    print(f"채팅 ID: {chat_id}")
    
    # 3. 질문하기 테스트
    test_questions = [
        "도서관 이용 시간이 어떻게 되나요?",
        "도서관에서 책을 빌릴 수 있나요?",
        "주차장 이용은 어떻게 되나요?"
    ]
    
    for question in test_questions:
        success = test_ask_question(chat_id, question)
        if not success:
            print(f"질문 '{question}' 처리에 실패했습니다.")
        time.sleep(1)  # API 호출 간격 조절
    
    # 4. 히스토리 조회
    test_get_history(chat_id)
    
    # 5. 채팅 종료
    test_end_chat(chat_id)
    
    print("\n=== 테스트 완료 ===")

if __name__ == "__main__":
    main()
