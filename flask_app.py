from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from datetime import datetime
from pathlib import Path

# 민원 마루 시스템 import
from minwon_maru.engine.chat import ProjPath, Chat, ChatManager

app = Flask(__name__)
CORS(app)  # CORS 설정으로 프론트엔드와 통신 가능

# 전역 변수로 ChatManager 관리
chat_manager = None

def initialize_chat_manager():
    """ChatManager 초기화"""
    global chat_manager
    
    # 현재 작업 디렉토리 기준으로 경로 설정
    current_dir = Path.cwd()
    data_root = current_dir / "datas"
    
    paths = ProjPath(
        data_root_str=str(data_root),
        metadata_name="metadata.json",
        raw_docs_dir="rawDocs",
        parsed_docs_dir="parsedDocs",
        workpages_name="work_page_info.json",
        reference_root_str="reference",
        departemt_info_name="department_info.json"
    )
    
    chat_manager = ChatManager(paths)
    print("ChatManager 초기화 완료")

@app.route('/api/chat/ask', methods=['POST'])
def ask_question():
    """질문하기 API"""
    try:
        # 요청 데이터 파싱
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': '요청 데이터가 없습니다.'
            }), 400
        
        chat_id = data.get('chatId')
        question = data.get('question')
        
        if not chat_id:
            return jsonify({
                'success': False,
                'message': 'chatId가 필요합니다.'
            }), 400
        
        if not question:
            return jsonify({
                'success': False,
                'message': '질문이 필요합니다.'
            }), 400
        
        # ChatManager가 초기화되지 않았다면 초기화
        if chat_manager is None:
            initialize_chat_manager()
        
        # 채팅이 시작되지 않았다면 시작
        if chat_manager.chat is None:
            chat_manager.start_chat()
        
        # 질문 처리
        try:
            response = chat_manager.ask(question)
            
            # 답변 추출
            if isinstance(response, dict) and 'generation' in response:
                answer = response['generation']
            else:
                answer = str(response)
            
            # 관련 부서 정보 가져오기
            relevant_departments = []
            try:
                if chat_manager.chat:
                    relevant_departments = chat_manager.chat.get_most_relevant_department(top_k=3, window=3)
            except Exception as e:
                print(f"부서 정보 조회 중 오류: {e}")
            
            # 컨텍스트 정보 (문서 정보)
            context = ""
            if isinstance(response, dict) and 'context' in response:
                context = response['context']
            
            return jsonify({
                'success': True,
                'answer': answer,
                'relevantDepartments': relevant_departments,
                'context': context,
                'message': '질문이 성공적으로 처리되었습니다.'
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'질문 처리 중 오류가 발생했습니다: {str(e)}'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'서버 오류가 발생했습니다: {str(e)}'
        }), 500

@app.route('/api/chat/start', methods=['POST'])
def start_chat():
    """채팅 시작 API"""
    try:
        # ChatManager가 초기화되지 않았다면 초기화
        if chat_manager is None:
            initialize_chat_manager()
        
        # 채팅 시작
        chat_id = chat_manager.start_chat()
        
        return jsonify({
            'success': True,
            'chatId': chat_id,
            'message': '채팅이 시작되었습니다.'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'채팅 시작 중 오류가 발생했습니다: {str(e)}'
        }), 500

@app.route('/api/chat/end', methods=['POST'])
def end_chat():
    """채팅 종료 API"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': '요청 데이터가 없습니다.'
            }), 400
        
        chat_id = data.get('chatId')
        
        if not chat_id:
            return jsonify({
                'success': False,
                'message': 'chatId가 필요합니다.'
            }), 400
        
        # 채팅 종료
        if chat_manager and chat_manager.chat:
            success = chat_manager.end_chat()
            
            if success:
                return jsonify({
                    'success': True,
                    'message': '채팅이 종료되었습니다.'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': '이미 종료된 채팅입니다.'
                }), 400
        else:
            return jsonify({
                'success': False,
                'message': '진행 중인 채팅이 없습니다.'
            }), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'채팅 종료 중 오류가 발생했습니다: {str(e)}'
        }), 500

@app.route('/api/chat/<chat_id>/history', methods=['GET'])
def get_chat_history(chat_id):
    """채팅 히스토리 조회 API"""
    try:
        if not chat_manager or not chat_manager.chat:
            return jsonify({
                'success': False,
                'message': '진행 중인 채팅이 없습니다.'
            }), 400
        
        # 현재 채팅의 히스토리 가져오기
        input_history, output_history = chat_manager.chat.get_chat_history()
        
        # 히스토리를 메시지 형태로 변환
        history = []
        for i, (user_input, assistant_output) in enumerate(zip(input_history, output_history)):
            # 사용자 메시지
            history.append({
                'type': 'user',
                'content': user_input,
                'timestamp': datetime.now().isoformat()  # 실제로는 저장된 타임스탬프 사용
            })
            
            # 어시스턴트 메시지
            history.append({
                'type': 'assistant',
                'content': assistant_output,
                'timestamp': datetime.now().isoformat()  # 실제로는 저장된 타임스탬프 사용
            })
        
        return jsonify({
            'success': True,
            'chatId': chat_id,
            'history': history,
            'message': '히스토리가 성공적으로 조회되었습니다.'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'히스토리 조회 중 오류가 발생했습니다: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """헬스 체크 API"""
    return jsonify({
        'success': True,
        'message': '서버가 정상적으로 실행 중입니다.',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # 서버 시작 시 ChatManager 초기화
    initialize_chat_manager()
    
    print("민원 마루 Flask 서버가 시작되었습니다.")
    print("API 엔드포인트:")
    print("- POST /api/chat/start: 채팅 시작")
    print("- POST /api/chat/ask: 질문하기")
    print("- POST /api/chat/end: 채팅 종료")
    print("- GET /api/chat/<chatId>/history: 채팅 히스토리")
    print("- GET /health: 서버 상태 확인")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
