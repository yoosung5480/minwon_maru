from flask import Flask
from flask_socketio import SocketIO, emit
from minwon_maru.engine.chat import ChatManager, ProjPath

# Flask + SocketIO 초기화
app = Flask(__name__)
app.config["SECRET_KEY"] = "secret!"
socketio = SocketIO(app, cors_allowed_origins="*")  # CORS 허용

# ChatManager 준비
paths = ProjPath(
    data_root_str="/Users/yujin/Desktop/코딩shit/python_projects/대한민국해커톤/민원마루ver1/datas",
    metadata_name="metadata.json",
    raw_docs_dir="rawDocs",
    parsed_docs_dir="parsedDocs",
    workpages_name="work_page_info.json",
    reference_root_str="reference",
    departemt_info_name="department_info.json",
)

manager = ChatManager(paths)

# 세션별 chat_id 저장 (간단 구현)
user_chat_sessions = {}


# -------------------------
# WebSocket 이벤트 핸들러
# -------------------------

@socketio.on("connect")
def handle_connect():
    print("클라이언트 연결됨")
    emit("server_response", {"msg": "민원마루 AI 챗봇에 연결되었습니다."})


@socketio.on("start_chat")
def handle_start_chat(data):
    """프런트에서 사용자가 접속했을 때 chat 세션 시작"""
    user_id = data.get("user_id")
    chat_id = manager.start_chat()
    user_chat_sessions[user_id] = chat_id
    emit("server_response", {"msg": f"새로운 상담 세션이 시작되었습니다.", "chat_id": chat_id})


@socketio.on("user_question")
def handle_user_question(data):
    """사용자가 질문을 보냈을 때 응답"""
    user_id = data.get("user_id")
    question = data.get("question")

    if user_id not in user_chat_sessions:
        emit("server_response", {"msg": "먼저 start_chat 이벤트를 호출하세요."})
        return

    chat_id = user_chat_sessions[user_id]

    # ChatManager에게 전달
    response = manager.ask(question)

    emit("server_response", {
        "user_id": user_id,
        "question": question,
        "answer": response["generation"],
        "relavent_workpages": response.get("relavent_workpages", ""),
        "departments": manager.chat.get_most_relevant_department(top_k=3),
    })


@socketio.on("disconnect")
def handle_disconnect():
    print("클라이언트 연결 종료")


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
