# 민원 마루 Flask API 서버

민원 마루 시스템을 Flask 기반의 REST API로 제공하는 서버입니다.

## 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 서버 실행
```bash
python flask_app.py
```

서버가 `http://localhost:5001`에서 실행됩니다.

## API 엔드포인트

### 1. 채팅 시작
**POST** `/api/chat/start`

채팅 세션을 시작합니다.

**Response:**
```json
{
  "success": true,
  "chatId": "20241201120000",
  "message": "채팅이 시작되었습니다."
}
```

### 2. 질문하기 (핵심 API)
**POST** `/api/chat/ask`

사용자의 질문을 처리하고 답변을 생성합니다.

**Request:**
```json
{
  "chatId": "20241201120000",
  "question": "도서관 이용 시간이 어떻게 되나요?"
}
```

**Response:**
```json
{
  "success": true,
  "answer": "도서관 이용 시간은 평일 오전 9시부터 오후 6시까지, 주말은 오전 9시부터 오후 5시까지입니다.",
  "relevantDepartments": [
    {
      "id": "dept001",
      "name": "도서관과",
      "phone": "051-123-4567"
    }
  ],
  "context": "도서관 운영 규정 제3조에 따르면...",
  "message": "질문이 성공적으로 처리되었습니다."
}
```

### 3. 채팅 종료
**POST** `/api/chat/end`

채팅 세션을 종료합니다.

**Request:**
```json
{
  "chatId": "20241201120000"
}
```

**Response:**
```json
{
  "success": true,
  "message": "채팅이 종료되었습니다."
}
```

### 4. 채팅 히스토리 조회
**GET** `/api/chat/{chatId}/history`

채팅 세션의 대화 기록을 조회합니다.

**Response:**
```json
{
  "success": true,
  "chatId": "20241201120000",
  "history": [
    {
      "type": "user",
      "content": "도서관 이용 시간이 어떻게 되나요?",
      "timestamp": "2024-12-01T12:00:00"
    },
    {
      "type": "assistant",
      "content": "도서관 이용 시간은 평일 오전 9시부터 오후 6시까지, 주말은 오전 9시부터 오후 5시까지입니다.",
      "timestamp": "2024-12-01T12:00:01"
    }
  ],
  "message": "히스토리가 성공적으로 조회되었습니다."
}
```

### 5. 서버 상태 확인
**GET** `/health`

서버의 상태를 확인합니다.

**Response:**
```json
{
  "success": true,
  "message": "서버가 정상적으로 실행 중입니다.",
  "timestamp": "2024-12-01T12:00:00"
}
```

## TypeScript 인터페이스

프론트엔드에서 사용할 수 있는 TypeScript 인터페이스입니다:

```typescript
interface AskQuestionRequest {
  chatId: string;
  question: string;
}

interface AskQuestionResponse {
  success: boolean;
  answer: string;
  relevantDepartments?: DepartmentInfo[];
  context?: string;
  message?: string;
}

interface DepartmentInfo {
  id: string;
  name: string;
  phone: string;
}

interface ChatMessage {
  type: 'user' | 'assistant';
  content: string;
  timestamp: string;
}
```

## 사용 예시

### JavaScript/TypeScript에서 API 호출

```typescript
// 채팅 시작
const startChat = async () => {
  const response = await fetch('/api/chat/start', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    }
  });
  const data = await response.json();
  return data.chatId;
};

// 질문하기
const askQuestion = async (chatId: string, question: string) => {
  const response = await fetch('/api/chat/ask', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      chatId,
      question
    })
  });
  return await response.json();
};

// 사용 예시
const chatExample = async () => {
  // 1. 채팅 시작
  const chatId = await startChat();
  console.log('채팅 시작:', chatId);
  
  // 2. 질문하기
  const result = await askQuestion(chatId, '도서관 이용 시간이 어떻게 되나요?');
  console.log('답변:', result.answer);
  console.log('관련 부서:', result.relevantDepartments);
};
```

## 에러 처리

모든 API는 에러 발생 시 다음과 같은 형식으로 응답합니다:

```json
{
  "success": false,
  "message": "에러 메시지"
}
```

HTTP 상태 코드:
- `200`: 성공
- `400`: 잘못된 요청 (필수 파라미터 누락 등)
- `500`: 서버 내부 오류

## 주의사항

1. **채팅 ID**: 모든 질문은 유효한 `chatId`가 필요합니다.
2. **세션 관리**: 서버는 메모리에서 채팅 세션을 관리합니다. 서버 재시작 시 세션이 초기화됩니다.
3. **동시 사용자**: 현재는 단일 사용자 환경을 가정하고 구현되어 있습니다.
4. **환경 변수**: OpenAI API 키 등 필요한 환경 변수가 설정되어 있어야 합니다.

## 개발 환경 설정

1. `.env` 파일에 OpenAI API 키 설정
2. `datas` 폴더에 필요한 문서 파일들이 있는지 확인
3. `python flask_app.py`로 서버 실행
4. `http://localhost:5000/health`로 서버 상태 확인
