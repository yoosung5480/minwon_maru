
# 민원마루 (Minwon Maru)

**지자체 행정 민원 AI 챗봇 서비스**

---

## 📌 프로젝트 개요
현재 행정동 홈페이지의 챗봇은 단순한 감정 "응대" 역할과 부서 연결 기능이 부족합니다.  
**민원마루**는 파인튜닝 없이 **RAG(Retrieval-Augmented Generation)** 기반으로 행정 문서를 활용하여,  
민원인에게 **정확하고 신뢰할 수 있는 답변**을 제공하고, 부서에는 민원 상담 로그를 정리하여 기록하는 **AI 민원 상담 챗봇 서비스**입니다.  

이를 통해:
- 지자체 행정부서의 감정 노동을 줄이고,  
- 간단한 정보성 민원은 24시간 자동으로 처리하며,  
- 복잡한 민원은 적합한 부서로 연결할 수 있습니다.

---

## 🎯 주요 기능

### 상담 챗봇 (민원인 응대)
1. 민원인의 채팅 내용에서 개인정보를 **자동 마스킹**  
2. `metadata.json` 기반 문서 요약본에서 **유사도 검색**  
3. 관련 문서를 기반으로 **RAG 체인 실행**  
4. 참조 문서를 `recent_docs` 메모리에 관리 (Queue 방식)  
5. 다음 질문 시 `recent_docs` → 전체 문서 순으로 검색  
6. 상담 종료 시, 채팅 로그와 유사도가 가장 높은 부서로 자동 연결  
   - 유사도 점수가 임계값 이하일 경우 전송하지 않음  

### 업무 어시스턴트 (부서 담당자 지원)
- 부서별로 접수된 민원 로그를 **기록/분류/조회** 가능  
- 향후 **부서 협업 챗봇**으로 확장 예정  

---

## 📂 프로젝트 구조

```

root/
├── cook\_books/                 # 참고용 코드 스니펫
├── minwon\_maru/                # 핵심 라이브러리
│   ├── engine/
│   │   ├── crag\_chain.py       # ★ 핵심 채팅 기능
│   │   ├── minwon\_maru\_counselor.py  # 민원 상담 챗봇
│   │   ├── minwon\_maru\_assistant.py  # 부서 협력 챗봇 (개발 예정)
│   │   └── chain.py            # 사용 안함
│   ├── init/
│   │   ├── load\_docs.py
│   │   ├── make\_metadata.py
│   │   └── check\_consistency.py
│   ├── prompts/
│   │   └── prompt.py
│   ├── tools/
│   │   ├── context.py          # 핵심 검색/컨텍스트 관리
│   │   ├── myPDFparser.py
│   │   ├── json\_tool.py
│   │   ├── llms.py
│   │   └── personal\_info\_keeper.py   # 개인정보 마스킹
│   └── workflows/
│       └── basicCrag.py        # 구현됨, 미사용
├── main.py                     # ★ 실제 웹소켓 서버 실행 진입점
├── app.py                      # Flask 실행용
├── test.ipynb                  # 테스트 노트북 (출력/코드 참고용)
├── datas/
│   ├── reference/
│   │   ├── work\_page\_info.json
│   │   └── department\_info.json
│   ├── metadata.json
│   ├── rawDocs/                # 원본 PDF 문서
│   └── parsedDocs/             # JSON 변환된 문서
└── outputs/
└── {날짜}/
├── 부서별민원현황/
└── 채팅로그/

````

---

## ⚙️ 개발 환경

- Python 3.10+
- Conda 가상환경 사용 권장
- 필수 라이브러리: `requirements.txt` 참고  

```bash
# 가상환경 생성
conda create -n minwon_maru python=3.10
conda activate minwon_maru

# 라이브러리 설치
pip install -r requirements.txt
````

---

## 🚀 실행 방법

1. **데이터 경로 설정**

   * `main.py`에서 `data_root_str` 값을 프로젝트 경로에 맞게 수정

2. **프로젝트 루트 이동**

   ```bash
   cd root
   ```

3. **서버 실행**

   ```bash
   python main.py
   ```

---

## 💬 사용 예시

### 사용자 질문 → 응답 예시

```python
from minwon_maru.engine.chat import ChatManager, ProjPath

paths = ProjPath(
    data_root_str="./datas",
    metadata_name="metadata.json",
    raw_docs_dir="rawDocs",
    parsed_docs_dir="parsedDocs",
    workpages_name="work_page_info.json",
    reference_root_str="reference",
    departemt_info_name="department_info.json"
)

manager = ChatManager(paths)
chat_id = manager.start_chat()

sample_q = "금정도서관 주차장 이용은 공짜야? 다음주 월요일은 몇시까지 운영해?"
response = manager.ask(sample_q)

print("질문:", sample_q)
print("응답:", response["generation"])
```

출력 예시:

```
질문: 금정도서관 주차장 이용은 공짜야? 다음주 월요일은 몇시까지 운영해?

응답: 
안녕하세요. 금정도서관 주차장 이용 안내드리겠습니다.
- 최초 30분은 무료, 이후 10분당 100원 부과 (1일 최대 4,700원)
- 월요일은 휴관일로 주차장 이용 불가
```

---

## 📌 향후 계획

* 부서별 민원 업무 자동 분류 및 통계 기능 강화
* 실시간 **WebSocket 스트리밍 응답** 지원
* 부서 협업 어시스턴트 챗봇 개발
* 배포용 Docker 환경 제공
* 행정 서비스 연계 API 확장 (정부24, 국민신문고 등)

---
