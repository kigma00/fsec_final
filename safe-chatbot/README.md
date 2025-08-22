# Safe Chatbot - LangChain Edition

LangChain을 기반으로 한 안전한 챗봇 웹 애플리케이션입니다.

## 🚀 주요 기능

### 🔍 ISMS-P 문서 분석
- **자동 문서 분석**: PDF, TXT, MD 파일 자동 처리
- **스마트 분할**: 대용량 문서 자동 섹션 분할 및 분석
- **기준 지침 통합**: KISA, PCI-DSS 등 표준 지침 + 기업 내부 지침
- **종합 평가**: 준수도 점수, 등급, 개선 권장사항 제공

### 💬 분석 결과 질문
- **자연어 질의**: 분석 결과에 대한 자연어 질문 지원
- **실시간 답변**: 점수, 약한 섹션, 누락된 내용 등 즉시 확인
- **대화형 인터페이스**: 직관적인 채팅 형태의 질의응답

### 🔍 JSON 결과 검증
- **결과 검증**: 업로드된 JSON 파일과 현재 분석 결과 비교
- **차이점 식별**: 준수도 점수, 등급, 섹션, 권장사항 차이 분석
- **데이터 무결성**: 분석 결과의 일관성 및 정확성 검증
- **실시간 비교**: 원본 JSON과 검증된 JSON 동시 표시

## 🏗️ 아키텍처

```
safe-chatbot/
├── app.py                      # 메인 Streamlit 애플리케이션
├── document_analyzer.py        # ISMS 문서 분석 엔진
├── feedback_verifier.py        # JSON 결과 검증 엔진
├── policy_engine.py            # 보안 정책 엔진
├── requirements.txt            # Python 의존성
├── README.md                   # 프로젝트 문서
├── .env                        # 환경 변수 (API 키 등)
├── data/                       # 기준 문서 폴더
│   ├── *.pdf                  # KISA, PCI-DSS 등 표준 지침
│   └── *.txt                  # 텍스트 기반 지침
└── docs/                       # 추가 문서
```

### 핵심 컴포넌트

- **DocumentAnalyzer**: ISMS 문서 분석 및 평가
- **FeedbackVerifier**: JSON 결과 검증 및 차이점 분석
- **PolicyEngine**: 보안 정책 및 규정 준수 검사
- **Streamlit UI**: 직관적인 웹 기반 사용자 인터페이스

## 📦 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 추가하세요:

```bash
# OpenAI API 설정
OPENAI_API_KEY=your_openai_api_key_here

# 모델 설정
MODEL_NAME=o4-mini

# 서버 설정
HOST=0.0.0.0
PORT=8000

# 로깅 레벨
LOG_LEVEL=INFO

# 보안 설정
MAX_TOKENS=4000
TEMPERATURE=0.7
SAFETY_THRESHOLD=0.5
```

### 3. 애플리케이션 실행

```bash
# Streamlit 앱 실행 (권장)
streamlit run app.py

# 또는 FastAPI 앱 실행
python langchain_app.py
```

## 🌐 웹 인터페이스

애플리케이션 실행 후 브라우저에서 자동으로 열리는 Streamlit 인터페이스를 사용할 수 있습니다.

### 주요 기능

- **💬 실시간 채팅**: LangChain 기반 AI 응답
- **📊 레드팀 테스트**: 보안 테스트 시나리오 실행 및 결과 분석
- **📚 문서 정보**: 문서 목록, 벡터스토어 상태, 검색 테스트
- **🔍 ISMS 문서 분석**: ISMS 준수성 분석 및 부족한 내용 식별
- **⚙️ 시스템 모니터링**: LLM, 벡터스토어 상태 실시간 확인

## 🔌 API 엔드포인트 (FastAPI)

### 1. 채팅 API

```http
POST /chat
Content-Type: application/json

{
  "message": "질문 내용",
  "user_id": "사용자_아이디",
  "conversation_id": "대화_아이디"
}
```

### 2. 레드팀 테스트 API

```http
POST /redteam
Content-Type: application/json

{
  "test_scenarios": [
    {
      "id": "RT-01",
      "prompt": "테스트 프롬프트"
    }
  ]
}
```

### 3. 시스템 상태 확인

```http
GET /health
```

## 🔧 LangChain 컴포넌트

### LangChainManager

- **LLM 초기화**: OpenAI GPT 모델 설정
- **임베딩 모델**: OpenAI 임베딩을 사용한 벡터화
- **벡터스토어**: FAISS를 사용한 문서 검색
- **체인 관리**: RAG 및 대화형 체인 설정

### 주요 체인

1. **RetrievalQA Chain**: 문서 기반 질의응답
2. **ConversationalRetrievalChain**: 대화 컨텍스트를 고려한 응답
3. **Memory Management**: 대화 히스토리 유지

## 📚 문서 처리

### 지원 형식

- `.txt`: 일반 텍스트 파일
- `.md`: Markdown 파일
- `.pdf`: PDF 파일 (PyPDF2 사용)

### 처리 과정

1. **문서 로딩**: 지정된 디렉토리에서 파일 읽기
2. **텍스트 분할**: RecursiveCharacterTextSplitter로 청크 생성
3. **벡터화**: OpenAI 임베딩으로 텍스트를 벡터로 변환
4. **인덱싱**: FAISS 벡터스토어에 저장

## 🛡️ 보안 기능

### 안전성 검사

- **키워드 필터링**: 유해한 키워드 감지
- **안전성 점수**: 0.0 ~ 1.0 범위의 점수 계산
- **응답 차단**: 낮은 안전성 점수의 응답 차단

### 보안 정책

- 입력 메시지 검증
- 응답 내용 필터링
- API 요청 제한 (구현 예정)

## 🧪 테스트

### 레드팀 테스트

```bash
# Streamlit 인터페이스에서 직접 실행
# 사이드바의 레드팀 테스트 섹션 사용
```

## 🚀 확장 가능성

### 추가 기능

1. **에이전트 시스템**: LangChain 에이전트를 사용한 복잡한 작업 처리
2. **멀티모달**: 이미지, 오디오 등 다양한 입력 처리
3. **실시간 스트리밍**: 응답 스트리밍 지원
4. **사용자 인증**: JWT 기반 인증 시스템
5. **데이터베이스 연동**: PostgreSQL, MongoDB 등 연동

### 성능 최적화

1. **캐싱**: Redis를 사용한 응답 캐싱
2. **로드 밸런싱**: 여러 인스턴스 간 부하 분산
3. **비동 처리**: Celery를 사용한 백그라운드 작업

## 📝 개발 가이드

### 코드 구조

```
app.py                    # Streamlit 메인 애플리케이션
langchain_app.py          # FastAPI 백엔드 서버
├── LangChainManager      # LangChain 컴포넌트 관리
├── API 엔드포인트        # FastAPI 라우터
└── 웹 인터페이스         # HTML/CSS/JavaScript

docs/                     # 문서 저장소
requirements.txt           # 의존성 목록
README.md                 # 프로젝트 문서
```

## 🔍 문제 해결

### 일반적인 문제

1. **API 키 오류**: `.env` 파일에 올바른 API 키 설정 확인
2. **메모리 부족**: FAISS 인덱스 크기 조정
3. **응답 지연**: LLM 모델 변경 또는 캐싱 구현

### 로그 확인

```bash
# 로그 레벨 설정
export LOG_LEVEL=DEBUG
streamlit run app.py
```

## 📄 라이선스

MIT License

## 🤝 기여

버그 리포트, 기능 제안, 풀 리퀘스트를 환영합니다!

## 📖 사용법

### 1. ISMS-P 문서 분석
1. **기준 문서 설정**: `data/` 폴더에 표준 지침 PDF 파일들을 배치
2. **기업 지침 추가** (선택사항): 기업 내부 ISMS 지침 파일 업로드
3. **분석할 문서 업로드**: ISMS 문서 파일(.txt, .md, .pdf) 업로드
4. **분석 실행**: "🔍 문서 분석 실행" 버튼 클릭
5. **결과 확인**: 준수도 점수, 등급, 개선사항 등 종합 분석 결과 확인

### 2. 분석 결과 질문
1. **질문 입력**: "💬 분석 결과 질문" 탭에서 자연어로 질문
2. **실시간 답변**: 점수, 약한 섹션, 누락된 내용 등에 대한 즉시 답변
3. **대화형 인터페이스**: 연속적인 질의응답으로 깊이 있는 분석

### 3. JSON 결과 검증
1. **JSON 파일 업로드**: 검증할 ISMS-P 분석 결과 JSON 파일 업로드
2. **검증 실행**: "🔍 JSON 결과 검증 실행" 버튼 클릭
3. **차이점 분석**: 현재 분석 결과와 업로드된 JSON 간의 차이점 식별
4. **결과 비교**: 원본 JSON과 검증된 JSON을 나란히 비교하여 확인
5. **데이터 무결성**: 분석 결과의 일관성 및 정확성 검증
