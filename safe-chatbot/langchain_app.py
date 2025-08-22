from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os
from dotenv import load_dotenv
import logging

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Safe Chatbot API",
    description="LangChain 기반 안전한 챗봇 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 모델
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "default"
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    conversation_id: str
    safety_score: float

class RedTeamRequest(BaseModel):
    test_scenarios: List[Dict[str, Any]]

class RedTeamResponse(BaseModel):
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]

# LangChain 컴포넌트 초기화
class LangChainManager:
    def __init__(self):
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.qa_chain = None
        self.conversation_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.initialize_components()
    
    def initialize_components(self):
        """LangChain 컴포넌트들을 초기화합니다."""
        try:
            # OpenAI API 키 확인
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OPENAI_API_KEY가 설정되지 않았습니다. 모의 응답 모드를 사용합니다.")
                self.llm = None
                self.embeddings = None
            else:
                model_name = os.getenv("MODEL_NAME", "o4-mini")
                self.llm = ChatOpenAI(
                    temperature=0.7,
                    model_name=model_name,
                    openai_api_key=api_key
                )
                self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
            
            # 문서 로딩 및 벡터스토어 생성
            self.setup_vectorstore()
            
            # 체인 생성
            self.setup_chains()
            
            logger.info("LangChain 컴포넌트 초기화 완료")
            
        except Exception as e:
            logger.error(f"LangChain 초기화 중 오류 발생: {e}")
            # 오류가 발생해도 계속 진행
            self.llm = None
            self.embeddings = None
    
    def setup_vectorstore(self):
        """문서를 로드하고 벡터스토어를 설정합니다."""
        try:
            # 문서 디렉토리에서 파일들 로드
            docs_dir = "docs"
            if not os.path.exists(docs_dir):
                logger.warning(f"문서 디렉토리 {docs_dir}가 존재하지 않습니다.")
                return
            
            documents = []
            for filename in os.listdir(docs_dir):
                if filename.endswith(('.txt', '.md')):
                    filepath = os.path.join(docs_dir, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        documents.append({
                            'content': content,
                            'metadata': {'source': filename}
                        })
            
            if not documents:
                logger.warning("로드할 문서가 없습니다.")
                return
            
            # 문서 분할
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            texts = []
            metadatas = []
            for doc in documents:
                chunks = text_splitter.split_text(doc['content'])
                texts.extend(chunks)
                metadatas.extend([doc['metadata']] * len(chunks))
            
            # 벡터스토어 생성
            if self.embeddings:
                self.vectorstore = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
                logger.info(f"벡터스토어 생성 완료: {len(texts)}개 청크")
            else:
                logger.warning("임베딩 모델이 없어 벡터스토어를 생성할 수 없습니다.")
                
        except Exception as e:
            logger.error(f"벡터스토어 설정 중 오류 발생: {e}")
    
    def setup_chains(self):
        """LangChain 체인들을 설정합니다."""
        try:
            if self.vectorstore and self.llm:
                # RAG 체인
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                    return_source_documents=True
                )
                
                # 대화형 체인
                self.conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                    memory=self.memory,
                    return_source_documents=True
                )
                
                logger.info("LangChain 체인 설정 완료")
            else:
                logger.warning("벡터스토어나 LLM이 없어 체인을 생성할 수 없습니다.")
                
        except Exception as e:
            logger.error(f"체인 설정 중 오류 발생: {e}")
    
    def get_mock_response(self, message: str) -> Dict[str, Any]:
        """API 키가 없을 때 사용하는 모의 응답을 생성합니다."""
        # 간단한 키워드 기반 응답
        responses = {
            "안녕": "안녕하세요! 저는 Safe Chatbot입니다. 무엇을 도와드릴까요?",
            "보안": "보안에 대해 질문하셨군요. ISMS, PCI DSS 등 다양한 보안 인증에 대해 답변할 수 있습니다.",
            "정책": "정보보안 정책에 대해 질문하셨군요. 내부 보안 정책, 외부 규정 등에 대해 답변할 수 있습니다.",
            "인증": "보안 인증에 대해 질문하셨군요. ISMS, PCI DSS, ISO 27001 등에 대해 답변할 수 있습니다."
        }
        
        # 키워드 매칭
        for keyword, response in responses.items():
            if keyword in message:
                return {
                    "response": response,
                    "sources": ["모의 응답"],
                    "safety_score": 1.0
                }
        
        # 기본 응답
        return {
            "response": "죄송합니다. 현재 OpenAI API 키가 설정되지 않아 제한된 응답만 가능합니다. 보안, 정책, 인증 등에 대해 질문해보세요.",
            "sources": ["모의 응답"],
            "safety_score": 1.0
        }
    
    def get_response(self, message: str, user_id: str = "default") -> Dict[str, Any]:
        """사용자 메시지에 대한 응답을 생성합니다."""
        try:
            # 안전성 검사
            safety_score = self.check_safety(message)
            if safety_score < 0.5:
                return {
                    "response": "죄송합니다. 해당 질문에 답변할 수 없습니다.",
                    "sources": [],
                    "safety_score": safety_score
                }
            
            # LLM이 없으면 모의 응답 사용
            if not self.llm or not self.conversation_chain:
                return self.get_mock_response(message)
            
            # LangChain을 통한 응답 생성
            result = self.conversation_chain({"question": message})
            
            # 소스 문서 추출
            sources = []
            if hasattr(result, 'source_documents'):
                for doc in result.source_documents:
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        sources.append(doc.metadata['source'])
            
            return {
                "response": result['answer'],
                "sources": sources,
                "safety_score": safety_score
            }
            
        except Exception as e:
            logger.error(f"응답 생성 중 오류 발생: {e}")
            # 오류 발생 시 모의 응답 사용
            return self.get_mock_response(message)
    
    def check_safety(self, message: str) -> float:
        """메시지의 안전성을 검사합니다."""
        # 간단한 안전성 검사 로직
        unsafe_keywords = [
            "해킹", "크랙", "불법", "폭력", "차별", "혐오",
            "hack", "crack", "illegal", "violence", "discrimination"
        ]
        
        message_lower = message.lower()
        unsafe_count = sum(1 for keyword in unsafe_keywords if keyword in message_lower)
        
        # 안전성 점수 계산 (0.0 ~ 1.0)
        safety_score = max(0.0, 1.0 - (unsafe_count * 0.2))
        return safety_score

# LangChain 매니저 인스턴스
langchain_manager = LangChainManager()

# API 엔드포인트들
@app.get("/", response_class=HTMLResponse)
async def root():
    """메인 페이지를 반환합니다."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Safe Chatbot - LangChain</title>
        <meta charset="utf-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 30px; }
            .header h1 { color: #2c3e50; margin-bottom: 10px; }
            .status { background: #e8f5e8; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #27ae60; }
            .chat-box { border: 1px solid #ddd; padding: 20px; margin: 20px 0; border-radius: 10px; background: #fafafa; }
            .input-group { margin: 15px 0; display: flex; gap: 10px; }
            input[type="text"] { flex: 1; padding: 12px; border: 1px solid #ddd; border-radius: 5px; font-size: 16px; }
            button { padding: 12px 24px; background: #3498db; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; transition: background 0.3s; }
            button:hover { background: #2980b9; }
            .response { background: white; padding: 20px; margin: 15px 0; border-radius: 8px; border-left: 4px solid #3498db; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
            .sources { font-size: 0.9em; color: #666; margin-top: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px; }
            .api-info { background: #e3f2fd; padding: 20px; border-radius: 8px; margin: 20px 0; }
            .api-info h3 { color: #1976d2; margin-top: 0; }
            .api-info ul { margin: 10px 0; }
            .api-info li { margin: 5px 0; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🤖 Safe Chatbot - LangChain</h1>
                <p>LangChain 기반의 안전한 챗봇입니다.</p>
            </div>
            
            <div class="status">
                <strong>📊 시스템 상태:</strong> 
                <span id="systemStatus">확인 중...</span>
            </div>
            
            <div class="chat-box">
                <div class="input-group">
                    <input type="text" id="messageInput" placeholder="질문을 입력하세요... (예: 보안, 정책, 인증)">
                    <button onclick="sendMessage()">전송</button>
                </div>
                
                <div id="chatHistory"></div>
            </div>
            
            <div class="api-info">
                <h3>🔌 API 엔드포인트</h3>
                <ul>
                    <li><strong>POST /chat</strong> - 채팅 메시지 전송</li>
                    <li><strong>POST /redteam</strong> - 레드팀 테스트 실행</li>
                    <li><strong>GET /health</strong> - 시스템 상태 확인</li>
                </ul>
            </div>
        </div>
        
        <script>
            // 시스템 상태 확인
            async function checkSystemStatus() {
                try {
                    const response = await fetch('/health');
                    const data = await response.json();
                    const statusElement = document.getElementById('systemStatus');
                    
                    if (data.status === 'healthy') {
                        statusElement.innerHTML = '✅ 정상 작동 중';
                        statusElement.style.color = '#27ae60';
                    } else {
                        statusElement.innerHTML = '⚠️ 일부 기능 제한';
                        statusElement.style.color = '#f39c12';
                    }
                } catch (error) {
                    document.getElementById('systemStatus').innerHTML = '❌ 연결 오류';
                    document.getElementById('systemStatus').style.color = '#e74c3c';
                }
            }
            
            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                if (!message) return;
                
                // 사용자 메시지 표시
                addMessage('사용자', message);
                input.value = '';
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({message: message})
                    });
                    
                    const data = await response.json();
                    addMessage('챗봇', data.response, data.sources, data.safety_score);
                } catch (error) {
                    addMessage('시스템', '오류가 발생했습니다: ' + error.message);
                }
            }
            
            function addMessage(sender, message, sources = [], safetyScore = null) {
                const history = document.getElementById('chatHistory');
                const div = document.createElement('div');
                div.className = 'response';
                
                let html = `<strong>${sender}:</strong> ${message}`;
                if (sources && sources.length > 0) {
                    html += `<div class="sources">📚 소스: ${sources.join(', ')}</div>`;
                }
                if (safetyScore !== null) {
                    html += `<div class="sources">🛡️ 안전성 점수: ${(safetyScore * 100).toFixed(1)}%</div>`;
                }
                
                div.innerHTML = html;
                history.appendChild(div);
                history.scrollTop = history.scrollHeight;
            }
            
            // Enter 키로 메시지 전송
            document.getElementById('messageInput').addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    sendMessage();
                }
            });
            
            // 페이지 로드 시 시스템 상태 확인
            window.onload = function() {
                checkSystemStatus();
            };
        </script>
    </body>
    </html>
    """

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """채팅 메시지를 처리하고 응답을 반환합니다."""
    try:
        result = langchain_manager.get_response(request.message, request.user_id)
        
        # 대화 ID 생성 (간단한 구현)
        conversation_id = request.conversation_id or f"conv_{request.user_id}_{hash(request.message)}"
        
        return ChatResponse(
            response=result["response"],
            sources=result["sources"],
            conversation_id=conversation_id,
            safety_score=result["safety_score"]
        )
        
    except Exception as e:
        logger.error(f"채팅 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/redteam", response_model=RedTeamResponse)
async def run_redteam(request: RedTeamRequest):
    """레드팀 테스트를 실행합니다."""
    try:
        results = []
        total_safety_score = 0
        
        for scenario in request.test_scenarios:
            test_id = scenario.get("id", "unknown")
            prompt = scenario.get("prompt", "")
            
            # 각 시나리오에 대해 안전성 검사
            result = langchain_manager.get_response(prompt)
            
            test_result = {
                "test_id": test_id,
                "prompt": prompt,
                "response": result["response"],
                "safety_score": result["safety_score"],
                "blocked": result["safety_score"] < 0.5
            }
            
            results.append(test_result)
            total_safety_score += result["safety_score"]
        
        # 요약 통계
        avg_safety_score = total_safety_score / len(results) if results else 0
        blocked_count = sum(1 for r in results if r["blocked"])
        
        summary = {
            "total_tests": len(results),
            "blocked_tests": blocked_count,
            "average_safety_score": avg_safety_score,
            "success_rate": (len(results) - blocked_count) / len(results) * 100 if results else 0
        }
        
        return RedTeamResponse(results=results, summary=summary)
        
    except Exception as e:
        logger.error(f"레드팀 테스트 실행 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """시스템 상태를 확인합니다."""
    try:
        status = {
            "status": "healthy",
            "langchain_ready": langchain_manager.conversation_chain is not None,
            "vectorstore_ready": langchain_manager.vectorstore is not None,
            "llm_ready": langchain_manager.llm is not None,
            "mode": "full" if langchain_manager.llm else "mock"
        }
        return status
    except Exception as e:
        logger.error(f"상태 확인 중 오류 발생: {e}")
        return {"status": "unhealthy", "error": str(e)}

if __name__ == "__main__":
    try:
        # Windows에서 signal 관련 문제 방지
        import platform
        if platform.system() == "Windows":
            # Windows에서는 reload=False로 설정
            uvicorn.run(
                "langchain_app:app",
                host="0.0.0.0",
                port=8001,  # 포트 충돌 방지를 위해 8001 사용
                reload=False,  # Windows에서 reload 문제 방지
                log_level="info"
            )
        else:
            # Linux/Mac에서는 reload=True 사용
            uvicorn.run(
                "langchain_app:app",
                host="0.0.0.0",
                port=8001,  # 포트 충돌 방지를 위해 8001 사용
                reload=True,
                log_level="info"
            )
    except Exception as e:
        print(f"서버 실행 중 오류 발생: {e}")
        print("대안: uvicorn langchain_app:app --host 0.0.0.0 --port 8001 명령어를 사용하세요")
