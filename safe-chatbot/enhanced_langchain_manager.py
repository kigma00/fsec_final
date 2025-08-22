"""
Enhanced LangChain Manager with Policy Engine Integration
정책 엔진과 통합된 강화된 LangChain 매니저
"""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import re

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from policy_engine import PolicyEngine

class EnhancedLangChainManager:
    """정책 엔진과 통합된 LangChain 매니저"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 정책 엔진 초기화
        self.policy_engine = PolicyEngine()
        
        # LangChain 컴포넌트
        self.llm = None
        self.embeddings = None
        self.vectorstore = None
        self.conversation_chain = None
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # 문서 정보
        self.documents = []
        self.document_metadata = {}
        
        # 초기화
        self.initialize_components()
    
    def initialize_components(self):
        """LangChain 컴포넌트들을 초기화합니다."""
        try:
            # OpenAI API 키 확인
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                self.logger.warning("OPENAI_API_KEY가 설정되지 않았습니다. 모의 응답 모드를 사용합니다.")
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
            
            self.logger.info("Enhanced LangChain 컴포넌트 초기화 완료")
            
        except Exception as e:
            self.logger.error(f"Enhanced LangChain 초기화 중 오류 발생: {e}")
            self.llm = None
            self.embeddings = None
    
    def setup_vectorstore(self):
        """문서를 로드하고 벡터스토어를 설정합니다."""
        try:
            # 문서 디렉토리에서 파일들 로드
            docs_dir = "docs"
            if not os.path.exists(docs_dir):
                self.logger.warning(f"문서 디렉토리 {docs_dir}가 존재하지 않습니다.")
                return
            
            documents = []
            for filename in os.listdir(docs_dir):
                if filename.endswith(('.txt', '.md', '.pdf')):
                    filepath = os.path.join(docs_dir, filename)
                    try:
                        with open(filepath, 'r', encoding='utf-8') as f:
                            content = f.read()
                            
                            # 문서 ID 생성 (파일명 기반)
                            doc_id = self._generate_document_id(filename)
                            
                            documents.append({
                                'content': content,
                                'metadata': {
                                    'source': filename,
                                    'doc_id': doc_id,
                                    'type': self._get_document_type(filename),
                                    'size': len(content),
                                    'last_updated': datetime.now().isoformat()
                                }
                            })
                            
                            # 문서 메타데이터 저장
                            self.document_metadata[doc_id] = {
                                'filename': filename,
                                'type': self._get_document_type(filename),
                                'size': len(content)
                            }
                            
                    except Exception as e:
                        self.logger.error(f"문서 로드 실패: {filename} - {e}")
            
            if not documents:
                self.logger.warning("로드할 문서가 없습니다.")
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
                self.logger.info(f"벡터스토어 생성 완료: {len(texts)}개 청크")
            else:
                self.logger.warning("임베딩 모델이 없어 벡터스토어를 생성할 수 없습니다.")
                
        except Exception as e:
            self.logger.error(f"벡터스토어 설정 중 오류 발생: {e}")
    
    def setup_chains(self):
        """LangChain 체인들을 설정합니다."""
        try:
            if self.vectorstore and self.llm:
                # 대화형 체인
                self.conversation_chain = ConversationalRetrievalChain.from_llm(
                    llm=self.llm,
                    retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
                    memory=self.memory,
                    return_source_documents=True
                )
                
                self.logger.info("LangChain 체인 설정 완료")
            else:
                self.logger.warning("벡터스토어나 LLM이 없어 체인을 생성할 수 없습니다.")
                
        except Exception as e:
            self.logger.error(f"체인 설정 중 오류 발생: {e}")
    
    def get_response(self, message: str, user_id: str = "default") -> Dict[str, Any]:
        """
        사용자 메시지에 대한 응답을 생성합니다.
        정책 엔진과 통합되어 인용 강제 및 도메인 게이트를 적용합니다.
        """
        try:
            # 1. 정책 엔진을 통한 요청 검증
            policy_check = self.policy_engine.validate_request(message)
            
            if not policy_check["allowed"]:
                return {
                    "response": f"❌ **요청이 차단되었습니다**\n\n**사유:** {policy_check['reason']}\n\n🛡️ **보안 정책:** {policy_check['reason']}",
                    "sources": [],
                    "safety_score": policy_check["risk_score"],
                    "policy_violation": True,
                    "violation_type": "request_blocked",
                    "citation_required": policy_check["citations_required"]
                }
            
            # 2. LLM이 없으면 모의 응답 사용
            if not self.llm or not self.conversation_chain:
                return self._get_mock_response(message, policy_check)
            
            # 3. LangChain을 통한 응답 생성
            result = self.conversation_chain({"question": message})
            
            # 4. 소스 문서 추출
            sources = []
            if hasattr(result, 'source_documents'):
                for doc in result.source_documents:
                    if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                        sources.append(doc.metadata['source'])
            
            # 5. 정책 엔진을 통한 응답 검증
            response_validation = self.policy_engine.validate_response(
                result['answer'], 
                sources, 
                policy_check["citations_required"]
            )
            
            # 6. 최종 응답 구성
            final_response = response_validation["formatted_response"] if response_validation["allowed"] else response_validation["formatted_response"]
            
            return {
                "response": final_response,
                "sources": sources,
                "safety_score": 1.0 - policy_check["risk_score"],
                "policy_violation": not response_validation["allowed"],
                "violation_type": "response_blocked" if not response_validation["allowed"] else None,
                "citation_required": policy_check["citations_required"],
                "citation_score": response_validation.get("citation_score", 1.0),
                "policy_status": "PASS" if response_validation["allowed"] else "FAIL"
            }
            
        except Exception as e:
            self.logger.error(f"응답 생성 중 오류 발생: {e}")
            return {
                "response": "죄송합니다. 응답 생성 중 오류가 발생했습니다.",
                "sources": [],
                "safety_score": 0.0,
                "policy_violation": False,
                "error": str(e)
            }
    
    def _get_mock_response(self, message: str, policy_check: Dict) -> Dict[str, Any]:
        """API 키가 없을 때 사용하는 모의 응답을 생성합니다."""
        # 정책 위반 시 모의 응답도 차단
        if not policy_check["allowed"]:
            return {
                "response": f"❌ **요청이 차단되었습니다**\n\n**사유:** {policy_check['reason']}",
                "sources": [],
                "safety_score": policy_check["risk_score"],
                "policy_violation": True,
                "violation_type": "request_blocked",
                "citation_required": policy_check["citations_required"]
            }
        
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
                    "safety_score": 1.0,
                    "policy_violation": False,
                    "citation_required": policy_check["citations_required"]
                }
        
        # 기본 응답
        return {
            "response": "죄송합니다. 현재 OpenAI API 키가 설정되지 않아 제한된 응답만 가능합니다. 보안, 정책, 인증 등에 대해 질문해보세요.",
            "sources": ["모의 응답"],
            "safety_score": 1.0,
            "policy_violation": False,
            "citation_required": policy_check["citations_required"]
        }
    
    def _generate_document_id(self, filename: str) -> str:
        """문서 ID 생성"""
        # 파일명에서 확장자 제거
        name_without_ext = os.path.splitext(filename)[0]
        
        # 특수문자 제거 및 대문자 변환
        clean_name = re.sub(r'[^a-zA-Z0-9가-힣]', '-', name_without_ext)
        
        # 짧은 ID 생성 (예: KISA-SW-2021)
        if "isms" in filename.lower():
            return "ISMS-KISA-2024"
        elif "pci" in filename.lower():
            return "PCI-DSS-4.0"
        elif "security" in filename.lower():
            return "SEC-POL-2024"
        else:
            return clean_name.upper()[:15]
    
    def _get_document_type(self, filename: str) -> str:
        """문서 유형 판별"""
        filename_lower = filename.lower()
        
        if "isms" in filename_lower:
            return "ISMS 인증 기준"
        elif "pci" in filename_lower:
            return "PCI DSS 표준"
        elif "security" in filename_lower:
            return "보안 정책"
        else:
            return "일반 문서"
    
    def get_document_info(self) -> Dict[str, Any]:
        """문서 정보 반환"""
        return {
            "total_documents": len(self.documents),
            "document_types": list(set(doc.get('type', 'unknown') for doc in self.documents)),
            "document_metadata": self.document_metadata,
            "vectorstore_ready": self.vectorstore is not None,
            "llm_ready": self.llm is not None
        }
    
    def get_policy_summary(self) -> Dict[str, Any]:
        """정책 엔진 요약 정보 반환"""
        return self.policy_engine.get_audit_summary()
    
    def export_audit_log(self, format: str = "json") -> str:
        """감사 로그 내보내기"""
        return self.policy_engine.export_audit_log(format)
    
    def reset_policy_counters(self):
        """정책 카운터 초기화"""
        self.policy_engine.reset_counters()
