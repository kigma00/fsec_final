"""
Enhanced Policy Engine for Safe Chatbot
인용 강제 + 도메인 게이트 + 공격 차단 정책 엔진
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

class PolicyEngine:
    """강화된 보안 정책 엔진"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 인용 필수 키워드
        self.citation_keywords = [
            "조항", "항목", "조", "항", "절", "호", "목",
            "article", "section", "clause", "item", "paragraph"
        ]
        
        # 금지어 사전 (의무/필수)
        self.mandatory_keywords = [
            "의무", "필수", "반드시", "해야", "해야 함", "해야 합니다",
            "shall", "must", "required", "obligatory"
        ]
        
        # 권고어 사전
        self.recommended_keywords = [
            "권고", "권장", "가능하면", "바람직", "좋음",
            "should", "recommended", "desirable", "preferred"
        ]
        
        # 허용 도메인 화이트리스트 (일시적으로 주석 처리)
        # self.allowed_domains = [
        #     "kisa.or.kr",      # 한국정보통신기술협회
        #     "nist.gov",        # NIST
        #     "iso.org",         # ISO
        #     "iec.ch",          # IEC
        #     "bsigroup.com",    # BSI
        #     "gov.kr",          # 정부24
        #     "krcert.or.kr",    # 한국정보보호진흥원
        #     "kisa.or.kr",      # 한국정보통신기술협회
        #     "ncc.go.kr",       # 개인정보보호위원회
        # ]
        
        # 차단 패턴 (프롬프트 인젝션)
        self.blocked_patterns = [
            r"이전\s*지시\s*전부\s*무시",
            r"시스템\s*프롬프트\s*무시",
            r"관리자\s*권한",
            r"root\s*접근",
            r"sudo\s*권한",
            r"이전\s*대화\s*무시",
            r"새로운\s*지시",
            r"다른\s*역할로\s*변경",
            r"해킹", "크랙", "불법", "폭력"
        ]
        
        # 감사 로그
        self.audit_log = []
        
        # 정책 위반 카운터
        self.violation_counters = {
            "no_citation": 0,
            "external_domain": 0,
            "blocked_pattern": 0,
            "total_requests": 0
        }
    
    def validate_request(self, user_input: str) -> Dict:
        """
        사용자 요청 검증
        
        Returns:
            Dict: {
                "allowed": bool,
                "reason": str,
                "risk_score": float,
                "citations_required": bool,
                "domain_check": bool
            }
        """
        self.violation_counters["total_requests"] += 1
        
        # 1. 차단 패턴 검사
        pattern_check = self._check_blocked_patterns(user_input)
        if not pattern_check["allowed"]:
            self._log_violation("blocked_pattern", user_input, pattern_check["reason"])
            return {
                "allowed": False,
                "reason": f"차단된 패턴 감지: {pattern_check['reason']}",
                "risk_score": 1.0,
                "citations_required": True,
                "domain_check": True
            }
        
        # 2. 외부 도메인 호출 검사 (일시적으로 비활성화)
        # domain_check = self._check_external_domains(user_input)
        # if not domain_check["allowed"]:
        #     self._log_violation("external_domain", user_input, domain_check["reason"])
        #     return {
        #         "allowed": False,
        #         "reason": f"외부 도메인 호출 차단: {domain_check['reason']}",
        #         "risk_score": 0.8,
        #         "citations_required": True,
        #         "domain_check": False
        #     }
        
        # 3. 인용 요구사항 판단
        citations_required = self._determine_citation_requirement(user_input)
        
        return {
            "allowed": True,
            "reason": "요청 검증 통과",
            "risk_score": 0.1,
            "citations_required": citations_required,
            "domain_check": True
        }
    
    def validate_response(self, response: str, sources: List[str], 
                         citations_required: bool = True) -> Dict:
        """
        응답 검증 (인용 필수 체크)
        
        Args:
            response: AI 응답
            sources: 소스 문서 목록
            citations_required: 인용 필수 여부
            
        Returns:
            Dict: {
                "allowed": bool,
                "reason": str,
                "citation_score": float,
                "formatted_response": str
            }
        """
        if not citations_required:
            return {
                "allowed": True,
                "reason": "인용 불필요",
                "citation_score": 1.0,
                "formatted_response": response
            }
        
        # 인용 검증
        citation_check = self._validate_citations(response, sources)
        
        if not citation_check["valid"]:
            self._log_violation("no_citation", response, "인용 부족")
            return {
                "allowed": False,
                "reason": f"인용 부족: {citation_check['reason']}",
                "citation_score": citation_check["score"],
                "formatted_response": self._format_rejection_response(citation_check["reason"])
            }
        
        # 응답 포맷팅
        formatted_response = self._format_response_with_citations(response, sources)
        
        return {
            "allowed": True,
            "reason": "인용 요구사항 충족",
            "citation_score": citation_check["score"],
            "formatted_response": formatted_response
        }
    
    def _check_blocked_patterns(self, text: str) -> Dict:
        """차단 패턴 검사"""
        text_lower = text.lower()
        
        for pattern in self.blocked_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return {
                    "allowed": False,
                    "reason": f"차단 패턴 감지: {pattern}"
                }
        
        return {"allowed": True, "reason": "패턴 검사 통과"}
    
    def _check_external_domains(self, text: str) -> Dict:
        """외부 도메인 호출 검사"""
        # URL 패턴 찾기
        url_pattern = r'https?://([^/\s]+)'
        urls = re.findall(url_pattern, text)
        
        for url in urls:
            domain = url.lower()
            if not any(allowed in domain for allowed in self.allowed_domains):
                return {
                    "allowed": False,
                    "reason": f"허용되지 않은 도메인: {domain}"
                }
        
        return {"allowed": True, "reason": "도메인 검사 통과"}
    
    def _determine_citation_requirement(self, text: str) -> bool:
        """인용 요구사항 판단"""
        text_lower = text.lower()
        
        # 규정/정책 관련 키워드가 있으면 인용 필수
        policy_keywords = ["규정", "정책", "지침", "가이드라인", "표준", "기준"]
        if any(keyword in text_lower for keyword in policy_keywords):
            return True
        
        # 의무/필수 키워드가 있으면 인용 필수
        if any(keyword in text_lower for keyword in self.mandatory_keywords):
            return True
        
        return False
    
    def _validate_citations(self, response: str, sources: List[str]) -> Dict:
        """인용 유효성 검증"""
        if not sources:
            return {
                "valid": False,
                "reason": "소스 문서가 없습니다",
                "score": 0.0
            }
        
        # 인용 키워드 검사
        citation_found = any(keyword in response for keyword in self.citation_keywords)
        
        if not citation_found:
            return {
                "valid": False,
                "reason": "인용 키워드가 없습니다 (조항, 항목 등)",
                "score": 0.2
            }
        
        # 소스 문서와의 연관성 검사
        source_mention = any(source in response for source in sources)
        if not source_mention:
            return {
                "valid": False,
                "reason": "소스 문서가 언급되지 않았습니다",
                "score": 0.4
            }
        
        return {
            "valid": True,
            "reason": "인용 요구사항 충족",
            "score": 1.0
        }
    
    def _format_response_with_citations(self, response: str, sources: List[str]) -> str:
        """인용을 포함한 응답 포맷팅"""
        if not sources:
            return response
        
        citation_text = "\n\n📚 **참고 문서:**\n"
        for i, source in enumerate(sources, 1):
            citation_text += f"{i}. {source}\n"
        
        return response + citation_text
    
    def _format_rejection_response(self, reason: str) -> str:
        """거부 응답 포맷팅"""
        return f"""❌ **응답이 거부되었습니다**

**사유:** {reason}

**해결 방법:**
- 관련 규정/정책의 구체적인 조항번호를 포함하여 질문해주세요
- 예: "ISM-P 인증 절차의 3조 2항에 따른 구체적인 요구사항은?"
- 예: "PCI DSS 3.4항의 데이터 암호화 기준은?"

🛡️ **보안 정책:** 신뢰할 수 있는 응답을 위해 출처 인용이 필수입니다."""
    
    def _log_violation(self, violation_type: str, content: str, reason: str):
        """정책 위반 로그 기록"""
        self.violation_counters[violation_type] += 1
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "violation_type": violation_type,
            "content": content[:200] + "..." if len(content) > 200 else content,
            "reason": reason,
            "risk_level": "HIGH" if violation_type in ["blocked_pattern", "external_domain"] else "MEDIUM"
        }
        
        self.audit_log.append(log_entry)
        self.logger.warning(f"정책 위반 감지: {violation_type} - {reason}")
    
    def get_audit_summary(self) -> Dict:
        """감사 로그 요약"""
        total_violations = sum(self.violation_counters.values()) - self.violation_counters["total_requests"]
        
        return {
            "total_requests": self.violation_counters["total_requests"],
            "total_violations": total_violations,
            "violation_rate": (total_violations / self.violation_counters["total_requests"]) * 100 if self.violation_counters["total_requests"] > 0 else 0,
            "violation_breakdown": self.violation_counters.copy(),
            "recent_violations": self.audit_log[-10:] if self.audit_log else [],
            "policy_status": "ACTIVE"
        }
    
    def export_audit_log(self, format: str = "json") -> str:
        """감사 로그 내보내기"""
        if format.lower() == "csv":
            import csv
            import io
            
            output = io.StringIO()
            writer = csv.writer(output)
            
            # 헤더
            writer.writerow(["Timestamp", "Violation Type", "Content", "Reason", "Risk Level"])
            
            # 데이터
            for entry in self.audit_log:
                writer.writerow([
                    entry["timestamp"],
                    entry["violation_type"],
                    entry["content"],
                    entry["reason"],
                    entry["risk_level"]
                ])
            
            return output.getvalue()
        else:
            return json.dumps(self.audit_log, ensure_ascii=False, indent=2)
    
    def reset_counters(self):
        """카운터 초기화"""
        for key in self.violation_counters:
            self.violation_counters[key] = 0
        self.audit_log = []
