"""
Document Analyzer for ISMS Compliance
ISMS 문서 분석 및 부족한 내용 식별
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import re

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("PyPDF2가 설치되지 않았습니다. PDF 처리가 불가능합니다.")

class DocumentAnalyzer:
    """ISMS 문서 분석기"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # ISMS 필수 항목 체크리스트
        self.isms_requirements = {
            "정보보호정책": {
                "description": "조직의 정보보호 방침과 목표",
                "required_sections": [
                    "정책 목적", "적용 범위", "책임과 권한", "정책 검토 및 개정 절차"
                ],
                "keywords": ["정보보호", "보안정책", "방침", "목표", "책임", "권한"]
            },
            "정보보호조직": {
                "description": "정보보호 업무를 담당하는 조직 구성",
                "required_sections": [
                    "조직도", "역할과 책임", "담당자 지정", "교육 계획"
                ],
                "keywords": ["조직", "역할", "책임", "담당자", "교육", "훈련"]
            },
            "인적보안": {
                "description": "직원의 정보보안 의식과 행동 규범",
                "required_sections": [
                    "채용 시 보안", "재직 중 보안", "퇴직 시 보안", "보안 서약서"
                ],
                "keywords": ["인적보안", "채용", "재직", "퇴직", "서약서", "교육"]
            },
            "물리적보안": {
                "description": "물리적 접근 통제 및 환경 보호",
                "required_sections": [
                    "접근 통제", "환경 보호", "자산 보호", "운반 및 폐기"
                ],
                "keywords": ["물리적보안", "접근통제", "환경보호", "자산", "운반", "폐기"]
            },
            "기술적보안": {
                "description": "기술적 보안 대책 및 접근 통제",
                "required_sections": [
                    "접근 통제", "암호화", "네트워크 보안", "악성코드 방지"
                ],
                "keywords": ["기술적보안", "접근통제", "암호화", "네트워크", "악성코드", "방화벽"]
            },
            "업무연속성": {
                "description": "재해 및 장애 시 업무 연속성 보장",
                "required_sections": [
                    "재해 복구 계획", "업무 연속성 계획", "정기 점검", "훈련"
                ],
                "keywords": ["업무연속성", "재해복구", "백업", "복구", "점검", "훈련"]
            },
            "법적준수": {
                "description": "관련 법령 및 규정 준수",
                "required_sections": [
                    "개인정보보호", "전자금융거래", "정보통신망", "준수 점검"
                ],
                "keywords": ["법적준수", "개인정보", "전자금융", "정보통신망", "준수", "점검"]
            }
        }
        
        # 기준 문서들
        self.reference_documents = {}
        
        # 분석 결과
        self.analysis_results = {}
        
        # 기업 지침 (기본값은 None)
        self.enterprise_guideline = None
        self.enterprise_guideline_content = None
    
    def load_reference_documents(self, data_dir: str = "data") -> Dict[str, Any]:
        """기준 문서들을 로드합니다."""
        if not os.path.exists(data_dir):
            self.logger.warning(f"데이터 디렉토리 {data_dir}가 존재하지 않습니다.")
            return {}
        
        documents = {}
        
        for filename in os.listdir(data_dir):
            if filename.endswith('.pdf'):
                filepath = os.path.join(data_dir, filename)
                try:
                    content = self._extract_pdf_text(filepath)
                    if content:
                        documents[filename] = {
                            'content': content,
                            'size': len(content),
                            'type': self._get_document_type(filename),
                            'extracted_at': datetime.now().isoformat()
                        }
                        self.logger.info(f"문서 로드 완료: {filename} ({len(content)} 문자)")
                except Exception as e:
                    self.logger.error(f"문서 로드 실패: {filename} - {e}")
        
        self.reference_documents = documents
        return documents
    
    def set_enterprise_guideline(self, guideline_content: str):
        """기업 내부 지침을 설정합니다."""
        self.enterprise_guideline = guideline_content
        self.enterprise_guideline_content = guideline_content
        
        # 기업 지침에서 키워드 추출
        extracted_keywords = self._extract_keywords_from_content(guideline_content)
        
        # 기업 지침에서 필수 하위섹션 추출
        extracted_subsections = self._extract_required_subsections(guideline_content)
        
        # 기존 요구사항에 기업 지침 내용 추가
        for section, requirements in self.isms_requirements.items():
            if section in extracted_subsections:
                # 기업 지침의 하위섹션을 기존 요구사항에 추가
                self.isms_requirements[section]["subsections"].extend(extracted_subsections[section])
                # 중복 제거
                self.isms_requirements[section]["subsections"] = list(set(self.isms_requirements[section]["subsections"]))
        
        # 기업 지침 키워드를 기존 키워드에 추가
        for keyword_type, keywords in extracted_keywords.items():
            if keyword_type in self.isms_requirements:
                self.isms_requirements[keyword_type]["keywords"].extend(keywords)
                # 중복 제거
                self.isms_requirements[keyword_type]["keywords"] = list(set(self.isms_requirements[keyword_type]["keywords"]))
    
    def feedback_verifier(self, json_data: dict) -> dict:
        """업로드된 JSON 데이터와 현재 분석 결과를 비교하여 검증합니다."""
        try:
            differences = {}
            verified_json = {}
            
            # 현재 분석 결과와 업로드된 JSON 비교
            if hasattr(self, 'last_analysis_result'):
                current_result = self.last_analysis_result
                
                # 준수도 점수 비교
                if 'document_analysis' in current_result and 'document_analysis' in json_data:
                    current_score = current_result['document_analysis'].get('compliance_score', 0)
                    json_score = json_data['document_analysis'].get('compliance_score', 0)
                    if abs(current_score - json_score) > 0.1:  # 0.1% 이상 차이
                        differences['compliance_score'] = f"현재: {current_score}%, JSON: {json_score}%"
                
                # 전체 평가 비교
                if 'overall_assessment' in current_result and 'overall_assessment' in json_data:
                    current_grade = current_result['overall_assessment'].get('grade', '')
                    json_grade = json_data['overall_assessment'].get('grade', '')
                    if current_grade != json_grade:
                        differences['overall_grade'] = f"현재: {current_grade}, JSON: {json_grade}"
                
                # 섹션 분석 비교
                if 'section_analysis' in current_result and 'section_analysis' in json_data:
                    current_sections = set(current_result['section_analysis'].keys())
                    json_sections = set(json_data['section_analysis'].keys())
                    
                    # 누락된 섹션
                    missing_in_json = current_sections - json_sections
                    if missing_in_json:
                        differences['missing_sections'] = f"JSON에 누락: {', '.join(missing_in_json)}"
                    
                    # 추가된 섹션
                    extra_in_json = json_sections - current_sections
                    if extra_in_json:
                        differences['extra_sections'] = f"JSON에 추가: {', '.join(extra_in_json)}"
                
                # 권장사항 비교
                if 'document_analysis' in current_result and 'document_analysis' in json_data:
                    current_recs = set(current_result['document_analysis'].get('recommendations', []))
                    json_recs = set(json_data['document_analysis'].get('recommendations', []))
                    
                    if current_recs != json_recs:
                        differences['recommendations'] = f"권장사항 차이: {len(current_recs)} vs {len(json_recs)}"
            
            # 검증 결과 생성
            is_valid = len(differences) == 0
            description = "모든 내용이 일치합니다." if is_valid else f"{len(differences)}개 항목에서 차이점이 발견되었습니다."
            
            # 검증된 JSON 생성 (현재 분석 결과 기반)
            if hasattr(self, 'last_analysis_result'):
                verified_json = self.last_analysis_result.copy()
            else:
                verified_json = {"error": "현재 분석 결과가 없습니다."}
            
            return {
                "is_valid": is_valid,
                "differences": differences,
                "description": description,
                "verified_json": verified_json,
                "verification_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "is_valid": False,
                "differences": {"error": str(e)},
                "description": f"검증 중 오류 발생: {e}",
                "verified_json": {},
                "verification_timestamp": datetime.now().isoformat()
            }
    
    def _extract_enterprise_requirements(self, guideline_content: str) -> None:
        """기업 지침에서 ISMS 요구사항을 추출하여 업데이트합니다."""
        # 기업 지침 내용을 분석하여 새로운 요구사항 구조 생성
        enterprise_requirements = {}
        
        # 기업 지침에서 섹션별로 분석
        sections = self._parse_enterprise_guideline(guideline_content)
        
        for section_name, section_content in sections.items():
            # 키워드 추출
            keywords = self._extract_keywords_from_content(section_content)
            
            # 필수 하위 섹션 추출
            required_subsections = self._extract_required_subsections(section_content)
            
            enterprise_requirements[section_name] = {
                "description": f"기업 내부 지침: {section_name}",
                "required_sections": required_subsections,
                "keywords": keywords,
                "source": "enterprise_guideline"
            }
        
        # 기존 요구사항과 병합 (기업 지침 우선)
        if enterprise_requirements:
            self.isms_requirements.update(enterprise_requirements)
            self.logger.info(f"기업 지침에서 {len(enterprise_requirements)}개 섹션 추출")
    
    def _parse_enterprise_guideline(self, content: str) -> Dict[str, str]:
        """기업 지침을 섹션별로 파싱합니다."""
        sections = {}
        
        # 일반적인 섹션 구분자들
        section_patterns = [
            r'^(\d+\.\s*[^\n]+)',  # 1. 제목
            r'^([가-힣\s]+)\s*$',   # 한글 제목
            r'^([A-Z][^\n]+)',      # 영문 제목
        ]
        
        lines = content.split('\n')
        current_section = "일반"
        current_content = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 새로운 섹션인지 확인
            is_new_section = False
            for pattern in section_patterns:
                if re.match(pattern, line):
                    # 이전 섹션 저장
                    if current_content:
                        sections[current_section] = '\n'.join(current_content)
                    
                    # 새 섹션 시작
                    current_section = line
                    current_content = []
                    is_new_section = True
                    break
            
            if not is_new_section:
                current_content.append(line)
        
        # 마지막 섹션 저장
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _extract_keywords_from_content(self, content: str) -> List[str]:
        """컨텐츠에서 키워드를 추출합니다."""
        # 한국어 키워드 추출
        korean_keywords = re.findall(r'[가-힣]{2,}', content)
        
        # 영문 키워드 추출
        english_keywords = re.findall(r'\b[A-Za-z]{3,}\b', content)
        
        # 빈도수 기반으로 상위 키워드 선택
        keyword_freq = {}
        for keyword in korean_keywords + english_keywords:
            if len(keyword) >= 2:  # 2글자 이상만
                keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
        
        # 빈도수 순으로 정렬하여 상위 10개 선택
        sorted_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)
        return [keyword for keyword, freq in sorted_keywords[:10]]
    
    def _extract_required_subsections(self, content: str) -> List[str]:
        """컨텐츠에서 필수 하위 섹션을 추출합니다."""
        subsections = []
        
        # 일반적인 하위 섹션 패턴
        subsection_patterns = [
            r'^\s*[-•]\s*([^\n]+)',  # - 항목
            r'^\s*\d+\.\s*([^\n]+)',  # 1. 항목
            r'^\s*[가-힣]+\s*:\s*([^\n]+)',  # 항목: 설명
        ]
        
        lines = content.split('\n')
        for line in lines:
            for pattern in subsection_patterns:
                match = re.match(pattern, line)
                if match:
                    subsection = match.group(1).strip()
                    if len(subsection) > 3:  # 3글자 이상만
                        subsections.append(subsection)
                    break
        
        return subsections[:5]  # 최대 5개까지만
    
    def _extract_pdf_text(self, filepath: str) -> Optional[str]:
        """PDF에서 텍스트를 추출합니다."""
        if not PDF_AVAILABLE:
            self.logger.error("PDF 처리가 불가능합니다. PyPDF2를 설치해주세요.")
            return None
        
        try:
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                
                return text.strip()
                
        except Exception as e:
            self.logger.error(f"PDF 텍스트 추출 실패: {filepath} - {e}")
            return None
    
    def _get_document_type(self, filename: str) -> str:
        """문서 유형을 판별합니다."""
        filename_lower = filename.lower()
        
        if "isms" in filename_lower:
            return "ISMS 인증 기준"
        elif "보안약점" in filename_lower:
            return "보안 약점 진단 가이드"
        elif "취약점" in filename_lower:
            return "취약점 분석 평가 가이드"
        elif "국가정보보안" in filename_lower:
            return "국가정보보안 기본지침"
        else:
            return "일반 문서"
    
    def analyze_isms_document(self, user_document: str) -> Dict[str, Any]:
        """사용자 ISMS 문서를 분석하여 부족한 내용을 식별합니다."""
        analysis_result = {
            "document_analysis": {
                "analyzed_at": datetime.now().isoformat(),
                "total_sections_found": 0,
                "total_sections_required": 0,
                "compliance_score": 0.0,
                "missing_sections": [],
                "weak_sections": [],
                "recommendations": []
            },
            "section_analysis": {},
            "reference_comparison": {},
            "overall_assessment": {}
        }
        
        # 각 ISMS 요구사항별 분석
        total_required = 0
        total_found = 0
        
        for requirement_name, requirement_info in self.isms_requirements.items():
            section_result = self._analyze_section(
                requirement_name, 
                requirement_info, 
                user_document
            )
            
            analysis_result["section_analysis"][requirement_name] = section_result
            
            total_required += len(requirement_info["required_sections"])
            total_found += section_result["sections_found"]
            
            # 부족한 섹션 추가
            if section_result["missing_sections"]:
                analysis_result["document_analysis"]["missing_sections"].extend(
                    [f"{requirement_name}: {section}" for section in section_result["missing_sections"]]
                )
            
            # 약한 섹션 추가
            if section_result["strength_score"] < 0.7:
                analysis_result["document_analysis"]["weak_sections"].append({
                    "section": requirement_name,
                    "score": section_result["strength_score"],
                    "issues": section_result["identified_issues"]
                })
        
        # 전체 점수 계산
        analysis_result["document_analysis"]["total_sections_required"] = total_required
        analysis_result["document_analysis"]["total_sections_found"] = total_found
        
        if total_required > 0:
            compliance_score = (total_found / total_required) * 100
            analysis_result["document_analysis"]["compliance_score"] = round(compliance_score, 2)
        
        # 권장사항 생성
        analysis_result["document_analysis"]["recommendations"] = self._generate_recommendations(
            analysis_result["section_analysis"]
        )
        
        # 기준 문서와의 비교 분석
        analysis_result["reference_comparison"] = self._compare_with_references(user_document)
        
        # 전체 평가
        analysis_result["overall_assessment"] = self._generate_overall_assessment(
            analysis_result["document_analysis"]
        )
        
        self.last_analysis_result = analysis_result # 분석 결과를 저장
        self.analysis_results = analysis_result
        return analysis_result
    
    def _analyze_section(self, requirement_name: str, requirement_info: Dict, 
                         user_document: str) -> Dict[str, Any]:
        """특정 섹션을 분석합니다."""
        required_sections = requirement_info["required_sections"]
        keywords = requirement_info["keywords"]
        
        # 키워드 기반 내용 강도 분석
        content_strength = self._analyze_content_strength(user_document, keywords)
        
        # 필수 섹션 존재 여부 확인
        found_sections = []
        missing_sections = []
        
        for section in required_sections:
            if self._section_exists(user_document, section, keywords):
                found_sections.append(section)
            else:
                missing_sections.append(section)
        
        # 섹션 강도 점수 계산
        sections_found = len(found_sections)
        total_sections = len(required_sections)
        strength_score = sections_found / total_sections if total_sections > 0 else 0.0
        
        # 식별된 문제점
        identified_issues = []
        if strength_score < 0.5:
            identified_issues.append("대부분의 필수 섹션이 누락됨")
        elif strength_score < 0.8:
            identified_issues.append("일부 필수 섹션이 누락됨")
        
        if content_strength < 0.3:
            identified_issues.append("관련 내용이 부족함")
        
        return {
            "requirement_name": requirement_name,
            "description": requirement_info["description"],
            "required_sections": required_sections,
            "found_sections": found_sections,
            "missing_sections": missing_sections,
            "sections_found": sections_found,
            "total_sections": total_sections,
            "strength_score": round(strength_score, 3),
            "content_strength": round(content_strength, 3),
            "identified_issues": identified_issues,
            "keywords_used": keywords
        }
    
    def _analyze_content_strength(self, document: str, keywords: List[str]) -> float:
        """키워드 기반으로 내용의 강도를 분석합니다."""
        if not keywords:
            return 0.0
        
        document_lower = document.lower()
        found_keywords = 0
        
        for keyword in keywords:
            if keyword.lower() in document_lower:
                found_keywords += 1
        
        return found_keywords / len(keywords)
    
    def _section_exists(self, document: str, section_name: str, keywords: List[str]) -> bool:
        """특정 섹션이 문서에 존재하는지 확인합니다."""
        # 섹션명과 키워드를 모두 확인
        section_found = section_name.lower() in document.lower()
        keyword_found = any(keyword.lower() in document.lower() for keyword in keywords)
        
        return section_found or keyword_found
    
    def _generate_recommendations(self, section_analysis: Dict) -> List[str]:
        """권장사항을 생성합니다."""
        recommendations = []
        
        for section_name, analysis in section_analysis.items():
            if analysis["strength_score"] < 0.5:
                recommendations.append(
                    f"'{section_name}' 섹션을 대폭 보완해야 합니다. "
                    f"누락된 섹션: {', '.join(analysis['missing_sections'])}"
                )
            elif analysis["strength_score"] < 0.8:
                recommendations.append(
                    f"'{section_name}' 섹션을 보완해야 합니다. "
                    f"부족한 섹션: {', '.join(analysis['missing_sections'])}"
                )
            
            if analysis["content_strength"] < 0.5:
                recommendations.append(
                    f"'{section_name}' 섹션의 구체적인 내용을 추가해야 합니다."
                )
        
        if not recommendations:
            recommendations.append("전반적으로 잘 작성된 ISMS 문서입니다.")
        
        return recommendations
    
    def _compare_with_references(self, user_document: str) -> Dict[str, Any]:
        """기준 문서들과 비교 분석합니다."""
        comparison = {
            "reference_documents_used": list(self.reference_documents.keys()),
            "similarity_scores": {},
            "best_practices": [],
            "gaps_identified": []
        }
        
        if not self.reference_documents:
            comparison["note"] = "기준 문서가 로드되지 않았습니다."
            return comparison
        
        # 각 기준 문서와의 유사도 분석
        for doc_name, doc_info in self.reference_documents.items():
            similarity = self._calculate_similarity(user_document, doc_info["content"])
            comparison["similarity_scores"][doc_name] = round(similarity, 3)
        
        # 모범 사례 식별
        comparison["best_practices"] = self._identify_best_practices(user_document)
        
        # 격차 식별
        comparison["gaps_identified"] = self._identify_gaps(user_document)
        
        return comparison
    
    def _calculate_similarity(self, doc1: str, doc2: str) -> float:
        """두 문서 간의 유사도를 계산합니다."""
        # 간단한 키워드 기반 유사도 계산
        words1 = set(re.findall(r'\w+', doc1.lower()))
        words2 = set(re.findall(r'\w+', doc2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _identify_best_practices(self, user_document: str) -> List[str]:
        """모범 사례를 식별합니다."""
        best_practices = []
        
        # 문서 구조 관련
        if "목차" in user_document or "차례" in user_document:
            best_practices.append("체계적인 목차 구성")
        
        if "정의" in user_document or "용어" in user_document:
            best_practices.append("용어 정의 포함")
        
        if "부록" in user_document:
            best_practices.append("부록을 통한 상세 정보 제공")
        
        # 보안 관련
        if "정기점검" in user_document or "정기검토" in user_document:
            best_practices.append("정기적인 점검 및 검토 체계")
        
        if "교육" in user_document and "훈련" in user_document:
            best_practices.append("교육 및 훈련 계획 포함")
        
        return best_practices
    
    def _identify_gaps(self, user_document: str) -> List[str]:
        """격차를 식별합니다."""
        gaps = []
        
        # 일반적인 격차 패턴
        if "위험도" not in user_document and "리스크" not in user_document:
            gaps.append("위험도 평가 체계 부재")
        
        if "사고대응" not in user_document and "침해사고" not in user_document:
            gaps.append("사고 대응 체계 부재")
        
        if "감사" not in user_document and "점검" not in user_document:
            gaps.append("감사 및 점검 체계 부재")
        
        return gaps
    
    def _generate_overall_assessment(self, document_analysis: Dict) -> Dict[str, Any]:
        """전체 평가를 생성합니다."""
        compliance_score = document_analysis["compliance_score"]
        
        if compliance_score >= 90:
            grade = "A"
            assessment = "우수"
            description = "ISMS 인증을 위한 충분한 수준의 문서입니다."
        elif compliance_score >= 80:
            grade = "B"
            assessment = "양호"
            description = "일부 보완이 필요하지만 전반적으로 잘 작성되었습니다."
        elif compliance_score >= 70:
            grade = "C"
            assessment = "보통"
            description = "상당한 보완이 필요합니다."
        elif compliance_score >= 60:
            grade = "D"
            assessment = "미흡"
            description = "대폭적인 보완이 필요합니다."
        else:
            grade = "F"
            assessment = "부족"
            description = "ISMS 인증을 위해 전면적인 재작성이 필요합니다."
        
        return {
            "grade": grade,
            "assessment": assessment,
            "description": description,
            "compliance_score": compliance_score,
            "priority_actions": self._get_priority_actions(document_analysis)
        }
    
    def _get_priority_actions(self, document_analysis: Dict) -> List[str]:
        """우선순위 행동을 제시합니다."""
        actions = []
        
        if document_analysis["compliance_score"] < 70:
            actions.append("누락된 필수 섹션을 우선적으로 작성")
            actions.append("기준 문서 참고하여 구조 보완")
        
        if document_analysis["compliance_score"] < 80:
            actions.append("약한 섹션의 내용 강화")
            actions.append("구체적인 실행 방안 추가")
        
        actions.append("정기적인 문서 검토 및 업데이트")
        
        return actions
    
    def export_analysis(self, format: str = "json") -> str:
        """분석 결과를 내보냅니다."""
        if format.lower() == "json":
            return json.dumps(self.analysis_results, ensure_ascii=False, indent=2)
        else:
            return str(self.analysis_results)
    
    def get_summary(self) -> Dict[str, Any]:
        """분석 결과 요약을 반환합니다."""
        if not self.analysis_results:
            return {"status": "분석이 수행되지 않았습니다."}
        
        return {
            "compliance_score": self.analysis_results["document_analysis"]["compliance_score"],
            "grade": self.analysis_results["overall_assessment"]["grade"],
            "missing_sections_count": len(self.analysis_results["document_analysis"]["missing_sections"]),
            "weak_sections_count": len(self.analysis_results["document_analysis"]["weak_sections"]),
            "recommendations_count": len(self.analysis_results["document_analysis"]["recommendations"])
        }
