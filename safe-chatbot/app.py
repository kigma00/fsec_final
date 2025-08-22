import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime
from document_analyzer import DocumentAnalyzer

# 페이지 설정
st.set_page_config(
    page_title="ISMS-P 문서 분석기",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem;
    }
    .file-upload-box {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f9f9f9;
        margin: 1rem 0;
    }
    .file-info {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4caf50;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if 'document_analyzer' not in st.session_state:
    st.session_state.document_analyzer = DocumentAnalyzer()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

if 'file_content' not in st.session_state:
    st.session_state.file_content = ""

if 'enterprise_guideline' not in st.session_state:
    st.session_state.enterprise_guideline = None

if 'enterprise_guideline_content' not in st.session_state:
    st.session_state.enterprise_guideline_content = ""

# 헬퍼 함수들
def generate_analysis_response(question: str, analysis_result: dict) -> str:
    """분석 결과를 기반으로 질문에 답변합니다."""
    question_lower = question.lower()
    
    # 준수도 점수 관련 질문
    if any(word in question_lower for word in ["점수", "등급", "평가", "수준"]):
        score = analysis_result["document_analysis"]["compliance_score"]
        grade = analysis_result["overall_assessment"]["grade"]
        assessment = analysis_result["overall_assessment"]["assessment"]
        return f"현재 ISMS 문서의 준수도 점수는 {score}%입니다. 등급은 {grade}이며, 평가는 {assessment}입니다."
    
    # 약한 섹션 관련 질문
    if any(word in question_lower for word in ["약한", "부족한", "문제", "취약"]):
        weak_sections = analysis_result["document_analysis"]["weak_sections"]
        if weak_sections:
            response = "현재 약한 섹션들은 다음과 같습니다:\n"
            for section in weak_sections:
                response += f"• {section['section']}: 점수 {section['score']*100:.1f}% - {', '.join(section['issues'])}\n"
            return response
        else:
            return "현재 약한 섹션이 없습니다. 모든 섹션이 적절한 수준으로 작성되었습니다."
    
    # 누락된 섹션 관련 질문
    if any(word in question_lower for word in ["누락", "빠진", "없는", "부족"]):
        missing_sections = analysis_result["document_analysis"]["missing_sections"]
        if missing_sections:
            response = "누락된 섹션들은 다음과 같습니다:\n"
            for section in missing_sections:
                response += f"• {section}\n"
            return response
        else:
            return "누락된 섹션이 없습니다. 모든 필수 섹션이 포함되어 있습니다."
    
    # 권장사항 관련 질문
    if any(word in question_lower for word in ["권장", "개선", "보완", "해야"]):
        recommendations = analysis_result["document_analysis"]["recommendations"]
        if recommendations:
            response = "다음과 같은 개선사항을 권장합니다:\n"
            for i, rec in enumerate(recommendations, 1):
                response += f"{i}. {rec}\n"
            return response
        else:
            return "현재 문서는 잘 작성되어 있어 특별한 개선사항이 없습니다."
    
    # 특정 섹션 관련 질문
    for section_name in analysis_result["section_analysis"].keys():
        if section_name.lower() in question_lower:
            section_info = analysis_result["section_analysis"][section_name]
            strength = section_info["strength_score"] * 100
            content = section_info["content_strength"] * 100
            
            response = f"{section_name} 섹션 분석 결과:\n"
            response += f"• 강도 점수: {strength:.1f}%\n"
            response += f"• 내용 강도: {content:.1f}%\n"
            
            if section_info["found_sections"]:
                response += f"• 포함된 섹션: {', '.join(section_info['found_sections'])}\n"
            
            if section_info["missing_sections"]:
                response += f"• 누락된 섹션: {', '.join(section_info['missing_sections'])}\n"
            
            if section_info["identified_issues"]:
                response += f"• 식별된 문제점: {', '.join(section_info['identified_issues'])}"
            
            return response
    
    # 기본 응답
    return "분석 결과에 대해 구체적으로 질문해주세요. 예를 들어:\n• 준수도 점수는 어떻게 되나요?\n• 어떤 섹션이 가장 약한가요?\n• 누락된 내용은 무엇인가요?\n• 개선사항은 무엇인가요?"

def split_document_by_sections(content: str, max_chunk_size: int = 30000) -> list:
    """큰 문서를 섹션별로 분할합니다."""
    sections = []
    
    # 섹션 구분자들
    section_markers = [
        "\n\n", "\n", ". ", " ", "."
    ]
    
    current_chunk = ""
    lines = content.split('\n')
    
    for line in lines:
        # 현재 청크에 새 줄 추가
        test_chunk = current_chunk + line + "\n"
        
        # 청크 크기 확인
        if len(test_chunk) > max_chunk_size and current_chunk:
            # 현재 청크 저장
            sections.append(current_chunk.strip())
            current_chunk = line + "\n"
        else:
            current_chunk = test_chunk
    
    # 마지막 청크 추가
    if current_chunk.strip():
        sections.append(current_chunk.strip())
    
    # 빈 섹션 제거
    sections = [s for s in sections if s.strip()]
    
    return sections

def combine_section_results(section_results: list, original_content: str) -> dict:
    """분할된 섹션 분석 결과를 통합합니다."""
    if not section_results:
        return {}
    
    # 첫 번째 결과를 기본으로 사용
    combined_result = section_results[0].copy()
    
    # 통계 정보 통합
    total_sections_found = sum(r["document_analysis"]["total_sections_found"] for r in section_results)
    total_sections_required = sum(r["document_analysis"]["total_sections_required"] for r in section_results)
    
    # 준수도 점수 재계산
    if total_sections_required > 0:
        compliance_score = (total_sections_found / total_sections_required) * 100
        combined_result["document_analysis"]["compliance_score"] = round(compliance_score, 2)
        combined_result["document_analysis"]["total_sections_found"] = total_sections_found
        combined_result["document_analysis"]["total_sections_required"] = total_sections_required
    
    # 섹션별 분석 통합
    combined_sections = {}
    for result in section_results:
        for section_name, section_info in result["section_analysis"].items():
            if section_name not in combined_sections:
                combined_sections[section_name] = section_info.copy()
            else:
                # 점수 평균 계산
                combined_sections[section_name]["strength_score"] = (
                    combined_sections[section_name]["strength_score"] + section_info["strength_score"]
                ) / 2
                combined_sections[section_name]["content_strength"] = (
                    combined_sections[section_name]["content_strength"] + section_info["content_strength"]
                ) / 2
    
    combined_result["section_analysis"] = combined_sections
    
    # 전체 평가 재생성
    combined_result["overall_assessment"] = combined_result["document_analysis"]["overall_assessment"]
    
    return combined_result

# JSON 검증 관련 함수들
def verify_json_structure(uploaded_json: dict, current_result: dict) -> dict:
    """JSON 구조를 검증합니다."""
    required_fields = [
        "document_analysis", "overall_assessment", "section_analysis", 
        "reference_comparison", "timestamp"
    ]
    
    required_fields_status = {}
    for field in required_fields:
        required_fields_status[field] = field in uploaded_json
    
    # 데이터 타입 검증
    data_types = {
        "document_analysis": dict,
        "overall_assessment": dict,
        "section_analysis": dict,
        "reference_comparison": dict
    }
    
    data_types_status = {}
    for field, expected_type in data_types.items():
        if field in uploaded_json:
            data_types_status[field] = isinstance(uploaded_json[field], expected_type)
        else:
            data_types_status[field] = False
    
    return {
        "required_fields": required_fields_status,
        "data_types": data_types_status
    }

def verify_json_content(uploaded_json: dict, current_result: dict, similarity_threshold: float) -> dict:
    """JSON 내용을 검증합니다."""
    # 수치 데이터 검증
    numeric_metrics = {}
    
    if "document_analysis" in uploaded_json and "document_analysis" in current_result:
        current_score = current_result["document_analysis"].get("compliance_score", 0)
        uploaded_score = uploaded_json["document_analysis"].get("compliance_score", 0)
        
        numeric_metrics["준수도 점수"] = {
            "value": uploaded_score,
            "expected": current_score,
            "match": abs(uploaded_score - current_score) <= 0.1,
            "difference": abs(uploaded_score - current_score)
        }
    
    # 텍스트 데이터 검증
    text_fields = {}
    
    if "overall_assessment" in uploaded_json and "overall_assessment" in current_result:
        current_grade = current_result["overall_assessment"].get("grade", "")
        uploaded_grade = uploaded_json["overall_assessment"].get("grade", "")
        
        # 등급 유사도 계산 (간단한 문자열 비교)
        if current_grade and uploaded_grade:
            similarity = 1.0 if current_grade == uploaded_grade else 0.0
        else:
            similarity = 0.0
        
        text_fields["전체 등급"] = {
            "value": uploaded_grade,
            "expected": current_grade,
            "similarity": similarity
        }
    
    return {
        "numeric_metrics": numeric_metrics,
        "text_fields": text_fields
    }

def categorize_differences(differences: dict) -> dict:
    """차이점을 카테고리별로 분류합니다."""
    categories = {
        "구조적 차이": [],
        "수치적 차이": [],
        "텍스트적 차이": [],
        "기타 차이": []
    }
    
    for field, description in differences.items():
        if "score" in field.lower() or "점수" in field:
            categories["수치적 차이"].append({
                "field": field,
                "description": description,
                "impact": "높음 - 분석 결과에 직접적 영향"
            })
        elif "grade" in field.lower() or "등급" in field:
            categories["텍스트적 차이"].append({
                "field": field,
                "description": description,
                "impact": "중간 - 평가 결과에 영향"
            })
        elif "section" in field.lower() or "섹션" in field:
            categories["구조적 차이"].append({
                "field": field,
                "description": description,
                "impact": "높음 - 문서 구조에 영향"
            })
        else:
            categories["기타 차이"].append({
                "field": field,
                "description": description,
                "impact": "낮음 - 제한적 영향"
            })
    
    # 빈 카테고리 제거
    return {k: v for k, v in categories.items() if v}

def generate_verification_report(verification_result: dict, structural_verification: dict, 
                               content_verification: dict, filename: str) -> str:
    """검증 보고서를 생성합니다."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
# ISMS-P JSON 검증 보고서

**검증 일시:** {timestamp}  
**검증 대상 파일:** {filename}  
**검증 상태:** {'✅ 성공' if verification_result['is_valid'] else '⚠️ 차이점 발견'}

## 📊 검증 결과 요약

- **전체 상태:** {verification_result['description']}
- **차이점 수:** {len(verification_result['differences'])}
- **검증 시간:** {verification_result.get('verification_timestamp', 'N/A')}

## 🔍 구조적 검증 결과

### 필수 필드 검증
"""
    
    for field, status in structural_verification["required_fields"].items():
        report += f"- {'✅' if status else '❌'} {field}\n"
    
    report += """
### 데이터 타입 검증
"""
    
    for field, status in structural_verification["data_types"].items():
        report += f"- {'✅' if status else '❌'} {field}\n"
    
    report += """
## 📝 내용 검증 결과

### 수치 데이터 검증
"""
    
    for metric, details in content_verification["numeric_metrics"].items():
        status = "✅" if details["match"] else "❌"
        report += f"- {status} {metric}: {details['value']} (예상: {details['expected']})\n"
    
    report += """
### 텍스트 데이터 검증
"""
    
    for field, details in content_verification["text_fields"].items():
        similarity = details["similarity"]
        status = "✅" if similarity >= 0.85 else "⚠️"
        report += f"- {status} {field}: {similarity:.1%} 유사도\n"
    
    if not verification_result["is_valid"]:
        report += """
## ⚠️ 발견된 차이점

"""
        for field, description in verification_result['differences'].items():
            report += f"- **{field}**: {description}\n"
    
    report += """
## 📋 권장사항

"""
    
    if verification_result["is_valid"]:
        report += "- ✅ 검증이 성공적으로 완료되었습니다.\n"
        report += "- 📊 데이터 무결성이 확인되었습니다.\n"
        report += "- 🔒 업로드된 JSON을 신뢰할 수 있습니다.\n"
    else:
        report += "- ⚠️ 발견된 차이점을 검토하세요.\n"
        report += "- 🔍 원본 데이터와 비교하여 정확성을 확인하세요.\n"
        report += "- 📝 필요시 데이터를 수정하고 재검증하세요.\n"
    
    report += f"""
---
*이 보고서는 ISMS-P 문서 분석기에서 자동으로 생성되었습니다.*
*생성 시간: {timestamp}*
"""
    
    return report

# 메인 헤더
st.markdown("""
<div class="main-header">
    <h1>🔍 ISMS-P 문서 분석기</h1>
    <p>ISMS-P 준수성 분석 및 부족한 내용 식별</p>
</div>
""", unsafe_allow_html=True)

# 메인 컨텐츠
tab1, tab2, tab3 = st.tabs(["🔍 ISMS-P 문서 분석", "💬 분석 결과 질문", "🔍 JSON 결과 검증"])

with tab1:
    st.header("🔍 ISMS-P 문서 분석")
    
    # 기준 문서 로드
    if 'reference_docs_loaded' not in st.session_state:
        with st.spinner("기준 문서를 로드하고 있습니다..."):
            st.session_state.document_analyzer.load_reference_documents("data")
            st.session_state.reference_docs_loaded = True
        st.success("기준 문서 로드 완료!")
    
    # 기업 내부 지침 추가 업로드 (선택사항)
    st.subheader("🏢 기업 내부 지침 추가 (선택사항)")
    st.info("💡 표준 지침(KISA, PCI-DSS 등)을 기본으로 사용하며, 기업 내부 지침이 있다면 추가로 업로드할 수 있습니다.")
    
    enterprise_guideline = st.file_uploader(
        "기업 내부 ISMS-P 지침 파일을 추가로 업로드하세요 (선택사항)",
        type=['txt', 'md', 'pdf'],
        help="기업 내부 ISMS-P 지침이 담긴 텍스트 파일(.txt), 마크다운(.md), PDF(.pdf) 파일을 추가로 업로드할 수 있습니다.",
        key="enterprise_guideline_uploader"
    )
    
    if enterprise_guideline is not None:
        st.session_state.enterprise_guideline = enterprise_guideline
        
        # 기업 지침 파일 정보 표시
        guideline_size = len(enterprise_guideline.getvalue())
        guideline_size_mb = guideline_size / (1024 * 1024)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("지침 파일명", enterprise_guideline.name)
        with col2:
            st.metric("파일 크기", f"{guideline_size_mb:.2f} MB")
        with col3:
            st.metric("문자 수", f"{guideline_size:,}")
        
        # 기업 지침 내용 추출
        try:
            if enterprise_guideline.type == "application/pdf":
                # PDF 파일 처리
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(enterprise_guideline)
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
                st.session_state.enterprise_guideline_content = content.strip()
                st.success(f"✅ 기업 지침 PDF에서 {len(content)} 문자를 추출했습니다.")
                
            else:
                # 텍스트 파일 처리
                content = enterprise_guideline.getvalue().decode('utf-8')
                st.session_state.enterprise_guideline_content = content
                st.success(f"✅ 기업 지침 텍스트 파일을 성공적으로 읽었습니다.")
            
            # 기업 지침 내용 미리보기
            with st.expander("📖 기업 지침 내용 미리보기"):
                preview_length = min(1000, len(st.session_state.enterprise_guideline_content))
                st.text(st.session_state.enterprise_guideline_content[:preview_length])
                if len(st.session_state.enterprise_guideline_content) > preview_length:
                    st.text("... (내용이 길어 일부만 표시)")
            
            # 기업 지침 기반 분석기 설정
            if st.button("🔧 기업 지침 추가 설정", type="secondary"):
                with st.spinner("기업 지침을 추가로 설정하고 있습니다..."):
                    # 기업 지침을 추가로 설정
                    st.session_state.document_analyzer.set_enterprise_guideline(
                        st.session_state.enterprise_guideline_content
                    )
                st.success("✅ 기업 지침이 표준 지침에 추가로 설정되었습니다!")
            
        except Exception as e:
            st.error(f"기업 지침 파일 읽기 오류: {e}")
            st.session_state.enterprise_guideline_content = ""
    
    elif st.session_state.enterprise_guideline_content:
        st.success("✅ 기업 지침이 이미 추가로 설정되어 있습니다.")
        
        # 기업 지침 내용 미리보기
        with st.expander("📖 현재 설정된 기업 지침"):
            preview_length = min(1000, len(st.session_state.enterprise_guideline_content))
            st.text(st.session_state.enterprise_guideline_content[:preview_length])
            if len(st.session_state.enterprise_guideline_content) > preview_length:
                st.text("... (내용이 길어 일부만 표시)")
    
    # 파일 업로드 섹션
    st.subheader("📁 파일 업로드")
    
    uploaded_file = st.file_uploader(
        "ISMS-P 문서 파일을 업로드하세요",
        type=['txt', 'md', 'pdf'],
        help="텍스트 파일(.txt), 마크다운(.md), PDF(.pdf) 파일을 지원합니다."
    )
    
    if uploaded_file is not None:
        st.session_state.uploaded_file = uploaded_file
        
        # 파일 정보 표시
        file_size = len(uploaded_file.getvalue())
        file_size_mb = file_size / (1024 * 1024)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("파일명", uploaded_file.name)
        with col2:
            st.metric("파일 크기", f"{file_size_mb:.2f} MB")
        with col3:
            st.metric("문자 수", f"{file_size:,}")
        
        # 파일 크기 경고
        if file_size_mb > 10:
            st.warning("⚠️ 파일이 큽니다 (10MB 초과). 분석 시간이 오래 걸릴 수 있습니다.")
        elif file_size_mb > 5:
            st.info("ℹ️ 파일이 중간 크기입니다. 분석에 시간이 걸릴 수 있습니다.")
        
        # 파일 내용 추출
        try:
            if uploaded_file.type == "application/pdf":
                # PDF 파일 처리
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
                st.session_state.file_content = content.strip()
                st.success(f"✅ PDF 파일에서 {len(content)} 문자를 추출했습니다.")
                
            else:
                # 텍스트 파일 처리
                content = uploaded_file.getvalue().decode('utf-8')
                st.session_state.file_content = content
                st.success(f"✅ 텍스트 파일을 성공적으로 읽었습니다.")
            
            # 파일 내용 미리보기
            with st.expander("📖 파일 내용 미리보기"):
                preview_length = min(1000, len(st.session_state.file_content))
                st.text(st.session_state.file_content[:preview_length])
                if len(st.session_state.file_content) > preview_length:
                    st.text("... (내용이 길어 일부만 표시)")
            
        except Exception as e:
            st.error(f"파일 읽기 오류: {e}")
            st.session_state.file_content = ""
    
    # 수동 입력 섹션
    st.subheader("📝 또는 직접 입력")
    manual_input = st.text_area(
        "분석할 ISMS-P 문서 내용을 직접 입력하세요:",
        height=200,
        placeholder="여기에 ISMS-P 문서 내용을 붙여넣거나 위에서 파일을 업로드하세요..."
    )
    
    # 분석 실행 버튼
    if st.button("🔍 문서 분석 실행", type="primary"):
        # 기업 지침 모드일 때 검증
        if st.session_state.enterprise_guideline_content:
            st.info("💡 표준 지침(KISA, PCI-DSS 등)과 업로드된 기업 내부 지침을 모두 기준으로 문서를 분석합니다.")
        else:
            st.warning("⚠️ 표준 지침(KISA, PCI-DSS 등)만 기준으로 문서를 분석합니다.")
        
        # 입력 내용 확인
        content_to_analyze = ""
        if st.session_state.file_content:
            content_to_analyze = st.session_state.file_content
        elif manual_input.strip():
            content_to_analyze = manual_input.strip()
        else:
            st.warning("분석할 문서 내용이 없습니다. 파일을 업로드하거나 내용을 입력해주세요.")
            st.stop()
        
        # 파일 크기 확인 및 분할 처리
        if len(content_to_analyze) > 50000:  # 50KB 초과시 분할
            st.info("📄 문서가 큽니다. 자동으로 분할하여 분석합니다...")
            
            # 문서를 섹션별로 분할
            sections = split_document_by_sections(content_to_analyze)
            
            with st.spinner(f"분할된 {len(sections)}개 섹션을 분석하고 있습니다..."):
                # 각 섹션별로 분석
                section_results = []
                for i, section in enumerate(sections):
                    st.write(f"섹션 {i+1}/{len(sections)} 분석 중...")
                    result = st.session_state.document_analyzer.analyze_isms_document(section)
                    section_results.append(result)
                
                # 결과 통합
                combined_result = combine_section_results(section_results, content_to_analyze)
                st.session_state.analysis_result = combined_result
                
        else:
            # 일반 분석
            with st.spinner("ISMS-P 문서를 분석하고 있습니다..."):
                analysis_result = st.session_state.document_analyzer.analyze_isms_document(content_to_analyze)
                st.session_state.analysis_result = analysis_result
        
        st.success("분석이 완료되었습니다!")
    
    # 분석 결과 표시
    if 'analysis_result' in st.session_state:
        result = st.session_state.analysis_result
        
        # 현재 분석 모드 표시
        if st.session_state.enterprise_guideline_content:
            st.success("🔍 **표준 지침 + 기업 내부 지침 기반 분석 결과**")
            st.info(f"📋 분석 기준: 표준 지침(KISA, PCI-DSS 등) + 업로드된 기업 지침 ({len(st.session_state.enterprise_guideline_content)} 문자)")
        else:
            st.success("📚 **표준 지침 기반 분석 결과**")
            st.info("📋 분석 기준: KISA, PCI-DSS 등 표준 ISMS 지침")
        
        # 전체 평가
        st.subheader("📊 전체 평가")
        overall = result["overall_assessment"]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("준수도 점수", f"{result['document_analysis']['compliance_score']}%")
        with col2:
            st.metric("등급", overall["grade"])
        with col3:
            st.metric("평가", overall["assessment"])
        with col4:
            st.metric("누락 섹션", len(result['document_analysis']['missing_sections']))
        
        # 평가 설명
        st.info(f"**평가 결과:** {overall['description']}")
        
        # 우선순위 행동
        if overall["priority_actions"]:
            st.subheader("🎯 우선순위 행동")
            for i, action in enumerate(overall["priority_actions"], 1):
                st.write(f"{i}. {action}")
        
        # 섹션별 분석
        st.subheader("📋 섹션별 분석")
        section_analysis = result["section_analysis"]
        
        for section_name, analysis in section_analysis.items():
            with st.expander(f"{section_name} ({analysis['strength_score']*100:.1f}%)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**설명:** {analysis['description']}")
                    st.write(f"**강도 점수:** {analysis['strength_score']*100:.1f}%")
                    st.write(f"**내용 강도:** {analysis['content_strength']*100:.1f}%")
                
                with col2:
                    if analysis['found_sections']:
                        st.write("**✅ 포함된 섹션:**")
                        for section in analysis['found_sections']:
                            st.write(f"  • {section}")
                    
                    if analysis['missing_sections']:
                        st.write("**❌ 누락된 섹션:**")
                        for section in analysis['missing_sections']:
                            st.write(f"  • {section}")
                
                if analysis['identified_issues']:
                    st.warning("**⚠️ 식별된 문제점:**")
                    for issue in analysis['identified_issues']:
                        st.write(f"  • {issue}")
        
        # 권장사항
        if result['document_analysis']['recommendations']:
            st.subheader("💡 권장사항")
            for i, rec in enumerate(result['document_analysis']['recommendations'], 1):
                st.write(f"{i}. {rec}")
        
        # 기준 문서 비교
        if result['reference_comparison'].get('similarity_scores'):
            st.subheader("📚 기준 문서 비교")
            similarity_df = pd.DataFrame.from_dict(
                result['reference_comparison']['similarity_scores'], 
                orient='index', 
                columns=['유사도']
            )
            st.dataframe(similarity_df, use_container_width=True)
        
        # 모범 사례
        if result['reference_comparison'].get('best_practices'):
            st.subheader("🏆 모범 사례")
            for practice in result['reference_comparison']['best_practices']:
                st.write(f"✅ {practice}")
        
        # 격차 식별
        if result['reference_comparison'].get('gaps_identified'):
            st.subheader("⚠️ 격차 식별")
            for gap in result['reference_comparison']['gaps_identified']:
                st.write(f"❌ {gap}")
        
        # JSON 내보내기
        st.subheader("📥 분석 결과 내보내기")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📄 JSON 다운로드"):
                json_str = st.session_state.document_analyzer.export_analysis("json")
                st.download_button(
                    label="💾 JSON 파일 다운로드",
                    data=json_str,
                    file_name=f"isms_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📊 CSV 다운로드"):
                csv_str = st.session_state.document_analyzer.export_analysis("csv")
                st.download_button(
                    label="💾 CSV 파일 다운로드",
                    data=csv_str,
                    file_name=f"isms_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        # 분석 요약
        st.subheader("📈 분석 요약")
        summary = st.session_state.document_analyzer.get_summary()
        summary_df = pd.DataFrame([summary])
        st.dataframe(summary_df, use_container_width=True)

with tab2:
    st.header("💬 분석 결과 질문")
    
    if 'analysis_result' not in st.session_state:
        st.info("먼저 ISMS 문서 분석을 실행해주세요!")
    else:
        st.success("분석 결과에 대해 질문할 수 있습니다!")
        
        # 채팅 입력
        col1, col2 = st.columns([4, 1])
        with col1:
            user_input = st.text_input("분석 결과에 대해 질문하세요...", placeholder="예: 어떤 섹션이 가장 약한가요? 누락된 내용은 무엇인가요?")
        with col2:
            send_button = st.button("전송", type="primary")
        
        # 메시지 전송 처리
        if send_button and user_input:
            # 사용자 메시지 추가
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            # 분석 결과 기반 응답 생성
            response = generate_analysis_response(user_input, st.session_state.analysis_result)
            
            # 챗봇 응답 추가
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
            
            # 페이지 새로고침
            st.rerun()
        
        # 채팅 히스토리 표시
        if st.session_state.chat_history:
            st.subheader("💭 대화 기록")
            
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    # 사용자 메시지
                    with st.container():
                        st.markdown(f"**👤 사용자 ({message['timestamp']}):**")
                        st.info(message['content'])
                else:
                    # 챗봇 응답
                    with st.container():
                        st.markdown(f"**🤖 분석 도우미 ({message['timestamp']}):**")
                        st.success(message['content'])
                    st.divider()
        
        # 채팅 히스토리 초기화
        if st.button("🗑️ 대화 기록 초기화"):
            st.session_state.chat_history = []
            st.rerun()

with tab3:
    st.header("🔍 JSON 결과 검증")
    
    st.info("💡 이 기능은 업로드된 JSON 파일과 현재 분석 결과를 비교하여 차이점을 검증하고 데이터 무결성을 보장합니다.")
    
    # 검증 설정
    st.subheader("⚙️ 검증 설정")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        similarity_threshold = st.slider(
            "유사도 임계값 (%)",
            min_value=0.0,
            max_value=100.0,
            value=85.0,
            step=5.0,
            help="이 값 이상의 유사도를 가진 결과를 '일치'로 간주합니다."
        )
    
    with col2:
        strict_mode = st.checkbox(
            "엄격 모드",
            value=False,
            help="엄격 모드에서는 모든 필드가 정확히 일치해야 합니다."
        )
    
    with col3:
        include_details = st.checkbox(
            "상세 분석 포함",
            value=True,
            help="상세한 분석 정보를 포함하여 검증합니다."
        )
    
    # 검증할 JSON 파일 업로드
    st.subheader("📁 검증할 JSON 파일 업로드")
    uploaded_json_file = st.file_uploader(
        "검증할 ISMS-P 문서 분석 JSON 파일을 업로드하세요:",
        type=["json"],
        help="분석된 ISMS-P 문서의 JSON 파일을 업로드하여 검증합니다."
    )
    
    if uploaded_json_file is not None:
        try:
            json_content = uploaded_json_file.getvalue().decode('utf-8')
            json_data = json.loads(json_content)
            st.session_state.uploaded_json_data = json_data
            st.success(f"✅ 업로드된 JSON 파일을 성공적으로 읽었습니다. ({len(json_content)} 문자)")
            
            # 파일 정보 표시
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("파일명", uploaded_json_file.name)
            with col2:
                st.metric("파일 크기", f"{len(json_content)/1024:.1f} KB")
            with col3:
                st.metric("JSON 키 수", len(json_data.keys()) if isinstance(json_data, dict) else "N/A")
            
            # JSON 내용 미리보기
            with st.expander("📖 JSON 내용 미리보기"):
                st.json(json_data)
            
            # 검증 버튼
            if st.button("🔍 JSON 결과 검증 실행", type="primary"):
                if 'analysis_result' in st.session_state:
                    with st.spinner("JSON 결과를 상세하게 검증하고 있습니다..."):
                        try:
                            verification_result = st.session_state.document_analyzer.feedback_verifier(json_data)
                            
                            # 검증 결과 요약
                            st.subheader("📊 검증 결과 요약")
                            
                            if verification_result["is_valid"]:
                                st.success("✅ JSON 결과 검증 성공!")
                                st.info(f"**검증 결과:** {verification_result['description']}")
                                
                                # 성공 메트릭
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("검증 상태", "성공", delta="✅")
                                with col2:
                                    st.metric("차이점 수", "0", delta="일치")
                                with col3:
                                    st.metric("신뢰도", "100%", delta="완벽")
                                
                            else:
                                st.warning("⚠️ JSON 결과 검증에서 차이점이 발견되었습니다.")
                                st.info(f"**검증 결과:** {verification_result['description']}")
                                
                                # 차이점 메트릭
                                difference_count = len(verification_result['differences'])
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("검증 상태", "차이점 발견", delta=f"⚠️ {difference_count}개")
                                with col2:
                                    st.metric("차이점 수", difference_count, delta="주의 필요")
                                with col3:
                                    confidence = max(0, 100 - (difference_count * 20))
                                    st.metric("신뢰도", f"{confidence}%", delta=f"감소 {100-confidence}%")
                            
                            # 상세 검증 결과
                            st.subheader("🔍 상세 검증 분석")
                            
                            # 1. 구조적 검증
                            st.markdown("#### 📋 구조적 검증")
                            structural_verification = verify_json_structure(json_data, st.session_state.analysis_result)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**필수 필드 검증:**")
                                for field, status in structural_verification["required_fields"].items():
                                    if status:
                                        st.write(f"✅ {field}")
                                    else:
                                        st.write(f"❌ {field}")
                            
                            with col2:
                                st.write("**데이터 타입 검증:**")
                                for field, status in structural_verification["data_types"].items():
                                    if status:
                                        st.write(f"✅ {field}")
                                    else:
                                        st.write(f"❌ {field}")
                            
                            # 2. 내용 검증
                            st.markdown("#### 📝 내용 검증")
                            content_verification = verify_json_content(json_data, st.session_state.analysis_result, similarity_threshold)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write("**수치 데이터 검증:**")
                                for metric, details in content_verification["numeric_metrics"].items():
                                    if details["match"]:
                                        st.write(f"✅ {metric}: {details['value']}")
                                    else:
                                        st.write(f"❌ {metric}: {details['value']} (예상: {details['expected']})")
                            
                            with col2:
                                st.write("**텍스트 데이터 검증:**")
                                for field, details in content_verification["text_fields"].items():
                                    similarity = details["similarity"]
                                    if similarity >= similarity_threshold / 100:
                                        st.write(f"✅ {field}: {similarity:.1%}")
                                    else:
                                        st.write(f"⚠️ {field}: {similarity:.1%}")
                            
                            # 3. 차이점 상세 분석
                            if not verification_result["is_valid"]:
                                st.markdown("#### ⚠️ 차이점 상세 분석")
                                
                                # 차이점 카테고리별 분류
                                categorized_differences = categorize_differences(verification_result['differences'])
                                
                                for category, items in categorized_differences.items():
                                    with st.expander(f"🔍 {category} ({len(items)}개)"):
                                        for item in items:
                                            st.write(f"• **{item['field']}**: {item['description']}")
                                            if 'impact' in item:
                                                st.info(f"영향도: {item['impact']}")
                                
                                # 시각적 차이점 표시
                                st.markdown("#### 📊 차이점 시각화")
                                
                                # 차이점 분포 차트
                                difference_data = {
                                    'category': list(categorized_differences.keys()),
                                    'count': [len(items) for items in categorized_differences.values()]
                                }
                                
                                if difference_data['count']:
                                    chart_df = pd.DataFrame(difference_data)
                                    st.bar_chart(chart_df.set_index('category'))
                            
                            # 4. 원본 vs 검증된 JSON 비교
                            st.markdown("#### 📚 상세 비교")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.subheader("📚 업로드된 JSON")
                                st.json(json_data)
                            
                            with col2:
                                st.subheader("🔍 검증된 JSON")
                                st.json(verification_result['verified_json'])
                            
                            # 5. 검증 보고서 생성
                            st.markdown("#### 📄 검증 보고서")
                            
                            report = generate_verification_report(
                                verification_result, 
                                structural_verification, 
                                content_verification,
                                uploaded_json_file.name
                            )
                            
                            with st.expander("📋 검증 보고서 보기"):
                                st.markdown(report)
                            
                            # 보고서 다운로드
                            if st.button("💾 검증 보고서 다운로드", type="secondary"):
                                st.download_button(
                                    label="📥 HTML 보고서 다운로드",
                                    data=report,
                                    file_name=f"verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                    mime="text/html"
                                )
                            
                        except Exception as e:
                            st.error(f"검증 중 오류 발생: {e}")
                            st.exception(e)
                else:
                    st.error("❌ 먼저 ISMS 문서 분석을 실행해주세요!")
                    
        except json.JSONDecodeError:
            st.error("❌ 업로드된 파일이 유효한 JSON 형식이 아닙니다.")
        except Exception as e:
            st.error(f"JSON 파일 읽기 오류: {e}")
    
    # 현재 분석 결과가 있는 경우 표시
    if 'analysis_result' in st.session_state:
        st.subheader("📊 현재 분석 결과 요약")
        current_result = st.session_state.analysis_result
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'document_analysis' in current_result:
                score = current_result['document_analysis'].get('compliance_score', 0)
                st.metric("준수도 점수", f"{score}%")
        
        with col2:
            if 'overall_assessment' in current_result:
                grade = current_result['overall_assessment'].get('grade', 'N/A')
                st.metric("전체 등급", grade)
        
        with col3:
            if 'section_analysis' in current_result:
                section_count = len(current_result['section_analysis'])
                st.metric("분석된 섹션", section_count)
        
        # 현재 결과 다운로드
        if st.button("💾 현재 분석 결과 다운로드", type="secondary"):
            json_str = json.dumps(current_result, ensure_ascii=False, indent=2)
            st.download_button(
                label="📥 JSON 다운로드",
                data=json_str,
                file_name=f"isms_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # 검증 히스토리 (선택사항)
    if 'verification_history' not in st.session_state:
        st.session_state.verification_history = []
    
    if st.session_state.verification_history:
        st.subheader("📈 검증 히스토리")
        history_df = pd.DataFrame(st.session_state.verification_history)
        st.dataframe(history_df, use_container_width=True)
