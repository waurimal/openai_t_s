import streamlit as st
import fitz
from openai import OpenAI
import os
import tempfile
import easyocr
import numpy as np
from PIL import Image

# OCR 리더 초기화 (세션 상태로 관리)
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['ko', 'en'])  # 한국어, 영어 지원

# 텍스트 추출 함수 정의 (OCR 포함)
def get_text_from_pages(pdf_bytes, start_page=None, end_page=None, use_ocr=False):
    """
    pdf_bytes에서 지정된 페이지 범위의 텍스트를 추출.
    start_page와 end_page는 1부터 시작하는 페이지 번호.
    use_ocr=True면 OCR을 사용하여 텍스트 추출
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # 범위 지정 없으면 전체
    if start_page is None or end_page is None:
        start_page = 1
        end_page = doc.page_count

    # 안전 범위 보정
    start_page = max(1, start_page)
    end_page = min(doc.page_count, end_page)

    texts = []
    
    if use_ocr:
        # OCR 사용
        reader = get_ocr_reader()
        
        for p in range(start_page - 1, end_page):
            page = doc.load_page(p)
            # 페이지를 이미지로 변환
            pix = page.get_pixmap(dpi=300)  # 높은 해상도로 OCR 정확도 향상
            img_data = pix.tobytes("png")
            
            # PIL Image로 변환
            img = Image.open(io.BytesIO(img_data))
            img_array = np.array(img)
            
            # OCR 수행
            try:
                result = reader.readtext(img_array)
                page_text = " ".join([detection[1] for detection in result])
                texts.append(page_text)
            except Exception as e:
                st.warning(f"페이지 {p+1} OCR 처리 중 오류: {e}")
                texts.append("")
    else:
        # 기본 텍스트 추출
        for p in range(start_page - 1, end_page):
            page = doc.load_page(p)
            texts.append(page.get_text("text"))
    
    doc.close()  # 문서 닫기
    return "\n".join(texts)

# 이미지 변환 함수 정의 (수정된 버전)
def convert_pdf_to_images(pdf_data):
    document = fitz.open(stream=pdf_data, filetype="pdf")
    images = []
    
    # 임시 디렉토리 생성
    temp_dir = tempfile.mkdtemp()
    
    for page_num in range(len(document)):
        page = document[page_num]
        pix = page.get_pixmap(dpi=150)        # 이미지 생성
        img_path = os.path.join(temp_dir, f"page_{page_num+1}.png")   # 이미지 저장 경로 설정
        pix.save(img_path)                    # 이미지 저장
        images.append(img_path)               # 이미지 저장 경로 리스트에 추가
    
    document.close()  # 문서 닫기
    return images

# 텍스트 추출 방법 자동 감지
def detect_text_extraction_method(pdf_bytes):
    """
    PDF에서 추출 가능한 텍스트가 있는지 확인하여 OCR 필요 여부 판단
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    # 첫 3페이지 정도만 체크
    pages_to_check = min(3, doc.page_count)
    total_text_length = 0
    
    for p in range(pages_to_check):
        page = doc.load_page(p)
        text = page.get_text("text").strip()
        total_text_length += len(text)
    
    doc.close()
    
    # 평균적으로 페이지당 50자 미만이면 OCR 필요할 가능성이 높음
    avg_text_per_page = total_text_length / pages_to_check
    return avg_text_per_page < 50

# 메인 화면 구성
def main():
    st.set_page_config(layout="wide", page_title="PDF OCR 번역/요약 프로그램")
    
    with st.sidebar:
        st.title("PDF OCR 번역/요약 프로그램")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        st.write("[OpenAI API Key 받기](https://platform.openai.com/account/api-keys)")
        
        # 파일 업로드 위젯 생성
        pdf_file = st.file_uploader("PDF 파일을 업로드 하세요.", type=["pdf"])
        mode = st.radio("페이지 선택 모드", ["단일 페이지", "페이지 범위", "전체 문서"])
        
        # OCR 옵션
        st.subheader("텍스트 추출 옵션")
        extraction_method = st.radio(
            "추출 방법",
            ["자동 감지", "일반 텍스트 추출", "OCR 사용"],
            help="자동 감지: PDF 내용을 분석하여 최적의 방법 선택\n일반 텍스트 추출: PDF에 포함된 텍스트 직접 추출\nOCR 사용: 이미지를 텍스트로 변환"
        )
        
        # 주요 세션 상태 초깃값 설정
        if "images" not in st.session_state:
            st.session_state.images = []
        if "page_number" not in st.session_state:
            st.session_state.page_number = 1
        if "start_page" not in st.session_state:
            st.session_state.start_page = 1           
        if "end_page" not in st.session_state:
            st.session_state.end_page = 1    

        client = None

        # OpenAI 클라이언트 생성
        if openai_api_key:
            try:
                client = OpenAI(api_key=openai_api_key)
            except Exception as e:
                st.error(f"OpenAI 클라이언트 생성 오류: {e}")

        if pdf_file:
            try:
                pdf_data = pdf_file.read()
                st.session_state.images = convert_pdf_to_images(pdf_data)
                total_pages = len(st.session_state.images)

                if mode == "단일 페이지":
                    st.session_state.page_number = st.number_input(
                        "페이지 번호 선택",
                        min_value=1,
                        max_value=total_pages,
                        value=st.session_state.page_number,
                    )
                elif mode == "페이지 범위":
                    st.session_state.start_page = st.number_input(
                        "시작 페이지",
                        min_value=1,
                        max_value=total_pages,
                        value=st.session_state.start_page,
                    )
                    st.session_state.end_page = st.number_input(
                        "끝 페이지",
                        min_value=st.session_state.start_page,
                        max_value=total_pages,
                        value=min(st.session_state.end_page, total_pages),
                    )
                    
                # 자동 감지 결과 표시
                if extraction_method == "자동 감지":
                    pdf_file.seek(0)
                    pdf_data_check = pdf_file.read()
                    needs_ocr = detect_text_extraction_method(pdf_data_check)
                    if needs_ocr:
                        st.info("📷 스캔된 문서로 감지됨 - OCR 사용 권장")
                    else:
                        st.info("📄 텍스트 PDF로 감지됨 - 일반 추출 사용")
                        
            except Exception as e:
                st.error(f"PDF 처리 오류: {e}")
                return

    if pdf_file:
        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.subheader("미리 보기")
            try:
                if mode == "단일 페이지":
                    if st.session_state.images and 0 <= st.session_state.page_number-1 < len(st.session_state.images):
                        st.image(
                            st.session_state.images[st.session_state.page_number-1],
                            caption=f"Page {st.session_state.page_number}",
                            use_container_width=True,
                        )
                elif mode == "페이지 범위":
                    for idx in range(st.session_state.start_page-1, min(st.session_state.end_page, len(st.session_state.images))):
                        if 0 <= idx < len(st.session_state.images):
                            st.image(st.session_state.images[idx], caption=f"Page {idx+1}", use_container_width=True)
                else:  # 전체 문서
                    for idx, img in enumerate(st.session_state.images):
                        st.image(img, caption=f"Page {idx+1}", use_container_width=True)
            except Exception as e:
                st.error(f"이미지 표시 오류: {e}")

        with right_col:
            st.subheader("텍스트 추출")
            try:
                pdf_file.seek(0)  # 파일 포인터를 처음으로 되돌림
                pdf_data = pdf_file.read()
                
                # 추출 방법 결정
                use_ocr = False
                if extraction_method == "OCR 사용":
                    use_ocr = True
                elif extraction_method == "자동 감지":
                    use_ocr = detect_text_extraction_method(pdf_data)
                
                # 텍스트 추출
                if mode == "단일 페이지":
                    pdf_text = get_text_from_pages(pdf_data, st.session_state.page_number, st.session_state.page_number, use_ocr)
                elif mode == "페이지 범위":
                    pdf_text = get_text_from_pages(pdf_data, st.session_state.start_page, st.session_state.end_page, use_ocr)
                else:  # 전체 문서
                    pdf_text = get_text_from_pages(pdf_data, use_ocr=use_ocr)

                # 추출 방법 표시
                method_text = "🔍 OCR" if use_ocr else "📄 일반 텍스트 추출"
                st.caption(f"사용된 추출 방법: {method_text}")
                
                st.text_area("추출된 텍스트", value=pdf_text, height=300, disabled=True)

                start_prompt_summary = """다음 문서를 개조식으로 요약하되 한글로 번역해 주세요.
- ~음, ~했음 어조 사용
- 핵심 내용 중심으로 간결하게
- 마크다운 형식으로 구조화"""
                
                start_prompt_translation = """다음 문서를 한글로 정확하게 전문 번역해 주세요.
- 원문의 의미를 최대한 유지
- 학술/전문 문체 사용"""

                prompt_summary = st.text_area("요약 프롬프트", value=start_prompt_summary, height=150)
                prompt_translation = st.text_area("번역 프롬프트", value=start_prompt_translation, height=150)

                if st.button("번역 & 요약 실행"):
                    if not client:
                        st.error("유효한 API Key를 입력하세요.")
                    elif not pdf_text.strip():
                        st.error("추출할 텍스트가 없습니다. PDF에 텍스트가 포함되어 있는지 확인하세요.")
                    else:
                        with st.spinner("처리 중입니다..."):
                            try:
                                # 요약
                                summary_response = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[{"role": "user", "content": prompt_summary + "\n\n" + pdf_text}],
                                    max_tokens=2000,
                                    temperature=0.3
                                )
                                summary_result = summary_response.choices[0].message.content

                                # 전문 번역
                                translation_response = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[{"role": "user", "content": prompt_translation + "\n\n" + pdf_text}],
                                    max_tokens=4000,
                                    temperature=0.1
                                )
                                translation_result = translation_response.choices[0].message.content

                                st.subheader("✂️ 요약 결과")
                                st.markdown(summary_result)
                                
                                st.subheader("📜 전문 번역 결과")
                                st.markdown(translation_result)

                            except Exception as e:
                                st.error(f"AI 처리 오류: {e}")
                                
            except Exception as e:
                st.error(f"텍스트 추출 오류: {e}")

if __name__ == "__main__":
    main()
