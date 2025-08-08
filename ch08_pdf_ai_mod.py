import streamlit as st
import pymupdf
import fitz  # PyMuPDF의 별칭
from openai import OpenAI
import os
import tempfile

# 텍스트 추출 함수 정의
def get_text_from_page(pdf_data, start_page=None, end_page=None):
    document = pymupdf.open(stream=pdf_data, filetype="pdf") 
    if start_page is None or end_page is None:
        start_page = 1
        end_page = len(document)
    text = ""
    for page_number in range(start_page-1, end_page):
        page = document[page_number]
        text += page.get_text() + "\n"
    document.close()  # 문서 닫기 추가
    return text

# 이미지 변환 함수 정의
def convert_pdf_to_images(pdf_data):
    document = pymupdf.open(stream=pdf_data, filetype="pdf")
    images = []
    
    # 임시 디렉토리 생성
    temp_dir = tempfile.mkdtemp()
    
    for page_num in range(len(document)):
        page = document[page_num]
        pix = page.get_pixmap(dpi=150)        # 이미지 생성
        img_path = os.path.join(temp_dir, f"page_{page_num+1}.png")  # 임시 디렉토리 사용
        pix.save(img_path)                    # 이미지 저장
        images.append(img_path)               # 이미지 저장 경로 리스트에 추가
    
    document.close()  # 문서 닫기 추가
    return images        

def get_text_from_pages(pdf_bytes, start_page=None, end_page=None):
    """
    pdf_bytes에서 지정된 페이지 범위의 텍스트를 추출.
    start_page와 end_page는 1부터 시작하는 페이지 번호.
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
    for p in range(start_page - 1, end_page):
        page = doc.load_page(p)
        texts.append(page.get_text("text"))
    
    doc.close()  # 문서 닫기 추가
    return "\n".join(texts)


# 메인 화면 구성
def main():
    st.set_page_config(layout="wide")
    
    with st.sidebar:
        st.title("PDF 번역/요약 프로그램")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        st.write("[OpenAI API Key 받기](https://platform.openai.com/account/api-keys)")
        
        # (2) 파일 업로드 위젯 생성
        pdf_file = st.file_uploader("PDF 파일을 업로드 하세요.", type=["pdf"])
        mode = st.radio("페이지 선택 모드", ["단일 페이지", "페이지 범위", "전체 문서"])
        
        # (3) 주요 세션 상태 초깃값 설정
        if "images" not in st.session_state:
            st.session_state.images = []
        if "page_number" not in st.session_state:
            st.session_state.page_number = 1
        if "start_page" not in st.session_state:
            st.session_state.start_page = 1           
        if "end_page" not in st.session_state:
            st.session_state.end_page = 1    

        client = None

        # (11) OpenAI 클라이언트 생성
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
                        value=min(st.session_state.page_number, total_pages),  # 범위 안전성 확보
                        key="single_page_input"
                    )
                elif mode == "페이지 범위":
                    st.session_state.start_page = st.number_input(
                        "시작 페이지",
                        min_value=1,
                        max_value=total_pages,
                        value=min(st.session_state.start_page, total_pages),
                        key="start_page_input"
                    )
                    st.session_state.end_page = st.number_input(
                        "끝 페이지",
                        min_value=st.session_state.start_page,
                        max_value=total_pages,
                        value=min(st.session_state.end_page, total_pages),
                        key="end_page_input"
                    )
            except Exception as e:
                st.error(f"PDF 파일 처리 오류: {e}")

    # 메인 컨텐츠 영역
    if pdf_file and st.session_state.images:
        left_col, right_col = st.columns([1, 1])

        with left_col:
            st.subheader("미리 보기")
            try:
                if mode == "단일 페이지":
                    if os.path.exists(st.session_state.images[st.session_state.page_number-1]):
                        st.image(
                            st.session_state.images[st.session_state.page_number-1],
                            caption=f"Page {st.session_state.page_number}",
                            use_container_width=True,
                        )
                elif mode == "페이지 범위":
                    for idx in range(st.session_state.start_page-1, st.session_state.end_page):
                        if idx < len(st.session_state.images) and os.path.exists(st.session_state.images[idx]):
                            st.image(st.session_state.images[idx], caption=f"Page {idx+1}", use_container_width=True)
                else:  # 전체 문서
                    for idx, img in enumerate(st.session_state.images):
                        if os.path.exists(img):
                            st.image(img, caption=f"Page {idx+1}", use_container_width=True)
            except Exception as e:
                st.error(f"이미지 표시 오류: {e}")

        with right_col:
            st.subheader("텍스트 추출")
            try:
                pdf_data = pdf_file.getvalue()  # read() 대신 getvalue() 사용
                
                if mode == "단일 페이지":
                    pdf_text = get_text_from_pages(pdf_data, st.session_state.page_number, st.session_state.page_number)
                elif mode == "페이지 범위":
                    pdf_text = get_text_from_pages(pdf_data, st.session_state.start_page, st.session_state.end_page)
                else:
                    pdf_text = get_text_from_pages(pdf_data)

                st.text_area("추출된 텍스트", pdf_text, height=200, disabled=True)

                start_prompt_summary = """다음 문서를 개조식으로 요약하되 한글로 번역해 주세요.
- ~음, ~했음 어조 사용
- 핵심 내용 중심으로 간결하게
- 마크다운 형식으로 구조화"""
                
                start_prompt_translation = """다음 문서를 한글로 정확하게 전문 번역해 주세요.
- 원문의 의미를 최대한 유지
- 학술/전문 문체 사용"""

                prompt_summary = st.text_area("요약 프롬프트", value=start_prompt_summary, height=100)
                prompt_translation = st.text_area("번역 프롬프트", value=start_prompt_translation, height=100)

                if st.button("번역 & 요약 실행"):
                    if not client:
                        st.error("유효한 API Key를 입력하세요.")
                    elif not pdf_text.strip():
                        st.error("추출할 텍스트가 없습니다. PDF 파일을 다시 확인해주세요.")
                    else:
                        with st.spinner("처리 중입니다..."):
                            try:
                                # 요약
                                summary_response = client.chat.completions.create(
                                    model="gpt-4o-mini",
                                    messages=[{"role": "user", "content": prompt_summary + "\n\n" + pdf_text}],
                                    max_tokens=4000,
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
                                st.error(f"OpenAI API 호출 오류: {e}")
                                
            except Exception as e:
                st.error(f"텍스트 추출 오류: {e}")

if __name__ == "__main__":
    main()