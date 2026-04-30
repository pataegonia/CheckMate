import io

import numpy as np
import streamlit as st
from PIL import Image

from answers import ANSWERS
from pipeline import grade_page


st.set_page_config(page_title="수학 문제집 채점 데모", layout="wide")
st.title("📐 수학 문제집 채점 데모")
st.caption("페이지 사진/PDF를 올리면 문제별로 분할 → 답 인식 → 정답과 비교해 점수를 보여줍니다.")

with st.sidebar:
    st.subheader("저장된 정답 (서버 흉내)")
    for k, v in sorted(ANSWERS.items()):
        st.write(f"{k}번: `{v}`")
    st.divider()
    pdf_dpi = st.slider("PDF 변환 해상도 (DPI)", 100, 400, 200, 50)
    debug_mode = st.checkbox("🔍 디버그 보기 (검출된 OCR/앵커 시각화)", value=False)


def pdf_to_images(file_bytes: bytes, dpi: int) -> list[Image.Image]:
    import fitz
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    images = []
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        images.append(img)
    doc.close()
    return images


uploaded = st.file_uploader("페이지 이미지/PDF 업로드", type=["jpg", "jpeg", "png", "pdf"])

if uploaded is None:
    st.info("samples/ 안의 페이지 사진(또는 PDF)을 업로드해보세요.")
    st.stop()

is_pdf = uploaded.name.lower().endswith(".pdf")
if is_pdf:
    with st.spinner("PDF를 페이지 이미지로 변환 중..."):
        pages = pdf_to_images(uploaded.read(), dpi=pdf_dpi)
    st.success(f"PDF에서 {len(pages)}페이지 추출됨")
else:
    pages = [Image.open(uploaded).convert("RGB")]

all_results = []
for page_idx, image in enumerate(pages):
    st.divider()
    st.header(f"📄 페이지 {page_idx + 1}")

    image_np = np.array(image)

    col_in, col_out = st.columns([1, 1])
    with col_in:
        st.subheader("입력")
        st.image(image, use_column_width=True)

    with col_out:
        st.subheader("채점 결과")
        with st.spinner("모델 로딩 + 채점 중... (최초 실행은 모델 다운로드로 시간이 좀 걸려요)"):
            if debug_mode:
                results, debug = grade_page(image_np, ANSWERS, return_debug=True)
            else:
                results = grade_page(image_np, ANSWERS)
                debug = None
        all_results.extend(results)

        if not results:
            st.warning("이 페이지에선 문제를 찾지 못했어요. 분할이 잘 되도록 문제 사이 여백을 늘려보세요.")
        else:
            correct = sum(1 for r in results if r["correct"])
            total = len(results)
            st.metric("이 페이지 점수", f"{correct} / {total}")

            found_nums = {r["problem_num"] for r in results}
            missing = sorted(set(ANSWERS.keys()) - found_nums)
            if missing:
                st.error(f"⚠️ 못 찾은 문제번호: {missing} (페이지 OCR이 헤더를 놓쳤거나 분할 실패)")

            for r in results:
                icon = "✅" if r["correct"] else "❌"
                st.markdown(
                    f"**{icon} {r['problem_num']}번** — 학생 답: `{r['student_answer']}` / 정답: `{r['correct_answer']}`"
                )
                st.image(r["crop_rgb"], use_column_width=True)

    if debug and debug.get("debug_image") is not None:
        st.divider()
        st.subheader("🔍 디버그: 페이지 전체 OCR/앵커 시각화")
        st.caption("파란 박스 = 정답 dict와 매칭된 문제번호 앵커 / 노란 얇은 박스 = EasyOCR이 검출한 모든 텍스트")
        st.image(debug["debug_image"], use_column_width=True)
        with st.expander(f"검출된 앵커 {len(debug['anchors'])}개"):
            for a in sorted(debug["anchors"], key=lambda a: a["num"]):
                st.write(
                    f"- **{a['num']}** at (x={a['x_left']}~{a['x_right']}, y={a['y_top']}~{a['y_bottom']}) "
                    f"— 인식 텍스트: `{a.get('text', '')}` / conf={a['conf']:.2f}"
                )
        with st.expander(f"전체 OCR 결과 {len(debug['all_ocr'])}건"):
            for box, text, conf in debug["all_ocr"]:
                if text.strip():
                    st.write(f"- `{text}` (conf={conf:.2f})")

if len(pages) > 1 and all_results:
    st.divider()
    total_correct = sum(1 for r in all_results if r["correct"])
    total_n = len(all_results)
    st.header(f"🏁 전체 합계: {total_correct} / {total_n}")
