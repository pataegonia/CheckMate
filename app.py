import io
import os

import numpy as np
import streamlit as st
from PIL import Image

from ai_grader import (
    BedrockConverseGrader,
    HybridAIGrader,
    MockAIGrader,
    build_submission_payload,
    dump_api_json,
)
from answers import ANSWERS, QUESTION_SPECS
from pipeline import grade_page, normalize


st.set_page_config(page_title="수학 문제집 채점 데모", layout="wide")
st.title("📐 수학 문제집 채점 데모")
st.caption("페이지 사진/PDF를 올리면 문제별로 분할 → 답 인식 → 정답과 비교해 점수를 보여줍니다.")

# 사용자 수정값을 페이지/문제별로 저장
if "manual_overrides" not in st.session_state:
    st.session_state.manual_overrides = {}  # (page_idx, problem_num) -> override text

with st.sidebar:
    st.subheader("저장된 정답 (서버 흉내)")
    for k, v in sorted(ANSWERS.items()):
        qtype = QUESTION_SPECS.get(k, {}).get("type", "short_answer")
        st.write(f"{k}번: `{v}` · {qtype}")
    st.divider()
    pdf_dpi = st.slider("PDF 변환 해상도 (DPI)", 100, 400, 200, 50)
    debug_mode = st.checkbox("🔍 디버그 보기 (검출된 OCR/앵커 시각화)", value=False)
    ai_mode = st.radio(
        "AI 채점 방식",
        [
            "빠른 모드: OCR/rule",
            "하이브리드: 서답형 오답/불확실/서술형만 Claude",
            "오답 전체 Claude 재검토",
            "서답형+서술형 Claude",
            "전체 Claude",
        ],
        index=0,
    )
    default_model_id = os.getenv("BEDROCK_MODEL_ID") or "us.anthropic.claude-sonnet-4-6"
    bedrock_model_id = st.text_input("Bedrock model ID", value=default_model_id)
    has_aws_creds = bool(
        os.getenv("AWS_PROFILE")
        or (os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
    )
    st.caption(
        f"Region: `{os.getenv('AWS_REGION') or os.getenv('AWS_DEFAULT_REGION') or 'us-east-1'}` · "
        f"AWS creds: `{'OK' if has_aws_creds else 'missing'}`"
    )
    if ai_mode != "빠른 모드: OCR/rule" and not has_aws_creds:
        st.warning("AWS credentials가 없어 Claude 호출 대신 mock 결과가 표시됩니다.")
    if st.button("✏️ 수동 수정 모두 초기화"):
        st.session_state.manual_overrides = {}
        st.rerun()
    if st.button("OCR 캐시 비우기"):
        st.cache_data.clear()
        st.rerun()


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


def image_to_png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


@st.cache_data(show_spinner=False)
def grade_page_cached(image_bytes: bytes, answer_items: tuple, return_debug: bool):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    answer_dict = dict(answer_items)
    if return_debug:
        return grade_page(image_np, answer_dict, return_debug=True)
    return grade_page(image_np, answer_dict), None


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

def _conf_badge(conf: float) -> str:
    """신뢰도를 시각적 뱃지로."""
    if conf < 0:
        return "❓ 알수없음"
    if conf >= 0.85:
        return f"🟢 높음 ({conf:.2f})"
    if conf >= 0.6:
        return f"🟡 보통 ({conf:.2f})"
    return f"🔴 낮음 ({conf:.2f})"


def _source_label(src: str) -> str:
    return {
        "circle": "🎯 객관식 동그라미",
        "trailing": "📝 본문 끝 답",
        "paddle": "PaddleOCR",
        "easy": "EasyOCR",
        "v1-fallback": "휴리스틱 폴백",
        "none": "없음",
    }.get(src, src)


def _type_label(qtype: str) -> str:
    return {
        "multiple_choice": "객관식",
        "short_answer": "서답형",
        "descriptive": "서술형",
    }.get(qtype, qtype)


def _grade_ai_payload(ai_payload):
    if ai_mode == "전체 Claude":
        try:
            return (
                BedrockConverseGrader(model_id=bedrock_model_id).grade_submission(ai_payload),
                "Bedrock/Claude 실제 호출 결과입니다.",
                None,
            )
        except Exception as e:
            return (
                MockAIGrader().grade_submission(ai_payload),
                "Bedrock 호출에 실패해 로컬 mock 결과를 표시합니다.",
                str(e),
            )

    if ai_mode in {
        "하이브리드: 서답형 오답/불확실/서술형만 Claude",
        "오답 전체 Claude 재검토",
        "서답형+서술형 Claude",
    }:
        if ai_mode == "서답형+서술형 Claude":
            mode = "short_descriptive"
        elif ai_mode == "오답 전체 Claude 재검토":
            mode = "review_incorrect"
        else:
            mode = "review_needed"
        try:
            return (
                HybridAIGrader(
                    mode=mode,
                    remote_grader=BedrockConverseGrader(model_id=bedrock_model_id),
                ).grade_submission(ai_payload),
                "쉬운 문항은 OCR/rule로 처리하고 필요한 문항만 Claude를 호출했습니다.",
                None,
            )
        except Exception as e:
            return (
                MockAIGrader().grade_submission(ai_payload),
                "Bedrock 호출에 실패해 로컬 mock 결과를 표시합니다.",
                str(e),
            )

    return (
        MockAIGrader().grade_submission(ai_payload),
        "빠른 OCR/rule 결과입니다. 사이드바에서 Claude 호출 범위를 선택할 수 있습니다.",
        None,
    )


page_records = []
all_results = []
with st.spinner("OCR/문항 분할 중... 같은 입력은 캐시됩니다."):
    for page_idx, image in enumerate(pages):
        page_bytes = image_to_png_bytes(image)
        results, debug = grade_page_cached(
            page_bytes,
            tuple(sorted(ANSWERS.items())),
            debug_mode,
        )

        for r in results:
            r["question_id"] = f"{page_idx + 1}-{r['problem_num']}"
            key = (page_idx, r["problem_num"])
            if key in st.session_state.manual_overrides:
                r["student_answer_override"] = st.session_state.manual_overrides[key]
                r["correct"] = (
                    normalize(r["student_answer_override"])
                    == normalize(r["correct_answer"])
                )

        page_records.append(
            {"page_idx": page_idx, "image": image, "results": results, "debug": debug}
        )
        all_results.extend(results)

ai_grade = None
ai_caption = ""
ai_error = None
ai_by_question_id = {}
if all_results:
    ai_payload = build_submission_payload(all_results, question_specs=QUESTION_SPECS)
    with st.spinner("AI 채점 결과 계산 중..."):
        ai_grade, ai_caption, ai_error = _grade_ai_payload(ai_payload)
    ai_by_question_id = {q.question_id: q for q in ai_grade.questions}


for record in page_records:
    page_idx = record["page_idx"]
    image = record["image"]
    results = record["results"]
    debug = record["debug"]

    st.divider()
    st.header(f"📄 페이지 {page_idx + 1}")

    col_in, col_out = st.columns([1, 1])
    with col_in:
        st.subheader("입력")
        st.image(image, use_container_width=True)

    with col_out:
        st.subheader("채점 결과")

        if not results:
            st.warning("이 페이지에선 문제를 찾지 못했어요. 분할이 잘 되도록 문제 사이 여백을 늘려보세요.")
        else:
            if ai_by_question_id:
                correct = sum(
                    1
                    for r in results
                    if (ai_q := ai_by_question_id.get(r["question_id"])) is not None
                    and ai_q.is_correct
                )
            else:
                correct = sum(1 for r in results if r["correct"])
            total = len(results)
            st.metric("이 페이지 점수", f"{correct} / {total}")

            found_nums = {r["problem_num"] for r in results}
            missing = sorted(set(ANSWERS.keys()) - found_nums)
            if missing:
                st.error(f"⚠️ 못 찾은 문제번호: {missing}")

            for r in results:
                ai_q = ai_by_question_id.get(r["question_id"])
                display_correct = ai_q.is_correct if ai_q else r["correct"]
                icon = "✅" if display_correct else "❌"
                key = (page_idx, r["problem_num"])
                manual = "student_answer_override" in r
                ocr_effective = r.get("student_answer_override") or r["student_answer"]
                effective = (
                    ai_q.extracted_answer
                    if ai_q and ai_q.extracted_answer
                    else ocr_effective
                )
                manual_tag = " ✏️ 수동수정" if manual else ""

                with st.container(border=True):
                    spec = QUESTION_SPECS.get(r["problem_num"], {})
                    qtype = spec.get("type", "short_answer")
                    cols = st.columns([3, 2])
                    with cols[0]:
                        st.markdown(
                            f"**{icon} {r['problem_num']}번** · {_type_label(qtype)}{manual_tag}\n\n"
                            f"학생 답: `{effective or '(인식 실패)'}`  /  정답: `{r['correct_answer']}`"
                        )
                        if ai_q and ai_q.extracted_answer != ocr_effective:
                            st.caption(f"OCR 원문: `{ocr_effective or '(없음)'}`")
                        if not manual:
                            st.caption(
                                f"신뢰도 {_conf_badge(r.get('confidence', -1))} · "
                                f"출처 {_source_label(r.get('source', 'none'))}"
                            )
                        if ai_q:
                            st.caption(
                                f"AI 판정: {ai_q.source} · "
                                f"conf={ai_q.confidence:.2f} · {ai_q.reason}"
                            )
                    with cols[1]:
                        new_val = st.text_input(
                            "수동 수정",
                            value=st.session_state.manual_overrides.get(key, ""),
                            placeholder="비우면 원래 OCR 결과 사용",
                            key=f"override_{page_idx}_{r['problem_num']}",
                        )
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("적용", key=f"apply_{page_idx}_{r['problem_num']}"):
                                if new_val.strip():
                                    st.session_state.manual_overrides[key] = new_val.strip()
                                else:
                                    st.session_state.manual_overrides.pop(key, None)
                                st.rerun()
                        with c2:
                            if manual and st.button("취소", key=f"reset_{page_idx}_{r['problem_num']}"):
                                st.session_state.manual_overrides.pop(key, None)
                                st.rerun()

                    cands = r.get("candidates") or []
                    if cands:
                        with st.expander(f"OCR 후보 {len(cands)}개"):
                            for c in cands:
                                st.write(
                                    f"- `{c['text']}` "
                                    f"(conf={c['conf']:.2f} · {_source_label(c['source'])})"
                                )
                    focus_crop = r.get("answer_crop_rgb")
                    if focus_crop is not None:
                        with st.expander("답안 확대 crop"):
                            st.image(focus_crop, use_container_width=True)
                    st.image(r["crop_rgb"], use_container_width=True)

    if debug and debug.get("debug_image") is not None:
        st.divider()
        st.subheader("🔍 디버그: 페이지 전체 OCR/앵커 시각화")
        st.caption("파란 박스 = 정답 dict와 매칭된 문제번호 앵커 / 노란 얇은 박스 = EasyOCR이 검출한 모든 텍스트")
        st.image(debug["debug_image"], use_container_width=True)
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

if all_results:
    st.divider()
    st.header("AI 채점 JSON")
    if ai_error:
        st.error(f"Bedrock 호출 실패: {ai_error}")
    st.caption(ai_caption)

    with st.expander("백엔드 전달용 JSON", expanded=False):
        st.code(dump_api_json(ai_grade), language="json")
