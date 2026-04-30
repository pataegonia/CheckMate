import re
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


_easyocr_reader = None
_trocr_processor = None
_trocr_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"


def _get_easyocr():
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["ko", "en"], gpu=(_device == "cuda"))
    return _easyocr_reader


def _get_trocr():
    global _trocr_processor, _trocr_model
    if _trocr_processor is None:
        model_id = "microsoft/trocr-base-handwritten"
        _trocr_processor = TrOCRProcessor.from_pretrained(model_id)
        _trocr_model = VisionEncoderDecoderModel.from_pretrained(model_id).to(_device)
        _trocr_model.eval()
    return _trocr_processor, _trocr_model


def _anchors_from_results(results, valid_keys: set) -> list[dict]:
    anchors = []
    for box, text, conf in results:
        for m in re.finditer(r"\d+", text):
            n = int(m.group())
            if n in valid_keys:
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                anchors.append(
                    {
                        "num": n,
                        "x_left": int(min(xs)),
                        "x_right": int(max(xs)),
                        "y_top": int(min(ys)),
                        "y_bottom": int(max(ys)),
                        "conf": float(conf),
                        "text": text,
                    }
                )
    return anchors


def _dedup_anchors(anchors: list[dict]) -> list[dict]:
    by_num: dict[int, dict] = {}
    for a in anchors:
        prev = by_num.get(a["num"])
        if prev is None or a["conf"] > prev["conf"]:
            by_num[a["num"]] = a
    return list(by_num.values())


def find_anchors(image_bgr: np.ndarray, valid_keys: set) -> tuple[list[dict], list]:
    """페이지 전체를 OCR해서 valid_keys 안의 문제번호 위치를 모두 찾는다.
    1차에서 빠진 게 있으면 더 민감한 옵션으로 재시도.
    반환: (anchors, all_ocr_results)"""
    reader = _get_easyocr()

    pass1 = reader.readtext(image_bgr)
    anchors = _anchors_from_results(pass1, valid_keys)
    all_results = list(pass1)

    found = {a["num"] for a in anchors}
    if valid_keys - found:
        pass2 = reader.readtext(
            image_bgr, text_threshold=0.5, low_text=0.3, link_threshold=0.3
        )
        more = _anchors_from_results(pass2, valid_keys)
        for a in more:
            if a["num"] not in found:
                anchors.append(a)
                found.add(a["num"])
        all_results.extend(pass2)

    return _dedup_anchors(anchors), all_results


def visualize_debug(image_bgr: np.ndarray, anchors: list[dict], all_ocr) -> np.ndarray:
    """앵커(파란 박스 + 번호) + 모든 OCR 박스(노란 얇은 선)를 그린 디버그 이미지."""
    img = image_bgr.copy()
    for box, text, conf in all_ocr:
        pts = np.array(box, dtype=np.int32)
        cv2.polylines(img, [pts], True, (0, 200, 200), 1)
    for a in anchors:
        cv2.rectangle(
            img,
            (a["x_left"] - 4, a["y_top"] - 4),
            (a["x_right"] + 4, a["y_bottom"] + 4),
            (255, 0, 0),
            3,
        )
        cv2.putText(
            img,
            str(a["num"]),
            (a["x_left"], max(20, a["y_top"] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
        )
    return img


def cluster_columns(anchors: list[dict], page_width: int) -> list[tuple[int, int]]:
    """앵커들의 x 좌표로 단(column) 경계 추정. 1단/2단까지 지원."""
    if len(anchors) <= 1:
        return [(0, page_width)]

    xs = sorted(a["x_left"] for a in anchors)
    gaps = [(xs[i + 1] - xs[i], xs[i], xs[i + 1]) for i in range(len(xs) - 1)]
    biggest = max(gaps, key=lambda g: g[0])

    if biggest[0] > page_width * 0.20:
        margin = max(20, int(page_width * 0.02))
        split = max(biggest[1] + margin, biggest[2] - margin)
        return [(0, split), (split, page_width)]
    return [(0, page_width)]


_PRINTED_HEAD_PATTERNS = [
    r"^\d{3,4}$",                  # 0815 같은 문제번호
    r"^[①-⑳]+$",                   # ①~⑳
    r"^[ㄱ-ㅎ][.,]?$",              # ㄱ. ㄴ. ㄷ.
    r"보기",
]


def _looks_like_handwritten_answer(text: str) -> bool:
    """간단한 휴리스틱: 짧고, 답 같은 문자(숫자/부등호/x 등)를 포함하고, 본문스러운 한글 덩어리는 아님."""
    t = text.strip()
    if not t or len(t) > 12:
        return False
    if re.search(r"[가-힣]{2,}", t):
        return False
    for pat in _PRINTED_HEAD_PATTERNS:
        if re.fullmatch(pat, t):
            return False
    if re.search(r"[\d=<>≤≥+\-xX/.,]", t):
        return True
    return False


def extract_handwritten_answer(crop_bgr: np.ndarray) -> str:
    """크롭 안의 모든 텍스트를 EasyOCR로 읽고, 답으로 보이는 후보만 골라서 합쳐 반환."""
    reader = _get_easyocr()
    results = reader.readtext(crop_bgr)

    h, w = crop_bgr.shape[:2]
    candidates = []
    for box, text, conf in results:
        cx = sum(p[0] for p in box) / 4
        cy = sum(p[1] for p in box) / 4
        if cx < w * 0.10 and cy < h * 0.20:
            continue
        if _looks_like_handwritten_answer(text):
            candidates.append((cy, cx, text.strip(), conf))

    if not candidates:
        return ""

    candidates.sort(key=lambda c: (-c[3], c[0]))
    return candidates[0][2]


def extract_handwritten_answer_trocr(crop_bgr: np.ndarray) -> str:
    """[참고용] 영문 손글씨 TrOCR. 한글/수식엔 환각하므로 기본 파이프라인에선 안 씀."""
    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    processor, model = _get_trocr()
    pixel_values = processor(images=pil, return_tensors="pt").pixel_values.to(_device)
    with torch.no_grad():
        generated = model.generate(pixel_values, max_length=32)
    text = processor.batch_decode(generated, skip_special_tokens=True)[0]
    return text.strip()


def normalize(s) -> str:
    s = str(s).strip().lower()
    s = s.replace("−", "-").replace(" ", "")
    return s


def grade_page(image_rgb: np.ndarray, answer_dict: dict, return_debug: bool = False):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    h, w = image_bgr.shape[:2]
    valid_keys = set(answer_dict.keys())

    anchors, all_ocr = find_anchors(image_bgr, valid_keys)
    if not anchors:
        if return_debug:
            return [], {"all_ocr": all_ocr, "anchors": [], "debug_image": image_rgb}
        return []

    cols = cluster_columns(anchors, w)

    by_col: dict[int, list[dict]] = defaultdict(list)
    for a in anchors:
        cx = (a["x_left"] + a["x_right"]) / 2
        col_idx = 0
        for i, (cx1, cx2) in enumerate(cols):
            if cx1 <= cx < cx2:
                col_idx = i
                break
        by_col[col_idx].append(a)

    for col_idx in by_col:
        by_col[col_idx].sort(key=lambda a: a["y_top"])

    results = []
    for col_idx, col_anchors in by_col.items():
        cx1, cx2 = cols[col_idx]
        for j, a in enumerate(col_anchors):
            y1 = max(0, a["y_top"] - 5)
            if j + 1 < len(col_anchors):
                y2 = max(y1 + 1, col_anchors[j + 1]["y_top"] - 5)
            else:
                y2 = h
            crop = image_bgr[y1:y2, cx1:cx2]

            student_ans = extract_handwritten_answer(crop)
            correct_ans = answer_dict.get(a["num"])
            is_correct = normalize(student_ans) == normalize(correct_ans)

            results.append(
                {
                    "problem_num": a["num"],
                    "bbox": (cx1, y1, cx2, y2),
                    "student_answer": student_ans,
                    "correct_answer": str(correct_ans),
                    "correct": is_correct,
                    "crop_rgb": cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                }
            )

    out = sorted(results, key=lambda r: r["problem_num"])
    if return_debug:
        debug_img_bgr = visualize_debug(image_bgr, anchors, all_ocr)
        return out, {
            "all_ocr": all_ocr,
            "anchors": anchors,
            "columns": cols,
            "debug_image": cv2.cvtColor(debug_img_bgr, cv2.COLOR_BGR2RGB),
        }
    return out
