"""EasyOCR vs PaddleOCR(한국어) 비교 — 손글씨 답 크롭에서.

사용:
    python bench_paddle.py debug_out/page_1_problem_*.png
"""
import re
import sys
from pathlib import Path

import cv2
import numpy as np

from answers import ANSWERS

ANSWER_ALLOWLIST = "0123456789xXyzabc=<>+-≤≥≠/.,()"


def normalize(s):
    if s is None:
        return ""
    return str(s).strip().lower().replace("−", "-").replace(" ", "")


def get_easyocr():
    import easyocr
    return easyocr.Reader(["ko", "en"], gpu=False)


def easyocr_top_text(reader, img_bgr, allowlist=None):
    kwargs = {"allowlist": allowlist} if allowlist else {}
    res = reader.readtext(img_bgr, **kwargs)
    if not res:
        return ""
    res.sort(key=lambda r: -r[2])
    return res[0][1].strip()


def easyocr_combine_answer_row(reader, img_bgr, allowlist=None):
    """여러 박스를 답 패턴으로 필터링한 뒤 row 단위로 합쳐 반환."""
    kwargs = {"allowlist": allowlist} if allowlist else {}
    res = reader.readtext(img_bgr, **kwargs)
    if not res:
        return ""
    h, w = img_bgr.shape[:2]
    cands = []
    for box, text, conf in res:
        cy = sum(p[1] for p in box) / 4
        cx = sum(p[0] for p in box) / 4
        # 좌상단(문제번호 영역) 무시
        if cx < w * 0.18 and cy < h * 0.18:
            continue
        t = text.strip()
        if not t:
            continue
        if not re.search(r"[\d=<>≤≥+\-xX]", t):
            continue
        if re.search(r"[가-힣]{2,}", t):
            continue
        cands.append((cy, cx, t, conf))
    if not cands:
        return ""
    cands.sort(key=lambda c: -c[3])
    return cands[0][2]


def paddle_top_text(img_bgr):
    import paddle_ocr
    res = paddle_ocr.readtext(img_bgr)
    if not res:
        return ""
    res.sort(key=lambda r: -r[2])
    return res[0][1].strip()


def paddle_filter_answer(img_bgr):
    import paddle_ocr
    res = paddle_ocr.readtext(img_bgr)
    if not res:
        return ""
    h, w = img_bgr.shape[:2]
    cands = []
    for box, text, conf in res:
        cy = sum(p[1] for p in box) / 4
        cx = sum(p[0] for p in box) / 4
        if cx < w * 0.18 and cy < h * 0.18:
            continue
        t = text.strip()
        if not t:
            continue
        if not re.search(r"[\d=<>≤≥+\-xX]", t):
            continue
        if re.search(r"[가-힣]{2,}", t):
            continue
        cands.append((cy, cx, t, conf))
    if not cands:
        return ""
    cands.sort(key=lambda c: -c[3])
    return cands[0][2]


def preprocess_bottom_two_thirds(crop_bgr):
    """문제번호/본문이 많은 상단 1/3을 잘라낸 뒤 OCR — 손글씨가 주로 있는 영역."""
    h = crop_bgr.shape[0]
    return crop_bgr[h // 3 :, :]


def main(crops):
    print("loading EasyOCR...")
    reader = get_easyocr()
    print("loading PaddleOCR...")
    import paddle_ocr
    paddle_ocr.get_paddle_ocr()  # warm up

    rows = []
    for p in crops:
        m = re.search(r"problem_(\d+)", p.name)
        if not m:
            continue
        num = int(m.group(1))
        correct = str(ANSWERS.get(num, "?"))
        img = cv2.imread(str(p))
        if img is None:
            continue

        bottom = preprocess_bottom_two_thirds(img)

        outs = {
            "easy/raw/top": easyocr_top_text(reader, img),
            "easy/raw/filt": easyocr_combine_answer_row(reader, img),
            "easy/btm/filt": easyocr_combine_answer_row(reader, bottom),
            "easy/raw/wl": easyocr_combine_answer_row(reader, img, ANSWER_ALLOWLIST),
            "easy/btm/wl": easyocr_combine_answer_row(reader, bottom, ANSWER_ALLOWLIST),
            "paddle/raw/top": paddle_top_text(img),
            "paddle/raw/filt": paddle_filter_answer(img),
            "paddle/btm/filt": paddle_filter_answer(bottom),
        }
        rows.append((num, correct, outs))

    keys = list(rows[0][2].keys()) if rows else []
    header = f"{'문제':>4} {'정답':>8} | " + " ".join(f"{k:>16}" for k in keys)
    print(header)
    print("-" * len(header))

    hits = {k: 0 for k in keys}
    for num, correct, outs in rows:
        cells = []
        for k in keys:
            v = outs.get(k, "")
            ok = normalize(v) == normalize(correct)
            if ok:
                hits[k] += 1
            cells.append(f"{'✓' if ok else ' '}{v[:14]:<14}")
        print(f"{num:>4} {correct:>8} | " + " ".join(cells))

    print()
    print("=== 정답 적중 횟수 (총 {}) ===".format(len(rows)))
    for k, v in sorted(hits.items(), key=lambda kv: -kv[1]):
        print(f"  {k:>16}: {v}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python bench_paddle.py <crop1.png> [crop2.png ...]")
        sys.exit(1)
    main([Path(p) for p in sys.argv[1:]])
