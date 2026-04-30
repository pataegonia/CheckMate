"""손글씨 답 크롭에 대해 여러 (모델, 전처리) 조합을 비교.

사용:
    python bench.py debug_out/page_1_problem_*.png

각 크롭마다 모델별 결과를 표로 출력. 정답은 answers.py 기준.
"""
import re
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from answers import ANSWERS

# math/answer 글자만 (한글 본문 무시, 수식 부호 통과)
ANSWER_ALLOWLIST = "0123456789xXyzabc=<>+-≤≥≠/.,()"


def get_easyocr():
    import easyocr
    return easyocr.Reader(["ko", "en"], gpu=False)


def get_trocr_handwritten_en():
    """영문 손글씨 TrOCR (참고용 — 한글/한자엔 환각)."""
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    proc = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    model.eval()
    return ("trocr-en-handwritten", proc, model)


def get_trocr_korean():
    """한국어 TrOCR — 인쇄체 위주이지만 손글씨도 일부 학습."""
    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    name = "team-lucid/trocr-small-korean"
    try:
        proc = TrOCRProcessor.from_pretrained(name)
        model = VisionEncoderDecoderModel.from_pretrained(name)
        model.eval()
        return ("trocr-korean", proc, model)
    except Exception as e:
        print(f"  (skip {name}: {e})")
        return None


def preprocess_variants(crop_bgr: np.ndarray) -> dict[str, np.ndarray]:
    """크롭에 대한 여러 전처리 변형."""
    out = {"raw": crop_bgr}

    # 1) 빨간 잉크만 추출 (학생이 빨간펜으로 쓴 경우)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, (0, 60, 60), (12, 255, 255)) | cv2.inRange(
        hsv, (165, 60, 60), (180, 255, 255)
    )
    if red_mask.sum() > 200:
        # red ink → black on white
        red_only = np.full_like(crop_bgr, 255)
        red_only[red_mask > 0] = [0, 0, 0]
        # dilate to thicken strokes for OCR
        red_only = cv2.morphologyEx(
            red_only,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        out["red_only"] = red_only

    # 2) 회색 + adaptive 이진화
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
    )
    out["binary"] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # 3) 손글씨 영역만 추출 시도: 빨간 마스크 bbox 또는 인쇄와 다른 위치
    if red_mask.sum() > 200:
        ys, xs = np.where(red_mask > 0)
        if len(ys) > 0:
            y1, y2 = max(0, ys.min() - 8), min(crop_bgr.shape[0], ys.max() + 8)
            x1, x2 = max(0, xs.min() - 8), min(crop_bgr.shape[1], xs.max() + 8)
            handwriting_crop = crop_bgr[y1:y2, x1:x2]
            if handwriting_crop.size > 0:
                # 2x 업스케일
                up = cv2.resize(
                    handwriting_crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
                )
                out["red_bbox_2x"] = up

    return out


def run_easyocr(reader, image_bgr, allowlist: Optional[str] = None) -> str:
    kwargs = {}
    if allowlist:
        kwargs["allowlist"] = allowlist
    results = reader.readtext(image_bgr, **kwargs)
    if not results:
        return ""
    # 가장 conf 높은 것을 선택
    results.sort(key=lambda r: -r[2])
    text = results[0][1].strip()
    return text


def run_trocr(model_tuple, image_bgr) -> str:
    name, proc, model = model_tuple
    import torch
    from PIL import Image
    pil = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    inputs = proc(images=pil, return_tensors="pt").pixel_values
    with torch.no_grad():
        gen = model.generate(inputs, max_length=32)
    text = proc.batch_decode(gen, skip_special_tokens=True)[0]
    return text.strip()


def normalize(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = s.replace("−", "-").replace(" ", "").replace(" ", "")
    return s


def run_bench(crops: list[Path]) -> None:
    print("=== EasyOCR 로드 중...")
    reader = get_easyocr()

    print("=== TrOCR(영문 손글씨) 로드 중...")
    try:
        trocr_en = get_trocr_handwritten_en()
    except Exception as e:
        print(f"  skip TrOCR-en: {e}")
        trocr_en = None

    print("=== TrOCR(Korean) 로드 중...")
    trocr_ko = get_trocr_korean()

    print()
    header = f"{'문제':>4} {'정답':>8} | {'easy/raw':>14} {'easy/red':>14} {'easy/bin':>14} {'easy/redbox':>14} {'easy/raw+wl':>14}"
    if trocr_en:
        header += f" {'troc-en/raw':>14} {'troc-en/red':>14}"
    if trocr_ko:
        header += f" {'troc-ko/raw':>14}"
    print(header)
    print("-" * len(header))

    summary = {"total": 0, "best_per_crop": {}}

    for crop_path in crops:
        m = re.search(r"problem_(\d+)", crop_path.name)
        if not m:
            continue
        num = int(m.group(1))
        correct = ANSWERS.get(num, "?")

        crop = cv2.imread(str(crop_path))
        if crop is None:
            continue

        variants = preprocess_variants(crop)
        results = {}

        # EasyOCR variants
        results["easy/raw"] = run_easyocr(reader, variants.get("raw", crop))
        results["easy/red"] = run_easyocr(reader, variants["red_only"]) if "red_only" in variants else "-"
        results["easy/bin"] = run_easyocr(reader, variants["binary"])
        results["easy/redbox"] = (
            run_easyocr(reader, variants["red_bbox_2x"]) if "red_bbox_2x" in variants else "-"
        )
        results["easy/raw+wl"] = run_easyocr(reader, variants.get("raw", crop), allowlist=ANSWER_ALLOWLIST)

        if trocr_en:
            results["troc-en/raw"] = run_trocr(trocr_en, variants.get("raw", crop))
            results["troc-en/red"] = (
                run_trocr(trocr_en, variants["red_only"]) if "red_only" in variants else "-"
            )
        if trocr_ko:
            results["troc-ko/raw"] = run_trocr(trocr_ko, variants.get("raw", crop))

        # 정답 매칭 표시
        def mark(s):
            ok = normalize(s) == normalize(correct)
            return f"{'✓' if ok else ' '}{s[:13]:<13}"

        row = f"{num:>4} {str(correct):>8} | "
        row += " ".join(mark(results.get(k, "-")) for k in [
            "easy/raw", "easy/red", "easy/bin", "easy/redbox", "easy/raw+wl"
        ])
        if trocr_en:
            row += " " + " ".join(mark(results.get(k, "-")) for k in ["troc-en/raw", "troc-en/red"])
        if trocr_ko:
            row += " " + mark(results.get("troc-ko/raw", "-"))
        print(row)

        summary["total"] += 1
        for k, v in results.items():
            if normalize(v) == normalize(correct):
                summary["best_per_crop"].setdefault(num, []).append(k)

    print()
    print("=== 모델/전처리별 정답 적중 횟수 (총 {} 문제) ===".format(summary["total"]))
    hits = {}
    for num, ks in summary["best_per_crop"].items():
        for k in ks:
            hits[k] = hits.get(k, 0) + 1
    for k, v in sorted(hits.items(), key=lambda kv: -kv[1]):
        print(f"  {k:>14}: {v}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python bench.py <crop1.png> [<crop2.png> ...]")
        sys.exit(1)
    crops = [Path(p) for p in sys.argv[1:]]
    run_bench(crops)
