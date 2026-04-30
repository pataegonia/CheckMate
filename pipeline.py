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


# OCR이 자주 헷갈리는 글자 → 숫자 매핑 (0814 → "OB14", "OOI4" 같은 오인식 흡수용)
_OCR_DIGIT_CONFUSIONS = str.maketrans({
    "O": "0", "o": "0", "D": "0", "Q": "0",
    "I": "1", "l": "1", "i": "1", "|": "1", "!": "1",
    "Z": "7", "z": "7",
    "B": "8",
    "S": "5", "s": "5",
    "G": "6", "b": "6",
    "T": "7",
    "g": "9", "q": "9",
})


def _extract_digit_runs(text: str) -> list[str]:
    """텍스트에서 숫자 시퀀스를 모두 추출. 혼동 글자도 한 번 흡수해 두 번 시도."""
    runs = re.findall(r"\d+", text)
    fuzzy = text.translate(_OCR_DIGIT_CONFUSIONS)
    runs.extend(re.findall(r"\d+", fuzzy))
    return runs


def _match_valid_key(runs: list[str], valid_keys: set) -> Optional[int]:
    """digit run 안에서 valid_key를 substring 매칭. "0817" 같은 4자리 형태 우선."""
    # 길이 긴 형태(0XXX)부터 → 짧은 형태(XXX) 순으로 시도해 더 구체적인 매칭을 선호
    forms_4 = sorted(valid_keys, key=lambda k: -k)  # arbitrary stable order
    candidates_4 = [(f"0{k}", k) for k in forms_4]
    candidates_3 = [(str(k), k) for k in forms_4]

    for run in runs:
        for s, n in candidates_4:
            if s in run:
                return n
    for run in runs:
        for s, n in candidates_3:
            if s in run:
                return n
    return None


def _anchors_from_results(results, valid_keys: set, fuzzy: bool = False) -> list[dict]:
    anchors = []
    for box, text, conf in results:
        runs = _extract_digit_runs(text) if fuzzy else re.findall(r"\d+", text)
        n = _match_valid_key(runs, valid_keys)
        if n is None:
            continue
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


def _recover_missing_anchors(
    image_bgr: np.ndarray,
    found_anchors: list[dict],
    valid_keys: set,
    cols: list[tuple[int, int]],
) -> tuple[list[dict], list]:
    """발견된 앵커들로부터 누락 앵커의 위치를 컬럼 안에서 기하학적으로 추정하고,
    그 좁은 ROI에 전처리(업스케일/대비)를 걸어 EasyOCR을 재호출.
    반환: (recovered_anchors, extra_ocr_results)"""
    h, w = image_bgr.shape[:2]
    missing = sorted(valid_keys - {a["num"] for a in found_anchors})
    if not missing or not found_anchors:
        return [], []

    def col_of(x_center: float) -> int:
        for ci, (cx1, cx2) in enumerate(cols):
            if cx1 <= x_center < cx2:
                return ci
        return 0

    by_col: dict[int, list[dict]] = defaultdict(list)
    for a in found_anchors:
        cx = (a["x_left"] + a["x_right"]) / 2
        by_col[col_of(cx)].append(a)
    for col_idx in by_col:
        by_col[col_idx].sort(key=lambda a: a["num"])

    reader = _get_easyocr()
    recovered: list[dict] = []
    extra_ocr: list = []

    for k in missing:
        # k와 가장 번호가 가까운 앵커가 있는 컬럼을 후보로 선택
        nearest = min(found_anchors, key=lambda a: abs(a["num"] - k))
        col_idx = col_of((nearest["x_left"] + nearest["x_right"]) / 2)
        col_anchors = by_col[col_idx]

        prev_a = next((a for a in reversed(col_anchors) if a["num"] < k), None)
        next_a = next((a for a in col_anchors if a["num"] > k), None)

        # 같은 컬럼의 평균 gap 추정 (번호 1당 y 픽셀)
        if len(col_anchors) >= 2:
            gaps = []
            for i in range(len(col_anchors) - 1):
                dy = col_anchors[i + 1]["y_top"] - col_anchors[i]["y_top"]
                dn = max(1, col_anchors[i + 1]["num"] - col_anchors[i]["num"])
                gaps.append(dy / dn)
            typical = sum(gaps) / len(gaps)
        else:
            typical = 350.0

        # 외삽(prev/next 한쪽만 있는 경우 — 컬럼 맨 위/맨 아래)은 위치 불확실성이 크므로 ROI를 넉넉히
        extrapolate = not (prev_a and next_a)
        if prev_a and next_a:
            frac = (k - prev_a["num"]) / max(1, next_a["num"] - prev_a["num"])
            pred_y = int(prev_a["y_top"] + frac * (next_a["y_top"] - prev_a["y_top"]))
        elif prev_a:
            pred_y = int(prev_a["y_top"] + (k - prev_a["num"]) * typical)
        elif next_a:
            pred_y = int(next_a["y_top"] - (next_a["num"] - k) * typical)
        else:
            continue

        cx1, cx2 = cols[col_idx]
        # 문제번호는 컬럼 좌측 ~45% 안에 위치. 외삽이면 ROI 더 넓게.
        pad_up = 200 if extrapolate else 80
        pad_down = 100 if extrapolate else 80
        roi_y1 = max(0, pred_y - pad_up)
        roi_y2 = min(h, pred_y + pad_down)
        roi_x1 = cx1
        roi_x2 = min(cx2, cx1 + int((cx2 - cx1) * 0.45))
        roi = image_bgr[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            continue

        # 전처리 변형: 업스케일 + 대비강화 + 그레이스케일
        up = cv2.resize(roi, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        contrast = cv2.convertScaleAbs(up, alpha=1.6, beta=15)
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        variants = [up, contrast, gray_3ch]

        best: Optional[dict] = None
        for var in variants:
            # 1) 숫자만 (allowlist) — 가장 깨끗한 결과
            # 2) 일반 OCR + fuzzy 글자→숫자 변환 (보조)
            for kwargs in (
                {"allowlist": "0123456789", "text_threshold": 0.3, "low_text": 0.15, "link_threshold": 0.2},
                {"text_threshold": 0.3, "low_text": 0.15, "link_threshold": 0.2},
            ):
                res = reader.readtext(var, **kwargs)
                # 이 누락 번호 한정으로 substring 매칭 (예: '08179'에서 '0817' 찾기)
                target_forms = (f"0{k}", str(k))
                for box, text, conf in res:
                    xs0 = [p[0] / 2.5 + roi_x1 for p in box]
                    ys0 = [p[1] / 2.5 + roi_y1 for p in box]
                    extra_ocr.append((
                        [(int(x), int(y)) for x, y in zip(xs0, ys0)],
                        text,
                        conf,
                    ))
                    runs = _extract_digit_runs(text)
                    matched = any(form in run for run in runs for form in target_forms)
                    if matched:
                        cand = {
                            "num": k,
                            "x_left": int(min(xs0)),
                            "x_right": int(max(xs0)),
                            "y_top": int(min(ys0)),
                            "y_bottom": int(max(ys0)),
                            "conf": float(conf),
                            "text": text,
                            "recovered": True,
                        }
                        if best is None or cand["conf"] > best["conf"]:
                            best = cand
                if best is not None and best["conf"] > 0.5:
                    break
            if best is not None and best["conf"] > 0.5:
                break
        if best is not None:
            recovered.append(best)

    return recovered, extra_ocr


def find_anchors(image_bgr: np.ndarray, valid_keys: set) -> tuple[list[dict], list]:
    """페이지 전체를 OCR해서 valid_keys 안의 문제번호 위치를 모두 찾는다.
    1차에서 빠진 게 있으면 더 민감한 옵션으로 재시도.
    그래도 누락된 게 있으면 컬럼 내 기하학적 추정 + 좁은 ROI 재OCR로 복구.
    반환: (anchors, all_ocr_results)"""
    reader = _get_easyocr()

    pass1 = reader.readtext(image_bgr)
    anchors = _anchors_from_results(pass1, valid_keys)
    all_results = list(pass1)

    found = {a["num"] for a in anchors}
    if valid_keys - found:
        # 2차: 더 민감한 옵션 + 글자→숫자 fuzzy 매칭 ("DBIZuC"→0817, "OB14"→0814 흡수)
        pass2 = reader.readtext(
            image_bgr, text_threshold=0.3, low_text=0.2, link_threshold=0.2
        )
        more = _anchors_from_results(pass2, valid_keys, fuzzy=True)
        for a in more:
            if a["num"] not in found:
                anchors.append(a)
                found.add(a["num"])
        all_results.extend(pass2)

    anchors = _dedup_anchors(anchors)

    # 3차 패스: 컬럼 추정 후 누락 앵커를 좁은 ROI로 복구
    found = {a["num"] for a in anchors}
    if valid_keys - found and anchors:
        h, w = image_bgr.shape[:2]
        cols = cluster_columns(anchors, w)
        recovered, extra = _recover_missing_anchors(image_bgr, anchors, valid_keys, cols)
        anchors.extend(recovered)
        all_results.extend(extra)
        anchors = _dedup_anchors(anchors)

    return anchors, all_results


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


_ANSWER_ALLOWLIST = "0123456789xXyzabc<>=+-/.,"


def _detect_handwriting_mask(crop_bgr: np.ndarray) -> np.ndarray:
    """학생 손글씨 추정 마스크: 채도 높은 빨강/파랑 잉크.
    상단 60px(인쇄된 문제번호 태그 영역)는 마스크에서 제외."""
    h = crop_bgr.shape[0]
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 60, 60), (12, 255, 255)) | cv2.inRange(
        hsv, (165, 60, 60), (180, 255, 255)
    )
    blue = cv2.inRange(hsv, (100, 80, 60), (135, 255, 255))
    mask = red | blue
    # 상단 인쇄 태그 영역 제거
    top_skip = min(60, h // 4)
    mask[:top_skip, :] = 0
    return mask


def _largest_handwriting_region(mask: np.ndarray, min_area: int = 60) -> Optional[tuple[int, int, int, int]]:
    """마스크에서 손글씨로 추정되는 큰 연결요소들의 bbox(union) 반환. (x1, y1, x2, y2)."""
    if mask.sum() < 200:
        return None
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    big = [
        stats[i] for i in range(1, n)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
    ]
    if not big:
        return None
    x1 = min(int(s[cv2.CC_STAT_LEFT]) for s in big)
    y1 = min(int(s[cv2.CC_STAT_TOP]) for s in big)
    x2 = max(int(s[cv2.CC_STAT_LEFT] + s[cv2.CC_STAT_WIDTH]) for s in big)
    y2 = max(int(s[cv2.CC_STAT_TOP] + s[cv2.CC_STAT_HEIGHT]) for s in big)
    return x1, y1, x2, y2


def _try_paddle_ocr(crop_bgr: np.ndarray) -> Optional[list[tuple[list, str, float]]]:
    """PaddleOCR이 설치되어 있으면 결과 반환, 아니면 None."""
    try:
        import paddle_ocr as _po
        return _po.readtext(crop_bgr)
    except ImportError:
        return None
    except Exception as e:
        print(f"[paddle skip: {e}]")
        return None


def _isolation_score(cy: float, cx: float, all_boxes: list, ignore_box=None) -> float:
    """주변 다른 OCR 박스로부터 떨어진 정도 (가까운 이웃까지의 거리, px).
    값이 클수록 고립 → 손글씨일 가능성 높음."""
    best = float("inf")
    for box, _, _ in all_boxes:
        if box is ignore_box:
            continue
        bcy = sum(p[1] for p in box) / 4
        bcx = sum(p[0] for p in box) / 4
        dx = bcx - cx; dy = bcy - cy
        if dx == 0 and dy == 0:
            continue
        d = (dx * dx + dy * dy) ** 0.5
        if d < best:
            best = d
    return best if best != float("inf") else 0.0


def _filter_handwritten_candidates(
    res: list[tuple[list, str, float]],
    crop_shape: tuple[int, int, int],
    problem_num: int,
) -> list[tuple[float, float, str, float]]:
    """OCR 결과에서 학생 답으로 보이는 후보만 골라 (cy, cx, text, conf) 반환.
    제외 규칙:
      - 좌상단(문제번호 영역) — 인쇄된 문제번호 박스
      - 텍스트가 인쇄된 문제번호 자체("0817", "817")
      - 한글 2자 이상 (본문)
      - 숫자/부등호/x 등 답 후보 글자 미포함
      - 6자 초과 (간단한 답이 아님)
    """
    h, w = crop_shape[:2]
    forms_to_skip = {f"0{problem_num}", str(problem_num)}
    cands = []
    for box, text, conf in res:
        if not text:
            continue
        t = text.strip()
        if not t or len(t) > 8:
            continue
        # 인쇄된 문제번호 자체 제거
        if any(form in t for form in forms_to_skip):
            continue
        cy = sum(p[1] for p in box) / 4
        cx = sum(p[0] for p in box) / 4
        # 좌상단 (문제번호 태그 영역)
        if cx < w * 0.18 and cy < h * 0.13:
            continue
        # 한글 2자 이상이면 본문/보기
        if re.search(r"[가-힣]{2,}", t):
            continue
        # 답에 등장 가능한 글자 포함 여부
        if not re.search(r"[\d=<>≤≥+\-xXyz]", t):
            continue
        # 인쇄 패턴 (①~⑤ 단독 등)
        if re.fullmatch(r"[①-⑳][.,]?", t):
            continue
        # 단일 한글 자음만은 옵션 마커 가능 — pass
        cands.append((cy, cx, t, float(conf)))
    return cands


def _normalize_handwriting_artifacts(text: str) -> str:
    """손글씨 OCR이 자주 헷갈리는 패턴 정규화.
    - "7∠"/"7L" → "x<" (x를 7로, <를 ∠/L로 읽는 경우)
    - "≤" → "<=" (시각적 동등하지만 답 dict 표기)
    - 공백/유니코드 마이너스 흡수
    """
    s = text.strip().replace(" ", "")
    s = s.replace("−", "-").replace("―", "-")
    # x 글자가 7로, < 가 ∠/L로 잘못 읽힌 경우만 좁게 적용 (`7` 다음에 < 같은 부등호 부호가 와야 함)
    s = re.sub(r"7\s*[∠L]", "x<", s)
    s = re.sub(r"7\s*>", "x>", s)
    return s


def _ocr_mask_region(crop_bgr: np.ndarray, mask: np.ndarray) -> list[tuple[float, str, float]]:
    """색마스크 bbox만 잘라 업스케일 후 OCR. (cy, text, conf) 리스트 반환."""
    h, w = crop_bgr.shape[:2]
    bbox = _largest_handwriting_region(mask)
    if bbox is None:
        return []
    x1, y1, x2, y2 = bbox
    pad = 14
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)

    hw_only = np.full_like(crop_bgr, 255)
    hw_only[mask > 0] = [0, 0, 0]
    hw_crop = hw_only[y1:y2, x1:x2]
    if hw_crop.size == 0 or hw_crop.shape[0] < 5 or hw_crop.shape[1] < 5:
        return []

    up = cv2.resize(hw_crop, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    reader = _get_easyocr()
    out: list[tuple[float, str, float]] = []
    # 두 가지 설정으로 시도: 답 글자만 / 일반
    for kwargs in (
        {"allowlist": _ANSWER_ALLOWLIST},
        {},
    ):
        res = reader.readtext(up, **kwargs)
        for box, text, conf in res:
            t = text.strip()
            if not t:
                continue
            cy = (sum(p[1] for p in box) / 4) / 3.0 + y1
            out.append((cy, t, float(conf)))
        if out:
            break
    return out


_OPTION_MARKS = {"①": "1", "②": "2", "③": "3", "④": "4", "⑤": "5"}


def _detect_circled_option(
    crop_bgr: np.ndarray, ocr_results: list
) -> Optional[str]:
    """학생이 인쇄된 ①~⑤ 중 하나에 빨간/검은 동그라미 표시한 경우 검출.
    1. 색마스크에서 큰 블롭(>500px) 찾고 원형(circularity>0.5) 필터
    2. OCR 결과에서 ①~⑤ 위치 모음
    3. 동그라미 중심에서 가장 가까운 옵션 → 해당 숫자 반환"""
    h, w = crop_bgr.shape[:2]
    mask = _detect_handwriting_mask(crop_bgr)
    if mask.sum() < 500:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circle_centers: list[tuple[float, float, float]] = []  # (cx, cy, area)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 300:
            continue
        perim = cv2.arcLength(c, True)
        if perim == 0:
            continue
        circularity = 4 * np.pi * area / (perim * perim)
        if circularity < 0.35:  # 너무 길쭉하면 글씨, 둥글면 동그라미
            continue
        (cx, cy), _r = cv2.minEnclosingCircle(c)
        circle_centers.append((cx, cy, area))

    if not circle_centers:
        return None

    # OCR 결과에서 ①~⑤ 위치 모으기
    option_positions: list[tuple[float, float, str]] = []  # (cx, cy, digit)
    for box, text, _conf in ocr_results:
        for ch, digit in _OPTION_MARKS.items():
            if ch in text:
                cx = sum(p[0] for p in box) / 4
                cy = sum(p[1] for p in box) / 4
                option_positions.append((cx, cy, digit))
                break

    if not option_positions:
        return None

    # 가장 큰 원형 블롭 → 가장 가까운 옵션
    circle_centers.sort(key=lambda c: -c[2])
    cx, cy, _ = circle_centers[0]
    nearest = min(
        option_positions,
        key=lambda o: (o[0] - cx) ** 2 + (o[1] - cy) ** 2,
    )
    # 거리가 너무 멀면 매칭 무시
    dist = ((nearest[0] - cx) ** 2 + (nearest[1] - cy) ** 2) ** 0.5
    if dist > max(w, h) * 0.25:
        return None
    return nearest[2]


def extract_handwritten_answer_v2(crop_bgr: np.ndarray, problem_num: int = 0) -> str:
    """v2: PaddleOCR(설치 시) → EasyOCR 순으로 시도 + 스마트 필터.
    - 인쇄된 문제번호("0815"), ①~⑤, 본문 한글 등 제거
    - 답 패턴(숫자/부등호/x) 통과 후보만 conf 순 정렬
    - 손글씨 OCR 흔한 패턴 정규화 적용 ("7∠" → "x<")
    - 모든 후보가 비면 v1 휴리스틱으로 폴백.
    """
    h, w = crop_bgr.shape[:2]

    def _score_candidates(cands, all_boxes):
        """conf + 고립도(주변 다른 박스에서 멀수록) 가중. 가장 높은 후보 반환."""
        if not cands:
            return None
        scored = []
        for cy, cx, t, conf in cands:
            iso = _isolation_score(cy, cx, all_boxes)
            iso_norm = min(iso / 100.0, 2.0)  # 100px 이상이면 만점, 200px 이상은 capped
            score = conf + 0.15 * iso_norm
            scored.append((score, cy, cx, t, conf))
        scored.sort(key=lambda s: -s[0])
        return scored[0][3]

    # 1) PaddleOCR 우선
    paddle_res = _try_paddle_ocr(crop_bgr)
    if paddle_res is not None:
        # 1a) 객관식 동그라미 표시 검출 (학생이 ①~⑤ 중 하나에 빨간 원)
        # 단, 학생 마크가 검정 잉크인 경우 색마스크가 비어 None 반환됨
        circled = _detect_circled_option(crop_bgr, paddle_res)
        if circled:
            return circled
        cands = _filter_handwritten_candidates(paddle_res, crop_bgr.shape, problem_num)
        picked = _score_candidates(cands, paddle_res)
        if picked:
            return _normalize_handwriting_artifacts(picked)

    # 2) EasyOCR 폴백 — 같은 필터 적용
    reader = _get_easyocr()
    easy_res = reader.readtext(crop_bgr)
    cands = _filter_handwritten_candidates(easy_res, crop_bgr.shape, problem_num)
    picked = _score_candidates(cands, easy_res)
    if picked:
        return _normalize_handwriting_artifacts(picked)

    # 3) 최종 폴백: v1 휴리스틱
    return extract_handwritten_answer(crop_bgr)


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

            student_ans = extract_handwritten_answer_v2(crop, problem_num=a["num"])
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
