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
        # k와 가장 번호가 가까운 앵커가 있는 컬럼만 보면 컬럼 끝(예: 817/818 경계)에서
        # 반대쪽 컬럼으로 외삽될 수 있으므로 모든 컬럼 후보를 시도한다.
        candidate_cols = []
        for ci in range(len(cols)):
            if by_col.get(ci):
                score = min(abs(a["num"] - k) for a in by_col[ci])
                candidate_cols.append((score, ci))
        candidate_cols.sort()

        best_for_k: Optional[dict] = None

        for _score, col_idx in candidate_cols:
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
            # 문제번호는 컬럼 좌측 ~55% 안에 위치. 외삽이면 ROI 더 넓게.
            pad_up = 220 if extrapolate else 90
            pad_down = 180 if extrapolate else 90
            roi_y1 = max(0, pred_y - pad_up)
            roi_y2 = min(h, pred_y + pad_down)
            roi_x1 = cx1
            roi_x2 = min(cx2, cx1 + int((cx2 - cx1) * 0.55))
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
                    {"allowlist": "0123456789", "text_threshold": 0.25, "low_text": 0.12, "link_threshold": 0.15},
                    {"text_threshold": 0.25, "low_text": 0.12, "link_threshold": 0.15},
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
                    if best is not None and best["conf"] > 0.45:
                        break
                if best is not None and best["conf"] > 0.45:
                    break
            if best is not None:
                if best_for_k is None or best["conf"] > best_for_k["conf"]:
                    best_for_k = best

        if best_for_k is not None:
            recovered.append(best_for_k)

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


def _trailing_handwritten_answer(text: str) -> Optional[str]:
    """텍스트의 마지막 한글 글자 이후에 매달린 짧은 답 후보 추출.
    예: '...실수이다.)4' → '4' (학생이 본문 끝에 답을 적은 경우 OCR이 한 박스로 합침)"""
    if not text:
        return None
    last_korean = -1
    for i in range(len(text) - 1, -1, -1):
        ch = text[i]
        if "가" <= ch <= "힣":
            last_korean = i
            break
    tail = text[last_korean + 1:] if last_korean >= 0 else text
    tail = tail.strip(" .,()[]?!:")
    if not tail or len(tail) > 8:
        return None
    if not re.search(r"[\d=<>≤≥+\-xX]", tail):
        return None
    if re.search(r"[가-힣]", tail):
        return None
    return tail


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
        raw = text
        t = text.strip()
        if not t:
            continue
        cy = sum(p[1] for p in box) / 4
        cx = sum(p[0] for p in box) / 4
        # 좌상단 (문제번호 태그 영역)
        if cx < w * 0.18 and cy < h * 0.13:
            continue

        is_korean_body = bool(re.search(r"[가-힣]{2,}", t))
        if is_korean_body:
            # 본문에 답이 매달린 경우만 trailing 추출 ('...실수이다.)4' → '4')
            tail = _trailing_handwritten_answer(raw)
            if tail and tail not in forms_to_skip:
                # 위치는 trailing 부분 추정이 어렵지만 conf는 살짝 깎아둠 (확신 낮음)
                cands.append((cy, cx, tail, float(conf) * 0.9))
            continue

        if len(t) > 8:
            continue
        if any(form in t for form in forms_to_skip):
            continue
        if not re.search(r"[\d=<>≤≥+\-xXyz]", t):
            continue
        if re.fullmatch(r"[①-⑳][.,]?", t):
            continue
        cands.append((cy, cx, t, float(conf)))
    return cands


def _normalize_handwriting_artifacts(text: str) -> str:
    """손글씨 OCR이 자주 헷갈리는 패턴 정규화.
    - "7∠"/"7L" → "x<" (x를 7로, <를 ∠/L로 읽는 경우)
    - "≤" → "<=" (시각적 동등하지만 답 dict 표기)
    - 공백/유니코드 마이너스 흡수
    """
    s = str(text or "").strip().replace(" ", "")
    replacements = {
        "−": "-",
        "―": "-",
        "–": "-",
        "—": "-",
        "﹣": "-",
        "－": "-",
        "＜": "<",
        "〈": "<",
        "‹": "<",
        "≤": "<=",
        "≦": "<=",
        "＞": ">",
        "〉": ">",
        "›": ">",
        "≥": ">=",
        "≧": ">=",
        "＝": "=",
        "×": "x",
        "χ": "x",
        "X": "x",
    }
    for src, dst in replacements.items():
        s = s.replace(src, dst)
    # x 글자가 7로, < 가 ∠/L로 잘못 읽힌 경우만 좁게 적용 (`7` 다음에 < 같은 부등호 부호가 와야 함)
    s = re.sub(r"7\s*[∠L]", "x<", s)
    s = re.sub(r"7\s*>", "x>", s)
    # OCR이 x를 생략하고 부등호식만 읽은 경우는 유지하되, 흔한 'xL-2'류를 보정
    s = re.sub(r"^x[∠L](-?\d+)$", r"x<\1", s)
    return s


def _answer_focus_crop(crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    """문항 crop 안에서 손글씨 답안으로 보이는 영역만 조금 더 확대해서 반환."""
    h, w = crop_bgr.shape[:2]
    if h < 20 or w < 20:
        return None

    # 상단 문제 본문을 어느 정도 제외하고 손글씨/마킹 후보를 찾는다.
    mask = _build_dark_ink_mask(crop_bgr, [], top_skip=max(50, int(h * 0.22)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    components = []
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < 20:
            continue
        if bw > w * 0.85 or bh > h * 0.55:
            continue
        components.append((x, y, x + bw, y + bh, area))

    if not components:
        return None

    # 너무 많은 인쇄 영역이 섞이면 전체 crop이 더 안전하다.
    x1 = min(c[0] for c in components)
    y1 = min(c[1] for c in components)
    x2 = max(c[2] for c in components)
    y2 = max(c[3] for c in components)
    if (x2 - x1) * (y2 - y1) > w * h * 0.45:
        return None

    pad_x = max(24, int((x2 - x1) * 0.55))
    pad_y = max(24, int((y2 - y1) * 0.90))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    focus = crop_bgr[y1:y2, x1:x2]
    if focus.size == 0:
        return None
    return cv2.resize(focus, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)


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


def _build_dark_ink_mask(
    crop_bgr: np.ndarray,
    ocr_boxes: list,
    top_skip: int = 60,
    suppress_conf_min: float = 0.65,
) -> np.ndarray:
    """학생 손글씨용 마스크: 저-Value(어두운) 픽셀 - 고신뢰 인쇄 텍스트 OCR 박스.
    - 상단 top_skip은 인쇄 태그 영역으로 제외
    - conf >= suppress_conf_min 인 박스만 마스크에서 차감 (학생이 쓴 마크는 보통 OCR이 낮은 conf로 읽음)
    - 한글 2자 이상 포함된 박스는 무조건 인쇄 본문 → 차감"""
    h, w = crop_bgr.shape[:2]
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, 80))
    color_ink = _detect_handwriting_mask(crop_bgr)
    mask = cv2.bitwise_or(dark, color_ink)
    if top_skip > 0:
        mask[:top_skip, :] = 0

    suppress = np.zeros_like(mask)
    pad = 4
    for box, text, conf in ocr_boxes:
        t = text or ""
        # 옵션 마커가 들어간 박스는 차감하지 않음 (학생이 그 위에 동그라미/체크할 수 있음)
        if any(ch in t for ch in _OPTION_MARKS):
            continue
        is_korean_body = bool(re.search(r"[가-힣]{2,}", t))
        if (conf is None or conf >= suppress_conf_min) or is_korean_body:
            xs = [int(p[0]) for p in box]; ys = [int(p[1]) for p in box]
            x1 = max(0, min(xs) - pad); y1 = max(0, min(ys) - pad)
            x2 = min(w, max(xs) + pad); y2 = min(h, max(ys) + pad)
            suppress[y1:y2, x1:x2] = 255
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(suppress))
    return mask


def _infer_missing_option_positions(
    detected: list[tuple[float, float, str]]
) -> list[tuple[float, float, str]]:
    """검출된 옵션 layout(행/열)을 분석해 누락 옵션의 추정 위치 반환.
    가정: 옵션은 행 단위로 좌→우, 위→아래 번호가 매겨지고 같은 열은 x가 비슷함."""
    if not detected:
        return []
    detected_digits = {d[2] for d in detected}
    missing = [d for d in "12345" if d not in detected_digits]
    if not missing:
        return []

    # 행 그룹화 (y 비슷한 것끼리)
    detected_by_y = sorted(detected, key=lambda d: d[1])
    rows: list[list[tuple[float, float, str]]] = []
    for d in detected_by_y:
        if rows and abs(d[1] - rows[-1][-1][1]) < 35:
            rows[-1].append(d)
        else:
            rows.append([d])
    # 각 행 좌→우 정렬, 행 평균 y 계산
    row_info = []
    for r in rows:
        r_sorted = sorted(r, key=lambda d: d[0])
        avg_y = sum(d[1] for d in r_sorted) / len(r_sorted)
        row_info.append((avg_y, r_sorted))
    row_info.sort(key=lambda ri: ri[0])

    # 글로벌 열 슬롯: 모든 행의 x 위치를 모아 중복 비슷한 것끼리 묶음
    all_xs = sorted(d[0] for r in rows for d in r)
    col_slots: list[float] = []
    col_threshold = 80
    for x in all_xs:
        if not col_slots or abs(x - col_slots[-1]) > col_threshold:
            col_slots.append(x)
        else:
            # 평균으로 갱신
            col_slots[-1] = (col_slots[-1] + x) / 2

    def column_index(x: float) -> int:
        return min(range(len(col_slots)), key=lambda i: abs(col_slots[i] - x))

    # 검출된 옵션의 행/열 매핑
    pos_to_row = {}
    for ri_idx, (_, r) in enumerate(row_info):
        for d in r:
            pos_to_row[d[2]] = ri_idx

    inferred: list[tuple[float, float, str]] = []
    for m in missing:
        m_int = int(m)
        prev = next((d for d in reversed(detected_by_y) if int(d[2]) == m_int - 1), None)
        nxt = next((d for d in detected_by_y if int(d[2]) == m_int + 1), None)

        target_row_idx = None
        target_col_idx = None
        if nxt is not None:
            # m은 nxt(=m+1) 직전 — 같은 행 nxt의 한 칸 왼쪽
            target_row_idx = pos_to_row[nxt[2]]
            target_col_idx = column_index(nxt[0]) - 1
            if target_col_idx < 0:
                # nxt가 행의 첫 칸이면 m은 이전 행 마지막 칸
                target_row_idx -= 1
                target_col_idx = len(col_slots) - 1
        elif prev is not None:
            target_row_idx = pos_to_row[prev[2]]
            target_col_idx = column_index(prev[0]) + 1
            if target_col_idx >= len(col_slots):
                target_row_idx += 1
                target_col_idx = 0

        if target_row_idx is None or not (0 <= target_row_idx < len(row_info)):
            continue
        if not (0 <= target_col_idx < len(col_slots)):
            continue
        cy = row_info[target_row_idx][0]
        cx = col_slots[target_col_idx]
        inferred.append((cx, cy, m))
    return inferred


def _detect_circled_option(
    crop_bgr: np.ndarray, ocr_results: list
) -> Optional[str]:
    """학생이 인쇄된 ①~⑤ 중 하나에 동그라미/체크 표시한 경우 검출.
    1. 다크잉크 마스크 - (옵션 마커 외) 인쇄 OCR 박스 영역 = 손글씨 후보 픽셀
    2. contour 추출 (circ/area 약한 임계값 — 체크 V도 포함)
    3. OCR로 검출된 ①~⑤ + 누락 옵션의 추정 위치 모음
    4. 가장 큰 contour 중심 → 가장 가까운 옵션 매칭"""
    h, w = crop_bgr.shape[:2]
    mask = _build_dark_ink_mask(crop_bgr, ocr_results, top_skip=60)
    if mask.sum() < 100:
        return None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[tuple[float, float, float]] = []  # (cx, cy, area)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 80:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        ratio = max(bw, bh) / max(1, min(bw, bh))
        if ratio > 5.0:
            continue
        (cx, cy), _ = cv2.minEnclosingCircle(c)
        candidates.append((cx, cy, area))

    if not candidates:
        return None

    # 검출된 ①~⑤ + 누락 옵션의 추정 위치
    detected_options: list[tuple[float, float, str]] = []
    for box, text, _conf in ocr_results:
        for ch, digit in _OPTION_MARKS.items():
            if ch in text:
                cx = sum(p[0] for p in box) / 4
                cy = sum(p[1] for p in box) / 4
                detected_options.append((cx, cy, digit))
                break

    if not detected_options:
        return None

    inferred = _infer_missing_option_positions(detected_options)
    all_options = detected_options + inferred

    # 가장 큰 candidate 우선
    candidates.sort(key=lambda c: -c[2])
    option_xs = [o[0] for o in all_options]
    option_ys = [o[1] for o in all_options]
    option_x1, option_x2 = min(option_xs), max(option_xs)
    option_y1, option_y2 = min(option_ys), max(option_ys)
    x_pad = max(50.0, w * 0.08)
    y_pad = max(45.0, h * 0.08)
    max_match_dist = max(42.0, min(w, h) * 0.06)

    for cx, cy, area in candidates:
        # 오른쪽 위에 따로 쓴 답안 숫자를 인쇄 선택지 근처 체크로 오판하지 않도록,
        # contour 중심이 선택지 영역 근처에 있을 때만 객관식 표시로 인정한다.
        if not (
            option_x1 - x_pad <= cx <= option_x2 + x_pad
            and option_y1 - y_pad <= cy <= option_y2 + y_pad
        ):
            continue
        nearest = min(
            all_options,
            key=lambda o: (o[0] - cx) ** 2 + (o[1] - cy) ** 2,
        )
        dist = ((nearest[0] - cx) ** 2 + (nearest[1] - cy) ** 2) ** 0.5
        # 인접 옵션과 충분히 가까워야 매칭 인정
        if dist <= max_match_dist:
            return nearest[2]
    return None


def extract_handwritten_answer_v2(crop_bgr: np.ndarray, problem_num: int = 0) -> str:
    """편의 함수 — extract_handwritten_answer_v2_full(...)["answer"]만 반환."""
    return extract_handwritten_answer_v2_full(crop_bgr, problem_num)["answer"]


def extract_handwritten_answer_v2_full(crop_bgr: np.ndarray, problem_num: int = 0) -> dict:
    """v2: PaddleOCR(설치 시) → EasyOCR 순으로 시도 + 스마트 필터.
    - 인쇄된 문제번호("0815"), ①~⑤, 본문 한글 등 제거
    - 답 패턴(숫자/부등호/x) 통과 후보만 conf 순 정렬
    - 손글씨 OCR 흔한 패턴 정규화 적용 ("7∠" → "x<")
    - 모든 후보가 비면 v1 휴리스틱으로 폴백.

    반환 dict:
      - answer: 최종 답 문자열
      - confidence: 0~1, 신뢰도 (-1: 알수없음)
      - source: "circle"/"trailing"/"paddle"/"easy"/"v1-fallback"
      - candidates: [{"text": str, "conf": float, "source": str}, ...] 후보 상위 5개
    """
    h, w = crop_bgr.shape[:2]

    def _empty_result():
        return {"answer": "", "confidence": -1.0, "source": "none", "candidates": []}

    def _score_candidates(cands, all_boxes, source_label: str):
        """conf + 고립도 가중. 점수 정렬된 (text, conf, score) 리스트 반환."""
        if not cands:
            return []
        scored = []
        for cy, cx, t, conf in cands:
            iso = _isolation_score(cy, cx, all_boxes)
            iso_norm = min(iso / 100.0, 2.0)
            score = conf + 0.15 * iso_norm
            scored.append((score, t, conf, source_label))
        scored.sort(key=lambda s: -s[0])
        return scored

    all_candidates: list[dict] = []

    # 1) PaddleOCR 우선
    paddle_res = _try_paddle_ocr(crop_bgr)
    if paddle_res is not None:
        # 옵션 마커 3개 이상 검출되면 객관식 문제로 간주
        n_options = sum(
            1 for _b, t, _c in paddle_res
            if any(ch in (t or "") for ch in _OPTION_MARKS)
        )
        if n_options >= 3:
            # 1a) 동그라미/체크 검출
            circled = _detect_circled_option(crop_bgr, paddle_res)
            if circled:
                return {
                    "answer": circled,
                    "confidence": 0.85,
                    "source": "circle",
                    "candidates": [{"text": circled, "conf": 0.85, "source": "circle"}],
                }
            # 1b) 본문 끝 trailing 답
            for box, text, conf in paddle_res:
                if re.search(r"[가-힣]{2,}", text or ""):
                    tail = _trailing_handwritten_answer(text)
                    if tail and tail in "12345" and len(tail) == 1:
                        return {
                            "answer": tail,
                            "confidence": float(conf) * 0.9,
                            "source": "trailing",
                            "candidates": [
                                {"text": tail, "conf": float(conf) * 0.9, "source": "trailing"}
                            ],
                        }
        cands = _filter_handwritten_candidates(paddle_res, crop_bgr.shape, problem_num)
        scored = _score_candidates(cands, paddle_res, "paddle")
        for _score, t, c, s in scored[:5]:
            all_candidates.append({"text": _normalize_handwriting_artifacts(t), "conf": c, "source": s})
        if scored:
            top = scored[0]
            return {
                "answer": _normalize_handwriting_artifacts(top[1]),
                "confidence": top[2],
                "source": "paddle",
                "candidates": all_candidates,
            }

    # 2) EasyOCR 폴백 — 같은 필터 적용
    reader = _get_easyocr()
    easy_res = reader.readtext(crop_bgr)
    cands = _filter_handwritten_candidates(easy_res, crop_bgr.shape, problem_num)
    scored = _score_candidates(cands, easy_res, "easy")
    for _score, t, c, s in scored[:5]:
        all_candidates.append({"text": _normalize_handwriting_artifacts(t), "conf": c, "source": s})
    if scored:
        top = scored[0]
        return {
            "answer": _normalize_handwriting_artifacts(top[1]),
            "confidence": top[2],
            "source": "easy",
            "candidates": all_candidates,
        }

    # 3) v1 폴백
    v1 = extract_handwritten_answer(crop_bgr)
    return {
        "answer": v1,
        "confidence": -1.0,
        "source": "v1-fallback",
        "candidates": [{"text": v1, "conf": -1.0, "source": "v1-fallback"}] if v1 else [],
    }


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
    s = _normalize_handwriting_artifacts(str(s)).strip().lower()
    s = re.sub(r"\s+", "", s)
    s = s.strip(".,;:()[]{}")
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
            answer_focus = _answer_focus_crop(crop)

            extract = extract_handwritten_answer_v2_full(crop, problem_num=a["num"])
            student_ans = extract["answer"]
            correct_ans = answer_dict.get(a["num"])
            is_correct = normalize(student_ans) == normalize(correct_ans)

            results.append(
                {
                    "problem_num": a["num"],
                    "bbox": (cx1, y1, cx2, y2),
                    "student_answer": student_ans,
                    "correct_answer": str(correct_ans),
                    "correct": is_correct,
                    "confidence": extract["confidence"],
                    "source": extract["source"],
                    "candidates": extract["candidates"],
                    "crop_rgb": cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                    "answer_crop_rgb": (
                        cv2.cvtColor(answer_focus, cv2.COLOR_BGR2RGB)
                        if answer_focus is not None
                        else None
                    ),
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
