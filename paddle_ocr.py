"""PaddleOCR(한국어) 래퍼. EasyOCR과 같은 인터페이스로 readtext 흉내."""
from typing import Optional

import numpy as np

_paddle_ocr = None


def get_paddle_ocr():
    global _paddle_ocr
    if _paddle_ocr is not None:
        return _paddle_ocr
    from paddleocr import PaddleOCR
    # Windows mkldnn 호환성 이슈 회피용 옵션 후보들
    for kwargs in (
        {"lang": "korean", "use_textline_orientation": True, "enable_mkldnn": False},
        {"lang": "korean", "use_textline_orientation": True},
        {"lang": "korean", "use_angle_cls": True, "enable_mkldnn": False},
        {"lang": "korean", "use_angle_cls": True},
        {"lang": "korean", "enable_mkldnn": False},
        {"lang": "korean"},
    ):
        try:
            _paddle_ocr = PaddleOCR(**kwargs)
            return _paddle_ocr
        except (TypeError, ValueError):
            continue
    raise RuntimeError("PaddleOCR 초기화 실패")


def readtext(image_bgr: np.ndarray) -> list[tuple[list[tuple[int, int]], str, float]]:
    """EasyOCR과 동일한 (box, text, conf) 형식으로 반환.
    box는 4-corner [(x,y), ...]"""
    ocr = get_paddle_ocr()
    # 신 API: predict(); 구 API: ocr()
    results: list = []
    try:
        out = ocr.predict(image_bgr)
    except (AttributeError, TypeError):
        out = ocr.ocr(image_bgr, cls=True)

    # 출력 구조 다양성 흡수
    if not out:
        return []
    page = out[0] if isinstance(out, list) else out
    if page is None:
        return []
    # 신 API: dict 형태
    if isinstance(page, dict):
        boxes = page.get("rec_polys", []) or page.get("dt_polys", [])
        texts = page.get("rec_texts", [])
        scores = page.get("rec_scores", [])
        for box, text, score in zip(boxes, texts, scores):
            pts = [(int(p[0]), int(p[1])) for p in box]
            results.append((pts, text, float(score)))
        return results
    # 구 API: 리스트(of [box, (text, conf)])
    for entry in page:
        if entry is None or len(entry) < 2:
            continue
        box, ts = entry[0], entry[1]
        text, conf = ts if isinstance(ts, (list, tuple)) else (ts, 1.0)
        pts = [(int(p[0]), int(p[1])) for p in box]
        results.append((pts, text, float(conf)))
    return results
