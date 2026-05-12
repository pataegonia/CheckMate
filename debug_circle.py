"""객관식 동그라미 검출 디버그: 다크 마스크, OCR 박스 빼기 결과, 후보 contour 시각화."""
import sys
from pathlib import Path

import cv2
import numpy as np

from pipeline import _build_dark_ink_mask, _try_paddle_ocr, _OPTION_MARKS


def main(crop_paths: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in crop_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        ocr = _try_paddle_ocr(img) or []
        mask = _build_dark_ink_mask(img, ocr, top_skip=60)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 시각화 이미지: 원본 위에 마스크 초록 오버레이 + contour 빨강
        viz = img.copy()
        viz[mask_closed > 0] = (
            viz[mask_closed > 0].astype(np.float32) * 0.4
            + np.array([0, 255, 0], dtype=np.float32) * 0.6
        ).astype(np.uint8)

        opts = []
        for box, text, _ in ocr:
            for ch, d in _OPTION_MARKS.items():
                if ch in text:
                    cx = sum(pt[0] for pt in box) / 4
                    cy = sum(pt[1] for pt in box) / 4
                    opts.append((cx, cy, d))
                    cv2.circle(viz, (int(cx), int(cy)), 8, (255, 200, 0), 2)
                    cv2.putText(viz, d, (int(cx) - 8, int(cy) - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
                    break

        print(f"\n=== {p.name} ===")
        print(f"  mask px: {int(mask_closed.sum() / 255)}")
        print(f"  options found: {opts}")
        print(f"  contours (area>=100):")
        for i, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area < 100:
                continue
            perim = cv2.arcLength(c, True)
            circ = 4 * np.pi * area / (perim * perim) if perim > 0 else 0
            x, y, bw, bh = cv2.boundingRect(c)
            ratio = max(bw, bh) / max(1, min(bw, bh))
            (cx, cy), _ = cv2.minEnclosingCircle(c)
            print(f"    [{i}] area={area:.0f} circ={circ:.2f} ratio={ratio:.1f} center=({cx:.0f},{cy:.0f}) bbox={x},{y},{bw}x{bh}")
            cv2.drawContours(viz, [c], -1, (0, 0, 255), 2)
            cv2.putText(viz, f"#{i}", (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        cv2.imwrite(str(out_dir / f"circle_{p.stem}.png"), viz)


if __name__ == "__main__":
    crops = [Path(p) for p in sys.argv[1:]]
    main(crops, Path("debug_out/circles"))
