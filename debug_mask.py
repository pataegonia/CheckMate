"""각 크롭의 손글씨 색마스크와 추출된 bbox를 시각화해서 저장."""
import sys
from pathlib import Path

import cv2
import numpy as np

from pipeline import _detect_handwriting_mask, _largest_handwriting_region


def main(crop_paths: list[Path], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in crop_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        mask = _detect_handwriting_mask(img)
        bbox = _largest_handwriting_region(mask)

        # mask overlay
        overlay = img.copy()
        overlay[mask > 0] = [0, 255, 0]  # green where mask
        blend = cv2.addWeighted(img, 0.5, overlay, 0.5, 0)

        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(blend, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # also save just hw_only image
        hw = np.full_like(img, 255)
        hw[mask > 0] = [0, 0, 0]

        out_path = out_dir / f"mask_{p.stem}.png"
        cv2.imwrite(str(out_path), blend)
        out_hw = out_dir / f"hwonly_{p.stem}.png"
        cv2.imwrite(str(out_hw), hw)

        print(f"{p.name}: mask px={int(mask.sum()/255)}, bbox={bbox} → {out_path.name}")


if __name__ == "__main__":
    crops = [Path(p) for p in sys.argv[1:]]
    main(crops, Path("debug_out/masks"))
