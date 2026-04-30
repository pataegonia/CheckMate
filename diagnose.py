"""현재 파이프라인이 sample PDF에서 어디서 실패하는지 진단.

사용:
    python diagnose.py samples/Notes_260430_134520.pdf

결과:
    - 콘솔: 페이지별 앵커 검출 결과 + 학생 답 추출 결과
    - debug_out/page_<n>_debug.png: 앵커/OCR 박스 시각화
    - debug_out/page_<n>_problem_<num>.png: 각 문제 크롭
"""

import sys
from pathlib import Path

import cv2
import fitz
import numpy as np
from PIL import Image

from answers import ANSWERS
from pipeline import (
    cluster_columns,
    extract_handwritten_answer_v2,
    find_anchors,
    visualize_debug,
)


def pdf_pages(path: Path, dpi: int = 200) -> list[np.ndarray]:
    doc = fitz.open(path)
    zoom = dpi / 72
    matrix = fitz.Matrix(zoom, zoom)
    pages = []
    for page in doc:
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        pages.append(img.copy())
    doc.close()
    return pages


def diagnose(input_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    valid_keys = set(ANSWERS.keys())

    if input_path.suffix.lower() == ".pdf":
        pages_rgb = pdf_pages(input_path)
    else:
        pages_rgb = [np.array(Image.open(input_path).convert("RGB"))]

    print(f"\n=== 입력: {input_path} / 페이지 {len(pages_rgb)}개 ===")
    print(f"=== 정답 dict 키: {sorted(valid_keys)} ===\n")

    for i, page_rgb in enumerate(pages_rgb, 1):
        print(f"\n----- 페이지 {i} (크기 {page_rgb.shape[1]}x{page_rgb.shape[0]}) -----")
        page_bgr = cv2.cvtColor(page_rgb, cv2.COLOR_RGB2BGR)
        h, w = page_bgr.shape[:2]

        anchors, all_ocr = find_anchors(page_bgr, valid_keys)
        found = sorted(a["num"] for a in anchors)
        missing = sorted(valid_keys - set(found))
        print(f"  검출된 앵커: {found}")
        print(f"  ❌ 누락된 앵커: {missing}")
        print(f"  전체 OCR 박스 수: {len(all_ocr)}")

        for a in sorted(anchors, key=lambda a: a["num"]):
            print(
                f"    · {a['num']:>4}: text=`{a['text']}` conf={a['conf']:.2f} "
                f"box=({a['x_left']},{a['y_top']})~({a['x_right']},{a['y_bottom']})"
            )

        # 누락된 앵커 영역에서 EasyOCR이 무엇을 읽었는지 살펴보기
        for miss_num in missing:
            print(f"\n  [missing {miss_num}] 페이지 OCR 중 위쪽 영역 텍스트:")
            for box, text, conf in all_ocr:
                ys = [p[1] for p in box]
                xs = [p[0] for p in box]
                yc = sum(ys) / 4
                xc = sum(xs) / 4
                # 페이지 상단 700px 안의 모든 결과
                if yc < 700:
                    print(
                        f"      · `{text}` conf={conf:.2f} "
                        f"at ({int(min(xs))},{int(min(ys))})~({int(max(xs))},{int(max(ys))})"
                    )

        debug_bgr = visualize_debug(page_bgr, anchors, all_ocr)
        cv2.imwrite(str(out_dir / f"page_{i}_debug.png"), debug_bgr)

        if not anchors:
            print("  앵커가 없어 답 추출 스킵")
            continue

        cols = cluster_columns(anchors, w)
        print(f"  컬럼 분할: {cols}")

        from collections import defaultdict
        by_col: dict[int, list[dict]] = defaultdict(list)
        for a in anchors:
            cx = (a["x_left"] + a["x_right"]) / 2
            col_idx = 0
            for ci, (cx1, cx2) in enumerate(cols):
                if cx1 <= cx < cx2:
                    col_idx = ci
                    break
            by_col[col_idx].append(a)
        for col_idx in by_col:
            by_col[col_idx].sort(key=lambda a: a["y_top"])

        print(f"\n  학생 답 추출 결과:")
        for col_idx, col_anchors in by_col.items():
            cx1, cx2 = cols[col_idx]
            for j, a in enumerate(col_anchors):
                y1 = max(0, a["y_top"] - 5)
                if j + 1 < len(col_anchors):
                    y2 = max(y1 + 1, col_anchors[j + 1]["y_top"] - 5)
                else:
                    y2 = h
                crop = page_bgr[y1:y2, cx1:cx2]
                cv2.imwrite(str(out_dir / f"page_{i}_problem_{a['num']}.png"), crop)

                student_ans = extract_handwritten_answer_v2(crop, problem_num=a["num"])
                correct_ans = ANSWERS.get(a["num"])
                ok = "✅" if student_ans.strip() == str(correct_ans).strip() else "❌"
                print(
                    f"    {ok} {a['num']}: 학생=`{student_ans}` / 정답=`{correct_ans}`"
                )

    print(f"\n=== 디버그 이미지 저장: {out_dir.resolve()} ===")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python diagnose.py <image-or-pdf>")
        sys.exit(1)
    diagnose(Path(sys.argv[1]), Path("debug_out"))
