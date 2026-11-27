#!/usr/bin/env python3
"""태블릿 필기(Ink 주석)가 있는 PDF → 완벽한 정답 쌍 학습 자료.

필기가 주석으로 저장된 쪽은 '필기 있음/없음' 두 장을 그대로 렌더링할 수 있어서
사람이 표시하지 않아도 정확한 정답이 생깁니다.

  python tools/annot_pairs.py 노트.pdf -o pairs/        # pairs/노트_p51/orig.jpg, result.jpg ...
  python tools/annot_pairs.py 노트.pdf -o pairs/ --stamps stamps.npz   # 손글씨 조각도 저장
학습: python train.py --resume eraser.pt --contrib-dir pairs/ --real-npz stamps.npz
"""
import argparse
import os

import cv2
import numpy as np
import pymupdf

HANDWRITING = {pymupdf.PDF_ANNOT_INK, pymupdf.PDF_ANNOT_HIGHLIGHT, pymupdf.PDF_ANNOT_UNDERLINE,
               pymupdf.PDF_ANNOT_STRIKE_OUT, pymupdf.PDF_ANNOT_SQUIGGLY, pymupdf.PDF_ANNOT_LINE,
               pymupdf.PDF_ANNOT_SQUARE, pymupdf.PDF_ANNOT_CIRCLE, pymupdf.PDF_ANNOT_POLYGON,
               pymupdf.PDF_ANNOT_POLY_LINE}


def render(page, dpi):
    px = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, annots=True)
    return np.frombuffer(px.samples, np.uint8).reshape(px.height, px.width).copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("-o", "--out", default="pairs")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--stamps", help="손글씨 조각 npz 저장 경로")
    a = ap.parse_args()
    name = os.path.splitext(os.path.basename(a.pdf))[0]
    with_ink = pymupdf.open(a.pdf)
    no_ink = pymupdf.open(a.pdf)
    stamps, n = [], 0
    for i in range(len(with_ink)):
        p0, p1 = with_ink[i], no_ink[i]
        hw = [an for an in p1.annots() if an.type[0] in HANDWRITING]
        if not hw:
            continue
        for an in list(p1.annots()):
            if an.type[0] in HANDWRITING:
                p1.delete_annot(an)
        orig, clean = render(p0, a.dpi), render(p1, a.dpi)
        d = os.path.join(a.out, f"{name}_p{i + 1}")
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, "orig.jpg"), orig, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(d, "result.jpg"), clean, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
        if a.stamps:  # 필기 부분만 잘라 조각으로
            m = (cv2.absdiff(orig, clean) > 25).astype(np.uint8)
            grp = cv2.dilate(m, np.ones((int(0.12 * a.dpi), int(0.2 * a.dpi)), np.uint8))
            k_, lab, st, _ = cv2.connectedComponentsWithStats(grp)
            keep = cv2.dilate(m, np.ones((3, 3), np.uint8))
            for k in range(1, k_):
                x, y, w, h, _ = st[k]
                if w < 15 or h < 15 or m[y:y + h, x:x + w].sum() < 150:
                    continue
                crop = np.where((lab[y:y + h, x:x + w] == k) & (keep[y:y + h, x:x + w] > 0), orig[y:y + h, x:x + w], 255)
                stamps.append(crop.astype(np.uint8))
    print(f"정답 쌍 {n}쪽 → {a.out}")
    if a.stamps:
        arr = np.empty(len(stamps), dtype=object)
        for j, s in enumerate(stamps):
            arr[j] = s
        np.savez_compressed(a.stamps, stamps=arr, pages=np.empty(0, dtype=object), ignore=np.empty(0, dtype=object))
        print(f"손글씨 조각 {len(stamps)}개 → {a.stamps}")


if __name__ == "__main__":
    main()
