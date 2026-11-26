#!/usr/bin/env python3
"""실제 스캔본으로 학습 자료 만들기.

boxes.json 형식 (pt 단위, 쪽 번호는 1부터):
  {"print_boxes": {"1": [75,100,345,470], ...}}   # 인쇄된 문제 영역. 이 밖의 잉크 = 손필기
출력 (npz):
  stamps: 진짜 손글씨 조각들 (흰 배경 255, 잉크만)   → 합성 필기 대신 붙여 넣음
  pages / ignore: 배경 페이지와 학습 제외 영역 (손필기가 있는 곳)

  python tools/prepare_real.py 스캔.pdf boxes.json -o real_data.npz
"""
import argparse
import json

import cv2
import numpy as np
import pymupdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("boxes")
    ap.add_argument("-o", "--out", default="real_data.npz")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--margin", type=float, default=14, help="페이지 가장자리 무시 (pt)")
    a = ap.parse_args()
    cfg = json.load(open(a.boxes))
    boxes = {int(k): v for k, v in cfg.get("print_boxes", {}).items()}
    s = a.dpi / 72
    doc = pymupdf.open(a.pdf)
    stamps, pages, ignores = [], [], []
    for i, page in enumerate(doc, 1):
        px = page.get_pixmap(dpi=a.dpi, colorspace=pymupdf.csGRAY)
        g = np.frombuffer(px.samples, np.uint8).reshape(px.height, px.width).copy()
        H, W = g.shape
        m = int(a.margin * s)
        ink = (g < 160).astype(np.uint8)
        hw = np.zeros_like(ink)
        if i in boxes:
            x0, y0, x1, y1 = [int(v * s) for v in boxes[i]]
            hw[m:H - m, m:W - m] = 1
            hw[y0:y1, x0:x1] = 0
            hw &= ink
            # 페이지 테두리 선/스캔 그림자 제거: 아주 길고 얇은 덩어리
            n, lab, st, _ = cv2.connectedComponentsWithStats(hw)
            for k in range(1, n):
                x, y, w, h, ar = st[k]
                if (w > 0.4 * W and h < 0.03 * H) or (h > 0.4 * H and w < 0.03 * W):
                    hw[lab == k] = 0
            # 단어/줄 단위로 묶어 조각 만들기
            grp = cv2.dilate(hw, np.ones((int(0.12 * a.dpi), int(0.2 * a.dpi)), np.uint8))
            n, lab, st, _ = cv2.connectedComponentsWithStats(grp)
            keep = cv2.dilate(hw, np.ones((3, 3), np.uint8))
            for k in range(1, n):
                x, y, w, h, ar = st[k]
                if w < 15 or h < 15 or (hw[y:y + h, x:x + w].sum() < 60):
                    continue
                if max(w, h) / min(w, h) > 5:  # 가늘고 긴 조각 = 테두리선/스캔 자국일 가능성
                    continue
                crop = np.where((lab[y:y + h, x:x + w] == k) & (keep[y:y + h, x:x + w] > 0), g[y:y + h, x:x + w], 255)
                ys, xs = np.nonzero(crop < 160)
                if len(ys) < 150:
                    continue
                ih, iw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
                if max(ih, iw) / max(1, min(ih, iw)) > 5:
                    continue
                stamps.append(crop.astype(np.uint8))
            ign = cv2.dilate(hw, np.ones((25, 25), np.uint8))
        else:
            ign = np.zeros_like(ink)
        pages.append(g)
        ignores.append(ign)
        print(f"{i}쪽: 손글씨 조각 누적 {len(stamps)}개")
    def objs(lst):  # 크기가 달라도/같아도 '배열 목록' 그대로 저장
        arr = np.empty(len(lst), dtype=object)
        for j, v in enumerate(lst):
            arr[j] = v
        return arr
    np.savez_compressed(a.out, stamps=objs(stamps), pages=objs(pages), ignore=objs(ignores))
    print("저장:", a.out)


if __name__ == "__main__":
    main()
