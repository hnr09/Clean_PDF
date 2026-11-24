"""인쇄된 그래프·도형·선을 찾아 '지우면 안 되는 영역'을 만든다."""
import cv2
import numpy as np


def _segments(lines, horizontal):
    n, lab, st, _ = cv2.connectedComponentsWithStats(lines)
    segs = []
    for k in range(1, n):
        x, y, w, h, _ = st[k]
        segs.append((x, x + w, y + h // 2) if horizontal else (x + w // 2, y, y + h))
    return segs


def line_masks(gray, dpi=200, length=0.35):
    ink = (gray < 170).astype(np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))  # 끊긴 스캔 선 잇기
    L = int(length * dpi)  # length 인치 이상 곧은 선
    hor = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1)))
    ver = cv2.morphologyEx(ink, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, L)))
    return hor, ver


def figure_boxes(gray, dpi=200):
    """좌표축(세로축 아래 끝 ≈ 가로축 왼쪽 끝) 을 찾아 그래프 영역 상자 목록을 반환."""
    hor, ver = line_masks(gray, dpi, length=0.22)
    H = _segments(hor, True)
    V = _segments(ver, False)
    tol = int(0.06 * dpi)
    boxes = []
    for vx, vy0, vy1 in V:
        for hx0, hx1, hy in H:
            if abs(hy - vy1) > tol or abs(hx0 - vx) > tol:
                continue
            # 표/상자 제외: 이 세로선 왼쪽 끝에서 시작하는 다른 가로선(위쪽 테두리, 행 구분선)이 있으면 표
            others = [1 for (a0, a1, yy) in H if abs(a0 - vx) <= tol and vy0 - tol <= yy < hy - tol
                      and (a1 - a0) >= 0.8 * (hx1 - hx0)]
            # 가로축 오른쪽 끝에 세로선이 있으면 닫힌 상자
            closed = any(abs(x - hx1) <= tol and y0 < hy - tol and abs(y1 - hy) <= tol for (x, y0, y1) in V if x != vx)
            if others or closed:
                continue
            m = int(0.25 * dpi)
            boxes.append((max(0, vx - m), max(0, vy0 - m // 2), hx1 + m // 2, hy + m))
    return boxes


def protect_mask(gray, dpi=200):
    """1 = 지우지 말 것."""
    hor, ver = line_masks(gray, dpi)
    keep = cv2.dilate(hor | ver, np.ones((3, 3), np.uint8))
    for x0, y0, x1, y1 in figure_boxes(gray, dpi):
        keep[y0:y1, x0:x1] = 1
    return keep


def printed_curves(gray, cand, dpi=200):
    """후보 마스크(cand) 덩어리 중 '인쇄된 함수 그래프 곡선'처럼 생긴 것을 골라 1로 표시.
    조건: 충분히 길고(가로 0.5인치+), 가로가 세로보다 길고, 대부분의 x 위치에서 획이 1~2개뿐(함수 모양)이며 휘어 있음."""
    keep = np.zeros_like(cand, dtype=np.uint8)
    ink = gray < 170
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    for k in range(1, n):
        x, y, w, h, area = st[k]
        if w < 0.5 * dpi or w < 1.2 * h or h < 0.08 * dpi:
            continue
        comp = (lab[y:y + h, x:x + w] == k) & ink[y:y + h, x:x + w]
        col = comp.any(0)
        if col.mean() < 0.7:
            continue
        runs = (np.diff(comp.astype(np.int8), axis=0) == 1).sum(0) + comp[0]
        runs = runs[col]
        one = (runs == 1).mean()
        if one < 0.55 or (runs >= 3).mean() > 0.15:
            continue
        # 직선(밑줄)은 제외: 각 열의 중심 y를 직선으로 맞췄을 때 잔차가 작으면 밑줄
        ys = np.array([np.nonzero(comp[:, c])[0].mean() for c in np.nonzero(col)[0]])
        xs = np.nonzero(col)[0]
        resid = np.abs(ys - np.polyval(np.polyfit(xs, ys, 1), xs)).mean()
        if resid < 0.02 * dpi:
            continue
        keep[lab == k] = 1
    return keep


def printed_blobs(gray, cand, dpi=200, max_pen=0.04, max_fill=0.52):
    """후보 덩어리 중 인쇄물로 보이는 것: 펜보다 훨씬 굵은 채워진 도형(나무 아이콘, ● 등)
    또는 잉크가 빽빽한 인쇄 글자 묶음(제목 배지 등)."""
    keep = np.zeros_like(cand, dtype=np.uint8)
    ink = (gray < 140).astype(np.uint8)
    dt = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    for k in range(1, n):
        x, y, w, h, area = st[k]
        comp = lab[y:y + h, x:x + w] == k
        sel = comp & (ink[y:y + h, x:x + w] > 0)
        if sel.sum() < 40:
            continue
        sw = 2 * np.percentile(dt[y:y + h, x:x + w][sel], 95)
        if sw > max_pen * dpi or sel.sum() / area > max_fill:
            keep[y:y + h, x:x + w][comp] = 1
    return keep
