#!/usr/bin/env python3
"""AI 필기 지우개: 검정 펜 필기까지 지우고, 필기에 가려진 인쇄 글씨를 복원합니다.

  python erase_ai.py 시험지.pdf                    # → 시험지_clean.pdf
  python erase_ai.py 시험지.pdf --model eraser.pt --dpi 200 --strength 0.5
"""
import argparse

import cv2
import numpy as np
import pymupdf
import torch
import torch.nn.functional as F

from model import EraserUNet


def load_model(path, dev):
    ck = torch.load(path, map_location=dev)
    net = EraserUNet(ck.get("width", 32)).to(dev).eval()
    net.load_state_dict(ck["model"])
    return net


@torch.no_grad()
def run_tiled(net, gray, dev, tile=512, pad=48):
    """큰 페이지를 겹치는 타일로 나눠 처리."""
    H, W = gray.shape
    x = torch.from_numpy(gray).float().div(255)[None, None]
    clean = torch.zeros_like(x)
    prob = torch.zeros_like(x)
    step = tile - 2 * pad
    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            ya, xa = max(0, y0 - pad), max(0, x0 - pad)
            yb, xb = min(H, y0 + step + pad), min(W, x0 + step + pad)
            t = x[..., ya:yb, xa:xb]
            ph, pw = (-t.shape[2]) % 16, (-t.shape[3]) % 16
            t = F.pad(t, (0, pw, 0, ph), mode="replicate").to(dev)
            c, l = net(t)
            c, l = c[..., :yb - ya, :xb - xa].cpu(), l[..., :yb - ya, :xb - xa].sigmoid().cpu()
            sy, sx = y0 - ya, x0 - xa
            ey, ex = min(step, H - y0), min(step, W - x0)
            clean[..., y0:y0 + ey, x0:x0 + ex] = c[..., sy:sy + ey, sx:sx + ex]
            prob[..., y0:y0 + ey, x0:x0 + ex] = l[..., sy:sy + ey, sx:sx + ex]
    return clean[0, 0].numpy(), prob[0, 0].numpy()


def hysteresis(prob, thr, min_area=30):
    """확실한 필기(prob>thr)와 이어진 약한 후보(prob>thr/2)까지 포함, 너무 작은 점은 무시."""
    weak = (prob > thr * 0.7).astype(np.uint8)
    strong = prob > thr
    n, lab, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
    keep = np.zeros(n, bool)
    s_cnt = np.bincount(lab[strong], minlength=n)
    keep[(s_cnt >= min_area)] = True
    keep[0] = False
    return keep[lab].astype(np.uint8)


def erase_page(net, gray, dev, thr=0.5, grow=2):
    clean, prob = run_tiled(net, gray, dev)
    # 필기로 판단된 곳(+주변)만 모델 복원본으로 교체, 나머지는 원본 그대로 → 인쇄 글씨 선명도 유지
    m = hysteresis(prob, thr, min_area=30)
    m = cv2.dilate(m, np.ones((3, 3), np.uint8), iterations=grow).astype(np.float32)
    m = cv2.GaussianBlur(m, (5, 5), 0)
    out = gray.astype(np.float32) / 255 * (1 - m) + clean * m
    return (np.clip(out, 0, 1) * 255).astype(np.uint8), prob


def main(argv=None):
    ap = argparse.ArgumentParser(description="AI 필기 지우개 (복원 포함)")
    ap.add_argument("input")
    ap.add_argument("-o", "--output")
    ap.add_argument("--model", default="eraser.pt")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--thr", type=float, default=0.5, help="필기 판정 기준 (낮추면 더 많이 지움)")
    ap.add_argument("--grow", type=int, default=2, help="지우는 영역 확장")
    ap.add_argument("--quality", type=int, default=90)
    ap.add_argument("--pages", help="처리할 쪽 (예: 1-3,5)")
    ap.add_argument("--save-mask", action="store_true", help="필기 감지 결과도 PDF로 저장")
    a = ap.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = load_model(a.model, dev)
    src = pymupdf.open(a.input)
    sel = range(len(src))
    if a.pages:
        sel = [i - 1 for part in a.pages.split(",") for i in
               (range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1))]
    out_path = a.output or a.input.rsplit(".", 1)[0] + "_clean.pdf"
    out, mdoc = pymupdf.open(), pymupdf.open()
    for i in sel:
        page = src[i]
        pix = page.get_pixmap(dpi=a.dpi, colorspace=pymupdf.csGRAY)
        gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
        res, prob = erase_page(net, gray, dev, a.thr, a.grow)
        for doc, im in [(out, res)] + ([(mdoc, (255 - (prob > a.thr) * 200).astype(np.uint8))] if a.save_mask else []):
            buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, a.quality])[1].tobytes()
            np_ = doc.new_page(width=page.rect.width, height=page.rect.height)
            np_.insert_image(np_.rect, stream=buf)
        print(f"[{i + 1}/{len(src)}] 필기 영역 {(prob > a.thr).mean() * 100:.1f}%", flush=True)
    out.save(out_path, garbage=3, deflate=True)
    if a.save_mask:
        mdoc.save(out_path.replace(".pdf", "_mask.pdf"))
    print("완료 →", out_path)


if __name__ == "__main__":
    main()
