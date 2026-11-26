#!/usr/bin/env python3
"""합성 데이터로 필기 지우개 모델 학습.

  python train.py --steps 6000 --out eraser.pt                # 합성 페이지만
  python train.py --clean-pdf 원본시험지.pdf --steps 6000       # 필기 없는 실제 PDF로 품질 향상
  python train.py --resume eraser.pt --steps 3000              # 이어서 학습
"""
import argparse
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

import synth
from model import EraserUNet


def load_clean_pdf_pages(paths, dpi=200):
    import pymupdf
    pages = []
    for p in paths:
        for pg in pymupdf.open(p):
            pix = pg.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
            pages.append(np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy())
    return pages


@torch.no_grad()
def real_handwriting_ignore(pages, model_path):
    """필기가 있는 실제 스캔: 기존 모델로 필기 위치를 찾아 학습에서 제외할 영역을 만든다.
    인쇄된 그래프·도형·직선은 제외하지 않아야 '지우면 안 되는 것'으로 학습된다."""
    import cv2
    from erase_ai import hysteresis, load_model, run_tiled
    from protect import printed_blobs, printed_curves, printed_symbols, protect_mask
    net = load_model(model_path, "cuda" if torch.cuda.is_available() else "cpu")
    out = []
    for g in pages:
        _, prob = run_tiled(net, g, next(net.parameters()).device)
        m = cv2.dilate(hysteresis(prob, 0.4, min_area=30), np.ones((3, 3), np.uint8), iterations=2)
        m[printed_curves(g, m) > 0] = 0
        m[printed_blobs(g, m, prob=prob) > 0] = 0
        m[protect_mask(g, prob=prob) > 0] = 0
        m[printed_symbols(g) > 0] = 0
        out.append(cv2.dilate(m, np.ones((21, 21), np.uint8)))
        print(f"  실제 필기 제외 영역 {out[-1].mean()*100:.1f}%")
    return out


def load_contrib(root):
    """웹에서 사용자가 보내 준 (원본, 결과) 쌍. 둘의 차이 = 지운 필기 → 그대로 정답으로 사용."""
    import glob
    import os

    import cv2
    pairs = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        o, r = os.path.join(d, "orig.jpg"), os.path.join(d, "result.jpg")
        if not (os.path.exists(o) and os.path.exists(r)):
            continue
        a, b = cv2.imread(o, 0), cv2.imread(r, 0)
        if a is None or b is None or a.shape != b.shape:
            continue
        m = cv2.dilate((cv2.absdiff(a, b) > 25).astype(np.uint8), np.ones((3, 3), np.uint8))
        pairs.append((a, b, m))
    print(f"기여 자료 {len(pairs)}쪽")
    return pairs


def contrib_sample(pair, size):
    import cv2
    a, b, m = pair
    H, W = a.shape
    s = random.uniform(0.85, 1.2)
    ps = int(size / s)
    y, x = random.randint(0, H - ps), random.randint(0, W - ps)
    crop = lambda z, it=cv2.INTER_AREA: cv2.resize(z[y:y + ps, x:x + ps], (size, size), interpolation=it)
    inp, tgt, mask = crop(a), crop(b), crop(m, cv2.INTER_NEAREST)
    if random.random() < 0.3:  # 합성 필기를 조금 더 얹어 데이터 늘리기
        hw = synth._pen_layer(size, size, scale=s)
        inp = np.minimum(inp, hw)
        mask = np.maximum(mask, (hw < 200).astype(np.uint8))
    return inp, tgt, mask, np.ones((size, size), np.float32)


class Synth(IterableDataset):
    def __init__(self, n_pages, real_pages, size, scans=(), scan_ign=(), contrib=()):
        self.n_pages, self.real, self.size = n_pages, real_pages, size
        self.scans, self.scan_ign = list(scans), list(scan_ign)
        self.contrib = list(contrib)

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        seed = (wi.id if wi else 0) * 9973 + int(time.time())
        random.seed(seed); np.random.seed(seed % 2**32)
        pages = [synth.render_clean_page() for _ in range(self.n_pages)]
        i = 0
        while True:
            r = random.random()
            if self.contrib and r < 0.5:
                inp, tgt, m, v = contrib_sample(random.choice(self.contrib), self.size)
            elif self.scans and r < 0.7:
                k = random.randrange(len(self.scans))
                inp, tgt, m, v = synth.make_sample(self.scans[k], self.size, self.scan_ign[k], real=True)
            else:
                pool = self.real if self.real and random.random() < 0.6 else pages
                inp, tgt, m, v = synth.make_sample(random.choice(pool), self.size, real=pool is self.real)
            yield (torch.from_numpy(inp)[None].float() / 255, torch.from_numpy(tgt)[None].float() / 255,
                   torch.from_numpy(m)[None].float(), torch.from_numpy(v)[None])
            i += 1
            if i % 200 == 0:  # 페이지를 조금씩 새로 그림
                pages[random.randrange(len(pages))] = synth.render_clean_page()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--pages", type=int, default=20, help="워커당 합성 페이지 수")
    ap.add_argument("--clean-pdf", nargs="*", default=[], help="필기 없는 실제 PDF (선택)")
    ap.add_argument("--scan-pdf", nargs="*", default=[],
                    help="필기가 있는 실제 스캔 PDF (--resume 모델로 기존 필기를 찾아 제외하고 배경으로 사용)")
    ap.add_argument("--contrib-dir", help="웹에서 모은 학습 자료 폴더 (예: /content/drive/MyDrive/CleanPDF_학습자료)")
    ap.add_argument("--real-npz", nargs="*", default=[], help="tools/prepare_real.py 로 만든 실제 자료 (손글씨 조각 + 배경)")
    ap.add_argument("--resume")
    ap.add_argument("--out", default="eraser.pt")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.resume, map_location=dev) if a.resume else None
    if ck:
        a.width = ck.get("width", a.width)  # 이어서 학습할 땐 기존 모델 크기를 따름
    net = EraserUNet(a.width).to(dev)
    if ck:
        net.load_state_dict(ck["model"])
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=a.steps, pct_start=0.05)
    real = load_clean_pdf_pages(a.clean_pdf)
    scans = load_clean_pdf_pages(a.scan_pdf)
    ign = real_handwriting_ignore(scans, a.resume) if scans else []
    for f in a.real_npz:
        d = np.load(f, allow_pickle=True)
        synth.STAMPS.extend([np.asarray(x, np.uint8) for x in d["stamps"]])
        pg = [np.asarray(x, np.uint8) for x in d["pages"]]
        auto = real_handwriting_ignore(pg, a.resume) if a.resume else [np.zeros_like(p) for p in pg]
        scans += pg
        ign += [np.maximum(i0.astype(np.uint8), i1.astype(np.uint8)) for i0, i1 in zip([np.asarray(x, np.uint8) for x in d["ignore"]], auto)]
    print(f"실제 손글씨 조각 {len(synth.STAMPS)}개, 실제 배경 {len(scans)}쪽")
    contrib = load_contrib(a.contrib_dir) if a.contrib_dir else []
    dl = DataLoader(Synth(a.pages, real, a.size, scans, ign, contrib), batch_size=a.batch, num_workers=a.workers)
    print(f"device={dev}, params={sum(p.numel() for p in net.parameters())/1e6:.2f}M, 실제페이지={len(real)}")

    t0, run = time.time(), None
    for step, (x, y, m, v) in enumerate(dl, 1):
        x, y, m, v = x.to(dev), y.to(dev), m.to(dev), v.to(dev)
        clean, logit = net(x)
        w = (1 + 10 * F.max_pool2d(m, 5, 1, 2)) * v  # 필기 주변을 더 중요하게, 제외영역은 0
        bce = F.binary_cross_entropy_with_logits(logit, m, pos_weight=torch.tensor(3.0, device=dev), reduction="none")
        loss = (w * (clean - y).abs()).mean() + 0.3 * (bce * v).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
        if step % 50 == 0:
            print(f"step {step}/{a.steps}  loss {run:.4f}  {time.time() - t0:.0f}s", flush=True)
        if step % 500 == 0 or step == a.steps:
            torch.save({"model": net.state_dict(), "width": a.width}, a.out)
        if step >= a.steps:
            break
    print("저장:", a.out)


if __name__ == "__main__":
    main()
