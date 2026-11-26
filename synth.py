"""합성 학습 데이터: (깨끗한 시험지 페이지) + (가짜 손필기) → (입력, 정답, 필기마스크)"""
import glob
import os
import random

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIRS = ["/usr/share/fonts/truetype/nanum", os.path.join(os.path.dirname(__file__), "fonts")]


def _fonts(names):
    out = []
    for d in FONT_DIRS:
        for n in names:
            out += glob.glob(os.path.join(d, n))
    return out


PRINT_FONTS = _fonts(["NanumGothic.ttf", "NanumGothicBold.ttf", "NanumGothicExtraBold.ttf",
                      "NanumMyeongjo.ttf", "NanumMyeongjoBold.ttf", "NanumBarunGothic.ttf",
                      "NanumBarunGothicBold.ttf", "NanumSquareR.ttf", "NanumSquareB.ttf"])
HAND_FONTS = _fonts(["NanumPen.ttf", "NanumBrush.ttf", "NanumBarunpenR.ttf", "NanumBarunpenB.ttf"])

SENTS = """표는 방형구법을 이용하여 어떤 지역의 식물 군집을 조사한 결과를 나타낸 것이다.
이 자료에 대한 설명으로 옳은 것만을 <보기>에서 있는 대로 고른 것은?
(단, A~D 이외의 종은 고려하지 않는다.) [3점]
그림은 세포 주기를 나타낸 것이다. 지표를 덮고 있는 면적이 가장 큰 종은 B이다.
A의 상대 빈도 값은 C의 상대 밀도 값보다 작다. 우점종은 C이다.
다음은 어떤 생태계에서 일어나는 물질 순환 과정에 대한 자료이다.
사람의 유전 형질 ㉠에 대한 자료이다. 이 가족 구성원의 핵형은 모두 정상이다.
그래프는 시간에 따른 개체군의 밀도 변화를 나타낸 것이다. 종 다양성은 Ⅰ에서가 Ⅱ에서보다 낮다.
실험 과정과 결과에 대한 설명으로 옳은 것은? 구간 Ⅰ에서 ATP가 합성된다.
흥분의 전도와 전달에 대한 자료이다. 막전위는 -70mV에서 +30mV로 변한다.
2026학년도 대학수학능력시험 문제지 과학탐구 영역 생명과학 홀수형
함수 f(x)=x^2-4x+3에 대하여 lim f(x)의 값은? 수열 {an}의 첫째항부터 제n항까지의 합을 Sn이라 할 때
다음 글의 목적으로 가장 적절한 것은? 밑줄 친 부분이 가리키는 대상이 나머지 넷과 다른 것은?
윗글의 내용과 일치하지 않는 것은? 그림과 같이 질량이 m인 물체가 빗면을 따라 운동한다.""".split("\n")
CELLS = ["A", "B", "C", "D", "Ⅰ", "Ⅱ", "?", "0", "0.1", "0.2", "0.4", "0.7", "12", "16", "20", "24",
         "30", "36", "40", "68", "112", "종", "빈도", "개체 수", "상대 밀도(%)", "상대 빈도(%)",
         "상대 피도(%)", "중요치", "지역", "시점", "t1", "t2", "㉠", "㉡", "?"]
HAND_TXT = list("0123456789") + ["24", "35", "40", "10", "45", "0.7", "56", "89", "50", "104",
                                 "?", "O", "X", "V", "답", "ㄱ", "ㄴ", "ㄷ", "A", "B", "C", "D",
                                 "×2", "+", "=", "→", "작다", "크다", "정답", "틀림", "1/2", "%"]


def _font(paths, size):
    return ImageFont.truetype(random.choice(paths), size)


def _wrap(draw, text, font, width):
    lines, cur = [], ""
    for ch in text:
        if draw.textlength(cur + ch, font=font) > width:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    return lines + [cur] if cur else lines


def render_clean_page(W=1654, H=2339):
    """시험지 느낌의 깨끗한 페이지를 그린다 (200dpi A4 기준)."""
    img = Image.new("L", (W, H), 255)
    d = ImageDraw.Draw(img)
    ncol = random.choice([1, 2, 2])
    margin = random.randint(60, 120)
    colw = (W - 2 * margin - (ncol - 1) * 60) // ncol
    if ncol == 2 and random.random() < 0.7:
        d.line([(W // 2, margin), (W // 2, H - margin)], fill=random.randint(0, 120), width=2)
    for c in range(ncol):
        x0, y = margin + c * (colw + 60), margin + random.randint(0, 60)
        while y < H - margin - 200:
            y = _block(d, x0, y, colw, H - margin) + random.randint(30, 140)
    return np.array(img)


def _block(d, x0, y, w, ymax):
    fs = random.randint(24, 36)
    lh = int(fs * random.uniform(1.4, 1.9))
    f = _font(PRINT_FONTS, fs)
    kind = random.random()
    text = random.choice(SENTS)
    if random.random() < 0.5:
        text = f"{random.randint(1, 20)}. " + text + " " + random.choice(SENTS)
    for line in _wrap(d, text, f, w):
        if y > ymax - lh:
            return y
        d.text((x0, y), line, font=f, fill=random.randint(0, 50))
        y += lh
    if kind < 0.3:  # 그래프/그림 (인쇄된 곡선은 지우면 안 됨)
        gh = random.randint(150, 320)
        if y + gh > ymax:
            return y
        _graph(d, x0 + random.randint(0, 40), y, min(w, random.randint(200, 480)), gh, fs)
        y += gh + lh // 2
    elif kind < 0.5:  # 표
        rows, cols = random.randint(3, 6), random.randint(3, 7)
        cw, rh = w // cols, int(fs * random.uniform(1.6, 2.2))
        if y + rows * rh > ymax:
            return y
        for r in range(rows + 1):
            d.line([(x0, y + r * rh), (x0 + cols * cw, y + r * rh)], fill=random.randint(0, 80), width=random.choice([1, 2, 2, 3]))
        for k in range(cols + 1):
            d.line([(x0 + k * cw, y), (x0 + k * cw, y + rows * rh)], fill=random.randint(0, 80), width=random.choice([1, 2]))
        cf = _font(PRINT_FONTS, int(fs * 0.85))
        for r in range(rows):
            for k in range(cols):
                t = random.choice(CELLS)
                tw = d.textlength(t, font=cf)
                if tw < cw - 4:
                    d.text((x0 + k * cw + (cw - tw) / 2, y + r * rh + rh * 0.2), t, font=cf, fill=random.randint(0, 50))
        y += rows * rh + lh // 2
    elif kind < 0.7:  # <보기> 박스
        n = random.randint(2, 4)
        bh = n * lh + lh
        if y + bh > ymax:
            return y
        d.rectangle([x0, y + lh // 2, x0 + w, y + bh], outline=random.randint(0, 80), width=2)
        d.text((x0 + w // 2 - fs * 1.5, y + 4), "<보기>", font=f, fill=0)
        yy = y + lh
        for j, ch in zip(range(n), "ㄱㄴㄷㄹ"):
            line = _wrap(d, ch + ". " + random.choice(SENTS), f, w - 40)[0]
            d.text((x0 + 20, yy), line, font=f, fill=random.randint(0, 50))
            yy += lh
        y += bh + lh // 2
    # 선택지
    opts = ["ㄱ", "ㄴ", "ㄷ", "ㄱ, ㄴ", "ㄱ, ㄷ", "ㄴ, ㄷ", "ㄱ, ㄴ, ㄷ", "1", "2", "3", "4", "5"]
    if random.random() < 0.8 and y + 2 * lh < ymax:
        for row in range(2):
            for j in range(3 if row == 0 else 2):
                d.text((x0 + j * w // 3, y), "①②③④⑤"[row * 3 + j] + " " + random.choice(opts), font=f, fill=0)
            y += lh
    return y


def _graph(d, x0, y0, w, h, fs):
    """축 + 곡선 + 점선 + 라벨이 있는 인쇄 그래프, 또는 도형 그림."""
    c = random.randint(0, 60)
    lw = random.choice([1, 2, 2, 3, 3, 4, 5])
    f = _font(PRINT_FONTS, int(fs * 0.7))
    r0 = random.random()
    if r0 < 0.2:  # 채워진 아이콘 (나무, 사람, ●▲■)
        for _ in range(random.randint(2, 7)):
            px, py = x0 + random.randint(0, max(1, w - 60)), y0 + random.randint(20, max(21, h - 40))
            s = random.randint(12, 40)
            k = random.random()
            if k < 0.4:  # 나무
                d.polygon([(px, py), (px - s // 2, py + s), (px + s // 2, py + s)], fill=c)
                d.rectangle([px - 2, py + s, px + 2, py + s + s // 3], fill=c)
            elif k < 0.7:
                d.ellipse([px, py, px + s, py + s], fill=c)
            else:
                d.rectangle([px, py, px + s, py + s // 2], fill=c)
        return
    if random.random() < 0.3:  # 축이 희미하거나 없는 그래프도 있음
        c_axis = random.randint(120, 200)
    else:
        c_axis = c
    if r0 < 0.8:
        ox, oy = x0 + 30, y0 + h - 25
        d.line([(ox, oy), (x0 + w, oy)], fill=c_axis, width=max(1, lw - 1))
        d.line([(ox, oy), (ox, y0)], fill=c_axis, width=max(1, lw - 1))
        d.polygon([(x0 + w, oy), (x0 + w - 8, oy - 4), (x0 + w - 8, oy + 4)], fill=c)
        d.polygon([(ox, y0), (ox - 4, y0 + 8), (ox + 4, y0 + 8)], fill=c)
        d.text((x0 + w - 30, oy + 4), random.choice(["시간", "t", "x", "온도"]), font=f, fill=c)
        for _ in range(random.randint(1, 3)):
            xs = np.linspace(ox, x0 + w - 15, 80)
            kind = random.random()
            u = (xs - ox) / max(1, (x0 + w - 15 - ox))
            a, b = random.uniform(0.1, 0.9), random.uniform(3, 12)
            if kind < 0.33:
                yv = 1 / (1 + np.exp(-b * (u - a)))
            elif kind < 0.66:
                yv = np.exp(-((u - a) ** 2) * b * 2)
            else:
                yv = 0.5 + 0.4 * np.sin(u * b + a * 6) * np.exp(-u * random.uniform(0, 2))
            ys = oy - 10 - yv * (h - 50)
            pts = list(zip(xs.tolist(), ys.tolist()))
            if random.random() < 0.3:  # 점선
                for k in range(0, len(pts) - 1, 3):
                    d.line(pts[k:k + 2], fill=c, width=lw)
            else:
                d.line(pts, fill=c, width=lw, joint="curve")
            d.text(pts[-1], random.choice(["A", "B", "I", "N1"]), font=f, fill=c)
        for _ in range(random.randint(0, 2)):  # 수직 점선 + 라벨
            xx = random.randint(ox + 20, x0 + w - 20)
            for yy in range(y0 + 10, oy, 8):
                d.line([(xx, yy), (xx, yy + 4)], fill=c, width=1)
            d.text((xx - 5, oy + 4), random.choice(["t1", "t2", "Ⅰ", "Ⅱ"]), font=f, fill=c)
    else:  # 도형 그림: 원, 화살표, 상자
        for _ in range(random.randint(3, 8)):
            px, py = x0 + random.randint(0, w - 60), y0 + random.randint(0, h - 60)
            r = random.randint(10, 50)
            t = random.random()
            if t < 0.35:
                d.ellipse([px, py, px + r, py + r], outline=c, width=lw)
            elif t < 0.6:
                d.rounded_rectangle([px, py, px + r * 2, py + r], radius=6, outline=c, width=lw)
                d.text((px + 6, py + 4), random.choice(["생태계", "A", "생산자", "물질"]), font=f, fill=c)
            else:
                q = (px + random.randint(-80, 80), py + random.randint(-40, 40))
                d.line([(px, py), q], fill=c, width=lw)
                d.polygon([q, (q[0] - 6, q[1] - 4), (q[0] - 6, q[1] + 4)], fill=c)


# ------------------------------------------------------------------ 손필기 합성
STAMPS = []  # 실제 스캔에서 잘라낸 진짜 손글씨 조각 (tools/prepare_real.py)


def _paste_stamp(lay, scale):
    st = random.choice(STAMPS)
    f = scale * random.uniform(0.7, 1.25)
    st = cv2.resize(st, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    if random.random() < 0.5:
        M = cv2.getRotationMatrix2D((st.shape[1] / 2, st.shape[0] / 2), random.uniform(-8, 8), 1)
        st = cv2.warpAffine(st, M, (st.shape[1], st.shape[0]), borderValue=255)
    if random.random() < 0.5:  # 펜 진하기 변화
        k = random.uniform(0.6, 1.3)
        st = np.where(st < 250, np.clip(255 - (255 - st.astype(np.float32)) * k, 0, 255), 255).astype(np.uint8)
    h, w = lay.shape
    sh, sw = st.shape
    # 패치보다 크면 일부만, 작으면 아무 위치에
    oy, ox = random.randint(-sh // 2, max(-sh // 2, h - sh // 2)), random.randint(-sw // 2, max(-sw // 2, w - sw // 2))
    ys, xs = max(0, oy), max(0, ox)
    ye, xe = min(h, oy + sh), min(w, ox + sw)
    if ye > ys and xe > xs:
        lay[ys:ye, xs:xe] = np.minimum(lay[ys:ye, xs:xe], st[ys - oy:ye - oy, xs - ox:xe - ox])


def _pen_layer(h, w, scale=1.0):
    """흰 바탕(255)에 필기 획(어두운 값)을 그린 레이어."""
    lay = np.full((h, w), 255, np.uint8)
    if STAMPS and random.random() < 0.6:  # 진짜 손글씨 조각
        for _ in range(random.choice([1, 1, 2, 3])):
            _paste_stamp(lay, scale)
        if random.random() < 0.5:
            return lay
    n = random.choice([0, 1, 2, 3, 3, 4, 5, 6, 8])
    for _ in range(n):
        val = random.randint(0, 110)
        th = max(1, int(random.choice([2, 2, 3, 3, 4, 5, 6]) * scale))
        x, y = random.randint(-20, w), random.randint(-20, h)
        r = random.random()
        if r < 0.40:  # 손글씨 글자
            lay = np.minimum(lay, _hand_text(h, w, x, y, val, scale))
        elif r < 0.55:  # 체크/엑스
            s = int(random.randint(20, 80) * scale)
            if random.random() < 0.5:
                pts = [(x, y), (x + s // 3, y + s // 2), (x + s, y - s // 2)]
                cv2.polylines(lay, [_jitter(pts)], False, val, th, cv2.LINE_AA)
            else:
                cv2.line(lay, (x, y), (x + s, y + s), val, th, cv2.LINE_AA)
                cv2.line(lay, (x + s, y), (x, y + s), val, th, cv2.LINE_AA)
        elif r < 0.70:  # 동그라미
            ax = (int(random.randint(15, 120) * scale), int(random.randint(12, 60) * scale))
            a0 = random.randint(0, 90)
            cv2.ellipse(lay, (x, y), ax, random.randint(-20, 20), a0, a0 + random.randint(250, 380), val, th, cv2.LINE_AA)
        elif r < 0.85:  # 직선 (밑줄, 사선, 세로줄)
            L = int(random.randint(30, 300) * scale)
            ang = random.choice([0, 0, 90, random.uniform(0, 180)])
            x2, y2 = int(x + L * np.cos(np.radians(ang))), int(y + L * np.sin(np.radians(ang)))
            cv2.polylines(lay, [_jitter([(x, y), ((x + x2) // 2, (y + y2) // 2), (x2, y2)], 3)], False, val, th, cv2.LINE_AA)
        else:  # 낙서 곡선
            pts = [(x, y)]
            for _ in range(random.randint(3, 12)):
                pts.append((pts[-1][0] + random.randint(-30, 30), pts[-1][1] + random.randint(-30, 30)))
            cv2.polylines(lay, [_smooth(pts)], False, val, th, cv2.LINE_AA)
    if random.random() < 0.5:
        lay = cv2.GaussianBlur(lay, (3, 3), 0)
    return lay


def _jitter(pts, a=4):
    return np.array([(p[0] + random.randint(-a, a), p[1] + random.randint(-a, a)) for p in pts], np.int32)


def _smooth(pts):
    p = np.array(pts, np.float32)
    t = np.linspace(0, len(p) - 1, len(p) * 8)
    return np.stack([np.interp(t, np.arange(len(p)), p[:, 0]), np.interp(t, np.arange(len(p)), p[:, 1])], 1).astype(np.int32)


def _hand_text(h, w, x, y, val, scale):
    size = int(random.randint(28, 90) * scale)
    t = random.choice(HAND_TXT)
    if random.random() < 0.3:
        t = t + random.choice(HAND_TXT)
    im = Image.new("L", (size * (len(t) + 1), int(size * 1.6)), 255)
    ImageDraw.Draw(im).text((size // 4, size // 8), t, font=_font(HAND_FONTS, size), fill=val)
    if random.random() < 0.6:  # 펜 굵기 변화
        k = random.choice([2, 3])
        im = Image.fromarray(cv2.erode(np.array(im), np.ones((k, k), np.uint8)))
    im = im.rotate(random.uniform(-20, 20), expand=True, fillcolor=255, resample=Image.BILINEAR)
    a = np.array(im)
    sh = random.uniform(-0.3, 0.3)  # 기울임
    M = np.float32([[1, sh, 0], [0, 1, 0]])
    a = cv2.warpAffine(a, M, (a.shape[1] + int(abs(sh) * a.shape[0]), a.shape[0]), borderValue=255)
    out = np.full((h, w), 255, np.uint8)
    ys, xs = max(0, y), max(0, x)
    ye, xe = min(h, y + a.shape[0]), min(w, x + a.shape[1])
    if ye > ys and xe > xs:
        out[ys:ye, xs:xe] = a[ys - y:ye - y, xs - x:xe - x]
    return out


def degrade(clean, hw):
    """스캔 느낌: 배경 톤, 대비, 블러, 노이즈, JPEG. (입력, 정답) 반환."""
    clean = clean.astype(np.float32)
    lo, hi = random.uniform(0, 40), random.uniform(200, 255)
    tgt = lo + clean / 255 * (hi - lo)
    if random.random() < 0.3:  # 그림자/조명 불균일
        g = cv2.resize(np.random.uniform(0.85, 1.0, (3, 3)).astype(np.float32), clean.shape[::-1])
        tgt = tgt * g
    hwf = hw.astype(np.float32) / 255
    inp = tgt * hwf + (1 - hwf) * np.minimum(tgt, lo + hw.astype(np.float32) / 255 * (hi - lo))
    if random.random() < 0.5:
        k = random.choice([3, 5])
        s = random.uniform(0.3, 1.0)
        inp = cv2.GaussianBlur(inp, (k, k), s)
        tgt = cv2.GaussianBlur(tgt, (k, k), s)
    inp = inp + np.random.normal(0, random.uniform(0, 6), inp.shape)
    inp = np.clip(inp, 0, 255).astype(np.uint8)
    if random.random() < 0.7:
        _, b = cv2.imencode(".jpg", inp, [cv2.IMWRITE_JPEG_QUALITY, random.randint(50, 95)])
        inp = cv2.imdecode(b, 0)
    return inp, np.clip(tgt, 0, 255).astype(np.uint8)


def scanify(clean):
    """인쇄 글씨를 실제 스캔처럼: 굵게, 거친 가장자리."""
    if random.random() < 0.6:
        k = random.choice([2, 2, 3])
        clean = cv2.erode(clean, np.ones((k, k), np.uint8))
    if random.random() < 0.4:  # 스캐너 이진화 느낌
        b = cv2.GaussianBlur(clean, (3, 3), 0)
        clean = np.where(b < random.randint(120, 200), random.randint(0, 40), 255).astype(np.uint8)
        clean = cv2.GaussianBlur(clean, (3, 3), 0.6)
    return clean


def make_sample(page, size=256, ignore=None, real=False):
    """페이지에서 무작위 패치를 잘라 필기를 합성. ignore: 학습에서 제외할 영역(실제 필기)."""
    H, W = page.shape
    s = random.uniform(0.8, 1.25) if real else random.uniform(0.7, 1.4)  # 해상도 다양화
    ps = int(size / s)
    for _ in range(6):  # 내용이 있는 곳을 우선 (빈 여백만 학습하지 않도록)
        y, x = random.randint(0, H - ps), random.randint(0, W - ps)
        if (page[y:y + ps:4, x:x + ps:4] < 160).mean() > 0.01:
            break
    clean = cv2.resize(page[y:y + ps, x:x + ps], (size, size), interpolation=cv2.INTER_AREA)
    valid = np.ones((size, size), np.float32)
    if ignore is not None:
        valid = 1 - cv2.resize(ignore[y:y + ps, x:x + ps], (size, size), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    if not real:
        clean = scanify(clean)
    hw = _pen_layer(size, size, scale=s)
    if real:  # 실제 스캔은 이미 스캔 노이즈가 있으므로 가볍게만
        inp = np.minimum(clean, hw).astype(np.uint8)
        tgt = clean
        if random.random() < 0.5:
            _, b = cv2.imencode(".jpg", inp, [cv2.IMWRITE_JPEG_QUALITY, random.randint(60, 95)])
            inp = cv2.imdecode(b, 0)
    else:
        inp, tgt = degrade(clean, hw)
    mask = (hw < 200).astype(np.uint8)
    return inp, tgt, mask, valid
