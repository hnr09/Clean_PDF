#!/usr/bin/env python3
"""
erase_handwriting.py — PDF에서 필기(손글씨)를 지워주는 도구

지원하는 경우
  1) 태블릿/PDF 앱 필기 (GoodNotes, Notability, 갤럭시 노트, Acrobat 펜 등)
     - 필기가 '주석(annotation)'으로 저장된 경우 → 주석만 삭제 (원본 화질·텍스트 그대로, 무손실)
     - 필기가 페이지에 '평탄화(flatten)'된 경우 → 이미지로 렌더링 후 색상/연필 필터로 제거
  2) 종이에 손필기 후 스캔/촬영한 PDF
     - 색 펜(파랑·빨강·초록·형광펜 등) → 색상 기반 제거
     - 연필(회색, 인쇄 글씨보다 연한 선) → 밝기 기반 제거

한계: 인쇄 글씨와 같은 '검정 펜' 필기는 색/밝기로 구분이 안 되어 완벽히 지우기 어렵습니다.

사용 예
  python erase_handwriting.py input.pdf -o output.pdf                 # 자동
  python erase_handwriting.py input.pdf --mode color                  # 색 펜만
  python erase_handwriting.py input.pdf --mode color,pencil --dpi 250 # 색 펜 + 연필
  python erase_handwriting.py input.pdf --mode annot                  # 태블릿 주석만 (무손실)
"""
import argparse
import sys

import cv2
import numpy as np
import pymupdf

# 지우지 않을 주석 종류 (링크, 양식 필드 등은 필기가 아님)
KEEP_ANNOTS = {pymupdf.PDF_ANNOT_LINK, pymupdf.PDF_ANNOT_WIDGET, pymupdf.PDF_ANNOT_POPUP}


# ----------------------------------------------------------------------------
# 1) 디지털 필기: 주석 삭제 (무손실)
# ----------------------------------------------------------------------------
def remove_annotations(page) -> int:
    n = 0
    annot = page.first_annot
    while annot:
        nxt = annot.next
        if annot.type[0] not in KEEP_ANNOTS:
            page.delete_annot(annot)
            n += 1
        annot = nxt
    return n


# ----------------------------------------------------------------------------
# 2) 이미지 기반 필기 제거
# ----------------------------------------------------------------------------
def estimate_background(img_bgr: np.ndarray) -> np.ndarray:
    """글씨를 없앤 '종이 배경'을 추정 (그림자·누런 종이도 자연스럽게 채우기 위함)."""
    h, w = img_bgr.shape[:2]
    k = max(15, (min(h, w) // 60) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    bg = cv2.morphologyEx(img_bgr, cv2.MORPH_CLOSE, kernel)  # 어두운 선 제거
    return cv2.medianBlur(bg, 21)


def color_ink_mask(img_bgr, sat_thr=60, val_min=50, keep_hues=None):
    """채도가 높은(=색이 있는) 픽셀 = 색 펜/형광펜."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (s > sat_thr) & (v > val_min)
    if keep_hues:  # 남기고 싶은 색 범위 (OpenCV hue 0~179)
        for lo, hi in keep_hues:
            mask &= ~((h >= lo) & (h <= hi))
    return mask.astype(np.uint8) * 255


def pencil_mask(img_bgr, ink_thr=200, dark_thr=110, sat_max=60, min_area=8):
    """연필 선: 회색이고, 선 전체에서 가장 진한 부분도 인쇄 글씨만큼 진하지 않은 덩어리."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    sat = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)[..., 1]
    ink = ((gray < ink_thr) & (sat < sat_max)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        return np.zeros_like(gray)
    darkest = np.full(n, 255, dtype=np.int32)
    np.minimum.at(darkest, labels.ravel(), gray.ravel().astype(np.int32))
    is_pencil = (darkest > dark_thr) & (stats[:, cv2.CC_STAT_AREA] >= min_area)
    is_pencil[0] = False
    return (is_pencil[labels]).astype(np.uint8) * 255


def clean_image(img_bgr, modes, opts):
    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    if "color" in modes:
        mask |= color_ink_mask(img_bgr, opts.sat_thr, keep_hues=opts.keep_hues)
    if "pencil" in modes:
        mask |= pencil_mask(img_bgr, dark_thr=opts.pencil_dark)
    if not mask.any():
        return img_bgr, 0.0
    # 선 가장자리(안티앨리어싱 번짐)까지 지우도록 살짝 확장
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=opts.grow)
    bg = estimate_background(img_bgr)
    out = img_bgr.copy()
    out[mask > 0] = bg[mask > 0]
    if opts.whiten:  # 배경을 깨끗한 흰색으로
        out = whiten_background(out)
    return out, float((mask > 0).mean())


def whiten_background(img_bgr, thr=225):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    out = img_bgr.copy()
    out[gray > thr] = 255
    return out


# ----------------------------------------------------------------------------
# 페이지 판단 도우미
# ----------------------------------------------------------------------------
def is_scanned(page) -> bool:
    """페이지 대부분이 이미지이고 텍스트가 거의 없으면 스캔본으로 판단."""
    area = abs(page.rect)
    img_area = 0.0
    for info in page.get_image_info():
        img_area += abs(pymupdf.Rect(info["bbox"]) & page.rect)
    return img_area > 0.5 * area and len(page.get_text().strip()) < 50


def has_colored_vector_ink(page, min_paths=5) -> bool:
    """평탄화된 태블릿 필기: 검정이 아닌 색의 곡선 경로가 여러 개 있는지."""
    cnt = 0
    for d in page.get_drawings():
        col = d.get("color") or d.get("fill")
        if not col:
            continue
        r, g, b = col[:3]
        if max(r, g, b) - min(r, g, b) > 0.25 and any(it[0] == "c" for it in d["items"]):
            cnt += 1
            if cnt >= min_paths:
                return True
    return False


def page_to_bgr(page, dpi):
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


# ----------------------------------------------------------------------------
# 메인 처리
# ----------------------------------------------------------------------------
def process(in_path, out_path, opts, log=print):
    modes = set(opts.mode.split(","))
    auto = "auto" in modes
    src = pymupdf.open(in_path)
    out = pymupdf.open()

    for i, page in enumerate(src):
        tag = f"[{i + 1}/{len(src)}]"
        n_annot = remove_annotations(page) if (auto or "annot" in modes) else 0

        raster_modes = modes & {"color", "pencil"}
        if auto:
            if is_scanned(page):
                raster_modes = {"color", "pencil"} if opts.auto_pencil else {"color"}
            elif has_colored_vector_ink(page):
                raster_modes = {"color"}

        if not raster_modes:
            out.insert_pdf(src, from_page=i, to_page=i)  # 무손실 복사 (텍스트 유지)
            log(f"{tag} 주석 {n_annot}개 삭제 (원본 유지)")
            continue

        img = page_to_bgr(page, opts.dpi)
        cleaned, ratio = clean_image(img, raster_modes, opts)
        ok, buf = cv2.imencode(".jpg", cleaned, [cv2.IMWRITE_JPEG_QUALITY, opts.quality])
        new = out.new_page(width=page.rect.width, height=page.rect.height)
        new.insert_image(new.rect, stream=buf.tobytes())
        log(f"{tag} 주석 {n_annot}개 삭제, {'+'.join(sorted(raster_modes))} 필터로 "
            f"{ratio * 100:.1f}% 영역 제거")

    out.save(out_path, garbage=3, deflate=True)
    log(f"완료 → {out_path}")
    return out_path


def build_parser():
    p = argparse.ArgumentParser(description="PDF 필기 지우개")
    p.add_argument("input")
    p.add_argument("-o", "--output", help="출력 PDF (기본: 입력이름_clean.pdf)")
    p.add_argument("--mode", default="auto",
                   help="auto | annot | color | pencil (쉼표로 조합, 예: annot,color,pencil)")
    p.add_argument("--dpi", type=int, default=200, help="이미지 처리 해상도 (기본 200)")
    p.add_argument("--quality", type=int, default=90, help="JPEG 품질 (기본 90)")
    p.add_argument("--sat-thr", type=int, default=60,
                   help="색 펜 판정 채도 기준 (낮출수록 연한 색도 지움, 기본 60)")
    p.add_argument("--pencil-dark", type=int, default=110,
                   help="연필 판정: 이 값보다 진한 부분이 없으면 연필로 봄 (기본 110)")
    p.add_argument("--grow", type=int, default=1, help="지우는 영역 확장 정도 (기본 1)")
    p.add_argument("--keep-hue", action="append", default=[],
                   help="남길 색 범위 lo-hi (OpenCV hue 0~179), 예: --keep-hue 0-10 (빨강 유지)")
    p.add_argument("--auto-pencil", action="store_true", help="auto 모드에서 연필도 지우기")
    p.add_argument("--whiten", action="store_true", help="배경을 흰색으로 정리")
    return p


def main(argv=None):
    opts = build_parser().parse_args(argv)
    opts.keep_hues = [tuple(map(int, s.split("-"))) for s in opts.keep_hue]
    out = opts.output or opts.input.rsplit(".", 1)[0] + "_clean.pdf"
    process(opts.input, out, opts)


if __name__ == "__main__":
    sys.exit(main())
