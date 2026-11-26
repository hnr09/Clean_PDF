// 필기 지우개 처리 로직 (브라우저/Node 공용). erase_ai.py + protect.py 를 그대로 옮긴 것.
// gray: Uint8Array (W*H), 0=검정 255=흰색

// ---------------------------------------------------------------- 기본 도구
export function labelComponents(bin, W, H) {
  // 8-연결 라벨링 (union-find). 반환: {lab:Int32Array, n}
  const lab = new Int32Array(W * H);
  const parent = [0];
  const find = (a) => { while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
  const union = (a, b) => { a = find(a); b = find(b); if (a !== b) { if (a < b) parent[b] = a; else parent[a] = b; } };
  let next = 1;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (!bin[i]) continue;
      let l = 0;
      const nb = [];
      if (x > 0 && bin[i - 1]) nb.push(lab[i - 1]);
      if (y > 0) {
        const r = i - W;
        if (x > 0 && bin[r - 1]) nb.push(lab[r - 1]);
        if (bin[r]) nb.push(lab[r]);
        if (x < W - 1 && bin[r + 1]) nb.push(lab[r + 1]);
      }
      if (nb.length === 0) { l = next++; parent.push(l); }
      else { l = nb[0]; for (let k = 1; k < nb.length; k++) { if (nb[k] < l) l = nb[k]; } for (const v of nb) union(l, v); }
      lab[i] = l;
    }
  }
  // 압축된 번호로 재매핑
  const map = new Int32Array(next);
  let n = 1;
  for (let l = 1; l < next; l++) { const r = find(l); if (map[r] === 0) map[r] = n++; map[l] = map[r]; }
  for (let i = 0; i < lab.length; i++) if (lab[i]) lab[i] = map[lab[i]];
  return { lab, n };
}

function componentStats(lab, n, W, H) {
  const x0 = new Int32Array(n).fill(W), y0 = new Int32Array(n).fill(H), x1 = new Int32Array(n).fill(-1), y1 = new Int32Array(n).fill(-1), area = new Int32Array(n);
  for (let y = 0, i = 0; y < H; y++) for (let x = 0; x < W; x++, i++) {
    const l = lab[i]; if (!l) continue;
    area[l]++;
    if (x < x0[l]) x0[l] = x; if (x > x1[l]) x1[l] = x;
    if (y < y0[l]) y0[l] = y; if (y > y1[l]) y1[l] = y;
  }
  return { x0, y0, x1, y1, area };
}

// 정사각형 k×k 팽창 (분리 가능한 max 필터)
export function dilate(bin, W, H, k) {
  const r = k >> 1;
  const tmp = new Uint8Array(W * H), out = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) {
    const row = y * W;
    let last = -1e9;
    for (let x = 0; x < W; x++) { if (bin[row + x]) last = x; }
    // 앞/뒤 두 번 훑기
    let prev = -1e9;
    for (let x = 0; x < W; x++) { if (bin[row + x]) prev = x; if (x - prev <= r) tmp[row + x] = 1; }
    prev = 1e9;
    for (let x = W - 1; x >= 0; x--) { if (bin[row + x]) prev = x; if (prev - x <= r) tmp[row + x] = 1; }
  }
  for (let x = 0; x < W; x++) {
    let prev = -1e9;
    for (let y = 0; y < H; y++) { if (tmp[y * W + x]) prev = y; if (y - prev <= r) out[y * W + x] = 1; }
    prev = 1e9;
    for (let y = H - 1; y >= 0; y--) { if (tmp[y * W + x]) prev = y; if (prev - y <= r) out[y * W + x] = 1; }
  }
  return out;
}

function erode(bin, W, H, k) {
  const inv = new Uint8Array(W * H);
  for (let i = 0; i < inv.length; i++) inv[i] = bin[i] ? 0 : 1;
  const d = dilate(inv, W, H, k);
  for (let i = 0; i < d.length; i++) d[i] = d[i] ? 0 : 1;
  return d;
}

// 길이 L 이상인 가로(또는 세로) 연속 구간만 남김 (= 1×L 열림 연산)
function openLine(bin, W, H, L, horizontal) {
  const out = new Uint8Array(W * H);
  const A = horizontal ? H : W, B = horizontal ? W : H;
  for (let a = 0; a < A; a++) {
    let start = -1;
    for (let b = 0; b <= B; b++) {
      const i = horizontal ? a * W + b : b * W + a;
      const on = b < B && bin[i];
      if (on && start < 0) start = b;
      if (!on && start >= 0) {
        if (b - start >= L) for (let c = start; c < b; c++) out[horizontal ? a * W + c : c * W + a] = 1;
        start = -1;
      }
    }
  }
  return out;
}

// OpenCV DIST_L2, mask 3 과 같은 챔퍼 거리 변환
function distanceTransform(bin, W, H) {
  const a = 0.955, b = 1.3693, INF = 1e9;
  const d = new Float32Array(W * H);
  for (let i = 0; i < d.length; i++) d[i] = bin[i] ? INF : 0;
  const g = (x, y) => (x < 0 || y < 0 || x >= W || y >= H) ? 0 : d[y * W + x];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const i = y * W + x; if (!d[i]) continue;
    d[i] = Math.min(d[i], g(x - 1, y) + a, g(x, y - 1) + a, g(x - 1, y - 1) + b, g(x + 1, y - 1) + b);
  }
  for (let y = H - 1; y >= 0; y--) for (let x = W - 1; x >= 0; x--) {
    const i = y * W + x; if (!d[i]) continue;
    d[i] = Math.min(d[i], g(x + 1, y) + a, g(x, y + 1) + a, g(x + 1, y + 1) + b, g(x - 1, y + 1) + b);
  }
  return d;
}

function gaussianBlur5(f, W, H) {
  // OpenCV GaussianBlur(5x5, sigma=0) → sigma 1.1, 경계 reflect101
  const s = 1.1, k = [];
  let sum = 0;
  for (let i = -2; i <= 2; i++) { const v = Math.exp(-(i * i) / (2 * s * s)); k.push(v); sum += v; }
  for (let i = 0; i < 5; i++) k[i] /= sum;
  const refl = (p, n) => (p < 0 ? -p : p >= n ? 2 * n - 2 - p : p);
  const tmp = new Float32Array(W * H), out = new Float32Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let v = 0; for (let j = -2; j <= 2; j++) v += k[j + 2] * f[y * W + refl(x + j, W)]; tmp[y * W + x] = v;
  }
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let v = 0; for (let j = -2; j <= 2; j++) v += k[j + 2] * tmp[refl(y + j, H) * W + x]; out[y * W + x] = v;
  }
  return out;
}

// ---------------------------------------------------------------- 모델 실행 (타일)
// runModel(Float32Array data, h, w) → {clean: Float32Array, logit: Float32Array}
export async function runTiled(gray, W, H, runModel, onProgress, tile = 512, pad = 48) {
  const clean = new Float32Array(W * H), prob = new Float32Array(W * H);
  const step = tile - 2 * pad;
  const tiles = [];
  for (let y0 = 0; y0 < H; y0 += step) for (let x0 = 0; x0 < W; x0 += step) tiles.push([y0, x0]);
  let done = 0;
  for (const [y0, x0] of tiles) {
    const ya = Math.max(0, y0 - pad), xa = Math.max(0, x0 - pad);
    const yb = Math.min(H, y0 + step + pad), xb = Math.min(W, x0 + step + pad);
    const th = yb - ya, tw = xb - xa;
    const ph = th + ((16 - (th % 16)) % 16), pw = tw + ((16 - (tw % 16)) % 16);
    const t = new Float32Array(ph * pw);
    for (let y = 0; y < ph; y++) {
      const sy = ya + Math.min(y, th - 1);
      for (let x = 0; x < pw; x++) t[y * pw + x] = gray[sy * W + xa + Math.min(x, tw - 1)] / 255; // replicate pad
    }
    const r = await runModel(t, ph, pw);
    const sy = y0 - ya, sx = x0 - xa;
    const ey = Math.min(step, H - y0), ex = Math.min(step, W - x0);
    for (let y = 0; y < ey; y++) for (let x = 0; x < ex; x++) {
      const si = (sy + y) * pw + sx + x, di = (y0 + y) * W + x0 + x;
      clean[di] = r.clean[si];
      prob[di] = 1 / (1 + Math.exp(-r.logit[si]));
    }
    done++;
    if (onProgress) onProgress(done / tiles.length);
  }
  return { clean, prob };
}

// ---------------------------------------------------------------- 후처리
export function hysteresis(prob, W, H, thr, minArea = 30) {
  const weak = new Uint8Array(W * H);
  for (let i = 0; i < weak.length; i++) weak[i] = prob[i] > thr * 0.7 ? 1 : 0;
  const { lab, n } = labelComponents(weak, W, H);
  const cnt = new Int32Array(n);
  for (let i = 0; i < lab.length; i++) if (lab[i] && prob[i] > thr) cnt[lab[i]]++;
  const out = new Uint8Array(W * H);
  for (let i = 0; i < lab.length; i++) if (lab[i] && cnt[lab[i]] >= minArea) out[i] = 1;
  return out;
}

function lineMasks(gray, W, H, dpi, length) {
  let ink = new Uint8Array(W * H);
  for (let i = 0; i < ink.length; i++) ink[i] = gray[i] < 170 ? 1 : 0;
  ink = erode(dilate(ink, W, H, 3), W, H, 3); // 끊긴 스캔 선 잇기
  const L = Math.floor(length * dpi);
  return { hor: openLine(ink, W, H, L, true), ver: openLine(ink, W, H, L, false) };
}

function segments(lines, W, H, horizontal) {
  const { lab, n } = labelComponents(lines, W, H);
  const st = componentStats(lab, n, W, H);
  const segs = [];
  for (let k = 1; k < n; k++) {
    const x = st.x0[k], y = st.y0[k], w = st.x1[k] - x + 1, h = st.y1[k] - y + 1;
    segs.push(horizontal ? [x, x + w, y + (h >> 1)] : [x + (w >> 1), y, y + h]);
  }
  return segs;
}

export function figureBoxes(gray, W, H, dpi = 200) {
  const { hor, ver } = lineMasks(gray, W, H, dpi, 0.22);
  const Hs = segments(hor, W, H, true), Vs = segments(ver, W, H, false);
  const tol = Math.floor(0.06 * dpi), boxes = [];
  for (const [vx, vy0, vy1] of Vs) {
    for (const [hx0, hx1, hy] of Hs) {
      if (Math.abs(hy - vy1) > tol || Math.abs(hx0 - vx) > tol) continue;
      // 표 제외: 이 세로선에서 시작하는 내부 가로선(행 구분선)이 있으면 표
      const others = Hs.some(([a0, a1, yy]) => Math.abs(a0 - vx) <= tol && vy0 + tol < yy && yy < hy - tol && (a1 - a0) >= 0.8 * (hx1 - hx0));
      // 닫힌 넓은 상자(<보기> 등) 제외. 위 테두리가 없으면 양쪽 y축 그래프, 좁은 틀은 테두리 있는 그래프
      const right = Vs.some(([x, y0, y1]) => x !== vx && Math.abs(x - hx1) <= tol && y0 < hy - tol && Math.abs(y1 - hy) <= tol);
      const top = Hs.some(([a0, a1, yy]) => Math.abs(a0 - vx) <= tol && Math.abs(yy - vy0) <= tol);
      const closed = right && top && (hx1 - hx0) > 0.3 * W;
      if (others || closed) continue;
      const m = Math.floor(0.25 * dpi);
      boxes.push([Math.max(0, vx - m), Math.max(0, vy0 - (m >> 1)), hx1 + (m >> 1), hy + m]);
    }
  }
  return boxes;
}

export function protectMask(gray, W, H, dpi = 200, prob = null, lineMax = 0.6, figMax = 0.5) {
  const { hor, ver } = lineMasks(gray, W, H, dpi, 0.35);
  let lines = new Uint8Array(W * H);
  for (let i = 0; i < lines.length; i++) lines[i] = hor[i] | ver[i];
  if (prob) { // AI가 확신하는 선(손으로 그린 선)은 보호하지 않음
    const { lab, n } = labelComponents(lines, W, H);
    const sum = new Float64Array(n), cnt = new Int32Array(n);
    for (let i = 0; i < lab.length; i++) if (lab[i]) { sum[lab[i]] += prob[i]; cnt[lab[i]]++; }
    for (let i = 0; i < lab.length; i++) lines[i] = lab[i] && sum[lab[i]] / cnt[lab[i]] < lineMax ? 1 : 0;
  }
  const keep = dilate(lines, W, H, 3);
  for (const [x0, y0, x1, y1] of figureBoxes(gray, W, H, dpi)) {
    if (prob) {
      let s = 0, c = 0;
      for (let y = y0; y < Math.min(H, y1); y++) for (let x = x0; x < Math.min(W, x1); x++) { const i = y * W + x; if (gray[i] < 170) { s += prob[i]; c++; } }
      if (c && s / c >= figMax) continue;
    }
    for (let y = y0; y < Math.min(H, y1); y++) for (let x = x0; x < Math.min(W, x1); x++) keep[y * W + x] = 1;
  }
  return keep;
}

export function printedCurves(gray, cand, W, H, dpi = 200) {
  const keep = new Uint8Array(W * H);
  const { lab, n } = labelComponents(cand, W, H);
  const st = componentStats(lab, n, W, H);
  for (let k = 1; k < n; k++) {
    const x = st.x0[k], y = st.y0[k], w = st.x1[k] - x + 1, h = st.y1[k] - y + 1;
    if (w < 0.5 * dpi || w < 1.2 * h || h < 0.08 * dpi) continue;
    const xs = [], ys = [], runs = [];
    let colsOn = 0;
    for (let c = 0; c < w; c++) {
      let r = 0, prev = 0, sy = 0, cnt = 0;
      for (let rr = 0; rr < h; rr++) {
        const i = (y + rr) * W + x + c;
        const v = lab[i] === k && gray[i] < 170 ? 1 : 0;
        if (v && !prev) r++;
        if (v) { sy += rr; cnt++; }
        prev = v;
      }
      if (cnt) { colsOn++; runs.push(r); xs.push(c); ys.push(sy / cnt); }
    }
    if (colsOn / w < 0.7) continue;
    const one = runs.filter((r) => r === 1).length / runs.length;
    const many = runs.filter((r) => r >= 3).length / runs.length;
    if (one < 0.55 || many > 0.15) continue;
    // 직선(밑줄) 제외
    const N = xs.length; let mx = 0, my = 0;
    for (let i = 0; i < N; i++) { mx += xs[i]; my += ys[i]; }
    mx /= N; my /= N;
    let sxy = 0, sxx = 0;
    for (let i = 0; i < N; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) ** 2; }
    const sl = sxx ? sxy / sxx : 0;
    let res = 0;
    for (let i = 0; i < N; i++) res += Math.abs(ys[i] - (my + sl * (xs[i] - mx)));
    if (res / N < 0.02 * dpi) continue;
    for (let i = 0; i < lab.length; i++) if (lab[i] === k) keep[i] = 1;
  }
  return keep;
}

export function printedBlobs(gray, cand, W, H, dpi = 200, maxPen = 0.04, maxFill = 0.52, prob = null, fillMaxProb = 0.65) {
  const keep = new Uint8Array(W * H);
  const ink = new Uint8Array(W * H);
  for (let i = 0; i < ink.length; i++) ink[i] = gray[i] < 140 ? 1 : 0;
  const dt = distanceTransform(ink, W, H);
  const { lab, n } = labelComponents(cand, W, H);
  const vals = Array.from({ length: n }, () => []);
  const area = new Int32Array(n), psum = new Float64Array(n);
  for (let i = 0; i < lab.length; i++) { const l = lab[i]; if (!l) continue; area[l]++; if (ink[i]) { vals[l].push(dt[i]); if (prob) psum[l] += prob[i]; } }
  const keepL = new Uint8Array(n);
  for (let k = 1; k < n; k++) {
    const v = vals[k]; if (v.length < 40) continue;
    const pm = prob ? psum[k] / v.length : 0;
    v.sort((a, b) => a - b);
    const pos = 0.95 * (v.length - 1), lo = Math.floor(pos), f = pos - lo;
    const p95 = v[lo] + (v[Math.min(lo + 1, v.length - 1)] - v[lo]) * f;
    let dense = v.length / area[k] > maxFill;
    if (dense && prob && pm >= fillMaxProb) dense = false;
    if (2 * p95 > maxPen * dpi || dense) keepL[k] = 1;
  }
  for (let i = 0; i < lab.length; i++) if (keepL[lab[i]]) keep[i] = 1;
  return keep;
}

// 작고 속이 꽉 찬 인쇄 기호(●, ■ 등)
export function printedSymbols(gray, W, H, dpi = 200) {
  const ink = new Uint8Array(W * H);
  for (let i = 0; i < ink.length; i++) ink[i] = gray[i] < 140 ? 1 : 0;
  const { lab, n } = labelComponents(ink, W, H);
  const st = componentStats(lab, n, W, H);
  const lo = 0.025 * dpi, hi = 0.15 * dpi;
  const keepL = new Uint8Array(n);
  for (let k = 1; k < n; k++) {
    const x = st.x0[k], y = st.y0[k], w = st.x1[k] - x + 1, h = st.y1[k] - y + 1;
    if (!(lo <= w && w <= hi && lo <= h && h <= hi && w / h >= 0.6 && w / h <= 1.6)) continue;
    if (st.area[k] / (w * h) < 0.6) continue;
    const cy0 = h >> 2, cy1 = Math.max(h - (h >> 2), cy0 + 1), cx0 = w >> 2, cx1 = Math.max(w - (w >> 2), cx0 + 1);
    let on = 0, tot = 0;
    for (let yy = cy0; yy < cy1; yy++) for (let xx = cx0; xx < cx1; xx++) { tot++; if (lab[(y + yy) * W + x + xx] === k) on++; }
    if (on / tot >= 0.9) keepL[k] = 1;
  }
  const keep = new Uint8Array(W * H);
  for (let i = 0; i < lab.length; i++) if (keepL[lab[i]]) keep[i] = 1;
  return dilate(keep, W, H, 5);
}

// ---------------------------------------------------------------- 한 페이지 처리
// 반환: {out: Uint8Array(gray 결과), mask: Uint8Array(지운 영역), ratio}
export async function erasePage(gray, W, H, runModel, opts = {}, onProgress) {
  const { thr = 0.5, grow = 2, protect = true, dpi = 200 } = opts;
  const { clean, prob } = await runTiled(gray, W, H, runModel, onProgress);
  let m = hysteresis(prob, W, H, thr, 30);
  m = dilate(m, W, H, 1 + 2 * grow);
  if (protect) {
    const c = printedCurves(gray, m, W, H, dpi);
    for (let i = 0; i < m.length; i++) if (c[i]) m[i] = 0;
    const b = printedBlobs(gray, m, W, H, dpi, 0.04, 0.52, prob);
    for (let i = 0; i < m.length; i++) if (b[i]) m[i] = 0;
    const p = protectMask(gray, W, H, dpi, prob);
    for (let i = 0; i < m.length; i++) if (p[i]) m[i] = 0;
    const sy = printedSymbols(gray, W, H, dpi);
    for (let i = 0; i < m.length; i++) if (sy[i]) m[i] = 0;
  }
  const mf = new Float32Array(W * H);
  let cnt = 0;
  for (let i = 0; i < m.length; i++) { mf[i] = m[i]; cnt += m[i]; }
  const mb = gaussianBlur5(mf, W, H);
  // 종이색 = 밝기 90% 지점 (지운 자리의 옅은 흔적 정리용)
  const hist = new Uint32Array(256); let hn = 0;
  for (let y = 0; y < H; y += 4) for (let x = 0; x < W; x += 4) { hist[gray[y * W + x]]++; hn++; }
  let acc = 0, paper = 255;
  for (let v = 0; v < 256; v++) { acc += hist[v]; if (acc >= 0.9 * hn) { paper = v / 255; break; } }
  const out = new Uint8Array(W * H);
  for (let i = 0; i < out.length; i++) {
    let v = (gray[i] / 255) * (1 - mb[i]) + clean[i] * mb[i];
    if (mb[i] > 0.05 && v > 0.7) v = Math.max(v, paper);
    out[i] = Math.max(0, Math.min(255, Math.round(v * 255)));
  }
  return { out, mask: m, ratio: cnt / m.length };
}
