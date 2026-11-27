import { labelComponents, dilate } from './pipeline.js';
import { CONTRIBUTE_URL } from './config.js';
const APP_VERSION = '2026.11';

const PDFJS = 'https://cdn.jsdelivr.net/npm/pdfjs-dist@4.4.168/build/';
const pdfjsLib = await import(PDFJS + 'pdf.min.mjs');
pdfjsLib.GlobalWorkerOptions.workerSrc = PDFJS + 'pdf.worker.min.mjs';
const PDFLib = await import('https://cdn.jsdelivr.net/npm/pdf-lib@1.17.1/dist/pdf-lib.esm.min.js');

const $ = (id) => document.getElementById(id);
const state = { bytes: null, name: '', pdf: null, results: [], dpi: 200 };

// ------------------------------------------------------------ 워커
const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
const pending = new Map();
let reqId = 0, backend = '';
worker.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === 'ready') { backend = m.backend; showBackend(); return; }
  const p = pending.get(m.id);
  if (!p) { if (m.type === 'error') showError(m.message); return; }
  if (m.type === 'progress') p.onProgress && p.onProgress(m.p);
  else if (m.type === 'done') { pending.delete(m.id); p.resolve(m); }
  else if (m.type === 'error') { pending.delete(m.id); p.reject(new Error(m.message)); }
};
function processInWorker(gray, W, H, opts, onProgress) {
  const id = ++reqId;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject, onProgress });
    worker.postMessage({ type: 'page', id, gray: gray.buffer, W, H, opts }, [gray.buffer]);
  });
}
function showBackend() {
  $('backend').textContent = backend === 'webgpu' ? '⚡ GPU 가속 사용' : backend ? 'CPU로 처리 (조금 느려요)' : '';
}

// ------------------------------------------------------------ 파일 선택
const drop = $('drop'), fileInput = $('file');
drop.addEventListener('click', () => fileInput.click());
drop.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') fileInput.click(); });
drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', (e) => { e.preventDefault(); drop.classList.remove('over'); if (e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]); });
fileInput.addEventListener('change', () => { if (fileInput.files[0]) loadFile(fileInput.files[0]); });

async function loadFile(f) {
  try {
    state.bytes = new Uint8Array(await f.arrayBuffer());
    state.name = f.name.replace(/\.pdf$/i, '');
    state.pdf = await pdfjsLib.getDocument({ data: state.bytes.slice() }).promise;
    $('fileinfo').textContent = `📄 ${f.name} · ${state.pdf.numPages}쪽`;
    $('start').disabled = false;
    $('results').style.display = 'none';
    worker.postMessage({ type: 'init' }); // 모델 미리 불러오기
  } catch (e) {
    $('fileinfo').textContent = 'PDF를 열 수 없어요: ' + e.message;
  }
}

// ------------------------------------------------------------ 설정
const strength = $('strength');
const strengthLabel = () => {
  const v = +strength.value;
  $('strengthVal').textContent = v < 0.4 ? '약하게' : v > 0.6 ? '강하게' : '보통';
};
strength.addEventListener('input', strengthLabel); strengthLabel();
document.querySelectorAll('input[name=mode]').forEach((r) => r.addEventListener('change', () => {
  $('aiopts').style.display = mode() === 'ai' ? '' : 'none';
}));
const mode = () => document.querySelector('input[name=mode]:checked').value;

function parsePages(txt, n) {
  if (!txt.trim()) return [...Array(n).keys()];
  const out = new Set();
  for (const part of txt.split(',')) {
    const m = part.trim().match(/^(\d+)(?:\s*-\s*(\d+))?$/);
    if (!m) continue;
    const a = +m[1], b = m[2] ? +m[2] : a;
    for (let i = Math.max(1, a); i <= Math.min(n, b); i++) out.add(i - 1);
  }
  return [...out].sort((x, y) => x - y);
}

// ------------------------------------------------------------ 렌더링 도구
async function renderGray(pageNo, dpi) {
  const page = await state.pdf.getPage(pageNo + 1);
  const vp = page.getViewport({ scale: dpi / 72 });
  const c = document.createElement('canvas');
  c.width = Math.ceil(vp.width); c.height = Math.ceil(vp.height);
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, c.width, c.height);
  await page.render({ canvasContext: ctx, viewport: vp }).promise;
  const d = ctx.getImageData(0, 0, c.width, c.height).data;
  const gray = new Uint8Array(c.width * c.height);
  for (let i = 0, j = 0; i < gray.length; i++, j += 4) gray[i] = (d[j] * 299 + d[j + 1] * 587 + d[j + 2] * 114 + 500) / 1000;
  const v1 = page.getViewport({ scale: 1 });
  return { gray, W: c.width, H: c.height, wPt: v1.width, hPt: v1.height };
}
function grayCanvas(gray, W, H) {
  const c = document.createElement('canvas'); c.width = W; c.height = H;
  const ctx = c.getContext('2d');
  const img = ctx.createImageData(W, H);
  for (let i = 0, j = 0; i < gray.length; i++, j += 4) { img.data[j] = img.data[j + 1] = img.data[j + 2] = gray[i]; img.data[j + 3] = 255; }
  ctx.putImageData(img, 0, 0);
  return c;
}
function maskOverlayCanvas(gray, mask, W, H) {
  const c = grayCanvas(gray, W, H), ctx = c.getContext('2d');
  const img = ctx.getImageData(0, 0, W, H);
  for (let i = 0, j = 0; i < mask.length; i++, j += 4) if (mask[i]) { img.data[j] = 235; img.data[j + 1] = Math.round(img.data[j + 1] * 0.45); img.data[j + 2] = Math.round(img.data[j + 2] * 0.45); }
  ctx.putImageData(img, 0, 0);
  return c;
}
const toBlob = (c, type = 'image/jpeg', q = 0.9) => new Promise((r) => c.toBlob(r, type, q));
async function thumbUrl(c, w = 640) {
  const t = document.createElement('canvas');
  t.width = w; t.height = Math.round(c.height * w / c.width);
  const ctx = t.getContext('2d'); ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(c, 0, 0, t.width, t.height);
  return URL.createObjectURL(await toBlob(t, 'image/jpeg', 0.85));
}
async function blobToGray(blob, W, H) {
  const bmp = await createImageBitmap(blob);
  const c = document.createElement('canvas'); c.width = W; c.height = H;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0, W, H);
  const d = ctx.getImageData(0, 0, W, H).data;
  const g = new Uint8Array(W * H);
  for (let i = 0, j = 0; i < g.length; i++, j += 4) g[i] = d[j];
  return g;
}

// ------------------------------------------------------------ 실행
$('start').addEventListener('click', run);
async function run() {
  if (!state.pdf) return;
  hideError();
  $('start').disabled = true;
  $('progress').style.display = 'block';
  $('results').style.display = 'none';
  try {
    if (mode() === 'annot') await runAnnot();
    else await runAI();
  } catch (e) {
    showError('처리 중 오류가 났어요: ' + e.message);
  } finally {
    $('start').disabled = false;
  }
}

const HANDWRITING_ANNOTS = new Set(['/Ink', '/Highlight', '/Underline', '/StrikeOut', '/Squiggly', '/FreeText',
  '/Line', '/Square', '/Circle', '/Polygon', '/PolyLine', '/Caret']);
async function runAnnot() {
  setProgress(0.3, '태블릿 주석 찾는 중…');
  const { PDFDocument, PDFName } = PDFLib;
  const doc = await PDFDocument.load(state.bytes, { ignoreEncryption: true });
  let removed = 0;
  for (const page of doc.getPages()) {
    const annots = page.node.lookup(PDFName.of('Annots'));
    if (!annots || !annots.size) continue;
    const keep = [];
    for (let i = 0; i < annots.size(); i++) {
      const ref = annots.get(i);
      const a = doc.context.lookup(ref);
      const st = a && a.get && a.get(PDFName.of('Subtype'));
      const s = st ? st.toString() : '';
      if (HANDWRITING_ANNOTS.has(s)) removed++; else keep.push(ref); // 붙여 넣은 이미지(Stamp)·링크 등은 남김
    }
    if (keep.length) page.node.set(PDFName.of('Annots'), doc.context.obj(keep));
    else page.node.delete(PDFName.of('Annots'));
  }
  setProgress(1, removed ? `주석 필기 ${removed}개를 지웠어요.` :
    '지울 주석이 없어요. 필기가 페이지에 합쳐진 PDF라면 "손필기 지우기 (AI)"를 사용하세요.');
  if (removed) {
    const out = await doc.save();
    download(new Blob([out], { type: 'application/pdf' }), `${state.name}_clean.pdf`);
  }
}

async function runAI() {
  const dpi = +$('dpi').value;
  state.dpi = dpi;
  const sel = parsePages($('pages').value, state.pdf.numPages);
  if (!sel.length) throw new Error('쪽 범위를 확인해 주세요.');
  const opts = { thr: 1 - +strength.value, grow: 2, protect: $('protect').checked, dpi };
  state.opts = { strength: +strength.value, protect: opts.protect };
  state.results = [];
  $('grid').innerHTML = '';
  setProgress(0, 'AI 모델 불러오는 중… (처음 한 번만, 약 5MB)');
  const t0 = performance.now();
  for (let k = 0; k < sel.length; k++) {
    const pageNo = sel[k];
    const label = `${k + 1}/${sel.length}쪽 처리 중 (${pageNo + 1}쪽)`;
    const r = await renderGray(pageNo, dpi);
    const before = grayCanvas(r.gray, r.W, r.H);
    const res = await processInWorker(r.gray, r.W, r.H, opts, (p) => {
      const el = (performance.now() - t0) / 1000;
      const frac = (k + p) / sel.length;
      const eta = frac > 0.02 ? Math.round(el / frac - el) : null;
      setProgress(frac, `${label}${eta !== null ? ` · 약 ${eta}초 남음` : ''}`);
    });
    const out = new Uint8Array(res.out), mask = new Uint8Array(res.mask);
    const afterC = grayCanvas(out, r.W, r.H);
    const item = {
      pageNo, W: r.W, H: r.H, wPt: r.wPt, hPt: r.hPt,
      jpeg: await toBlob(afterC),
      mask: await toBlob(grayCanvas(mask.map((v) => v * 255), r.W, r.H), 'image/png'),
      beforeUrl: await thumbUrl(before),
      afterUrl: await thumbUrl(afterC),
      maskUrl: await thumbUrl(maskOverlayCanvas(r.gray, mask, r.W, r.H)),
      ratio: res.ratio,
    };
    state.results.push(item);
    addCard(item);
    $('results').style.display = 'block';
  }
  setProgress(1, `완료! ${sel.length}쪽 · ${Math.round((performance.now() - t0) / 1000)}초. 결과를 확인하고, 잘못 지워진 곳은 ✏️ 수정으로 되살릴 수 있어요.`);
  $('restitle').textContent = `결과 (${sel.length}쪽)`;
  if (CONTRIBUTE_URL) { $('contrib').style.display = 'block'; $('cstatus').textContent = ''; $('csend').disabled = !$('cconsent').checked; }
}

// ------------------------------------------------------------ 결과 카드
function addCard(item) {
  const card = document.createElement('div');
  card.className = 'page';
  card.innerHTML = `
    <div class="img"><img alt="${item.pageNo + 1}쪽 결과"></div>
    <div class="bar2">
      <b>${item.pageNo + 1}쪽</b>
      <div class="tog">
        <button data-v="after" class="on">결과</button><button data-v="before">원본</button><button data-v="mask">지운 곳</button>
      </div>
      <button class="edit">✏️ 수정</button>
    </div>`;
  const img = card.querySelector('img');
  let view = 'after';
  const show = () => { img.src = item[view + 'Url']; };
  show();
  card.querySelectorAll('.tog button').forEach((b) => b.addEventListener('click', () => {
    view = b.dataset.v;
    card.querySelectorAll('.tog button').forEach((x) => x.classList.toggle('on', x === b));
    show();
  }));
  card.querySelector('.img').addEventListener('click', () => openEditor(item, () => show()));
  card.querySelector('.edit').addEventListener('click', () => openEditor(item, () => show()));
  item.refresh = show;
  $('grid').appendChild(card);
}

// ------------------------------------------------------------ 다운로드
$('download').addEventListener('click', async () => {
  const { PDFDocument } = PDFLib;
  // 원본 PDF 구조는 그대로 두고, 처리한 쪽만 결과 이미지로 교체
  const doc = await PDFDocument.load(state.bytes, { ignoreEncryption: true });
  for (const r of state.results) {
    const jpg = await doc.embedJpg(await r.jpeg.arrayBuffer());
    doc.removePage(r.pageNo);
    const p = doc.insertPage(r.pageNo, [r.wPt, r.hPt]);
    p.drawImage(jpg, { x: 0, y: 0, width: r.wPt, height: r.hPt });
  }
  download(new Blob([await doc.save()], { type: 'application/pdf' }), `${state.name}_clean.pdf`);
});
function download(blob, name) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}

// ------------------------------------------------------------ 수정 편집기
const ed = { item: null, orig: null, cur: null, out: null, lab: null, erased: null, W: 0, H: 0, undo: [], tool: 'click', size: 24, showMask: true, paper: 255, onDone: null };
const edCanvas = $('edcanvas'), edCtx = edCanvas.getContext('2d');

async function openEditor(item, onDone) {
  $('editor').style.display = 'flex';
  $('edtitle').textContent = `${item.pageNo + 1}쪽 수정 · 불러오는 중…`;
  ed.item = item; ed.onDone = onDone; ed.W = item.W; ed.H = item.H; ed.undo = [];
  const r = await renderGray(item.pageNo, state.dpi);
  ed.orig = r.gray;
  ed.out = await blobToGray(item.jpeg, item.W, item.H);
  ed.cur = ed.out.slice();
  const mask = await blobToGray(item.mask, item.W, item.H);
  const bin = new Uint8Array(mask.length);
  for (let i = 0; i < bin.length; i++) bin[i] = mask[i] > 127 ? 1 : 0;
  const { lab, n } = labelComponents(dilate(bin, item.W, item.H, 5), item.W, item.H);
  ed.lab = lab; ed.erased = new Uint8Array(n).fill(1); ed.erased[0] = 0;
  // 종이 색 = 원본 밝은 쪽 98% 값
  const hist = new Uint32Array(256); for (const v of ed.orig) hist[v]++;
  let acc = 0; for (let v = 255; v >= 0; v--) { acc += hist[v]; if (acc > ed.orig.length * 0.3) { ed.paper = v; break; } }
  edCanvas.width = item.W; edCanvas.height = item.H;
  $('edtitle').textContent = `${item.pageNo + 1}쪽 수정`;
  draw();
}
function draw() {
  const { W, H, cur, lab, erased } = ed;
  const img = edCtx.createImageData(W, H), d = img.data;
  for (let i = 0, j = 0; i < cur.length; i++, j += 4) {
    const v = cur[i];
    if (ed.showMask && lab[i] && erased[lab[i]]) { d[j] = 255; d[j + 1] = Math.round(v * 0.55 + 60); d[j + 2] = Math.round(v * 0.55 + 60); }
    else if (ed.showMask && lab[i]) { d[j] = Math.round(v * 0.6 + 40); d[j + 1] = Math.round(v * 0.7 + 70); d[j + 2] = 255; }
    else { d[j] = d[j + 1] = d[j + 2] = v; }
    d[j + 3] = 255;
  }
  edCtx.putImageData(img, 0, 0);
}
function pushUndo() { ed.undo.push({ cur: ed.cur.slice(), erased: ed.erased.slice() }); if (ed.undo.length > 30) ed.undo.shift(); }
function toggleComponent(x, y) {
  const { W, H, lab } = ed;
  let k = 0;
  for (let r = 0; r <= 10 && !k; r += 2) {
    for (let dy = -r; dy <= r && !k; dy++) for (let dx = -r; dx <= r && !k; dx++) {
      const xx = x + dx, yy = y + dy;
      if (xx >= 0 && yy >= 0 && xx < W && yy < H && lab[yy * W + xx]) k = lab[yy * W + xx];
    }
  }
  if (!k) return;
  pushUndo();
  ed.erased[k] ^= 1;
  const src = ed.erased[k] ? ed.out : ed.orig;
  for (let i = 0; i < lab.length; i++) if (lab[i] === k) ed.cur[i] = src[i];
  draw();
}
function brush(x, y) {
  const { W, H } = ed, r = ed.size;
  for (let yy = Math.max(0, y - r); yy < Math.min(H, y + r); yy++) for (let xx = Math.max(0, x - r); xx < Math.min(W, x + r); xx++) {
    if ((xx - x) ** 2 + (yy - y) ** 2 > r * r) continue;
    const i = yy * W + xx;
    ed.cur[i] = ed.tool === 'erase' ? Math.max(ed.cur[i], ed.paper) : ed.orig[i];
  }
}
function evPos(e) {
  const rc = edCanvas.getBoundingClientRect();
  return [Math.round((e.clientX - rc.left) * ed.W / rc.width), Math.round((e.clientY - rc.top) * ed.H / rc.height)];
}
let painting = false, lastP = null;
edCanvas.addEventListener('pointerdown', (e) => {
  const [x, y] = evPos(e);
  if (ed.tool === 'click') { toggleComponent(x, y); return; }
  pushUndo(); painting = true; lastP = [x, y]; brush(x, y); draw();
  edCanvas.setPointerCapture(e.pointerId);
});
edCanvas.addEventListener('pointermove', (e) => {
  if (!painting) return;
  const [x, y] = evPos(e);
  const steps = Math.max(1, Math.ceil(Math.hypot(x - lastP[0], y - lastP[1]) / (ed.size / 2)));
  for (let s = 1; s <= steps; s++) brush(Math.round(lastP[0] + (x - lastP[0]) * s / steps), Math.round(lastP[1] + (y - lastP[1]) * s / steps));
  lastP = [x, y]; draw();
});
edCanvas.addEventListener('pointerup', () => { painting = false; });
document.querySelectorAll('#edtools [data-tool]').forEach((b) => b.addEventListener('click', () => {
  ed.tool = b.dataset.tool;
  document.querySelectorAll('#edtools [data-tool]').forEach((x) => x.classList.toggle('on', x === b));
  edCanvas.style.cursor = ed.tool === 'click' ? 'pointer' : 'crosshair';
}));
$('edsize').addEventListener('input', (e) => { ed.size = +e.target.value; });
$('edmask').addEventListener('change', (e) => { ed.showMask = e.target.checked; draw(); });
$('edundo').addEventListener('click', () => { const u = ed.undo.pop(); if (u) { ed.cur = u.cur; ed.erased = u.erased; draw(); } });
$('edcancel').addEventListener('click', closeEditor);
$('edsave').addEventListener('click', async () => {
  const it = ed.item;
  const c = grayCanvas(ed.cur, ed.W, ed.H);
  it.jpeg = await toBlob(c);
  URL.revokeObjectURL(it.afterUrl); it.afterUrl = await thumbUrl(c);
  const m = new Uint8Array(ed.W * ed.H);
  for (let i = 0; i < m.length; i++) if (ed.lab[i] && ed.erased[ed.lab[i]]) m[i] = 255;
  it.mask = await toBlob(grayCanvas(m, ed.W, ed.H), 'image/png');
  it.edited = true;
  URL.revokeObjectURL(it.maskUrl); it.maskUrl = await thumbUrl(maskOverlayCanvas(ed.orig, m, ed.W, ed.H));
  closeEditor();
  ed.onDone && ed.onDone();
});
function closeEditor() { $('editor').style.display = 'none'; ed.orig = ed.cur = ed.out = ed.lab = null; }
document.addEventListener('keydown', (e) => {
  if ($('editor').style.display !== 'flex') return;
  if (e.key === 'Escape') closeEditor();
  if ((e.ctrlKey || e.metaKey) && e.key === 'z') { e.preventDefault(); $('edundo').click(); }
});

// ------------------------------------------------------------ 학습에 기여하기 (사용자가 동의한 경우에만)
const blobB64 = (b) => new Promise((r) => { const fr = new FileReader(); fr.onload = () => r(fr.result); fr.readAsDataURL(b); });
$('cconsent').addEventListener('change', (e) => { $('csend').disabled = !e.target.checked; });
$('csend').addEventListener('click', async () => {
  if (!CONTRIBUTE_URL || !$('cconsent').checked) return;
  $('csend').disabled = true;
  let n = 0;
  try {
    for (const it of state.results) {
      $('cstatus').textContent = `보내는 중… ${n + 1}/${state.results.length}쪽`;
      const r = await renderGray(it.pageNo, state.dpi);
      const orig = await toBlob(grayCanvas(r.gray, r.W, r.H), 'image/jpeg', 0.92);
      const body = JSON.stringify({
        consent: true, page: it.pageNo + 1, dpi: state.dpi, edited: !!it.edited, version: APP_VERSION,
        strength: state.opts && state.opts.strength, protect: state.opts && state.opts.protect,
        orig: await blobB64(orig), result: await blobB64(it.jpeg), mask: await blobB64(it.mask),
      });
      await fetch(CONTRIBUTE_URL, { method: 'POST', mode: 'no-cors', headers: { 'Content-Type': 'text/plain' }, body });
      n++;
    }
    $('cstatus').textContent = `고마워요! ${n}쪽을 보냈어요. 모델 개선에 사용할게요.`;
  } catch (e) {
    $('cstatus').textContent = '보내지 못했어요: ' + e.message;
    $('csend').disabled = false;
  }
});

// ------------------------------------------------------------ 진행/오류 표시
function setProgress(p, text) { $('barfill').style.width = `${Math.round(p * 100)}%`; $('status').textContent = text; }
function showError(t) { $('error').style.display = 'block'; $('error').textContent = t; $('progress').style.display = 'block'; }
function hideError() { $('error').style.display = 'none'; }
window.__cleanpdf = state; // 디버그용
