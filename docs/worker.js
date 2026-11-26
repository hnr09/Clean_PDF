// 모델 실행 + 후처리를 UI와 분리된 스레드에서 수행
import { erasePage } from './pipeline.js';

const ORT_VER = '1.20.1';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
let ort = null;
let session = null;
let backend = '';

async function init() {
  if (session) return;
  ort = await import(ORT_BASE + 'ort.webgpu.min.mjs');
  ort.env.wasm.wasmPaths = ORT_BASE;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;
  const model = await (await fetch('eraser.onnx')).arrayBuffer();
  const tries = [];
  if (self.navigator && navigator.gpu) tries.push('webgpu');
  tries.push('wasm');
  let lastErr = null;
  for (const ep of tries) {
    try {
      session = await ort.InferenceSession.create(model, { executionProviders: [ep] });
      // WebGPU가 실제로 동작하는지 작은 입력으로 확인
      await session.run({ image: new ort.Tensor('float32', new Float32Array(64 * 64).fill(1), [1, 1, 64, 64]) });
      backend = ep;
      return;
    } catch (e) {
      lastErr = e;
      session = null;
    }
  }
  throw lastErr;
}

async function runModel(data, h, w) {
  const r = await session.run({ image: new ort.Tensor('float32', data, [1, 1, h, w]) });
  return { clean: r.clean.data, logit: r.logit.data };
}

self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') {
      await init();
      self.postMessage({ type: 'ready', backend });
    } else if (msg.type === 'page') {
      await init();
      const { gray, W, H, opts, id } = msg;
      const res = await erasePage(new Uint8Array(gray), W, H, runModel, opts, (p) =>
        self.postMessage({ type: 'progress', id, p }));
      self.postMessage({ type: 'done', id, out: res.out.buffer, mask: res.mask.buffer, ratio: res.ratio },
        [res.out.buffer, res.mask.buffer]);
    }
  } catch (e) {
    self.postMessage({ type: 'error', id: msg.id, message: String(e && e.message || e) });
  }
};
