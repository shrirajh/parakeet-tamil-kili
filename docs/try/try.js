/* parakeet-tamil-kili in the browser: CPU only (ONNX Runtime Web, WebAssembly).
   The same pipeline as kili.infer.onnx_run: NeMo's log-mel front end (numpy port), the int8 ONNX encoder, the fp32
   prediction network + joint, greedy TDT decoding. Audio never leaves the device. */
"use strict";

const REPO = "shrirajh/parakeet-tamil-kili";
const ORT_VER = "1.31.1";
const ORT_CDN = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VER}/dist/`;
const qs = new URLSearchParams(location.search);
const BASE = (qs.get("base") || `https://huggingface.co/${REPO}/resolve/main/`).replace(/\/?$/, "/");
const FILES = [
  { path: "onnx/meta.json", size: 117776 },
  { path: "onnx/frontend.npz", size: 133686 },
  { path: "onnx/encoder_int8.onnx", size: 3495327 },
  { path: "onnx/encoder_int8.onnx.data", size: 879004160 },
  { path: "onnx/pred.onnx", size: 54107242 },
  { path: "onnx/joint.onnx", size: 26261256 },
];
const TOTAL = FILES.reduce((a, f) => a + f.size, 0);
const SR = 16000, HOP = 1280, MAX_ONE_PASS_S = 120;
const CACHE = "kili-model-v1";

const $ = (s) => document.querySelector(s);
const L = (o) => (document.documentElement.lang === "ta" ? o.ta : o.en);
const fmtMB = (b) => (b / 1e6).toFixed(b < 1e7 ? 1 : 0) + " MB";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* ---------------------------------------------------------------------------------------------------- errors */
class StageError extends Error {
  constructor(stage, message, hint = "") { super(message); this.stage = stage; this.hint = hint; }
}
function showError(e, stage) {
  const box = $("#errors");
  const st = e.stage || stage || "unknown";
  let hint = e.hint || "";
  const msg = String(e && e.message ? e.message : e);
  if (!hint && /memory|allocation|RangeError|out of bounds/i.test(msg))
    hint = L({ en: "The browser ran out of memory. The model needs about 2 GB: try a desktop browser and close other tabs.",
               ta: "உலாவியில் நினைவகம் போதவில்லை. மாதிரிக்குச் சுமார் 2 GB தேவை: கணினியில் உள்ள உலாவியில், மற்ற தாவல்களை மூடிவிட்டு முயலுங்கள்." });
  const div = document.createElement("div");
  div.className = "err";
  div.innerHTML = `<b>${st}</b> <span></span>${hint ? `<p class="hint"></p>` : ""}<details><summary>${L({ en: "details", ta: "விவரம்" })}</summary><pre></pre></details>`;
  div.querySelector("span").textContent = msg;
  if (hint) div.querySelector(".hint").textContent = hint;
  div.querySelector("pre").textContent = `${new Date().toISOString()}\nstage: ${st}\n${msg}\n\n${(e && e.stack) || ""}\n\nbrowser: ${navigator.userAgent}\nisolated: ${self.crossOriginIsolated}\nbase: ${BASE}`;
  box.insertBefore(div, box.querySelector("h2").nextSibling);
  box.hidden = false;
  console.error(`[${st}]`, e);
}
window.addEventListener("error", (e) => showError(e.error || e.message, "page"));
window.addEventListener("unhandledrejection", (e) => showError(e.reason, "page"));

/* ------------------------------------------------------------------------------------------ cross-origin isolation */
// WebAssembly threads need cross-origin isolation; sw.js provides it on hosts that cannot set headers.
async function ensureIsolation() {
  if (self.crossOriginIsolated || !("serviceWorker" in navigator) || qs.has("nosw")) return;
  try {
    const reg = await navigator.serviceWorker.register("sw.js");
    await navigator.serviceWorker.ready;
    if (!sessionStorage.getItem("kili-coi-reload")) {
      sessionStorage.setItem("kili-coi-reload", "1");
      if (reg.active && !navigator.serviceWorker.controller) location.reload();
      else if (!self.crossOriginIsolated) location.reload();
    }
  } catch (e) {
    console.warn("service worker unavailable, running single-threaded", e);
  }
}

/* --------------------------------------------------------------------------------------------------- downloads */
async function openCache() {
  try { return "caches" in self ? await caches.open(CACHE) : null; } catch { return null; }
}
async function sha256hex(buf) {
  const d = await crypto.subtle.digest("SHA-256", buf);
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, "0")).join("");
}
async function fetchOk(url, stage) {
  let res;
  try { res = await fetch(url, { mode: "cors" }); }
  catch (e) {
    throw new StageError(stage, `network error fetching ${url}: ${e.message}`,
      L({ en: "Check your connection. If you are offline, a model you downloaded before is still cached.",
          ta: "இணைய இணைப்பைச் சரிபாருங்கள். முன்பு பதிவிறக்கிய மாதிரி சேமிப்பில் இருந்தால் அது இயங்கும்." }));
  }
  if (!res.ok) {
    const hint = [401, 403, 404].includes(res.status)
      ? L({ en: "The model files are not reachable at this address. The Hugging Face repository may not be public yet.",
            ta: "இந்த முகவரியில் மாதிரிக் கோப்புகள் கிடைக்கவில்லை. Hugging Face களஞ்சியம் இன்னும் பொதுவில் வராமல் இருக்கலாம்." })
      : "";
    throw new StageError(stage, `HTTP ${res.status} ${res.statusText} for ${url}`, hint);
  }
  return res;
}
async function cachedKeys() {
  const c = await openCache();
  if (!c) return new Set();
  return new Set((await c.keys()).map((r) => r.url));
}
// Download (or read from the browser cache) every file, verify SHA-256, report progress.
async function loadFiles(onProgress) {
  const sums = {};
  const cache = await openCache();
  let sumsText = null;
  const sumsKey = BASE + "SHA256SUMS";
  try { sumsText = await (await fetchOk(sumsKey, "download")).text(); if (cache) cache.put(sumsKey, new Response(sumsText)).catch(() => {}); }
  catch (e) { const hit = cache && (await cache.match(sumsKey)); if (!hit) throw e; sumsText = await hit.text(); }
  for (const line of sumsText.split("\n")) { const [h, p] = line.trim().split(/\s+/); if (p) sums[p] = h; }
  const out = {};
  const done = new Array(FILES.length).fill(0);
  const t0 = performance.now();
  let lastT = t0, lastB = 0, speed = 0;
  const report = (i, state) => {
    const got = done.reduce((a, b) => a + b, 0), now = performance.now();
    if (now - lastT > 400) { const s = (got - lastB) / ((now - lastT) / 1000); speed = speed ? 0.7 * speed + 0.3 * s : s; lastT = now; lastB = got; }
    onProgress({ i, state, got, total: TOTAL, speed, file: FILES[i], fileGot: done[i] });
  };
  for (let i = 0; i < FILES.length; i++) {
    const f = FILES[i], url = BASE + f.path, want = sums[f.path];
    if (!want) throw new StageError("download", `${f.path} is missing from SHA256SUMS`);
    const key = `${url}#${want}`;
    const hit = cache && (await cache.match(key));
    if (hit) {
      out[f.path] = new Uint8Array(await hit.arrayBuffer());
      done[i] = f.size; report(i, "cached");
      continue;
    }
    const res = await fetchOk(url, "download");
    const total = +res.headers.get("content-length") || f.size;
    let buf = new Uint8Array(total), got = 0;
    const reader = res.body.getReader();
    for (;;) {
      const { done: end, value } = await reader.read();
      if (end) break;
      if (got + value.length > buf.length) { const b2 = new Uint8Array(Math.max(buf.length * 1.5, got + value.length) | 0); b2.set(buf); buf = b2; }
      buf.set(value, got); got += value.length;
      done[i] = got; report(i, "downloading");
    }
    buf = buf.subarray(0, got);
    report(i, "verifying");
    const h = await sha256hex(buf);
    if (h !== want) throw new StageError("verify", `${f.path}: SHA-256 ${h.slice(0, 12)}… does not match ${want.slice(0, 12)}…`,
      L({ en: "The download was corrupted or the file changed. Reload and try again.", ta: "பதிவிறக்கம் பழுதாகிவிட்டது அல்லது கோப்பு மாறியிருக்கிறது. மீண்டும் ஏற்றி முயலுங்கள்." }));
    if (cache) {
      try { await cache.put(key, new Response(buf)); }
      catch (e) { showError(new StageError("cache", `could not cache ${f.path} (${e.message}); it will download again next time`), "cache"); }
    }
    out[f.path] = buf;
    report(i, "ok");
  }
  return { files: out, seconds: (performance.now() - t0) / 1000 };
}

/* --------------------------------------------------------------------------------------------------- npz / npy */
function readNpz(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  let e = u8.length - 22;
  while (e >= 0 && dv.getUint32(e, true) !== 0x06054b50) e--;
  if (e < 0) throw new StageError("model", "frontend.npz: no zip directory");
  let n = dv.getUint16(e + 10, true), cd = dv.getUint32(e + 16, true);
  const z64 = e - 20;
  if ((cd === 0xffffffff || n === 0xffff) && z64 >= 0 && dv.getUint32(z64, true) === 0x07064b50) {
    const rec = Number(dv.getBigUint64(z64 + 8, true));
    n = Number(dv.getBigUint64(rec + 32, true)); cd = Number(dv.getBigUint64(rec + 48, true));
  }
  const arrays = {};
  for (let k = 0, p = cd; k < n; k++) {
    const method = dv.getUint16(p + 10, true);
    let csize = dv.getUint32(p + 20, true), lho = dv.getUint32(p + 42, true);
    const nl = dv.getUint16(p + 28, true), xl = dv.getUint16(p + 30, true), cl = dv.getUint16(p + 32, true);
    const name = new TextDecoder().decode(u8.subarray(p + 46, p + 46 + nl));
    for (let x = p + 46 + nl; x < p + 46 + nl + xl;) {  // zip64 extra: sizes / offset
      const id = dv.getUint16(x, true), sz = dv.getUint16(x + 2, true);
      if (id === 1) { let q = x + 4; if (dv.getUint32(p + 24, true) === 0xffffffff) q += 8; if (csize === 0xffffffff) { csize = Number(dv.getBigUint64(q, true)); q += 8; } if (lho === 0xffffffff) lho = Number(dv.getBigUint64(q, true)); }
      x += 4 + sz;
    }
    if (method !== 0) throw new StageError("model", `frontend.npz: ${name} is compressed (method ${method})`);
    const start = lho + 30 + dv.getUint16(lho + 26, true) + dv.getUint16(lho + 28, true);
    arrays[name.replace(/\.npy$/, "")] = readNpy(u8.subarray(start, start + csize));
    p += 46 + nl + xl + cl;
  }
  return arrays;
}
function readNpy(u8) {
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  const major = u8[6], hl = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true), off = major === 1 ? 10 : 12;
  const head = new TextDecoder().decode(u8.subarray(off, off + hl));
  const descr = /'descr':\s*'([^']+)'/.exec(head)[1];
  const shape = /'shape':\s*\(([^)]*)\)/.exec(head)[1].split(",").map((s) => s.trim()).filter(Boolean).map(Number);
  if (/True/.test(/'fortran_order':\s*(\w+)/.exec(head)[1])) throw new StageError("model", "npy: fortran order not supported");
  const body = u8.slice(off + hl);  // copy: aligned
  if (descr === "<f4") return { data: new Float32Array(body.buffer), shape };
  if (descr === "<f8") return { data: Float32Array.from(new Float64Array(body.buffer)), shape };
  throw new StageError("model", `npy: dtype ${descr} not supported`);
}

/* -------------------------------------------------------------------------------------------------- front end */
// NeMo's FilterbankFeatures at inference, as kili.infer.onnx_run.Frontend: pre-emphasis, centred STFT (n_fft 512,
// hop 160, zero padding), power spectrum, mel filterbank, log(x + guard), per-feature normalisation over the valid
// frames (std with n - 1, + eps), frames past the valid length set to 0.
class Frontend {
  constructor(cfg, fb, win) {
    this.nfft = cfg.n_fft; this.hop = cfg.hop_length; this.pre = cfg.preemph; this.guard = cfg.log_guard; this.eps = cfg.std_eps;
    this.nm = fb.shape[0]; this.nb = fb.shape[1]; this.fb = fb.data;
    this.win = new Float32Array(this.nfft);
    const off = (this.nfft - win.data.length) >> 1;
    this.win.set(win.data, off);
    const N = this.nfft, lg = Math.log2(N);
    this.rev = new Uint32Array(N);
    for (let i = 0; i < N; i++) { let r = 0; for (let b = 0; b < lg; b++) r |= ((i >> b) & 1) << (lg - 1 - b); this.rev[i] = r; }
    this.cos = new Float64Array(N / 2); this.sin = new Float64Array(N / 2);
    for (let i = 0; i < N / 2; i++) { this.cos[i] = Math.cos((-2 * Math.PI * i) / N); this.sin[i] = Math.sin((-2 * Math.PI * i) / N); }
  }
  fft(re, im) {
    const N = this.nfft;
    for (let i = 0; i < N; i++) { const j = this.rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
    for (let size = 2; size <= N; size <<= 1) {
      const half = size >> 1, step = N / size;
      for (let s = 0; s < N; s += size) for (let k = 0; k < half; k++) {
        const c = this.cos[k * step], si = this.sin[k * step], a = s + k, b = a + half;
        const tr = re[b] * c - im[b] * si, ti = re[b] * si + im[b] * c;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
      }
    }
  }
  run(x) {
    const n = x.length, N = this.nfft, p = N >> 1, T = Math.floor(n / this.hop) + 1, nm = this.nm, nb = this.nb;
    const y = new Float32Array(n + 2 * p);
    if (n) y[p] = x[0];
    for (let i = 1; i < n; i++) y[p + i] = x[i] - this.pre * x[i - 1];
    const feats = new Float32Array(nm * T), re = new Float64Array(N), im = new Float64Array(N), pw = new Float64Array(nb);
    for (let t = 0; t < T; t++) {
      const o = t * this.hop;
      for (let i = 0; i < N; i++) { re[i] = y[o + i] * this.win[i]; im[i] = 0; }
      this.fft(re, im);
      for (let k = 0; k < nb; k++) pw[k] = re[k] * re[k] + im[k] * im[k];
      for (let m = 0; m < nm; m++) { let s = 0; const r = m * nb; for (let k = 0; k < nb; k++) s += this.fb[r + k] * pw[k]; feats[m * T + t] = Math.log(s + this.guard); }
    }
    const tv = Math.max(Math.floor(n / this.hop), 1);
    for (let m = 0; m < nm; m++) {
      const r = m * T; let mu = 0;
      for (let t = 0; t < tv; t++) mu += feats[r + t];
      mu /= tv;
      let v = 0;
      for (let t = 0; t < tv; t++) { const d = feats[r + t] - mu; v += d * d; }
      const sd = Math.sqrt(v / Math.max(tv - 1, 1)) + this.eps;
      for (let t = 0; t < T; t++) feats[r + t] = t < tv ? (feats[r + t] - mu) / sd : 0;
    }
    return { feats, T, tv };
  }
}

/* ------------------------------------------------------------------------------------------------------ model */
const M = { ready: false };
async function loadOrt(threads) {
  if (!self.ort) {
    await new Promise((ok, bad) => {
      const s = document.createElement("script");
      s.src = ORT_CDN + "ort.wasm.min.js"; s.crossOrigin = "anonymous";
      s.onload = ok;
      s.onerror = () => bad(new StageError("runtime", `could not load ONNX Runtime Web from ${s.src}`,
        L({ en: "A content blocker or firewall may be blocking cdn.jsdelivr.net.", ta: "cdn.jsdelivr.net-ஐ ஏதோ ஒன்று தடுக்கலாம்." })));
      document.head.appendChild(s);
    });
  }
  ort.env.wasm.wasmPaths = ORT_CDN;
  ort.env.wasm.numThreads = threads;
  ort.env.wasm.simd = true;
  ort.env.logLevel = "error";
}
async function createSessions(files, threads) {
  const opt = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
  const enc = await ort.InferenceSession.create(files["onnx/encoder_int8.onnx"], {
    ...opt, externalData: [{ path: "encoder_int8.onnx.data", data: files["onnx/encoder_int8.onnx.data"] }],
  });
  const pred = await ort.InferenceSession.create(files["onnx/pred.onnx"], opt);
  const joint = await ort.InferenceSession.create(files["onnx/joint.onnx"], opt);
  return { enc, pred, joint, threads };
}
async function loadModel(onProgress, onStage) {
  const { files, seconds } = await loadFiles(onProgress);
  onStage("runtime");
  const meta = JSON.parse(new TextDecoder().decode(files["onnx/meta.json"]));
  const fz = readNpz(files["onnx/frontend.npz"]);
  M.front = new Frontend(meta.features, fz.fb, fz.window);
  M.pieces = meta.pieces; M.blank = meta.blank; M.durations = meta.durations; M.maxSym = meta.max_symbols || 10;
  let threads = self.crossOriginIsolated ? Math.max(1, Math.min(navigator.hardwareConcurrency || 4, 8)) : 1;
  const t0 = performance.now();
  await loadOrt(threads);
  try { Object.assign(M, await createSessions(files, threads)); }
  catch (e) {
    if (threads > 1) {  // threaded runtime failed to start: report it and fall back to one thread
      showError(new StageError("runtime", `multi-threaded start failed (${e.message}); retrying with 1 thread`), "runtime");
      threads = 1; ort.env.wasm.numThreads = 1;
      Object.assign(M, await createSessions(files, 1));
    } else throw new StageError("runtime", `could not start the model: ${e.message}`);
  }
  M.loadS = (performance.now() - t0) / 1000;
  M.downloadS = seconds;
  for (const k in files) delete files[k];  // the sessions hold their own copies
  M.ready = true;
}

const i64 = (v) => new ort.Tensor("int64", BigInt64Array.of(BigInt(v)), [1]);
async function encode(audio) {
  const t0 = performance.now();
  const { feats, T, tv } = M.front.run(audio);
  const t1 = performance.now();
  const out = await M.enc.run({ feats: new ort.Tensor("float32", feats, [1, M.front.nm, T]), flen: i64(tv) });
  const t2 = performance.now();
  const f = out.f.data, n = Number(out.n.data[0]), D = out.f.dims[2];
  return { f, n, D, frontMs: t1 - t0, encMs: t2 - t1 };
}
async function predStep(label, h, c) {
  const o = await M.pred.run({ label: new ort.Tensor("int64", BigInt64Array.of(BigInt(label)), [1, 1]), h, c });
  return { g: o.g, h: o.h2, c: o.c2 };
}
async function newState() {
  const L_ = 2, H = 640;  // pred.onnx state h, c: [2, 1, 640]
  const z = () => new ort.Tensor("float32", new Float32Array(L_ * H), [L_, 1, H]);
  const p = await predStep(M.blank, z(), z());  // the blank id is the start symbol
  return { g: p.g, h: p.h, c: p.c, ids: [], ts: [], f0: 0, sym: 0, decMs: 0, steps: 0 };
}
// Greedy TDT over global frames [t, tEnd) of f (f row i = global frame st.f0 + i), as kili.infer.onnx_run.tdt_greedy.
async function tdtGreedy(enc, t, tEnd, st) {
  const V1 = M.blank + 1, D = enc.D, t0 = performance.now();
  while (t < tEnd) {
    const row = enc.f.subarray((t - st.f0) * D, (t - st.f0 + 1) * D);
    const lo = (await M.joint.run({ f: new ort.Tensor("float32", row, [1, D]), g: st.g })).logits.data;
    st.steps++;
    let k = 0, kv = -Infinity, d = 0, dv = -Infinity;
    for (let i = 0; i < V1; i++) if (lo[i] > kv) { kv = lo[i]; k = i; }
    for (let i = V1; i < lo.length; i++) if (lo[i] > dv) { dv = lo[i]; d = M.durations[i - V1]; }
    if (k === M.blank) { t += Math.max(d, 1); st.sym = 0; continue; }
    st.ids.push(k); st.ts.push([t, d]);  // global frame and duration: word timestamps
    const p = await predStep(k, st.h, st.c);
    st.g = p.g; st.h = p.h; st.c = p.c;
    if (d === 0) { if (++st.sym >= M.maxSym) { t++; st.sym = 0; } }
    else { t += d; st.sym = 0; }
  }
  st.decMs += performance.now() - t0;
  return t;
}
const text = (ids) => ids.map((i) => M.pieces[i]).join("").replace(/▁/g, " ").trim();
// tokens -> [word, start s, end s], as kili.infer.onnx_run.OnnxASR.words (NeMo's TDT word offsets): a "▁" piece starts a
// word; start = first token's frame, end = last token's frame + duration; punctuation joins without moving the end
function words(ids, ts) {
  const out = [];
  ids.forEach((i, n) => {
    const p = M.pieces[i], core = p.replace(/^▁/, ""), [t, d] = ts[n];
    if (out.length && core && !/[\p{L}\p{M}\p{N}]/u.test(core)) { out[out.length - 1][0] += core; return; }
    if (p.startsWith("▁") || !out.length) out.push([core, t, t + d]);
    else { const w = out[out.length - 1]; w[0] += core; w[2] = t + d; }
  });
  const fs = HOP / SR;
  return out.filter((w) => w[0]).map(([w, s, e]) => [w, +(s * fs).toFixed(2), +(e * fs).toFixed(2)]);
}

async function transcribeOffline(audio, onStats) {
  const t0 = performance.now(), st = await newState();
  let frontMs = 0, encMs = 0;
  if (audio.length <= MAX_ONE_PASS_S * SR) {
    const e = await encode(audio);
    frontMs += e.frontMs; encMs += e.encMs;
    await tdtGreedy(e, 0, e.n, st);
  } else {  // long audio: 60 s chunks with 20.48 s of context each side, decoder state carried across
    const chunk = Math.round((60 * SR) / HOP), ctx = Math.round((20.48 * SR) / HOP), nF = Math.ceil(audio.length / HOP);
    let t = 0;
    for (let c0 = 0; ; c0 += chunk) {
      const c1 = Math.min(c0 + chunk, nF), w0 = Math.max(0, c0 - ctx), w1 = Math.min(c1 + ctx, nF), last = c1 >= nF;
      const e = await encode(audio.subarray(w0 * HOP, w1 * HOP));
      frontMs += e.frontMs; encMs += e.encMs; st.f0 = w0;
      t = await tdtGreedy(e, Math.max(t, w0), last ? w0 + e.n : c1, st);
      onStats({ partial: text(st.ids) });
      if (last) break;
    }
  }
  const ms = performance.now() - t0, sec = audio.length / SR;
  return { text: text(st.ids), words: words(st.ids, st.ts), stats: { audio: sec, frontMs, encMs, decMs: st.decMs, totalMs: ms, rtfx: sec / (ms / 1000), tokens: st.ids.length, steps: st.steps } };
}

/* ------------------------------------------------------------------------------------------------------- audio */
async function decodeFile(blob) {
  const buf = await blob.arrayBuffer();
  let ab;
  try { ab = await new OfflineAudioContext(1, 1, SR).decodeAudioData(buf); }
  catch (e) { throw new StageError("audio", `could not decode this file (${e.message || e})`, L({ en: "Try WAV, MP3, OGG, M4A or FLAC.", ta: "WAV, MP3, OGG, M4A அல்லது FLAC கோப்பைப் பயன்படுத்துங்கள்." })); }
  return toMono16k(ab);
}
async function toMono16k(ab) {
  if (ab.sampleRate !== SR) {
    const oc = new OfflineAudioContext(1, Math.ceil((ab.duration * SR)), SR);
    const s = oc.createBufferSource(); s.buffer = ab; s.connect(oc.destination); s.start();
    ab = await oc.startRendering();
  }
  if (ab.numberOfChannels === 1) return ab.getChannelData(0).slice();
  const out = new Float32Array(ab.length);
  for (let c = 0; c < ab.numberOfChannels; c++) { const d = ab.getChannelData(c); for (let i = 0; i < d.length; i++) out[i] += d[i] / ab.numberOfChannels; }
  return out;
}
async function resampleTo16k(x, rate) {
  if (rate === SR) return x;
  const oc = new OfflineAudioContext(1, Math.max(1, Math.round((x.length * SR) / rate)), SR);
  const b = oc.createBuffer(1, x.length, rate); b.copyToChannel(x, 0);
  const s = oc.createBufferSource(); s.buffer = b; s.connect(oc.destination); s.start();
  return (await oc.startRendering()).getChannelData(0).slice();
}

// Microphone: raw PCM via an AudioWorklet (no compression), kept at the device rate.
const WORKLET = `class Tap extends AudioWorkletProcessor { process(i) { const c = i[0] && i[0][0]; if (c) this.port.postMessage(c.slice(0)); return true; } } registerProcessor("tap", Tap);`;
class Mic {
  async start() {
    try { this.stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: true } }); }
    catch (e) { throw new StageError("microphone", `${e.name}: ${e.message}`, L({ en: "Allow microphone access for this page (the lock icon in the address bar).", ta: "இந்தப் பக்கத்துக்கு மைக்ரோஃபோன் அனுமதி கொடுங்கள் (முகவரிப் பட்டியில் உள்ள பூட்டுக் குறி)." })); }
    try { this.ctx = new AudioContext({ sampleRate: SR }); } catch { this.ctx = new AudioContext(); }
    let src;
    try { src = this.ctx.createMediaStreamSource(this.stream); }
    catch { await this.ctx.close(); this.ctx = new AudioContext(); src = this.ctx.createMediaStreamSource(this.stream); }  // Firefox: no rate change
    this.rate = this.ctx.sampleRate;
    const url = URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" }));
    await this.ctx.audioWorklet.addModule(url);
    this.node = new AudioWorkletNode(this.ctx, "tap");
    this.chunks = []; this.n = 0;
    this.node.port.onmessage = (e) => { this.chunks.push(e.data); this.n += e.data.length; if (this.onLevel) this.onLevel(e.data); };
    const sink = this.ctx.createGain(); sink.gain.value = 0;
    src.connect(this.node).connect(sink).connect(this.ctx.destination);
    this.t0 = performance.now();
  }
  samples(from = 0, to = this.n) {  // device-rate samples [from, to)
    const out = new Float32Array(Math.max(0, to - from));
    let pos = 0;
    for (const c of this.chunks) {
      const a = Math.max(from - pos, 0), b = Math.min(to - pos, c.length);
      if (b > a) out.set(c.subarray(a, b), pos + a - from);
      pos += c.length;
      if (pos >= to) break;
    }
    return out;
  }
  async stop() {
    try { this.stream.getTracks().forEach((t) => t.stop()); await this.ctx.close(); } catch {}
  }
}

/* ------------------------------------------------------------------------------------------------- live mode */
// Chunked streaming as measured for the CPU runtime: every `chunk` seconds, encode the chunk with `left` s of past
// and `right` s of future audio around it, decode only the chunk's frames, carry the decoder state forward.
async function live(mic, cfg, ui) {
  const chunk = Math.round((cfg.chunk * SR) / HOP), left = Math.round((cfg.left * SR) / HOP), right = Math.round((cfg.right * SR) / HOP);
  const st = await newState();
  let t = 0, c0 = 0, encMs = 0, frontMs = 0, windows = 0, audioDone = 0;
  const toFrames = (n) => Math.floor((n * SR) / mic.rate / HOP);
  for (;;) {
    const stopped = mic.stopped, avail = toFrames(mic.n);
    const c1 = c0 + chunk;
    const last = stopped && avail < c1 + right;
    if (!last && avail < c1 + right) { await sleep(60); continue; }
    const end = last ? avail : c1;
    if (end <= c0 && last) break;
    const w0 = Math.max(0, c0 - left), w1 = last ? avail : c1 + right;
    const r = mic.rate / SR;
    const x = await resampleTo16k(mic.samples(Math.floor(w0 * HOP * r), Math.floor(w1 * HOP * r)), mic.rate);
    const e = await encode(x);
    encMs += e.encMs; frontMs += e.frontMs; windows++;
    st.f0 = w0;
    t = await tdtGreedy(e, Math.max(t, w0), last ? w0 + e.n : end, st);
    audioDone = end * HOP / SR;
    const heard = mic.n / mic.rate;
    ui({ text: text(st.ids), lag: Math.max(0, heard - audioDone), windowMs: e.encMs + e.frontMs, encMs, decMs: st.decMs, windows, audioDone, heard });
    if (last) break;
    c0 = c1;
  }
  return text(st.ids);
}

/* ---------------------------------------------------------------------------------------------------------- UI */
function setLang(l) {
  document.documentElement.lang = l; localStorage.setItem("lang", l);
  document.querySelectorAll("[data-set-lang]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.setLang === l)));
}
function bar(el, frac) { el.style.width = `${Math.max(0, Math.min(1, frac)) * 100}%`; }
function stat(id, v) { const el = $(id); if (el) el.textContent = v; }
function busy(on) { document.querySelectorAll("[data-needs-model]").forEach((b) => (b.disabled = on || !M.ready)); }

async function boot() {
  setLang(localStorage.getItem("lang") || "en");
  document.querySelectorAll("[data-set-lang]").forEach((b) => b.addEventListener("click", () => setLang(b.dataset.setLang)));
  await ensureIsolation();
  stat("#s-threads", self.crossOriginIsolated ? Math.min(navigator.hardwareConcurrency || 4, 8) : "1");
  stat("#s-iso", self.crossOriginIsolated ? "yes" : "no");
  $("#base").textContent = BASE;
  $("#total").textContent = fmtMB(TOTAL);
  const keys = await cachedKeys();
  const cached = FILES.filter((f) => [...keys].some((k) => k.startsWith(BASE + f.path + "#"))).length;
  $("#cache-state").textContent = cached === FILES.length ? L({ en: "already downloaded on this device", ta: "இந்தக் கருவியில் ஏற்கெனவே பதிவிறக்கப்பட்டது" })
    : cached ? `${cached}/${FILES.length}` : L({ en: "not downloaded yet", ta: "இன்னும் பதிவிறக்கவில்லை" });
  const rows = $("#files");
  rows.innerHTML = FILES.map((f, i) => `<div class="frow" id="f${i}"><span class="mono">${f.path.replace("onnx/", "")}</span><span class="mono">${fmtMB(f.size)}</span><span class="fst mono">–</span></div>`).join("");
  busy(true);

  $("#load").addEventListener("click", async () => {
    $("#load").disabled = true;
    $("#stage").textContent = L({ en: "downloading", ta: "பதிவிறக்குகிறது" });
    try {
      await loadModel((p) => {
        bar($("#dlbar"), p.got / p.total);
        $("#pbytes").textContent = `${fmtMB(p.got)} / ${fmtMB(p.total)}`;
        $("#pspeed").textContent = p.speed ? `${(p.speed / 1e6).toFixed(1)} MB/s` : "";
        $("#peta").textContent = p.speed > 1e5 ? `${L({ en: "about", ta: "சுமார்" })} ${Math.ceil((p.total - p.got) / p.speed)} s` : "";
        const fs = $(`#f${p.i} .fst`);
        fs.textContent = p.state === "downloading" ? `${Math.floor((100 * p.fileGot) / p.file.size)}%` : p.state;
        fs.dataset.state = p.state;
      }, (stage) => { $("#stage").textContent = L({ en: "starting ONNX Runtime", ta: "ONNX Runtime தொடங்குகிறது" }); });
      $("#stage").textContent = L({ en: "ready", ta: "தயார்" });
      stat("#s-threads", M.threads);
      stat("#s-load", `${M.loadS.toFixed(1)} s`);
      stat("#s-dl", `${M.downloadS.toFixed(1)} s`);
      $("#model").classList.add("ready");
      busy(false);
    } catch (e) {
      $("#stage").textContent = L({ en: "failed", ta: "தோல்வி" });
      showError(e, "load");
      $("#load").disabled = false;
    }
  });
  $("#clear").addEventListener("click", async () => { await caches.delete(CACHE); location.reload(); });

  // word timestamps: hoverable words, play along, download
  let last = null, player = null;
  const stopPlay = () => { if (player) { player.src.onended = null; try { player.src.stop(); } catch (e) {} player.ctx.close(); player = null; } $("#out").classList.remove("playing"); $("#playalong").setAttribute("aria-pressed", "false"); };
  const showWords = (ws, audio, label) => {
    last = { ws, audio, label };
    $("#out").innerHTML = ws.map(([w, s, e]) => `<span class="w" data-s="${s}" data-e="${e}" title="${s.toFixed(2)}–${e.toFixed(2)} s">${w.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c])}</span>`).join(" ");
    $("#wordbar").hidden = false;
  };
  $("#playalong").addEventListener("click", () => {
    if (player) return stopPlay();
    if (!last) return;
    const ctx = new AudioContext(), buf = ctx.createBuffer(1, last.audio.length, SR);
    buf.copyToChannel(last.audio, 0);
    const src = ctx.createBufferSource(); src.buffer = buf; src.connect(ctx.destination);
    player = { ctx, src, t0: ctx.currentTime }; src.start();
    $("#out").classList.add("playing"); $("#playalong").setAttribute("aria-pressed", "true");
    const sp = [...document.querySelectorAll("#out .w")];
    const tick = () => {
      if (!player) { sp.forEach((s) => s.classList.remove("on", "now")); return; }
      const tt = player.ctx.currentTime - player.t0; let cur = -1;
      sp.forEach((s, i) => { const on = +s.dataset.s <= tt; s.classList.toggle("on", on); if (on) cur = i; });
      sp.forEach((s, i) => s.classList.toggle("now", i === cur && tt <= +sp[i].dataset.e + 0.3));
      requestAnimationFrame(tick);
    };
    src.onended = stopPlay; tick();
  });
  $("#dlwords").addEventListener("click", () => {
    if (!last) return;
    const blob = new Blob([JSON.stringify({ source: last.label, words: last.ws.map(([w, s, e]) => ({ word: w, start: s, end: e })) }, null, 1)], { type: "application/json" });
    const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "words.json"; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  const run = async (audio, label) => {
    busy(true); stopPlay(); $("#wordbar").hidden = true; last = null;
    $("#out").textContent = "…"; $("#src").textContent = label;
    try {
      const r = await transcribeOffline(audio, (s) => s.partial && ($("#out").textContent = s.partial + " …"));
      $("#out").textContent = r.text || L({ en: "(nothing recognised)", ta: "(எதுவும் கேட்கவில்லை)" });
      if (r.words.length) showWords(r.words, audio, label);
      showStats(r.stats);
    } catch (e) { showError(e, "transcribe"); $("#out").textContent = ""; }
    busy(false);
  };
  const showStats = (s) => {
    stat("#s-audio", `${s.audio.toFixed(2)} s`); stat("#s-front", `${s.frontMs.toFixed(0)} ms`);
    stat("#s-enc", `${s.encMs.toFixed(0)} ms`); stat("#s-dec", `${s.decMs.toFixed(0)} ms`);
    stat("#s-total", `${s.totalMs.toFixed(0)} ms`); stat("#s-rtfx", `${s.rtfx.toFixed(1)}×`);
    stat("#s-tok", `${s.tokens} / ${s.steps}`);
    if (performance.memory) stat("#s-mem", fmtMB(performance.memory.usedJSHeapSize));
  };

  // presets: the write-up's clips
  try {
    const seen = new Set();
    const clips = (await (await fetch("../data/clips.json")).json()).filter((c) => !seen.has(c.audio) && seen.add(c.audio));
    $("#presets").innerHTML = clips.map((c, i) => `<button data-needs-model data-clip="${i}"><span data-l="en">${c.label ? c.label.en : c.source.split(" ·")[0]}</span><span data-l="ta">${c.label ? c.label.ta : c.source.split(" ·")[0]}</span> <small class="mono">${c.dur ? c.dur.toFixed(1) + " s" : ""}</small></button>`).join("");
    $("#presets").querySelectorAll("[data-clip]").forEach((b) => b.addEventListener("click", async () => {
      const c = clips[+b.dataset.clip];
      try { await run(await decodeFile(await (await fetchOk("../" + c.audio, "audio")).blob()), c.label ? `${L(c.label)} · ${c.source}` : c.source); }
      catch (e) { showError(e, "audio"); busy(false); }
    }));
  } catch (e) { showError(e, "presets"); }
  busy(true);

  $("#file").addEventListener("change", async (ev) => {
    const f = ev.target.files[0];
    if (!f) return;
    try { await run(await decodeFile(f), f.name); } catch (e) { showError(e, "audio"); busy(false); }
    ev.target.value = "";
  });

  // record, then transcribe
  let mic = null;
  const level = (d) => { let m = 0; for (let i = 0; i < d.length; i += 4) m = Math.max(m, Math.abs(d[i])); bar($("#lvl"), Math.min(1, m * 1.5)); };
  $("#rec").addEventListener("click", async () => {
    if (!mic) {
      try {
        mic = new Mic(); mic.onLevel = level; await mic.start();
        $("#rec").setAttribute("aria-pressed", "true"); $("#live").disabled = true;
        const tick = setInterval(() => { if (!mic) return clearInterval(tick); $("#rtime").textContent = `${(mic.n / mic.rate).toFixed(1)} s`; }, 200);
      } catch (e) { showError(e, "microphone"); mic = null; }
      return;
    }
    const m = mic; mic = null; await m.stop();
    $("#rec").setAttribute("aria-pressed", "false"); $("#live").disabled = !M.ready; bar($("#lvl"), 0);
    await run(await resampleTo16k(m.samples(), m.rate), L({ en: "your recording", ta: "உங்கள் பதிவு" }));
  });

  // live (chunked streaming)
  let liveMic = null;
  $("#live").addEventListener("click", async () => {
    if (liveMic) { liveMic.stopped = true; $("#live").disabled = true; return; }
    const cfg = { chunk: +$("#chunk").value, left: +$("#left").value, right: +$("#right").value };
    try {
      liveMic = new Mic(); liveMic.onLevel = level; await liveMic.start();
    } catch (e) { showError(e, "microphone"); liveMic = null; return; }
    $("#live").setAttribute("aria-pressed", "true"); $("#rec").disabled = true;
    $("#src").textContent = L({ en: "live microphone", ta: "நேரடி மைக்ரோஃபோன்" }); $("#out").textContent = "…"; stopPlay(); $("#wordbar").hidden = true; last = null;
    const t0 = performance.now();
    try {
      await live(liveMic, cfg, (u) => {
        $("#out").textContent = u.text || "…";
        stat("#s-lag", `${u.lag.toFixed(1)} s`); stat("#s-win", `${u.windowMs.toFixed(0)} ms`);
        stat("#s-audio", `${u.audioDone.toFixed(1)} s`); stat("#s-enc", `${u.encMs.toFixed(0)} ms`); stat("#s-dec", `${u.decMs.toFixed(0)} ms`);
        stat("#s-rtfx", `${(u.audioDone / ((u.encMs + u.decMs) / 1000)).toFixed(1)}×`);
        $("#lagwarn").hidden = u.lag < cfg.chunk + cfg.right + 2;
      });
    } catch (e) { showError(e, "live"); }
    await liveMic.stop(); liveMic = null; bar($("#lvl"), 0);
    $("#live").setAttribute("aria-pressed", "false"); $("#live").disabled = !M.ready; $("#rec").disabled = !M.ready;
  });
}
boot();
