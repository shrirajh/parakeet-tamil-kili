// parakeet-tamil · தமிழ் கிளி. Vanilla JS, no dependencies; loads only ./data and ./audio.
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const lang = () => document.documentElement.lang;
const L = (o) => (o && typeof o === "object" ? o[lang()] ?? o.en : o);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
const J = async (p) => (await fetch(p)).json();

/* ---------- language, theme, progress, reveal ---------- */
const redraw = [];
function setLang(l) {
  document.documentElement.lang = l; localStorage.setItem("lang", l);
  $$("[data-set-lang]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.setLang === l)));
  redraw.forEach((f) => f());
}
$$("[data-set-lang]").forEach((b) => b.addEventListener("click", () => setLang(b.dataset.setLang)));
$("#theme").addEventListener("click", () => {
  const t = document.documentElement.dataset.theme === "light" ? "" : "light";
  if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
  localStorage.setItem("theme", t); redraw.forEach((f) => f());
});
addEventListener("scroll", () => { const h = document.documentElement; $("#pbar").style.transform = `scaleX(${h.scrollTop / (h.scrollHeight - h.clientHeight)})`; }, { passive: true });
const io = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); e.target.dispatchEvent(new Event("shown")); io.unobserve(e.target); } }), { threshold: 0.12 });
const observe = () => $$(".reveal:not(.in)").forEach((el) => io.observe(el));

/* ---------- tooltip ---------- */
const tip = Object.assign(document.createElement("div"), { className: "tip" }); document.body.append(tip);
const showTip = (e, h) => { tip.innerHTML = h; tip.style.opacity = 1; tip.style.left = Math.min(e.clientX + 14, innerWidth - 310) + "px"; tip.style.top = e.clientY + 16 + "px"; };
const hideTip = () => (tip.style.opacity = 0);
const tips = (root) => $$("[data-tip]", root).forEach((el) => { el.addEventListener("mousemove", (e) => showTip(e, el.dataset.tip)); el.addEventListener("mouseleave", hideTip); });

/* ---------- audio: one player at a time, waveform peaks via WebAudio ---------- */
let current = null;
const peaksCache = {};
async function peaks(url, n) {
  if (!peaksCache[url]) peaksCache[url] = (async () => {
    const buf = await (await fetch(url)).arrayBuffer();
    const ctx = new (window.OfflineAudioContext || window.webkitOfflineAudioContext)(1, 1, 24000);
    return (await ctx.decodeAudioData(buf)).getChannelData(0);
  })();
  const d = await peaksCache[url], step = Math.floor(d.length / n), out = [];
  for (let i = 0; i < n; i++) { let m = 0; for (let j = i * step; j < (i + 1) * step; j += 4) m = Math.max(m, Math.abs(d[j] || 0)); out.push(m); }
  const mx = Math.max(...out) || 1; return out.map((v) => v / mx);
}
async function drawWave(canvas, url, frac = 0) {
  const r = canvas.getBoundingClientRect(); if (!r.width) return;
  const dpr = devicePixelRatio || 1; canvas.width = r.width * dpr; canvas.height = r.height * dpr;
  const g = canvas.getContext("2d"), n = Math.max(40, Math.floor(r.width / 5)), p = await peaks(url, n);
  g.clearRect(0, 0, canvas.width, canvas.height);
  const w = canvas.width / n, mid = canvas.height / 2;
  p.forEach((v, i) => { g.fillStyle = i / n < frac ? css("--ours") : css("--rule"); const h = Math.max(2 * dpr, v * canvas.height * 0.92); g.fillRect(i * w + w * 0.2, mid - h / 2, w * 0.6, h); });
}
function wirePlayer({ url, button, canvas, onTime }) {
  const audio = new Audio(url); audio.preload = "none";
  const paint = () => drawWave(canvas, url, audio.currentTime / (audio.duration || 1));
  button.addEventListener("click", () => { if (audio.paused) { if (current && current !== audio) current.pause(); current = audio; audio.play(); } else audio.pause(); });
  audio.addEventListener("play", () => { button.setAttribute("aria-pressed", "true"); button.textContent = "❚❚"; tick(); });
  audio.addEventListener("pause", () => { button.setAttribute("aria-pressed", "false"); button.textContent = "▶"; });
  audio.addEventListener("ended", () => { onTime?.(1e9); paint(); });
  const tick = () => { paint(); onTime?.(audio.currentTime); if (!audio.paused) requestAnimationFrame(tick); };
  canvas.parentElement.addEventListener("click", (e) => { const r = canvas.getBoundingClientRect(); if (!audio.duration) { button.click(); return; } audio.currentTime = ((e.clientX - r.left) / r.width) * audio.duration; if (audio.paused) button.click(); });
  const io2 = new IntersectionObserver((es) => { if (es[0].isIntersecting) { paint(); io2.disconnect(); } }); io2.observe(canvas);
  addEventListener("resize", paint);
  return paint;
}

/* ---------- word diff: punctuation ignored, other-script loanwords marked as fold ---------- */
const key = (w) => w.toLowerCase().replace(/[.,?!;:"'“”‘’()…]/g, "");
const latin = (w) => /[a-z]/i.test(w);
function diffHTML(ref, hyp) {
  const a = ref.split(/\s+/).filter(key), b = hyp.split(/\s+/).filter(key), ka = a.map(key), kb = b.map(key);
  const n = a.length, m = b.length, D = Array.from({ length: n + 1 }, (_, i) => [i, ...Array(m).fill(0)]);
  for (let j = 1; j <= m; j++) D[0][j] = j;
  for (let i = 1; i <= n; i++) for (let j = 1; j <= m; j++) D[i][j] = Math.min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + (ka[i - 1] === kb[j - 1] ? 0 : 1));
  const out = []; let i = n, j = m;
  while (i > 0 || j > 0) {
    if (i && j && D[i][j] === D[i - 1][j - 1] + (ka[i - 1] === kb[j - 1] ? 0 : 1)) {
      const w = esc(b[j - 1]), r = esc(a[i - 1]);
      out.push(ka[i - 1] === kb[j - 1] ? w : latin(b[j - 1]) !== latin(a[i - 1]) ? `<span class="w-fold" title="${r}">${w}</span>` : `<span class="w-sub" title="${r}">${w}</span>`); i--; j--;
    } else if (j && D[i][j] === D[i][j - 1] + 1) { out.push(`<span class="w-ins">${esc(b[j - 1])}</span>`); j--; }
    else { out.push(`<span class="w-del">${esc(a[i - 1])}</span>`); i--; }
  }
  return out.reverse().join(" ");
}

/* ---------- hero ---------- */
function hero(clips) {
  const c = clips.find((x) => x.group === "hero"); if (!c) return;
  const box = $("#hero-words");
  box.innerHTML = c.words.map(([w, t]) => `<span data-t="${t}" class="${latin(w) ? "lat" : ""}">${esc(w)}</span>`).join(" ");
  const ws = $$("span", box);
  redraw.push(wirePlayer({ url: c.audio, button: $("#hero-play"), canvas: $("#hero-wave canvas"), onTime: (t) => ws.forEach((w) => w.classList.toggle("on", +w.dataset.t <= t + 0.05)) }));
}
function counters() {
  $$("[data-count]").forEach((el) => {
    const to = +el.dataset.count, dec = +el.dataset.dec, suf = el.dataset.suffix || "", t0 = performance.now(), dur = reduce ? 0 : 1400;
    const f = (t) => { const k = dur ? Math.min(1, (t - t0) / dur) : 1, e = 1 - Math.pow(1 - k, 3); el.textContent = (to * e).toFixed(dec) + suf; if (k < 1) requestAnimationFrame(f); };
    requestAnimationFrame(f);
  });
}

/* ---------- quiz ---------- */
const score = { right: 0, done: 0 };
function quiz(clips, metrics) {
  $$("[data-quiz]").forEach((host) => {
    host.innerHTML = "";
    clips.filter((c) => c.group === host.dataset.quiz).forEach((c, k) => {
      const opts = [["ours", c.ours], ["itcore", c.itcore]]; if (k % 2) opts.reverse();
      const right = c.answer || "ours", M = metrics[c.id] || {};
      const card = document.createElement("div"); card.className = "qcard reveal in";
      const mline = (w) => (M[w] ? `<div class="small mono mline" style="display:none;margin-top:.3rem">${L({ en: "vs label", ta: "லேபிளுடன்" })}: WER ${M[w].wer_fold ?? M[w].wer} · PER ${M[w].per_fold ?? "–"} · EER ${M[w].eer}</div>` : "");
      card.innerHTML = `<div class="qhead"><button class="qplay" aria-pressed="false" aria-label="play">▶</button><div class="mini"><canvas></canvas></div><span class="qsrc">${esc(c.source)}</span></div>
        <div class="opts">${opts.map(([w, t]) => `<button class="opt ${w}" data-w="${w}"><span class="who">${w === "ours" ? "parakeet-tamil" : "indic-transcribe-core"}</span><span class="txt" data-raw="${esc(t)}">${esc(t)}</span>${mline(w)}</button>`).join("")}
          <div class="opt ref" style="display:none"><span class="who">${L({ en: "label", ta: "லேபிள்" })}</span>${esc(c.ref)}</div></div>
        <div class="verdict">${esc(L(c.note))}</div>`;
      wirePlayer({ url: c.audio, button: $(".qplay", card), canvas: $("canvas", card) });
      $$("button.opt", card).forEach((b) => b.addEventListener("click", () => {
        if (card.classList.contains("done")) return;
        card.classList.add("done"); score.done++; if (b.dataset.w === right) score.right++;
        $(`.opt.${right}`, card).classList.add("right");
        $(".opt.ref", card).style.display = "block";
        $$(".mline", card).forEach((x) => (x.style.display = "block"));
        $$("button.opt .txt", card).forEach((t) => (t.innerHTML = diffHTML(c.ref, t.dataset.raw)));
        $$("#score-line").forEach((sl) => (sl.textContent = L({ en: `You: ${score.right} of ${score.done}`, ta: `நீங்கள்: ${score.done}-இல் ${score.right}` })));
      }));
      host.append(card);
    });
  });
}

/* ---------- scoreboard dumbbell ---------- */
let SC, metric = "wer", withLM = false;
function dumbbell() {
  if (!SC) return;
  const host = $("#dumbbell"), S = SC.scores;
  const sets = Object.keys(SC.sets).filter((s) => metric === "wer" || SC.sets[s].kind !== "english");
  const vals = sets.flatMap((s) => ["e5", "itcore", "e5_lm"].map((m) => S[m]?.[s]?.[metric]).filter((v) => v != null));
  const W = 1000, rowH = 44, GAPH = 30, P = { l: 220, r: 150, t: 20 };
  const kinds = [...new Set(sets.map((s) => SC.sets[s].kind))], H = P.t + sets.length * rowH + kinds.length * GAPH + 30;
  const mx = Math.max(...vals), stp = mx > 30 ? 10 : mx > 12 ? 5 : mx > 6 ? 2 : mx > 3 ? 1 : 0.5, lo = 0, hi = Math.ceil((mx * 1.04) / stp) * stp, x = (v) => P.l + ((v - lo) / (hi - lo)) * (W - P.l - P.r);
  const kindLbl = { spoken: { en: "spoken Tamil", ta: "பேச்சுத் தமிழ்" }, read: { en: "read Tamil", ta: "வாசிப்புத் தமிழ்" }, english: { en: "English", ta: "ஆங்கிலம்" } };
  let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="scoreboard">`;
  for (let v = 0; v <= hi + 1e-9; v += stp) { g += `<g class="grid"><line x1="${x(v)}" x2="${x(v)}" y1="${P.t}" y2="${H - 24}"/></g><text x="${x(v)}" y="${H - 6}" text-anchor="middle">${+v.toFixed(1)}</text>`; }
  let last = null, yy = P.t;
  sets.forEach((s) => {
    const k = SC.sets[s].kind;
    if (k !== last) { yy += GAPH; g += `<text x="0" y="${yy - 8}" style="fill:var(--fg-3);font-size:10px;letter-spacing:.08em">${esc(L(kindLbl[k]).toUpperCase())}</text>`; last = k; }
    const y = yy + rowH / 2; yy += rowH;
    const o = S.e5[s][metric], it = S.itcore[s]?.[metric], lm = S.e5_lm[s]?.[metric];
    const ours = withLM && lm != null ? lm : o;
    g += `<text class="lbl" x="0" y="${y + 5}">${esc(L(SC.sets[s]))}</text>`;
    if (it != null) g += `<line x1="${x(ours)}" x2="${x(it)}" y1="${y}" y2="${y}" stroke="var(--fg-3)" stroke-width="2" opacity=".45"/>`;
    if (withLM && lm != null) g += `<circle cx="${x(o)}" cy="${y}" r="7" fill="none" stroke="var(--ours)" stroke-width="2" opacity=".55" data-tip="parakeet-tamil · ${o.toFixed(2)}"/>`;
    if (it == null) g += `<circle cx="${x(ours)}" cy="${y}" r="8" fill="var(--ours)" stroke="var(--bg-2)" stroke-width="2" data-tip="parakeet-tamil${withLM ? " + LM" : ""} · ${ours.toFixed(2)}"/>`;
    if (it != null) {
      g += `<circle cx="${x(it)}" cy="${y}" r="9" fill="var(--itcore)" stroke="var(--bg-2)" stroke-width="2" data-tip="indic-transcribe-core · ${it.toFixed(2)}"/>`;
    g += `<circle cx="${x(ours)}" cy="${y}" r="8" fill="var(--ours)" stroke="var(--bg-2)" stroke-width="2" data-tip="parakeet-tamil${withLM ? " + LM" : ""} · ${ours.toFixed(2)}"/>`;
      const d = ours - it, win = d < 0;
      g += `<text x="${W - P.r + 18}" y="${y + 5}" style="fill:var(--fg);font-weight:700;font-size:14px">${win ? "▲" : "▼"} ${Math.abs(d).toFixed(metric === "wer" || metric === "cer" ? 1 : 2)}</text>`;
      g += `<text x="${W - 2}" y="${y + 5}" text-anchor="end" style="font-size:11px">${win ? L({ en: "we lead", ta: "நாம் முன்னால்" }) : L({ en: "behind", ta: "பின்னால்" })}</text>`;
    }
  });
  host.innerHTML = g + "</svg>"; tips(host);
  $("#dumb-legend").innerHTML = [["--ours", "parakeet-tamil" + (withLM ? " + LM" : "")], ["--itcore", "indic-transcribe-core"]].map(([c, n]) => `<span class="key"><span class="dot" style="background:var(${c})"></span>${n}</span>`).join("") + `<span class="key dim">${L({ en: "lower is better", ta: "குறைவே சிறந்தது" })} · ${metric.toUpperCase()} %</span>`;
}
$$("[data-metric]").forEach((b) => b.addEventListener("click", () => { metric = b.dataset.metric; $$("[data-metric]").forEach((x) => x.setAttribute("aria-pressed", String(x === b))); dumbbell(); }));
$("#lm-toggle").addEventListener("click", (e) => { withLM = !withLM; e.currentTarget.setAttribute("aria-pressed", String(withLM)); dumbbell(); });

/* ---------- epochs ---------- */
let eSet = "eval_kathbath";
function epochs() {
  if (!SC) return;
  const sets = ["eval_indicvoices", "eval_kathbath", "eval_fleurs_ta", "eval_cv_ta", "eval_openslr65", "eval_cv_en", "eval_svarah"];
  $("#epoch-sets").innerHTML = sets.map((s) => `<button data-eset="${s}" aria-pressed="${s === eSet}">${esc(L(SC.sets[s]))}</button>`).join("");
  $$("[data-eset]").forEach((b) => b.addEventListener("click", () => { eSet = b.dataset.eset; epochs(); }));
  const ys = ["e1", "e2", "e3", "e4", "e5", "e6"].map((e) => SC.scores[e][eSet].wer), ref = SC.scores.itcore[eSet]?.wer;
  const all = [...ys, ref].filter((v) => v != null), lo = Math.floor(Math.min(...all) - 1.5), hi = Math.ceil(Math.max(...all) + 1.5);
  const W = 640, H = 340, P = { l: 44, r: 24, t: 20, b: 40 }, x = (i) => P.l + (i * (W - P.l - P.r)) / 5, y = (v) => P.t + ((hi - v) * (H - P.t - P.b)) / (hi - lo);
  const best = ys.indexOf(Math.min(...ys)), path = ys.map((v, i) => `${i ? "L" : "M"}${x(i)},${y(v)}`).join("");
  let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="WER by epoch">`;
  for (let k = 0; k <= 4; k++) { const v = lo + ((hi - lo) * k) / 4; g += `<g class="grid"><line x1="${P.l}" x2="${W - P.r}" y1="${y(v)}" y2="${y(v)}"/></g><text x="${P.l - 8}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(0)}</text>`; }
  if (ref != null) g += `<line x1="${P.l}" x2="${W - P.r}" y1="${y(ref)}" y2="${y(ref)}" stroke="var(--itcore)" stroke-width="2" stroke-dasharray="6 5"/><text x="${W - P.r}" y="${y(ref) - 8}" text-anchor="end" style="fill:var(--fg)">indic-transcribe-core ${ref.toFixed(2)}</text>`;
  g += `<path id="ep-path" d="${path}" fill="none" stroke="var(--ours)" stroke-width="3" stroke-linejoin="round"/>`;
  ys.forEach((v, i) => { g += `<circle cx="${x(i)}" cy="${y(v)}" r="${i === best ? 8 : 6}" fill="${i === best ? "var(--ours)" : "var(--bg-2)"}" stroke="${i === 5 && v > ys[4] ? "var(--bad)" : "var(--ours)"}" stroke-width="3" data-tip="${L({ en: "epoch", ta: "சுற்று" })} ${i + 1} · WER ${v.toFixed(2)}"/><text x="${x(i)}" y="${H - 14}" text-anchor="middle">${i + 1}</text>`; });
  g += `<text x="${x(best)}" y="${y(ys[best]) + 28}" text-anchor="middle" style="fill:var(--fg);font-weight:700">${ys[best].toFixed(2)}</text>`;
  $("#epoch-chart").innerHTML = g + "</svg>"; tips($("#epoch-chart"));
  const p = $("#ep-path"); if (p && !reduce) { const len = p.getTotalLength(); p.style.strokeDasharray = len; p.style.strokeDashoffset = len; p.getBoundingClientRect(); p.style.transition = "stroke-dashoffset 1.1s ease"; p.style.strokeDashoffset = 0; }
}

/* ---------- runs: English gate + strip ---------- */
const RUNS = [
  { k: "stock", d: "", s: 16.5, t: { en: "stock parakeet v3", ta: "அடிப்படை parakeet v3" }, p: { en: "No Tamil: Kathbath WER 117%.", ta: "தமிழே இல்லை: Kathbath WER 117%." } },
  { k: "run 1", d: "9 Nov", s: 43.5, t: { en: "Tamil in Latin letters", ta: "லத்தீன் எழுத்தில் தமிழ்" }, p: { en: "Tamil improved. English came out spelled like Tamil: \"Chuus tistrikt… veekchineeshan\".", ta: "தமிழ் மேம்பட்டது. ஆங்கிலம் தமிழ் போல எழுதப்பட்டது: \"Chuus tistrikt… veekchineeshan\"." } },
  { k: "run 2", d: "10 Nov", s: 29.3, t: { en: "2,044 Tamil tokens", ta: "2,044 தமிழ் token-கள்" }, p: { en: "Native Tamil script. Before training, identical to the stock model.", ta: "தமிழ் எழுத்து. பயிற்சிக்கு முன் அடிப்படை மாதிரிக்கு இணை." } },
  { k: "run 3", d: "10 Nov", s: 30.5, t: { en: "extra heads", ta: "கூடுதல் தலைகள்" }, p: { en: "Phoneme and language-ID heads, English distillation.", ta: "ஒலியன், மொழி அடையாளத் தலைகள், ஆங்கில வடிகட்டல்." } },
  { k: "run 4", d: "10 Nov", s: 45.4, t: { en: "synthetic Tanglish", ta: "செயற்கை Tanglish" }, p: { en: "Read Tamil up, English down again.", ta: "வாசிப்புத் தமிழ் மேலே, ஆங்கிலம் மீண்டும் கீழே." } },
  { k: "run 5", d: "11 Nov", s: 12.4, t: { en: "one spelling rule", ta: "ஒரே எழுத்து விதி" }, p: { en: "English loanwords in English letters (meeting-க்கு). English came back.", ta: "ஆங்கிலக் கடன்சொற்கள் ஆங்கில எழுத்தில் (meeting-க்கு). ஆங்கிலம் திரும்பியது." } },
  { k: "run 6", d: "11 Nov", s: 13.5, t: { en: "noise and phones", ta: "இரைச்சல், தொலைபேசி" }, p: { en: "Noise, reverb, phone codecs, babble, long audio.", ta: "இரைச்சல், எதிரொலி, தொலைபேசி codec, கூட்டப் பேச்சு, நீண்ட ஒலி." } },
  { k: "run 7", d: "11 Nov", s: 11.5, t: { en: "evicted at 91%", ta: "91%-இல் வெளியேற்றம்" }, p: { en: "1,400 h of world English. Finished on a single GPU after the eviction.", ta: "1,400 மணி உலக ஆங்கிலம். வெளியேற்றத்துக்குப் பின் ஒரே GPU-வில் முடிந்தது." } },
  { k: "run 8", d: "12 Nov", s: 12.07, t: { en: "punctuation", ta: "நிறுத்தக்குறிகள்" }, p: { en: "Gemma punctuation, 30,895 bad clips dropped.", ta: "Gemma நிறுத்தக்குறிகள், 30,895 மோசமான ஒலிக்கோப்புகள் நீக்கம்." } },
  { k: "run 9", d: "13 Nov", s: null, t: { en: "formal Tamil targets", ta: "எழுத்துத் தமிழ் இலக்குகள்" }, p: { en: "Spoken Tamil got worse. Dropped.", ta: "பேச்சுத் தமிழ் மோசமானது. கைவிடப்பட்டது." } },
  { k: "run 10", d: "14 Nov", s: 10.67, ship: 1, t: { en: "from scratch", ta: "புதிதாக" }, p: { en: "18,746 audio hours per 3 epochs, about 3 h on 8 GPUs. Extended to 6. Epoch 5 ships.", ta: "3 சுற்றுகளுக்கு 18,746 ஒலி மணிகள், 8 GPU-களில் சுமார் 3 மணி. 6 வரை நீட்டிப்பு. 5-ஆவது சுற்று வெளியீடு." } },
];
function runs() {
  const pts = RUNS.map((r, i) => [i, r.s]).filter(([, s]) => s != null);
  const W = 1000, H = 300, P = { l: 40, r: 20, t: 20, b: 40 }, hi = 50, x = (i) => P.l + (i * (W - P.l - P.r)) / (RUNS.length - 1), y = (v) => P.t + ((hi - v) * (H - P.t - P.b)) / hi;
  let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Svarah WER by run">`;
  [0, 10, 20, 30, 40, 50].forEach((v) => (g += `<g class="grid"><line x1="${P.l}" x2="${W - P.r}" y1="${y(v)}" y2="${y(v)}"/></g><text x="${P.l - 8}" y="${y(v) + 4}" text-anchor="end">${v}</text>`));
  g += `<rect x="${P.l}" y="${y(hi)}" width="${W - P.l - P.r}" height="${y(17) - y(hi)}" fill="var(--bad)" opacity=".08"/><line x1="${P.l}" x2="${W - P.r}" y1="${y(17)}" y2="${y(17)}" stroke="var(--bad)" stroke-dasharray="6 5" stroke-width="2"/><text x="${W - P.r}" y="${y(17) - 8}" text-anchor="end" style="fill:var(--fg)">${L({ en: "English gate: 17", ta: "ஆங்கில வாயில்: 17" })}</text>`;
  g += `<path d="${pts.map(([i, s], k) => `${k ? "L" : "M"}${x(i)},${y(s)}`).join("")}" fill="none" stroke="var(--ours)" stroke-width="3" stroke-linejoin="round"/>`;
  RUNS.forEach((r, i) => {
    if (r.s != null) g += `<circle class="rp" data-i="${i}" cx="${x(i)}" cy="${y(r.s)}" r="${r.ship ? 9 : 6}" fill="${r.ship ? "var(--ours)" : "var(--bg)"}" stroke="${r.s > 17 ? "var(--bad)" : "var(--ours)"}" stroke-width="3" data-tip="${esc(r.k)} · Svarah WER ${r.s}"/>`;
    else g += `<text x="${x(i)}" y="${y(8)}" text-anchor="middle">?</text>`;
    g += `<text x="${x(i)}" y="${H - 12}" text-anchor="middle">${r.k.replace("run ", "")}</text>`;
  });
  $("#gate-chart").innerHTML = g + "</svg>"; tips($("#gate-chart"));
  $("#run-strip").innerHTML = RUNS.map((r, i) => `<article class="run ${r.ship ? "ship" : ""}" data-i="${i}"><div class="day">${esc(r.k)}${r.d ? " · " + esc(r.d) : ""}</div><h3>${esc(L(r.t))}</h3><p>${esc(L(r.p))}</p>${r.s != null ? `<div class="res">Svarah ${r.s}</div>` : ""}</article>`).join("");
  $$("#gate-chart .rp").forEach((c) => c.addEventListener("mouseenter", () => { $$(".run").forEach((a) => a.classList.toggle("hl", a.dataset.i === c.dataset.i)); const a = $(`.run[data-i="${c.dataset.i}"]`); a?.parentElement.scrollTo({ left: a.offsetLeft - 40, behavior: "smooth" }); }));
}

/* ---------- eviction simulator ---------- */
function sim() {
  const TOT = 21619, CK = 540; let step = 0, ck = 0, state = "run";
  const bar = $("#sim-track i"), track = $("#sim-track"), log = $("#sim-log");
  for (let s = CK; s < TOT; s += CK) { const m = document.createElement("span"); m.className = "ck"; m.style.left = (100 * s) / TOT + "%"; track.append(m); }
  const say = (t) => { log.textContent = (log.textContent + "\n" + t).split("\n").slice(-6).join("\n").trim(); };
  const set = (id, cls, txt) => { $(id).className = "box " + cls; if (txt) $(id + "-v").textContent = txt; };
  const loop = () => {
    if (state === "run") { step = step >= TOT ? 0 : step + 23; if (Math.floor(step / CK) * CK > ck) { ck = Math.floor(step / CK) * CK; $("#sim-s3-v").textContent = `checkpoint @ ${ck.toLocaleString()}`; } if (step === 0) ck = 0; }
    bar.style.width = (100 * step) / TOT + "%"; $("#sim-step").textContent = `step ${step.toLocaleString()} / ${TOT.toLocaleString()}`;
    setTimeout(loop, 60);
  };
  loop();
  $("#sim-evict").addEventListener("click", () => {
    if (state !== "run") return; state = "dead"; $("#sim-evict").disabled = true;
    const lost = step - ck; set("#sim-vm", "dead", L({ en: "gone", ta: "மறைந்தது" })); log.textContent = "";
    const T = [[0, "t+0 s    the cloud takes the VM back"], [700, "t+30 s   supervisor: API says 404, VM is gone"], [1300, "t+34 s   launch 8 × RTX PRO 6000 on the kept disk"],
      [2100, "t+74 s   booted, bootstrap starts"], [2800, `t+92 s   resume from checkpoint ${ck.toLocaleString()} (${lost} steps to redo)`], [3400, "t+105 s  training again"]];
    set("#sim-sup", "busy", L({ en: "relaunching", ta: "மீண்டும் தொடங்குகிறது" }));
    T.forEach(([ms, t]) => setTimeout(() => say(t), reduce ? 0 : ms));
    setTimeout(() => { step = ck; set("#sim-vm", "busy", L({ en: "booting", ta: "தொடங்குகிறது" })); }, reduce ? 0 : 2100);
    setTimeout(() => { state = "run"; set("#sim-vm", "live", "8 × RTX PRO 6000"); set("#sim-sup", "live", L({ en: "watching", ta: "கண்காணிக்கிறது" })); $("#sim-evict").disabled = false; }, reduce ? 0 : 3400);
  });
}

/* ---------- teacher ---------- */
const PUNCT = { raw: "இப்ப இப்ப ம இப்ப மதியம் ஆயிடுச்சி இப்ப நீங்க ஒன்னு பண்ணுங்க நீங்க சாய்ங்காலத்துக்குள்ள அதுக்கான procedure என்னன்னு பாத்து",
  on: "இப்ப இப்ப ம இப்ப மதியம் ஆயிடுச்சி. இப்ப நீங்க ஒன்னு பண்ணுங்க. நீங்க சாய்ங்காலத்துக்குள்ள அதுக்கான procedure என்னன்னு பாத்து." };
let punctOn = false;
function teacher() {
  const b = $("#punct-toggle"), t = $("#punct-text");
  t.innerHTML = punctOn ? esc(PUNCT.on).replace(/([.?,])/g, '<b style="color:var(--ours-2)">$1</b>') : esc(PUNCT.raw);
  b.setAttribute("aria-pressed", String(punctOn)); b.textContent = L({ en: `Gemma: ${punctOn ? "on" : "off"}`, ta: `Gemma: ${punctOn ? "இயக்கம்" : "அணைப்பு"}` });
}
$("#punct-toggle").addEventListener("click", () => { punctOn = !punctOn; teacher(); });

/* ---------- flip cards ---------- */
const DIRTY = [
  ["sir-ஆ", { en: "Our own loanword converter turned \"சார் நா\" (sir, I…) into \"sir-ஆ\". 670 times. Fixed before run 10.", ta: "எங்கள் கடன்சொல் மாற்றி \"சார் நா\" என்பதை \"sir-ஆ\" ஆக்கியது. 670 முறை. run 10-க்கு முன் சரிசெய்யப்பட்டது." }, "×670"],
  ["police சாமியார்", { en: "போலிச் சாமியார் means \"fake godman\". The same converter made him a policeman.", ta: "போலிச் சாமியார் — அதே மாற்றி அவரைக் காவலர் ஆக்கியது." }, "our bug"],
  ["touch-உ", { en: "டச்சு (Dutch) became touch-உ. It also turned up inside a Tirumantiram verse.", ta: "டச்சு, touch-உ ஆனது. திருமந்திரப் பாடல் ஒன்றிலும் வந்தது." }, "our bug"],
  ["veekchineeshan", { en: "Run 1 learned to write English the Tamil way: \"Choose district… vaccination\" came out as \"Chuus tistrikt… veekchineeshan\".", ta: "run 1 ஆங்கிலத்தைத் தமிழ் முறையில் எழுதக் கற்றது: \"Chuus tistrikt… veekchineeshan\"." }, "run 1"],
  ["56.7%", { en: "of one SPRING-INX release duplicates the previous one: 29,482 clips, 67 h.", ta: "ஒரு SPRING-INX வெளியீட்டின் 56.7% முந்தையதன் நகல்: 29,482 ஒலிக்கோப்புகள், 67 மணி." }, "duplicates"],
  ["ஏலு", { en: "IndicVoices read labels write ழ as ல: ஏலு for ஏழு (7), about 15 times per 100 labels.", ta: "IndicVoices வாசிப்பு லேபிள்கள் ழ-வை ல ஆக எழுதுகின்றன: ஏழு → ஏலு, 100 லேபிள்களுக்கு சுமார் 15 முறை." }, "label"],
  ["6.4", { en: "errors per 100 words in Shrutilipi labels, the sloppiest source. A third of its labels have one.", ta: "Shrutilipi லேபிள்களில் 100 சொற்களுக்குப் பிழைகள். மூன்றில் ஒரு லேபிளில் பிழை." }, "per 100 words"],
  ["\"okay mam\"", { en: "The whole label for 10 seconds of fluent Tamil speech. 30,895 clips like it were dropped.", ta: "10 வினாடி சரளமான தமிழ்ப் பேச்சுக்கு முழு லேபிள் இதுதான். இப்படி 30,895 நீக்கப்பட்டன." }, "label"],
];
const HOMO = [["மழை", "மலை", "rain", "hill", "மழை", "மலை"], ["வலி", "வழி", "pain", "way", "வலி", "வழி"], ["கொல்ல", "கொள்ள", "kill", "take", "கொல்", "கொள்"], ["அழிக்க", "அளிக்க", "destroy", "give", "அழி", "அளி"], ["இருந்தனர்", "இறந்தனர்", "were there", "died", "இருந்தனர்", "இறந்தனர்"], ["கழிவறை", "கலிவரை", "toilet", "not a word", "கழிவறை", "—"]];
function flips() {
  $("#dirty-cards").innerHTML = DIRTY.map(([f, b, x]) => `<button class="flip"><div class="in"><div class="f"><span class="x">${esc(x)}</span><span class="w">${esc(f)}</span><span class="x">↻</span></div><div class="b"><span class="g">${esc(L(b))}</span></div></div></button>`).join("");
  $("#homophones").innerHTML = HOMO.map(([a, b, ga, gb]) => `<button class="flip" style="height:116px"><div class="in"><div class="f"><span class="w">${a}</span><span class="x">${lang() === "en" ? ga : ""} ↻</span></div><div class="b"><span class="w" style="color:var(--bad)">${b}</span><span class="x">${lang() === "en" ? gb : ""}</span></div></div></button>`).join("");
  $$(".flip").forEach((f) => f.addEventListener("click", () => f.classList.toggle("on")));
}

/* ---------- scorer demo ---------- */
const FOLD = { ref: ["இன்னைக்கு", "ஆஃபீஸ்ல", "2", "மீட்டிங்"], hyp: ["இன்னைக்கு", "office-ல", "ரெண்டு", "meeting."],
  rules: [["punct", { en: "ignore punctuation", ta: "நிறுத்தக்குறி தவிர்" }], ["script", { en: "English in either script", ta: "ஆங்கிலம் எந்த எழுத்திலும்" }], ["num", { en: "2 = ரெண்டு", ta: "2 = ரெண்டு" }]] };
const foldOn = { punct: false, script: false, num: false };
function scorer() {
  $("#fold-btns").innerHTML = FOLD.rules.map(([k, l]) => `<button data-fold="${k}" aria-pressed="${foldOn[k]}">${esc(L(l))}</button>`).join("");
  $$("[data-fold]").forEach((b) => b.addEventListener("click", () => { foldOn[b.dataset.fold] = !foldOn[b.dataset.fold]; scorer(); }));
  const ok = FOLD.ref.map((r, i) => r === FOLD.hyp[i] || (i === 1 && foldOn.script) || (i === 2 && foldOn.num) || (i === 3 && foldOn.script && foldOn.punct));
  $("#fold-pair").innerHTML = `<span class="who">${L({ en: "label", ta: "லேபிள்" })}</span><span>${FOLD.ref.join(" ")}</span><span class="who">model</span><span>${FOLD.hyp.map((w, i) => (ok[i] ? esc(w) : `<span class="w-sub">${esc(w)}</span>`)).join(" ")}</span>`;
  const wer = (100 * ok.filter((x) => !x).length) / ok.length, b = $("#fold-wer"); b.textContent = wer.toFixed(0) + "%"; b.style.color = wer ? "var(--bad)" : "var(--good)";
}
let PERD = [], perI = 0;
function perDemo() {
  const host = $("#per-demo"); if (!host || !PERD.length) return;
  const d = PERD[perI];
  host.innerHTML = `<div class="controls">${PERD.map((p, i) => `<button data-per="${i}" aria-pressed="${i === perI}" class="ta">${esc(p.ref)} → ${esc(p.hyp)}</button>`).join("")}</div>
    <div style="display:flex;flex-wrap:wrap;gap:4px;margin:.6rem 0">${d.ops.map(([r, h, c]) => `<div style="text-align:center;min-width:36px;border-radius:8px;padding:.35rem .25rem;background:${c === 0 ? "var(--bg-2)" : c < 0.5 ? "var(--sub)" : "var(--del)"}" data-tip="${c ? L({ en: "cost", ta: "செலவு" }) + " " + c : "="}"><div class="ta" style="font-size:1.15rem">${esc(r || "∅")}</div><div class="ta" style="font-size:1.15rem;color:var(--fg-2)">${esc(h || "∅")}</div><div class="mono" style="font-size:.65rem;color:var(--fg-3);height:1em">${c || ""}</div></div>`).join("")}</div>
    <div class="meter"><b>${d.per}%</b><span class="dim">PER · ${esc(L(d.label))}</span></div>`;
  $$("[data-per]", host).forEach((b) => b.addEventListener("click", () => { perI = +b.dataset.per; perDemo(); })); tips(host);
}

/* ---------- speed race + ladder ---------- */
const LANES = [
  { n: { en: "RTX 3090, batched", ta: "RTX 3090, தொகுப்பாக" }, x: 646, c: "--ours" },
  { n: { en: "8 CPU threads, home server", ta: "8 CPU இழைகள், வீட்டு சர்வர்" }, x: 28, c: "--ours-2" },
  { n: { en: "a human typist", ta: "ஒரு மனிதத் தட்டச்சாளர்" }, x: 0.25, c: "--itcore" },
];
const fmtT = (s) => (s < 90 ? `${s.toFixed(s < 10 ? 1 : 0)} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`);
let raceRun = 0;
function race(animate) {
  const H = +$("#hours").value; $$(".hours-v").forEach((e) => (e.textContent = H));
  $("#race").innerHTML = LANES.map((l, i) => `<div class="lane"><span class="name">${esc(L(l.n))}</span><div class="tr"><i id="lane${i}" style="background:var(${l.c})"></i></div><span class="t">${fmtT((H * 3600) / l.x)}</span></div>`).join("");
  const id = ++raceRun;
  if (!animate) { LANES.forEach((l, i) => ($(`#lane${i}`).style.width = "100%")); return; }
  const t0 = performance.now(), D = 8000;
  const f = (t) => { if (id !== raceRun) return; const el = (t - t0) / D; LANES.forEach((l, i) => { const need = l.x === 0.25 ? 1 : Math.max(0.004, 0.25 / l.x); $(`#lane${i}`).style.width = Math.min(100, (100 * el) / need) + "%"; }); if (el < 1) requestAnimationFrame(f); };
  requestAnimationFrame(f);
}
const LADDER = [["baseline", 358], ["fast local attention, 1000 s micro-batches", 592], ["no activation checkpointing", 733], ["40 length buckets", 762], ["bf16 SDPA", 810], ["Triton banded attention", 838], ["60 buckets", 850], ["fast conv", 866], ["fused add + LayerNorm", 874], ["bias in row kernels", 893], ["Triton depthwise conv", 911]];
function ladder() {
  const W = 1000, rh = 26, P = { l: 300, r: 60, t: 30 }, H = P.t + LADDER.length * rh + 10, x = (v) => P.l + (v / 950) * (W - P.l - P.r);
  let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="training throughput"><text x="0" y="16" style="fill:var(--fg);font-weight:700">${L({ en: "training speed: audio hours per GPU-hour, RTX PRO 6000", ta: "பயிற்சி வேகம்: ஒரு GPU-மணிக்கு ஒலி மணிகள், RTX PRO 6000" })}</text>`;
  LADDER.forEach(([n, v], i) => { const y = P.t + i * rh; g += `<text x="${P.l - 10}" y="${y + 17}" text-anchor="end">${esc(n)}</text><rect x="${P.l}" y="${y + 4}" width="${x(v) - P.l}" height="${rh - 8}" rx="4" fill="var(--ours)" opacity="${0.45 + (0.55 * i) / (LADDER.length - 1)}" data-tip="${esc(n)}: ${v}"/><text x="${x(v) + 6}" y="${y + 17}" style="fill:var(--fg)">${v}</text>`; });
  $("#ladder").innerHTML = g + "</svg>"; tips($("#ladder"));
}

/* ---------- FLEURS gap ---------- */
const GAP = [[{ en: "near miss: one sound off", ta: "ஒரு ஒலி வேறுபாடு" }, 3.38, 35], [{ en: "a different word", ta: "வேறு சொல்" }, 1.98, 27], [{ en: "joined, split or suffix", ta: "சேர்த்தல், பிரித்தல், விகுதி" }, 1.67, 25],
  [{ en: "sandhi consonant (spelling rule)", ta: "புணர்ச்சி ஒற்று (எழுத்து விதி)" }, 1.19, 32], [{ en: "dropped word", ta: "விடுபட்ட சொல்" }, 1.13, 23], [{ en: "ல/ள/ழ, ன/ண/ந, ர/ற only", ta: "ல/ள/ழ, ன/ண/ந, ர/ற மட்டும்" }, 0.76, 27], [{ en: "English word, other script", ta: "ஆங்கிலச் சொல், வேறு எழுத்து" }, 0.47, 22]];
function gap() {
  const W = 1000, rh = 34, P = { l: 330, r: 160, t: 34 }, H = P.t + GAP.length * rh + 6, x = (v) => P.l + (v / 3.6) * (W - P.l - P.r);
  let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="FLEURS gap"><text x="0" y="16" style="fill:var(--fg);font-weight:700">${L({ en: "FLEURS: WER points we lose to indic-transcribe-core, by kind", ta: "FLEURS: indic-transcribe-core-இடம் இழக்கும் WER புள்ளிகள், வகைப்படி" })}</text>`;
  GAP.forEach(([n, v, lm], i) => { const y = P.t + i * rh; g += `<text x="${P.l - 10}" y="${y + 21}" text-anchor="end" class="lbl" style="font-weight:500">${esc(L(n))}</text><rect x="${P.l}" y="${y + 6}" width="${x(v) - P.l}" height="${rh - 12}" rx="4" fill="var(--itcore)" data-tip="${v} WER · LM ${lm}%"/><text x="${x(v) + 8}" y="${y + 21}" style="fill:var(--fg)">${v.toFixed(2)}</text><text x="${W}" y="${y + 21}" text-anchor="end">${L({ en: "LM fixes", ta: "LM சரிசெய்வது" })} ${lm}%</text>`; });
  $("#gap-chart").innerHTML = g + "</svg>"; tips($("#gap-chart"));
}

/* ---------- use tabs ---------- */
const CODE = {
  nemo: `# pip install nemo_toolkit[asr]\nimport nemo.collections.asr as nemo_asr\n\nm = nemo_asr.models.ASRModel.from_pretrained("shrirajh/parakeet-tamil-kili")  # coming soon\nm.change_attention_model("rel_pos_local_attn", [256, 256])\nprint(m.transcribe(["clip.wav"])[0].text)`,
  onnx: `# int8 ONNX encoder + joint, numpy greedy decoder, no PyTorch\n# about 28x real time on 8 CPU threads, 2.8 GB peak memory\n# export and runtime scripts: coming soon`,
  ha: `# Wyoming speech-to-text server for Home Assistant\n# CPU only, next to your voice pipeline\n# setup guide: coming soon`,
};
let tab = "nemo";
function use() { $("#use-code").textContent = CODE[tab]; $$("[data-tab]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.tab === tab))); }
$$("[data-tab]").forEach((b) => b.addEventListener("click", () => { tab = b.dataset.tab; use(); }));

/* ---------- boot ---------- */
(async () => {
  setLang(document.documentElement.lang || "en");
  const [clips, metrics, per, sc] = await Promise.all([J("data/clips.json"), J("data/metrics.json").catch(() => ({})), J("data/per_demo.json").catch(() => []), J("data/scores.json")]);
  SC = sc; PERD = per;
  hero(clips); quiz(clips, metrics); dumbbell(); epochs(); runs(); sim(); teacher(); flips(); scorer(); perDemo(); ladder(); gap(); use(); race(false);
  $("#hours").addEventListener("input", () => race(false)); $("#race-go").addEventListener("click", () => race(true));
  const cs = $(".counters"); cs.classList.add("reveal"); cs.addEventListener("shown", counters);
  observe();
  redraw.push(() => { dumbbell(); epochs(); runs(); teacher(); flips(); scorer(); perDemo(); ladder(); gap(); race(false); quiz(clips, metrics); });
})();
