// parakeet-tamil · தமிழ் கிளி. Vanilla JS, no dependencies; loads only ./data and ./audio.
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const lang = () => document.documentElement.lang;
const L = (o) => (o && typeof o === "object" ? o[lang()] ?? o.en : o);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
const J = async (p) => (await fetch(p, { cache: "no-cache" })).json();
let metric = "per", withLM = false;

/* ---------- language, theme, progress, reveal ---------- */
const redraw = [];
const NARROW = 760;
const isNarrow = () => innerWidth <= NARROW;
// width to draw a chart at: the desktop viewBox width, or on phones the host box's pixel width
const fitW = (host, desktopW, pad = 0) => (isNarrow() ? Math.max(280, Math.floor((host && host.clientWidth ? host.clientWidth : innerWidth - 56) - pad)) : desktopW);
const wrapText = (s, width, font = 13) => {
  const cw = font * (/[஀-௿]/.test(s) ? 0.7 : 0.64), max = Math.max(8, Math.floor(width / cw)), out = [];
  let line = "";
  for (const w of String(s).split(" ")) { if (line && (line + " " + w).length > max) { out.push(line); line = w; } else line = line ? line + " " + w : w; }
  if (line) out.push(line);
  return out;
};
const tspans = (lines, x, lh) => lines.map((l, i) => `<tspan x="${x}" dy="${i ? lh : 0}">${esc(l)}</tspan>`).join("");
const fitSvg = (W, H, label) => `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${label}"${isNarrow() ? ' class="fit"' : ""}>`;
function setLang(l) {
  document.documentElement.lang = l; localStorage.setItem("lang", l);
  $$("[data-set-lang]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.setLang === l)));
  redraw.forEach((f) => f());
}
$$("[data-set-lang]").forEach((b) => b.addEventListener("click", () => keepPlace(() => setLang(b.dataset.setLang))));
function setDepth(d) {
  document.documentElement.dataset.depth = d; localStorage.setItem("depth", d);
  $$("[data-set-depth]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.setDepth === d)));
  if (d === "simple") { metric = "per"; withLM = false; }
  redraw.forEach((f) => f());
}
$$("[data-set-depth]").forEach((b) => b.addEventListener("click", () => keepPlace(() => setDepth(b.dataset.setDepth))));
/* keep the reader's place when switching language or depth: anchor on the most specific block under a line
   30% down the screen, remember how far into it we are, and after the switch scroll so the same block (or its
   other-language twin, or its nearest surviving ancestor) sits at the same point */
function langPartner(el) {
  const other = el.dataset.l === "en" ? "ta" : "en", sib = [...el.parentElement.children];
  const mine = sib.filter((c) => c.dataset.l === el.dataset.l), theirs = sib.filter((c) => c.dataset.l === other);
  return theirs[mine.indexOf(el)] || null;
}
function keepPlace(change) {
  const probe = Math.round(innerHeight * 0.3), seen = (e) => e && e.getClientRects().length > 0;
  let best = null;
  for (const e of $$("header.hero, section, footer, section *, header.hero *")) {
    if (e.closest("svg, .bar") || !seen(e) || getComputedStyle(e).display.startsWith("inline")) continue;
    const r = e.getBoundingClientRect();
    if (r.top <= probe && r.bottom > probe && (!best || r.height <= best.r.height)) best = { e, r };
  }
  if (!best) return change();
  const chain = [];
  for (let e = best.e; e && e !== document.body; e = e.parentElement) { const r = e.getBoundingClientRect(); chain.push({ e, f: (probe - r.top) / (r.height || 1) }); }
  const lang0 = document.documentElement.lang;
  change();
  const swapped = document.documentElement.lang !== lang0;
  const place = () => {
    for (const { e, f } of chain) {
      let m = e;
      if (swapped) { const l = e.closest("[data-l]"); if (l) { if (l !== e) continue; m = langPartner(l); if (!m) continue; } }
      if (!seen(m)) continue;
      const r = m.getBoundingClientRect(), d = r.top + f * r.height - probe;
      if (Math.abs(d) > 1) window.scrollTo({ top: scrollY + d, behavior: "instant" });
      return;
    }
  };
  place(); requestAnimationFrame(place);
}
$("#theme").addEventListener("click", () => {
  const flip = () => {
    const t = document.documentElement.dataset.theme === "light" ? "" : "light";
    if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
    localStorage.setItem("theme", t); redraw.forEach((f) => f());
  };
  // cross-fade the whole page (text, charts, canvases) between themes; instant where unsupported or reduced motion
  if (!document.startViewTransition || matchMedia("(prefers-reduced-motion: reduce)").matches) return flip();
  document.startViewTransition(flip);
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
  return { paint, audio };
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

/* ---------- hero: pick a recording ---------- */
let heroClips = [], heroI = 0, heroAudio = null;
function hero(clips) {
  heroClips = clips.filter((x) => x.group === "hero");
  $("#hero-pick").innerHTML = heroClips.map((c, i) => `<button data-hero="${i}" aria-pressed="${i === heroI}">${esc(L(c.label))}</button>`).join("");
  $$("[data-hero]").forEach((b) => b.addEventListener("click", () => { heroI = +b.dataset.hero; heroLoad(true); }));
  heroLoad(false);
}
function heroLoad(autoplay) {
  const c = heroClips[heroI]; if (!c) return;
  $$("[data-hero]").forEach((b) => b.setAttribute("aria-pressed", String(+b.dataset.hero === heroI)));
  if (heroAudio) heroAudio.pause();
  const box = $("#hero-words");
  box.innerHTML = c.words.map(([w, t]) => `<span data-t="${t}" class="${latin(w) ? "lat" : ""}">${esc(w)}</span>`).join(" ");
  const ws = $$("span", box), btn = $("#hero-play"), fresh = btn.cloneNode(false); fresh.textContent = "▶"; fresh.setAttribute("aria-pressed", "false"); btn.replaceWith(fresh);
  const w = $("#hero-wave"), w2 = w.cloneNode(false), cv2 = document.createElement("canvas"); w2.append(cv2); w.replaceWith(w2);
  const r = wirePlayer({ url: c.audio, button: fresh, canvas: cv2, onTime: (t) => ws.forEach((x) => x.classList.toggle("on", +x.dataset.t <= t + 0.05)) });
  heroAudio = r.audio; r.paint(); numbersStrip(c.audio);
  if (autoplay) fresh.click();
}
// The HTML holds the real values (crawlers, no-JS readers, link previews); with JS they count up once when seen.
function counters() {
  if (reduce) return;
  const run = (el) => {
    const to = +el.dataset.count, dec = +el.dataset.dec, suf = el.dataset.suffix || "", t0 = performance.now(), dur = 1400;
    const f = (t) => { const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3); el.textContent = (to * e).toFixed(dec) + suf; if (k < 1) requestAnimationFrame(f); };
    requestAnimationFrame(f);
  };
  const ob = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { run(e.target); ob.unobserve(e.target); } }), { threshold: 0.6 });
  $$("[data-count]").forEach((el) => ob.observe(el));
}
counters();

/* ---------- quiz ---------- */
const score = { right: 0, done: 0 };
function quiz(clips, metrics) {
  $$("[data-quiz]").forEach((host) => {
    host.innerHTML = "";
    clips.filter((c) => c.group === host.dataset.quiz).forEach((c, k) => {
      const opts = [["ours", c.ours], ["itcore", c.itcore]]; if (k % 2) opts.reverse();
      const right = c.answer || "ours", M = metrics[c.id] || {};
      const card = document.createElement("div"); card.className = "qcard reveal in";
      const mline = (w) => (M[w] ? `<div class="small mono mline" data-d="d" style="display:none;margin-top:.3rem">${L({ en: "vs label", ta: "எழுத்துப்படியுடன் ஒப்பிட" })}: WER ${M[w].wer_fold ?? M[w].wer} · PER ${M[w].per_fold ?? "–"} · EER ${M[w].eer}</div>` : "");
      card.innerHTML = `<div class="qhead"><button class="qplay" aria-pressed="false" aria-label="play">▶</button><div class="mini"><canvas></canvas></div><span class="qsrc">${esc(c.source)}</span></div>
        <div class="opts">${opts.map(([w, t]) => `<button class="opt ${w}" data-w="${w}"><span class="who">${w === "ours" ? "parakeet-tamil" : "indic-transcribe-core"}</span><span class="txt" data-raw="${esc(t)}">${esc(t)}</span>${mline(w)}</button>`).join("")}
          <div class="opt ref" style="display:none"><span class="who">${L({ en: "label", ta: "எழுத்துப்படி" })}</span>${esc(c.ref)}</div></div>
        <div class="verdict">${esc(L(c.note))}</div>`;
      wirePlayer({ url: c.audio, button: $(".qplay", card), canvas: $("canvas", card) });
      $$("button.opt", card).forEach((b) => b.addEventListener("click", () => {
        if (card.classList.contains("done")) return;
        card.classList.add("done"); score.done++; if (b.dataset.w === right) score.right++;
        $(`.opt.${right}`, card).classList.add("right");
        $(".opt.ref", card).style.display = "block";
        $$(".mline", card).forEach((x) => (x.style.display = "block"));
        $$("button.opt .txt", card).forEach((t) => (t.innerHTML = diffHTML(c.ref, t.dataset.raw)));
        $$("#score-line").forEach((sl) => (sl.textContent = L({ en: `You: ${score.right} of ${score.done}`, ta: `உங்கள் மதிப்பெண்: ${score.done}-க்கு ${score.right}` })));
      }));
      host.append(card);
    });
  });
}

/* ---------- scoreboard dumbbell ---------- */
let SC;
function dumbbell() {
  if (!SC) return;
  const host = $("#dumbbell"), S = SC.scores;
  const sets = Object.keys(SC.sets).filter((s) => metric === "wer" || SC.sets[s].kind !== "english");
  const vals = sets.flatMap((s) => ["e5", "itcore", "e5_lm"].map((m) => S[m]?.[s]?.[metric]).filter((v) => v != null));
  const nar = isNarrow(), W = nar ? fitW(host, 1000) : 1000, rowH = nar ? 50 : 44, GAPH = 30, P = nar ? { l: 8, r: 64, t: 20 } : { l: 220, r: 210, t: 20 };
  const kinds = [...new Set(sets.map((s) => SC.sets[s].kind))], H = P.t + sets.length * rowH + kinds.length * GAPH + 30;
  const mx = Math.max(...vals), stp = mx > 30 ? 10 : mx > 12 ? 5 : mx > 6 ? 2 : mx > 3 ? 1 : 0.5, lo = 0, hi = Math.ceil((mx * 1.04) / stp) * stp, x = (v) => P.l + ((v - lo) / (hi - lo)) * (W - P.l - P.r);
  const kindLbl = { spoken: { en: "spoken Tamil", ta: "பேச்சுத் தமிழ்" }, read: { en: "read Tamil", ta: "வாசித்த தமிழ்" }, english: { en: "English", ta: "ஆங்கிலம்" } };
  let g = fitSvg(W, H, "scoreboard");
  for (let v = 0; v <= hi + 1e-9; v += stp) { g += `<g class="grid"><line x1="${x(v)}" x2="${x(v)}" y1="${P.t}" y2="${H - 24}"/></g><text x="${x(v)}" y="${H - 6}" text-anchor="middle">${+v.toFixed(1)}</text>`; }
  let last = null, yy = P.t;
  sets.forEach((s) => {
    const k = SC.sets[s].kind;
    if (k !== last) { yy += GAPH; g += `<text x="0" y="${yy - 8}" style="fill:var(--fg-3);font-size:10px;${lang() === "en" ? "letter-spacing:.08em" : ""}">${esc(L(kindLbl[k]).toUpperCase())}</text>`; last = k; }
    const y = yy + rowH / 2 + (nar ? 8 : 0); yy += rowH;
    const o = S.e5[s][metric], it = S.itcore[s]?.[metric], lm = S.e5_lm[s]?.[metric];
    const ours = withLM && lm != null ? lm : o;
    g += nar ? `<text class="lbl" x="0" y="${y - 14}" style="font-size:12px">${esc(L(SC.sets[s]))}</text>` : `<text class="lbl" x="0" y="${y + 5}">${esc(L(SC.sets[s]))}</text>`;
    if (it != null) g += `<line x1="${x(ours)}" x2="${x(it)}" y1="${y}" y2="${y}" stroke="var(--fg-3)" stroke-width="2" opacity=".45"/>`;
    if (withLM && lm != null) g += `<circle cx="${x(o)}" cy="${y}" r="7" fill="none" stroke="var(--ours)" stroke-width="2" opacity=".55" data-tip="parakeet-tamil · ${o.toFixed(2)}"/>`;
    if (it == null) g += `<circle cx="${x(ours)}" cy="${y}" r="8" fill="var(--ours)" stroke="var(--bg-2)" stroke-width="2" data-tip="parakeet-tamil${withLM ? " + LM" : ""} · ${ours.toFixed(2)}"/>`;
    if (it != null) {
      g += `<circle cx="${x(it)}" cy="${y}" r="9" fill="var(--itcore)" stroke="var(--bg-2)" stroke-width="2" data-tip="indic-transcribe-core · ${it.toFixed(2)}"/>`;
    g += `<circle cx="${x(ours)}" cy="${y}" r="8" fill="var(--ours)" stroke="var(--bg-2)" stroke-width="2" data-tip="parakeet-tamil${withLM ? " + LM" : ""} · ${ours.toFixed(2)}"/>`;
      const d = ours - it, win = d < 0;
      g += `<text x="${nar ? W - 2 : W - P.r + 18}" y="${y + 5}" ${nar ? 'text-anchor="end"' : ""} style="fill:var(--fg);font-weight:700;font-size:14px">${win ? "▲" : "▼"} ${Math.abs(d).toFixed(metric === "wer" || metric === "cer" ? 1 : 2)}</text>`;
      if (!nar) g += `<text x="${W - 2}" y="${y + 5}" text-anchor="end" style="font-size:11px">${win ? L({ en: "we lead", ta: "நாங்கள் முன்னால்" }) : L({ en: "behind", ta: "நாங்கள் பின்னால்" })}</text>`;
    }
  });
  host.innerHTML = g + "</svg>"; tips(host);
  $("#dumb-legend").innerHTML = [["--ours", "parakeet-tamil" + (withLM ? " + LM" : "")], ["--itcore", "indic-transcribe-core"]].map(([c, n]) => `<span class="key"><span class="dot" style="background:var(${c})"></span>${n}</span>`).join("") + `<span class="key dim">${L({ en: "lower is better", ta: "குறைவாக இருந்தால் நல்லது" })} · ${metric.toUpperCase()} %${metric === "per" || metric === "eer" ? L({ en: " · English tests appear under WER", ta: " · ஆங்கிலத் தேர்வுகளை WER-இல் பாருங்கள்" }) : ""}</span>`;
}
$$("[data-metric]").forEach((b) => b.addEventListener("click", () => { metric = b.dataset.metric; $$("[data-metric]").forEach((x) => x.setAttribute("aria-pressed", String(x === b))); dumbbell(); openTable(); }));
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
  { k: "stock", d: "", s: 16.5, t: { en: "stock parakeet v3", ta: "அசல் parakeet v3" }, p: { en: "No Tamil: Kathbath WER 117%.", ta: "தமிழ் தெரியாது: Kathbath WER 117%." } },
  { k: "run 1", d: "9 Nov", s: 43.5, t: { en: "Tamil in Latin letters", ta: "ஆங்கில எழுத்தில் தமிழ்" }, p: { en: "Tamil improved. English came out spelled like Tamil: \"Chuus tistrikt… veekchineeshan\".", ta: "தமிழ் முன்னேறியது. ஆனால் ஆங்கிலம், தமிழ் உச்சரிப்பில் எழுதப்பட்டு வந்தது: \"Chuus tistrikt… veekchineeshan\"." } },
  { k: "run 2", d: "10 Nov", s: 29.3, t: { en: "2,044 Tamil tokens", ta: "2,044 தமிழ் token-கள்" }, p: { en: "Native Tamil script. Before training, identical to the stock model.", ta: "தமிழ் எழுத்திலேயே. பயிற்சி தொடங்கும் முன், அசல் மாதிரியும் இதுவும் ஒன்றே." } },
  { k: "run 3", d: "10 Nov", s: 30.5, t: { en: "extra heads", ta: "கூடுதல் பகுதிகள் (heads)" }, p: { en: "Phoneme and language-ID heads, English distillation.", ta: "ஒலியன்களையும், பேசும் மொழியையும் கண்டறியும் கூடுதல் பகுதிகள். அசல் மாதிரியிடமிருந்து ஆங்கிலத்தைக் கற்றல் (distillation)." } },
  { k: "run 4", d: "10 Nov", s: 45.4, t: { en: "synthetic Tanglish", ta: "கணினிக் குரலில் Tanglish" }, p: { en: "Read Tamil up, English down again.", ta: "வாசித்த தமிழ் மேம்பட்டது. ஆங்கிலம் மீண்டும் சரிந்தது." } },
  { k: "run 5", d: "11 Nov", s: 12.4, t: { en: "one spelling rule", ta: "ஒரே ஒரு எழுத்து விதி" }, p: { en: "English loanwords in English letters (meeting-க்கு). English came back.", ta: "ஆங்கிலக் கடன் சொற்கள் ஆங்கில எழுத்தில் (meeting-க்கு). ஆங்கிலம் மீண்டு வந்தது." } },
  { k: "run 6", d: "11 Nov", s: 13.5, t: { en: "noise and phones", ta: "பின்னணிச் சத்தம், தொலைபேசி" }, p: { en: "Noise, reverb, phone codecs, babble, long audio.", ta: "பின்னணிச் சத்தம், எதிரொலி, தொலைபேசி ஒலித்தரம், பலர் பேசும் சத்தம், நீண்ட பதிவுகள்." } },
  { k: "run 7", d: "11 Nov", s: 11.5, t: { en: "evicted at 91%", ta: "91%-இல் கணினி பறிபோனது" }, p: { en: "1,400 h of world English. Finished on a single GPU after the eviction.", ta: "உலகெங்கிலுமிருந்து 1,400 மணி நேர ஆங்கிலம். கணினி பறிபோன பிறகு, ஒரே GPU-வில் பயிற்சி முடிந்தது." } },
  { k: "run 8", d: "12 Nov", s: 12.07, t: { en: "punctuation", ta: "நிறுத்தக்குறிகள்" }, p: { en: "Gemma punctuation, 30,895 bad clips dropped.", ta: "Gemma இட்ட நிறுத்தக்குறிகள். 30,895 மோசமான பதிவுகள் நீக்கப்பட்டன." } },
  { k: "run 9", d: "13 Nov", s: null, t: { en: "formal Tamil targets", ta: "எழுத்துத் தமிழ் இலக்கு" }, p: { en: "Spoken Tamil got worse. Dropped.", ta: "பேச்சுத் தமிழில் தவறுகள் கூடின. கைவிட்டோம்." } },
  { k: "run 10", d: "14 Nov", s: 10.67, ship: 1, t: { en: "from scratch", ta: "மீண்டும் முதலிலிருந்து" }, p: { en: "18,746 audio hours per 3 epochs, about 3 h on 8 GPUs. Extended to 6. Epoch 5 ships.", ta: "3 சுற்றுகளுக்கு 18,746 மணி நேர ஒலி; 8 GPU-களில் சுமார் 3 மணி நேரம். பிறகு 6 சுற்றுகள் வரை நீட்டினோம். 5-ஆவது சுற்றுதான் வெளியிடப்படுகிறது." } },
];
function runs() {
  const pts = RUNS.map((r, i) => [i, r.s]).filter(([, s]) => s != null);
  const W = 1000, H = 300, P = { l: 40, r: 20, t: 20, b: 40 }, hi = 50, x = (i) => P.l + (i * (W - P.l - P.r)) / (RUNS.length - 1), y = (v) => P.t + ((hi - v) * (H - P.t - P.b)) / hi;
  let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Svarah WER by run">`;
  [0, 10, 20, 30, 40, 50].forEach((v) => (g += `<g class="grid"><line x1="${P.l}" x2="${W - P.r}" y1="${y(v)}" y2="${y(v)}"/></g><text x="${P.l - 8}" y="${y(v) + 4}" text-anchor="end">${v}</text>`));
  g += `<rect x="${P.l}" y="${y(hi)}" width="${W - P.l - P.r}" height="${y(17) - y(hi)}" fill="var(--bad)" opacity=".08"/><line x1="${P.l}" x2="${W - P.r}" y1="${y(17)}" y2="${y(17)}" stroke="var(--bad)" stroke-dasharray="6 5" stroke-width="2"/><text x="${W - P.r}" y="${y(17) - 8}" text-anchor="end" style="fill:var(--fg)">${L({ en: "English gate: 17", ta: "ஆங்கில எல்லைக்கோடு: 17" })}</text>`;
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
    const lost = step - ck; set("#sim-vm", "dead", L({ en: "gone", ta: "போய்விட்டது" })); log.textContent = "";
    const T = [[0, "t+0 s    the cloud takes the VM back"], [700, "t+30 s   supervisor: API says 404, VM is gone"], [1300, "t+34 s   launch 8 × RTX PRO 6000 on the kept disk"],
      [2100, "t+74 s   booted, bootstrap starts"], [2800, `t+92 s   resume from checkpoint ${ck.toLocaleString()} (${lost} steps to redo)`], [3400, "t+105 s  training again"]];
    set("#sim-sup", "busy", L({ en: "relaunching", ta: "மீண்டும் தொடங்குகிறது" }));
    T.forEach(([ms, t]) => setTimeout(() => say(t), reduce ? 0 : ms));
    setTimeout(() => { step = ck; set("#sim-vm", "busy", L({ en: "booting", ta: "இயங்கத் தொடங்குகிறது" })); }, reduce ? 0 : 2100);
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
  b.setAttribute("aria-pressed", String(punctOn)); b.textContent = L({ en: `Gemma: ${punctOn ? "on" : "off"}`, ta: `Gemma: ${punctOn ? "இயக்கத்தில்" : "அணைந்துள்ளது"}` });
}
$("#punct-toggle").addEventListener("click", () => { punctOn = !punctOn; teacher(); });

/* ---------- flip cards ---------- */
const DIRTY = [
  ["sir-ஆ", { en: "Our own loanword converter turned \"சார் நா\" (sir, I…) into \"sir-ஆ\". 670 times. Fixed before run 10.", ta: "ஆங்கிலச் சொற்களை மாற்ற நாங்களே எழுதிய கருவி, \"சார் நா\" (சார், நான்…) என்பதை \"sir-ஆ\" ஆக்கியது. 670 முறை. 10-ஆவது முயற்சிக்கு முன் சரிசெய்தோம்." }, "×670"],
  ["police சாமியார்", { en: "போலிச் சாமியார் means \"fake priest\". The same converter made him a policeman.", ta: "போலிச் சாமியார் என்றால் பொய்யான சாமியார். அதே கருவி அவரைப் போலீஸ்காரர் ஆக்கிவிட்டது." }, "our bug"],
  ["touch-உ", { en: "டச்சு (Dutch) became touch-உ. It also turned up inside a Tirumantiram verse.", ta: "டச்சு என்பது touch-உ ஆனது. திருமந்திரப் பாடல் ஒன்றுக்குள்ளும் இது புகுந்துவிட்டது." }, "our bug"],
  ["veekchineeshan", { en: "Run 1 learned to write English the Tamil way: \"Choose district… vaccination\" came out as \"Chuus tistrikt… veekchineeshan\".", ta: "முதல் முயற்சி ஆங்கிலத்தைத் தமிழ் உச்சரிப்பில் எழுதப் பழகியது: \"Choose district… vaccination\" என்பது \"Chuus tistrikt… veekchineeshan\" என்று வந்தது." }, "run 1"],
  ["56.7%", { en: "of one SPRING-INX release duplicates the previous one: 29,482 clips, 67 h.", ta: "ஒரு SPRING-INX வெளியீட்டில் 56.7%, அதற்கு முந்தைய வெளியீட்டின் பிரதிதான்: 29,482 பதிவுகள், 67 மணி நேரம்." }, "duplicates"],
  ["ஏலு", { en: "IndicVoices read labels write ழ as ல: ஏலு for ஏழு (7), about 15 times per 100 labels.", ta: "IndicVoices-இன் வாசிப்புப் பதிவுகளுக்கான எழுத்துப்படிகளில் ழ-வை ல என்று எழுதியிருக்கிறார்கள்: ஏழு (7) என்பதற்கு ஏலு. 100 எழுத்துப்படிகளுக்குச் சுமார் 15 முறை." }, "label"],
  ["6.4", { en: "errors per 100 words in Shrutilipi labels, the sloppiest source. A third of its labels have one. How we counted: below.", ta: "Shrutilipi எழுத்துப்படிகளில் 100 சொற்களுக்கு 6.4 பிழைகள். எங்கள் மூலங்களிலேயே மிக அலட்சியமானது இதுதான். அதன் எழுத்துப்படிகளில் மூன்றில் ஒன்றில் பிழை இருக்கிறது. எப்படி எண்ணினோம் என்பது கீழே." }, "per 100 words"],
  ["\"okay mam\"", { en: "The whole text for 13 seconds of fluent Tamil speech. 30,895 clips like it were dropped.", ta: "13 வினாடி சரளமான தமிழ்ப் பேச்சுக்கு, எழுத்துப்படியில் இருந்தது இது மட்டும்தான். இது போன்ற 30,895 பதிவுகளை நீக்கினோம்." }, "label"],
];
const HOMO = [["மழை", "மலை", "rain", "hill", "மழை", "மலை"], ["வலி", "வழி", "pain", "way", "வலி", "வழி"], ["கொல்ல", "கொள்ள", "kill", "take", "கொல்", "கொள்"], ["அழிக்க", "அளிக்க", "destroy", "give", "அழி", "அளி"], ["இருந்தனர்", "இறந்தனர்", "were there", "died", "இருந்தனர்", "இறந்தனர்"], ["கழிவறை", "கலிவரை", "toilet", "not a word", "கழிவறை", "—"]];
function flips() {
  $("#dirty-cards").innerHTML = DIRTY.map(([f, b, x]) => `<button class="flip"><div class="in"><div class="f"><span class="x">${esc(x)}</span><span class="w">${esc(f)}</span><span class="x">↻</span></div><div class="b"><span class="g">${esc(L(b))}</span></div></div></button>`).join("");
  $("#homophones").innerHTML = HOMO.map(([a, b, ga, gb]) => `<button class="flip" style="height:116px"><div class="in"><div class="f"><span class="w">${a}</span><span class="x">${lang() === "en" ? ga : ""} ↻</span></div><div class="b"><span class="w" style="color:var(--bad)">${b}</span><span class="x">${lang() === "en" ? gb : ""}</span></div></div></button>`).join("");
}

/* ---------- scorer demo ---------- */
const FOLD = { ref: ["இன்னைக்கு", "ஆஃபீஸ்ல", "2", "மீட்டிங்"], hyp: ["இன்னைக்கு", "office-ல", "ரெண்டு", "meeting."],
  rules: [["punct", { en: "ignore punctuation", ta: "நிறுத்தக்குறிகளைப் பொருட்படுத்த வேண்டாம்" }], ["script", { en: "English in either script", ta: "ஆங்கிலம் எந்த எழுத்திலும் சரி" }], ["num", { en: "2 = ரெண்டு", ta: "2 = ரெண்டு" }]] };
const foldOn = { punct: false, script: false, num: false };
function scorer() {
  $("#fold-btns").innerHTML = FOLD.rules.map(([k, l]) => `<button data-fold="${k}" aria-pressed="${foldOn[k]}">${esc(L(l))}</button>`).join("");
  $$("[data-fold]").forEach((b) => b.addEventListener("click", () => { foldOn[b.dataset.fold] = !foldOn[b.dataset.fold]; scorer(); }));
  const ok = FOLD.ref.map((r, i) => r === FOLD.hyp[i] || (i === 1 && foldOn.script) || (i === 2 && foldOn.num) || (i === 3 && foldOn.script && foldOn.punct));
  $("#fold-pair").innerHTML = `<span class="who">${L({ en: "label", ta: "எழுத்துப்படி" })}</span><span>${FOLD.ref.join(" ")}</span><span class="who">model</span><span>${FOLD.hyp.map((w, i) => (ok[i] ? esc(w) : `<span class="w-sub">${esc(w)}</span>`)).join(" ")}</span>`;
  const wer = (100 * ok.filter((x) => !x).length) / ok.length, b = $("#fold-wer"); b.textContent = wer.toFixed(0) + "%"; b.style.color = wer ? "var(--bad)" : "var(--good)";
}
let PERD = [], perI = 0;
function perDemo() {
  const host = $("#per-demo"); if (!host || !PERD.length) return;
  const d = PERD[perI];
  host.innerHTML = `<div class="controls">${PERD.map((p, i) => `<button data-per="${i}" aria-pressed="${i === perI}" class="ta">${esc(p.ref)} → ${esc(p.hyp)}</button>`).join("")}</div>
    <div style="display:flex;flex-wrap:wrap;gap:4px;margin:.6rem 0">${d.ops.map(([r, h, c]) => `<div style="text-align:center;min-width:36px;border-radius:8px;padding:.35rem .25rem;background:${c === 0 ? "var(--bg-2)" : c < 0.5 ? "var(--sub)" : "var(--del)"}" data-tip="${c ? L({ en: "cost", ta: "தவறுப் புள்ளி" }) + " " + c : "="}"><div class="ta" style="font-size:1.15rem">${esc(r || "∅")}</div><div class="ta" style="font-size:1.15rem;color:var(--fg-2)">${esc(h || "∅")}</div><div class="mono" style="font-size:.65rem;color:var(--fg-3);height:1em">${c || ""}</div></div>`).join("")}</div>
    <div class="meter"><b>${d.per}%</b><span class="dim">PER · ${esc(L(d.label))}</span></div>`;
  $$("[data-per]", host).forEach((b) => b.addEventListener("click", () => { perI = +b.dataset.per; perDemo(); })); tips(host);
}

/* ---------- speed race: real time by default ---------- */
const LANES = [
  { n: { en: "gaming graphics card (RTX 3090)", ta: "கேமிங் graphics card (RTX 3090)" }, x: 646, c: "--ours" },
  { n: { en: "ordinary home computer, no graphics card", ta: "சாதாரண வீட்டுக் கணினி, graphics card இல்லாமல்" }, x: 28, c: "--ours-2" },
  { n: { en: "a fast human typist", ta: "வேகமாகத் தட்டச்சு செய்யும் ஒருவர்" }, x: 0.25, c: "--itcore" },
];
const fmtT = (s) => (s < 90 ? `${s.toFixed(s < 10 ? 1 : 0)} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`);
const clock = (s) => { s = Math.floor(s); return `${Math.floor(s / 3600)}:${String(Math.floor(s / 60) % 60).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`; };
let raceSpeed = 1, raceT = 0, raceOn = false, raceLast = 0;
function raceDraw() {
  const H = +$("#hours").value; $$(".hours-v").forEach((e) => (e.textContent = H));
  if (!$("#lane0")) $("#race").innerHTML = LANES.map((l, i) => `<div class="lane"><span class="name">${esc(L(l.n))}</span><div class="tr"><i id="lane${i}" style="background:var(${l.c})"></i></div><span class="t" id="lt${i}"></span></div>`).join("");
  LANES.forEach((l, i) => { const need = (H * 3600) / l.x, f = Math.min(1, raceT / need); $(`#lane${i}`).style.width = 100 * f + "%"; $(`#lane${i}`).classList.toggle("done", f >= 1); $(`#lt${i}`).textContent = (f >= 1 ? "✓ " : "") + fmtT(need); });
  $("#race-clock").textContent = clock(raceT);
}
function raceTick(t) {
  if (!raceOn) return;
  raceT += ((t - raceLast) / 1000) * raceSpeed; raceLast = t; raceDraw();
  const H = +$("#hours").value;
  if (raceT >= (H * 3600) / LANES[LANES.length - 1].x) { raceOn = false; $("#race-go").textContent = "↺"; } else requestAnimationFrame(raceTick);
}
function raceInit() {
  $("#race-go").addEventListener("click", () => {
    if (raceOn) { raceOn = false; $("#race-go").textContent = "▶"; return; }
    const H = +$("#hours").value; if (raceT >= (H * 3600) / LANES[LANES.length - 1].x) raceT = 0;
    raceOn = true; raceLast = performance.now(); $("#race-go").textContent = "❚❚"; requestAnimationFrame(raceTick);
  });
  $$("[data-speed]").forEach((b) => b.addEventListener("click", () => { raceSpeed = +b.dataset.speed; $$("[data-speed]").forEach((x) => x.setAttribute("aria-pressed", String(x === b))); }));
  $("#hours").addEventListener("input", () => { raceT = 0; raceDraw(); });
  raceDraw();
}
const LADDER = [["baseline", 358], ["local attention, 1000 s batches", 592], ["no activation checkpointing", 733], ["40 length buckets", 762], ["bf16 SDPA", 810], ["Triton banded attention", 838], ["60 buckets", 850], ["fast conv", 866], ["fused add + LayerNorm", 874], ["bias in row kernels", 893], ["Triton depthwise conv", 911]];
function ladder() {
  if (isNarrow()) return hbarsNarrow($("#ladder"), LADDER.map(([n, v], i) => [n, v, "--ours", `${v}`, 0.45 + (0.55 * i) / (LADDER.length - 1)]), 950,
    "training throughput", L({ en: "training speed: audio hours per GPU-hour, RTX PRO 6000", ta: "பயிற்சி வேகம்: ஒரு GPU-மணியில் கற்கும் ஒலி மணிகள், RTX PRO 6000" }));
  const W = 1000, rh = 26, P = { l: 300, r: 60, t: 30 }, H = P.t + LADDER.length * rh + 10, x = (v) => P.l + (v / 950) * (W - P.l - P.r);
  let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="training throughput"><text x="0" y="16" style="fill:var(--fg);font-weight:700">${L({ en: "training speed: audio hours per GPU-hour, RTX PRO 6000", ta: "பயிற்சி வேகம்: ஒரு GPU ஒரு மணி நேரத்தில் கற்கும் ஒலி மணிகள், RTX PRO 6000" })}</text>`;
  LADDER.forEach(([n, v], i) => { const y = P.t + i * rh; g += `<text x="${P.l - 10}" y="${y + 17}" text-anchor="end">${esc(n)}</text><rect x="${P.l}" y="${y + 4}" width="${x(v) - P.l}" height="${rh - 8}" rx="4" fill="var(--ours)" opacity="${0.45 + (0.55 * i) / (LADDER.length - 1)}" data-tip="${esc(n)}: ${v}"/><text x="${x(v) + 6}" y="${y + 17}" style="fill:var(--fg)">${v}</text>`; });
  $("#ladder").innerHTML = g + "</svg>"; tips($("#ladder"));
}

/* ---------- FLEURS gap ---------- */
const GAP = [[{ en: "near miss: one sound off", ta: "ஒரே ஒரு ஒலி மாறியது" }, 3.38, 35], [{ en: "a different word", ta: "வேறு சொல்" }, 1.98, 27], [{ en: "joined, split or suffix", ta: "சேர்த்து/பிரித்து எழுதியது, விகுதி" }, 1.67, 25],
  [{ en: "sandhi consonant (spelling rule)", ta: "புணர்ச்சி ஒற்று (எழுத்து விதி)" }, 1.19, 32], [{ en: "dropped word", ta: "விடுபட்ட சொல்" }, 1.13, 23], [{ en: "ல/ள/ழ, ன/ண/ந, ர/ற only", ta: "ல/ள/ழ, ன/ண/ந, ர/ற மட்டும்" }, 0.76, 27], [{ en: "English word, other script", ta: "ஆங்கிலச் சொல், வேறு எழுத்தில்" }, 0.47, 22]];
function gap() {
  if (isNarrow()) return hbarsNarrow($("#gap-chart"), GAP.map(([n, v, lm]) => [L(n), v, "--itcore", `${v} · LM ${lm}%`]), 3.6,
    "FLEURS gap", L({ en: "FLEURS: WER points we lose to indic-transcribe-core, by kind", ta: "FLEURS: indic-transcribe-core-ஐ விட நாங்கள் பின்தங்கும் WER புள்ளிகள், வகைவாரியாக" }));
  const W = 1000, rh = 34, P = { l: 330, r: 160, t: 34 }, H = P.t + GAP.length * rh + 6, x = (v) => P.l + (v / 3.6) * (W - P.l - P.r);
  let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="FLEURS gap"><text x="0" y="16" style="fill:var(--fg);font-weight:700">${L({ en: "FLEURS: WER points we lose to indic-transcribe-core, by kind", ta: "FLEURS: indic-transcribe-core-ஐ விட நாங்கள் பின்தங்கும் WER புள்ளிகள், வகை வாரியாக" })}</text>`;
  GAP.forEach(([n, v, lm], i) => { const y = P.t + i * rh; g += `<text x="${P.l - 10}" y="${y + 21}" text-anchor="end" class="lbl" style="font-weight:500">${esc(L(n))}</text><rect x="${P.l}" y="${y + 6}" width="${x(v) - P.l}" height="${rh - 12}" rx="4" fill="var(--itcore)" data-tip="${v} WER · LM ${lm}%"/><text x="${x(v) + 8}" y="${y + 21}" style="fill:var(--fg)">${v.toFixed(2)}</text><text x="${W}" y="${y + 21}" text-anchor="end">${L({ en: "LM fixes", ta: "LM சரிசெய்வது" })} ${lm}%</text>`; });
  $("#gap-chart").innerHTML = g + "</svg>"; tips($("#gap-chart"));
}

/* ---------- use tabs ---------- */
const CODE = {
  nemo: `# git clone https://github.com/shrirajh/parakeet-tamil-kili && cd parakeet-tamil-kili && uv sync --extra nemo
from kili.infer.transcribe import load_model

m = load_model("shrirajh/parakeet-tamil-kili")  # local attention [256, 256], as trained
print(m.transcribe(["clip.wav"])[0].text)`,
  onnx: `# int8 ONNX encoder + numpy decoder, no PyTorch: about 28x real time on 8 CPU threads
uv sync --extra onnx
uvx hf download shrirajh/parakeet-tamil-kili --include "onnx/*" --local-dir kili
uv run python -m kili.infer.onnx_run --dir kili/onnx clip.wav`,
  ha: `# Wyoming speech-to-text server for Home Assistant, CPU only
uv sync --extra wyoming
uv run python -m kili.infer.wyoming_server --dir kili/onnx --uri tcp://0.0.0.0:10300
# Home Assistant: Settings > Devices & services > Add integration > Wyoming Protocol, this host, port 10300`,
};
let tab = "nemo";
function use() { $("#use-code").textContent = CODE[tab]; $$("[data-tab]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.tab === tab))); }
$$("[data-tab]").forEach((b) => b.addEventListener("click", () => { tab = b.dataset.tab; use(); }));

/* ---------- boot ---------- */
(async () => {
  setLang(document.documentElement.lang || "en");
  $$("[data-set-depth]").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.setDepth === (document.documentElement.dataset.depth || "simple"))));
  const [clips, metrics, per, sc] = await Promise.all([J("data/clips.json"), J("data/metrics.json").catch(() => ({})), J("data/per_demo.json").catch(() => []), J("data/scores.json")]);
  SC = sc; PERD = per;
  hero(clips); quiz(clips, metrics); dumbbell(); epochs(); runs(); sim(); teacher(); flips(); scorer(); perDemo(); ladder(); gap(); use(); raceInit(); hardCharts(); recipe(); openTable();
  const cs = $(".counters"); cs.classList.add("reveal"); cs.addEventListener("shown", counters);
  observe();
  redraw.push(() => { dumbbell(); epochs(); runs(); teacher(); flips(); scorer(); perDemo(); ladder(); gap(); $("#race").innerHTML = ""; raceDraw(); quiz(clips, metrics); hero(clips); hardCharts(); recipe(); openTable(); });
})();

/* ---------- reveal card ---------- */
$("#secret-btn").addEventListener("click", () => $("#secret").classList.toggle("open"));
document.addEventListener("click", (e) => {
  if (!e.target.closest("[data-dflip]")) return;
  const w = $("#dflip"), on = !w.classList.contains("on");
  w.classList.toggle("on", on);
  $("#okm").inert = on; $("#secret").inert = !on;
  if (on && $("#okm-play").getAttribute("aria-pressed") === "true") $("#okm-play").click();
  if (e.detail === 0) (on ? $("#secret") : $("#okm")).querySelector("[data-dflip]").focus({ preventScroll: true });
});

/* ---------- alternate chapter shading ---------- */
$$("section.chapter").forEach((s, i) => s.classList.toggle("alt", i % 2 === 1));

/* ---------- two Tamils: Gemma's formal twins ---------- */
const TWINS = [
  ["ஏத்துக்கல", "ஏற்றுக்கொள்ளுங்கள்", { en: "\"didn't accept\" became \"please accept\". The meaning flipped.", ta: "\"ஏற்கவில்லை\" என்ற பொருள், \"ஏற்றுக்கொள்ளுங்கள்\" என்று மாறியது. பொருளே தலைகீழாகிவிட்டது." }],
  ["போகுல", "போகிறோம்", { en: "\"not going\" became \"we are going\". Flipped again.", ta: "\"போகவில்லை\" என்பது \"போகிறோம்\" என்று ஆனது. இங்கும் பொருள் தலைகீழ்." }],
  ["இல்லாம", "இல்லாமால்", { en: "A word that doesn't exist. The formal form is இல்லாமல்.", ta: "தமிழில் இப்படி ஒரு சொல்லே இல்லை. சரியான எழுத்துத் தமிழ் வடிவம்: இல்லாமல்." }],
  ["பசங்கள", "பசங்களിനെ", { en: "Malayalam letters (ിനെ) inside a Tamil word.", ta: "ஒரு தமிழ்ச் சொல்லுக்குள் மலையாள எழுத்துகள் (ിനെ)." }],
  ["எதுவா", "எதுவா든", { en: "A Korean syllable (든) inside a Tamil word.", ta: "தமிழ்ச் சொல்லுக்குள் ஒரு கொரிய எழுத்து (든)." }],
];
let twinI = 0;
function twins() {
  const [a, b, g] = TWINS[twinI];
  $("#twin-card").innerHTML = `<div class="pair" style="font-size:1.6rem;grid-template-columns:6.5rem 1fr"><span class="who">${L({ en: "spoken", ta: "பேச்சு" })}</span><span>${a}</span><span class="who">Gemma</span><span style="color:var(--bad)">${b}</span></div><p style="margin:0">${esc(L(g))}</p><div class="small mono" style="margin-top:.6rem">${twinI + 1} / ${TWINS.length}</div>`;
}
$("#twin-next").addEventListener("click", () => { twinI = (twinI + 1) % TWINS.length; twins(); });
twins(); redraw.push(twins);

/* ---------- the map: chapters as a graph along the arc ---------- */
const MAP = {
  stages: [[90, { en: "the problem", ta: "சிக்கல்" }], [330, { en: "building it", ta: "உருவாக்கம்" }], [570, { en: "measuring honestly", ta: "நேர்மையான அளவீடு" }], [770, { en: "where it stands", ta: "இப்போதைய நிலை" }], [930, { en: "hand it on", ta: "உங்கள் கையில்" }]],
  nodes: {
    why: [90, 170, { en: "A stranger in the house", ta: "வீட்டில் ஓர் அந்நியர்" }],
    hard: [90, 360, { en: "Why is this so hard?", ta: "ஏன் இவ்வளவு கடினம்?" }],
    runs: [260, 110, { en: "10 attempts, 5 days", ta: "10 முயற்சிகள், 5 நாட்கள்" }],
    epochs: [410, 60, { en: "Epoch 6 💀", ta: "6-ஆவது சுற்று 💀" }],
    recipe: [410, 170, { en: "What it learned from", ta: "எதைக் கேட்டுக் கற்றது" }],
    spot: [260, 260, { en: "Vanishing computers", ta: "காணாமல் போகும் கணினிகள்" }],
    kernels: [300, 330, { en: "A faster engine", ta: "வேகமான இதயப் பகுதி" }],
    speed: [680, 320, { en: "Small enough for home", ta: "வீட்டுக்கு ஏற்ற அளவு" }],
    teacher: [260, 400, { en: "Gemma adds full stops", ta: "Gemma இட்ட முற்றுப்புள்ளிகள்" }],
    "two-tamils": [410, 460, { en: "Two Tamils", ta: "இரண்டு தமிழ்" }],
    loans: [580, 470, { en: "English stays English", ta: "ஆங்கிலம் ஆங்கிலத்திலேயே" }],
    data: [580, 190, { en: "As you sow", ta: "நூலைப் போலச் சேலை" }],
    measure: [580, 370, { en: "Marking a Tamil test", ta: "தமிழ்த் தேர்வு திருத்தம்" }],
    listen: [770, 110, { en: "Who heard it right?", ta: "எது சரியாகக் கேட்டது?" }],
    scores: [770, 260, { en: "Scoreboard", ta: "மதிப்பெண் பலகை" }],
    verbatim: [770, 330, { en: "What you said", ta: "நீங்கள் சொன்னது" }],
    lose: [770, 400, { en: "Where it struggles", ta: "தடுமாறும் இடங்கள்" }],
    lm: [770, 480, { en: "A second opinion", ta: "இன்னொரு கருத்து" }],
    claude: [935, 170, { en: "Made with Claude", ta: "Claude-இன் துணையோடு" }],
    use: [935, 350, { en: "Use it", ta: "பயன்படுத்துங்கள்" }],
  },
  edges: [["why", "hard"], ["hard", "recipe"], ["hard", "data"], ["hard", "lose"], ["recipe", "epochs"], ["recipe", "runs"], ["why", "listen"], ["why", "speed"], ["why", "runs"], ["runs", "epochs"], ["runs", "spot"], ["runs", "two-tamils"], ["spot", "kernels"], ["kernels", "runs"], ["kernels", "speed"], ["teacher", "two-tamils"], ["teacher", "data"],
    ["two-tamils", "listen"], ["data", "measure"], ["data", "scores"], ["measure", "scores"], ["measure", "lose"], ["scores", "listen"], ["scores", "verbatim"], ["verbatim", "lose"], ["two-tamils", "verbatim"], ["verbatim", "lm"], ["epochs", "scores"], ["lose", "use"], ["lose", "lm"], ["loans", "runs"], ["loans", "measure"], ["loans", "data"], ["hard", "loans"], ["lm", "use"], ["lm", "scores"], ["claude", "runs"], ["speed", "use"]],
};
function mapDraw() {
  const W = 1020, H = 520, N = MAP.nodes;
  let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="chapter map">`;
  g += `<path d="M20,300 C 200,${H + 40} 420,-20 600,260 S 900,${H} ${W - 10},250" fill="none" stroke="var(--ours)" stroke-width="40" opacity=".07" stroke-linecap="round"/>`;
  MAP.stages.forEach(([x, l]) => (g += `<text x="${x}" y="16" text-anchor="middle" style="fill:var(--fg-3);font-size:11px;${lang() === "en" ? "letter-spacing:.08em" : ""}">${esc(L(l).toUpperCase())}</text>`));
  MAP.edges.forEach(([a, b]) => { const [x1, y1] = N[a], [x2, y2] = N[b], mx = (x1 + x2) / 2; g += `<path class="me" data-a="${a}" data-b="${b}" d="M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}" fill="none" stroke="var(--fg-3)" stroke-width="1.5" opacity=".35" style="transition:opacity .2s,stroke .2s"/>`; });
  Object.entries(N).forEach(([id, [x0, y, l]]) => { let x = x0;
    const t = L(l), w = Math.max(110, t.length * (lang() === "ta" ? 8.6 : 7.4) + 24); x = Math.min(W - w / 2 - 4, Math.max(w / 2 + 4, x));
    g += `<g class="mn" data-id="${id}" style="cursor:pointer" tabindex="0" role="link" aria-label="${esc(t)}"><rect x="${x - w / 2}" y="${y - 17}" width="${w}" height="34" rx="17" fill="var(--bg-3)" stroke="var(--rule)" style="transition:stroke .2s,fill .2s"/><text x="${x}" y="${y + 5}" text-anchor="middle" class="lbl" style="font-size:12.5px">${esc(t)}</text></g>`;
  });
  $("#map-svg").innerHTML = g + "</svg>";
  $$("#map-svg .mn").forEach((n) => {
    const id = n.dataset.id, on = (v) => {
      $$("#map-svg .me").forEach((e) => { const hit = e.dataset.a === id || e.dataset.b === id; e.style.opacity = v ? (hit ? 1 : 0.12) : 0.35; e.style.stroke = v && hit ? "var(--ours)" : "var(--fg-3)"; });
      $$("#map-svg .mn").forEach((m) => { const linked = m.dataset.id === id || MAP.edges.some(([a, b]) => (a === id && b === m.dataset.id) || (b === id && a === m.dataset.id)); $("rect", m).style.stroke = v && linked ? "var(--ours)" : "var(--rule)"; m.style.opacity = v && !linked ? 0.4 : 1; });
    };
    n.addEventListener("mouseenter", () => on(true)); n.addEventListener("mouseleave", () => on(false)); n.addEventListener("focus", () => on(true)); n.addEventListener("blur", () => on(false));
    const go = () => document.getElementById(id)?.scrollIntoView({ behavior: reduce ? "auto" : "smooth" });
    n.addEventListener("click", go); n.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
  });
}
mapDraw(); redraw.push(mapDraw);

/* ---------- table of contents: recessed sidebar on wide screens, slide-in panel elsewhere ---------- */
// Short names come from the chapter map, grouped under its stages, in page order.
const TOC = [
  [null, [["top", { en: "Start", ta: "தொடக்கம்" }], ["map", { en: "The map", ta: "வரைபடம்" }]]],
  [MAP.stages[0][1], ["why", "hard"]],
  [MAP.stages[1][1], ["recipe", "data", "loans", "teacher", "two-tamils", "spot", "kernels", "runs", "epochs"]],
  [MAP.stages[2][1], ["measure", "listen", "scores", "verbatim"]],
  [MAP.stages[3][1], ["lose", "lm", "speed"]],
  [MAP.stages[4][1], ["claude", "use"]],
];
const tocIds = TOC.flatMap(([, ids]) => ids.map((x) => (Array.isArray(x) ? x[0] : x)));
const plain = (el) => { const c = el.cloneNode(true); c.querySelectorAll("rt").forEach((r) => r.remove()); return c.textContent.replace(/\s+/g, " ").trim(); };
const tocSub = {};  // chapter id -> [{en, ta}] of its visible subheadings (h3 pairs)
function tocSubs(id) {
  const sec = document.getElementById(id);
  if (!sec || sec.tagName !== "SECTION") return [];
  const detail = document.documentElement.dataset.depth === "detail";
  const shown = (el) => [...ancestors(el)].every((a) => !a.dataset || !a.dataset.d || (a.dataset.d === "d") === detail);
  return [...sec.querySelectorAll('h3[data-l="en"]')].filter(shown).map((en) => ({ en, ta: langPartner(en) || en }));
}
function* ancestors(el) { for (let e = el; e && e !== document.body; e = e.parentElement) yield e; }
function tocDraw() {
  const html = TOC.map(([stage, ids]) => (stage ? `<div class="stg">${esc(L(stage))}</div>` : "") + ids.map((x) => {
    const [id, label] = Array.isArray(x) ? x : [x, MAP.nodes[x][2]];
    const subs = (tocSub[id] = tocSubs(id));
    const sub = subs.length ? `<div class="subs" data-subs="${id}">${subs.map((p, i) => `<a href="#${id}" class="sub" data-sub="${id}:${i}">${esc(L({ en: plain(p.en), ta: plain(p.ta) }))}</a>`).join("")}</div>` : "";
    return `<a href="#${id}" data-toc="${id}">${esc(L(label))}</a>${sub}`;
  }).join("")).join("");
  $("#toc").innerHTML = html;
  $("#tocpanel").innerHTML = `<div class="tochead">${L({ en: "Contents", ta: "உள்ளடக்கம்" })}</div>` + html;
  tocSpy(true);
}
let tocCur = null;
function tocSpy(force) {
  const y = innerHeight * 0.35;
  let cur = "top";
  for (const id of tocIds) { const el = document.getElementById(id); if (el && el.getBoundingClientRect().top <= y) cur = id; }
  $("#toc").classList.toggle("on", scrollY > innerHeight * 0.6);
  let sub = null;
  (tocSub[cur] || []).forEach((p, i) => { const el = lang() === "ta" ? p.ta : p.en; if (el.getBoundingClientRect().top <= y) sub = `${cur}:${i}`; });
  $$("[data-sub]").forEach((a) => (a.dataset.sub === sub ? a.setAttribute("aria-current", "true") : a.removeAttribute("aria-current")));
  if (cur === tocCur && !force) return;
  tocCur = cur;
  $$("[data-toc]").forEach((a) => (a.dataset.toc === cur ? a.setAttribute("aria-current", "true") : a.removeAttribute("aria-current")));
  $$("#toc [data-subs]").forEach((d) => d.classList.toggle("open", d.dataset.subs === cur));
  const box = $("#toc"), it = box.querySelector("[aria-current]");  // keep the current entry visible inside the sidebar
  if (it && box.scrollHeight > box.clientHeight) {
    const top = it.offsetTop - box.clientHeight / 2 + it.offsetHeight / 2;
    box.scrollTo({ top: Math.max(0, top), behavior: "smooth" });
  }
}
let tocTick = false;
addEventListener("scroll", () => { if (!tocTick) { tocTick = true; requestAnimationFrame(() => { tocTick = false; tocSpy(); }); } }, { passive: true });
addEventListener("resize", () => tocSpy(true));
let lastW = innerWidth, rzT = null;
addEventListener("resize", () => { clearTimeout(rzT); rzT = setTimeout(() => { const w = innerWidth; if ((w <= NARROW) !== (lastW <= NARROW) || (w <= NARROW && Math.abs(w - lastW) > 30)) { lastW = w; redraw.forEach((f) => f()); } }, 200); });
function tocOpen(open) {
  $("#tocpanel").classList.toggle("open", open); $("#tocscrim").classList.toggle("open", open);
  $("#tocbtn").setAttribute("aria-expanded", String(open));
  if (open) $("#tocpanel [aria-current]")?.scrollIntoView({ block: "center" });
}
$("#tocbtn").addEventListener("click", () => tocOpen(!$("#tocpanel").classList.contains("open")));
$("#tocscrim").addEventListener("click", () => tocOpen(false));
$("#tocpanel").addEventListener("click", (e) => { if (e.target.closest("a")) tocOpen(false); });
document.addEventListener("click", (e) => {
  const a = e.target.closest("[data-sub]");
  if (!a) return;
  e.preventDefault();
  const [id, i] = a.dataset.sub.split(":"), p = tocSub[id][+i], el = lang() === "ta" ? p.ta : p.en;
  scrollTo({ top: el.getBoundingClientRect().top + scrollY - 90, behavior: "smooth" });
});
addEventListener("keydown", (e) => { if (e.key === "Escape") tocOpen(false); });
tocDraw(); redraw.push(tocDraw);
let hintT = setTimeout(() => scrollY < 40 && $("#scrollhint").classList.add("on"), 1400);
addEventListener("scroll", () => { if (scrollY > 40) { clearTimeout(hintT); $("#scrollhint").classList.remove("on"); } }, { passive: true });

/* ---------- "a computer can't hear": the voice as numbers ---------- */
let numTimer = null;
async function numbersStrip(url) {
  const host = $("#numbers"); if (!host) return;
  await peaks(url, 10); const d = await peaksCache[url];
  let pos = Math.floor(d.length * 0.3);
  clearInterval(numTimer);
  const draw = () => { const out = []; for (let i = 0; i < 48; i++) { const v = d[pos + i * 7] || 0; out.push(Math.abs(v) > 0.2 ? `<b>${v.toFixed(3)}</b>` : v.toFixed(3)); } host.innerHTML = out.join(" "); pos = (pos + 37) % (d.length - 400); };
  draw(); if (!reduce) numTimer = setInterval(draw, 110);
}

/* ---------- phone layout for horizontal bar charts: label above a full-width bar, value at its end ---------- */
function hbarsNarrow(host, rows, max, aria, title) {
  const W = fitW(host, 1000), LH = 17, bh = 14, vw = 64, x = (v) => Math.max(3, (v / max) * (W - vw));
  const tl = title ? wrapText(title, W, 13) : [], top = title ? tl.length * LH + 10 : 4;
  const lab = rows.map(([n]) => wrapText(n, W, 13)), hs = lab.map((l) => l.length * LH + bh + 12);
  const H = top + hs.reduce((a, b) => a + b, 0);
  let g = fitSvg(W, H, esc(aria));
  if (title) g += `<text x="0" y="14" style="fill:var(--fg);font-weight:700;font-size:13px">${tspans(tl, 0, LH)}</text>`;
  let y = top;
  rows.forEach(([n, v, c, label, op], i) => {
    const w = x(v), by = y + lab[i].length * LH + 2;
    g += `<text x="0" y="${y + 13}" class="lbl" style="font-size:13px">${tspans(lab[i], 0, LH)}</text>`;
    g += `<rect x="0" y="${by}" width="${w}" height="${bh}" rx="4" fill="var(${c})"${op ? ` opacity="${op}"` : ""}/>`;
    const lw = String(label).length * 7, inside = w + 6 + lw > W && w > lw + 12;  // no room after the bar: write it inside the bar's end
    g += `<text x="${inside ? w - 6 : Math.min(w + 6, W - lw)}" y="${by + 11}" ${inside ? 'text-anchor="end"' : ""} style="fill:var(${inside ? "--bg" : "--fg"});font-weight:700;font-size:12px">${esc(label)}</text>`;
    y += hs[i];
  });
  host.innerHTML = g + "</svg>";
}

/* ---------- Whisper's hours: English vs Tamil ---------- */
function hardCharts() {
  const B = [[{ en: "Whisper: English", ta: "Whisper: ஆங்கிலம்" }, 438218, "--fg-3"], [{ en: "Whisper: Tamil", ta: "Whisper: தமிழ்" }, 136, "--itcore"], [{ en: "our Tamil, collected from scraps", ta: "நாங்கள் அங்கிங்கே சேகரித்த தமிழ்" }, 4760, "--ours"]];
  if (isNarrow()) return hbarsNarrow($("#gap-hours"), B.map(([n, v, c]) => [L(n), v, c, `${v.toLocaleString()} h`]), 438218, "hours of training audio");
  const W = 1000, rh = 46, P = { l: lang() === "ta" ? 360 : 260, r: 120, t: 6 }, H = P.t + B.length * rh, x = (v) => P.l + (v / 438218) * (W - P.l - P.r);
  let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="hours of training audio">`;
  B.forEach(([n, v, c], i) => { const y = P.t + i * rh; g += `<text x="${P.l - 12}" y="${y + 27}" text-anchor="end" class="lbl">${esc(L(n))}</text><rect x="${P.l}" y="${y + 10}" width="${Math.max(3, x(v) - P.l)}" height="${rh - 20}" rx="4" fill="var(${c})" data-tip="${v.toLocaleString()} h"/><text x="${Math.max(P.l + 3, x(v)) + 10}" y="${y + 27}" style="fill:var(--fg);font-weight:700">${v.toLocaleString()} h</text>`; });
  $("#gap-hours").innerHTML = g + "</svg>"; tips($("#gap-hours"));
}

/* ---------- what it learned from ---------- */
let MIX = null;
const KIND = { spoken: { en: "conversations", ta: "உரையாடல்கள்" }, read: { en: "read aloud", ta: "உரக்க வாசித்தவை" }, youtube: { en: "YouTube", ta: "YouTube" }, synthetic: { en: "computer voices, Tamil + English", ta: "கணினிக் குரல்கள், தமிழ் + ஆங்கிலம்" }, english: { en: "English", ta: "ஆங்கிலம்" }, negatives: { en: "silence and noise", ta: "பேச்சு இல்லாத ஒலிகள்" } };
const KORDER = ["youtube", "spoken", "read", "english", "synthetic", "negatives"];
function recipe() {
  if (!MIX) return;
  const by = {}; MIX.rows.forEach((r) => { (by[r.kind] ||= { h: 0, seen: 0 }); by[r.kind].h += r.hours; by[r.kind].seen += r.seen_5ep; });
  const W = 1000, rh = 40, P = { l: 290, r: 150, t: 6 }, H = P.t + KORDER.length * rh, mx = Math.max(...KORDER.map((k) => by[k].h)), x = (v) => P.l + (v / mx) * (W - P.l - P.r);
  let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="hours by kind">`;
  const mixNarrow = isNarrow();
  if (mixNarrow) hbarsNarrow($("#mix-bars"), KORDER.map((k) => [L(KIND[k]), by[k].h, k === "english" ? "--fg-3" : "--ours", `${Math.round(by[k].h).toLocaleString()} h`]), mx, "hours by kind");
  KORDER.forEach((k, i) => { const y = P.t + i * rh, v = by[k].h; g += `<text x="${P.l - 12}" y="${y + 25}" text-anchor="end" class="lbl">${esc(L(KIND[k]))}</text><rect x="${P.l}" y="${y + 8}" width="${Math.max(3, x(v) - P.l)}" height="${rh - 16}" rx="4" fill="var(${k === "english" ? "--fg-3" : "--ours"})" data-tip="${v.toLocaleString()} h · ${L({ en: "heard", ta: "கேட்டது" })} ${by[k].seen.toLocaleString()} h"/><text x="${x(v) + 10}" y="${y + 25}" style="fill:var(--fg);font-weight:700">${v.toLocaleString()} h</text>`; });
  if (!mixNarrow) { $("#mix-bars").innerHTML = g + "</svg>"; tips($("#mix-bars")); }
  const hd = [L({ en: "source", ta: "மூலம்" }), L({ en: "what", ta: "என்ன" }), L({ en: "split", ta: "பகுதி" }), L({ en: "licence", ta: "உரிமம்" }), L({ en: "hours", ta: "மணி" }), L({ en: "passes per epoch", ta: "ஒரு சுற்றில் எத்தனை முறை" }), L({ en: "heard by epoch 5", ta: "5 சுற்றுகளில் கேட்டது" })];
  let t = `<table class="dt"><thead><tr>${hd.map((h, i) => `<th style="text-align:${i <= 3 ? "left" : "right"}">${esc(h)}</th>`).join("")}</tr></thead><tbody>`;
  KORDER.forEach((k) => { t += `<tr class="grp"><td colspan="7">${esc(L(KIND[k]))}</td></tr>`; MIX.rows.filter((r) => r.kind === k).forEach((r) => (t += `<tr><td>${esc(r.name)}</td><td style="text-align:left">${esc(r.what)}</td><td style="text-align:left;white-space:normal;min-width:12rem">${esc(r.split)}</td><td style="text-align:left">${esc(r.licence)}</td><td>${r.hours.toLocaleString()}</td><td>${r.per_epoch === 0.33 ? "⅓" : r.per_epoch}</td><td>${r.seen_5ep.toLocaleString()}</td></tr>`)); });
  t += `<tr><td><b>${L({ en: "total", ta: "மொத்தம்" })}</b></td><td></td><td></td><td></td><td><b>${MIX.unique_hours.toLocaleString()}</b></td><td></td><td><b>${MIX.seen_hours_5ep.toLocaleString()}</b></td></tr></tbody></table>`;
  $("#mix-table").innerHTML = t;
  const S = MIX.schedule, T = S.total, E = S.steps_per_epoch, D = S.decay, pk = S.peak;
  const dec = (s, a) => pk * (1 - Math.sqrt(Math.min(1, Math.max(0, (s - a) / D))));
  const main = (s) => (s <= S.warmup ? (pk * s) / S.warmup : s < 3 * E - D ? pk : s < 3 * E ? dec(s, 3 * E - D) : s < 3 * E + S.rewarm[1] ? (pk * (s - 3 * E)) / S.rewarm[1] : s < T - D ? pk : dec(s, T - D));
  const SW = 1000, SH = 260, Q = { l: 60, r: 20, t: 30, b: 36 }, sx = (s) => Q.l + (s / T) * (SW - Q.l - Q.r), sy = (v) => Q.t + (1 - v / pk) * (SH - Q.t - Q.b);
  let p = ""; for (let s = 0; s <= T; s += 40) p += `${s ? "L" : "M"}${sx(s).toFixed(1)},${sy(main(s)).toFixed(1)}`;
  let g2 = `<svg viewBox="0 0 ${SW} ${SH}" role="img" aria-label="learning rate schedule">`;
  for (let k = 0; k <= 6; k++) g2 += `<g class="grid"><line x1="${sx(k * E)}" x2="${sx(k * E)}" y1="${Q.t}" y2="${SH - Q.b}"/></g><text x="${sx(k * E)}" y="${SH - 14}" text-anchor="${k === 6 ? "end" : k === 0 ? "start" : "middle"}">${k ? L({ en: "epoch ", ta: "சுற்று " }) + k : "0"}</text>`;
  g2 += `<text x="${Q.l - 8}" y="${sy(pk) + 4}" text-anchor="end">1.49e-4</text><text x="${Q.l - 8}" y="${sy(0) + 4}" text-anchor="end">0</text>`;
  [1, 2, 4, 5].forEach((k) => { const a = k * E - D; let q = ""; for (let s = a; s <= k * E; s += 20) q += `${s === a ? "M" : "L"}${sx(s).toFixed(1)},${sy(dec(s, a)).toFixed(1)}`; const rel = k === 5; g2 += `<path d="${q}" fill="none" stroke="var(${rel ? "--good" : "--ours-2"})" stroke-width="${rel ? 3 : 2}" stroke-dasharray="${rel ? "" : "4 3"}"/><circle cx="${sx(k * E)}" cy="${sy(0)}" r="${rel ? 7 : 5}" fill="var(${rel ? "--good" : "--ours-2"})" data-tip="${L({ en: "epoch", ta: "சுற்று" })} ${k}"/>`; });
  g2 += `<path d="${p}" fill="none" stroke="var(--ours)" stroke-width="2.5"/>`;
  g2 += `<circle cx="${sx(3 * E)}" cy="${sy(0)}" r="5" fill="var(--ours)" data-tip="${L({ en: "epoch", ta: "சுற்று" })} 3"/><circle cx="${sx(T)}" cy="${sy(0)}" r="5" fill="var(--bad)" data-tip="${L({ en: "epoch", ta: "சுற்று" })} 6"/>`;
  g2 += `<text x="${sx(5 * E) - 12}" y="${sy(0) - 12}" text-anchor="end" style="fill:var(--fg);font-weight:700">${L({ en: "release: epoch 5", ta: "வெளியீடு: சுற்று 5" })}</text>`;
  $("#sched-chart").innerHTML = g2 + "</svg>"; tips($("#sched-chart"));
}

/* ---------- every open model on the same clips ---------- */
let OPEN = {};
const ONAME = { e5: "parakeet-tamil", e5_lm: "parakeet-tamil + LM", itcore: "indic-transcribe-core", hima: "Hima-ASR", nemotron: "Nemotron 3.5 Transcribe", stock_v3: "original parakeet v3", shruti: "shruti-tamil", indicwhisper: "IndicWhisper", omni: "Omni 7B", gemma4_e4b: "Gemma 4 E4B" };
const OCOLS = ["e5", "e5_lm", "itcore", "hima", "nemotron", "stock_v3", "shruti", "indicwhisper", "omni", "gemma4_e4b"];
// parameters, from each model's published weights; LM = the 5-gram language model
const TSY = { en: "yes, natively", ta: "ஆம், நேரடியாக" }, TSA = { en: "approximately", ta: "தோராயமாக" }, TSN = { en: "no", ta: "இல்லை" };
// word timestamps: native (TDT / Nemotron's aligner), approximate (Whisper attention), or none in the released code
const OTS = { e5: TSY, e5_lm: TSN, itcore: TSN, hima: TSN, nemotron: TSY, stock_v3: TSY, shruti: TSN, indicwhisper: TSA, omni: TSN, gemma4_e4b: TSN };
// word timestamps on Tamil: share of word starts within 0.2 s of an independent aligner / share of words put in silence
const OTSACC = { e5: "94% / 1.1%", nemotron: "24% / 0.3%", indicwhisper: "90% / 7.5%" };
const OPARAMS = { e5: "0.63B", e5_lm: "0.63B + LM", itcore: "1.22B", hima: "0.46B", nemotron: "33.7B (4.3B active)", stock_v3: "0.63B", shruti: "0.06B", indicwhisper: "0.77B", omni: "7.8B", gemma4_e4b: "8.0B" };
// may include clips the model trained on (its training data lists Common Voice and FLEURS): marked *, never "best"
const OFLAG = { nemotron: ["eval_cv_ta", "eval_fleurs_ta", "eval_cv_en"] };
function openTable() {
  if (!SC) return;
  const keyOf = (c) => Object.keys(OPEN).find((k) => k === c || k.startsWith(c)) || c;
  const val = (c, s) => (c === "e5" || c === "e5_lm" || c === "itcore" ? SC.scores[c]?.[s] : OPEN[keyOf(c)]?.[s]) || null;
  const m = metric;
  let t = `<table class="dt"><thead><tr><th>${L({ en: "test set", ta: "தேர்வு" })} · ${m.toUpperCase()}</th>${OCOLS.map((c) => `<th>${esc(ONAME[c])}</th>`).join("")}</tr></thead><tbody>`;
  t += `<tr class="params"><td>${L({ en: "parameters", ta: "அளவுருக்கள்" })}</td>${OCOLS.map((c) => `<td>${esc(OPARAMS[c] || "")}</td>`).join("")}</tr>`;
  t += `<tr class="params"><td>${L({ en: "word timestamps", ta: "சொல் நேரங்கள்" })}</td>${OCOLS.map((c) => `<td>${esc(L(OTS[c]))}</td>`).join("")}</tr>`;
  t += `<tr class="params"><td>${L({ en: "on Tamil: within 0.2 s / in silence", ta: "தமிழில்: 0.2 வினாடிக்குள் / அமைதியில்" })}</td>${OCOLS.map((c) => `<td>${esc(OTSACC[c] || "–")}</td>`).join("")}</tr>`;
  Object.keys(SC.sets).forEach((s) => {
    const en = SC.sets[s].kind === "english", mm = en ? "wer" : m;
    const vs = OCOLS.map((c) => { const r = val(c, s); return r ? r[mm] : undefined; });
    const flag = (i) => (OFLAG[OCOLS[i]] || []).includes(s);
    const ok = vs.filter((v, i) => v != null && !flag(i)), best = ok.length ? Math.min(...ok) : null;
    t += `<tr><td>${esc(L(SC.sets[s]))}</td>${vs.map((v, i) => { const c = OCOLS[i]; if (v == null) return (c === "shruti" || c === "indicwhisper") && en ? `<td class="pending" title="Tamil-only model">–</td>` : `<td class="pending">${L({ en: "scoring…", ta: "மதிப்பிடப்படுகிறது…" })}</td>`; return flag(i) ? `<td title="${esc(L({ en: "may include clips this model trained on", ta: "இந்த மாதிரி பயிற்சியில் கேட்ட ஒலிக்கோப்புகள் இதில் இருக்கலாம்" }))}">${v.toFixed(2)}*</td>` : `<td class="${v === best ? "best" : ""}">${v.toFixed(2)}</td>`; }).join("")}</tr>`;
  });
  $("#open-table").innerHTML = t + "</tbody></table>";
}
(async () => { try { MIX = await J("data/mix.json"); recipe(); } catch (e) { console.warn(e); } try { OPEN = await J("data/open_models.json"); openTable(); } catch (e) {} })();

/* ---------- story-mode scoreboard: sounds misheard (PER) and meaning drift (EER) ---------- */
function storyScore() {
  if (!SC) return;
  const S = SC.scores, READ = ["eval_kathbath", "eval_mile_ta", "eval_fleurs_ta", "eval_cv_ta", "eval_openslr65"];
  const avg = (m, k) => READ.reduce((a, s) => a + S[m][s][k], 0) / READ.length;
  const Q = [
    { k: "per", t: { en: "Sounds misheard", ta: "தவறாகக் கேட்ட ஒலிகள்" }, u: { en: "out of every 100 sounds", ta: "ஒவ்வொரு 100 ஒலிகளுக்கும் எத்தனை" } },
    { k: "eer", t: { en: "Meaning drift", ta: "பொருள் எவ்வளவு மாறியது" }, u: { en: "0 means exactly the same meaning", ta: "0 என்றால் பொருள் கொஞ்சமும் மாறவில்லை" } },
  ];
  const G = [[{ en: "everyday conversation", ta: "அன்றாட உரையாடல்" }, (m, k) => S[m].eval_indicvoices[k]], [{ en: "reading aloud", ta: "உரக்க வாசித்தல்" }, avg]];
  $("#story-score").innerHTML = Q.map((q) => {
    const vals = G.flatMap(([, f]) => [f("e5", q.k), f("itcore", q.k)]), mx = Math.max(...vals) * 1.15;
    const W = fitW($("#story-score"), 520, 40), rh = 34, H = 2 * (2 * rh + 34) + 10;
    let g = fitSvg(W, H, esc(L(q.t)));
    G.forEach(([gl, f], gi) => {
      const y0 = gi * (2 * rh + 34) + 4;
      g += `<text x="0" y="${y0 + 14}" class="lbl">${esc(L(gl))}</text>`;
      [["e5", "--ours", L({ en: "ours", ta: "எங்களுடையது" })], ["itcore", "--itcore", L({ en: "the other model", ta: "மற்றது" })]].forEach(([m, c, nm], i) => {
        const v = f(m, q.k), y = y0 + 22 + i * rh, w = (v / mx) * (W - 190);
        g += `<text x="0" y="${y + 18}">${esc(nm)}</text><rect x="140" y="${y + 4}" width="${w}" height="${rh - 10}" rx="4" fill="var(${c})"/><text x="${146 + w}" y="${y + 19}" style="fill:var(--fg);font-weight:700;font-size:14px">${v.toFixed(1)}</text>`;
      });
    });
    return `<div class="scorer"><h3 style="margin:0">${esc(L(q.t))}</h3><div class="small" style="margin-bottom:.6rem">${esc(L(q.u))}</div><div class="chart">${g}</svg></div></div>`;
  }).join("");
}
(function waitSC() { if (SC) { storyScore(); redraw.push(storyScore); } else setTimeout(waitSC, 100); })();

/* ---------- one sentence per line, everywhere (static and script-rendered text) ---------- */
const SENT_SEL = "h2, h3, p, li, .vlist li > span, .okmstat span, .verdict, .flip .g, .small, .tile p, .hint, .tag, figcaption";
const SENT_SKIP = ".lines, svg, table, .karaoke, .numbers, .simlog, .pair, .prompt, .heard, pre, .opt, .loanrow, .bar, footer, .techdlg";
// widow control: the last space inside an element becomes a no-break space, across inline tags
function glueNode(el) {
  const w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT), nodes = [];
  while (w.nextNode()) nodes.push(w.currentNode);
  for (let i = nodes.length - 1; i >= 0; i--) {
    const t = nodes[i].nodeValue, trimmed = t.replace(/\s+$/, ""), k = trimmed.search(/\s\S*$/);
    if (k > 0) {
      // glue only short endings: two long (Tamil) words glued together cannot fit and get split mid-word
      const prev = trimmed.slice(0, k).match(/\S*$/)[0], last = trimmed.slice(k + 1);
      if (prev.length + last.length <= (/[\u0B80-\u0BFF]/.test(prev + last) ? 12 : 20)) nodes[i].nodeValue = t.slice(0, k) + "\u00a0" + t.slice(k + 1);
      return;
    }
    if (trimmed.trim().length > 14) return;
  }
}
function splitSentences(root = document) {
  const els = [...(root.matches?.(SENT_SEL) ? [root] : []), ...root.querySelectorAll(SENT_SEL)];
  els.forEach((p) => {
    if (p.dataset.split === "1" || p.closest(SENT_SKIP) || p.querySelector(".s") || (p.tagName === "LI" && p.querySelector(":scope > span, :scope > b"))) return;
    const html = p.innerHTML.trim();
    const parts = html.split(/(?<=[.?!…])\s+(?=(?:<[^>]+>)*["“(]?[A-Z0-9஀-௿])/);
    p.dataset.split = "1";
    // glue the last 2 words of every sentence so no sentence ends with a lone short word on its own line
    if (parts.length < 2) { if (!/^H[1-3]$/.test(p.tagName)) glueNode(p); return; }
    p.innerHTML = parts.map((x) => `<span class="s">${x}</span>`).join("");
    p.classList.add("sents");
    if (!/^H[1-3]$/.test(p.tagName)) $$(".s", p).forEach(glueNode);
  });
}
splitSentences();
// keep a Tamil suffix on the same line as the English word it is attached to ("AI-க்கு"): word joiner after the hyphen
function joinSuffixes(root) {
  const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
  let n;
  while ((n = w.nextNode())) if (/[-/–]/.test(n.nodeValue) && !n.parentElement.closest("script, style")) n.nodeValue = n.nodeValue.replace(/-(?=[\u0B80-\u0BFF])/g, "-\u2060").replace(/(?<=[\u0B80-\u0BFF])\/(?=[\u0B80-\u0BFF])/g, "\u2060/\u2060").replace(/(?<=\d)–(?=\d)/g, "–\u2060");
}
joinSuffixes(document.body);
new MutationObserver((ms) => { for (const m of ms) m.addedNodes.forEach((n) => { if (n.nodeType === 1) { joinSuffixes(n); splitSentences(n); } }); }).observe(document.body, { childList: true, subtree: true });
/* ---------- voice map: stepped dial, infographic, real clip with synced caption ---------- */
let VD = null, vdAudio = null, vdI = 2;
const vdA = [-1, -0.5, 0, 0.5, 1];
function vmap() {
  const W = 500, H = 290, Y = 190, X0 = 180, X1 = 460;
  const p = (a) => X0 + ((a + 1) / 2) * (X1 - X0);
  let rnd = 11; const r = () => ((rnd = (rnd * 16807) % 2147483647) / 2147483647);
  let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="voice map">`;
  g += `<defs><marker id="vm-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="var(--ours-2)"/></marker></defs>`;
  // Maya2's built-in voices: a cluster, top left, label above it
  g += `<text x="6" y="22" class="lbl" style="font-size:13px">${esc(L({ en: "Maya2's 20 built-in voices", ta: "Maya2-இன் 20 உள்ளமைந்த குரல்கள்" }))}</text>`;
  for (let i = 0; i < 20; i++) { const t2 = r() * 6.283, rr = Math.sqrt(r()) * 44; g += `<circle cx="${80 + Math.cos(t2) * rr * 1.2}" cy="${92 + Math.sin(t2) * rr}" r="5" fill="var(--fg-3)" opacity=".6"/>`; }
  // the new voice v, far from the cluster: a faint "distance" link
  g += `<text x="${p(0)}" y="${Y - 38}" text-anchor="middle" class="lbl" style="font-size:13px">${esc(L({ en: "v: a new voice nobody has", ta: "v: யாருக்கும் இல்லாத புதிய குரல்" }))}</text>`;
  g += `<line x1="${p(0)}" y1="${Y - 30}" x2="${p(0)}" y2="${Y - 13}" stroke="var(--fg-3)" stroke-width="1"/>`;
  // direction d as a horizontal axis with 5 stops (same order as the slider)
  g += `<line x1="${X0 - 22}" y1="${Y}" x2="${X1 + 22}" y2="${Y}" stroke="var(--ours-2)" stroke-width="2" stroke-dasharray="6 5" marker-end="url(#vm-arr)"/>`;
  vdA.forEach((a, i) => { const x = p(a); g += `<circle cx="${x}" cy="${Y}" r="${i === vdI ? 10 : 5}" fill="${i === vdI ? "var(--ours)" : "var(--bg-3)"}" stroke="var(--ours)" stroke-width="2.5" style="cursor:pointer" data-vstep="${i}"/>`; });
  g += `<text x="${X0}" y="${Y + 30}" text-anchor="middle" style="fill:var(--fg)">${esc(L({ en: "deeper", ta: "ஆழ்ந்த" }))}</text>`;
  g += `<text x="${X1}" y="${Y + 30}" text-anchor="middle" style="fill:var(--fg)">${esc(L({ en: "higher", ta: "உயர்ந்த" }))}</text>`;
  g += `<text x="${(X0 + X1) / 2}" y="${Y + 58}" text-anchor="middle" style="font-size:12px;fill:var(--fg)">${esc(L({ en: "direction d", ta: "திசை d" }))}</text>`;
  g += `<text x="${(X0 + X1) / 2}" y="${Y + 76}" text-anchor="middle" style="font-size:11px;fill:var(--fg-2)">${esc(L({ en: "learned from pitch only, no labels", ta: "சுருதியிலிருந்து மட்டும், வகைப்படுத்தல் இல்லாமல்" }))}</text>`;
  $("#vmap").innerHTML = g + "</svg>";
  $$("#vmap [data-vstep]").forEach((c) => c.addEventListener("click", () => vdSet(+c.dataset.vstep, true)));
}
function vdial() {
  if (!VD) return;
  const st = VD.steps[vdI], hz = st.f0;
  $("#vdial").value = vdI; $("#vdial-hz").textContent = `${hz} Hz`;
  const W = 440, H = 86, lo = 90, hi = 290, x = (v) => 8 + ((v - lo) / (hi - lo)) * (W - 16);
  let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="voice pitch">`;
  g += `<text x="${x(104)}" y="13" style="font-size:11px">${L({ en: "real Tamil speakers, 95%", ta: "உண்மையான தமிழ்ப் பேச்சாளர்கள், 95%" })}</text>`;
  g += `<rect x="${x(104)}" y="20" width="${x(266) - x(104)}" height="16" rx="8" fill="var(--fg-3)" opacity=".3"/>`;
  g += `<rect x="${x(112)}" y="44" width="${x(271) - x(112)}" height="5" rx="2.5" fill="var(--ours)" opacity=".45"/>`;
  g += `<text x="${x(271)}" y="64" text-anchor="end" style="font-size:11px">${L({ en: "our dial", ta: "எங்கள் குமிழ்" })}</text>`;
  g += `<line x1="${x(hz)}" x2="${x(hz)}" y1="16" y2="52" stroke="var(--fg)" stroke-width="2.5"/><circle cx="${x(hz)}" cy="46.5" r="7" fill="var(--ours)" stroke="var(--bg-3)" stroke-width="2.5"/>`;
  [100, 150, 200, 250].forEach((v) => (g += `<text x="${x(v)}" y="${H - 2}" text-anchor="middle" style="font-size:10px">${v} Hz</text>`));
  $("#vdial-chart").innerHTML = g + "</svg>";
  $("#vdial-words").innerHTML = st.words.map(([w, t]) => `<span data-t="${t}">${esc(w)}</span>`).join(" ");
  $("#vdial-words").classList.add("idle");
  vmap();
}
function vdSet(i, play) {
  vdI = i; if (vdAudio) vdAudio.pause(); vdAudio = null;
  const b = $("#vdial-play"); b.textContent = "▶"; b.setAttribute("aria-pressed", "false");
  vdial(); if (play) vdPlay();
}
function vdPlay() {
  const st = VD.steps[vdI], b = $("#vdial-play");
  if (vdAudio && !vdAudio.paused) { vdAudio.pause(); return; }
  if (current && current !== vdAudio) current.pause();
  vdAudio = new Audio(st.audio); current = vdAudio;
  const ws = $$("#vdial-words span"), box = $("#vdial-words");
  vdAudio.addEventListener("play", () => { b.textContent = "❚❚"; b.setAttribute("aria-pressed", "true"); box.classList.remove("idle"); });
  vdAudio.addEventListener("pause", () => { b.textContent = "▶"; b.setAttribute("aria-pressed", "false"); });
  vdAudio.addEventListener("ended", () => box.classList.add("idle"));
  const tick = () => { if (!vdAudio) return; const tm = vdAudio.currentTime; ws.forEach((x, k) => { const on = +x.dataset.t <= tm + 0.05; x.classList.toggle("on", on); x.classList.toggle("now", on && (!ws[k + 1] || +ws[k + 1].dataset.t > tm + 0.05)); }); if (!vdAudio.paused) requestAnimationFrame(tick); };
  vdAudio.addEventListener("play", tick); vdAudio.play();
}
$("#vdial").addEventListener("input", (e) => vdSet(+e.target.value, false));
$("#vdial").addEventListener("change", (e) => vdSet(+e.target.value, true));
$("#vdial-play").addEventListener("click", vdPlay);
$("#vdial-rand").addEventListener("click", () => { let k; do k = Math.floor(Math.random() * 5); while (k === vdI); vdSet(k, true); });
(async () => { VD = await J("data/vdial.json"); vdial(); redraw.push(vdial); })();

/* ---------- language model chapter ---------- */
const LMX = [
  ["தொலைக்காட்சியில் ஒலிபரப்பப்பட்ட", "தொலைக்காட்சியில் ஒளிபரப்பப்பட்ட", { en: "\"on television\": a TV broadcast is ஒளிபரப்பு (light), radio is ஒலிபரப்பு (sound). Both are real words. Only the context decides.", ta: "\"தொலைக்காட்சியில்\" என்று வருகிறது. தொலைக்காட்சி என்றால் ஒளிபரப்பு (ஒளி), வானொலி என்றால் ஒலிபரப்பு (ஒலி). இரண்டுமே சரியான சொற்கள். எது என்பதைச் சூழல்தான் தீர்மானிக்கும்." }],
  ["கருவரை வாயிலிலே", "கருவறை வாயிலிலே", { en: "the sanctum of a temple is கருவறை.", ta: "கோயிலின் சன்னதியைக் குறிக்கும் சொல் கருவறை." }],
  ["அந்த அரக்கட்டளை", "அந்த அறக்கட்டளை", { en: "a charitable trust is அறக்கட்டளை (அறம், virtue).", ta: "தொண்டு நிறுவனம் அறக்கட்டளை (அறம் என்ற சொல்லிலிருந்து)." }],
  ["நிளம் சமூகத்திலிருந்து", "நிலம் சமூகத்திலிருந்து", { en: "land is நிலம்; நிளம் is not a word.", ta: "நிலம் என்பதே சரி. நிளம் என்று ஒரு சொல்லே இல்லை." }],
];
let lmI = 0;
const hl = (x, y, c) => { const yw = y.split(" "); return x.split(" ").map((w, i) => (w === yw[i] ? esc(w) : `<b style="color:var(${c})">${esc(w)}</b>`)).join(" "); };
function lmCard() {
  const [a, b, g] = LMX[lmI];
  $("#lm-card").innerHTML = `<div class="pair" style="font-size:1.35rem;grid-template-columns:7rem 1fr"><span class="who">${L({ en: "heard", ta: "கேட்டது" })}</span><span>${hl(a, b, "--bad")}</span><span class="who">${L({ en: "with LM", ta: "மொழி மாதிரியுடன்" })}</span><span>${hl(b, a, "--good")}</span></div><p style="margin:0">${esc(L(g))}</p><div class="small mono" style="margin-top:.6rem">${lmI + 1} / ${LMX.length}</div>`;
}
$("#lm-next").addEventListener("click", () => { lmI = (lmI + 1) % LMX.length; lmCard(); });
lmCard(); redraw.push(lmCard);
function lmChart() {
  if (!SC) return;
  const S = SC.scores, READ = ["eval_kathbath", "eval_mile_ta", "eval_fleurs_ta", "eval_cv_ta", "eval_openslr65"], av = (m, k) => READ.reduce((a, s) => a + S[m][s][k], 0) / READ.length;
  const rows = [
    [{ en: "Sounds misheard, reading aloud", ta: "தவறாகக் கேட்ட ஒலிகள், உரக்க வாசித்தல்" }, av("e5", "per"), av("e5_lm", "per")],
    [{ en: "Meaning drift, reading aloud", ta: "பொருள் மாற்றம், உரக்க வாசித்தல்" }, av("e5", "eer"), av("e5_lm", "eer")],
    [{ en: "Meaning drift, conversation", ta: "பொருள் மாற்றம், உரையாடல்" }, S.e5.eval_indicvoices.eer, S.e5_lm.eval_indicvoices.eer],
  ];
  const nar = isNarrow(), W = fitW($("#lm-chart"), 520), heads = rows.map(([n]) => (nar ? wrapText(L(n), W, 12) : [L(n)]));
  const hh = heads.map((h) => (h.length - 1) * 15), mx = Math.max(...rows.flatMap((r) => [r[1], r[2]])) * 1.2, x = (v) => (v / mx) * (W - 170);
  const H = rows.length * 76 + hh.reduce((a, b) => a + b, 0) + 10;
  let g = fitSvg(W, H, "effect of the language model"), y = 4;
  rows.forEach(([n, a, b], i) => {
    g += `<text x="0" y="${y + 14}" class="lbl"${nar ? ' style="font-size:12px"' : ""}>${tspans(heads[i], 0, 15)}</text>`;
    const o = y + hh[i];
    g += `<text x="0" y="${o + 36}">${L({ en: "without", ta: "இல்லாமல்" })}</text><rect x="110" y="${o + 24}" width="${x(a)}" height="16" rx="4" fill="var(--ours)" opacity=".55"/><text x="${116 + x(a)}" y="${o + 37}" style="fill:var(--fg);font-weight:700">${a.toFixed(2)}</text>`;
    g += `<text x="0" y="${o + 58}">${L({ en: "with LM", ta: "உடன்" })}</text><rect x="110" y="${o + 46}" width="${x(b)}" height="16" rx="4" fill="var(--ours)"/><text x="${116 + x(b)}" y="${o + 59}" style="fill:var(--fg);font-weight:700">${b.toFixed(2)}</text>`;
    y += 76 + hh[i];
  });
  $("#lm-chart").innerHTML = `<div class="small" style="margin-bottom:.4rem">${L({ en: "It fixes spelling and meaning, not hearing. Lower is better.", ta: "மொழி மாதிரி எழுத்துப் பிழையையும் பொருளையும் சரிசெய்கிறது. காதால் கேட்பதை அல்ல. எண் குறைவாக இருந்தால் நல்லது." })}</div><div class="chart">${g}</svg></div>`;
  const sets = Object.keys(SC.sets).filter((s) => SC.sets[s].kind !== "english");
  let t = `<table class="dt"><thead><tr><th style="text-align:left">${L({ en: "test set", ta: "தேர்வு" })}</th>${["WER", "CER", "PER", "EER"].map((m) => `<th>${m}</th><th>${m} + LM</th>`).join("")}</tr></thead><tbody>`;
  sets.forEach((s) => { t += `<tr><td>${esc(L(SC.sets[s]))}</td>${["wer", "cer", "per", "eer"].map((m) => { const a = S.e5[s][m], b = S.e5_lm[s][m]; return `<td>${a.toFixed(2)}</td><td class="${b < a ? "best" : ""}">${b.toFixed(2)}</td>`; }).join("")}</tr>`; });
  $("#lm-table").innerHTML = t + "</tbody></table>";
}
(function waitLM() { if (SC) { lmChart(); redraw.push(lmChart); } else setTimeout(waitLM, 100); })();

/* ---------- same family: keyboard, our LM, ChatGPT/Claude ---------- */
function family() {
  const F = [
    { r: 26, t: { en: "phone keyboard", ta: "கைபேசி விசைப்பலகை" }, d: { en: "guesses your next word from the last few", ta: "நீங்கள் தட்டச்சு செய்த கடைசி சில சொற்களை வைத்து, அடுத்த சொல்லை ஊகிக்கிறது" } },
    { r: 40, t: { en: "our language model", ta: "எங்கள் மொழி மாதிரி" }, d: { en: "looks at the last 4 words · read 8 million Tamil sentences · runs on a home computer", ta: "கடைசி 4 சொற்களைப் பார்க்கிறது · 80 லட்சம் தமிழ் வாக்கியங்களைப் படித்தது · வீட்டுக் கணினியில் இயங்குகிறது" } },
    { r: 110, t: { en: "ChatGPT, Claude", ta: "ChatGPT, Claude" }, d: { en: "look at the whole conversation · read a large part of the internet · run in huge data centres", ta: "முழு உரையாடலையும் பார்க்கின்றன · இணையத்தின் பெரும் பகுதியைப் படித்தவை · பெரிய தரவு மையங்களில் இயங்குகின்றன" } },
  ];
  const W = 1000, H = 360, cx = [150, 420, 770], base = 250;
  let g = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="language model family">`;
  g += `<text x="${W / 2}" y="22" text-anchor="middle" class="lbl" style="font-size:15px">${esc(L({ en: "Same family: every one of them guesses the next word.", ta: "எல்லாம் ஒரே குடும்பம்: எல்லாமே அடுத்த சொல்லை ஊகிக்கின்றன." }))}</text>`;
  F.forEach((f, i) => {
    g += `<circle cx="${cx[i]}" cy="${base - f.r}" r="${f.r}" fill="var(--ours)" opacity="${0.35 + i * 0.25}"/>`;
    g += `<text x="${cx[i]}" y="${base + 30}" text-anchor="middle" class="lbl">${esc(L(f.t))}</text>`;
    g += `<foreignObject x="${cx[i] - 150}" y="${base + 38}" width="300" height="70"><div xmlns="http://www.w3.org/1999/xhtml" style="text-align:center;font-size:12px;color:var(--fg-3);line-height:1.4">${esc(L(f.d))}</div></foreignObject>`;
  });
  g += `<line x1="40" x2="${W - 40}" y1="${base}" y2="${base}" stroke="var(--rule)"/><text x="${W - 4}" y="${H - 4}" text-anchor="end" style="font-size:10px">${L({ en: "illustration, not to scale", ta: "விளக்கத்துக்கான படம்; அளவுகள் உண்மையான விகிதத்தில் இல்லை" })}</text>`;
  $("#family").innerHTML = `<div class="chart">${g}</svg></div>`;
}
/* ---------- next word: real counts from the language model's text ---------- */
const NW = [
  { c: "தொலைக்காட்சியில்", g: { en: "on television", ta: "தொலைக்காட்சியில்" }, a: 754, b: 1 },
  { c: "வானொலியில்", g: { en: "on the radio", ta: "வானொலியில்" }, a: 1, b: 39 },
];
let nwI = 0;
function nextword() {
  const n = NW[nwI], mx = Math.max(n.a, n.b);
  const row = (w, gl, v) => `<div class="lane" style="grid-template-columns:12rem 1fr 5rem"><span class="name ta" style="font-size:1.15rem">${w}<br><small class="dim">${esc(L(gl))}</small></span><div class="tr"><i style="width:${Math.max(0.6, (100 * v) / mx)}%;background:var(--ours)"></i></div><span class="t">${v.toLocaleString()}×</span></div>`;
  $("#nextword").innerHTML = `<div class="controls">${NW.map((x, i) => `<button data-nw="${i}" aria-pressed="${i === nwI}"><span class="ta">${x.c}</span>${L({ en: ` · ${esc(x.g.en)}`, ta: "" })}</button>`).join("")}</div>
    <div class="ta" style="font-size:1.4rem;margin:.6rem 0 .8rem">${n.c} <span class="dim">___</span></div>
    ${row("ஒளிபரப்பு…", { en: "broadcast (light: TV)", ta: "ஒளி: தொலைக்காட்சி" }, n.a)}${row("ஒலிபரப்பு…", { en: "broadcast (sound: radio)", ta: "ஒலி: வானொலி" }, n.b)}
    <div class="small" style="margin-top:.6rem">${L({ en: "How often Tamil writers used each word after these words, in the 8 million sentences our language model read.", ta: "எங்கள் மொழி மாதிரி படித்த 80 லட்சம் வாக்கியங்களில், இந்தச் சொற்களுக்குப் பிறகு, தமிழில் எழுதுபவர்கள் ஒவ்வொரு சொல்லையும் எத்தனை முறை பயன்படுத்தினார்கள் என்பது." })}</div>`;
  $$("[data-nw]").forEach((x) => x.addEventListener("click", () => { nwI = +x.dataset.nw; nextword(); }));
}
family(); nextword(); redraw.push(family, nextword);

/* ---------- loanwords: the rule, and the traps ---------- */
const LOANS = [
  ["நாளைக்கு மீட்டிங்க்கு நீ வர்றியா", "நாளைக்கு meeting-க்கு நீ வர்றியா"],
  ["பையன் ஆஃபீஸ்ல சப்பாத்தி இவ்ளோ சாஃப்டா", "பையன் office-ல சப்பாத்தி இவ்ளோ soft-ஆ"],
  ["சாட்டர்டே சண்டே லீவ் போடுங்க", "saturday sunday leave போடுங்க"],
];
function loans() {
  const mark = (x, y, c) => { const yw = y.split(" "); return x.split(" ").map((w, i) => (w !== yw[i] ? `<b style="color:var(${c})">${esc(w)}</b>` : esc(w))).join(" "); };
  $("#loan-lines").innerHTML = `<div class="loanhead"><span>${L({ en: "how most datasets write it", ta: "பெரும்பாலான பதிவுத் தொகுப்புகள் எழுதுவது" })}</span><span></span><span>${L({ en: "our rule", ta: "எங்கள் விதி" })}</span></div>` +
    LOANS.map(([x, y]) => `<div class="loanrow"><span class="ta">${mark(x, y, "--itcore")}</span><span class="arr">→</span><span class="ta">${mark(y, x, "--ours-2")}</span></div>`).join("");
  const TR = [["போன", { en: "\"went\", not phone", ta: "போன = சென்ற; phone அல்ல" }], ["கால", { en: "\"leg\" or \"time\", not call", ta: "கால = கால்/காலம்; call அல்ல" }], ["சீனா", { en: "China, not scene", ta: "சீனா நாடு; scene அல்ல" }], ["சாப்ட்டு", { en: "\"having eaten\", not soft", ta: "சாப்ட்டு = சாப்பிட்டு; soft அல்ல" }], ["பேசிக் கொண்டு", { en: "\"while speaking\", not basic", ta: "பேசிக்கொண்டு இருக்கும்போது; basic அல்ல" }], ["போலிச் சாமியார்", { en: "\"fake priest\", not police", ta: "போலி = பொய்யான; police அல்ல" }]];
  $("#loan-traps").innerHTML = TR.map(([w, g]) => `<button class="flip" style="height:104px"><div class="in"><div class="f"><span class="w">${w}</span><span class="x">↻</span></div><div class="b"><span class="g">${esc(L(g))}</span></div></div></button>`).join("");
}
loans(); redraw.push(loans);

/* ---------- flip cards: one delegated handler, so re-renders never double-bind ---------- */
document.addEventListener("click", (e) => { const f = e.target.closest(".flip"); if (f) f.classList.toggle("on"); });

/* ---------- story-mode bad label: the "okay mam" phone call ---------- */
(async () => {
  const d = await J("data/okm.json");
  $("#okm-words").classList.add("idle"); $("#okm-words").innerHTML = d.words.map(([w, s]) => `<span data-t="${s}">${esc(w)}</span>`).join(" ");
  const ws = $$("#okm-words span");
  wirePlayer({ url: "audio/spring_okay_mam.mp3", button: $("#okm-play"), canvas: $("#okm-canvas"),
    onTime: (tm) => { $("#okm-words").classList.toggle("idle", tm <= 0.01 || tm >= 1e8); ws.forEach((x, i) => { const on = +x.dataset.t <= tm + 0.05; x.classList.toggle("on", on); x.classList.toggle("now", on && (!ws[i + 1] || +ws[i + 1].dataset.t > tm + 0.05) && tm < 1e8); }); } });
})();

/* ---------- augmentation demo: WebAudio effects on a real clip ---------- */
const AUG = [
  ["noise", { en: "background noise", ta: "பின்னணிச் சத்தம்" }],
  ["echo", { en: "echo", ta: "எதிரொலி" }],
  ["phone", { en: "phone line", ta: "தொலைபேசி இணைப்பு" }],
  ["babble", { en: "other people talking", ta: "மற்றவர்கள் பேசும் சத்தம்" }],
  ["fast", { en: "10% faster", ta: "10% வேகமாக" }],
  ["slow", { en: "10% slower", ta: "10% மெதுவாக" }],
];
const augOn = new Set();
let augCtx = null, augG = null, augPlaying = false;
async function augBuf(url) { augCtx ||= new (window.AudioContext || window.webkitAudioContext)(); const b = await (await fetch(url)).arrayBuffer(); return augCtx.decodeAudioData(b); }
function augApply() {
  if (!augG) return;
  const now = augCtx.currentTime, set = (p, v) => p.setTargetAtTime(v, now, 0.03);
  augG.src.playbackRate.setValueAtTime(augOn.has("fast") ? 1.1 : augOn.has("slow") ? 0.9 : 1, now);
  set(augG.hp.frequency, augOn.has("phone") ? 300 : 10); set(augG.lp.frequency, augOn.has("phone") ? 3400 : 20000);
  set(augG.dry.gain, augOn.has("echo") ? 0.5 : 1); set(augG.wet.gain, augOn.has("echo") ? 0.6 : 0);
  set(augG.noise.gain, augOn.has("noise") ? 0.35 : 0); set(augG.babble.gain, augOn.has("babble") ? 0.3 : 0);
}
function augRender() {
  $("#aug-btns").innerHTML = AUG.map(([k, l]) => `<button data-aug="${k}" aria-pressed="${augOn.has(k)}">${esc(L(l))}</button>`).join("");
  $$("[data-aug]").forEach((b) => b.addEventListener("click", () => { const k = b.dataset.aug; if (augOn.has(k)) augOn.delete(k); else { augOn.add(k); if (k === "fast") augOn.delete("slow"); if (k === "slow") augOn.delete("fast"); } augRender(); augApply(); }));
  $("#aug-desc").textContent = augOn.size ? L({ en: `${augOn.size} effect${augOn.size > 1 ? "s" : ""} on`, ta: `${augOn.size} ${augOn.size > 1 ? "மாற்றங்கள் சேர்க்கப்பட்டுள்ளன" : "மாற்றம் சேர்க்கப்பட்டுள்ளது"}` }) : L({ en: "clean recording", ta: "தெளிவான பதிவு" });
  $("#aug-play").innerHTML = augPlaying ? "❚❚" : `▶ ${esc(L({ en: "play", ta: "கேளுங்கள்" }))}`;
}
function augStop() { if (augG) { [augG.src, augG.ns, augG.bs].forEach((n) => { try { n.stop(); } catch {} }); augG = null; } augPlaying = false; augRender(); }
async function augPlay() {
  if (augPlaying) { augStop(); return; }
  const main = await augBuf("audio/eval_indicvoices_0000_01_00082.mp3"), bab = await augBuf("audio/eval_indicvoices_0000_00_00698.mp3"), ctx = augCtx; await ctx.resume();
  const g = {}; const out = ctx.createGain(); out.connect(ctx.destination);
  g.src = ctx.createBufferSource(); g.src.buffer = main;
  g.hp = ctx.createBiquadFilter(); g.hp.type = "highpass"; g.lp = ctx.createBiquadFilter(); g.lp.type = "lowpass";
  const n = Math.floor(ctx.sampleRate * 0.9), ir = ctx.createBuffer(1, n, ctx.sampleRate), d = ir.getChannelData(0); for (let i = 0; i < n; i++) d[i] = (Math.random() * 2 - 1) * Math.exp((-6 * i) / n); d[0] = 1;
  const cv = ctx.createConvolver(); cv.buffer = ir; g.dry = ctx.createGain(); g.wet = ctx.createGain();
  g.src.connect(g.hp); g.hp.connect(g.lp); g.lp.connect(g.dry); g.lp.connect(cv); cv.connect(g.wet); g.dry.connect(out); g.wet.connect(out);
  const nn = Math.floor(ctx.sampleRate * 4), nb = ctx.createBuffer(1, nn, ctx.sampleRate), nd = nb.getChannelData(0); let br = 0; for (let i = 0; i < nn; i++) { br = 0.97 * br + 0.03 * (Math.random() * 2 - 1); nd[i] = br * 2.2; }
  g.ns = ctx.createBufferSource(); g.ns.buffer = nb; g.ns.loop = true; g.noise = ctx.createGain(); g.ns.connect(g.noise); g.noise.connect(out);
  g.bs = ctx.createBufferSource(); g.bs.buffer = bab; g.bs.loop = true; g.babble = ctx.createGain(); g.bs.connect(g.babble); g.babble.connect(out);
  [g.dry.gain, g.wet.gain, g.noise.gain, g.babble.gain].forEach((p) => (p.value = 0));
  augG = g; augApply();
  g.src.onended = () => { if (augG === g) augStop(); };
  const t0 = ctx.currentTime + 0.05; g.src.start(t0); g.ns.start(t0); g.bs.start(t0 + 0.4);
  augPlaying = true; augRender();
}
$("#aug-play").addEventListener("click", augPlay);
augRender(); redraw.push(augRender);

/* ---------- press and hold any text to peek at the other language ----------
   Nothing reflows: the original keeps its space (invisible), the translation floats on top, anchored so the
   pressed point stays put. Words are drawn as ghosts and morphed: shared words (numbers, English) glide to
   their new place, the rest dissolve out and write in with a lag, in the spirit of 3Blue1Brown's text morphs. */
(() => {
  const HOLD = 380, D = 620, SKIP = "button, a, input, select, summary, svg, canvas, audio, .bar, .opt, .flip, .qcard, [data-dflip], .sync";
  const ease = "cubic-bezier(.45, 0, .2, 1)";
  let timer = null, start = null, cur = null;
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const partner = (el) => {
    const other = el.dataset.l === "en" ? "ta" : "en", sib = [...el.parentElement.children];
    const mine = sib.filter((c) => c.dataset.l === el.dataset.l), theirs = sib.filter((c) => c.dataset.l === other);
    return theirs[mine.indexOf(el)] || null;
  };
  const words = (root) => {
    const out = [], w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    let n;
    while ((n = w.nextNode())) {
      const cs = getComputedStyle(n.parentElement), re = /[^ \t\n\r]+/g;
      let m;
      while ((m = re.exec(n.nodeValue))) {
        const r = document.createRange(); r.setStart(n, m.index); r.setEnd(n, m.index + m[0].length);
        const rs = [...r.getClientRects()].filter((x) => x.width > 0);
        if (!rs.length) continue;
        const b = r.getBoundingClientRect();
        out.push({ text: m[0], x: b.left + scrollX, y: b.top + scrollY, w: b.width, h: rs[0].height, cs, key: m[0].replace(/\u2060/g, "").match(/[A-Za-z0-9$%.,']+/)?.[0].replace(/[.,']+$/, "").toLowerCase() || null });
      }
    }
    return out;
  };
  const ghost = (layer, o) => {
    const g = document.createElement("span"), s = g.style, c = o.cs;
    g.textContent = o.text;
    Object.assign(s, { position: "absolute", left: o.x + "px", top: o.y + "px", lineHeight: o.h + "px", whiteSpace: "pre", fontFamily: c.fontFamily, fontSize: c.fontSize, fontWeight: c.fontWeight, fontStyle: c.fontStyle, letterSpacing: c.letterSpacing, textTransform: c.textTransform, color: c.color, transformOrigin: "0 50%", willChange: "transform, opacity, filter" });
    layer.appendChild(g);
    return g;
  };
  const bgOf = (el) => { for (let e = el; e; e = e.parentElement) { const b = getComputedStyle(e).backgroundColor; if (b && !/rgba\(.*,\s*0\)$|transparent/.test(b)) return b; } return getComputedStyle(document.body).backgroundColor; };
  const cleanup = (p) => {
    if (!p) return;
    p.anims.forEach((a) => a.cancel());
    p.layer.remove();
    p.src.style.visibility = "";
    p.dst.classList.remove("peek-show");
    ["position", "left", "top", "width", "margin", "visibility", "zIndex"].forEach((k) => (p.dst.style[k] = ""));
    document.documentElement.classList.remove("peeking");
    if (cur === p) cur = null;
  };
  const begin = (src, dst, px, py) => {
    const sr = src.getBoundingClientRect(), fy = Math.min(1, Math.max(0, (py - sr.top) / (sr.height || 1)));
    if (src.tagName === "IMG") {  // images have no words to morph: cross-fade the other language's picture over it
      dst.loading = "eager";
      dst.classList.add("peek-show", "in");
      Object.assign(dst.style, { position: "absolute", left: "0px", top: "0px", width: sr.width + "px", margin: "0", zIndex: "40" });
      const dr = dst.getBoundingClientRect();
      dst.style.left = sr.left - dr.left + "px"; dst.style.top = sr.top - dr.top + "px";
      const layer = document.createElement("div");
      document.body.appendChild(layer);
      document.documentElement.classList.add("peeking");
      const anims = [dst.animate([{ opacity: 0, filter: "blur(8px)" }, { opacity: 1, filter: "blur(0)" }], { duration: reduce ? 1 : D * 0.6, easing: ease, fill: "both" })];
      cur = { src, dst, layer, anims, ending: false };
      return;
    }
    // lay the translation out invisibly on top of the original, same width, no reflow
    dst.classList.add("peek-show", "in");
    Object.assign(dst.style, { position: "absolute", left: "0px", top: "0px", width: sr.width + "px", margin: "0", visibility: "hidden", zIndex: "40" });
    let dr = dst.getBoundingClientRect();
    const left = sr.left - dr.left, top = sr.top + fy * sr.height - fy * dr.height - dr.top;
    dst.style.left = left + "px"; dst.style.top = top + "px";
    dr = dst.getBoundingClientRect();
    const A = words(src), B = words(dst);
    const layer = document.createElement("div");
    Object.assign(layer.style, { position: "absolute", left: "0", top: "0", width: "0", height: "0", pointerEvents: "none", zIndex: "60" });
    document.body.appendChild(layer);
    const pad = 12, bx = Math.min(sr.left, dr.left) - pad + scrollX, by = dr.top - pad + scrollY;
    const back = document.createElement("div");
    Object.assign(back.style, { position: "absolute", left: bx + "px", top: by + "px", width: Math.max(sr.width, dr.width) + 2 * pad + "px", height: dr.height + 2 * pad + "px", background: bgOf(src.parentElement), borderRadius: "14px", boxShadow: "0 10px 40px rgba(0,0,0,.25)" });
    layer.appendChild(back);
    src.style.visibility = "hidden";
    document.documentElement.classList.add("peeking");
    window.getSelection()?.removeAllRanges();
    // pair up shared words (numbers, English words, "AI" in "AI-க்கு"), in reading order
    const used = new Set(), pair = new Map();
    B.forEach((b, j) => { if (!b.key) return; const i = A.findIndex((a, k) => !used.has(k) && a.key === b.key); if (i >= 0) { used.add(i); pair.set(j, i); } });
    const anims = [], T = reduce ? 1 : D, lag = (i, n) => (n > 1 ? (i / (n - 1)) * T * 0.45 : 0);
    const go = (el, kf, o) => anims.push(el.animate(kf, { duration: T, easing: ease, fill: "both", ...o }));
    go(back, [{ opacity: 0 }, { opacity: 1 }], { duration: T * 0.35 });
    A.forEach((a, i) => {
      const g = ghost(layer, a);
      if (used.has(i)) return go(g, [{ opacity: 1 }, { opacity: 0 }], { duration: T * 0.5 });
      go(g, [{ opacity: 1, transform: "none", filter: "blur(0)" }, { opacity: 0, transform: "translateY(-6px) scale(.96)", filter: "blur(5px)" }], { duration: T * 0.45, delay: lag(i, A.length) * 0.6 });
    });
    B.forEach((b, j) => {
      const g = ghost(layer, b);
      if (pair.has(j)) {
        const a = A[pair.get(j)], k = a.h / b.h;
        return go(g, [{ transform: `translate(${a.x - b.x}px, ${a.y - b.y}px) scale(${k})`, opacity: 0.85 }, { transform: "none", opacity: 1 }], { duration: T * 0.8, delay: T * 0.05 });
      }
      go(g, [{ opacity: 0, transform: "translateY(7px)", filter: "blur(6px)" }, { opacity: 1, transform: "none", filter: "blur(0)" }], { duration: T * 0.5, delay: T * 0.25 + lag(j, B.length) * 0.8 });
    });
    cur = { src, dst, layer, anims, ending: false };
  };
  const end = () => {
    clearTimeout(timer); timer = null;
    const p = cur;
    if (!p || p.ending) return;
    p.ending = true;
    p.anims.forEach((a) => { a.updatePlaybackRate(1.6); a.reverse(); });
    Promise.all(p.anims.map((a) => a.finished)).then(() => cleanup(p), () => {});
  };
  document.addEventListener("pointerdown", (e) => {
    if (cur) cleanup(cur);
    if (e.button !== 0 || e.target.closest(SKIP)) return;
    const el = e.target.closest("[data-l]");
    if (!el || el.dataset.l !== document.documentElement.lang) return;
    const p = partner(el);
    if (!p) return;
    if (p.tagName === "IMG") p.loading = "eager";
    start = [e.clientX, e.clientY];
    timer = setTimeout(() => { timer = null; begin(el, p, start[0], start[1]); }, HOLD);
  });
  document.addEventListener("pointermove", (e) => { if (timer && Math.hypot(e.clientX - start[0], e.clientY - start[1]) > 10) { clearTimeout(timer); timer = null; } });
  ["pointerup", "pointercancel", "blur"].forEach((ev) => window.addEventListener(ev, end, { passive: true }));
  document.addEventListener("contextmenu", (e) => { if (cur || timer) e.preventDefault(); });
  document.addEventListener("click", (e) => { if (cur) { e.preventDefault(); e.stopPropagation(); } }, true);
})();

/* ---------- ISO 15919 romanisation under Tamil words, like furigana ---------- */
const ISO_C = { "க": "k", "ங": "ṅ", "ச": "c", "ஞ": "ñ", "ட": "ṭ", "ண": "ṇ", "த": "t", "ந": "n", "ப": "p", "ம": "m", "ய": "y", "ர": "r", "ல": "l", "வ": "v", "ழ": "ḻ", "ள": "ḷ", "ற": "ṟ", "ன": "ṉ", "ஜ": "j", "ஷ": "ṣ", "ஸ": "s", "ஹ": "h", "ஶ": "ś" };
const ISO_V = { "அ": "a", "ஆ": "ā", "இ": "i", "ஈ": "ī", "உ": "u", "ஊ": "ū", "எ": "e", "ஏ": "ē", "ஐ": "ai", "ஒ": "o", "ஓ": "ō", "ஔ": "au", "ஃ": "ḵ" };
const ISO_S = { "ா": "ā", "ி": "i", "ீ": "ī", "ு": "u", "ூ": "ū", "ெ": "e", "ே": "ē", "ை": "ai", "ொ": "o", "ோ": "ō", "ௌ": "au", "ௗ": "" };
function iso(w) {
  const s = [...w.normalize("NFC")];
  let o = "";
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (ISO_C[ch]) {
      o += ISO_C[ch];
      const nx = s[i + 1];
      if (nx === "்") i++;
      else if (nx in ISO_S) { o += ISO_S[nx]; i++; }
      else o += "a";
    } else if (ISO_V[ch]) o += ISO_V[ch];
    else if (!(ch in ISO_S) && ch !== "்") o += ch;
  }
  return o;
}
const TA_RUN = /[\u0B80-\u0BFF]+/g, ISO_SKIP = "script, style, svg, rt, ruby, textarea, code, pre, .bar, [data-noiso]";
function rubify(root) {
  const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT), list = [];
  let n;
  while ((n = w.nextNode())) if (/[\u0B80-\u0BFF]/.test(n.nodeValue) && !n.parentElement.closest(ISO_SKIP)) list.push(n);
  list.forEach((n) => {
    const f = document.createDocumentFragment(), v = n.nodeValue;
    let last = 0;
    v.replace(TA_RUN, (m, at) => {
      if (at > last) f.append(v.slice(last, at));
      const r = document.createElement("ruby"), rt = document.createElement("rt");
      rt.textContent = iso(m); r.append(m, rt); f.append(r);
      last = at + m.length;
    });
    if (last < v.length) f.append(v.slice(last));
    n.replaceWith(f);
  });
}
function unrubify() {
  const parents = new Set();
  $$("ruby").forEach((r) => { if (!r.closest(ISO_SKIP.replace("ruby, ", "").replace("rt, ", ""))) { parents.add(r.parentNode); r.replaceWith(r.firstChild.nodeValue); } });
  parents.forEach((p) => p.normalize());
}
let isoOn = localStorage.getItem("iso") === "1";
function setIso(on) {
  isoOn = on; localStorage.setItem("iso", on ? "1" : "");
  document.documentElement.dataset.iso = on ? "1" : "";
  $("#iso").setAttribute("aria-pressed", String(on));
  if (on) rubify(document.body); else unrubify();
}
$("#iso").addEventListener("click", () => keepPlace(() => setIso(!isoOn)));
new MutationObserver((ms) => {
  if (!isoOn) return;
  for (const m of ms) m.addedNodes.forEach((n) => {
    if (n.nodeType === 1 && n.tagName !== "RUBY" && n.tagName !== "RT") rubify(n);
    else if (n.nodeType === 3 && n.parentElement && n.parentElement.tagName !== "RUBY" && n.parentElement.tagName !== "RT") rubify(n.parentElement);
  });
}).observe(document.body, { childList: true, subtree: true });
if (isoOn) setIso(true);

/* ---------- "it writes what you said": real clips, model tabs, 976 dots, then tidy ---------- */
let VB = null, vbClip = 0, vbModel = "itcore", vbDotModel = "itcore", vbPlayer = null;
const VBM = [["kili", { en: "ours", ta: "எங்களுடையது" }], ["itcore", { en: "indic-transcribe-core", ta: "indic-transcribe-core" }], ["hima", { en: "Hima-ASR", ta: "Hima-ASR" }], ["nemotron", { en: "Nemotron 3.5", ta: "Nemotron 3.5" }]];
function vbTabs(box, cur, onPick) {
  box.innerHTML = VBM.map(([k, n]) => `<button role="tab" data-k="${k}" aria-pressed="${k === cur}">${esc(L(n))}</button>`).join("");
  $$("button", box).forEach((b) => b.addEventListener("click", () => onPick(b.dataset.k)));
}
function vbRefTimes(c) {  // reference word i -> start time of the matching word in our transcript
  const ref = c.ref.split(" "), out = Array(ref.length).fill(null); let ri = 0, hi = 0;
  c.models.kili.forEach((x) => { if ("=wx".includes(x.k)) { out[ri++] = c.kili_words[hi++][1]; } else if (x.k === "-") ri++; else hi++; });
  for (let i = 0; i < out.length; i++) if (out[i] == null) out[i] = i ? out[i - 1] : 0;
  return out;
}
function vbOut(toks) {
  return toks.map((x) => {
    if (x.k === "=") return x.f ? `<span class="vb-f">${esc(x.t)}</span>` : esc(x.t);
    if (x.k === "w") return `<span class="vb-w"><s>${esc(x.r)}</s>${esc(x.t)}</span>`;
    if (x.k === "x") return latin(x.t) !== latin(x.r) ? `<span class="w-fold" title="${esc(x.r)}">${esc(x.t)}</span>` : `<span class="w-sub" title="${esc(x.r)}">${esc(x.t)}</span>`;
    if (x.k === "-") return `<span class="w-del">${esc(x.t)}</span>`;
    return `<span class="w-ins">${esc(x.t)}</span>`;
  }).join(" ");
}
function vbDraw() {
  if (!VB) return;
  const root = $("#vb-player"), c = VB.clips[vbClip];
  $(".vb-clips", root).innerHTML = VB.clips.map((x, i) => `<button role="tab" data-i="${i}" aria-pressed="${i === vbClip}">${esc(L(x.label))}</button>`).join("");
  $$(".vb-clips button", root).forEach((b) => b.addEventListener("click", () => { vbClip = +b.dataset.i; vbPlayer?.audio.pause(); vbPlayer = null; vbDraw(); }));
  if (!vbPlayer) {
    const head = $(".qhead", root);
    head.innerHTML = `<button class="qplay" aria-pressed="false" aria-label="play">▶</button><div class="mini"><canvas></canvas></div><span class="qsrc">${esc(c.source)}</span>`;
    const times = vbRefTimes(c);
    vbPlayer = wirePlayer({ url: c.audio, button: $(".qplay", head), canvas: $("canvas", head), onTime: (tt) => {
      const sp = $$(".vb-ref span", root); let last = -1;
      const end = tt >= 1e8;
      sp.forEach((s, i) => { const on = end || times[i] <= tt + 0.05; s.classList.toggle("on", on); if (on) last = i; });
      sp.forEach((s, i) => s.classList.toggle("now", !end && i === last));
    } });
  }
  $(".vb-ref", root).innerHTML = c.ref.split(" ").map((w) => `<span>${esc(w)}</span>`).join(" ");
  vbTabs($(".vb-models", root), vbModel, (k) => { vbModel = k; vbDraw(); });
  const toks = c.models[vbModel];
  $(".vb-out", root).innerHTML = `<span class="vb-lab">${esc(L(VBM.find((m) => m[0] === vbModel)[1]))}</span>${vbOut(toks)}`;
  const w = toks.filter((x) => x.k === "w").length, mis = toks.filter((x) => (x.k === "x" && latin(x.t) === latin(x.r)) || x.k === "-" || x.k === "+").length;
  $(".vb-count", root).textContent = L({ en: `${w} spoken ${w === 1 ? "word" : "words"} rewritten as written Tamil · ${mis} misheard`, ta: `${w} பேச்சுச் சொற்கள் எழுத்துத் தமிழாக மாற்றப்பட்டன · ${mis} தவறாகக் கேட்டவை` });
  // dots
  const D = $("#vb-dots"), s = VB.dots[vbDotModel];
  vbTabs($(".vb-models", D), vbDotModel, (k) => { vbDotModel = k; vbDraw(); });
  const g = $(".vb-grid", D);
  if (g.children.length !== s.length) g.innerHTML = Array.from(s, (_, i) => `<i style="transition-delay:${(i % 48) * 7}ms"></i>`).join("");
  Array.from(g.children).forEach((el, i) => { el.className = s[i]; });
  const n = (ch) => s.split(ch).length - 1;
  $(".vb-dotkey", D).innerHTML = `<span><i style="background:var(--itcore)"></i><b>${n("w")}</b>${L({ en: "rewritten as written Tamil", ta: "எழுத்துத் தமிழாக மாற்றப்பட்டவை" })}</span><span><i style="background:var(--fg-2);opacity:.55"></i><b>${n("k")}</b>${L({ en: "kept as said", ta: "சொன்னபடியே" })}</span><span><i style="box-shadow:inset 0 0 0 1.5px var(--fg-3)"></i><b>${n("o")}</b>${L({ en: "misheard or missed", ta: "தவறாகக் கேட்டவை, விடுபட்டவை" })}</span>`;
  // then tidy
  const P = $("#vb-pipe");
  $(".vb-p0", P).textContent = c.kili_words.map((x) => x[0]).join(" ");
  $(".vb-p1", P).textContent = c.tidy.formal;
  $(".vb-p2", P).textContent = c.tidy.english;
}
(async () => { try { VB = await J("data/verbatim.json"); vbDraw(); redraw.push(vbDraw); } catch (e) { console.warn(e); } })();
$$("[data-dialog]").forEach((b) => b.addEventListener("click", (e) => { e.preventDefault(); $("#" + b.dataset.dialog).showModal(); }));
$$(".techdlg").forEach((d) => { $(".techclose", d).addEventListener("click", () => d.close()); d.addEventListener("click", (e) => { if (e.target === d) d.close(); }); });
