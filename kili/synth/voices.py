"""Random Maya2 speakers and the label-free voice-range dial.

Voices: Maya2 (maya-research/maya2-native, Apache-2.0) has a speaker-vector conditioning mode ("spk": a 192-d ECAPA
vector, no reference audio). Each synthetic voice is a seeded random unit vector v in R^192, moved along a pitch
direction d:  voice = unit(v + a * 0.47 * d),  a ~ U[-1, 1]. Random vectors give new voices, not copies: their output
has max cosine 0.47 to any of Maya2's 20 preset voices (a preset's own centroid reproduces it at 0.77). No real
speaker's audio is cloned.

Direction d (`voice_range_dir.npy`, shipped): ridge regression of log-F0 on the output ECAPA of 748 Maya2-synthesised
clips (random, preset-mixture and preset voices; 8 Tamil / Tanglish sentences); no gender labels, no real audio.
Voice-grouped 5-fold CV R^2 of log-F0 from the 1-D projection: 0.68. Scale 0.47 = the spread of the preset voices'
projections. Sliding a random voice from a = -1 to +1 moves its median F0 from 112 to 271 Hz (real Tamil speakers,
2.5-97.5%: 104-266 Hz); intelligibility holds (PER <= 1.8% at a = +-0.5 and +-1, 3.2% for the unsteered a = 0).

Validation on real voices (measured only; nothing is synthesised from them): the projection on d orders real Tamil
speakers by F0 at Spearman 0.77 (ceiling 0.87 for a direction fit on the real voices themselves) and separates the
female/male labels at AUC 0.92 (Indian-English speakers: 0.73 / 0.90).

  python -m kili.synth.voices probe ./probe                      # 748 Maya2 clips + output ECAPA (GPU, maya3)
  python -m kili.synth.voices fit ./probe                        # -> ./probe/voice_range_dir.npy, fit.json
  python -m kili.synth.voices sweep ./sweep                      # F0 by a for 6 random voices (GPU, maya3)
  python -m kili.synth.voices validate --manifest real.jsonl --ecapa-dir ECAPA_DIR --out validation.json
      real.jsonl: rows {audio_filepath (16 kHz), gender (female/male), speaker_id?, duration?} from a public corpus
      with speaker gender labels (e.g. Kathbath, Svarah); ECAPA_DIR: a snapshot of speechbrain/spkrec-ecapa-voxceleb.

Maya2 runtime: `maya3` from github.com/MayaResearch/maya2-native (vetted at commit c6b376868e26), weights
maya-research/maya2-native @ 87a83efdd6551e038246b6b37836a6264713eec3, kyutai/mimi @
89091b3e466eb6a9d11e537bf26b144f194978f7, speechbrain/spkrec-ecapa-voxceleb @ 0f99f2d0ebe89ac095bcc5903c4dd8f72b367286.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random

import numpy as np

DIM = 192
SCALE = 0.47
A_MAX = 1.0
DIRECTION = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_range_dir.npy")
MAYA_REPO = "maya-research/maya2-native"
TA = ["இன்று காலை மழை பெய்ததால் பள்ளிக்கு செல்ல தாமதமானது.",
      "என் அம்மா சமைத்த சாம்பார் மிகவும் சுவையாக இருந்தது.",
      "நாளை மாலை நாங்கள் கோயிலுக்கு போகலாம் என்று நினைக்கிறேன்.",
      "இந்த புத்தகத்தை படித்து முடிக்க எனக்கு இரண்டு வாரங்கள் ஆனது."]
TANGLISH = ["நாளைக்கு office-ல ஒரு important meeting இருக்கு, அதனால late ஆ வருவேன்.",
            "இந்த phone-ஓட battery ரொம்ப சீக்கிரம் drain ஆகுது.",
            "நீ அந்த email-ஐ manager-க்கு forward பண்ணிட்டியா?",
            "weekend-ல நாங்க friends கூட movie பார்க்க போறோம்."]
A_LEVELS = [-1.0, -0.5, 0.0, 0.5, 1.0]


# ------------------------------------------------------------------------------------------------- sampling
def unit(v) -> np.ndarray:
    v = np.asarray(v, np.float64)
    return v / np.linalg.norm(v)


def load_direction(path: str | None = None) -> np.ndarray:
    return unit(np.load(path or DIRECTION).astype(np.float64))


def voice_seed(source: str, voice_pass: int, shard: int, slot: int) -> int:
    """Stable 63-bit seed per (source, pass, shard, slot): a voice is rebuilt from these four values alone."""
    return int.from_bytes(hashlib.sha256(f"{source}|{voice_pass}|{shard}|{slot}".encode()).digest()[:8], "big") >> 1


def random_voice(seed: int, d: np.ndarray, scale: float = SCALE, a_max: float = A_MAX) -> tuple[np.ndarray, float]:
    """(speaker vector, a): unit(v + a * scale * d), v a uniform random unit vector, a ~ U[-a_max, a_max]."""
    rng = np.random.default_rng(seed)
    base = unit(rng.standard_normal(DIM))
    a = float(rng.uniform(-a_max, a_max))
    return unit(base + a * scale * d).astype(np.float32), a


# ------------------------------------------------------------------------------------------------ pitch (F0)
def f0_median(y: np.ndarray, sr: int = 16000) -> float | None:
    """Median voiced F0 (Hz) by normalised autocorrelation: 40 ms frames, 10 ms hop, 60-400 Hz, frames quieter than
    10% of the clip RMS skipped, voiced if the peak correlation > 0.6; None with <= 10 voiced frames."""
    y = np.asarray(y, np.float32)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        import soxr
        y, sr = soxr.resample(y, sr, 16000), 16000
    n, hop = int(0.04 * sr), int(0.01 * sr)
    lo, hi = sr // 400, sr // 60
    thr = 0.1 * np.sqrt(np.mean(y ** 2)) if len(y) else 0
    f0s = []
    for i in range(0, len(y) - n - hi, hop):
        x = y[i:i + n + hi]
        if np.sqrt(np.mean(x[:n] ** 2)) < thr:
            continue
        x = x - x.mean()
        a = x[:n]
        num = np.array([np.dot(a, x[k:k + n]) for k in range(lo, hi)])
        den = np.sqrt(np.dot(a, a) * np.array([np.dot(x[k:k + n], x[k:k + n]) for k in range(lo, hi)])) + 1e-9
        r = num / den
        k = int(np.argmax(r))
        if r[k] > 0.6:
            f0s.append(sr / (lo + k))
    return float(np.median(f0s)) if len(f0s) > 10 else None


def f0_file(path: str) -> float | None:
    import soundfile as sf
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    return f0_median(y.mean(axis=1), sr)


# --------------------------------------------------------------------------------------------- direction fit
def ridge_dir(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    Xc, yc = X - X.mean(0), y - y.mean()
    return np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ yc)


def fit_direction(X: np.ndarray, log_f0: np.ndarray, groups, lam: float = 1.0, folds: int = 5,
                  seed: int = 0) -> tuple[np.ndarray, float]:
    """Label-free direction: unit ridge coefficients of log-F0 on unit ECAPA rows, oriented + = higher pitch.
    Also returns the group-wise (voice-wise) k-fold CV R^2 of log-F0 predicted from the 1-D projection."""
    X = np.stack([unit(x) for x in X])
    groups = np.asarray(groups)
    g = sorted(set(groups.tolist()))
    np.random.default_rng(seed).shuffle(g)
    fold_of = {v: i % folds for i, v in enumerate(g)}
    fid = np.array([fold_of[v] for v in groups])
    pred, true = [], []
    for f in range(folds):
        tr, te = fid != f, fid == f
        if not te.any():
            continue
        d = unit(ridge_dir(X[tr], log_f0[tr], lam))
        A = np.vstack([X[tr] @ d, np.ones(tr.sum())]).T
        c = np.linalg.lstsq(A, log_f0[tr], rcond=None)[0]
        pred += list(c[0] * (X[te] @ d) + c[1]); true += list(log_f0[te])
    pred, true = np.array(pred), np.array(true)
    r2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - true.mean()) ** 2)
    d = unit(ridge_dir(X, log_f0, lam))
    if np.corrcoef(X @ d, log_f0)[0, 1] < 0:
        d = -d
    return d, float(r2)


# ------------------------------------------------------------------------------------------ real-voice checks
def ranks(a) -> np.ndarray:
    a = np.asarray(a, np.float64)
    r = np.empty(len(a)); r[np.argsort(a, kind="mergesort")] = np.arange(len(a))
    for v in np.unique(a):  # average ties
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(x, y) -> float:
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def auc(score, positive) -> float:
    """P(score of a positive > score of a negative), ties count half (Mann-Whitney)."""
    positive = np.asarray(positive).astype(bool)
    r = ranks(score); n1 = positive.sum(); n0 = len(positive) - n1
    return float((r[positive].sum() - n1 * (n1 - 1) / 2) / (n1 * n0))


def evaluate_direction(X: np.ndarray, f0: np.ndarray, female: np.ndarray, d: np.ndarray, seed: int = 0) -> dict:
    """How well `d` orders real voices by F0 (Spearman) and separates the gender labels (AUC), against ceilings from
    directions fit on the real voices themselves under 5-fold CV (female-minus-male mean; ridge of log-F0)."""
    X = np.stack([unit(x) for x in X]); f0 = np.asarray(f0, np.float64); female = np.asarray(female).astype(bool)
    ok = ~np.isnan(f0)
    res = {"n_female": int(female.sum()), "n_male": int((~female).sum()),
           "direction": {"auc_female_vs_male": round(auc(X @ d, female), 3),
                         "spearman_f0": round(spearman((X @ d)[ok], f0[ok]), 3)}}
    perm = np.random.default_rng(seed).permutation(len(X))
    for name in ("real_mean_diff", "real_ridge_logf0"):
        sc = np.full(len(X), np.nan)
        for k in range(5):
            te = perm[k::5]; tr = np.setdiff1d(perm, te)
            if name == "real_mean_diff":
                dv = unit(X[tr][female[tr]].mean(0) - X[tr][~female[tr]].mean(0))
            else:
                t = tr[ok[tr]]
                dv = unit(ridge_dir(X[t], np.log2(f0[t]), 1.0))
            t = tr[ok[tr]]
            if np.corrcoef(X[t] @ dv, np.log2(f0[t]))[0, 1] < 0:
                dv = -dv
            sc[te] = X[te] @ dv
        res[name + " (5-fold CV ceiling)"] = {"auc_female_vs_male": round(auc(sc, female), 3),
                                              "spearman_f0": round(spearman(sc[ok], f0[ok]), 3)}
    res["f0_alone_auc"] = round(auc(np.where(ok, f0, np.nanmedian(f0)), female), 3)
    res["f0_hz_percentiles"] = {f"p{q}": round(float(np.nanpercentile(f0, q)), 1) for q in (2.5, 50, 97.5)}
    return res


# ------------------------------------------------------------------------------------------- Maya2 commands
def _maya(seats: int = 16):
    from maya3 import Maya3TTS
    tts = Maya3TTS.from_pretrained(MAYA_REPO, max_seats=seats)
    tts.generate("வணக்கம்.", lang="ta", voice="Kayal")  # warm-up
    return tts


def _synth_rows(tts, jobs, out):
    """jobs: (strategy, vid, voice, i, text, seed) -> rows with the wav path and the output's ECAPA."""
    import soundfile as sf
    from concurrent.futures import ThreadPoolExecutor

    def one(j):
        s, vid, voice, i, text, seed = j
        w = tts.generate(text, lang="ta", voice=voice, seed=seed)
        d = os.path.join(out, "wav", s); os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f"{vid}_{i}.wav")
        sf.write(p, w, 24000, subtype="PCM_16")
        e = tts.engine.call(lambda: tts._embed(w)).tolist() if len(w) >= 2400 else None  # Maya2's own ECAPA
        return {"strategy": s, "vid": vid, "i": i, "text": text, "path": p, "dur": round(len(w) / 24000, 3), "ecapa": e}
    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(one, jobs))


def probe(out: str, k: int = 12) -> None:
    """The 748 clips the direction was fit on: preset centroids (20 x 3), five sampling strategies (12 voices x 8
    sentences each: uniform random; mixtures of 2-3 preset centroids + small / wide noise; Gaussian fit to the
    centroids; centroid mean + large noise), spk mode fed preset centroids, the presets themselves, no speaker."""
    os.makedirs(out, exist_ok=True)
    tts = _maya()
    texts = TA + TANGLISH
    named = [v for v in tts.voices if v != "plain"]
    crow = _synth_rows(tts, [("centroid", n, n, i, TA[i], 100 + i) for n in named for i in range(3)], out)
    cent = {n: unit(np.mean([r["ecapa"] for r in crow if r["vid"] == n and r["ecapa"]], 0)) for n in named
            if any(r["vid"] == n and r["ecapa"] for r in crow)}
    C = np.stack(list(cent.values()))
    rng = np.random.default_rng(20261111)
    req = {}
    for j in range(k):
        req[("A", f"A{j:02d}")] = unit(rng.standard_normal(DIM))
        m = rng.choice(len(C), size=rng.integers(2, 4), replace=False)
        b = unit((rng.dirichlet(np.ones(len(m)))[:, None] * C[m]).sum(0))
        req[("B", f"B{j:02d}")] = unit(b + rng.standard_normal(DIM) / np.sqrt(DIM) * 0.3)
    mu = C.mean(0)
    cov = np.cov((C - mu).T)
    cov = 0.7 * cov + 0.3 * np.trace(cov) / DIM * np.eye(DIM)
    for j in range(k):
        req[("C", f"C{j:02d}")] = unit(rng.multivariate_normal(mu, cov))
    for j in range(k):
        m = rng.choice(len(C), size=rng.integers(2, 4), replace=False)
        b = unit((rng.dirichlet(np.ones(len(m)))[:, None] * C[m]).sum(0))
        req[("B_wide", f"W{j:02d}")] = unit(b + rng.standard_normal(DIM) / np.sqrt(DIM) * 0.7)
        req[("D", f"D{j:02d}")] = unit(unit(mu) + rng.standard_normal(DIM) / np.sqrt(DIM) * 1.0)
    for n in list(cent)[:k]:
        req[("spk_centroid", f"S_{n}")] = cent[n]
    for (s, vid), v in req.items():
        tts.register_voice(vid, mode="spk", ecapa=v.astype(np.float32))
    jobs = [(s, vid, vid, i, t, 1000 + i) for (s, vid) in req for i, t in enumerate(texts)]
    jobs += [("named", n, n, i, t, 1000 + i) for n in list(cent)[:k] for i, t in enumerate(texts)]
    jobs += [("plain", f"P{j}", "plain", i, t, 2000 + 37 * j + i) for j in range(2) for i, t in enumerate(texts)]
    rows = _synth_rows(tts, jobs, out)
    tts.close()
    with open(os.path.join(out, "gen.jsonl"), "w", encoding="utf-8") as f:
        for r in crow + rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[probe] {len(crow) + len(rows)} clips -> {out}/gen.jsonl", flush=True)


def fit(out: str) -> None:
    from concurrent.futures import ProcessPoolExecutor
    rows = [json.loads(x) for x in open(os.path.join(out, "gen.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if r.get("ecapa") and os.path.exists(r["path"])]
    with ProcessPoolExecutor() as ex:
        f0 = list(ex.map(f0_file, [r["path"] for r in rows], chunksize=8))
    rows, f0 = [r for r, f in zip(rows, f0) if f], np.array([f for f in f0 if f])
    X = np.stack([unit(r["ecapa"]) for r in rows])
    d, r2 = fit_direction(X, np.log2(f0), [f"{r['strategy']}/{r['vid']}" for r in rows])
    preset = {}
    for r, x in zip(rows, X):
        if r["strategy"] in ("centroid", "named"):
            preset.setdefault(r["vid"], []).append(x)
    proj = [float(unit(np.mean(v, 0)) @ d) for v in preset.values()]
    half = (max(proj) - min(proj)) / 2
    np.save(os.path.join(out, "voice_range_dir.npy"), d.astype(np.float32))
    res = {"n_clips": len(rows), "cv_r2_logf0": round(r2, 3), "preset_projection_half_range": round(half, 3),
           "scale": round(2 * half, 3)}
    json.dump(res, open(os.path.join(out, "fit.json"), "w"), indent=1)
    print(json.dumps(res), flush=True)


def sweep(out: str, direction: str | None, scale: float) -> None:
    """Causal check: 6 random voices x a in {-1, -0.5, 0, 0.5, 1} x 4 sentences; median F0 per a."""
    os.makedirs(out, exist_ok=True)
    d = load_direction(direction)
    rng = np.random.default_rng(20261112)
    bases = {f"R{j}": unit(rng.standard_normal(DIM)) for j in range(6)}
    texts = [TA[0], TA[2], TANGLISH[0], TANGLISH[2]]
    tts = _maya()
    jobs, a_of = [], {}
    for b, v in bases.items():
        for a in A_LEVELS:
            vid = f"{b}_a{a:+.1f}".replace("+", "p").replace("-", "m")
            tts.register_voice(vid, mode="spk", ecapa=unit(v + a * scale * d).astype(np.float32))
            a_of[vid] = a
            jobs += [("sweep", vid, vid, i, t, 3000 + i) for i, t in enumerate(texts)]
    rows = _synth_rows(tts, jobs, out)
    tts.close()
    for r in rows:
        r["f0"] = f0_file(r["path"])
    res = {}
    for a in A_LEVELS:
        fs = [r["f0"] for r in rows if a_of[r["vid"]] == a and r["f0"]]
        res[f"a={a:+.1f}"] = {"clips": sum(a_of[r["vid"]] == a for r in rows),
                              "median_f0_hz": round(float(np.median(fs)), 1) if fs else None}
    with open(os.path.join(out, "sweep.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("vid", "i", "text", "path", "f0")}, ensure_ascii=False) + "\n")
    json.dump(res, open(os.path.join(out, "sweep.json"), "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


def validate(manifest: str, ecapa_dir: str, out: str, direction: str | None, per_gender: int, seed: int = 7) -> None:
    """Real-voice check of the direction: one clean 3-20 s clip per speaker, ECAPA with Maya2's extractor (16 kHz
    path) and F0. Writes aggregate numbers only."""
    import soundfile as sf
    import torch
    from maya3.api import Ecapa
    rows = [json.loads(x) for x in open(manifest, encoding="utf-8") if x.strip()]
    by_spk: dict = {}
    for i, r in enumerate(rows):
        g = str(r.get("gender", "")).lower()[:1]
        if g not in ("f", "m") or not 3.0 <= float(r.get("duration", 5.0)) <= 20.0:
            continue
        by_spk.setdefault((g, r.get("speaker_id") or f"row{i}"), []).append(r)
    rng = random.Random(seed)
    chosen = []
    for g in ("f", "m"):
        keys = sorted(k for k in by_spk if k[0] == g)
        rng.shuffle(keys)
        chosen += [(g, rng.choice(by_spk[k])) for k in keys[:per_gender or None]]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ec = Ecapa(ecapa_dir, dev)
    X, f0, female = [], [], []
    for g, r in chosen:
        y, sr = sf.read(r["audio_filepath"], dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        assert sr == 16000, (r["audio_filepath"], sr)
        with torch.inference_mode():  # Ecapa.__call__ without its 24 -> 16 kHz resample
            w = torch.from_numpy(y).to(dev)[None]
            L = torch.ones(1, device=dev)
            e = ec.emb(ec.norm(ec.fbank(w), L), L).reshape(-1)
        X.append(torch.nn.functional.normalize(e, dim=0).cpu().numpy())
        f = f0_median(y, sr)
        f0.append(np.nan if f is None else f)
        female.append(g == "f")
    res = evaluate_direction(np.stack(X), np.array(f0), np.array(female), load_direction(direction))
    res["manifest"] = os.path.basename(manifest)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe").add_argument("out")
    sub.add_parser("fit").add_argument("out")
    s = sub.add_parser("sweep")
    s.add_argument("out")
    s.add_argument("--direction", default=None, help="default: the shipped voice_range_dir.npy")
    s.add_argument("--scale", type=float, default=SCALE)
    v = sub.add_parser("validate")
    v.add_argument("--manifest", required=True)
    v.add_argument("--ecapa-dir", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--direction", default=None)
    v.add_argument("--per-gender", type=int, default=200, help="speakers per gender (0 = all)")
    a = ap.parse_args()
    if a.cmd == "probe":
        probe(a.out)
    elif a.cmd == "fit":
        fit(a.out)
    elif a.cmd == "sweep":
        sweep(a.out, a.direction, a.scale)
    else:
        validate(a.manifest, a.ecapa_dir, a.out, a.direction, a.per_gender)


if __name__ == "__main__":
    main()
