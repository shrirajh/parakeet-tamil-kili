"""Long-form eval: stitched multi-minute recordings, decoded whole, against the same audio decoded clip by clip.

Fine-tuning only sees clips <= 30 s, so nothing in training protects the base model's long-audio ability; this
measures it. Recordings are built deterministically (seed 7) from short-form manifests:
  long_ta      2-5 min of Tamil (clips drawn at random from the --ta manifests)
  long_en      2-5 min of English (--en manifests)
  long_hybrid  2-5 min cycling through the --en then --ta manifests clip by clip (many language switches)
Clips of 2-25 s are joined with 0.4 s of silence. Each recording is decoded whole with local attention
--att-context L R (default 256 256, the window the model was trained with; -1 -1 = full attention) AND as its
separate clips (short-form, the model as loaded), so `degradation_wer` (`degradation_wer_fold`) = long-form minus
short-form wer (wer_fold) on identical audio.

Metrics per category, for both decodes (fractions):
  wer_fold / cer_fold   the convention fold of kili.text.fold (loanword script, spoken number words) on the
                        mixed-script normalised text: use these
  per_fold              PER after the phonetic fold
  wer / cer             on the mixed-script normalised text only (counts a Latin-written loanword against a
                        Tamil-script reference as an error)
Run 10, 5 epochs, long_ta (20 recordings from IndicVoices + Kathbath test clips): wer_fold 34.37, cer_fold 10.44,
per_fold 7.39 (%). 128-frame windows (the base model's long-audio default) lost ~3.6 WER vs short-form.

  python -m kili.eval.long_form --model parakeet-tamil-kili.nemo --out ./results/long_form \\
      --ta indicvoices_test.jsonl kathbath_test.jsonl [--en ls_test_other.jsonl svarah.jsonl] [--n 20]
Recordings depend on the manifests' contents and order: the same manifests give the same recordings.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import unicodedata

import numpy as np

from kili.infer.transcribe import set_attention

SR = 16000
GAP = 0.4  # s of silence between stitched clips
MIN_S, MAX_S = 2.0, 25.0


def norm_mixed(s: str) -> str:
    """Lowercase; keep letters (incl. Tamil vowel signs), digits, apostrophes; collapse spaces."""
    s = unicodedata.normalize("NFC", s).lower()
    s = "".join(c if (unicodedata.category(c)[0] in "LMN" or c in "' " or c.isspace()) else " " for c in s)
    return re.sub(r"\s+", " ", s).strip()


def _duration(r: dict) -> float:
    if "duration" in r:
        return float(r["duration"])
    import soundfile as sf
    return sf.info(r["audio_filepath"]).duration


def _pool(path: str) -> list[dict]:
    from kili.eval.evaluate import load_manifest
    return [r for r in load_manifest(path) if MIN_S <= _duration(r) <= MAX_S]


def build(ta: list[str], en: list[str], n: int, seed: int = 7) -> dict[str, list[dict]]:
    """-> {category: [{id, segments: [{audio_filepath, ref, lang, duration}]}]}. Clips are drawn without
    replacement across categories."""
    rng = random.Random(seed)
    cats = {"long_ta": [(p, "ta") for p in ta], "long_en": [(p, "en") for p in en],
            "long_hybrid": [(p, "en") for p in en] + [(p, "ta") for p in ta] if ta and en else []}
    pools: dict[str, list[dict]] = {}
    out = {}
    for cat, sets in cats.items():
        if not sets:
            continue
        for s, _ in sets:
            if s not in pools:
                pools[s] = _pool(s)
                rng.shuffle(pools[s])
        recs = []
        for i in range(n):
            target = rng.uniform(120, 300)
            segs, dur, k = [], 0.0, 0
            while dur < target:
                s, lang = sets[k % len(sets)] if cat == "long_hybrid" else rng.choice(sets)
                k += 1
                if not pools[s]:
                    break
                r = pools[s].pop()
                d = _duration(r)
                segs.append({"audio_filepath": r["audio_filepath"], "ref": r["text"], "lang": lang, "duration": d})
                dur += d + GAP
            if segs:
                recs.append({"id": f"{cat}_{i:03d}", "segments": segs})
        out[cat] = recs
    return out


def load_audio(path: str) -> np.ndarray:
    import librosa
    x, _ = librosa.load(path, sr=SR, mono=True)
    return x.astype(np.float32)


def score(refs: list[str], hyps: list[str]) -> dict:
    """refs / hyps: one norm_mixed text per recording."""
    from kili.metrics.per import per
    from kili.metrics.wer import cer, wer
    from kili.text.fold import fold, phonetic_fold
    R = [r or "-" for r in refs]
    H = [h or "-" for h in hyps]
    rf, hf = [fold(r) for r in R], [fold(h) or "-" for h in H]
    res = {"wer": wer(R, H), "cer": cer(R, H), "wer_fold": wer(rf, hf), "cer_fold": cer(rf, hf),
           "per_fold": per([phonetic_fold(r) for r in R], [phonetic_fold(h) or "-" for h in H])["per"]}
    return {k: round(v, 4) for k, v in res.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="long-form (stitched) vs short-form decoding of the same audio")
    ap.add_argument("--model", required=True, help=".nemo file or Hugging Face model id")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ta", nargs="*", default=[], help="Tamil manifests (audio_filepath, text[, duration])")
    ap.add_argument("--en", nargs="*", default=[], help="English manifests")
    ap.add_argument("--n", type=int, default=20, help="recordings per category")
    ap.add_argument("--att-context", type=int, nargs=2, default=[256, 256],
                    help="local attention window for the long decode; -1 -1 = full attention")
    ap.add_argument("--batch-size", type=int, default=32, help="short-form batch size")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if not (a.ta or a.en):
        ap.error("give --ta and/or --en manifests")
    import torch

    from kili.infer.transcribe import load_model
    from kili.eval.evaluate import transcribe, write_json, write_jsonl
    model = load_model(a.model, a.device)
    cats = build(a.ta, a.en, a.n)
    os.makedirs(a.out, exist_ok=True)
    summary = {"model": a.model, "att_context": a.att_context}
    gap = np.zeros(int(GAP * SR), np.float32)
    for cat, recs in cats.items():
        if not recs:
            continue
        longs, clips, refs = [], [], []
        for rec in recs:
            parts, words = [], []
            for sg in rec["segments"]:
                x = load_audio(sg["audio_filepath"])
                parts += [x, gap]
                clips.append(x)
                words += norm_mixed(sg["ref"]).split()
            longs.append(np.concatenate(parts))
            refs.append(" ".join(words))
        short = transcribe(model, clips, a.batch_size)
        k, short_rec = 0, []
        for rec in recs:
            m = len(rec["segments"])
            short_rec.append(norm_mixed(" ".join(short[k:k + m])))
            k += m
        orig_kind = model.cfg.encoder.get("self_attention_model", "rel_pos")
        orig_ctx = list(model.cfg.encoder.get("att_context_size", [-1, -1]))
        full = list(a.att_context) == [-1, -1]
        set_attention(model, "rel_pos" if full else "rel_pos_local_attn", a.att_context)
        with torch.inference_mode():
            long_h = [norm_mixed(h) for h in transcribe(model, longs, 1)]
        set_attention(model, orig_kind, orig_ctx)
        res_long, res_short = score(refs, long_h), score(refs, short_rec)
        res = {"set": cat, "n": len(recs), "audio_min": round(sum(len(x) for x in longs) / SR / 60, 1),
               "long": res_long, "short": res_short,
               "degradation_wer": round(res_long["wer"] - res_short["wer"], 4),
               "degradation_wer_fold": round(res_long["wer_fold"] - res_short["wer_fold"], 4)}
        print(json.dumps(res), flush=True)
        write_json(os.path.join(a.out, f"{cat}.json"), res)
        write_jsonl(os.path.join(a.out, f"{cat}.hyps.jsonl"),
                    [{"id": r["id"], "ref": ref, "hyp_long": lh, "hyp_short": sh}
                     for r, ref, lh, sh in zip(recs, refs, long_h, short_rec)])
        summary[cat] = res
    write_json(os.path.join(a.out, "summary.json"), summary)


if __name__ == "__main__":
    main()
