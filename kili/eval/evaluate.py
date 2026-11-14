"""Score a NeMo ASR model on manifests of {"audio_filepath", "text"} (NeMo JSON lines).

  python -m kili.eval.evaluate --model parakeet-tamil-kili.nemo --out ./results/eval \\
      kathbath_test.jsonl fleurs_ta_test.jsonl svarah.jsonl [--lm lm.bin] [--eer] [--max-per-set 2000]
  python -m kili.eval.evaluate --rescore ./results/eval       # recompute metrics from saved *.hyps.jsonl (no GPU)

Decoding: greedy (NeMo's default TDT greedy_batch) unless --lm is given, which decodes with kili.lm.decode.transcribe_lm
(gated beam 4, alpha 0.2). The model is loaded with kili.infer.transcribe.load_model (local attention [256, 256], as
trained).

Each manifest is one set, named after its file. Its language is detected from the references (Tamil script -> "ta")
unless --lang is given. Metrics (fractions; the website shows them x100):
  Tamil sets   wer_fold / cer_fold   kili.metrics.wer.fold_wer: digits -> number words, then the convention fold
                                     (loanword script, spoken number words, punctuation). The headline WER / CER.
               per_fold              PER after the phonetic fold (the headline PER); per / per_flat on norm_ta text
               wer / cer             plain: reference with loans latinized vs hypothesis, both norm_ta
               eer                   with --eer: embedding error rate on folded text (kili.metrics.eer)
  English sets wer / cer             Open ASR Leaderboard normaliser (Whisper's EnglishTextNormalizer via
                                     transformers; norm_en if it is missing)
               wer_basic / cer_basic lowercase + strip punctuation
               tamil_script_utts     share of utterances with any Tamil-script character in the output
With --max-per-set N the set is shuffled with a fixed seed (0) and the first N clips are scored (the website uses
2000 per set). --ref-fixes takes a JSON {audio file basename: corrected reference, or null to drop the clip}.

Public test sets the website reports (build the manifests from the original releases):
  Tamil    IndicVoices (ta test), Kathbath (ta test), MILE Tamil (held-out speakers), FLEURS ta (test),
           Common Voice ta (test), OpenSLR-65 (Tamil)
  English  Common Voice en (test), LibriSpeech test-clean / test-other, Svarah (Indian-accented English)
Run 10, 5 epochs, greedy (wer_fold / cer_fold / per_fold, %): Kathbath 29.15 / 5.56 / 2.41, IndicVoices
34.05 / 11.62 / 9.44, FLEURS ta 30.46 / 9.71 / 6.66; English WER: LibriSpeech clean 2.30, other 4.07, Svarah 10.67.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import time

TAMIL = re.compile(r"[஀-௿]")


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)


def detect_lang(refs: list[str]) -> str:
    return "ta" if sum(bool(TAMIL.search(r)) for r in refs) > len(refs) / 2 else "en"


def load_manifest(path: str, max_n: int = 0, seed: int = 0) -> list[dict]:
    """Rows with audio paths resolved (relative paths are taken relative to the manifest's directory)."""
    base = os.path.dirname(os.path.abspath(path))
    rows = [r for r in read_jsonl(path) if not r.get("_skipme")]
    for r in rows:
        p = r["audio_filepath"]
        if not os.path.isabs(p) and not os.path.exists(p):
            r["audio_filepath"] = os.path.join(base, p)
    if max_n:
        random.Random(seed).shuffle(rows)
        rows = rows[:max_n]
    return rows


def transcribe(model, paths: list[str], batch_size: int) -> list[str]:
    """Greedy transcripts (NeMo's model.transcribe; handles the return types of NeMo 2.x)."""
    import torch
    with torch.inference_mode():
        hyps = model.transcribe(paths, batch_size=batch_size, verbose=False)
    if isinstance(hyps, tuple):
        hyps = hyps[0]
    hyps = [h[0] if isinstance(h, list) and h else h for h in hyps]  # beam strategies: n-best lists, best first
    return [h.text if hasattr(h, "text") else str(h) for h in hyps]


def score_tamil(refs: list[str], hyps: list[str]) -> dict:
    from kili.metrics.per import per, per_fold
    from kili.metrics.wer import cer, fold_wer, wer
    from kili.text.loanwords import latinize
    from kili.text.normalise import norm_ta
    from kili.text.numbers_ta import verbalize_ta
    refs_v = [verbalize_ta(r) or r for r in refs]
    hyps_v = [verbalize_ta(h) or h for h in hyps]
    rt = [norm_ta(latinize(r)) for r in refs_v]
    ht = [norm_ta(h) or "-" for h in hyps_v]
    p = per(rt, ht)
    return {**fold_wer(refs, hyps), "per_fold": per_fold(refs, hyps), "wer": wer(rt, ht), "cer": cer(rt, ht),
            "per": p["per"], "per_flat": p["per_flat"]}


def score_english(refs: list[str], hyps: list[str]) -> dict:
    from kili.metrics.wer import cer, english_wer, wer
    from kili.text.normalise import norm_en
    r, h = [norm_en(x) for x in refs], [norm_en(x) or "-" for x in hyps]
    return {**english_wer(refs, hyps), "wer_basic": wer(r, h), "cer_basic": cer(r, h),
            "tamil_script_utts": sum(bool(TAMIL.search(x)) for x in hyps) / max(len(hyps), 1)}


def score(rows: list[dict], lang: str, with_eer: bool = False, ref_fixes: dict | None = None) -> dict:
    """All metrics for one set from rows {audio_filepath, ref, hyp}."""
    if ref_fixes:
        key = lambda o: os.path.basename(o["audio_filepath"])  # noqa: E731
        rows = [{**o, "ref": ref_fixes[key(o)]} if ref_fixes.get(key(o)) else o
                for o in rows if not (key(o) in ref_fixes and ref_fixes[key(o)] is None)]
    refs, hyps = [o["ref"] for o in rows], [o["hyp"] for o in rows]
    res = {"n": len(rows), "lang": lang}
    res.update(score_tamil(refs, hyps) if lang == "ta" else score_english(refs, hyps))
    if with_eer and lang == "ta":
        from kili.metrics.eer import eer
        res["eer"] = eer(refs, hyps)["eer"]
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}


def rescore(d: str, with_eer: bool, ref_fixes: dict | None, lang: str) -> None:
    for hp in sorted(glob.glob(os.path.join(d, "*.hyps.jsonl"))):
        rows = read_jsonl(hp)
        jp = hp[: -len(".hyps.jsonl")] + ".json"
        old = json.load(open(jp, encoding="utf-8")) if os.path.exists(jp) else {}
        lg = lang if lang != "auto" else old.get("lang") or detect_lang([r["ref"] for r in rows])
        res = {**old, "set": os.path.basename(hp)[: -len(".hyps.jsonl")], **score(rows, lg, with_eer, ref_fixes)}
        write_json(jp, res)
        print(json.dumps(res, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="score a NeMo model on NeMo manifests")
    ap.add_argument("manifests", nargs="*", help="JSON lines manifests with audio_filepath and text")
    ap.add_argument("--model", help=".nemo file or Hugging Face model id")
    ap.add_argument("--out", help="output directory: <set>.hyps.jsonl, <set>.json, summary.json")
    ap.add_argument("--rescore", metavar="DIR", help="recompute metrics from DIR/*.hyps.jsonl and exit")
    ap.add_argument("--lang", default="auto", choices=["auto", "ta", "en"])
    ap.add_argument("--max-per-set", type=int, default=0, help="0 = every clip")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lm", default=None, help="token-level KN 5-gram (NGPU-LM .nemo or ARPA from kili.lm.kn_arpa) for kili.lm.decode.transcribe_lm")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--no-gate", action="store_true", help="LM on every utterance (default: gated)")
    ap.add_argument("--eer", action="store_true", help="also embedding error rate on Tamil sets (EmbeddingGemma 2)")
    ap.add_argument("--ref-fixes", default=None, help="JSON {audio basename: corrected reference | null = drop}")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    fixes = json.load(open(a.ref_fixes, encoding="utf-8")) if a.ref_fixes else None
    if a.rescore:
        return rescore(a.rescore, a.eer, fixes, a.lang)
    if not (a.model and a.out and a.manifests):
        ap.error("--model, --out and at least one manifest are required (or --rescore DIR)")
    from kili.infer.transcribe import load_model
    model = load_model(a.model, a.device)
    os.makedirs(a.out, exist_ok=True)
    summary = {"model": a.model, "lm": a.lm, "results": []}
    for man in a.manifests:
        name = os.path.basename(man).rsplit(".", 1)[0]
        rows = load_manifest(man, a.max_per_set)
        paths = [r["audio_filepath"] for r in rows]
        t0 = time.time()
        if a.lm:
            from kili.lm.decode import transcribe_lm
            hyps = transcribe_lm(model, paths, a.lm, alpha=a.alpha, beam=a.beam, gate=not a.no_gate,
                                 batch_size=a.batch_size)
        else:
            hyps = transcribe(model, paths, a.batch_size)
        secs = time.time() - t0
        out = [{"audio_filepath": r["audio_filepath"], "ref": r["text"], "hyp": h} for r, h in zip(rows, hyps)]
        lang = a.lang if a.lang != "auto" else detect_lang([r["text"] for r in rows])
        audio_s = sum(r.get("duration", 0.0) for r in rows)
        res = {"set": name, **score(out, lang, a.eer, fixes), "decode_s": round(secs, 1)}
        if audio_s:
            res["rtfx"] = round(audio_s / secs, 1)
        print(json.dumps(res, ensure_ascii=False), flush=True)
        write_jsonl(os.path.join(a.out, f"{name}.hyps.jsonl"), out)
        write_json(os.path.join(a.out, f"{name}.json"), res)
        summary["results"].append(res)
    write_json(os.path.join(a.out, "summary.json"), summary)


if __name__ == "__main__":
    main()
