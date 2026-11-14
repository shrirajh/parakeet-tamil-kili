"""Word / character error rates, plain and convention-folded.

  wer(refs, hyps), cer(refs, hyps)   corpus rates: total edits / total reference words (characters). CER counts
                                     every code point including spaces (as jiwer does). Inputs are used as given.
  fold_wer(refs, hyps)               Tamil: digits -> Tamil number words on both sides (numbers_ta.verbalize_ta),
                                     then kili.text.fold.fold (spoken number words, loanword script, punctuation)
                                     -> {"wer_fold", "cer_fold"}. These are the "WER" / "CER" the website reports
                                     for Tamil sets (run 10, 5 epochs, Kathbath: 29.15 / 5.56).
  english_wer(refs, hyps)            English: the Open ASR Leaderboard normaliser (norm_en_std) -> {"wer", "cer"}.

An empty hypothesis is scored as "-" (one wrong word), as in every reported number.

  python -m kili.metrics.wer pairs.tsv             # TSV columns ref<TAB>hyp, or id<TAB>ref<TAB>hyp, or a header
  python -m kili.metrics.wer hyps.jsonl --lang en  # JSON lines with "ref" and "hyp" fields
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Sequence

import numpy as np


def edit_distance(ref: Sequence, hyp: Sequence) -> int:
    """Levenshtein distance between two token sequences (unit costs). Row-vectorised: the in-row insertion chain
    D[j] = min_k<=j (T[k] + (j - k)) is j + cummin(T[k] - k)."""
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    ids: dict = {}
    r = np.fromiter((ids.setdefault(t, len(ids)) for t in ref), dtype=np.int64, count=len(ref))
    h = np.fromiter((ids.setdefault(t, len(ids)) for t in hyp), dtype=np.int64, count=len(hyp))
    m = len(h)
    js = np.arange(m + 1, dtype=np.int64)
    prev = js.copy()
    t = np.empty(m + 1, dtype=np.int64)
    for i, x in enumerate(r, 1):
        t[0] = i
        np.minimum(prev[:-1] + (h != x), prev[1:] + 1, out=t[1:])
        prev = js + np.minimum.accumulate(t - js)
    return int(prev[m])


def _rate(refs: Sequence[Sequence], hyps: Sequence[Sequence]) -> float:
    if len(refs) != len(hyps):
        raise ValueError(f"{len(refs)} references vs {len(hyps)} hypotheses")
    n = sum(len(r) for r in refs)
    if n == 0:
        raise ValueError("all references are empty")
    return sum(edit_distance(r, h) for r, h in zip(refs, hyps)) / n


def wer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    return _rate([r.split() for r in refs], [h.split() for h in hyps])


def cer(refs: Sequence[str], hyps: Sequence[str]) -> float:
    return _rate([" ".join(r.split()) for r in refs], [" ".join(h.split()) for h in hyps])


def fold_wer(refs: Sequence[str], hyps: Sequence[str], verbalize: bool = True) -> dict:
    """Folded WER / CER for Tamil (and Tamil-English) text. verbalize=False skips the digits -> words step (the
    long-form scorer and EER fold without it)."""
    from kili.text.fold import fold
    if verbalize:
        from kili.text.numbers_ta import verbalize_ta
        refs = [verbalize_ta(r) or r for r in refs]
        hyps = [verbalize_ta(h) or h for h in hyps]
    rf, hf = [fold(r) for r in refs], [fold(h) or "-" for h in hyps]
    return {"wer_fold": wer(rf, hf), "cer_fold": cer(rf, hf)}


def english_wer(refs: Sequence[str], hyps: Sequence[str]) -> dict:
    from kili.text.normalise import norm_en_std
    r, h = [norm_en_std(x) for x in refs], [norm_en_std(x) or "-" for x in hyps]
    return {"wer": wer(r, h), "cer": cer(r, h)}


def read_pairs(path: str, ref_field: str = "ref", hyp_field: str = "hyp") -> tuple[list[str], list[str]]:
    """(refs, hyps) from JSON lines (fields ref_field / hyp_field) or TSV (ref, hyp | id, ref, hyp | a header row
    naming ref_field and hyp_field)."""
    refs, hyps = [], []
    with open(path, encoding="utf-8", newline="") as f:
        if path.endswith((".jsonl", ".json")):
            for line in f:
                if line.strip():
                    o = json.loads(line)
                    refs.append(o.get(ref_field) or "")
                    hyps.append(o.get(hyp_field) or "")
            return refs, hyps
        rows = [r for r in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)]
    if rows and ref_field in rows[0] and hyp_field in rows[0]:
        ri, hi = rows[0].index(ref_field), rows[0].index(hyp_field)
        rows = rows[1:]
    else:
        ri, hi = (0, 1) if rows and len(rows[0]) == 2 else (1, 2)
    for r in rows:
        if r:
            refs.append(r[ri])
            hyps.append(r[hi] if len(r) > hi else "")
    return refs, hyps


def main() -> None:
    ap = argparse.ArgumentParser(description="fold WER / CER (Tamil) or normalised WER / CER (English)")
    ap.add_argument("pairs", help="TSV (ref, hyp) or JSON lines with ref / hyp fields")
    ap.add_argument("--lang", default="ta", choices=["ta", "en"])
    ap.add_argument("--ref-field", default="ref")
    ap.add_argument("--hyp-field", default="hyp")
    ap.add_argument("--no-verbalize", action="store_true", help="Tamil: skip the digits -> number words step")
    a = ap.parse_args()
    refs, hyps = read_pairs(a.pairs, a.ref_field, a.hyp_field)
    res = english_wer(refs, hyps) if a.lang == "en" else fold_wer(refs, hyps, verbalize=not a.no_verbalize)
    print(json.dumps({"n": len(refs), **{k: round(v, 4) for k, v in res.items()}}))


if __name__ == "__main__":
    sys.exit(main())
