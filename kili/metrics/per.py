"""Phonetic error rate (PER) for Tamil: an edit distance over phonemes where substituting a similar phoneme costs
less than a full error, normalised by the number of reference phonemes. Spaces are ignored.

Why: WER punishes one wrong suffix as a whole wrong word, and CER counts ண/ந/ன or ல/ள/ழ swaps as full errors even
though they are the most common (and least harmful) confusions, often not distinguished in casual speech. PER answers
"how close is what the model heard to what was said". It does not measure meaning: மழை -> மலை (rain -> hill) is a
near miss (6.25%); that is what EER (kili.metrics.eer) is for.

  per       feature-weighted (costs below)        <- the headline phonetic metric
  per_flat  every phoneme substitution costs 1    (plain phoneme error rate, for comparison)
  per_fold  `per` after kili.text.fold.phonetic_fold on both sides (Latin loans back in Tamil script, spoken number
            words written, sandhi joins). The website's "PER" is per_fold (run 10, 5 epochs, Kathbath: 2.41%).

Units come from Tamil script: consonants, vowels (the inherent 'a' made explicit; virama = no vowel), aytham, and ஃப
as one unit (/f/). Other letters and digits (Latin) are their own units, similar to nothing.

Substitution costs (insertion / deletion always 1):
  consonants: same manner, same place (all coronals share one place)  0.25  ண ந ன, ட த, ல ள ழ, ர ற, ஸ ஷ
              same manner, neighbouring place                         0.5
              same manner, farther place                              0.75
              related manner at the same place                        0.5   (nasal-stop ண/ட, lateral-rhotic ல/ர,
                                                                             approximant-liquid, stop-fricative)
              special pairs                                           0.25  ச~ஸ, ச~ஶ, ஜ~ச (Tamil ச is often [s]/[dʒ])
  vowels:     same quality, different length (அ/ஆ, இ/ஈ ...)           0.25
              neighbouring quality (i-e, e-a, a-o, o-u, ai-e, ai-a, au-o, au-a)  0.5
              other vowel pairs                                       0.75
  anything else (consonant <-> vowel, Latin units, unrelated manners) 1

  python -m kili.metrics.per pairs.tsv        # TSV ref<TAB>hyp (or JSON lines with ref / hyp), Tamil script
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata

import numpy as np

VIRAMA, AYTHAM = "்", "ஃ"
CONSONANTS = set("கஙசஞடணதநனபமயரறலளழவஜஷஸஹஶ")
VOWELS = {"அ", "ஆ", "இ", "ஈ", "உ", "ஊ", "எ", "ஏ", "ஐ", "ஒ", "ஓ", "ஔ"}
SIGN_TO_VOWEL = {"ா": "ஆ", "ி": "இ", "ீ": "ஈ", "ு": "உ", "ூ": "ஊ", "ெ": "எ", "ே": "ஏ", "ை": "ஐ", "ொ": "ஒ", "ோ": "ஓ",
                 "ௌ": "ஔ"}

# consonant -> (manner, place); places on a line, coronal places share one slot
_LABIAL, _LABIODENTAL, _CORONAL, _PALATAL, _VELAR, _GLOTTAL = 0, 1, 2, 3, 4, 5
_CONS_FEAT = {
    "க": ("stop", _VELAR), "ங": ("nasal", _VELAR), "ச": ("stop", _PALATAL), "ஞ": ("nasal", _PALATAL),
    "ட": ("stop", _CORONAL), "ண": ("nasal", _CORONAL), "த": ("stop", _CORONAL), "ந": ("nasal", _CORONAL),
    "ன": ("nasal", _CORONAL), "ப": ("stop", _LABIAL), "ம": ("nasal", _LABIAL), "ய": ("approx", _PALATAL),
    "ர": ("rhotic", _CORONAL), "ற": ("rhotic", _CORONAL), "ல": ("lateral", _CORONAL), "ள": ("lateral", _CORONAL),
    "ழ": ("lateral", _CORONAL), "வ": ("approx", _LABIODENTAL), "ஜ": ("stop", _PALATAL),
    "ஷ": ("fric", _CORONAL), "ஸ": ("fric", _CORONAL), "ஹ": ("fric", _GLOTTAL), "ஶ": ("fric", _PALATAL),
    "ஃப": ("fric", _LABIODENTAL), AYTHAM: ("fric", _VELAR),
}
_RELATED_MANNER = {frozenset(p) for p in [("nasal", "stop"), ("lateral", "rhotic"), ("approx", "lateral"),
                                          ("approx", "rhotic"), ("stop", "fric")]}
_SPECIAL = {frozenset(p): 0.25 for p in [("ச", "ஸ"), ("ச", "ஶ"), ("ஜ", "ச")]}

# vowel -> (quality, long); qualities on a chain i - e - a - o - u, diphthongs attached to their neighbours
_VOWEL_FEAT = {"அ": ("a", 0), "ஆ": ("a", 1), "இ": ("i", 0), "ஈ": ("i", 1), "உ": ("u", 0), "ஊ": ("u", 1),
               "எ": ("e", 0), "ஏ": ("e", 1), "ஐ": ("ai", 1), "ஒ": ("o", 0), "ஓ": ("o", 1), "ஔ": ("au", 1)}
_VOWEL_NEIGH = {frozenset(p) for p in [("i", "e"), ("e", "a"), ("a", "o"), ("o", "u"), ("ai", "e"), ("ai", "a"),
                                       ("au", "o"), ("au", "a")]}


def tamil_phonemes(text: str) -> list[str]:
    """Tamil script -> phoneme units (consonant letters, vowel letters; spaces / punctuation dropped)."""
    s = unicodedata.normalize("NFC", text).replace("ஃப", "\x00")
    out: list[str] = []
    i = 0
    while i < len(s):
        ch = s[i]
        c = "ஃப" if ch == "\x00" else ch
        if c in CONSONANTS or c == "ஃப":
            out.append(c)
            nxt = s[i + 1] if i + 1 < len(s) else ""
            if nxt in SIGN_TO_VOWEL:
                out.append(SIGN_TO_VOWEL[nxt]); i += 2; continue
            if nxt == VIRAMA:
                i += 2; continue
            out.append("அ")  # inherent vowel
        elif ch in VOWELS or ch == AYTHAM:
            out.append(ch)
        elif ch.isalnum():
            out.append("x:" + ch.lower())  # non-Tamil letter / digit: its own unit
        # spaces, punctuation, stray signs: ignored
        i += 1
    return out


def sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if frozenset((a, b)) in _SPECIAL:
        return _SPECIAL[frozenset((a, b))]
    if a in _CONS_FEAT and b in _CONS_FEAT:
        (ma, pa), (mb, pb) = _CONS_FEAT[a], _CONS_FEAT[b]
        if ma == mb:
            d = abs(pa - pb)
            return 0.25 if d == 0 else 0.5 if d == 1 else 0.75
        if pa == pb and frozenset((ma, mb)) in _RELATED_MANNER:
            return 0.5
        return 1.0
    if a in _VOWEL_FEAT and b in _VOWEL_FEAT:
        (qa, _), (qb, _) = _VOWEL_FEAT[a], _VOWEL_FEAT[b]
        if qa == qb:
            return 0.25
        return 0.5 if frozenset((qa, qb)) in _VOWEL_NEIGH else 0.75
    return 1.0


class _Inventory:
    """Unit ids and the substitution cost matrix over every unit seen so far."""

    def __init__(self):
        self.ids: dict[str, int] = {}
        self.cost = np.ones((0, 0), dtype=np.float32)

    def encode(self, units: list[str]) -> np.ndarray:
        new = [u for u in dict.fromkeys(units) if u not in self.ids]
        if new:
            names = list(self.ids) + new
            for u in new:
                self.ids[u] = len(self.ids)
            n = len(names)
            c = np.empty((n, n), dtype=np.float32)
            for i, a in enumerate(names):
                for j, b in enumerate(names):
                    c[i, j] = sub_cost(a, b)
            self.cost = c
        return np.fromiter((self.ids[u] for u in units), dtype=np.int64, count=len(units))


_INV = _Inventory()


def weighted_edit_distance(ref: np.ndarray, hyp: np.ndarray, cost: np.ndarray) -> float:
    """Levenshtein with a substitution cost matrix, unit insert / delete. Row-vectorised: the in-row insertion chain
    D[j] = min_k<=j (T[k] + (j - k)) is j + cummin(T[k] - k)."""
    m = len(hyp)
    if len(ref) == 0:
        return float(m)
    prev = np.arange(m + 1, dtype=np.float64)
    js = np.arange(m + 1, dtype=np.float64)
    for i, r in enumerate(ref, 1):
        t = np.empty(m + 1)
        t[0] = i
        if m:
            t[1:] = np.minimum(prev[:-1] + cost[r, hyp], prev[1:] + 1.0)
        prev = js + np.minimum.accumulate(t - js)
    return float(prev[m])


def per(refs: list[str], hyps: list[str]) -> dict:
    """Corpus PER over Tamil-script pairs: sum of edit costs / reference phonemes -> {per, per_flat, ref_phonemes}."""
    tot_w = tot_f = n = 0.0
    for r, h in zip(refs, hyps):
        ru, hu = tamil_phonemes(r), tamil_phonemes(h)
        ri, hi = _INV.encode(ru), _INV.encode(hu)
        tot_w += weighted_edit_distance(ri, hi, _INV.cost)
        tot_f += weighted_edit_distance(ri, hi, (_INV.cost > 0).astype(np.float32))
        n += len(ru)
    return {"per": tot_w / max(n, 1), "per_flat": tot_f / max(n, 1), "ref_phonemes": int(n)}


def per_fold(refs: list[str], hyps: list[str], verbalize: bool = True) -> float:
    """PER after the phonetic fold. verbalize: digits -> Tamil number words first (as the short-form scorer does)."""
    from kili.text.fold import phonetic_fold
    if verbalize:
        from kili.text.numbers_ta import verbalize_ta
        refs = [verbalize_ta(r) or r for r in refs]
        hyps = [verbalize_ta(h) or h for h in hyps]
    return per([phonetic_fold(r) for r in refs], [phonetic_fold(h) or "-" for h in hyps])["per"]


def main() -> None:
    from kili.metrics.wer import read_pairs
    ap = argparse.ArgumentParser(description="feature-weighted Tamil phoneme error rate")
    ap.add_argument("pairs", help="TSV (ref, hyp) or JSON lines with ref / hyp fields")
    ap.add_argument("--ref-field", default="ref")
    ap.add_argument("--hyp-field", default="hyp")
    a = ap.parse_args()
    refs, hyps = read_pairs(a.pairs, a.ref_field, a.hyp_field)
    p = per(refs, hyps)
    out = {"n": len(refs), "per": p["per"], "per_flat": p["per_flat"], "per_fold": per_fold(refs, hyps),
           "ref_phonemes": p["ref_phonemes"]}
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in out.items()}))


if __name__ == "__main__":
    sys.exit(main())
