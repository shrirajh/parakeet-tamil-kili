"""Language gate for LM decoding: the Tamil n-gram LM may act on a clip only if its greedy transcript is not mostly
English.

Rule: decode greedy first; if more than half of the transcript's letters are Latin (A-Z, a-z) keep the greedy output,
otherwise use the beam + LM output for that clip. Digits, punctuation and spaces do not count; a transcript with no
letters goes to the LM. Without the gate the Tamil LM deletes English words (Svarah WER 11.8 -> 21.1 on run 7); with
it every English set is unchanged or within 0.05 WER, and 93.5-100% of Tamil clips take the LM path.

Merge saved hypotheses (JSONL rows with `audio_filepath` and `hyp`, e.g. from two decode runs):

  python -m kili.lm.gate --greedy greedy.jsonl --lm lm.jsonl --out gated.jsonl
"""
from __future__ import annotations

import argparse
import json
import re

LATIN = re.compile(r"[A-Za-z]")
TAMIL = re.compile(r"[஀-௿]")
THRESHOLD = 0.5


def latin_share(text: str) -> float:
    """Latin letters / (Latin letters + Tamil-block characters); 0.0 if there are none."""
    la, ta = len(LATIN.findall(text)), len(TAMIL.findall(text))
    return la / (la + ta) if la + ta else 0.0


def use_lm(greedy_text: str, threshold: float = THRESHOLD) -> bool:
    """True if the LM may rewrite this clip (its greedy output is not mostly Latin script)."""
    return latin_share(greedy_text) <= threshold


def merge(greedy: list[str], lm: list[str | None], threshold: float = THRESHOLD) -> list[str]:
    """Per clip: the LM output where use_lm(greedy) holds and an LM output exists, else the greedy output."""
    assert len(greedy) == len(lm)
    return [h if (h is not None and use_lm(g, threshold)) else g for g, h in zip(greedy, lm)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--greedy", required=True)
    ap.add_argument("--lm", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    a = ap.parse_args()
    g = [json.loads(x) for x in open(a.greedy, encoding="utf-8") if x.strip()]
    lm = {o["audio_filepath"]: o["hyp"] for o in (json.loads(x) for x in open(a.lm, encoding="utf-8") if x.strip())}
    hyps = merge([o["hyp"] for o in g], [lm.get(o["audio_filepath"]) for o in g], a.threshold)
    n_lm = sum(use_lm(o["hyp"], a.threshold) for o in g)
    with open(a.out, "w", encoding="utf-8") as f:
        for o, h in zip(g, hyps):
            f.write(json.dumps({**o, "hyp": h}, ensure_ascii=False) + "\n")
    print(f"[gate] {len(g)} clips, {n_lm} on the LM path ({100 * n_lm / max(len(g), 1):.1f}%) -> {a.out}")


if __name__ == "__main__":
    main()
