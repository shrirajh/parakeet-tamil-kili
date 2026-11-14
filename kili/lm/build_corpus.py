r"""LM text: Sangraha web sentences and (optionally) your own human transcripts, normalised to the training-label
convention (kili.text.normalise: loans to Latin, numbers to Tamil words, punctuation kept), with every evaluation
sentence removed.

  python -m kili.lm.build_corpus download --out ./data/sangraha --files 4
  python -m kili.lm.build_corpus build --out ./data/lm_corpus \
      --sangraha ./data/sangraha/verified/tam --max-sentences 8000000 \
      --exclude ./data/eval/kathbath.jsonl ./data/eval/mile.jsonl ./data/eval/fleurs_ta.jsonl ...
  -> <out>/sangraha.txt, <out>/exclusion.json; then kili.lm.kn_arpa --text sangraha.txt (the released LM).

The released LM is Sangraha only. --own adds your own human transcripts (<out>/own.txt; weight them with
kn_arpa --weights).

Sources:
- Sangraha `verified/tam` (AI4Bharat, CC BY 4.0, arXiv 2403.06350), the first 4 parquet files in sorted order, first
  8M sentences. Only the `verified` config: `synthetic` is machine-translated and `unverified` is not human-checked.
  Documents are split into sentences (bracketed spans dropped; mostly-Tamil sentences of 3-60 words); punctuation is
  kept, as in the labels.
- Own transcripts (optional, not in the released LM): human-transcribed training text only. Never add model
  pseudo-labels or LLM-written text here.

--exclude (eval manifests, JSONL with `text` or `text_ta`, or directories of manifest_*.jsonl): a corpus sentence is
dropped if, in the space fold WER scores in (digits -> words, kili.text.fold.fold, lowercased), it
  exact      equals a reference with spaces ignored,
  ngram      shares any word n-gram with a reference (n = --shared-ngram, default 6 as released), or
  contains   contains a whole 5-7-word reference.
Pass EVERY test set you will report on. Counts and examples per rule go to <out>/exclusion.json. The released LM
excluded the references of 15 test sets (91,212 sentences: Kathbath, IndicVoices, MILE, FLEURS, Common Voice ta and
en, OpenSLR-65, Sri Lankan parliament, Svarah, LibriSpeech, GigaSpeech, Wikimedia and the synthetic Tanglish tests),
removing 2,279 of the first 8M Sangraha sentences.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import unicodedata
from multiprocessing import Pool

TA = re.compile(r"[஀-௿]")
LAT = re.compile(r"[A-Za-z]")
_PUNCT = re.compile(r"[^\w\s஀-௿-]")
SANGRAHA = "ai4bharat/sangraha"


def nfc(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s or "")).strip()


def normalise(s: str) -> str | None:
    """The Tamil training-label rule (kili.text.normalise, lang "ta"); None (sentence dropped) for a number that can't
    be verbalised confidently."""
    from kili.text.normalise import normalise_label
    return normalise_label(nfc(s), "ta")


def key_words(s: str) -> list[str]:
    """The comparison space for exclusion, as the fold-WER scorer sees text: digits -> Tamil number words, then
    kili.text.fold.fold (number register, loan script, punctuation off), lowercased."""
    from kili.text.fold import fold
    from kili.text.numbers_ta import verbalize_ta
    t = _PUNCT.sub(" ", fold(verbalize_ta(s) or s)).lower()
    return t.split()


def doc_sentences(text: str) -> list[str]:
    """Document -> mostly-Tamil sentences of 3-60 words (>= 10 Tamil characters, Latin <= 20% of letters)."""
    out = []
    for para in text.split("\n"):
        para = re.sub(r"\([^()]*\)|\[[^\[\]]*\]|\{[^{}]*\}", " ", para)
        for s in re.split(r"(?<=[.!?।])\s+", para):
            s = nfc(s)
            ta, la = len(TA.findall(s)), len(LAT.findall(s))
            if ta < 10 or la > 0.2 * (ta + la):
                continue
            if 3 <= len(s.split()) <= 60:
                out.append(s)
    return out


def _doc_chunk(texts: list[str]) -> list[str]:
    res = []
    for tx in texts:
        for s in doc_sentences(tx or ""):
            v = normalise(s)
            if v and len(v.split()) >= 3:
                res.append(v)
    return res


def _manifest_files(paths: list[str]) -> list[str]:
    files = []
    for p in paths:
        files += sorted(glob.glob(os.path.join(p, "**", "*.jsonl"), recursive=True)) if os.path.isdir(p) else [p]
    assert files, f"no manifests in {paths}"
    return files


def read_texts(paths: list[str]) -> list[str]:
    """`text_ta` or `text` of every non-skipped row of JSONL manifests (files or directories)."""
    out = []
    for p in _manifest_files(paths):
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                t = r.get("text_ta") or r.get("text") or ""
                if t and not r.get("_skipme"):
                    out.append(t)
    return out


def _own_chunk(texts: list[str]) -> list[str]:
    return [v for v in (normalise(t) for t in texts) if v]


class Exclusion:
    """Exact / shared-n-gram / contained-reference matches against a set of references (see the module docstring)."""

    def __init__(self, refs: list[str], ngram: int = 8, key=key_words):
        self.key, self.n = key, ngram
        self.exact, self.grams, self.short = set(), set(), {5: set(), 6: set(), 7: set()}
        for r in refs:
            w = key(r)
            if not w:
                continue
            self.exact.add("".join(w))
            for i in range(len(w) - ngram + 1):
                self.grams.add(tuple(w[i:i + ngram]))
            if 5 <= len(w) <= 7:
                self.short[len(w)].add(tuple(w))

    def check(self, s: str) -> str | None:
        """The rule that drops `s`, or None if it is kept."""
        w = self.key(s)
        if "".join(w) in self.exact:
            return "exact"
        n = self.n
        if any(tuple(w[i:i + n]) in self.grams for i in range(len(w) - n + 1)):
            return "ngram"
        for k, st in self.short.items():
            if st and any(tuple(w[i:i + k]) in st for i in range(len(w) - k + 1)):
                return "contains"
        return None


_EXCL: Exclusion | None = None


def _excl_init(refs, ngram):
    global _EXCL
    _EXCL = Exclusion(refs, ngram)


def _excl_chunk(lines):
    return [_EXCL.check(s) for s in lines]


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def download(out: str, n_files: int) -> None:
    """The first n parquet files (sorted) of Sangraha verified/tam."""
    from huggingface_hub import HfApi, hf_hub_download
    files = sorted(f for f in HfApi().list_repo_files(SANGRAHA, repo_type="dataset")
                   if f.startswith("verified/tam/") and f.endswith(".parquet"))
    print(f"[sangraha] verified/tam: {len(files)} parquet files; taking {n_files}", flush=True)
    for f in files[:n_files]:
        print("[sangraha]", hf_hub_download(SANGRAHA, f, repo_type="dataset", local_dir=out), flush=True)


def build(a) -> None:
    os.makedirs(a.out, exist_ok=True)
    refs = [r for r in read_texts(a.exclude) if r]
    print(f"[exclude] {len(refs)} references from {len(a.exclude)} paths", flush=True)
    corpora: dict[str, list[str]] = {}
    with Pool(a.procs) as pool:
        if a.own:
            own = []
            for res in pool.imap(_own_chunk, chunks(read_texts(a.own), 5000)):
                own += res
            corpora["own"] = own
            print(f"[own] {len(own)} sentences", flush=True)
        if a.sangraha:
            import pyarrow.parquet as pq
            sg = []
            for p in sorted(glob.glob(os.path.join(a.sangraha, "*.parquet"))):
                if a.max_sentences and len(sg) >= a.max_sentences:
                    break
                texts = pq.read_table(p, columns=["text"]).column("text").to_pylist()
                for res in pool.imap(_doc_chunk, chunks(texts, 200)):
                    sg += res
                print(f"[sangraha] {os.path.basename(p)}: {len(texts)} docs -> {len(sg)} sentences so far", flush=True)
            corpora["sangraha"] = sg[:a.max_sentences] if a.max_sentences else sg
    report = {"n_refs": len(refs), "exclude": a.exclude, "shared_ngram": a.shared_ngram, "corpora": {}}
    with Pool(a.procs, initializer=_excl_init, initargs=(refs, a.shared_ngram)) as pool:
        for name, lines in corpora.items():
            hits = []
            for h in pool.imap(_excl_chunk, chunks(lines, 2000)):
                hits += h
            kept = [s for s, h in zip(lines, hits) if h is None]
            report["corpora"][name] = {
                "sentences_in": len(lines), "kept": len(kept),
                "removed": {k: sum(h == k for h in hits) for k in ("exact", "ngram", "contains")},
                "removed_examples": [s for s, h in zip(lines, hits) if h][:20]}
            with open(os.path.join(a.out, f"{name}.txt"), "w", encoding="utf-8") as f:
                f.writelines(s + "\n" for s in kept)
            print(f"[exclude] {name}: {len(lines)} in, {report['corpora'][name]['removed']} removed, {len(kept)} kept",
                  flush=True)
    with open(os.path.join(a.out, "exclusion.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="Sangraha verified/tam parquet files")
    d.add_argument("--out", required=True)
    d.add_argument("--files", type=int, default=4)
    b = sub.add_parser("build", help="normalise, exclude eval sentences, write <out>/<corpus>.txt")
    b.add_argument("--out", required=True)
    b.add_argument("--sangraha", help="directory of Sangraha parquet files (`text` column)")
    b.add_argument("--max-sentences", type=int, default=8_000_000, help="Sangraha sentences to keep (0 = all)")
    b.add_argument("--own", nargs="*", default=[], help="human-transcript manifests (JSONL files or directories)")
    b.add_argument("--exclude", nargs="+", required=True, help="eval manifests whose sentences must not be in the LM")
    b.add_argument("--shared-ngram", type=int, default=6)
    b.add_argument("--procs", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    if a.cmd == "download":
        download(a.out, a.files)
    else:
        assert a.sangraha or a.own, "nothing to build: pass --sangraha and/or --own"
        build(a)


if __name__ == "__main__":
    main()
