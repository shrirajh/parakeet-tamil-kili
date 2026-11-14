r"""Interpolated modified Kneser-Ney n-gram LM over the ASR model's own subword ids, written as ARPA for NeMo's GPU
n-gram fusion (NGPU-LM, nemo.collections.asr.parts.submodules.ngram_lm). Pure numpy; no KenLM binaries.

Text is encoded the way NeMo's train_kenlm does it: each sentence -> the model's sentencepiece ids, each id written as
chr(id + 100). The released LM (gated beam 4, alpha 0.2):

  python -m kili.lm.kn_arpa --model kili.nemo --text corpus/sangraha.txt \
      --order 5 --prune 0 0 1 1 2 --out lm/sangraha_o5.arpa --nemo-out lm/sangraha_o5.nemo

(8M Sangraha sentences with every test sentence removed, 5-gram; corpus from kili.lm.build_corpus).

Smoothing (Chen & Goodman, as KenLM lmplz): the highest order uses raw counts, lower orders continuation counts (number
of distinct left extensions; n-grams starting with <s> keep raw counts); discounts D1, D2, D3+ per order from the
count-of-counts; unigrams interpolate with the uniform distribution over vocab + </s>, and the unseen-token mass goes
to <unk> (NGPU-LM spreads it over the unseen ids). --prune k: drop n-grams of that order with raw count <= k
(non-decreasing over orders, so every kept n-gram's prefix and suffix are kept); the backoff weight of every context is
then recomputed so each conditional distribution sums to 1: bo(h) = (1 - sum_kept p(w|h)) / (1 - sum_kept p(w|h')).
"""
from __future__ import annotations

import argparse
import os
import tarfile
import time
from multiprocessing import Pool

import numpy as np

OFFSET = 100  # NeMo DEFAULT_TOKEN_OFFSET
_SP = None


def load_spm(nemo_path: str, out_dir: str) -> str:
    """Extract the tokenizer.model from a .nemo archive."""
    with tarfile.open(nemo_path) as tf:
        mem = [m for m in tf.getmembers() if m.name.endswith("tokenizer.model")]
        assert len(mem) == 1, [m.name for m in mem]
        p = os.path.join(out_dir, "tokenizer.model")
        with open(p, "wb") as f:
            f.write(tf.extractfile(mem[0]).read())
    return p


def _init(spm_path):
    global _SP
    import sentencepiece as spm
    _SP = spm.SentencePieceProcessor(model_file=spm_path)


def _encode(lines):
    """Sentences -> one flat id stream, each sentence wrapped in <s> = V and </s> = V + 1."""
    V = _SP.get_piece_size()
    out = []
    for x in _SP.encode(lines, out_type=int):
        if x:
            out.append(V); out.extend(x); out.append(V + 1)
    return np.asarray(out, dtype=np.int32)


def read_lines(paths: list[str], weights: list[int]) -> list[str]:
    lines = []
    for p, w in zip(paths, weights):
        with open(p, encoding="utf-8") as f:
            ls = [x.strip() for x in f if x.strip()]
        lines += ls * w
        print(f"[kn] {p}: {len(ls)} sentences x{w}", flush=True)
    return lines


def build(tok: np.ndarray, V: int, order: int) -> list[dict]:
    """Count n-grams of every order up to `order` in a <s>/</s>-wrapped id stream.

    Returns one table per order: `last` (final symbol), `pref` (id of the (k-1)-gram prefix in the previous table),
    `suf` (id of the (k-1)-gram suffix in the previous table), `cnt` (raw count). Symbols: ids 0..V-1, <s> = V,
    </s> = V + 1. N-grams never cross a sentence start."""
    S = V + 2
    T = len(tok)
    starts = np.flatnonzero(tok == V)
    pos = np.arange(T, dtype=np.int64) - np.repeat(starts, np.diff(np.append(starts, T)))  # offset from <s>
    tok64 = tok.astype(np.int64)
    u, inv, c = np.unique(tok64, return_inverse=True, return_counts=True)
    pid = inv.astype(np.int64)  # id of the (k-1)-gram ending at each position
    tabs = [dict(last=u, pref=np.full(len(u), -1), suf=np.full(len(u), -1), cnt=c)]
    for k in range(2, order + 1):
        valid = np.flatnonzero(pos >= k - 1)
        key = pid[valid - 1] * S + tok64[valid]  # prefix = the (k-1)-gram ending one position earlier
        u, first, inv, c = np.unique(key, return_index=True, return_inverse=True, return_counts=True)
        suf = pid[valid[first]]  # suffix = the (k-1)-gram ending at the same position
        tabs.append(dict(last=u % S, pref=u // S, suf=suf, cnt=c))
        new = np.full(T, -1, dtype=np.int64)
        new[valid] = inv
        pid = new
        print(f"[kn] order {k}: {len(u)} n-grams, {len(valid)} tokens", flush=True)
    return tabs


def kn(tabs: list[dict], V: int, prune: list[int]):
    """Modified KN probabilities, pruning and renormalised backoffs. Returns (probs, bows, keep, unk_p)."""
    BOS = V
    N = len(tabs)
    swb = [tabs[0]["last"] == BOS]  # starts with <s>
    for k in range(1, N):
        swb.append(swb[k - 1][tabs[k]["pref"]])
    adj = []  # adjusted counts
    for k in range(N):
        if k == N - 1:
            a = tabs[k]["cnt"].astype(np.float64)
        else:
            ext = np.bincount(tabs[k + 1]["suf"], minlength=len(tabs[k]["cnt"])).astype(np.float64)
            a = np.where(swb[k], tabs[k]["cnt"], ext).astype(np.float64)
        adj.append(a)
    Ds = []
    for k in range(N):
        a = adj[k][tabs[k]["last"] != BOS] if k == 0 else adj[k]
        n = [np.sum(a == i) for i in range(1, 5)]
        if min(n) == 0:
            D = np.array([0.0, 0.5, 1.0, 1.5])
        else:
            Y = n[0] / (n[0] + 2 * n[1])
            D = np.array([0.0, 1 - 2 * Y * n[1] / n[0], 2 - 3 * Y * n[2] / n[1], 3 - 4 * Y * n[3] / n[2]])
            D = np.clip(D, 0.0, [0, 0.99, 1.98, 2.97])
        Ds.append(D)
        print(f"[kn] order {k + 1}: n1..n4={n} D={np.round(D[1:], 3).tolist()}", flush=True)

    def dfun(k, a):
        return Ds[k][np.minimum(a, 3).astype(np.int64)]
    probs = []
    last = tabs[0]["last"]
    a = adj[0].copy(); a[last == BOS] = 0
    denom = a.sum()
    disc = np.where(a > 0, a - dfun(0, a), 0.0)
    gamma0 = (a.sum() - disc.sum()) / denom
    Vp = V + 1  # vocab + </s>
    p1 = disc / denom + gamma0 / Vp
    p1[last == BOS] = 0.0
    n_seen_tok = int(np.sum(last < V))
    unk_p = gamma0 * max(V - n_seen_tok, 1) / Vp
    probs.append(p1)
    for k in range(1, N):
        t = tabs[k]; a = adj[k]
        npref = len(tabs[k - 1]["cnt"])
        denom = np.bincount(t["pref"], weights=a, minlength=npref)
        dsum = np.bincount(t["pref"], weights=dfun(k, a), minlength=npref)
        gamma = np.divide(dsum, denom, out=np.zeros(npref), where=denom > 0)
        probs.append((a - dfun(k, a)) / denom[t["pref"]] + gamma[t["pref"]] * probs[k - 1][t["suf"]])
    assert all(prune[i] <= prune[i + 1] for i in range(len(prune) - 1)), "prune thresholds must be non-decreasing"
    keep = [np.ones(len(tabs[0]["cnt"]), bool)]
    for k in range(1, N):
        keep.append(tabs[k]["cnt"] > prune[k])
    for k in range(1, N):  # every kept n-gram's prefix and suffix must be kept (ARPA closure)
        assert keep[k - 1][tabs[k]["pref"][keep[k]]].all() and keep[k - 1][tabs[k]["suf"][keep[k]]].all()
    bows = []
    for k in range(N - 1):
        t = tabs[k + 1]; kk = keep[k + 1]
        n_ctx = len(tabs[k]["cnt"])
        s_hi = np.bincount(t["pref"][kk], weights=probs[k + 1][kk], minlength=n_ctx)
        s_lo = np.bincount(t["pref"][kk], weights=probs[k][t["suf"][kk]], minlength=n_ctx)
        bows.append(np.clip(1.0 - s_hi, 1e-12, None) / np.clip(1.0 - s_lo, 1e-12, None))
    bows.append(np.ones(len(tabs[N - 1]["cnt"])))
    return probs, bows, keep, unk_p


def write_arpa(path, tabs, probs, bows, keep, unk_p, V) -> list[int]:
    BOS = V
    sym = np.array([chr(i + OFFSET) for i in range(V)] + ["<s>", "</s>"], dtype=object)
    N = len(tabs)
    counts = [int(keep[k].sum()) + (1 if k == 0 else 0) for k in range(N)]  # + <unk>
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\data\\\n")
        for k in range(N):
            f.write(f"ngram {k + 1}={counts[k]}\n")
        for k in range(N):
            f.write(f"\n\\{k + 1}-grams:\n")
            idx = np.flatnonzero(keep[k])
            cols = [tabs[k]["last"][idx]]
            ref = idx
            for j in range(k, 0, -1):  # walk the prefix chain back to the first symbol
                ref = tabs[j]["pref"][ref]
                cols.insert(0, tabs[j - 1]["last"][ref])
            lp = np.log10(np.clip(probs[k][idx], 1e-99, None))
            if k == 0:
                lp = np.where(tabs[0]["last"][idx] == BOS, -99.0, lp)
                f.write(f"{np.log10(unk_p):.6f}\t<unk>\n")
            bo = np.log10(bows[k][idx]) if k < N - 1 else None
            strs = [sym[c] for c in cols]
            B = 1_000_000
            for s in range(0, len(idx), B):
                e = min(s + B, len(idx))
                parts = [strs[j][s:e] for j in range(len(strs))]
                names = parts[0] if len(parts) == 1 else [" ".join(x) for x in zip(*parts)]
                if bo is None:
                    f.writelines(f"{p:.6f}\t{n}\n" for p, n in zip(lp[s:e], names))
                else:
                    f.writelines(f"{p:.6f}\t{n}\t{b:.6f}\n" if b != 0.0 else f"{p:.6f}\t{n}\n"
                                 for p, n, b in zip(lp[s:e], names, bo[s:e]))
        f.write("\n\\end\\\n")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help=".nemo whose tokenizer defines the LM vocabulary")
    ap.add_argument("--text", nargs="+", required=True, help="one sentence per line")
    ap.add_argument("--weights", nargs="+", type=int, default=None, help="repeat each text file this many times")
    ap.add_argument("--order", type=int, default=5)
    ap.add_argument("--prune", nargs="+", type=int, default=None, help="per order: drop raw count <= k")
    ap.add_argument("--out", required=True, help="output .arpa")
    ap.add_argument("--nemo-out", default=None, help="also save an NGPU-LM .nemo (much faster to load)")
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    t0 = time.time()
    out_dir = os.path.dirname(a.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    spm_path = load_spm(a.model, out_dir)
    _init(spm_path)
    V = _SP.get_piece_size()
    prune = a.prune or [0] * a.order
    assert len(prune) == a.order
    lines = read_lines(a.text, a.weights or [1] * len(a.text))
    with Pool(a.procs, initializer=_init, initargs=(spm_path,)) as pool:
        parts = pool.map(_encode, [lines[i:i + 5000] for i in range(0, len(lines), 5000)])
    tok = np.concatenate(parts)
    del lines, parts
    print(f"[kn] V={V} tokens={len(tok)} (incl <s></s>) encode {time.time() - t0:.0f}s", flush=True)
    tabs = build(tok, V, a.order)
    probs, bows, keep, unk_p = kn(tabs, V, prune)
    counts = write_arpa(a.out, tabs, probs, bows, keep, unk_p, V)
    print(f"[kn] wrote {a.out}: ngrams {counts} ({os.path.getsize(a.out) / 1e6:.0f} MB) {time.time() - t0:.0f}s",
          flush=True)
    if a.nemo_out:
        from nemo.collections.asr.parts.submodules.ngram_lm import NGramGPULanguageModel
        lm = NGramGPULanguageModel.from_arpa(a.out, vocab_size=V)
        lm.save_to(a.nemo_out)
        print(f"[kn] saved {a.nemo_out} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
