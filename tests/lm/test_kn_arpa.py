"""KN 5-gram LM on a toy id corpus: counts, ARPA closure, and every conditional distribution summing to 1."""
import math

import numpy as np

from kili.lm.kn_arpa import OFFSET, build, kn, write_arpa

V = 12  # ids 0..11; ids 9-11 never occur (their mass goes to <unk>)
SENTS = [[0, 1, 2, 3], [0, 1, 2, 4], [0, 1, 5], [6, 1, 2, 3], [7, 8, 2, 3, 0], [0, 1, 2, 3], [4, 4, 4], [5, 1, 2]]


def stream(sents):
    out = []
    for s in sents:
        out += [V, *s, V + 1]
    return np.asarray(out, np.int32)


def table_counts(tabs, k):
    """{n-gram tuple: raw count} for order k (1-based)."""
    t = tabs[k - 1]
    out = {}
    for i in range(len(t["cnt"])):
        gram, ref = [int(t["last"][i])], i
        for j in range(k - 1, 0, -1):
            ref = tabs[j]["pref"][ref]
            gram.insert(0, int(tabs[j - 1]["last"][ref]))
        out[tuple(gram)] = int(t["cnt"][i])
    return out


def test_counts():
    tabs = build(stream(SENTS), V, 3)
    c1, c2, c3 = (table_counts(tabs, k) for k in (1, 2, 3))
    assert c1[(1,)] == 6 and c1[(V,)] == len(SENTS) and c1[(V + 1,)] == len(SENTS)
    assert c2[(0, 1)] == 4 and c2[(V, 0)] == 4 and c2[(3, V + 1)] == 3
    assert c3[(0, 1, 2)] == 3 and c3[(1, 2, 3)] == 3
    assert (V + 1, V) not in c2  # n-grams never cross a sentence boundary
    # suffix pointers: the suffix of (0, 1, 2) is (1, 2)
    t3, t2 = tabs[2], tabs[1]
    i = next(i for i, g in enumerate(table_counts(tabs, 3)) if g == (0, 1, 2))
    j = t3["suf"][i]
    assert (int(t2["last"][j]), int(tabs[0]["last"][t2["pref"][j]])) == (2, 1)


def read_arpa(path):
    grams, unk = {}, None
    with open(path, encoding="utf-8") as f:
        section = 0
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("\\") and line.endswith("-grams:"):
                section = int(line[1:line.index("-")])
                continue
            if not section or not line or line == "\\end\\":
                continue
            parts = line.split("\t")
            toks = tuple(parts[1].split(" "))
            if toks == ("<unk>",):
                unk = 10 ** float(parts[0])
                continue
            grams[toks] = (10 ** float(parts[0]), 10 ** float(parts[2]) if len(parts) > 2 else 1.0)
    return grams, unk


def sym(i):
    return "<s>" if i == V else "</s>" if i == V + 1 else chr(i + OFFSET)


def check_normalised(order, prune, tmp_path):
    tabs = build(stream(SENTS), V, order)
    probs, bows, keep, unk_p = kn(tabs, V, prune)
    path = tmp_path / "lm.arpa"
    counts = write_arpa(str(path), tabs, probs, bows, keep, unk_p, V)
    grams, unk = read_arpa(path)
    assert sum(counts) == len(grams) + 1  # + <unk>
    seen = {g[0] for g in grams if len(g) == 1}
    unseen = [sym(i) for i in range(V) if sym(i) not in seen]
    for g in grams:  # ARPA closure: every n-gram's prefix and suffix exist
        if len(g) > 1:
            assert g[:-1] in grams and g[1:] in grams

    def p(w, h):
        if h + (w,) in grams:
            return grams[h + (w,)][0]
        if not h:
            return unk / len(unseen)
        bo = grams[h][1] if h in grams else 1.0
        return bo * p(w, h[1:])
    vocab = [sym(i) for i in range(V)] + ["</s>"]
    contexts = [g for g in grams if len(g) < order and g[-1] != "</s>"]
    assert contexts
    for h in contexts:
        total = sum(p(w, h) for w in vocab)
        assert math.isclose(total, 1.0, abs_tol=1e-4), (h, total)


def test_distributions_sum_to_one(tmp_path):
    check_normalised(5, [0, 0, 0, 0, 0], tmp_path)


def test_distributions_sum_to_one_pruned(tmp_path):
    check_normalised(4, [0, 0, 1, 1], tmp_path)
