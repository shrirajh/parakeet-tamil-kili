"""WER / CER, PER and the evaluate scorer on CPU; EER only where torch, transformers and the model are available.

  python -m pytest tests/metrics      (or: python tests/metrics/test_metrics.py)
"""
import json
import os
import random
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from kili.metrics.per import per, per_fold, sub_cost, tamil_phonemes  # noqa: E402
from kili.metrics.wer import cer, edit_distance, fold_wer, read_pairs, wer  # noqa: E402


def _naive(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def test_edit_distance():
    rng = random.Random(0)
    for _ in range(300):
        a = [rng.choice("abcd") for _ in range(rng.randint(0, 12))]
        b = [rng.choice("abcd") for _ in range(rng.randint(0, 12))]
        assert edit_distance(a, b) == _naive(a, b), (a, b)


def test_wer_cer():
    assert wer(["a b c d"], ["a b c d"]) == 0.0
    assert wer(["a b c d"], ["a x c"]) == 0.5  # 1 substitution + 1 deletion over 4 words
    assert wer(["a b", "c d e f"], ["a b", "c d e"]) == 1 / 6  # corpus rate, not a mean of rates
    assert cer(["ab cd"], ["ab cd"]) == 0.0
    assert cer(["ab cd"], ["abcd"]) == 1 / 5  # spaces are characters


def test_fold_wer():
    # loanword script, number register and digits are conventions, not errors
    r = fold_wer(["நாளைக்கு மீட்டிங்க்கு 5 மணிக்கு வா."], ["நாளைக்கு meeting-க்கு அஞ்சு மணிக்கு வா"])
    assert r == {"wer_fold": 0.0, "cer_fold": 0.0}, r
    r = fold_wer(["அவன் வீட்டுக்கு போனான்"], ["அவன் வீட்டுக்கு போன"])
    assert abs(r["wer_fold"] - 1 / 3) < 1e-9


# the PER examples shown on the website: (ref, hyp, PER %, reference phonemes)
PER_PAIRS = [
    ("மழை", "மலை", 6.25, 4),  # ழ -> ல 0.25: rain -> hill, a near miss phonetically
    ("கொல்ல", "கொள்ள", 10.0, 5),  # two ல -> ள, 0.25 each
    ("பெருநிறுவன மற்றும்", "பெரு நிறுவன மற்றும்", 0.0, 18),  # spaces are ignored
    ("வீட்டில்", "வீட்ல", 45.83, 6),  # written -> spoken form
    ("அதிசயங்களில்", "திசீங்கள்", 44.23, 13),  # a real mishearing
]


def test_per_documented_pairs():
    for ref, hyp, want, n in PER_PAIRS:
        p = per([ref], [hyp])
        assert p["ref_phonemes"] == n, (ref, p)
        assert abs(100 * p["per"] - want) < 0.01, (ref, hyp, 100 * p["per"], want)


def test_per_costs():
    assert tamil_phonemes("கா க் கி") == ["க", "ஆ", "க", "க", "இ"]
    assert tamil_phonemes("ஃபோன்") == ["ஃப", "ஓ", "ன"]
    for a, b in [("ல", "ள"), ("ல", "ழ"), ("ள", "ழ"), ("ண", "ந"), ("ண", "ன"), ("ந", "ன"), ("ர", "ற"), ("அ", "ஆ"),
                 ("இ", "ஈ"), ("ச", "ஸ")]:
        assert sub_cost(a, b) == 0.25, (a, b)
    assert sub_cost("இ", "எ") == 0.5  # neighbouring vowels
    assert sub_cost("ண", "ட") == 0.5  # nasal / stop at the same place
    assert sub_cost("க", "ப") == 0.75  # stops, far apart
    assert sub_cost("இ", "அ") == 0.75  # vowels, not neighbours
    assert sub_cost("க", "அ") == 1.0 and sub_cost("க", "x:k") == 1.0
    p = per(["பணம்"], ["பனம்"])
    assert p["per"] == 0.25 / 5 and p["per_flat"] == 1 / 5
    assert per(["அ"], [""])["per"] == 1.0


def test_per_fold():
    assert per_fold(["நாளைக்கு மீட்டிங்க்கு 5 மணிக்கு"], ["நாளைக்கு meeting-க்கு அஞ்சு மணிக்கு"]) == 0.0
    assert per_fold(["அப்புச்சாமி"], ["அப்பு சாமி"]) == 0.0


def test_read_pairs_and_scorer():
    from kili.eval.evaluate import score
    with tempfile.TemporaryDirectory() as d:
        tsv, jl = os.path.join(d, "p.tsv"), os.path.join(d, "p.jsonl")
        with open(tsv, "w", encoding="utf-8") as f:
            f.write("id\tref\thyp\nu1\tமழை பெய்தது\tமலை பெய்தது\n")
        rows = [{"audio_filepath": "u1.flac", "ref": "மழை பெய்தது", "hyp": "மலை பெய்தது"}]
        with open(jl, "w", encoding="utf-8") as f:
            f.write("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
        assert read_pairs(tsv) == read_pairs(jl) == (["மழை பெய்தது"], ["மலை பெய்தது"])
    s = score(rows, "ta")
    assert s["wer_fold"] == 0.5 and s["per_fold"] > 0 and s["per"] < s["per_flat"]
    s = score([{"audio_filepath": "a.flac", "ref": "Hello, world.", "hyp": "hello world"},
               {"audio_filepath": "b.flac", "ref": "Good morning", "hyp": "good mourning"}], "en")
    assert s["wer_basic"] == 0.25 and s["tamil_script_utts"] == 0.0
    s = score([{"audio_filepath": "c.flac", "ref": "Good morning", "hyp": "good மார்னிங்"}], "en")
    assert s["tamil_script_utts"] == 1.0


def test_eer():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        from kili.metrics.eer import Embedder, eer
        emb = Embedder()
    except Exception as e:  # no torch / transformers / model access on this machine
        try:
            import pytest
            pytest.skip(f"EER needs torch, transformers and google/embeddinggemma-2: {e}")
        except ImportError:
            print("skip test_eer:", e)
            return
    same = eer(["நாளைக்கு மீட்டிங்க்கு வா"], ["நாளைக்கு meeting-க்கு வா"], emb)["eer"]
    diff = eer(["மழை பெய்தது"], ["மலை பெய்தது"], emb)["eer"]
    assert same < 1e-3 < diff


def test_long_form_build_and_score():
    from kili.eval.long_form import build, norm_mixed, score
    with tempfile.TemporaryDirectory() as d:
        paths = {}
        for lang, word in (("ta", "வணக்கம்"), ("en", "hello")):
            paths[lang] = os.path.join(d, f"{lang}.jsonl")
            with open(paths[lang], "w", encoding="utf-8") as f:
                for i in range(200):
                    f.write(json.dumps({"audio_filepath": f"{lang}_{i}.flac", "text": f"{word} {i}",
                                        "duration": 5.0 + i % 10}, ensure_ascii=False) + "\n")
        a = build([paths["ta"]], [paths["en"]], n=2)
        b = build([paths["ta"]], [paths["en"]], n=2)
    assert a == b and set(a) == {"long_ta", "long_en", "long_hybrid"}  # deterministic
    for cat, recs in a.items():
        for rec in recs:
            dur = sum(s["duration"] + 0.4 for s in rec["segments"])
            assert 120 <= dur <= 300 + 15.4, (cat, dur)
    langs = [s["lang"] for s in a["long_hybrid"][0]["segments"][:4]]
    assert langs == ["en", "ta", "en", "ta"]
    assert norm_mixed("Hello, WORLD! வணக்கம்.") == "hello world வணக்கம்"
    r = score(["நாளைக்கு மீட்டிங்க்கு வா"], ["நாளைக்கு meeting-க்கு வா"])
    assert r["wer_fold"] == 0.0 and r["per_fold"] == 0.0 and r["wer"] > 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
