"""LM corpus: sentence splitting/filtering, manifest reading and eval-sentence exclusion (fold-space key replaced by a
plain lowercase split so the test does not depend on kili.text)."""
import json

from kili.lm.build_corpus import Exclusion, doc_sentences, read_texts


def key(s):
    return s.lower().replace(".", " ").replace(",", " ").split()


def test_doc_sentences():
    doc = ("இன்று காலை சென்னையில் கனமழை பெய்தது. (இது ஒரு குறிப்பு) சாலை போக்குவரத்து பாதிக்கப்பட்டது!\n"
           "This line is English only and is dropped.\n"
           "ஒரு")
    out = doc_sentences(doc)
    assert out == ["இன்று காலை சென்னையில் கனமழை பெய்தது.", "சாலை போக்குவரத்து பாதிக்கப்பட்டது!"]


def test_exclusion_rules():
    refs = ["one two three four five six seven eight nine ten", "alpha beta gamma delta epsilon", "x y"]
    ex = Exclusion(refs, ngram=8, key=key)
    assert ex.check("One two three four five six seven eight nine ten.") == "exact"
    assert ex.check("onetwo three four five six seven eight nine ten") == "exact"  # spaces ignored
    assert ex.check("zero two three four five six seven eight nine eleven") == "ngram"  # shares an 8-gram
    assert ex.check("he said alpha beta gamma delta epsilon today") == "contains"  # whole 5-word reference
    assert ex.check("x y z") is None  # references under 5 words only match exactly
    assert ex.check("three four five six seven eight nine") is None  # 7 shared words < 8
    assert Exclusion(refs, ngram=6, key=key).check("zero three four five six seven eight") == "ngram"


def test_read_texts(tmp_path):
    d = tmp_path / "m"
    d.mkdir()
    rows = [{"text": "a b"}, {"text_ta": "c d", "text": "ignored"}, {"text": "e f", "_skipme": True}, {"text": ""}]
    (d / "manifest_0.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    assert read_texts([str(d)]) == ["a b", "c d"]
