"""Language gate logic (CPU) and the gated LM decode (GPU + NeMo; skipped otherwise)."""
import json
import os

import pytest

from kili.lm.decode import lm_decoding_override
from kili.lm.gate import latin_share, merge, use_lm


def test_latin_share():
    assert latin_share("") == 0.0
    assert latin_share("123 ?!") == 0.0
    assert latin_share("hello world") == 1.0
    assert latin_share("வணக்கம்") == 0.0
    assert 0.0 < latin_share("நாளைக்கு ஆபீஸ்ல ஒரு meeting இருக்கு") < 0.5
    assert latin_share("நாளைக்கு office-ல meeting") > 0.5  # letters, not words: Tamil signs count one each


def test_use_lm():
    assert use_lm("நாளைக்கு office-ல ஒரு meeting இருக்கு")
    assert not use_lm("I will call you tomorrow")
    assert not use_lm("please send the report today சரி")
    assert use_lm("")  # no letters: the LM may act
    assert use_lm("ab அஆ")  # exactly half Latin is not "more than half"


def test_merge():
    greedy = ["வணக்கம் எப்படி இருக்கீங்க", "see you tomorrow", "நன்றி"]
    lm = ["வணக்கம், எப்படி இருக்கீங்க", "see you to morrow", None]
    assert merge(greedy, lm) == ["வணக்கம், எப்படி இருக்கீங்க", "see you tomorrow", "நன்றி"]


def test_override():
    o = lm_decoding_override("lm.nemo")
    assert o["strategy"] == "malsd_batch"
    assert o["beam"] == {"beam_size": 4, "ngram_lm_model": "lm.nemo", "ngram_lm_alpha": 0.2,
                         "pruning_mode": "late", "blank_lm_score_mode": "lm_weighted_full"}


def test_transcribe_lm_gpu():
    """KILI_TEST_MODEL (.nemo or HF id), KILI_TEST_LM (.arpa/.nemo) and KILI_TEST_AUDIO (comma-separated 16 kHz files,
    Tamil and English) must be set."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("nemo")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    model_id, lm, audio = (os.environ.get(k) for k in ("KILI_TEST_MODEL", "KILI_TEST_LM", "KILI_TEST_AUDIO"))
    if not (model_id and lm and audio):
        pytest.skip("set KILI_TEST_MODEL, KILI_TEST_LM, KILI_TEST_AUDIO")
    from kili.infer.transcribe import load_model
    from kili.lm.decode import _transcribe, transcribe_lm
    model = load_model(model_id)
    paths = audio.split(",")
    before = json.dumps(dict(model.cfg.decoding), default=str)
    greedy = _transcribe(model, paths, 4)
    out = transcribe_lm(model, paths, lm, batch_size=4)
    assert len(out) == len(paths)
    for g, h in zip(greedy, out):
        if not use_lm(g):
            assert h == g  # English clips keep the greedy output
    assert json.dumps(dict(model.cfg.decoding), default=str) == before  # decoding config restored
