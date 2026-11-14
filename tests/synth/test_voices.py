"""Random-voice sampling, the shipped voice-range direction, pitch tracking and the direction fit (numpy only)."""
import json

import numpy as np

from kili.synth.synth import read_text, tts_seed
from kili.synth.voices import (DIM, SCALE, auc, evaluate_direction, f0_median, fit_direction, load_direction,
                               random_voice, spearman, unit, voice_seed)


def test_shipped_direction():
    raw = np.load(__import__("kili.synth.voices", fromlist=["DIRECTION"]).DIRECTION)
    assert raw.shape == (DIM,) and raw.dtype == np.float32
    assert abs(np.linalg.norm(raw) - 1) < 1e-5


def test_voice_seed_stable():
    s = voice_seed("synth_tanglish", 0, 3, 7)
    assert s == voice_seed("synth_tanglish", 0, 3, 7)
    assert 0 <= s < 2 ** 63
    assert len({voice_seed("x", p, 0, k) for p in range(3) for k in range(50)}) == 150


def test_random_voice():
    d = load_direction()
    v1, a1 = random_voice(123, d)
    v2, a2 = random_voice(123, d)
    assert np.array_equal(v1, v2) and a1 == a2
    assert v1.dtype == np.float32 and abs(np.linalg.norm(v1) - 1) < 1e-5
    vs, as_ = zip(*(random_voice(s, d) for s in range(500)))
    as_ = np.array(as_)
    assert as_.min() >= -1 and as_.max() <= 1 and abs(as_.mean()) < 0.1
    proj = np.array([float(v @ d) for v in vs])
    assert np.corrcoef(as_, proj)[0, 1] > 0.9  # the dial moves voices along d
    assert abs(proj[as_ > 0.9].mean() - SCALE) < 0.1  # a = +1 shifts the projection by ~ the scale
    bases = [unit(np.random.default_rng(s).standard_normal(DIM)) for s in range(500)]
    assert abs(np.mean([b @ d for b in bases])) < 0.02  # random bases are unbiased on d


def voiced(f0, sr=16000, secs=1.0):
    """5 harmonics with 2% / 5 Hz vibrato (a perfectly periodic tone makes every pitch multiple an equal peak)."""
    t = np.arange(int(sr * secs)) / sr
    phase = 2 * np.pi * np.cumsum(f0 * (1 + 0.02 * np.sin(2 * np.pi * 5 * t))) / sr
    return sum(np.sin(h * phase) / h for h in range(1, 6)).astype(np.float32)


def test_f0_median():
    for f in (110.0, 150.0, 240.0):
        est = f0_median(voiced(f))
        assert abs(est - f) / f < 0.03, (f, est)
    assert f0_median(np.zeros(16000, np.float32)) is None


def test_fit_direction_recovers_planted_axis():
    rng = np.random.default_rng(0)
    true = unit(rng.standard_normal(DIM))
    groups = np.repeat(np.arange(200), 8)
    voice = rng.standard_normal((200, DIM))
    X = voice[groups] + 0.3 * rng.standard_normal((len(groups), DIM))
    Xu = np.stack([unit(x) for x in X])
    y = 7.3 + 3.0 * (Xu @ true) + 0.02 * rng.standard_normal(len(groups))
    d, r2 = fit_direction(X, y, groups)
    assert d @ true > 0.9 and r2 > 0.7
    d2, _ = fit_direction(X, -y, groups)
    assert d2 @ true < -0.8  # oriented: + = higher target


def test_rank_metrics():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert auc([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0]) == 1.0
    assert auc([0.5, 0.5], [1, 0]) == 0.5


def test_evaluate_direction():
    rng = np.random.default_rng(1)
    d = load_direction()
    n = 200
    female = np.arange(n) < n // 2
    f0 = np.where(female, 210.0, 130.0) * np.exp(0.1 * rng.standard_normal(n))
    X = rng.standard_normal((n, DIM)) * 0.05 + np.outer(np.log2(f0) - 7.4, d)
    res = evaluate_direction(X, f0, female, d)
    assert res["direction"]["auc_female_vs_male"] > 0.9 and res["direction"]["spearman_f0"] > 0.8
    assert res["n_female"] == res["n_male"] == n // 2
    json.dumps(res)


def test_read_text_and_seeds(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text('{"tts": "office-ல meeting", "label": "office-ல meeting"}\n{"text": "வணக்கம்"}\n', encoding="utf-8")
    assert read_text(str(p)) == [{"tts": "office-ல meeting", "label": "office-ல meeting"},
                                 {"tts": "வணக்கம்", "label": "வணக்கம்"}]
    q = tmp_path / "t.txt"
    q.write_text("ஒன்று\n\nஇரண்டு\n", encoding="utf-8")
    assert [r["tts"] for r in read_text(str(q))] == ["ஒன்று", "இரண்டு"]
    assert tts_seed(5, 0) == 7005 and tts_seed(5, 1) == 7005 + 1_000_003
