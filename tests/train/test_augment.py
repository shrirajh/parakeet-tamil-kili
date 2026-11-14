"""kili.train.augment (CPU, or CUDA when available; needs torch; the pairing tests need lhotse + soundfile + numpy).

  pytest tests/train/test_augment.py
"""
from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from kili.train import augment  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000


def _tones():
    t = torch.arange(SR * 3, device=DEV) / SR
    x = torch.zeros(3, SR * 4, device=DEV)
    for b in range(3):
        x[b, :SR * 3] = 0.3 * torch.sin(2 * torch.pi * (300 + 2000 * b) * t) * (torch.rand(SR * 3, device=DEV) > 0.5)
    return x, torch.tensor([SR * 3] * 3, device=DEV)


def test_reverb_keeps_level_and_padding():
    x, lens = _tones()
    y = augment.gpu_augment(x, lens, SR, p_rir=1.0, p_lowpass=0.0)
    r_in, r_out = x[:, :SR * 3].pow(2).mean(1).sqrt(), y[:, :SR * 3].pow(2).mean(1).sqrt()
    assert bool(torch.isfinite(y).all()) and bool((y[:, SR * 3:] == 0).all())
    assert float(((r_out / r_in) - 1).abs().max()) < 0.02
    assert float((y - x).abs().mean()) > 1e-3


def test_lowpass_removes_high_band():
    x, lens = _tones()
    y = augment.gpu_augment(x, lens, SR, p_rir=0.0, p_lowpass=1.0, hz=(1000.0, 1000.0))
    f = torch.fft.rfftfreq(SR * 3, 1 / SR).to(DEV)
    s_in, s_out = torch.fft.rfft(x[2, :SR * 3]).abs(), torch.fft.rfft(y[2, :SR * 3]).abs()  # 4.3 kHz tone
    assert float(s_out[f > 1400].pow(2).sum() / s_in[f > 1400].pow(2).sum()) < 1e-3
    assert bool(torch.isfinite(y).all())


def test_long_clip():
    x = torch.randn(1, SR * 120, device=DEV) * 0.1
    y = augment.gpu_augment(x, torch.tensor([SR * 120], device=DEV), SR, p_rir=1.0, p_lowpass=1.0)
    assert bool(torch.isfinite(y).all())
    assert abs(float(y.pow(2).mean().sqrt() / x.pow(2).mean().sqrt()) - 1) < 0.5


def test_speed_perturbation():
    """f x faster: a 1 kHz tone becomes f kHz, L samples become round(L / f), amplitude kept, padding zero."""
    L, T = 32000, 40000
    t = torch.arange(L) / SR
    x = torch.zeros(2, T)
    x[:, :L] = 0.5 * torch.sin(2 * torch.pi * 1000 * t)
    lens = torch.tensor([L, L])
    g = torch.Generator().manual_seed(0)
    y, ny = augment.speed(x, lens, p=1.0, factors=(0.9, 0.9), gen=g)
    M = round(L / 0.9)
    assert ny.tolist() == [M, M] and y.shape[1] == max(T, M)
    peak = torch.fft.rfft(y[0, :M]).abs().argmax().item() * SR / M
    assert abs(peak - 900.0) < 5
    assert abs(y[0, :M].pow(2).mean().sqrt().item() - 0.5 / 2 ** 0.5) < 0.02
    assert y[0, M:].abs().max().item() == 0.0
    y2, n2 = augment.speed(x, lens, p=1.0, factors=(1.1, 1.1), gen=g)
    M2 = round(L / 1.1)
    assert n2.tolist() == [M2, M2]
    assert abs(torch.fft.rfft(y2[0, :M2]).abs().argmax().item() * SR / M2 - 1100.0) < 5
    y3, n3 = augment.speed(x, lens, p=0.0)
    assert y3 is x and n3 is lens


# ---------------------------------------------------------------------------------------------- cut stream

def _cut(cid, text, dur, freq, seed):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    lhotse = pytest.importorskip("lhotse")
    import io

    from lhotse.audio import AudioSource
    t = np.arange(int(dur * SR)) / SR
    rng = np.random.default_rng(seed)
    x = 0.3 * np.sin(2 * np.pi * freq * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)) + 0.01 * rng.standard_normal(len(t))
    b = io.BytesIO()
    sf.write(b, x.astype(np.float32), SR, format="WAV", subtype="FLOAT")
    rec = lhotse.Recording(id=cid, sources=[AudioSource(type="memory", channels=[0], source=b.getvalue())],
                           sampling_rate=SR, num_samples=len(t), duration=dur)
    return lhotse.MonoCut(id=cid, start=0, duration=dur, channel=0, recording=rec,
                          supervisions=[lhotse.SupervisionSegment(id=cid, recording_id=cid, start=0, duration=dur,
                                                                  text=text)])


def _text(c):
    return " ".join(s.text for s in c.supervisions)


def test_babble_and_overlap():
    np = pytest.importorskip("numpy")
    a = _cut("a", "வணக்கம் நண்பர்களே", 4.0, 220, 1)
    b = _cut("b", "சரி போகலாம்", 3.0, 330, 2)
    ps = augment.PairingStream([], 0.0, 24.0, 0.2, 0, p_overlap=1.0, overlap_s=(1.0, 1.0), overlap_snr=(0.0, 0.0),
                               p_babble=1.0, babble_snr=(10.0, 10.0))
    rng = random.Random(0)
    m = ps._single(a, rng, [b])  # babble: main label and duration kept, background near 10 dB
    assert abs(m.duration - a.duration) < 1e-6 and _text(m) == _text(a)
    xa, xm = a.load_audio()[0], m.load_audio()[0]
    snr = 10 * np.log10(np.mean(xa ** 2) / max(np.mean((xm - xa) ** 2), 1e-12))
    assert 6.0 < snr < 14.0
    j = ps._pair(a, b, rng)  # overlap at the turn change: 4 + 3 - 1 s, label a then b
    assert abs(j.duration - 6.0) < 1e-3
    assert _text(j) == "வணக்கம் நண்பர்களே சரி போகலாம்"


def test_stream_pairs():
    src = [_cut(f"s{i}", f"t{i}", 2.0 + (i % 3), 200 + i, i) for i in range(60)]
    out = list(augment.PairingStream(src, 0.4, 24.0, 0.2, 7))
    joined = [c for c in out if len(c.supervisions) > 1 or " " in _text(c)]
    assert 0 < len(joined) < len(out) and all(c.duration <= 24.0 + 1e-6 for c in out)
    assert sorted(" ".join(_text(c) for c in out).split()) == sorted(f"t{i}" for i in range(60))  # every label once
