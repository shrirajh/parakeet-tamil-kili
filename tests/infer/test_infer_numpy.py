"""numpy-only tests of the CPU runtime: greedy TDT loop, windowed decoding, mel front end."""
import numpy as np

from kili.infer.onnx_run import (HOP, SR, Frontend, NumpyDecoder, OnnxASR, hz_to_mel, mel_filterbank, mel_to_hz,
                                 new_state, tdt_greedy)

BLANK = 3  # tokens 0..2, blank 3
DUR = [0, 1, 2, 3, 4]


class ScriptDec:
    """Fake decoder: f_t[0] = frame index, g[0] = labels emitted so far; script(frame, n) -> (token, duration)."""

    def __init__(self, script):
        self.script = script

    def init_state(self, blank):
        return np.array([0.0]), 0

    def pred(self, label, state):
        return np.array([state + 1.0]), state + 1

    def joint(self, f_t, g):
        k, d = self.script(int(f_t[0]), int(g[0]))
        lo = np.full(BLANK + 1 + len(DUR), -10.0)
        lo[k] = 1.0
        lo[BLANK + 1 + DUR.index(d)] = 1.0
        return lo


_dec = ScriptDec


def _run(dec, T, max_symbols=10):
    f = np.arange(T, dtype=np.float32)[:, None]
    st = new_state(dec, BLANK)
    tdt_greedy(dec, f, 0, T, st, BLANK, DUR, max_symbols)
    return st["ids"]


def test_tdt_semantics():
    table = {(0, 0): (1, 0), (0, 1): (2, 2), (2, 2): (BLANK, 0), (3, 2): (BLANK, 2), (5, 2): (0, 1)}
    # label with duration 0 stays on the frame; blank with duration 0 advances 1; durations jump
    assert _run(_dec(lambda t, n: table[(t, n)]), 6) == [1, 2, 0]


def test_max_symbols_per_frame():
    dec = _dec(lambda t, n: (1, 0) if t == 0 else (BLANK, 1))
    assert _run(dec, 4, max_symbols=3) == [1, 1, 1]


def test_split_calls_equal_one_pass():
    script = lambda t, n: (BLANK, (t + n) % 3) if t % 4 == 0 else (t % 3, 1 + (t + n) % 4)  # noqa: E731
    T = 40
    one = _run(_dec(script), T)
    dec = _dec(script)
    f = np.arange(T, dtype=np.float32)[:, None]
    st, t = new_state(dec, BLANK), 0
    for c0 in range(0, T, 7):  # chunks of 7 frames, each seen through a window starting 2 frames earlier
        w0 = max(0, c0 - 2)
        st["f0"] = w0
        t = tdt_greedy(dec, f[w0:], max(t, w0), min(c0 + 7, T), st, BLANK, DUR)
    assert st["ids"] == one


def test_decode_windows_equals_offline():
    script = lambda t, n: (BLANK, 0) if t % 5 == 0 else (t % 3, (t * 7 + n) % 5)  # noqa: E731
    asr = OnnxASR.__new__(OnnxASR)
    asr.dec, asr.blank, asr.durations, asr.max_symbols = _dec(script), BLANK, DUR, 10

    def encode(audio):  # frame k of a window = global frame index (audio holds the frame index of each sample)
        n = len(audio) // HOP + 1
        return (audio[0] + np.arange(n, dtype=np.float32))[:, None], n
    asr.encode = encode
    audio = (np.arange(61 * HOP) // HOP).astype(np.float32)
    off = asr.decode_offline(audio)
    assert len(off) > 10
    for chunk_s, ctx_s in ((0.24, 0.16), (0.08, 0.0), (1.0, 0.4)):
        assert asr.decode_windows(audio, chunk_s, ctx_s) == off


def test_mel_scale():
    assert abs(float(hz_to_mel(1000.0)) - 15.0) < 1e-9
    f = np.array([0.0, 300.0, 1000.0, 4000.0, 8000.0])
    assert np.allclose(mel_to_hz(hz_to_mel(f)), f)


def test_filterbank():
    fb = mel_filterbank(16000, 512, 128)
    assert fb.shape == (128, 257) and fb.dtype == np.float32
    assert (fb >= 0).all() and (fb.max(axis=1) > 0).all()
    assert fb[:, -1].max() == 0.0  # nothing above fmax = Nyquist peaks at the last bin's upper edge


def test_frontend_shapes_and_normalisation():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(SR).astype(np.float32) * 0.1
    feats, n = Frontend()(x)
    assert feats.shape == (1, 128, SR // 160 + 1) and feats.dtype == np.float32
    assert n.dtype == np.int64 and int(n[0]) == 101
    assert np.allclose(feats[0].mean(axis=1), 0, atol=1e-4)
    assert np.allclose(feats[0].std(axis=1, ddof=1), 1, atol=1e-3)
    assert Frontend()(x[:1000])[0].shape == (1, 128, 1000 // 160 + 1)


def test_filterbank_tone_band():
    """A 1 kHz tone's power spectrum peaks in the mel band centred nearest 1 kHz."""
    fe = Frontend()
    frame = np.sin(2 * np.pi * 1000 * np.arange(512) / SR) * fe.win
    band = int(np.argmax(fe.fb @ np.abs(np.fft.rfft(frame)) ** 2))
    centres = mel_to_hz(np.linspace(0, hz_to_mel(8000), 130))[1:-1]
    assert band == int(np.argmin(np.abs(centres - 1000)))


def test_numpy_decoder_lstm_step():
    """One-layer LSTM step against the textbook equations (PyTorch gate order i, f, g, o)."""
    rng = np.random.default_rng(1)
    E, H, V = 4, 3, 5
    w = {"embed": rng.standard_normal((V + 1, E)).astype(np.float32), "wih0": rng.standard_normal((4 * H, E)).astype(np.float32),
         "whh0": rng.standard_normal((4 * H, H)).astype(np.float32), "b0": rng.standard_normal(4 * H).astype(np.float32),
         "pred_w": np.eye(H, dtype=np.float32), "pred_b": np.zeros(H, np.float32),
         "joint_w": rng.standard_normal((V + 6, H)).astype(np.float32), "joint_b": np.zeros(V + 6, np.float32)}
    w["embed"][V] = 0
    dec = NumpyDecoder(w)
    h0, c0 = rng.standard_normal((1, H)).astype(np.float32), rng.standard_normal((1, H)).astype(np.float32)
    g, (h1, c1) = dec.pred(2, (h0, c0))
    z = w["wih0"] @ w["embed"][2] + w["whh0"] @ h0[0] + w["b0"]
    s = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
    c_ref = s(z[H:2 * H]) * c0[0] + s(z[:H]) * np.tanh(z[2 * H:3 * H])
    h_ref = s(z[3 * H:]) * np.tanh(c_ref)
    assert np.allclose(c1[0], c_ref, atol=1e-6) and np.allclose(h1[0], h_ref, atol=1e-6) and np.allclose(g, h_ref)
    g0, (h, c) = dec.init_state(V)  # start symbol = blank (zero embedding row), zero state
    assert np.allclose(c[0], s(w["b0"][H:2 * H]) * 0 + s(w["b0"][:H]) * np.tanh(w["b0"][2 * H:3 * H]), atol=1e-6)
