"""CPU inference without PyTorch: ONNX Runtime encoder + numpy mel front end + numpy greedy TDT decoder.

Reads a directory written by `kili.infer.onnx_export` (encoder_int8.onnx, dec.npz, frontend.npz, meta.json).

  python -m kili.infer.onnx_run --dir onnx_dir a.wav b.wav [--threads 8] [--precision int8|fp32] [--decoder np|int8]

Default = the deployed configuration: int8 encoder (ONNX Runtime dynamic quantisation, per-channel weights) and the
prediction network + joint in numpy fp32. Measured on run 10 (5 epochs): int8 vs the bf16 GPU eval differs by
<= 0.3 WER on 100-clip checks; ~28x faster than real time on 8 CPU threads; ~2.8 GB peak memory; median 0.25 s
from the end of an utterance to its text. The fp32 encoder with this decoder reproduces NeMo's greedy transcripts
string for string (900 / 900 clips in the CPU study).

Audio is decoded in one pass up to `max_seconds` (120 s). The exported encoder uses dense attention restricted to
the same +-256-frame band as NeMo's rel_pos_local_attn, so memory grows with the square of the length; longer audio
is decoded in overlapping windows (`decode_windows`): 60 s chunks, each encoded with 20.48 s (the attention band)
of audio on both sides, decoder state carried across chunks.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

SR = 16000
HOP = 1280  # samples per encoder frame: 10 ms mel hop x 8 subsampling


# --------------------------------------------------------------------------------------------------------------- audio
def load_audio(path: str) -> np.ndarray:
    """Any soundfile-readable file -> float32 mono 16 kHz (channels averaged; resampled with scipy if needed)."""
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        try:
            from scipy.signal import resample_poly
        except ImportError as e:
            raise RuntimeError(f"{path}: {sr} Hz audio needs scipy to resample to 16 kHz (pip install scipy)") from e
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    return np.ascontiguousarray(x, dtype=np.float32)


# ----------------------------------------------------------------------------------------------------------- front end
def hz_to_mel(f):
    """Slaney mel scale (librosa htk=False): linear below 1 kHz, logarithmic above."""
    f = np.asarray(f, np.float64)
    lin = f / (200.0 / 3)
    return np.where(f >= 1000.0, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / (np.log(6.4) / 27.0), lin)


def mel_to_hz(m):
    m = np.asarray(m, np.float64)
    return np.where(m >= 15.0, 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - 15.0)), m * (200.0 / 3))


def mel_filterbank(sr: int = 16000, n_fft: int = 512, n_mels: int = 128, fmin: float = 0.0,
                   fmax: float | None = None) -> np.ndarray:
    """librosa.filters.mel(norm='slaney', htk=False) -> [n_mels, n_fft // 2 + 1]. Fallback only: the export writes
    NeMo's own filterbank to frontend.npz."""
    fmax = sr / 2 if fmax is None else fmax
    fft_f = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mel_f = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, None] - fft_f[None, :]
    lower = -ramps[:n_mels] / fdiff[:n_mels, None]
    upper = ramps[2:] / fdiff[1:, None]
    w = np.maximum(0.0, np.minimum(lower, upper))
    w *= (2.0 / (mel_f[2:] - mel_f[:n_mels]))[:, None]
    return w.astype(np.float32)


DEFAULT_FEATURES = {"sample_rate": 16000, "n_fft": 512, "win_length": 400, "hop_length": 160, "n_mels": 128,
                    "preemph": 0.97, "log_guard": 2.0 ** -24, "std_eps": 1e-5}


class Frontend:
    """NeMo's AudioToMelSpectrogramPreprocessor (FilterbankFeatures) at inference, in numpy: pre-emphasis 0.97,
    STFT (n_fft 512, hop 160, symmetric Hann of 400 centred in 512, center=True zero padding), power spectrum,
    128 Slaney mel bands, log(x + 2^-24), per-feature mean / std over the frames (std with n - 1, + 1e-5).
    No dither (NeMo applies it only in training)."""

    def __init__(self, cfg: dict | None = None, fb: np.ndarray | None = None, window: np.ndarray | None = None):
        c = {**DEFAULT_FEATURES, **(cfg or {})}
        self.n_fft, self.hop = int(c["n_fft"]), int(c["hop_length"])
        self.preemph, self.guard, self.eps = float(c["preemph"]), float(c["log_guard"]), float(c["std_eps"])
        if fb is None:
            fb = mel_filterbank(int(c["sample_rate"]), self.n_fft, int(c["n_mels"]))
        if window is None:
            window = np.hanning(int(c["win_length"]))  # torch.hann_window(periodic=False)
        self.fb = np.asarray(fb, np.float32)
        self.win = np.zeros(self.n_fft, np.float32)
        off = (self.n_fft - len(window)) // 2
        self.win[off:off + len(window)] = window

    @classmethod
    def from_dir(cls, d: str, meta: dict) -> "Frontend":
        p = os.path.join(d, "frontend.npz")
        if not os.path.exists(p):
            return cls(meta.get("features"))
        z = np.load(p, allow_pickle=False)
        return cls(meta.get("features"), z["fb"], z["window"])

    def __call__(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """audio [N] -> feats [1, n_mels, T], length [1] with T = N // hop + 1."""
        x = np.asarray(x, np.float32)
        n = len(x)
        x = np.concatenate([x[:1], x[1:] - self.preemph * x[:-1]])
        p = self.n_fft // 2
        x = np.pad(x, (p, p))
        frames = np.lib.stride_tricks.sliding_window_view(x, self.n_fft)[::self.hop] * self.win
        spec = np.fft.rfft(frames, axis=-1)
        power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
        m = np.log(self.fb @ power.T + self.guard)  # [n_mels, T]
        t = n // self.hop + 1
        mean = m[:, :t].mean(axis=1, keepdims=True)
        std = np.sqrt(((m[:, :t] - mean) ** 2).sum(axis=1, keepdims=True) / max(t - 1, 1)) + self.eps
        m = (m - mean) / std
        return m[None].astype(np.float32), np.array([t], np.int64)


# ------------------------------------------------------------------------------------------------------------- decoders
def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


class NumpyDecoder:
    """Prediction network (embedding + L-layer LSTM + joint.pred projection) and joint output layer in numpy fp32:
    one BLAS matrix-vector product per weight, no framework dispatch per step. dec.npz holds the weights."""

    def __init__(self, w: dict):
        self.w = w
        self.L = sum(1 for k in w if k.startswith("wih"))
        self.H = w["whh0"].shape[1]

    def init_state(self, blank: int):
        """Start of a sequence: the blank id is the start symbol (its embedding row is zero)."""
        z = np.zeros((self.L, self.H), np.float32)
        return self.pred(blank, (z, z.copy()))

    def pred(self, label: int, state):
        w, (h, c) = self.w, state
        x = w["embed"][label]
        h2, c2 = np.empty_like(h), np.empty_like(c)
        for k in range(self.L):
            gates = w[f"wih{k}"] @ x + w[f"whh{k}"] @ h[k] + w[f"b{k}"]
            i, f, g, o = np.split(gates, 4)  # PyTorch LSTM gate order
            c2[k] = _sig(f) * c[k] + _sig(i) * np.tanh(g)
            h2[k] = _sig(o) * np.tanh(c2[k])
            x = h2[k]
        return w["pred_w"] @ x + w["pred_b"], (h2, c2)

    def joint(self, f_t: np.ndarray, g: np.ndarray) -> np.ndarray:
        return self.w["joint_w"] @ np.maximum(f_t + g, 0) + self.w["joint_b"]


class OrtDecoder:
    """Prediction network and joint as int8 ONNX models (pred_int8.onnx, joint_int8.onnx), one thread each."""

    def __init__(self, d: str, ort):
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = 1, 1
        prov = ["CPUExecutionProvider"]
        self.p = ort.InferenceSession(os.path.join(d, "pred_int8.onnx"), so, providers=prov)
        self.j = ort.InferenceSession(os.path.join(d, "joint_int8.onnx"), so, providers=prov)
        shape = self.p.get_inputs()[1].shape  # h: [L, 1, H]
        self.L, self.H = int(shape[0]), int(shape[2])

    def init_state(self, blank: int):
        z = np.zeros((self.L, 1, self.H), np.float32)
        return self.pred(blank, (z, z))

    def pred(self, label: int, state):
        g, h, c = self.p.run(None, {"label": np.array([[label]], np.int64), "h": state[0], "c": state[1]})
        return g, (h, c)

    def joint(self, f_t: np.ndarray, g: np.ndarray) -> np.ndarray:
        return self.j.run(None, {"f": f_t[None], "g": g})[0][0]


# ------------------------------------------------------------------------------------------------------ greedy TDT loop
def new_state(dec, blank: int) -> dict:
    g, state = dec.init_state(blank)
    return {"g": g, "state": state, "ids": [], "f0": 0, "sym": 0}


def tdt_greedy(dec, f: np.ndarray, t: int, t_end: int, st: dict, blank: int, durations, max_symbols: int = 10) -> int:
    """Greedy TDT with NeMo's semantics over global frames [t, t_end) of f (f[i] is global frame st['f0'] + i).

    Per step: token = argmax of the first blank + 1 logits, duration = argmax of the rest. A blank advances
    max(duration, 1) frames; a label is emitted, the prediction network steps, and time advances by its duration;
    at most `max_symbols` labels per frame. Emits into st['ids'] and returns the next frame to decode, which may lie
    past t_end (a duration jump into the next window)."""
    V1 = blank + 1
    g, state, ids, f0, sym = st["g"], st["state"], st["ids"], st["f0"], st["sym"]
    while t < t_end:
        lo = dec.joint(f[t - f0], g)
        k, d = int(lo[:V1].argmax()), durations[int(lo[V1:].argmax())]
        if k == blank:
            t += max(d, 1)
            sym = 0
            continue
        ids.append(k)
        g, state = dec.pred(k, state)
        if d == 0:
            sym += 1
            if sym >= max_symbols:
                t += 1
                sym = 0
        else:
            t += d
            sym = 0
    st["g"], st["state"], st["sym"] = g, state, sym
    return t


# ---------------------------------------------------------------------------------------------------------------- model
class OnnxASR:
    """d: export directory; threads: ONNX Runtime intra-op threads for the encoder (None = ORT default);
    precision: encoder file (int8 / fp32); decoder: np (numpy fp32, default) or int8 (ONNX int8 pred + joint)."""

    def __init__(self, d: str, threads: int | None = None, precision: str = "int8", decoder: str = "np"):
        import onnxruntime as ort
        meta = json.load(open(os.path.join(d, "meta.json"), encoding="utf-8"))
        self.pieces, self.blank = meta["pieces"], int(meta["blank"])
        self.durations = list(meta["durations"])
        self.max_symbols = int(meta.get("max_symbols", 10))
        self.front = Frontend.from_dir(d, meta)
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")  # idle threads do not burn CPU
        enc = {"int8": "encoder_int8.onnx", "fp32": "encoder.onnx"}[precision]
        self.enc = ort.InferenceSession(os.path.join(d, enc), so, providers=["CPUExecutionProvider"])
        if decoder == "np":
            self.dec = NumpyDecoder(dict(np.load(os.path.join(d, "dec.npz"), allow_pickle=False)))
        elif decoder == "int8":
            self.dec = OrtDecoder(d, ort)
        else:
            raise ValueError(f"decoder must be np or int8, not {decoder!r}")

    def encode(self, audio: np.ndarray) -> tuple[np.ndarray, int]:
        """audio -> joint-projected encoder frames f [T, 640], T."""
        feats, flen = self.front(audio)
        f, n = self.enc.run(None, {"feats": feats, "flen": flen})
        return f[0], int(n[0])

    def text(self, ids: list[int]) -> str:
        return "".join(self.pieces[i] for i in ids).replace("▁", " ").strip()

    def decode_offline(self, audio: np.ndarray) -> list[int]:
        f, n = self.encode(audio)
        st = new_state(self.dec, self.blank)
        tdt_greedy(self.dec, f, 0, n, st, self.blank, self.durations, self.max_symbols)
        return st["ids"]

    def decode_windows(self, audio: np.ndarray, chunk_s: float = 60.0, context_s: float = 20.48) -> list[int]:
        """Long audio: each chunk is encoded together with context_s of audio before and after it, and only the
        chunk's frames are decoded, carrying the decoder state (last label, LSTM state, duration overshoot) across
        chunks. Window edges are whole encoder frames, so frame k of every window is the same 80 ms of audio."""
        chunk, ctx = max(1, round(chunk_s * SR / HOP)), round(context_s * SR / HOP)
        n_frames = -(-len(audio) // HOP)
        st = new_state(self.dec, self.blank)
        t, c0 = 0, 0
        while True:
            c1 = min(c0 + chunk, n_frames)
            w0, w1 = max(0, c0 - ctx), min(c1 + ctx, n_frames)
            last = c1 >= n_frames
            f, n = self.encode(audio[w0 * HOP: w1 * HOP])
            st["f0"] = w0
            t = tdt_greedy(self.dec, f, max(t, w0), w0 + n if last else c1, st, self.blank, self.durations,
                           self.max_symbols)
            if last:
                return st["ids"]
            c0 = c1

    def transcribe(self, audio: np.ndarray, max_seconds: float = 120.0) -> str:
        """float32 mono 16 kHz audio -> text."""
        if len(audio) == 0:
            return ""
        ids = self.decode_offline(audio) if len(audio) <= max_seconds * SR else self.decode_windows(audio)
        return self.text(ids)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True, help="export directory (kili.infer.onnx_export)")
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--precision", choices=["int8", "fp32"], default="int8")
    ap.add_argument("--decoder", choices=["np", "int8"], default="np")
    ap.add_argument("--max-seconds", type=float, default=120.0, help="longer audio is decoded in windows")
    a = ap.parse_args()
    asr = OnnxASR(a.dir, a.threads, a.precision, a.decoder)
    tot_audio = tot_s = 0.0
    for p in a.audio:
        x = load_audio(p)
        t0 = time.perf_counter()
        text = asr.transcribe(x, a.max_seconds)
        tot_s += time.perf_counter() - t0
        tot_audio += len(x) / SR
        print(f"{p}\t{text}", flush=True)
    print(f"[onnx_run] {tot_audio:.1f} s of audio in {tot_s:.2f} s (RTFx {tot_audio / max(tot_s, 1e-9):.1f})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
