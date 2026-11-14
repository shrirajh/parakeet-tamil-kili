"""Training augmentation used for run 10: GPU waveform effects and multi-talker clips on the lhotse cut stream.

Run 10's augmentation, per training clip (probabilities as trained):
  background noise  0.3, SNR 0-20 dB, room-tone clips      NeMo lhotse option (noise_path / noise_snr / noise_mix_prob)
  compression       0.1, opus / mp3 / gsm (0.4 / 0.3 / 0.3)  NeMo lhotse option (compression_*)
  echo (reverb)     0.3, RT60 0.2-1.0 s, DRR -3..12 dB       gpu_augment() below
  phone line        0.1, low-pass at 3.4-8 kHz              gpu_augment() below
  speed             0.5, 0.9-1.1x (pitch moves with it)      speed() below
  pairs             0.25 of output clips are two clips joined (different speakers / languages), <= 24 s
  overlap           0.3 of joined pairs overlap 0.2-1.5 s at the turn change, second voice -6..+3 dB vs the first
  babble            0.25 of single clips get 1-3 background talkers at 3-20 dB below the main voice
(the last three: PairingStream below; the noise / compression rows are configs/train_r10.yaml train_ds options).

GPU effects run on the raw waveform right before the preprocessor (training only; a few ms per batch, no dataloader
CPU): install(model, ...) wraps model.preprocessor.forward. Long clips use a block-wise overlap-add FFT convolution
with a fixed FFT size; a clip whose augmentation fails passes through unchanged; padding stays zero.

The cut stream is wrapped before NeMo's tokenisation and duration bucketing (joined clips are bucketed by their joined
duration): patch_nemo(...) wraps nemo.collections.common.data.lhotse.dataloader.read_cutset_from_config for shuffled
(training) loaders only.

  from kili.train import augment
  augment.install(model, p_rir=0.3, p_lowpass=0.1, p_speed=0.5)
  augment.patch_nemo(p=0.25, max_s=24.0, gap_s=0.2, p_overlap=0.3, p_babble=0.25, babble_snr=(3.0, 20.0),
                     babble_n=(1, 3), src_max_s=125.0)
"""
from __future__ import annotations

import math
import random
from collections import deque

import torch

# ---------------------------------------------------------------------------------------------- GPU waveform effects


def _rir(n: int, sr: int, rt60: float, drr_db: float, device) -> torch.Tensor:
    """Synthetic room response: direct path + exponentially decaying noise tail at the given RT60 / DRR."""
    t = torch.arange(n, device=device, dtype=torch.float32) / sr
    tail = torch.randn(n, device=device) * torch.exp(-6.9078 * t / rt60)
    tail[0] = 0.0
    e_tail = tail.pow(2).sum().clamp_min(1e-12)
    g = math.sqrt(1.0 / (10 ** (drr_db / 10)))  # direct energy 1 -> tail energy 10^(-drr/10)
    h = tail * (g / e_tail.sqrt())
    h[0] = 1.0
    return h


def fft_conv(s: torch.Tensor, h: torch.Tensor, block: int = 1 << 16) -> torch.Tensor:
    """Linear convolution of s [L] with h [n], first L samples (overlap-add, fixed FFT size)."""
    L, n = s.shape[0], h.shape[0]
    nfft = 1 << (block + n - 1).bit_length()
    H = torch.fft.rfft(h, nfft)
    out = torch.zeros(L + n - 1, device=s.device, dtype=torch.float32)
    for i in range(0, L, block):
        seg = s[i:i + block]
        y = torch.fft.irfft(torch.fft.rfft(seg, nfft) * H, nfft)[: seg.shape[0] + n - 1]
        out[i:i + y.shape[0]] += y
    return out[:L]


def _lowpass_fir(fc: float, sr: int, taps: int, device) -> torch.Tensor:
    m = torch.arange(taps, device=device, dtype=torch.float32) - (taps - 1) / 2
    h = torch.sinc(2 * fc / sr * m) * (2 * fc / sr) * torch.hamming_window(taps, periodic=False, device=device)
    return h / h.sum()


def gpu_augment(x: torch.Tensor, lengths: torch.Tensor, sr: int, p_rir: float, rt60=(0.2, 1.0), drr_db=(-3.0, 12.0),
                p_lowpass: float = 0.0, hz=(3400.0, 8000.0), gen=None) -> torch.Tensor:
    """x [B, T] waveform, lengths [B] samples. Per clip: reverb with prob p_rir (output RMS matched to the input),
    then a 511-tap windowed-sinc low-pass at fc ~ U(hz) with prob p_lowpass (zero phase). Returns a new tensor."""
    B, T = x.shape
    if B == 0 or (p_rir <= 0 and p_lowpass <= 0):
        return x
    dev = x.device
    u = torch.rand(B, 2, generator=gen, device="cpu")
    rir_rows = [b for b in range(B) if u[b, 0] < p_rir]
    lp_rows = [b for b in range(B) if u[b, 1] < p_lowpass]
    if not rir_rows and not lp_rows:
        return x
    y = x.clone()
    lens = lengths.tolist()
    for b in rir_rows:
        L = int(lens[b])
        if L < sr // 10:
            continue
        try:
            r60 = float(torch.empty(1).uniform_(*rt60, generator=gen))
            drr = float(torch.empty(1).uniform_(*drr_db, generator=gen))
            n = min(int(sr * min(1.2, 1.5 * r60)), L)
            s = y[b, :L].float()
            out = fft_conv(s, _rir(n, sr, r60, drr, dev))
            rms_in, rms_out = s.pow(2).mean().sqrt(), out.pow(2).mean().sqrt().clamp_min(1e-8)
            y[b, :L] = (out * (rms_in / rms_out)).to(y.dtype)
        except RuntimeError:  # e.g. no cuFFT workspace on a nearly full GPU: skip this clip, never the run
            pass
    for b in lp_rows:
        L = int(lens[b])
        if L < sr // 10:
            continue
        fc = float(torch.empty(1).uniform_(*hz, generator=gen))
        if fc >= sr / 2 - 50:
            continue
        try:
            taps = 511
            h = _lowpass_fir(fc, sr, taps, dev)
            s = y[b, :L].float()
            out = fft_conv(torch.nn.functional.pad(s, (0, taps // 2)), h)[taps // 2: taps // 2 + L]  # zero phase
            y[b, :L] = out.to(y.dtype)
        except RuntimeError:
            pass
    return y


def resample_sinc(s: torch.Tensor, M: int, half: int = 16, block: int = 32768) -> torch.Tensor:
    """s [L] -> [M] samples spanning the same signal: windowed-sinc interpolation (2 x half taps, Hann window),
    low-pass at min(1, M/L) of Nyquist so speeding up does not alias. Block-wise over the output (bounded memory)."""
    L = s.shape[0]
    step = L / M
    cut = min(1.0, M / L)
    offs = torch.arange(-half + 1, half + 1, device=s.device)
    out = torch.empty(M, device=s.device, dtype=torch.float32)
    sf = s.float()
    for k0 in range(0, M, block):
        pos = torch.arange(k0, min(M, k0 + block), device=s.device, dtype=torch.float64) * step
        i0 = pos.floor().long()
        idx = i0[:, None] + offs[None, :]
        d = (pos[:, None] - idx.double()).float()
        w = cut * torch.sinc(cut * d) * (0.5 + 0.5 * torch.cos(torch.pi * d / half)) * (d.abs() < half)
        v = sf[idx.clamp(0, L - 1)] * ((idx >= 0) & (idx < L))
        out[k0: k0 + pos.shape[0]] = (v * w).sum(1)
    return out


def speed(x: torch.Tensor, lengths: torch.Tensor, p: float, factors=(0.9, 1.1), gen=None):
    """Speed perturbation: with prob p a clip plays f ~ U(factors) times faster (resampled to round(L / f) samples,
    level kept; transcripts unchanged). Returns (x', lengths'); x' is wider than x when a clip is slowed down.
    The sampler packs batches by the original durations: cap micro-batches at duration x min(factors) of memory."""
    B, T = x.shape
    if B == 0 or p <= 0:
        return x, lengths
    u = torch.rand(B, generator=gen)
    f = torch.empty(B).uniform_(float(factors[0]), float(factors[1]), generator=gen)
    rows = [(b, float(f[b])) for b in range(B) if u[b] < p]
    if not rows:
        return x, lengths
    lens = [int(v) for v in lengths.tolist()]
    new, outs = list(lens), {}
    for b, fb in rows:
        L = lens[b]
        M = max(1, int(round(L / fb)))
        if L < 160 or M == L:
            continue
        try:
            outs[b] = resample_sinc(x[b, :L], M)
            new[b] = M
        except RuntimeError:
            pass
    if not outs:
        return x, lengths
    y = x.new_zeros(B, max(T, max(new)))
    y[:, :T] = x
    for b, o in outs.items():
        y[b].zero_()
        y[b, : o.shape[0]] = o.to(y.dtype)
    return y, torch.tensor(new, device=lengths.device, dtype=lengths.dtype)


def install(model, p_rir: float = 0.0, rt60=(0.2, 1.0), drr_db=(-3.0, 12.0), p_lowpass: float = 0.0,
            hz=(3400.0, 8000.0), p_speed: float = 0.0, speed_factors=(0.9, 1.1)) -> None:
    """Run speed, then reverb / low-pass on the waveform before model.preprocessor, in training mode only."""
    pre = model.preprocessor
    orig = pre.forward
    sr = int(getattr(model.cfg.preprocessor, "sample_rate", 16000))

    def forward(*args, **kw):
        if model.training and (p_rir > 0 or p_lowpass > 0 or p_speed > 0):
            a = dict(zip(["input_signal", "length"], args))
            a.update(kw)
            if a.get("input_signal") is not None and a.get("length") is not None:
                with torch.no_grad():
                    if p_speed > 0:  # first: reverb / band limit then act on the re-timed audio
                        a["input_signal"], a["length"] = speed(a["input_signal"], a["length"], p_speed, speed_factors)
                    a["input_signal"] = gpu_augment(a["input_signal"], a["length"], sr, p_rir, rt60, drr_db,
                                                    p_lowpass, hz)
                return orig(**a)
        return orig(*args, **kw)

    pre.forward = forward


# ---------------------------------------------------------------------------------------------- cut stream (lhotse)


def _first(key, values):
    values = [v for v in values if v is not None]
    return values[0] if values else None


def _text(cut) -> str:
    return " ".join((s.text or "") for s in (cut.supervisions or [])).strip()


def _unlabelled(cut):
    """A background-speech track: supervisions dropped, so its text never reaches the target."""
    from lhotse.utils import fastcopy
    return cut.drop_supervisions() if hasattr(cut, "drop_supervisions") else fastcopy(cut, supervisions=[])


class PairingStream:
    """Lazy, re-iterable wrapper of a lhotse cut stream.

    Pairs: each cut starts a pair with probability q = p / (2 - p) (~p of output clips are joined) if the next cut
    fits within max_s; the pair is appended with a gap_s pause, labels joined with a space (so a Tamil + English pair
    switches script where the audio switches language). With p_overlap the second clip instead starts overlap_s
    before the first ends, mixed at overlap_snr dB below the first (negative = louder); the label is both texts in
    order of start time (a transducer emits monotonically, so only boundary overlap is simulated).
    Babble: a single clip gets, with p_babble, 1..babble_n different recent clips (labels dropped) mixed under it, each
    at snr + 10 log10(n) dB below it so the babble as a whole sits at ~snr ~ U(babble_snr). Empty-label clips (non-speech
    negatives) get neither overlap nor babble. Source cuts longer than src_max_s are dropped."""

    def __init__(self, src, p: float, max_s: float, gap_s: float, seed: int = 1234, p_overlap: float = 0.0,
                 overlap_s=(0.2, 1.5), overlap_snr=(-3.0, 6.0), p_babble: float = 0.0, babble_snr=(10.0, 20.0),
                 babble_n=(1, 1), src_max_s: float = 30.0):
        self.src, self.max_s, self.gap_s, self.seed = src, max_s, gap_s, seed
        self.p_overlap, self.overlap_s, self.overlap_snr = p_overlap, tuple(overlap_s), tuple(overlap_snr)
        self.p_babble, self.babble_snr, self.babble_n = p_babble, tuple(babble_snr), tuple(int(x) for x in babble_n)
        self.q = p / max(2 - p, 1e-6)
        self.src_max_s = src_max_s
        self.epoch = 0

    def _pair(self, a, b, rng):
        if self.p_overlap and rng.random() < self.p_overlap and _text(a) and _text(b):
            ov = min(rng.uniform(*self.overlap_s), 0.5 * a.duration, 0.5 * b.duration)
            if ov > 0.05:
                j = a.mix(b, offset_other_by=a.duration - ov, allow_padding=True, snr=rng.uniform(*self.overlap_snr))
                return j.merge_supervisions(custom_merge_fn=_first)
        return a.pad(duration=a.duration + self.gap_s).append(b).merge_supervisions(custom_merge_fn=_first)

    def _single(self, c, rng, recent):
        if self.p_babble and recent and _text(c) and rng.random() < self.p_babble:
            n, out = rng.randint(*self.babble_n), c
            for _ in range(n):
                d = recent[rng.randrange(len(recent))]
                if d.id == c.id or d.duration <= 1.0:
                    continue
                off = rng.uniform(0.0, max(0.0, c.duration - 1.0))
                d = _unlabelled(d.truncate(duration=min(d.duration, c.duration - off)))
                snr = rng.uniform(*self.babble_snr) + 10 * math.log10(n)
                out = out.mix(d, offset_other_by=off, snr=snr, preserve_id="left")
            return out
        return c

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        recent = deque(maxlen=256)  # babble donors: recent speech clips (any language)
        held = None
        for c in self.src:
            if c.duration > self.src_max_s:
                continue
            if self.p_babble and _text(c):
                recent.append(c)
            if held is not None:
                if held.duration + self.gap_s + c.duration <= self.max_s:
                    yield self._pair(held, c, rng)
                    held = None
                    continue
                yield self._single(held, rng, recent)
                held = None
            if rng.random() < self.q and c.duration + self.gap_s + 0.5 <= self.max_s:
                held = c
            else:
                yield self._single(c, rng, recent)
        if held is not None:
            yield held


def patch_nemo(p: float, max_s: float, gap_s: float, seed: int = 1234, **kw) -> None:
    """Wrap the cut stream NeMo's lhotse dataloader reads for shuffled (training) loaders; validation is untouched.
    kw: PairingStream options (p_overlap, overlap_s, overlap_snr, p_babble, babble_snr, babble_n, src_max_s)."""
    import nemo.collections.common.data.lhotse.dataloader as dl
    from lhotse import CutSet

    orig = dl.read_cutset_from_config

    def read_cutset_from_config(config, *a, **k):
        cuts, iterable = orig(config, *a, **k)
        shuffle = config.get("shuffle", False) if hasattr(config, "get") else getattr(config, "shuffle", False)
        if not shuffle:
            return cuts, iterable
        return CutSet(PairingStream(cuts, p, max_s, gap_s, seed, **kw)), iterable

    dl.read_cutset_from_config = read_cutset_from_config
