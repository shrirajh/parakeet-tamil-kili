"""Transcribe audio files with the NeMo model (GPU or CPU, PyTorch).

  python -m kili.infer.transcribe --model shrirajh/parakeet-tamil-kili a.wav b.wav [--timestamps] [--lm lm.arpa]

`--model` takes a local .nemo file or a Hugging Face id. Decoding is NeMo's greedy TDT (greedy_batch); `--lm` uses
the gated beam-4 decode with a KN 5-gram LM (kili.lm.decode, alpha 0.2). Output: one line per file, `path<TAB>text`.

Attention: the model was trained with rel_pos_local_attn and a [256, 256] frame window (+-20.48 s) and is decoded
with the same window; v3's long-audio default of [128, 128] costs ~3.6 WER on long-form audio. With local attention
memory grows linearly with length, so long recordings (minutes to an hour) are decoded whole, one file per batch.

Audio: any soundfile-readable file, averaged to mono and resampled to 16 kHz (kili.infer.onnx_run.load_audio).
Precision: bf16 on CUDA GPUs that support it (as in every reported eval), fp32 otherwise.
"""
from __future__ import annotations

import argparse

SR = 16000
ATTENTION = ("rel_pos_local_attn", [256, 256])
LONG_S = 30.0  # clips longer than this are decoded one per batch


def set_attention(model, kind: str, ctx) -> None:
    """model.change_attention_model() without its bias problem: NeMo rebuilds every attention module with biases
    (use_bias defaults to True) and loads the old weights with strict=False, so on parakeet-tdt-0.6b-v3
    (use_bias: false) every q/k/v/out projection would get a randomly initialised bias. Remove those biases after
    switching, leaving exactly the original weights. No-op when the model is already configured this way."""
    enc = model.cfg.encoder
    if enc.get("self_attention_model") == kind and list(enc.get("att_context_size", [-1, -1])) == list(ctx):
        return
    ref = next(model.encoder.parameters())
    model.change_attention_model(kind, list(ctx))
    for layer in model.encoder.layers:  # rebuilt modules come out fp32 on the default device
        layer.self_attn.to(device=ref.device, dtype=ref.dtype)
    if not enc.get("use_bias", True):
        for layer in model.encoder.layers:
            for name in ("linear_q", "linear_k", "linear_v", "linear_out"):
                lin = getattr(layer.self_attn, name, None)
                if lin is not None and lin.bias is not None:
                    lin.register_parameter("bias", None)


def load_model(path_or_hf_id: str, device: str | None = None, bf16: bool = True):
    """Load a .nemo file or a Hugging Face id, set local attention [256, 256], eval mode.
    device: None = cuda if available, else cpu. bf16 applies on CUDA only."""
    import torch
    from nemo.collections.asr.models import ASRModel

    if path_or_hf_id.endswith(".nemo"):
        m = ASRModel.restore_from(path_or_hf_id, map_location="cpu")
    else:
        m = ASRModel.from_pretrained(path_or_hf_id, map_location="cpu")
    set_attention(m, *ATTENTION)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = bf16 and device.startswith("cuda") and torch.cuda.is_bf16_supported()
    return m.to(device=device, dtype=torch.bfloat16 if use_bf16 else torch.float32).eval()


def _texts(hyps) -> list:
    if isinstance(hyps, tuple):
        hyps = hyps[0]
    return [h[0] if isinstance(h, list) and h else h for h in hyps]


def transcribe_arrays(model, audios: list, batch_size: int = 16, timestamps: bool = False) -> list:
    """float32 mono 16 kHz arrays -> NeMo hypotheses (with .text, and .timestamp if timestamps) or strings.
    Clips up to 30 s are batched; longer ones are decoded whole, one per batch."""
    import torch

    out = [None] * len(audios)
    short = [i for i, x in enumerate(audios) if len(x) <= LONG_S * SR]
    long = [i for i, x in enumerate(audios) if len(x) > LONG_S * SR]
    kw = {"timestamps": True, "return_hypotheses": True} if timestamps else {}
    for idx, bs in ((short, batch_size), (long, 1)):
        if not idx:
            continue
        with torch.inference_mode():
            hyps = _texts(model.transcribe([audios[i] for i in idx], batch_size=bs, verbose=False, **kw))
        for i, h in zip(idx, hyps):
            out[i] = h
    return out


def text_of(h) -> str:
    return h.text if hasattr(h, "text") else str(h)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, help=".nemo file or Hugging Face id (shrirajh/parakeet-tamil-kili)")
    ap.add_argument("audio", nargs="+")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--fp32", action="store_true", help="fp32 on GPU too (default bf16 where supported)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--timestamps", action="store_true", help="also print word start / end times (s)")
    ap.add_argument("--lm", default=None, help="ARPA LM: gated beam decode (kili.lm.decode)")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--no-gate", action="store_true", help="--lm: apply the LM to every utterance")
    a = ap.parse_args()
    if a.lm and a.timestamps:
        ap.error("--timestamps is not available with --lm")
    model = load_model(a.model, a.device, bf16=not a.fp32)
    if a.lm:
        from kili.lm.decode import transcribe_lm
        texts = transcribe_lm(model, a.audio, a.lm, alpha=a.alpha, beam=a.beam, gate=not a.no_gate,
                              batch_size=a.batch_size)
        for p, t in zip(a.audio, texts):
            print(f"{p}\t{t}", flush=True)
        return
    from kili.infer.onnx_run import load_audio
    hyps = transcribe_arrays(model, [load_audio(p) for p in a.audio], a.batch_size, a.timestamps)
    for p, h in zip(a.audio, hyps):
        print(f"{p}\t{text_of(h)}", flush=True)
        if a.timestamps:
            ts = getattr(h, "timestamp", None) or {}
            for w in ts.get("word", []):
                print(f"\t{w.get('start', 0):.2f}\t{w.get('end', 0):.2f}\t{w.get('word', '')}", flush=True)


if __name__ == "__main__":
    main()
