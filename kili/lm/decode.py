"""Gated beam-4 decoding with the KN 5-gram LM (NeMo GPU n-gram fusion, NGPU-LM).

Released setting: greedy decode of every clip; clips whose greedy output is not mostly Latin script (kili.lm.gate)
are decoded again with NeMo's batched TDT beam search (malsd_batch, beam 4) fused with the n-gram LM at alpha 0.2.
On run 10 (5 epochs) this takes Kathbath 29.15 -> 25.39 fold WER, MILE 27.89 -> 26.31, FLEURS ta 30.46 -> 27.44,
CV ta 35.22 -> 32.62, OpenSLR-65 24.84 -> 20.73, IndicVoices 34.05 -> 32.75 (it closes 30-45% of the read-speech
gap to indic-transcribe-core); English outputs unchanged. Cost: beam 4 + LM decodes ~40% slower than greedy, and the
gate decodes the LM share twice (greedy, then beam + LM); ~16 GB GPU memory with the 2.27 GB Sangraha LM.

  python -m kili.lm.decode --model kili.nemo --lm own_sg_o5.nemo audio1.wav audio2.wav > hyps.jsonl

The LM must be built over the same tokenizer as the model (kili.lm.kn_arpa --model). Both .arpa and NGPU-LM .nemo
work; the .nemo loads much faster.
"""
from __future__ import annotations

import argparse
import copy
import json

from kili.lm.gate import use_lm


def lm_decoding_override(lm_path: str, alpha: float = 0.2, beam: int = 4) -> dict:
    """The decoding-config override used for every reported LM number."""
    return {"strategy": "malsd_batch",
            "beam": {"beam_size": beam, "ngram_lm_model": lm_path, "ngram_lm_alpha": alpha,
                     "pruning_mode": "late", "blank_lm_score_mode": "lm_weighted_full"}}


def _set_decoding(model, cfg) -> None:
    model.change_decoding_strategy(cfg, verbose=False)


def _transcribe(model, paths: list[str], batch_size: int) -> list[str]:
    import torch
    if not paths:
        return []
    with torch.inference_mode():
        hyps = model.transcribe(paths, batch_size=batch_size, verbose=False)
    if isinstance(hyps, tuple):
        hyps = hyps[0]
    hyps = [h[0] if isinstance(h, list) and h else h for h in hyps]  # beam strategies return an n-best list per clip
    return [h.text if hasattr(h, "text") else str(h) for h in hyps]


def transcribe_lm(model, paths: list[str], lm_path: str, alpha: float = 0.2, beam: int = 4, gate: bool = True,
                  batch_size: int = 16) -> list[str]:
    """Transcribe audio files with beam search + n-gram LM fusion; with gate=True only clips whose greedy output is
    not mostly Latin script use the LM (the rest keep the greedy output). The model's decoding config is restored
    before returning."""
    from omegaconf import OmegaConf, open_dict
    orig = copy.deepcopy(model.cfg.decoding)
    try:
        if gate:
            greedy_cfg = copy.deepcopy(orig)
            with open_dict(greedy_cfg):
                greedy_cfg.strategy = "greedy_batch"
            _set_decoding(model, greedy_cfg)
            out = _transcribe(model, paths, batch_size)
            idx = [i for i, t in enumerate(out) if use_lm(t)]
        else:
            out, idx = [""] * len(paths), list(range(len(paths)))
        if idx:
            cfg = copy.deepcopy(orig)
            with open_dict(cfg):
                cfg = OmegaConf.merge(cfg, OmegaConf.create(lm_decoding_override(lm_path, alpha, beam)))
            _set_decoding(model, cfg)
            for i, h in zip(idx, _transcribe(model, [paths[i] for i in idx], batch_size)):
                out[i] = h
        return out
    finally:
        _set_decoding(model, orig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help=".nemo path or Hugging Face id")
    ap.add_argument("--lm", required=True, help="n-gram LM (.arpa or NGPU-LM .nemo) over the model's tokenizer")
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--beam", type=int, default=4)
    ap.add_argument("--no-gate", action="store_true", help="LM on every clip (hurts English)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("audio", nargs="+")
    a = ap.parse_args()
    from kili.infer.transcribe import load_model
    model = load_model(a.model)
    hyps = transcribe_lm(model, a.audio, a.lm, a.alpha, a.beam, not a.no_gate, a.batch_size)
    for p, h in zip(a.audio, hyps):
        print(json.dumps({"audio_filepath": p, "hyp": h}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
