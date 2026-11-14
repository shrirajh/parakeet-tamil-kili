"""Training kernels for NeMo FastConformer models (Triton), as used to train parakeet-tamil-kili run 10.

On one RTX PRO 6000 Blackwell these kernels, together with the batch settings they made possible, took run 10's
training of Parakeet-TDT 0.6B from 358 to 911 audio-hours per GPU-hour (2.54x). Every kernel was checked against
NeMo's own modules (float64 references, bf16 error ratios, same-batch training replays). They change speed and
memory only, not the model or the recipe; checkpoints load and save as plain NeMo models.

Usage (training only; Triton + a CUDA GPU; developed on Parakeet-TDT 0.6B v3):

    from nemo.collections.asr.models import ASRModel
    import kili.kernels

    model = ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v3")
    kili.kernels.apply(model)   # local attention [256, 256] + all four kernels, as trained
    # ... train with bf16-mixed precision as usual (Lightning trainer.fit) ...
    kili.kernels.remove(model)  # optional: back to NeMo's forwards (e.g. before export); weights are unchanged
    model.save_to("model.nemo")

Runs on any CUDA GPU Triton supports with bf16 (sm_80+); launch configs were tuned on the Blackwell card.
Or patch individually: attention.install, conv.install, layer.install, tdt_loss.install (see each module).
apply() switches the encoder to NeMo's local attention (rel_pos_local_attn, [256, 256] frames: what run 10 trained
and decodes with) unless att_context=None; the attention and fused-layer kernels only act on that attention model.
The encoder kernels work for any FastConformer (TDT, RNN-T, CTC); the loss kernel is for TDT models.

| kernel                        | what it replaces                                                   | module      |
|-------------------------------|--------------------------------------------------------------------|-------------|
| banded rel-pos attention      | NeMo's Longformer local attention core (dropout in-kernel)         | attention   |
| conv module + depthwise conv  | ConformerConvolution's transposes / pad copies, cudnn depthwise    | conv        |
| residual + dropout + LayerNorm| the 4 add / dropout / LayerNorm steps of every ConformerLayer      | layer       |
| gathered TDT loss             | the full B x T x U x V joint, NeMo's numba TDT loss                | tdt_loss    |

Throughput, run 10's recipe on one RTX PRO 6000 Blackwell (audio-hours per GPU-hour, bf16; cumulative):
    358  baseline: NeMo local attention, 500 s micro-batches, layer checkpointing, gathered TDT loss
    592  + banded attention (dense exact version) and 1000 s micro-batches (memory freed by the attention)
    733  + no layer checkpointing, joint logits kept for backward up to 3M cells
    762  + 40 duration buckets
    810  + attention core in bf16 (as NeMo's own core under autocast)
    838  + fused banded attention in Triton with in-kernel dropout             (attention)
    850  + 60 duration buckets
    866  + conv module without layout copies                                  (conv, triton_dw=False)
    874  + fused residual + dropout + LayerNorm                               (layer)
    893  + joint output bias inside the TDT row kernels                       (tdt_loss, bias_in_rows)
    911  + Triton depthwise conv                                              (conv) -> 2.54x
Steps differing by a few percent are within run-to-run noise (each run draws different batches).

Exactness (bf16 ratio = the kernel's error vs a float64 reference divided by NeMo's own bf16 error):
- attention: vs float64, output and gradient errors equal to or below PyTorch's dense bf16 SDPA (16-120 s inputs); with
  dropout 0.1 the same error level against the kernel's own mask (kept fraction 0.9000, masks independent per call);
  NeMo module level bf16 ratio median 0.91-1.00, worst 1.24; 24-layer encoder on real audio, mean error 1.8e-4 vs
  NeMo's own bf16 2.1e-4.
- conv module: fp32 4e-6 relative vs NeMo (output, all gradients, BatchNorm running stats); bf16 ratio median 1.00,
  worst 1.17. Depthwise kernel vs F.conv1d fp32: output and input gradient identical, weight gradient 2e-7; bf16
  ratio 1.00; 3.8-5x faster than cudnn forward + backward (RTX 3090).
- residual + dropout + LayerNorm: fp32 1e-7 vs torch; bf16 ratio 1.00; kept fraction 0.9002, mask regenerated in
  backward. Whole conformer layer vs NeMo's: fp32 2e-6 to 3e-5, bf16 ratio median 1.00, worst 1.39.
- TDT loss: loss and every gradient vs NeMo's per-utterance reference within fp32 round-off (tests: loss rel < 1e-5,
  grad rel < 1e-3); Triton recursion vs NeMo's numba kernel likewise, for TDT and NeMo's RNN-T (omega) branch.
  Bias in the row kernels: fp32 outputs identical to addmm, bias gradient 1.9e-7; bf16 ratio vs the addmm path
  0.81-1.12 (bias gradient 0.47: more accurate).
- Training replay (same batches, dropout 0, 60 steps), everything on vs the fp32 reference path: loss mean |diff|
  0.0009, max 0.0021; for scale, NeMo's own bf16 attention vs that path: 0.0011 / 0.0039.

Tests: tests/kernels (CUDA + Triton; NeMo for the module tests). Requires torch, triton (bundled with Linux torch
wheels), nemo_toolkit[asr].
"""
from __future__ import annotations


def apply(model, *, att_context=(256, 256), attention: bool = True, conv: bool = True, layer: bool = True,
          tdt_loss: bool = True, triton_dw: bool = True, max_T: int = 2048, chunk_elems: int = 128 * 2**20,
          recompute_cells: float = 3_000_000, bias_in_rows: bool = True) -> dict:
    """Install the kernels on a NeMo ASR model, in place; returns {kernel: modules patched}. Defaults = run 10.

    att_context: switch the encoder to local attention with this [left, right] window first (None: leave the
      attention model as it is; the attention / layer kernels then only act if it is already rel_pos_local_attn)
    max_T: inputs longer than this many encoder frames use NeMo's attention (2048 frames = 164 s)
    chunk_elems, recompute_cells, bias_in_rows: TDT loss options (kili.kernels.tdt_loss.install)"""
    from kili.kernels import attention as _attention
    out = {}
    if att_context is not None:
        _attention.set_local_attention(model, tuple(att_context))
    if layer:  # first: it must be the innermost wrapper of layer.forward
        from kili.kernels import layer as _layer
        out["layer"] = _layer.install(model)
    if attention:
        out["attention"] = _attention.install(model, max_T=max_T)
    if conv:
        from kili.kernels import conv as _conv
        out["conv"] = _conv.install(model, triton_dw=triton_dw)
    if tdt_loss:
        from kili.kernels import tdt_loss as _tdt
        _tdt.install(model, chunk_elems=chunk_elems, recompute_cells=recompute_cells, bias_in_rows=bias_in_rows)
        out["tdt_loss"] = 1
    return out


def remove(model) -> None:
    """Undo apply(): NeMo's own forwards and loss again (the attention model stays as set). Drops any instance-level
    forward on the encoder layers and their attention / conv modules, including wrappers installed after apply()."""
    for lyr in model.encoder.layers:
        for mod in (lyr, getattr(lyr, "self_attn", None), getattr(lyr, "conv", None)):
            if mod is not None:
                mod.__dict__.pop("forward", None)
    if hasattr(model, "_kili_tdt_orig"):
        from kili.kernels import tdt_loss as _tdt
        _tdt.uninstall(model)
