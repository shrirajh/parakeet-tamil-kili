"""Exact TDT loss without materialising the B x T x (U+1) x (V+1+D) joint logits (NeMo TDT models).

NeMo's TDT path computes the full joint (V + 1 + D logits per lattice cell, upcast to fp32) plus an equally large
gradient: for a 30 s Tamil clip with ~230 BPE tokens and Parakeet's 8,198 outputs that is gigabytes per clip.

Exact reduction: the TDT forward-backward only reads, per lattice cell (b, t, u), log p(blank), log p(y_{u+1}) (the
next target label) and the D duration log-probs. So per cell we compute
    r = [log p(label), log p(other), log p(blank), duration logits]      other = every token except label and blank
with log p(other) = logsumexp(masked logits) - logsumexp(all logits) (stable, no log1p(-x)). The joint runs only on
each utterance's valid cells (t < T_b, u <= U_b; "ragged", no padding FLOPs), its full-vocabulary row work is one
Triton kernel each way, and the TDT recursion is a Triton kernel on the reduced lattice (kili.kernels.tdt_kernels).
Gradients flow through r back to the joint weights, encoder and prediction network by autograd.

Memory: the bf16 logits of the valid cells are kept for backward (~16 kB per cell) unless a batch has more than
`recompute_cells` cells; then each chunk of cells is recomputed in backward (activation checkpointing, RNG state
preserved so joint dropout masks match).

bias_in_rows: the joint's output projection runs as a plain GEMM z = h W^T and the row kernels add the bias on load;
the bias gradient comes from per-row-group column sums in the backward kernel (no addmm, no full reduction over dz).
The output width is padded to a multiple of 64 with zero rows of W (aligned GEMMs); the kernels ignore the padding.

Supported: TDT losses with NeMo's durations / sigma / omega and reductions mean_volume, mean, mean_batch, sum, None;
not fastemit or clamp (Parakeet uses neither). The model's joint must have fuse_loss_wer off (stock v3 ships with it on:
install() switches it off).

  from kili.kernels.tdt_loss import install; install(model)       # model.loss / model.joint patched in place
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from kili.kernels.tdt_kernels import gathered_rows, tdt_loss_from_reduced

ALIGN = 64  # pad the joint output width (V + 1 + D) to a multiple of this: unaligned widths force slow GEMM kernels


class LazyJoint:
    """Stand-in for the joint logits: carries the projected encoder / prediction-net outputs until the loss needs
    them."""

    def __init__(self, f: torch.Tensor, g: torch.Tensor):
        self.f, self.g = f, g  # f: [B, T, H], g: [B, U+1, H] (already projected)


def joint_padded(lin: torch.nn.Linear, h: torch.Tensor, split_bias: bool):
    """lin(h) with the output padded to a multiple of ALIGN columns (padded rows of W are zero).
    Returns (z [..., Wp], real width, bias or None); with split_bias, z excludes the bias (the row kernels add it)."""
    w = lin.out_features
    pad = (-w) % ALIGN
    if split_bias and lin.bias is not None:
        wp = F.pad(lin.weight, (0, 0, 0, pad)) if pad else lin.weight
        return F.linear(h, wp, None), w, lin.bias
    if not pad:
        return F.linear(h, lin.weight, lin.bias), w, None
    wp = F.pad(lin.weight, (0, 0, 0, pad))
    bp = F.pad(lin.bias, (0, pad)) if lin.bias is not None else None
    return F.linear(h, wp, bp), w, None


def _cells_logprobs(joint_net, f2, g2, fi, gi, y_cells, blank: int, n_dur: int, split_bias: bool):
    """Joint on the listed cells (flat encoder row fi, prediction row gi). Returns [n, 3 + n_dur] fp32."""
    drop = joint_net[1] if len(joint_net) > 2 and isinstance(joint_net[1], torch.nn.Dropout) else None
    p, training = (drop.p, drop.training) if drop is not None else (0.0, False)
    # index_select: its backward is an index_add_ (advanced indexing backprops through a slower sort-based kernel)
    h = F.dropout(torch.relu(f2.index_select(0, fi) + g2.index_select(0, gi)), p, training)
    z, w, bias = joint_padded(joint_net[-1], h, split_bias)  # [n, Wp >= V+1+D]
    lse_o, zl, zb, zd = gathered_rows(z, y_cells, blank, n_dur, width=w, bias=bias)
    lse = torch.logaddexp(lse_o, torch.logaddexp(zl, zb))  # == logsumexp over all tokens, exactly
    return torch.cat([torch.stack([zl - lse, lse_o - lse, zb - lse], dim=-1), zd], dim=-1)


def ragged_logprobs(joint_net, f, g, targets, enc_len, tgt_len, blank: int, n_dur: int, chunk_elems: int,
                    recompute: bool, split_bias: bool):
    """Reduced lattice [B, T, U+1, 3 + n_dur] (fp32) computed on valid cells only; invalid cells are 0 (never read)."""
    assert isinstance(joint_net[0], torch.nn.ReLU) and isinstance(joint_net[-1], torch.nn.Linear), joint_net
    B, T, H = f.shape
    U1 = g.shape[1]
    y_ext = torch.zeros(B, U1, dtype=torch.long, device=f.device)
    n = min(targets.shape[1], U1 - 1)
    y_ext[:, :n] = targets[:, :n].long()
    tt = torch.arange(T, device=f.device)
    uu = torch.arange(U1, device=f.device)
    valid = (tt[None, :, None] < enc_len[:, None, None]) & (uu[None, None, :] <= tgt_len[:, None, None])
    bi, ti, ui = valid.nonzero(as_tuple=True)  # one host sync per step
    y_cells = y_ext[bi, ui].to(torch.int32)
    fi, gi = bi * T + ti, bi * U1 + ui
    f2, g2 = f.reshape(-1, H), g.reshape(-1, H)
    step = max(1, chunk_elems // joint_net[-1].out_features)
    vals = []
    for s0 in range(0, bi.numel(), step):
        sl = slice(s0, s0 + step)
        args = (joint_net, f2, g2, fi[sl], gi[sl], y_cells[sl], blank, n_dur, split_bias)
        if torch.is_grad_enabled() and recompute:
            vals.append(checkpoint(_cells_logprobs, *args, use_reentrant=False, preserve_rng_state=True))
        else:
            vals.append(_cells_logprobs(*args))
    vals = torch.cat(vals, 0)
    flat = (bi * T + ti) * U1 + ui
    out = torch.zeros(B * T * U1, vals.shape[-1], device=f.device, dtype=vals.dtype)
    return out.index_copy(0, flat, vals).view(B, T, U1, -1)


class GatheredTDTLoss(torch.nn.Module):
    """Drop-in for model.loss when model.joint returns a LazyJoint. Reuses the original loss's durations, sigma, omega
    and reduction."""

    def __init__(self, orig_loss, joint, chunk_elems: int, recompute_cells: float, bias_in_rows: bool):
        super().__init__()
        inner = orig_loss._loss
        assert type(inner).__name__ == "TDTLossNumba", f"expected NeMo's TDT loss, got {type(inner)}"
        if inner.fastemit_lambda or inner.clamp > 0:
            raise ValueError(f"fastemit_lambda={inner.fastemit_lambda} / clamp={inner.clamp} are not supported")
        self._orig = [orig_loss]  # lists: keep these out of the module tree (no duplicate registration)
        self.joint_ref = [joint]
        self.blank = inner.blank
        self.durations = list(inner.durations)
        self.n_dur = len(self.durations)
        self.sigma, self.omega = float(inner.sigma), float(inner.omega)
        self.reduction = orig_loss.reduction
        self.chunk_elems = chunk_elems
        self.recompute_cells = recompute_cells
        self.bias_in_rows = bias_in_rows

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        if not isinstance(log_probs, LazyJoint):  # real logits (e.g. someone called the original joint): fall back
            return self._orig[0](log_probs=log_probs, targets=targets, input_lengths=input_lengths,
                                 target_lengths=target_lengths)
        target_lengths = target_lengths.long()
        loss = self.per_utterance(log_probs, targets, input_lengths, target_lengths)
        # NeMo's mean_volume divides by target_lengths.sum(): a micro-batch of only empty-label clips (non-speech
        # negatives) would give inf. The denominator is clamped on the device (no sync).
        if self.reduction == "mean_volume":
            loss = loss.sum() / target_lengths.sum().clamp(min=1)
        elif self.reduction == "mean":
            loss = (loss / target_lengths.clamp(min=1)).mean()
        elif self.reduction == "mean_batch":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction is not None:
            raise ValueError(f"unsupported TDT reduction {self.reduction!r}")
        return loss

    def per_utterance(self, log_probs: LazyJoint, targets, input_lengths, target_lengths):
        """Unreduced TDT costs [B] (-log P(y|x) with the loss's sigma / omega); differentiable."""
        input_lengths, target_lengths = input_lengths.long(), target_lengths.long()
        # one sync for the max lengths and the valid (t, u) cell count
        T, U, cells = torch.stack([input_lengths.max(), target_lengths.max(),
                                   (input_lengths * (target_lengths + 1)).sum()]).tolist()
        recompute = cells > self.recompute_cells
        f = log_probs.f[:, :T]
        g = log_probs.g[:, : U + 1]
        targets = targets[:, :U].long().contiguous()
        red = ragged_logprobs(self.joint_ref[0].joint_net, f, g, targets, input_lengths, target_lengths, self.blank,
                              self.n_dur, self.chunk_elems, recompute, self.bias_in_rows)
        return tdt_loss_from_reduced(red, input_lengths, target_lengths, self.durations, self.sigma, self.omega)


def install(model, chunk_elems: int = 128 * 2**20, recompute_cells: float = 3_000_000,
            bias_in_rows: bool = True) -> None:
    """Patch a NeMo TDT model in place (runtime only; nothing changes in a saved .nemo).
    chunk_elems: joint logits per chunk (cells per chunk = chunk_elems / (V + 1 + D)).
    recompute_cells: batches with more valid (t, u) cells recompute the joint in backward instead of keeping logits."""
    joint = model.joint
    joint.set_fuse_loss_wer(False)  # NeMo's fused sub-batch joint/loss is replaced, not combined
    try:
        from omegaconf import open_dict
        with open_dict(model.cfg):
            model.cfg.joint.fuse_loss_wer = False
    except Exception:  # config without a joint section: the runtime flag above is what matters
        pass

    def forward(encoder_outputs, decoder_outputs=None, **kw):
        f = joint.project_encoder(encoder_outputs.transpose(1, 2))
        g = joint.project_prednet(decoder_outputs.transpose(1, 2))
        if joint.training and getattr(joint, "masking_prob", 0) > 0:
            B, U, _ = g.shape
            g = g * torch.gt(torch.rand([B, U, 1], device=g.device), joint.masking_prob)
        return LazyJoint(f, g)

    model._kili_tdt_orig = (joint.__dict__.get("forward"), model.loss)
    joint.forward = forward
    model.loss = GatheredTDTLoss(model.loss, joint, chunk_elems, float(recompute_cells), bias_in_rows)


def uninstall(model) -> None:
    """Restore the original joint forward and loss."""
    fwd, loss = model._kili_tdt_orig
    if fwd is None:
        del model.joint.forward  # back to the class method
    else:
        model.joint.forward = fwd
    model.loss = loss
    del model._kili_tdt_orig
