"""Fused banded relative-position attention (Triton) for NeMo's local attention, with in-kernel dropout.

Replaces the core of NeMo's RelPositionMultiHeadAttentionLongformer (self_attention_model "rel_pos_local_attn",
att_context_size [w, w]). For query i and key j with |j - i| <= w and j inside the utterance:
    s[i, j] = (q_u[i] . k[j] + bd[i, j - i + w]) / sqrt(d_k)       q_u = q + pos_bias_u,  bd = (q + pos_bias_v) p^T
    P = softmax_j(s);  P = dropout(P);  out[i] = sum_j P[i, j] v[j]
which is exactly NeMo's Longformer maths (its chunked einsums compute the same band; its -10000 fills underflow to
exact zeros). NeMo spends most of this module in pads, as_strided views, slice assignments and their backward copies.

Kernel: flash-attention style forward and two backward kernels (dK/dV/d(bd) and dQ) that visit only key blocks inside
the band; the rel-pos term is read from bd [B, H, T, 2w+1] inside the kernel (no dense [B, H, T, T] bias) and d(bd)
is written directly (each (i, j) belongs to one key block: no atomics). bf16 operands, fp32 accumulation and softmax
statistics, as NeMo's own core under bf16 autocast (its einsums run in bf16 there). Dropout: Philox (tl.rand) keyed by a
device seed and the element index ((b*H + h)*T + i)*T + j, regenerated in backward. The seed is drawn from torch's
CUDA generator, so it is reproducible under torch.manual_seed, identical when activation checkpointing recomputes a
forward, and CUDA-graph safe.

Projections (q/k/v/pos/out) run exactly as in NeMo. Anything the kernel does not cover (caches / streaming, global
tokens, an asymmetric window, head dim not in {16, 32, 64, 128}, T > max_T) falls back to NeMo's own forward.

  from kili.kernels.attention import set_local_attention, install
  set_local_attention(model, (256, 256)); install(model)
"""
from __future__ import annotations

import types

import torch
import triton
import triton.language as tl


@triton.jit
def _fwd(Q, K, V, BD, LENS, O, LSE, SEED,
         sq_b, sq_h, sq_t, sk_b, sk_h, sk_t, sv_b, sv_h, sv_t, sbd_b, sbd_h, sbd_t, so_b, so_h, so_t,
         H, T, W, scale, p_drop,
         BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr, DROP: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    L = tl.load(LENS + b)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    mrow = offs_m < T
    q = tl.load(Q + b * sq_b + h * sq_h + offs_m[:, None] * sq_t + offs_d[None, :], mask=mrow[:, None], other=0.0)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    lo = tl.maximum(pid_m * BM - W, 0)
    lo = (lo // BN) * BN
    hi = tl.minimum(tl.minimum(T, L), pid_m * BM + BM + W)
    if DROP:
        seed = tl.load(SEED)
    for start_n in range(lo, hi, BN):
        offs_n = start_n + tl.arange(0, BN)
        k = tl.load(K + b * sk_b + h * sk_h + offs_n[:, None] * sk_t + offs_d[None, :], mask=(offs_n < T)[:, None],
                    other=0.0)
        s = tl.dot(q, tl.trans(k))
        rel = offs_n[None, :] - offs_m[:, None]
        valid = (rel <= W) & (rel >= -W) & (offs_n[None, :] < L) & mrow[:, None]
        bdv = tl.load(BD + b * sbd_b + h * sbd_h + offs_m[:, None] * sbd_t + (rel + W), mask=valid, other=0.0)
        s = (s + bdv.to(tl.float32)) * scale
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(s - m_safe[:, None])
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, 1)
        if DROP:
            offs = ((bh * T + offs_m[:, None]) * T + offs_n[None, :]).to(tl.uint32)
            keep = tl.rand(seed, offs) >= p_drop
            p = tl.where(keep, p / (1.0 - p_drop), 0.0)
        v = tl.load(V + b * sv_b + h * sv_h + offs_n[:, None] * sv_t + offs_d[None, :], mask=(offs_n < T)[:, None],
                    other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    l_safe = tl.where(l_i == 0.0, 1.0, l_i)
    m_fin = tl.where(m_i == float("-inf"), 0.0, m_i)
    out = acc / l_safe[:, None]
    tl.store(O + b * so_b + h * so_h + offs_m[:, None] * so_t + offs_d[None, :], out.to(O.dtype.element_ty),
             mask=mrow[:, None])
    # rows without a valid key: +inf so that exp(s - lse) = 0 in backward
    lse = tl.where(l_i == 0.0, float("inf"), m_fin + tl.log(l_safe))
    tl.store(LSE + bh * T + offs_m, lse, mask=mrow)


@triton.jit
def _bwd_dkdv(Q, K, V, BD, LENS, DO, LSE, DELTA, DK, DV, DBD, SEED,
              sq_b, sq_h, sq_t, sk_b, sk_h, sk_t, sv_b, sv_h, sv_t, sbd_b, sbd_h, sbd_t, so_b, so_h, so_t,
              H, T, W, scale, p_drop,
              BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr, DROP: tl.constexpr):
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    L = tl.load(LENS + b)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    ncol = offs_n < T
    k = tl.load(K + b * sk_b + h * sk_h + offs_n[:, None] * sk_t + offs_d[None, :], mask=ncol[:, None], other=0.0)
    v = tl.load(V + b * sv_b + h * sv_h + offs_n[:, None] * sv_t + offs_d[None, :], mask=ncol[:, None], other=0.0)
    dk = tl.zeros([BN, D], tl.float32)
    dv = tl.zeros([BN, D], tl.float32)
    if DROP:
        seed = tl.load(SEED)
    lo = tl.maximum(pid_n * BN - W, 0)
    lo = (lo // BM) * BM
    hi = tl.minimum(T, pid_n * BN + BN + W)
    if pid_n * BN < L:  # key blocks entirely past this batch row's length get no gradient (zeros)
        for start_m in range(lo, hi, BM):
            offs_m = start_m + tl.arange(0, BM)
            mrow = offs_m < T
            q = tl.load(Q + b * sq_b + h * sq_h + offs_m[:, None] * sq_t + offs_d[None, :], mask=mrow[:, None],
                        other=0.0)
            do = tl.load(DO + b * so_b + h * so_h + offs_m[:, None] * so_t + offs_d[None, :], mask=mrow[:, None],
                         other=0.0)
            lse = tl.load(LSE + bh * T + offs_m, mask=mrow, other=float("inf"))
            delta = tl.load(DELTA + bh * T + offs_m, mask=mrow, other=0.0)
            s = tl.dot(q, tl.trans(k))
            rel = offs_n[None, :] - offs_m[:, None]
            valid = (rel <= W) & (rel >= -W) & (offs_n[None, :] < L) & mrow[:, None]
            bd_ptr = BD + b * sbd_b + h * sbd_h + offs_m[:, None] * sbd_t + (rel + W)
            bdv = tl.load(bd_ptr, mask=valid, other=0.0)
            s = (s + bdv.to(tl.float32)) * scale
            p = tl.where(valid, tl.exp(s - lse[:, None]), 0.0)
            dp = tl.dot(do, tl.trans(v))
            if DROP:
                offs = ((bh * T + offs_m[:, None]) * T + offs_n[None, :]).to(tl.uint32)
                keep = tl.rand(seed, offs) >= p_drop
                pd = tl.where(keep, p / (1.0 - p_drop), 0.0)
                dp = tl.where(keep, dp / (1.0 - p_drop), 0.0)
            else:
                pd = p
            dv += tl.dot(tl.trans(pd.to(do.dtype)), do)
            ds = p * (dp - delta[:, None])
            dk += tl.dot(tl.trans(ds.to(q.dtype)), q)
            dbd_ptr = DBD + b * sbd_b + h * sbd_h + offs_m[:, None] * sbd_t + (rel + W)
            tl.store(dbd_ptr, ds * scale, mask=valid)
    tl.store(DK + b * sk_b + h * sk_h + offs_n[:, None] * sk_t + offs_d[None, :], (dk * scale).to(DK.dtype.element_ty),
             mask=ncol[:, None])
    tl.store(DV + b * sv_b + h * sv_h + offs_n[:, None] * sv_t + offs_d[None, :], dv.to(DV.dtype.element_ty),
             mask=ncol[:, None])


@triton.jit
def _bwd_dq(Q, K, V, BD, LENS, DO, LSE, DELTA, DQ, SEED,
            sq_b, sq_h, sq_t, sk_b, sk_h, sk_t, sv_b, sv_h, sv_t, sbd_b, sbd_h, sbd_t, so_b, so_h, so_t,
            H, T, W, scale, p_drop,
            BM: tl.constexpr, BN: tl.constexpr, D: tl.constexpr, DROP: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    L = tl.load(LENS + b)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    mrow = offs_m < T
    q = tl.load(Q + b * sq_b + h * sq_h + offs_m[:, None] * sq_t + offs_d[None, :], mask=mrow[:, None], other=0.0)
    do = tl.load(DO + b * so_b + h * so_h + offs_m[:, None] * so_t + offs_d[None, :], mask=mrow[:, None], other=0.0)
    lse = tl.load(LSE + bh * T + offs_m, mask=mrow, other=float("inf"))
    delta = tl.load(DELTA + bh * T + offs_m, mask=mrow, other=0.0)
    dq = tl.zeros([BM, D], tl.float32)
    if DROP:
        seed = tl.load(SEED)
    lo = tl.maximum(pid_m * BM - W, 0)
    lo = (lo // BN) * BN
    hi = tl.minimum(tl.minimum(T, L), pid_m * BM + BM + W)
    for start_n in range(lo, hi, BN):
        offs_n = start_n + tl.arange(0, BN)
        k = tl.load(K + b * sk_b + h * sk_h + offs_n[:, None] * sk_t + offs_d[None, :], mask=(offs_n < T)[:, None],
                    other=0.0)
        v = tl.load(V + b * sv_b + h * sv_h + offs_n[:, None] * sv_t + offs_d[None, :], mask=(offs_n < T)[:, None],
                    other=0.0)
        s = tl.dot(q, tl.trans(k))
        rel = offs_n[None, :] - offs_m[:, None]
        valid = (rel <= W) & (rel >= -W) & (offs_n[None, :] < L) & mrow[:, None]
        bdv = tl.load(BD + b * sbd_b + h * sbd_h + offs_m[:, None] * sbd_t + (rel + W), mask=valid, other=0.0)
        s = (s + bdv.to(tl.float32)) * scale
        p = tl.where(valid, tl.exp(s - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        if DROP:
            offs = ((bh * T + offs_m[:, None]) * T + offs_n[None, :]).to(tl.uint32)
            keep = tl.rand(seed, offs) >= p_drop
            dp = tl.where(keep, dp / (1.0 - p_drop), 0.0)
        ds = p * (dp - delta[:, None])
        dq += tl.dot(ds.to(k.dtype), k)
    tl.store(DQ + b * sq_b + h * sq_h + offs_m[:, None] * sq_t + offs_d[None, :], (dq * scale).to(DQ.dtype.element_ty),
             mask=mrow[:, None])


@triton.jit
def _keep_mask(OUT, SEED, H, T, p_drop, BM: tl.constexpr, BN: tl.constexpr):
    """The dropout keep mask the kernels use, materialised [B*H, T, T] (tests only)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    bh = tl.program_id(2)
    seed = tl.load(SEED)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs = ((bh * T + offs_m[:, None]) * T + offs_n[None, :]).to(tl.uint32)
    keep = tl.rand(seed, offs) >= p_drop
    tl.store(OUT + bh * T * T + offs_m[:, None] * T + offs_n[None, :], keep.to(tl.int8),
             mask=(offs_m[:, None] < T) & (offs_n[None, :] < T))


# launch configs "BM, BN, warps, stages" (forward tuned on an RTX PRO 6000 Blackwell)
_FWD = (128, 64, 8, 2)
_BWD = (64, 64, 4, 2)


def next_seed(device) -> torch.Tensor:
    """Per-call dropout seed drawn on the device from torch's CUDA generator (one int32, no host sync): reproducible
    under torch.manual_seed, identical when activation checkpointing recomputes a forward (checkpoint restores the CUDA
    RNG state), CUDA-graph safe. Saved by forward for backward."""
    return torch.randint(0, 2**31 - 1, (1,), device=device, dtype=torch.int32)


def _strides(t):
    return t.stride(0), t.stride(1), t.stride(2)


class BandAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, qu, k, v, bd, lens, w: int, p_drop: float, seed: torch.Tensor | None):
        B, H, T, D = qu.shape
        assert D in (16, 32, 64, 128) and bd.shape[-1] == 2 * w + 1
        qu, k, v, bd = (t.contiguous() for t in (qu, k, v, bd))
        o = torch.empty_like(v)
        lse = torch.empty(B * H, T, device=qu.device, dtype=torch.float32)
        drop = p_drop > 0.0
        if seed is None:
            seed = next_seed(qu.device) if drop else torch.zeros(1, dtype=torch.int32, device=qu.device)
        lens = lens.to(torch.int32)
        scale = D ** -0.5
        fm, fn, fw, fs = _FWD
        grid = (triton.cdiv(T, fm), B * H)
        _fwd[grid](qu, k, v, bd, lens, o, lse, seed, *_strides(qu), *_strides(k), *_strides(v), *_strides(bd),
                   *_strides(o), H, T, w, scale, p_drop, BM=fm, BN=fn, D=D, DROP=drop, num_warps=fw, num_stages=fs)
        ctx.save_for_backward(qu, k, v, bd, lens, o, lse, seed)
        ctx.w, ctx.p_drop = w, p_drop
        return o

    @staticmethod
    def backward(ctx, do):
        qu, k, v, bd, lens, o, lse, seed = ctx.saved_tensors
        w, p_drop = ctx.w, ctx.p_drop
        B, H, T, D = qu.shape
        do = do.contiguous()
        delta = (do.float() * o.float()).sum(-1).reshape(B * H, T)
        dq, dk, dv = torch.empty_like(qu), torch.empty_like(k), torch.empty_like(v)
        dbd = torch.zeros(bd.shape, device=bd.device, dtype=torch.float32)
        assert dbd.stride() == bd.stride(), "bd must be contiguous"
        scale = D ** -0.5
        drop = p_drop > 0.0
        args = (*_strides(qu), *_strides(k), *_strides(v), *_strides(bd), *_strides(o), H, T, w, scale, p_drop)
        bm, bn, bw, bs = _BWD
        _bwd_dkdv[(triton.cdiv(T, bn), B * H)](qu, k, v, bd, lens, do, lse, delta, dk, dv, dbd, seed, *args,
                                                BM=bm, BN=bn, D=D, DROP=drop, num_warps=bw, num_stages=bs)
        _bwd_dq[(triton.cdiv(T, bm), B * H)](qu, k, v, bd, lens, do, lse, delta, dq, seed, *args,
                                              BM=bm, BN=bn, D=D, DROP=drop, num_warps=bw, num_stages=bs)
        return dq, dk, dv, dbd.to(bd.dtype), None, None, None, None


def band_attention(qu, k, v, bd, lens, w: int, p_drop: float = 0.0, seed: torch.Tensor | None = None):
    """qu, k, v [B, H, T, D] (bf16), bd [B, H, T, 2w+1] (raw (q + pos_bias_v) p^T, unscaled), lens [B] key lengths.
    Returns [B, H, T, D]; query rows with no valid key are 0."""
    return BandAttn.apply(qu, k, v, bd.contiguous(), lens, int(w), float(p_drop), seed)


def keep_mask(seed: torch.Tensor, B: int, H: int, T: int, p_drop: float) -> torch.Tensor:
    """[B, H, T, T] bool: the keep mask band_attention draws for this seed (tests)."""
    out = torch.empty(B * H, T, T, dtype=torch.int8, device=seed.device)
    _keep_mask[(triton.cdiv(T, 64), triton.cdiv(T, 64), B * H)](out, seed, H, T, p_drop, BM=64, BN=64)
    return out.view(B, H, T, T).bool()


def fast_forward(self, query, key, value, pad_mask, pos_emb, cache=None, *, _orig, max_T: int = 2048):
    """Drop-in forward for one RelPositionMultiHeadAttentionLongformer module."""
    T = query.shape[1]
    L, R = self.att_context_size
    if (cache is not None or self.global_tokens > 0 or T > max_T or L != R or self.d_k not in (16, 32, 64, 128)):
        return _orig(query, key, value, pad_mask, pos_emb, cache=cache)
    w = L
    # projections as NeMo: its fp32 casts are undone by autocast (avoid_float16_autocast_context only acts on fp16),
    # so under bf16 autocast q/k/v/pos/out are bf16 GEMMs in both paths
    if torch.is_autocast_enabled():
        query, key, value = query.to(torch.float32), key.to(torch.float32), value.to(torch.float32)
    q, k, v = self.forward_qkv(query, key, value)  # [B, H, T, d]
    B = q.shape[0]
    q_u = q + self.pos_bias_u.unsqueeze(1)
    q_v = q + self.pos_bias_v.unsqueeze(1)
    nb = pos_emb.size(0)
    p = self.linear_pos(pos_emb).view(nb, -1, self.h, self.d_k).transpose(1, 2)  # [1, H, 2w+1, d]
    with torch.autocast("cuda", enabled=False):
        bf = torch.bfloat16
        bd = torch.matmul(q_v.to(bf), p.to(bf).transpose(-2, -1))  # [B, H, T, 2w+1]
        if pad_mask is not None:
            lens = (~pad_mask[:, :T].bool()).sum(1).to(torch.int32)
        else:
            lens = torch.full((B,), T, dtype=torch.int32, device=q.device)
        pdrop = self.dropout.p if self.training else 0.0
        out = band_attention(q_u.to(bf), k.to(bf), v.to(bf), bd, lens, w, pdrop)
        if pad_mask is not None:  # NeMo: softmax(...).masked_fill(padded query, 0) -> those rows are 0
            out = out.masked_fill(pad_mask[:, None, :T, None].bool(), 0.0)
        out = out.transpose(1, 2).reshape(B, T, self.h * self.d_k)
    if not torch.is_autocast_enabled():  # fp32 without autocast (tests, eval): linear_out weights are fp32
        out = out.to(query.dtype)
    return self.linear_out(out)


def set_local_attention(model, att_context=(256, 256)) -> None:
    """Switch a FastConformer encoder to NeMo's local attention (rel_pos_local_attn, [L, R] frames), keeping exactly
    the original weights (see kili.infer.transcribe.set_attention for the NeMo bias issue it works around)."""
    from kili.infer.transcribe import set_attention
    set_attention(model, "rel_pos_local_attn", list(att_context))


def install(model, max_T: int = 2048) -> int:
    """Patch every local-attention module of model.encoder; returns the number patched (0 if the encoder does not use
    rel_pos_local_attn: call set_local_attention first). max_T: longer inputs (frames after subsampling) use NeMo's
    path; 2048 frames = 164 s at 80 ms per frame."""
    from nemo.collections.asr.parts.submodules.multi_head_attention import RelPositionMultiHeadAttentionLongformer
    n = 0
    for layer in model.encoder.layers:
        att = layer.self_attn
        if not isinstance(att, RelPositionMultiHeadAttentionLongformer) or "forward" in att.__dict__:
            continue
        orig = att.forward

        def fwd(self, query, key, value, pad_mask, pos_emb, cache=None, _o=orig):
            return fast_forward(self, query, key, value, pad_mask, pos_emb, cache, _orig=_o, max_T=max_T)

        att.forward = types.MethodType(fwd, att)
        n += 1
    return n
