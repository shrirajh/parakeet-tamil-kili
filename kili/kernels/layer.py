"""Conformer layer with the four residual + dropout + LayerNorm steps fused (Triton, forward + backward).

NeMo's ConformerLayer does, four times per layer:
    residual = residual + dropout(x) * factor        (factor 0.5 after the feed-forwards, 1 otherwise)
    y = LayerNorm(residual)                          (fp32 under autocast; the next GEMM casts it to bf16)
i.e. a dropout kernel, a mul, an fp32 add, LayerNorm forward, a cast, and in backward the LayerNorm backward (two
kernels), the dropout backward, the mul and the cast: ~10 memory-bound passes over [B*T, d_model]. Here: one forward
kernel (new residual in the residual's dtype, y in the consumer's dtype, mean/rstd saved) and one backward kernel
(d residual, d x with the regenerated dropout mask, per-block partial LayerNorm weight/bias grads summed in torch).
Statistics and arithmetic in fp32. Dropout: Philox keyed by a device seed (kili.kernels.attention.next_seed) and the
element index row * C + col, regenerated in backward.

The layer forward keeps NeMo's order and dtypes for Parakeet's configuration (rel_pos_local_attn, no caches, no
adapters; anything else falls back to NeMo's forward). Dropout p is the layer's own dropout module's, off in eval. The
residual keeps torch's promotion of (residual, x); LayerNorm outputs that feed GEMMs are produced directly in bf16
under autocast (NeMo makes them fp32 and the GEMM casts them: same values); the layer output stays fp32 as in NeMo.

Install before anything that wraps layer.forward (e.g. activation checkpointing): the checkpointed recompute draws the
same seeds and regenerates the same masks.

  from kili.kernels.layer import install; install(model)
"""
from __future__ import annotations

import types

import torch
import triton
import triton.language as tl

from kili.kernels.attention import next_seed


@triton.jit
def _fwd(R, X, RO, Y, W, Bb, MEAN, RSTD, SEED, C, factor, p, eps,
         BLOCK_C: tl.constexpr, DROP: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    m = offs < C
    r = tl.load(R + row * C + offs, mask=m, other=0.0).to(tl.float32)
    x = tl.load(X + row * C + offs, mask=m, other=0.0).to(tl.float32)
    if DROP:
        seed = tl.load(SEED)
        keep = tl.rand(seed, (row * C + offs).to(tl.uint32)) >= p
        x = tl.where(keep, x / (1.0 - p), 0.0)
    rn = r + x * factor
    tl.store(RO + row * C + offs, rn.to(RO.dtype.element_ty), mask=m)
    rn = rn.to(RO.dtype.element_ty).to(tl.float32)  # LayerNorm sees the stored residual, as in NeMo
    mean = tl.sum(tl.where(m, rn, 0.0), 0) / C
    xc = tl.where(m, rn - mean, 0.0)
    var = tl.sum(xc * xc, 0) / C
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + offs, mask=m, other=0.0).to(tl.float32)
    b = tl.load(Bb + offs, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + row * C + offs, (xc * rstd * w + b).to(Y.dtype.element_ty), mask=m)
    tl.store(MEAN + row, mean)
    tl.store(RSTD + row, rstd)


@triton.jit
def _bwd(DRO, DY, RO, W, MEAN, RSTD, DR, DX, DWP, DBP, SEED, N, C, factor, p,
         BLOCK_C: tl.constexpr, ROWS: tl.constexpr, DROP: tl.constexpr, HAS_DRO: tl.constexpr):
    g = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    m = offs < C
    w = tl.load(W + offs, mask=m, other=0.0).to(tl.float32)
    dw = tl.zeros([BLOCK_C], tl.float32)
    db = tl.zeros([BLOCK_C], tl.float32)
    if DROP:
        seed = tl.load(SEED)
    for i in range(ROWS):
        row = g * ROWS + i
        if row < N:
            rn = tl.load(RO + row * C + offs, mask=m, other=0.0).to(tl.float32)
            dy = tl.load(DY + row * C + offs, mask=m, other=0.0).to(tl.float32)
            mean = tl.load(MEAN + row)
            rstd = tl.load(RSTD + row)
            xhat = tl.where(m, (rn - mean) * rstd, 0.0)
            wdy = w * dy
            c1 = tl.sum(xhat * wdy, 0) / C
            c2 = tl.sum(wdy, 0) / C
            dr = (wdy - (xhat * c1 + c2)) * rstd
            if HAS_DRO:
                dr += tl.load(DRO + row * C + offs, mask=m, other=0.0).to(tl.float32)
            tl.store(DR + row * C + offs, dr.to(DR.dtype.element_ty), mask=m)
            dx = dr * factor
            if DROP:
                keep = tl.rand(seed, (row * C + offs).to(tl.uint32)) >= p
                dx = tl.where(keep, dx / (1.0 - p), 0.0)
            tl.store(DX + row * C + offs, dx.to(DX.dtype.element_ty), mask=m)
            dw += dy * xhat
            db += dy
    tl.store(DWP + g * C + offs, dw, mask=m)
    tl.store(DBP + g * C + offs, db, mask=m)


ROWS = 16  # rows per backward program (partial LayerNorm weight/bias sums per program)


class AddLN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, r, x, w, b, factor: float, p: float, eps: float, y_dtype, seed):
        C = r.shape[-1]
        r2, x2 = r.reshape(-1, C).contiguous(), x.reshape(-1, C).contiguous()
        N = r2.shape[0]
        ro = torch.empty(r2.shape, dtype=torch.promote_types(r.dtype, x.dtype), device=r.device)
        y = torch.empty(r2.shape, dtype=y_dtype, device=r.device)
        mean = torch.empty(N, dtype=torch.float32, device=r.device)
        rstd = torch.empty(N, dtype=torch.float32, device=r.device)
        BLOCK_C = triton.next_power_of_2(C)
        drop = p > 0.0
        _fwd[(N,)](r2, x2, ro, y, w, b, mean, rstd, seed, C, factor, p, eps, BLOCK_C=BLOCK_C, DROP=drop,
                   num_warps=4 if BLOCK_C <= 1024 else 8)
        ctx.save_for_backward(ro, w, mean, rstd, seed)
        ctx.factor, ctx.p, ctx.shape, ctx.x_dtype, ctx.r_dtype = factor, p, r.shape, x.dtype, r.dtype
        return ro.view(r.shape), y.view(r.shape)

    @staticmethod
    def backward(ctx, dro, dy):
        ro, w, mean, rstd, seed = ctx.saved_tensors
        N, C = ro.shape
        dy = torch.zeros_like(ro) if dy is None else dy.reshape(N, C).contiguous()
        has_dro = dro is not None
        dro2 = dro.reshape(N, C).contiguous() if has_dro else dy
        dr = torch.empty(N, C, dtype=ctx.r_dtype, device=ro.device)
        dx = torch.empty(N, C, dtype=ctx.x_dtype, device=ro.device)
        G = triton.cdiv(N, ROWS)
        dwp = torch.empty(G, C, dtype=torch.float32, device=ro.device)
        dbp = torch.empty(G, C, dtype=torch.float32, device=ro.device)
        BLOCK_C = triton.next_power_of_2(C)
        _bwd[(G,)](dro2, dy, ro, w, mean, rstd, dr, dx, dwp, dbp, seed, N, C, ctx.factor, ctx.p,
                   BLOCK_C=BLOCK_C, ROWS=ROWS, DROP=ctx.p > 0.0, HAS_DRO=has_dro, num_warps=4)
        dw, db = dwp.sum(0).to(w.dtype), dbp.sum(0).to(w.dtype)
        return dr.view(ctx.shape), dx.view(ctx.shape), dw, db, None, None, None, None, None


def add_ln(r, x, ln: torch.nn.LayerNorm, factor: float, p: float, y_dtype=None, seed=None):
    """(residual + dropout(x) * factor, LayerNorm of it). y_dtype: bf16 when the consumer is a GEMM under autocast."""
    if seed is None:
        seed = next_seed(r.device) if p > 0.0 else torch.zeros(1, dtype=torch.int32, device=r.device)
    if y_dtype is None:
        y_dtype = torch.promote_types(r.dtype, x.dtype)
    return AddLN.apply(r, x, ln.weight, ln.bias, float(factor), float(p), float(ln.eps), y_dtype, seed)


def fused_forward(self, x, att_mask=None, pos_emb=None, pad_mask=None, cache_last_channel=None, cache_last_time=None,
                  *, _orig):
    """Drop-in forward for one NeMo ConformerLayer."""
    if (cache_last_channel is not None or cache_last_time is not None or self.self_attention_model != "rel_pos_local_attn"
            or self.is_adapter_available() or self.is_access_enabled(getattr(self, "model_guid", None))):
        return _orig(x, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask, cache_last_channel=cache_last_channel,
                     cache_last_time=cache_last_time)
    p = self.dropout.p if self.training else 0.0
    ac = torch.is_autocast_enabled()
    yd = torch.bfloat16 if ac else None
    residual = x
    x = self.norm_feed_forward1(x)
    x = self.feed_forward1(x)
    residual, x = add_ln(residual, x, self.norm_self_att, self.fc_factor, p, y_dtype=yd)
    x = self.self_attn(query=x, key=x, value=x, pad_mask=pad_mask, pos_emb=pos_emb, cache=None)
    residual, x = add_ln(residual, x, self.norm_conv, 1.0, p, y_dtype=yd)
    x = self.conv(x, pad_mask=pad_mask, cache=None)
    residual, x = add_ln(residual, x, self.norm_feed_forward2, 1.0, p, y_dtype=yd)
    x = self.feed_forward2(x)
    _, x = add_ln(residual, x, self.norm_out, self.fc_factor, p, y_dtype=torch.float32 if ac else None)
    return x


def install(model) -> int:
    """Patch every ConformerLayer of model.encoder; returns the number patched. Must run before anything else wraps
    layer.forward."""
    n = 0
    for layer in model.encoder.layers:
        if "forward" in layer.__dict__:
            raise RuntimeError("install the fused layer before anything that wraps layer.forward")
        orig = layer.forward

        def fwd(self, *args, _o=orig, **kw):
            return fused_forward(self, *args, _orig=_o, **kw)

        layer.forward = types.MethodType(fwd, layer)
        n += 1
    return n
