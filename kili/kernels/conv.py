"""Conformer convolution module without layout copies, with a Triton depthwise convolution on [B, T, C].

NeMo's ConformerConvolution: x [B, T, C] -> transpose -> pointwise Conv1d (k=1) -> GLU(dim=1) -> masked_fill ->
F.pad (copy) -> depthwise Conv1d (k=9) -> BatchNorm1d -> SiLU -> pointwise Conv1d -> transpose. Every transpose and
pad is a full copy of a [B, C, T] activation in forward and again in backward.

Here (same maths):
- pointwise convs are F.linear on [B, T, C] (a k=1 conv is a matmul), GLU on the last dim;
- the depthwise conv runs directly on [B, T, C] (Triton kernels below: C is contiguous, so every load is a coalesced
  row of channels; the kernel's zero padding replaces F.pad), or, with triton_dw=False, through cudnn on a
  [B, C, T, 1] channels_last *view* (no copy);
- BatchNorm + SiLU run on the [B, C, T, 1] view: BatchNorm2d over (B, T, 1) gives exactly BatchNorm1d's statistics
  over (B, T); running stats / num_batches_tracked follow torch.
Only the configuration Parakeet uses (batch_norm, glu, symmetric padding, stride 1, no cache); anything else falls back
to NeMo's forward.

Depthwise kernels (k = 9, groups = C, zero padding P = 4, stride 1), fp32 accumulation, activation dtype in/out:
  y[b, t, c]  = sum_k w[c, k] x[b, t + k - P, c]  (+ bias[c])
  dx[b, t, c] = sum_k w[c, k] dy[b, t - k + P, c]
  dw[c, k]    = sum_{b, t} dy[b, t, c] x[b, t + k - P, c]     (per-tile partials, summed in torch in fp32)

  from kili.kernels.conv import install; install(model)
"""
from __future__ import annotations

import types

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _dw_fwd(X, Wt, Bias, Y, T, C, P: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr,
            HAS_BIAS: tl.constexpr):
    pt, pc, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t = pt * BT + tl.arange(0, BT)
    c = pc * BC + tl.arange(0, BC)
    cm = c < C
    base = b.to(tl.int64) * T * C
    acc = tl.zeros([BT, BC], tl.float32)
    for k in tl.static_range(K):
        ts = t + (k - P)
        m = (ts >= 0) & (ts < T)
        x = tl.load(X + base + ts[:, None] * C + c[None, :], mask=m[:, None] & cm[None, :], other=0.0)
        w = tl.load(Wt + c * K + k, mask=cm, other=0.0).to(tl.float32)
        acc += x.to(tl.float32) * w[None, :]
    if HAS_BIAS:
        acc += tl.load(Bias + c, mask=cm, other=0.0).to(tl.float32)[None, :]
    tl.store(Y + base + t[:, None] * C + c[None, :], acc.to(Y.dtype.element_ty), mask=(t < T)[:, None] & cm[None, :])


@triton.jit
def _dw_bwd(X, Wt, DY, DX, DWP, T, C, NT, P: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr):
    """dx for this tile, and this tile's partial dw[k, c] (written to DWP[b * NT + pt, k, c])."""
    pt, pc, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    t = pt * BT + tl.arange(0, BT)
    c = pc * BC + tl.arange(0, BC)
    cm = c < C
    tm = t < T
    base = b.to(tl.int64) * T * C
    dy = tl.load(DY + base + t[:, None] * C + c[None, :], mask=tm[:, None] & cm[None, :], other=0.0).to(tl.float32)
    dx = tl.zeros([BT, BC], tl.float32)
    tile = b * NT + pt
    for k in tl.static_range(K):
        w = tl.load(Wt + c * K + k, mask=cm, other=0.0).to(tl.float32)
        # dx[t] += w[k] dy[t - k + P]
        ts = t - (k - P)
        m = (ts >= 0) & (ts < T)
        dys = tl.load(DY + base + ts[:, None] * C + c[None, :], mask=m[:, None] & cm[None, :], other=0.0)
        dx += dys.to(tl.float32) * w[None, :]
        # dw[k] += sum_t dy[t] x[t + k - P]
        tx = t + (k - P)
        mx = (tx >= 0) & (tx < T) & tm
        x = tl.load(X + base + tx[:, None] * C + c[None, :], mask=mx[:, None] & cm[None, :], other=0.0)
        part = tl.sum(dy * x.to(tl.float32), 0)
        tl.store(DWP + (tile.to(tl.int64) * K + k) * C + c, part, mask=cm)
    tl.store(DX + base + t[:, None] * C + c[None, :], dx.to(DX.dtype.element_ty), mask=tm[:, None] & cm[None, :])


BT, BC = 64, 128


class DWConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, bias, P: int):
        B, T, C = x.shape
        K = w.shape[-1]
        x = x.contiguous()
        wt = w.reshape(C, K).contiguous()
        y = torch.empty_like(x)
        grid = (triton.cdiv(T, BT), triton.cdiv(C, BC), B)
        _dw_fwd[grid](x, wt, bias if bias is not None else wt, y, T, C, P=P, K=K, BT=BT, BC=BC,
                      HAS_BIAS=bias is not None, num_warps=4)
        ctx.save_for_backward(x, wt)
        ctx.P, ctx.wshape, ctx.has_bias = P, w.shape, bias is not None
        return y

    @staticmethod
    def backward(ctx, dy):
        x, wt = ctx.saved_tensors
        B, T, C = x.shape
        K = wt.shape[1]
        dy = dy.contiguous()
        dx = torch.empty_like(x)
        NT = triton.cdiv(T, BT)
        dwp = torch.empty(B * NT, K, C, device=x.device, dtype=torch.float32)
        _dw_bwd[(NT, triton.cdiv(C, BC), B)](x, wt, dy, dx, dwp, T, C, NT, P=ctx.P, K=K, BT=BT, BC=BC, num_warps=4)
        dw = dwp.sum(0).t().reshape(ctx.wshape).to(wt.dtype)
        db = dy.float().sum((0, 1)).to(wt.dtype) if ctx.has_bias else None
        return dx, dw, db, None


def dwconv_btc(x, weight, padding: int, bias=None):
    """Depthwise conv over time. x [B, T, C]; weight [C, 1, K] (Conv1d depthwise layout); symmetric zero padding;
    returns [B, T, C]. Under autocast the weight is cast to x's dtype, as autocast's conv does."""
    if torch.is_autocast_enabled() and x.dtype != torch.float32:
        weight = weight.to(x.dtype)
        bias = bias.to(x.dtype) if bias is not None else None
    elif x.dtype != weight.dtype:
        x = x.to(weight.dtype)
    return DWConv.apply(x, weight, bias, int(padding))


def _bn(bn, y):
    """nn.BatchNorm's forward on a 4D [B, C, T, 1] input (BatchNorm1d's _check_input_dim rejects 4D)."""
    if bn.training and bn.track_running_stats:
        bn.num_batches_tracked.add_(1)
        mom = (1.0 / float(bn.num_batches_tracked)) if bn.momentum is None else bn.momentum
    else:
        mom = 0.0 if bn.momentum is None else bn.momentum
    use_batch = bn.training or bn.running_mean is None
    return F.batch_norm(y, bn.running_mean if not bn.training or bn.track_running_stats else None,
                        bn.running_var if not bn.training or bn.track_running_stats else None,
                        bn.weight, bn.bias, use_batch, mom, bn.eps)


def fast_forward(self, x, pad_mask=None, cache=None, *, _orig, triton_dw: bool = True):
    """Drop-in forward for one NeMo ConformerConvolution module."""
    dw = self.depthwise_conv
    if (cache is not None or self.norm_type != "batch_norm" or self.pointwise_activation != "glu_"
            or dw._left_padding != dw._right_padding or dw.stride != (1,) or dw.dilation != (1,)
            or type(self.batch_norm) is not torch.nn.BatchNorm1d):
        return _orig(x, pad_mask=pad_mask, cache=cache)
    x = F.linear(x, self.pointwise_conv1.weight.squeeze(-1), self.pointwise_conv1.bias)  # [B, T, 2C]
    x = F.glu(x, dim=-1)  # [B, T, C]
    if pad_mask is not None:
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
    if triton_dw:  # BN / SiLU then run on the result's channels_last view
        y = dwconv_btc(x, dw.weight, dw._left_padding, dw.bias).unsqueeze(2).permute(0, 3, 1, 2)
    else:
        xc = x.unsqueeze(2).permute(0, 3, 1, 2)  # [B, C, T, 1] view with channels_last strides (TC, 1, C, C)
        y = F.conv2d(xc, dw.weight.unsqueeze(-1), dw.bias, stride=1, padding=(dw._left_padding, 0), groups=dw.groups)
    y = _bn(self.batch_norm, y)
    y = self.activation(y)
    y = y.permute(0, 2, 3, 1).squeeze(2)  # [B, T, C] (a view when the layout stayed channels_last)
    return F.linear(y, self.pointwise_conv2.weight.squeeze(-1), self.pointwise_conv2.bias)


def install(model, triton_dw: bool = True) -> int:
    """Patch every ConformerConvolution of model.encoder; returns the number patched.
    triton_dw: Triton depthwise conv (as trained); False: cudnn on the channels_last view."""
    n = 0
    for layer in model.encoder.layers:
        conv = getattr(layer, "conv", None)
        if conv is None or "forward" in conv.__dict__:
            continue
        orig = conv.forward

        def fwd(self, x, pad_mask=None, cache=None, _o=orig):
            return fast_forward(self, x, pad_mask, cache, _orig=_o, triton_dw=triton_dw)

        conv.forward = types.MethodType(fwd, conv)
        n += 1
    return n
