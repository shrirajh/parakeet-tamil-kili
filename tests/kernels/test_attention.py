"""Exactness of kili.kernels.attention (fused banded rel-pos attention) on a CUDA GPU.

1. kernel, dropout 0: output and grads of q_u, k, v, bd (bf16 operands) vs a float64 reference, next to the same
   errors of PyTorch's dense bf16 SDPA; the kernel may be at most 2x worse (measured: equal or better).
2. kernel, dropout 0.1: the kernel's own keep mask in the float64 reference: same error level (backward regenerates
   the forward's mask); kept fraction ~0.9; consecutive calls draw different masks.
3. module: NeMo's RelPositionMultiHeadAttentionLongformer [256, 256] (d 1024, 8 heads, random weights) under bf16
   autocast, mixed lengths up to 1563 frames (125 s): error vs NeMo in float64 / NeMo's own bf16 error <= 1.5 for the
   output, the input gradient and every parameter gradient (measured: median 0.91-1.00, worst 1.24).
4. fallback: inputs longer than max_T go through NeMo's own forward unchanged.
"""
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required", allow_module_level=True)

import torch.nn.functional as F  # noqa: E402

from kili.kernels.attention import band_attention, fast_forward, keep_mask, next_seed  # noqa: E402

NEG = -10000.0
SHAPES = [(5, 200), (3, 600), (2, 1500)]  # (B, T): 16-120 s at 80 ms per frame


def rel(a, b, valid=None) -> float:
    d = (a.double() - b.double()).abs()
    if valid is not None:
        d = d * valid
    return float(d.max() / b.double().abs().max().clamp_min(1e-30))


def dense_bias(bd, T, w, lens):
    i = torch.arange(T, device=bd.device)
    r = i[None, :] - i[:, None]
    dense = bd.gather(-1, (r + w).clamp(0, 2 * w).expand(bd.shape[0], bd.shape[1], T, T))
    bad = ((r < -w) | (r > w))[None, None] | (i[None, :] >= lens[:, None])[:, None, None, :]
    return dense.masked_fill(bad, NEG)


def reference(qu, k, v, bd, lens, w, keep=None, p=0.0):
    s = qu.shape[-1] ** 0.5
    a = torch.softmax((torch.matmul(qu, k.transpose(-2, -1)) + dense_bias(bd, qu.shape[2], w, lens)) / s, -1)
    if keep is not None:
        a = a * keep / (1 - p)
    return torch.matmul(a, v)


def sdpa_bf16(qu, k, v, bd, lens, w):
    s = qu.shape[-1] ** 0.5
    return F.scaled_dot_product_attention(qu, k, v, attn_mask=dense_bias(bd, qu.shape[2], w, lens) / s, scale=1 / s)


def grads(fn, xs, go):
    xs = [x.detach().clone().requires_grad_(True) for x in xs]
    o = fn(*xs)
    (o.float() * go).sum().backward()
    return [o.detach().float()] + [x.grad.float() for x in xs]


def _inputs(B, T, H=8, D=128, w=256):
    mk = lambda *s: torch.randn(*s, device="cuda") * 0.5  # noqa: E731
    qu, k, v = (mk(B, H, T, D) for _ in range(3))
    bd = mk(B, H, T, 2 * w + 1) * 4
    lens = torch.randint(T // 2, T + 1, (B,), device="cuda")
    lens[0] = T
    # the attention module zeroes padded query rows (NeMo's masked_fill), so they never receive gradient
    qvalid = (torch.arange(T, device="cuda")[None, :] < lens[:, None])[:, None, :, None].double()
    go = torch.randn(B, H, T, D, device="cuda") * qvalid.float()
    return (qu, k, v, bd), lens, go, qvalid


@pytest.mark.parametrize("B,T", SHAPES)
def test_kernel_vs_float64(B, T):
    torch.manual_seed(0)
    w = 256
    xs, lens, go, qvalid = _inputs(B, T, w=w)
    xs64, xs16 = [t.double() for t in xs], [t.bfloat16() for t in xs]
    ref = grads(lambda *a: reference(*a, lens, w), xs64, go.double())
    ker = grads(lambda *a: band_attention(*a, lens, w), xs16, go)
    den = grads(lambda *a: sdpa_bf16(*a, lens, w), xs16, go)
    for i, name in enumerate(("out", "d q_u", "d k", "d v", "d bd")):
        m = qvalid if i == 0 else None
        ek, ed = rel(ker[i], ref[i], m), rel(den[i], ref[i], m)
        assert ek <= 2 * max(ed, 1e-6), f"{name}: kernel {ek:.3e} vs dense bf16 SDPA {ed:.3e}"


@pytest.mark.parametrize("B,T", SHAPES)
def test_kernel_dropout(B, T):
    torch.manual_seed(0)
    w, p = 256, 0.1
    xs, lens, go, qvalid = _inputs(B, T, w=w)
    xs64, xs16 = [t.double() for t in xs], [t.bfloat16() for t in xs]
    seed = next_seed("cuda")
    keep = keep_mask(seed, B, 8, T, p)
    assert abs(float(keep.float().mean()) - (1 - p)) < 0.005
    ref = grads(lambda *a: reference(*a, lens, w, keep.double(), p), xs64, go.double())
    ker = grads(lambda *a: band_attention(*a, lens, w, p, seed), xs16, go)
    for i, name in enumerate(("out", "d q_u", "d k", "d v", "d bd")):
        e = rel(ker[i], ref[i], qvalid if i == 0 else None)
        assert e <= 2e-2, f"{name}: {e:.3e} vs float64 with the same mask"
    k2 = keep_mask(next_seed("cuda"), B, 8, T, p)
    assert float((k2 != keep).float().mean()) > 0.1  # independent masks per call


def _nemo_module():
    pytest.importorskip("nemo.collections.asr")
    from nemo.collections.asr.parts.submodules.multi_head_attention import (
        LocalAttRelPositionalEncoding, RelPositionMultiHeadAttentionLongformer)
    torch.manual_seed(0)
    d, H = 1024, 8
    u = torch.nn.Parameter(torch.randn(H, d // H) * 0.1)
    v = torch.nn.Parameter(torch.randn(H, d // H) * 0.1)
    att = RelPositionMultiHeadAttentionLongformer(H, d, 0.0, u, v, [256, 256], use_bias=False).cuda().train()
    pe = LocalAttRelPositionalEncoding([256, 256], d_model=d, dropout_rate=0.0, max_len=5000).cuda()
    pe.extend_pe(1, "cuda", torch.float32)
    return att, pe, d


def _run(mod, pe, x0, pad, fast: bool, autocast: bool, max_T: int = 4096):
    x = x0.clone().requires_grad_(True)
    mod.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        xs, pos = pe(x)
        pos = pos.to(x.dtype)
        if fast:
            out = fast_forward(mod, xs, xs, xs, pad, pos, None, _orig=mod.forward, max_T=max_T)
        else:
            out = mod(xs, xs, xs, pad, pos)
    g = torch.randn(out.shape, device="cuda", generator=torch.Generator("cuda").manual_seed(1), dtype=torch.float64)
    (out.double() * g).sum().backward()
    return [out.detach()] + [x.grad] + [p.grad for p in mod.parameters() if p.grad is not None]


@pytest.mark.parametrize("lens", [[300, 200, 37], [700, 512, 513, 64], [1563], [1367]])
def test_module_bf16_vs_nemo(lens):
    att, pe, d = _nemo_module()
    att64 = copy.deepcopy(att).double()
    B, T = len(lens), max(lens)
    x0 = torch.randn(B, T, d, device="cuda")
    pad = torch.arange(T, device="cuda")[None, :] >= torch.tensor(lens, device="cuda")[:, None]
    truth = _run(att64, pe, x0.double(), pad, fast=False, autocast=False)
    nemo = _run(att, pe, x0, pad, fast=False, autocast=True)
    fast = _run(att, pe, x0, pad, fast=True, autocast=True)
    ratios = [rel(f, t) / max(rel(n, t), 1e-12) for f, n, t in zip(fast, nemo, truth)]
    assert max(ratios) <= 1.5, f"error / NeMo's own bf16 error: {[round(r, 2) for r in ratios]}"


def test_fallback_above_max_T():
    att, pe, d = _nemo_module()
    x0 = torch.randn(2, 300, d, device="cuda")
    pad = torch.zeros(2, 300, dtype=torch.bool, device="cuda")
    a = _run(att, pe, x0, pad, fast=False, autocast=False)
    b = _run(att, pe, x0, pad, fast=True, autocast=False, max_T=100)
    assert max(rel(p, q) for p, q in zip(b, a)) <= 1e-6  # same NeMo path (backward atomics may reorder sums)
