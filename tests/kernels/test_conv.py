"""Exactness of kili.kernels.conv (Triton depthwise conv; conv module without layout copies) on a CUDA GPU.

1. depthwise kernel vs F.conv1d: fp32 (TF32 off) y, dx, dw rel <= 1e-5 (measured: y, dx identical, dw 2e-7);
   bf16: error vs float64 / cudnn's bf16 error <= 1.5 (measured 1.00).
2. conv module vs NeMo's ConformerConvolution (real Parakeet weights, 2 training steps): fp32 output, input grad,
   every parameter grad and BatchNorm running stats rel <= 1e-5 (measured 4e-6); bf16 error vs float64 / NeMo's own
   bf16 error <= 1.5 (measured median 1.00, worst 1.17); eval mode rel <= 1e-5. Both depthwise paths.
"""
from __future__ import annotations

import copy
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required", allow_module_level=True)

import torch.nn.functional as F  # noqa: E402

from kili.kernels import conv as kconv  # noqa: E402


def rel(a, b) -> float:
    return float((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30))


def _ref_dw(x, w, P):  # x [B, T, C]
    return F.conv1d(x.transpose(1, 2), w, None, padding=P, groups=w.shape[0]).transpose(1, 2)


def _cudnn_cl(x, w, P):  # cudnn on the channels_last view
    xc = x.unsqueeze(2).permute(0, 3, 1, 2)
    return F.conv2d(xc, w.unsqueeze(-1), None, padding=(P, 0), groups=w.shape[0]).permute(0, 2, 3, 1).squeeze(2)


def _dw_grads(fn, x, w, go, ac=False):
    xi, wi = x.detach().clone().requires_grad_(True), w.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ac):
        y = fn(xi, wi, 4)
    (y.double() * go).sum().backward()
    return y.detach(), xi.grad, wi.grad


@pytest.mark.parametrize("B,T,C", [(80, 150, 1024), (6, 1500, 1024), (3, 37, 1024)])
def test_depthwise(B, T, C):
    torch.manual_seed(0)
    x = torch.randn(B, T, C, device="cuda")
    w = torch.randn(C, 1, 9, device="cuda") * 0.3
    go = torch.randn(B, T, C, device="cuda").double()
    r, k = _dw_grads(_ref_dw, x, w, go), _dw_grads(kconv.dwconv_btc, x, w, go)
    assert max(rel(a, b) for a, b in zip(k, r)) <= 1e-5
    r64 = _dw_grads(_ref_dw, x.double(), w.double(), go)
    xb = x.bfloat16()
    cb = _dw_grads(_cudnn_cl, xb, w, go, ac=True)
    kb = _dw_grads(kconv.dwconv_btc, xb, w, go, ac=True)
    ratios = [rel(a, t) / max(rel(c, t), 1e-12) for a, c, t in zip(kb, cb, r64)]
    assert max(ratios) <= 1.5, ratios


def _patched(mod, triton_dw):
    m = copy.deepcopy(mod)
    m.forward = types.MethodType(lambda self, x, pad_mask=None, cache=None: kconv.fast_forward(
        self, x, pad_mask, cache, _orig=type(self).forward.__get__(self), triton_dw=triton_dw), m)
    return m


def _run(mod, x, pad, go, ac, dtype=torch.float32, steps=2):
    mod = mod.to(dtype)
    outs = None
    for _ in range(steps):
        xi = x.detach().to(dtype).requires_grad_(True)
        mod.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ac):
            y = mod(xi, pad_mask=pad)
        (y.to(go.dtype) * go).sum().backward()
        outs = [y.detach()] + [xi.grad] + [p.grad for p in mod.parameters()]
    return outs + [mod.batch_norm.running_mean.clone(), mod.batch_norm.running_var.clone()]


@pytest.mark.parametrize("triton_dw", [True, False])
@pytest.mark.parametrize("B,T", [(6, 237), (2, 1500)])
def test_conv_module_vs_nemo(parakeet, triton_dw, B, T):
    base = copy.deepcopy(parakeet.encoder.layers[7].conv).cuda().float().train()
    d = base.pointwise_conv2.out_channels
    torch.manual_seed(0)
    x = torch.randn(B, T, d, device="cuda")
    lens = torch.randint(T // 2, T + 1, (B,), device="cuda")
    lens[0] = T
    pad = torch.arange(T, device="cuda")[None, :] >= lens[:, None]
    go = torch.randn(B, T, d, device="cuda")
    r32 = _run(copy.deepcopy(base), x, pad, go, False)
    f32 = _run(_patched(base, triton_dw), x, pad, go, False)
    assert max(rel(p, q) for p, q in zip(f32, r32)) <= 1e-5
    r64 = _run(copy.deepcopy(base), x.double(), pad, go.double(), False, torch.float64)
    rb = _run(copy.deepcopy(base), x, pad, go, True)
    fb = _run(_patched(base, triton_dw), x, pad, go, True)
    ratios = [rel(p, t) / max(rel(q, t), 1e-12) for p, q, t in zip(fb, rb, r64)]
    assert max(ratios) <= 1.5, f"worst {max(ratios):.2f}"
    ev, fe = copy.deepcopy(base).eval(), _patched(base, triton_dw).eval()
    with torch.no_grad():
        assert rel(fe(x, pad_mask=pad), ev(x, pad_mask=pad)) <= 1e-5
