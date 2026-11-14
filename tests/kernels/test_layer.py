"""Exactness of kili.kernels.layer (fused residual + dropout + LayerNorm; fused ConformerLayer) on a CUDA GPU.

1. add_ln, fp32, dropout 0: new residual, y, d residual, d x, d w, d b vs torch: rel <= 1e-5 (measured 1e-7).
2. dropout 0.1: the kernel's own mask (recovered with r = 0, x = 1) in a float64 reference: same error level; kept
   fraction ~0.9; d x is zero exactly where dropped (backward regenerates the mask).
3. training dtypes (fp32 residual, bf16 x, bf16 y): error vs float64 / torch's bf16-path error <= 1.5 (measured 1.00).
4. whole ConformerLayer vs NeMo's (real Parakeet weights, local attention [256, 256], dropout 0, train mode):
   fp32 rel <= 1e-4 (measured 2e-6..3e-5); bf16 error vs float64 / NeMo's own <= 1.5 (measured median 1.00, worst
   1.39).
"""
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required", allow_module_level=True)

import torch.nn.functional as F  # noqa: E402

from kili.kernels.attention import next_seed  # noqa: E402
from kili.kernels.layer import add_ln  # noqa: E402

N, C = 12500, 1024


class _LN:
    def __init__(self, w, b):
        self.weight, self.bias, self.eps = w, b, 1e-5


def rel(a, b) -> float:
    return float((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30))


def _ref(r, x, w, b, factor, keep=None, p=0.0):
    xd = x if keep is None else x * keep / (1 - p)
    ro = r + xd * factor
    return ro, F.layer_norm(ro, (r.shape[-1],), w, b, 1e-5)


def _grads(fn, inputs, g1, g2):
    xs = [t.detach().clone().requires_grad_(True) for t in inputs]
    ro, y = fn(*xs)
    ((ro.double() * g1).sum() + (y.double() * g2).sum()).backward()
    return [ro.detach(), y.detach()] + [t.grad for t in xs]


@pytest.fixture(scope="module")
def data():
    torch.manual_seed(0)
    ln = torch.nn.LayerNorm(C).cuda()
    with torch.no_grad():
        ln.weight.normal_(1.0, 0.2)
        ln.bias.normal_(0.0, 0.1)
    r = torch.randn(N, C, device="cuda") * 3
    x = torch.randn(N, C, device="cuda")
    g1, g2 = torch.randn(N, C, device="cuda").double(), torch.randn(N, C, device="cuda").double()
    return ln, r, x, g1, g2


@pytest.mark.parametrize("factor", [0.5, 1.0])
def test_add_ln_fp32(data, factor):
    ln, r, x, g1, g2 = data
    t = _grads(lambda r_, x_, w_, b_: _ref(r_, x_, w_, b_, factor), [r, x, ln.weight, ln.bias], g1, g2)
    k = _grads(lambda r_, x_, w_, b_: add_ln(r_, x_, _LN(w_, b_), factor, 0.0), [r, x, ln.weight, ln.bias], g1, g2)
    assert max(rel(a, b) for a, b in zip(k, t)) <= 1e-5


def test_add_ln_dropout(data):
    ln, r, x, g1, g2 = data
    p = 0.1
    seed = next_seed("cuda")
    ro1, _ = add_ln(torch.zeros_like(r), torch.ones_like(x), ln, 1.0, p, seed=seed)
    keep = (ro1 > 0).double()
    assert abs(float(keep.mean()) - (1 - p)) < 0.005
    t = _grads(lambda r_, x_, w_, b_: _ref(r_, x_, w_, b_, 0.5, keep, p),
               [r.double(), x.double(), ln.weight.double(), ln.bias.double()], g1, g2)
    k = _grads(lambda r_, x_, w_, b_: add_ln(r_, x_, _LN(w_, b_), 0.5, p, seed=seed), [r, x, ln.weight, ln.bias],
               g1, g2)
    assert max(rel(a, b) for a, b in zip(k, t)) <= 1e-5
    assert bool(((k[3] == 0) | (keep > 0)).all())  # d x is zero where the forward dropped


def test_add_ln_bf16(data):
    ln, r, x, g1, g2 = data
    xb = x.bfloat16()
    t64 = _grads(lambda r_, x_, w_, b_: _ref(r_, x_, w_, b_, 0.5),
                 [r.double(), xb.double(), ln.weight.double(), ln.bias.double()], g1, g2)
    tb = _grads(lambda r_, x_, w_, b_: (lambda ro: (ro, F.layer_norm(ro, (C,), w_, b_, 1e-5).bfloat16()))(r_ + x_ * 0.5),
                [r, xb, ln.weight, ln.bias], g1, g2)
    kb = _grads(lambda r_, x_, w_, b_: add_ln(r_, x_, _LN(w_, b_), 0.5, 0.0, y_dtype=torch.bfloat16),
                [r, xb, ln.weight, ln.bias], g1, g2)
    ratios = [rel(a, c) / max(rel(b_, c), 1e-12) for a, b_, c in zip(kb, tb, t64)]
    assert max(ratios) <= 1.5, ratios


@pytest.fixture(scope="module")
def local_encoder(fresh_parakeet):
    from kili.kernels.attention import set_local_attention
    set_local_attention(fresh_parakeet, (256, 256))
    return fresh_parakeet.encoder.cuda().float().train()


@pytest.mark.parametrize("B,T", [(6, 237), (2, 1500)])
def test_layer_vs_nemo(local_encoder, B, T):
    from kili.kernels import layer as klayer
    enc = local_encoder
    for mod in enc.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
    d = enc.d_model
    torch.manual_seed(0)
    x = torch.randn(B, T, d, device="cuda")
    lens = torch.randint(T // 2, T + 1, (B,), device="cuda")
    lens[0] = T
    with torch.no_grad():
        _, pos = enc.pos_enc(x=x, cache_len=0)
    pad_mask, att_mask = enc._create_masks(att_context_size=enc.att_context_size, padding_length=lens,
                                           max_audio_length=T, offset=None, device=x.device)
    go = torch.randn(B, T, d, device="cuda")
    valid = (~pad_mask)[:, :, None].double()

    def run(layer, ac, dtype=torch.float32):
        layer = layer.to(dtype)
        xi = x.detach().to(dtype).requires_grad_(True)
        layer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ac):
            y = layer(x=xi, att_mask=att_mask, pos_emb=pos.to(dtype), pad_mask=pad_mask)
        ((y.double() * valid) * go.double()).sum().backward()
        return [(y.detach().double() * valid), xi.grad] + [p.grad for p in layer.parameters()]

    base = enc.layers[9]

    def fused():
        lay = copy.deepcopy(base)
        holder = torch.nn.Module()
        holder.encoder = torch.nn.Module()
        holder.encoder.layers = [lay]
        klayer.install(holder)
        return lay

    r32, f32 = run(copy.deepcopy(base), False), run(fused(), False)
    assert max(rel(p, q) for p, q in zip(f32, r32)) <= 1e-4
    r64 = run(copy.deepcopy(base), False, torch.float64)
    rb, fb = run(copy.deepcopy(base), True), run(fused(), True)
    ratios = [rel(p, t) / max(rel(q, t), 1e-12) for p, q, t in zip(fb, rb, r64)]
    assert max(ratios) <= 1.5, f"worst {max(ratios):.2f}"
