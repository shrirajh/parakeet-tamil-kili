"""Exactness of kili.kernels.tdt_kernels / tdt_loss (gathered TDT loss) on a CUDA GPU, against NeMo.

1. row kernels vs a plain PyTorch reference (fp32 and bf16 logits, odd sizes): outputs and logit gradients.
2. bias inside the row kernels vs addmm + row kernels: fp32 outputs and grads of h, W, b rel <= 1e-5 (measured:
   identical, bias grad 1.9e-7); bf16 error vs float64 / the addmm path's <= 1.5 (measured 0.81-1.12).
3. Triton TDT recursion vs NeMo's numba TDT loss on identical reduced lattices, TDT and the RNN-T (omega) branch:
   cost rel < 1e-4, gradient rel < 1e-3.
4. the installed loss on Parakeet v3 (fp32, eval) vs NeMo's own joint + loss run one utterance at a time: loss rel
   < 1e-5, gradients w.r.t. encoder output, prediction-net output and every joint parameter rel < 1e-3; with and
   without bias_in_rows, with logits kept and recomputed.
5. empty-label clips (non-speech negatives) in a batch: per-utterance costs finite and equal to NeMo's where NeMo's
   are finite; every reduction finite on an all-empty batch.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required", allow_module_level=True)

from kili.kernels import tdt_loss as tg  # noqa: E402
from kili.kernels.tdt_kernels import gathered_rows, tdt_loss_from_reduced  # noqa: E402


def rel(a, b) -> float:
    return float((a.double() - b.double()).abs().max() / b.double().abs().max().clamp_min(1e-30))


def rows_torch(z, y, blank: int, n_dur: int):
    """Reference for gathered_rows (no bias)."""
    Vt = z.shape[-1] - n_dur
    z_lab = z[..., :Vt].float()
    idx = y.long().unsqueeze(-1)
    zl = torch.gather(z_lab, -1, idx).squeeze(-1)
    zb = z_lab[..., blank]
    mask = torch.zeros_like(z_lab, dtype=torch.bool).scatter(-1, idx, True)
    mask[..., blank] = True
    lse_o = torch.logsumexp(z_lab.masked_fill(mask, float("-inf")), -1)
    return lse_o, zl, zb, z[..., Vt:].float()


# ---------------------------------------------------------------------------------------------- 1. row kernels

@pytest.mark.parametrize("dt", ["float32", "bfloat16"])
@pytest.mark.parametrize("N,Vt,D", [(5, 37, 5), (300, 8193, 5), (1000, 2049, 3)])
def test_rows_vs_torch(dt, N, Vt, D):
    dt = getattr(torch, dt)
    gen = torch.Generator(device="cpu").manual_seed(N)
    z0 = (3 * torch.randn(N, Vt + D, generator=gen)).to("cuda", dt)
    y = torch.randint(0, Vt - 1, (N,), generator=gen).to("cuda")
    gs = [torch.randn(N, generator=gen).to("cuda") for _ in range(3)] + [torch.randn(N, D, generator=gen).to("cuda")]
    outs, grads = [], []
    for fn in (rows_torch, gathered_rows):
        z = z0.clone().requires_grad_(True)
        o = fn(z, y, Vt - 1, D)
        sum((a * b).sum() for a, b in zip(o, gs)).backward()
        outs.append([t.detach() for t in o])
        grads.append(z.grad.float())
    eo = max((a - b).abs().max().item() for a, b in zip(*outs))
    eg = ((grads[0] - grads[1]).abs().max() / grads[0].abs().max()).item()
    fp32 = dt == torch.float32
    assert eo < (1e-4 if fp32 else 1e-2) and eg < (1e-4 if fp32 else 2e-2), (eo, eg)


# ---------------------------------------------------------------------------------------------- 2. bias in rows

def _rows_bias_run(h, W, b, y, blank, D, split, dtype, ac):
    lin = torch.nn.Linear(W.shape[1], W.shape[0]).cuda().to(dtype)
    with torch.no_grad():
        lin.weight.copy_(W)
        lin.bias.copy_(b)
    hh = h.detach().clone().to(dtype).requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ac):
        z, w, bias = tg.joint_padded(lin, hh, split_bias=split)
        outs = gathered_rows(z, y, blank, D, width=w, bias=bias)
    torch.manual_seed(1)
    loss = sum((o.double() * torch.randn(o.shape, device="cuda", dtype=torch.float64)).sum() for o in outs)
    loss.backward()
    return [o.detach() for o in outs] + [hh.grad, lin.weight.grad, lin.bias.grad]


def test_bias_in_rows():
    torch.manual_seed(0)
    n, H, V, D, blank = 20000, 640, 10241, 5, 10240
    h = torch.relu(torch.randn(n, H, device="cuda"))
    W = torch.randn(V + D, H, device="cuda") * 0.05
    b = torch.randn(V + D, device="cuda") * 0.5
    y = torch.randint(0, V - 1, (n,), device="cuda", dtype=torch.int32)
    a = _rows_bias_run(h, W, b, y, blank, D, False, torch.float32, False)
    s = _rows_bias_run(h, W, b, y, blank, D, True, torch.float32, False)
    assert max(rel(p, q) for p, q in zip(s, a)) <= 1e-5
    r64 = _rows_bias_run(h.double(), W.double(), b.double(), y, blank, D, False, torch.float64, False)
    ab = _rows_bias_run(h, W, b, y, blank, D, False, torch.float32, True)
    sb = _rows_bias_run(h, W, b, y, blank, D, True, torch.float32, True)
    ratios = [rel(p, t) / max(rel(q, t), 1e-12) for p, q, t in zip(sb, ab, r64)]
    assert max(ratios) <= 1.5, ratios


# ---------------------------------------------------------------------------------------------- model fixture

@pytest.fixture(scope="module")
def model(fresh_parakeet):
    m = fresh_parakeet.cuda().eval()
    m.joint.set_fuse_loss_wer(False)  # reference = plain NeMo joint + TDT loss
    # omega > 0 makes the TDT loss fall back to the RNN-T loss on a random fraction of calls: not comparable run to
    # run. Exactness is checked with omega = 0 (the recursion test covers the omega branch).
    m.loss._loss.omega = 0.0
    m.joint.requires_grad_(True)
    return m


def _dims(m):
    return m.cfg.joint.jointnet.encoder_hidden, m.cfg.joint.jointnet.pred_hidden, m.loss._loss.blank


# ---------------------------------------------------------------------------------------------- 3. recursion

@pytest.mark.parametrize("omega", [0.0, 1.0])
@pytest.mark.parametrize("B,T,U", [(1, 5, 2), (3, 40, 12), (4, 200, 90), (2, 375, 300)])
def test_recursion_vs_numba(model, omega, B, T, U):
    from nemo.collections.asr.parts.numba.rnnt_loss.rnnt_pytorch import TDTLossNumba
    inner = model.loss._loss
    gen = torch.Generator(device="cpu").manual_seed(B * 1000 + T)
    z0 = torch.randn(B, T, U + 1, 3 + len(inner.durations), generator=gen).to("cuda")
    xl = torch.tensor([T - 2 * i for i in range(B)], device="cuda").clamp(min=1)
    yl = torch.tensor([U - 3 * i for i in range(B)], device="cuda").clamp(min=1)
    res = []
    for impl in ("numba", "triton"):
        # leaf = pre-softmax logits as in the model: the two kernels' gradients w.r.t. the log-probs differ by the
        # softmax Jacobian, which cancels after this chain rule
        z = z0.clone().requires_grad_(True)
        red = torch.cat([torch.log_softmax(z[..., :3], -1), z[..., 3:]], -1)
        if impl == "numba":
            nl = TDTLossNumba(blank=2, durations=inner.durations, reduction=None, sigma=inner.sigma, omega=omega)
            cost = nl(acts=red, labels=torch.zeros(B, U, dtype=torch.long, device="cuda"), act_lens=xl, label_lens=yl)
        else:
            cost = tdt_loss_from_reduced(red, xl, yl, inner.durations, inner.sigma, omega)
        (cost * torch.arange(1, B + 1, device="cuda", dtype=torch.float32)).sum().backward()
        res.append((cost.detach(), z.grad.detach().clone()))
    (cn, gn), (ct, gt) = res
    assert ((cn - ct).abs() / cn.abs().clamp(min=1e-6)).max().item() < 1e-4
    assert ((gn - gt).abs().max() / gn.abs().max().clamp(min=1e-12)).item() < 1e-3


# ---------------------------------------------------------------------------------------------- 4. installed loss

def _inputs(B, T, U, enc_dim, pred_dim, V, seed, scale=0.1):
    # scale 0.1 keeps joint logits in a realistic range (unit-normal inputs give ~1000-nat losses)
    g = torch.Generator(device="cpu").manual_seed(seed)
    enc = scale * torch.randn(B, enc_dim, T, generator=g).to("cuda")
    dec = scale * torch.randn(B, pred_dim, U + 1, generator=g).to("cuda")
    targets = torch.randint(0, V, (B, U), generator=g).to("cuda")
    enc_len = torch.tensor([T - 3 * i for i in range(B)], device="cuda").clamp(min=2)
    tgt_len = torch.tensor([U - 2 * i for i in range(B)], device="cuda").clamp(min=1)
    return enc, dec, targets, enc_len, tgt_len


def _loss_and_grads(m, enc, dec, targets, enc_len, tgt_len, per_utt=False):
    enc = enc.clone().requires_grad_(True)
    dec = dec.clone().requires_grad_(True)
    m.zero_grad(set_to_none=True)
    if per_utt:  # NeMo's reference, one utterance's lattice at a time (as fused_batch_size=1)
        losses = []
        red, m.loss.reduction = m.loss.reduction, None
        try:
            for b in range(enc.shape[0]):
                T, U = int(enc_len[b]), int(tgt_len[b])
                j = m.joint(encoder_outputs=enc[b:b + 1, :, :T], decoder_outputs=dec[b:b + 1, :, :U + 1])
                losses.append(m.loss(log_probs=j, targets=targets[b:b + 1, :U], input_lengths=enc_len[b:b + 1],
                                     target_lengths=tgt_len[b:b + 1]))
        finally:
            m.loss.reduction = red
        loss = m.loss.reduce(torch.cat(losses), tgt_len)
    else:
        j = m.joint(encoder_outputs=enc, decoder_outputs=dec)
        loss = m.loss(log_probs=j, targets=targets, input_lengths=enc_len, target_lengths=tgt_len)
    loss.backward()
    grads = {"enc": enc.grad.detach().clone(), "dec": dec.grad.detach().clone()}
    grads.update({n: p.grad.detach().clone() for n, p in m.joint.named_parameters() if p.grad is not None})
    return loss.detach().float().item(), grads


@pytest.mark.parametrize("bias_in_rows", [True, False])
@pytest.mark.parametrize("recompute_cells", [float("inf"), 0])
def test_loss_vs_nemo(model, bias_in_rows, recompute_cells):
    enc_dim, pred_dim, V = _dims(model)
    for (B, T, U) in [(1, 7, 3), (3, 40, 12), (2, 120, 37)]:
        x = _inputs(B, T, U, enc_dim, pred_dim, V, seed=B * 100 + T)
        ref = _loss_and_grads(model, *x, per_utt=True)
        tg.install(model, chunk_elems=2 * 2**20, recompute_cells=recompute_cells, bias_in_rows=bias_in_rows)
        try:
            got = _loss_and_grads(model, *x)
        finally:
            tg.uninstall(model)
        assert abs(ref[0] - got[0]) / abs(ref[0]) < 1e-5, (B, T, U, ref[0], got[0])
        worst = max(((ref[1][k] - got[1][k]).abs().max() / (ref[1][k].abs().max() + 1e-12)).item() for k in ref[1])
        assert worst < 1e-3, (B, T, U, worst)


# ---------------------------------------------------------------------------------------------- 5. empty labels

EMPTY_CASES = {  # (encoder frames, target tokens) per utterance
    "empty shortest": ([153, 75, 15], [52, 33, 0]),
    "empty shortest, first": ([48, 67, 60, 201], [0, 22, 4, 62]),
    "empty, two clips": ([49, 262], [0, 78]),
    "empty not shortest": ([41, 28, 170], [0, 7, 81]),
    "empty in the middle": ([14, 43, 169], [8, 0, 37]),
    "single empty": ([30], [0]),
    "all empty": ([30, 20], [0, 0]),
}


def _ref_costs(m, enc, dec, tgt, el, tl):
    out = []
    red, m.loss.reduction = m.loss.reduction, None
    try:
        for b in range(enc.shape[0]):
            T, U = int(el[b]), int(tl[b])
            j = m.joint(encoder_outputs=enc[b:b + 1, :, :T], decoder_outputs=dec[b:b + 1, :, :U + 1])
            out.append(float(m.loss(log_probs=j, targets=tgt[b:b + 1, :U], input_lengths=el[b:b + 1],
                                    target_lengths=tl[b:b + 1])))
    finally:
        m.loss.reduction = red
    return out


def test_empty_labels(model):
    enc_dim, pred_dim, V = _dims(model)
    cases = {}
    with torch.no_grad():
        for k, (T, U) in EMPTY_CASES.items():
            enc, dec, tgt, _, _ = _inputs(len(T), max(T), max(max(U), 1), enc_dim, pred_dim, V, seed=1)
            el, tl = torch.tensor(T, device="cuda"), torch.tensor(U, device="cuda")
            cases[k] = (enc, dec, tgt, el, tl, _ref_costs(model, enc, dec, tgt, el, tl))
    tg.install(model)
    red0 = model.loss.reduction
    try:
        model.loss.reduction = None
        for k, (enc, dec, tgt, el, tl, rc) in cases.items():
            with torch.no_grad():
                c = model.loss(log_probs=model.joint(encoder_outputs=enc, decoder_outputs=dec), targets=tgt,
                               input_lengths=el, target_lengths=tl).float()
            assert bool(torch.isfinite(c).all()), k
            r = torch.tensor(rc)
            ok = torch.isfinite(r)  # compare where NeMo's own per-utterance cost is finite
            assert torch.allclose(c.cpu()[ok], r[ok], rtol=1e-4, atol=1e-3), (k, c.tolist(), rc)
        enc, dec, tgt, el, tl, _ = cases["all empty"]
        for red in ("mean_volume", "mean", "mean_batch", "sum"):
            model.loss.reduction = red
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                v = model.loss(log_probs=model.joint(encoder_outputs=enc, decoder_outputs=dec), targets=tgt,
                               input_lengths=el, target_lengths=tl)
            assert bool(torch.isfinite(v)), red
    finally:
        model.loss.reduction = red0
        tg.uninstall(model)
