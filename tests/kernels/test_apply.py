"""kili.kernels.apply / remove on the real Parakeet model, one bf16 training step (CUDA GPU, NeMo).

Same inputs, dropout / SpecAugment / dither off: the loss with every kernel installed must match NeMo's own path
(local attention [256, 256]) to bf16 noise, the encoder gradients must be finite, and remove() must restore NeMo's
forwards and loss.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA GPU required", allow_module_level=True)

import kili.kernels  # noqa: E402
from kili.kernels.attention import set_local_attention  # noqa: E402


def test_apply_training_step(fresh_parakeet):
    m = fresh_parakeet.cuda().train()
    set_local_attention(m, (256, 256))
    m.joint.set_fuse_loss_wer(False)
    m.spec_augmentation = None
    m.preprocessor.featurizer.dither = 0.0
    m.loss._loss.omega = 0.0  # omega > 0 randomly switches the TDT loss to plain RNN-T on some calls
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
        if isinstance(mod, torch.nn.LSTM):  # the prediction net's inter-layer dropout is not an nn.Dropout module
            mod.dropout = 0.0
    g = torch.Generator().manual_seed(0)
    wav = (0.05 * torch.randn(3, 16000 * 12, generator=g)).cuda()
    wav_len = torch.tensor([16000 * 12, 16000 * 9, 16000 * 4], device="cuda")
    tlen = torch.tensor([40, 25, 9], device="cuda")
    targets = torch.randint(0, m.loss._loss.blank, (3, 40), generator=g).cuda()

    def step(bf16=True):
        m.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            enc, el = m(input_signal=wav, input_signal_length=wav_len)
            dec, _, _ = m.decoder(targets=targets, target_length=tlen)
            j = m.joint(encoder_outputs=enc, decoder_outputs=dec)
            loss = m.loss(log_probs=j, targets=targets, input_lengths=el, target_lengths=tlen)
        loss.backward()
        grad = torch.cat([p.grad.float().flatten() for p in m.encoder.parameters() if p.grad is not None])
        return float(loss), grad

    cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    _, g32 = step(bf16=False)            # fp32 NeMo: the reference gradient
    ref_loss, g16 = step()               # bf16 NeMo: how far NeMo's own training precision is from it
    assert step()[0] == ref_loss         # deterministic with dropout off
    out = kili.kernels.apply(m, att_context=None)
    n = len(m.encoder.layers)
    assert out == {"layer": n, "attention": n, "conv": n, "tdt_loss": 1}
    loss, gk = step()
    assert abs(loss - ref_loss) / abs(ref_loss) < 5e-3, (loss, ref_loss)
    # bf16 kernels must be at least as faithful to the fp32 gradient as NeMo's own bf16 path (measured on a 3090:
    # NeMo 0.959, kili 0.963)
    assert torch.isfinite(gk).all() and cos(gk, g32) > cos(g16, g32) - 0.01, (cos(gk, g32), cos(g16, g32))
    kili.kernels.remove(m)
    assert not hasattr(m, "_kili_tdt_orig")
    assert all("forward" not in lyr.__dict__ and "forward" not in lyr.self_attn.__dict__
               and "forward" not in lyr.conv.__dict__ for lyr in m.encoder.layers)
    assert step()[0] == ref_loss         # remove() restores NeMo exactly
