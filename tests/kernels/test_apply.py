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
    m.loss._loss.omega = 0.0
    for mod in m.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0
    g = torch.Generator().manual_seed(0)
    wav = (0.05 * torch.randn(3, 16000 * 12, generator=g)).cuda()
    wav_len = torch.tensor([16000 * 12, 16000 * 9, 16000 * 4], device="cuda")
    tlen = torch.tensor([40, 25, 9], device="cuda")
    targets = torch.randint(0, m.loss._loss.blank, (3, 40), generator=g).cuda()

    def step():
        m.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            enc, el = m(input_signal=wav, input_signal_length=wav_len)
            dec, _, _ = m.decoder(targets=targets, target_length=tlen)
            j = m.joint(encoder_outputs=enc, decoder_outputs=dec)
            loss = m.loss(log_probs=j, targets=targets, input_lengths=el, target_lengths=tlen)
        loss.backward()
        gn = torch.stack([p.grad.float().norm() for p in m.encoder.parameters() if p.grad is not None]).norm()
        return float(loss), float(gn)

    ref_loss, ref_gn = step()
    out = kili.kernels.apply(m, att_context=None)
    n = len(m.encoder.layers)
    assert out == {"layer": n, "attention": n, "conv": n, "tdt_loss": 1}
    loss, gn = step()
    assert abs(loss - ref_loss) / abs(ref_loss) < 1e-2, (loss, ref_loss)
    assert abs(gn - ref_gn) / ref_gn < 5e-2 and torch.isfinite(torch.tensor(gn)), (gn, ref_gn)
    kili.kernels.remove(m)
    assert not hasattr(m, "_kili_tdt_orig")
    assert all("forward" not in lyr.__dict__ and "forward" not in lyr.self_attn.__dict__
               and "forward" not in lyr.conv.__dict__ for lyr in m.encoder.layers)
    loss2, _ = step()
    assert abs(loss2 - ref_loss) / abs(ref_loss) < 1e-3
