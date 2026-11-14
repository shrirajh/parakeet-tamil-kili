"""Parity of the numpy runtime with PyTorch / NeMo. Skipped without torch / NeMo; the model tests also need

  KILI_NEMO_MODEL=path/to/model.nemo (or a HF id)   KILI_ONNX_DIR=export dir   KILI_TEST_AUDIO="a.wav b.wav ..."
"""
import os

import numpy as np
import pytest

from kili.infer.onnx_run import Frontend, NumpyDecoder

torch = pytest.importorskip("torch")


def test_numpy_lstm_matches_torch():
    torch.manual_seed(0)
    V, H = 7, 16
    emb, lstm, proj = torch.nn.Embedding(V + 1, H, padding_idx=V), torch.nn.LSTM(H, H, 2), torch.nn.Linear(H, 9)
    w = {"embed": emb.weight, "pred_w": proj.weight, "pred_b": proj.bias,
         "joint_w": torch.zeros(3, 9), "joint_b": torch.zeros(3)}
    for k in range(2):
        w[f"wih{k}"], w[f"whh{k}"] = getattr(lstm, f"weight_ih_l{k}"), getattr(lstm, f"weight_hh_l{k}")
        w[f"b{k}"] = getattr(lstm, f"bias_ih_l{k}") + getattr(lstm, f"bias_hh_l{k}")
    dec = NumpyDecoder({k: v.detach().numpy() for k, v in w.items()})
    g, state = dec.init_state(V)
    st = (torch.zeros(2, 1, H), torch.zeros(2, 1, H))
    with torch.no_grad():
        for label in (V, 3, 1, 5):
            out, st = lstm(emb(torch.tensor([[label]])), st)
            g_ref = proj(out[0, 0]).numpy()
            if label != V:
                g, state = dec.pred(label, state)
            assert np.allclose(g, g_ref, atol=1e-5)
            assert np.allclose(state[0], st[0][:, 0].numpy(), atol=1e-5)


def test_frontend_matches_nemo_preprocessor():
    nemo_asr = pytest.importorskip("nemo.collections.asr")
    pre = nemo_asr.modules.AudioToMelSpectrogramPreprocessor(
        sample_rate=16000, normalize="per_feature", window_size=0.025, window_stride=0.01, window="hann",
        features=128, n_fft=512, log=True, frame_splicing=1, dither=1e-5, pad_to=0, pad_value=0.0).eval()
    fz = pre.featurizer
    fe = Frontend(None, fz.fb.squeeze(0).numpy(), fz.window.numpy())
    rng = np.random.default_rng(0)
    t = np.arange(16000 * 3) / 16000
    for x in (rng.standard_normal(16000 * 3) * 0.05, 0.3 * np.sin(2 * np.pi * 440 * t) + rng.standard_normal(len(t)) * 0.01):
        x = x.astype(np.float32)
        with torch.inference_mode():
            a, an = pre(input_signal=torch.from_numpy(x)[None], length=torch.tensor([len(x)]))
        b, bn = fe(x)
        assert int(an[0]) == int(bn[0]) and tuple(a.shape) == b.shape
        assert np.abs(a.numpy() - b).max() < 2e-3


def _env():
    m, d, audio = (os.environ.get(k) for k in ("KILI_NEMO_MODEL", "KILI_ONNX_DIR", "KILI_TEST_AUDIO"))
    if not m:
        pytest.skip("set KILI_NEMO_MODEL (and KILI_ONNX_DIR, KILI_TEST_AUDIO for the ONNX parity test)")
    return m, d, (audio or "").split()


def test_load_model_attention():
    pytest.importorskip("nemo")
    from kili.infer.transcribe import load_model
    m = load_model(_env()[0], device="cpu")
    assert m.cfg.encoder.self_attention_model == "rel_pos_local_attn"
    assert list(m.cfg.encoder.att_context_size) == [256, 256]
    if not m.cfg.encoder.get("use_bias", True):
        assert all(layer.self_attn.linear_q.bias is None for layer in m.encoder.layers)


def test_onnx_matches_nemo():
    pytest.importorskip("nemo")
    pytest.importorskip("onnxruntime")
    m, d, audio = _env()
    if not d or not audio:
        pytest.skip("set KILI_ONNX_DIR and KILI_TEST_AUDIO")
    from kili.infer.onnx_export import check
    res = check(m, d, audio)
    assert res["feature_max_abs_diff"] < 2e-3
    if "identical_fp32" in res:
        assert res["identical_fp32"] == res["n"]  # fp32 ONNX + numpy decoder = NeMo greedy, string for string
