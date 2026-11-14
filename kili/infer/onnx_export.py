"""Export the NeMo model to ONNX for CPU inference (kili.infer.onnx_run), quantise to int8, check parity with NeMo.

  python -m kili.infer.onnx_export export   --model parakeet-tamil-kili.nemo --dir onnx_dir
  python -m kili.infer.onnx_export quantize --dir onnx_dir
  python -m kili.infer.onnx_export check    --model parakeet-tamil-kili.nemo --dir onnx_dir a.wav b.wav ...

Files written to --dir:
  encoder.onnx (+ .data)       mel [1, 128, T], length [1] -> encoder frames already projected by the joint's
                               encoder layer [1, T', 640], length [1]
  pred.onnx                    one prediction-network step: label [1, 1], h, c [2, 1, 640] -> g [1, 640], h, c
  joint.onnx                   f [1, 640], g [1, 640] -> logits [1, V + 1 + 5] (tokens, blank, TDT durations)
  encoder_int8.onnx (+ .data), pred_int8.onnx, joint_int8.onnx
                               ONNX Runtime dynamic int8 (MatMul / Gemm / LSTM weights per output channel,
                               activations quantised per tensor at run time; conv layers stay fp32)
  dec.npz                      prediction network + joint output layer as fp32 arrays (the numpy decoder)
  frontend.npz                 NeMo's mel filterbank [128, 257] and window [400] (the numpy front end)
  meta.json                    vocabulary pieces, blank id, TDT durations, max symbols per frame, feature config

The encoder is exported with NeMo's dense rel_pos attention restricted to the same +-256-frame band as
rel_pos_local_attn (whose sliding-chunk kernel uses as_strided and does not export): the same maths, identical
output. The band mask is dense, so encoder memory grows with T^2; the positional table covers 5000 frames (400 s).
Sizes for this 0.6B model: encoder fp32 ~2.5 GB, int8 ~0.9 GB (the ~75M conv parameters stay fp32).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

import numpy as np

EXPORT_ATTENTION = ("rel_pos", [256, 256])


def _wrappers(m):
    import torch

    class EncWrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder, self.proj = m.encoder, m.joint.enc

        def forward(self, feats, flen):
            enc, elen = self.encoder(audio_signal=feats, length=flen)
            return self.proj(enc.transpose(1, 2).float()), elen

    class PredWrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dec, self.proj = m.decoder, m.joint.pred

        def forward(self, label, h, c):
            g, (h2, c2) = self.dec.predict(label, (h, c), add_sos=False, batch_size=1)
            return self.proj(g[:, 0]), h2, c2

    class JointWrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = m.joint.joint_net  # ReLU, Dropout, Linear

        def forward(self, f, g):  # 3-D through the Linear so it exports as MatMul (quantisable)
            return self.net((f + g).unsqueeze(1)).squeeze(1)

    return EncWrap().eval(), PredWrap().eval(), JointWrap().eval()


def dec_weights(m) -> dict:
    """Prediction network + joint output layer as fp32 numpy arrays (the numpy decoder's weights)."""
    dec, j = m.decoder, m.joint
    lstm = dec.prediction["dec_rnn"].lstm
    w = {"embed": dec.prediction["embed"].weight, "pred_w": j.pred.weight, "pred_b": j.pred.bias,
         "joint_w": j.joint_net[-1].weight, "joint_b": j.joint_net[-1].bias}
    for k in range(lstm.num_layers):
        w[f"wih{k}"] = getattr(lstm, f"weight_ih_l{k}")
        w[f"whh{k}"] = getattr(lstm, f"weight_hh_l{k}")
        w[f"b{k}"] = getattr(lstm, f"bias_ih_l{k}") + getattr(lstm, f"bias_hh_l{k}")
    w = {k: v.detach().float().cpu().numpy() for k, v in w.items()}
    w["blank"] = np.array(dec.blank_idx)
    return w


def frontend_config(m) -> tuple[dict, dict]:
    """NeMo's featurizer -> (feature config for meta.json, arrays for frontend.npz). Fails loudly on any setting
    the numpy front end does not implement."""
    fz = m.preprocessor.featurizer
    guard = fz.log_zero_guard_value
    unsupported = {"normalize": fz.normalize != "per_feature", "log": not fz.log,
                   "frame_splicing": fz.frame_splicing != 1, "mag_power": float(fz.mag_power) != 2.0,
                   "log_zero_guard_type": fz.log_zero_guard_type != "add", "log_zero_guard_value": isinstance(guard, str),
                   "exact_pad": getattr(fz, "exact_pad", False), "pad_to": fz.pad_to not in (0, None)}
    bad = [k for k, v in unsupported.items() if v]
    assert not bad, f"numpy front end does not implement these preprocessor settings: {bad}"
    cfg = {"sample_rate": int(m.cfg.preprocessor.sample_rate), "n_fft": int(fz.n_fft), "win_length": int(fz.win_length),
           "hop_length": int(fz.hop_length), "n_mels": int(fz.fb.shape[1]), "preemph": float(fz.preemph or 0.0),
           "log_guard": float(guard), "std_eps": 1e-5}
    arrays = {"fb": fz.fb.squeeze(0).float().cpu().numpy(), "window": fz.window.float().cpu().numpy()}
    return cfg, arrays


def export(model_path: str, d: str) -> None:
    import onnx
    import torch

    from kili.infer.transcribe import load_model, set_attention
    torch.set_grad_enabled(False)
    m = load_model(model_path, device="cpu")  # fp32
    set_attention(m, *EXPORT_ATTENTION)
    os.makedirs(d, exist_ok=True)
    enc, pred, joint = _wrappers(m)

    # the fp32 encoder exceeds protobuf's 2 GB: the exporter writes one file per tensor, so export into a scratch
    # directory and re-save with a single external data file
    tmp = os.path.join(d, "_encoder_tmp")
    os.makedirs(tmp, exist_ok=True)
    t = time.time()
    torch.onnx.export(enc, (torch.randn(1, 128, 801), torch.tensor([801])), os.path.join(tmp, "encoder.onnx"),
                      input_names=["feats", "flen"], output_names=["f", "n"], opset_version=17, dynamo=False,
                      dynamic_axes={"feats": {2: "T"}, "f": {1: "Te"}})
    onnx.save(onnx.load(os.path.join(tmp, "encoder.onnx")), os.path.join(d, "encoder.onnx"),
              save_as_external_data=True, all_tensors_to_one_file=True, location="encoder.onnx.data")
    shutil.rmtree(tmp)
    print(f"[export] encoder {time.time() - t:.0f} s", flush=True)

    lstm = m.decoder.prediction["dec_rnn"].lstm
    z = torch.zeros(lstm.num_layers, 1, lstm.hidden_size)
    torch.onnx.export(pred, (torch.tensor([[5]]), z, z), os.path.join(d, "pred.onnx"),
                      input_names=["label", "h", "c"], output_names=["g", "h2", "c2"], opset_version=17, dynamo=False)
    Hj = m.joint.joint_net[-1].in_features
    torch.onnx.export(joint, (torch.zeros(1, Hj), torch.zeros(1, Hj)), os.path.join(d, "joint.onnx"),
                      input_names=["f", "g"], output_names=["logits"], opset_version=17, dynamo=False)

    np.savez(os.path.join(d, "dec.npz"), **dec_weights(m))
    feats, arrays = frontend_config(m)
    np.savez(os.path.join(d, "frontend.npz"), **arrays)
    durations = list(m.cfg.model_defaults.get("tdt_durations", [0, 1, 2, 3, 4]))
    greedy = m.cfg.decoding.get("greedy", {}) or {}
    meta = {"pieces": m.tokenizer.ids_to_tokens(list(range(m.tokenizer.vocab_size))), "blank": int(m.decoder.blank_idx),
            "durations": [int(x) for x in durations], "max_symbols": int(greedy.get("max_symbols", 10) or 10),
            "features": feats, "subsampling": int(m.cfg.encoder.subsampling_factor),
            "attention": {"kind": EXPORT_ATTENTION[0], "context": EXPORT_ATTENTION[1]},
            "source": os.path.basename(model_path.rstrip("/"))}
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False)
    print(f"[export] done: {d}", flush=True)


def quantize(d: str) -> None:
    """ONNX Runtime dynamic int8, weights per output channel."""
    from onnxruntime.quantization import QuantType, quantize_dynamic
    for src, dst, ops, ext in (("encoder.onnx", "encoder_int8.onnx", ["MatMul", "Gemm"], True),
                               ("pred.onnx", "pred_int8.onnx", ["MatMul", "Gemm", "LSTM"], False),
                               ("joint.onnx", "joint_int8.onnx", ["MatMul", "Gemm"], False)):
        t = time.time()
        quantize_dynamic(os.path.join(d, src), os.path.join(d, dst), weight_type=QuantType.QInt8, per_channel=True,
                         use_external_data_format=ext, op_types_to_quantize=ops)
        print(f"[quantize] {dst} {time.time() - t:.0f} s", flush=True)


def check(model_path: str, d: str, wavs: list[str], threads: int | None = None) -> dict:
    """Parity with NeMo on the given files (fp32 on CPU, local attention as trained): max |numpy front end - NeMo
    preprocessor|, and how many ONNX transcripts equal NeMo's for each available encoder (fp32 should match all;
    int8 differs on some clips by design)."""
    import torch

    from kili.infer.onnx_run import OnnxASR, load_audio
    from kili.infer.transcribe import load_model, text_of, transcribe_arrays
    m = load_model(model_path, device="cpu")
    audios = [load_audio(p) for p in wavs]
    ref = [text_of(h) for h in transcribe_arrays(m, audios, batch_size=1)]
    res = {"n": len(audios)}
    for prec in ("fp32", "int8"):
        if not os.path.exists(os.path.join(d, "encoder.onnx" if prec == "fp32" else "encoder_int8.onnx")):
            continue
        asr = OnnxASR(d, threads, prec, "np")
        if "feature_max_abs_diff" not in res:
            worst = 0.0
            for x in audios:
                with torch.inference_mode():
                    a, an = m.preprocessor(input_signal=torch.from_numpy(x)[None], length=torch.tensor([len(x)]))
                b, bn = asr.front(x)
                assert int(an[0]) == int(bn[0]) and tuple(a.shape) == b.shape, (a.shape, b.shape)
                worst = max(worst, float(np.abs(a.numpy() - b).max()))
            res["feature_max_abs_diff"] = worst
        hyps = [asr.transcribe(x) for x in audios]
        res[f"identical_{prec}"] = sum(h == r for h, r in zip(hyps, ref))
        for p, h, r in zip(wavs, hyps, ref):
            if h != r:
                print(f"[check] {prec} differs: {p}\n  nemo: {r}\n  onnx: {h}", flush=True)
    print(f"[check] {json.dumps(res)}", flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--model", required=True, help=".nemo file or Hugging Face id")
    e.add_argument("--dir", required=True)
    e.add_argument("--threads", type=int, default=8)
    q = sub.add_parser("quantize")
    q.add_argument("--dir", required=True)
    c = sub.add_parser("check")
    c.add_argument("--model", required=True)
    c.add_argument("--dir", required=True)
    c.add_argument("--threads", type=int, default=None)
    c.add_argument("audio", nargs="+")
    a = ap.parse_args()
    if a.cmd == "export":
        import torch
        torch.set_num_threads(a.threads)
        export(a.model, a.dir)
    elif a.cmd == "quantize":
        quantize(a.dir)
    else:
        check(a.model, a.dir, a.audio, a.threads)


if __name__ == "__main__":
    main()
