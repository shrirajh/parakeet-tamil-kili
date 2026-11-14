"""End-to-end CPU runtime on a tiny synthetic export (needs onnx + onnxruntime; skipped otherwise)."""
import json

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from kili.infer.onnx_run import Frontend, NumpyDecoder, OnnxASR, new_state, tdt_greedy  # noqa: E402

V, D, H = 6, 8, 5  # tokens, joint width, LSTM width
DUR = [0, 1, 2, 3, 4]


def _write_export(d, rng):
    """encoder: feats [1, 128, T] -> f = feats^T @ W [1, T, D] (no subsampling), n = flen."""
    W = rng.standard_normal((128, D)).astype(np.float32) * 0.3
    nodes = [helper.make_node("Transpose", ["feats"], ["x"], perm=[0, 2, 1]),
             helper.make_node("MatMul", ["x", "W"], ["f"]),
             helper.make_node("Identity", ["flen"], ["n"])]
    g = helper.make_graph(nodes, "enc",
                          [helper.make_tensor_value_info("feats", TensorProto.FLOAT, [1, 128, "T"]),
                           helper.make_tensor_value_info("flen", TensorProto.INT64, [1])],
                          [helper.make_tensor_value_info("f", TensorProto.FLOAT, [1, "T", D]),
                           helper.make_tensor_value_info("n", TensorProto.INT64, [1])],
                          [numpy_helper.from_array(W, "W")])
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)], ir_version=8), str(d / "encoder.onnx"))
    w = {"embed": rng.standard_normal((V + 1, H)).astype(np.float32), "pred_w": rng.standard_normal((D, H)).astype(np.float32),
         "pred_b": rng.standard_normal(D).astype(np.float32),
         "joint_w": rng.standard_normal((V + 1 + len(DUR), D)).astype(np.float32),
         "joint_b": rng.standard_normal(V + 1 + len(DUR)).astype(np.float32), "blank": np.array(V)}
    w["embed"][V] = 0
    for k in range(2):
        w[f"wih{k}"] = rng.standard_normal((4 * H, H)).astype(np.float32)
        w[f"whh{k}"] = rng.standard_normal((4 * H, H)).astype(np.float32)
        w[f"b{k}"] = rng.standard_normal(4 * H).astype(np.float32)
    np.savez(d / "dec.npz", **w)
    meta = {"pieces": ["▁a", "b", "▁c", "d", "▁e", "f"], "blank": V, "durations": DUR, "max_symbols": 10}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return W, w


def test_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    W, w = _write_export(tmp_path, rng)
    asr = OnnxASR(str(tmp_path), threads=1, precision="fp32")
    x = rng.standard_normal(16000 * 2).astype(np.float32) * 0.1
    feats, n = Frontend()(x)
    f = feats[0].T @ W
    dec = NumpyDecoder(dict(np.load(tmp_path / "dec.npz")))
    st = new_state(dec, V)
    tdt_greedy(dec, f, 0, int(n[0]), st, V, DUR)
    assert asr.decode_offline(x) == st["ids"]
    pieces = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))["pieces"]
    assert asr.transcribe(x) == "".join(pieces[i] for i in st["ids"]).replace("▁", " ").strip()
    assert asr.transcribe(np.zeros(0, np.float32)) == ""
