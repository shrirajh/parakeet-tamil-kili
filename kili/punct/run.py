"""Punctuate training labels with Gemma 4 12B-it on a vLLM server (prompts: kili.punct.prompts), then write the
punctuated manifests.

Server (vLLM 0.32.0, one replica per GPU; fp8 weights as used for the labels; 16k context):

  vllm serve google/gemma-4-12B-it --revision 43c9d014b231474fa7212dbb48a762566c2b255d --served-model-name gemma12 \
      --data-parallel-size <n_gpus> --max-model-len 16384 --quantization fp8_per_tensor --kv-cache-dtype fp8 \
      --gpu-memory-utilization 0.95 --max-num-seqs 1024 --max-num-batched-tokens 65536 \
      --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 \
      --limit-mm-per-prompt '{"image": 0, "audio": 0, "video": 0}'

Client:

  python -m kili.punct.run marks --url http://127.0.0.1:8000/v1 --manifests ./data/manifests/indicvoices \
      --out ./data/punct/indicvoices.jsonl
  python -m kili.punct.run apply --marks ./data/punct/indicvoices.jsonl --manifests ./data/manifests/indicvoices \
      --out ./data/manifests_punct/indicvoices

`marks`: one request per unique label text, results appended as they land ({k, text, marks} or {k, text, err});
resumable (texts already done are skipped, failed ones retried on the next run). Keep a few hundred requests in
flight per replica (--workers); for many GPUs run several client processes with --shard i/n. Throughput reference:
all YouTube labels in 13.5 min on 8 GPUs.
`apply`: `text` <- the punctuated label (only marks added; a row whose result would change a letter is left
unpunctuated and counted), the original kept as `text_unpunct`. Rows without a result stay unpunctuated; coverage is
printed. Rows with `_skipme` pass through.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import http.client
import json
import os
import random
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

from kili.punct.prompts import SERVED_NAME, apply, dense, request_body


def key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def manifest_files(path: str) -> list[str]:
    return sorted(glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True)) if os.path.isdir(path) else [path]


def unique_texts(paths: list[str]) -> list[str]:
    """Unique `text` of non-skipped manifest rows, first-seen order."""
    seen, out = set(), []
    for p in paths:
        for f in manifest_files(p):
            for line in open(f, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                t = (r.get("text") or "").strip()
                if t and not r.get("_skipme") and t not in seen:
                    seen.add(t); out.append(t)
    return out


def call(url: str, body: dict, timeout: float = 600) -> dict:
    """POST /chat/completions; the forced tool call's arguments, or {} if the reply has none."""
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    try:
        return json.loads(r["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    except (KeyError, IndexError, TypeError, ValueError):
        return {}


def one(url: str, served: str, text: str) -> dict:
    out = {"k": key(text), "text": text}
    for attempt in range(8):  # connection-level failures: retry with jittered backoff
        try:
            out["marks"] = dense(call(url, request_body(text, served)).get("marks"), len(text.split()))
            out.pop("err", None)
            return out
        except (ConnectionError, urllib.error.URLError, http.client.HTTPException) as e:
            out["err"] = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(min(30.0, 0.5 * 2 ** attempt) * (0.5 + random.random()))
        except Exception as e:  # noqa: BLE001 - recorded, retried on the next run
            out["err"] = f"{type(e).__name__}: {str(e)[:200]}"
            return out
    return out


def load_results(paths: list[str]) -> dict[str, dict]:
    res = {}
    for p in paths:
        if os.path.exists(p):
            for line in open(p, encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    if "err" not in r:
                        res[r["k"]] = r
    return res


def run_marks(a) -> None:
    texts = unique_texts(a.manifests)
    si, sn = (int(v) for v in a.shard.split("/"))
    if sn > 1:
        texts = [t for t in texts if int(key(t), 16) % sn == si]
    if a.limit:
        texts = texts[:a.limit]
    done = load_results([a.out])
    todo = [t for t in texts if key(t) not in done]
    print(f"[punct] {len(texts)} unique texts, {len(texts) - len(todo)} done, {len(todo)} to go", flush=True)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    t0, n_err, lock = time.time(), 0, threading.Lock()
    with open(a.out, "a", encoding="utf-8") as f, ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(one, a.url, a.served, t) for t in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            with lock:
                f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            n_err += "err" in r
            if i % 2000 == 0 or i == len(todo):
                rate = i / max(time.time() - t0, 1e-9)
                print(f"[punct] {i}/{len(todo)} ({n_err} errors) {rate:.1f} texts/s", flush=True)


def run_apply(a) -> None:
    res = load_results(a.marks)
    n = n_punct = n_reject = 0
    for src in manifest_files(a.manifests):
        rel = os.path.relpath(src, a.manifests) if os.path.isdir(a.manifests) else os.path.basename(src)
        dst = os.path.join(a.out, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "w", encoding="utf-8") as f:
            for line in open(src, encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                t = (r.get("text") or "").strip()
                if r.get("_skipme") or not t:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    continue
                n += 1
                c = res.get(key(t))
                if c:
                    try:
                        r["text"], r["text_unpunct"] = apply(t, [{"after": i + 1, "mark": m}
                                                                 for i, m in enumerate(c["marks"]) if m]), t
                        n_punct += 1
                    except ValueError:
                        n_reject += 1
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[punct] {n} rows, {n_punct} punctuated ({100 * n_punct / max(n, 1):.1f}%), {n_reject} rejected", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("marks", help="query the server for every unique label text")
    m.add_argument("--url", default="http://127.0.0.1:8000/v1")
    m.add_argument("--served", default=SERVED_NAME)
    m.add_argument("--manifests", nargs="+", required=True, help="JSONL manifests (files or directories)")
    m.add_argument("--out", required=True)
    m.add_argument("--workers", type=int, default=256, help="requests in flight")
    m.add_argument("--shard", default="0/1", help="i/n: only texts whose key hashes to shard i of n")
    m.add_argument("--limit", type=int, default=0, help="first N texts (smoke test)")
    p = sub.add_parser("apply", help="write punctuated manifests")
    p.add_argument("--marks", nargs="+", required=True, help="outputs of `marks`")
    p.add_argument("--manifests", required=True, help="JSONL manifest file or directory")
    p.add_argument("--out", required=True, help="output directory (same file layout)")
    a = ap.parse_args()
    run_marks(a) if a.cmd == "marks" else run_apply(a)


if __name__ == "__main__":
    main()
