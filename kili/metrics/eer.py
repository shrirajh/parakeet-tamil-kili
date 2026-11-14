"""Embedding error rate (EER): how far the meaning of a transcript is from the reference.

EER = mean over utterances of (1 - cosine(embed(reference), embed(hypothesis))), reported in %. Close in spirit to
SemDist (Kim et al., 2021). PER says whether the right sounds were heard; EER says whether the meaning came through
(மழை -> மலை, rain -> hill, is a small PER error but a meaning error).

Embedder: google/embeddinggemma-2 (Apache-2.0, `transformers`, text tower) in bf16 on GPU (fp32 on CPU), with the
symmetric sentence-similarity prompt "task: sentence similarity | query: " prepended to every text, max 512 tokens,
mean pooling over the attention mask, L2-normalised (the model's sentence-transformers configuration). Both texts are
folded first (kili.text.fold.fold: loanword script, spoken number words, punctuation); without the fold the embedder
penalises Latin-script loanwords, since it is not script-blind. An empty text is embedded as "-".

Validation: on 5,232 Kathbath sentences rated by 3 blind LLM (Claude) meaning judges as same / minor / wrong (system
names hidden), mean cosine was 0.985 / 0.968 / 0.954, and cosine separates same from wrong with AUC 0.88 (same vs
minor 0.80). The scale is compressed (most sentences sit above 0.95), but it tracks meaning. Run 10, 5 epochs,
fold EER: IndicVoices 2.66, Kathbath 1.79, MILE 1.57, FLEURS ta 2.47, Common Voice ta 2.27, OpenSLR-65 1.37.

  python -m kili.metrics.eer pairs.tsv [--device cuda] [--no-fold]   # TSV ref<TAB>hyp or JSON lines ref / hyp
Needs torch and transformers (and access to the model on the Hugging Face hub, or a local copy via --model).
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

MODEL = "google/embeddinggemma-2"
PROMPT = "task: sentence similarity | query: "


class Embedder:
    def __init__(self, model: str = MODEL, device: str | None = None, batch_size: int = 128):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model, dtype=dtype).to(self.device).eval()
        self.bs = batch_size
        self.cache: dict[str, np.ndarray] = {}

    def __call__(self, texts: list[str]) -> np.ndarray:
        """L2-normalised sentence embeddings, one row per text (cached by text)."""
        torch = self.torch
        need = [t for t in dict.fromkeys(texts) if t not in self.cache]
        with torch.inference_mode():
            for i in range(0, len(need), self.bs):
                chunk = need[i:i + self.bs]
                enc = self.tok([PROMPT + (t or "-") for t in chunk], padding=True, truncation=True, max_length=512,
                               return_tensors="pt").to(self.device)
                out = self.model(**enc)
                h = (out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]).float()
                m = enc["attention_mask"].unsqueeze(-1).float()
                v = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1).cpu().numpy()
                for t, e in zip(chunk, v):
                    self.cache[t] = e
        return np.stack([self.cache[t] for t in texts])


def cosines(refs: list[str], hyps: list[str], embedder: Embedder | None = None, fold: bool = True) -> np.ndarray:
    if fold:
        from kili.text.fold import fold as f
        refs, hyps = [f(t) for t in refs], [f(t) for t in hyps]
    emb = embedder or Embedder()
    return (emb(refs) * emb(hyps)).sum(1)


def eer(refs: list[str], hyps: list[str], embedder: Embedder | None = None, fold: bool = True) -> dict:
    """-> {eer, share_cos_lt_0.9, share_cos_lt_0.8} as fractions (x100 for the % the website shows)."""
    cos = cosines(refs, hyps, embedder, fold)
    return {"eer": float(np.mean(1 - cos)), "share_cos_lt_0.9": float(np.mean(cos < 0.9)),
            "share_cos_lt_0.8": float(np.mean(cos < 0.8))}


def main() -> None:
    from kili.metrics.wer import read_pairs
    ap = argparse.ArgumentParser(description="embedding error rate (1 - cosine of EmbeddingGemma 2 embeddings)")
    ap.add_argument("pairs", help="TSV (ref, hyp) or JSON lines with ref / hyp fields")
    ap.add_argument("--ref-field", default="ref")
    ap.add_argument("--hyp-field", default="hyp")
    ap.add_argument("--model", default=MODEL, help="hub id or local directory of the embedding model")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-fold", action="store_true", help="embed the raw texts (not comparable with reported EER)")
    a = ap.parse_args()
    refs, hyps = read_pairs(a.pairs, a.ref_field, a.hyp_field)
    res = eer(refs, hyps, Embedder(a.model, a.device), fold=not a.no_fold)
    print(json.dumps({"n": len(refs), **{k: round(v, 4) for k, v in res.items()}}))


if __name__ == "__main__":
    sys.exit(main())
