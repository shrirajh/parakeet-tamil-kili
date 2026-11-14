r"""Tamil tokens for parakeet-tdt-0.6b-v3: extend its tokenizer and resize the model without touching English.

  python -m kili.tokenizer.extend --corpus ta_corpus.txt --out ./native [--size 2048] [--english en_sample.txt]
  -> ./native/tokenizer/{tokenizer.model, vocab.txt, tokenizer.vocab}, ./native/base_native.nemo, ./native/report.json

--corpus: Tamil-script transcripts, one per line (the released model used ~800k training transcripts).

1. Train a Tamil-script BPE SentencePiece model on the corpus. BPE, not unigram: pieces appended to v3's BPE model
   must form a self-consistent merge chain.
2. Append its Tamil-script pieces after v3's 8,192, scored below v3's lowest in their own rank order. Latin, digits
   and punctuation keep v3's pieces and ids, so English tokenisation is unchanged; Tamil text segments exactly as
   the Tamil model would. Released model: 2,044 new pieces (4 already existed), vocabulary 8,192 -> 10,236; Tamil
   sentences went from 56.9 tokens (romanised, through v3's pieces) to 27.1, so the joint costs ~0.6x as much.
3. Model surgery (NeMo change_vocabulary rebuilds the prediction-net embedding and the joint output): every old row
   is copied back (tokens, blank, the 5 TDT duration outputs; LSTM and projections untouched), and the blank moves
   from 8,192 to the end. Each new Tamil row is **warm-started from the mean of v3's rows for the piece's ASCII
   romanisation** (kili.tokenizer.romanize): the new piece "க்கு" starts where v3 already kept "kku", a sound-based
   prior instead of random noise. Pieces whose romanisation does not tokenise get the mean row with a -2 output bias.
   Released model: 1,855 of 2,044 warm-started. Right after surgery the model scored identically to stock v3 (to the
   digit) on Svarah and Kathbath: old weights are bit-identical, and the new rows only compete once trained.
   (We did not run a warm-versus-random comparison for this tokenizer.)
4. Checks: English ids identical (built-in sentences + --english), Tamil segmentation identical to the Tamil model,
   decode(encode(x)) == x on mixed text, no <unk> in Tamil; old rows bit-identical after surgery.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import unicodedata

TA_CHARS = re.compile(r"[஀-௿]")


def train_ta_bpe(corpus: str, size: int, prefix: str) -> str:
    import sentencepiece as spm
    spm.SentencePieceTrainer.train(input=corpus, model_prefix=prefix, vocab_size=size, model_type="bpe",
                                   character_coverage=1.0, normalization_rule_name="identity",
                                   input_sentence_size=2_000_000, shuffle_input_sentence=True,
                                   num_threads=os.cpu_count(), minloglevel=2)
    return prefix + ".model"


def merge(v3_proto_bytes: bytes, ta_model: str) -> tuple[bytes, list[str]]:
    """v3's BPE proto + the Tamil model's Tamil-script pieces appended (new ids after v3's), lower scores."""
    from sentencepiece import sentencepiece_model_pb2 as pb
    v3 = pb.ModelProto(); v3.ParseFromString(v3_proto_bytes)
    ta = pb.ModelProto(); ta.ParseFromString(open(ta_model, "rb").read())
    have = {p.piece for p in v3.pieces}
    low = min(p.score for p in v3.pieces)
    added = []
    for p in ta.pieces:  # BPE proto order == merge rank (score descending)
        if p.type != pb.ModelProto.SentencePiece.NORMAL or p.piece in have:
            continue
        if not TA_CHARS.search(p.piece):  # only Tamil-script pieces; Latin / punctuation stay v3's
            continue
        q = v3.pieces.add()
        q.piece, q.score, q.type = p.piece, low - 1.0 - len(added), pb.ModelProto.SentencePiece.NORMAL
        added.append(p.piece)
    return v3.SerializeToString(), added


def write_tokenizer_dir(proto: bytes, d: str) -> str:
    """tokenizer.model + the vocab.txt / tokenizer.vocab files NeMo's BPE tokenizer expects."""
    import sentencepiece as spm
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "tokenizer.model")
    open(path, "wb").write(proto)
    sp = spm.SentencePieceProcessor(model_file=path)
    with open(os.path.join(d, "vocab.txt"), "w", encoding="utf-8") as f:
        for i in range(sp.get_piece_size()):
            f.write(sp.id_to_piece(i) + "\n")
    with open(os.path.join(d, "tokenizer.vocab"), "w", encoding="utf-8") as f:
        for i in range(sp.get_piece_size()):
            f.write(f"{sp.id_to_piece(i)}\t{sp.get_score(i)}\n")
    return path


def check_tokenizer(v3_sp, new_sp, ta_sp, en_texts, ta_texts, mixed) -> dict:
    """English: identical ids. Tamil: identical pieces to the Tamil model. Round trips exact. No <unk> in Tamil."""
    bad_en = [t for t in en_texts if v3_sp.encode(t) != new_sp.encode(t)]
    bad_ta = []
    for t in ta_texts:
        a = new_sp.encode(t, out_type=str)
        b = ta_sp.encode(unicodedata.normalize("NFKC", t), out_type=str)
        if a != b:
            bad_ta.append((t[:60], a[:8], b[:8]))
    bad_rt = [t for t in mixed if new_sp.decode(new_sp.encode(t)) != unicodedata.normalize("NFKC", t).strip()]
    unk = sum(new_sp.encode(t).count(new_sp.unk_id()) for t in ta_texts)
    return {"en_mismatch": len(bad_en), "ta_mismatch": len(bad_ta), "roundtrip_mismatch": len(bad_rt),
            "ta_unk_tokens": unk, "examples": {"ta": bad_ta[:3], "rt": bad_rt[:3]}}


def warm_rows(added: list[str], old_tok, old_V: int) -> list[list[int] | None]:
    """For each new piece, the v3 token ids of its romanisation (None: fall back to the mean row)."""
    from kili.tokenizer.romanize import romanize
    out = []
    for piece in added:
        try:
            rom = romanize(piece.lstrip("▁"))
        except Exception:
            rom = ""
        ids = [i for i in old_tok.text_to_ids(rom) if i < old_V] if rom else []
        out.append(ids or None)
    return out


def surgery(model, tok_dir: str, added: list[str]) -> dict:
    """Grow the vocabulary in place (NeMo change_vocabulary), restoring every old weight; warm-start the new rows."""
    import torch
    old_V, old_tok = model.tokenizer.vocab_size, model.tokenizer
    dec_sd = {k: v.detach().clone() for k, v in model.decoder.state_dict().items()}
    joint_sd = {k: v.detach().clone() for k, v in model.joint.state_dict().items()}
    model.change_vocabulary(new_tokenizer_dir=tok_dir, new_tokenizer_type="bpe")
    V = model.tokenizer.vocab_size
    assert V - old_V == len(added), (V - old_V, len(added))
    emb_key = next(k for k in dec_sd if k.endswith("embed.weight"))
    out_w = next(k for k in joint_sd if k.endswith(f"joint_net.{len(model.joint.joint_net) - 1}.weight"))
    out_b = out_w[: -len("weight")] + "bias"
    new_dec, new_joint = model.decoder.state_dict(), model.joint.state_dict()
    for k, v in dec_sd.items():
        if k != emb_key:
            new_dec[k].copy_(v)
    for k, v in joint_sd.items():
        if k not in (out_w, out_b):
            new_joint[k].copy_(v)
    E_old, E = dec_sd[emb_key], new_dec[emb_key]  # [old_V+1, H] -> [V+1, H]; blank (padding) is the last row
    W_old, W = joint_sd[out_w], new_joint[out_w]  # [old_V+1+D, H] -> [V+1+D, H]
    b_old, b = joint_sd[out_b], new_joint[out_b]
    E[:old_V] = E_old[:old_V]; E[V] = E_old[old_V]
    W[:old_V] = W_old[:old_V]; W[V:] = W_old[old_V:]  # blank + durations keep their order after the tokens
    b[:old_V] = b_old[:old_V]; b[V:] = b_old[old_V:]
    warm = 0
    for j, ids in enumerate(warm_rows(added, old_tok, old_V)):
        if ids:  # mean of the v3 pieces of the romanised spelling this Tamil piece replaces
            E[old_V + j] = E_old[ids].mean(0); W[old_V + j] = W_old[ids].mean(0); b[old_V + j] = b_old[ids].mean()
            warm += 1
        else:
            E[old_V + j] = E_old[:old_V].mean(0); W[old_V + j] = W_old[:old_V].mean(0); b[old_V + j] = b_old[:old_V].mean() - 2.0
    assert torch.equal(model.decoder.state_dict()[emb_key][:old_V], E_old[:old_V])
    assert torch.equal(model.joint.state_dict()[out_w][V:], W_old[old_V:])
    return {"old_V": old_V, "V": V, "added": len(added), "warm_started": warm, "blank": model.decoder.blank_idx}


EN_CHECK = ["Hello, how are you doing today?", "The price mentioned for National Gardens seeds is not correct.",
            "I'll call you at 5 p.m. tomorrow, OK?", "Mr. Sharma's flight AI-302 lands at 10:45."]
MIXED_CHECK = ["நாளைக்கு meeting எத்தனை மணிக்கு?", "இந்த app ரொம்ப slow-ஆ இருக்கு.", "நான் free-ஆ இருக்கேன், OK?"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--corpus", required=True, help="Tamil-script text, one sentence per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=2048, help="Tamil BPE vocabulary size (released: 2048)")
    ap.add_argument("--base", default="nvidia/parakeet-tdt-0.6b-v3")
    ap.add_argument("--english", default=None, help="extra English sentences (one per line) for the identical-ids check")
    a = ap.parse_args()
    import sentencepiece as spm
    from nemo.collections.asr.models import ASRModel

    os.makedirs(a.out, exist_ok=True)
    ta_model = train_ta_bpe(a.corpus, a.size, os.path.join(a.out, f"ta_bpe_{a.size}"))
    model = ASRModel.from_pretrained(a.base)
    v3_sp = model.tokenizer.tokenizer
    proto, added = merge(v3_sp.serialized_model_proto(), ta_model)
    path = write_tokenizer_dir(proto, os.path.join(a.out, "tokenizer"))
    new_sp, ta_sp = spm.SentencePieceProcessor(model_file=path), spm.SentencePieceProcessor(model_file=ta_model)
    lines = [l.strip() for l in open(a.corpus, encoding="utf-8") if l.strip()]
    ta_texts = random.Random(0).sample(lines, min(500, len(lines)))
    en_texts = EN_CHECK + ([l.strip() for l in open(a.english, encoding="utf-8") if l.strip()] if a.english else [])
    rep = {"tamil_pieces_added": len(added), "tokenizer": check_tokenizer(v3_sp, new_sp, ta_sp, en_texts, ta_texts, MIXED_CHECK)}
    print("[tokenizer]", json.dumps(rep, ensure_ascii=False), flush=True)
    rep["surgery"] = surgery(model, os.path.join(a.out, "tokenizer"), added)
    print("[surgery]", json.dumps(rep["surgery"]), flush=True)
    model.save_to(os.path.join(a.out, "base_native.nemo"))
    json.dump(rep, open(os.path.join(a.out, "report.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
