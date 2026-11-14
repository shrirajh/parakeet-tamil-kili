"""Batch speech synthesis with Maya2 random voices (kili.synth.voices) into NeMo tarred shards, and the ASR round-trip
filter that drops bad clips. Run 10's 525 h of synthetic Tanglish (four text sets) were made this way.

  python -m kili.synth.synth synth --text sentences.jsonl --source synth_tanglish --lang ta --out ./data
      -> ./data/shards/<source>/audio_<n>.tar (16 kHz Opus ~31 kbps), ./data/manifests/<source>/manifest_<n>.jsonl,
         ./data/manifests/<source>/voices_<n>.json (every voice's seed, a and full vector)
  python -m kili.synth.synth filter --data ./data --source synth_tanglish --model kili.nemo

Text rows: {"tts": what is spoken, "label": the training transcript} (or {"text"}, or a .txt file, one sentence per
line, label = text). Each shard of 1,000 rows gets 50 new random voices (unit(v + a * 0.47 * d), a ~ U[-1, 1]) that
rotate over its rows; a voice is rebuilt from (source, pass, shard, slot) alone, and each row's sampling seed is
7000 + row + 1,000,003 * pass, so a re-render with --voice-pass N gives the same text new voices. Clips not ended by
Maya2's audio end-of-speech head (truncated or runaway) and clips outside 0.6-30 s are not written.

filter: transcribe every clip and mark rows `_skipme` (the tar member stays; the manifest row says skip) when the
fold CER between label and transcript exceeds --max-err (0.3). Rows get `asr_err` and `asr_hyp`.

Runtime: maya3 (github.com/MayaResearch/maya2-native) for `synth`; NeMo for `filter`. Pinned weights: see
kili.synth.voices.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from kili.synth.voices import A_MAX, MAYA_REPO, SCALE, load_direction, random_voice, voice_seed

MAYA_REVISION = "87a83efdd6551e038246b6b37836a6264713eec3"


def read_text(path: str) -> list[dict]:
    if path.endswith(".jsonl"):
        rows = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
        return [{"tts": r.get("tts") or r["text"], "label": r.get("label") or r.get("text") or r["tts"]} for r in rows]
    return [{"tts": t, "label": t} for t in (x.strip() for x in open(path, encoding="utf-8")) if t]


def tts_seed(row: int, voice_pass: int) -> int:
    return 7000 + row + 1_000_003 * voice_pass


_RS: dict = {}


def resample(wav: np.ndarray, sr: int, to: int = 16000) -> np.ndarray:
    import torch
    import torchaudio.transforms as T
    if (sr, to) not in _RS:
        _RS[(sr, to)] = T.Resample(sr, to)  # cached sinc kernel
    with torch.no_grad():
        return _RS[(sr, to)](torch.from_numpy(np.asarray(wav, np.float32))).numpy()


def encode_opus(w16: np.ndarray) -> bytes:
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, w16, 16000, format="OGG", subtype="OPUS", compression_level=0.9)
    return buf.getvalue()


def synth(a) -> None:
    from maya3 import Maya3TTS
    rows = read_text(a.text)
    rows = rows[:a.max_rows] if a.max_rows else rows
    sdir, mdir = os.path.join(a.out, "shards", a.source), os.path.join(a.out, "manifests", a.source)
    os.makedirs(sdir, exist_ok=True); os.makedirs(mdir, exist_ok=True)
    d = load_direction(a.direction)
    dsha = hashlib.sha256(np.asarray(d, np.float32).tobytes()).hexdigest()
    tts = Maya3TTS.from_pretrained(MAYA_REPO, max_seats=a.seats)
    tts.generate("வணக்கம்." if a.lang == "ta" else "Hello.", lang=a.lang, voice="Kayal")  # warm-up
    t0, total_s = time.time(), 0.0
    voices: list[dict] = []

    def one(i_r):
        i, r = i_r
        v = voices[i % len(voices)]
        wav, info = tts.generate(r["tts"], lang=a.lang, voice=v["vid"], return_info=True, seed=tts_seed(i, a.voice_pass))
        if not str(info.get("reason", "aeos")).startswith("aeos"):
            return i, r, None, 0.0  # not ended by the audio-EOS head: truncated or runaway
        w16 = resample(wav, 24000)
        dur = len(w16) / 16000
        if dur < 0.6 or dur > 30.0:
            return i, r, None, 0.0
        return i, r, encode_opus(w16), dur

    for sid, s0 in enumerate(range(0, len(rows), a.per_shard)):
        if sid % a.parts != a.part:
            continue
        man, tar = os.path.join(mdir, f"manifest_{sid}.jsonl"), os.path.join(sdir, f"audio_{sid}.tar")
        if os.path.exists(man) and os.path.exists(tar):
            continue  # done (both are written atomically)
        voices.clear()
        side = []
        for k in range(a.voices_per_shard):
            seed = voice_seed(a.source, a.voice_pass, sid, k)
            vec, ra = random_voice(seed, d, a.scale, a.a_max)
            vid = f"r{sid}_{k}"
            tts.register_voice(vid, mode="spk", ecapa=vec)
            voices.append({"vid": vid, "voice_seed": seed, "range_a": round(ra, 6)})
            side.append({"vid": vid, "voice_seed": seed, "range_a": ra, "ecapa": [float(x) for x in vec]})
        with open(os.path.join(mdir, f"voices_{sid}.json"), "w", encoding="utf-8") as f:
            json.dump({"source": a.source, "shard_id": sid, "voice_pass": a.voice_pass, "range_scale": a.scale,
                       "range_max": a.a_max, "direction_sha256": dsha, "maya_revision": MAYA_REVISION,
                       "voices": side}, f)
        chunk = list(enumerate(rows[s0:s0 + a.per_shard], start=s0))
        out_rows = []
        with tarfile.open(tar + ".part", "w") as tf, ThreadPoolExecutor(a.seats) as ex:
            for i, r, data, dur in ex.map(one, chunk):
                if data is None:
                    continue
                name = f"{a.source}_{sid:04d}_{i:07d}.opus"
                ti = tarfile.TarInfo(name); ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
                total_s += dur
                v = voices[i % len(voices)]
                out_rows.append({"audio_filepath": name, "duration": round(dur, 3), "shard_id": sid, "source": a.source,
                                 "lang": a.lang, "text": r["label"], "tts": r["tts"], "synthetic": True,
                                 "voice": v["vid"], "voice_seed": v["voice_seed"], "range_a": v["range_a"],
                                 "tts_seed": tts_seed(i, a.voice_pass), "voice_pass": a.voice_pass})
        with open(man + ".part", "w", encoding="utf-8") as f:
            f.writelines(json.dumps(o, ensure_ascii=False) + "\n" for o in out_rows)
        os.replace(tar + ".part", tar); os.replace(man + ".part", man)
        for v in voices:
            tts.remove_voice(v["vid"])
        el = time.time() - t0
        print(f"[synth] {a.source} shard {sid}: {len(out_rows)}/{len(chunk)} kept, {total_s / 3600:.2f} h audio in "
              f"{el / 60:.1f} min ({total_s / max(el, 1):.1f} audio-s/s)", flush=True)
    tts.close()


def tar_audio(tar: str, names: set[str]):
    import soundfile as sf
    with tarfile.open(tar) as tf:
        for ti in tf:
            if ti.name in names:
                y, sr = sf.read(io.BytesIO(tf.extractfile(ti).read()), dtype="float32")
                assert sr == 16000, (ti.name, sr)
                yield ti.name, y


def round_trip_error(label: str, hyp: str) -> float:
    import jiwer
    from kili.text.fold import fold
    return float(jiwer.cer(fold(label) or "-", fold(hyp) or "-"))


def filter_shards(a) -> None:
    import torch
    from kili.infer.transcribe import load_model
    model = load_model(a.model)
    kept = total = 0
    for man in sorted(glob.glob(os.path.join(a.data, "manifests", a.source, "manifest_*.jsonl"))):
        rows = [json.loads(x) for x in open(man, encoding="utf-8") if x.strip()]
        if not rows or all("asr_err" in r for r in rows):
            kept += sum(not r.get("_skipme") for r in rows); total += len(rows)
            continue  # already filtered
        by = {r["audio_filepath"]: r for r in rows}
        tar = os.path.join(a.data, "shards", a.source, f"audio_{rows[0]['shard_id']}.tar")
        names, audio = zip(*tar_audio(tar, set(by))) if by else ((), ())
        with torch.inference_mode():
            hyps = model.transcribe(list(audio), batch_size=a.batch_size, verbose=False)
        hyps = hyps[0] if isinstance(hyps, tuple) else hyps
        for nm, h in zip(names, hyps):
            h = h.text if hasattr(h, "text") else str(h)
            r = by[nm]
            r["asr_err"], r["asr_hyp"] = round(round_trip_error(r["text"], h), 3), h
            if r["asr_err"] > a.max_err:
                r["_skipme"] = True
        for r in rows:
            if r.setdefault("asr_err", None) is None:
                r["_skipme"] = True  # not decodable
        with open(man, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        k = sum(not r.get("_skipme") for r in rows)
        kept += k; total += len(rows)
        print(f"[filter] {os.path.basename(man)}: kept {k}/{len(rows)}", flush=True)
    print(f"[filter] {a.source}: kept {kept}/{total}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("synth")
    s.add_argument("--text", required=True)
    s.add_argument("--source", required=True)
    s.add_argument("--lang", choices=["ta", "en"], required=True)
    s.add_argument("--out", default="./data")
    s.add_argument("--per-shard", type=int, default=1000)
    s.add_argument("--voices-per-shard", type=int, default=50)
    s.add_argument("--voice-pass", type=int, default=0, help="re-render the same text with new voices and seeds")
    s.add_argument("--seats", type=int, default=64, help="concurrent Maya2 generations (scale with GPU memory)")
    s.add_argument("--part", type=int, default=0, help="this process handles shards with id %% parts == part")
    s.add_argument("--parts", type=int, default=1)
    s.add_argument("--max-rows", type=int, default=0)
    s.add_argument("--direction", default=None, help="voice-range direction (default: the shipped one)")
    s.add_argument("--scale", type=float, default=SCALE)
    s.add_argument("--a-max", type=float, default=A_MAX)
    f = sub.add_parser("filter")
    f.add_argument("--data", default="./data")
    f.add_argument("--source", required=True)
    f.add_argument("--model", required=True, help=".nemo path or Hugging Face id of a Tamil-script ASR model")
    f.add_argument("--max-err", type=float, default=0.3)
    f.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()
    synth(a) if a.cmd == "synth" else filter_shards(a)


if __name__ == "__main__":
    main()
