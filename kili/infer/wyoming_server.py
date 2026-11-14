"""Wyoming speech-to-text server (Home Assistant) on the CPU runtime kili.infer.onnx_run (no PyTorch).

  python -m kili.infer.wyoming_server --dir onnx_dir --uri tcp://0.0.0.0:10300 [--threads 8]

Then add a "Wyoming Protocol" integration in Home Assistant pointing at this host and port. One model serves Tamil,
English and code-switched speech; the language hint from Home Assistant is ignored, and the transcript is tagged
`ta` if it contains Tamil script, else `en`.

Audio is buffered while the person speaks and decoded once at AudioStop (utterance mode). The model is not a
streaming model: chunked encoding needs ~2 s of lookahead to keep offline accuracy, while one full pass returns the
text a median 0.25 s after the end of speech on 8 CPU threads.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import re
import time
from functools import partial

import numpy as np
from wyoming.asr import Transcribe, Transcript
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStart, AudioStop
from wyoming.event import Event
from wyoming.info import AsrModel, AsrProgram, Attribution, Describe, Info
from wyoming.server import AsyncEventHandler, AsyncServer

from kili.infer.onnx_run import SR, OnnxASR

LOG = logging.getLogger("kili.wyoming")
NAME = "parakeet-tamil-kili"
LANGS = ["ta", "en"]
TAMIL = re.compile("[஀-௿]")
MIN_SAMPLES = SR // 10  # < 0.1 s: nothing to decode


def build_info(version: str = "1.0") -> Info:
    att = Attribution(name="shrirajh/parakeet-tamil-kili (fine-tuned from NVIDIA parakeet-tdt-0.6b-v3, CC BY 4.0)",
                      url="https://huggingface.co/shrirajh/parakeet-tamil-kili")
    model = AsrModel(name=NAME, attribution=att, installed=True, version=version, languages=LANGS,
                     description="Tamil / English / code-switching speech recognition, 0.6B, int8 on CPU")
    return Info(asr=[AsrProgram(name=NAME, attribution=att, installed=True, version=version, models=[model],
                                description="parakeet-tamil-kili on ONNX Runtime")])


class Handler(AsyncEventHandler):
    def __init__(self, info: Info, asr: OnnxASR, lock: asyncio.Lock, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.info, self.asr, self.lock = info, asr, lock
        self.conv = AudioChunkConverter(rate=SR, width=2, channels=1)
        self.buf = bytearray()

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.info.event())
            return True
        if Transcribe.is_type(event.type) or AudioStart.is_type(event.type):
            self.buf = bytearray()
            return True
        if AudioChunk.is_type(event.type):
            self.buf += self.conv.convert(AudioChunk.from_event(event)).audio
            return True
        if AudioStop.is_type(event.type):
            audio = np.frombuffer(bytes(self.buf), dtype=np.int16).astype(np.float32) / 32768.0
            self.buf = bytearray()
            t0 = time.perf_counter()
            text = ""
            if len(audio) >= MIN_SAMPLES:
                async with self.lock:  # one decode at a time: the encoder already uses every thread
                    text = await asyncio.get_running_loop().run_in_executor(None, self.asr.transcribe, audio)
            lang = "ta" if TAMIL.search(text) else "en"
            LOG.info("%.1f s audio -> %.2f s [%s] %s", len(audio) / SR, time.perf_counter() - t0, lang, text)
            await self.write_event(Transcript(text=text, language=lang).event())
            return False
        return True


async def serve(a) -> None:
    t0 = time.time()
    asr = OnnxASR(a.dir, a.threads, a.precision, a.decoder)
    asr.transcribe(np.zeros(SR, np.float32))  # warm-up: ONNX Runtime allocates on the first run
    LOG.info("ready on %s (loaded in %.1f s)", a.uri, time.time() - t0)
    await AsyncServer.from_uri(a.uri).run(partial(Handler, build_info(), asr, asyncio.Lock()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", required=True, help="export directory (kili.infer.onnx_export)")
    ap.add_argument("--uri", default="tcp://0.0.0.0:10300")
    ap.add_argument("--threads", type=int, default=None, help="encoder threads (default: ONNX Runtime's choice)")
    ap.add_argument("--precision", choices=["int8", "fp32"], default="int8")
    ap.add_argument("--decoder", choices=["np", "int8"], default="np")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    asyncio.run(serve(a))


if __name__ == "__main__":
    main()
