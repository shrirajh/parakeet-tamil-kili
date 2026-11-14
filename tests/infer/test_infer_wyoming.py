"""Wyoming handler with a stub model (needs the wyoming package; skipped otherwise)."""
import asyncio

import numpy as np
import pytest

pytest.importorskip("wyoming")
pytest.importorskip("onnxruntime")
from wyoming.asr import Transcribe, Transcript  # noqa: E402
from wyoming.audio import AudioChunk, AudioStart, AudioStop  # noqa: E402
from wyoming.info import Describe, Info  # noqa: E402

from kili.infer.wyoming_server import Handler, build_info  # noqa: E402


class StubASR:
    def __init__(self):
        self.seen = []

    def transcribe(self, audio):
        self.seen.append(len(audio))
        return "வணக்கம் hello"


def test_info_languages():
    info = build_info()
    assert info.asr[0].models[0].languages == ["ta", "en"]


def test_utterance_round_trip():
    asr, out = StubASR(), []
    h = Handler(build_info(), asr, asyncio.Lock(), None, None)

    async def write_event(event):
        out.append(event)
    h.write_event = write_event

    async def run():
        assert await h.handle_event(Describe().event())
        await h.handle_event(Transcribe(language="ta").event())
        await h.handle_event(AudioStart(rate=16000, width=2, channels=1).event())
        pcm = (np.zeros(16000, np.int16)).tobytes()
        await h.handle_event(AudioChunk(rate=16000, width=2, channels=1, audio=pcm).event())
        assert not await h.handle_event(AudioStop().event())
    asyncio.run(run())
    assert Info.is_type(out[0].type)
    t = Transcript.from_event(out[1])
    assert t.text == "வணக்கம் hello" and t.language == "ta" and asr.seen == [16000]
