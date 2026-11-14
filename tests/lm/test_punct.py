"""Punctuation prompts (kili.punct): prompt layout, mark parsing and the no-letter-change guarantee."""
import json

import pytest

from kili.punct.prompts import (SHOTS, SYSTEM, apply, dense, marks_of, messages, numbered, punctuate, request_body,
                                strip_added)


def test_numbered():
    assert numbered("அ ஆ இ") == "N = 3\n1. அ\n2. ஆ\n3. இ"


def test_shots_are_word_aligned():
    assert len(SHOTS) == 12
    for src, ref in SHOTS:
        words, marks = marks_of(ref)
        assert words == src.split()
        assert punctuate(words, marks) == ref


def test_messages_layout():
    msgs = messages("சரி நான் வரேன்")
    assert msgs[0] == {"role": "system", "content": SYSTEM}
    assert len(msgs) == 1 + 3 * len(SHOTS) + 1
    assert msgs[-1] == {"role": "user", "content": "N = 3\n1. சரி\n2. நான்\n3. வரேன்"}
    call = msgs[5]["tool_calls"][0]["function"]  # second example: "அடுத்தது, ... இருக்கணும். ..."
    assert call["name"] == "submit_marks"
    assert json.loads(call["arguments"]) == {"marks": [{"after": 1, "mark": ","}, {"after": 5, "mark": "."}]}
    body = request_body("சரி")
    assert body["temperature"] == 0 and body["tool_choice"]["function"]["name"] == "submit_marks"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_dense_ignores_bad_items():
    marks = [{"after": 1, "mark": ","}, {"after": 3, "mark": "?"}, {"after": 9, "mark": "."},
             {"after": 2, "mark": ";"}, {"mark": "."}, {"after": "x", "mark": "."}]
    assert dense(marks, 3) == [",", "", "?"]
    assert dense(None, 2) == ["", ""]


def test_apply_and_no_double_marks():
    assert apply("சரி நான் வரேன்", [{"after": 1, "mark": ","}, {"after": 3, "mark": "."}]) == "சரி, நான் வரேன்."
    assert apply("சரி. நான் வரேன்", [{"after": 1, "mark": "?"}]) == "சரி. நான் வரேன்"


def test_strip_added_rejects_changed_letters(monkeypatch):
    assert strip_added("சரி நான் வரேன்", "சரி, நான் வரேன்?")
    assert not strip_added("சரி நான் வரேன்", "சரி, நான் வருகிறேன்.")
    assert not strip_added("சரி நான் வரேன்", "சரி நான்")
    import kili.punct.prompts as P
    monkeypatch.setattr(P, "punctuate", lambda words, marks: " ".join(reversed(words)))
    with pytest.raises(ValueError):
        P.apply("அ ஆ", [])
