"""Loanword engine: conversions, false friends that must stay Tamil, and the context rules.

  python -m pytest tests/text      (or: python tests/text/test_loanwords.py)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from kili.text.loanwords import Latinizer, english_to_tamil, latinize  # noqa: E402
from kili.text.loanwords.engine import fmt, splits  # noqa: E402

CONVERT = {
    "நாளைக்கு மீட்டிங்க்கு நீ வர்றியா": "நாளைக்கு meeting-க்கு நீ வர்றியா",
    "ஃபோன்ல பேசு": "phone-ல பேசு",
    "போன்ல பேசு": "phone-ல பேசு",
    "ஆபீஸ்ல இருக்கேன்": "office-ல இருக்கேன்",
    "பாஸ்வேர்ட மாத்து": "password-அ மாத்து",
    "பாஸ்வேர்ட்ட மறந்துட்டேன்": "password-அ மறந்துட்டேன்",
    "ஸ்கூல்ல படிக்கிறான்": "school-ல படிக்கிறான்",
    "ஸ்கூலுக்கு போனான்": "school-க்கு போனான்",
    "ஸ்டார்ட் பண்ணு": "start பண்ணு",
    "ஸ்கூல் தான்": "school-தான்",  # clitic: hyphen, no compound
    "மியூசிக் கேட்டேன்": "music கேட்டேன்",
    "பேசிக் சம்பளம் எவ்வளவு": "basic சம்பளம் எவ்வளவு",  # basic before a non-க word
    "சார் சொன்னாரு": "sir சொன்னாரு",
    "ஓகே சார்": "ok sir",
    # split compounds
    "நம்ம ஸ்டார்ட் அப்புக்கு வர்ற முதலீடு": "நம்ம startup-க்கு வர்ற முதலீடு",
    "ஆன் லைன் வாயிலாக": "online வாயிலாக",
    "பர்த் டே பார்ட்டி": "birthday party",
    "ஃபோன் பே மூலம் அனுப்பு": "PhonePe மூலம் அனுப்பு",  # not phone-ஏ
    "குருப் போட்டோ எடுத்தோம்": "group photo எடுத்தோம்",  # next word a loan: no sandhi block
    # Latin already in the source: suffix tokens join with a hyphen
    "pen க்கு": "pen-க்கு", "penக்கு": "pen-க்கு", "help பு": "help-உ", "jolly யா": "jolly-ஆ", "ஓகே வா": "ok-ஆ",
    "itemsலாம்": "items-லாம்",
}

KEEP_TAMIL = [
    "அவன் வீட்டுக்கு போன",  # போன = went, not phone
    "போன வாரம் கால வலிக்குது நான் தான் வந்தேன்",  # போன = last, கால = leg
    "கால் பண்ணு",  # கால் = leg / quarter as well as call: ambiguous, kept
    "காலையில வந்தேன்",
    "சீனா சாப்ட்டு",
    "சீனா போனார்",  # China
    "கடல் சார் ஆய்வு",  # postposition சார் (related to), not sir
    "இருப்பினும் பலர் உலகம் அழியப் போவதைப் பற்றி பேசிக் கொண்டு தான் இருக்கிறார்கள்",  # பேசி + sandhi க்
    "முப்பத்து ஆறு வயதான டச்சு நடிகர்",
    "அவங்களுக்கு ரொம்ப ஷேஃபு",
    "போலிச் சாமியார் வர்றாங்க",
    "ஆசப் பட்றேன்",
    "குருப் பெயர்ச்சி பலன்கள்",
]

PARTIAL = {
    "டூ டேஸ் லீவ்": "டூ days லீவ்",  # never "today's"
    "ஃபங்க்ஷன் ஸோ நாங்க": "function ஸோ நாங்க",  # an inflection is no compound
    "ரன் சிங் அடிச்சாரு": "run சிங் அடிச்சாரு",  # not running
}


def test_convert():
    bad = [(s, latinize(s), w) for s, w in {**CONVERT, **PARTIAL}.items() if latinize(s) != w]
    assert not bad, "\n".join(f"{s!r} -> {g!r}, want {w!r}" for s, g, w in bad)


def test_false_friends_stay_tamil():
    bad = [(s, latinize(s)) for s in KEEP_TAMIL if latinize(s) != s]
    assert not bad, bad


def test_punctuation_and_idempotence():
    assert latinize("மீட்டிங்க்கு, வா!") == "meeting-க்கு, வா!"
    for s in CONVERT.values():
        assert latinize(s) == s, s


def test_splits_and_fmt():
    assert ("மீட்டிங்", "க்கு") in splits("மீட்டிங்க்கு")
    assert ("பாஸ்வேர்ட்", "அ") in splits("பாஸ்வேர்ட")
    assert fmt("item", "ஸுக்கு") == "items-க்கு"
    assert fmt("meeting", "") == "meeting"


def test_table():
    lat = Latinizer()
    assert len(lat.surf) > 20000 and len(lat.stem) > 2000 and len(lat.nat) > 1000 and len(lat.sandhi) >= 7
    assert english_to_tamil()["meeting"].startswith("மீட்டிங்")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
