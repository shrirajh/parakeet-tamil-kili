"""Label normalisation, number words and the scoring folds.

  python -m pytest tests/text      (or: python tests/text/test_normalise.py)
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from kili.text.fold import detin, fold, phonetic_fold, sandhi_join  # noqa: E402
from kili.text.normalise import norm_en, norm_ta, normalise, normalise_label  # noqa: E402
from kili.text.numbers_en import verbalize  # noqa: E402
from kili.text.numbers_ta import attach_suffix, fold_number_words, spoken_form, verbalize_ta  # noqa: E402

NUMBERS_TA = {
    "ஆயிரத்து தொள்ளாயிரத்து 94ல், அஜர்பைஜானின்": "ஆயிரத்து தொள்ளாயிரத்து தொண்ணூற்றுநான்கில், அஜர்பைஜானின்",
    "1994ல், அஜர்பைஜானின்": "ஆயிரத்து தொள்ளாயிரத்து தொண்ணூற்றுநான்கில், அஜர்பைஜானின்",
    "2010 இல், தேசீயமயமாக்கப்பட்ட": "இரண்டாயிரத்து பத்தில், தேசீயமயமாக்கப்பட்ட",
    "3 ஆயிரத்தில் இருந்து 4 ஆயிரம் வரை": "மூன்றாயிரத்தில் இருந்து நான்காயிரம் வரை",
    "1930 களில்": "ஆயிரத்து தொள்ளாயிரத்து முப்பதுகளில்",
    "2020 க்குள்": "இரண்டாயிரத்து இருபதுக்குள்",
    "2005 ஆம் ஆண்டிலிருந்து": "இரண்டாயிரத்து ஐந்தாம் ஆண்டிலிருந்து",
    "அவை இரண்டு 2 வகைப்படும்": "அவை இரண்டு வகைப்படும்",
    "2017ஆம் ஆண்டு": "இரண்டாயிரத்து பதினேழாம் ஆண்டு",
    "1990களில்": "ஆயிரத்து தொள்ளாயிரத்து தொண்ணூறுகளில்",
    "5ம் நிலை": "ஐந்தாம் நிலை",
    "190வது இடத்தை": "நூற்று தொண்ணூறாவது இடத்தை",
    "60 வது கோல்": "அறுபதாவது கோல்",
    "600 க்கும் மேற்பட்ட": "அறுநூறுக்கும் மேற்பட்ட",
    "1000ல்": "ஆயிரத்தில்", "1000க்கும்": "ஆயிரத்துக்கும்", "100இல் ஒருவர்": "நூற்றில் ஒருவர்",
    "10 ல பாத்தேன்": "பத்துல பாத்தேன்", "45%க்கும்": "நாற்பத்தைந்து சதவீதத்துக்கும்",
    "90கிகி எடை": "தொண்ணூறு கிகி எடை",  # a unit, not a suffix: its own word
    "8. ம், எது": "எட்டு. ம், எது",  # spaced ம் is also the filler "hmm": not an ordinal
    "2.5 சதவீதம் 45%": "இரண்டு புள்ளி ஐந்து சதவீதம் நாற்பத்தைந்து சதவீதம்",
    "3,00,948 சதுர": "மூன்று இலட்சத்து தொள்ளாயிரத்து நாற்பத்தெட்டு சதுர",
    "100000": "இலட்சம்", "2000": "இரண்டாயிரம்", "25000": "இருபத்தைந்தாயிரம்",
    "௨௦௧௭": "இரண்டாயிரத்து பதினேழு",  # Tamil digits
    "சொல்லுங்க": "சொல்லுங்க",
}
# codes / digits glued to letters: not verbalised, the label is dropped
CODES = ["ஃப்ளைட் சிஜி4684 விழுந்ததாக", "கோவிட்19 குறித்து", "A4 size", "4GB RAM", "5G network", "US$30 கட்டணம்",
         "1234567890"]


def test_numbers_ta():
    bad = [(s, verbalize_ta(s), w) for s, w in NUMBERS_TA.items() if verbalize_ta(s) != w]
    assert not bad, "\n".join(f"{s!r} -> {g!r}, want {w!r}" for s, g, w in bad)
    assert all(verbalize_ta(s) is None for s in CODES)


def test_numbers_ta_spoken():
    assert verbalize_ta("2017ஆம் ஆண்டு 5 பேர்", spoken=True) == "ரெண்டாயிரத்தி பதினேழாம் ஆண்டு அஞ்சு பேர்"
    assert spoken_form("ஐம்பத்தைந்து") == "அம்பத்தஞ்சு"
    assert fold_number_words("அஞ்சு பேர் நாலுல") == "ஐந்து பேர் நான்குல"
    assert fold_number_words(spoken_form("இரண்டாயிரத்து ஐம்பத்தைந்து")) == "இரண்டாயிரத்து ஐம்பத்தைந்து"
    assert attach_suffix("நூறு", "ஆவது") == "நூறாவது"
    assert attach_suffix("கோடி", "இல்") == "கோடியில்"
    assert attach_suffix("ஆயிரம்", "கள்") == "ஆயிரங்கள்"


def test_numbers_en():
    assert verbalize("over 2,000 people in 1995 paid $5.50, 20% more on the 3rd") == \
        "over two thousand people in nineteen ninety five paid five dollars fifty, twenty percent more on the third"
    assert verbalize("call 04425 now") == "call zero four four two five now"
    assert verbalize("in 2005") == "in two thousand five"
    assert verbalize("1,23,456") is None  # Indian grouping in English text: not confident


def test_normalise_label():
    assert normalise_label("  நாளைக்கு   மீட்டிங்க்கு 5 மணிக்கு வா.", "ta") == "நாளைக்கு meeting-க்கு ஐந்து மணிக்கு வா."
    assert normalise_label("1994ல் 2010 இல்", "ta") == "ஆயிரத்து தொள்ளாயிரத்து தொண்ணூற்றுநான்கில் இரண்டாயிரத்து பத்தில்"
    assert normalise_label("1994ல் 2010 இல் சிஜி4684", "ta") is None
    assert normalise_label("எனக்கு Meeting இருக்கு, BHK வீடு", "ta") == "எனக்கு meeting இருக்கு, BHK வீடு"
    assert normalise_label("Hello, I paid $5.", "en") == "Hello, I paid five dollars."
    assert normalise_label("Hello World") == "Hello World"  # auto -> en: casing kept
    assert normalise("கோவிட்19 குறித்து") == "கோவிட்19 குறித்து"  # unreadable digits kept, no None


def test_scoring_normalisers():
    assert norm_ta("வணக்கம்,  நண்பா!") == "வணக்கம் நண்பா"  # vowel signs and virama kept
    assert norm_en("Hello, World's END.") == "hello world's end"


def test_fold():
    # loanword script and number register are not errors; punctuation goes
    assert fold("நாளைக்கு மீட்டிங்க்கு அஞ்சு மணிக்கு.") == fold("நாளைக்கு meeting-க்கு ஐந்து மணிக்கு")
    assert fold("நாளைக்கு மீட்டிங்க்கு அஞ்சு மணிக்கு.") == "நாளைக்கு meeting க்கு ஐந்து மணிக்கு"  # hyphen -> space
    assert fold("போன வாரம்") == "போன வாரம்"


def test_phonetic_fold():
    assert sandhi_join("அப்பு சாமி") == sandhi_join("அப்புச்சாமி") == sandhi_join("அப்புச் சாமி")
    assert sandhi_join("கரை உடைப்பை") == sandhi_join("கரையுடைப்பை")
    assert sandhi_join("சீரமைக்கக்") == sandhi_join("சீரமைக்க")
    assert detin("meeting") == "மீட்டிங்"
    assert phonetic_fold("meeting-க்கு") == phonetic_fold("மீட்டிங்க்கு")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
