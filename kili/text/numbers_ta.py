"""Tamil numbers in text -> Tamil number words (standard written forms), so training labels and eval references /
hypotheses use one convention (spoken form, never digits).

  verbalize_ta("2017ஆம் ஆண்டு 15 பேர், 2.5 சதவீதம் 45%")
    -> "இரண்டாயிரத்து பதினேழாம் ஆண்டு பதினைந்து பேர், இரண்டு புள்ளி ஐந்து சதவீதம் நாற்பத்தைந்து சதவீதம்"

Cardinals 0 .. 99,99,99,999 (Indian grouping: ஆயிரம், இலட்சம், கோடி), decimals ("புள்ளி" + digits), "%" ->
"சதவீதம்", Tamil digits (௦-௯) as well as ASCII; thousand separators are dropped. A suffix after the number joins the
last number word ("1994ல்" -> "... தொண்ணூற்றுநான்கில்", "2010 இல்" -> "இரண்டாயிரத்து பத்தில்"). A digit run it cannot
read (a code glued to letters such as "4GB" / "கோவிட்19", a 10+ digit string) -> None: the caller drops the row.

fold_number_words() maps spoken (colloquial) number words to written ones (அஞ்சு -> ஐந்து); the fold metrics use it so
the register of a number is not counted as an error.

  python -m kili.text.numbers_ta "1994ல் 2010 இல்"
"""
from __future__ import annotations

import re
import sys

UNITS = ["பூஜ்ஜியம்", "ஒன்று", "இரண்டு", "மூன்று", "நான்கு", "ஐந்து", "ஆறு", "ஏழு", "எட்டு", "ஒன்பது"]
TEENS = ["பத்து", "பதினொன்று", "பன்னிரண்டு", "பதின்மூன்று", "பதினான்கு", "பதினைந்து", "பதினாறு", "பதினேழு",
         "பதினெட்டு", "பத்தொன்பது"]
TENS = {2: "இருபது", 3: "முப்பது", 4: "நாற்பது", 5: "ஐம்பது", 6: "அறுபது", 7: "எழுபது", 8: "எண்பது", 9: "தொண்ணூறு"}
TENS_JOIN = {2: ("இருபத்", "த"), 3: ("முப்பத்", "த"), 4: ("நாற்பத்", "த"), 5: ("ஐம்பத்", "த"), 6: ("அறுபத்", "த"),
             7: ("எழுபத்", "த"), 8: ("எண்பத்", "த"), 9: ("தொண்ணூற்", "ற")}
HUNDREDS = {1: "நூறு", 2: "இருநூறு", 3: "முந்நூறு", 4: "நானூறு", 5: "ஐந்நூறு", 6: "அறுநூறு", 7: "எழுநூறு",
            8: "எண்ணூறு", 9: "தொள்ளாயிரம்"}
HUNDREDS_JOIN = {1: "நூற்று", 2: "இருநூற்று", 3: "முந்நூற்று", 4: "நானூற்று", 5: "ஐந்நூற்று", 6: "அறுநூற்று",
                 7: "எழுநூற்று", 8: "எண்ணூற்று", 9: "தொள்ளாயிரத்து"}
VOWEL_SIGN = {"அ": "", "ஆ": "ா", "இ": "ி", "ஈ": "ீ", "உ": "ு", "ஊ": "ூ", "எ": "ெ", "ஏ": "ே", "ஐ": "ை", "ஒ": "ொ",
              "ஓ": "ோ", "ஔ": "ௌ"}
VIRAMA = "்"


def _join_vowel(stem_cons: str, word: str) -> str:
    """'த' + 'ஒன்று' -> 'தொன்று' (the consonant takes the word's initial vowel)."""
    return stem_cons + VOWEL_SIGN[word[0]] + word[1:]


def _below_100(n: int) -> str:
    if n < 10:
        return UNITS[n]
    if n < 20:
        return TEENS[n - 10]
    t, u = divmod(n, 10)
    if u == 0:
        return TENS[t]
    stem, c = TENS_JOIN[t]
    w = UNITS[u]
    return stem + (_join_vowel(c, w) if w[0] in VOWEL_SIGN else c + "ு" + w)


def _below_1000(n: int) -> str:
    h, r = divmod(n, 100)
    if h == 0:
        return _below_100(r)
    return HUNDREDS[h] if r == 0 else HUNDREDS_JOIN[h] + " " + _below_100(r)


def _scale(n: int, word: str, join: str) -> tuple[str, str]:
    """n x scale word: (final form, form when more follows). 1 x ஆயிரம் = ஆயிரம்; 2 x = இரண்டாயிரம்."""
    if n == 1:
        return word, join
    head = cardinal(n)
    # இரண்டு + ஆயிரம் -> இரண்டாயிரம்; இலட்சம் stays a separate word (இரண்டு இலட்சம், not இரண்டிலட்சம்)
    if word == "ஆயிரம்" and head.endswith("ு"):
        base = head[:-1]
        return base + VOWEL_SIGN[word[0]] + word[1:], base + VOWEL_SIGN[join[0]] + join[1:]
    return head + " " + word, head + " " + join


def cardinal(n: int) -> str:
    if n < 1000:
        return _below_1000(n)
    for div, word, join in ((10 ** 7, "கோடி", "கோடியே"), (10 ** 5, "இலட்சம்", "இலட்சத்து"), (1000, "ஆயிரம்", "ஆயிரத்து")):
        if n >= div:
            q, r = divmod(n, div)
            final, joined = _scale(q, word, join)
            return final if r == 0 else joined + " " + cardinal(r)
    raise ValueError(n)


def _digits_only(s: str) -> str:
    return " ".join(UNITS[int(d)] for d in s)


_TA_DIGITS = str.maketrans("௦௧௨௩௪௫௬௭௮௯", "0123456789")
_TA = "஀-௿"  # the Tamil block, incl. vowel signs and virama (Python's \w does NOT match those: Mc/Mn)
# A number is a digit run not glued to a word on its left (Latin, Tamil letter or vowel sign / virama, digit, "." or a
# currency sign) nor to Latin letters on its right: those are codes ("சிஜி4684", "CG4684", "4GB", "கோவிட்19", "US$30"),
# left as digits, so verbalize_ta returns None and the row is not trained on.
# A Tamil remainder glued on the right ("1994ல்", "2005ஆம்", "1990களில்") is the number's suffix (group sfx); so is a
# suffix-only token after one space ("2010 இல்", "1930 களில்", "60 வது"; group ssfx).
_SPACED_SUFFIX = ("ஆம்", "ஆவது", "வது", "இல்", "இன்", "ல்", "ல", "இலிருந்து", "லிருந்து")
_NUM = re.compile(
    rf"(?<![\w.$£€₹{_TA}])(\d{{1,3}}(?:,\d{{2,3}})+|\d+)(?:\.(\d+))?(%)?(?![A-Za-z\d])"
    rf"(?:(?P<sfx>[{_TA}]+)|(?P<ssfx> (?:{'|'.join(_SPACED_SUFFIX)}|[கசதபலளநன]்[{_TA}]*|கள[{_TA}]*))(?![{_TA}])"
    rf"|(?P<kilo> ஆயிர[{_TA}]*))?")
# (a spaced "ம்" is not taken as the ordinal: in conversational transcripts it is also the filler "hmm")
# glued remainders that are case / ordinal / plural / clitic suffixes attach to the number word; anything else after
# the digits (units like "90கிகி") is written as a separate word
_SUFFIX = re.compile(r"^(?:ஆ|இ|ஈ|உ|ஊ|எ|ஏ|ஐ|ஒ|ஓ|ம்$|வது|வதாக|ல|க்க|க்கு|கள|த்த|ற்க|டன்|போது)")
_ORDINAL = ("ஆம்", "ஆவது", "ஆவதாக", "ஆவதும்", "ஆவதை", "ஆவதில்")


def _canon_suffix(s: str) -> str:
    """Abbreviated suffixes as written after digits -> the suffix read aloud: 1994ல் = ...நான்கில் (-இல்),
    5ம் = ஐந்தாம் (ordinal -ஆம்), 60வது = அறுபதாவது (-ஆவது)."""
    if s == "ல்":
        return "இல்"
    if s.startswith(("லிருந்து", "லும்")):
        return "இ" + s
    if s == "ம்":
        return "ஆம்"
    if s.startswith("வத"):
        return "ஆ" + s
    return s


def attach_suffix(words: str, suf: str) -> str:
    """Number words + a Tamil suffix, joined as written Tamil joins them (the last word takes the suffix):
    நான்கு+இல் = நான்கில், ஆயிரம்+இல் = ஆயிரத்தில், ஆயிரம்+க்கு = ஆயிரத்துக்கு, நூறு+இல் = நூற்றில்,
    நூறு+ஆவது = நூறாவது, கோடி+இல் = கோடியில், எழுபது+களில் = எழுபதுகளில். Unknown remainders stay a separate word."""
    if not suf:
        return words
    if not _SUFFIX.match(suf):
        return words + " " + suf
    suf = _canon_suffix(suf)
    head, _, w = words.rpartition(" ")
    if suf[0] in VOWEL_SIGN:  # vowel-initial: the vowel merges into the word's last consonant
        sign = VOWEL_SIGN[suf[0]] + suf[1:]
        if w.endswith("ு"):
            if w.endswith(("நூறு", "ணூறு")) and not suf.startswith(_ORDINAL):  # நூற்றில், தொண்ணூற்றில்
                w = w[:-2] + "ற்ற" + sign
            else:
                w = w[:-1] + sign
        elif w.endswith("ம்"):  # ஆயிரத்தில் (oblique -த்த-); ordinal ஆயிரமாவது
            w = w[:-1] + sign if suf.startswith(_ORDINAL) else w[:-2] + "த்த" + sign
        elif w.endswith(("ி", "ை")):  # கோடியில்
            w = w + "ய" + sign
        else:
            w = w + suf
    elif w.endswith("ம்"):
        if suf.startswith("த்த"):  # ஆயிரத்தில் written as 1000த்தில்
            w = w[:-2] + suf
        elif suf.startswith("கள"):  # ஆயிரங்கள்
            w = w[:-2] + "ங" + VIRAMA + suf
        else:  # ஆயிரத்துக்கு, ஆயிரத்துல
            w = w[:-2] + "த்து" + suf
    else:
        w = w + suf
    return (head + " " if head else "") + w


def _one(m: re.Match) -> str:
    whole, frac, pct = m.group(1).replace(",", ""), m.group(2), m.group(3)
    if len(whole) > 9:
        raise ValueError("too long")
    out = _digits_only(whole) if len(whole) > 1 and whole.startswith("0") else cardinal(int(whole))
    if frac:
        out += " புள்ளி " + _digits_only(frac)
    if pct:
        out += " சதவீதம்"
    return out


def verbalize_ta(text: str, spoken: bool = False) -> str | None:
    """Tamil number words for every digit run (written forms; spoken=True: colloquial forms); None if a digit run
    can't be read (a code glued to letters, a 10+ digit string)."""
    t = text.translate(_TA_DIGITS)
    if not re.search(r"\d", t):
        return text

    def rep(m: re.Match) -> str:
        w = _one(m)
        if spoken:
            w = spoken_form(w)
        kilo = (m.group("kilo") or "").strip()
        if kilo:  # "3 ஆயிரத்தில்" -> மூன்றாயிரத்தில் (as cardinal(3000) writes மூன்றாயிரம்); 1 ஆயிரம் -> ஆயிரம்
            if m.group(2) or m.group(3) or not w.endswith("ு"):
                return w + " " + kilo
            return kilo if w in ("ஒன்று", "ஒண்ணு") else w[:-1] + "ா" + kilo[1:]
        sfx = m.group("sfx") or (m.group("ssfx") or "").strip()
        # "இரண்டு 2 வகைப்படும்" (some references write the word and the digit): the digit repeats the word before it
        before = " " + t[:m.start()]
        if not sfx and before.endswith(" " + w + " "):
            return "\0"
        return attach_suffix(w, sfx)

    try:
        out = _NUM.sub(rep, t)
    except ValueError:
        return None
    if re.search(r"\d", out):
        return None
    return re.sub(" ?\0", "", out)


# ---- spoken (colloquial) vs written number words -------------------------------------------------------------------
# 5 = ஐந்து (written) / அஞ்சு (spoken). The fold metrics treat the two registers as the same number.
SPOKEN = {  # spoken form -> written form (whole words; compounds handled by the endings below)
    "ஒண்ணு": "ஒன்று", "ரெண்டு": "இரண்டு", "மூணு": "மூன்று", "நாலு": "நான்கு", "அஞ்சு": "ஐந்து", "ஒம்பது": "ஒன்பது",
    "பதினொண்ணு": "பதினொன்று", "பன்னெண்டு": "பன்னிரண்டு", "பன்னண்டு": "பன்னிரண்டு", "பதிமூணு": "பதின்மூன்று",
    "பதினாலு": "பதினான்கு", "பதினஞ்சு": "பதினைந்து", "பத்தொம்பது": "பத்தொன்பது", "இருவது": "இருபது",
    "நாப்பது": "நாற்பது", "அம்பது": "ஐம்பது", "அறுவது": "அறுபது", "எழுவது": "எழுபது", "எம்பது": "எண்பது",
    "எரநூறு": "இருநூறு", "எறநூறு": "இருநூறு", "எரநூத்தி": "இருநூற்று", "முன்னூறு": "முந்நூறு", "அஞ்ஞூறு": "ஐந்நூறு",
    "அறநூறு": "அறுநூறு", "நூத்தி": "நூற்று", "ஆயிரத்தி": "ஆயிரத்து", "தொள்ளாயிரத்தி": "தொள்ளாயிரத்து",
    "ரெண்டாயிரம்": "இரண்டாயிரம்", "ரெண்டாயிரத்தி": "இரண்டாயிரத்து", "லட்சம்": "இலட்சம்", "லட்சத்தி": "இலட்சத்து",
}
_SPOKEN_TENS = [("ாயிரத்தி", "ாயிரத்து"), ("ரெண்டாயிர", "இரண்டாயிர"), ("அம்பத்", "ஐம்பத்"), ("நாப்பத்", "நாற்பத்"),
                ("எம்பத்", "எண்பத்"), ("அறுவத்", "அறுபத்"), ("எழுவத்", "எழுபத்"), ("இருவத்", "இருபத்")]
_SPOKEN_ENDINGS = [("த்தஞ்சு", "த்தைந்து"), ("த்திரெண்டு", "த்திரண்டு"), ("த்துமூணு", "த்துமூன்று"),
                   ("த்துநாலு", "த்துநான்கு"), ("த்தொம்பது", "த்தொன்பது"), ("த்தொண்ணு", "த்தொன்று"),
                   ("த்தி ", "த்து ")]
_STEM = re.compile("(?<![஀-௿])(" + "|".join(sorted(map(re.escape, SPOKEN), key=len, reverse=True)) + ")")


def fold_number_words(text: str) -> str:
    """Spoken number words -> written ones (also at the start of a suffixed word: நாலுல -> நான்குல)."""
    t = _STEM.sub(lambda m: SPOKEN[m.group(1)], text)
    for a, b in _SPOKEN_ENDINGS[:-1] + _SPOKEN_TENS:
        t = t.replace(a, b)
    return t


_WRITTEN_TO_SPOKEN = {"ஒன்று": "ஒண்ணு", "இரண்டு": "ரெண்டு", "மூன்று": "மூணு", "நான்கு": "நாலு", "ஐந்து": "அஞ்சு",
                      "ஒன்பது": "ஒம்பது", "பதினொன்று": "பதினொண்ணு", "பன்னிரண்டு": "பன்னெண்டு",
                      "பதின்மூன்று": "பதிமூணு", "பதினான்கு": "பதினாலு", "பதினைந்து": "பதினஞ்சு",
                      "பத்தொன்பது": "பத்தொம்பது", "இருபது": "இருவது", "நாற்பது": "நாப்பது", "ஐம்பது": "அம்பது",
                      "அறுபது": "அறுவது", "எழுபது": "எழுவது", "எண்பது": "எம்பது", "இருநூறு": "எரநூறு",
                      "முந்நூறு": "முன்னூறு", "ஐந்நூறு": "அஞ்ஞூறு", "அறுநூறு": "அறநூறு", "இலட்சம்": "லட்சம்"}
_WRITTEN_ENDINGS = [("த்தைந்து", "த்தஞ்சு"), ("த்திரண்டு", "த்திரெண்டு"), ("த்துமூன்று", "த்துமூணு"),
                    ("த்துநான்கு", "த்துநாலு"), ("த்தொன்பது", "த்தொம்பது"), ("த்தொன்று", "த்தொண்ணு"),
                    ("நூற்று", "நூத்தி"), ("ஆயிரத்து", "ஆயிரத்தி"), ("ாயிரத்து", "ாயிரத்தி"), ("இலட்சத்து", "லட்சத்தி"),
                    ("இரண்டாயிர", "ரெண்டாயிர")]


def spoken_form(written: str) -> str:
    """A written number phrase in its common spoken form: ஐம்பத்தைந்து -> அம்பத்தஞ்சு."""
    t = written
    for a, b in _WRITTEN_ENDINGS + [(w, s_) for s_, w in _SPOKEN_TENS]:
        t = t.replace(a, b)
    return " ".join(_WRITTEN_TO_SPOKEN.get(w, w) for w in t.split(" "))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(verbalize_ta(" ".join(sys.argv[1:])))
