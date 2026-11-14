"""Deterministic, invertible ASCII romanisation of Tamil script ("ta-rom").

Used to warm-start the new Tamil tokens (kili.tokenizer.extend): each Tamil piece starts from the parakeet v3
rows of its ASCII romanisation. Design goals:
  * ASCII only: Parakeet v3's tokenizer has no byte pieces, and ISO 15919 dot-below letters are <unk>.
  * Every Tamil letter stays distinct (5 n's, 3 l's, 2 r's, 2 t's) and round-trips exactly.
  * No case distinctions: case is cosmetic and stripped before inversion.
  * One Tamil word -> one romanised word (timestamps and confidence carry back).
  * Readable and Tanglish-like where possible, so letters keep their usual phonetic prior.

Mapping
  vowels   a aa i ii u uu e ee ai o oo au        (independent letter or vowel sign; inherent vowel = a)
  stops    k ng ch ny t nh th n p m              (க ங ச ஞ ட ண த ந/ன ப ம)
  others   y r rh l lh zh v                       (ய ர ற ல ள ழ வ)
  grantha  j sh s h sx                            (ஜ ஷ ஸ ஹ ஶ)
  aytham   q, and f for ஃப (Tamil's spelling of /f/)  (ஃ)
  dead consonant (virama) = bare consonant letters.

ந vs ன: both are written "n" in their usual positions, and the decoder decides from context:
  * "n" at the start of a word, or "n" directly followed by "th", is ந.
  * any other "n" is ன.
  * Exceptions are marked: ந elsewhere = "nd"; ன in a ந position = "nx".

Separator "'" is inserted only where concatenation would be ambiguous:
  * a vowel letter following a vowel (e.g. கஇ -> ka'i, not kai = கை)
  * an independent vowel after a dead consonant (க்அ -> k'a)
  * two consonants whose letters would merge (ட்ஹ -> t'h, not th = த)
"""
from __future__ import annotations

import re
import unicodedata

VIRAMA = "்"
AYTHAM = "ஃ"

CONS = {
    "க": "k", "ங": "ng", "ச": "ch", "ஞ": "ny", "ட": "t", "ண": "nh", "த": "th",
    "ந": "n", "ன": "n",  # contextual, see module docstring
    "ப": "p", "ம": "m", "ய": "y", "ர": "r", "ற": "rh", "ல": "l", "ள": "lh", "ழ": "zh", "வ": "v",
    "ஜ": "j", "ஷ": "sh", "ஸ": "s", "ஹ": "h", "ஶ": "sx",
    "ஃப": "f",  # aytham + pa is how Tamil writes the English /f/ sound
}
NA, NNA = "ந", "ன"
NA_MARK, NNA_MARK = "nd", "nx"

VOWELS = {"அ": "a", "ஆ": "aa", "இ": "i", "ஈ": "ii", "உ": "u", "ஊ": "uu",
          "எ": "e", "ஏ": "ee", "ஐ": "ai", "ஒ": "o", "ஓ": "oo", "ஔ": "au"}
SIGNS = {"ா": "aa", "ி": "i", "ீ": "ii", "ு": "u", "ூ": "uu",
         "ெ": "e", "ே": "ee", "ை": "ai", "ொ": "o", "ோ": "oo", "ௌ": "au"}
SEP = "'"

# ---- inverse tables
_CONS_INV = {v: k for k, v in CONS.items() if k not in (NA, NNA)}
_CONS_INV.update({NA_MARK: NA, NNA_MARK: NNA, "q": AYTHAM})
_VOW_INV = {v: k for k, v in VOWELS.items()}
_SIGN_INV = {v: k for k, v in SIGNS.items()}
_SIGN_INV["a"] = ""
_UNITS = sorted(set(_CONS_INV) | {"n"} | set(_VOW_INV), key=len, reverse=True)
_CONS_UNITS = set(_CONS_INV) | {"n"}
_VOW_UNITS = set(_VOW_INV)

TAMIL_RE = re.compile(r"[஀-௿]+")

# Common typing errors in the source corpora (visually identical renderings):
#   "ா்" typed for "ர்" (old fonts drew ர like ா), and two-part vowel signs typed in reverse order.
#   "ுா" typed for "ூ" (old fonts drew ூ as ு + ா).
_FIXES = [("ாே", "ோ"), ("ாெ", "ொ"), ("ௗெ", "ௌ"), ("ுா", "ூ")]
_AA_AS_RA = re.compile("ா(?=[ா-ௌ்])")  # ா followed by another vowel sign / virama is really ர


def normalize_tamil(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("‌", "").replace("‍", "")
    for a, b in _FIXES:
        text = text.replace(a, b)
    text = _AA_AS_RA.sub("ர", text)
    return unicodedata.normalize("NFC", text)


def _tokenize(word: str, strict: bool = True) -> list[str]:
    """Greedy longest-match split of a romanised word into units (separator kept as a unit).
    With strict=False, characters outside the scheme (e.g. 'b', 'd', 'w' in model output) become
    single-character passthrough units instead of raising."""
    out, i = [], 0
    while i < len(word):
        if word[i] == SEP:
            out.append(SEP); i += 1; continue
        for u in _UNITS:
            if word.startswith(u, i):
                out.append(u); i += len(u); break
        else:
            if strict:
                raise ValueError(f"cannot parse {word!r} at {i}")
            out.append(word[i]); i += 1
    return out


# ------------------------------------------------------------------ encode
def _syllables(word: str) -> list[tuple[str, str]]:
    """Tamil word -> list of (kind, letter) with kind in C (consonant), S (vowel sign), V (vowel), D (virama), Q."""
    out = []
    word = word.replace("ஃப", "\x00")  # treat aytham+pa as one consonant letter
    for ch in word:
        if ch == "\x00":
            out.append(("C", "ஃப"))
        elif ch in CONS:
            out.append(("C", ch))
        elif ch in SIGNS:
            out.append(("S", ch))
        elif ch == VIRAMA:
            out.append(("D", ch))
        elif ch in VOWELS:
            out.append(("V", ch))
        elif ch == AYTHAM:
            out.append(("Q", ch))
        else:
            raise ValueError(f"unsupported Tamil char U+{ord(ch):04X} in {word!r}")
    return out


def _encode_word(word: str) -> str:
    syl = _syllables(word)
    # units: list of (string, type) with type C (consonant letters) or V (vowel letters)
    units: list[tuple[str, str]] = []
    i = 0
    while i < len(syl):
        kind, ch = syl[i]
        if kind == "C":
            nxt = syl[i + 1] if i + 1 < len(syl) else None
            dead = nxt is not None and nxt[0] == "D"
            if ch in (NA, NNA):
                after = syl[i + 2] if dead and i + 2 < len(syl) else None
                na_pos = i == 0 or (dead and after is not None and after == ("C", "த"))
                if ch == NA:
                    s = "n" if na_pos else NA_MARK
                else:
                    s = NNA_MARK if na_pos else "n"
            else:
                s = CONS[ch]
            units.append((s, "C"))
            if dead:
                i += 2
            elif nxt is not None and nxt[0] == "S":
                units.append((SIGNS[nxt[1]], "VS")); i += 2
            else:
                units.append(("a", "VS")); i += 1
        elif kind == "V":
            units.append((VOWELS[ch], "VI")); i += 1
        elif kind == "Q":
            units.append(("q", "C")); i += 1
        else:
            raise ValueError(f"stray {'vowel sign' if kind == 'S' else 'virama'} in {word!r}")
    out = []
    prev = None
    for s, t in units:
        if prev is not None:
            ps, pt = prev
            need = (
                (t == "VI" and pt in ("VS", "VI"))            # vowel after vowel
                or (t == "VI" and pt == "C")                  # independent vowel after dead consonant
                or (t == "C" and pt == "C" and _tokenize(ps + s) != [ps, s])  # consonant merge
            )
            if need:
                out.append(SEP)
        out.append(s)
        prev = (s, t)
    return "".join(out)


def romanize(text: str) -> str:
    """Romanise every Tamil-script run in `text`; anything else (Latin, digits, punctuation) passes through."""
    text = normalize_tamil(text)
    return TAMIL_RE.sub(lambda m: _encode_word(m.group(0)), text)


# ------------------------------------------------------------------ decode
def _decode_word(word: str, strict: bool = False) -> str:
    """Romanised word -> Tamil script. Non-strict: characters outside the scheme pass through as-is."""
    units = _tokenize(word.lower(), strict=strict)
    out = []
    n = len(units)
    i = 0
    while i < n:
        u = units[i]
        if u == SEP:
            i += 1; continue
        if u in _CONS_UNITS:
            if u == "q":
                out.append(AYTHAM); i += 1; continue
            nxt = units[i + 1] if i + 1 < n else None
            if u == "n":
                word_start = i == 0
                before_tha = nxt == "th"
                ch = NA if (word_start or before_tha) else NNA
            else:
                ch = _CONS_INV[u]
            if nxt is not None and nxt in _VOW_UNITS:
                out.append(ch + _SIGN_INV[nxt]); i += 2
            else:
                out.append(ch + VIRAMA); i += 1
        elif u in _VOW_INV:
            out.append(_VOW_INV[u]); i += 1
        else:  # passthrough (non-strict)
            out.append(u); i += 1
    return "".join(out)


_ROM_WORD_RE = re.compile(r"[a-z']+")


def deromanize_word(word: str, strict: bool = False) -> str:
    return _decode_word(word, strict)


def deromanize(text: str, strict: bool = False) -> str:
    """Invert `romanize` for text known to be romanised Tamil (all [a-z'] runs are treated as Tamil).
    strict=True raises on characters outside the scheme; the default passes them through (for model output)."""
    return _ROM_WORD_RE.sub(lambda m: _decode_word(m.group(0), strict), text.lower())
