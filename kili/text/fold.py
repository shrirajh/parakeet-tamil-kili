"""Convention folds for scoring: make a metric blind to writing choices that are not recognition errors.

  fold(text)      the fold behind fold WER / CER (and EER): spoken number words -> written (அஞ்சு = ஐந்து), English
                  loans in Tamil script -> Latin with the hyphenated suffix (மீட்டிங்க்கு = meeting-க்கு), then norm_ta
                  (punctuation dropped, spaces collapsed; the hyphen too, so meeting-க்கு scores as two words
                  "meeting க்கு" on both sides). Applied to both reference and hypothesis, so a system is
                  neither rewarded nor punished for the script it writes an English loanword in, or for the register
                  of a number. Case is kept.
  phonetic_fold(text)
                  the fold behind per_fold: spoken number words -> written, Latin words -> Tamil script (loan table
                  stem, else a rule transliteration), norm_ta, then sandhi_join. PER itself already ignores spaces.
  sandhi_join(s)  join words the way Tamil sandhi does, so split vs joined writing compares equal:
                  C+virama + vowel -> C+sign (உம் இடம் ~ உமிடம்); vowel + vowel -> glide (கரை உடைப்பை ~ கரையுடைப்பை;
                  ய/வ glides unified); doubled stops collapsed (அப்பு சாமி ~ அப்புச்சாமி ~ அப்புச் சாமி); a word-final
                  stop+virama dropped (சீரமைக்கக் ~ சீரமைக்க). Lenient (symmetric): also collapses lexical gemination.
  detin(s)        Latin words -> Tamil script.

The short-form scorer (kili.eval.evaluate) first writes digits as Tamil number words on both sides
(kili.text.numbers_ta.verbalize_ta), then folds; kili.metrics.wer.fold_wer does the same by default.

  python -m kili.text.fold "நாளைக்கு மீட்டிங்க்கு அஞ்சு மணிக்கு"
"""
from __future__ import annotations

import re
import sys
import unicodedata

from kili.text.loanwords import english_to_tamil, latinize
from kili.text.normalise import norm_ta
from kili.text.numbers_ta import fold_number_words

V = "்"
STOPS = "கசடதபற"
VOW = {"அ": "", "ஆ": "ா", "இ": "ி", "ஈ": "ீ", "உ": "ு", "ஊ": "ூ", "எ": "ெ", "ஏ": "ே", "ஐ": "ை", "ஒ": "ொ", "ஓ": "ோ",
       "ஔ": "ௌ"}
YGLIDE = ("ி", "ீ", "ை", "ெ", "ே", "இ", "ஈ", "ஐ", "எ", "ஏ")


def fold(text: str) -> str:
    return norm_ta(latinize(fold_number_words(text or "")))


def phonetic_fold(text: str) -> str:
    return sandhi_join(norm_ta(detin(fold_number_words(text or ""))))


def sandhi_join(s: str) -> str:
    acc = ""
    for w in s.split():
        if acc and w[0] in VOW:
            if acc.endswith(V):
                acc = acc[:-1] + VOW[w[0]] + w[1:]
                continue
            g = "ய" if acc.endswith(YGLIDE) else "வ"
            acc = acc + g + VOW[w[0]] + w[1:]
            continue
        acc += w
    acc = acc.replace("வ", "ய")
    acc = re.sub(r"([" + STOPS + "])" + V + r"\1", r"\1", acc)
    return re.sub(r"[" + STOPS + "]" + V + "$", "", acc)


def detin(s: str) -> str:
    m = english_to_tamil()
    return re.sub(r"[A-Za-z']+", lambda x: m.get(x.group(0).lower()) or rom_to_ta(x.group(0).lower(), english=True), s)


# ---- rule transliteration Latin -> Tamil script (fallback for words not in the loan table; crude by design) --------
_T_CONS = [("nth", "ந்த"), ("ndh", "ந்த"), ("zh", "ழ"), ("sh", "ஷ"), ("ch", "ச"), ("th", "த"), ("dh", "த"), ("ph", "ஃப"),
           ("bh", "ப"), ("kh", "க"), ("gh", "க"), ("nj", "ஞ்ச"), ("ng", "ங்க"), ("nd", "ண்ட"), ("nt", "ண்ட"),
           ("tr", "ட்ர"), ("k", "க"), ("g", "க"), ("c", "க"), ("q", "க"), ("j", "ஜ"), ("s", "ச"), ("z", "ச"), ("t", "ட"),
           ("d", "ட"), ("n", "ன"), ("p", "ப"), ("b", "ப"), ("f", "ஃப"), ("m", "ம"), ("y", "ய"), ("r", "ர"),
           ("l", "ல"), ("v", "வ"), ("w", "வ"), ("h", "ஹ"), ("x", "க்ஸ")]
_T_VOW = [("aa", "ஆ", "ா"), ("ai", "ஐ", "ை"), ("ay", "ஐ", "ை"), ("au", "ஔ", "ௌ"), ("ee", "ஈ", "ீ"),
          ("ii", "ஈ", "ீ"), ("oo", "ஊ", "ூ"), ("uu", "ஊ", "ூ"), ("ea", "ஈ", "ீ"), ("ei", "ஏ", "ே"),
          ("ae", "ஏ", "ே"), ("ou", "ஒ", "ொ"), ("a", "அ", ""), ("e", "எ", "ெ"), ("i", "இ", "ி"), ("o", "ஒ", "ொ"),
          ("u", "உ", "ு"), ("y", "இ", "ி")]


def rom_to_ta(w: str, english: bool = False) -> str:
    """Rule transliteration of a romanised (or, english=True, English) word into Tamil script."""
    w = re.sub(r"(.)\1{2,}", r"\1\1", w.lower())  # "seriyaaa" -> "seriyaa"
    if english:
        w = re.sub(r"(?<=[^aeiou])e$", "", w)  # silent final e
        w = w.replace("ck", "k").replace("ph", "f")
    out = ""
    i = 0
    after_cons = False
    while i < len(w):
        for a, ind, sign in _T_VOW:
            if w.startswith(a, i) and not (a == "y" and not after_cons):
                if after_cons:
                    out = out[:-1] + sign if out.endswith(V) else out + sign
                elif out and not out.endswith(V):  # vowel after vowel: glide
                    out += ("ய" if out[-1] in "ிீெேை" or out[-1] in "இஈஎஏஐ" else "வ") + (sign or "")
                else:
                    out += ind
                after_cons = False
                i += len(a)
                break
        else:
            for a, ta in _T_CONS:
                if w.startswith(a, i):
                    double = i + len(a) < len(w) and w.startswith(a, i + len(a))
                    if a == "n" and i == 0:
                        ta = "ந"
                    out += ta + V
                    if double:
                        out += ta + V
                        i += len(a)
                    after_cons = True
                    i += len(a)
                    break
            else:
                i += 1
    return unicodedata.normalize("NFC", out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    t = " ".join(sys.argv[1:])
    print("fold:          ", fold(t))
    print("phonetic_fold: ", phonetic_fold(t))
