"""Text normalisation: the training-label normaliser and the scoring normalisers.

Training labels (one convention for every source):
  normalise(text, "ta")  Tamil (incl. Tamil-English code-switching): NFC, single spaces, English loans in Tamil script
                         -> Latin with the Tamil suffix hyphenated (kili.text.loanwords: மீட்டிங்க்கு -> meeting-க்கு),
                         numbers -> Tamil number words (kili.text.numbers_ta), Latin words lowercased unless an acronym
                         (BHK, ECG: 2-5 capitals). Punctuation and Tamil text are kept as they are.
  normalise(text, "en")  English: NFC, single spaces, numbers / currency / % in spoken words (kili.text.numbers_en);
                         punctuation and casing kept (Parakeet style).
  lang="auto" (default) picks "ta" if the text contains Tamil script, else "en".
  normalise_label() is the same but returns None when a number can't be verbalised confidently (a code glued to
  letters such as "4GB", a 10+ digit string); training drops those rows. normalise() leaves such digits in place.

Scoring (applied to references and hypotheses before WER / CER):
  norm_ta      Tamil script: drop punctuation, collapse spaces; keeps letters, digits AND combining marks (Python's
               \\w does not match Tamil vowel signs / virama, so a [^\\w\\s] filter would delete them).
  norm_en      lowercase, strip punctuation (keep apostrophes), collapse spaces.
  norm_en_std  the Open ASR Leaderboard English normaliser (Whisper's EnglishTextNormalizer, shipped in
               `transformers`): spelled-out numbers and digits score as equal. Falls back to norm_en (with a warning)
               when transformers is not installed. English WER on the website uses this.

  python -m kili.text.normalise --lang ta < in.txt > out.txt      (one text per line)
"""
from __future__ import annotations

import re
import sys
import unicodedata

_LATIN_WORD = re.compile(r"[A-Za-z][A-Za-z'.]*")
_ACRONYM = re.compile(r"[A-Z]{2,5}s?")
_TAMIL = re.compile(r"[஀-௿]")


def _space(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


def _latin_case(m: re.Match) -> str:
    w = m.group(0)
    return w if _ACRONYM.fullmatch(w) else w.lower()


def _lang(text: str, lang: str) -> str:
    if lang == "auto":
        return "ta" if _TAMIL.search(text) else "en"
    if lang not in ("ta", "en"):
        raise ValueError(f"lang must be 'ta', 'en' or 'auto', not {lang!r}")
    return lang


def normalise_label(text: str, lang: str = "auto") -> str | None:
    """The training-label rule; None = a number that can't be verbalised confidently (drop the row)."""
    t = _space(text or "")
    if _lang(t, lang) == "en":
        from kili.text.numbers_en import verbalize
        return verbalize(t) if re.search(r"\d", t) else t
    from kili.text.loanwords import latinize
    t = _space(latinize(t))
    if re.search(r"[0-9௦-௯]", t):
        from kili.text.numbers_ta import verbalize_ta
        t = verbalize_ta(t)
        if t is None:
            return None
    return _LATIN_WORD.sub(_latin_case, t)


def normalise(text: str, lang: str = "auto") -> str:
    """normalise_label(), except that numbers it can't verbalise are left as digits instead of returning None."""
    out = normalise_label(text, lang)
    if out is not None:
        return out
    t = _space(text or "")
    if _lang(t, lang) == "en":
        return t
    from kili.text.loanwords import latinize
    return _LATIN_WORD.sub(_latin_case, _space(latinize(t)))


# ---- scoring normalisers ------------------------------------------------------------------------------------------
_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)


def norm_ta(s: str) -> str:
    s = unicodedata.normalize("NFC", s)
    s = "".join(c if (unicodedata.category(c)[0] in "LMN" or c.isspace()) else " " for c in s)
    return re.sub(r"\s+", " ", s).strip()


def norm_en(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = _PUNCT.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


_WN = None


def norm_en_std(s: str) -> str:
    global _WN
    if _WN is None:
        try:
            from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
            _WN = EnglishTextNormalizer({})
        except Exception:
            print("[normalise] transformers not installed: norm_en_std falls back to norm_en", file=sys.stderr)
            _WN = False
    return re.sub(r"\s+", " ", _WN(s)).strip() if _WN else norm_en(s)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="normalise one text per line (stdin -> stdout)")
    ap.add_argument("--lang", default="auto", choices=["auto", "ta", "en"])
    a = ap.parse_args()
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    for line in sys.stdin:
        print(normalise(line.rstrip("\n"), a.lang))


if __name__ == "__main__":
    main()
