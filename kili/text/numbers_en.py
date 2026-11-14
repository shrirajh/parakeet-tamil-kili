"""English numbers in labels -> spoken words, so every training source uses one convention (spoken form).

Most English training labels are spoken-form already; People's Speech writes digits in ~7% of its labels ("over 2,000
people"). Pure Python and conservative: anything it can't verbalise confidently makes verbalize() return None (the
caller drops that row).

    verbalize("over 2,000 people in 1995 paid $5.50, 20% more on the 3rd") ->
    "over two thousand people in nineteen ninety five paid five dollars fifty, twenty percent more on the third"

  python -m kili.text.numbers_en "in 1995 we paid $5.50"
"""
from __future__ import annotations

import re
import sys

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
        "seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = [(10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")]
_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
        "twelve": "twelfth"}


def cardinal(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + cardinal(n % 100))
    for v, name in _SCALES:
        if n >= v:
            return cardinal(n // v) + " " + name + ("" if n % v == 0 else " " + cardinal(n % v))
    raise ValueError(n)


def year(n: int) -> str:
    if n % 1000 < 10 and n >= 2000:  # 2000-2009: "two thousand five"
        return cardinal(n)
    hi, lo = divmod(n, 100)
    if lo == 0:
        return cardinal(hi) + " hundred"
    return cardinal(hi) + " " + ("oh " + _ONES[lo] if lo < 10 else cardinal(lo))


def ordinal(n: int) -> str:
    w = cardinal(n).split()
    last = w[-1]
    if last in _ORD:
        w[-1] = _ORD[last]
    elif last.endswith("y"):
        w[-1] = last[:-1] + "ieth"
    else:
        w[-1] = last + "th"
    return " ".join(w)


def digits(s: str) -> str:
    return " ".join(_ONES[int(c)] for c in s)


_TOKEN = re.compile(r"(?<![\w.])(\$|£|€|₹|rs\.?\s?)?(\d[\d,]*)(\.\d+)?(%|st|nd|rd|th)?(?![\w])", re.I)
_CUR = {"$": ("dollar", "dollars", "cent", "cents"), "£": ("pound", "pounds", "penny", "pence"),
        "€": ("euro", "euros", "cent", "cents"), "₹": ("rupee", "rupees", "paisa", "paise")}


def _one(m: re.Match) -> str:
    cur, num, frac, suf = m.group(1), m.group(2), m.group(3), (m.group(4) or "").lower()
    plain = num.replace(",", "")
    if "," in num and not re.fullmatch(r"\d{1,3}(,\d{3})+", num):
        raise ValueError("odd grouping")
    if cur and cur.lower().startswith("rs"):
        cur = "₹"
    if cur:
        n = int(plain)
        one, many, sub1, subn = _CUR[cur]
        out = cardinal(n) + " " + (one if n == 1 else many)
        if frac:
            c = int((frac[1:] + "0")[:2])
            if c:
                out += " " + cardinal(c)
        return out
    if suf in ("st", "nd", "rd", "th") and not frac:
        return ordinal(int(plain))
    if len(plain) > 1 and plain.startswith("0") or (len(plain) >= 5 and "," not in num):
        out = digits(plain)  # phone / account / PIN-style strings: digit by digit
    elif len(plain) == 4 and "," not in num and 1100 <= int(plain) <= 2099 and not frac and suf != "%":
        out = year(int(plain))
    else:
        out = cardinal(int(plain))
    if frac:
        out += " point " + digits(frac[1:])
    if suf == "%":
        out += " percent"
    return out


def verbalize(text: str) -> str | None:
    """Spoken-form numbers; None if something numeric couldn't be verbalised confidently (the caller drops the row)."""
    try:
        out = _TOKEN.sub(_one, text)
    except (ValueError, KeyError):
        return None
    return None if re.search(r"\d", out) else out


if __name__ == "__main__":
    print(verbalize(" ".join(sys.argv[1:])))
