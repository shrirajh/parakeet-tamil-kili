"""Loanword engine: English loanwords written in Tamil script -> Latin, with the Tamil suffix attached by a hyphen.

  "நாளைக்கு மீட்டிங்க்கு நீ வர்றியா" -> "நாளைக்கு meeting-க்கு நீ வர்றியா"

This is the label convention of parakeet-tamil-kili: English words in Tamil speech are written in Latin, the Tamil
suffix follows a hyphen (meeting-க்கு, password-அ, app-ல; a vowel-initial suffix is written as an independent vowel).
Precision first: only confident loans are converted, so ambiguous words and native words stay in Tamil script
(போன "went" is not phone, கால "leg / time" is not call, சீனா is China in Tamil, not "cheena").

Data: loans.tsv next to this file. Row kinds:
  surf    <tamil word>  <english>  <suffix>  a verified surface form (suffix "-" = none)
  stem    <tamil stem>  <english>  -         a verified stem; generalises to unseen inflections with a known suffix
  nat     <tamil word>  -          -         a native Tamil word (never converted; also blocks stem parses)
  sandhi  <tamil word>  -          -         a bare loan form that is also a native word + a doubled stop (sandhi):
                                             பேசிக் = basic, but in "பேசிக் கொண்டு" it is பேசி (speaking) + க். Left in
                                             Tamil when the next word starts with that consonant.
The table was built offline from Tamil transcripts of public speech corpora: candidate word types that parse as an
English stem + whitelisted suffix were judged by LLMs (out of context, then blind on transcript snippets), checked for
phonetic plausibility, and reviewed by hand. Names of people and places are kept in Tamil script.

Context rules: a word is not looked up alone where its neighbour changes what it is.
  - sandhi doubling: a `sandhi` form followed by a word starting with its final consonant stays Tamil;
  - split compounds: a loan written as two words where only one half converts on its own ("ஸ்டார்ட் அப்புக்கு",
    "ஆன் லைன்") becomes one word when the joined form is a loan whose English extends that half ("startup-க்கு",
    "online"); never with a clitic / suffix-only second word (தான், ஏ, ஓ, வா).
Transcripts that already contain Latin are normalised too: "pen க்கு" / "penக்கு" -> "pen-க்கு", "jolly யா" ->
"jolly-ஆ". Digits and everything else are left alone.

  python -m kili.text.loanwords in.txt out.txt                       # one transcript per line
  python -m kili.text.loanwords --jsonl in.jsonl out.jsonl --field text
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

VIRAMA = "்"
CONS = set("கஙசஞடணதநபமயரலவழளறனஜஷஸஹஶ")
SIGN2V = {"ா": "ஆ", "ி": "இ", "ீ": "ஈ", "ு": "உ", "ூ": "ஊ", "ெ": "எ", "ே": "ஏ", "ை": "ஐ", "ொ": "ஒ", "ோ": "ஓ",
          "ௌ": "ஔ"}
V2SIGN = {v: s for s, v in SIGN2V.items()}
V2SIGN["அ"] = ""
# canonical suffixes (as written after the hyphen). A parse is accepted only if its suffix is listed here.
SUFFIXES = {
    "", "க்கு", "க்கும்", "க்கே", "க்குத்தான்", "க்குதான்", "க்காக", "க்குள்ள", "க்குள்ளே", "க்கெல்லாம்",
    "ல", "ல்ல", "லே", "லயும்", "லயே", "லேயே", "லதான்", "லத்தான்", "லாம்", "லெல்லாம்", "லேர்ந்து", "லிருந்து",
    "ல்லாம்", "லும்", "அ", "ஆ", "ஐ", "ஏ", "ஓ", "உ", "உம்", "உமே", "ஐயும்", "ஐயே", "ஆவும்", "ஆவே", "ஆன",
    "ஆக", "ஆகவும்", "ஆல", "ஆல்", "ஆலே", "ஓட", "ஓட்ட", "ஓடு", "ஓடது", "இல்", "இல", "இலே", "கிட்ட", "கிட்டே",
    "ன்னு", "னு", "ன்னா", "னா", "தான்", "தானே", "ஏதான்", "எல்லாம்", "பத்தி", "கள்", "களை", "களுக்கு", "களுக்கும்",
    "களோட", "கள", "களும்", "களில்", "களில", "களா", "கள்ல", "ஸ்", "ஸை", "ஸ்ல", "ஸுக்கு", "ஸும்", "ஆனா", "ஆகி",
    "ஏன்னு", "உன்னு", "உனு", "இடம்", "உங்க", "லருந்து", "க்குலாம்", "க்குல்லாம்", "இலிருந்து", "ஆதான்", "ஓட்டு", "ஏயும்",
}
# clitics / particles written as their own word after a loan ("ஸ்கூல் தான்"): never the second half of a compound
CLITICS = {"தான்", "தானே", "ஏ", "ஓ", "ஆ", "வா", "யா", "உம்", "எல்லாம்", "லாம்", "கூட", "மட்டும்", "னு", "ன்னு", "னா",
           "ன்னா", "ங்க", "ம்"}
# Latin + Tamil remainder glued ("itemsலாம்") or spaced ("pen க்கு"); spaced forms need a suffix-only token
SPACED_OK = SUFFIXES - {""}
LAT_TOKEN = re.compile(r"^[A-Za-z][A-Za-z'’.&]*$")
LAT_GLUED = re.compile(r"^([A-Za-z][A-Za-z'’.&]*)-?([஀-௿]+)$")
TA_WORD = re.compile(r"^[஀-௿]+$")
PUNCT = ".,?!\"'()[]:;…“”‘’–—-"
DATA = Path(__file__).with_name("loans.tsv")
MIN_STEM = 3  # stem generalisation only for stems of >= 3 letters (graphemes)


def graphemes(w: str) -> int:
    return sum(1 for c in w if c not in SIGN2V and c != VIRAMA)


def canon_suffix(rem: str) -> str:
    """Normalise a parsed remainder: உக்கு -> க்கு (மீட்டிங்குக்கு = meeting-க்கு)."""
    if rem.startswith("உக்கு"):
        return rem[1:]
    return rem


def splits(w: str) -> list[tuple[str, str]]:
    """Every (stem, suffix) reading of a Tamil-script word: the stem is the English word as spelt in Tamil script
    (consonant-final stems end in pulli), the suffix is canonical (independent vowel first)."""
    w = unicodedata.normalize("NFC", w)
    out: list[tuple[str, str]] = [(w, "")]
    n = len(w)
    if n > 1 and w[-1] in CONS:  # பாஸ்வேர்ட = password-அ
        out.append((w + VIRAMA, "அ"))
    elif n > 1 and w[-1] in SIGN2V and w[-2] in CONS:  # பாஸ்வேர்டை = password-ஐ
        out.append((w[:-1] + VIRAMA, SIGN2V[w[-1]]))
    for i in range(n - 1, 0, -1):
        pre, rest = w[:i], w[i:]
        last = pre[-1]
        if rest[0] in SIGN2V or rest[0] == VIRAMA:
            continue  # not a letter boundary
        if last == VIRAMA:
            c = pre[-2] if len(pre) > 1 else ""
            out.append((pre, rest))  # மீட்டிங் + க்கு (before gemination: ஸ்கூல்ல = school-ல, not school-அ)
            # gemination: stem C் + C(+sign) (பாஸ்வேர்ட்ட = password-அ, பஸ்ஸு = bus-உ)
            if c and rest[0] == c and (len(rest) == 1 or rest[1] != VIRAMA):
                if len(rest) > 1 and rest[1] in SIGN2V:
                    out.append((pre, SIGN2V[rest[1]] + rest[2:]))
                else:
                    out.append((pre, "அ" + rest[1:]))
        elif last in SIGN2V:
            if len(pre) > 1 and pre[-2] in CONS:
                out.append((pre[:-1] + VIRAMA, SIGN2V[last] + rest))  # டாக்டர் + ஓட = டாக்டரோட
            if rest[0] in "யவ" and len(rest) > 1 and rest[1] in SIGN2V:  # மூவி + யா = movie-ஆ
                out.append((pre, SIGN2V[rest[1]] + rest[2:]))
            elif rest[0] in "யவ" and (len(rest) == 1 or rest[1] != VIRAMA):
                out.append((pre, "அ" + rest[1:]))
            out.append((pre, rest))  # மூவி + ல
        elif last in CONS:
            out.append((pre + VIRAMA, "அ" + rest))
            out.append((pre, rest))
        else:
            out.append((pre, rest))
    return [(s, canon_suffix(r)) for s, r in out if not _bad_stem(s)]


def _bad_stem(s: str) -> bool:
    """Stems no English word has: a doubled final consonant (குட்ட்) or a glide left over from a vowel suffix (ஐடிய்)."""
    if len(s) >= 4 and s[-1] == VIRAMA and s[-3] == VIRAMA and s[-2] == s[-4]:
        return True
    return len(s) >= 3 and s[-1] == VIRAMA and s[-2] in "யவ" and s[-3] in "ிீுூ"


def fmt(en: str, suf: str) -> str:
    """English + suffix as written; plural ஸ் joins the English word (ஐட்டம்ஸ் = items, ஐட்டம்ஸுக்கு = items-க்கு)."""
    if suf.startswith("ஸ்") or (len(suf) > 1 and suf[0] == "ஸ" and suf[1] in SIGN2V):
        rest = suf[2:] if suf[1] == VIRAMA else canon_suffix(SIGN2V[suf[1]] + suf[2:])
        en = en if en.endswith("s") else en + "s"
        suf = rest
    return en + ("-" + suf if suf else "")


class Latinizer:
    def __init__(self, path: str | Path = DATA):
        self.surf: dict[str, tuple[str, str]] = {}
        self.stem: dict[str, str] = {}
        self.nat: set[str] = set()
        self.sandhi: set[str] = set()
        for line in open(path, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) < 4 or line.startswith("#"):
                continue
            kind, ta, en, suf = p[:4]
            if kind == "surf":
                self.surf[ta] = (en, "" if suf == "-" else suf)
            elif kind == "stem":
                self.stem[ta] = en
            elif kind == "nat":
                self.nat.add(ta)
            elif kind == "sandhi":
                self.sandhi.add(ta)
            else:
                raise ValueError(f"{path}: unknown row kind {kind!r}")

    @lru_cache(maxsize=500_000)
    def word(self, w: str) -> str | None:
        """Tamil-script word -> 'english' / 'english-suffix', or None (leave as is)."""
        if w in self.surf:
            return fmt(*self.surf[w])
        if w in self.nat:
            return None
        for stem, suf in splits(w):
            # unseen words: a verified stem + a non-empty whitelisted suffix (bare forms are judged as surfaces).
            # Not gemination + உம் (பிளக்கும் is the native verb "will split", not plug-உம்)
            if suf and suf in SUFFIXES and stem in self.stem and graphemes(stem) >= MIN_STEM:
                if suf == "உம்" and w[len(stem):len(stem) + 1] == stem[-2]:
                    return None
                return fmt(self.stem[stem], suf)
        return None

    def joined(self, a: str, b: str) -> str | None:
        """Split compound: "ஸ்டார்ட்" + "அப்புக்கு" -> "startup-க்கு" when only one half converts on its own and the
        joined word's English extends that half's English (start -> startup, line -> online). Not when the other half
        is a clitic / suffix-only word (ஓகே தான், சிக்ஸ் ஓ), and not when the English doesn't extend
        (டூ டேஸ் = two days, never today's)."""
        if not a or not b or a in self.sandhi or _suffix_token(b, gemination=False) is not None or b in CLITICS:
            return None
        j = self.word(a + b)
        if not j:  # a verified bare form as the stem (ஸ்டார்ட்அப் = startup -> ஸ்டார்ட்அப்புக்கு = startup-க்கு)
            j = next((fmt(self.surf[s][0], f) for s, f in splits(a + b)
                      if f and f in SUFFIXES and self.surf.get(s, ("", "x"))[1] == "" and graphemes(s) >= MIN_STEM),
                     None)
        if not j:
            return None
        ra, rb = self.word(a), self.word(b)
        if (ra is None) == (rb is None):
            return None
        je = j.split("-")[0].lower()
        if ra is not None:
            ae = ra.split("-")[0].lower()
            # an inflection is no compound: "ஃபங்க்ஷன் ஸோ" = function so (not functions-ஓ), "லவ் டு" = love to,
            # "ரன் சிங்" = run + a name (not running)
            ext = je[len(ae):]
            infl = ("s", "es", "d", "ed", "ing", "ings", "er", "ers")
            ok = ("-" not in ra and je.startswith(ae) and len(je) > len(ae) and ext not in infl
                  and not (ext[:1] == ae[-1:] and ext[1:] in infl))  # runn-ing
        else:
            be = rb.split("-")[0].lower()
            ok = je.endswith(be) and len(je) > len(be)
        return j if ok else None

    def __call__(self, text: str) -> str:
        toks = text.split(" ")
        cores = [unicodedata.normalize("NFC", t.strip(PUNCT)) for t in toks]
        out: list[str] = []
        converted: set[int] = set()  # positions in out that this call converted (not Latin from the source)
        skip = False
        for i, tok in enumerate(toks):
            if skip:
                skip = False
                continue
            if not tok:
                out.append(tok)
                continue
            core = tok.strip(PUNCT)
            if not core:
                out.append(tok)
                continue
            lo = tok.index(core)
            pre, post = tok[:lo], tok[lo + len(core):]
            if TA_WORD.match(core):
                # "pen க்கு" -> "pen-க்கு": a suffix-only token right after a bare Latin word. After a word converted
                # here, no gemination residue reading: "ஃபோன் பே" is PhonePe, not phone-ஏ
                if out and not pre and LAT_TOKEN.match(out[-1]):
                    s = _suffix_token(core, gemination=len(out) - 1 not in converted)
                    if s is not None:
                        out[-1] = out[-1] + "-" + s
                        if post:
                            out[-1] += post
                        continue
                core = unicodedata.normalize("NFC", core)
                nxt = cores[i + 1] if i + 1 < len(toks) else ""
                # split compound: this word + the next one (no punctuation between them) as one loan
                if not post and nxt and TA_WORD.match(nxt) and toks[i + 1].startswith(nxt):
                    j = self.joined(core, unicodedata.normalize("NFC", nxt))
                    if j:
                        converted.add(len(out))
                        out.append(pre + j + toks[i + 1][len(nxt):])
                        skip = True
                        continue
                r = self.word(core)
                if r and core in SAR and _sar_native(toks[i - 1].strip(PUNCT) if i else "", nxt):
                    r = None
                # sandhi doubling: "பேசிக் கொண்டு" is பேசி + க் before கொண்டு, not "basic"
                # (unless the next word is a loan itself: "குருப் போட்டோ" = group photo)
                if r and core in self.sandhi and nxt.startswith(core[-2]) and not (
                        TA_WORD.match(nxt) and self.word(nxt)):
                    r = None
                if r:
                    converted.add(len(out))
                out.append(pre + (r if r else core) + post)
                continue
            m = LAT_GLUED.match(core)
            if m and _suffix_token(m.group(2)) is not None:  # "itemsலாம்" / "items-லாம்"
                out.append(pre + m.group(1) + "-" + _suffix_token(m.group(2)) + post)
                continue
            out.append(tok)
        return " ".join(out)


# சார் is "sir" in speech, but in formal text also a postposition/prefix: கடல் சார் (sea-related), சார் ஆட்சியர்
# (sub-collector), சார் பதிவாளர். Kept Tamil in those contexts.
SAR = {"சார்"}
SAR_NEXT = re.compile(r"^(ஆட்சியர்|பதிவாள|பதிவக|கருவூல|நிலை|ஆய்வாளர்|நீதிமன்ற|துறை|தொழில்|பாதுகாப்பு|"
                      r"கொள்கை|நெறிமுறை|சொத்|பணியாளர்|உணவூட்ட|உளவு|போக்குவரத்|சேவை|சட்ட|உறுப்பினர்|அலுவல)")
SAR_PREV = {"கடல்", "புவி", "மருத்துவம்", "மருத்துவ", "அறிவு", "துறை", "திறன்", "அலுவல்", "சுங்கம்", "கணிணி", "கணினி",
            "தொழில்", "மண்", "நிலம்", "இயற்கை", "அரசு", "வேளாண்", "விவசாயம்", "கலை", "மொழி", "இனம்", "சாதி",
            "கட்சி", "மதம்", "வணிகம்", "அறிவியல்", "சுற்றுச்சூழல்", "தொழில்நுட்பம்", "மாவட்ட", "இசை", "பெண்ணியம்",
            "மக்கள்", "சமூகம்", "நீர்", "வனம்", "கல்வி", "உணவு", "நலம்", "சுகாதாரம்", "கூட்டுறவு", "அரசியல்"}


def _sar_native(prev: str, nxt: str) -> bool:
    return prev in SAR_PREV or bool(nxt and SAR_NEXT.match(nxt))


def _suffix_token(t: str, gemination: bool = True) -> str | None:
    """A Tamil token that is only a suffix, as written after a Latin word ("க்கு", "யா", "உம்", "பு") -> canonical
    suffix. gemination=False: no consonant residue readings (பு -> உ, டே -> ஏ), only real suffix tokens and glides
    (யா, வா); used after a word this engine converted, where "ஃபோன் பே" is PhonePe and "பர்த் டே" birthday."""
    t = unicodedata.normalize("NFC", t)
    if t in SPACED_OK:
        return t
    # glide/gemination residue: யா -> ஆ, வா -> ஆ, பு -> உ, டு -> உ, ஸ்ஸு -> உ
    if len(t) >= 2 and t[0] in CONS and t[1] in SIGN2V:
        s = SIGN2V[t[1]] + t[2:]
        if s in SPACED_OK and (t[0] in "யவ" or gemination and s in ("உ", "உம்", "ஐ", "அ", "ஆ", "ஏ", "ஓ")):
            return s
    if len(t) >= 1 and t[0] in "யவ" and (len(t) == 1 or t[1] not in SIGN2V and t[1] != VIRAMA):
        s = "அ" + t[1:]
        if s in SPACED_OK:
            return s
    return None


_DEFAULT: Latinizer | None = None


def default() -> Latinizer:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Latinizer()
    return _DEFAULT


def latinize(text: str) -> str:
    """Tamil transcript -> the same transcript with confident English loans in Latin (suffix hyphenated)."""
    return default()(text)


@lru_cache(maxsize=1)
def english_to_tamil() -> dict[str, str]:
    """Lowercased English word -> the Tamil-script stem first listed for it (the reverse direction, used by the
    phonetic fold to write Latin loans back in Tamil script)."""
    m: dict[str, str] = {}
    for line in open(DATA, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 3 and f[0] == "stem":
            m.setdefault(f[2].lower(), f[1])
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description="Tamil-script English loans -> Latin + hyphenated Tamil suffix")
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--jsonl", action="store_true", help="JSON lines input; convert one field")
    ap.add_argument("--field", default="text")
    ap.add_argument("--data", default=str(DATA), help="loan table (default: the bundled loans.tsv)")
    a = ap.parse_args()
    lat = Latinizer(a.data)
    with open(a.inp, encoding="utf-8") as f, open(a.out, "w", encoding="utf-8") as g:
        for line in f:
            if a.jsonl:
                if not line.strip():
                    continue
                r = json.loads(line)
                r[a.field] = lat(r.get(a.field) or "")
                g.write(json.dumps(r, ensure_ascii=False) + "\n")
            else:
                g.write(lat(line.rstrip("\n")) + "\n")


if __name__ == "__main__":
    sys.exit(main())
