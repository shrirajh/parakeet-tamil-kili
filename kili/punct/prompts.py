"""Prompts that punctuated the Tamil training labels (Gemma 4 12B-it on vLLM, forced tool call, temperature 0).

The model never rewrites text. It sees the label as a numbered word list and returns only the positions that take a
mark ({"after": k, "mark": m}), so letters cannot change by construction; `apply` still checks it. A word that already
ends in a mark (some sources ship punctuation) gets no second one.

Few-shot examples: 12 colloquial IndicVoices training transcripts (CC BY 4.0) with reference punctuation, shown as
prior tool-call turns. 12 examples scored like 30 on 75 held-out sentences (mark F1 59.5 vs 59.7) with a much
shorter prompt.

  from kili.punct.prompts import request_body, apply
  body = request_body("சரி நான் நாளைக்கு வரேன் நீங்க எப்ப வருவீங்க")   # POST to <server>/v1/chat/completions
  apply(text, tool_call_arguments["marks"])  ->  "சரி, நான் நாளைக்கு வரேன். நீங்க எப்ப வருவீங்க?"

Not included: the same pipeline also drafted written-form ("centamil") rewrites of conversational labels for a
spoken/written mode switch. That was dropped and is not used by the released model.
"""
from __future__ import annotations

import json
import re

MODEL_ID = "google/gemma-4-12B-it"
MODEL_REVISION = "43c9d014b231474fa7212dbb48a762566c2b255d"
SERVED_NAME = "gemma12"

MARKS = (",", ".", "?", "!")
SPARSE_RULE = ('Punctuate as a list of marks: {"after": k, "mark": m} puts mark m right after word k (words are numbered '
               'from 1); list only punctuated positions. Sentence ends get "." or "?" (questions, including -ஆ / -ஓ '
               "question endings, get \"?\"); commas only at clear clause breaks, after discourse openers, in lists and around "
               "vocatives or backchannels. Do not comma every filler.")
SYSTEM = "Punctuate this transcript of spoken Tamil (a numbered list of words). Never change the words. " + SPARSE_RULE
TOOL = {"type": "function", "function": {
    "name": "submit_marks", "description": "Punctuation marks by position.",
    "parameters": {"type": "object", "properties": {"marks": {
        "type": "array", "description": "only the punctuated positions",
        "items": {"type": "object", "properties": {"after": {"type": "integer", "description": "word number (1-based)"},
                                                  "mark": {"type": "string", "enum": list(MARKS)}},
                  "required": ["after", "mark"]}}}, "required": ["marks"]}}}

# (unpunctuated transcript, reference punctuation); IndicVoices training transcripts, in prompt order
SHOTS = [
    ("செலவழித்த அத்தனை நேரங்களும் நம்மளுக்கு மலிவான காய்கறிகள் கிடைக்கிதுனு நம்ம சந்தோஷப்பட்டு நம்ம மிக வேகமாக எல்லா வேலையும் செஞ்சமுன்னால்",
     "செலவழித்த அத்தனை நேரங்களும் நம்மளுக்கு மலிவான காய்கறிகள் கிடைக்கிதுனு நம்ம சந்தோஷப்பட்டு நம்ம மிக வேகமாக எல்லா வேலையும் செஞ்சமுன்னால்"),
    ("அடுத்தது நம்ம உடம்புலையும் வலிமையும் இருக்கணும் அதோட சேர்த்து நம்ம டீம்மை provide பண்ணுறதுலையும் ஒரு",
     "அடுத்தது, நம்ம உடம்புலையும் வலிமையும் இருக்கணும். அதோட சேர்த்து நம்ம டீம்மை provide பண்ணுறதுலையும் ஒரு"),
    ("அதனால அரசு தக்க நடிவு நடவடிக்கை எடுத்து நிறையா உதவிகளெலாம் பண்ணிணாதான் விவசாயத்தை காப்பாற்ற முடியும்",
     "அதனால, அரசு தக்க நடிவு நடவடிக்கை எடுத்து நிறையா உதவிகளெலாம் பண்ணிணாதான் விவசாயத்தை காப்பாற்ற முடியும்."),
    ("normal-ஆக எல்லாேரும் சமைக்கிறதும் ஒன்றுதான் நம்மளுக்கு விருப்பப்பட்டு ஒருத்தருக்கு செஞ்சுத்தரணுன்னு நினைக்குறது வந்து ஒரு ஒரு ஒரு விதமான அன்பு அப்படின்றத நான் வந்து நினைச்சுக்குறேன்",
     "normal-ஆக எல்லாேரும் சமைக்கிறதும் ஒன்றுதான். நம்மளுக்கு விருப்பப்பட்டு ஒருத்தருக்கு செஞ்சுத்தரணுன்னு நினைக்குறது வந்து ஒரு ஒரு ஒரு விதமான அன்பு அப்படின்றத நான் வந்து நினைச்சுக்குறேன்."),
    ("அப்புறம் காலையில எந்திரிச்சு கொஞ்சம் easy-ஆக வேலை முடியுங்கிறக்காக எல்லாமே செஞ்சுருவேன் அது அதனால் அந்த மாதிரி செய்யறது வந்து எனக்கு கொஞ்சம் easy-ஆக இருக்குன்னு வைங்களேன்",
     "அப்புறம் காலையில எந்திரிச்சு கொஞ்சம் easy-ஆக வேலை முடியுங்கிறக்காக எல்லாமே செஞ்சுருவேன். அது அதனால் அந்த மாதிரி செய்யறது வந்து எனக்கு கொஞ்சம் easy-ஆக இருக்குன்னு வைங்களேன்."),
    ("நன அடையறத்துக்கு வந்து drawing வந்து நல்ல மாஸ்டர் ஆஃப் degree அந்த விதத்தில் வந்து நான் நல்லா வரைவேன்",
     "நன அடையறத்துக்கு வந்து drawing வந்து நல்ல மாஸ்டர் ஆஃப் degree, அந்த விதத்தில் வந்து நான் நல்லா வரைவேன்."),
    ("அதனால் நம்ம அவங்கக்கிட்ட கத்துக்கிறது ரொம்ப நல்லது ஏன்னா வந்து ஒரு சிறப்பான முறையில் நம்ம கத்துக்கிறதுக்கு வாய்ப்பு இருக்கும் அவங்கக்கிட்ட போய்ட்டு கற்றுகிட்டோன்னா",
     "அதனால் நம்ம அவங்கக்கிட்ட கத்துக்கிறது ரொம்ப நல்லது. ஏன்னா வந்து ஒரு சிறப்பான முறையில் நம்ம கத்துக்கிறதுக்கு வாய்ப்பு இருக்கும், அவங்கக்கிட்ட போய்ட்டு கற்றுகிட்டோன்னா."),
    ("எவ்வளோ point collect பண்ணுறோம் ஒவ்வொன்றுக்கும் ஒவ்வொரு point மாதிரி இருக்கும் அது வந்து அது பேஸ் பண்ணி போட்டோம்ன்னால் நம்மளுக்கு இன்னும் கொஞ்சம் point கிடைக்கும் ஓசரம் அது கொஞ்சம் easy-ஆக இருக்கும்",
     "எவ்வளோ point collect பண்ணுறோம், ஒவ்வொன்றுக்கும் ஒவ்வொரு point மாதிரி இருக்கும். அது வந்து அது பேஸ் பண்ணி போட்டோம்ன்னால் நம்மளுக்கு இன்னும் கொஞ்சம் point கிடைக்கும். ஓசரம் அது கொஞ்சம் easy-ஆக இருக்கும்."),
    ("அங்கே வந்து வழிபாட்டு நேரன்னு பார்த்துக்கிட்டிங்கனா காலைல பதினோரு மணிலருந்து மத்யானம் ஒன்றரை வரையும் இருக்கும்",
     "அங்கே வந்து வழிபாட்டு நேரன்னு பார்த்துக்கிட்டிங்கனா, காலைல பதினோரு மணிலருந்து மத்யானம் ஒன்றரை வரையும் இருக்கும்."),
    ("பா நம்ம bank-இல் பண்ணணும் அப்படிங்கிறது எல்லாமே அது மூலயுமா நம்ம தெரிஞ்சு அதன்படி வந்து நம்ம செயல்படலாம்",
     "பா நம்ம bank-இல் பண்ணணும் அப்படிங்கிறது எல்லாமே அது மூலயுமா நம்ம தெரிஞ்சு, அதன்படி வந்து நம்ம செயல்படலாம்."),
    ("இப்போ இந்த பருவ காலத்துலலாம் பார்த்திங்கன்னு சொன்னால் மக்களை பாதிக்கக்கூடிய நோய்கள் வந்து நிறையா இருக்கு",
     "இப்போ இந்த பருவ காலத்துலலாம் பார்த்திங்கன்னு சொன்னால், மக்களை பாதிக்கக்கூடிய நோய்கள் வந்து நிறையா இருக்கு."),
    ("ஒரு இடோம் இருக்கு அங்கே வந் அங்கே வந்து சுற்றுலா பயணிகள் வரவங்க அப்படி இல்லைனா அவங்க வந்து அவங்க வந்து தங்கிக்குவாங்க",
     "ஒரு இடோம் இருக்கு. அங்கே வந் அங்கே வந்து சுற்றுலா பயணிகள் வரவங்க, அப்படி இல்லைனா அவங்க வந்து அவங்க வந்து தங்கிக்குவாங்க."),
]

ENDS_IN_MARK = tuple(".?,!;:…")


def numbered(text: str) -> str:
    """'a b c' -> 'N = 3\\n1. a\\n2. b\\n3. c' (the user turn)."""
    ws = text.split()
    return f"N = {len(ws)}\n" + "\n".join(f"{i + 1}. {w}" for i, w in enumerate(ws))


def marks_of(punctuated: str) -> tuple[list[str], list[str]]:
    """'சரி, நான் வருகிறேன்.' -> (['சரி', 'நான்', 'வருகிறேன்'], [',', '', '.'])"""
    words, marks = [], []
    for tok in punctuated.split():
        m = re.match(r"^(.*?)([.,?!]*)$", tok)
        words.append(m.group(1))
        marks.append(m.group(2)[-1:] if m.group(2) else "")
    return words, marks


def sparse(marks: list[str]) -> list[dict]:
    return [{"after": k + 1, "mark": m} for k, m in enumerate(marks) if m]


def dense(marks: list | None, n: int) -> list[str]:
    """Tool-call marks -> one mark (or "") per word. Out-of-range positions and unknown marks are ignored."""
    out = [""] * n
    for x in marks or []:
        try:
            k, m = int(x["after"]) - 1, x["mark"]
        except (KeyError, TypeError, ValueError):
            continue
        if 0 <= k < n and m in MARKS:
            out[k] = m
    return out


def messages(text: str, shots=SHOTS) -> list[dict]:
    """System prompt, each example as a user turn + forced tool call + tool reply, then the text to punctuate."""
    msgs = [{"role": "system", "content": SYSTEM}]
    for k, (src, ref) in enumerate(shots):
        _, pm = marks_of(ref)
        msgs += [{"role": "user", "content": numbered(src)},
                 {"role": "assistant", "content": None, "tool_calls": [{"id": f"c{k}", "type": "function", "function": {
                     "name": "submit_marks", "arguments": json.dumps({"marks": sparse(pm)}, ensure_ascii=False)}}]},
                 {"role": "tool", "tool_call_id": f"c{k}", "content": "ok"}]
    return msgs + [{"role": "user", "content": numbered(text)}]


def request_body(text: str, served: str = SERVED_NAME, shots=SHOTS) -> dict:
    """OpenAI-compatible /chat/completions body: greedy, thinking off, the tool call forced."""
    return {"model": served, "messages": messages(text, shots), "tools": [TOOL], "temperature": 0, "max_tokens": 900,
            "tool_choice": {"type": "function", "function": {"name": "submit_marks"}},
            "chat_template_kwargs": {"enable_thinking": False}}


def punctuate(words: list[str], marks: list[str]) -> str:
    """Attach marks to words; a word that already ends in a mark keeps only its own."""
    return " ".join(w + (marks[i] if i < len(marks) and marks[i] and not w.endswith(ENDS_IN_MARK) else "")
                    for i, w in enumerate(words))


def strip_added(src: str, out: str) -> bool:
    """True if `out` is `src` with marks added after words and nothing else changed."""
    sw, ow = src.split(), out.split()
    return len(sw) == len(ow) and all(o == s or (o.startswith(s) and o[len(s):] in MARKS) for s, o in zip(sw, ow))


def apply(text: str, tool_marks: list | None) -> str:
    """The punctuated label for `text` from the model's `marks` argument. Raises if any letter would change."""
    words = text.split()
    out = punctuate(words, dense(tool_marks, len(words)))
    if not strip_added(text, out):
        raise ValueError(f"punctuation changed the text: {text!r} -> {out!r}")
    return out
