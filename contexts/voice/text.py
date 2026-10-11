"""What Atlas says aloud. A compact Python version of web/lib/speechText.ts (the browser's voice): no markdown, links,
frame refs or ids, times the way a person says them, and a few sentences at most."""

import re

MAX_SPOKEN_CHARS = 900  # room for a whole day walked through ("run me through my day" is said in full, never cut off); was 420
NOTHING_TO_SAY = "I don't have an answer to read out for that."

_URL = re.compile(r"\b(?:https?://|www\.)[^\s<>()\]]+", re.I)
_FRAME_REF = re.compile(r"\bframes?:[\w.-]+:\d{6,}\b", re.I)
_EPISODE_ID = re.compile(r"\b[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*[-_]\d{10,}\b", re.I)
_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
_LONG_NUMBER = re.compile(r"\b\d{10,}\b")
_ISO_STAMP = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ](\d{2}):(\d{2})(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?\b")
_CLOCK_24 = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)(?::[0-5]\d)?\b(?!\s*[AP]M)")
_FAKE_EVIDENCE = re.compile(r"^\s*\(fake\)\s*Evidence:\s*\n([\s\S]*?)(?:\n\s*\n\s*Question:[\s\S]*)?$", re.I)

_REF = "\x01"  # stands in for a removed link or ref until the words that only pointed at it are removed too
_POINTER = re.compile(
    r"(?:\b(?:see|visit|open|view|source|sources|link|links|ref|refs|frame|frames|more at|details at|here|at|via|from)\b[\s:,]*)*"
    + _REF + r"(?:\s*(?:,|and|or)?\s*" + _REF + r")*(?:\s+for (?:more|details|more details|more info|more information)\b)?", re.I)


def _say_clock(hour24: int, minute: int) -> str:
    suffix = "PM" if hour24 >= 12 else "AM"
    h = hour24 % 12 or 12
    if minute == 0 and hour24 == 12:
        return "noon"
    if minute == 0 and hour24 == 0:
        return "midnight"
    return f"{h} {suffix}" if minute == 0 else f"{h}:{minute:02d} {suffix}"


def _strip_markdown(t: str) -> str:
    t = re.sub(r"```[\s\S]*?(```|$)", " ", t)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)
    t = re.sub(r"\[([^\]]+)\]\((?:[^()]|\([^)]*\))*\)", r"\1", t)
    t = re.sub(r"\[\^?\d+\]", "", t)
    t = re.sub(r"<\/?[a-z][^>]*>", " ", t, flags=re.I)
    t = re.sub(r"^\s{0,3}#{1,6}\s+(.*)$", r"\1.", t, flags=re.M)
    t = re.sub(r"^\s{0,3}>\s?", "", t, flags=re.M)
    t = re.sub(r"^\s*(?:[-*+•]|\d{1,2}[.)])\s+(.*)$",
               lambda m: m[1].strip() if re.search(r"[.!?:;,]$", m[1].strip()) else m[1].strip() + ".", t, flags=re.M)
    t = re.sub(r"(\*\*|__)(.+?)\1", r"\2", t)
    t = re.sub(r"~~(.+?)~~", r"\1", t)
    return re.sub(r"`([^`]*)`", r"\1", t)


def _strip_refs(t: str) -> str:
    for pattern in (_URL, _FRAME_REF, _UUID, _EPISODE_ID, _LONG_NUMBER):
        t = pattern.sub(_REF, t)
    t = _POINTER.sub(" ", t).replace(_REF, " ")
    t = re.sub(r"\b(?:see|source|sources|link|links|frame|frames)\s*:\s*(?=[.,;!?]|$)", " ", t, flags=re.I | re.M)
    t = re.sub(r"[([{]\s*(?:see|source|sources|ref|refs|link|links|via|from|frame|frames|id|and|or|at|,|;|:|\s)*[)\]}]", " ", t, flags=re.I)
    return re.sub(r"\bcam-([a-z][a-z0-9]*)(?:-([a-z][a-z0-9]*))?\b",
                  lambda m: f"the {m[1]} {m[2]} camera" if m[2] else f"the {m[1]} camera", t, flags=re.I)


def _clock_24(m: re.Match) -> str:
    """Only a clock that can only be a 24-hour one is reworded: 13:00 and later, a leading zero ("07:05"), "0:15", or one
    with seconds. A bare "10:40" or "2:30" is already how a person says it (the answers written by contexts/day/domain/ear.py
    say a morning time that way) and is spoken as written: appending "AM" to it was a guess."""
    if 1 <= int(m[1]) <= 12 and not m[1].startswith("0") and m[0].count(":") == 1:
        return m[0]
    return _say_clock(int(m[1]), int(m[2]))


def _natural_times(t: str) -> str:
    t = _ISO_STAMP.sub(lambda m: _say_clock(int(m[1]), int(m[2])), t)
    return _CLOCK_24.sub(_clock_24, t)


def _tidy(t: str) -> str:
    t = re.sub(r"\s*&\s*", " and ", t)
    t = re.sub(r"\s*(?:→|->|=>)\s*", " to ", t)
    t = re.sub(r"\s*[·•]\s*", ", ", t)
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"\s+([.,;:!?])", r"\1", t)
    t = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", t)
    t = re.sub(r"[,;:]\s*([.!?])", r"\1", t)
    t = re.sub(r"([.!?])(?:\s*\.)+", r"\1", t)
    t = re.sub(r"\(\s*\)|\[\s*\]", "", t)
    return re.sub(r"^[\s.,;:!?-]+", "", t).strip()


def clean_for_speech(text: str) -> str:
    return _tidy(_natural_times(_strip_refs(_strip_markdown(text))))


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def _cap(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    out = ""
    for s in _sentences(text):
        if out and len(out) + len(s) + 1 > limit:
            break
        if not out and len(s) > limit:
            return re.sub(r"\s+\S*$", "", s[:limit]) + "."
        out = f"{out} {s}" if out else s
    return out


def speech_for_answer(answer: str, citation_count: int = 0) -> str:
    """The answer cut to a few sentences, then at most "Based on N pieces of evidence." The evidence stays on screen."""
    text = answer or ""
    if m := _FAKE_EVIDENCE.match(text):  # the offline demo model echoes its prompt; the evidence lines are the content
        lines = [re.sub(r"^\s*[-*•]\s*", "", ln).strip() for ln in m[1].split("\n")]
        text = "\n".join(f"- {ln}" for ln in [ln for ln in lines if ln][:3])
    said = _cap(clean_for_speech(text), MAX_SPOKEN_CHARS)
    if not said:
        return NOTHING_TO_SAY
    if not re.search(r"[.!?]$", said):
        said += "."
    if citation_count > 0:
        said += " " + evidence_line(citation_count)
    return said


def evidence_line(citation_count: int) -> str:
    return f"Based on {citation_count} {'piece' if citation_count == 1 else 'pieces'} of evidence."


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
MIN_SPOKEN_CHUNK = 20  # "Sure." or "1." is joined to the next sentence, so the speaker is not fed one word at a time


class SentenceStream:
    """Cuts an answer that is still being written into sentences, as soon as each one is finished."""

    def __init__(self) -> None:
        self._buf = ""
        self._held = ""  # a sentence too short to say alone
        self._gap = " "  # what stood between the held words and what follows: a line break is kept, so a short lead-in
        #                  ("Here's the plan:") joined to a list item does not hide the "- " from clean_for_speech

    def feed(self, piece: str) -> list[str]:
        self._buf += piece
        out = []
        while m := _SENTENCE_END.search(self._buf):
            sentence, self._buf = f"{self._held}{self._gap}{self._buf[:m.start()]}".strip(), self._buf[m.end():]
            if len(sentence) >= MIN_SPOKEN_CHUNK:
                out.append(sentence)
                self._held, self._gap = "", " "
            else:
                self._held, self._gap = sentence, "\n" if "\n" in m[0] else " "
        return out

    def flush(self) -> str:
        """Whatever is left once the answer has ended."""
        rest, self._buf, self._held, self._gap = f"{self._held}{self._gap}{self._buf}".strip(), "", "", " "
        return rest


# whisper tiny.en spells the name a few ways: "Atlas", "Atlus", "At last", and sometimes "Alice". "Hey" is required, so the name
# alone inside a sentence ("the atlas is on the shelf") is not a wake; the name as the only word said is.
_NAME = r"(?:atlas|atlus|atlis|atlast|athlas|adlas|alas|alice|atlantis|atlanta|at\s+las|at\s+last)"
_WAKE = re.compile(r"\b(?:hey|hay|hi|okay|ok)\W+" + _NAME + r"\b", re.I)
_NAME_ALONE = re.compile(r"^\W*" + _NAME + r"\W*$", re.I)
MIN_WORDS_FOR_QUESTION = 2  # "Hey Atlas. you" is a wake and a stray word, not a question


def without_wake(text: str) -> str:
    """The words with every "Hey Atlas" taken out: said again and again while waiting, it is not part of the question."""
    spaced = re.sub(r"\s+", " ", _WAKE.sub(" ", text))
    return re.sub(r"^[\W_]+", "", spaced).strip()


def find_wake(transcript: str) -> str | None:
    """None: this was not said to Atlas. Otherwise what followed "Hey Atlas": "" when that was all (Atlas answers "What's up?"
    and listens), or the question when it came in the same breath."""
    m = _WAKE.search(transcript)
    if not m:
        return "" if _NAME_ALONE.match(transcript) else None
    rest = without_wake(transcript[m.end():])
    return rest if len(re.findall(r"\w+", rest)) >= MIN_WORDS_FOR_QUESTION else ""


# What whisper writes for a cough, a fan or a door when there are no words (it learned them from video subtitles)
_NOISE_ONLY = re.compile(r"^\W*(?:you|thank you|thanks|thank you(?: so much)? for watching|thanks for watching|bye|okay|so|um+|uh+|hmm+"
                         r"|please subscribe|subtitles by[\w\s.]*)\W*$", re.I)
# the same words twice or more in a row: whisper looping on a short clip ("Nothing much are you. Nothing much are you.")
_REPEATED = re.compile(r"^\W*(.{4,}?)(?:[\s.,!?]+\1)+\W*?([.!?]?)$", re.I)


def clean_transcript(raw: str) -> str:
    """Whisper marks silence and noise as [BLANK_AUDIO], (music) and the like; those are not a question, and neither is a
    lone "Thank you." it invented for a noise. A phrase it wrote twice in a row is kept once."""
    t = re.sub(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*", " ", raw)
    t = re.sub(r"\s+", " ", t).strip()
    if _NOISE_ONLY.match(t) or not re.search(r"[a-z]", t, re.I):  # "♪♪♪" for a hiss
        return ""
    if m := _REPEATED.match(t):
        t = m[1].rstrip(" .,!?") + (m[2] or ("?" if m[1].rstrip().endswith("?") else ""))
    return t
