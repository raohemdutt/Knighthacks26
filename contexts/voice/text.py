"""What Atlas says aloud. A compact Python version of web/lib/speechText.ts (the browser's voice): no markdown, links,
frame refs or ids, times the way a person says them, and a few sentences at most."""

import re

MAX_SPOKEN_CHARS = 420
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


def _natural_times(t: str) -> str:
    t = _ISO_STAMP.sub(lambda m: _say_clock(int(m[1]), int(m[2])), t)
    return _CLOCK_24.sub(lambda m: _say_clock(int(m[1]), int(m[2])), t)


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
        said += f" Based on {citation_count} {'piece' if citation_count == 1 else 'pieces'} of evidence."
    return said


def clean_transcript(raw: str) -> str:
    """Whisper marks silence and noise as [BLANK_AUDIO], (music) and the like; those are not a question."""
    t = re.sub(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*", " ", raw)
    return re.sub(r"\s+", " ", t).strip()
