"""The file inbox's rules (W1): what a shared file may be, and the text Atlas reads out of it.

Text comes out of plain-text files on any box. PDF and Word files need the optional readers
(`pip install -e ".[files]"`: pypdf, python-docx); without them the file is still kept, just not searchable.
"""

import hashlib
import io
import re
import unicodedata
from datetime import datetime
from pathlib import PurePath

from pydantic import BaseModel

MAX_BYTES = 20 * 1024 * 1024  # one file; a syllabus PDF or a photo of a whiteboard, not a video
TEXT_LIMIT = 8000  # characters of extracted text carried in workspace.file.added and indexed for search
MAX_PDF_PAGES = 50

TEXT_TYPES = {"application/json", "application/xml", "application/x-yaml", "application/yaml", "application/csv",
              "application/x-sh", "application/javascript", "application/sql"}
TEXT_EXTS = {".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".yaml", ".yml", ".xml", ".log", ".py", ".js", ".ts",
             ".ics", ".rst", ".tex", ".sql", ".ini", ".toml", ".cfg"}
HTML_EXTS = {".html", ".htm"}
PDF = "application/pdf"
DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class FileRecord(BaseModel):
    """One file in the inbox, as the phone lists it. Rebuilt from workspace.file.added."""

    file_id: str
    name: str
    content_type: str
    size: int
    sha256: str
    added_at: datetime
    text_chars: int
    extractor: str | None
    excerpt: str  # the first line or so of the text, for the list


def safe_name(name: str | None) -> str:
    """A display name with no path, no control characters, at most 120 characters."""
    base = PurePath((name or "").replace("\\", "/")).name
    base = "".join(ch for ch in base if unicodedata.category(ch)[0] != "C").strip().strip(".")
    return (base or "shared-file")[:120]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tidy(text: str) -> str:
    """Whitespace collapsed: spaces within a line, at most one blank line between paragraphs."""
    lines = [re.sub(r"[ \t\f\v\u00a0]+", " ", line).strip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _html(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", raw)
    raw = re.sub(r"(?i)<br\s*/?>|</p>|</div>|</li>|</h[1-6]>", "\n", raw)
    import html

    return html.unescape(re.sub(r"<[^>]+>", " ", raw))


def _pdf(data: bytes) -> str | None:
    try:
        from pypdf import PdfReader
    except ImportError:
        return None
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages[:MAX_PDF_PAGES])
    except Exception:  # a damaged or encrypted PDF is kept, just not read
        return ""


def _docx(data: bytes) -> str | None:
    try:
        import docx
    except ImportError:
        return None
    try:
        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    except Exception:
        return ""


def extract_text(name: str, content_type: str, data: bytes) -> tuple[str, str | None]:
    """(text, extractor). ("", None) when the type is not one Atlas can read, or its reader is not installed."""
    ext = PurePath(name).suffix.lower()
    ctype = content_type.split(";")[0].strip().lower()
    if ext in HTML_EXTS or ctype == "text/html":
        return tidy(_html(data.decode("utf-8", "replace"))), "html"
    if ctype.startswith("text/") or ctype in TEXT_TYPES or ext in TEXT_EXTS:
        return tidy(data.decode("utf-8", "replace")), "text"
    if ctype == PDF or ext == ".pdf":
        text = _pdf(data)
        return (tidy(text), "pdf") if text is not None else ("", None)
    if ctype == DOCX or ext == ".docx":
        text = _docx(data)
        return (tidy(text), "docx") if text is not None else ("", None)
    return "", None


def excerpt(text: str, limit: int = 140) -> str:
    one = " ".join(text.split())
    return one if len(one) <= limit else one[: limit - 1].rstrip() + "…"
