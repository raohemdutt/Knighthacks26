"""The file inbox (W1): files shared from the phone, kept encrypted on the box, read, and handed to Memory.

`add` stores the bytes (encrypted), extracts the text and returns `workspace.file.added`; the caller publishes
it. Memory indexes that event for search, and this inbox rebuilds its list from the same events (replay=True),
so the list and the index always agree with the log. Sharing the same bytes twice keeps the first copy.

Adding a file is something the person does from their own phone, not an outside effect: no approval card.
"""

from uuid import uuid4

from contexts.workspace.domain.files import (MAX_BYTES, TEXT_LIMIT, FileRecord, excerpt, extract_text, safe_name,
                                             sha256)
from contexts.workspace.infrastructure.file_store import EncryptedFileStore
from contracts.workspace import FileAdded


class FileTooLarge(ValueError):
    pass


class FileInbox:
    def __init__(self, store: EncryptedFileStore) -> None:
        self.store = store
        self._files: dict[str, FileRecord] = {}
        self._by_hash: dict[str, str] = {}

    def add(self, name: str | None, content_type: str | None, data: bytes) -> tuple[FileRecord, FileAdded | None]:
        """(the record, the event to publish), or (the existing record, None) for bytes already in the inbox."""
        if len(data) > MAX_BYTES:
            raise FileTooLarge(f"files are limited to {MAX_BYTES // (1024 * 1024)} MB")
        if not data:
            raise ValueError("the file is empty")
        digest = sha256(data)
        if (known := self._by_hash.get(digest)) and known in self._files and self.store.exists(known):
            return self._files[known], None
        name = safe_name(name)
        ctype = (content_type or "application/octet-stream").split(";")[0].strip().lower() or "application/octet-stream"
        file_id = uuid4().hex
        self.store.save(file_id, data)  # on disk before the event: a logged file always has its bytes
        text, extractor = extract_text(name, ctype, data)
        event = FileAdded(file_id=file_id, name=name, content_type=ctype, size=len(data), sha256=digest,
                          text=text[:TEXT_LIMIT], text_chars=len(text), extractor=extractor if text else None)
        return self.on_added(event), event

    def on_added(self, e: FileAdded) -> FileRecord:
        """Projection: the list rebuilds from the log. Also called by `add`, so the list is current at once."""
        rec = FileRecord(file_id=e.file_id, name=e.name, content_type=e.content_type, size=e.size, sha256=e.sha256,
                         added_at=e.ts, text_chars=e.text_chars, extractor=e.extractor, excerpt=excerpt(e.text))
        self._files[e.file_id] = rec
        self._by_hash.setdefault(e.sha256, e.file_id)
        return rec

    def handle(self, e: FileAdded) -> list:
        """Bus handler form of `on_added` (event in, no events out)."""
        self.on_added(e)
        return []

    def list(self) -> list[FileRecord]:
        """Newest first. A file whose bytes are gone (wiped) is not listed."""
        return sorted((r for r in self._files.values() if self.store.exists(r.file_id)), key=lambda r: r.added_at, reverse=True)

    def get(self, file_id: str) -> FileRecord | None:
        rec = self._files.get(file_id)
        return rec if rec and self.store.exists(file_id) else None

    def read(self, file_id: str) -> bytes:
        if file_id not in self._files:
            raise KeyError(file_id)
        return self.store.load(file_id)
