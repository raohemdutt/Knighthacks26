import asyncio
import io
from urllib.parse import quote

import pytest
from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

from bus import EventBus
from contexts.memory.application.projection import MemoryProjection
from contexts.memory.application.reader import SqliteMemory
from contexts.memory.infrastructure.store import MemoryStore
from contexts.vault import KeyRing, Vault
from contexts.workspace.api import files_router
from contexts.workspace.application.inbox import FileInbox, FileTooLarge
from contexts.workspace.domain.files import MAX_BYTES, TEXT_LIMIT, extract_text, safe_name
from contexts.workspace.infrastructure.file_store import EncryptedFileStore

SYLLABUS = b"COP 4600 Operating Systems\n\nExam 1 is on October 14 in HEC 101.\nBring a calculator.\n"


def inbox(tmp_path) -> FileInbox:
    return FileInbox(EncryptedFileStore(tmp_path / "files", Vault(KeyRing(tmp_path))))


# --- reading text out of files ---


def test_plain_text_html_and_unknown_types():
    assert extract_text("notes.md", "application/octet-stream", b"# Week 3\n\n\n\n  lab   due Friday ") == ("# Week 3\n\nlab due Friday", "text")
    text, how = extract_text("page.html", "text/html", b"<html><style>x{}</style><h1>Club fair</h1><p>Thu&nbsp;6 PM</p><script>bad()</script>")
    assert how == "html" and "Club fair" in text and "Thu 6 PM" in text and "bad()" not in text and "x{}" not in text
    assert extract_text("photo.jpg", "image/jpeg", b"\xff\xd8\xff") == ("", None)


def test_a_word_file_is_read_when_python_docx_is_installed():
    docx = pytest.importorskip("docx")
    doc = docx.Document()
    doc.add_paragraph("Lab report: ")
    doc.add_paragraph("measured 9.81 m/s2")
    buf = io.BytesIO()
    doc.save(buf)
    text, how = extract_text("report.docx", "application/octet-stream", buf.getvalue())
    assert how == "docx" and "measured 9.81 m/s2" in text


def test_a_broken_pdf_is_kept_but_not_read(tmp_path):
    pytest.importorskip("pypdf")
    rec, event = inbox(tmp_path).add("scan.pdf", "application/pdf", b"%PDF-1.4 not really a pdf")
    assert event.text == "" and event.extractor is None and rec.size == 25


def test_names_lose_paths_and_control_characters():
    assert safe_name("../../etc/passwd") == "passwd"
    assert safe_name("C:\\Users\\me\\Syllabus.pdf") == "Syllabus.pdf"
    assert safe_name("a\x00b\nc.txt") == "abc.txt"
    assert safe_name("") == safe_name(None) == "shared-file"
    assert len(safe_name("x" * 500 + ".txt")) == 120


# --- encrypted on disk ---


def test_bytes_on_disk_are_sealed_and_ids_cannot_walk_out(tmp_path):
    box = inbox(tmp_path)
    rec, _ = box.add("syllabus.txt", "text/plain", SYLLABUS)
    blob = (tmp_path / "files" / f"{rec.file_id}.bin").read_bytes()
    assert b"Operating Systems" not in blob and blob[:1] == b"\x01"  # AES-GCM vault blob, version 1
    assert box.read(rec.file_id) == SYLLABUS
    with pytest.raises(KeyError):
        box.store.load("../vault/master")
    assert not list((tmp_path / "files").glob("*.tmp"))


def test_the_same_bytes_twice_keep_one_copy_and_limits_hold(tmp_path):
    box = inbox(tmp_path)
    first, event = box.add("a.txt", "text/plain", SYLLABUS)
    again, none = box.add("copy of a.txt", "text/plain", SYLLABUS)
    assert event is not None and none is None and again.file_id == first.file_id
    assert len(list((tmp_path / "files").iterdir())) == 1
    with pytest.raises(FileTooLarge):
        box.add("big.bin", None, b"x" * (MAX_BYTES + 1))
    with pytest.raises(ValueError):
        box.add("empty.txt", "text/plain", b"")


def test_long_text_is_capped_in_the_event_but_its_length_is_kept(tmp_path):
    _, event = inbox(tmp_path).add("book.txt", "text/plain", ("word " * 5000).encode())
    assert len(event.text) == TEXT_LIMIT and event.text_chars == len(("word " * 5000).strip())


# --- through the app: event, memory search, replay ---


def app_for(tmp_path, token=None):
    bus = EventBus(tmp_path / "events")
    box = inbox(tmp_path)
    bus.subscribe("workspace.file.added", box.handle, replay=True)
    store = MemoryStore(tmp_path / "memory.sqlite")
    MemoryProjection(store).subscribe(bus)

    def guard(authorization: str | None = Header(default=None)) -> None:
        if token and authorization != f"Bearer {token}":
            raise HTTPException(401, "missing or wrong token")

    app = FastAPI()
    app.include_router(files_router(box, bus, guard=guard))
    return TestClient(app), bus, box, store


def test_upload_emits_the_event_and_memory_search_finds_the_file(tmp_path):
    c, bus, box, store = app_for(tmp_path)
    r = c.post("/api/files", params={"name": "COP4600 syllabus.txt"}, content=SYLLABUS, headers={"Content-Type": "text/plain"})
    assert r.status_code == 200, r.text
    rec = r.json()
    assert rec["duplicate"] is False and rec["extractor"] == "text" and rec["excerpt"].startswith("COP 4600 Operating Systems")
    [logged] = [e for e in bus.read_log() if e.type == "workspace.file.added"]
    assert logged.file_id == rec["file_id"] and "Exam 1 is on October 14" in logged.text

    hits = SqliteMemory(store).search("operating systems exam")
    assert hits and hits[0].kind == "file" and hits[0].title == "COP4600 syllabus.txt"
    assert hits[0].url == f"/api/files/{rec['file_id']}" and hits[0].camera_id == "files"

    assert [f["name"] for f in c.get("/api/files").json()] == ["COP4600 syllabus.txt"]
    d = c.get(f"/api/files/{rec['file_id']}")
    assert d.content == SYLLABUS and d.headers["content-disposition"].startswith("attachment;")
    assert c.get("/api/files/" + "0" * 32).status_code == 404
    assert c.post("/api/files", params={"name": "again.txt"}, content=SYLLABUS).json()["duplicate"] is True


def test_the_list_and_the_index_rebuild_from_the_log(tmp_path):
    c, bus, _, _ = app_for(tmp_path)
    file_id = c.post("/api/files", headers={"X-File-Name": quote("Café menu.txt")}, content=b"Soup of the day: lentil").json()["file_id"]

    fresh_bus = EventBus(tmp_path / "events")  # a restart: same log, same encrypted files, a new memory
    fresh = inbox(tmp_path)
    fresh_bus.subscribe("workspace.file.added", fresh.handle, replay=True)
    store = MemoryStore(tmp_path / "memory2.sqlite")
    MemoryProjection(store).subscribe(fresh_bus)
    asyncio.run(fresh_bus.replay_into())
    assert [r.name for r in fresh.list()] == ["Café menu.txt"]
    assert fresh.read(file_id) == b"Soup of the day: lentil"
    assert SqliteMemory(store).search("lentil soup")[0].episode_id == f"item:files:{file_id}"


def test_the_share_sheet_posts_a_form_and_lands_on_the_list(tmp_path):
    c, _, box, _ = app_for(tmp_path)
    r = c.post("/api/files/share", data={"title": "", "text": ""}, files={"file": ("whiteboard.txt", b"Sprint goals: demo", "text/plain")},
               follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/files/?shared=1"
    r = c.post("/api/files/share", data={"title": "Club fair", "url": "https://example.com/fair"}, follow_redirects=False)
    assert r.status_code == 303
    names = [f.name for f in box.list()]
    assert set(names) == {"whiteboard.txt", "Club fair.txt"}
    note = next(f for f in box.list() if f.name == "Club fair.txt")
    assert box.read(note.file_id) == b"Club fair\nhttps://example.com/fair"
    assert c.post("/api/files/share", data={"title": " "}, follow_redirects=False).status_code == 400
    assert c.post("/api/files/share", content=b"x", headers={"Content-Type": "text/plain"}).status_code == 415


def test_binary_files_survive_the_form_untouched(tmp_path):
    c, _, box, _ = app_for(tmp_path)
    data = bytes(range(256)) * 40 + b"\r\n--not-a-boundary\r\n" + bytes(range(256))
    assert c.post("/api/files/share", files={"file": ("raw.bin", data, "application/octet-stream")}, follow_redirects=False).status_code == 303
    [rec] = box.list()
    assert box.read(rec.file_id) == data and rec.extractor is None


def test_writes_need_the_token_when_one_is_set(tmp_path):
    c, _, _, _ = app_for(tmp_path, token="s3cret")
    assert c.post("/api/files", params={"name": "a.txt"}, content=b"hi").status_code == 401
    ok = c.post("/api/files", params={"name": "a.txt"}, content=b"hi", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
    assert c.get("/api/files").status_code == 200  # reads stay open on the tailnet, like the rest of the API


def test_too_large_is_refused_before_it_is_read(tmp_path):
    c, _, _, _ = app_for(tmp_path)
    r = c.post("/api/files", params={"name": "big.bin"}, content=b"x" * (MAX_BYTES + 1))
    assert r.status_code == 413
