"""A stand-in llama-server for lab/desk_score.py, and the builder of the synthetic desk fixture. No network, no model.

    from lab.desk_fake import FakeDeskServer, fixture_policy
    server = FakeDeskServer(fixture_policy("fixtures/learning/desk-fixture"))
    scores = desk_score.score("http://desk.invalid:8080", examples, prompt=..., transport=server.transport)

    python -m lab.desk_fake build-fixture [--out fixtures/learning/desk-fixture]

The server answers what the scorer asks of llama-server: GET /v1/models lists the served model (SERVED_MODEL), and
POST /v1/chat/completions answers one choice with the text the policy gives for the request body and the picture's
bytes. As a Qwen chat template does, a request whose chat_template_kwargs does not set enable_thinking to false gets a
thinking block in front of its answer, and an answer longer than max_tokens (one token per four characters) is cut
there with finish reason "length". fail(status) answers the next request with that status. Every request is kept in
requests: {"path", "body", "finish_reason"}.

The fixture (fixtures/learning/desk-fixture) is a desk-object export in the box's format (contracts/learning.py): 12
examples on 8 frame_refs (four frames asked twice), labels mug, keys, stapler and phone charger in target, the box's
caption in context, a 64x64 grayscale PNG per frame (stored uncompressed, so the bytes are the same on every
machine), split.json as python -m lab desk-score split writes it with FIXTURE_HELD_OUT, and fake_answers.json: what
the fake model says for each example, some of them wrong on purpose so a score is neither 0 nor 1. Stdlib and httpx.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import struct
import sys
import zlib
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if not __package__:  # run by path: the repo root, one folder up, is what holds lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from contracts.learning import DatasetExample, DatasetManifest, TaskSpec

__all__ = ["DATASET_ID", "FIXTURE_DIR", "FIXTURE_HELD_OUT", "SERVED_MODEL", "PROMPT", "Policy", "FakeDeskServer",
           "fixture_answers", "fixture_policy", "png", "build_fixture", "main"]

DATASET_ID = "desk-fixture"
FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "learning" / DATASET_ID
FIXTURE_HELD_OUT = 5
SERVED_MODEL = "atlas-fake-Q4_K_M.gguf"
PROMPT = "Name the one object on the desk in this picture. Reply with only its name."
ANSWERS = "fake_answers.json"
THINKING = "<think>\nThe user wants the object's name.\n</think>\n\n"
CREATED = datetime(2026, 10, 11, 9, 30, tzinfo=timezone.utc)

Policy = Callable[[Mapping[str, Any], bytes], str]

# (frame, label, caption, what the fake model answers) per example; frames 0-3 are each asked twice
_EXAMPLES = [
    (0, "mug", "A white mug on the desk next to a laptop.", "A mug."),
    (0, "mug", "A white mug on the desk next to a laptop.", "A mug."),
    (1, "keys", "A ring of keys by the keyboard.", "the Keys"),
    (1, "keys", "A ring of keys by the keyboard.", "the Keys"),
    (2, "stapler", "A black stapler at the edge of the desk.", "stapler."),
    (2, "stapler", "A black stapler at the edge of the desk.", "stapler."),
    (3, "phone charger", "A phone charger plugged in beside the monitor.", "cable"),
    (3, "phone charger", "A phone charger plugged in beside the monitor.", "cable"),
    (4, "mug", "A mug of coffee by the notebook.", "cup"),
    (5, "keys", "Keys on top of a notebook.", "keys"),
    (6, "stapler", "A stapler beside a stack of paper.", "A Stapler"),
    (7, "phone charger", "A charger coiled on the desk.", "Phone chargers"),
]


class FakeDeskServer:
    """llama-server's chat and model routes over an httpx.MockTransport (see the module docstring)."""

    def __init__(self, policy: Policy, *, model: str = SERVED_MODEL) -> None:
        self.policy = policy
        self.model = model
        self.requests: list[dict[str, Any]] = []
        self._failures: list[int] = []
        self.transport = httpx.MockTransport(self._handle)

    def fail(self, status: int, times: int = 1) -> None:
        self._failures.extend([status] * times)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/models" and request.method == "GET":
            self.requests.append({"path": path, "body": None, "finish_reason": None})
            return httpx.Response(200, json={"object": "list", "data": [{"id": self.model, "object": "model",
                                                                         "owned_by": "llamacpp"}]})
        if path != "/v1/chat/completions" or request.method != "POST":
            return httpx.Response(404, json={"error": {"code": 404, "message": "File Not Found"}})
        body = json.loads(request.content)
        if self._failures:
            self.requests.append({"path": path, "body": body, "finish_reason": None})
            status = self._failures.pop(0)
            return httpx.Response(status, json={"error": {"code": status, "message": "fake failure"}})
        text = self.policy(body, _picture(body))
        if (body.get("chat_template_kwargs") or {}).get("enable_thinking") is not False:
            text = THINKING + text
        finish = "stop"
        limit = body.get("max_tokens")
        if isinstance(limit, int) and len(text) > 4 * limit:
            text, finish = text[:4 * limit], "length"
        self.requests.append({"path": path, "body": body, "finish_reason": finish})
        return httpx.Response(200, json={
            "object": "chat.completion", "model": self.model,
            "choices": [{"index": 0, "finish_reason": finish, "message": {"role": "assistant", "content": text}}],
            "usage": {"completion_tokens": max(1, len(text) // 4)}})


def _picture(body: Mapping[str, Any]) -> bytes:
    """The bytes of the first picture in the request's messages, or b"" when it has none."""
    for message in body.get("messages") or []:
        content = message.get("content")
        for part in content if isinstance(content, list) else []:
            url = (part.get("image_url") or {}).get("url", "") if part.get("type") == "image_url" else ""
            if url.startswith("data:") and "," in url:
                return base64.b64decode(url.split(",", 1)[1])
    return b""


def fixture_answers(root: str | Path = FIXTURE_DIR) -> dict[str, str]:
    """What the fake model answers for each example id of the fixture."""
    return json.loads((Path(root) / ANSWERS).read_text(encoding="utf-8"))


def fixture_policy(root: str | Path = FIXTURE_DIR) -> Policy:
    """Answer each of the fixture's pictures as fake_answers.json says; an unknown picture gets "I don't know"."""
    root = Path(root)
    answers = fixture_answers(root)
    by_picture: dict[str, str] = {}
    for line in (root / "examples.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            by_picture[hashlib.sha256((root / row["image"]).read_bytes()).hexdigest()] = answers[row["id"]]
    return lambda body, picture: by_picture.get(hashlib.sha256(picture).hexdigest(), "I don't know")


# --- the fixture --------------------------------------------------------------------------

def png(width: int, height: int, pixel: Callable[[int, int], int]) -> bytes:
    """A grayscale PNG, its pixel data stored uncompressed (zlib level 0) so the bytes never depend on the zlib."""
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    rows = b"".join(b"\x00" + bytes(pixel(x, y) & 0xFF for x in range(width)) for y in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 0)) + chunk(b"IEND", b""))


def _frame_picture(frame: int) -> bytes:
    """A 64x64 picture per frame: a background shade and a square whose place and shade depend on the frame."""
    left, top, shade = 8 + 5 * frame, 40 - 4 * frame, 255 - 20 * frame
    return png(64, 64, lambda x, y: shade if left <= x < left + 16 and top <= y < top + 16 else 30 + 10 * frame)


def build_fixture(out: str | Path = FIXTURE_DIR) -> Path:
    """Write the synthetic desk fixture into out (made new; it must not exist yet) and its split; the folder."""
    from lab import desk_score   # here, not at the top: desk_score's tests import this module

    out = Path(out)
    (out / "images").mkdir(parents=True)
    rows, answers, seen = [], {}, {}
    for frame in sorted({frame for frame, *_ in _EXAMPLES}):
        (out / "images" / f"desk-{frame:02d}.png").write_bytes(_frame_picture(frame))
    for frame, label, caption, answer in _EXAMPLES:
        asked = seen[frame] = seen.get(frame, -1) + 1
        example = DatasetExample(id=f"desk-{frame:02d}-{asked}", image=f"images/desk-{frame:02d}.png",
                                 frame_ref=f"frames:desk:{1760175000000 + 60000 * frame}", ts=CREATED,
                                 context=caption, target=label)
        rows.append(example.model_dump_json() + "\n")
        answers[example.id] = answer
    manifest = DatasetManifest(dataset_id=out.name, spec=TaskSpec(task="desk-objects", instruction=PROMPT),
                               created=CREATED, examples=len(rows), action_id="fixture:desk-synthetic")
    (out / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    (out / "examples.jsonl").write_text("".join(rows), encoding="utf-8", newline="\n")
    (out / ANSWERS).write_text(json.dumps(answers, indent=2) + "\n", encoding="utf-8", newline="\n")
    desk_score.write_split(out, held_out=FIXTURE_HELD_OUT)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lab.desk_fake", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("build-fixture", help="write the synthetic desk fixture (the folder must not exist)")
    build.add_argument("--out", type=Path, default=FIXTURE_DIR)
    args = parser.parse_args(argv)
    print(json.dumps({"ok": True, "fixture": str(build_fixture(args.out))}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
