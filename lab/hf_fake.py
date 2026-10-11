"""A stand-in for the Hugging Face Hub client that tests and dry runs can drive: no network, no token, no library.

    client = FakeHubClient()                       # the base's card answers apache-2.0
    client = FakeHubClient(cards={"Qwen/Qwen3.5-4B": canned_card("mit")})

It has the two calls lab/publish_adapter.py makes through its client, `read_card(repo_id)` and
`upload_folder(repo_id, folder, private=...)`, and records them instead of sending anything:

    client.events       [("read_card", repo), ("upload", repo), ...] in call order
    client.card_reads   the repos whose card was read
    client.uploads      one Upload per upload: repo, private, files {relative name: bytes} and folder_copy, a
                        private copy of the folder on disk for checks that need a directory
"""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

BASE_REPO = "Qwen/Qwen3.5-4B"


def canned_card(license_id: str | None = "apache-2.0") -> str:
    """A model card as the Hub serves one: YAML metadata, then text. license_id None leaves the key out."""
    meta = f"license: {license_id}\n" if license_id else ""
    return f"---\n{meta}pipeline_tag: image-text-to-text\n---\n\n# Qwen3.5-4B\n\nA canned card for tests.\n"


@dataclass
class Upload:
    repo: str
    private: bool
    files: dict[str, bytes]
    folder_copy: Path


@dataclass
class FakeHubClient:
    cards: dict[str, str] = field(default_factory=lambda: {BASE_REPO: canned_card()})
    events: list[tuple[str, str]] = field(default_factory=list)
    card_reads: list[str] = field(default_factory=list)
    uploads: list[Upload] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._scratch = tempfile.TemporaryDirectory(prefix="hf-fake-")

    def read_card(self, repo_id: str) -> str:
        self.events.append(("read_card", repo_id))
        self.card_reads.append(repo_id)
        if repo_id not in self.cards:
            raise LookupError(f"fake hub has no card for {repo_id}")
        return self.cards[repo_id]

    def upload_folder(self, repo_id: str, folder: Path, *, private: bool = False) -> str:
        self.events.append(("upload", repo_id))
        folder = Path(folder)
        copy = Path(self._scratch.name) / f"upload-{len(self.uploads)}"
        shutil.copytree(folder, copy)
        files = {p.relative_to(copy).as_posix(): p.read_bytes() for p in sorted(copy.rglob("*")) if p.is_file()}
        self.uploads.append(Upload(repo_id, private, files, copy))
        return f"https://huggingface.co/{repo_id}"
