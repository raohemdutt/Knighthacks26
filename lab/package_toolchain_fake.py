"""A stand-in for the trainer pod's packaging toolchain, for lab/package.py's tests and rehearsals.

    fake = FakeToolchain()
    package.build(..., toolchain=fake)
    fake.calls                       # every argv lab/package.py asked to run, in order

lab/package.py never runs a tool itself: it hands each command (an argv list) to a toolchain, whose run() answers with
a lab.package.ToolResult (returncode, stdout, stderr), and asks it for the llama.cpp commit. The real toolchain
(lab.package.Toolchain) runs them with subprocess on pod B. This one reads the same argv and does on disk what the
real tool leaves behind, small:

    python -m lab.package merge --base-dir B --adapter A --out M    M/config.json (the base's, copied), M/model.safetensors
                                                                    (placeholder bytes made from the base's config and the
                                                                    adapter's weights), M/merge.json (what was merged)
    python <llama.cpp>/convert_hf_to_gguf.py M --outtype bf16 --outfile F    F: b"GGUF" and bytes made from M's files
    <llama.cpp>/build/bin/llama-quantize F Q Q4_K_M                         Q: b"GGUF" and half of F's size, made from F
    git -C <llama.cpp> rev-parse HEAD                                       COMMIT

Placeholder bytes are a sha256 stream seeded by the inputs, so two different adapters give two different GGUFs and
the sha256 and size logic in lab/package.py run on real files. Nothing here imports torch or PEFT, runs a process or
touches the network. `fail_on` makes one step answer a non-zero code ("merge", "convert", "quantize" or "commit"),
as a failing tool would, and leaves no output behind.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from lab.package import ToolResult

__all__ = ["COMMIT", "FakeToolchain", "ToolResult", "placeholder"]

COMMIT = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c"
GGUF_MAGIC = b"GGUF"
CONVERTED_BYTES = 8192           # the fake bf16 GGUF; the Q4_K_M one is half of it, as 4.9 bits is about a third of 16


def placeholder(seed: bytes, size: int) -> bytes:
    """size bytes that depend on every byte of seed: a sha256 chain."""
    out = bytearray()
    block = hashlib.sha256(seed).digest()
    while len(out) < size:
        out += block
        block = hashlib.sha256(block).digest()
    return bytes(out[:size])


def _digest_folder(folder: Path) -> bytes:
    digest = hashlib.sha256()
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        digest.update(path.relative_to(folder).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.digest()


def _option(argv: Sequence[str], name: str) -> str:
    return argv[list(argv).index(name) + 1]


class FakeToolchain:
    """The packaging toolchain of pod B, on placeholder files."""

    def __init__(self, *, commit: str = COMMIT, fail_on: str | None = None) -> None:
        self.commit = commit
        self.fail_on = fail_on
        self.calls: list[list[str]] = []

    def llama_cpp_commit(self, llama_dir: str | Path) -> str:
        argv = ["git", "-C", str(llama_dir), "rev-parse", "HEAD"]
        result = self.run(argv)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or "git rev-parse failed")
        return result.stdout.strip()

    def run(self, argv: Sequence[str]) -> ToolResult:
        argv = [str(part) for part in argv]
        self.calls.append(argv)
        step = self._step(argv)
        if step is None:
            return ToolResult(127, stderr=f"fake toolchain: no such tool {argv[0]!r}")
        if step == self.fail_on:
            return ToolResult(1, stderr=f"fake toolchain: {step} failed as asked")
        return getattr(self, f"_{step}")(argv)

    @staticmethod
    def _step(argv: list[str]) -> str | None:
        if argv[:2] == ["git", "-C"] and argv[3:] == ["rev-parse", "HEAD"]:
            return "commit"
        if len(argv) > 3 and argv[1:4] == ["-m", "lab.package", "merge"]:
            return "merge"
        if len(argv) > 1 and Path(argv[1]).name == "convert_hf_to_gguf.py":
            return "convert"
        if Path(argv[0]).name == "llama-quantize":
            return "quantize"
        return None

    def _commit(self, argv: list[str]) -> ToolResult:
        return ToolResult(0, stdout=self.commit + "\n")

    def _merge(self, argv: list[str]) -> ToolResult:
        base, adapter, out = Path(_option(argv, "--base-dir")), Path(_option(argv, "--adapter")), Path(_option(argv, "--out"))
        weights = adapter / "adapter_model.safetensors"
        if not weights.is_file():
            return ToolResult(1, stderr=f"fake toolchain: no adapter weights at {weights}")
        out.mkdir(parents=True, exist_ok=True)
        config = (base / "config.json").read_bytes()
        (out / "config.json").write_bytes(config)
        (out / "model.safetensors").write_bytes(placeholder(config + weights.read_bytes(), 4096))
        (out / "merge.json").write_text(json.dumps({"base": str(base), "adapter": str(adapter), "dtype": "bfloat16"}))
        return ToolResult(0, stdout=f"merged {adapter} into {base} -> {out}\n")

    def _convert(self, argv: list[str]) -> ToolResult:
        merged, outfile = Path(argv[2]), Path(_option(argv, "--outfile"))
        if _option(argv, "--outtype") != "bf16":
            return ToolResult(2, stderr="fake toolchain: the packaging step converts in bf16 only")
        outfile.write_bytes(GGUF_MAGIC + placeholder(_digest_folder(merged), CONVERTED_BYTES - len(GGUF_MAGIC)))
        return ToolResult(0, stdout=f"Model successfully exported to {outfile}\n")

    def _quantize(self, argv: list[str]) -> ToolResult:
        source, target, kind = Path(argv[1]), Path(argv[2]), argv[3]
        data = source.read_bytes()
        if not data.startswith(GGUF_MAGIC):
            return ToolResult(1, stderr=f"fake toolchain: {source} is not a GGUF file")
        size = len(data) // 2
        target.write_bytes(GGUF_MAGIC + placeholder(data + kind.encode(), size - len(GGUF_MAGIC)))
        return ToolResult(0, stdout=f"llama_model_quantize_impl: model size = {len(data)} B, quant size = {size} B\n")

