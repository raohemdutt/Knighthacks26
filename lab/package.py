"""Package a trained LoRA adapter for the Jetson: merge in bf16, GGUF, Q4_K_M, the base mmproj, manifest.json (L5).

    python -m lab package build --adapter DIR --run-id RUN --task TASK --mmproj FILE
                                [--base Qwen/Qwen3.5-4B] [--base-dir ~/models/Qwen3.5-4B] [--llama-cpp ~/llama.cpp]
                                [--out ~/data/packages] [--version V] [--node pod-b] [--toolchain real|fake]
    python -m lab package scores --package DIR --run-id RUN
    python -m lab package dry-run --package DIR [--scratch DIR]
    python -m lab.package merge --base-dir DIR --adapter DIR --out DIR      (run by build on pod B, through the toolchain)

Run it from the repo root. Every call answers with one JSON object and an exit code, as the other lab verbs do: 0 when
it worked, 1 when it did not (`kind` says why), 2 for bad arguments.

The adapter folder. The trainer (L3) has no script on main yet, so this is the layout the step expects, the one PEFT's
save_pretrained writes: DIR/adapter_config.json (its base_model_name_or_path names the base, by id or by a local
folder of the same name) and DIR/adapter_model.safetensors (or adapter_model.bin). The adapter must train only the
language side: a target or saved module naming `visual` or `vision` is refused, because the package keeps the base's
vision projector unchanged.

build. Checks the inputs, then runs three commands through the toolchain, in this order:

    1. python -m lab.package merge --base-dir B --adapter A --out W/merged      PEFT merge_and_unload on the bf16 base
    2. python <llama.cpp>/convert_hf_to_gguf.py W/merged --outtype bf16 --outfile W/<version>-bf16.gguf
    3. <llama.cpp>/build/bin/llama-quantize W/<version>-bf16.gguf P/<version>-Q4_K_M.gguf Q4_K_M

W is <out>/.work-<version> (removed afterwards, whatever happens) and P the package being built inside it. A base
whose config.json carries a quantization_config is refused before anything runs: the merge goes into the unquantized
weights only. The base mmproj is copied into P unchanged (its sha256 checked against the source), manifest.json is
written last (jetson/make_manifest.py's folder rule: exactly one model gguf, at most one mmproj, a sha256 for each),
and P is renamed to <out>/<version>, which the pod's package server (lab/package_server.py) serves. An existing
<out>/<version> is never overwritten. The version defaults to atlas-<task>-<run id>.

Scores are never typed. The gate (L4) scores the package and writes a ledger row whose refs.run_id is the run id and
whose metrics carry learning.heldout.<name> (current, candidate, teacher, and dotted keys such as candidate.bf16).
`scores` copies them into manifest.json as {<name>: value}, from the newest such row of the run (a row whose verb
starts with `package` is never read as the gate's); a row that names another package version (conditions.version) is
refused. build does the same when the gate's row is already there, and otherwise leaves scores empty.

The run's ledger row (verb `package build`, refs.run_id, node pod-b): metrics learning.package_mb (the model gguf plus
the mmproj, in MB of 10^6 bytes), conditions {llama_cpp_commit, quant, base, version}, the fields commands (the three
argv lists), wall_seconds (the packaging wall-clock, from the commit read to the rename), version and files. A step that
fails writes the row with outcome fail, the commands run so far and the reason in note. `scores` writes a
`package scores` row whose source is the gate row's seq.

dry-run. Installs the package under <scratch>/data/models/<version>/ (a fresh temporary folder unless --scratch names
one; never the repo's own data/models), switches <scratch>/data/models/current to it with runtime/modelstore.py (which
checks every sha256) and reads it back through runtime/registry.py's model_version(), which must give the manifest's
version, base, task and scores.

The toolchain. build never runs a tool itself: it hands argv lists to a toolchain (Toolchain here, subprocess on the
pod; lab/package_toolchain_fake.py in tests and rehearsals, --toolchain fake) and asks it for the llama.cpp commit.
lab/L5_REAL_RUN.md has the commands for the real run on pod B.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

if __package__ in (None, ""):          # started as `python lab/package.py`: put the repo root on the path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import package_server
from lab.machine import ledger
from lab.machine.record import record

__all__ = ["PackageError", "Toolchain", "ToolResult", "build", "copy_scores", "dry_run", "folder_manifest",
           "gate_scores", "merge_adapter", "main"]

CLI_VERB = "package"   # registers python -m lab package
REPO = Path(__file__).resolve().parents[1]
DEFAULT_BASE = "Qwen/Qwen3.5-4B"
QUANT = "Q4_K_M"
MANIFEST = package_server.MANIFEST
SCORE_PREFIX = "learning.heldout."
GATE_VERB, GATE_PREFIX = "eval-gate", "eval."   # the decision row lab/eval_gate.py writes, and its metric names
GATE_SCORE_NAMES = ("current", "candidate", "teacher")   # contracts/learning.py: the scores of a run's evaluation
BUILD_ROW, SCORES_ROW = "package build", "package scores"
PACKAGE_MB = "learning.package_mb"
ADAPTER_CONFIG = "adapter_config.json"
ADAPTER_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")
VISION_WORDS = ("visual", "vision")
SIDE_FILE_SUFFIXES = (".json", ".jinja", ".txt", ".model")    # tokenizer, chat template, processor configs


class PackageError(Exception):
    """A package that cannot be built, scored or installed. kind says why, in one word."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


# --- the toolchain ---------------------------------------------------------------------------------------------

@dataclasses.dataclass
class ToolResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class Runner(Protocol):
    def run(self, argv: Sequence[str]) -> ToolResult: ...

    def llama_cpp_commit(self, llama_dir: str | Path) -> str: ...


class Toolchain:
    """The real toolchain: each command as a process from the repo root, its output captured."""

    def run(self, argv: Sequence[str]) -> ToolResult:
        done = subprocess.run([str(part) for part in argv], cwd=REPO, capture_output=True, text=True, check=False)
        return ToolResult(done.returncode, done.stdout, done.stderr)

    def llama_cpp_commit(self, llama_dir: str | Path) -> str:
        result = self.run(["git", "-C", str(llama_dir), "rev-parse", "HEAD"])
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "git rev-parse failed")
        return result.stdout.strip()


def _toolchain(name: str) -> Runner:
    if name == "fake":
        from lab.package_toolchain_fake import FakeToolchain
        return FakeToolchain()
    return Toolchain()


# --- checks before anything runs -------------------------------------------------------------------------------

def _read_json(path: Path, kind: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PackageError(kind, f"cannot read {path}: {exc}") from None
    if not isinstance(data, dict):
        raise PackageError(kind, f"{path} is not a JSON object")
    return data


def check_base(base_dir: Path) -> None:
    """The base checkpoint is a Hugging Face folder of unquantized weights."""
    config = _read_json(Path(base_dir) / "config.json", "bad_base")
    if config.get("quantization_config"):
        raise PackageError("quantized_base", f"{base_dir} is a quantized checkpoint: the adapter is merged into the "
                                             "unquantized bf16 base only")


def check_adapter(adapter: Path, base: str) -> None:
    config = _read_json(Path(adapter) / ADAPTER_CONFIG, "bad_adapter")
    if not any((Path(adapter) / name).is_file() for name in ADAPTER_WEIGHTS):
        raise PackageError("bad_adapter", f"{adapter} has no {' or '.join(ADAPTER_WEIGHTS)}")
    named = str(config.get("base_model_name_or_path") or "")
    if named and named != base and Path(named).name != base.split("/")[-1]:
        raise PackageError("wrong_base", f"the adapter was trained on {named!r}, not {base!r}")
    modules = []
    for key in ("target_modules", "modules_to_save"):
        value = config.get(key) or []
        modules += [value] if isinstance(value, str) else list(value)
    vision = [m for m in modules if any(word in str(m).lower() for word in VISION_WORDS)]
    if vision:
        raise PackageError("vision_adapter", f"the adapter touches vision modules {vision}: the package keeps the base "
                                             "projector unchanged, so only a language-side adapter can be packaged")


def check_mmproj(mmproj: Path) -> None:
    if not Path(mmproj).is_file():
        raise PackageError("bad_mmproj", f"no base projector at {mmproj}")
    if not (Path(mmproj).name.startswith("mmproj") and Path(mmproj).suffix == ".gguf"):
        raise PackageError("bad_mmproj", f"{Path(mmproj).name}: the projector's file name must be mmproj*.gguf, "
                                         "the name jetson/make_manifest.py and the box tell it apart by")


def check_version(version: str) -> str:
    from lab.transfer import RESERVED_VERSIONS
    if not package_server.NAME_RE.match(version) or version in RESERVED_VERSIONS:
        raise PackageError("bad_version", f"{version!r} cannot be a version: letters, digits, dot, dash and underscore, "
                                          f"not one of {sorted(RESERVED_VERSIONS)}")
    return version


# --- the manifest ----------------------------------------------------------------------------------------------

def _make_manifest_module() -> Any:
    """jetson/make_manifest.py, which is a script beside the box's files, not a package."""
    spec = importlib.util.spec_from_file_location("make_manifest", REPO / "jetson" / "make_manifest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def folder_manifest(folder: Path, *, base: str, version: str, task: str | None) -> dict[str, Any]:
    """The manifest jetson/make_manifest.py writes for the folder, its folder rule included."""
    try:
        return _make_manifest_module().build_manifest(Path(folder), base, version, task)
    except SystemExit as exc:
        raise PackageError("folder_rule", str(exc)) from None


def _validated(manifest: Mapping[str, Any]) -> dict[str, Any]:
    from contracts.learning import ModelManifest
    try:
        return ModelManifest.model_validate(manifest).model_dump(mode="json")
    except ValueError as exc:
        raise PackageError("bad_manifest", f"manifest.json would not validate as ModelManifest: {exc}") from None


def _write_manifest(folder: Path, manifest: Mapping[str, Any]) -> None:
    text = json.dumps(_validated(manifest), indent=2) + "\n"
    part = folder / (MANIFEST + ".tmp")
    part.write_text(text, encoding="utf-8")
    os.replace(part, folder / MANIFEST)


def _read_manifest(folder: Path) -> dict[str, Any]:
    return _validated(_read_json(Path(folder) / MANIFEST, "bad_manifest"))


# --- the gate's scores -----------------------------------------------------------------------------------------

def gate_scores(run_id: str, *, state_dir: str | os.PathLike[str] | None = None
                ) -> tuple[dict[str, float] | None, dict[str, Any] | None]:
    """The scores of the newest gate row of the run, as {current: .., candidate: .., ...}, and the row; (None, None)
    when the gate has not scored the run. A gate row is the decision row of `python -m lab eval-gate` (verb eval-gate,
    metrics eval.current, eval.candidate and eval.teacher; its per-score rows, verb eval-gate.score, and a refused
    run carry no decision and are not one), or any other row whose metrics are named learning.heldout.<name>."""
    rows = [(row, scores) for row in ledger.read(state_dir=state_dir, refs={"run_id": run_id})
            if not str(row.get("verb", "")).startswith("package") and (scores := _row_scores(row))]
    if not rows:
        return None, None
    row, scores = rows[-1]
    return scores, row


def _row_scores(row: Mapping[str, Any]) -> dict[str, float]:
    metrics = row.get("metrics") or {}
    if row.get("verb") == GATE_VERB and all(GATE_PREFIX + name in metrics for name in ("current", "candidate")):
        return {name: float(metrics[GATE_PREFIX + name]) for name in GATE_SCORE_NAMES if GATE_PREFIX + name in metrics}
    return {name[len(SCORE_PREFIX):]: float(value) for name, value in metrics.items() if name.startswith(SCORE_PREFIX)}


def _check_gate_version(row: Mapping[str, Any], version: str) -> None:
    scored = (row.get("conditions") or {}).get("version") or row.get("version")
    if scored and scored != version:
        raise PackageError("gate_mismatch", f"the gate's row (seq {row.get('seq')}) scored package {scored!r}, "
                                            f"not {version!r}")


# --- build -----------------------------------------------------------------------------------------------------

def commands(*, base_dir: Path, adapter: Path, llama_dir: Path, work: Path, staging: Path, version: str
             ) -> tuple[list[str], list[str], list[str]]:
    merged, bf16 = work / "merged", work / f"{version}-bf16.gguf"
    return ([sys.executable, "-m", "lab.package", "merge", "--base-dir", str(base_dir), "--adapter", str(adapter),
             "--out", str(merged)],
            [sys.executable, str(llama_dir / "convert_hf_to_gguf.py"), str(merged), "--outtype", "bf16",
             "--outfile", str(bf16)],
            [str(llama_dir / "build" / "bin" / "llama-quantize"), str(bf16), str(staging / f"{version}-{QUANT}.gguf"),
             QUANT])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build(*, adapter: str | Path, run_id: str, task: str, mmproj: str | Path, base_dir: str | Path,
          llama_dir: str | Path, out_root: str | Path, base: str = DEFAULT_BASE, version: str | None = None,
          toolchain: Runner | None = None, node: str = "pod-b", state_dir: str | os.PathLike[str] | None = None,
          clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    """Merge, convert and quantize the adapter, keep the base mmproj, write manifest.json; one ledger row."""
    adapter, mmproj, base_dir, llama_dir, out_root = map(Path, (adapter, mmproj, base_dir, llama_dir, out_root))
    toolchain = toolchain or Toolchain()
    version = check_version(version or f"atlas-{task}-{run_id}")
    check_base(base_dir)
    check_adapter(adapter, base)
    check_mmproj(mmproj)
    folder, work = out_root / version, out_root / f".work-{version}"
    if folder.exists():
        raise PackageError("exists", f"{folder} exists: a package is never overwritten; pick another --version")
    scores, gate = gate_scores(run_id, state_dir=state_dir)
    if gate is not None:
        _check_gate_version(gate, version)
    staging = work / "package"
    planned = commands(base_dir=base_dir, adapter=adapter, llama_dir=llama_dir, work=work, staging=staging,
                       version=version)
    ran: list[list[str]] = []
    conditions: dict[str, Any] = {"quant": QUANT, "base": base, "version": version}
    started = clock()
    try:
        try:
            conditions["llama_cpp_commit"] = toolchain.llama_cpp_commit(llama_dir)
        except Exception as exc:
            raise PackageError("toolchain", f"cannot read the llama.cpp commit in {llama_dir}: {exc}") from None
        shutil.rmtree(work, ignore_errors=True)
        staging.mkdir(parents=True)
        for step, argv in zip(("merge", "convert", "quantize"), planned):
            ran.append(argv)
            result = toolchain.run(argv)
            if result.returncode != 0:
                tail = (result.stderr or result.stdout).strip().splitlines()[-1:] or [""]
                raise PackageError("toolchain", f"{step} exited {result.returncode}: {tail[0]}")
        shutil.copyfile(mmproj, staging / mmproj.name)
        if _sha256(staging / mmproj.name) != _sha256(mmproj):
            raise PackageError("bad_mmproj", f"the copy of {mmproj.name} does not match its source")
        manifest = folder_manifest(staging, base=base, version=version, task=task)
        manifest["scores"] = scores or {}
        _write_manifest(staging, manifest)
        os.replace(staging, folder)
    except PackageError as exc:
        record(BUILD_ROW, "fail", refs={"run_id": run_id}, node=node, conditions=conditions, commands=ran,
               version=version, note=f"{exc.kind}: {exc.message}"[:300], state_dir=state_dir)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)
    wall = round(clock() - started, 3)
    files = {name: (folder / name).stat().st_size for name in manifest["sha256"]}
    package_mb = round(sum(files.values()) / 1e6, 6)
    seq = record(BUILD_ROW, "ok", metrics={PACKAGE_MB: package_mb}, refs={"run_id": run_id}, node=node,
                 units={PACKAGE_MB: "MB"}, conditions=conditions, commands=ran, wall_seconds=wall, version=version,
                 files=files, state_dir=state_dir)
    return {"ok": True, "outcome": "ok", "version": version, "folder": str(folder), "package_mb": package_mb,
            "wall_seconds": wall, "llama_cpp_commit": conditions["llama_cpp_commit"], "commands": ran,
            "scores": scores or {}, "gate_seq": gate["seq"] if gate else None, "ledger_seq": seq}


# --- scores ----------------------------------------------------------------------------------------------------

def copy_scores(package_dir: str | Path, run_id: str, *, state_dir: str | os.PathLike[str] | None = None
                ) -> dict[str, Any]:
    """Copy the gate's scores for the run into the package's manifest.json; one ledger row."""
    package_dir = Path(package_dir)
    manifest = _read_manifest(package_dir)
    scores, gate = gate_scores(run_id, state_dir=state_dir)
    if gate is None:
        raise PackageError("no_gate_row", f"no gate row with refs.run_id {run_id!r} in the ledger (an {GATE_VERB} "
                                          f"decision row, or {SCORE_PREFIX}* metrics): the gate (L4) has not scored "
                                          "this package yet")
    _check_gate_version(gate, manifest["version"])
    manifest["scores"] = scores
    _write_manifest(package_dir, manifest)
    seq = record(SCORES_ROW, "ok", refs={"run_id": run_id}, conditions={"version": manifest["version"]},
                 source=f"ledger seq {gate['seq']}", state_dir=state_dir)
    return {"ok": True, "outcome": "ok", "version": manifest["version"], "scores": scores, "gate_seq": gate["seq"],
            "ledger_seq": seq}


# --- the dry run -----------------------------------------------------------------------------------------------

def dry_run(package_dir: str | Path, scratch: str | Path) -> dict[str, Any]:
    """Install the package under <scratch>/data/models/<version>/ and read it back through model_version()."""
    from runtime.modelstore import ModelStore, ModelStoreError
    from runtime.registry import model_version

    package_dir = Path(package_dir)
    models = Path(scratch) / "data" / "models"
    if models.resolve() == (REPO / "data" / "models").resolve():
        raise PackageError("not_scratch", f"{models} is the repo's own data/models: a dry run installs off the box")
    manifest = _read_manifest(package_dir)
    version = manifest["version"]
    installed = models / version
    shutil.rmtree(installed, ignore_errors=True)
    installed.mkdir(parents=True)
    for name in [MANIFEST, manifest["gguf"], *([manifest["mmproj"]] if manifest["mmproj"] else [])]:
        shutil.copyfile(package_dir / name, installed / name)
    try:
        ModelStore(models).activate(version)
    except ModelStoreError as exc:
        raise PackageError("verify", str(exc)) from None
    found = model_version({"manifest": str(models / "current" / MANIFEST)})
    read_back = {"version": found.version, "base": found.base, "task": found.task, "scores": dict(found.scores)}
    expected = {key: manifest[key] for key in ("version", "base", "task", "scores")}
    if read_back != expected:
        raise PackageError("read_back", f"model_version() read back {read_back}, the manifest says {expected}")
    return {"ok": True, "outcome": "ok", "installed": str(installed), "read_back": read_back}


# --- the merge, as it runs on pod B ----------------------------------------------------------------------------

def merge_adapter(base_dir: str | Path, adapter: str | Path, out: str | Path) -> dict[str, Any]:
    """PEFT merge_and_unload of the adapter into the bf16 base, saved as a Hugging Face folder for the converter."""
    base_dir, out = Path(base_dir), Path(out)
    check_base(base_dir)
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16, device_map={"": "cpu"})
    merged = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(base_dir)).save_pretrained(str(out))
    copied = []
    for path in sorted(base_dir.iterdir()):          # chat template and processor files the converter reads
        if path.is_file() and path.suffix in SIDE_FILE_SUFFIXES and "index" not in path.name \
                and not (out / path.name).exists():
            shutil.copyfile(path, out / path.name)
            copied.append(path.name)
    return {"ok": True, "outcome": "ok", "merged": str(out), "copied": copied}


# --- the verb --------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lab package", description=main.__doc__)
    verbs = parser.add_subparsers(dest="verb", required=True)
    home = Path.home()
    b = verbs.add_parser("build", help="merge, convert, quantize and write the package")
    b.add_argument("--adapter", required=True)
    b.add_argument("--run-id", required=True)
    b.add_argument("--task", required=True)
    b.add_argument("--mmproj", required=True, help="the base's vision projector, mmproj*.gguf")
    b.add_argument("--base", default=DEFAULT_BASE)
    b.add_argument("--base-dir", default=str(home / "models" / DEFAULT_BASE.split("/")[-1]))
    b.add_argument("--llama-cpp", default=str(home / "llama.cpp"))
    b.add_argument("--out", default=str(home / "data" / "packages"))
    b.add_argument("--version")
    b.add_argument("--node", default="pod-b")
    b.add_argument("--toolchain", choices=("real", "fake"), default="real")
    s = verbs.add_parser("scores", help="copy the gate's scores for the run into manifest.json")
    s.add_argument("--package", required=True)
    s.add_argument("--run-id", required=True)
    d = verbs.add_parser("dry-run", help="install under a scratch data/models/ and read it back")
    d.add_argument("--package", required=True)
    d.add_argument("--scratch")
    m = verbs.add_parser("merge", help="PEFT merge_and_unload into the bf16 base (run by build on pod B)")
    m.add_argument("--base-dir", required=True)
    m.add_argument("--adapter", required=True)
    m.add_argument("--out", required=True)
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Package a LoRA adapter for the Jetson: merge, GGUF, Q4_K_M, manifest, scores, dry-run install."""
    args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.verb == "build":
            return build(adapter=args.adapter, run_id=args.run_id, task=args.task, mmproj=args.mmproj, base=args.base,
                         base_dir=args.base_dir, llama_dir=args.llama_cpp, out_root=args.out, version=args.version,
                         node=args.node, toolchain=_toolchain(args.toolchain))
        if args.verb == "scores":
            return copy_scores(args.package, args.run_id)
        if args.verb == "dry-run":
            if args.scratch:
                return dry_run(args.package, args.scratch)
            with tempfile.TemporaryDirectory(prefix="l5-dry-run-") as scratch:
                return dry_run(args.package, scratch)
        return merge_adapter(args.base_dir, args.adapter, args.out)
    except PackageError as exc:
        return {"ok": False, "outcome": "error", "kind": exc.kind, "error": exc.message}


if __name__ == "__main__":
    answer = main()
    print(json.dumps(answer))
    sys.exit(0 if answer.get("ok", True) else 1)
