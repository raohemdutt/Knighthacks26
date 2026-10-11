"""Group fine-tune (L10): one LoRA trained on the pooled opt-in exports of a Circle's members, offered to every box.

    python -m lab group-finetune --circle C --task T [--pool ~/pool] [--run-id ID] [--work DIR] [--packages DIR]
        --teacher-url URL [--teacher-token-env AUP_HUB_TOKEN_C] --model ~/models/Qwen3.5-4B --mmproj FILE
        [--llama-cpp ~/llama.cpp] [--trainer-python PY] [--epochs 3 --lr 2e-4 --rank 16 ...] [--min-held-out 50]
        [--decode-attested] [--keep] [--dry-run]
    python -m lab group-finetune pool-add --circle C --task T --member M --export data/exports/<dataset_id> [--pool DIR]
    python -m lab group-finetune take --offer group_offer.json --member M [--account D] [--rollback]

The pool. Each member's box makes its own approved export (contexts/learning/application/export.py: an approval card,
then data/exports/<dataset_id>/ with the approval's action id in its DatasetManifest and blurred keyframes only), and
that member opts in to the group run by uploading it to the network GPU (pod D) under

    ~/pool/<circle>/<task>/pool.json                       the roster: every member of the Circle, uploader or not
    ~/pool/<circle>/<task>/<member>/<dataset_id>/          one approved export per folder, as the box wrote it

pool.json is {"version": 1, "circle", "task", "members": [{"member", "current_version", "current_url"?, "exports":
{dataset_id: examples}}]}: who is in the Circle, which model each box serves now (the rollback version), and how many
examples each member's exports hold. read_pool() reads the folders through lab/dataset.py load() (an export without
its approval's action id, or whose examples do not hold up, is refused), refuses an export of another task and a
member folder that is not on the roster, and counts. add_export() is how an export lands there off the pod.

The run, in STEPS order. Every external call goes through an object a test replaces with a fake:

    pool      read and count the pool, write the counts into pool.json
    merge     one dataset at <work>/data/exports/<pooled id>/: every example, its id, frame_ref and image prefixed
              by its member, the manifest's action_id listing each member's approval; episodes.json maps each
              frame_ref to itself (one episode per frame), which the gate needs for an image task
    split     split.json frozen per member: whole frame_ref groups in the order of their sha256, a quarter of each
              uploader's groups (at least one) held out, so every uploader's data is on the held-out side
    label     the teacher labels the pooled set through lab/label.py label_dataset(), the code path L13 used
    rows      one training row per labeled train example in the student's chat format (picture, instruction, the
              teacher's answer after an empty thinking block); held-out examples never get a row
    train     one LoRA on those rows with the L3 trainer (lab/train.py, the same flags and recorded config) on a copy
              of the pooled dataset, which the trainer deletes on the pod
    package   the L5 step (lab/package.py build): merge, GGUF, Q4_K_M, the base mmproj, manifest.json
    gate      the L4 gate (lab/eval_gate.py run_gate) once per member: that member's current model against the
              group model and the teacher on the pooled held-out set, run id <run id>.<member>
              (a desk-objects run is scored by lab/desk_score.py through DeskGateClient, the gate having no
              registered desk scorer; the teacher's score carries the hub token)
    offer     <package>/group_offer.json: every member of the Circle (an uploader or not) with their gate decision,
              scores and rollback version; the package server serves it beside the model
    delete    the pooled exports and every copy of them are deleted on every way out (--keep keeps them, off a pod
              only); a row records it

What it records. A group-finetune.run ledger row (members, counts, run id, the training config, scores per member),
a group-finetune.delete row, and metrics rows (lab/refine/metrics.jsonl unless --metrics-file): the pooled example
count and, per member, the group model's score and its delta over that member's current model.

Labeling on the pod. lab/label.py refuses an export on a pod because the copy would outlive the run. This verb
deletes the pooled copies itself on every way out, so it asks label.py to label them there; --keep is refused on a
pod for that reason.

take. A box takes the offer: it must be on the offer's roster (uploaded or not), pulls the package
(python -m lab.package_pull pull), swaps to it (python jetson/atlas_swap.py swap, which checks every sha256 and
falls back itself), and on a failed swap rolls back to its version before. --rollback goes back on purpose.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ContextManager, Protocol

import httpx

from contracts.learning import DatasetExample, DatasetManifest
from lab import dataset as ds
from lab import hub
from lab import eval_gate, package
from lab.label import Decode, LabelRefused, Teacher, label_dataset, running_on_pod
from lab.machine import metrics as metrics_file
from lab.machine.record import record

__all__ = ["CLI_VERB", "POOL_FILE", "OFFER_FILE", "STEPS", "GroupError", "Export", "Member", "Pool", "Merged",
           "TrainConfig", "Trainer", "RealTrainer", "DeskGateClient", "gate_client_for", "Servers", "RealServers",
           "Box", "RealBox", "read_pool",
           "add_export", "merge_pool", "freeze_split", "image_rows", "run_group_finetune", "plan", "take_offer",
           "main"]

CLI_VERB = "group-finetune"   # registers python -m lab group-finetune
REPO = Path(__file__).resolve().parents[1]
POOL_FILE = "pool.json"
OFFER_FILE = "group_offer.json"
EPISODES_FILE = "episodes.json"
STEPS = ("pool", "merge", "split", "label", "rows", "train", "package", "gate", "offer", "delete")
HELD_OUT_SHARE = 0.25
DEFAULT_POOL = "~/pool"
DEFAULT_WORK = "~/group-runs"
DEFAULT_PACKAGES = "~/data/packages"
DEFAULT_CURRENT = "atlas-base"
NODE = "pod-b"                 # the trainer's role (SJC account D), as lab/train.py and lab/package.py name it
SOURCE = "lab/group_finetune.py"
IM_START, IM_END = "<|im_start|>", "<|im_end|>"
IMAGE = "<|vision_start|><|image_pad|><|vision_end|>"
EMPTY_THINK = "<think>\n\n</think>\n\n"
DESK_TASK = "desk-objects"      # an image task the gate has no registered scorer for; DeskGateClient scores it


class GroupError(RuntimeError):
    """The group run cannot go ahead; str() says why and never quotes a caption or a label."""


# --- the pool -------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Export:
    member: str
    dataset_id: str
    root: Path
    examples: int
    action_id: str


@dataclass(frozen=True)
class Member:
    member: str
    current_version: str = DEFAULT_CURRENT
    current_url: str | None = None
    exports: tuple[Export, ...] = ()

    @property
    def examples(self) -> int:
        return sum(e.examples for e in self.exports)


@dataclass(frozen=True)
class Pool:
    circle: str
    task: str
    folder: Path
    members: tuple[Member, ...]

    @property
    def uploaders(self) -> list[str]:
        return [m.member for m in self.members if m.exports]

    @property
    def examples(self) -> int:
        return sum(m.examples for m in self.members)

    def counts(self) -> dict[str, int]:
        return {m.member: m.examples for m in self.members}


def pool_folder(pool_root: str | os.PathLike[str], circle: str, task: str) -> Path:
    for name, value in (("circle", circle), ("task", task)):
        if not package.package_server.NAME_RE.match(value):
            raise GroupError(f"{name} {value!r} is not a folder name (letters, digits, dot, dash, underscore)")
    return Path(pool_root).expanduser() / circle / task


def _read_roster(folder: Path, circle: str, task: str) -> dict[str, Any]:
    path = folder / POOL_FILE
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise GroupError(f"no {POOL_FILE} in {folder}: the Circle's roster is missing") from None
    except ValueError:
        raise GroupError(f"{path} is not JSON") from None
    if body.get("circle") != circle or body.get("task") != task:
        raise GroupError(f"{POOL_FILE} is for circle {body.get('circle')!r} task {body.get('task')!r}, "
                         f"not {circle!r} {task!r}")
    members = body.get("members")
    if not isinstance(members, list) or not members or not all(isinstance(m, dict) and m.get("member")
                                                                for m in members):
        raise GroupError(f"{POOL_FILE} lists no members")
    names = [m["member"] for m in members]
    if len(set(names)) != len(names):
        raise GroupError(f"{POOL_FILE} lists a member twice")
    return body


def read_pool(pool_root: str | os.PathLike[str], circle: str, task: str) -> Pool:
    """The roster and every member's approved exports for the task. GroupError naming every problem."""
    folder = pool_folder(pool_root, circle, task)
    body = _read_roster(folder, circle, task)
    roster = {m["member"]: m for m in body["members"]}
    problems: list[str] = []
    found: dict[str, list[Export]] = {name: [] for name in roster}
    for sub in sorted(p for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []:
        if sub.name not in roster:
            problems.append(f"{sub.name}: has uploads but is not a member of the Circle's roster")
            continue
        for export in sorted(p for p in sub.iterdir() if p.is_dir()):
            try:
                data = ds.load(export, require_images=True)
            except ds.DatasetError as err:
                problems.append(f"{sub.name}/{export.name}: {err}")
                continue
            if data.manifest.spec.task != task:
                problems.append(f"{sub.name}/{export.name}: an export of task {data.manifest.spec.task!r}, not {task!r}")
                continue
            if export.name != data.dataset_id:
                problems.append(f"{sub.name}/{export.name}: the folder is not named for its dataset id")
                continue
            found[sub.name].append(Export(sub.name, data.dataset_id, export, len(data.examples),
                                          data.manifest.action_id))
    if problems:
        raise GroupError("the pool does not hold up: " + "; ".join(problems))
    members = tuple(Member(name, roster[name].get("current_version") or DEFAULT_CURRENT,
                           roster[name].get("current_url"), tuple(found[name])) for name in roster)
    return Pool(circle, task, folder, members)


def _write_json(path: Path, body: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_counts(pool: Pool, **extra: Any) -> None:
    """pool.json with each member's exports and example counts as they are on disk now."""
    body = _read_roster(pool.folder, pool.circle, pool.task)
    by_name = {m.member: m for m in pool.members}
    for entry in body["members"]:
        member = by_name[entry["member"]]
        entry["exports"] = {e.dataset_id: e.examples for e in member.exports}
    body.update(extra)
    _write_json(pool.folder / POOL_FILE, body)


def add_export(pool_root: str | os.PathLike[str], circle: str, task: str, member: str,
               export: str | os.PathLike[str]) -> dict[str, Any]:
    """Copy one approved export into the pool under the member's folder and count it. The member must be on the
    roster; the export must hold up (lab/dataset.py) and be of the task."""
    folder = pool_folder(pool_root, circle, task)
    body = _read_roster(folder, circle, task)
    if member not in {m["member"] for m in body["members"]}:
        raise GroupError(f"{member!r} is not a member of the Circle's roster")
    try:
        data = ds.load(export, require_images=True)
    except ds.DatasetError as err:
        raise GroupError(f"the export does not hold up: {err}") from None
    if data.manifest.spec.task != task:
        raise GroupError(f"the export is of task {data.manifest.spec.task!r}, not {task!r}")
    dest = folder / member / data.dataset_id
    if dest.exists():
        raise GroupError(f"{member}/{data.dataset_id} is already in the pool")
    shutil.copytree(export, dest, symlinks=False)
    pool = read_pool(pool_root, circle, task)
    write_counts(pool)
    return {"ok": True, "outcome": "added", "member": member, "dataset_id": data.dataset_id,
            "examples": len(data.examples), "counts": pool.counts()}


# --- merge and split -------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Merged:
    root: Path
    dataset_id: str
    member_of: dict[str, str]          # pooled example id -> member
    episodes: Path


def merge_pool(pool: Pool, exports_dir: str | os.PathLike[str], *, run_id: str,
               now: dt.datetime | None = None) -> Merged:
    """One dataset holding every uploaded example, ids, frame_refs and images prefixed by member, with its split."""
    if len(pool.uploaders) < 1:
        raise GroupError("no member uploaded an export for this task")
    specs = {(e.member, e.dataset_id): ds.load(e.root).manifest.spec for m in pool.members for e in m.exports}
    instructions = {s.instruction for s in specs.values()}
    if len(instructions) != 1:
        raise GroupError(f"the exports ask {len(instructions)} different instructions for the task; a group run "
                         "trains on one")
    dataset_id = f"group-{pool.circle}-{pool.task}-{run_id}"
    package.check_version(dataset_id)
    root = Path(exports_dir) / dataset_id
    if root.exists():
        raise GroupError(f"{root} exists: a pooled dataset is never overwritten")
    (root / ds.IMAGES).mkdir(parents=True)
    examples: list[DatasetExample] = []
    member_of: dict[str, str] = {}
    approvals: list[str] = []
    for member in pool.members:
        for export in member.exports:
            data = ds.load(export.root)
            approvals.append(f"{member.member}:{data.manifest.action_id}")
            copied: dict[str, str] = {}
            for example in data.examples:
                if example.image not in copied:
                    suffix = Path(example.image).suffix
                    name = f"{ds.IMAGES}/{member.member}-{export.dataset_id}-{len(copied):05d}{suffix}"
                    shutil.copyfile(data.image_path(example), root / name)
                    copied[example.image] = name
                new_id = f"{member.member}--{example.id}"
                member_of[new_id] = member.member
                examples.append(example.model_copy(update={"id": new_id, "image": copied[example.image],
                                                           "frame_ref": f"{member.member}:{example.frame_ref}"}))
    spec = next(iter(specs.values()))
    manifest = DatasetManifest(dataset_id=dataset_id, spec=spec, created=now or dt.datetime.now(dt.timezone.utc),
                               examples=len(examples), action_id=";".join(approvals))
    (root / ds.EXAMPLES).write_text("".join(e.model_dump_json() + "\n" for e in examples), encoding="utf-8",
                                    newline="\n")
    (root / ds.MANIFEST).write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    episodes = root.parent / f"{dataset_id}.{EPISODES_FILE}"
    _write_json(episodes, {e.frame_ref: e.frame_ref for e in examples})
    freeze_split(root, member_of)
    return Merged(root, dataset_id, member_of, episodes)


def _order(frame_ref: str) -> str:
    return hashlib.sha256(frame_ref.encode("utf-8")).hexdigest()


def freeze_split(root: Path, member_of: Mapping[str, str], share: float = HELD_OUT_SHARE) -> dict[str, Any]:
    """split.json, once: per member, whole frame_ref groups in sha256 order, ceil(share x groups) of them held out
    (at least one, and never every group of a member who has two or more)."""
    data = ds.load(root, require_images=False)
    groups: dict[str, dict[str, list[str]]] = {}
    for example in data.examples:
        groups.setdefault(member_of[example.id], {}).setdefault(example.frame_ref, []).append(example.id)
    train: list[str] = []
    held: list[str] = []
    for member in sorted(groups):
        refs = sorted(groups[member], key=_order)
        n_held = max(1, math.ceil(share * len(refs)))
        if len(refs) > 1:
            n_held = min(n_held, len(refs) - 1)
        for index, ref in enumerate(refs):
            (held if index < n_held else train).extend(sorted(groups[member][ref]))
    rule = (f"Per member: whole frame_ref groups in the order of the sha256 of their frame_ref, the first "
            f"ceil({share} x groups) (at least one, never all) held out, the rest train. Written once by "
            "python -m lab group-finetune.")
    body = {"version": 1, "dataset_id": data.dataset_id, "rule": rule, "train": train, "held_out": held}
    with open(root / ds.SPLIT, "x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(body, indent=2) + "\n")
    return body


# --- training rows ------------------------------------------------------------------------------------------------------

def image_rows(split: ds.Split) -> list[dict[str, Any]]:
    """One row per labeled train example, in the student's chat format with thinking off: the picture and the
    instruction as the user turn (what the box sends), the teacher's label as the answer. Loss on the answer only."""
    instruction = split.data.manifest.spec.instruction
    rows = []
    for example in split.train:
        if not ds.has_target(example):
            continue
        prompt = f"{IM_START}user\n{IMAGE}{instruction}{IM_END}\n{IM_START}assistant\n{EMPTY_THINK}"
        answer = f"{example.target.strip()}{IM_END}"
        text = prompt + answer + "\n"
        messages = [{"role": "user", "content": [{"type": "image", "image": example.image},
                                                 {"type": "text", "text": instruction}]},
                    {"role": "assistant", "content": example.target.strip()}]
        rows.append({"id": example.id, "frame_ref": example.frame_ref, "messages": messages, "text": text,
                     "spans": [[len(prompt), len(prompt) + len(answer)]], "images": [example.image]})
    return rows


# --- the outside calls ---------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    """The L3 trainer's flags this run sets; their defaults are lab/train.py's (R36's pinned baseline)."""

    epochs: float = 3.0
    lr: float = 2e-4
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    batch: int = 4
    grad_accum: int = 4
    seed: int = 42
    max_steps: int | None = None
    base_revision: str | None = None

    def fields(self) -> dict[str, Any]:
        return asdict(self)

    def argv(self) -> list[str]:
        out = ["--epochs", str(self.epochs), "--lr", str(self.lr), "--rank", str(self.rank), "--alpha",
               str(self.alpha), "--dropout", str(self.dropout), "--batch", str(self.batch), "--grad-accum",
               str(self.grad_accum), "--seed", str(self.seed)]
        if self.max_steps is not None:
            out += ["--max-steps", str(self.max_steps)]
        if self.base_revision:
            out += ["--base-revision", self.base_revision]
        return out


class Trainer(Protocol):
    def train(self, *, dataset: Path, rows: Path, model: str, out: Path, run_id: str,
              config: TrainConfig) -> dict[str, Any]:
        """Train one LoRA; the trainer's one JSON answer (ok, outcome, refs, metrics). The adapter is <out>/adapter."""


def _last_json(text: str) -> dict[str, Any] | None:
    for line in reversed(text.strip().splitlines()):
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


class RealTrainer:
    """lab/train.py as a subprocess of this checkout, with python (default: this interpreter)."""

    def __init__(self, python: str | None = None, extra: Sequence[str] = ()) -> None:
        self.python = python or sys.executable
        self.extra = list(extra)

    def command(self, *, dataset: Path, rows: Path, model: str, out: Path, run_id: str,
                config: TrainConfig) -> list[str]:
        return [self.python, "-u", "lab/train.py", "--dataset", str(dataset), "--rows", str(rows), "--model", model,
                "--out", str(out), "--run-id", run_id, *config.argv(), *self.extra]

    def train(self, **kwargs: Any) -> dict[str, Any]:
        done = subprocess.run(self.command(**kwargs), cwd=REPO, stdout=subprocess.PIPE, stderr=sys.stderr, text=True,
                              check=False)
        return _last_json(done.stdout) or {"ok": False, "outcome": "failed",
                                           "error": f"the trainer exited {done.returncode} with no answer"}


class Servers(Protocol):
    teacher_url: str

    def candidate(self, package_dir: Path) -> ContextManager[str]:
        """A server for the packaged group model while the context is open; yields its base URL."""

    def current(self, member: Member) -> ContextManager[str]:
        """A server for the member's current model; yields its base URL."""


def _wait_healthy(url: str, timeout_s: float, sleep: Callable[[float], None] = time.sleep) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url + "/health", timeout=3) as reply:
                if reply.status == 200:
                    return True
        except OSError:
            pass
        sleep(2)
    return False


class RealServers:
    """llama-server from llama.cpp on the pod, one model at a time, on 127.0.0.1, with the gate's one decode setting.
    A member's current model is their current_url when pool.json gives one, else <packages>/<current_version>."""

    def __init__(self, *, llama_dir: Path, packages: Path, teacher_url: str, port: int = 8091,
                 timeout_s: float = 600.0) -> None:
        self.llama_dir, self.packages, self.teacher_url = Path(llama_dir), Path(packages), teacher_url
        self.port, self.timeout_s = port, timeout_s

    @contextlib.contextmanager
    def _serve(self, folder: Path) -> Iterator[str]:
        manifest = json.loads((folder / package.MANIFEST).read_text(encoding="utf-8"))
        argv = [str(self.llama_dir / "build" / "bin" / "llama-server"), "-m", str(folder / manifest["gguf"]),
                "--host", "127.0.0.1", "--port", str(self.port), "--jinja", "--reasoning-budget", "0",
                "--temp", "0", "--seed", "0", "-n", str(eval_gate.DECODE["max_tokens"])]
        if manifest.get("mmproj"):
            argv += ["--mmproj", str(folder / manifest["mmproj"])]
        url = f"http://127.0.0.1:{self.port}"
        process = subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=sys.stderr)
        try:
            if not _wait_healthy(url, self.timeout_s):
                raise eval_gate.GateError(f"llama-server for {folder.name} did not become healthy")
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()

    def candidate(self, package_dir: Path) -> ContextManager[str]:
        return self._serve(package_dir)

    def current(self, member: Member) -> ContextManager[str]:
        if member.current_url:
            return contextlib.nullcontext(member.current_url)
        folder = self.packages / member.current_version
        if not (folder / package.MANIFEST).is_file():
            raise eval_gate.GateError(f"no current_url for {member.member} and no package {member.current_version} "
                                      f"in {self.packages}")
        return self._serve(folder)


class _Headers(httpx.BaseTransport):
    """Adds the hub's headers (the token, a browser's agent) to every request for one base URL."""

    def __init__(self, headers: Mapping[str, str], inner: httpx.BaseTransport | None = None) -> None:
        self.headers, self.inner = dict(headers), inner or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        request.headers.update(self.headers)
        return self.inner.handle_request(request)


class DeskGateClient(eval_gate.RealClient):
    """The gate's client for an image task the gate has no scorer for (desk-objects): lab/desk_score.py asks each
    held-out picture the way the box does and compares the normalised answer with the label; each result is given
    the scorer's Result fields (answer_ok is the match) and its example id, so the gate pairs it by id. A request to
    the teacher's URL carries teacher_headers (the hub token), since the teacher sits behind the hub proxy."""

    def __init__(self, dataset_root: str | os.PathLike[str], *, teacher_url: str | None = None,
                 teacher_headers: Mapping[str, str] | None = None,
                 transport: httpx.BaseTransport | None = None) -> None:
        self.dataset_root, self.transport = Path(dataset_root), transport
        self.teacher_url, self.teacher_headers = teacher_url, dict(teacher_headers or {})

    def score(self, base_url: str, *, model_version: str | None = None) -> dict[str, Any]:
        from lab import desk_score
        split = ds.load_split(self.dataset_root, require_images=True, min_held_out=1)
        transport = self.transport
        if self.teacher_url and base_url.rstrip("/") == self.teacher_url.rstrip("/") and self.teacher_headers:
            transport = _Headers({"User-Agent": hub.USER_AGENT, **self.teacher_headers}, transport)
        out = desk_score.score(base_url, desk_score.held_out_examples(split),
                               prompt=split.data.manifest.spec.instruction, transport=transport, model=model_version)
        out["results"] = [{"id": r["id"], "q": r["id"], "answer": r["answer"], "cited": [], "citations_ok": True,
                           "answer_ok": bool(r["match"]), "latency_ms": 0} for r in out["results"]]
        return out


def gate_client_for(task: str, dataset: Path, *, teacher_url: str | None = None,
                    teacher_headers: Mapping[str, str] | None = None) -> eval_gate.Client:
    """The gate's registered scorer for the task, else DeskGateClient for desk-objects, else the gate's default."""
    if task not in eval_gate.SCORERS and task == DESK_TASK:
        return DeskGateClient(dataset, teacher_url=teacher_url, teacher_headers=teacher_headers)
    return eval_gate.client_for(task, dataset)


# --- the run ------------------------------------------------------------------------------------------------------------

def _delete(paths: Sequence[Path]) -> tuple[bool, list[str]]:
    left = []
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink(missing_ok=True)
        if path.exists():
            left.append(str(path))
    return not left, left


def _pooled_paths(pool: Pool, work: Path) -> list[Path]:
    members = [pool.folder / m.member for m in pool.members if (pool.folder / m.member).exists()]
    return [*members, work / "data", work / "train-data"]


def _member_entry(member: Member, gate: Mapping[str, Any]) -> dict[str, Any]:
    decision = gate.get("decision")
    status = {"deploy": "recommended", "reject": "available"}.get(decision or "", "unscored")
    return {"member": member.member, "uploaded_examples": member.examples,
            "exports": [e.dataset_id for e in member.exports], "current_version": member.current_version,
            "rollback_version": member.current_version, "status": status,
            "gate": {"decision": decision or "unscored", "scores": gate.get("scores", {}),
                     "p_value": gate.get("p_value"), "reasons": gate.get("reasons", []),
                     **({"error": gate["error"]} if gate.get("error") else {})}}


def run_group_finetune(*, pool_root: str | os.PathLike[str], circle: str, task: str, run_id: str,
                       work: str | os.PathLike[str], packages: str | os.PathLike[str], teacher: Teacher,
                       trainer: Trainer, gate_client: Callable[[str, Path], eval_gate.Client], servers: Servers,
                       model: str, base_dir: str | os.PathLike[str], mmproj: str | os.PathLike[str],
                       llama_dir: str | os.PathLike[str], toolchain: Any = None, config: TrainConfig = TrainConfig(),
                       min_held_out: int = eval_gate.MIN_ITEMS, decode_attested: bool = False, keep: bool = False,
                       label_workers: int = 1, metrics_path: str | os.PathLike[str] | None = None,
                       node: str = NODE, account: str = "D") -> dict[str, Any]:
    """The whole group run (see the module docstring). GroupError before anything is written when the inputs do
    not hold up; after that, every way out deletes the pooled data (unless keep) and writes the run's rows."""
    if keep and running_on_pod():
        raise GroupError("--keep is refused on a pod: the pooled exports must not outlive the run there")
    work, packages = Path(work).expanduser(), Path(packages).expanduser()
    pool = read_pool(pool_root, circle, task)
    if not pool.uploaders:
        raise GroupError("no member uploaded an export for this task")
    refs = {"run_id": run_id}
    result: dict[str, Any] = {"ok": False, "outcome": "failed", "run_id": run_id, "circle": circle, "task": task,
                              "members": [m.member for m in pool.members], "uploaders": pool.uploaders}
    counts: dict[str, Any] = {"examples_by_member": pool.counts(), "exports_by_member":
                              {m.member: len(m.exports) for m in pool.members}, "examples": pool.examples}
    scores: dict[str, dict[str, float]] = {}
    gates: dict[str, dict[str, Any]] = {}
    step = "pool"
    try:
        write_counts(pool)
        step = "merge"
        merged = merge_pool(pool, work / "data" / "exports", run_id=run_id)
        step = "label"
        labeled = label_dataset(merged.root, teacher, decode=Decode(), workers=label_workers, run_id=f"{run_id}.label",
                                environ={})   # the pooled copy is deleted by this run on every way out
        result["labels"] = {"outcome": labeled.outcome, "accepted": labeled.accepted, "asked": labeled.asked,
                            "pending": labeled.pending_after, "checkpoint": labeled.checkpoint}
        if labeled.outcome != "ok":
            raise GroupError(f"labeling ended {labeled.outcome}: {labeled.error}")
        step = "rows"
        split = ds.load_split(merged.root, min_held_out=1)   # rows need only the train side
        rows = image_rows(split)
        if not rows:
            raise GroupError("the teacher's labels left no training row")
        counts.update(train_rows=len(rows), held_out=len(split.held_out), labeled=labeled.accepted)
        train_dir = work / "train-data"
        train_copy = train_dir / merged.dataset_id
        shutil.copytree(merged.root, train_copy)
        rows_path = train_dir / "rows.jsonl"
        rows_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8",
                             newline="\n")
        step = "train"
        out = work / "runs" / run_id
        trained = trainer.train(dataset=train_copy, rows=rows_path, model=model, out=out, run_id=run_id,
                                config=config)
        result["train"] = {"outcome": trained.get("outcome"), "metrics": trained.get("metrics", {})}
        if not trained.get("ok"):
            raise GroupError(f"the trainer answered {trained.get('outcome')}: {trained.get('error')}")
        step = "package"
        version = f"atlas-group-{task}-{run_id}"
        built = package.build(adapter=out / "adapter", run_id=run_id, task=task, mmproj=mmproj, base_dir=base_dir,
                              llama_dir=llama_dir, out_root=packages, version=version, toolchain=toolchain,
                              node=node)
        folder = Path(built["folder"])
        result.update(model_version=version, package=str(folder))
        step = "gate"
        with servers.candidate(folder) as candidate_url:
            for member in pool.members:
                try:
                    with servers.current(member) as current_url:
                        gate = eval_gate.run_gate(
                            client=gate_client(task, merged.root), dataset=merged.root, current_url=current_url,
                            candidate_url=candidate_url, teacher_url=servers.teacher_url,
                            candidate_manifest=folder / package.MANIFEST, run_id=f"{run_id}.{member.member}",
                            out_dir=work / "gate" / member.member, current_version=member.current_version,
                            episodes=merged.episodes, decode_attested=decode_attested, min_held_out=min_held_out,
                            task=task)
                except eval_gate.GateError as err:
                    gate = {"decision": None, "error": str(err)[:300]}
                gates[member.member] = gate
                if gate.get("scores"):
                    scores[member.member] = {k: gate["scores"][k] for k in ("current", "candidate", "teacher")}
        step = "offer"
        offer = {"version": 1, "circle": circle, "task": task, "run_id": run_id, "model_version": version,
                 "created": dt.datetime.now(dt.timezone.utc).isoformat(), "account": account,
                 "members": [_member_entry(m, gates.get(m.member, {})) for m in pool.members],
                 "take": {"pull": f"python -m lab.package_pull pull --account {account} --version {version} "
                                  "--dest data/models",
                          "swap": f"python jetson/atlas_swap.py swap {version}",
                          "rollback": "python jetson/atlas_swap.py rollback"}}
        _write_json(folder / OFFER_FILE, offer)
        result.update(ok=True, outcome="offered", offer=str(folder / OFFER_FILE))
    except (GroupError, ds.DatasetError, LabelRefused, package.PackageError, eval_gate.GateError) as err:
        result.update(ok=False, outcome="failed", step=step, error=str(err)[:500])
    finally:
        paths = _pooled_paths(pool, work)
        if keep:
            result["data_deleted"] = "no: --keep"
            record("group-finetune.delete", "kept", refs=refs, node=node, note="--keep: the pooled data was kept",
                   paths=[str(p) for p in paths if p.exists()])
        else:
            gone, left = _delete(paths)
            result["data_deleted"] = "yes" if gone else f"no: {len(left)} paths left"
            write_counts(read_pool(pool_root, circle, task), last_run={"run_id": run_id, "data_deleted": gone})
            record("group-finetune.delete", "deleted" if gone else "fail",
                   {"group.deleted_paths": len(paths) - len(left)}, refs, node=node,
                   note=None if gone else "left: " + ", ".join(left)[:300])
    result["counts"] = counts
    result["scores"] = scores
    metrics = {"group.members": len(pool.members), "group.uploaders": len(pool.uploaders),
               "group.examples": pool.examples, **({"group.train_rows": counts["train_rows"]}
                                                     if "train_rows" in counts else {})}
    record("group-finetune.run", "ok" if result["ok"] else "failed", metrics, refs, node=node, source=SOURCE,
           conditions={"circle": circle, "task": task, "config": config.fields(), "min_held_out": min_held_out},
           members=result["members"], counts=counts, scores=scores,
           decisions={m: (g.get("decision") or "unscored") for m, g in gates.items()},
           note=result.get("error"))
    if result["ok"]:
        _metrics_rows(pool, scores, run_id=run_id, version=result["model_version"], node=node, path=metrics_path)
    return result


def _metrics_rows(pool: Pool, scores: Mapping[str, Mapping[str, float]], *, run_id: str, version: str, node: str,
                  path: str | os.PathLike[str] | None) -> None:
    where = {"path": Path(path)} if path else {}
    base = {"circle": pool.circle, "task": pool.task, "run_id": run_id, "model_version": version}
    metrics_file.add({"metric": "group_finetune.pooled_examples", "value": pool.examples, "unit": "examples",
                      "node": node, "conditions": {**base, "uploaders": len(pool.uploaders)}, "source": SOURCE,
                      "low_confidence": False}, **where)
    for member, score in scores.items():
        conditions = {**base, "member": member}
        metrics_file.add({"metric": "group_finetune.candidate_score", "value": score["candidate"], "unit": "score",
                          "node": node, "conditions": conditions, "source": SOURCE, "low_confidence": False}, **where)
        metrics_file.add({"metric": "group_finetune.candidate_delta", "value": score["candidate"] - score["current"],
                          "unit": "score", "node": node, "conditions": conditions, "source": SOURCE,
                          "low_confidence": False}, **where)


def plan(pool_root: str | os.PathLike[str], circle: str, task: str, *, run_id: str, work: Path,
         packages: Path, keep: bool) -> dict[str, Any]:
    """What a run would do, read-only: the pool's counts and each step's inputs and outputs."""
    pool = read_pool(pool_root, circle, task)
    dataset_id = f"group-{circle}-{task}-{run_id}"
    version = f"atlas-group-{task}-{run_id}"
    steps = [
        {"step": "pool", "reads": str(pool.folder), "writes": str(pool.folder / POOL_FILE)},
        {"step": "merge", "writes": str(work / "data" / "exports" / dataset_id)},
        {"step": "split", "writes": str(work / "data" / "exports" / dataset_id / ds.SPLIT)},
        {"step": "label", "examples": pool.examples},
        {"step": "rows", "writes": str(work / "train-data" / "rows.jsonl")},
        {"step": "train", "out": str(work / "runs" / run_id / "adapter")},
        {"step": "package", "writes": str(packages / version)},
        {"step": "gate", "members": [m.member for m in pool.members]},
        {"step": "offer", "writes": str(packages / version / OFFER_FILE)},
        {"step": "delete", "paths": [str(p) for p in _pooled_paths(pool, work)], "keep": keep},
    ]
    return {"members": pool.counts(), "uploaders": pool.uploaders, "examples": pool.examples,
            "dataset_id": dataset_id, "model_version": version, "steps": steps}


# --- the box side ---------------------------------------------------------------------------------------------------------

class Box(Protocol):
    def pull(self, version: str) -> bool: ...
    def swap(self, version: str) -> bool: ...
    def rollback(self, version: str) -> bool: ...


class RealBox:
    """The box's own commands: lab/package_pull.py, then jetson/atlas_swap.py (which checks every sha256)."""

    def __init__(self, account: str = "D", models_dir: str = "data/models") -> None:
        self.account, self.models_dir = account, models_dir

    def _run(self, argv: list[str]) -> bool:
        return subprocess.run(argv, cwd=REPO, check=False).returncode == 0

    def pull(self, version: str) -> bool:
        return self._run([sys.executable, "-m", "lab.package_pull", "pull", "--account", self.account, "--version",
                          version, "--dest", self.models_dir])

    def swap(self, version: str) -> bool:
        return self._run([sys.executable, "jetson/atlas_swap.py", "swap", version, "--models-dir", self.models_dir])

    def rollback(self, version: str) -> bool:
        return self._run([sys.executable, "jetson/atlas_swap.py", "rollback", "--models-dir", self.models_dir])


def take_offer(offer_path: str | os.PathLike[str], member: str, box: Box, *, rollback: bool = False) -> dict[str, Any]:
    """Take the group model on one member's box (any member on the roster, uploader or not), or roll back."""
    offer = json.loads(Path(offer_path).read_text(encoding="utf-8"))
    entries = {m["member"]: m for m in offer.get("members", [])}
    if member not in entries:
        raise GroupError(f"{member!r} is not a member of the Circle this model was offered to")
    entry, version = entries[member], offer["model_version"]
    refs = {"run_id": offer.get("run_id")}
    if rollback:
        ok = box.rollback(entry["rollback_version"])
        record("group-finetune.take", "rolled-back" if ok else "fail", refs=refs, node="jetson", member=member,
               version=entry["rollback_version"])
        return {"ok": ok, "outcome": "rolled-back" if ok else "fail", "member": member,
                "version": entry["rollback_version"]}
    if not box.pull(version):
        record("group-finetune.take", "fail", refs=refs, node="jetson", member=member, version=version,
               note="pull failed")
        return {"ok": False, "outcome": "pull-failed", "member": member, "version": version}
    if box.swap(version):
        record("group-finetune.take", "swapped", refs=refs, node="jetson", member=member, version=version,
               status=entry.get("status"))
        return {"ok": True, "outcome": "swapped", "member": member, "version": version, "status": entry.get("status"),
                "rollback_version": entry["rollback_version"]}
    back = box.rollback(entry["rollback_version"])
    record("group-finetune.take", "fail", refs=refs, node="jetson", member=member, version=version,
           note=f"swap failed; rollback {'ok' if back else 'failed'}")
    return {"ok": False, "outcome": "swap-failed", "member": member, "version": version,
            "rolled_back_to": entry["rollback_version"] if back else None}


# --- the verb -----------------------------------------------------------------------------------------------------------

def _run_parser() -> argparse.ArgumentParser:
    home = Path("~")
    p = argparse.ArgumentParser(prog="python -m lab group-finetune", description=main.__doc__, allow_abbrev=False)
    p.add_argument("--circle", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--pool", default=DEFAULT_POOL, help="the pool root: <pool>/<circle>/<task>/ (default ~/pool)")
    p.add_argument("--run-id", help="default group-<UTC time>")
    p.add_argument("--work", help=f"default {DEFAULT_WORK}/<run id>")
    p.add_argument("--packages", default=DEFAULT_PACKAGES, help="where the package goes (the package server's root)")
    p.add_argument("--teacher-url", help="the teacher's address without /v1 (the hub proxy URL of pod C's port 8000)")
    p.add_argument("--teacher-token-env", default="AUP_HUB_TOKEN_C",
                   help="the environment variable holding the hub token for the teacher URL")
    p.add_argument("--gate-teacher-url", help="the teacher URL the gate's scorer uses (default: --teacher-url)")
    p.add_argument("--model", default=str(home / "models" / "Qwen3.5-4B"), help="the base model folder")
    p.add_argument("--mmproj", default=str(home / "models" / "Qwen3.5-4B-GGUF" / "mmproj-F16.gguf"))
    p.add_argument("--llama-cpp", default=str(home / "llama.cpp"))
    p.add_argument("--trainer-python", help="the interpreter lab/train.py runs under (default: this one)")
    p.add_argument("--trainer-arg", action="append", default=[], help="one more flag for lab/train.py (repeat)")
    for name, kind in (("epochs", float), ("lr", float), ("rank", int), ("alpha", float), ("dropout", float),
                       ("batch", int), ("grad-accum", int), ("seed", int), ("max-steps", int)):
        p.add_argument(f"--{name}", type=kind)
    p.add_argument("--base-revision")
    p.add_argument("--min-held-out", type=int, default=eval_gate.MIN_ITEMS)
    p.add_argument("--decode-attested", action="store_true")
    p.add_argument("--label-workers", type=int, default=4)
    p.add_argument("--port", type=int, default=8091, help="the port the gate's llama-servers use on the pod")
    p.add_argument("--metrics-file", help="default lab/refine/metrics.jsonl")
    p.add_argument("--account", default="D", help="the hub account the boxes pull the package from")
    p.add_argument("--keep", action="store_true", help="keep the pooled data (debugging; refused on a pod)")
    p.add_argument("--dry-run", action="store_true", help="print the plan; write nothing")
    return p


def _take(argv: Sequence[str]) -> dict[str, Any]:
    p = argparse.ArgumentParser(prog="python -m lab group-finetune take", allow_abbrev=False)
    p.add_argument("--offer", required=True, help="the group_offer.json pulled from the pod")
    p.add_argument("--member", required=True)
    p.add_argument("--account", default="D")
    p.add_argument("--models-dir", default="data/models")
    p.add_argument("--rollback", action="store_true")
    args = p.parse_args(argv)
    return take_offer(args.offer, args.member, RealBox(args.account, args.models_dir), rollback=args.rollback)


def _pool_add(argv: Sequence[str]) -> dict[str, Any]:
    p = argparse.ArgumentParser(prog="python -m lab group-finetune pool-add", allow_abbrev=False)
    for name in ("--circle", "--task", "--member", "--export"):
        p.add_argument(name, required=True)
    p.add_argument("--pool", default=DEFAULT_POOL)
    args = p.parse_args(argv)
    return add_export(args.pool, args.circle, args.task, args.member, args.export)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Group fine-tune (L10): pool a Circle's opt-in exports, label, train one LoRA, gate per member, offer to all."""
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv[:1] == ["take"]:
            return _take(argv[1:])
        if argv[:1] == ["pool-add"]:
            return _pool_add(argv[1:])
        args = _run_parser().parse_args(argv)
        run_id = args.run_id or f"group-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
        work = Path(args.work or f"{DEFAULT_WORK}/{run_id}").expanduser()
        packages = Path(args.packages).expanduser()
        if args.dry_run:
            return {"ok": True, "outcome": "dry-run", "run_id": run_id,
                    "plan": plan(args.pool, args.circle, args.task, run_id=run_id, work=work, packages=packages,
                                 keep=args.keep)}
        if not args.teacher_url:
            return {"ok": False, "outcome": "refused", "error": "--teacher-url is required for a run"}
        token = os.environ.get(args.teacher_token_env or "", "")
        headers = {"Authorization": f"token {token}"} if token else {}
        overrides = {k: v for k, v in {"epochs": args.epochs, "lr": args.lr, "rank": args.rank, "alpha": args.alpha,
                                       "dropout": args.dropout, "batch": args.batch, "grad_accum": args.grad_accum,
                                       "seed": args.seed, "max_steps": args.max_steps,
                                       "base_revision": args.base_revision}.items() if v is not None}
        llama = Path(args.llama_cpp).expanduser()
        servers = RealServers(llama_dir=llama, packages=packages, teacher_url=args.gate_teacher_url or args.teacher_url,
                              port=args.port)
        with Teacher(args.teacher_url, headers=headers,
                     scrub=(lambda text: text.replace(token, "***")) if token else None, node="pod-c") as teacher:
            return run_group_finetune(
                pool_root=args.pool, circle=args.circle, task=args.task, run_id=run_id, work=work, packages=packages,
                teacher=teacher, trainer=RealTrainer(args.trainer_python, args.trainer_arg),
                gate_client=lambda task, dataset: gate_client_for(task, dataset, teacher_url=servers.teacher_url,
                                                                  teacher_headers=headers),
                servers=servers,
                model=str(Path(args.model).expanduser()), base_dir=Path(args.model).expanduser(),
                mmproj=Path(args.mmproj).expanduser(), llama_dir=llama, config=TrainConfig(**overrides),
                min_held_out=args.min_held_out, decode_attested=args.decode_attested, keep=args.keep,
                label_workers=args.label_workers, metrics_path=args.metrics_file, account=args.account)
    except GroupError as err:
        return {"ok": False, "outcome": "refused", "error": str(err)}
