"""Read, check and label in place a dataset export from the box.

An approved export is a folder named for its dataset id (data/exports/<dataset_id>/ on the box) that holds
what contracts/learning.py describes:

    manifest.json    a DatasetManifest: the dataset id, the task spec, the example count, the approval's action id
    examples.jsonl   one DatasetExample per line: id, image, frame_ref, ts, context (the box's own caption), target
    images/*.jpg     the blurred keyframes the examples point at

    from lab.dataset import load, write_target

    data = load("data/exports/desk-objects-3fa91c2b")     # DatasetError, naming every problem, if it does not hold up
    for example in data.pending():                         # the examples that have no target yet
        write_target(data.root, example.id, teacher_label(example, data.image_path(example)))

load() reads the folder and checks it against the contract models. Beyond what pydantic enforces it requires
that the manifest names the approval that allowed the export (an empty action_id is refused), that the
manifest's example count is the number of lines, that example ids are unique and not empty, and that every
image is a relative path to a file inside the folder (no absolute path, no "..", no backslash, no symlink out).
A problem is named by file and line and never quotes a caption or a label, so it can go in a log. Text-only
datasets, such as the one lab/fixture_dataset.py builds, have an empty image on every example, which load()
refuses unless it is called with require_images=False: an image task whose pictures are missing should fail
before it spends teacher time, not label blank prompts.

The lab's labeling loop fills `target`. write_targets() and write_target() put labels into examples.jsonl so a
run can be killed at any moment and started again:

    * every call replaces the file whole (a temporary file in the same folder, fsynced, then os.replace), so a
      crash leaves the old file or the new one, never half of one;
    * the read, change and replace happen under a lock file in the dataset folder (.examples.lock), so labelers
      that run at the same time lose no one's labels;
    * the lines that are not changed stay byte for byte as they were, and a changed line keeps its other fields
      as they were, including fields a later contract version adds;
    * a target already filled with the same text is left alone, and one filled with different text is an error
      unless overwrite=True, so a second run never pays for or clobbers labels it already has;
    * pending() is the work left: the examples whose target is missing, empty or blank.

A call that fails writes nothing: unknown ids, a target that is not text, an example the contract would refuse.

A dataset can carry a frozen split: split.json, beside examples.jsonl, lists the example ids that train and the ids
held out for the eval gate. load_split() is the one place the trainer and the gate decide membership. Nothing else
counts: not a question's origin, not a hash of its id, not the order of examples.jsonl.

    from lab.dataset import load_split

    split = load_split("fixtures/learning/tool-calling-fixture", require_images=False)
    split.train          # DatasetExample objects, in the order split.json lists them
    split.held_out       # the same for the gate's examples, which no training run may see
    split.sha256         # of split.json; the ledger row of every run that used the split records it

It refuses a split that is not whole and clean, so a run cannot start on a leaky one: the dataset id must match;
every example must be listed once, on one side; no listed id may be missing from examples.jsonl; no frame_ref may
be on both sides (a frame stays on one side, and with it the group of questions that share it); the same question
(a text-only example's `context`, with case, spacing and closing punctuation folded) may not be on both sides; and
the held-out side must hold at least MIN_HELD_OUT examples, the floor for a gate that may deploy. A smaller floor
can be asked for by name, to report the scores of a short held-out set; split.meets_deploy_floor then says False,
and a gate that reads it that way reports the scores and never deploys. A problem names ids and counts and never
quotes a caption, a question or a label.

    python lab/dataset.py <dataset folder> [--text-only] [--split]

prints one JSON line (ok, dataset_id, examples, labeled, pending, problems) and exits 0 when the folder holds up
and 1 when it does not. With --split it also checks split.json as load_split() does and adds a `split` entry
(sha256, train, held_out, held_out_frame_refs) to the line; the ledger row then carries refs.split_sha256, which is
the record that the split was frozen as it stands. The entry point writes a ledger row (verb dataset-check) and
refuses to run while lab/state/STOP exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, NamedTuple

if not __package__:  # run by path on a pod: the repo root, one folder up, is what holds contracts/ and lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pydantic import BaseModel, ConfigDict, ValidationError

from contracts.learning import DatasetExample, DatasetManifest
from lab.machine import stop
from lab.machine.lock import FileLock
from lab.machine.record import recorded

__all__ = ["MANIFEST", "EXAMPLES", "IMAGES", "SPLIT", "MIN_HELD_OUT", "LOCK_NAME", "LOCK_TIMEOUT", "DatasetError",
           "Dataset", "Split", "Written", "has_target", "load", "check", "load_split", "write_target",
           "write_targets", "main"]

MANIFEST = "manifest.json"
EXAMPLES = "examples.jsonl"
IMAGES = "images"
SPLIT = "split.json"
MIN_HELD_OUT = 50              # held-out examples a task needs before the eval gate may deploy (the owner's ruling)
LOCK_NAME = ".examples.lock"   # in the dataset folder; held only while labels are written
LOCK_TIMEOUT = 30.0            # seconds a write waits for another labeler
REPLACE_TRIES = 6              # os.replace attempts; Windows refuses while a scanner holds the file for a moment


class DatasetError(ValueError):
    """The folder does not meet the contract, or a write was refused. problems lists every reason found."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        shown = "; ".join(self.problems[:5]) + ("; ..." if len(self.problems) > 5 else "")
        super().__init__(f"{len(self.problems)} problem{'s' if len(self.problems) != 1 else ''}: {shown}")


class Written(NamedTuple):
    """What a write-back did: targets stored, and targets that were already filled with the same text."""

    written: int
    unchanged: int


@dataclass(frozen=True)
class Dataset:
    """A dataset folder that passed load(): its manifest and its examples in file order."""

    root: Path
    manifest: DatasetManifest
    examples: tuple[DatasetExample, ...]

    @property
    def dataset_id(self) -> str:
        return self.manifest.dataset_id

    def pending(self) -> list[DatasetExample]:
        """The examples still to label, in file order."""
        return [example for example in self.examples if not has_target(example)]

    def labeled(self) -> list[DatasetExample]:
        return [example for example in self.examples if has_target(example)]

    def image_path(self, example: DatasetExample) -> Path | None:
        """The example's picture, or None for a text-only example (image is empty)."""
        return _image_file(self.root, example.image) if example.image else None


def has_target(example: DatasetExample) -> bool:
    """Whether the teacher's label is in: a target that is neither missing, empty nor blank."""
    return bool(example.target and example.target.strip())


# --- reading ----------------------------------------------------------------------------

def _why(err: ValidationError) -> str:
    """The contract's complaint as field and message only: pydantic's own text quotes the offending value."""
    return "; ".join(f"{'.'.join(str(part) for part in item['loc']) or 'value'}: {item['msg']}"
                     for item in err.errors(include_url=False, include_context=False, include_input=False))


def _text(path: Path) -> str:
    return path.read_bytes().decode("utf-8-sig")


def _image_file(root: Path, image: str) -> Path:
    return root.joinpath(*PurePosixPath(image).parts)


def _image_problem(root: Path, image: str) -> str | None:
    """Why image is not a file inside the dataset folder, or None when it is."""
    parts = PurePosixPath(image).parts
    if "\\" in image or image.startswith("/") or image[1:2] == ":" or ".." in parts or not parts:
        return "is not a relative path inside the dataset folder"
    file = _image_file(root, image)
    if not file.is_file():
        return "is not a file in the dataset folder"
    if not file.resolve().is_relative_to(root.resolve()):
        return "leaves the dataset folder through a link"
    return None


def _read_manifest(root: Path, problems: list[str]) -> DatasetManifest | None:
    try:
        text = _text(root / MANIFEST)
    except FileNotFoundError:
        problems.append(f"{MANIFEST}: missing")
        return None
    except (OSError, UnicodeDecodeError) as exc:
        problems.append(f"{MANIFEST}: cannot be read ({type(exc).__name__})")
        return None
    try:
        return DatasetManifest.model_validate_json(text)
    except ValidationError as err:
        problems.append(f"{MANIFEST}: {_why(err)}")
        return None


def _read_examples(root: Path, problems: list[str]) -> list[tuple[int, DatasetExample]] | None:
    """(line number, example) for every line; None, with the problems of every bad line, when any line is bad."""
    try:
        text = _text(root / EXAMPLES)
    except FileNotFoundError:
        problems.append(f"{EXAMPLES}: missing")
        return None
    except (OSError, UnicodeDecodeError) as exc:
        problems.append(f"{EXAMPLES}: cannot be read ({type(exc).__name__})")
        return None
    found: list[tuple[int, DatasetExample]] = []
    clean = True
    for number, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        try:
            found.append((number, DatasetExample.model_validate_json(line)))
        except ValidationError as err:
            problems.append(f"{EXAMPLES} line {number}: {_why(err)}")
            clean = False
    return found if clean else None


def _cross_check(root: Path, manifest: DatasetManifest, examples: list[tuple[int, DatasetExample]],
                 require_images: bool, problems: list[str]) -> None:
    if manifest.examples != len(examples):
        problems.append(f"{MANIFEST}: says {manifest.examples} examples but {EXAMPLES} has {len(examples)}")
    if not manifest.action_id.strip():
        problems.append(f"{MANIFEST}: action_id is empty; an export is read only with the approval that allowed it")
    first_seen: dict[str, int] = {}
    for number, example in examples:
        where = f"{EXAMPLES} line {number} (id {example.id!r})"
        if not example.id.strip():
            problems.append(f"{where}: the id is empty")
        elif example.id in first_seen:
            problems.append(f"{where}: the id repeats line {first_seen[example.id]}")
        else:
            first_seen[example.id] = number
        if not example.image:
            if require_images:
                problems.append(f"{where}: has no image (a text-only dataset is read with require_images=False)")
        elif (why := _image_problem(root, example.image)) is not None:
            problems.append(f"{where}: image {why}")


def _inspect(root: Path, require_images: bool) -> tuple[Dataset | None, list[str]]:
    if not root.is_dir():
        return None, [f"{root.name or str(root)}: not a folder"]
    problems: list[str] = []
    manifest = _read_manifest(root, problems)
    examples = _read_examples(root, problems)
    if manifest is not None and examples is not None:
        _cross_check(root, manifest, examples, require_images, problems)
    if problems or manifest is None or examples is None:
        return None, problems
    return Dataset(root, manifest, tuple(example for _, example in examples)), []


def load(root: str | os.PathLike[str], *, require_images: bool = True) -> Dataset:
    """The dataset in the folder. DatasetError, listing every problem, when it does not hold up."""
    data, problems = _inspect(Path(root), require_images)
    if data is None:
        raise DatasetError(problems)
    return data


def check(root: str | os.PathLike[str], *, require_images: bool = True) -> list[str]:
    """Every problem load() would raise for this folder; empty when it holds up."""
    return _inspect(Path(root), require_images)[1]


# --- writing ----------------------------------------------------------------------------

def _replace(temporary: Path, destination: Path) -> None:
    """os.replace, tried a few times: on Windows a scanner or indexer can hold the file for a moment."""
    for attempt in range(REPLACE_TRIES):
        try:
            os.replace(temporary, destination)
            return
        except PermissionError:
            if attempt == REPLACE_TRIES - 1:
                raise
            time.sleep(0.05 * (attempt + 1))


def _store(path: Path, payload: bytes) -> None:
    """Replace the file whole: the new bytes go to a temporary file beside it, fsynced, then take its place."""
    handle, name = tempfile.mkstemp(dir=path.parent, prefix=".examples-", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(payload)
            out.flush()
            os.fsync(out.fileno())
        shutil.copymode(path, temporary)
        _replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _write_locked(root: Path, targets: Mapping[str, str], overwrite: bool) -> Written:
    path = root / EXAMPLES
    try:
        text = _text(path)
    except FileNotFoundError:
        raise DatasetError([f"{EXAMPLES}: missing"]) from None
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetError([f"{EXAMPLES}: cannot be read ({type(exc).__name__})"]) from None
    lines = text.split("\n")
    problems: list[str] = []
    rows: dict[int, dict[str, Any]] = {}
    index_of: dict[str, int] = {}
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            problems.append(f"{EXAMPLES} line {index + 1}: not JSON")
            continue
        key = row.get("id") if isinstance(row, dict) else None
        if not isinstance(key, str):
            problems.append(f"{EXAMPLES} line {index + 1}: not an object with a text id")
        elif key in index_of:
            problems.append(f"{EXAMPLES} line {index + 1} (id {key!r}): the id repeats line {index_of[key] + 1}")
        else:
            rows[index] = row
            index_of[key] = index
    if problems:
        raise DatasetError(problems)

    changed: dict[int, dict[str, Any]] = {}
    unchanged = 0
    for key, label in targets.items():
        index = index_of.get(key)
        if index is None:
            problems.append(f"id {key!r}: not in {EXAMPLES}")
            continue
        row = rows[index]
        current = row.get("target")
        if isinstance(current, str) and current.strip():
            if current == label:
                unchanged += 1
                continue
            if not overwrite:
                problems.append(f"id {key!r}: already has a different target (overwrite=True replaces it)")
                continue
        row["target"] = label
        try:
            DatasetExample.model_validate(row)
        except ValidationError as err:
            problems.append(f"{EXAMPLES} line {index + 1} (id {key!r}): {_why(err)}")
            continue
        changed[index] = row
    if problems:
        raise DatasetError(problems)
    if not changed:
        return Written(0, unchanged)

    for index, row in changed.items():
        ending = "\r" if lines[index].endswith("\r") else ""
        lines[index] = json.dumps(row, ensure_ascii=False, separators=(",", ":")) + ending
    payload = "\n".join(lines)
    _store(path, (payload if payload.endswith("\n") else payload + "\n").encode("utf-8"))
    return Written(len(changed), unchanged)


def write_targets(root: str | os.PathLike[str], targets: Mapping[str, str], *, overwrite: bool = False,
                  timeout: float = LOCK_TIMEOUT) -> Written:
    """Store {example id: label} in the folder's examples.jsonl, all or nothing (see the module docstring).

    DatasetError, with nothing written, for an unknown id, a label that is not non-blank text, an example the
    contract would refuse, a target already filled with different text (unless overwrite), or an examples.jsonl
    that is not one JSON object with a text id per line. LockTimeout when another labeler holds the folder's
    lock for the whole timeout."""
    root = Path(root)
    problems = [f"id {key!r}: the target must be non-blank text" for key, label in targets.items()
                if not isinstance(label, str) or not label.strip()]
    if not root.is_dir():
        problems.append(f"{root.name or str(root)}: not a folder")
    if problems:
        raise DatasetError(problems)
    if not targets:
        return Written(0, 0)
    with FileLock(root / LOCK_NAME, timeout=timeout) as lock:
        lock.write_note(f"pid {os.getpid()} writing targets")
        return _write_locked(root, targets, overwrite)


def write_target(root: str | os.PathLike[str], example_id: str, label: str, *, overwrite: bool = False,
                 timeout: float = LOCK_TIMEOUT) -> Written:
    """write_targets() for one example: the call a labeling loop makes after each answer from the teacher."""
    return write_targets(root, {example_id: label}, overwrite=overwrite, timeout=timeout)


# --- the frozen split -------------------------------------------------------------------

class _SplitFile(BaseModel):
    """split.json as stored: the ids that train, the ids held out, and a line on how they were drawn."""

    model_config = ConfigDict(extra="forbid")

    version: Literal[1]
    dataset_id: str
    rule: str = ""
    train: list[str]
    held_out: list[str]


@dataclass(frozen=True)
class Split:
    """A dataset and its frozen split: the examples that train and the examples held out, as split.json lists them."""

    data: Dataset
    train: tuple[DatasetExample, ...]      # in the order split.json lists them
    held_out: tuple[DatasetExample, ...]   # likewise; no training run may see these
    sha256: str                            # of split.json, its line endings read as LF

    @property
    def dataset_id(self) -> str:
        return self.data.dataset_id

    @property
    def held_out_frame_refs(self) -> int:
        """How many distinct frame_refs the held-out examples stand on. Report it beside len(held_out): examples that
        share a frame_ref are one case asked several ways, not independent items."""
        return len({example.frame_ref for example in self.held_out})

    @property
    def meets_deploy_floor(self) -> bool:
        """Whether the held-out side holds the MIN_HELD_OUT examples a gate needs before it may deploy. False only for
        a split read with a smaller min_held_out to report scores: those are reported and never deploy."""
        return len(self.held_out) >= MIN_HELD_OUT


def _question_key(text: str | None) -> str:
    """A question's wording with case, spacing and closing punctuation folded away, to tell the same question twice."""
    return " ".join((text or "").lower().split()).rstrip("?.! ")


def _listed(ids: Sequence[str], shown: int = 5) -> str:
    """The first few ids, quoted, and how many more there are."""
    text = ", ".join(repr(name) for name in ids[:shown])
    return text + (f" and {len(ids) - shown} more" if len(ids) > shown else "")


def _shared(train: Sequence[DatasetExample], held_out: Sequence[DatasetExample],
            key: Callable[[DatasetExample], str]) -> tuple[int, str, str] | None:
    """How many keys both sides have, with a train id and a held-out id that share one; None when they share none.
    An empty key is no key."""
    first: dict[str, str] = {}
    for example in train:
        if name := key(example):
            first.setdefault(name, example.id)
    clashes: dict[str, tuple[str, str]] = {}
    for example in held_out:
        if (name := key(example)) and name in first:
            clashes.setdefault(name, (first[name], example.id))
    if not clashes:
        return None
    train_id, held_out_id = next(iter(clashes.values()))
    return len(clashes), train_id, held_out_id


def _read_split(root: Path, problems: list[str]) -> tuple[_SplitFile, str] | None:
    """split.json parsed, with its sha256; None, with the problem, when it is missing or is not a split file."""
    try:
        text = _text(root / SPLIT)
    except FileNotFoundError:
        problems.append(f"{SPLIT}: missing")
        return None
    except (OSError, UnicodeDecodeError) as exc:
        problems.append(f"{SPLIT}: cannot be read ({type(exc).__name__})")
        return None
    try:
        listed = _SplitFile.model_validate_json(text)
    except ValidationError as err:
        problems.append(f"{SPLIT}: {_why(err)}")
        return None
    return listed, hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def _inspect_split(data: Dataset, min_held_out: int) -> tuple[Split | None, list[str]]:
    problems: list[str] = []
    read = _read_split(data.root, problems)
    if read is None:
        return None, problems
    listed, digest = read
    if listed.dataset_id != data.dataset_id:
        problems.append(f"{SPLIT}: is for dataset {listed.dataset_id!r}, but the folder holds {data.dataset_id!r}")
    by_id = {example.id: example for example in data.examples}
    for side, ids in (("train", listed.train), ("held_out", listed.held_out)):
        missing = sorted({name for name in ids if name not in by_id})
        if missing:
            problems.append(f"{SPLIT}: {side} ids that {EXAMPLES} does not have ({len(missing)}): {_listed(missing)}")
        repeated = sorted(name for name, times in Counter(ids).items() if times > 1)
        if repeated:
            problems.append(f"{SPLIT}: {side} ids listed more than once ({len(repeated)}): {_listed(repeated)}")
    train_ids, held_out_ids = set(listed.train), set(listed.held_out)
    both = sorted(train_ids & held_out_ids)
    if both:
        problems.append(f"{SPLIT}: ids listed in both train and held_out ({len(both)}): {_listed(both)}")
    on_a_side = train_ids | held_out_ids
    unlisted = [example.id for example in data.examples if example.id not in on_a_side]
    if unlisted:
        problems.append(f"{SPLIT}: examples listed on neither side ({len(unlisted)}): {_listed(unlisted)}")

    train = [by_id[name] for name in listed.train if name in by_id and name not in held_out_ids]
    held_out = [by_id[name] for name in listed.held_out if name in by_id and name not in train_ids]
    bare = [example.id for example in (*train, *held_out) if not example.frame_ref.strip()]
    if bare:
        problems.append(f"{SPLIT}: examples with no frame_ref, which the split cannot keep on one side "
                        f"({len(bare)}): {_listed(bare)}")
    for what, key in (("frame_refs", lambda example: example.frame_ref.strip()),
                      # a picture with a caption is a different input from the same caption beside another picture
                      ("questions", lambda example: "" if example.image else _question_key(example.context))):
        if clash := _shared(train, held_out, key):
            count, train_id, held_out_id = clash
            problems.append(f"{SPLIT}: {what} on both sides ({count}), for example train {train_id!r} and "
                            f"held_out {held_out_id!r}")
    held_out_count = len({example.id for example in held_out})   # real examples, once each
    if held_out_count < min_held_out:
        problems.append(f"{SPLIT}: held_out has {held_out_count} examples, below the {min_held_out} a gate "
                        "needs before it may deploy")
    if not train:
        problems.append(f"{SPLIT}: train is empty")
    if problems:
        return None, problems
    return Split(data, tuple(train), tuple(held_out), digest), []


def load_split(root: str | os.PathLike[str], *, require_images: bool = True,
               min_held_out: int = MIN_HELD_OUT) -> Split:
    """The dataset in the folder with its frozen split (see the module docstring).

    DatasetError, listing every problem, when the dataset does not hold up (as load() would say) or split.json is
    missing, is not a split file, or does not fit the dataset. min_held_out is the floor for a gate that may deploy;
    pass a smaller number only to report the scores of a short held-out set (the split then has meets_deploy_floor
    False and must not deploy) or in a test."""
    split, problems = _inspect_split(load(root, require_images=require_images), min_held_out)
    if split is None:
        raise DatasetError(problems)
    return split


# --- the entry point ----------------------------------------------------------------------

@recorded("dataset-check")
@stop.guarded
def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Check a dataset export against contracts/learning.py and count what is labeled; with --split, check its
    frozen split too."""
    parser = argparse.ArgumentParser(prog="python lab/dataset.py", description=main.__doc__, allow_abbrev=False)
    parser.add_argument("dataset", help="the <dataset_id> folder: manifest.json, examples.jsonl, images/")
    parser.add_argument("--text-only", action="store_true",
                        help="the examples carry no picture (an empty image is allowed)")
    parser.add_argument("--split", action="store_true",
                        help="also check split.json, and record its sha256 in the ledger row")
    args = parser.parse_args(argv)
    data, problems = _inspect(Path(args.dataset), require_images=not args.text_only)
    if data is None:
        return {"ok": False, "outcome": "invalid", "problems": problems,
                "error": str(DatasetError(problems))}
    labeled = len(data.labeled())
    answer: dict[str, Any] = {"ok": True, "outcome": "ok", "dataset_id": data.dataset_id,
                              "examples": len(data.examples), "labeled": labeled,
                              "pending": len(data.examples) - labeled, "problems": [],
                              "refs": {"dataset_id": data.dataset_id}}
    if args.split:
        split, problems = _inspect_split(data, MIN_HELD_OUT)
        if split is None:
            return {"ok": False, "outcome": "invalid", "problems": problems,
                    "error": str(DatasetError(problems))}
        answer["split"] = {"sha256": split.sha256, "train": len(split.train), "held_out": len(split.held_out),
                           "held_out_frame_refs": split.held_out_frame_refs}
        answer["refs"]["split_sha256"] = split.sha256
    return answer


if __name__ == "__main__":
    answer = main()
    print(json.dumps(answer))
    sys.exit(0 if answer.get("ok", True) else 1)
