"""LoRA SFT of the student (the base model the Jetson runs) on the training rows lab/format_traces.py writes.

On the trainer pod, from the folder lab/ and contracts/ were uploaded to (the team repo cannot be cloned there):

    python lab/train.py --dataset datasets/<dataset_id> --rows datasets/rows.jsonl --model ~/models/Qwen3.5-4B
    python lab/train.py ... --epochs 3 --lr 2e-4 --batch 4 --rank 16 --max-steps 20      # a short check run

Anywhere else, as a rehearsal on the CPU with a tiny model (nothing is deleted, no GPU is needed):

    python lab/train.py --cpu --dataset fixtures/learning/train-tiny/tool-calling-tiny \\
        --rows fixtures/learning/train-tiny/rows.jsonl --model <a tiny model folder> --base-revision <40 hex>

What it installs. The pod image has torch and triton and nothing else, so before a run the entry point compares
lab/requirements-train.txt with what is installed and, when a pin is missing or differs, runs pip on that file with
a constraint file that pins the torch, torchvision, torchaudio and triton already installed, then starts itself
again. pip can therefore never swap the pod's torch 2.9.1+rocm7.13.0 for a build from PyPI: a pin that needs another
torch fails to resolve instead. It refuses to install anything where torch is missing. --no-install skips the step.
The install writes its own ledger row (verb train-install).

Where it runs. Without --cpu it refuses to start unless torch is a ROCm build, a GPU is visible and the GPU is the
--gfx chip (default gfx1151, the Strix Halo of the SJC pods that hold pod B's role); it checks this before it reads
anything. With --cpu it trains on the CPU, refuses a model over CPU_PARAM_CAP parameters, never deletes data, and
swaps TRL's fused LM-head loss (a Triton GPU kernel in TRL 1.15) for its plain torch equivalent. --torch-loss-head
makes that swap on the pod too, for a GPU whose Triton cannot build the kernel (the row says which head ran).

What it trains on, and what it checks first:

    --dataset   an approved export (a <dataset_id>/ folder: manifest.json, examples.jsonl, images/) or the no-camera
                fixture. It is read with lab/dataset.py load_split(), the one place membership is decided: the
                folder must hold a frozen split.json that is whole and clean, with at least MIN_HELD_OUT examples
                held out, and the folder's name must be its dataset_id.
    --rows      the JSONL lab/format_traces.py writes (id, frame_ref, messages, tools, text, spans), made with
                --split. Every row must be an example the split lists under train: a row that is held out, or that
                the dataset does not have, refuses the run. A row's frame_ref must be its example's, and an image
                row must point at its example's own image and nothing else.

Loss on the spans. Each row is tokenized whole, and only the tokens of its spans (what the model writes in each
assistant turn, from just after the empty thinking block through <|im_end|>) carry the loss. A span must start and
end on a token boundary of the whole text: the tokens of text[:start] are exactly the first tokens of the text,
which is what the box feeds the model before that turn. A row that breaks this, or that the tokenizer does not give
back unchanged, refuses the run; nothing is truncated (--max-length below the longest example refuses the run).

The model. The weights are loaded in bfloat16 by this script (TRL would load a model it is handed by name in
float32), the text-only class for text rows and the image-text class with its processor for image rows. LoRA goes on
every linear layer of the language model (PEFT's all-linear, which leaves out the LM head) and, for the image-text
class, excludes the vision encoder and its projector by VISION_MODULES; before the first step the run checks that
every trainable parameter is a LoRA weight outside the vision side, and that the base weights are bfloat16.
DataLoader workers start with forkserver; torch.compile is off. For image rows the processor expands each image
placeholder into its image tokens and the labels follow that expansion; a row whose tokens do not line up refuses
the run. The Qwen3.5 processor needs torchvision (transformers builds its video processor only with it), which the
requirements file cannot pin without replacing the pod's torch, so an image run without it is refused up front.

One ledger row per run (verb train), whatever the outcome: refs carry run_id, dataset_id, action_id, the sha256 of
examples.jsonl (line endings read as LF, as load_split reads split.json), the split's sha256, the rows file's
sha256, the base model's Hub repo and commit, the seed and data_seed, the torch, ROCm, PEFT, TRL, transformers,
accelerate and datasets versions, and the full LoraConfig and SFTConfig as JSON text (lora_config, sft_config);
metrics carry trainer.samples_per_s, measured as lab/refine/metrics.jsonl:13 measures it (examples seen, that is
steps x per-device batch x gradient accumulation, over the wall time of trainer.train()), with the steps, batch,
max_length, rank and examples beside it. The same answer is written to <out>/run.json.

Per step. Every optimizer step appends one telemetry.train.step event (contracts/telemetry.py) to
<out>/telemetry.jsonl, on node pod-b (the trainer's role; cpu for a rehearsal): run_id, step, loss (null when not
finite), step_ms, tokens, socket_w and energy_j. The trainer never reads amd-smi itself: pod B's power has one
reading path. The sampler lab/pod_power.py runs on the pod and writes telemetry.gpu.power samples to
<--power-dir>/samples-<UTC day>.jsonl (default ~/atlas-power), each with its sampler, source tool, power field and
period_ms; the run reads the samples of --power-node (gpu-b, the name the sampler is started with; pod-b means the
same pod) as they are written, and every window, a step's or the whole run's, goes through lab/pod_power.py's
summarize(), the same summary `python -m lab gpu-power record` stores. socket_w is the step's mean_w; energy_j is set
only when the window's energy came from the card's energy counter, which reads N/A on the Strix Halo, so it is null
there. A step shorter than the sampler's period can hold no reading, and with no sampler running none does: socket_w
is then null, and trainer.steps_with_power counts the steps that had one. The run's own window gives
trainer.mean_socket_w, trainer.peak_socket_w and trainer.energy_j (integrated power on the Strix Halo), with the
sampler, tool, field, period, energy source and the whole summary (power_window) in the row.

Settings. The defaults are R36's pinned baseline for this step on gfx1151 (lab/refine/research/kernels/
settings_training_gfx1151.md section 13): LoRA r 16, alpha 32, dropout 0 on all-linear; learning rate 2e-4; batch 4
by gradient accumulation 4 (effective batch 16, the one its search space holds fixed); gradient checkpointing on
with use_reentrant False; SDPA attention; fused AdamW; two forkserver DataLoader workers with pinned memory and no
persistent workers; seed 42; no packing, no length grouping, no torch.compile. On the pod the run exports the pinned
environment (MIOPEN_FIND_MODE=NORMAL, every other variable it names unset) before torch is imported. The settings of
its search space that a trainer flag can move are --no-gradient-checkpointing, --batch with --grad-accum,
--group-by-length, --aotriton-experimental, --blas-backend hipblaslt and --tunableop <results file>; packing and
torch.compile stay off. refs.settings names the run's value of every search-space setting under the search space's
own parameter names (gdn_kernel and causal_conv1d_kernel as installed), refs.env the environment it trained under.

The data is deleted when the run ends. On the pod, once the node check has passed, the dataset folder and the rows
file are deleted on every way out (trained, refused, failed or interrupted), and the row of every run that was not
interrupted records whether that worked (refs.data_deleted). A folder that does not look like a dataset export is
never deleted: refs.data_deleted says why, and a run that trained then ends data-left (exit 1). Only the adapter, the
telemetry and run.json stay, under --out (default ~/runs/<run_id>).

The entry point prints one JSON line and exits 0 when the run trained and its data is gone (or, with --cpu, kept),
1 otherwise. It refuses to run while lab/state/STOP exists. Nothing it prints or records quotes a question, an
answer or a caption.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if not __package__:  # run by path on a pod: the repo root, one folder up, is what holds contracts/ and lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import pod_power
from lab.machine import stop
from lab.machine.record import recorded

if TYPE_CHECKING:
    from lab.dataset import Split

__all__ = ["VERB", "INSTALL_VERB", "REQUIREMENTS", "DEFAULT_BASE_REPO", "POD_GFX", "POD_NODE", "POWER_NODE",
           "CPU_NODE", "VISION_MODULES", "CPU_PARAM_CAP", "IGNORE", "SAMPLES_MEASURE", "BASELINE", "PINNED_ENV",
           "TrainError", "Row", "Base", "Encoded", "PowerWindow", "PowerFeed", "StepLog", "run_env", "apply_env",
           "read_rows", "train_rows", "resolve_base", "span_labels",
           "expand_image_labels", "ImageSpanCollator", "pod_gate", "delete_data", "file_sha256", "read_pins",
           "pins_to_install", "protected_constraints", "install_command", "install", "torch_loss_head", "main"]

VERB = "train"
INSTALL_VERB = "train-install"
LAB_DIR = Path(__file__).resolve().parent
REQUIREMENTS = LAB_DIR / "requirements-train.txt"
PROTECTED = ("torch", "torchvision", "torchaudio", "triton", "pytorch-triton-rocm")   # the pod image's own builds
RESTARTED_ENV = "ATLAS_TRAIN_INSTALLED"   # set on the second start, after an install, so it cannot loop

DEFAULT_BASE_REPO = "Qwen/Qwen3.5-4B"     # the student: the base model the Jetson runs
POD_GFX = pod_power.POD_GFX               # the SJC pods' Strix Halo (Radeon 8060S), which hold pod B's role
POWER_NODE = "gpu-b"                      # the name the trainer pod's sampler is started with (pod_power.NODES)
POD_NODE = pod_power.LEDGER_NODE[POWER_NODE]   # pod-b, the trainer's role: telemetry's name (ruling 3) and the ledger's
LEDGER_NODE = POD_NODE
CPU_NODE = "cpu"                          # what a --cpu rehearsal puts in node, so it is never read as pod B
CPU_PARAM_CAP = 50_000_000                # parameters a --cpu run accepts: a tiny model, never the real student
DEFAULT_RUNS = "~/runs"
DEFAULT_POWER_DIR = pod_power.DEFAULT_DIR    # where lab/pod_power.py writes its samples on the pod
POWER_FILE = "samples-{day}.jsonl"        # one file per UTC day, named as pod_power.DailyJsonl names them
POWER_EVENT = pod_power.GPU_POWER
POWER_KEEP = pod_power.KEEP               # samples kept in memory: about 5.5 hours at one a second
HUB_COMMIT = re.compile(r"[0-9a-f]{40}")
# Module names of the vision side across image-text families (Qwen's visual tower and its merger, LLaVA-style
# vision_tower and multi_modal_projector): PEFT reads this as a full match against each module's name.
VISION_MODULES = r"(.*\.)?(visual|vision_tower|vision_model|multi_modal_projector|mm_projector)(\..*)?"
IGNORE = -100                             # the label of a token that carries no loss
SAMPLES_MEASURE = "examples seen (steps x per-device batch x gradient accumulation) over trainer.train() wall time"
SHOWN = 5                                 # problems an answer lists in full

# R36's pinned baseline for the student's LoRA step on gfx1151 (lab/refine/research/kernels/
# settings_training_gfx1151.md section 13, and the "baseline" and "pinned" of data/searchspace-training-gfx1151.json,
# which a test holds these to). A run uses it unless a flag moves one of the search space's settings, and its ledger
# row names every one of them (refs.settings) under the search space's own parameter names, so an Ax trial (R39) and
# a run of this trainer read the same way. Effective batch 16 (4 by accumulation 4) is the one the search holds fixed.
BASELINE = {"gdn_kernel": "torch", "gradient_checkpointing": True, "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4, "packing": False, "group_by_length": False,
            "aotriton_experimental": False, "torch_compile": False, "tunableop": False,
            "causal_conv1d_kernel": "torch", "blas_backend": "default"}
PINNED = {"lora_r": 16, "lora_alpha": 32.0, "lora_dropout": 0.0, "learning_rate": 2e-4, "attn_implementation": "sdpa",
          "optim": "adamw_torch_fused", "dataloader_num_workers": 2, "dataloader_pin_memory": True,
          "dataloader_persistent_workers": False, "seed": 42}
# The pinned environment: a value is exported, None is unset, so a pod profile cannot leak a value into a run.
PINNED_ENV: dict[str, str | None] = {
    "MIOPEN_FIND_MODE": "NORMAL", "PYTORCH_ALLOC_CONF": None, "PYTORCH_NO_HIP_MEMORY_CACHING": None,
    "HIPBLAS_WORKSPACE_CONFIG": None, "HIPBLASLT_ALLOW_TF32": None, "TORCH_ROCM_FA_PREFER_CK": None,
    "GPU_MAX_HW_QUEUES": None, "AMD_SERIALIZE_KERNEL": None, "HIP_FORCE_DEV_KERNARG": None, "HSA_XNACK": None,
    "HSA_ENABLE_SDMA": None, "GPU_MAX_HEAP_SIZE": None, "GPU_SINGLE_ALLOC_PERCENT": None}
KNOB_ENV = ("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "TORCH_BLAS_PREFER_HIPBLASLT", "PYTORCH_TUNABLEOP_ENABLED",
            "PYTORCH_TUNABLEOP_TUNING", "PYTORCH_TUNABLEOP_FILENAME")   # set only by the flags of their settings
BLAS_BACKENDS = ("default", "hipblaslt")

DEFAULTS = {"epochs": 3.0, "lr": PINNED["learning_rate"], "batch": BASELINE["per_device_train_batch_size"],
            "grad_accum": BASELINE["gradient_accumulation_steps"], "rank": PINNED["lora_r"],
            "alpha": PINNED["lora_alpha"], "dropout": PINNED["lora_dropout"], "seed": PINNED["seed"],
            "workers": PINNED["dataloader_num_workers"]}


class TrainError(ValueError):
    """The run cannot go ahead. problems lists every reason found; none quotes the text of a row."""

    def __init__(self, problems: Sequence[str], outcome: str = "invalid") -> None:
        self.problems = list(problems)
        self.outcome = outcome
        shown = "; ".join(self.problems[:SHOWN]) + ("; ..." if len(self.problems) > SHOWN else "")
        super().__init__(f"{len(self.problems)} problem{'s' if len(self.problems) != 1 else ''}: {shown}")


# --- the rows -------------------------------------------------------------------------------

@dataclass(frozen=True)
class Row:
    """One training row: the rendered conversation, the [start, end) character spans that carry the loss, and the
    pictures it shows (paths inside the dataset folder; none for a text row)."""

    id: str
    frame_ref: str | None
    text: str
    spans: tuple[tuple[int, int], ...]
    images: tuple[str, ...] = ()


def _inside_folder(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return bool(parts) and "\\" not in path and not path.startswith("/") and path[1:2] != ":" and ".." not in parts


def _row_problems(raw: Any) -> list[str]:
    if not isinstance(raw, dict):
        return ["is not a JSON object"]
    problems = []
    if not isinstance(raw.get("id"), str) or not raw["id"].strip():
        problems.append("has no id")
    if raw.get("frame_ref") is not None and not isinstance(raw["frame_ref"], str):
        problems.append("frame_ref is not text")
    text = raw.get("text")
    if not isinstance(text, str) or not text:
        problems.append("has no text")
        text = ""
    spans = raw.get("spans")
    if not isinstance(spans, list) or not spans:
        problems.append("has no spans")
    else:
        last = 0
        for number, span in enumerate(spans, 1):
            if (not isinstance(span, list) or len(span) != 2
                    or not all(isinstance(edge, int) and not isinstance(edge, bool) for edge in span)):
                problems.append(f"span {number} is not two whole numbers")
                continue
            start, end = span
            if not last <= start < end <= len(text):
                problems.append(f"span {number} is not inside the text after the span before it")
            last = max(last, end)
    images = raw.get("images", [])
    if not isinstance(images, list) or not all(isinstance(image, str) and _inside_folder(image) for image in images):
        problems.append("images is not a list of paths inside the dataset folder")
    return problems


def read_rows(path: str | os.PathLike[str]) -> list[Row]:
    """The rows of the JSONL file, in file order. TrainError naming every bad line (by line and id) otherwise."""
    file = Path(path)
    name = file.name
    try:
        text = file.read_bytes().decode("utf-8-sig")
    except FileNotFoundError:
        raise TrainError([f"{name}: missing"]) from None
    except (OSError, UnicodeDecodeError) as exc:
        raise TrainError([f"{name}: cannot be read ({type(exc).__name__})"]) from None
    rows: list[Row] = []
    problems: list[str] = []
    seen: dict[str, int] = {}
    for number, line in enumerate(text.split("\n"), 1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except ValueError:
            problems.append(f"{name} line {number}: not JSON")
            continue
        found = _row_problems(raw)
        key = raw.get("id") if isinstance(raw, dict) else None
        where = f"{name} line {number}" + (f" (id {key!r})" if isinstance(key, str) else "")
        if found:
            problems.extend(f"{where}: {problem}" for problem in found)
            continue
        if key in seen:
            problems.append(f"{where}: the id repeats line {seen[key]}")
            continue
        seen[key] = number
        rows.append(Row(key, raw.get("frame_ref"), raw["text"], tuple((start, end) for start, end in raw["spans"]),
                        tuple(raw.get("images", []))))
    if not rows and not problems:
        problems.append(f"{name}: has no rows")
    if problems:
        raise TrainError(problems)
    return rows


def train_rows(rows: Sequence[Row], split: Split) -> tuple[list[Row], list[str]]:
    """The rows a run may train on, in file order, and the train-side example ids that have no row (labels the
    formatter rejected, or examples still waiting for one). Membership is the split's: TrainError when a row is held
    out for the gate, is not in the dataset, names another frame_ref, or shows a picture other than its example's."""
    examples = {example.id: example for example in split.data.examples}
    train = {example.id for example in split.train}
    held_out = {example.id for example in split.held_out}
    problems = []
    for row in rows:
        example = examples.get(row.id)
        if example is None:
            problems.append(f"row {row.id!r}: the dataset has no such example")
        elif row.id in held_out:
            problems.append(f"row {row.id!r}: held out for the eval gate; no training run may see it")
        elif row.id not in train:
            problems.append(f"row {row.id!r}: not on the train side of the split")
        else:
            if row.frame_ref is not None and row.frame_ref != example.frame_ref:
                problems.append(f"row {row.id!r}: its frame_ref is not its example's")
            if row.images != ((example.image,) if example.image else ()):
                problems.append(f"row {row.id!r}: its images are not its example's own image")
    if problems:
        raise TrainError(problems)
    have = {row.id for row in rows}
    return list(rows), [example.id for example in split.train if example.id not in have]


def file_sha256(path: str | os.PathLike[str]) -> str:
    """sha256 of a text file with its line endings read as LF, so a checkout with CRLF gives the same hash."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# --- the base model ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Base:
    """The base model: the folder from_pretrained reads, the Hub repo it came from and the commit it was taken at."""

    path: str
    repo: str
    revision: str


def _folder_commit(folder: Path) -> str | None:
    """The Hub commit `hf download --local-dir` recorded for the folder (the first line of each file's
    .cache/huggingface/download/<file>.metadata); None when it recorded none. TrainError when the files disagree."""
    metadata = folder / ".cache" / "huggingface" / "download"
    commits = set()
    for file in sorted(metadata.rglob("*.metadata")) if metadata.is_dir() else ():
        try:
            first = file.read_text(encoding="utf-8").split("\n", 1)[0].strip()
        except (OSError, UnicodeDecodeError):
            continue
        if first:
            commits.add(first)
    if len(commits) > 1:
        raise TrainError([f"the model folder holds files from {len(commits)} Hub commits; download it again"],
                         outcome="refused")
    return commits.pop() if commits else None


def resolve_base(model: str, repo: str = DEFAULT_BASE_REPO, revision: str | None = None) -> Base:
    """The base model, with the Hub commit a run records. A local folder's commit is the one `hf download` wrote
    into it (and must equal --base-revision when both are known); anything else is a Hub id, downloaded at
    --base-revision (or main) into the Hugging Face cache, whose snapshot folder is named for the commit."""
    if revision is not None and not HUB_COMMIT.fullmatch(revision):
        raise TrainError(["--base-revision must be a 40-character Hub commit hash"], outcome="refused")
    folder = Path(model).expanduser()
    if folder.is_dir():
        recorded_commit = _folder_commit(folder)
        if recorded_commit is not None and revision is not None and recorded_commit != revision:
            raise TrainError(["--base-revision is not the commit the model folder was downloaded at"],
                             outcome="refused")
        commit = recorded_commit or revision
        if commit is None:
            raise TrainError(["the base model's Hub commit is unknown: the folder has no `hf download` metadata, "
                              "so pass --base-revision"], outcome="refused")
        if not HUB_COMMIT.fullmatch(commit):
            raise TrainError(["the model folder's recorded commit is not a Hub commit hash"], outcome="refused")
        return Base(str(folder), repo, commit)
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", model):
        raise TrainError([f"--model is neither a folder nor a Hub repo id ({Path(model).name})"], outcome="refused")
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(repo_id=model, revision=revision or "main"))
    if not HUB_COMMIT.fullmatch(path.name):
        raise TrainError(["the downloaded snapshot is not named for a Hub commit"], outcome="refused")
    return Base(str(path), model, path.name)


# --- labels on the spans -----------------------------------------------------------------------

@dataclass(frozen=True)
class Encoded:
    """A row as token ids, with the label of each token: the id itself on the spans, IGNORE elsewhere."""

    ids: tuple[int, ...]
    labels: tuple[int, ...]

    @property
    def trained(self) -> int:
        return sum(label != IGNORE for label in self.labels)


def span_labels(encode: Callable[[str], Sequence[int]], text: str, spans: Sequence[tuple[int, int]],
                decode: Callable[[Sequence[int]], str] | None = None) -> Encoded:
    """Token ids for the whole text and labels that put the loss on the spans only.

    The tokens of text[:start] must be the first tokens of the whole text, and the same for text[:end]: a span
    that starts or ends inside a token is one the box would tokenize differently, so it is refused (ValueError),
    as is a text the tokenizer does not give back unchanged and a span that holds no token."""
    ids = list(encode(text))
    if decode is not None and decode(ids) != text:
        raise ValueError("the tokenizer does not give the text back unchanged")
    labels = [IGNORE] * len(ids)
    for number, (start, end) in enumerate(spans, 1):
        head, upto = list(encode(text[:start])), list(encode(text[:end]))
        if ids[:len(head)] != head:
            raise ValueError(f"span {number} starts inside a token")
        if ids[:len(upto)] != upto:
            raise ValueError(f"span {number} ends inside a token")
        if len(upto) <= len(head):
            raise ValueError(f"span {number} holds no token")
        labels[len(head):len(upto)] = ids[len(head):len(upto)]
    return Encoded(tuple(ids), tuple(labels))


def expand_image_labels(ids: Sequence[int], labels: Sequence[int], expanded: Sequence[int],
                        image_token: int) -> list[int]:
    """The labels of `ids` carried over to `expanded`, the processor's tokens for the same text, in which each image
    placeholder token became a run of image tokens. Image tokens never carry loss. ValueError when the two do not
    line up: a placeholder with no run, a run where the text has none, or any other token that differs."""
    out: list[int] = []
    i = j = 0
    while i < len(ids):
        if ids[i] == image_token:
            if labels[i] != IGNORE:
                raise ValueError("an image placeholder is inside a span")
            if i + 1 < len(ids) and ids[i + 1] == image_token:
                raise ValueError("two image placeholders touch, so their image tokens cannot be told apart")
            run = j
            while run < len(expanded) and expanded[run] == image_token:
                run += 1
            if run == j:
                raise ValueError("an image placeholder has no image tokens in the processor's tokens")
            out.extend([IGNORE] * (run - j))
            i, j = i + 1, run
            continue
        if j >= len(expanded) or expanded[j] != ids[i]:
            raise ValueError("the processor's tokens do not line up with the text's")
        out.append(labels[i])
        i, j = i + 1, j + 1
    if j != len(expanded):
        raise ValueError("the processor's tokens do not line up with the text's")
    return out


class ImageSpanCollator:
    """Batches image rows for the image-text class: the processor reads each row's pictures and expands its image
    placeholders, the labels follow the expansion, and the batch is padded on the right (labels with IGNORE).

    It lives at module level so DataLoader workers started with forkserver can unpickle it."""

    def __init__(self, processor: Any, root: str | os.PathLike[str]) -> None:
        self.processor = processor
        self.root = str(root)
        tokenizer = processor.tokenizer
        self.image_token = getattr(processor, "image_token_id", None)
        if self.image_token is None:
            self.image_token = tokenizer.convert_tokens_to_ids(getattr(processor, "image_token", "<|image_pad|>"))
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        if pad is None:
            raise ValueError("the tokenizer has neither a pad token nor an end-of-sequence token to pad with")
        self.pad = pad

    def _images(self, paths: Sequence[str]) -> list[Any]:
        from PIL import Image

        pictures = []
        for relative in paths:
            with Image.open(os.path.join(self.root, *PurePosixPath(relative).parts)) as picture:
                pictures.append(picture.convert("RGB"))
        return pictures

    def encode(self, example: Mapping[str, Any]) -> dict[str, Any]:
        """One row through the processor: input_ids, attention_mask, labels, and the image tensors."""
        tokenizer = self.processor.tokenizer
        spans = [tuple(span) for span in example["spans"]]
        plain = span_labels(lambda text: tokenizer(text, add_special_tokens=False)["input_ids"], example["text"],
                            spans, lambda ids: tokenizer.decode(ids, skip_special_tokens=False))
        out = self.processor(text=[example["text"]], images=self._images(example["images"]) or None,
                             return_tensors="pt", add_special_tokens=False)
        expanded = out["input_ids"][0].tolist()
        labels = expand_image_labels(plain.ids, plain.labels, expanded, self.image_token)
        unknown = set(out) - {"input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids"}
        if unknown:
            raise ValueError(f"the processor returned inputs this collator does not batch: {sorted(unknown)}")
        encoded = {"input_ids": expanded, "attention_mask": out["attention_mask"][0].tolist(), "labels": labels}
        for key in ("pixel_values", "image_grid_thw"):
            if key in out:
                encoded[key] = out[key]
        if "mm_token_type_ids" in out:
            encoded["mm_token_type_ids"] = out["mm_token_type_ids"][0].tolist()
        return encoded

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        encoded = [self.encode(example) for example in examples]
        width = max(len(item["input_ids"]) for item in encoded)

        def padded(key: str, value: int) -> Any:
            return torch.tensor([item[key] + [value] * (width - len(item[key])) for item in encoded])

        batch = {"input_ids": padded("input_ids", self.pad), "attention_mask": padded("attention_mask", 0),
                 "labels": padded("labels", IGNORE)}
        if "mm_token_type_ids" in encoded[0]:
            batch["mm_token_type_ids"] = padded("mm_token_type_ids", 0)
        for key in ("pixel_values", "image_grid_thw"):
            parts = [item[key] for item in encoded if key in item]
            if parts:
                batch[key] = torch.cat(parts, dim=0)
        return batch


# --- per-step power, read from the pod sampler's own samples -------------------------------------

@dataclass(frozen=True)
class PowerWindow:
    """What the pod sampler's samples say about one window (a step, or the whole run), taken from
    lab/pod_power.py's summarize(), the summary `python -m lab gpu-power record` stores for a window: watts is its
    mean_w (None without a reading), samples its readings (samples that carry a power reading), then where they
    came from and the window's energy with where that came from (energy_counter, or avg_power_integral, which is the
    Strix Halo's case since its counter reads N/A). Several samplers, tools, fields or periods in one window are
    listed together, separated by "; "."""

    watts: float | None
    samples: int
    sampler: str | None = None
    source: str | None = None
    power_field: str | None = None
    period_ms: float | str | None = None
    energy_j: float | None = None
    energy_source: str | None = None
    measured_period_ms: float | None = None

    @classmethod
    def of(cls, summary: Mapping[str, Any]) -> PowerWindow:
        """The window pod_power.summarize() described."""
        period = summary.get("period_ms")
        return cls(summary.get("mean_w"), int(summary.get("readings") or 0), _listed(summary.get("sampler")),
                   _listed(summary.get("source")), _listed(summary.get("power_field")),
                   period if period is None or isinstance(period, (int, float)) else _listed(period),
                   summary.get("energy_j"), summary.get("energy_source"), summary.get("measured_period_ms"))

    @property
    def counter_energy_j(self) -> float | None:
        """The window's energy when it came from the card's energy counter, which is all TrainStep.energy_j takes."""
        return self.energy_j if self.energy_source == "energy_counter" else None


def _listed(value: Any) -> str | None:
    """summarize's one value, or its list of different values as text; None for none."""
    if value is None or value == []:
        return None
    if isinstance(value, list):
        return "; ".join(f"{item:g}" if isinstance(item, float) else str(item) for item in value)
    return f"{value:g}" if isinstance(value, float) else str(value)


def _when(text: Any) -> dt.datetime | None:
    if not isinstance(text, str):
        return None
    try:
        moment = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo is not None else None


def _sampler_node(node: str) -> str:
    """The name the pod sampler gives the pod of `node`, which may be given by either name (gpu-b or pod-b)."""
    names = {ledger: sampler for sampler, ledger in pod_power.LEDGER_NODE.items()}
    return names.get(node, node)


class PowerFeed:
    """The pod sampler's telemetry.gpu.power samples for one node, read from its daily files as they grow, and
    summarized per window by lab/pod_power.py's own summarize(): one reading path for pod B's power across this
    trainer, the sampler and `python -m lab gpu-power record`.

    The sampler names each file for the UTC day of its samples, so a window reads only the files of the days it
    spans, and each file from where the last read stopped. Only whole lines are read (a line the sampler is still
    writing waits for the next read); lines that are not JSON or have no time are counted in bad_lines, and other
    event types and other nodes are skipped. The node may be given by either of its names (gpu-b, pod-b), and samples
    carrying either are read as the sampler's. A socket_w that is not a finite number is read as no reading. The
    newest POWER_KEEP samples are kept."""

    def __init__(self, directory: str | os.PathLike[str], node: str = POWER_NODE) -> None:
        self.directory = Path(directory).expanduser()
        self.node = _sampler_node(node)
        self._names = {self.node, pod_power.LEDGER_NODE.get(self.node, self.node)}
        self.samples: deque[tuple[dt.datetime, dict[str, Any]]] = deque(maxlen=POWER_KEEP)
        self.bad_lines = 0
        self._offsets: dict[Path, int] = {}
        self._pending: dict[Path, bytes] = {}

    def _read_new(self, first: dt.date, last: dt.date) -> None:
        day = first
        while day <= last:
            file = self.directory / POWER_FILE.format(day=day.isoformat())
            day += dt.timedelta(days=1)
            if not file.is_file():
                continue
            try:
                with file.open("rb") as handle:
                    handle.seek(self._offsets.get(file, 0))
                    chunk = handle.read()
                    self._offsets[file] = handle.tell()
            except OSError:
                continue
            *lines, rest = (self._pending.pop(file, b"") + chunk).split(b"\n")
            if rest:
                self._pending[file] = rest
            for line in lines:
                self._take(line)

    def _take(self, line: bytes) -> None:
        if not line.strip():
            return
        try:
            event = json.loads(line)
        except ValueError:
            self.bad_lines += 1
            return
        if not isinstance(event, dict) or event.get("type") != POWER_EVENT or event.get("node") not in self._names:
            return
        moment = _when(event.get("ts"))
        if moment is None:
            self.bad_lines += 1
            return
        watts = event.get("socket_w")
        if not (isinstance(watts, (int, float)) and not isinstance(watts, bool) and math.isfinite(watts)):
            watts = None
        self.samples.append((moment, {**event, "node": self.node, "socket_w": watts}))

    def summary(self, start: dt.datetime, end: dt.datetime) -> dict[str, Any] | None:
        """pod_power.summarize() over the samples with start <= ts <= end, after reading what the sampler has
        written since the last call; None when no sample falls inside (or the samples cannot be summarized)."""
        self._read_new(start.astimezone(dt.timezone.utc).date(), end.astimezone(dt.timezone.utc).date())
        inside = [event for moment, event in self.samples if start <= moment <= end]
        if not inside:
            return None
        try:
            return pod_power.summarize(inside, node=self.node)
        except (ValueError, TypeError, KeyError):
            return None

    def window(self, start: dt.datetime, end: dt.datetime) -> PowerWindow:
        """The PowerWindow of start <= ts <= end; PowerWindow(None, 0) when no sample falls inside."""
        summary = self.summary(start, end)
        return PowerWindow(None, 0) if summary is None else PowerWindow.of(summary)

    def kept_since(self, start: dt.datetime) -> bool:
        """Whether every sample since `start` is still held (False once POWER_KEEP newer ones pushed some out)."""
        return len(self.samples) < POWER_KEEP or self.samples[0][0] <= start


# --- one telemetry.train.step event per optimizer step ------------------------------------------

@dataclass
class StepLog:
    """Times each optimizer step and, once its loss is logged, appends its telemetry.train.step event to `path`.

    begin() and end() bracket a step (the trainer's on_step_begin and on_step_end), logged() takes the trainer's
    log of the same step; `clock` gives wall time in seconds since the epoch (for ts and the power window) and
    `timer` a monotonic time for step_ms. socket_w is the mean_w of pod_power.summarize() over the step's samples,
    and energy_j the window's energy only when it came from the card's energy counter (as TrainStep defines it);
    `node` is the trainer pod's role, pod-b (cpu for a rehearsal)."""

    path: Path
    run_id: str
    node: str
    power: PowerFeed | None = None
    clock: Callable[[], float] = time.time
    timer: Callable[[], float] = time.perf_counter
    events: list[dict[str, Any]] = field(default_factory=list)
    windows: list[PowerWindow] = field(default_factory=list)
    tokens_total: int | None = None    # the trainer's running count of tokens seen, at the last logged step
    _started: tuple[float, float] | None = None
    _ended: dict[int, tuple[float, float, float]] = field(default_factory=dict)

    def begin(self) -> None:
        self._started = (self.clock(), self.timer())

    def end(self, step: int) -> None:
        if self._started is None:
            return
        wall, mark = self._started
        self._ended[step] = (wall, self.clock(), (self.timer() - mark) * 1000.0)
        self._started = None

    def logged(self, step: int, logs: Mapping[str, Any]) -> dict[str, Any] | None:
        """The event of a step whose log carries its loss; None for any other log (the run's closing summary)."""
        if "loss" not in logs or step not in self._ended:
            return None
        from contracts.telemetry import TrainStep

        began, ended, step_ms = self._ended.pop(step)
        loss = logs.get("loss")
        loss = float(loss) if isinstance(loss, (int, float)) and math.isfinite(loss) else None
        tokens = None
        total = logs.get("num_tokens")
        if isinstance(total, (int, float)) and math.isfinite(total):
            tokens = int(total) - (self.tokens_total or 0)
            self.tokens_total = int(total)
        window = PowerWindow(None, 0)
        if self.power is not None:
            window = self.power.window(dt.datetime.fromtimestamp(began, dt.timezone.utc),
                                       dt.datetime.fromtimestamp(ended, dt.timezone.utc))
        event = TrainStep(ts=dt.datetime.fromtimestamp(ended, dt.timezone.utc), node=self.node, run_id=self.run_id,
                          step=step, loss=loss, step_ms=round(step_ms, 3), socket_w=window.watts,
                          energy_j=window.counter_energy_j, tokens=tokens)
        with self.path.open("a", encoding="utf-8", newline="\n") as out:
            out.write(event.model_dump_json() + "\n")
        record = json.loads(event.model_dump_json())
        self.events.append(record)
        self.windows.append(window)
        return record


def _callback(steps: StepLog) -> Any:
    from transformers import TrainerCallback

    class Steps(TrainerCallback):
        def on_step_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            steps.begin()

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            steps.end(state.global_step)

        def on_log(self, args: Any, state: Any, control: Any, logs: Mapping[str, Any] | None = None,
                   **kwargs: Any) -> None:
            steps.logged(state.global_step, logs or {})

    return Steps()


# --- the pinned environment and the search space's settings -----------------------------------------

def run_env(aotriton: bool = False, blas: str = "default", tunableop: str | None = None) -> dict[str, str | None]:
    """The environment a run on the pod trains under: PINNED_ENV, and the variables of the three environment
    settings of the search space (aotriton_experimental, blas_backend, tunableop with tuning off reading the results
    file named), each unset unless its flag asks for it. A value is exported, None is unset."""
    if blas not in BLAS_BACKENDS:
        raise ValueError(f"blas backend {blas!r} is not one of {', '.join(BLAS_BACKENDS)}")
    env = dict(PINNED_ENV)
    env.update(dict.fromkeys(KNOB_ENV))
    if aotriton:
        env["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
    if blas == "hipblaslt":
        env["TORCH_BLAS_PREFER_HIPBLASLT"] = "1"
    if tunableop:
        env.update(PYTORCH_TUNABLEOP_ENABLED="1", PYTORCH_TUNABLEOP_TUNING="0", PYTORCH_TUNABLEOP_FILENAME=tunableop)
    return env


def apply_env(env: Mapping[str, str | None], environ: Any = None) -> None:
    """Export each value of `env` and unset each None, in os.environ unless another mapping is given. On the pod
    this runs before torch is imported, so the HIP runtime and torch's ROCm switches read the pinned values."""
    environ = os.environ if environ is None else environ
    for name, value in env.items():
        if value is None:
            environ.pop(name, None)
        else:
            environ[name] = value


# --- where it runs, and the data afterwards ----------------------------------------------------------

def pod_gate(torch: Any, gfx: str = POD_GFX, rocminfo: Callable[[], str] | None = None) -> str | None:
    """Why this process may not train here, or None when it may: torch is a ROCm build, a GPU is visible, and the
    GPU (torch's gcnArchName, else rocminfo) is the expected chip."""
    if not getattr(getattr(torch, "version", None), "hip", None):
        return "this torch is not a ROCm build, so this is not the trainer pod (use --cpu for a rehearsal)"
    if not torch.cuda.is_available():
        return "no GPU is visible to torch, so this is not the trainer pod (use --cpu for a rehearsal)"
    arch = str(getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or "")
    if not arch and rocminfo is not None:
        try:
            arch = gfx if re.search(rf"\b{re.escape(gfx)}\b", rocminfo()) else "another chip"
        except (OSError, subprocess.SubprocessError):
            arch = ""
    if arch.split(":")[0] != gfx:
        return f"the GPU is {arch or 'of an unknown kind'}, not {gfx}: this is not the trainer pod"
    return None


def _rocminfo() -> str:
    tool = shutil.which("rocminfo") or "/opt/rocm/bin/rocminfo"
    return subprocess.run([tool], capture_output=True, text=True, timeout=60, check=False).stdout


def delete_data(dataset: str | os.PathLike[str], rows: str | os.PathLike[str]) -> tuple[bool, str]:
    """Delete the run's copy of the dataset folder and of the rows file; (True, "deleted") when both are gone.

    A folder is deleted only when it looks like a dataset export (manifest.json and examples.jsonl at its top) and
    is not the home folder, a folder above it, or a checkout (.git): anything else is left and the reason given."""
    from lab.dataset import EXAMPLES, MANIFEST

    reasons = []
    folder = Path(dataset).expanduser()
    if folder.exists():
        real = folder.resolve()
        home = Path.home().resolve()
        if real == Path(real.anchor) or real == home or home.is_relative_to(real):
            reasons.append("the dataset path is the home folder or above it")
        elif not ((real / MANIFEST).is_file() and (real / EXAMPLES).is_file()) or (real / ".git").exists():
            reasons.append("the dataset path does not look like a dataset export, so it was left")
        else:
            try:
                shutil.rmtree(real)
            except OSError as exc:
                reasons.append(f"the dataset folder could not be deleted ({type(exc).__name__})")
    file = Path(rows).expanduser()
    if file.is_dir():
        reasons.append("the rows path is a folder, so it was left")
    else:
        try:
            file.unlink(missing_ok=True)
        except OSError as exc:
            reasons.append(f"the rows file could not be deleted ({type(exc).__name__})")
    if folder.exists() and not reasons:
        reasons.append("the dataset folder is still there")
    return (False, "; ".join(reasons)) if reasons else (True, "deleted")


# --- the pinned requirements ----------------------------------------------------------------------

def read_pins(path: str | os.PathLike[str] = REQUIREMENTS) -> dict[str, str]:
    """{package: version} for every `name==version` line of the requirements file."""
    pins = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            name, sep, version = line.partition("==")
            if not sep or not name.strip() or not version.strip():
                raise ValueError(f"{Path(path).name}: {line!r} is not name==version")
            pins[name.strip()] = version.strip()
    return pins


def _installed(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def pins_to_install(pins: Mapping[str, str], version_of: Callable[[str], str | None] = _installed) -> list[str]:
    """The pins that are missing or installed at another version, as name==version."""
    return [f"{name}=={version}" for name, version in pins.items() if version_of(name) != version]


def protected_constraints(version_of: Callable[[str], str | None] = _installed) -> list[str]:
    """name==version for each of PROTECTED that is installed: pip may resolve around them, never replace them."""
    return [f"{name}=={version}" for name in PROTECTED if (version := version_of(name)) is not None]


def install_command(python: str, requirements: str, constraints: str, in_venv: bool) -> list[str]:
    command = [python, "-m", "pip", "install", "--disable-pip-version-check", "--no-input"]
    if not in_venv:
        command.append("--user")
    return command + ["-r", requirements, "-c", constraints]


@recorded(INSTALL_VERB)
def install(requirements: str | os.PathLike[str] = REQUIREMENTS,
            run: Callable[[list[str]], int] | None = None,
            version_of: Callable[[str], str | None] = _installed) -> dict[str, Any]:
    """Install the pinned training stack around the torch this machine already has."""
    pins = read_pins(requirements)
    missing = pins_to_install(pins, version_of)
    refs = {"requirements_sha256": file_sha256(requirements), "missing": ", ".join(missing) or None}
    if not missing:
        return {"ok": True, "outcome": "present", "refs": refs}
    if version_of("torch") is None:
        return {"ok": False, "outcome": "refused", "refs": refs,
                "error": "torch is not installed; install the platform's own torch first (the pod image has it), "
                         "so pip never brings one from PyPI"}
    constraints = protected_constraints(version_of)
    refs["constraints"] = ", ".join(constraints)
    handle, name = tempfile.mkstemp(prefix="train-constraints-", suffix=".txt")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write("\n".join(constraints) + "\n")
        command = install_command(sys.executable, str(requirements), name, sys.prefix != sys.base_prefix)
        code = (run or (lambda argv: subprocess.run(argv, check=False).returncode))(command)
    finally:
        Path(name).unlink(missing_ok=True)
    if code != 0:
        return {"ok": False, "outcome": "failed", "refs": refs, "error": f"pip exited with {code}"}
    still = pins_to_install(pins, version_of)
    if still:
        return {"ok": False, "outcome": "failed", "refs": refs,
                "error": f"pip finished but these pins are still not installed: {', '.join(still)}"}
    return {"ok": True, "outcome": "installed", "refs": refs}


# --- training ---------------------------------------------------------------------------------------

@contextlib.contextmanager
def torch_loss_head() -> Any:
    """For a --cpu run: TRL 1.15 computes the SFT loss with a Triton GPU kernel (trl.kernels.ChunkedLogProbFunction,
    bound in trl.trainer.utils); inside this block the same computation runs in plain torch, which autograd
    differentiates, and the kernel is put back afterwards."""
    import torch
    import trl.trainer.utils as trl_utils

    if not hasattr(trl_utils, "_ChunkedLogProbFunction"):
        raise RuntimeError("this TRL has no _ChunkedLogProbFunction to replace; check the pinned TRL version")
    kernel = trl_utils._ChunkedLogProbFunction

    class TorchLogProb:
        @staticmethod
        def apply(hidden: Any, weight: Any, bias: Any, targets: Any, temperature: float, chunk_size: int,
                  softcap: float | None = None, logit_scale: float = 1.0, outputs: Sequence[str] = ()) -> tuple:
            logits = torch.nn.functional.linear(hidden, weight, bias).float() * logit_scale
            if softcap:
                logits = softcap * torch.tanh(logits / softcap)
            logits = logits / temperature
            logp = torch.log_softmax(logits, dim=-1)
            log_probs = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            entropy = -(logp.exp() * logp).sum(-1) if "entropy" in outputs else None
            log_sum_sq = torch.logsumexp(2 * logp, dim=-1).detach() if "log_sum_sq_probs" in outputs else None
            mean_logits = logits.mean(-1).detach() if "mean_logits" in outputs else None
            is_top1 = (logits.argmax(-1) == targets) if "is_top1" in outputs else None
            return log_probs, entropy, log_sum_sq, mean_logits, is_top1

    trl_utils._ChunkedLogProbFunction = TorchLogProb
    try:
        yield
    finally:
        trl_utils._ChunkedLogProbFunction = kernel


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return getattr(value, "value", None) if hasattr(value, "value") and isinstance(value.value, str) else str(value)


def _config_text(config: Mapping[str, Any]) -> str:
    return json.dumps(_jsonable(config), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class Plan:
    """Everything a run needs once its inputs have been checked."""

    run_id: str
    rows: tuple[Row, ...]
    dataset: Path
    base: Base
    out: Path
    cpu: bool
    node: str
    epochs: float
    max_steps: int | None
    lr: float
    batch: int
    grad_accum: int
    rank: int
    alpha: float
    dropout: float
    max_length: int | None
    seed: int
    data_seed: int
    workers: int
    gradient_checkpointing: bool
    power: PowerFeed | None
    torch_loss: bool = False     # the loss head in plain torch instead of TRL's Triton kernel (always with --cpu)
    group_by_length: bool = False
    env: Mapping[str, str | None] = field(default_factory=run_env)   # what the run trains under (run_env)
    env_applied: bool = False    # exported before torch was imported (on the pod; never in a --cpu rehearsal)


@dataclass
class Trained:
    """What a finished trainer.train() measured, and the configs it ran with."""

    metrics: dict[str, float]
    refs: dict[str, Any]


def _weights_bytes(folder: Path) -> int:
    """Bytes of the weight files in a model folder: an upper bound a --cpu run checks before it loads anything."""
    return sum(file.stat().st_size for pattern in ("*.safetensors", "*.bin") for file in folder.glob(pattern))


def _versions(torch: Any) -> dict[str, Any]:
    out = {"torch": torch.__version__, "rocm": getattr(torch.version, "hip", None) or None,
           "python": platform.python_version()}
    for name in ("peft", "trl", "transformers", "accelerate", "datasets", "tokenizers"):
        out[name] = _installed(name)
    return out


def _check_frozen(model: Any) -> tuple[int, int]:
    """(trainable, total) parameters, after checking that every trainable one is a LoRA weight outside the vision
    side. TrainError otherwise: the run would change the base, or the vision tower the package ships."""
    trainable = total = 0
    wrong = []
    for name, parameter in model.named_parameters():
        total += parameter.numel()
        if not parameter.requires_grad:
            continue
        trainable += parameter.numel()
        inner = re.sub(r"^(base_model\.model\.)", "", name)
        if ".lora_" not in name or re.fullmatch(VISION_MODULES, inner.rsplit(".lora_", 1)[0]):
            wrong.append(inner)
    if wrong:
        raise TrainError([f"{len(wrong)} trainable parameters are not language-model LoRA weights, for example "
                          f"{wrong[0]}"], outcome="refused")
    if not trainable:
        raise TrainError(["LoRA put no trainable parameter on the model"], outcome="refused")
    return trainable, total


def _settings(config: Any, plan: Plan) -> dict[str, Any]:
    """The run's value of every setting in R36's search space, under the search space's parameter names."""
    return {
        "gdn_kernel": "fla" if _installed("flash-linear-attention") else "torch",
        "gradient_checkpointing": bool(config.gradient_checkpointing),
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "packing": bool(config.packing), "group_by_length": config.train_sampling_strategy == "group_by_length",
        "aotriton_experimental": plan.env.get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL") == "1",
        "torch_compile": bool(config.torch_compile), "tunableop": plan.env.get("PYTORCH_TUNABLEOP_ENABLED") == "1",
        "causal_conv1d_kernel": "causal_conv1d" if _installed("causal-conv1d") else "torch",
        "blas_backend": "hipblaslt" if plan.env.get("TORCH_BLAS_PREFER_HIPBLASLT") == "1" else "default",
    }


def train(plan: Plan, callbacks: Sequence[Any] = ()) -> Trained:
    """Load the base in bfloat16, put LoRA on its language model, train on the rows, save the adapter. callbacks
    are more TrainerCallbacks for the trainer, after the step log's (lab/profile_train.py steps its profiler so)."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    image = any(row.images for row in plan.rows)
    if image:
        if _installed("torchvision") is None:
            raise TrainError(["image rows need torchvision: the Qwen3.5 processor builds a video processor, which "
                              "transformers makes only with torchvision. Use the torchvision built for this torch "
                              "(on the pod, the ROCm build that matches torch 2.9.1+rocm7.13.0), never PyPI's"],
                             outcome="refused")
        processing = AutoProcessor.from_pretrained(plan.base.path)
        tokenizer = processing.tokenizer
        model_class = AutoModelForImageTextToText
    else:
        processing = tokenizer = AutoTokenizer.from_pretrained(plan.base.path)
        model_class = AutoModelForCausalLM
    if plan.cpu and _weights_bytes(Path(plan.base.path)) > 4 * CPU_PARAM_CAP:
        raise TrainError([f"--cpu takes a model of at most {CPU_PARAM_CAP:,} parameters; this folder's weights are "
                          "larger"], outcome="refused")
    model = model_class.from_pretrained(plan.base.path, dtype=torch.bfloat16,
                                        attn_implementation=PINNED["attn_implementation"])
    model.config.use_cache = False
    dtypes = sorted({str(p.dtype).replace("torch.", "") for p in model.parameters() if p.is_floating_point()})
    if dtypes != ["bfloat16"]:
        raise TrainError([f"the base weights loaded as {', '.join(dtypes)}, not bfloat16"], outcome="refused")
    parameters = sum(p.numel() for p in model.parameters())
    if plan.cpu and parameters > CPU_PARAM_CAP:
        raise TrainError([f"--cpu takes a model of at most {CPU_PARAM_CAP:,} parameters; this one has "
                          f"{parameters:,}"], outcome="refused")

    problems: list[str] = []
    encoded: list[Encoded] = []
    lengths: list[int] = []
    trained_tokens = 0
    collator = None
    if image:
        collator = ImageSpanCollator(processing, plan.dataset)
        for row in plan.rows:   # each row through the processor once, for its length; its tensors are not kept
            try:
                item = collator.encode({"text": row.text, "spans": row.spans, "images": row.images})
            except (ValueError, OSError) as why:
                problems.append(f"row {row.id!r}: {why}")
                continue
            lengths.append(len(item["input_ids"]))
            trained_tokens += sum(label != IGNORE for label in item["labels"])
    else:
        def encode(text: str) -> list[int]:
            return tokenizer(text, add_special_tokens=False)["input_ids"]

        def decode(ids: Sequence[int]) -> str:
            return tokenizer.decode(list(ids), skip_special_tokens=False)

        for row in plan.rows:
            try:
                encoded.append(span_labels(encode, row.text, row.spans, decode))
            except ValueError as why:
                problems.append(f"row {row.id!r}: {why}")
        lengths = [len(item.ids) for item in encoded]
        trained_tokens = sum(item.trained for item in encoded)
    if problems:
        raise TrainError(problems)
    longest = max(lengths)
    if plan.max_length is not None and plan.max_length < longest:
        cut = sum(length > plan.max_length for length in lengths)
        raise TrainError([f"--max-length {plan.max_length} would cut {cut} of {len(lengths)} examples (the longest "
                          f"is {longest} tokens); nothing is truncated, so raise it or leave it unset"],
                         outcome="refused")
    if image:   # a length column for group_by_length: the processor's tokens, which the default lengths cannot see
        data = Dataset.from_list([{"text": row.text, "spans": [list(span) for span in row.spans],
                                   "images": list(row.images), **({"length": length} if plan.group_by_length else {})}
                                  for row, length in zip(plan.rows, lengths)])
    else:
        data = Dataset.from_list([{"input_ids": list(item.ids), "labels": list(item.labels)} for item in encoded])

    vision = any(re.fullmatch(VISION_MODULES, name) for name, _ in model.named_modules())
    lora = LoraConfig(r=plan.rank, lora_alpha=plan.alpha, lora_dropout=plan.dropout, target_modules="all-linear",
                      exclude_modules=VISION_MODULES if vision else None, bias="none", task_type="CAUSAL_LM")
    settings: dict[str, Any] = dict(
        output_dir=str(plan.out / "trainer"), num_train_epochs=plan.epochs, max_steps=plan.max_steps or -1,
        per_device_train_batch_size=plan.batch, gradient_accumulation_steps=plan.grad_accum,
        learning_rate=plan.lr, bf16=True, logging_strategy="steps", logging_steps=1, logging_nan_inf_filter=False,
        save_strategy="no", eval_strategy="no", report_to=[], disable_tqdm=True,
        dataloader_num_workers=plan.workers, dataloader_persistent_workers=PINNED["dataloader_persistent_workers"],
        dataloader_pin_memory=PINNED["dataloader_pin_memory"] and not plan.cpu, optim=PINNED["optim"],
        torch_compile=False, max_length=plan.max_length, packing=False, seed=plan.seed, data_seed=plan.data_seed,
        use_cpu=plan.cpu, gradient_checkpointing=plan.gradient_checkpointing,
        train_sampling_strategy="group_by_length" if plan.group_by_length else "random",
        dataset_kwargs={"skip_prepare_dataset": True}, remove_unused_columns=not image)
    if plan.gradient_checkpointing:
        settings["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if plan.workers:
        if "forkserver" not in multiprocessing.get_all_start_methods():
            raise TrainError(["DataLoader workers start with forkserver, which this platform lacks; pass "
                              "--workers 0"], outcome="refused")
        settings["dataloader_multiprocessing_context"] = "forkserver"
    config = SFTConfig(**settings)
    steps = StepLog(plan.out / "telemetry.jsonl", plan.run_id, plan.node, plan.power)
    trainer = SFTTrainer(model=model, args=config, train_dataset=data, processing_class=processing,
                         peft_config=lora, data_collator=collator, callbacks=[_callback(steps), *callbacks])
    trainable, total = _check_frozen(trainer.model)
    if config.torch_compile or config.packing:
        raise TrainError(["torch.compile or packing is on; both stay off (R36: packing needs fla and a varlen "
                          "attention path on gfx1151)"], outcome="refused")
    if torch.cuda.is_available() and not plan.cpu:
        torch.cuda.reset_peak_memory_stats()

    with torch_loss_head() if plan.torch_loss else contextlib.nullcontext():
        began = dt.datetime.now(dt.timezone.utc)
        clock = time.perf_counter()
        result = trainer.train()
        seconds = time.perf_counter() - clock
        ended = dt.datetime.now(dt.timezone.utc)
    trainer.save_model(str(plan.out / "adapter"))
    # The whole run's power, summarized exactly as `python -m lab gpu-power record` summarizes a window.
    run_power = plan.power.summary(began, ended) if plan.power is not None else None

    state = trainer.state
    seen = state.global_step * config.per_device_train_batch_size * config.gradient_accumulation_steps
    losses = [event["loss"] for event in steps.events if event["loss"] is not None]
    powered = [window for window in steps.windows if window.watts is not None]
    metrics = {
        "trainer.samples_per_s": seen / seconds if seconds > 0 else 0.0, "trainer.train_s": seconds,
        "trainer.steps": state.global_step, "trainer.examples_seen": seen, "trainer.examples": len(plan.rows),
        "trainer.epochs": state.epoch or 0.0, "trainer.train_loss": result.training_loss,
        "trainer.loss_first": losses[0] if losses else None, "trainer.loss_last": losses[-1] if losses else None,
        "trainer.tokens": steps.tokens_total, "trainer.span_tokens": trained_tokens,
        "trainer.longest_example_tokens": longest, "trainer.trainable_params": trainable,
        "trainer.base_params": total - trainable, "trainer.steps_with_power": len(powered),
        "trainer.mean_socket_w": run_power["mean_w"] if run_power else None,
        "trainer.peak_socket_w": run_power["max_w"] if run_power else None,
        "trainer.energy_j": run_power["energy_j"] if run_power else None,
        "trainer.batch": config.per_device_train_batch_size, "trainer.grad_accum": config.gradient_accumulation_steps,
        "trainer.lora_rank": plan.rank,
    }
    if config.max_length is not None:
        metrics["trainer.max_length"] = config.max_length
    if torch.cuda.is_available() and not plan.cpu:
        metrics["trainer.peak_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
    device = "cpu"
    if not plan.cpu and torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        device = f"{torch.cuda.get_device_name(0)} ({getattr(properties, 'gcnArchName', '') or 'unknown arch'})"
        metrics["trainer.gpu_mem_total_gb"] = properties.total_memory / 2**30   # 64 GB shared on the Strix Halo
    effective_lora = trainer.model.peft_config["default"].to_dict()
    refs = {
        "model_class": type(model).__name__, "weights_dtype": "bfloat16", "device": device,
        "loss_head": "torch" if plan.torch_loss else "trl triton kernel",
        "deltanet_kernel": "fla" if _installed("flash-linear-attention") else "torch fallback",
        "settings": _config_text(_settings(config, plan)),
        "effective_batch": config.per_device_train_batch_size * config.gradient_accumulation_steps,
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "env": _config_text(plan.env), "env_applied": "yes" if plan.env_applied else "no: a --cpu rehearsal",
        "lora_config": _config_text(effective_lora), "sft_config": _config_text(config.to_dict()),
        "samples_measure": SAMPLES_MEASURE, "adapter": str(plan.out / "adapter"),
        "telemetry": str(plan.out / "telemetry.jsonl"),
        **_versions(torch),
    }
    if run_power is not None:
        whole = PowerWindow.of(run_power)
        refs.update(power_sampler=whole.sampler, power_source=whole.source, power_field=whole.power_field,
                    power_period_ms=_listed(whole.period_ms), power_energy_source=whole.energy_source,
                    power_window=_config_text(run_power))
        if plan.power is not None and not plan.power.kept_since(began):
            refs["power_window_note"] = f"the run outlasted the {POWER_KEEP} samples kept, so the window is its end"
    else:
        refs["power_source"] = "none: no sampler sample fell inside the run"
    return Trained({key: value for key, value in metrics.items() if value is not None}, refs)


# --- the entry point ------------------------------------------------------------------------------------

def _run_id() -> str:
    return f"train-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"


def _positive(kind: type) -> Callable[[str], Any]:
    def parse(text: str) -> Any:
        value = kind(text)
        if not value > 0 or (isinstance(value, float) and not math.isfinite(value)):
            raise argparse.ArgumentTypeError(f"{text!r} is not a positive number")
        return value
    return parse


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python lab/train.py", allow_abbrev=False,
                                     description="LoRA SFT of the student on the training rows of an approved export.")
    parser.add_argument("--dataset", required=True, help="the <dataset_id> folder with its frozen split.json")
    parser.add_argument("--rows", required=True, help="the JSONL lab/format_traces.py --split wrote for it")
    parser.add_argument("--model", required=True, help="the base model: a folder (hf download --local-dir) or a Hub id")
    parser.add_argument("--base-repo", default=DEFAULT_BASE_REPO, help="the Hub repo a model folder came from")
    parser.add_argument("--base-revision", help="the Hub commit of the base, when the folder does not record it")
    parser.add_argument("--out", help=f"where the adapter, telemetry and run.json go (default {DEFAULT_RUNS}/<run_id>)")
    parser.add_argument("--run-id", help="the run's id (default train-<UTC time>-<6 hex>)")
    parser.add_argument("--epochs", type=_positive(float), default=DEFAULTS["epochs"])
    parser.add_argument("--max-steps", type=_positive(int), help="stop after this many optimizer steps")
    parser.add_argument("--lr", type=_positive(float), default=DEFAULTS["lr"])
    parser.add_argument("--batch", type=_positive(int), default=DEFAULTS["batch"], help="per-device batch size")
    parser.add_argument("--grad-accum", type=_positive(int), default=DEFAULTS["grad_accum"])
    parser.add_argument("--rank", type=_positive(int), default=DEFAULTS["rank"])
    parser.add_argument("--alpha", type=_positive(float), default=DEFAULTS["alpha"])
    parser.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    parser.add_argument("--max-length", type=_positive(int), help="refuse examples longer than this (default: none)")
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--data-seed", type=int, help="the data order's seed (default: --seed)")
    parser.add_argument("--workers", type=int, default=DEFAULTS["workers"], help="DataLoader workers (forkserver)")
    checkpointing = parser.add_mutually_exclusive_group()
    checkpointing.add_argument("--gradient-checkpointing", dest="checkpointing", action="store_true",
                               default=BASELINE["gradient_checkpointing"], help="on by default (R36's baseline)")
    checkpointing.add_argument("--no-gradient-checkpointing", dest="checkpointing", action="store_false")
    parser.add_argument("--group-by-length", action="store_true", help="length-grouped sampling (R36's TS14)")
    parser.add_argument("--aotriton-experimental", action="store_true",
                        help="TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1: SDPA's AOTriton kernels on gfx1151 (TS6)")
    parser.add_argument("--blas-backend", choices=BLAS_BACKENDS, default=BASELINE["blas_backend"],
                        help="hipblaslt sets TORCH_BLAS_PREFER_HIPBLASLT=1 (TS4)")
    parser.add_argument("--tunableop", metavar="RESULTS_CSV",
                        help="read this TunableOp results file with tuning off (TS5)")
    parser.add_argument("--power-dir", default=DEFAULT_POWER_DIR, help="the pod sampler's samples folder")
    parser.add_argument("--power-node", default=POWER_NODE,
                        help="the pod whose sampler samples are read (gpu-b or pod-b, the same pod)")
    parser.add_argument("--gfx", default=POD_GFX, help="the GPU the trainer pod must have")
    parser.add_argument("--cpu", action="store_true", help="a rehearsal on the CPU with a tiny model")
    parser.add_argument("--torch-loss-head", action="store_true",
                        help="compute the loss in plain torch instead of TRL's Triton kernel (always on with --cpu), "
                             "for a GPU whose Triton cannot build that kernel; it holds the batch's full logits")
    parser.add_argument("--no-install", action="store_true", help="do not install lab/requirements-train.txt")
    return parser


def _metrics(values: Mapping[str, Any]) -> dict[str, float]:
    return {key: value for key, value in values.items() if isinstance(value, (int, float))
            and not isinstance(value, bool) and math.isfinite(value)}


def _refs(values: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in values.items():
        if value is None or (isinstance(value, (str, int)) and not isinstance(value, bool)):
            out[key] = value
        else:
            out[key] = json.dumps(value) if isinstance(value, bool) else str(value)
    return out


def _answer(outcome: str, refs: Mapping[str, Any], metrics: Mapping[str, Any] | None = None,
            error: str | None = None, problems: Sequence[str] = ()) -> dict[str, Any]:
    answer: dict[str, Any] = {"ok": error is None, "outcome": outcome, "refs": _refs(refs),
                              "metrics": _metrics(metrics or {})}
    if problems:
        answer["problems"] = list(problems)
    if error is not None:
        answer["error"] = error
    return answer


def _prepare(args: argparse.Namespace, run_id: str, refs: dict[str, Any], node: str) -> Plan:
    from lab.dataset import EXAMPLES, DatasetError, load_split

    if not 0 <= args.dropout < 1:
        raise TrainError(["--dropout must be at least 0 and below 1"], outcome="usage")
    if args.workers < 0:
        raise TrainError(["--workers must be 0 or more"], outcome="usage")
    if args.tunableop and not Path(args.tunableop).expanduser().is_file():
        raise TrainError(["--tunableop names no results file; tune once at this sequence length first"],
                         outcome="usage")
    dataset = Path(args.dataset).expanduser()
    rows = read_rows(Path(args.rows).expanduser())
    try:
        split = load_split(dataset, require_images=any(row.images for row in rows))
    except DatasetError as err:
        raise TrainError(err.problems) from None
    refs.update(dataset_id=split.dataset_id, action_id=split.data.manifest.action_id,
                task=split.data.manifest.spec.task, split_sha256=split.sha256,
                examples_sha256=file_sha256(dataset / EXAMPLES), rows_sha256=file_sha256(Path(args.rows).expanduser()))
    if dataset.resolve().name != split.dataset_id:
        raise TrainError([f"the dataset folder is not named for its dataset_id ({split.dataset_id!r})"])
    chosen, without_row = train_rows(rows, split)
    refs["train_without_row"] = len(without_row)
    refs["held_out_examples"] = len(split.held_out)
    base = resolve_base(args.model, args.base_repo, args.base_revision)
    refs.update(base_model=base.repo, base_revision=base.revision)
    out = Path(args.out).expanduser() if args.out else Path(DEFAULT_RUNS).expanduser() / run_id
    if out.resolve().is_relative_to(dataset.resolve()):
        raise TrainError(["--out is inside the dataset folder, which is deleted after the run"], outcome="usage")
    out.mkdir(parents=True, exist_ok=True)
    seed = args.seed
    data_seed = args.seed if args.data_seed is None else args.data_seed
    refs.update(seed=seed, data_seed=data_seed, out=str(out))
    return Plan(run_id, tuple(chosen), dataset, base, out, args.cpu, node, args.epochs, args.max_steps, args.lr,
                args.batch, args.grad_accum, args.rank, args.alpha, args.dropout, args.max_length, seed, data_seed,
                args.workers, args.checkpointing, PowerFeed(args.power_dir, args.power_node),
                args.cpu or args.torch_loss_head, args.group_by_length, _env_of(args), not args.cpu)


def _env_of(args: argparse.Namespace) -> dict[str, str | None]:
    tunableop = str(Path(args.tunableop).expanduser().resolve()) if args.tunableop else None
    return run_env(args.aotriton_experimental, args.blas_backend, tunableop)


@recorded(VERB)
@stop.guarded
def main(argv: Sequence[str] | None = None, callbacks: Sequence[Any] = ()) -> dict[str, Any]:
    """LoRA SFT of the student on the training rows of an approved export (see the module docstring). callbacks go
    to train() unchanged."""
    args = _parser().parse_args(argv)
    run_id = args.run_id or _run_id()
    mode = "cpu" if args.cpu else "pod"
    refs: dict[str, Any] = {"run_id": run_id, "mode": mode, "node": CPU_NODE if args.cpu else LEDGER_NODE}
    if not args.cpu:   # the pinned environment goes in before torch is imported, so the HIP runtime reads it
        apply_env(_env_of(args))
        try:
            import torch
        except ImportError:
            return _answer("refused", refs, error="torch is not installed, so this is not the trainer pod")
        why = pod_gate(torch, args.gfx, _rocminfo)
        if why is not None:
            return _answer("refused", refs, error=why)
    answer: dict[str, Any]
    try:
        with contextlib.redirect_stdout(sys.stderr):   # the one JSON line is the only thing on stdout
            plan = _prepare(args, run_id, refs, CPU_NODE if args.cpu else POD_NODE)
            trained = train(plan, callbacks)
        refs.update(trained.refs)
        answer = _answer("trained", refs, trained.metrics)
    except TrainError as err:
        answer = _answer(err.outcome, refs, error=str(err), problems=err.problems)
    except Exception as exc:  # a crash inside the training stack: keep the run's refs in its row
        traceback.print_exc(file=sys.stderr)
        answer = _answer("failed", refs, error=f"{type(exc).__name__}: {exc}"[:400])
    finally:
        if not args.cpu:   # past the node check, on the trainer pod: the copy goes however the run ended
            deleted, how = delete_data(args.dataset, args.rows)
            refs["data_deleted"] = "yes" if deleted else f"no: {how}"
        else:
            refs["data_deleted"] = "no: a --cpu rehearsal keeps its data"
    answer["refs"] = _refs(refs)
    if not args.cpu and refs["data_deleted"] != "yes" and answer.get("ok"):
        answer.update(ok=False, outcome="data-left", error=f"the dataset copy was not deleted: {refs['data_deleted']}")
    out = refs.get("out")
    if out and Path(out).is_dir():
        (Path(out) / "run.json").write_text(json.dumps(answer, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return answer


def _start() -> int:
    """Install the pinned stack when needed (then start again), run, print the one JSON line."""
    argv = sys.argv[1:]
    wants_help = any(arg in ("-h", "--help") for arg in argv)
    if not wants_help and "--no-install" not in argv and not os.environ.get(RESTARTED_ENV):
        if pins_to_install(read_pins()):
            done = install()
            if not done.get("ok"):
                print(json.dumps(done))
                return 1
            return subprocess.run([sys.executable, str(Path(__file__).resolve()), *argv],
                                   env={**os.environ, RESTARTED_ENV: "1"}, check=False).returncode
    answer = main(argv)
    print(json.dumps(answer))
    return 0 if answer.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(_start())
