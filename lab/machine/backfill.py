"""Backfill: put what the lab already measured and set up into the ledger, so
the ledger holds the past before anything compares against it.

    python -m lab ledger backfill [--metrics FILE] [--dry-run]

backfill is a sub-verb of the ledger verb: lab/machine/ledger_cli.py hands the words after
"ledger backfill" to main() here.

Two kinds of row are written, both with verb "backfill", outcome "ok" (the
outcome of the write, not of the thing recorded), actor "backfill" and no
tier. Every row carries a backfill field: {key, kind, origin} plus a topic
for facts.

  metric rows  one per row of the metrics file (lab/refine/metrics.jsonl
               unless --metrics names another). A row keeps the file row's
               measurement exactly: ts, metrics {metric: value}, units
               {metric: unit}, node, conditions, source and low_confidence.
               ts stays the time the number was observed or recorded, never
               the time of the backfill, so the ledger's newest-by-ts rule
               (ledger.latest_metric) still finds the current number, and
               latest_metric answers what metrics.latest answers for every
               row the file holds. task is None: a measurement is not about
               one task.
  fact rows    one per entry of FACTS: what was built and started when the two
               R9700 pods were set up (the build fixes, the checkpoints in
               use, the pod sessions). task, refs, node, note (the fact in
               plain sentences), conditions (its details) and source (the
               records it comes from) are set by the fact; metrics is {} so
               a fact never passes for a measurement. A fact's ts is the time
               of the earliest record that states it: a commit's author time,
               or when a pull request was opened or merged.

Idempotent. Each candidate row has a key: "metric:<16 hex of the sha256 of
the file row as normalized JSON>", with "#2", "#3" ... after it for a second
and later identical row, or "fact:<fact key>". A run reads the keys already
in the ledger (rows with verb "backfill") and appends only the rows whose key
is missing, so a second run adds nothing, a run that was cut short finishes
the job, and rows added to the metrics file later are picked up. The key
follows the content, not the position, so reordering the file imports nothing
twice. The ledger is append-only: a metrics row that changed in the file, or a
fact whose key was reused for other text, would not replace the row already
written; use a new fact key for a corrected fact.

Every row is validated before anything is written, so a corrupt metrics file
or a bad fact refuses the whole run. Runs take backfill.lock in the state dir,
so two runs at once cannot both decide a row is missing. --dry-run reports
what would be added, takes no lock and writes nothing.

The answer is one JSON object: ok, outcome (backfilled, up_to_date, dry_run,
refused or locked), dry_run, ledger, metrics_file, rows {metrics, facts,
total}, present (already in the ledger), to_add, added, first_seq and
last_seq (of the rows this run appended, else null). A refused run answers
ok false with error, ledger and metrics_file and writes nothing. A locked run
(another backfill, or the ledger itself, stayed locked) answers ok false with
error and the rows it did append; running again finishes the job."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lab.machine import ledger, metrics
from lab.machine.lock import FileLock, LockTimeout

__all__ = ["VERB", "ACTOR", "LOCK_FILE", "LOCK_TIMEOUT", "FACT_TASK", "FACT_REFS", "Fact", "FACTS", "Candidate",
           "BackfillError", "fact_row", "metric_row", "metric_candidates", "fact_candidates", "candidates",
           "present_keys", "run", "main"]

VERB = "backfill"
ACTOR = "backfill"
OUTCOME = "ok"
LOCK_FILE = "backfill.lock"
LOCK_TIMEOUT = 30.0          # seconds a run waits for another backfill run before giving up
REPO_ROOT = Path(__file__).resolve().parents[2]

FACT_TASK = "R2"
FACT_REFS = {"pr": 37, "commit": "bb351e6"}
FACT_ORIGIN = "R2 task page and PR #37"

Row = dict[str, Any]


class BackfillError(ValueError):
    """The run cannot start: the metrics file is missing or something it names is not usable."""


@dataclass(frozen=True)
class Fact:
    """One thing that was true of the lab setup, as a ledger row waiting to be written."""

    key: str                       # stable identity inside FACTS; never reused for other text
    topic: str                     # build, build_fix, checkpoint, serving or pod_session
    ts: str                        # the time of the earliest record that states it (UTC, ending in Z)
    note: str                      # the fact in plain sentences
    source: str                    # the records it comes from
    node: str | None = None        # pod-a or pod-b when the fact is about one pod
    conditions: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    """A row backfill would write and the key that says whether it already has."""

    key: str
    kind: str                      # "metric" or "fact"
    row: Row


_MODEL = "RedHatAI/Qwen3.8-27B-INT4"
_VLLM = "vllm@b1388b1"
_TORCH = "2.9.1+rocm7.13.0"
_DIFF = "lab/vllm-gfx1201.diff"
_SRC_A = "PR #37"

FACTS: tuple[Fact, ...] = (
    Fact(
        key="build.vllm_from_source", topic="build", node="pod-a", ts="2026-10-10T10:22:13Z",
        note="vLLM was built from source on pod A at commit b1388b1 (April 2026 main) for gfx1201 on ROCm 7.13.0, "
             f"against the pod's own torch {_TORCH}, with the patch set {_DIFF} applied. The build step takes about "
             "1 h.",
        conditions={"engine": _VLLM, "rocm": "7.13.0", "torch": _TORCH, "gpu_target": "gfx1201", "patch": _DIFF,
                    "build_time": "about 1 h"},
        source=f"{_SRC_A} body ('builds vLLM from source on the pod'); lab/README.md (Teacher, pod A); "
               "lab/pod_a_teacher.sh (VLLM_COMMIT, build step); R2 task page progress line",
    ),
    Fact(
        key="build.fix.attention_num_stages", topic="build_fix", node="pod-a", ts="2026-10-10T10:22:13Z",
        note="The Triton unified attention kernel now launches with num_stages=1. RDNA4 (gfx1201) has 64 KB of LDS and "
             "the default pipelining needs 66560 bytes, which makes Triton raise OutOfResources; num_stages=1 removes "
             "the double-buffering. Both launches in unified_attention() carry it.",
        conditions={"file": "vllm/v1/attention/ops/triton_unified_attention.py", "num_stages": 1, "lds_kb": 64,
                    "needed_bytes": 66560, "error_without": "triton OutOfResources"},
        source=f"{_SRC_A}, {_DIFF} (triton_unified_attention.py hunks); lab/README.md (Teacher, pod A)",
    ),
    Fact(
        key="build.fix.rocm_detection_via_torch", topic="build_fix", node="pod-a", ts="2026-10-10T10:22:13Z",
        note="ROCm platform detection in vllm/platforms/__init__.py now falls back to torch.version.hip together with "
             "torch.cuda.is_available() when amdsmi fails. The patch comment names the amdsmi status "
             "AMDSMI_STATUS_DRIVER_NOT_LOADED, a GPU exposed through /dev/dxg instead of /dev/kfd, and vLLM issue "
             "#40081.",
        conditions={"file": "vllm/platforms/__init__.py", "fallback": "torch.version.hip and torch.cuda.is_available()",
                    "amdsmi_status": "AMDSMI_STATUS_DRIVER_NOT_LOADED", "upstream_issue": "vLLM #40081"},
        source=f"{_SRC_A}, {_DIFF} (vllm/platforms/__init__.py hunk); lab/README.md ('two platform-detection fixes')",
    ),
    Fact(
        key="build.fix.gcn_arch_log_level", topic="build_fix", node="pod-a", ts="2026-10-10T10:22:13Z",
        note="In vllm/platforms/rocm.py the amdsmi failure path of _get_gcn_arch now logs at debug level instead of "
             "calling logger.warning_once. warning_once imports vllm.distributed, which imports current_platform and "
             "re-enters platform resolution while _GCN_ARCH is still being computed at module import (a circular "
             "import).",
        conditions={"file": "vllm/platforms/rocm.py", "function": "_get_gcn_arch", "log_call": "logger.debug"},
        source=f"{_SRC_A}, {_DIFF} (vllm/platforms/rocm.py hunk); lab/README.md ('two platform-detection fixes')",
    ),
    Fact(
        key="build.fix.torch_pins_removed", topic="build_fix", node="pod-a", ts="2026-10-10T10:22:13Z",
        note="The torch pins are removed (torch==2.10.0 in pyproject.toml, requirements/build.txt and "
             "requirements/rocm-build.txt; torchvision, torchaudio and the PyTorch rocm7.1 extra index in "
             "requirements/rocm-build.txt; uv's no-build-isolation-package setting for torch) so the build uses the "
             f"pod's own torch {_TORCH}. The build step also runs use_existing_torch.py.",
        conditions={"pinned_torch": "2.10.0", "pod_torch": _TORCH,
                    "files": ["pyproject.toml", "requirements/build.txt", "requirements/rocm-build.txt"],
                    "script": "use_existing_torch.py"},
        source=f"{_SRC_A}, {_DIFF} (pyproject.toml and requirements hunks); lab/README.md; "
               "lab/pod_a_teacher.sh (build step)",
    ),
    Fact(
        key="build.fix.gptq_half_atomicadd", topic="build_fix", node="pod-a", ts="2026-10-10T10:22:13Z",
        note="csrc/quantization/gptq/compat.cuh no longer defines atomicAdd for half and half2 when USE_ROCM is set, "
             "because ROCm 7.13 ships atomicAdd(__half*, __half) and atomicAdd(__half2*, __half2) itself and a second "
             "definition is ambiguous.",
        conditions={"file": "csrc/quantization/gptq/compat.cuh", "rocm": "7.13", "guard": "!defined(USE_ROCM)",
                    "types": ["half", "half2"]},
        source=f"{_SRC_A}, {_DIFF} (compat.cuh hunks); lab/README.md (GPTQ compat.cuh fallbacks)",
    ),
    Fact(
        key="build.fix.pip_user_toolchain", topic="build_fix", node="pod-a", ts="2026-10-10T10:22:13Z",
        note="There is no root on the pod, so cmake, ninja and pkgconf come from pip install --user (binaries land in "
             "~/.local/bin, and pkg-config is linked to pkgconf when the image has none).",
        conditions={"root": False, "installed_with": "pip install --user", "tools": ["cmake", "ninja", "pkgconf"]},
        source=f"{_SRC_A}, lab/README.md (build gotchas); lab/pod_a_teacher.sh (deps step)",
    ),
    Fact(
        key="build.fix.libdrm_stub", topic="build_fix", node="pod-a", ts="2026-10-10T10:22:13Z",
        note="The image has no libdrm, but torch's LoadHIP.cmake (through rocm_smi-config.cmake) requires a libdrm "
             "pkg-config entry, so the build script writes a stub libdrm.pc and an empty libdrm.so under ~/.local.",
        conditions={"missing": "libdrm", "required_by": "torch LoadHIP.cmake", "stub": ["libdrm.pc", "libdrm.so"],
                    "stub_dir": "~/.local"},
        source=f"{_SRC_A}, lab/README.md (build gotchas); lab/pod_a_teacher.sh (deps step)",
    ),
    Fact(
        key="build.fix.scm_pretend_version", topic="build_fix", node="pod-a", ts="2026-10-10T10:22:13Z",
        note="The vLLM build sets SETUPTOOLS_SCM_PRETEND_VERSION (0.19.2+gfx1201) because git describe on the NFS home "
             "directory times out (40 s) and would otherwise kill the build.",
        conditions={"variable": "SETUPTOOLS_SCM_PRETEND_VERSION", "value": "0.19.2+gfx1201",
                    "git_describe_timeout_s": 40},
        source=f"{_SRC_A}, lab/README.md (build gotchas); lab/pod_a_teacher.sh (build step)",
    ),
    Fact(
        key="build.llama_cpp_gfx1201", topic="build", node="pod-b", ts="2026-10-10T09:48:20Z",
        note="llama.cpp was built on pod B for gfx1201 (GGML_HIP=ON, GPU_TARGETS=gfx1201, Release, Ninja) with CMake's "
             "HIP compiler set to the ROCm clang++ itself, because CMake's HIP language wants clang rather than the "
             "hipcc wrapper. llama-quantize, llama-cli and llama-server were built.",
        conditions={"gpu_target": "gfx1201", "hip_compiler": "/opt/rocm/llvm/bin/clang++",
                    "targets": ["llama-quantize", "llama-cli", "llama-server"]},
        source=f"{_SRC_A} body ('llama-quantize, llama-cli, llama-server built'); lab/pod_b_trainer.sh (llama step); "
               "R2 task page progress line",
    ),
    Fact(
        key="build.trainer_stack", topic="build", node="pod-b", ts="2026-10-10T09:48:20Z",
        note=f"Pod B trains on the pod image's own torch {_TORCH}; torch, torchvision and triton are never "
             "reinstalled. peft, trl, transformers, accelerate, datasets and gguf are installed with pip install "
             "--user.",
        conditions={"torch": _TORCH, "installed_with": "pip install --user",
                    "packages": ["peft", "trl", "transformers", "accelerate", "datasets", "gguf"]},
        source=f"{_SRC_A} body ('PEFT/TRL on the pod's torch'); lab/README.md (Trainer, pod B); "
               "lab/pod_b_trainer.sh (deps step)",
    ),
    Fact(
        key="checkpoint.teacher_in_use", topic="checkpoint", node="pod-a", ts="2026-10-10T10:22:13Z",
        note=f"Pod A serves {_MODEL} (compressed-tensors W4A16, 18 GB) as model name teacher on port 8000, with bf16 "
             "activations, a 16384-token maximum length, gpu-memory-utilization 0.90 and one image per prompt. It is "
             "the checkpoint that serves on this vLLM commit.",
        conditions={"model": _MODEL, "engine": _VLLM, "served_model_name": "teacher", "port": 8000,
                    "format": "compressed-tensors W4A16", "size_gb": 18, "dtype": "bfloat16", "max_model_len": 16384,
                    "gpu_memory_utilization": 0.9, "images_per_prompt": 1},
        source=f"{_SRC_A} body ('vllm serve is up on {_MODEL}'); lab/pod_a_teacher.sh (TEACHER, serve step); "
               "lab/README.md (Teacher, pod A); R2 task page progress line",
    ),
    Fact(
        key="serving.w4a16_kernel", topic="serving", node="pod-a", ts="2026-10-10T10:22:13Z",
        note="The W4A16 layers run on ConchLinearKernel, the Triton fallback, because Marlin and Machete do not exist "
             "on ROCm. That is why decode is far below the 20 to 30 tok/s the design doc estimated, and it is the "
             "first target for the gfx1201 kernel work.",
        conditions={"model": _MODEL, "engine": _VLLM, "kernel": "ConchLinearKernel", "kind": "Triton fallback",
                    "design_doc_estimate_tok_s": "20 to 30"},
        source=f"{_SRC_A}, lab/README.md (Teacher, pod A); R2 task page progress line",
    ),
    Fact(
        key="checkpoint.quark_not_loaded", topic="checkpoint", node="pod-a", ts="2026-10-10T10:22:13Z",
        note="AMD's Quark exports amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16 and its -MXFP4 sibling (Aug 2026) do not load "
             "on this vLLM commit: the Quark weight mapper (quark.py apply_vllm_mapper) rejects their config. Two "
             "routes would make them load: a vLLM commit from late August 2026 or later (it pins torch 2.10+, so "
             "torch would come from repo.radeon.com/rocm/manylinux), or backporting that one Python file's mapper, "
             "which needs no C++ rebuild. Qwen/Qwen3.8-27B-FP8 (28.8 GB) is the last resort and leaves little KV "
             "room.",
        conditions={"engine": _VLLM, "rejected_by": "quark.py:apply_vllm_mapper",
                    "checkpoints": ["amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16", "-MXFP4"],
                    "routes": ["vLLM commit from late August 2026 or later", "backport the mapper"],
                    "last_resort": "Qwen/Qwen3.8-27B-FP8"},
        source=f"{_SRC_A} body ('The AMD Quark checkpoints do not load on this vLLM commit'); lab/README.md "
               "(Teacher, pod A); R2 task page progress line",
    ),
    Fact(
        key="checkpoint.student", topic="checkpoint", node="pod-b", ts="2026-10-10T09:48:20Z",
        note="Pod B holds Qwen/Qwen3.5-4B, the student the Jetson runs, in ~/models; the smoke LoRA trains on it.",
        conditions={"model": "Qwen/Qwen3.5-4B", "dir": "~/models", "role": "student"},
        source=f"{_SRC_A}, lab/README.md (Trainer, pod B); lab/pod_b_trainer.sh (STUDENT, download step); "
               "R2 task page progress line",
    ),
    Fact(
        key="session.pod_a_running", topic="pod_session", node="pod-a", ts="2026-10-10T10:23:23Z",
        note=f"Pod A's session (account A, the teacher) was up: vllm serve was running {_MODEL} and answered text and "
             "720p image prompts called from outside through the hub proxy "
             "(https://dub.aupcloud.io/user/<account>/proxy/8000/v1/chat/completions, with the account's hub token).",
        conditions={"role": "teacher", "account": "A", "port": 8000, "reached_through": "hub proxy",
                    "prompts_answered": ["text", "720p image"]},
        source=f"{_SRC_A} body (How it was verified, Pod A); R2 task page progress line ('both pods set up and "
               "running')",
    ),
    Fact(
        key="session.pod_b_running", topic="pod_session", node="pod-b", ts="2026-10-10T10:23:23Z",
        note="Pod B's session (account B, the trainer) was up: llama-quantize, llama-cli and llama-server were built "
             "there and lab/smoke_lora.py --steps 20 finished on one R9700 (loss 2.23 to 0.16, 0.51 samples/s, 8.7 GB "
             "peak; the numbers are also in the trainer.* metric rows).",
        conditions={"role": "trainer", "account": "B", "smoke_steps": 20},
        source=f"{_SRC_A} body (How it was verified, Pod B); R2 task page progress line ('both pods set up and "
               "running')",
    ),
    Fact(
        key="session.both_left_running", topic="pod_session", ts="2026-10-10T10:41:19Z",
        note="Both pod sessions were left running when the lab setup merged, on 600-minute sessions.",
        conditions={"left_running": True, "runtime_minutes": 600},
        source="R2 task page Done line ('Both pods left running'); Build Log R2 entry ('left running on 600-minute "
               "sessions'); merge time of PR #37",
    ),
    Fact(
        key="session.idle_culler", topic="pod_session", ts="2026-10-10T10:22:13Z",
        note="A 60-minute idle culler stops a pod server that receives no requests, so the model server has to be kept "
             "busy or pinged. The server enforces no runtime cap of its own, and stopping a pod is immediate.",
        conditions={"idle_culler_minutes": 60, "runtime_cap": "none enforced by the server", "stop": "immediate"},
        source=f"{_SRC_A} body ('the 60-minute idle culler stops a pod that receives no requests'); lab/README.md "
               "(What the pods are)",
    ),
    Fact(
        key="session.tailscale_blocked", topic="pod_session", ts="2026-10-10T10:22:13Z",
        note="Tailscale is blocked on the pods (every tailscale.com host is connection-reset), so the box reaches a "
             "pod only through the hub proxy. GitHub, Hugging Face and PyPI are reachable from the pods.",
        conditions={"blocked": "tailscale.com", "reachable": ["GitHub", "Hugging Face", "PyPI"],
                    "box_route": "hub proxy"},
        source=f"{_SRC_A}, lab/README.md (What the pods are); R2 task page Done-when text",
    ),
    Fact(
        key="session.launch_without_repo_clone", topic="pod_session", ts="2026-10-10T10:22:13Z",
        note="The private team repo cannot be cloned when a pod launches (there is no GitHub app on the hub), so lab/ "
             "is uploaded after the pod starts, by JupyterLab upload or PUT /user/<account>/api/contents/<path>.",
        conditions={"clone_at_launch": False, "reason": "no GitHub app on the hub",
                    "upload": ["JupyterLab upload", "PUT /user/<account>/api/contents/<path>"]},
        source=f"{_SRC_A}, lab/README.md (Launching a pod)",
    ),
)


# --- candidate rows -----------------------------------------------------------------------

def fact_row(fact: Fact) -> Row:
    """The ledger row for one fact."""
    row: Row = {
        "ts": fact.ts, "actor": ACTOR, "tier": None, "task": FACT_TASK, "verb": VERB, "outcome": OUTCOME,
        "metrics": {}, "refs": dict(FACT_REFS),
    }
    if fact.node is not None:
        row["node"] = fact.node
    row.update({
        "conditions": copy.deepcopy(dict(fact.conditions)), "low_confidence": False, "source": fact.source,
        "note": fact.note,
        "backfill": {"key": f"fact:{fact.key}", "kind": "fact", "topic": fact.topic, "origin": FACT_ORIGIN},
    })
    return row


def metric_row(entry: Mapping[str, Any], key: str, origin: str) -> Row:
    """The ledger row for one row of the metrics file."""
    name = entry["metric"]
    return {
        "ts": entry["ts"], "actor": ACTOR, "tier": None, "task": None, "verb": VERB, "outcome": OUTCOME,
        "metrics": {name: entry["value"]}, "refs": {},
        "node": entry["node"], "units": {name: entry["unit"]},
        "conditions": copy.deepcopy(entry["conditions"]), "low_confidence": entry["low_confidence"],
        "source": entry["source"],
        "backfill": {"key": key, "kind": "metric", "origin": origin},
    }


def _origin(path: Path) -> str:
    """The metrics file as the rows name it: relative to the repo when it lies inside, else as given."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def metric_candidates(metrics_path: str | os.PathLike[str] | None = None) -> list[Candidate]:
    """One candidate per row of the metrics file, in file order. BackfillError when the file does not exist;
    metrics.MetricsFormatError when a line breaks the metrics schema; ledger.RowError when a row is one the ledger
    would refuse."""
    path = Path(metrics.DEFAULT_PATH if metrics_path is None else metrics_path)
    if not path.is_file():
        raise BackfillError(f"metrics file {path} does not exist")
    origin = _origin(path)
    seen: dict[str, int] = {}
    found: list[Candidate] = []
    for entry in metrics.load(path):
        text = json.dumps(entry, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        digest = hashlib.sha256(text.encode("ascii")).hexdigest()[:16]
        seen[digest] = seen.get(digest, 0) + 1
        key = f"metric:{digest}" if seen[digest] == 1 else f"metric:{digest}#{seen[digest]}"
        row = metric_row(entry, key, origin)
        ledger.validate(row)
        found.append(Candidate(key, "metric", row))
    return found


def fact_candidates() -> list[Candidate]:
    """One candidate per entry of FACTS, in table order. BackfillError for a repeated fact key; ledger.RowError for a
    fact the ledger would refuse."""
    found: list[Candidate] = []
    keys: set[str] = set()
    for fact in FACTS:
        if fact.key in keys:
            raise BackfillError(f"fact key {fact.key!r} appears twice in FACTS")
        keys.add(fact.key)
        row = fact_row(fact)
        ledger.validate(row)
        found.append(Candidate(row["backfill"]["key"], "fact", row))
    return found


def candidates(metrics_path: str | os.PathLike[str] | None = None) -> list[Candidate]:
    """Every row a full backfill writes: the metrics file's rows, then the facts."""
    return metric_candidates(metrics_path) + fact_candidates()


def present_keys(state_dir: str | os.PathLike[str] | None = None) -> set[str]:
    """The backfill keys already in the ledger."""
    keys: set[str] = set()
    for row in ledger.read(state_dir=state_dir, verb=VERB):
        info = row.get("backfill")
        if isinstance(info, dict) and isinstance(info.get("key"), str):
            keys.add(info["key"])
    return keys


# --- the run ------------------------------------------------------------------------------

def _answer(ok: bool, outcome: str, **more: Any) -> dict[str, Any]:
    return {"ok": ok, "outcome": outcome, **more}


def run(*, metrics_path: str | os.PathLike[str] | None = None, state_dir: str | os.PathLike[str] | None = None,
        dry_run: bool = False, lock_timeout: float = LOCK_TIMEOUT) -> dict[str, Any]:
    """Append the backfill rows the ledger does not hold yet and answer with what was done. Nothing is written
    when any row is refused (outcome refused). dry_run only counts. See the module docstring for the answer."""
    path = Path(metrics.DEFAULT_PATH if metrics_path is None else metrics_path)
    state = ledger.resolve_state_dir(state_dir)
    where = {"ledger": str(ledger.ledger_path(state)), "metrics_file": str(path)}
    try:
        wanted = candidates(path)
    except (BackfillError, metrics.MetricsFormatError, metrics.MetricsLocked, ledger.RowError) as exc:
        return _answer(False, "refused", error=str(exc), **where)
    rows = {"metrics": sum(c.kind == "metric" for c in wanted), "facts": sum(c.kind == "fact" for c in wanted),
            "total": len(wanted)}

    def report(outcome: str, present: int, to_add: int, seqs: list[int], **more: Any) -> dict[str, Any]:
        return _answer(outcome in ("backfilled", "up_to_date", "dry_run"), outcome, dry_run=dry_run, **where,
                       rows=rows, present=present, to_add=to_add, added=len(seqs),
                       first_seq=seqs[0] if seqs else None, last_seq=seqs[-1] if seqs else None, **more)

    if dry_run:
        have = present_keys(state)
        missing = [c for c in wanted if c.key not in have]
        return report("dry_run", len(wanted) - len(missing), len(missing), [])

    seqs: list[int] = []
    present = to_add = 0
    try:
        with FileLock(state / LOCK_FILE, timeout=lock_timeout) as lock:
            lock.write_note(f"pid {os.getpid()} backfilling since {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
            have = present_keys(state)
            missing = [c for c in wanted if c.key not in have]
            present, to_add = len(wanted) - len(missing), len(missing)
            for candidate in missing:
                seqs.append(ledger.append(candidate.row, state_dir=state))
    except LockTimeout as exc:
        return report("locked", present, to_add, seqs, error=str(exc))
    return report("backfilled" if seqs else "up_to_date", present, to_add, seqs)


def main(argv: Sequence[str]) -> dict[str, Any]:
    """Backfill the ledger with the lab's past: the metrics file's rows and the lab setup facts (safe to repeat)."""
    parser = argparse.ArgumentParser(prog="python -m lab ledger backfill", description=main.__doc__)
    parser.add_argument("--metrics", default=None, metavar="FILE",
                        help="metrics file to import (default: lab/refine/metrics.jsonl)")
    parser.add_argument("--dry-run", action="store_true", help="report what would be added and write nothing")
    args = parser.parse_args(argv)
    return run(metrics_path=args.metrics, dry_run=args.dry_run)
