"""quark-compare (R23): AMD's Quark INT4 checkpoint against the RedHat INT4 teacher on pod B, with a verdict.

    python -m lab quark-compare preflight [--checkpoint DIR] [plan options]      checkpoint, config keys, time plan
    python -m lab quark-compare slot [--row ID] [--reply TEXT]                     take the 60-minute pod B slot
    python -m lab quark-compare run [--checkpoint DIR] [--vllm-dir DIR] [plan options]   on pod B, inside the slot
    python -m lab quark-compare record --result FILE [--result FILE ...] --control-rows FILE [--reference FILE]
                                       [--row ID]                                  verdict, ledger, T0 text, release
    python -m lab quark-compare all [every option above]                           the four in one process

What it compares. Arm A is RedHatAI/Qwen3.8-27B-INT4, the teacher (compressed-tensors W4A16); arm B is
amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16 (Quark AWQ, int4 weight-only, group 128, symmetric). Both are served by the
same build (vLLM b1388b1 plus lab/vllm-gfx1201.diff) with the same settings sheet: the serve line of
lab/pod_a_teacher.sh as the pinned baseline of lab/refine/research/kernels/data/searchspace-inference-gfx1151.json
(R35's settings_inference_gfx1151.md) gives it, every fixed argument, every pinned variable set and every unpinned one
unset. The metrics are decode tok/s and joules per token at each batch size of the baseline matrix (lab/bench.py,
one cold spawn per repeat) and the exact-match rate of B's greedy outputs against A's on the bench prompt set.

The steps.

preflight   reads <checkpoint>/config.json and runs lab/quark_keys.audit on its quantization_config (every key
            known, none dropped that holds a setting, the W4A16 scheme found, the zero-point layout named), checks
            every file of the pinned revision is there at its size (the checkpoint is copied from pod A's home
            volume, never downloaded here), and prints the time plan: a rough estimate from the README's measured
            numbers (cold start 523 s, warm 297 s, 2.88 tok/s at batch 1) against the slot's 60 minutes and its one
            15-minute extension. A plan over 75 minutes is refused unless --over-budget.
slot        asks the GPU Queue for a 60-minute pod B row (lab/machine/slots.py, with the justification a long row
            needs) or starts the row given; a long row starts only once the referee has critiqued it and --reply
            answered. Anything short of "started" is outcome waiting, with the row and the next command.
run         on pod B: refuses unless the checkpoint passes preflight and the loader hunks of the diff either apply
            or are already in ~/vllm; when no reference outputs exist yet, serves arm A once at batch 1 (eager) for
            them; puts the loader hunks on (only when it had to), runs the bench matrix on arm B and its correctness
            pass (eager, batch 1, greedy, the same prompts), takes the hunks off again (only those it put on), and
            checks the pod is as found: the same `git status` and `git diff` in ~/vllm, nothing listening on the
            bench port. Writes <out>/<session>/result.json. It needs no Notion and no ledger beyond bench's own rows.
record      on the box: reads one or more result.json (the matrix may take two slots), A's bench rows from R20
            (--control-rows, a bench rows.jsonl) and the reference outputs (--reference, else the ones run made);
            reads each batch size's decode tok/s and joules per token B against A with lab/machine/verdict.py,
            the exact-match rate as counts beside A's self-match floor when the reference holds two spawns; appends
            one `quark-compare` ledger row (task R23, node pod-b), records T0's E5 result with arms A and B (the
            single-stream joules per token, E5's metric; E5 stays running while its arm C, the FP8 checkpoint, has
            no reading), writes R23_T0_result.txt (the E5 checklist line and the R23 paragraph, for the dispatcher
            to post on T0) and, with --row, releases the slot with a one-line result.
all         slot, run and record in one process; it releases the slot in every case once it started it.

Reference outputs (--reference, default fixtures/kernels/baseline_outputs.jsonl when R21 has written it): JSON lines
{"prompt_id", "text", "spawn" (optional, 1 or 2), "checkpoint", "eager", "max_tokens"}. Exact match is the whole
text equal; a mismatch records the first character where the two differ. Two checkpoints quantized two ways are not
expected to agree token for token, so a low rate against A is a finding about the quantization, not a kernel
defect; the floor (A against itself across two spawns) says what agreement the server itself allows.

Every call prints one JSON line. Exit 0 when it worked, 1 when it was refused or failed, 2 for bad arguments.
Tested against lab/machine/slots_fake.py (the queue), lab/bench_fake.py (the server and the pod) and a local git
repository standing in for ~/vllm. What only pod B can show is in lab/R23_REAL_RUN.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shlex
import subprocess
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lab import bench, quark_keys
from lab.machine import experiments, ledger, record, slots
from lab.machine import verdict as vmath

REPO = Path(__file__).resolve().parents[1]
TASK = "R23"
EXPERIMENT = "E5"
NODE = "pod-b"
VERB = "quark-compare"
RESULT_SCHEMA = "lab.quark_compare.result/1"

CHECKPOINT = quark_keys.CHECKPOINT_REPO
CONTROL = "RedHatAI/Qwen3.8-27B-INT4"
DEFAULT_CHECKPOINT = "~/models/Qwen3.8-27B-Quark-AWQ-INT4-W4A16"
DEFAULT_CONTROL_CHECKPOINT = "~/models/Qwen3.8-27B-INT4"
DEFAULT_VLLM_DIR = "~/vllm"
PATCH = REPO / "lab" / "vllm-gfx1201.diff"
SETTINGS = REPO / "lab" / "refine" / "research" / "kernels" / "data" / "searchspace-inference-gfx1151.json"
DEFAULT_REFERENCE = REPO / "fixtures" / "kernels" / "baseline_outputs.jsonl"
DEFAULT_OUT = REPO / "lab" / "state" / "quark_compare"
T0_FILE = "R23_T0_result.txt"
QUANTIZATION = "quark-awq-int4-w4a16-g128"

#: Every file of the checkpoint at quark_keys.CHECKPOINT_REVISION, with its size (the hub's tree listing).
CHECKPOINT_FILES: dict[str, int] = {
    ".gitattributes": 1570, "LICENSE": 11544, "README.md": 5776, "chat_template.jinja": 8952, "config.json": 10840,
    "crc32.txt": 238, "generation_config.json": 213, "merges.txt": 3353259, "model.safetensors": 19512909752,
    "preprocessor_config.json": 482, "processor_config.json": 1300, "tokenizer.json": 12809320,
    "tokenizer_config.json": 17928, "video_preprocessor_config.json": 385, "vocab.json": 6722759,
}
#: The weights' sha256 (the hub's LFS id), for the copy from pod A: lab/transfer.py pull-file --sha256.
MODEL_SHA256 = "f1699476b8c79a9a3121469b8bdfd84c0ee55417d2af02260b0d8982257c6346"

SLOT_MINUTES = slots.LONG_MINUTES
EXTENSION_MINUTES = slots.EXTENSION_MINUTES
SLOT_WHAT = "R23 bench: serve amd Quark AWQ INT4 checkpoint, baseline matrix and correctness pass against RedHat INT4"
SLOT_WHY = "5 cold spawns of a 27B server plus an eager correctness spawn; each spawn takes 5 to 9 minutes to load"
SLOT_MEASURED = ("pod D cold start 523 s, pod C warm start 297 s, 2.88 tok/s at batch 1 (lab/README.md); the "
                 "loader hunks pass git apply --check on b1388b1")

#: The timing model of the plan, from lab/README.md (pods C and D, 2026-10-10). Rough on purpose: it decides
#: whether a plan fits the slot, not what anything will measure.
READY_COLD_S = 523.0
READY_WARM_S = 297.0
DECODE_TOK_S = 2.88
BATCH_SLOWDOWN = 1.5          # per-request time per token at batch 4 to 16, the upper end of the research note
SETUP_S = 120.0               # the patch on and off, the checks, the idle reads outside spawns

Runner = Callable[[Sequence[str], Path], subprocess.CompletedProcess]


class Refused(Exception):
    """A step that stops before it changes anything, with the outcome it answers."""

    def __init__(self, outcome: str, message: str, **more: Any) -> None:
        super().__init__(message)
        self.outcome = outcome
        self.more = more


# --- the checkpoint ---------------------------------------------------------------------------------------------

def check_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Is the copied checkpoint whole (every file of the pinned revision at its size) and does its config load with
    the patched loader (quark_keys.audit)? Nothing is downloaded and nothing is hashed here."""
    folder = Path(path).expanduser()
    missing, wrong = [], {}
    for name, size in CHECKPOINT_FILES.items():
        f = folder / name
        if not f.is_file():
            missing.append(name)
        elif f.stat().st_size != size:
            wrong[name] = {"size": f.stat().st_size, "expected": size}
    audit: dict[str, Any] | None = None
    problem = None
    try:
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        audit = quark_keys.audit(config["quantization_config"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        problem = f"cannot read the quantization config: {type(exc).__name__}: {exc}"
    ok = not missing and not wrong and audit is not None and audit["ok"]
    return {"ok": ok, "path": str(folder), "revision": quark_keys.CHECKPOINT_REVISION, "missing": missing,
            "wrong_size": wrong, "problem": problem,
            "audit": None if audit is None else {k: audit[k] for k in ("ok", "unknown", "blocking", "scheme",
                                                                       "zero_point")}}


# --- the settings sheet -----------------------------------------------------------------------------------------

def settings_sheet(path: str | os.PathLike[str] = SETTINGS, *, run_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """The pinned baseline of the inference search space as launch arguments, variables to set and to unset."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    fixed = data["fixed_launch"]
    args, env, unset = list(fixed["args"]), dict(fixed["env"]), list(fixed["unset_env"])
    for name, value in data["baseline"].items():
        options = data["mapping"].get(name)
        if options is None:
            continue
        choice = options[str(value).lower() if isinstance(value, bool) else str(value)]
        args += choice["args"]
        env.update({k: str(v).replace("{run_dir}", str(run_dir)) for k, v in choice["env"].items()})
    clash = sorted(set(env) & set(unset))
    if clash:
        raise ValueError(f"the settings sheet both sets and unsets {', '.join(clash)}")
    return {"source": str(Path(path).resolve().relative_to(REPO)) if Path(path).resolve().is_relative_to(REPO)
            else str(path), "args": args, "env": env, "unset": unset}


def serve_template(sheet: Mapping[str, Any]) -> list[str]:
    """The vLLM launch as lab/bench.py fills it ({checkpoint}, {model}, {port}): `env -u` for each unpinned variable,
    then pod_a_teacher.sh's server with the sheet's arguments, the served name left to the harness."""
    args = list(sheet["args"])
    if "--served-model-name" in args:
        i = args.index("--served-model-name")
        del args[i:i + 2]
    head = ["env"] + [x for name in sheet["unset"] for x in ("-u", name)]
    return head + ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", "{checkpoint}",
                   "--served-model-name", "{model}", "--port", "{port}", "--host", "127.0.0.1", *args]


def eager(template: Sequence[str]) -> tuple[list[str], bool]:
    """The template with CUDA graphs off for the correctness pass (kernels rubric K4 a): the sheet's
    --compilation-config pair becomes --enforce-eager. A template without one (a stand-in server) is kept."""
    argv = list(template)
    if "--compilation-config" not in argv:
        return argv, False
    i = argv.index("--compilation-config")
    return argv[:i] + ["--enforce-eager"] + argv[i + 2:], True


# --- the plan and its time --------------------------------------------------------------------------------------

@dataclass
class Plan:
    batches: list[int] = field(default_factory=lambda: [1, 4])
    with_64: bool = False
    repeats: int = vmath.MIN_RUNS_PER_ARM
    max_tokens: int = 64
    requests_per_stream: int = 1
    warmup_rounds: int = 1
    idle_s: float = bench.DEFAULT_IDLE_S
    correctness_max_tokens: int = 64
    control_reference: bool = False      # serve arm A once for reference outputs (no --reference file)


def estimate(plan: Plan, prompts: int) -> dict[str, Any]:
    """Seconds the plan needs on pod B by the timing model, and whether that fits the slot."""
    per_request = plan.max_tokens / DECODE_TOK_S
    spawn_s = 2 * plan.idle_s + per_request                        # idle reads, the probe
    for b in plan.batches + ([64] if plan.with_64 else []):
        slow = 1.0 if b == 1 else BATCH_SLOWDOWN
        spawn_s += prompts * per_request                           # the sequential warm-up, one per prompt
        spawn_s += (plan.warmup_rounds + plan.requests_per_stream) * per_request * slow
    bench_s = plan.repeats * (READY_WARM_S + spawn_s) + (READY_COLD_S - READY_WARM_S)
    correctness_s = READY_WARM_S + prompts * plan.correctness_max_tokens / DECODE_TOK_S
    total = SETUP_S + bench_s + correctness_s * (2 if plan.control_reference else 1)
    minutes = round(total / 60.0, 1)
    return {"seconds": round(total), "minutes": minutes, "spawns": plan.repeats + 1 + int(plan.control_reference),
            "slot_minutes": SLOT_MINUTES, "fits_slot": minutes <= SLOT_MINUTES,
            "needs_extension": SLOT_MINUTES < minutes <= SLOT_MINUTES + EXTENSION_MINUTES,
            "fits_with_extension": minutes <= SLOT_MINUTES + EXTENSION_MINUTES,
            "model": f"ready {READY_COLD_S:g} s cold and {READY_WARM_S:g} s warm, {DECODE_TOK_S} tok/s at batch 1, "
                     f"x{BATCH_SLOWDOWN} per request above it (lab/README.md)"}


# --- the pod: ~/vllm, the loader hunks, the state it is left in -------------------------------------------------

def run_cmd(argv: Sequence[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(list(argv), cwd=str(cwd), capture_output=True, text=True, timeout=120)


def snapshot(vllm_dir: Path, port: int, runner: Runner = run_cmd) -> dict[str, Any]:
    """What the run must leave as it found it: the vLLM tree's status and diff, and the bench port."""
    status = runner(["git", "status", "--porcelain"], vllm_dir)
    diff = runner(["git", "diff", "--binary"], vllm_dir)
    return {"git_status": status.stdout, "git_diff_sha256": hashlib.sha256(diff.stdout.encode()).hexdigest(),
            "port": port, "port_open": bool(port) and bench.port_open(port)}


def patch_state(vllm_dir: Path, loader_patch: Path, runner: Runner = run_cmd) -> str:
    """applied (the loader hunks are in the tree), absent (they apply cleanly) or conflict."""
    if runner(["git", "apply", "--check", "-R", str(loader_patch)], vllm_dir).returncode == 0:
        return "applied"
    if runner(["git", "apply", "--check", str(loader_patch)], vllm_dir).returncode == 0:
        return "absent"
    return "conflict"


def apply_patch(vllm_dir: Path, loader_patch: Path, *, reverse: bool = False, runner: Runner = run_cmd) -> None:
    argv = ["git", "apply", *(["-R"] if reverse else []), str(loader_patch)]
    done = runner(argv, vllm_dir)
    if done.returncode != 0:
        raise RuntimeError(f"{' '.join(argv[:3])} failed in {vllm_dir}: {done.stderr.strip()[:300]}")


# --- the correctness pass ---------------------------------------------------------------------------------------

def collect(endpoint: Any, body: Mapping[str, Any]) -> dict[str, Any]:
    """One streamed greedy completion: its text, token count and finish reason."""
    parts: list[str] = []
    usage: dict[str, Any] = {}
    finish = error = None
    try:
        for _, data in endpoint.events(body):
            if data.strip() == "[DONE]":
                break
            chunk = json.loads(data)
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                parts.append((choice.get("delta") or {}).get("content") or "")
                finish = choice.get("finish_reason") or finish
    except Exception as exc:  # a failed request is a mismatch with a reason, not a crash
        error = f"{type(exc).__name__}: {' '.join(str(exc).split())[:200]}"
    return {"text": "".join(parts), "tokens": usage.get("completion_tokens"), "finish_reason": finish, "error": error}


def correctness_pass(template: Sequence[str], checkpoint: str, *, out_dir: Path, tag: str, max_tokens: int,
                     prompts_path: Path, port: int, launch_env: Mapping[str, str],
                     ready_timeout_s: float = bench.DEFAULT_READY_TIMEOUT_S) -> dict[str, Any]:
    """One cold spawn, eager when the template allows it, every prompt once at batch 1, greedy, thinking off, no
    nonce (the same text a reference recorded), then the server stopped."""
    argv_t, is_eager = eager(template)
    prompts = bench.PromptSet.load(prompts_path)
    settings = bench.Settings(model="teacher", max_tokens=max_tokens, ignore_eos=False, prefix_caching=False)
    log_path = out_dir / f"correctness-{tag}.log"
    port_file = out_dir / f"correctness-{tag}.port"
    values = {"port": port, "port_file": port_file, "log": log_path, "batch": 1, "max_len": 16384,
              "model": "teacher", "checkpoint": checkpoint}
    argv = bench.fill_command(argv_t, values)
    own_log = any("{log}" in str(a) for a in argv_t)
    spawned = bench.spawn_server(argv, port=port, log_path=log_path, env=launch_env, cwd=out_dir, own_log=own_log)
    outputs: list[dict[str, Any]] = []
    endpoint = None
    try:
        bound = port or bench.wait_port(spawned, port_file, ready_timeout_s)
        endpoint = bench.DirectEndpoint(f"http://127.0.0.1:{bound}")
        bench.wait_ready(spawned, endpoint, ready_timeout_s)
        for i in range(len(prompts.prompts)):
            body = prompts.request_body(i, settings, None)
            outputs.append({"prompt_id": prompts.prompt_id(i), **collect(endpoint, body)})
    finally:
        if endpoint is not None:
            endpoint.close()
        stopped = spawned.stop()
    return {"checkpoint": checkpoint, "eager": is_eager, "max_tokens": max_tokens, "batch": 1,
            "prompt_set": prompts.describe(), "stopped": stopped, "outputs": outputs}


# --- run: on pod B ----------------------------------------------------------------------------------------------

@dataclass
class RunOptions:
    checkpoint: str = DEFAULT_CHECKPOINT
    control_checkpoint: str = DEFAULT_CONTROL_CHECKPOINT
    vllm_dir: str = DEFAULT_VLLM_DIR
    patch: Path = PATCH
    settings: Path = SETTINGS
    reference: Path | None = DEFAULT_REFERENCE
    out: Path = DEFAULT_OUT
    prompts: Path = bench.DEFAULT_PROMPTS
    port: int = bench.DEFAULT_PORT
    serve_cmd: list[str] | None = None
    over_budget: bool = False
    ready_timeout_s: float = bench.DEFAULT_READY_TIMEOUT_S
    request_timeout_s: float = bench.DEFAULT_REQUEST_TIMEOUT_S
    python: str = "python3"
    plan: Plan = field(default_factory=Plan)


def preflight(opts: RunOptions) -> dict[str, Any]:
    prompts = bench.PromptSet.load(opts.prompts)
    plan = opts.plan
    plan.control_reference = not (opts.reference and Path(opts.reference).is_file())
    check = check_checkpoint(opts.checkpoint)
    time_plan = estimate(plan, len(prompts.prompts))
    ok = check["ok"] and (time_plan["fits_with_extension"] or opts.over_budget)
    return {"ok": ok, "outcome": "ok" if ok else ("checkpoint" if not check["ok"] else "over_budget"),
            "checkpoint": check, "plan": asdict(plan), "time": time_plan}


def run_pod(opts: RunOptions, *, host: Any = None, power: Any = None, runner: Runner = run_cmd,
            on_bench_done: Callable[[Mapping[str, Any]], Any] | None = None) -> dict[str, Any]:
    """The pod side of R23 (see the module docstring); the result as written to result.json. on_bench_done is
    called with the bench answer before the correctness pass (all uses it to extend the slot)."""
    pre = preflight(opts)
    if not pre["ok"]:
        raise Refused(pre["outcome"], "preflight refused the run", preflight=pre)
    plan = opts.plan
    session = "quark-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:4]
    out = Path(opts.out) / session
    out.mkdir(parents=True, exist_ok=True)
    vllm_dir = Path(opts.vllm_dir).expanduser()
    loader = out / "loader.diff"
    loader.write_text(quark_keys.split_patch(Path(opts.patch).read_text(encoding="utf-8"))[1], encoding="utf-8")
    state = patch_state(vllm_dir, loader, runner)
    if state == "conflict":
        raise Refused("patch_conflict", f"the loader hunks neither apply to nor are already in {vllm_dir}",
                      vllm_dir=str(vllm_dir))
    sheet = settings_sheet(opts.settings, run_dir=out)
    template = list(opts.serve_cmd) if opts.serve_cmd else serve_template(sheet)
    before = snapshot(vllm_dir, opts.port, runner)
    started = time.time()
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA, "task": TASK, "session": session, "node": NODE, "checkpoint": CHECKPOINT,
        "checkpoint_dir": str(Path(opts.checkpoint).expanduser()), "control": CONTROL,
        "revision": quark_keys.CHECKPOINT_REVISION, "quantization": QUANTIZATION, "preflight": pre,
        "settings_sheet": sheet, "command": template, "patch": {"state_before": state, "applied_by_run": False,
                                                                 "reverted": False}}
    applied = False
    try:
        if plan.control_reference:      # arm A's reference outputs, on the build as found
            result["control_correctness"] = correctness_pass(
                template, str(Path(opts.control_checkpoint).expanduser()), out_dir=out, tag="control",
                max_tokens=plan.correctness_max_tokens, prompts_path=opts.prompts, port=opts.port,
                launch_env=sheet["env"], ready_timeout_s=opts.ready_timeout_s)
        if state == "absent":
            apply_patch(vllm_dir, loader, runner=runner)
            applied = True
            result["patch"]["applied_by_run"] = True
        config = bench.Config(
            engine="vllm", command=template, checkpoint=str(Path(opts.checkpoint).expanduser()), model="teacher",
            quantization=QUANTIZATION, engine_dir=str(vllm_dir), patches=[str(opts.patch)],
            launch_env=dict(sheet["env"]), node=NODE, port=opts.port, batches=list(plan.batches),
            add_64=plan.with_64, repeats=plan.repeats, share_spawn=True, max_tokens=plan.max_tokens,
            requests_per_stream=plan.requests_per_stream, warmup_rounds=plan.warmup_rounds, idle_s=plan.idle_s,
            ready_timeout_s=opts.ready_timeout_s, request_timeout_s=opts.request_timeout_s, out_dir=out / "bench",
            prompts=Path(opts.prompts), metric_prefix="quark", python=opts.python)
        session_result = bench.run_session(config, host=host, power=power)
        result["bench"] = {"session": session_result["session"], "rows_path": session_result["rows_path"],
                           "summary_path": session_result["summary_path"], "rows": session_result["rows"]}
        if on_bench_done is not None:
            result["extension"] = on_bench_done(result["bench"])
        result["correctness"] = correctness_pass(
            template, str(Path(opts.checkpoint).expanduser()), out_dir=out, tag="quark",
            max_tokens=plan.correctness_max_tokens, prompts_path=opts.prompts, port=opts.port,
            launch_env=sheet["env"], ready_timeout_s=opts.ready_timeout_s)
    finally:
        if applied:
            apply_patch(vllm_dir, loader, reverse=True, runner=runner)
            result["patch"]["reverted"] = True
        after = snapshot(vllm_dir, opts.port, runner)
        result["pod_as_found"] = {"ok": before == after, "before": before, "after": after}
        result["wall_s"] = round(time.time() - started, 1)
        (out / "result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    result["result_path"] = str(out / "result.json")
    return result


# --- record: the comparison, the ledger, T0 ---------------------------------------------------------------------

def read_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def arm_values(rows: Sequence[Mapping[str, Any]], batch: int, metric: str) -> list[float]:
    """One value per ok run row at that batch size (one row per spawn)."""
    return [float(r[metric]) for r in rows
            if r.get("outcome") == "ok" and r.get("batch_size") == batch and r.get(metric) is not None]


def compare_arms(control: Sequence[Mapping[str, Any]], quark: Sequence[Mapping[str, Any]],
                 batches: Sequence[int]) -> dict[str, Any]:
    """B (quark) against A (control) at each batch size: decode tok/s (higher better), joules per token (lower)."""
    out: dict[str, Any] = {}
    for b in batches:
        entry: dict[str, Any] = {}
        for metric, higher in (("decode_tok_s", True), ("joules_per_token", False)):
            a, q = arm_values(control, b, metric), arm_values(quark, b, metric)
            reading: dict[str, Any] = {"a": _describe(a), "b": _describe(q)}
            if a and q:
                reading["verdict"] = vmath.verdict(a, q, higher_is_better=higher).as_dict()
            else:
                reading["verdict"] = None
                reading["why"] = "no ok run of " + ("arm A" if not a else "arm B") + f" at batch {b}"
            entry[metric] = reading
        out[f"b{b}"] = entry
    return out


def _describe(values: Sequence[float]) -> dict[str, Any]:
    return {"n": len(values), "median": round(sorted(values)[len(values) // 2], 4) if values else None,
            "values": [round(v, 6) for v in values]}


def load_reference(path: str | os.PathLike[str]) -> dict[int, dict[str, str]]:
    """{spawn: {prompt_id: text}} from a reference outputs file (spawn 1 when a row names none)."""
    found: dict[int, dict[str, str]] = {}
    for row in read_jsonl(path):
        text = row.get("text", row.get("output"))
        if row.get("prompt_id") is None or text is None:
            raise ValueError(f"{path}: a reference row needs prompt_id and text")
        found.setdefault(int(row.get("spawn") or 1), {})[str(row["prompt_id"])] = str(text)
    return found


def first_divergence(a: str, b: str) -> int | None:
    """The first character index where the two texts differ, or None when they are equal."""
    if a == b:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def exact_match(reference: Mapping[str, str], outputs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The share of reference prompts whose output equals the reference text, with each mismatch's divergence."""
    by_id = {o["prompt_id"]: o for o in outputs}
    x, mismatches = 0, []
    for pid, text in reference.items():
        got = by_id.get(pid)
        if got is None or got.get("error"):
            mismatches.append({"prompt_id": pid, "why": "no output" if got is None else got["error"]})
            continue
        at = first_divergence(text, got["text"])
        if at is None:
            x += 1
        else:
            near = slice(max(0, at - 20), at + 20)
            mismatches.append({"prompt_id": pid, "first_divergence_char": at, "reference": text[near],
                               "output": got["text"][near]})
    n = len(reference)
    return {"x": x, "n": n, "rate": round(x / n, 4) if n else None, "mismatches": mismatches}


def _reference_from(results: Sequence[Mapping[str, Any]], path: Path | None) -> tuple[dict[int, dict[str, str]],
                                                                                         str] | None:
    if path is not None and Path(path).is_file():
        return load_reference(path), str(path)
    for r in results:
        cc = r.get("control_correctness")
        if cc:
            return {1: {o["prompt_id"]: o["text"] for o in cc["outputs"] if not o.get("error")}}, \
                f"{r['session']} control_correctness"
    return None


def _label_words(label: str | None) -> str:
    return {"b_ahead": "Quark ahead", "a_ahead": "RedHat ahead", "no_difference": "no difference",
            "too_early": "too early to call", None: "not read"}[label]


def verdict_text(comparisons: Mapping[str, Any], match: Mapping[str, Any] | None,
                 floor: Mapping[str, Any] | None, e5: Mapping[str, Any] | None) -> str:
    """The R23 result in one paragraph of plain ASCII."""
    parts = []
    for key, entry in comparisons.items():
        bits = []
        for metric, unit in (("decode_tok_s", "tok/s"), ("joules_per_token", "J/token")):
            r = entry[metric]
            v = r.get("verdict")
            bits.append(f"{unit} A {r['a']['median']} (n {r['a']['n']}) vs B {r['b']['median']} (n {r['b']['n']}), "
                        f"{_label_words(v['label'] if v else None)}")
        parts.append(f"batch {key[1:]}: " + "; ".join(bits))
    text = "R23, amd Quark AWQ INT4 (B) against RedHat INT4 (A) on pod B, same build and settings. " + ". ".join(parts)
    if match is not None:
        text += (f". Exact match of B against A's greedy outputs: {match['x']}/{match['n']}"
                 + (f" ({100 * match['rate']:.1f}%)" if match["rate"] is not None else ""))
        text += (f", A against itself {floor['x']}/{floor['n']}" if floor else ", A's self-match floor not measured")
        text += "; different quantizations, so a mismatch is not by itself a kernel defect"
    else:
        text += ". Exact match not measured: no reference outputs"
    if e5:
        text += f". T0 E5 (single-stream J/token, A vs B): {e5.get('label')}"
    return text + "."


def slot_result_line(comparisons: Mapping[str, Any], match: Mapping[str, Any] | None, session: str) -> str:
    b1 = comparisons.get("b1", {})
    def med(metric: str, arm: str) -> Any:
        return ((b1.get(metric) or {}).get(arm) or {}).get("median")
    line = (f"R23 {session}: b1 tok/s A {med('decode_tok_s', 'a')} B {med('decode_tok_s', 'b')}, "
            f"J/token A {med('joules_per_token', 'a')} B {med('joules_per_token', 'b')}")
    if match is not None:
        line += f", exact match {match['x']}/{match['n']}"
    return line[:slots.MAX_NOTE_CHARS]


def record_results(results: Sequence[Mapping[str, Any]], *, control_rows: Sequence[Mapping[str, Any]],
                   reference: Path | None, out_dir: Path, state_dir: Any = None,
                   control_source: str | None = None) -> dict[str, Any]:
    """Verdicts, the ledger row, T0's E5 result and the T0 text (see the module docstring)."""
    quark_rows = [row for r in results for row in (r.get("bench") or {}).get("rows", [])]
    if not quark_rows:
        raise Refused("no_rows", "the results hold no bench rows of arm B")
    batches = sorted({row["batch_size"] for row in quark_rows if row.get("outcome") == "ok"})
    comparisons = compare_arms(control_rows, quark_rows, batches)
    conditions_check = _conditions_check(control_rows, quark_rows)
    outputs = [o for r in results for o in (r.get("correctness") or {}).get("outputs", [])]
    found = _reference_from(results, reference)
    match = floor = match_verdict = None
    reference_source = None
    if found and outputs:
        spawns, reference_source = found
        match = exact_match(spawns[min(spawns)], outputs)
        if len(spawns) >= 2:
            first, second = sorted(spawns)[:2]
            floor = exact_match(spawns[first], [{"prompt_id": k, "text": v} for k, v in spawns[second].items()])
            match_verdict = vmath.verdict(vmath.Counts(floor["x"], floor["n"]),
                                          vmath.Counts(match["x"], match["n"])).as_dict()
    e5 = _record_e5(comparisons, results, state_dir)
    session = results[0]["session"]
    text = verdict_text(comparisons, match, floor, e5)
    metrics: dict[str, float] = {}
    for key, entry in comparisons.items():
        for metric in ("decode_tok_s", "joules_per_token"):
            median = entry[metric]["b"]["median"]
            if median is not None:
                metrics[f"quark.{metric}.{key}"] = median
    if match is not None and match["rate"] is not None:
        metrics["teacher.output_match_rate"] = match["rate"]
    b1 = (comparisons.get("b1") or {}).get("joules_per_token", {}).get("verdict")
    label = b1["label"] if b1 else vmath.TOO_EARLY
    seq = record.record(
        VERB, label, metrics=metrics, refs={"run_id": session}, task=TASK, state_dir=state_dir, node=NODE,
        units={k: ("tok/s" if "decode" in k else "J/token" if "joules" in k else "fraction") for k in metrics},
        conditions={"checkpoint": CHECKPOINT, "revision": quark_keys.CHECKPOINT_REVISION, "control": CONTROL,
                    "control_rows": control_source, "reference": reference_source,
                    "sessions": [r["session"] for r in results], "zero_point":
                        (((results[0].get("preflight") or {}).get("checkpoint") or {}).get("audit") or {}).get(
                            "zero_point"), "same_conditions": conditions_check},
        low_confidence=label == vmath.TOO_EARLY or not conditions_check["ok"], note=text)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0_lines = []
    if e5 and e5.get("experiment"):
        t0_lines.append(experiments.checklist_line(e5["experiment"]))
    t0_lines.append(text)
    t0_path = out_dir / T0_FILE
    t0_path.write_text("\n\n".join(t0_lines) + "\n", encoding="utf-8")
    verdict = {"label": label, "basis": "batch 1 joules per token, B against A (E5's metric)",
               "comparisons": comparisons, "exact_match": match, "self_match_floor": floor,
               "exact_match_verdict": match_verdict, "same_conditions": conditions_check, "text": text}
    (out_dir / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return {"verdict": verdict, "ledger_seq": seq, "e5": {k: v for k, v in (e5 or {}).items() if k != "experiment"},
            "t0_path": str(t0_path), "verdict_path": str(out_dir / "verdict.json"),
            "slot_result": slot_result_line(comparisons, match, session)}


_SAME = ("engine_commit", "prompt_set", "max_tokens", "thinking", "max_model_len")


def _conditions_check(control: Sequence[Mapping[str, Any]], quark: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Do the two arms' rows share what must be equal for a comparison (commit, prompts, decode settings)?"""
    def first(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        return next((r.get("conditions") or {} for r in rows if r.get("outcome") == "ok"), {})
    a, b = first(control), first(quark)
    differ = {k: {"a": a.get(k), "b": b.get(k)} for k in _SAME if a.get(k) != b.get(k)}
    return {"ok": not differ, "differ": differ}


def _record_e5(comparisons: Mapping[str, Any], results: Sequence[Mapping[str, Any]],
               state_dir: Any) -> dict[str, Any] | None:
    """T0's E5 with arms A and B at batch 1 (its metric is single-stream joules per token); kept running while arm C
    has no reading. None when either arm has no batch-1 run."""
    b1 = (comparisons.get("b1") or {}).get("joules_per_token")
    if not b1 or not b1["a"]["values"] or not b1["b"]["values"]:
        return None
    seeded = None
    if EXPERIMENT not in experiments.load(state_dir):
        seeded = experiments.seed(state_dir=state_dir)["added"]
    note = (f"R23 on pod B: arm A from R20's rows, arm B {', '.join(r['session'] for r in results)}; batch 1 "
            "joules per token per cold spawn; arm C (FP8) not run")
    folded = experiments.record_result(EXPERIMENT, {"A": b1["a"]["values"], "B": b1["b"]["values"]}, note=note,
                                       refs={"run_id": results[0]["session"]}, state_dir=state_dir)
    label = folded["result"]["label"]
    if folded["status"] == experiments.DONE:
        folded = experiments.amend(EXPERIMENT, {"status": experiments.RUNNING},
                                   reason="R23 read arms A and B; arm C (Qwen3.8-27B-FP8) has no reading yet",
                                   state_dir=state_dir)
    return {"label": label, "status": folded["status"], "seeded": seeded, "experiment": folded}


# --- slot -------------------------------------------------------------------------------------------------------

def take_slot(row: str | None, reply: str | None, *, agent: str = slots.DEFAULT_AGENT, client: Any = None,
              environ: Mapping[str, str] | None = None, now: Any = None, state_dir: Any = None) -> dict[str, Any]:
    """Request the 60-minute pod B row (or use the one given) and start it. outcome started, or waiting with why.
    agent is the queue's Agent column the rows are written under (the slot verbs' --agent)."""
    common = {"client": client, "environ": environ, "now": now, "state_dir": state_dir}
    if row is None:
        asked, code = slots.run(["request", "--task", TASK, "--what", SLOT_WHAT, "--minutes", str(SLOT_MINUTES),
                                 "--why", SLOT_WHY, "--measured", SLOT_MEASURED, "--kind", "bench",
                                 "--agent", agent], **common)
        if not asked.get("ok") or not asked.get("row"):
            return {"ok": False, "outcome": asked.get("outcome", "error"), "slot": asked}
        row = asked["row"]
    argv = ["start", "--row", row, "--agent", agent] + (["--reply", reply] if reply else [])
    started, code = slots.run(argv, **common)
    if started.get("outcome") == "started":
        return {"ok": True, "outcome": "started", "row": row, "slot": started}
    return {"ok": False, "outcome": "waiting", "row": row, "slot": started,
            "next": f"python -m lab quark-compare slot --row {row}" + (" --reply \"<answer to the critique>\""
                                                                       if started.get("outcome") ==
                                                                       "reply_required" else "")}


def release_slot(row: str, result: str, *, agent: str = slots.DEFAULT_AGENT, client: Any = None,
                 environ: Mapping[str, str] | None = None, now: Any = None, state_dir: Any = None) -> dict[str, Any]:
    answer, _ = slots.run(["done", "--row", row, "--agent", agent, "--result", result[:slots.MAX_NOTE_CHARS]],
                          client=client, environ=environ, now=now, state_dir=state_dir)
    return answer


# --- the command line -------------------------------------------------------------------------------------------

def _plan_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--batches", type=lambda t: [int(x) for x in t.split(",") if x.strip()], default=[1, 4],
                   help="comma-separated batch sizes (default 1,4: the matrix's 8 and 16 take a second slot)")
    p.add_argument("--with-64", action="store_true", help="add the 64 arm when it fits in memory")
    p.add_argument("--repeats", type=int, default=vmath.MIN_RUNS_PER_ARM)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--requests-per-stream", type=int, default=1)
    p.add_argument("--correctness-max-tokens", type=int, default=64)
    p.add_argument("--idle-s", type=float, default=bench.DEFAULT_IDLE_S, help="idle watts read before each spawn")
    p.add_argument("--over-budget", action="store_true", help="run a plan the time model says will not fit")


def _run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--control-checkpoint", default=DEFAULT_CONTROL_CHECKPOINT)
    p.add_argument("--vllm-dir", default=DEFAULT_VLLM_DIR)
    p.add_argument("--patch", default=str(PATCH))
    p.add_argument("--settings", default=str(SETTINGS))
    p.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    p.add_argument("--prompts", default=str(bench.DEFAULT_PROMPTS))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--port", type=int, default=bench.DEFAULT_PORT)
    p.add_argument("--serve-cmd", default=None, help="a launch template (JSON list or shell words) instead of the "
                   "settings sheet's vLLM line; the rehearsals use lab/bench_fake.py's")
    p.add_argument("--python", default="python3")
    _plan_args(p)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m lab quark-compare", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="sub", required=True)
    _run_args(sub.add_parser("preflight", help="checkpoint, config keys and the time plan; changes nothing"))
    s = sub.add_parser("slot", help="request and start the 60-minute pod B slot")
    s.add_argument("--row", default=None)
    s.add_argument("--reply", default=None)
    s.add_argument("--agent", default=slots.DEFAULT_AGENT, help="the queue's Agent column (the slot verbs' --agent)")
    _run_args(sub.add_parser("run", help="on pod B inside the slot: serve, bench, correctness, leave the pod"))
    r = sub.add_parser("record", help="verdict, ledger row, E5 result, T0 text, slot release")
    r.add_argument("--result", action="append", required=True)
    r.add_argument("--control-rows", required=True, help="arm A's bench rows.jsonl (R20)")
    r.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    r.add_argument("--row", default=None, help="release this slot row with the result line")
    r.add_argument("--out", default=None)
    r.add_argument("--agent", default=slots.DEFAULT_AGENT)
    a = sub.add_parser("all", help="slot, run and record in one process")
    _run_args(a)
    a.add_argument("--row", default=None)
    a.add_argument("--reply", default=None)
    a.add_argument("--control-rows", required=True)
    a.add_argument("--agent", default=slots.DEFAULT_AGENT)
    return p


def options_from(args: argparse.Namespace) -> RunOptions:
    serve = None
    if args.serve_cmd:
        text = args.serve_cmd.strip()
        serve = json.loads(text) if text.startswith("[") else shlex.split(text)
    plan = Plan(batches=args.batches, with_64=args.with_64, repeats=args.repeats, max_tokens=args.max_tokens,
                requests_per_stream=args.requests_per_stream, correctness_max_tokens=args.correctness_max_tokens,
                idle_s=args.idle_s)
    return RunOptions(checkpoint=args.checkpoint, control_checkpoint=args.control_checkpoint, vllm_dir=args.vllm_dir,
                      patch=Path(args.patch), settings=Path(args.settings),
                      reference=Path(args.reference) if args.reference else None, out=Path(args.out),
                      prompts=Path(args.prompts), port=args.port, serve_cmd=serve, over_budget=args.over_budget,
                      python=args.python, plan=plan)


def run(argv: Sequence[str], *, client: Any = None, environ: Mapping[str, str] | None = None, now: Any = None,
        state_dir: Any = None, host: Any = None, power: Any = None, ready_timeout_s: float | None = None,
        runner: Runner = run_cmd) -> dict[str, Any]:
    """One call: the answer dict. client, environ and now reach the slot queue; host and power reach the bench."""
    args = build_parser().parse_args(list(argv))
    try:
        if args.sub == "slot":
            return take_slot(args.row, args.reply, agent=args.agent, client=client, environ=environ, now=now,
                             state_dir=state_dir)
        if args.sub == "record":
            results = [json.loads(Path(f).read_text(encoding="utf-8")) for f in args.result]
            out = Path(args.out) if args.out else Path(args.result[0]).parent
            done = record_results(results, control_rows=read_jsonl(args.control_rows),
                                  reference=Path(args.reference) if args.reference else None, out_dir=out,
                                  state_dir=state_dir, control_source=args.control_rows)
            answer = {"ok": True, "outcome": done["verdict"]["label"], **done}
            if args.row:
                answer["released"] = release_slot(args.row, done["slot_result"], agent=args.agent, client=client,
                                                  environ=environ, now=now, state_dir=state_dir)
            return answer
        opts = options_from(args)
        if ready_timeout_s is not None:
            opts.ready_timeout_s = ready_timeout_s
        if args.sub == "preflight":
            return preflight(opts)
        if args.sub == "run":
            result = run_pod(opts, host=host, power=power, runner=runner)
            return {"ok": result["pod_as_found"]["ok"], "outcome": "ok" if result["pod_as_found"]["ok"] else
                    "pod_changed", "session": result["session"], "result_path": result["result_path"],
                    "pod_as_found": result["pod_as_found"]["ok"], "patch": result["patch"]}
        return run_all(args, opts, client=client, environ=environ, now=now, state_dir=state_dir, host=host,
                       power=power, runner=runner)
    except Refused as exc:
        return {"ok": False, "outcome": exc.outcome, "error": str(exc), **exc.more}


def run_all(args: argparse.Namespace, opts: RunOptions, **kw: Any) -> dict[str, Any]:
    """slot, run, record and release in one process; the slot is released whatever happens once it started."""
    pre = preflight(opts)
    if not pre["ok"]:
        return {"ok": False, "outcome": pre["outcome"], "preflight": pre}
    slot_kw = {k: kw[k] for k in ("client", "environ", "now", "state_dir")}
    took = take_slot(args.row, args.reply, agent=args.agent, **slot_kw)
    if took["outcome"] != "started":
        return took
    row = took["row"]
    answer: dict[str, Any] = {"row": row}
    result_line = "R23 failed before any result; see the runner's answer"
    needs_extension = pre["time"]["needs_extension"]

    def extend(bench_answer: Mapping[str, Any]) -> dict[str, Any] | None:
        """The plan runs past 60 minutes by the time model: take the one 15-minute extension while still inside
        the cap, saying what the slot has proven (the bench rows)."""
        if not needs_extension:
            return None
        ok = sum(1 for r in bench_answer["rows"] if r.get("outcome") == "ok")
        note = (f"R23 bench done: {ok} ok rows of {len(bench_answer['rows'])} in {bench_answer['session']}; "
                "correctness pass next")
        answer_, _ = slots.run(["extend", "--row", row, "--agent", args.agent, "--note", note], **slot_kw)
        return answer_

    try:
        result = run_pod(opts, host=kw["host"], power=kw["power"], runner=kw["runner"], on_bench_done=extend)
        answer.update(session=result["session"], result_path=result["result_path"],
                      pod_as_found=result["pod_as_found"]["ok"], patch=result["patch"])
        done = record_results([result], control_rows=read_jsonl(args.control_rows), reference=opts.reference,
                              out_dir=Path(result["result_path"]).parent, state_dir=kw["state_dir"],
                              control_source=args.control_rows)
        result_line = done["slot_result"]
        answer.update(ok=result["pod_as_found"]["ok"], outcome=done["verdict"]["label"], **done)
    except Refused as exc:
        answer.update(ok=False, outcome=exc.outcome, error=str(exc), **exc.more)
        result_line = f"R23 refused: {exc}"
    except Exception as exc:
        answer.update(ok=False, outcome="error", error=f"{type(exc).__name__}: {exc}")
        result_line = f"R23 failed: {type(exc).__name__}: {exc}"
    finally:
        answer["released"] = release_slot(row, result_line, agent=args.agent, **slot_kw)
    return answer


def main(argv: Sequence[str]) -> dict[str, Any]:
    """R23: the AMD Quark INT4 checkpoint against RedHat INT4 on pod B (preflight, slot, run, record, all)."""
    return run(argv)
