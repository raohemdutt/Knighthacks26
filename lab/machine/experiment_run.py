"""experiment run: one kernels-track experiment on pod B, from the ledger's definition to its verdict.

    python -m lab experiment run E3 [--account D] [--config FILE] [--repeats N] [--task ID] [--row ROW]
                                    [--reply TEXT] [--minutes 15|60 --why TEXT --measured TEXT] [--agent NAME]
                                    [--dry-run]

What one call does, in this order, and what it never does:

  check      the experiment exists in the ledger (lab/machine/experiments.py), is on the kernels track, its metric is
             one the harness measures (lab/bench.py METRIC_UNITS, after its "teacher." prefix), it has a run config
             (RUN_CONFIGS, or --config), the account fills the node the experiment names (pod-b unless it names
             another), and the placement plan (lab/placement.py) puts benchmarks on that node. A pod the plan forbids
             is refused here, before anything is asked of the queue or the pod.
  prepare    the pod script and its plan, written locally under <state dir>/experiment_runs/<run id>/: one
             `python3 lab/bench.py run ...` per distinct set of arguments the arms need (arms that differ only by
             batch size share one invocation and are told apart by the rows' batch_size), with --node, --out and
             --repeats (the experiment's min_n unless the config or --repeats says otherwise) added. The harness
             itself travels inside the script as a base64 tarball (BUNDLE_FILES), so the pod needs no clone of
             this repo, only python3 and httpx.
  pod        the pod must be up (lab/controller.py status); a pod that is not is reported, never launched.
  upload     the script (lab_exp_<run id>.sh) and the plan (lab_exp_<run id>.json) go into the pod's Jupyter root
             through the contents API, before any slot is asked for, so the slot is spent on the GPU only.
  slot       python -m lab slot request (task, "<id> bench run: <title>", kind bench, the config's minutes and,
             for a 60-minute slot, its justification), then slot start (--reply answers a referee's critique).
             Asking again with the same words returns the row already Queued, so a call that had to wait can be
             run again as it is. A start the queue refuses (busy, not next, a critique to answer) ends the call
             with outcome waiting: the uploads are removed and the Queued row stays in line.
  run        lab/controller.py run-script runs `bash lab_exp_<run id>.sh` in a terminal on the pod and waits for its
             exit code. The script unpacks the harness under lab_exp/<run id>/, runs each invocation with
             LAB_STATE_DIR inside the run folder (the pod's own ledger is never written) and copies each
             invocation's rows to lab_exp/<run id>/results/.
  pull       lab/transfer.py pull_file brings the rows, the harness's answer and its log back to the local folder.
  record     one bench.run ledger row per measured run (the harness's metrics under the experiment's prefix, refs
             run_id, slot, experiment and arm), then the verdict: experiments.record_result with the arms' values,
             the slot and the run id as refs.
  release    python -m lab slot done with one result line (the verdict and the arms' means, or what failed).
  clean      lab_exp/<run id>/ and every file the call put in the Jupyter root are removed (lab_exp/ too when the
             call made it), and the pod is left as it was found.
  ledger     one "experiment run" row with the outcome, the step that failed and why, and refs slot and run_id.

Every step after the slot start runs inside try/finally: whatever fails (the script, the pull, the ledger, an
interrupt), the slot is released, the pod is cleaned and the "experiment run" row says which step failed. Nothing
here launches or stops a pod, and nothing touches a server the pod was running: the only server is the harness's
own cold spawn on port 8100, which the harness starts and stops for each measured run and which refuses a port
that is already taken.

--dry-run does the check and the prepare and prints where the script is; nothing is sent anywhere.

The answer is one JSON line: ok, outcome (dry_run, recorded, waiting, refused, failed), the run id, the slot row,
the steps done, the failed step and error, the readings per arm and the verdict. Exit code 0 for recorded and
dry_run, 1 otherwise, 2 for bad arguments.

For other code: plan_run(), pod_script(), readings(), HubPod (the real pod) and Runner (the steps, with the pod and
the queue client handed in) are what the verb uses; the tests drive Runner against lab/hub_fake.py and
lab/machine/slots_fake.py.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import gzip
import io
import json
import os
import secrets
import shlex
import tarfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lab.machine import experiments, ledger, record, slots, stop

__all__ = ["TRACK", "RUN_ROOT", "BUNDLE_FILES", "RUN_CONFIGS", "RunRefused", "Plan", "plan_run", "pod_script",
           "cleanup_script", "bundle", "readings", "HubPod", "Runner", "main"]

TRACK = "kernels"
VERB = "experiment run"
READING_VERB = "bench.run"
DEFAULT_ACCOUNT = "D"
RUN_ROOT = "lab_exp"
WORK_DIR = "experiment_runs"
SCRIPT_MARGIN_S = 120               # the run-script wait stops this long before the slot's cap
HARNESS_MARGIN_S = 240              # the harness is interrupted this long before the cap: time to pull and clean
HARNESS_KILL_S = 60                 # after the interrupt, the harness has this long to stop its server and exit
RESULT_CHARS = slots.MAX_NOTE_CHARS
REPO = Path(__file__).resolve().parents[2]

#: What the harness needs on the pod: lab/bench.py, every lab module it imports (lab.proxy for the token stream's
#: parser, and what that imports), the prompt set and the patch it hashes.
BUNDLE_FILES: tuple[str, ...] = (
    "lab/__init__.py", "lab/bench.py", "lab/hub.py", "lab/pod_power.py", "lab/proxy.py", "lab/vllm-gfx1201.diff",
    "lab/machine/__init__.py", "lab/machine/ledger.py", "lab/machine/lock.py", "lab/machine/metrics.py",
    "lab/machine/ready.py", "lab/machine/record.py", "lab/machine/stop.py", "lab/machine/verdict.py",
    "fixtures/lab/bench/prompts.json", "fixtures/lab/bench/images/bars_1280x720.png",
    "fixtures/lab/bench/images/shapes_640x480.png",
)

#: How each kernels experiment runs on the harness. "bench" is the argument list every arm shares; each arm adds
#: "args" (none: the shared list alone) and may pick its rows by "batch". "~/" starts a path in the pod's home and
#: "{repo}/" one inside the unpacked harness. minutes, why and measured are the slot request's.
RUN_CONFIGS: dict[str, dict[str, Any]] = {
    "E3": {
        "about": "the baseline experiment: the stock record (vLLM b1388b1 with lab/vllm-gfx1201.diff serving "
                 "Qwen3.8-27B-INT4 on pod B, R30) at batch 1 against batches 4, 16 and 64, one shared cold spawn "
                 "per repeat (KE1); 64 runs only when the KV cache holds it (owner ruling 5), else a skipped row",
        "bench": ["--engine", "vllm", "--checkpoint", "~/models/Qwen3.8-27B-INT4", "--engine-dir", "~/vllm",
                  "--patch", "{repo}/lab/vllm-gfx1201.diff", "--batches", "1,4,16", "--share-spawn"],
        "arms": {"A": {"batch": 1}, "B": {"batch": 4}, "C": {"batch": 16}, "D": {"batch": 64}},
        "minutes": 60,
        "why": "3 cold spawns of the 27B teacher, each about 9 min to ready on pod D, cannot fit 15-minute slots",
        "measured": "R30: engine ready 523 s after a cold start on pod D; R29: 2.88 tok/s and 120 W at batch 1 on "
                    "pod C",
    },
}


class RunRefused(Exception):
    """A call refused before anything was sent: outcome and the reasons."""

    def __init__(self, outcome: str, problems: Sequence[str]) -> None:
        self.outcome = outcome
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


class StepFailed(Exception):
    """A step that failed after the slot started."""

    def __init__(self, step: str, error: str) -> None:
        self.step = step
        self.error = error
        super().__init__(f"{step}: {error}")


# --- the plan --------------------------------------------------------------------------------------------------

@dataclass
class Plan:
    experiment: dict[str, Any]
    config: dict[str, Any]
    run_id: str
    node: str
    field: str                                   # the bench row field the experiment's metric reads
    prefix: str                                  # the metric's prefix, e.g. teacher
    invocations: dict[str, list[str]]            # name -> bench run arguments, in run order
    arms: dict[str, dict[str, Any]]              # arm -> {"invocation": name, "batch": int | None}
    minutes: int
    repeats: int

    @property
    def run_dir(self) -> str:
        return f"{RUN_ROOT}/{self.run_id}"

    @property
    def script_name(self) -> str:
        return f"lab_exp_{self.run_id}.sh"

    @property
    def plan_name(self) -> str:
        return f"lab_exp_{self.run_id}.json"

    def as_dict(self) -> dict[str, Any]:
        return {"run_id": self.run_id, "experiment": self.experiment["id"], "node": self.node,
                "metric": self.experiment["metric"], "field": self.field, "repeats": self.repeats,
                "minutes": self.minutes, "run_dir": self.run_dir, "invocations": self.invocations, "arms": self.arms}


def new_run_id(eid: str, now: dt.datetime | None = None) -> str:
    moment = now or dt.datetime.now(dt.timezone.utc)
    return f"{eid}-{moment:%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


def _has_flag(args: Sequence[str], flag: str) -> bool:
    return any(a == flag or a.startswith(flag + "=") for a in args)


def plan_run(experiment: Mapping[str, Any], config: Mapping[str, Any] | None, *, run_id: str, node: str,
             repeats: int | None = None) -> Plan:
    """The plan for one experiment, or RunRefused (outcome refused) listing every reason it cannot run."""
    from lab import bench   # imported here: the harness is heavy and only a run needs it

    eid = experiment["id"]
    problems: list[str] = []
    if experiment.get("track") != TRACK:
        problems.append(f"{eid} is on the {experiment.get('track')} track; this runner runs {TRACK} experiments")
    if experiment.get("kind") != experiments.MEASURE:
        problems.append(f"{eid} is a {experiment.get('kind')} experiment; the harness gives run values (measure)")
    if experiment.get("status") in (experiments.DONE, experiments.DROPPED):
        problems.append(f"{eid} is {experiment['status']}; amend it back to queued to run it again")
    prefix, _, name = str(experiment.get("metric") or "").partition(".")
    if name not in bench.METRIC_UNITS:
        problems.append(f"{eid}'s metric {experiment.get('metric')} is not one the harness measures "
                        f"(<prefix>.{', <prefix>.'.join(bench.METRIC_UNITS)})")
    wanted = experiment.get("node")
    if wanted and wanted != node:
        problems.append(f"{eid} runs on {wanted}, and this account fills {node}")
    if config is None:
        problems.append(f"{eid} has no run config: add one to RUN_CONFIGS or pass --config FILE")
        raise RunRefused("refused", problems)
    common = [str(a) for a in config.get("bench") or []]
    arm_specs = config.get("arms") or {}
    declared = list(experiment.get("arms") or {})
    missing = [arm for arm in declared if arm not in arm_specs]
    unknown = [arm for arm in arm_specs if arm not in declared]
    if missing:
        problems.append(f"the run config has no arm {', '.join(missing)} of {eid}")
    if unknown:
        problems.append(f"the run config names arms {eid} does not have: {', '.join(unknown)}")
    for flag in ("--node", "--out"):
        if _has_flag(common, flag) or any(_has_flag(spec.get("args") or [], flag) for spec in arm_specs.values()):
            problems.append(f"the run config may not set {flag}: the runner sets it")
    count = repeats if repeats is not None else config.get("repeats") or experiment.get("min_n") or 1
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        problems.append(f"repeats must be a whole number of at least 1, not {count!r}")
    minutes = config.get("minutes", slots.DEFAULT_MINUTES)
    if minutes not in slots.SLOT_MINUTES:
        problems.append(f"minutes must be one of {', '.join(map(str, slots.SLOT_MINUTES))}, not {minutes!r}")
    if problems:
        raise RunRefused("refused", problems)
    invocations: dict[str, list[str]] = {}
    by_args: dict[tuple[str, ...], str] = {}
    arms: dict[str, dict[str, Any]] = {}
    for arm in declared:
        spec = arm_specs[arm]
        args = ["run", *common, *(str(a) for a in spec.get("args") or [])]
        if not _has_flag(args, "--repeats"):
            args += ["--repeats", str(count)]
        args += ["--node", node]
        key = tuple(args)
        if key not in by_args:
            by_args[key] = f"i{len(by_args) + 1}"
            invocations[by_args[key]] = args
        batch = spec.get("batch")
        arms[arm] = {"invocation": by_args[key], "batch": int(batch) if batch is not None else None}
    return Plan(dict(experiment), dict(config), run_id, node, name, prefix, invocations, arms, int(minutes), count)


# --- the scripts -----------------------------------------------------------------------------------------------

def bundle(files: Sequence[str] = BUNDLE_FILES, root: Path = REPO) -> str:
    """The harness as a base64 gzip tarball, its members in the order given with fixed times, so it is the same
    text for the same files."""
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name in files:
            data = (root / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tar.addfile(info, io.BytesIO(data))
    return base64.encodebytes(gzip.compress(out.getvalue(), mtime=0)).decode("ascii")


def shell_word(arg: str) -> str:
    """One argument in the pod script: ~/x is under $HOME, {repo}/x inside the unpacked harness, the rest quoted."""
    if arg.startswith("~/"):
        return '"$HOME"/' + shlex.quote(arg[2:])
    if arg.startswith("{repo}/"):
        return '"$REPO"/' + shlex.quote(arg[len("{repo}/"):])
    return shlex.quote(arg)


def pod_script(plan: Plan, *, bundle_text: str | None = None) -> str:
    """The script the pod runs from its Jupyter root: unpack the harness under the run folder, run each invocation,
    copy its rows and answer to results/. It exits with the first invocation's non-zero code, else 0."""
    eid = plan.experiment["id"]
    budget = max(60, plan.minutes * 60 - HARNESS_MARGIN_S)
    lines = [
        "#!/usr/bin/env bash",
        f"# experiment run {plan.run_id}: {eid} on {plan.node}. Written by lab/machine/experiment_run.py; the runner",
        "# removes the run folder and this script when it is done. Each harness call is interrupted (SIGINT, so it",
        "# stops the server it spawned) when the slot's time is nearly up.",
        "set -u",
        f"RUN={shlex.quote(plan.run_dir)}",
        f"DEADLINE=$(( $(date +%s) + {budget} ))",
        'PY="${LAB_PYTHON:-python3}"',
        f'[ -d {RUN_ROOT} ] || {{ mkdir {RUN_ROOT} && MADE_ROOT=1; }}',
        'if [ -e "$RUN" ]; then echo "### the run folder $RUN is already there"; exit 3; fi',
        'mkdir -p "$RUN/repo" "$RUN/results" "$RUN/state" "$RUN/out" || exit 3',
        '[ -n "${MADE_ROOT:-}" ] && touch "$RUN/made_root"',
        'REPO="$PWD/$RUN/repo"',
        "base64 -d > \"$RUN/bundle.tgz\" <<'LAB_BUNDLE_END' || exit 3",
        (bundle_text if bundle_text is not None else bundle()).rstrip("\n"),
        "LAB_BUNDLE_END",
        'tar -xzf "$RUN/bundle.tgz" -C "$REPO" || exit 3',
        '"$PY" -c "import httpx" || { echo "### httpx is missing on the pod: pip install --user httpx"; exit 4; }',
        "FIRST_RC=0",
    ]
    for name, args in plan.invocations.items():
        words = " ".join(shell_word(a) for a in args)
        lines += [
            f'echo "### {name} start"',
            'LEFT=$(( DEADLINE - $(date +%s) ))',
            f'if [ "$LEFT" -le 0 ]; then echo "### {name} skipped: the slot time is up"; RC=124; else',
            f'LAB_STATE_DIR="$PWD/$RUN/state" timeout -s INT -k {HARNESS_KILL_S} "$LEFT" '
            f'"$PY" "$REPO/lab/bench.py" {words} --out "$PWD/$RUN/out/{name}" '
            f'> "$RUN/results/{name}.answer.json" 2> "$RUN/results/{name}.log"',
            "RC=$?; fi",
            f'echo "### {name} rc=$RC"',
            f'cat "$RUN"/out/{name}/*/rows.jsonl > "$RUN/results/{name}.rows.jsonl" 2>/dev/null || true',
            '[ "$RC" -ne 0 ] && [ "$FIRST_RC" -eq 0 ] && FIRST_RC=$RC',
        ]
    lines += ['echo "### results in $RUN/results"', 'exit "$FIRST_RC"', ""]
    return "\n".join(lines)


def cleanup_script(plan: Plan, run_script_id: str) -> str:
    """What removes everything the run put on the pod: the run folder (and lab_exp/ when the run made it), the
    uploaded script and plan, and the run-script files of the run itself."""
    leftovers = [plan.script_name, plan.plan_name, f"lab_run_{run_script_id}.sh", f"lab_run_{run_script_id}.out"]
    return "\n".join([
        "#!/usr/bin/env bash",
        f"# cleanup of experiment run {plan.run_id}. Written by lab/machine/experiment_run.py.",
        "set -u",
        f"RUN={shlex.quote(plan.run_dir)}",
        'if [ -e "$RUN/made_root" ]; then MADE_ROOT=1; fi',
        '# a harness of this run still going (the wait for it gave up): interrupt it, so it stops its own server',
        'HARNESS="$PWD/$RUN/repo/lab/bench.py"',
        'if pgrep -f "$HARNESS" > /dev/null 2>&1; then',
        '  pkill -INT -f "$HARNESS"',
        f'  for _ in $(seq 1 {HARNESS_KILL_S + 30}); do pgrep -f "$HARNESS" > /dev/null 2>&1 || break; sleep 1; done',
        '  if pgrep -f "$HARNESS" > /dev/null 2>&1; then echo "### the harness of $RUN would not stop"; exit 1; fi',
        'fi',
        'rm -rf "$RUN"',
        f'[ -n "${{MADE_ROOT:-}}" ] && rmdir {RUN_ROOT} 2>/dev/null',
        "rm -f " + " ".join(shlex.quote(name) for name in leftovers),
        'if [ -e "$RUN" ]; then echo "### $RUN is still there"; exit 1; fi',
        "exit 0",
        "",
    ])


# --- reading the results ---------------------------------------------------------------------------------------

def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def readings(plan: Plan, rows_by_invocation: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, list[dict]]:
    """Per arm, the measured runs that read it: [{run_id, value, row}], one per ok row of its invocation (of its
    batch size when the arm names one), in row order; value is None when the row lacks the experiment's metric."""
    out: dict[str, list[dict[str, Any]]] = {}
    for arm, spec in plan.arms.items():
        taken = []
        for row in rows_by_invocation.get(spec["invocation"], ()):
            if row.get("outcome") != "ok":
                continue
            if spec["batch"] is not None and row.get("batch_size") != spec["batch"]:
                continue
            value = ledger.finite_number(row.get(plan.field))
            taken.append({"run_id": row["run_id"], "value": None if value is None else float(value), "row": row})
        out[arm] = taken
    return out


def reading_metrics(prefix: str, row: Mapping[str, Any]) -> tuple[dict[str, float], dict[str, str]]:
    """The metrics a harness row carries under the experiment's prefix, and their units (lab/bench.py's)."""
    from lab import bench

    values = {name: row.get(name) for name in ("decode_tok_s", "decode_tok_s_total", "ttft_s", "joules_per_token")}
    watts = row.get("watts") or {}
    values.update(watts_avg=watts.get("avg_w"), watts_peak=watts.get("peak_w"))
    metrics, units = {}, {}
    for name, value in values.items():
        number = ledger.finite_number(value)
        if number is not None:
            metrics[f"{prefix}.{name}"] = number
            units[f"{prefix}.{name}"] = bench.METRIC_UNITS[name]
    return metrics, units


# --- the pod ---------------------------------------------------------------------------------------------------

class HubPod:
    """One pod as the runner uses it, through the existing helpers: lab/controller.py for its state and to run a
    script, lab/proxy.py's PodLink for the contents API, lab/transfer.py to pull files."""

    def __init__(self, letter: str, controller: Any, link: Any) -> None:
        from lab import transfer

        self.letter = letter
        self.controller = controller
        self.link = link
        self.files = transfer.PodFiles(link)
        self.node = controller.accounts[letter].node

    @classmethod
    def connect(cls, letter: str, *, environ: Mapping[str, str] | None = None, transport: Any = None,
                sleep: Callable[[float], None] | None = None, clock: Callable[[], float] | None = None,
                base_url: str | None = None) -> "HubPod":
        import time

        from lab import controller as controller_module
        from lab import hub
        from lab.proxy import PodLink

        accounts = hub.load_accounts([letter], environ=environ, require_role=True)
        ctl = controller_module.Controller(accounts, base_url=base_url, transport=transport,
                                           sleep=sleep or time.sleep, clock=clock or time.time, environ=environ)
        link = PodLink.for_account(accounts[letter], base_url=base_url, transport=transport, environ=environ,
                                   sleep=sleep or time.sleep, clock=clock or time.perf_counter)
        return cls(letter, ctl, link)

    def close(self) -> None:
        self.link.close()
        self.controller.close()

    def ready(self) -> tuple[bool, str | None]:
        answer = self.controller.status([self.letter])
        entry = (answer.get("accounts") or {}).get(self.letter) or {}
        if not answer.get("ok", True) or not entry.get("reachable", True):
            return False, entry.get("error") or answer.get("error") or "the hub did not answer"
        if not entry.get("running") or not entry.get("ready") or entry.get("pending"):
            return False, "the pod is not up (the runner never launches one: python -m lab.controller launch)"
        return True, None

    def put_text(self, remote: str, text: str) -> None:
        self.link.request("PUT", self.files.contents_rel(remote),
                          json={"type": "file", "format": "text", "content": text}, retry=True)

    def run_script(self, script: Path, *, run_id: str, timeout_s: float) -> dict[str, Any]:
        return self.controller.run_script(self.letter, script, timeout_s=timeout_s, run_id=run_id)

    def pull(self, remote: str, local: Path) -> None:
        from lab import transfer

        transfer.pull_file(self.files, remote, local)

    def exists(self, remote: str) -> bool:
        return self.files.stat(remote) is not None

    def delete(self, remote: str) -> None:
        self.files.delete(remote)


# --- the run ---------------------------------------------------------------------------------------------------

def _clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


@dataclass
class Run:
    """What a call has done so far; the answer and the ledger row are made from it."""
    eid: str
    run_id: str | None = None
    row: str | None = None
    steps: list[str] = field(default_factory=list)
    failed_step: str | None = None
    error: str | None = None
    slot_started: bool = False
    released: bool | None = None
    cleaned: bool | None = None
    leftovers: list[str] = field(default_factory=list)
    readings: dict[str, Any] = field(default_factory=dict)
    verdict: dict[str, Any] | None = None
    script_exit: int | None = None
    current: str = "run"                         # the step under way after the slot started

    def fail(self, step: str, error: str) -> None:
        if self.failed_step is None:
            self.failed_step, self.error = step, _clip(error, 500)


class Runner:
    """The steps of one experiment run. pod is a HubPod (or anything with its methods); slot_client the Notion
    client of the queue (None: the token's, as python -m lab slot finds it); now the queue's clock."""

    def __init__(self, pod: Any, *, slot_client: Any = None, environ: Mapping[str, str] | None = None,
                 now: Callable[[], dt.datetime] | None = None, state_dir: str | os.PathLike[str] | None = None,
                 agent: str = slots.DEFAULT_AGENT) -> None:
        self.pod = pod
        self.slot_client = slot_client
        self.environ = os.environ if environ is None else environ
        self.now = now
        self.state_dir = state_dir
        self.agent = agent

    # the queue

    def _slot(self, *argv: str) -> tuple[dict[str, Any], int]:
        try:
            return slots.run([*argv, "--agent", self.agent], client=self.slot_client, environ=self.environ,
                             now=self.now, state_dir=self.state_dir)
        except SystemExit as exc:            # argparse refusing an argument: an answer, never an exit mid-run
            return {"ok": False, "outcome": "bad_arguments", "error": f"slot {argv[0]}: bad arguments ({exc.code})"}, 2

    # the steps

    def check(self, eid: str, config: Mapping[str, Any] | None, *, run_id: str | None,
              repeats: int | None) -> Plan:
        experiment = experiments.load(self.state_dir).get(eid)
        if experiment is None:
            raise RunRefused("refused", [f"no experiment {eid} in the ledger (python -m lab experiment list)"])
        node = self.pod.node
        plan = plan_run(experiment, config if config is not None else RUN_CONFIGS.get(eid),
                        run_id=run_id or new_run_id(eid), node=node, repeats=repeats)
        from lab import placement

        letter = placement.GPU_OF.get(node)
        if letter is None:
            raise RunRefused("refused", [f"{node} is not a card of the GPU queue"])
        verdict = placement.check_request(gpu=letter, text=self.what(plan), kind="bench", environ=self.environ)
        if verdict.get("contradicts"):
            raise RunRefused("placement_forbids", [verdict.get("reason") or f"the placement plan forbids {node}"])
        plan.config["placement"] = {k: verdict.get(k) for k in ("verdict", "kind", "node", "reason")}
        return plan

    @staticmethod
    def check_slot(plan: Plan, *, task: str | None, row: str | None, reply: str | None, why: str | None,
                   measured: str | None) -> dict[str, Any]:
        """The slot request's arguments, checked before anything is sent; RunRefused lists what is wrong."""
        problems = []
        task = task or plan.experiment.get("task") or experiments.DEFAULT_TASK
        if not slots.TASK_ID.fullmatch(task):
            problems.append(f"--task must be a Task Board ID such as T0 or O29, not {task!r}")
        why = why or plan.config.get("why")
        measured = measured or plan.config.get("measured")
        if row is None and plan.minutes > slots.LONG_ABOVE_MINUTES:
            for name, text, limit in (("why", why, slots.MAX_WHY_CHARS), ("measured", measured,
                                                                         slots.MAX_MEASURED_CHARS)):
                if not text:
                    problems.append(f"a {plan.minutes}-minute slot needs --{name} (or the config's {name})")
                elif len(text) > limit:
                    problems.append(f"--{name} is {len(text)} characters; the queue takes {limit}")
        if reply is not None and not 0 < len(reply) <= slots.MAX_REPLY_CHARS:
            problems.append(f"--reply takes 1 to {slots.MAX_REPLY_CHARS} characters")
        if problems:
            raise RunRefused("refused", problems)
        return {"task": task, "row": row, "reply": reply, "why": why, "measured": measured}

    @staticmethod
    def what(plan: Plan) -> str:
        title = plan.experiment.get("title") or plan.experiment["lever"]
        return _clip(f"{plan.experiment['id']} bench run: {title}", slots.MAX_WHAT_CHARS)

    def prepare(self, plan: Plan) -> Path:
        work = ledger.resolve_state_dir(self.state_dir) / WORK_DIR / plan.run_id
        work.mkdir(parents=True, exist_ok=True)
        (work / plan.script_name).write_text(pod_script(plan), encoding="utf-8", newline="\n")
        (work / plan.plan_name).write_text(json.dumps(plan.as_dict(), indent=2), encoding="utf-8")
        launcher = work / "launch.sh"
        launcher.write_text(f"bash ./{plan.script_name}\n", encoding="utf-8", newline="\n")
        return work

    def run(self, eid: str, *, config: Mapping[str, Any] | None = None, repeats: int | None = None,
            task: str | None = None, row: str | None = None, reply: str | None = None, minutes: int | None = None,
            why: str | None = None, measured: str | None = None, dry_run: bool = False,
            run_id: str | None = None) -> dict[str, Any]:
        state = Run(eid)
        try:
            plan = self.check(eid, config, run_id=run_id, repeats=repeats)
            if minutes is not None:
                plan.minutes = minutes
            slot_args = self.check_slot(plan, task=task, row=row, reply=reply, why=why, measured=measured)
        except RunRefused as exc:
            state.error = _clip(str(exc), 500)
            return self._finish(state, None, exc.outcome, dry_run=dry_run, problems=exc.problems, error=str(exc))
        state.run_id = plan.run_id
        state.steps.append("check")
        work = self.prepare(plan)
        state.steps.append("prepare")
        if dry_run:
            return self._finish(state, plan, "dry_run", dry_run=True, work=str(work),
                                script=str(work / plan.script_name),
                                what=self.what(plan))
        ok, why_not = self.pod.ready()
        if not ok:
            state.fail("pod", why_not or "the pod is not up")
            return self._finish(state, plan, "failed", work=str(work))
        state.steps.append("pod")
        uploaded: list[str] = []
        try:
            for name in (plan.script_name, plan.plan_name):
                self.pod.put_text(name, (work / name).read_text(encoding="utf-8"))
                uploaded.append(name)
        except Exception as exc:
            state.fail("upload", f"{type(exc).__name__}: {exc}")
            self._remove(state, uploaded)
            return self._finish(state, plan, "failed", work=str(work))
        state.steps.append("upload")
        started, outcome, extra = self._take_slot(state, plan, **slot_args)
        if not started:
            self._remove(state, uploaded)
            return self._finish(state, plan, outcome, work=str(work), **extra)
        try:
            self._measure(state, plan, work)
        except StepFailed as exc:
            state.fail(exc.step, exc.error)
        except Exception as exc:
            state.fail(state.current, f"{type(exc).__name__}: {exc}")
        except BaseException as exc:          # an interrupt or the kill switch: put the slot and the pod right first
            state.fail(state.current, f"{type(exc).__name__}: {exc}")
            self._release(state, plan)
            self._clean(state, plan)
            self._finish(state, plan, "failed", work=str(work))
            raise
        self._release(state, plan)
        self._clean(state, plan)
        return self._finish(state, plan, "recorded" if state.failed_step is None else "failed", work=str(work))

    def _take_slot(self, state: Run, plan: Plan, *, task: str, row: str | None, reply: str | None,
                   why: str | None, measured: str | None) -> tuple[bool, str, dict[str, Any]]:
        if row is None:
            argv = ["request", "--task", task, "--what", self.what(plan), "--kind", "bench",
                    "--minutes", str(plan.minutes)]
            if plan.minutes > slots.LONG_ABOVE_MINUTES:
                argv += ["--why", why, "--measured", measured]
            answer, code = self._slot(*argv)
            if not answer.get("ok"):
                state.fail("slot_request", answer.get("reason") or answer.get("error") or answer.get("outcome", ""))
                outcome = "no_token" if answer.get("outcome") == "no_token" else "failed"
                return False, outcome, {"slot": _brief(answer)}
            row = answer.get("row")
            state.steps.append("slot_request")
        state.row = row
        argv = ["start", "--row", row]
        if reply:
            argv += ["--reply", reply]
        answer, code = self._slot(*argv)
        if answer.get("outcome") != "started":
            state.error = _clip(answer.get("reason") or answer.get("error") or answer.get("outcome", ""), 500)
            waiting = answer.get("outcome") in ("busy", "not_next", "critique_pending", "reply_required",
                                                "lost_race")
            if not waiting:
                state.failed_step = "slot_start"
            rerun = f"python -m lab experiment run {plan.experiment['id']} --row {row}"
            if answer.get("outcome") == "reply_required":
                rerun += ' --reply "..."'
            return False, "waiting" if waiting else "failed", {"slot": _brief(answer), "next": rerun}
        state.slot_started = True
        state.steps.append("slot_start")
        return True, "started", {}

    def _measure(self, state: Run, plan: Plan, work: Path) -> None:
        timeout_s = max(60.0, plan.minutes * 60.0 - SCRIPT_MARGIN_S)
        state.current = "run"
        answer = self.pod.run_script(work / "launch.sh", run_id=plan.run_id, timeout_s=timeout_s)
        state.script_exit = answer.get("exit_code")
        state.steps.append("run")
        if answer.get("outcome") not in ("ok", "fail"):
            raise StepFailed("run", answer.get("error") or answer.get("outcome") or "the script did not finish")
        if answer.get("outcome") == "fail":
            state.fail("run", f"the pod script exited {state.script_exit}: {answer.get('output_tail', '')[-300:]}")
        state.current = "pull"
        rows_by_invocation: dict[str, list[dict[str, Any]]] = {}
        pulled = work / "results"
        missing = []
        for name in plan.invocations:
            for suffix in ("rows.jsonl", "answer.json", "log"):
                remote = f"{plan.run_dir}/results/{name}.{suffix}"
                try:
                    self.pod.pull(remote, pulled / f"{name}.{suffix}")
                except Exception as exc:
                    if suffix == "rows.jsonl":
                        missing.append(f"{remote} ({type(exc).__name__}: {exc})")
            rows_by_invocation[name] = read_rows(pulled / f"{name}.rows.jsonl")
        state.steps.append("pull")
        if missing:
            state.fail("pull", "could not pull " + "; ".join(missing))
        taken = readings(plan, rows_by_invocation)
        state.readings = {arm: [r["run_id"] for r in runs] for arm, runs in taken.items()}
        state.current = "ledger"
        self._record_readings(state, plan, taken)
        state.steps.append("ledger")
        state.current = "verdict"
        self._verdict(state, plan, taken)

    def _record_readings(self, state: Run, plan: Plan, taken: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        eid = plan.experiment["id"]
        written: set[str] = set()
        for arm, runs in taken.items():
            for run in runs:
                row = run["row"]
                if row["run_id"] in written:
                    continue
                written.add(row["run_id"])
                metrics, units = reading_metrics(plan.prefix, row)
                conditions = dict(row.get("conditions") or {}, run_id=row["run_id"],
                                  batch_size=row.get("batch_size"), experiment_run=plan.run_id)
                record.record(READING_VERB, row.get("outcome", "ok"), metrics,
                              {"run_id": row["run_id"], "slot": state.row, "experiment": eid, "arm": arm},
                              task=plan.experiment.get("task"), node=row.get("node") or plan.node, units=units,
                              conditions=conditions, low_confidence=True, state_dir=self.state_dir,
                              source=f"lab/bench.py on {plan.node}, pulled by experiment run {plan.run_id}",
                              note=row.get("reason"))

    def _verdict(self, state: Run, plan: Plan, taken: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        eid = plan.experiment["id"]
        control = next(iter(plan.arms))
        values = {arm: [r["value"] for r in runs if r["value"] is not None] for arm, runs in taken.items()}
        arms = {arm: found for arm, found in values.items() if found}
        if control not in arms or len(arms) < 2:
            empty = [arm for arm in plan.arms if arm not in arms]
            state.fail("verdict", f"no verdict: no run with {plan.experiment['metric']} for arm(s) "
                                  f"{', '.join(empty)} (the control is {control})")
            return
        note = f"experiment run {plan.run_id} on {plan.node}"
        empty = [arm for arm in plan.arms if arm not in arms]
        if empty:
            note += f"; no run with {plan.experiment['metric']} for {', '.join(empty)}"
        result = experiments.record_result(eid, arms, control=control, note=note,
                                           refs={"slot": state.row, "run_id": plan.run_id}, state_dir=self.state_dir)
        state.verdict = {"label": result["result"]["label"], "winner": result["result"]["winner"],
                         "status": result["status"], "seq": result["seq"],
                         "summary": experiments._result_text(result["result"])}
        state.steps.append("verdict")

    def _result_line(self, state: Run, plan: Plan) -> str:
        head = f"{plan.experiment['id']} run {plan.run_id}"
        if state.verdict is not None and state.failed_step is None:
            return _clip(f"{head}: {state.verdict['summary']}", RESULT_CHARS)
        return _clip(f"{head} failed at {state.failed_step}: {state.error}", RESULT_CHARS)

    def _release(self, state: Run, plan: Plan) -> None:
        if not state.slot_started or state.released:
            return
        try:
            answer, _ = self._slot("done", "--row", state.row, "--result", self._result_line(state, plan))
            state.released = answer.get("outcome") == "done"
            if not state.released:
                state.fail("release", answer.get("reason") or answer.get("error") or answer.get("outcome", ""))
        except Exception as exc:
            state.released = False
            state.fail("release", f"{type(exc).__name__}: {exc}")
        if state.released:
            state.steps.append("release")

    def _remove(self, state: Run, names: Sequence[str]) -> None:
        """Delete files this call put in the Jupyter root; what cannot be deleted is a leftover."""
        for name in names:
            try:
                self.pod.delete(name)
            except Exception as exc:
                state.leftovers.append(f"{name} ({type(exc).__name__}: {exc})")
        state.cleaned = not state.leftovers

    def _clean(self, state: Run, plan: Plan) -> None:
        clean_id = f"{plan.run_id}-clean"
        work = ledger.resolve_state_dir(self.state_dir) / WORK_DIR / plan.run_id
        script = work / "cleanup.sh"
        script.write_text(cleanup_script(plan, plan.run_id), encoding="utf-8", newline="\n")
        try:
            answer = self.pod.run_script(script, run_id=clean_id, timeout_s=300.0)
            if not answer.get("ok"):
                state.leftovers.append(f"{plan.run_dir} (cleanup script: {answer.get('outcome')}: "
                                       f"{answer.get('error') or answer.get('output_tail', '')[-200:]})")
        except (Exception, stop.Stopped) as exc:     # the kill switch refuses run-script; delete what can be
            state.leftovers.append(f"{plan.run_dir} (cleanup script: {type(exc).__name__}: {exc})")
            self._remove(state, [plan.script_name, plan.plan_name, f"lab_run_{plan.run_id}.sh",
                                 f"lab_run_{plan.run_id}.out"])
        for name in (f"lab_run_{clean_id}.sh", f"lab_run_{clean_id}.out"):
            try:
                self.pod.delete(name)
            except Exception as exc:
                state.leftovers.append(f"{name} ({type(exc).__name__}: {exc})")
        state.cleaned = not state.leftovers
        if state.cleaned:
            state.steps.append("clean")
        else:
            state.fail("clean", "left on the pod: " + "; ".join(state.leftovers))

    def _finish(self, state: Run, plan: Plan | None, outcome: str, *, dry_run: bool = False,
                **more: Any) -> dict[str, Any]:
        ok = outcome in ("recorded", "dry_run")
        answer: dict[str, Any] = {"ok": ok, "outcome": outcome, "verb": VERB, "id": state.eid,
                                  "run_id": state.run_id, "row": state.row, "steps": state.steps}
        if state.failed_step or state.error:
            answer.update(failed_step=state.failed_step, error=state.error)
        if plan is not None:
            answer.update(node=plan.node, metric=plan.experiment["metric"], repeats=plan.repeats,
                          minutes=plan.minutes, invocations=plan.invocations,
                          placement=plan.config.get("placement"))
        if state.slot_started:
            answer["slot_released"] = bool(state.released)
        if state.cleaned is not None:
            answer["pod_cleaned"] = state.cleaned
        if state.leftovers:
            answer["leftovers"] = state.leftovers
        if state.readings:
            answer["readings"] = state.readings
        if state.verdict is not None:
            answer["verdict"] = state.verdict
        if state.script_exit is not None:
            answer["script_exit"] = state.script_exit
        answer.update(more)
        if not dry_run:
            try:
                answer["ledger_seq"] = record.record(
                    VERB, outcome, {}, {"slot": state.row, "run_id": state.run_id, "experiment": state.eid},
                    task=(plan.experiment.get("task") if plan else None) or record.current_task(self.environ),
                    node=plan.node if plan else None, state_dir=self.state_dir,
                    note=None if not (state.failed_step or state.error)
                    else _clip(f"{state.failed_step or outcome}: {state.error}", 500),
                    conditions={"steps": state.steps, "failed_step": state.failed_step,
                                "slot_released": answer.get("slot_released"), "pod_cleaned": state.cleaned,
                                "label": (state.verdict or {}).get("label")})
            except Exception as exc:
                answer["ledger_seq"] = None
                answer["ledger_error"] = f"{type(exc).__name__}: {exc}"
        return answer


def _brief(answer: Mapping[str, Any]) -> dict[str, Any]:
    return {key: answer.get(key) for key in ("outcome", "row", "reason", "next", "error") if answer.get(key)}


# --- the verb --------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m lab experiment run", description=main.__doc__)
    p.add_argument("id", help="the experiment, e.g. E3")
    p.add_argument("--account", default=DEFAULT_ACCOUNT, help=f"the hub account of the pod (default {DEFAULT_ACCOUNT})")
    p.add_argument("--config", help="a JSON run config (the shape of RUN_CONFIGS) instead of the built-in one")
    p.add_argument("--repeats", type=int, help="spawns per configuration (default: the config's, else min_n)")
    p.add_argument("--task", help="the Task Board ID of the slot request (default: the experiment's task)")
    p.add_argument("--row", help="a GPU Queue row already requested for this run")
    p.add_argument("--reply", help="the answer to the referee's critique, for slot start")
    p.add_argument("--minutes", type=int, choices=slots.SLOT_MINUTES, help="the slot's minutes (default: the config's)")
    p.add_argument("--why", help="why a 60-minute slot cannot be split (default: the config's)")
    p.add_argument("--measured", help="what was already measured (default: the config's)")
    p.add_argument("--agent", default=slots.DEFAULT_AGENT, help=f"whose queue rows (default {slots.DEFAULT_AGENT})")
    p.add_argument("--dry-run", action="store_true", help="check and prepare only; send nothing")
    return p


def main(argv: Sequence[str]) -> dict[str, Any]:
    """Run a kernels experiment on pod B: upload, slot, harness, pull, ledger rows and verdict, release, clean."""
    args = build_parser().parse_args(list(argv))
    config = None
    if args.config:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    environ = os.environ
    from lab import hub

    if args.dry_run:
        try:
            pod = _OfflinePod(hub.node_of(args.account.upper(), hub.pod_roles(environ)))
        except hub.PodRoleError as exc:
            return {"ok": False, "outcome": "no_pod_role", "verb": VERB, "id": args.id, "error": str(exc)}
        return Runner(pod, environ=environ, agent=args.agent).run(
            args.id, config=config, repeats=args.repeats, task=args.task, row=args.row, reply=args.reply,
            minutes=args.minutes, why=args.why, measured=args.measured, dry_run=True)
    try:
        pod = HubPod.connect(args.account.upper(), environ=environ)
    except (hub.MissingCredentials, hub.PodRoleError) as exc:
        return {"ok": False, "outcome": "no_credentials", "verb": VERB, "id": args.id, "error": str(exc)}
    try:
        return Runner(pod, environ=environ, agent=args.agent).run(
            args.id, config=config, repeats=args.repeats, task=args.task, row=args.row, reply=args.reply,
            minutes=args.minutes, why=args.why, measured=args.measured)
    finally:
        pod.close()


class _OfflinePod:
    """The pod of --dry-run: only the node its account fills (LAB_POD_ROLES); nothing is sent to it."""

    def __init__(self, node: str) -> None:
        self.node = node
