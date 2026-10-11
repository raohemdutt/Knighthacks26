"""The correctness harness: greedy outputs recorded from the baseline, compared on every later run (R21).

A speedup that changes outputs is a bug (kernels rubric K4). This records what the stock server answers, token for
token, on the R11 prompt set, and later checks a changed build against it.

    python -m lab correctness record  --account C --engine vllm --chip gfx1151 --node pod-a --engine-commit b1388b1 \\
        --patch lab/vllm-gfx1201.diff --quantization int4 --rocm 7.13 --torch 2.9.1+rocm7.13.0 --triton 3.5.1
    python -m lab correctness compare --account C --baseline fixtures/lab/correctness/vllm-gfx1151-2026-10-10.json \\
        [--allow-condition-drift] [--metrics-file]

Both talk to a server that is already running (`--account C [--port 8000]` through the hub proxy, or `--url
http://127.0.0.1:8000`); nothing is spawned, restarted or changed. Requests are sent one at a time, so a run is one
prompt after another on a server that may be doing other work: greedy outputs do not depend on load, but they can
depend on batch composition, which is why rubric K4 (b) judges one prompt at a time. Every call prints one JSON line.

The requests are lab/bench.py's: its fixed prompt set (fixtures/lab/bench/prompts.json: text and drawn images, never
camera frames), its PromptSet and Settings, its DirectEndpoint and ProxyEndpoint. They are greedy (temperature 0, top_p
1, seed 0), have a fixed --max-tokens (128) with ignore_eos so every prompt is compared over the same number of tokens,
ask the Qwen chat template for no thinking, and carry no per-request nonce (the same prompt must be the same bytes on
every run). They also ask for logprobs with return_tokens_as_token_ids, so an output is a list of token ids with the
logprob of each chosen token. A server that sends no logprobs is recorded by its streamed chunks instead
(`token_kind` "chunk_text"), and a baseline and a run must be of the same kind.

The fixture, fixtures/lab/correctness/<engine>-<chip>-<date>.json, holds the conditions (engine, engine commit, sha256
of each patch, checkpoint, quantization, max model length, served model name, node, chip, ROCm, torch, Triton, the
sampling settings, the prompt-set hash) and, per prompt, the text, the token ids, their logprobs, the finish reason
and the token counts. What the box cannot read from a pod it is told with flags (`--engine-commit`, `--rocm`, ...,
`--set NAME=VALUE` for anything else, such as `--set enforce_eager=true`); what is neither given nor served is null
with its reason under `unavailable`.

Compare. It sends the same prompts, then reports per prompt: the exact token match, the first divergence index (the
first position where the token ids differ, or where one output ends before the other), the logprob of the baseline's
token and of the run's token there and their difference (rubric K4 d), and a normalised-text match (Unicode NFC,
whitespace runs collapsed, ends trimmed, case kept). The summary row is the prompts, matches, divergences, the match
rate, the normalised-text matches and the first prompt that diverged.

Conditions. A run is only comparable when what decides the answer is the same. The checkpoint, quantization, max model
length, served model name, prompt set, sampling settings, token kind, engine, engine commit, chip, ROCm, torch,
Triton, enforce_eager and any `--set` value that the baseline and the run both know must be equal, or the compare is
refused with the list of what differs, unless `--allow-condition-drift` (the row then says it drifted and lists the
differences). The patch hash is not in that list: a changed patch is the change under test, so it is reported as
`under_test` beside the result, not refused. A condition one side does not know (null) cannot be checked and is listed
under `unverified`.

Rows. Every call is a `correctness` ledger row (the @recorded hook: LAB_STATE_DIR names the ledger), with the
summary as its metrics (`teacher.output_matches`, `.output_divergences`, `.output_prompts`, `.output_match_rate`;
the prefix is --metric-prefix) and the first diverging prompt in its refs. With --metrics-file the match rate is also
appended to lab/refine/metrics.jsonl (or the file named) as `teacher.output_match_rate` (rubric section on metrics),
low_confidence while it is one run. A compare also writes its full result to lab/state/correctness/<run id>.json.
Tested against lab/bench_fake.py.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import re
import sys
import time
import unicodedata
import uuid
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

if not __package__:  # run by path on a pod: the repo root, one folder up, holds lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import bench  # noqa: E402
from lab.machine import metrics as metrics_file  # noqa: E402
from lab.machine.record import recorded  # noqa: E402

CLI_VERB = "correctness"   # registers python -m lab correctness
HARNESS = "lab/correctness.py 1"
BASELINE_SCHEMA = "lab.correctness.baseline/1"
RESULT_SCHEMA = "lab.correctness.result/1"
REPO = Path(__file__).resolve().parents[1]
DEFAULT_DIR = REPO / "fixtures" / "lab" / "correctness"
DEFAULT_STATE = REPO / "lab" / "state" / "correctness"
DEFAULT_MAX_TOKENS = 128
DEFAULT_REQUEST_TIMEOUT_S = 900.0
# What must be equal for two runs to be comparable (patch_hash is the change under test, so it is not here).
MUST_MATCH = ("engine", "engine_commit", "checkpoint", "quantization", "max_model_len", "served_model", "chip",
              "rocm", "torch", "triton", "token_kind")
SETTINGS_KEYS = ("max_tokens", "temperature", "top_p", "seed", "ignore_eos", "thinking", "nonce")
_TOKEN_ID = re.compile(r"token_id:(\d+)$")


class CorrectnessError(Exception):
    """A call the harness refuses or cannot finish; its message is the answer's error."""


def log(message: str) -> None:
    print(f"[correctness {time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def normalise(text: str) -> str:
    """Text as the normalised match reads it: Unicode NFC, whitespace runs one space, ends trimmed, case kept."""
    return " ".join(unicodedata.normalize("NFC", text).split())


# --- one greedy output ------------------------------------------------------------------------------------------

def generate(endpoint: Any, prompts: bench.PromptSet, settings: bench.Settings, index: int) -> dict[str, Any]:
    """Send prompt `index` and read the stream: its text, the token ids (or chunk texts) and their logprobs."""
    body = prompts.request_body(index, settings, None)
    body.update({"logprobs": True, "top_logprobs": 0, "return_tokens_as_token_ids": True})
    text: list[str] = []
    ids: list[int | str] = []
    chunks: list[str] = []
    logprobs: list[float | None] = []
    usage: dict[str, Any] = {}
    finish = error = None
    started = time.perf_counter()
    try:
        for _, data in endpoint.events(body):
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                error = "a stream event that is not JSON"
                break
            if not isinstance(chunk, dict):
                continue
            if "error" in chunk and not chunk.get("choices"):
                error = f"the server streamed an error: {str(chunk['error'])[:200]}"
                break
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                content = (choice.get("delta") or {}).get("content")
                if content:
                    text.append(content)
                    chunks.append(content)
                entries = ((choice.get("logprobs") or {}).get("content")) or []
                for entry in entries:
                    token = str(entry.get("token"))
                    found = _TOKEN_ID.match(token)
                    ids.append(int(found.group(1)) if found else token)
                    logprobs.append(entry.get("logprob"))
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
    except Exception as exc:  # a failed request is that prompt's error, never a crash of the call
        error = f"{type(exc).__name__}: {' '.join(str(exc).split())[:300]}"
    if error is None and not chunks:
        error = "no token arrived"
    if ids and all(isinstance(i, int) for i in ids):
        kind, tokens, lps = "token_id", ids, logprobs
    else:   # no logprobs (or not as ids): the streamed chunks stand for the tokens
        kind, tokens, lps = "chunk_text", chunks, [None] * len(chunks)
    return {"prompt": prompts.prompt_id(index), "kind": prompts.prompts[index]["kind"], "text": "".join(text),
            "token_kind": kind, "tokens": tokens, "logprobs": lps, "finish_reason": finish,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": int(usage.get("completion_tokens") or len(tokens)),
            "seconds": round(time.perf_counter() - started, 3), "error": error}


def run_prompts(endpoint: Any, prompts: bench.PromptSet, settings: bench.Settings) -> list[dict[str, Any]]:
    """Every prompt of the set once, one request at a time, in the set's order."""
    outputs = []
    for index in range(len(prompts.prompts)):
        out = generate(endpoint, prompts, settings, index)
        log(f"{out['prompt']}: {out['completion_tokens']} tokens in {out['seconds']} s"
            + (f" ERROR {out['error']}" if out["error"] else ""))
        outputs.append(out)
    return outputs


def token_kind(outputs: Sequence[Mapping[str, Any]]) -> str:
    kinds = {o["token_kind"] for o in outputs}
    return kinds.pop() if len(kinds) == 1 else "mixed"


# --- conditions -------------------------------------------------------------------------------------------------

def collect_conditions(args: argparse.Namespace, endpoint: Any, prompts: bench.PromptSet,
                       settings: bench.Settings, kind: str, hub: Mapping[str, Any]) -> dict[str, Any]:
    """The conditions of this run: what the server reports (GET /v1/models), what the flags say and the settings."""
    unavailable: dict[str, str] = {}
    served: Mapping[str, Any] = {}
    try:
        served = next((e for e in endpoint.models() if e.get("id") == args.model), {})
        if not served:
            raise CorrectnessError(f"the server does not serve {args.model!r}")
    except CorrectnessError:
        raise
    except Exception as exc:
        unavailable["served_model"] = f"GET /v1/models: {type(exc).__name__}"
    patches: dict[str, str] = {}
    for path in args.patch:
        try:
            patches[Path(path).name] = bench.sha256_file(path)
        except OSError as exc:
            raise CorrectnessError(f"cannot read the patch {path}: {exc}") from exc
    extra: dict[str, Any] = {}
    for key, value in args.set:
        try:
            extra[key] = json.loads(value)
        except ValueError:
            extra[key] = value
    cond: dict[str, Any] = {
        "engine": args.engine, "engine_commit": args.engine_commit, "patch_hash": patches,
        "checkpoint": args.checkpoint or served.get("root"), "quantization": args.quantization,
        "max_model_len": args.max_len or served.get("max_model_len"), "served_model": args.model,
        "node": args.node, "chip": args.chip, "rocm": args.rocm, "torch": args.torch, "triton": args.triton,
        "hub": hub.get("hub"), "account": hub.get("account"), "prompt_set": prompts.describe(),
        "max_tokens": settings.max_tokens, "temperature": settings.temperature, "top_p": 1.0, "seed": 0,
        "ignore_eos": settings.ignore_eos, "thinking": settings.thinking, "nonce": settings.nonce_on,
        "token_kind": kind, "harness": HARNESS, **extra}
    for key in ("engine_commit", "checkpoint", "quantization", "max_model_len", "chip", "rocm", "torch", "triton"):
        if cond[key] is None:
            unavailable[key] = f"not served by /v1/models and not given with --{key.replace('_', '-')}"
    cond["unavailable"] = unavailable
    return cond


def condition_differences(baseline: Mapping[str, Any], run: Mapping[str, Any]) -> dict[str, Any]:
    """What differs between two runs' conditions, split: `differ` (both know it and it is not equal), `unverified`
    (one side does not know it) and `under_test` (the patch hashes, when they differ)."""
    differ: dict[str, Any] = {}
    unverified: list[str] = []
    pairs: list[tuple[str, Any, Any]] = [(k, baseline.get(k), run.get(k)) for k in MUST_MATCH]
    pairs += [("prompt_set.sha256", (baseline.get("prompt_set") or {}).get("sha256"),
               (run.get("prompt_set") or {}).get("sha256"))]
    pairs += [(k, baseline.get(k), run.get(k)) for k in SETTINGS_KEYS]
    known = set(MUST_MATCH) | set(SETTINGS_KEYS) | {"prompt_set", "patch_hash", "node", "hub", "account", "harness",
                                                    "unavailable"}
    pairs += [(k, baseline.get(k), run.get(k)) for k in sorted((set(baseline) | set(run)) - known)]
    for key, was, now in pairs:
        if was is None or now is None:
            if was != now:
                unverified.append(key)
        elif was != now:
            differ[key] = {"baseline": was, "run": now}
    out: dict[str, Any] = {"differ": differ, "unverified": unverified}
    if baseline.get("patch_hash") != run.get("patch_hash"):
        out["under_test"] = {"patch_hash": {"baseline": baseline.get("patch_hash"), "run": run.get("patch_hash")}}
    return out


# --- the baseline fixture ---------------------------------------------------------------------------------------

def baseline_path(directory: Path, engine: str, chip: str, date: str) -> Path:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{engine}-{chip}-{date}")
    return directory / f"{clean}.json"


def write_baseline(path: Path, conditions: Mapping[str, Any], outputs: Sequence[Mapping[str, Any]],
                   recorded_at: str) -> None:
    doc = {"schema": BASELINE_SCHEMA, "harness": HARNESS, "recorded_at": recorded_at, "conditions": conditions,
           "outputs": list(outputs)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def load_baseline(path: str | Path) -> dict[str, Any]:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CorrectnessError(f"cannot read the baseline {path}: {exc}") from exc
    if not isinstance(doc, dict) or doc.get("schema") != BASELINE_SCHEMA:
        raise CorrectnessError(f"{path} is not a {BASELINE_SCHEMA} fixture")
    if not isinstance(doc.get("outputs"), list) or not doc["outputs"] or not isinstance(doc.get("conditions"), dict):
        raise CorrectnessError(f"{path} has no outputs or no conditions")
    return doc


# --- the comparison ---------------------------------------------------------------------------------------------

def first_divergence(a: Sequence[Any], b: Sequence[Any]) -> int | None:
    """The first position where the sequences differ, or where the shorter one ends; None when they are equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare_outputs(baseline: Sequence[Mapping[str, Any]], run: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per prompt of the baseline: token match, first divergence, logprobs there, text matches."""
    by_prompt = {o["prompt"]: o for o in run}
    rows = []
    for base in baseline:
        now = by_prompt.get(base["prompt"])
        if now is None:
            raise CorrectnessError(f"the run has no output for prompt {base['prompt']!r}")
        at = first_divergence(base["tokens"], now["tokens"])
        lp_base = base["logprobs"][at] if at is not None and at < len(base["logprobs"]) else None
        lp_run = now["logprobs"][at] if at is not None and at < len(now["logprobs"]) else None
        rows.append({
            "prompt": base["prompt"], "kind": base.get("kind"), "tokens_match": at is None, "first_divergence": at,
            "baseline_tokens": len(base["tokens"]), "run_tokens": len(now["tokens"]),
            "baseline_token_at": base["tokens"][at] if at is not None and at < len(base["tokens"]) else None,
            "run_token_at": now["tokens"][at] if at is not None and at < len(now["tokens"]) else None,
            "logprob_baseline": lp_base, "logprob_run": lp_run,
            "logprob_difference": None if lp_base is None or lp_run is None else round(lp_run - lp_base, 6),
            "text_equal": base["text"] == now["text"],
            "text_normalised_equal": normalise(base["text"]) == normalise(now["text"]),
            "finish_reason": {"baseline": base.get("finish_reason"), "run": now.get("finish_reason")}})
    return rows


def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    divergent = [r for r in rows if not r["tokens_match"]]
    first = divergent[0] if divergent else None
    return {"prompts": len(rows), "matches": len(rows) - len(divergent), "divergences": len(divergent),
            "match_rate": round((len(rows) - len(divergent)) / len(rows), 6) if rows else None,
            "normalised_text_matches": sum(1 for r in rows if r["text_normalised_equal"]),
            "first_divergence_prompt": first["prompt"] if first else None,
            "first_divergence_index": first["first_divergence"] if first else None}


# --- the command line -------------------------------------------------------------------------------------------

def _pair(text: str) -> tuple[str, str]:
    key, sep, value = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected NAME=VALUE, not {text!r}")
    return key, value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m lab correctness", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    record = sub.add_parser("record", help="send the prompt set to a running server and write the baseline fixture")
    compare = sub.add_parser("compare", help="send it again and compare with a recorded baseline")
    for s in (record, compare):
        where = s.add_mutually_exclusive_group()
        where.add_argument("--url", default=None, help="the server's base URL, reached directly")
        where.add_argument("--account", default=None, help="hub account letter (C, D): reach it through the proxy")
        s.add_argument("--port", type=int, default=8000, help="the pod port, with --account")
        s.add_argument("--model", default="teacher", help="the served model name")
        s.add_argument("--engine", default="vllm")
        s.add_argument("--engine-commit", default=None)
        s.add_argument("--patch", action="append", default=[], help="a patch applied to the engine (repeatable)")
        s.add_argument("--checkpoint", default=None, help="default: the root /v1/models reports")
        s.add_argument("--quantization", default=None)
        s.add_argument("--max-len", type=int, default=None, help="default: the max_model_len /v1/models reports")
        s.add_argument("--node", default="pod-a", help="the ledger's name for the machine")
        s.add_argument("--chip", default=None, help="for example gfx1151")
        s.add_argument("--rocm", default=None)
        s.add_argument("--torch", default=None)
        s.add_argument("--triton", default=None)
        s.add_argument("--set", action="append", type=_pair, default=[], metavar="NAME=VALUE",
                       help="any other condition, such as enforce_eager=true (repeatable)")
        s.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
        s.add_argument("--thinking", choices=("off", "on", "unset"), default="off")
        s.add_argument("--prompts", default=str(bench.DEFAULT_PROMPTS))
        s.add_argument("--request-timeout-s", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
        s.add_argument("--metric-prefix", default="teacher")
    record.add_argument("--out-dir", default=str(DEFAULT_DIR), help="where the fixture goes")
    record.add_argument("--date", default=None, help="the date in the file name (default: today, UTC)")
    record.add_argument("--force", action="store_true", help="overwrite a fixture of the same name")
    compare.add_argument("--baseline", required=True, help="a fixture written by record")
    compare.add_argument("--allow-condition-drift", action="store_true",
                         help="compare although the conditions differ (the row says so)")
    compare.add_argument("--state-dir", default=str(DEFAULT_STATE), help="where the full result is written")
    compare.add_argument("--metrics-file", nargs="?", const=str(metrics_file.DEFAULT_PATH), default=None,
                         help="also append the match rate to the metrics file (default lab/refine/metrics.jsonl)")
    return p


@contextlib.contextmanager
def open_endpoint(args: argparse.Namespace) -> Iterator[tuple[Any, dict[str, Any]]]:
    """The endpoint (lab/bench.py's) and the hub facts for the conditions."""
    if bool(args.url) == bool(args.account):
        raise CorrectnessError("name the server with --url or --account, one of them")
    if args.url:
        endpoint = bench.DirectEndpoint(args.url, timeout_s=args.request_timeout_s)
        try:
            yield endpoint, {"hub": None, "account": None}
        finally:
            endpoint.close()
        return
    from lab import hub as hub_module
    from lab.proxy import PodLink

    letter = args.account.upper()
    account = hub_module.load_accounts([letter])[letter]
    with PodLink.for_account(account) as link:
        yield bench.ProxyEndpoint(link, args.port), {"hub": link.base_url, "account": account.user}


def _setup(args: argparse.Namespace) -> tuple[bench.PromptSet, bench.Settings]:
    try:
        prompts = bench.PromptSet.load(args.prompts)
    except bench.BenchError as exc:
        raise CorrectnessError(str(exc)) from exc
    settings = bench.Settings(model=args.model, max_tokens=args.max_tokens, ignore_eos=True,
                              thinking={"off": False, "on": True, "unset": None}[args.thinking],
                              prefix_caching=False)   # False: no per-request nonce, the prompt is the same every run
    return prompts, settings


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _failed(outputs: Sequence[Mapping[str, Any]]) -> str | None:
    bad = [f"{o['prompt']}: {o['error']}" for o in outputs if o["error"]]
    return "; ".join(bad) if bad else None


def do_record(args: argparse.Namespace) -> dict[str, Any]:
    prompts, settings = _setup(args)
    if not args.chip:
        raise CorrectnessError("record needs --chip (for example gfx1151): it is in the fixture's file name")
    date = args.date or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    path = baseline_path(Path(args.out_dir), args.engine, args.chip, date)
    if path.exists() and not args.force:
        raise CorrectnessError(f"{path} already exists: a baseline is not overwritten without --force")
    with open_endpoint(args) as (endpoint, hub):
        outputs = run_prompts(endpoint, prompts, settings)
        failure = _failed(outputs)
        if failure:
            raise CorrectnessError(f"no baseline written, a prompt failed: {failure}")
        conditions = collect_conditions(args, endpoint, prompts, settings, token_kind(outputs), hub)
    write_baseline(path, conditions, outputs, _now())
    return {"ok": True, "outcome": "ok", "action": "record", "fixture": str(path), "prompts": len(outputs),
            "tokens": [len(o["tokens"]) for o in outputs], "token_kind": conditions["token_kind"],
            "metrics": {f"{args.metric_prefix}.output_prompts": len(outputs)},
            "refs": {"run_id": path.stem}}


def do_compare(args: argparse.Namespace) -> dict[str, Any]:
    prompts, settings = _setup(args)
    baseline = load_baseline(args.baseline)
    with open_endpoint(args) as (endpoint, hub):
        # The conditions are checked before any prompt is sent: a refused compare costs the server nothing.
        probe = collect_conditions(args, endpoint, prompts, settings, baseline["conditions"].get("token_kind"), hub)
        pre = condition_differences(baseline["conditions"], probe)
        if pre["differ"] and not args.allow_condition_drift:
            names = ", ".join(f"{k} (baseline {v['baseline']!r}, run {v['run']!r})" for k, v in pre["differ"].items())
            raise CorrectnessError(f"the baseline's conditions differ from this run's: {names}; nothing was sent. "
                                   "Pass --allow-condition-drift to compare anyway")
        outputs = run_prompts(endpoint, prompts, settings)
    failure = _failed(outputs)
    if failure:
        raise CorrectnessError(f"a prompt failed, nothing compared: {failure}")
    conditions = dict(probe, token_kind=token_kind(outputs))
    diff = condition_differences(baseline["conditions"], conditions)
    if diff["differ"] and not args.allow_condition_drift:    # the token kind came back different
        names = ", ".join(f"{k} (baseline {v['baseline']!r}, run {v['run']!r})" for k, v in diff["differ"].items())
        raise CorrectnessError(f"the baseline's conditions differ from this run's: {names}. "
                               "Pass --allow-condition-drift to compare anyway")
    rows = compare_outputs(baseline["outputs"], outputs)
    summary = summarize(rows)
    run_id = "correctness-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:4]
    result = {"schema": RESULT_SCHEMA, "harness": HARNESS, "run_id": run_id, "at": _now(),
              "baseline": str(args.baseline), "baseline_recorded_at": baseline.get("recorded_at"),
              "conditions": conditions, "condition_drift": diff, "summary": summary, "prompts": rows,
              "outputs": outputs}
    state = Path(args.state_dir)
    state.mkdir(parents=True, exist_ok=True)
    result_path = state / f"{run_id}.json"
    result_path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    prefix = args.metric_prefix
    answer: dict[str, Any] = {
        "action": "compare", "run_id": run_id, "baseline": str(args.baseline), **summary,
        "condition_drift": bool(diff["differ"]), "unverified": diff["unverified"],
        "under_test": diff.get("under_test"), "result_path": str(result_path),
        "diverged": [{"prompt": r["prompt"], "first_divergence": r["first_divergence"],
                      "logprob_difference": r["logprob_difference"]} for r in rows if not r["tokens_match"]],
        "metrics": {f"{prefix}.output_matches": summary["matches"],
                    f"{prefix}.output_divergences": summary["divergences"],
                    f"{prefix}.output_prompts": summary["prompts"],
                    f"{prefix}.output_match_rate": summary["match_rate"]},
        "refs": {"run_id": run_id, "baseline": Path(args.baseline).name,
                 "first_divergence_prompt": summary["first_divergence_prompt"],
                 "first_divergence_index": summary["first_divergence_index"]}}
    if args.metrics_file:
        low = "one compare of one server state, under the 5 runs verdict() needs to call a difference"
        metrics_file.add({
            "metric": f"{prefix}.output_match_rate", "value": summary["match_rate"], "unit": "fraction",
            "node": args.node,
            "conditions": {k: conditions.get(k) for k in ("engine", "engine_commit", "patch_hash", "checkpoint",
                                                          "quantization", "max_model_len", "served_model", "chip",
                                                          "rocm", "torch", "triton", "prompt_set", "max_tokens",
                                                          "temperature", "ignore_eos", "thinking", "token_kind")}
                          | {"batch_size": 1, "baseline": Path(args.baseline).name,
                             "baseline_recorded_at": baseline.get("recorded_at"),
                             "matches": summary["matches"], "divergences": summary["divergences"],
                             "first_divergence_prompt": summary["first_divergence_prompt"],
                             "condition_drift": bool(diff["differ"]), "run_id": run_id,
                             "low_confidence_reason": low},
            "source": f"{HARNESS} {run_id}", "ts": _now(), "low_confidence": True}, path=args.metrics_file)
        answer["metrics_file"] = str(args.metrics_file)
    if summary["divergences"]:
        answer.update({"ok": False, "outcome": "diverged",
                       "error": f"{summary['divergences']} of {summary['prompts']} prompts diverge from the "
                                f"baseline; the first is {summary['first_divergence_prompt']} at token "
                                f"{summary['first_divergence_index']}"})
    else:
        answer.update({"ok": True, "outcome": "ok"})
    return answer


@recorded("correctness")
def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Record greedy outputs from a running server as a baseline, or compare a later run with it."""
    args = build_parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        return do_record(args) if args.cmd == "record" else do_compare(args)
    except CorrectnessError as exc:
        return {"ok": False, "outcome": "refused", "error": str(exc)}


if __name__ == "__main__":
    from lab.__main__ import main as lab_main

    raise SystemExit(lab_main(["correctness", *sys.argv[1:]]))
