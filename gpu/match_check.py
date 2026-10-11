"""R12: the changed build's greedy outputs against the stock build's, token for token, before any speed counts.

Collect one file per cold spawn (on the pod, against the server just started on it, alone):

    python gpu/match_check.py collect --url http://127.0.0.1:8100 --build stock --spawn stock-1 --pid <server pid> \\
        --out gpu/results/match/stock-1.json [--max-tokens 128] [--model teacher] [--prompts fixtures/lab/bench/prompts.json]

Then compare (anywhere):

    python gpu/match_check.py compare --stock stock-1.json --stock stock-2.json --changed stages-1.json --out match.json

The server for every file runs with --enforce-eager and one request at a time (batch 1); collect reads the server's
own command line from /proc/<pid>/cmdline, records it, and refuses a server without --enforce-eager. Each request is
greedy (temperature 0, top_p 1, seed 0) with a fixed max_tokens and ignore_eos, thinking off, no per-request nonce, and
asks for the chosen token's logprob and the top 5 (vLLM's return_tokens_as_token_ids gives token ids, not text).

Match rate: the share of prompts whose generated token ids are identical. For each mismatch the record keeps the first
divergent position, both tokens, and the logprob gap there: the reference build's margin between its own token and the
other build's token when that token is in the reference's top list (`reference_margin`; near 0 is a near tie), else the
difference of the two builds' logprobs for their own tokens (`chosen_logprob_diff`).

The pass rule (R12 Done when): the changed build's match rate against stock is no lower than stock's own match rate
against itself across two cold spawns. The changed build is compared with each stock spawn and the lower rate counts.
Exit 0 pass, 1 fail, 2 when the files cannot be compared (not batch 1, not eager, not greedy, different prompt sets or
settings, or the same stock spawn twice).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

if not __package__:  # run by path on a pod: the repo root, one folder up, holds lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OUTPUTS_SCHEMA = "gpu.r12.outputs/1"
REPORT_SCHEMA = "gpu.r12.match/1"
TOP_LOGPROBS = 5
SAME_SETTINGS = ("model", "max_tokens", "temperature", "top_p", "seed", "ignore_eos", "thinking")


class CompareError(Exception):
    pass


# --- collect ----------------------------------------------------------------------------------------------------

def _token_id(token: str) -> int:
    if not token.startswith("token_id:"):
        raise ValueError(f"expected token_id:<n>, got {token!r} (send return_tokens_as_token_ids)")
    return int(token.split(":", 1)[1])


def parse_choice(response: dict[str, Any]) -> dict[str, Any]:
    """Token ids, the chosen tokens' logprobs and the top logprobs of one non-streamed chat completion."""
    content = response["choices"][0]["logprobs"]["content"]
    return {"token_ids": [_token_id(c["token"]) for c in content],
            "logprobs": [float(c["logprob"]) for c in content],
            "top_logprobs": [{str(_token_id(t["token"])): float(t["logprob"]) for t in c.get("top_logprobs") or []}
                             for c in content]}


def server_argv(pid: int) -> list[str]:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    return [a.decode(errors="replace") for a in raw.split(b"\0") if a]


def collect(args: argparse.Namespace) -> int:
    import httpx

    from lab.bench import PromptSet, Settings
    argv = server_argv(args.pid)
    if "--enforce-eager" not in argv:
        print(json.dumps({"error": f"the server (pid {args.pid}) was not started with --enforce-eager"}))
        return 2
    prompts = PromptSet.load(args.prompts)
    settings = Settings(model=args.model, max_tokens=args.max_tokens, prefix_caching=False)
    outputs = []
    with httpx.Client(base_url=args.url.rstrip("/"), timeout=args.timeout_s) as client:
        for i in range(len(prompts.prompts)):        # one request at a time: batch 1
            body = prompts.request_body(i, settings, None)
            body.update(stream=False, logprobs=True, top_logprobs=TOP_LOGPROBS, return_tokens_as_token_ids=True)
            body.pop("stream_options", None)
            resp = client.post("/v1/chat/completions", json=body)
            resp.raise_for_status()
            outputs.append({"prompt_id": prompts.prompt_id(i), **parse_choice(resp.json())})
    doc = {"schema": OUTPUTS_SCHEMA, "build": args.build, "spawn": args.spawn, "cold": True, "batch": 1,
           "enforce_eager": True, "server_argv": argv, "prompt_set": prompts.describe(),
           "settings": {"model": args.model, "max_tokens": args.max_tokens, "temperature": 0.0, "top_p": 1.0,
                        "seed": 0, "ignore_eos": True, "thinking": False, "top_logprobs": TOP_LOGPROBS},
           "collected_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "outputs": outputs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps({"out": args.out, "prompts": len(outputs),
                      "tokens": sum(len(o["token_ids"]) for o in outputs)}))
    return 0


# --- compare ----------------------------------------------------------------------------------------------------

def validate(docs: list[dict[str, Any]]) -> None:
    for d in docs:
        name = d.get("spawn") or "?"
        if d.get("schema") != OUTPUTS_SCHEMA:
            raise CompareError(f"{name}: not a {OUTPUTS_SCHEMA} file")
        if d.get("batch") != 1:
            raise CompareError(f"{name}: batch {d.get('batch')}, not 1")
        if d.get("enforce_eager") is not True or "--enforce-eager" not in (d.get("server_argv") or []):
            raise CompareError(f"{name}: enforce_eager is not set (the server must run with --enforce-eager)")
        if d.get("cold") is not True:
            raise CompareError(f"{name}: not a cold spawn")
        if (d.get("settings") or {}).get("temperature") != 0.0:
            raise CompareError(f"{name}: not greedy")
    first = docs[0]
    for d in docs[1:]:
        if d["prompt_set"].get("sha256") != first["prompt_set"].get("sha256"):
            raise CompareError(f"{d['spawn']}: a different prompt set from {first['spawn']}")
        diff = [k for k in SAME_SETTINGS if d["settings"].get(k) != first["settings"].get(k)]
        if diff:
            raise CompareError(f"{d['spawn']}: settings differ from {first['spawn']}: {diff}")
        if [o["prompt_id"] for o in d["outputs"]] != [o["prompt_id"] for o in first["outputs"]]:
            raise CompareError(f"{d['spawn']}: prompts differ from {first['spawn']}")


def _divergence(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any] | None:
    ta, tb = a["token_ids"], b["token_ids"]
    pos = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
    if pos is None:
        if len(ta) == len(tb):
            return None
        pos = min(len(ta), len(tb))
    ref_tok = ta[pos] if pos < len(ta) else None
    oth_tok = tb[pos] if pos < len(tb) else None
    lp_a = a["logprobs"][pos] if pos < len(a["logprobs"]) else None
    lp_b = b["logprobs"][pos] if pos < len(b["logprobs"]) else None
    top = a["top_logprobs"][pos] if pos < len(a.get("top_logprobs") or []) else {}
    if lp_a is not None and oth_tok is not None and str(oth_tok) in top:
        gap, kind = lp_a - top[str(oth_tok)], "reference_margin"
    elif lp_a is not None and lp_b is not None:
        gap, kind = lp_a - lp_b, "chosen_logprob_diff"
    else:
        gap, kind = None, "unavailable"
    return {"prompt_id": a["prompt_id"], "first_divergence": pos, "reference_token": ref_tok, "other_token": oth_tok,
            "reference_logprob": lp_a, "other_logprob": lp_b,
            "logprob_gap": round(gap, 6) if gap is not None else None, "logprob_gap_kind": kind,
            "reference_tokens": len(ta), "other_tokens": len(tb)}


def compare_pair(reference: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    mismatches = []
    same_tokens = total_tokens = 0
    for a, b in zip(reference["outputs"], other["outputs"]):
        d = _divergence(a, b)
        total_tokens += len(a["token_ids"])
        same_tokens += len(a["token_ids"]) if d is None else d["first_divergence"]
        if d is not None:
            mismatches.append(d)
    n = len(reference["outputs"])
    return {"reference": reference["spawn"], "other": other["spawn"], "prompts": n, "matched": n - len(mismatches),
            "match_rate": round((n - len(mismatches)) / n, 6) if n else None,
            "tokens_before_divergence_rate": round(same_tokens / total_tokens, 6) if total_tokens else None,
            "mismatches": mismatches}


def report(stock: list[dict[str, Any]], changed: dict[str, Any]) -> dict[str, Any]:
    if len(stock) != 2:
        raise CompareError("the rule needs stock's own rate across two cold spawns: pass --stock twice")
    if stock[0]["spawn"] == stock[1]["spawn"]:
        raise CompareError("the two stock files are the same spawn; the rule needs two cold spawns")
    validate([*stock, changed])
    stock_self = compare_pair(stock[0], stock[1])
    vs = [compare_pair(s, changed) for s in stock]
    worst = min(vs, key=lambda p: p["match_rate"])
    passed = worst["match_rate"] >= stock_self["match_rate"]
    return {"schema": REPORT_SCHEMA, "changed_build": changed["build"], "changed_spawn": changed["spawn"],
            "prompt_set": changed["prompt_set"], "settings": changed["settings"],
            "stock_self": stock_self, "changed_vs_stock": worst, "changed_vs_each_stock": vs,
            "rule": "changed_vs_stock.match_rate (the lower against either stock spawn) >= stock_self.match_rate",
            "pass": passed}


def compare(args: argparse.Namespace) -> int:
    try:
        stock = [json.loads(Path(p).read_text()) for p in args.stock]
        changed = json.loads(Path(args.changed).read_text())
        rep = report(stock, changed)
    except (OSError, ValueError, KeyError, CompareError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=1) + "\n")
    for m in rep["changed_vs_stock"]["mismatches"]:
        print(f"mismatch {m['prompt_id']}: first divergence at token {m['first_divergence']}, "
              f"{m['reference_token']} vs {m['other_token']}, logprob gap {m['logprob_gap']} ({m['logprob_gap_kind']})")
    print(json.dumps({"stock_self": rep["stock_self"]["match_rate"],
                      "changed_vs_stock": rep["changed_vs_stock"]["match_rate"], "pass": rep["pass"]}))
    return 0 if rep["pass"] else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--url", required=True)
    c.add_argument("--pid", type=int, required=True, help="the server's pid, for its command line")
    c.add_argument("--build", required=True, help="stock, stages or block")
    c.add_argument("--spawn", required=True, help="a name unique to this cold spawn, e.g. stock-1")
    c.add_argument("--out", required=True)
    c.add_argument("--model", default="teacher")
    c.add_argument("--max-tokens", type=int, default=128)
    c.add_argument("--timeout-s", type=float, default=900.0)
    c.add_argument("--prompts", default=str(Path(__file__).resolve().parents[1] / "fixtures/lab/bench/prompts.json"))
    m = sub.add_parser("compare")
    m.add_argument("--stock", action="append", required=True)
    m.add_argument("--changed", required=True)
    m.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    return collect(args) if args.cmd == "collect" else compare(args)


if __name__ == "__main__":
    sys.exit(main())
