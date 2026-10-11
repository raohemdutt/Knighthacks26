"""A stand-in OpenAI-compatible server for lab/spot_score.py, and a dry run of the eval gate on the SPOT set. No
network, no model.

    from lab.spot_fake import FakeSpotServer, role_policy
    server = FakeSpotServer(role_policy(split))
    scores = spot_score.score("http://teacher.spot.invalid", examples, instruction=..., transport=server.transport)

    python -m lab.spot_fake dry-run --out <dir>

The server answers what the scorer asks: GET /v1/models lists SERVED_MODEL, POST /v1/chat/completions answers one
choice with the text the policy gives for (host, request body). Every request is kept in requests.

role_policy answers by the host the request went to, so one server plays the three roles of the gate: the teacher
(host teacher.*) replies with each example's label, the candidate (candidate.*) gets about one example in four
wrong and the current model (current.*) about three in four, each mistake one of the kinds the score must catch: a
citation dropped, the numbers written as words, or a reply that is not JSON. Which examples go wrong is fixed by the
sha256 of the example id, so every run gives the same scores.

dry-run writes the set with contexts.spot.dataset (its own code), joins it with spot-score split, and runs the gate
(lab/eval_gate.py run_gate) on it with the registered SPOT client over this server: current twice, candidate and
teacher once, on all 240 held-out questions, grouped by anchor mission. The ledger rows go to <out>/state unless
LAB_STATE_DIR is set. It prints the gate's decision, the three scores and where scores.json is.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

if not __package__:  # run by path: the repo root, one folder up, is what holds lab/
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx

from lab.dataset import Split

__all__ = ["SERVED_MODEL", "CURRENT", "CANDIDATE", "TEACHER", "Policy", "FakeSpotServer", "labels_by_context",
           "mistake", "role_policy", "dry_run", "main"]

SERVED_MODEL = "atlas-spot-fake"
CURRENT, CANDIDATE, TEACHER = ("http://current.spot.invalid", "http://candidate.spot.invalid",
                               "http://teacher.spot.invalid")
WRONG_IN_FOUR = {"current": 3, "candidate": 1, "teacher": 0}
DIGIT_WORDS = {str(d): w for d, w in enumerate("zero one two three four five six seven eight nine".split())}

Policy = Callable[[str, Mapping[str, Any]], str]


class FakeSpotServer:
    """An OpenAI-compatible chat and model route over an httpx.MockTransport (see the module docstring)."""

    def __init__(self, policy: Policy, *, model: str = SERVED_MODEL) -> None:
        self.policy = policy
        self.model = model
        self.requests: list[dict[str, Any]] = []
        self.failing: int | None = None
        self.transport = httpx.MockTransport(self._handle)

    def fail(self, status: int) -> None:
        """Answer the next request with this HTTP status."""
        self.failing = status

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if self.failing is not None:
            status, self.failing = self.failing, None
            return httpx.Response(status, json={"error": "fake failure"})
        if request.method == "GET" and request.url.path == "/v1/models":
            return httpx.Response(200, json={"object": "list", "data": [{"id": self.model, "object": "model"}]})
        if request.method == "POST" and request.url.path == "/v1/chat/completions":
            body = json.loads(request.content)
            self.requests.append({"host": request.url.host, "body": body})
            text = self.policy(request.url.host, body)
            return httpx.Response(200, json={"choices": [{"index": 0, "finish_reason": "stop",
                                                          "message": {"role": "assistant", "content": text}}]})
        return httpx.Response(404, json={"error": "not found"})


def labels_by_context(split: Split) -> dict[str, tuple[str, dict[str, Any]]]:
    """{user message: (example id, its label)} for every example of the split."""
    return {e.context or "": (e.id, json.loads(e.target or "{}")) for e in split.data.examples}


def mistake(label: Mapping[str, Any], kind: int) -> str:
    """The label as a reply with one kind of mistake: 0 a citation dropped, 1 the digits spelled out, 2 not JSON."""
    answer, cited = label["answer"], list(label["citations"])
    if kind == 0 and cited:
        return json.dumps({"answer": answer, "citations": cited[1:]})
    if kind in (0, 1):
        return json.dumps({"answer": "".join(DIGIT_WORDS.get(c, c) for c in answer), "citations": cited})
    return f"Here is what I found: {answer}"


def role_policy(split: Split) -> Policy:
    """The three roles' replies (see the module docstring), decided by the host's first label."""
    labels = labels_by_context(split)

    def policy(host: str, body: Mapping[str, Any]) -> str:
        example_id, label = labels[body["messages"][-1]["content"]]
        digest = hashlib.sha256(example_id.encode("utf-8")).digest()
        if digest[0] % 4 >= WRONG_IN_FOUR[host.split(".")[0]]:
            return json.dumps(label, ensure_ascii=False)
        return mistake(label, digest[1] % 3)
    return policy


def dry_run(out: str | Path) -> dict[str, Any]:
    """Write the SPOT set, join it, and run the gate on it over the fake server (see the module docstring)."""
    from contexts.spot import dataset as spot_dataset
    from contracts.learning import ModelManifest
    from lab import eval_gate, spot_score
    from lab.dataset import load_split

    out = Path(out)
    os.environ.setdefault("LAB_STATE_DIR", str(out / "state"))
    with contextlib.redirect_stdout(sys.stderr):   # its report, kept off the dry run's own answer
        wrote = spot_dataset.main(["--out", str(out / "set")])
    if wrote != 0:
        raise RuntimeError("contexts.spot.dataset wrote no set (no SPOT kit?)")
    joined = spot_score.join(out / "set", out / "set" / spot_score.DATASET_ID)
    gguf = "atlas-spot-dry-run-Q4_K_M.gguf"
    manifest = out / "pkg" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(ModelManifest(version="atlas-spot-dry-run", base="Qwen/Qwen3.5-4B", task="spot-qa", gguf=gguf,
                                      sha256={gguf: "00" * 32},
                                      created=datetime(2026, 10, 11, tzinfo=UTC)).model_dump_json(), encoding="utf-8")
    server = FakeSpotServer(role_policy(load_split(joined["folder"], require_images=False)))
    client = eval_gate.client_for(spot_score.DATASET_TASK, joined["folder"])
    client.transport = server.transport
    client.scorer_commit = lambda: "d0" * 20   # a scratch folder is no commit; the real run uses the checkout's
    result = eval_gate.run_gate(client=client, dataset=joined["folder"], current_url=CURRENT, candidate_url=CANDIDATE,
                                teacher_url=TEACHER, candidate_manifest=manifest, run_id="spot-dry-run",
                                out_dir=out / "gate", current_version="atlas-base", decode_attested=True)
    return {"decision": result["decision"], "reasons": result["reasons"],
            "scores": {k: result["scores"][k] for k in ("current", "candidate", "teacher")},
            "n_items": result["n_items"], "n_groups": result["n_groups"], "p_value": result["p_value"],
            "requests": len(server.requests), "scores_json": str(out / "gate" / eval_gate.SCORES_FILE)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m lab.spot_fake", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("dry-run", help="run the gate on the SPOT set over the fake server")
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(dry_run(args.out), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
