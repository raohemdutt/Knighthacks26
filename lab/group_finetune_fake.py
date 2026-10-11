"""Stand-ins for lab/group_finetune.py's outside calls: the trainer, the model servers the gate scores, and a box.

    from lab.group_finetune_fake import FakeBox, FakeServers, FakeTrainer

    FakeTrainer()            lab/train_fake.py, in process, with the same arguments the real trainer gets (it writes
                             the telemetry and run.json and deletes its dataset copy and rows, as the real one does on
                             the pod), then a PEFT-shaped adapter folder (adapter_config.json naming the base, and
                             placeholder weights) for lab/package.py; fail=True answers as a failed run instead
    FakeServers()            no server: the candidate is FakeServers.CANDIDATE, each member's current model
                             FakeServers.current_url(member), the teacher FakeServers.TEACHER; lab/eval_gate_fake.py's
                             FakeClient is keyed by those URLs
    FakeBox(swap_fails=...)  a box's pull, swap and rollback, recorded in calls

The teacher's fake is lab/teacher_fake.py and the packaging toolchain's lab/package_toolchain_fake.py. Nothing here
opens a socket, starts a process or touches a GPU.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any, Iterator

from lab import train_fake

__all__ = ["FakeTrainer", "FakeServers", "FakeBox"]


class FakeTrainer:
    def __init__(self, *, fail: bool = False, steps: int = 3) -> None:
        self.fail, self.steps = fail, steps
        self.calls: list[dict[str, Any]] = []

    def train(self, *, dataset: Path, rows: Path, model: str, out: Path, run_id: str, config: Any) -> dict[str, Any]:
        n_rows = sum(1 for line in Path(rows).read_text(encoding="utf-8").splitlines() if line.strip())
        self.calls.append({"dataset": str(dataset), "rows": n_rows, "model": model, "out": str(out),
                           "run_id": run_id, "config": config.fields(), "argv": config.argv()})
        if self.fail:
            return {"ok": False, "outcome": "failed", "error": "the fake trainer was told to fail"}
        argv = ["--dataset", str(dataset), "--rows", str(rows), "--model", model, "--out", str(out), "--run-id",
                run_id, "--max-steps", str(config.max_steps or self.steps), "--batch", str(config.batch),
                "--grad-accum", str(config.grad_accum), "--step-s", "0"]
        with contextlib.redirect_stdout(io.StringIO()):
            train_fake.main(argv)
        answer = json.loads((Path(out) / "run.json").read_text(encoding="utf-8"))
        adapter = Path(out) / "adapter"
        adapter.mkdir(parents=True, exist_ok=True)
        (adapter / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA", "base_model_name_or_path": model, "r": config.rank, "lora_alpha": config.alpha,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}))
        (adapter / "adapter_model.safetensors").write_bytes(f"fake adapter {run_id} {n_rows}".encode() * 16)
        answer.setdefault("refs", {}).update(lora_config=json.dumps({"r": config.rank, "lora_alpha": config.alpha}))
        return answer


class FakeServers:
    CANDIDATE = "http://candidate.test"
    TEACHER = "http://teacher.test"
    teacher_url = TEACHER

    def __init__(self) -> None:
        self.served: list[str] = []

    @staticmethod
    def current_url(member: str) -> str:
        return f"http://current-{member}.test"

    @contextlib.contextmanager
    def candidate(self, package_dir: Path) -> Iterator[str]:
        self.served.append(Path(package_dir).name)
        yield self.CANDIDATE

    @contextlib.contextmanager
    def current(self, member: Any) -> Iterator[str]:
        self.served.append(f"{member.member}:{member.current_version}")
        yield self.current_url(member.member)


class FakeBox:
    def __init__(self, *, pull_fails: bool = False, swap_fails: bool = False) -> None:
        self.pull_fails, self.swap_fails = pull_fails, swap_fails
        self.calls: list[tuple[str, str]] = []

    def pull(self, version: str) -> bool:
        self.calls.append(("pull", version))
        return not self.pull_fails

    def swap(self, version: str) -> bool:
        self.calls.append(("swap", version))
        return not self.swap_fails

    def rollback(self, version: str) -> bool:
        self.calls.append(("rollback", version))
        return True
