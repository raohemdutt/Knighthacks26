"""Writes fixtures/lab/train_profile/rocm/: a synthetic torch profiler trace of three training steps on a GPU, with
totals per step that can be checked by hand, and the numbers a reader must get back (expected.json).

    python fixtures/lab/train_profile/make_rocm_synthetic.py

Per step (kernel time, ms): Gated DeltaNet 2.6 (forward 1.0, backward 1.6), attention 0.74 (0.3, 0.44), MLP 0.5
(0.2, 0.3), LoRA 0.045 (0.015, 0.03; inside the attention range, so never counted there), optimizer 0.3, host to
device copies 0.02; data loading 0.2 ms of wall time. Steps last 10.0, 10.5 and 9.5 ms (mean 10). Kernel names are
generic (a GEMM, an elementwise kernel), so the layer of a kernel can only come from the ranges and the backward
links, never from its name; none is a flash-linear-attention kernel.

Threads: tid 1 runs the step, the forward and the optimizer; tid 2 is the autograd thread. A backward op is linked to
its forward op by an fwdbwd flow (ph s at the forward op, ph f at the backward function), as the torch profiler
writes them.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent / "rocm"
STEP_US = (10000.0, 10500.0, 9500.0)
# kind -> (forward us, backward us)
LAYERS = {"gdn": (1000.0, 1600.0), "attention": (300.0, 440.0), "mlp": (200.0, 300.0)}
LORA = (15.0, 30.0)
OPTIMIZER_US = 300.0
COPY_US = 20.0
COPY_BYTES = 65536
DATA_US = 200.0
MODULES = {"gdn": "model.layers.0.linear_attn", "attention": "model.layers.1.self_attn", "mlp": "model.layers.1.mlp"}
LORA_MODULE = "model.layers.1.self_attn.q_proj.lora_A.default"
# The matmul each layer's forward op runs: [M, K] x [K, N] in bf16.
MM = {"gdn": (8, 64, 192), "attention": (8, 64, 128), "mlp": (8, 64, 256), "lora": (8, 64, 16)}
GIB = 2**30


class Writer:
    def __init__(self) -> None:
        self.events: list[dict] = []
        self.corr = 1000
        self.flow = 1
        self.gpu_t = 0.0

    def x(self, name: str, cat: str, ts: float, dur: float, tid: int, **args) -> dict:
        event = {"ph": "X", "cat": cat, "name": name, "pid": 1, "tid": tid, "ts": ts, "dur": dur}
        if args:
            event["args"] = args
        self.events.append(event)
        return event

    def launch(self, ts: float, tid: int, kernel: str, dur: float, *, copy: bool = False) -> None:
        """A runtime call at ts that starts one kernel (or copy) of dur on the GPU, after any earlier GPU work."""
        self.corr += 1
        self.x("hipMemcpyAsync" if copy else "hipLaunchKernel", "cuda_runtime", ts, 4.0, tid, correlation=self.corr)
        start = max(self.gpu_t, ts + 5.0)
        if copy:
            self.events.append({"ph": "X", "cat": "gpu_memcpy", "name": "Memcpy HtoD (Host -> Device)", "pid": 0,
                                "tid": 7, "ts": start, "dur": dur,
                                "args": {"correlation": self.corr, "bytes": COPY_BYTES}})
        else:
            self.events.append({"ph": "X", "cat": "kernel", "name": kernel, "pid": 0, "tid": 7, "ts": start,
                                "dur": dur, "args": {"correlation": self.corr, "grid": [64, 1, 1],
                                                     "block": [256, 1, 1]}})
        self.gpu_t = start + dur

    def mm_op(self, ts: float, tid: int, kind: str, kernel_us: float, kernel: str) -> dict:
        m, k, n = MM[kind]
        op = self.x("aten::mm", "cpu_op", ts, 20.0, tid, **{"Input Dims": [[m, k], [k, n]],
                                                           "Input type": ["c10::BFloat16", "c10::BFloat16"]})
        self.launch(ts + 2.0, tid, kernel, kernel_us)
        return op

    def flow_pair(self, fwd: dict, bwd: dict) -> None:
        self.events.append({"ph": "s", "cat": "fwdbwd", "name": "fwdbwd", "id": self.flow, "pid": 1,
                            "tid": fwd["tid"], "ts": fwd["ts"]})
        self.events.append({"ph": "f", "cat": "fwdbwd", "name": "fwdbwd", "id": self.flow, "pid": 1,
                            "tid": bwd["tid"], "ts": bwd["ts"], "bp": "e"})
        self.flow += 1


def build() -> dict:
    w = Writer()
    t = 1000.0
    for step, length in enumerate(STEP_US):
        s = t
        w.x(f"ProfilerStep#{step}", "user_annotation", s, length, 1)
        w.x("enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__", "user_annotation", s + 10, DATA_US, 1)
        w.x("aten::to", "cpu_op", s + 215, 10.0, 1)
        w.launch(s + 216, 1, "", COPY_US, copy=True)
        w.gpu_t = max(w.gpu_t, s + 300)
        fwd_ops: dict[str, dict] = {}
        cursor = s + 300
        for kind in ("gdn", "attention", "mlp"):
            w.x(f"atlas::{kind}::{MODULES[kind]}", "user_annotation", cursor, 150.0, 1)
            fwd_ops[kind] = w.mm_op(cursor + 5, 1, kind, LAYERS[kind][0], "Cijk_Alik_Bljk_BBS_BH_MT64x64x32")
            if kind == "attention":
                w.x(f"atlas::lora::{LORA_MODULE}", "user_annotation", cursor + 40, 40.0, 1)
                fwd_ops["lora"] = w.mm_op(cursor + 45, 1, "lora", LORA[0], "Cijk_Alik_Bljk_BBS_BH_MT16x16x16")
            cursor += 200
        w.x("aten::cross_entropy_loss", "cpu_op", cursor, 20.0, 1)
        cursor += 100
        for kind in ("mlp", "lora", "attention", "gdn"):
            bwd_us = LORA[1] if kind == "lora" else LAYERS[kind][1]
            outer = w.x("autograd::engine::evaluate_function: MmBackward0", "cpu_op", cursor, 80.0, 2)
            fn = w.x("MmBackward0", "cpu_op", cursor + 5, 60.0, 2)
            w.flow_pair(fwd_ops[kind], fn)
            # two kernels per backward matmul: grad of the input (a GEMM) and an elementwise scale; outer has no link
            w.x("aten::mm", "cpu_op", cursor + 10, 20.0, 2, **{"Input Dims": [[8, 8], [8, 8]],
                                                              "Input type": ["c10::BFloat16", "c10::BFloat16"]})
            w.launch(cursor + 12, 2, "Cijk_Ailk_Bljk_BBS_BH_MT64x64x32", bwd_us * 0.75)
            w.x("aten::mul", "cpu_op", cursor + 35, 20.0, 2)
            w.launch(cursor + 37, 2, "elementwise_kernel_mul", bwd_us * 0.25)
            del outer
            cursor += 100
        w.x("Optimizer.step#AdamW.step", "user_annotation", cursor, 100.0, 1)
        w.x("aten::_fused_adamw_", "cpu_op", cursor + 5, 50.0, 1)
        w.launch(cursor + 7, 1, "fused_adamw_kernel_impl", OPTIMIZER_US)
        # allocator samples: allocated climbs to 3 GiB in the last step, reserved to 4 GiB
        w.events.append({"ph": "i", "s": "t", "cat": "cpu_instant_event", "name": "[memory]", "pid": 1, "tid": 1,
                         "ts": s + 500, "args": {"Total Allocated": (step + 1) * GIB, "Total Reserved": 4 * GIB,
                                                 "Device Type": 1, "Device Id": 0, "Bytes": 1024}})
        t += length
    return {"schemaVersion": 1, "deviceProperties": [{"id": 0, "name": "Radeon 8060S Graphics",
                                                      "totalGlobalMem": 64 * GIB, "computeMajor": 11,
                                                      "computeMinor": 5, "numSms": 20}],
            "baseTimeNanoseconds": 1760000000000000000, "traceEvents": w.events}


def expected() -> dict:
    steps = len(STEP_US)
    total_ms = sum(STEP_US) / 1000.0
    per_step = {"gdn": sum(LAYERS["gdn"]) / 1000.0, "attention": sum(LAYERS["attention"]) / 1000.0,
                "mlp": sum(LAYERS["mlp"]) / 1000.0, "lora": sum(LORA) / 1000.0, "optimizer": OPTIMIZER_US / 1000.0,
                "copies": COPY_US / 1000.0, "data_loading": DATA_US / 1000.0}
    flops = sum(2 * m * k * n for m, k, n in MM.values()) + 4 * 2 * 8 * 8 * 8
    return {"steps": steps, "step_ms_mean": total_ms / steps, "ms_per_step": per_step,
            "shares": {k: round(v * steps / total_ms, 4) for k, v in per_step.items()},
            "flops_per_step": flops, "peak_allocated_gib": 3.0, "peak_reserved_gib": 4.0, "fla_present": False}


if __name__ == "__main__":
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "trace.json").write_text(json.dumps(build(), indent=1) + "\n", encoding="utf-8")
    (HERE / "expected.json").write_text(json.dumps(expected(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (HERE / "profile.json").write_text(json.dumps({"cpu": False, "date": "2026-10-11", "steps": 3}, indent=2) + "\n",
                                       encoding="utf-8")
    run = {"ok": True, "outcome": "trained", "metrics": {"trainer.samples_per_s": 1.6, "trainer.steps": 3,
                                                        "trainer.batch": 4, "trainer.grad_accum": 4,
                                                        "trainer.lora_rank": 16},
           "refs": {"run_id": "synthetic-rocm", "deltanet_kernel": "torch fallback", "device": "Radeon 8060S Graphics "
                    "(gfx1151)", "torch": "2.9.1+rocm7.13.0", "rocm": "7.13.0"}}
    (HERE / "run.json").write_text(json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {HERE}")
