"""A prior for LLM decode speed on a Strix Halo (Radeon 8060S, gfx1151), fitted to published benchmarks.

    python lab/public_prior.py                                    # fit, then the teacher and student table
    python lab/public_prior.py --gb 15.5 --batch 8 --backend vllm # one prediction for 15.5 GB read per token
    python lab/public_prior.py --params-b 4 --bits 16 --backend llama.cpp-rocm
    python lab/public_prior.py --fit                              # the fitted coefficients and fit quality only

The data is lab/refine/research/kernels/data/public-strix-halo-benchmarks.csv: one row per published measurement,
columns empty where the source is silent. The note lab/refine/research/kernels/public_strix_halo_benchmarks.md says
where each row came from and what the fit below can and cannot carry. Every number printed here is a prior, to be
calibrated by our own trials on the pods; it is never a measurement.

The model is a bandwidth-bound decode model fitted by weighted least squares, nothing more:

1. Bytes read per decoded token (GB). Decode at batch 1 streams every active weight once per token, so a row's bytes
   are its stated model file size (llama-bench prints it in GiB) scaled by active over total parameters for a
   mixture of experts; failing a size, active parameters times stated bits per weight over 8; failing those, active
   parameters times the format's nominal bits per weight (NOMINAL_BITS). A row with none of these is not used.
2. Single stream: log(tok/s) = a[backend] - beta * log(GB) - g * (context depth in thousands) + m * [MoE], fitted on
   rows at batch 1 with a decode rate. exp(a[backend]) is that backend's effective bandwidth in GB/s at 1 GB read per
   token; beta near 1 means decode really is bandwidth-bound across the rows. Each row is weighted 1/sqrt(n) where n
   is the number of rows its source contributed, so one large table does not outvote every other source.
3. Batch: aggregate tok/s at batch B is B * single / (1 + k * (B - 1)), where k is fitted (through the origin) from
   rows that report batch or concurrency B > 1 next to a batch-1 row of the same source, model, quantization and
   backend. k = 0 is perfect scaling (pure bandwidth bound); k = 1 is no gain from batching. A backend with no batch
   rows uses the pooled k, and the answer says so.

Fit quality is printed with every answer: rows and sources used, R^2 and RMS error of log(tok/s) (as a typical
factor), and the median absolute error when each source is left out of the fit in turn.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

__all__ = [
    "CSV_PATH",
    "NOMINAL_BITS",
    "TARGETS",
    "Fit",
    "Row",
    "backend_key",
    "bytes_per_token_gb",
    "fit",
    "load_rows",
    "main",
    "predict",
]

CSV_PATH = Path(__file__).resolve().parent / "refine" / "research" / "kernels" / "data" / "public-strix-halo-benchmarks.csv"

GIB = 1.073741824  # GB per GiB

# Nominal bits per weight of common formats, used only when a row states neither a file size nor its bits. The GGUF
# k-quant figures are llama.cpp's own averages for a 7B model (tools/quantize README); a 4-bit group-128 checkpoint
# (AWQ, GPTQ, compressed-tensors) carries a 16-bit scale per 128 weights.
NOMINAL_BITS = {
    "f32": 32.0, "fp32": 32.0, "f16": 16.0, "fp16": 16.0, "bf16": 16.0,
    "fp8": 8.0, "int8": 8.0, "q8_0": 8.5, "q6_k": 6.56, "q5_k_m": 5.69, "q5_k_s": 5.54, "q5_0": 5.5,
    "q4_k_m": 4.85, "q4_k_s": 4.58, "q4_0": 4.5, "q4_1": 5.0, "iq4_xs": 4.25, "iq4_nl": 4.5, "mxfp4": 4.25,
    "q3_k_m": 3.91, "q3_k_s": 3.5, "q2_k": 3.35, "awq-int4": 4.125, "gptq-int4": 4.125, "int4": 4.125,
    "w4a16": 4.125,
}

# What the prior is asked about by default. The teacher's bytes are the byte budget of one decode step of
# RedHatAI/Qwen3.8-27B-INT4 (lab/refine/research/kernels/w4a16_decode_gfx1201.md, section 1.2: 12.60 GB of int4
# layers and scales, 2.54 GB of bf16 lm_head, about 0.3 GB of recurrent state). The student, Qwen/Qwen3.5-4B, is sized
# from its nominal 4B parameters: bf16 as vLLM would load it, Q4_K_M as the Jetson runs it.
TARGETS = {
    "teacher-27b-int4": 15.5,
    "student-4b-bf16": 4.0 * 16.0 / 8.0,
    "student-4b-q4_k_m": 4.0 * NOMINAL_BITS["q4_k_m"] / 8.0,
}
BATCHES = (1, 4, 8, 16)
PREDICT_BACKENDS = ("vllm", "llama.cpp-rocm", "llama.cpp-vulkan")


def _num(text: str | None) -> float | None:
    if text is None:
        return None
    text = text.strip().replace(",", "")
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def backend_key(engine: str, backend: str) -> str:
    """One label per serving path: 'vllm', 'llama.cpp-rocm', 'llama.cpp-vulkan', 'llama.cpp-cpu' or 'other'.

    LM Studio, Ollama and lemonade run llama.cpp underneath, so they fold into it by their GPU backend.
    """
    e, b = engine.strip().lower(), backend.strip().lower()
    if "vllm" in e:
        return "vllm"
    if "sglang" in e:
        return "sglang"
    llama_family = any(name in e for name in ("llama.cpp", "llama-cpp", "llamacpp", "lm studio", "lmstudio", "ollama", "lemonade", "koboldcpp", "llamafile", "jan"))
    if llama_family:
        if "vulkan" in b:
            return "llama.cpp-vulkan"
        if any(word in b for word in ("rocm", "hip")):
            return "llama.cpp-rocm"
        if b.startswith("cpu"):
            return "llama.cpp-cpu"
        return "llama.cpp-unknown"
    return "other"


@dataclass(frozen=True)
class Row:
    source: str
    model: str
    quant: str
    backend: str
    gb: float  # bytes read per decoded token, GB
    batch: int  # requests decoded together (batch or concurrency); 1 when the source states neither
    depth_k: float  # context depth in thousands of tokens at which decode was timed (0 when unstated)
    moe: bool
    tg: float  # decode tok/s, summed over the batch when batch > 1


def bytes_per_token_gb(rec: dict[str, str]) -> float | None:
    """GB of weights one decoded token reads, from the row's own columns; None when they do not say."""
    total = _num(rec.get("params_b"))
    active = _num(rec.get("active_params_b")) or total
    share = (active / total) if (total and active and active <= total) else 1.0
    size = _num(rec.get("model_size_gb"))
    if size:
        if "size_unit=gib" in (rec.get("notes") or "").lower().replace(" ", ""):
            size *= GIB
        return size * share
    bits = _num(rec.get("bits_per_weight"))
    if active and bits:
        return active * bits / 8.0
    nominal = NOMINAL_BITS.get((rec.get("quant_format") or "").strip().lower())
    if active and nominal:
        return active * nominal / 8.0
    return None


def _depth_k(rec: dict[str, str]) -> float:
    notes = (rec.get("notes") or "").lower()
    for token in notes.replace(";", " ").replace(",", " ").split():
        if token.startswith("depth=") or (token.startswith("d") and token[1:].isdigit()):
            value = _num(token.split("=", 1)[1] if "=" in token else token[1:])
            if value is not None:
                return value / 1000.0
    return 0.0


def load_rows(path: Path = CSV_PATH) -> tuple[list[Row], list[dict[str, str]]]:
    """(usable decode rows, every CSV record). A record is usable when it has a decode rate and known bytes."""
    with open(path, newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    rows: list[Row] = []
    for rec in records:
        tg = _num(rec.get("tg_tps"))
        gb = bytes_per_token_gb(rec)
        if not tg or not gb or tg <= 0 or gb <= 0:
            continue
        batch = max(int(_num(rec.get("concurrency")) or 1), 1)
        per_request = batch > 1 and "per-request" in (rec.get("notes") or "").lower()
        rows.append(
            Row(
                source=rec.get("source_url", "").split("#")[0],
                model=rec.get("model", ""),
                quant=(rec.get("quant_format") or "").lower(),
                backend=backend_key(rec.get("engine", ""), rec.get("backend", "")),
                gb=gb,
                batch=batch,
                depth_k=_depth_k(rec),
                moe=(rec.get("arch") or "").strip().lower() == "moe",
                tg=tg * batch if per_request else tg,
            )
        )
    return rows, records


def _solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve a small square system by Gaussian elimination with partial pivoting (ridge-stabilized by the caller)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[pivot] = m[pivot], m[col]
        if abs(m[col][col]) < 1e-12:
            continue
        for r in range(n):
            if r != col:
                f = m[r][col] / m[col][col]
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] if abs(m[i][i]) >= 1e-12 else 0.0 for i in range(n)]


def _wls(x: list[list[float]], y: list[float], w: list[float], ridge: float = 1e-6) -> list[float]:
    k = len(x[0])
    xtx = [[sum(w[i] * x[i][p] * x[i][q] for i in range(len(y))) + (ridge if p == q else 0.0) for q in range(k)] for p in range(k)]
    xty = [sum(w[i] * x[i][p] * y[i] for i in range(len(y))) for p in range(k)]
    return _solve(xtx, xty)


@dataclass
class Fit:
    backends: list[str]
    a: dict[str, float]  # log effective GB/s at 1 GB per token, per backend
    beta: float
    g: float  # per 1k tokens of context depth
    m: float  # MoE offset
    k: dict[str, float]  # batch contention per backend
    k_pooled: float
    k_rows: dict[str, int]
    rows: int
    sources: int
    backend_rows: dict[str, int]
    r2: float
    rms_log: float
    loso_median_abs_pct: float | None

    def single(self, gb: float, backend: str, depth_k: float = 0.0, moe: bool = False) -> float:
        if backend not in self.a:
            raise KeyError(f"no public batch-1 rows for backend {backend!r}; have {sorted(self.a)}")
        return math.exp(self.a[backend] - self.beta * math.log(gb) - self.g * depth_k + (self.m if moe else 0.0))

    def contention(self, backend: str) -> tuple[float, bool]:
        """(k, pooled): the backend's own k when it has batch rows, else the pooled k."""
        if self.k_rows.get(backend):
            return self.k[backend], False
        return self.k_pooled, True

    def as_dict(self) -> dict:
        return {
            "model": "log(tok/s) = a[backend] - beta*log(GB per token) - g*depth_k + m*moe; aggregate(B) = B*single/(1+k*(B-1))",
            "method": "weighted least squares, weight 1/sqrt(rows from the same source)",
            "rows": self.rows,
            "sources": self.sources,
            "rows_per_backend": self.backend_rows,
            "effective_gb_s_at_1gb": {b: round(math.exp(v), 1) for b, v in self.a.items()},
            "beta": round(self.beta, 3),
            "g_per_1k_depth": round(self.g, 4),
            "moe_factor": round(math.exp(self.m), 3),
            "batch_k": {b: round(v, 3) for b, v in self.k.items()},
            "batch_k_rows": self.k_rows,
            "batch_k_pooled": round(self.k_pooled, 3),
            "r2_log": round(self.r2, 3),
            "rms_log": round(self.rms_log, 3),
            "typical_factor_error": round(math.exp(self.rms_log), 2),
            "leave_one_source_out_median_abs_pct": None if self.loso_median_abs_pct is None else round(self.loso_median_abs_pct, 1),
        }


def _single_fit(rows: list[Row], backends: list[str]) -> tuple[dict[str, float], float, float, float]:
    x, y, w = [], [], []
    per_source: dict[str, int] = defaultdict(int)
    for r in rows:
        per_source[r.source] += 1
    for r in rows:
        x.append([1.0 if r.backend == b else 0.0 for b in backends] + [-math.log(r.gb), -r.depth_k, 1.0 if r.moe else 0.0])
        y.append(math.log(r.tg))
        w.append(1.0 / math.sqrt(per_source[r.source]))
    coef = _wls(x, y, w)
    a = {b: coef[i] for i, b in enumerate(backends)}
    return a, coef[len(backends)], coef[len(backends) + 1], coef[len(backends) + 2]


def fit(rows: list[Row], min_backend_rows: int = 1) -> Fit:
    """Fit the single-stream model on batch-1 rows and the batch term on matched pairs."""
    single = [r for r in rows if r.batch == 1]
    counts: dict[str, int] = defaultdict(int)
    for r in single:
        counts[r.backend] += 1
    backends = sorted(b for b, n in counts.items() if n >= min_backend_rows)
    single = [r for r in single if r.backend in backends]
    if not single:
        raise ValueError("no usable batch-1 rows")
    a, beta, g, m = _single_fit(single, backends)

    def predict_log(r: Row, a_: dict[str, float], beta_: float, g_: float, m_: float) -> float:
        return a_[r.backend] - beta_ * math.log(r.gb) - g_ * r.depth_k + (m_ if r.moe else 0.0)

    ys = [math.log(r.tg) for r in single]
    resid = [ys[i] - predict_log(r, a, beta, g, m) for i, r in enumerate(single)]
    mean = sum(ys) / len(ys)
    ss_tot = sum((v - mean) ** 2 for v in ys) or 1.0
    r2 = 1.0 - sum(e * e for e in resid) / ss_tot
    rms = math.sqrt(sum(e * e for e in resid) / len(resid))

    sources = sorted({r.source for r in single})
    errors: list[float] = []
    if len(sources) >= 3:
        for s in sources:
            train = [r for r in single if r.source != s]
            test = [r for r in single if r.source == s]
            train_backends = sorted({r.backend for r in train})
            test = [r for r in test if r.backend in train_backends]
            if not test:
                continue
            a2, b2, g2, m2 = _single_fit(train, train_backends)
            errors.extend(abs(math.exp(predict_log(r, a2, b2, g2, m2)) / r.tg - 1.0) * 100.0 for r in test)

    # Batch term from matched pairs: B/ratio - 1 = k (B - 1), fitted through the origin per backend.
    base: dict[tuple, float] = {}
    for r in single:
        base.setdefault((r.source, r.model, r.quant, r.backend, r.depth_k), r.tg)
    num: dict[str, float] = defaultdict(float)
    den: dict[str, float] = defaultdict(float)
    k_rows: dict[str, int] = defaultdict(int)
    for r in rows:
        if r.batch <= 1:
            continue
        tg1 = base.get((r.source, r.model, r.quant, r.backend, r.depth_k))
        if not tg1:
            continue
        ratio = r.tg / tg1
        lhs, rhs = r.batch / ratio - 1.0, r.batch - 1.0
        num[r.backend] += lhs * rhs
        den[r.backend] += rhs * rhs
        k_rows[r.backend] += 1
    k = {b: min(max(num[b] / den[b], 0.0), 1.0) for b in den if den[b] > 0}
    k_pooled = min(max(sum(num.values()) / sum(den.values()), 0.0), 1.0) if sum(den.values()) > 0 else 1.0

    return Fit(
        backends=backends, a=a, beta=beta, g=g, m=m, k=k, k_pooled=k_pooled, k_rows=dict(k_rows),
        rows=len(single), sources=len(sources), backend_rows={b: counts[b] for b in backends},
        r2=r2, rms_log=rms, loso_median_abs_pct=median(errors) if errors else None,
    )


def predict(f: Fit, gb: float, backend: str, batch: int = 1, depth_k: float = 0.0, moe: bool = False) -> dict:
    """The prior's decode rate for one model size, backend and batch: per request and aggregate tok/s."""
    one = f.single(gb, backend, depth_k, moe)
    k, pooled = f.contention(backend)
    aggregate = batch * one / (1.0 + k * (batch - 1))
    return {
        "backend": backend,
        "gb_per_token": round(gb, 3),
        "batch": batch,
        "per_request_tok_s": round(aggregate / batch, 2),
        "aggregate_tok_s": round(aggregate, 2),
        "batch_k": round(k, 3),
        "batch_k_pooled": pooled,
        "prior": True,
    }


def table(f: Fit) -> list[dict]:
    out = []
    for name, gb in TARGETS.items():
        for backend in PREDICT_BACKENDS:
            if backend not in f.a:
                out.append({"target": name, "backend": backend, "error": "no public batch-1 rows for this backend"})
                continue
            for batch in BATCHES:
                out.append({"target": name, **predict(f, gb, backend, batch)})
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", type=Path, default=CSV_PATH)
    p.add_argument("--gb", type=float, help="GB read per decoded token (overrides --params-b and --bits)")
    p.add_argument("--params-b", type=float, help="active parameters in billions")
    p.add_argument("--bits", type=float, help="bits per weight")
    p.add_argument("--backend", default="vllm", help="vllm, llama.cpp-rocm or llama.cpp-vulkan")
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--depth", type=float, default=0.0, help="context depth in tokens")
    p.add_argument("--moe", action="store_true")
    p.add_argument("--fit", action="store_true", help="print the fit only")
    args = p.parse_args(argv)

    rows, records = load_rows(args.csv)
    f = fit(rows)
    answer: dict = {"ok": True, "csv_rows": len(records), "usable_decode_rows": len(rows), "fit": f.as_dict()}
    if args.fit:
        pass
    elif args.gb or (args.params_b and args.bits):
        gb = args.gb if args.gb else args.params_b * args.bits / 8.0
        try:
            answer["prediction"] = predict(f, gb, args.backend, args.batch, args.depth / 1000.0, args.moe)
        except KeyError as exc:
            answer.update(ok=False, error=str(exc))
    else:
        answer["table"] = table(f)
    answer["note"] = "a prior from public data, to be calibrated by our own trials; not a measurement"
    print(json.dumps(answer))
    return 0 if answer["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
