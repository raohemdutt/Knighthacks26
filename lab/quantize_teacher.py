"""Quantizes the teacher (Qwen/Qwen3.8-27B) to 3-bit or 2-bit weights in compressed-tensors format, on a CUDA GPU.

    python lab/quantize_teacher.py --dry-run                         # print the plan (the committed recipe); no torch
    python lab/quantize_teacher.py --width 3 --calibration calibration.jsonl --out /content/quant
    python lab/quantize_teacher.py --width 2 --calibration calibration.jsonl --out /content/quant --upload

The two widths and their methods (lab/refine/research/kernels/lowbit_teacher.md says why):

  3 bits  GPTQ (llm-compressor GPTQModifier), preset W3A16: symmetric int3, group size 128, static activation order.
  2 bits  AutoRound (llm-compressor AutoRoundModifier), preset W2A16: symmetric int2, group size 128, 200 iterations.

Both keep in bf16 what the serving INT4 teacher keeps in bf16: the vision tower, lm_head, the embeddings and the Gated
DeltaNet in_proj_a and in_proj_b. The calibration set is fixtures/lab/quant/calibration.jsonl (Atlas's own chat
turns, made by fixtures/lab/quant/make_calibration.py); upload it next to this file and pass its path.

This file is self-contained (standard library at import time; torch, transformers, datasets, llmcompressor and
huggingface_hub only inside the run) so the Colab CLI can ship it alone with `colab exec -f`. The GPU stack is
lab/requirements-quantize.txt. The run refuses a ROCm torch and a CPU-only torch: the quantization is meant for the
Colab A100, never a Strix Halo pod or the Jetson.

Output, per width, in <out>/<name>/: the compressed checkpoint, the tokenizer and processor, recipe.yaml, and
atlas_quant.json (the plan, the calibration file's sha256, package versions, timings, free GPU memory and a short
sample answer). With --upload the folder goes to the Hugging Face Hub as a private model repo named
<namespace>/<name>, where the namespace comes from the environment variable QUANT_HUB_NAMESPACE and the token from
HF_TOKEN. The token is read from the environment only, is never printed and is never written to a file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3.8-27B"
GROUP_SIZE = 128
NAMESPACE_ENV = "QUANT_HUB_NAMESPACE"
TOKEN_ENV = "HF_TOKEN"
DEFAULT_CALIBRATION = "fixtures/lab/quant/calibration.jsonl"

# Modules kept in bf16, as in the serving INT4 teacher and llm-compressor's Qwen3.8 W4A16 example.
IGNORE = [
    "re:visual.*",
    "re:model.visual.*",
    r"re:.*lm_head",
    "re:.*embed_tokens$",
    r"re:.*linear_attn\.in_proj_a$",
    r"re:.*linear_attn\.in_proj_b$",
]

WIDTHS = {
    3: {
        "method": "gptq",
        "modifier": "GPTQModifier",
        "scheme": "W3A16",
        "params": {"block_size": 128, "dampening_frac": 0.01, "actorder": "static"},
        "max_seq_length": 4096,
        "pad_to_max": False,
        "estimate_minutes": 75,
    },
    2: {
        "method": "autoround",
        "modifier": "AutoRoundModifier",
        "scheme": "W2A16",
        "params": {"iters": 200, "batch_size": 8},
        "max_seq_length": 2048,
        "pad_to_max": True,
        "estimate_minutes": 180,
    },
}

# Disk and memory the run needs on the Colab machine: the bf16 base (about 54 GB of safetensors in the HF cache),
# one compressed output, and headroom. The model loads to CPU memory and llm-compressor onloads one layer at a time.
NEED_DISK_GB = 90
NEED_GPU_GB = 39

SAMPLE_QUESTION = "Where did I leave my keys?"


class QuantError(RuntimeError):
    """A refusal or a failed precondition; the message says which."""


def checkpoint_name(width: int) -> str:
    w = WIDTHS[width]
    return f"Qwen3.8-27B-{w['scheme']}-G{GROUP_SIZE}-{w['method'].upper()}-atlas"


def recipe(width: int) -> dict:
    """The llm-compressor recipe for one width, as the dict its YAML form holds (JSON is valid YAML)."""
    w = WIDTHS[width]
    args = {"targets": ["Linear"], "scheme": w["scheme"], "ignore": list(IGNORE), **w["params"]}
    return {"quant_stage": {"quant_modifiers": {w["modifier"]: args}}}


def plan(widths: list[int], calibration: str = DEFAULT_CALIBRATION, namespace: str | None = None) -> dict:
    """Everything a run will do, without doing it: what --dry-run prints and what atlas_quant.json records."""
    out = {"base_model": BASE_MODEL, "group_size": GROUP_SIZE, "calibration": calibration, "variants": []}
    for width in widths:
        w = WIDTHS[width]
        name = checkpoint_name(width)
        out["variants"].append({
            "width": width,
            "name": name,
            "method": w["method"],
            "scheme": w["scheme"],
            "symmetric": True,
            "max_seq_length": w["max_seq_length"],
            "pad_to_max": w["pad_to_max"],
            "estimate_minutes": w["estimate_minutes"],
            "hub_repo": f"{namespace}/{name}" if namespace else f"<${NAMESPACE_ENV}>/{name}",
            "recipe": recipe(width),
        })
    out["needs"] = {"gpu": "CUDA (Colab A100)", "gpu_memory_gb": NEED_GPU_GB, "free_disk_gb": NEED_DISK_GB}
    return out


def check_torch(torch) -> None:
    """Refuses a ROCm build and a torch without CUDA: the run belongs on the Colab A100 only."""
    if getattr(getattr(torch, "version", None), "hip", None):
        raise QuantError("refusing a ROCm torch: this run is for the Colab A100, never a Strix Halo pod")
    if not torch.cuda.is_available():
        raise QuantError("refusing a CPU-only torch: no CUDA device is visible")
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_gb < NEED_GPU_GB:
        raise QuantError(f"GPU has {total_gb:.0f} GB, the run needs at least {NEED_GPU_GB} GB")


def check_disk(path: Path, need_gb: float = NEED_DISK_GB) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(path).free / 1e9
    if free_gb < need_gb:
        raise QuantError(f"{path} has {free_gb:.0f} GB free, the run needs {need_gb} GB")


def hub_target(width: int, upload: bool, environ=os.environ) -> str | None:
    """The Hub repo an upload goes to, checked before the hours of quantization rather than after."""
    if not upload:
        return None
    namespace = environ.get(NAMESPACE_ENV, "").strip()
    if not namespace:
        raise QuantError(f"--upload needs the environment variable {NAMESPACE_ENV} (the Hub account or org)")
    if not environ.get(TOKEN_ENV, "").strip():
        raise QuantError(f"--upload needs the environment variable {TOKEN_ENV}")
    return f"{namespace}/{checkpoint_name(width)}"


def read_calibration(path: Path) -> list[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    texts = [r["text"] for r in rows if r.get("text")]
    if not texts:
        raise QuantError(f"{path} holds no calibration text")
    return texts


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def versions() -> dict:
    from importlib import metadata

    found = {}
    for pkg in ("torch", "transformers", "datasets", "llmcompressor", "compressed-tensors", "auto-round", "accelerate"):
        try:
            found[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            found[pkg] = None
    return found


def run(width: int, calibration: Path, out_dir: Path, upload: bool) -> dict:  # pragma: no cover - needs the GPU
    """Quantizes one width and saves (and optionally uploads) it. Imports the GPU stack only here."""
    repo = hub_target(width, upload)
    import torch

    check_torch(torch)
    check_disk(out_dir)
    from datasets import Dataset
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    from llmcompressor import oneshot
    from llmcompressor.utils import dispatch_for_generation, load_context

    w = WIDTHS[width]
    texts = read_calibration(calibration)
    started = time.time()
    with load_context(Qwen3_5ForConditionalGeneration):
        model = Qwen3_5ForConditionalGeneration.from_pretrained(BASE_MODEL, dtype="auto")
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    tokenizer = processor.tokenizer
    loaded = time.time()

    pad = "max_length" if w["pad_to_max"] else False
    enc = tokenizer(texts, add_special_tokens=False, truncation=True, max_length=w["max_seq_length"], padding=pad)
    dataset = Dataset.from_dict({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]})
    if w["method"] == "autoround":
        from llmcompressor.modifiers.autoround import fix_batch_if_needed

        dataset = dataset.map(fix_batch_if_needed)

    oneshot(
        model=model,
        processor=processor,
        recipe=json.dumps(recipe(width)),
        dataset=dataset,
        max_seq_length=w["max_seq_length"],
        num_calibration_samples=len(texts),
        shuffle_calibration_samples=False,
    )
    quantized = time.time()

    dispatch_for_generation(model)
    messages = [{"role": "user", "content": SAMPLE_QUESTION}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    sample = tokenizer.decode(model.generate(**ids, max_new_tokens=48, do_sample=False)[0][ids["input_ids"].shape[1]:],
                              skip_special_tokens=True)

    target = out_dir / checkpoint_name(width)
    model.save_pretrained(target, save_compressed=True)
    processor.save_pretrained(target)
    (target / "recipe.yaml").write_text(json.dumps(recipe(width), indent=2) + "\n", encoding="utf-8")
    manifest = {
        "plan": plan([width], str(calibration), repo.split("/")[0] if repo else None)["variants"][0],
        "base_model": BASE_MODEL,
        "calibration_sha256": sha256(calibration),
        "calibration_rows": len(texts),
        "versions": versions(),
        "gpu": torch.cuda.get_device_name(0),
        "minutes": {"load": round((loaded - started) / 60, 1), "quantize": round((quantized - loaded) / 60, 1)},
        "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
        "sample": {"question": SAMPLE_QUESTION, "answer": sample},
        "hub_repo": repo,
    }
    (target / "atlas_quant.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if repo:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ[TOKEN_ENV].strip())
        api.create_repo(repo, private=True, exist_ok=True, repo_type="model")
        api.upload_folder(folder_path=str(target), repo_id=repo, repo_type="model",
                          commit_message=f"{checkpoint_name(width)} from lab/quantize_teacher.py")
    return manifest


def parse_widths(value: str) -> list[int]:
    widths = sorted({int(v) for v in value.split(",") if v.strip()}, reverse=True)
    bad = [w for w in widths if w not in WIDTHS]
    if bad or not widths:
        raise argparse.ArgumentTypeError(f"widths must be among {sorted(WIDTHS)}, got {value!r}")
    return widths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--width", type=parse_widths, default=parse_widths("3,2"), help="3, 2 or 3,2 (default both)")
    ap.add_argument("--calibration", default=DEFAULT_CALIBRATION, help="the calibration jsonl")
    ap.add_argument("--out", default="quant-out", help="output folder (one subfolder per width)")
    ap.add_argument("--upload", action="store_true", help=f"upload to the Hub ({NAMESPACE_ENV}, {TOKEN_ENV})")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit; imports no GPU package")
    args = ap.parse_args(argv)

    if args.dry_run:
        print(json.dumps(plan(args.width, args.calibration, os.environ.get(NAMESPACE_ENV) or None), indent=2))
        return 0
    try:
        for width in args.width:
            hub_target(width, args.upload)  # fail fast, before any download
        calibration = Path(args.calibration)
        if not calibration.is_file():
            raise QuantError(f"calibration file {calibration} not found: upload it next to this script")
        for width in args.width:
            manifest = run(width, calibration, Path(args.out), args.upload)
            print(json.dumps({k: manifest[k] for k in ("hub_repo", "minutes", "peak_gpu_gb", "sample")}))
    except QuantError as exc:
        print(f"quantize_teacher: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
