"""Repacks a 3-bit or 2-bit compressed-tensors checkpoint (from lab/quantize_teacher.py) into GPTQ format for vLLM.

    python lab/ct_to_gptq.py --dry-run --src <ct checkpoint>           # print the GPTQ config it would write
    python lab/ct_to_gptq.py --src /content/quant/<name> --dst /content/quant/<name>-gptq

Why: stock vLLM 0.19 (the lab's 0.19.2+gfx1151 build) loads compressed-tensors WNA16 weights only at 4 and 8 bits
(WNA16_SUPPORTED_TYPES_MAP), but its GPTQ path takes 2, 3, 4 and 8 bits and runs them through `gptq_gemm`
(csrc/quantization/gptq/q_gemm.cu, the exllama-derived kernel). That kernel is in the common source list, so it is
built for ROCm as well as CUDA. The GPTQ path takes float16 activations only, so serve with `--dtype float16`.

The repack is exact; no weight is re-rounded:

  qweight  compressed-tensors packs the unsigned codes (signed code + 2**(bits-1)) densely along the input dimension,
           element i at bit i*bits, little-endian across int32 words. GPTQ packs the same codes the same dense way
           along the input dimension of the transposed weight. So qweight is weight_packed transposed. pack_dense and
           unpack_dense below are the reference that the tests check against GPTQ's own 3-bit loop.
  qzeros   symmetric quantization has one zero point, 2**(bits-1). GPTQ v1 checkpoints store zero - 1 (the kernel adds
           1 back), packed densely along the output dimension.
  scales   weight_scale transposed to [in/group, out], as float16.
  g_idx    input channel // group_size. The recipe's static activation order keeps groups contiguous, so desc_act is
           false and no permutation is needed.

Every other tensor (the bf16 vision tower, embeddings, lm_head, norms, in_proj_a and in_proj_b) is copied unchanged,
and the config's `dynamic` rules tell vLLM to skip quantization for those modules.

Packing helpers need numpy only; reading and writing safetensors needs torch and safetensors (the Colab VM has both).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PACKED = ".weight_packed"
SCALE = ".weight_scale"
SHAPE = ".weight_shape"
ZERO = ".weight_zero_point"

# Modules kept in bf16 by the recipe, as vLLM names them (it fuses in_proj_a and in_proj_b into in_proj_ba).
SKIP_RULES = {
    r"-:.*visual\..*": {},
    r"-:.*linear_attn\.in_proj_(a|b|ba)$": {},
    r"-:.*lm_head$": {},
    r"-:.*mtp\..*": {},
}


class ConvertError(RuntimeError):
    """The checkpoint is not one this script can repack exactly; the message says why."""


def pack_dense(values, bits: int):
    """Packs unsigned codes [rows, cols] into int32 words [rows, cols*bits/32], element i at bit i*bits."""
    import numpy as np

    v = np.asarray(values, dtype=np.uint64)
    rows, cols = v.shape
    if (cols * bits) % 32:
        raise ConvertError(f"{cols} values of {bits} bits do not fill whole int32 words")
    if v.size and int(v.max()) >= (1 << bits):
        raise ConvertError(f"a code does not fit in {bits} bits")
    words = cols * bits // 32
    out = np.zeros((rows, words), dtype=np.uint64)
    for i in range(cols):
        start = i * bits
        w, off = divmod(start, 32)
        out[:, w] |= (v[:, i] << np.uint64(off)) & np.uint64(0xFFFFFFFF)
        spill = off + bits - 32
        if spill > 0:
            out[:, w + 1] |= v[:, i] >> np.uint64(bits - spill)
    return out.astype(np.uint32).view(np.int32)


def unpack_dense(words, bits: int, cols: int):
    """The inverse of pack_dense: int32 words [rows, n] to unsigned codes [rows, cols] (uint8)."""
    import numpy as np

    w = np.asarray(words).view(np.uint32).astype(np.uint64)
    rows = w.shape[0]
    out = np.zeros((rows, cols), dtype=np.uint64)
    mask = np.uint64((1 << bits) - 1)
    for i in range(cols):
        start = i * bits
        k, off = divmod(start, 32)
        val = w[:, k] >> np.uint64(off)
        spill = off + bits - 32
        if spill > 0:
            val |= w[:, k + 1] << np.uint64(bits - spill)
        out[:, i] = val & mask
    return out.astype(np.uint8)


def convert_layer(weight_packed, weight_scale, shape, bits: int, group_size: int) -> dict:
    """One compressed-tensors Linear to its GPTQ v1 tensors (numpy arrays): qweight, qzeros, scales, g_idx."""
    import numpy as np

    out_f, in_f = (int(s) for s in shape)
    if in_f % group_size or in_f % 32 or (out_f * bits) % 32:
        raise ConvertError(f"shape {out_f}x{in_f} does not pack into GPTQ groups of {group_size} at {bits} bits")
    packed = np.asarray(weight_packed)
    if packed.shape != (out_f, in_f * bits // 32):
        raise ConvertError(f"weight_packed is {packed.shape}, expected {(out_f, in_f * bits // 32)}")
    scale = np.asarray(weight_scale, dtype=np.float32)
    if scale.shape != (out_f, in_f // group_size):
        raise ConvertError(f"weight_scale is {scale.shape}, expected {(out_f, in_f // group_size)}")
    groups = in_f // group_size
    zero_v1 = (1 << (bits - 1)) - 1
    return {
        "qweight": np.ascontiguousarray(packed.T),
        "qzeros": pack_dense(np.full((groups, out_f), zero_v1, dtype=np.uint8), bits),
        "scales": np.ascontiguousarray(scale.T.astype(np.float16)),
        "g_idx": (np.arange(in_f, dtype=np.int32) // group_size).astype(np.int32),
    }


def dequantize_gptq(qweight, qzeros, scales, g_idx, bits: int):
    """Reference dequantization of GPTQ v1 tensors to a float32 [out, in] weight (what gptq_gemm computes)."""
    import numpy as np

    in_f = int(np.asarray(g_idx).shape[0])
    out_f = int(np.asarray(scales).shape[1])
    codes = unpack_dense(np.asarray(qweight).T, bits, in_f).astype(np.float32)  # [out, in]
    zeros = unpack_dense(np.asarray(qzeros), bits, out_f).astype(np.float32) + 1.0  # [groups, out]
    g = np.asarray(g_idx)
    s = np.asarray(scales, dtype=np.float32)
    return (codes - zeros[g].T) * s[g].T


def weight_args(config: dict) -> tuple[int, int]:
    """(bits, group_size) from a compressed-tensors config; refuses anything this repack is not exact for."""
    qc = config.get("quantization_config") or {}
    if qc.get("quant_method") != "compressed-tensors":
        raise ConvertError("config.json has no compressed-tensors quantization_config")
    groups = qc.get("config_groups") or {}
    if len(groups) != 1:
        raise ConvertError(f"expected one config group, found {len(groups)}")
    (group,) = groups.values()
    w = group.get("weights") or {}
    fmt = group.get("format") or qc.get("format")
    if fmt != "pack-quantized":
        raise ConvertError(f"format {fmt!r} is not pack-quantized")
    if w.get("type") != "int" or w.get("strategy") != "group" or not w.get("symmetric"):
        raise ConvertError("only symmetric int group quantization repacks exactly")
    if w.get("actorder") not in (None, "static", False):
        raise ConvertError(f"actorder {w.get('actorder')!r} reorders groups; only static is supported")
    if group.get("input_activations"):
        raise ConvertError("activation quantization is not a W*A16 scheme")
    bits, gs = int(w["num_bits"]), int(w["group_size"])
    if bits not in (2, 3):
        raise ConvertError(f"{bits}-bit weights already load in vLLM's compressed-tensors path; nothing to do")
    return bits, gs


def gptq_config(bits: int, group_size: int) -> dict:
    return {
        "quant_method": "gptq",
        "bits": bits,
        "group_size": group_size,
        "desc_act": False,
        "sym": True,
        "lm_head": False,
        "checkpoint_format": "gptq",
        "dynamic": dict(SKIP_RULES),
        "true_sequential": True,
        "damp_percent": 0.01,
        "converted_from": "compressed-tensors pack-quantized (lab/ct_to_gptq.py)",
    }


def convert_dir(src: Path, dst: Path) -> dict:  # pragma: no cover - needs torch and the real checkpoint
    import numpy as np
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    config = json.loads((src / "config.json").read_text(encoding="utf-8"))
    bits, gs = weight_args(config)
    dst.mkdir(parents=True, exist_ok=True)
    shards = sorted(p.name for p in src.glob("*.safetensors"))
    weight_map: dict[str, str] = {}
    layers = 0
    total = 0
    for shard in shards:
        with safe_open(src / shard, framework="pt") as f:
            keys = list(f.keys())
            tensors = {k: f.get_tensor(k) for k in keys}
        out: dict = {}
        for k in keys:
            if k.endswith(ZERO):
                raise ConvertError(f"{k}: asymmetric checkpoints are not supported")
            if k.endswith(PACKED):
                base = k[: -len(PACKED)]
                shape = tensors[base + SHAPE].tolist()
                conv = convert_layer(tensors[k].numpy(), tensors[base + SCALE].float().numpy(), shape, bits, gs)
                for name, arr in conv.items():
                    out[f"{base}.{name}"] = torch.from_numpy(np.ascontiguousarray(arr))
                layers += 1
            elif k.endswith(SCALE) or k.endswith(SHAPE):
                continue
            else:
                out[k] = tensors[k].contiguous()
        save_file(out, str(dst / shard), metadata={"format": "pt"})
        for k, v in out.items():
            weight_map[k] = shard
            total += v.numel() * v.element_size()
    (dst / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2) + "\n", encoding="utf-8")
    qcfg = gptq_config(bits, gs)
    config["quantization_config"] = qcfg
    (dst / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (dst / "quantize_config.json").write_text(json.dumps(qcfg, indent=2) + "\n", encoding="utf-8")
    for p in src.iterdir():
        if p.is_file() and p.suffix != ".safetensors" and p.name not in (
                "config.json", "model.safetensors.index.json", "quantize_config.json"):
            shutil.copy2(p, dst / p.name)
    return {"bits": bits, "group_size": gs, "layers": layers, "shards": len(shards), "bytes": total}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="the compressed-tensors checkpoint folder")
    ap.add_argument("--dst", help="the GPTQ output folder (default <src>-gptq)")
    ap.add_argument("--dry-run", action="store_true", help="check the source config and print the GPTQ config")
    args = ap.parse_args(argv)
    src = Path(args.src)
    try:
        config = json.loads((src / "config.json").read_text(encoding="utf-8"))
        bits, gs = weight_args(config)
        if args.dry_run:
            print(json.dumps(gptq_config(bits, gs), indent=2))
            return 0
        dst = Path(args.dst) if args.dst else src.with_name(src.name + "-gptq")
        print(json.dumps(convert_dir(src, dst)))
    except (ConvertError, FileNotFoundError) as exc:
        print(f"ct_to_gptq: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
