"""The Quark config keys of amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16 (an August 2026 Quark export) and what vLLM's Quark
loader at b1388b1, the commit the pods build, does with each, before and after the loader hunks of
lab/vllm-gfx1201.diff. Standard library only, so it is testable without vLLM or torch and runs on the box and the pod.

What the export changed. Its key names are the ones b1388b1 reads: nothing was renamed (AUGUST_2026_KEYS gives each
key's disposition). Two things keep it from loading at b1388b1:

1. The mapper. `QuarkConfig.apply_vllm_mapper` hands every list in the config to `WeightsMapper.apply_list`, which
   maps module names. The export carries `algo_config`, a list of dicts (the AWQ scaling layers), and the mapper calls
   `str.startswith` on a dict: AttributeError at model construction, for any model with an hf_to_vllm_mapper
   (Qwen3_5ForConditionalGeneration inherits qwen3_vl's). The patch maps a list only when every item is a string and
   carries any other list unchanged (apply_vllm_mapper below, patched=True).
2. The scheme. With activations unquantized (`input_tensors: null`) and int4 weights, none of b1388b1's Quark schemes
   matches and get_scheme raises "No quark compatible scheme was found". The patch backports QuarkW4A16Int4 from vLLM
   PR #48606: it parses group_size and symmetric (parse_w4a16_int4_weight_config, copied verbatim), turns Quark's
   packed nibbles into the AWQ layout (canonicalize_word mirrors canonicalize_quark_packed_int4 one int32 at a time)
   and hands the layer to choose_mp_linear_kernel, which on ROCm at b1388b1 is [ConchLinearKernel,
   ExllamaLinearKernel]: the same selection the RedHat INT4 checkpoint goes through.

audit(quant_config) is the pre-flight lab/quark_compare.py runs on the checkpoint's config.json before it takes a
slot: every key known, none dropped that carries a setting, a scheme found, and the zero-point layout named.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

__all__ = [
    "AUGUST_2026_KEYS", "CHECKPOINT_REPO", "CHECKPOINT_REVISION", "CHECKPOINT_TENSORS", "CONFIG_BLOB",
    "LOADER_FILES", "SAMPLE_WORDS", "PrefixMapper", "apply_vllm_mapper", "audit", "awq_unpack", "canonicalize_word",
    "int4_to_uint4b8", "is_w4a16_int4", "key_paths", "parse_w4a16_int4_weight_config", "patch_files",
    "scheme_for", "scheme_param_shapes", "split_patch",
]

CHECKPOINT_REPO = "amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16"
CHECKPOINT_REVISION = "0f7ee2559e8dbc25879e1fe1677b2b10708b91a9"
#: The git blob id the hub lists for that revision's config.json (fixtures/kernels/quark_config_august_2026.json).
CONFIG_BLOB = "6507a23b09df4b7e5191acae391dac61be11fb65"

#: The files the loader hunks of lab/vllm-gfx1201.diff change, all Python.
LOADER_FILES = (
    "vllm/model_executor/layers/quantization/awq_marlin.py",
    "vllm/model_executor/layers/quantization/quark/quark.py",
    "vllm/model_executor/layers/quantization/quark/schemes/__init__.py",
    "vllm/model_executor/layers/quantization/quark/schemes/quark_w4a16_int4.py",
    "vllm/model_executor/layers/quantization/quark/utils.py",
)

READ, IGNORED, CARRIED = "read", "ignored", "carried"

#: Every key path of the export's quantization_config: (what b1388b1 does with it, the key it reads it under).
#: "read": quark.py or a scheme reads it under the same name. "ignored": nothing reads it. "carried": nothing reads
#: it, and only the patched mapper lets it through (the legacy one raises on it).
AUGUST_2026_KEYS: dict[str, tuple[str, str | None]] = {
    "quant_method": (READ, "quant_method"),
    "exclude": (READ, "exclude"),
    "export": (READ, "export"),
    "export.kv_cache_group": (READ, "export.kv_cache_group"),
    "export.pack_method": (READ, "export.pack_method"),
    "export.min_kv_scale": (IGNORED, None),
    "export.weight_format": (IGNORED, None),
    "export.weight_merge_groups": (IGNORED, None),
    "global_quant_config": (READ, "global_quant_config"),
    "global_quant_config.bias": (READ, "global_quant_config.bias"),
    "global_quant_config.input_tensors": (READ, "global_quant_config.input_tensors"),
    "global_quant_config.output_tensors": (READ, "global_quant_config.output_tensors"),
    "global_quant_config.target_device": (IGNORED, None),
    "global_quant_config.weight": (READ, "global_quant_config.weight"),
    "global_quant_config.weight.block_size": (IGNORED, None),
    "global_quant_config.weight.ch_axis": (READ, "global_quant_config.weight.ch_axis"),
    "global_quant_config.weight.dtype": (READ, "global_quant_config.weight.dtype"),
    "global_quant_config.weight.enable_buffer_reuse": (IGNORED, None),
    "global_quant_config.weight.group_size": (READ, "global_quant_config.weight.group_size"),
    "global_quant_config.weight.is_dynamic": (READ, "global_quant_config.weight.is_dynamic"),
    "global_quant_config.weight.is_scale_quant": (IGNORED, None),
    "global_quant_config.weight.max_input_numel": (IGNORED, None),
    "global_quant_config.weight.mx_element_dtype": (IGNORED, None),
    "global_quant_config.weight.observer_cls": (IGNORED, None),
    "global_quant_config.weight.qscheme": (READ, "global_quant_config.weight.qscheme"),
    "global_quant_config.weight.round_method": (IGNORED, None),
    "global_quant_config.weight.scale_calculation_mode": (IGNORED, None),
    "global_quant_config.weight.scale_format": (READ, "global_quant_config.weight.scale_format"),
    "global_quant_config.weight.scale_type": (IGNORED, None),
    "global_quant_config.weight.symmetric": (READ, "global_quant_config.weight.symmetric"),
    "layer_quant_config": (READ, "layer_quant_config"),
    "layer_type_quant_config": (READ, "layer_type_quant_config"),
    "algo_config": (CARRIED, None),
    "kv_cache_quant_config": (IGNORED, None),
    "kv_cache_post_rope": (IGNORED, None),
    "softmax_quant_spec": (IGNORED, None),
    "quant_mode": (IGNORED, None),
    "version": (IGNORED, None),
}

#: Keys whose value is a record of how the export was made, not a setting the loader would have to honour.
_CALIBRATION_FACTS = {
    "quant_mode", "version", "export.weight_format", "global_quant_config.weight.round_method",
    "global_quant_config.weight.observer_cls", "global_quant_config.weight.scale_type",
    "global_quant_config.weight.max_input_numel", "global_quant_config.weight.enable_buffer_reuse",
}
#: Values that hold no setting: an ignored key with one of these changes nothing when it is dropped.
_EMPTY = (None, False, 0, 0.0, "", [], {})
#: The dicts whose own keys are paths too (the rest are values).
_NESTED = ("export", "global_quant_config", "global_quant_config.weight")

#: Tensors of the checkpoint (safetensors header at CHECKPOINT_REVISION, layer 3 and layer 0): shapes as stored.
CHECKPOINT_TENSORS: dict[str, dict[str, Any]] = {
    "layers.3.self_attn.k_proj": {"in_features": 5120, "out_features": 1024, "weight": (5120, 128),
                                  "weight_scale": (40, 1024), "weight_zero_point": (40, 128)},
    "layers.3.self_attn.o_proj": {"in_features": 6144, "out_features": 5120, "weight": (6144, 640),
                                  "weight_scale": (48, 5120), "weight_zero_point": (48, 640)},
    "layers.3.mlp.gate_proj": {"in_features": 5120, "out_features": 17408, "weight": (5120, 2176),
                               "weight_scale": (40, 17408), "weight_zero_point": (40, 2176)},
    "layers.3.mlp.down_proj": {"in_features": 17408, "out_features": 5120, "weight": (17408, 640),
                               "weight_scale": (136, 5120), "weight_zero_point": (136, 640)},
    "layers.0.linear_attn.in_proj_a": {"in_features": 5120, "out_features": 48, "weight": (5120, 6),
                                       "weight_scale": (40, 48), "weight_zero_point": (40, 6)},
}
#: The first eight int32 words of layers.3.self_attn.k_proj.weight (read by byte range from the hub).
SAMPLE_WORDS = (19017715, -868163539, 248380717, 253689884, 2031837, -736027840, -301801428, -317672974)

#: vLLM's AWQ nibble order (awq_marlin._REVERSE_AWQ_PACK_ORDER at b1388b1): logical value i sits in slot ORDER[i].
_REVERSE_AWQ_PACK_ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


# --- the loader logic, copied out of the patch ------------------------------------------------------------------

def parse_w4a16_int4_weight_config(weight_config: Mapping[str, Any]) -> tuple[int, bool]:
    """Parse required W4A16 INT4/UINT4 weight fields from Quark config."""
    if "group_size" not in weight_config:
        raise ValueError(
            "Quark W4A16 INT4/UINT4 configs must specify weight.group_size"
        )
    if "symmetric" not in weight_config:
        raise ValueError("Quark W4A16 INT4/UINT4 configs must specify weight.symmetric")

    group_size = weight_config["group_size"]
    is_symmetric = weight_config["symmetric"]
    if not isinstance(group_size, int) or group_size <= 0:
        raise ValueError(
            f"Quark W4A16 weight.group_size must be a positive int, got {group_size!r}"
        )
    if not isinstance(is_symmetric, bool):
        raise ValueError(
            f"Quark W4A16 weight.symmetric must be a bool, got {is_symmetric!r}"
        )
    return group_size, is_symmetric


def is_w4a16_int4(weight_quant: Mapping[str, Any] | None, input_quant: Mapping[str, Any] | None,
                  pack_method: str | None) -> bool:
    """QuarkConfig._is_w4a16_int4 of the patch: INT4/UINT4 weight-only (W4A16) with packed int32 weights."""
    if weight_quant is None or input_quant is not None:
        return False
    is_int4 = weight_quant.get("dtype") in ("int4", "uint4")
    is_packed = pack_method in ("order", "reorder")
    if not (is_int4 and is_packed):
        return False
    parse_w4a16_int4_weight_config(weight_quant)
    return True


class PrefixMapper:
    """WeightsMapper(orig_to_new_prefix=...) at b1388b1: _map_name, apply_list and apply_dict, prefixes only."""

    def __init__(self, prefixes: Mapping[str, str | None]) -> None:
        self.prefixes = dict(prefixes)

    def map_name(self, key: Any) -> Any:
        for prefix, new in self.prefixes.items():
            if key.startswith(prefix):          # a dict here is the b1388b1 crash
                if new is None:
                    return None
                key = key.replace(prefix, new, 1)
        return key

    def apply_list(self, values: list[Any]) -> list[Any]:
        return [out for name in values if (out := self.map_name(name)) is not None]

    def apply_dict(self, values: Mapping[str, Any]) -> dict[str, Any]:
        return {out: value for name, value in values.items() if (out := self.map_name(name)) is not None}


def apply_vllm_mapper(quant_config: Mapping[str, Any], mapper: PrefixMapper, *, patched: bool = True) -> dict:
    """QuarkConfig.apply_vllm_mapper, as the patch has it (patched) or as b1388b1 has it (not patched)."""
    out: dict[str, Any] = {}
    for k, v in quant_config.items():
        if isinstance(v, list) and (not patched or all(isinstance(item, str) for item in v)):
            out[k] = mapper.apply_list(v)
        elif isinstance(v, list):
            out[k] = v
        elif isinstance(v, dict):
            out[k] = mapper.apply_dict(v)
        elif isinstance(v, str):
            mapped = mapper.apply_list([v])
            if mapped:
                out[k] = mapped[0]
        else:
            out[k] = v
    return out


def int4_to_uint4b8(nibble: int) -> int:
    """A symmetric Quark nibble (two's complement int4) as uint4b8: the xor with 8 the patch applies."""
    return (nibble & 0xF) ^ 0x8


def canonicalize_word(word: int, *, pack_reorder: bool, is_symmetric: bool, pack_factor: int = 8) -> int:
    """canonicalize_quark_packed_int4 of the patch, for one int32 word: the nibbles moved into vLLM's AWQ order (a
    "reorder" export is in that order already) and, when symmetric, turned into uint4b8."""
    source = _REVERSE_AWQ_PACK_ORDER if pack_reorder else tuple(range(pack_factor))
    values = [((word & 0xFFFFFFFF) >> (4 * slot)) & 0xF for slot in source]
    if is_symmetric:
        values = [v ^ 0x8 for v in values]
    packed = 0
    for i, v in enumerate(values):
        packed |= v << (4 * _REVERSE_AWQ_PACK_ORDER[i])
    return packed


def awq_unpack(word: int) -> list[int]:
    """The eight values of a word in logical order, as _convert_awq_to_standard_format unpacks AWQ order."""
    return [(word >> (4 * _REVERSE_AWQ_PACK_ORDER[i])) & 0xF for i in range(8)]


def scheme_param_shapes(in_features: int, out_features: int, group_size: int,
                        pack_factor: int = 8) -> dict[str, tuple[int, int]]:
    """The shapes QuarkW4A16Int4.create_weights registers for one unsharded linear layer (TP 1)."""
    packed_out = -(-out_features // pack_factor)
    groups = in_features // (group_size if group_size != -1 else in_features)
    return {"weight": (in_features, packed_out), "weight_scale": (groups, packed_out * pack_factor),
            "weight_zero_point": (groups, packed_out)}


def scheme_for(quant_config: Mapping[str, Any], *, patched: bool = True) -> str | None:
    """The Quark scheme the global config selects for a linear layer: None when b1388b1 would raise "No quark
    compatible scheme was found". Only the weight-only int4 branch is modelled; anything else is None."""
    glob = quant_config.get("global_quant_config") or {}
    pack = (quant_config.get("export") or {}).get("pack_method")
    if patched and is_w4a16_int4(glob.get("weight"), glob.get("input_tensors"), pack):
        return "QuarkW4A16Int4"
    return None


# --- the audit --------------------------------------------------------------------------------------------------

def key_paths(quant_config: Mapping[str, Any]) -> set[str]:
    """Every key path of a quantization_config, as AUGUST_2026_KEYS spells them."""
    paths: set[str] = set()

    def walk(node: Mapping[str, Any], prefix: str) -> None:
        for key, value in node.items():
            path = f"{prefix}{key}"
            paths.add(path)
            if path in _NESTED and isinstance(value, Mapping):
                walk(value, path + ".")
    walk(quant_config, "")
    return paths


def _value(quant_config: Mapping[str, Any], path: str) -> Any:
    node: Any = quant_config
    for part in path.split("."):
        node = node.get(part) if isinstance(node, Mapping) else None
    return node


def audit(quant_config: Mapping[str, Any]) -> dict[str, Any]:
    """Each key's disposition (read, carried, inert, dropped) and whether the checkpoint loads with the patch.

    dropped is an ignored key that carries a setting (kv_cache_quant_config with an entry, a softmax spec): the
    loader would load without it and say nothing, so it blocks the run. A key AUGUST_2026_KEYS does not know is
    listed under unknown and blocks it too."""
    keys: dict[str, dict[str, Any]] = {}
    unknown: list[str] = []
    blocking: list[str] = []
    for path in sorted(key_paths(quant_config)):
        known = AUGUST_2026_KEYS.get(path)
        if known is None:
            unknown.append(path)
            continue
        kind, legacy = known
        value = _value(quant_config, path)
        if kind == READ:
            entry = {"disposition": "read", "legacy_key": legacy}
        elif kind == CARRIED:
            entry = {"disposition": "carried", "why": "a list of dicts (the AWQ scaling layers): the patched mapper "
                     "passes it through; the b1388b1 mapper raises AttributeError on it"}
        elif value in _EMPTY or path in _CALIBRATION_FACTS:
            entry = {"disposition": "inert", "why": "not read at b1388b1; " + (
                "a record of how the export was made" if path in _CALIBRATION_FACTS else "empty here")}
        else:
            entry = {"disposition": "dropped", "why": "not read at b1388b1, and it holds a setting the loader would "
                     "silently drop"}
            blocking.append(path)
        keys[path] = entry
    scheme = scheme_for(quant_config)
    if scheme is None:
        blocking.append("global_quant_config.weight (no Quark scheme matches)")
    weight = _value(quant_config, "global_quant_config.weight") or {}
    zero_point = None
    if scheme:
        group, symmetric = parse_w4a16_int4_weight_config(weight)
        zero_point = (f"none (symmetric {weight.get('dtype')}, group {group}); zero-point tensors are stored and "
                      "ignored" if symmetric else f"asymmetric {weight.get('dtype')}, group {group}: packed int32 "
                      "zero points are loaded")
    return {"ok": not unknown and not blocking, "keys": keys, "unknown": unknown, "blocking": blocking,
            "scheme": scheme, "zero_point": zero_point}


# --- the patch file ---------------------------------------------------------------------------------------------

_DIFF_HEADER = re.compile(r"^diff --git a/(\S+) ", re.M)


def patch_files(diff_text: str) -> list[str]:
    """The loader files lab/vllm-gfx1201.diff changes, sorted."""
    return sorted(f for f in _DIFF_HEADER.findall(diff_text) if f in LOADER_FILES)


def split_patch(diff_text: str) -> tuple[str, str]:
    """(the build-enabling part, the loader part): the diff cut at the first loader file's header. The loader part
    is itself a patch that git apply takes, to put on or take off a tree that already has the build part."""
    for match in _DIFF_HEADER.finditer(diff_text):
        if match.group(1) in LOADER_FILES:
            return diff_text[:match.start()], diff_text[match.start():]
    return diff_text, ""
