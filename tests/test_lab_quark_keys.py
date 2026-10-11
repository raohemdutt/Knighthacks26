"""lab/quark_keys.py: the Quark config keys of amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16 against vLLM b1388b1's loader.

Every test reads fixtures/kernels/quark_config_august_2026.json, the checkpoint's own config.json at revision
0f7ee2559e8d (byte for byte: its git blob id is the one the hub lists, 6507a23b09df). No vLLM, torch or network."""
from __future__ import annotations

import copy
import json
import re
import subprocess
from pathlib import Path

import pytest

from lab import quark_keys as qk

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "fixtures" / "kernels" / "quark_config_august_2026.json"
DIFF = REPO / "lab" / "vllm-gfx1201.diff"

#: Qwen3_5ForConditionalGeneration's hf_to_vllm_mapper at b1388b1 (inherited from qwen3_vl.py).
QWEN3_VL_PREFIXES = {"model.visual.": "visual.", "lm_head.": "language_model.lm_head.",
                     "model.language_model.": "language_model.model."}


@pytest.fixture
def config():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture
def quant(config):
    return config["quantization_config"]


def test_the_fixture_is_the_checkpoints_own_config_json():
    blob = subprocess.run(["git", "hash-object", str(FIXTURE)], capture_output=True, text=True, check=True)
    assert blob.stdout.strip() == qk.CONFIG_BLOB


# --- every key of the August 2026 export has a mapping ----------------------------------------------------------

def test_every_august_key_is_mapped(quant):
    """No key of the export is left without a disposition: a new key would fail here before it fails on a pod."""
    found = qk.key_paths(quant)
    assert found == set(qk.AUGUST_2026_KEYS)
    report = qk.audit(quant)
    assert set(report["keys"]) == found
    assert not report["unknown"]


@pytest.mark.parametrize("path, legacy", [
    ("quant_method", "quant_method"),
    ("exclude", "exclude"),
    ("export.pack_method", "export.pack_method"),
    ("export.kv_cache_group", "export.kv_cache_group"),
    ("global_quant_config.weight.dtype", "global_quant_config.weight.dtype"),
    ("global_quant_config.weight.qscheme", "global_quant_config.weight.qscheme"),
    ("global_quant_config.weight.group_size", "global_quant_config.weight.group_size"),
    ("global_quant_config.weight.symmetric", "global_quant_config.weight.symmetric"),
    ("global_quant_config.weight.is_dynamic", "global_quant_config.weight.is_dynamic"),
    ("global_quant_config.weight.ch_axis", "global_quant_config.weight.ch_axis"),
    ("global_quant_config.weight.scale_format", "global_quant_config.weight.scale_format"),
    ("global_quant_config.input_tensors", "global_quant_config.input_tensors"),
    ("global_quant_config.output_tensors", "global_quant_config.output_tensors"),
    ("global_quant_config.bias", "global_quant_config.bias"),
    ("layer_quant_config", "layer_quant_config"),
    ("layer_type_quant_config", "layer_type_quant_config"),
])
def test_the_keys_the_loader_reads_keep_their_names(quant, path, legacy):
    """The loader at b1388b1 reads these under the same name: there is no renamed key to translate."""
    assert qk.AUGUST_2026_KEYS[path] == ("read", legacy)
    assert qk.audit(quant)["keys"][path]["disposition"] == "read"


@pytest.mark.parametrize("path", [
    "kv_cache_quant_config", "kv_cache_post_rope", "softmax_quant_spec", "quant_mode", "version",
    "export.min_kv_scale", "export.weight_format", "export.weight_merge_groups",
    "global_quant_config.target_device", "global_quant_config.weight.round_method",
    "global_quant_config.weight.observer_cls", "global_quant_config.weight.scale_type",
    "global_quant_config.weight.is_scale_quant", "global_quant_config.weight.scale_calculation_mode",
    "global_quant_config.weight.max_input_numel", "global_quant_config.weight.enable_buffer_reuse",
    "global_quant_config.weight.mx_element_dtype", "global_quant_config.weight.block_size",
])
def test_keys_the_loader_does_not_read_are_inert_in_this_export(quant, path):
    """Not read at b1388b1, and empty, null or a calibration-time fact here: dropping them changes nothing."""
    assert qk.AUGUST_2026_KEYS[path][0] == "ignored"
    assert qk.audit(quant)["keys"][path]["disposition"] == "inert"


def test_algo_config_is_carried_through_unmapped(quant):
    assert qk.AUGUST_2026_KEYS["algo_config"] == ("carried", None)
    entry = qk.audit(quant)["keys"]["algo_config"]
    assert entry["disposition"] == "carried"
    assert "list of dicts" in entry["why"]


def test_a_non_empty_key_the_loader_drops_blocks_the_run(quant):
    """kv_cache_quant_config is not read at b1388b1: a checkpoint that quantizes the KV cache through it would load
    with a bf16 cache and no word said, so the audit refuses it."""
    changed = copy.deepcopy(quant)
    changed["kv_cache_quant_config"] = {"*k_proj": {"dtype": "fp8_e4m3"}}
    report = qk.audit(changed)
    assert report["keys"]["kv_cache_quant_config"]["disposition"] == "dropped"
    assert not report["ok"]
    assert "kv_cache_quant_config" in report["blocking"]


def test_an_unknown_key_is_reported(quant):
    changed = copy.deepcopy(quant)
    changed["export"]["new_thing"] = 1
    report = qk.audit(changed)
    assert report["unknown"] == ["export.new_thing"]
    assert not report["ok"]


def test_the_real_config_passes_the_audit(quant):
    report = qk.audit(quant)
    assert report["ok"], report
    assert report["scheme"] == "QuarkW4A16Int4"
    assert report["zero_point"] == "none (symmetric int4, group 128); zero-point tensors are stored and ignored"


# --- the mapper: the crash at b1388b1, the fix ------------------------------------------------------------------

def test_the_legacy_mapper_rejects_the_august_config(quant):
    """The failure lab/README.md names: b1388b1 sends every list to WeightsMapper.apply_list, and algo_config is a
    list of dicts, so the prefix mapping calls str methods on a dict."""
    mapper = qk.PrefixMapper(QWEN3_VL_PREFIXES)
    with pytest.raises(AttributeError, match="startswith"):
        qk.apply_vllm_mapper(quant, mapper, patched=False)


def test_the_patched_mapper_maps_names_and_carries_the_rest(quant):
    mapper = qk.PrefixMapper(QWEN3_VL_PREFIXES)
    mapped = qk.apply_vllm_mapper(quant, mapper)
    assert mapped["algo_config"] == quant["algo_config"]
    assert "visual.blocks.0.attn.qkv" in mapped["exclude"]
    assert "visual.merger.linear_fc2" in mapped["exclude"]
    assert "lm_head" in mapped["exclude"]                      # no trailing dot: the "lm_head." prefix leaves it
    assert "mtp.fc" in mapped["exclude"]
    assert not any(name.startswith("model.visual.") for name in mapped["exclude"])
    assert len(mapped["exclude"]) == len(quant["exclude"])
    assert mapped["global_quant_config"]["weight"]["dtype"] == "int4"
    assert mapped["export"]["pack_method"] == "reorder"


def test_the_mapper_in_quark_keys_is_the_one_in_the_patch():
    """lab/quark_keys.py copies the loader logic out of the diff; this keeps the two from drifting apart."""
    text = DIFF.read_text(encoding="utf-8")
    assert "if isinstance(v, list) and all(isinstance(item, str) for item in v):" in text
    for name in ("parse_w4a16_int4_weight_config", "canonicalize_quark_packed_int4", "_is_w4a16_int4",
                 "class QuarkW4A16Int4"):
        assert name in text
    patched = qk.patch_files(text)
    assert patched == sorted(qk.LOADER_FILES)


def test_the_diff_appends_python_only_and_keeps_the_build_hunks():
    text = DIFF.read_text(encoding="utf-8")
    files = re.findall(r"^diff --git a/(\S+) ", text, flags=re.M)
    build = ["pyproject.toml", "requirements/build.txt", "requirements/rocm-build.txt", "vllm/platforms/__init__.py",
             "vllm/platforms/rocm.py", "vllm/v1/attention/ops/triton_unified_attention.py",
             "csrc/quantization/gptq/compat.cuh"]
    assert files[:7] == build
    assert files[7:] and all(f.endswith(".py") for f in files[7:])
    assert set(files[7:]) == set(qk.LOADER_FILES)
    assert qk.split_patch(text)[0] + qk.split_patch(text)[1] == text


# --- scheme selection and the weight layout ---------------------------------------------------------------------

def test_the_scheme_is_found_only_with_the_patch(quant):
    assert qk.scheme_for(quant, patched=False) is None
    assert qk.scheme_for(quant) == "QuarkW4A16Int4"


@pytest.mark.parametrize("weight, message", [
    ({"dtype": "int4", "symmetric": True}, "group_size"),
    ({"dtype": "int4", "group_size": 128}, "symmetric"),
    ({"dtype": "int4", "group_size": 0, "symmetric": True}, "positive int"),
    ({"dtype": "int4", "group_size": 128, "symmetric": "yes"}, "bool"),
])
def test_parse_refuses_an_incomplete_weight_config(weight, message):
    with pytest.raises(ValueError, match=message):
        qk.parse_w4a16_int4_weight_config(weight)


def test_is_w4a16_int4():
    weight = {"dtype": "int4", "group_size": 128, "symmetric": True}
    assert qk.is_w4a16_int4(weight, None, "reorder")
    assert qk.is_w4a16_int4({**weight, "dtype": "uint4"}, None, "order")
    assert not qk.is_w4a16_int4(weight, {"dtype": "int8"}, "reorder")     # activations quantized: not W4A16
    assert not qk.is_w4a16_int4({**weight, "dtype": "fp4"}, None, "reorder")
    assert not qk.is_w4a16_int4(weight, None, "unpacked")
    assert not qk.is_w4a16_int4(None, None, "reorder")


@pytest.mark.parametrize("name", sorted(qk.CHECKPOINT_TENSORS))
def test_the_scheme_shapes_match_the_checkpoint_tensors(name):
    """The parameters QuarkW4A16Int4.create_weights registers have the shapes the safetensors header gives."""
    facts = qk.CHECKPOINT_TENSORS[name]
    k, n = facts["in_features"], facts["out_features"]
    assert qk.scheme_param_shapes(k, n, 128) == {"weight": facts["weight"], "weight_scale": facts["weight_scale"],
                                                 "weight_zero_point": facts["weight_zero_point"]}


def test_symmetric_nibbles_become_uint4b8():
    """Quark stores symmetric int4 as two's complement nibbles (the k_proj histogram peaks at 0, 1 and 15 and is
    lowest at 8), so the loader's xor with 8 turns -8..7 into the 0..15 that uint4b8 means."""
    for value in range(-8, 8):
        assert qk.int4_to_uint4b8(value & 0xF) == value + 8


def test_reorder_words_keep_their_slots_and_only_flip_the_sign_bit():
    for word in qk.SAMPLE_WORDS:
        out = qk.canonicalize_word(word, pack_reorder=True, is_symmetric=True)
        for slot in range(8):
            assert (out >> (4 * slot)) & 0xF == ((word >> (4 * slot)) & 0xF) ^ 0x8


def test_order_words_move_to_the_awq_slots_and_unpack_in_logical_order():
    values = [3, -2, 7, -8, 0, 1, -1, 5]
    word = 0
    for i, v in enumerate(values):                     # "order": logical value i in nibble slot i
        word |= (v & 0xF) << (4 * i)
    out = qk.canonicalize_word(word, pack_reorder=False, is_symmetric=True)
    assert qk.awq_unpack(out) == [v + 8 for v in values]
