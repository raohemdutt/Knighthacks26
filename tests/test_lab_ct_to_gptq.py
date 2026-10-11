"""Guards on lab/ct_to_gptq.py, the exact repack of 3-bit and 2-bit compressed-tensors weights into GPTQ format.

The real checkpoint is 13 GB on the Colab VM; what is checked here is the arithmetic: the dense packing against
GPTQ's own 3-bit loop and an independent big-integer reference, the zero point the kernel expects, the round trip
from signed codes and scales to the dequantized weight, the configs it refuses, and the skip rules against the
module names vLLM gives the Qwen3.5-family teacher.
"""

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from lab import ct_to_gptq as cg  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "lab" / "ct_to_gptq.py"
STDLIB = set(sys.stdlib_module_names) | {"__future__"}


def gptq_pack_3bit_loop(intweight):
    """AutoGPTQ's QuantLinear.pack for bits=3, on [in, out] codes, packing along the input dimension."""
    intweight = intweight.astype(np.uint32)
    qweight = np.zeros((intweight.shape[0] // 32 * 3, intweight.shape[1]), dtype=np.uint32)
    i = row = 0
    while row < qweight.shape[0]:
        for j in range(i, i + 10):
            qweight[row] |= intweight[j] << (3 * (j - i))
        i += 10
        qweight[row] |= intweight[i] << 30
        row += 1
        qweight[row] |= (intweight[i] >> 2) & 1
        i += 1
        for j in range(i, i + 10):
            qweight[row] |= intweight[j] << (3 * (j - i) + 1)
        i += 10
        qweight[row] |= intweight[i] << 31
        row += 1
        qweight[row] |= (intweight[i] >> 1) & 0x3
        i += 1
        for j in range(i, i + 10):
            qweight[row] |= intweight[j] << (3 * (j - i) + 2)
        i += 10
        row += 1
    return qweight.view(np.int32)


def bigint_pack(codes, bits):
    """Independent reference: element i at bit i*bits of one big integer per row, cut into 32-bit words."""
    rows, cols = codes.shape
    out = np.zeros((rows, cols * bits // 32), dtype=np.uint32)
    for r in range(rows):
        acc = 0
        for i, c in enumerate(codes[r]):
            acc |= int(c) << (i * bits)
        for k in range(out.shape[1]):
            out[r, k] = (acc >> (32 * k)) & 0xFFFFFFFF
    return out.view(np.int32)


@pytest.fixture
def rng():
    return np.random.default_rng(55)


def test_top_level_imports_are_standard_library_only_so_it_ships_alone():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [(node.module or "").split(".")[0]]
        else:
            continue
        assert set(names) <= STDLIB, names


@pytest.mark.parametrize("bits", [2, 3])
def test_dense_packing_matches_an_independent_bigint_reference_and_round_trips(rng, bits):
    codes = rng.integers(0, 1 << bits, size=(5, 128), dtype=np.uint8)
    packed = cg.pack_dense(codes, bits)
    assert packed.dtype == np.int32 and packed.shape == (5, 128 * bits // 32)
    assert np.array_equal(packed, bigint_pack(codes, bits))
    assert np.array_equal(cg.unpack_dense(packed, bits, 128), codes)


def test_dense_3bit_packing_along_the_input_dimension_is_gptqs_own_layout(rng):
    intweight = rng.integers(0, 8, size=(256, 64), dtype=np.uint8)  # [in, out], as GPTQ packs it
    assert np.array_equal(cg.pack_dense(intweight.T, 3).T, gptq_pack_3bit_loop(intweight))


def test_pack_refuses_codes_that_do_not_fit_or_do_not_fill_words():
    with pytest.raises(cg.ConvertError):
        cg.pack_dense(np.full((1, 32), 8, dtype=np.uint8), 3)
    with pytest.raises(cg.ConvertError):
        cg.pack_dense(np.zeros((1, 10), dtype=np.uint8), 3)


@pytest.mark.parametrize("bits", [2, 3])
def test_a_compressed_tensors_layer_repacks_to_gptq_tensors_that_dequantize_to_the_same_weight(rng, bits):
    out_f, in_f, gs = 64, 256, 128
    half = 1 << (bits - 1)
    signed = rng.integers(-half, half, size=(out_f, in_f))
    scale = rng.uniform(0.001, 0.05, size=(out_f, in_f // gs)).astype(np.float32)
    # compressed-tensors: unsigned codes (signed + 2**(bits-1)) packed densely along the input dimension
    weight_packed = bigint_pack((signed + half).astype(np.uint8), bits)

    t = cg.convert_layer(weight_packed, scale, (out_f, in_f), bits, gs)
    assert t["qweight"].shape == (in_f * bits // 32, out_f) and t["qweight"].dtype == np.int32
    assert t["qzeros"].shape == (in_f // gs, out_f * bits // 32) and t["qzeros"].dtype == np.int32
    assert t["scales"].shape == (in_f // gs, out_f) and t["scales"].dtype == np.float16
    assert t["g_idx"].tolist() == [i // gs for i in range(in_f)]
    # GPTQ v1 stores zero - 1; the kernel adds the 1 back
    assert set(cg.unpack_dense(t["qzeros"], bits, out_f).ravel().tolist()) == {half - 1}

    expected = signed * np.repeat(scale, gs, axis=1)
    got = cg.dequantize_gptq(t["qweight"], t["qzeros"], t["scales"], t["g_idx"], bits)
    assert np.allclose(got, expected, rtol=1e-3, atol=1e-6)


def test_convert_layer_refuses_shapes_that_do_not_match():
    with pytest.raises(cg.ConvertError):
        cg.convert_layer(np.zeros((48, 24), np.int32), np.ones((48, 2)), (48, 256), 3, 128)  # 48*3 % 32 != 0
    with pytest.raises(cg.ConvertError):
        cg.convert_layer(np.zeros((64, 23), np.int32), np.ones((64, 2)), (64, 256), 3, 128)


def ct_config(bits=3, symmetric=True, actorder="static", fmt="pack-quantized", inputs=None):
    weights = {"num_bits": bits, "group_size": 128, "type": "int", "strategy": "group", "symmetric": symmetric,
               "actorder": actorder}
    return {"quantization_config": {"quant_method": "compressed-tensors", "format": fmt, "config_groups": {
        "group_0": {"format": fmt, "targets": ["Linear"], "weights": weights, "input_activations": inputs}}}}


def test_weight_args_reads_bits_and_group_and_refuses_what_it_cannot_repack_exactly():
    assert cg.weight_args(ct_config(3)) == (3, 128)
    assert cg.weight_args(ct_config(2)) == (2, 128)
    for bad in (ct_config(4), ct_config(symmetric=False), ct_config(actorder="group"), ct_config(fmt="int-quantized"),
                ct_config(inputs={"num_bits": 8}), {"quantization_config": {"quant_method": "gptq"}}, {}):
        with pytest.raises(cg.ConvertError):
            cg.weight_args(bad)


def skipped(layer: str) -> bool:
    """vLLM's gptq_utils.get_dynamic_override: a matching "-:" rule means the module stays unquantized."""
    return any(re.match(rule[2:], layer) for rule in cg.gptq_config(3, 128)["dynamic"] if rule.startswith("-:"))


@pytest.mark.parametrize("layer", [
    "visual.blocks.0.attn.qkv", "model.visual.blocks.26.mlp.linear_fc2", "visual.merger.linear_fc1",
    "language_model.model.layers.0.linear_attn.in_proj_ba", "model.language_model.layers.4.linear_attn.in_proj_a",
    "model.language_model.layers.4.linear_attn.in_proj_b", "language_model.lm_head", "lm_head", "mtp.layers.0.mlp.gate_up_proj",
])
def test_modules_the_recipe_keeps_in_bf16_are_skipped_by_vllm(layer):
    assert skipped(layer)


@pytest.mark.parametrize("layer", [
    "language_model.model.layers.0.linear_attn.in_proj_qkvz", "language_model.model.layers.0.linear_attn.out_proj",
    "language_model.model.layers.3.self_attn.qkv_proj", "language_model.model.layers.3.self_attn.o_proj",
    "language_model.model.layers.63.mlp.gate_up_proj", "language_model.model.layers.63.mlp.down_proj",
])
def test_quantized_text_layers_are_not_skipped(layer):
    assert not skipped(layer)


def test_gptq_config_is_what_vllm_reads():
    c = cg.gptq_config(3, 128)
    assert (c["quant_method"], c["bits"], c["group_size"], c["desc_act"], c["sym"], c["lm_head"]) == (
        "gptq", 3, 128, False, True, False)
    assert c["checkpoint_format"] == "gptq"  # v1 zero points; "gptq_v2" would make the kernel skip the +1


def test_dry_run_prints_the_gptq_config_and_refuses_a_4bit_source(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(ct_config(3)), encoding="utf-8")
    run = subprocess.run([sys.executable, str(SCRIPT), "--dry-run", "--src", str(tmp_path)],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)["bits"] == 3
    (tmp_path / "config.json").write_text(json.dumps(ct_config(4)), encoding="utf-8")
    run = subprocess.run([sys.executable, str(SCRIPT), "--dry-run", "--src", str(tmp_path)],
                         capture_output=True, text=True, timeout=60)
    assert run.returncode == 2 and "nothing to do" in run.stderr
