# Checks vLLM's own gptq_gemm (the kernel the GPTQ path runs, on CUDA and ROCm alike) against the reference
# dequantization of lab/ct_to_gptq.py on real layers of the converted checkpoint. Usage: kcheck.py <gptq dir> <bits>
import json, sys, torch
from safetensors import safe_open
sys.path.insert(0, "/content/Knighthacks26/lab")
import ct_to_gptq as cg
from vllm import _custom_ops as ops
d, bits = sys.argv[1], int(sys.argv[2])
idx = json.load(open(f"{d}/model.safetensors.index.json"))["weight_map"]
names = sorted(k[: -len(".qweight")] for k in idx if k.endswith(".qweight"))
want = ("layers.0.linear_attn.in_proj_qkv", "layers.0.linear_attn.out_proj", "layers.3.self_attn.q_proj",
        "layers.3.self_attn.o_proj", "layers.0.mlp.down_proj", "layers.63.mlp.gate_proj")
pick = [n for n in names if any(n.endswith(w) for w in want)]
print("quantized linears:", len(names), "checked:", len(pick))
worst = 0.0
for n in pick:
    t = {}
    with safe_open(f"{d}/{idx[n + '.qweight']}", "pt") as f:
        for s in ("qweight", "qzeros", "scales", "g_idx"):
            t[s] = f.get_tensor(f"{n}.{s}")
    ref = torch.from_numpy(cg.dequantize_gptq(*(t[s].numpy() for s in ("qweight", "qzeros", "scales", "g_idx")), bits))
    x = torch.randn(16, ref.shape[1], dtype=torch.float16, device="cuda") * 0.1
    qw = t["qweight"].cuda().contiguous()
    g = torch.empty((0,), dtype=torch.int, device="cuda")
    import inspect
    if len(inspect.signature(ops.gptq_shuffle).parameters) == 3:  # vLLM 0.19: (q_weight, g_idx, bit)
        ops.gptq_shuffle(qw, g, bits)
        y = ops.gptq_gemm(x, qw, t["qzeros"].cuda(), t["scales"].cuda(), g, True, False, bits)
    else:  # later vLLM dropped g_idx
        ops.gptq_shuffle(qw, bits)
        y = ops.gptq_gemm(x, qw, t["qzeros"].cuda(), t["scales"].cuda(), True, False, bits)
    yref = x.float() @ ref.cuda().t()
    rel = ((y.float() - yref).norm() / yref.norm()).item()
    worst = max(worst, rel)
    print(f"{n} {tuple(ref.shape)} rel_err={rel:.2e}")
print("WORST", worst)
sys.exit(0 if worst < 5e-3 else 1)
