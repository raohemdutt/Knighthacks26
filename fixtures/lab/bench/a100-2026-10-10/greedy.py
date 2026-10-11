"""Greedy outputs of the bench prompt set on one vLLM spawn (temperature 0, top_p 1, seed 0, thinking off, no nonce,
max_tokens 256, stopping at EOS), for R21's correctness harness to compare chips."""
import json, subprocess, sys, time, datetime as dt
from pathlib import Path
sys.path.insert(0, "/content/Knighthacks26")
import httpx
from lab.bench import PromptSet, Settings, default_command, fill_command

OUT = Path("/content/r57")
ps = PromptSet.load()
argv = fill_command(default_command("vllm", None), {"checkpoint": "/content/models/Qwen3.8-27B-INT4", "model": "teacher",
                                                  "port": 8100, "max_len": 16384, "batch": 1, "log": "", "port_file": ""})
log = open(OUT / "greedy_serve.log", "w")
proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
base = "http://127.0.0.1:8100"
for _ in range(900):
    try:
        if httpx.get(base + "/v1/models", timeout=5).status_code == 200:
            break
    except Exception:
        pass
    time.sleep(2)
settings = Settings(model="teacher", max_tokens=256, temperature=0.0, ignore_eos=False, thinking=False, prefix_caching=False)
rows = []
for i, p in enumerate(ps.prompts):
    body = ps.request_body(i, settings, None)
    body["stream"] = False
    body.pop("stream_options", None)
    body["logprobs"] = True
    body["top_logprobs"] = 1
    r = httpx.post(base + "/v1/chat/completions", json=body, timeout=600).json()
    ch = r["choices"][0]
    lp = (ch.get("logprobs") or {}).get("content") or []
    rows.append({"prompt": p["id"], "kind": p["kind"], "text": ch["message"]["content"],
                 "finish_reason": ch.get("finish_reason"), "usage": r.get("usage"),
                 "tokens": [t["token"] for t in lp], "logprobs": [round(t["logprob"], 5) for t in lp],
                 "settings": {"temperature": 0.0, "top_p": 1.0, "seed": 0, "max_tokens": 256, "ignore_eos": False,
                              "enable_thinking": False, "nonce": None}})
    print(p["id"], ch.get("finish_reason"), (r.get("usage") or {}).get("completion_tokens"), flush=True)
import vllm, torch
meta = {"prompt_set": ps.describe(), "engine": "vllm", "vllm": vllm.__version__, "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "checkpoint": "RedHatAI/Qwen3.8-27B-INT4", "served_model_name": "teacher", "launch": argv,
        "recorded": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
(OUT / "greedy_outputs.json").write_text(json.dumps({"meta": meta, "outputs": rows}, indent=1))
import os, signal
os.killpg(proc.pid, signal.SIGTERM)
proc.wait(60)
print("greedy done")
