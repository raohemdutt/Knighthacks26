# lab/ — the R9700 side (AUP Learning Cloud)

Two pods, one R9700 (gfx1201, 32 GB) each, on https://dub.aupcloud.io. Account A is the **teacher**
(serves Qwen3.8-27B), account B is the **trainer** (LoRA, eval, packaging). Logins and hub API tokens
are on the Notion Credentials page and in your local `.env`; never put them in this repo.

## What the pods are

- ROCm 7.13.0, `torch 2.9.1+rocm7.13.0`, `triton 3.5.1+rocm7.13.0`, `hipcc` on PATH, `python3` only.
- No root, no Docker, no SSH. JupyterLab terminal and the Jupyter REST API only. `pip install --user` works
  (binaries land in `~/.local/bin`; the scripts add it to PATH).
- `/home/jovyan` is a persistent NFS volume (models, venv packages, builds survive a relaunch).
  `/opt/workspace` (the Jupyter root) is ephemeral.
- Billing is per elapsed minute (4 credits/min on the R9700, ~60,000 per account). The server enforces no
  runtime cap; AMD staff are fine with long sessions. Stopping is immediate. A 60-minute idle culler
  stops a server that gets no requests, so keep the model server busy or ping it.
- **Tailscale is blocked** on the pods (every tailscale.com host is connection-reset). GitHub, Hugging Face
  and PyPI are reachable.

## Launching a pod

Web form: resource "Basic GPU Environment" (key `gpu`), accelerator `r9700`, runtime up to 240 (the API has
no cap). API, with the account's hub token:

```
POST https://dub.aupcloud.io/hub/api/users/<account>/server
Authorization: token <hub token>
{"runtime_minutes": 600, "resource_type": "gpu", "gpu_selection": "r9700"}
```

Poll `GET /hub/api/users/<account>` until `servers[""].ready`. Stop with `DELETE .../server`.
The private team repo cannot be cloned at launch (no GitHub app on the hub); upload `lab/` after launch
(JupyterLab upload, or `PUT /user/<account>/api/contents/<path>`).

## Box ↔ lab: the hub proxy

Once `pip install --user jupyter-server-proxy` has run on an account and the pod has been relaunched once,
anything listening on a pod port is reachable from outside:

```
https://dub.aupcloud.io/user/<account>/proxy/<port>/...
Authorization: token <hub token>
```

That is how the Jetson calls the teacher (`/proxy/8000/v1/chat/completions`) and how it will pull packaged
models. The `/proxy/absolute/<port>/` form does not work here.

## Teacher (pod A)

```
bash lab/pod_a_teacher.sh proxy      # once per account, then relaunch the pod
bash lab/pod_a_teacher.sh build      # vLLM from source at commit b1388b1 + lab/vllm-gfx1201.diff, ~1 h
bash lab/pod_a_teacher.sh download   # amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16 (18 GB) into ~/models
bash lab/pod_a_teacher.sh serve      # OpenAI-compatible server on 127.0.0.1:8000, model name "teacher"
```

`lab/vllm-gfx1201.diff` is the set of changes vLLM needs on RDNA4: `num_stages=1` in the Triton unified
attention kernel (64 KB LDS), two platform-detection fixes, and the torch pins removed so the pod's own
torch is used (`use_existing_torch.py`). Alternative checkpoints: `TEACHER=RedHatAI/Qwen3.8-27B-INT4` or
`TEACHER=Qwen/Qwen3.8-27B-FP8` (28.8 GB, tight).

Call it from anywhere:

```
curl https://dub.aupcloud.io/user/<account>/proxy/8000/v1/chat/completions \
  -H "Authorization: token $AUP_HUB_TOKEN_A" -H "Content-Type: application/json" \
  -d '{"model":"teacher","messages":[{"role":"user","content":"Describe this table in one sentence."}]}'
```

Images go inline as `{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,..."}}` content parts.

## Trainer (pod B)

```
bash lab/pod_b_trainer.sh proxy      # once, then relaunch
bash lab/pod_b_trainer.sh deps       # peft, trl, transformers, accelerate, datasets, gguf (torch stays the pod's)
bash lab/pod_b_trainer.sh llama      # llama-quantize / llama-cli / llama-server built for gfx1201
bash lab/pod_b_trainer.sh download   # Qwen/Qwen3.5-4B (the student the Jetson runs)
bash lab/pod_b_trainer.sh smoke      # lab/smoke_lora.py: tiny LoRA SFT, prints loss and samples/s
```

`smoke_lora.py` is the template for the real trainer (L3): bf16, PEFT LoRA on all linear layers, TRL
`SFTTrainer`, forkserver DataLoader workers, no `torch.compile`.
