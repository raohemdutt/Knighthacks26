#!/bin/bash
# L16v2 setup on the Colab A100 (a copy of the R57 setup.sh idea): the student's base model at the pinned commit and
# the trainer's pinned stack, installed against the torch the VM already has. Runs detached; SETUP_DONE marks the end.
set -x
OUT=/content/l16v2
mkdir -p $OUT
stamp() { echo "$1 $(date -u +%FT%TZ)" >> $OUT/times.txt; }
stamp setup_start
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > $OUT/versions.txt
python3 --version >> $OUT/versions.txt
python3 - >> $OUT/versions.txt <<'PY'
import importlib.metadata as m
for name in ("torch", "torchvision", "torchaudio", "triton", "transformers", "peft", "trl", "accelerate", "datasets",
             "pydantic", "tokenizers", "pyyaml", "jsonschema", "httpx", "pillow", "huggingface_hub"):
    try:
        print("before", name, m.version(name))
    except m.PackageNotFoundError:
        print("before", name, "missing")
PY
python3 -c "from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3.5-4B', revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a', local_dir='/content/models/Qwen3.5-4B'))" > $OUT/model_download.log 2>&1
echo "model rc $?" >> $OUT/model_download.log
du -sh /content/models/Qwen3.5-4B >> $OUT/versions.txt
stamp model_done
# The trainer's pins (lab/requirements-train.txt), with the VM's own torch held where it is, as lab/train.py install() does.
python3 - > $OUT/constraints.txt <<'PY'
import importlib.metadata as m
for name in ("torch", "torchvision", "torchaudio", "triton"):
    try:
        print(f"{name}=={m.version(name)}")
    except m.PackageNotFoundError:
        pass
PY
pip install -q -r /content/requirements-train.txt -c $OUT/constraints.txt pyyaml jsonschema httpx > $OUT/pip.log 2>&1
echo "pip rc $?" >> $OUT/pip.log
# Colab ships torchao 0.10.0, which PEFT 0.21.2 refuses (it wants 0.16 or newer) when it puts LoRA on a model; the
# trainer uses none, so it is removed rather than upgraded.
pip uninstall -y torchao >> $OUT/pip.log 2>&1
python3 - >> $OUT/versions.txt 2>&1 <<'PY'
import importlib.metadata as m
for name in ("torch", "triton", "transformers", "peft", "trl", "accelerate", "datasets", "pydantic", "tokenizers"):
    try:
        print("after", name, m.version(name))
    except m.PackageNotFoundError:
        print("after", name, "missing")
import torch
print("cuda", torch.version.cuda, "available", torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
stamp setup_done
echo SETUP_DONE > $OUT/SETUP_DONE
