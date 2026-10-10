#!/usr/bin/env bash
# Start llama-server for the model that data/models/current points at (R8, R7).
# The service unit runs this; R7's hot-swap starts a second copy with ATLAS_LLM_PORT set.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pkg="$(readlink -f "${ATLAS_MODEL_DIR:-$repo/data/models/current}")"  # the real folder, so /props names the version
llama_server="${LLAMA_SERVER:-$HOME/atlas/llama.cpp/build/bin/llama-server}"

read -r gguf mmproj < <(python3 -c '
import json, sys
m = json.load(open(sys.argv[1] + "/manifest.json"))
print(m["gguf"], m.get("mmproj") or "")
' "$pkg")

args=(-m "$pkg/$gguf" --jinja --host 127.0.0.1 --port "${ATLAS_LLM_PORT:-8080}" -c 4096 -ngl 99 -np 1 --reasoning-budget 0)
[ -n "$mmproj" ] && args+=(--mmproj "$pkg/$mmproj")

exec "$llama_server" "${args[@]}"
