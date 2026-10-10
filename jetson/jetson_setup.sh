#!/usr/bin/env bash
# Install and start the Atlas services on the Jetson (R8 model, C6 gateway + cameras). Run from the repo:
#     sudo ./jetson/jetson_setup.sh
# Safe to re-run. It does not touch network settings or upgrade packages.
# The gateway runs in the Python venv named by ATLAS_VENV (default ~/atlas/venv-yolo: torch, ultralytics, TensorRT).
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 1; }
user="${SUDO_USER:?run with sudo, not as root}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="${ATLAS_VENV:-$(getent passwd "$user" | cut -d: -f6)/atlas/venv-yolo}"

[ -x "$venv/bin/python" ] || { echo "no Python venv at $venv (set ATLAS_VENV)" >&2; exit 1; }
[ -e "$repo/yolo11n.engine" ] || echo "note: no $repo/yolo11n.engine yet: the gateway will run without cameras (see jetson/README.md, detector)"

chmod +x "$repo/jetson/atlas-llm.sh"
for unit in atlas-llm atlas-gateway; do
    sed -e "s|@USER@|$user|g" -e "s|@REPO@|$repo|g" -e "s|@VENV@|$venv|g" "$repo/jetson/$unit.service" > "/etc/systemd/system/$unit.service"
done
systemctl daemon-reload
systemctl enable atlas-llm atlas-gateway
systemctl restart atlas-llm
systemctl restart atlas-gateway
echo "installed atlas-llm and atlas-gateway; follow them with: journalctl -u atlas-llm -u atlas-gateway -f"
