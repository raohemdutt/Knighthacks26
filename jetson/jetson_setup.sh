#!/usr/bin/env bash
# Install and start the atlas-llm service (R8). Run on the Jetson from the repo:
#     sudo ./jetson/jetson_setup.sh
# Safe to re-run. It does not touch network settings or upgrade packages.
set -euo pipefail

[ "$(id -u)" -eq 0 ] || { echo "run with sudo" >&2; exit 1; }
user="${SUDO_USER:?run with sudo, not as root}"
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

chmod +x "$repo/jetson/atlas-llm.sh"
sed -e "s|@USER@|$user|g" -e "s|@REPO@|$repo|g" "$repo/jetson/atlas-llm.service" > /etc/systemd/system/atlas-llm.service
systemctl daemon-reload
systemctl enable atlas-llm
systemctl restart atlas-llm
echo "atlas-llm installed; follow the log with: journalctl -u atlas-llm -f"
