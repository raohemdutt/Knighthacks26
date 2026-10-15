#!/usr/bin/env bash
# Install and start the Atlas services on the Jetson (R8 model, C6 gateway + cameras, V6 egress firewall). Run from the repo:
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

# `sudo ./jetson/jetson_setup.sh voice` installs only the voice unit (X16) and touches no other service.
if [ "${1:-}" = "voice" ]; then
    sed -e "s|@USER@|$user|g" -e "s|@REPO@|$repo|g" -e "s|@VENV@|$venv|g" -e "s|@UID@|$(id -u "$user")|g" "$repo/jetson/atlas-voice.service" > /etc/systemd/system/atlas-voice.service
    # Without linger, PulseAudio (and the Bluetooth speaker with it) dies when the last SSH session closes.
    loginctl enable-linger "$user"
    systemctl daemon-reload
    systemctl enable atlas-voice
    systemctl restart atlas-voice
    echo "installed atlas-voice; follow it with: journalctl -u atlas-voice -f"
    exit 0
fi

# `sudo ./jetson/jetson_setup.sh deploy` installs only the deploy timer (the box pulls the prod branch's bundle).
if [ "${1:-}" = "deploy" ]; then
    for unit in atlas-deploy.service atlas-deploy.timer; do
        sed -e "s|@USER@|$user|g" -e "s|@REPO@|$repo|g" "$repo/jetson/$unit" > "/etc/systemd/system/$unit"
    done
    systemctl daemon-reload
    systemctl enable --now atlas-deploy.timer
    echo "installed atlas-deploy.timer; follow it with: journalctl -u atlas-deploy -f (pause: touch ~/.atlas-deploy-paused)"
    exit 0
fi

chmod +x "$repo/jetson/atlas-llm.sh"
for unit in atlas-llm atlas-gateway atlas-egress; do
    sed -e "s|@USER@|$user|g" -e "s|@REPO@|$repo|g" -e "s|@VENV@|$venv|g" "$repo/jetson/$unit.service" > "/etc/systemd/system/$unit.service"
done
# The box must never sleep: GNOME's idle timer would suspend it after a while.
systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target
systemctl daemon-reload
systemctl enable atlas-llm atlas-gateway atlas-egress
systemctl restart atlas-llm
systemctl restart atlas-gateway
systemctl restart atlas-egress
echo "installed atlas-llm, atlas-gateway and atlas-egress; follow them with: journalctl -u atlas-llm -u atlas-gateway -u atlas-egress -f"
