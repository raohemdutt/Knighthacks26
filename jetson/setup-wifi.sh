#!/usr/bin/env bash
# Run ON the Jetson (from the serial console) to join Wi-Fi and print the IP.
# Usage: sudo ./setup-wifi.sh "SSID" "password"
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: sudo $0 \"SSID\" \"password\""
    exit 1
fi

nmcli device wifi rescan || true
nmcli device wifi connect "$1" password "$2"

IP=$(ip -4 -o addr show wlan0 | awk '{print $4}' | cut -d/ -f1)
echo "Connected. wlan0 IP: ${IP:-unknown}"
echo "SSH from another machine on the same network: ssh $(logname 2>/dev/null || echo "$USER")@${IP}"
