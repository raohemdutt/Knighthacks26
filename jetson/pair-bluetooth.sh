#!/usr/bin/env bash
# Run ON the Jetson to pair, trust and connect a Bluetooth device.
# Usage: ./pair-bluetooth.sh            (scan for 15s and list devices)
#        ./pair-bluetooth.sh XX:XX:XX:XX:XX:XX
set -euo pipefail

bluetoothctl power on
bluetoothctl agent on
bluetoothctl default-agent

if [ $# -lt 1 ]; then
    echo "Scanning for 15 seconds..."
    bluetoothctl --timeout 15 scan on
    bluetoothctl devices
    echo "Re-run with the MAC address of the device to pair."
    exit 0
fi

MAC="$1"
bluetoothctl --timeout 15 scan on >/dev/null
bluetoothctl pair "$MAC"
bluetoothctl trust "$MAC"
bluetoothctl connect "$MAC"
