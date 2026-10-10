#!/usr/bin/env bash
# Open the Jetson Orin Nano serial console over the USB-C cable (macOS / Linux).
# Usage: ./connect-serial.sh [port]
set -euo pipefail

BAUD=115200
PORT="${1:-}"

if [ -z "$PORT" ]; then
    case "$(uname)" in
        Darwin) PORT=$(ls /dev/cu.usbmodem* 2>/dev/null | head -n1 || true) ;;
        *)      PORT=$(ls /dev/ttyACM* 2>/dev/null | head -n1 || true) ;;
    esac
fi

if [ -z "$PORT" ]; then
    echo "No Jetson serial port found."
    echo "Check: USB-C cable is data-capable, board is powered from the barrel jack, waited ~1 min after power-on."
    echo "If nothing ever appears, the board likely needs a firmware update (boot JetPack 5.1.3 once, then JetPack 6)."
    exit 1
fi

echo "Connecting to $PORT at $BAUD baud (exit screen with Ctrl-A then K)..."
if [ "$(uname)" = "Darwin" ]; then
    screen "$PORT" "$BAUD"
else
    sudo screen "$PORT" "$BAUD"
fi
