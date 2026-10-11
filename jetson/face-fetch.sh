#!/usr/bin/env bash
# Run on a laptop WITH internet (the Jetson's firewall blocks GitHub on purpose). Downloads the two face models that are
# not in the repo, copies them to the box over the LAN and restarts the app. The app itself never downloads anything:
# without a file its feature reports itself unavailable (GET /identity/settings).
#
#     ./jetson/face-fetch.sh                          # alexander@ubuntu.local, ~/Knighthacks26
#     ./jetson/face-fetch.sh alexander@172.20.10.6
#     ./jetson/face-fetch.sh --local                  # only into this checkout's contexts/perception/models/ (for a laptop)
#
# What lands in contexts/perception/models/ (both from OpenCV Zoo, Apache 2.0; YuNet is already in the repo):
#     face_recognition_sface_2021dec.onnx                         37 MB  owner recognition (P9): "available" turns true
#     facial_expression_recognition_mobilefacenet_2022july.onnx    5 MB  apparent expression: "expressions_available" turns true
# The box's OpenCV must have cv2.FaceRecognizerSF and cv2.dnn (opencv-python 4.5.4 or newer).
set -euo pipefail

ZOO=https://github.com/opencv/opencv_zoo/raw/main/models
SFACE=face_recognition_sface_2021dec.onnx
FER=facial_expression_recognition_mobilefacenet_2022july.onnx
FER_SHA256=4f61307602fc089ce20488a31d4e4614e3c9753a7d6c41578c854858b183e1a9

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${1:-alexander@ubuntu.local}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cd "$work"

curl -fsSL -o "$SFACE" "$ZOO/face_recognition_sface/$SFACE"
curl -fsSL -o "$FER" "$ZOO/facial_expression_recognition/$FER"
sum="$(shasum -a 256 "$FER" | cut -d' ' -f1)"
if [ "$sum" != "$FER_SHA256" ]; then
    echo "the expression model's checksum is $sum, expected $FER_SHA256: not installing it" >&2
    exit 1
fi
echo "sha256 $(shasum -a 256 "$SFACE")"

if [ "$target" = "--local" ]; then
    cp "$SFACE" "$FER" "$repo/contexts/perception/models/"
    echo "done: restart the app, then GET /identity/settings"
    exit 0
fi
ssh "$target" 'mkdir -p ~/Knighthacks26/contexts/perception/models'
scp -q "$SFACE" "$FER" "$target:Knighthacks26/contexts/perception/models/"
ssh "$target" 'sudo systemctl restart atlas-gateway'
echo "done. Check: ssh $target 'curl -s localhost:8000/identity/settings'  (available and expressions_available should be true)"
