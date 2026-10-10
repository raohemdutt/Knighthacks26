#!/usr/bin/env bash
# Run on a laptop WITH internet (the Jetson's firewall blocks GitHub, Hugging Face and PyPI, on purpose). Downloads the
# speech tools for voice on the box (X16), copies them to the Jetson over the LAN and builds whisper.cpp there.
#
#     ./jetson/voice-fetch.sh                          # alexander@192.168.1.126
#     ./jetson/voice-fetch.sh alexander@ubuntu.local
#
# What lands in ~/atlas/voice on the box (about 120 MB on disk, nothing loaded until someone speaks):
#     src/whisper.cpp-1.7.6/build/bin/whisper-cli   speech to text (built on the box, one job at a time, low priority)
#     models/ggml-tiny.en-q5_1.bin                   the whisper model, 31 MB
#     piper/piper + libraries                        text to speech (Piper's own aarch64 release)
#     models/en_US-lessac-low.onnx                   the voice, 60 MB
#     debs/pulseaudio-module-bluetooth_arm64.deb     lets PulseAudio use a Bluetooth speaker; install it with sudo (jetson/README.md)
set -euo pipefail

target="${1:-alexander@192.168.1.126}"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
cd "$work"

WHISPER_TAG=v1.7.6
PIPER_TAG=2023.11.14-2
HF=https://huggingface.co

curl -fsSL -o whisper.tar.gz "https://github.com/ggml-org/whisper.cpp/archive/refs/tags/$WHISPER_TAG.tar.gz"
curl -fsSL -o ggml-tiny.en-q5_1.bin "$HF/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q5_1.bin"
curl -fsSL -o piper.tar.gz "https://github.com/rhasspy/piper/releases/download/$PIPER_TAG/piper_linux_aarch64.tar.gz"
curl -fsSL -o en_US-lessac-low.onnx "$HF/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/en_US-lessac-low.onnx"
curl -fsSL -o en_US-lessac-low.onnx.json "$HF/rhasspy/piper-voices/resolve/main/en/en_US/lessac/low/en_US-lessac-low.onnx.json"
curl -fsSL -o pulseaudio-module-bluetooth_arm64.deb \
    "http://ports.ubuntu.com/ubuntu-ports/pool/main/p/pulseaudio/pulseaudio-module-bluetooth_15.99.1+dfsg1-1ubuntu2.2_arm64.deb"

ssh "$target" 'mkdir -p ~/atlas/voice/src ~/atlas/voice/models ~/atlas/voice/piper ~/atlas/voice/debs'
scp -q whisper.tar.gz "$target:atlas/voice/src/"
scp -q ggml-tiny.en-q5_1.bin en_US-lessac-low.onnx en_US-lessac-low.onnx.json "$target:atlas/voice/models/"
scp -q piper.tar.gz "$target:atlas/voice/"
scp -q pulseaudio-module-bluetooth_arm64.deb "$target:atlas/voice/debs/"

# -j1 and nice: the LLM and the detector are running on this box and must not be starved while it compiles
ssh "$target" 'set -e; cd ~/atlas/voice
    tar xzf piper.tar.gz -C piper --strip-components=1
    tar xzf src/whisper.tar.gz -C src
    cd src/whisper.cpp-'"${WHISPER_TAG#v}"'
    nice -n 19 cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DWHISPER_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF >/dev/null
    nice -n 19 cmake --build build -j1 --target whisper-cli >/dev/null
    ls -la build/bin/whisper-cli'
echo "done. Next: sudo dpkg -i ~/atlas/voice/debs/pulseaudio-module-bluetooth_arm64.deb  (see jetson/README.md, Voice)"
