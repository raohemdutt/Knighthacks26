# Connecting to the Jetson Orin Nano

The Orin Nano dev kit has Wi-Fi and Bluetooth built in. For the first boot you
reach it over the USB-C cable (serial console); after that you can SSH over
Wi-Fi or the USB network link.

## 1. First boot over USB-C (headless)

1. Flash the JetPack SD card image and insert the card.
2. Connect the Orin Nano's **USB-C** port to your laptop. That port is data
   only; power still comes from the **barrel jack**.
3. Power it on and wait about a minute.
4. Open the serial console (115200 baud):

| OS | Command |
|----|---------|
| Windows | `.\jetson\connect-serial.ps1` (needs PuTTY: `winget install PuTTY.PuTTY`) |
| macOS | `./jetson/connect-serial.sh` (`screen /dev/cu.usbmodem* 115200`) |
| Linux | `./jetson/connect-serial.sh` (`sudo screen /dev/ttyACM0 115200`) |

The setup wizard runs in that terminal: create your user, set the timezone, and so on.

## 2. Join Wi-Fi (run on the Jetson)

Copy `setup-wifi.sh` over, or type the commands directly in the serial console:

```bash
sudo nmcli device wifi connect "YourSSID" password "YourPassword"
ip addr show wlan0
```

Don't commit Wi-Fi passwords to this repo.

## 3. SSH in

- Over Wi-Fi: `ssh <user>@<wlan0 IP>`
- Over the USB-C cable: `ssh <user>@192.168.55.1`

## 4. Pair Bluetooth (run on the Jetson)

```bash
./pair-bluetooth.sh                    # scan and list nearby devices
./pair-bluetooth.sh XX:XX:XX:XX:XX:XX  # pair, trust and connect
```

Or manually:

```
bluetoothctl
  power on
  agent on
  scan on
  pair XX:XX:XX:XX:XX:XX
  trust XX:XX:XX:XX:XX:XX
  connect XX:XX:XX:XX:XX:XX
```

## 5. On-device model (`atlas-llm`, R8)

Qwen3.5-4B (Q4_K_M) plus its vision projector, served by llama.cpp on port 8080
(localhost only). The files live in `data/models/<version>/` with a `manifest.json`;
`data/models/current` points at the active version.

```bash
python3 jetson/make_manifest.py data/models/atlas-base --base Qwen/Qwen3.5-4B   # sha256s + manifest.json
ln -sfn atlas-base data/models/current
sudo ./jetson/jetson_setup.sh      # installs and starts the atlas-llm service (safe to re-run)
journalctl -u atlas-llm -f
```

The gateway reaches it with `ATLAS_LLM_URL=http://127.0.0.1:8080` in `.env`; the
checked-in `config/models.yaml` stays on `fake` so laptops and CI keep working.
Thinking mode is off (`--reasoning-budget 0`): the runtime's small token limits
would otherwise be spent before the answer starts.

### Switching models (R7)

```bash
PYTHONPATH=. python jetson/atlas_swap.py status                 # installed versions, current, previous
PYTHONPATH=. python jetson/atlas_swap.py swap atlas-v2-desk     # verify sha256s, switch, restart, test, roll back on failure
PYTHONPATH=. python jetson/atlas_swap.py rollback               # back to the version before the last switch
```

A new version is a folder `data/models/<version>/` with its `manifest.json` (`ModelManifest`: the `.gguf`, the
mmproj, and a sha256 for each). Nothing is switched to until every checksum matches. The 4B model does not fit
twice in 8 GB, so there is a ~40 s gap while the new one loads (a full swap took ~2 min including the checksums,
which are read before the old model is stopped). If the new model does not load and answer a test question, the old
one is put back automatically. `GET /runtime/models` on the gateway shows the current version and what llama-server
really serves (`in_sync`).

## 6. Detector (P1)

yolo11n as a TensorRT FP16 engine, called as `fn(frame) -> list[Track]` (normalized 0-1 boxes, top-left origin).
The venv `~/atlas/venv-yolo` has torch and ultralytics; the system TensorRT 10.3 is linked into it
(`ln -s /usr/lib/python3.10/dist-packages/tensorrt <venv>/lib/python3.10/site-packages/tensorrt`).

```bash
pip install onnx onnxslim                                  # the engine export needs them
python -m contexts.perception.infrastructure.detector export yolo11n.pt   # ~7 min, once per TensorRT version
PYTHONPATH=. python jetson/bench_detector.py yolo11n.engine some.mp4      # end-to-end FPS
```

The box's Python is 3.10; the repo asks for 3.11 only because of `from datetime import UTC`. A one-line
`.pth` in the venv fixes that without touching the code:
`echo 'import datetime; datetime.UTC = datetime.timezone.utc' > <venv>/lib/python3.10/site-packages/atlas_py310_compat.pth`.

## 7. Always on (C6)

Two systemd services, both `Restart=always` and enabled at boot: `atlas-llm` (the model, section 5) and
`atlas-gateway` (`python -m contexts.gateway`: the app the phone talks to on port 8000, plus the cameras, detector,
tracker and blurred keyframe store, all in one process so they share the event bus).

```bash
sudo ./jetson/jetson_setup.sh                       # installs and starts both (safe to re-run; ATLAS_VENV picks the Python venv)
journalctl -u atlas-llm -u atlas-gateway -f         # logs
curl localhost:8000/capture/status                  # per camera: ok / degraded / down, fps, frames detected
curl localhost:8000/runtime/models                  # current model version and what llama-server serves
```

Cameras are listed in `config/cameras.yaml` (id, `/dev/video0` or a video file, fps). Capture starts only where
`yolo11n.engine` exists (section 6), so laptops and CI skip it; `ATLAS_CAPTURE=0` turns it off. A camera that is not
plugged in is retried every 2 s and shows as `down`. The Vault policy is honoured live: faces are blurred before a
keyframe is stored unless `blur_faces` is off, and a paused camera is not looked at. Settings and secrets go in the
repo's `.env` (never in git). The unit sets `ATLAS_MEMORY_FIXTURE=` empty, so a real box holds only what its cameras
saw; put `ATLAS_MEMORY_FIXTURE=fixtures/memory/memory.yaml` in `.env` for the canned demo night.

`jetson_setup.sh` also masks the sleep, suspend and hibernate targets (GNOME's default idle timer would suspend the box).

## 8. Egress firewall and the offline switch (V6, V5, H4)

The `atlas-egress` service (root, installed by `jetson_setup.sh`) keeps an iptables/ip6tables chain `ATLAS_EGRESS` on
the OUTPUT side only (this Jetson kernel has no nftables counters, so iptables), so ssh in and the phone talking to the
box are not affected. What may leave: loopback and replies, any private or link-local address (so any venue's Wi-Fi
works), NTP, DNS to the box's own resolvers, the hub proxy `dub.aupcloud.io`, each connector's `hosts` from
`config/connectors.yaml`, Neon (from `NEON_DATABASE_URL`), a non-LAN ntfy relay, and the Ubuntu connectivity check.
Anything else is logged (`journalctl -k | grep atlas-egress-drop`) and rejected. Add hosts for a one-off need in
`config/egress.yaml` (`groups: {name: [host, ...]}`); names are re-resolved every 5 minutes.

```bash
make offline        # LAN only; the firewall follows within ~10 s. Detection, answers and alerts carry on; connectors pause
make online         # back to the allowlist (the phone's Privacy switch does the same through PUT /vault/policy)
curl localhost:8000/vault/egress                       # bytes per group (hub, each connector, LAN, NTP, DNS), blocked count, recent blocks
sudo python jetson/egress.py apply --revert-after 90   # try new rules by hand: undone in 90 s unless `touch /run/atlas-egress.ok`
sudo python jetson/egress.py clear                     # remove the firewall (for apt/pip on the box), then `sudo systemctl restart atlas-egress`
```

While the firewall is on, `apt` and `pip` on the box are blocked: run `clear` first and restart the service after.

### Updating the code on the box

The box has no GitHub login, so code reaches it from a laptop: `./jetson/push.sh [alexander@<ip>]` copies the working
tree (not `.env`, `data/`, the engine or the models) and restarts `atlas-gateway`; the app rebuilds its memory from the
event log. `--web` builds the phone app first. The default address is `alexander@ubuntu.local`.

## 9. Voice on the box (X16)

The Jetson listens through the USB webcam's built-in microphone and speaks through a Bluetooth speaker. Speech to text is
whisper.cpp (`tiny.en`), text to speech is Piper; both run on the box, one short process per use (peak about 200 MB and
150 MB, nothing resident between questions). A question goes in through the same `POST /api/ask` as the phone's Ask
screen, so it is logged as `experience.question.asked` and the spoken text is the `agent.answer.generated` text. Only
text leaves the microphone path: no audio leaves the box and none is stored (the recording lives in memory, plus one
temp file in RAM for the 2 seconds whisper reads it). It needs nothing from the internet, so `make offline` is fine.

**One-time setup** (the tools come from a laptop with internet, because the box's firewall blocks GitHub and Hugging Face):

```bash
./jetson/voice-fetch.sh alexander@192.168.1.126      # on the laptop: downloads, copies, builds whisper.cpp on the box
sudo dpkg -i ~/atlas/voice/debs/pulseaudio-module-bluetooth_arm64.deb && pulseaudio -k   # on the box: lets PulseAudio use a Bluetooth speaker
```

**Pair the speaker** (on the box; put the speaker in pairing mode first, then run):

```bash
./jetson/pair-bluetooth.sh                     # scans 15 s and lists devices: note the speaker's MAC
./jetson/pair-bluetooth.sh AA:BB:CC:DD:EE:FF   # pair, trust, connect
pactl list short sinks                         # a bluez_sink... line means it is an audio output now
```

Then set `speaker_mac: "AA:BB:CC:DD:EE:FF"` in `config/voice.yaml`: the service reconnects it before every answer and at
start, so a speaker that went to sleep or a restarted service comes back by itself (trusted devices also reconnect on power-up).

**Start it** (only `atlas-voice`, no other service is touched; the gateway must be restarted once to get `/api/voice/*`):

```bash
sudo ./jetson/jetson_setup.sh voice        # installs and starts atlas-voice (memory-capped, killed first if memory runs out)
sudo systemctl restart atlas-gateway       # once, to add the /api/voice proxy and the pre-flight line
journalctl -u atlas-voice -f
```

**Use it:** open `/operator/` and tap **Tap to talk** (the panel shows Listening, Thinking, Speaking or Off), or
`curl -X POST localhost:8000/api/voice/talk` (add `-H "Authorization: Bearer $ATLAS_TOKEN"` if a token is set). Tap again
when done (it also ends by itself after 1 s of quiet), tap while it speaks to stop it. `GET /api/voice/status` says what
it heard, what it answered, and whether the mic and speaker are fine. The box never speaks while the microphone is open.
`python -m contexts.demo preflight` has a "voice on the box" line.

**Hands-free:** with `wake_word: true` in `config/voice.yaml` (the default) the box is always listening: say "Hey Atlas", it
answers "What's up?", then ask your question. The microphone stays open and, every 1.5 s, the last 3 s of sound are
transcribed on the box by whisper, so music or a crowd in the room does not hide the wake phrase. Everything that is not
the wake phrase is dropped unread; nothing is kept or sent anywhere. Tapping still works, and `wake_word: false` goes back to
push to talk only.

## 10. Zones in a room you could not preset

Open `http://<jetson>:8000/zones/` on a laptop or phone (or "Set zones" on `/operator/`): pick the camera and the zone,
drag a box on the picture, **Save**. The box rewrites its own `config/zones.yaml` and the zone engine uses the new boxes
at once, with no restart (`PUT /api/zones/<camera>`; the token goes in Settings if `ATLAS_TOKEN` is set). **Use demo zones**
(`POST /api/zones/demo`) puts the big, forgiving boxes from `config/zones.demo.yaml` over the same cameras and ids when
there is no time to draw. Only zones that already exist can be moved, because rules and Waste Watch refer to their
names. Sort It reads its bin boxes at start, so restart `atlas-gateway` after moving Recycling or Trash.

`./jetson/push.sh` keeps the box's `config/zones.yaml` when it differs from the tree's (so a push cannot undo zones drawn
on site); `./jetson/push.sh --zones` overwrites it. To keep what was drawn, copy it back: `scp <box>:Knighthacks26/config/zones.yaml config/`.

## 11. Face models (owner recognition and the face profile)

YuNet (finds and blurs faces) is in the repo. The two models behind the opt-in People screen are not, and the app never
downloads anything; from a laptop with internet:

```bash
./jetson/face-fetch.sh [alexander@<ip>]     # SFace (37 MB) and the expression model (5 MB) into contexts/perception/models/, then restarts the app
curl localhost:8000/identity/settings       # available: true (SFace), expressions_available: true, photos_available: true (OpenCV)
```

Without SFace recognition cannot be switched on; without the expression model "Read my expressions" and the anonymous
room mode stay unavailable and everything else works. `./jetson/push.sh` leaves the box's model files alone. Everything is
off until someone switches it on in the app, and stored pictures are blurred in every mode.

## Firmware update (boards with old factory firmware)

JetPack 6 and 7 need firmware 36.x on the board. Boards with older firmware
won't boot those images and show nothing over USB-C. Check the version from a
booted Jetson with `sudo nvbootctrl dump-slots-info` (`Current version:` line).

If it's older than 36.0, use JetPack 5.1.3 as a bridge:

1. Write `JP513-orin-nano-sd-card-image_b29.zip`
   ([download](https://developer.nvidia.com/downloads/embedded/l4t/r35_release_v5.0/jp513-orin-nano-sd-card-image.zip))
   to a 64GB+ microSD card with balenaEtcher and boot it.
2. Finish first-boot setup over the serial console and get it online (`setup-wifi.sh`).
3. Run the updates:

   ```bash
   sudo systemctl status nv-l4t-bootloader-config   # should be inactive / exited successfully
   sudo reboot                                      # firmware update runs; don't cut power
   sudo apt update
   sudo apt install nvidia-l4t-jetson-orin-nano-qspi-updater
   sudo reboot                                      # QSPI update runs; don't cut power
   ```

4. Power off and swap in the JetPack 6.2.1 SD card image
   ([download page](https://developer.nvidia.com/embedded/jetpack-sdk-621)).

**JetPack 7.x is not an SD card image.** It ships as an installer ISO that you
write to a USB flash drive, which then installs onto the microSD card or an
NVMe SSD. Flashing the ISO straight to the SD card will not boot.
See NVIDIA's [Quick Start](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/quick_start.html).

## Troubleshooting

- **No serial port appears.** If the board shipped with older firmware and you
  flashed a JetPack 6 image, it may not boot at all. Boot a JetPack 5.1.3 image
  once to update the firmware, then switch to JetPack 6. If nothing shows up on
  USB-C after a couple of minutes, that's the likely cause.
- **Windows shows only "Standard Serial over Bluetooth link" ports.** Those are
  not the Jetson. Look for a new COM port that appears when you plug the board in.
- Use a USB-C cable that carries data; some charge-only cables won't enumerate.
