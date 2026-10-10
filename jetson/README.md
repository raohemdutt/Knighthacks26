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
