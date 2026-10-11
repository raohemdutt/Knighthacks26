# Demo staging kit (D6)

What has to be on the table for the five-minute demo, which beat each item serves, and the command that
checks the parts software can see. The Stagehand owns this list; tick it at T-15 together with the
preflight (D9) and the table-day checklist in `docs/demo-run-sheet.md`.

## Check it

```
python -m contexts.demo kit --gateway http://100.124.34.36:8000 --devpost https://devpost.com/software/<project>
```

- Prints the physical checklist below, then checks the box: the phone app is served and has its manifest,
  the gateway is on https (phones need it to install the app and use the mic), ntfy answers on `:8090` and the
  alert topic can be read, and the box's own preflight passes.
- `--devpost` writes `data/kit/devpost-qr.html` (one printable page) and `data/kit/devpost-qr.svg`. It needs
  `pip install qrcode`. The Devpost page doesn't exist yet; until it does, the QR check fails on purpose.
- Every check is a GET. Nothing is pushed to the phones and nothing on the box changes.
- `--list` prints only the physical checklist. `--ntfy` and `--topic` default to `NTFY_URL` and `NTFY_TOPIC`
  (else the gateway's host on `:8090` and `atlas`).

## The physical kit

| Item | Beat | Ready at T-15 |
| --- | --- | --- |
| Bright keychain | Where are my keys? (1:05) | On the desk in view of cam-desk at least 2 minutes before the run, not moved since. |
| Cardboard "package" | It watches for you (1:35), Unplug the internet (3:15) | Out of frame, within the Stagehand's reach. Try it once at T-15: the detector has to call it a package. |
| Tape-marked door zone | It watches for you (1:35) | The tape on the floor matches the Door zone drawn on cam-door in `/live/`; the package fits inside it. |
| Camera mounts / tripods | Hook (0:00), every camera beat | cam-desk sees the desk, keys and laptop; cam-door sees the taped zone; both tiles live in `/live/`. |
| Power strip | everything | Jetson, both cameras, the speaker and the phone chargers on one strip, plugged into a live outlet. |
| Long USB cables | Hook (0:00), every camera beat | Both cameras reach the Jetson without pulling on the mounts; taped down where people walk. |
| Phone hotspot | Unplug the internet (3:15); backup if venue Wi-Fi fails | Hotspot on, the Jetson joins it, the box answers on its LAN address from the demo phone. |
| 2 charged demo phones, ntfy + PWA installed | It watches for you, It learns, Unplug | Above 80%, Do Not Disturb off, ntfy subscribed to the alert topic on the box, Atlas on the home screen, the Tailscale and LAN addresses bookmarked. |
| Printed QR to the Devpost | Close | Printed from `kit --devpost <url>`, taped to the table, scans to the project page. |
| Extra: "This table is recorded, faces are blurred on-device" sign | the whole table | Visible from the aisle (Demo Plan T-15). |
| Extra: an empty can | v2: Waste Watch | Within reach; the Recycling and Trash zones are drawn. |

The first nine are D6's "Done when"; the two extras come from the Demo Plan.
