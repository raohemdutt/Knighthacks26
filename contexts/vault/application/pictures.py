"""Which camera pictures were cleared to leave the box (V7).

Saved pictures are always blurred, nothing with a face leaves the box, and only the owner's own signed-in screen
shows the live view clear. Video never leaves. A camera picture leaves only inside a training export (L1), which
runs only after the owner approves the `export_training_data` card and passes every picture through the outbound
gate (outbound.py: faces blurred again until none is found, whatever is on disk). Nothing else in the box sends a
picture: phone alerts carry a link the phone fetches from the box itself, and that link is the gated copy too. So
the count of camera pictures that may have left is exactly what the approved exports hold, and this projection
keeps that count from the log (`learning.dataset.exported`, replayed).

The firewall's numbers (jetson/egress.py, GET /vault/egress) say how many bytes went to each destination; this says
how many of them could have been pictures, and which approval let each batch go.
"""

from pathlib import Path

from bus import EventBus
from contracts.learning import DatasetExported


def _size(folder: Path) -> int:
    return sum(f.stat().st_size for f in folder.glob("*") if f.is_file()) if folder.is_dir() else 0


class PictureLedger:
    def __init__(self, data_dir: Path) -> None:
        self.exports_dir = Path(data_dir) / "exports"
        self.batches: dict[str, dict] = {}

    def subscribe(self, bus: EventBus) -> None:
        bus.subscribe("learning.dataset.exported", self.on_exported, replay=True)

    def on_exported(self, e: DatasetExported) -> None:
        self.batches[e.dataset_id] = {
            "dataset_id": e.dataset_id,
            "task": e.task,
            "pictures": e.examples,
            "bytes": _size(self.exports_dir / e.dataset_id / "images"),  # 0 once wiped: the folder is gone
            "approval": e.action_id,
            "ts": e.ts.isoformat(),
        }

    def snapshot(self) -> dict:
        batches = sorted(self.batches.values(), key=lambda b: b["ts"])
        return {
            "video_bytes": 0,  # no code path sends video: the box stores keyframes, never clips
            "pictures": sum(b["pictures"] for b in batches),
            "picture_bytes": sum(b["bytes"] for b in batches),
            "approved_exports": len(batches),
            "batches": batches[-5:],
        }
