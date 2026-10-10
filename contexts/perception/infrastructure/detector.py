"""The real-time detector (P1): ultralytics YOLO on a TensorRT engine, as `fn(frame) -> list[Track]`.

    detector = build_detector("config/models.yaml")      # None when there is no engine (laptops, CI)
    LlamaServerRuntime(model, detector=detector)         # then runtime.detect_track(frame)

Build the engine once, on the Jetson (the engine is tied to that GPU and TensorRT version):

    python -m contexts.perception.infrastructure.detector export yolo11n.pt

ultralytics is imported only when a detector is built, so nothing else needs it installed.
"""

import sys
from pathlib import Path
from typing import Any

from contracts.types import Box, Track
from runtime.registry import load_models_config


class YoloDetector:
    """Frame (BGR array or JPEG bytes) -> tracks with normalized 0-1 boxes, origin top-left.

    `model` is anything with ultralytics' predict()/track() interface; the default loads `weights`.
    With `track=True` the built-in ByteTrack keeps `track_id` stable across calls (P2); without it
    ids are only the detection order within one frame.
    """

    def __init__(self, weights: str | Path = "yolo11n.engine", *, conf: float = 0.35, imgsz: int = 640,
                 track: bool = False, model: Any = None) -> None:
        if model is None:
            from ultralytics import YOLO

            model = YOLO(str(weights), task="detect")
        self.model = model
        self.conf = conf
        self.imgsz = imgsz
        self.track = track

    def __call__(self, frame: Any) -> list[Track]:
        if isinstance(frame, bytes | bytearray):
            frame = _decode(frame)
        kw = {"conf": self.conf, "imgsz": self.imgsz, "verbose": False}
        results = self.model.track(frame, persist=True, **kw) if self.track else self.model.predict(frame, **kw)
        return tracks_from(results[0])


def tracks_from(result: Any) -> list[Track]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxyn, conf, cls = boxes.xyxyn.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
    ids = boxes.id.tolist() if getattr(boxes, "id", None) is not None else list(range(len(xyxyn)))
    out = []
    for (x1, y1, x2, y2), c, k, i in zip(xyxyn, conf, cls, ids, strict=True):
        box = Box(x1=min(max(x1, 0.0), 1.0), y1=min(max(y1, 0.0), 1.0), x2=min(max(x2, 0.0), 1.0), y2=min(max(y2, 0.0), 1.0))
        out.append(Track(label=result.names[int(k)], confidence=float(c), box=box, track_id=int(i)))
    return out


def _decode(jpeg: bytes) -> Any:
    import cv2
    import numpy as np

    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("not an image")
    return frame


def detector_available(config_path: str | Path = "config/models.yaml") -> bool:
    """Whether models.yaml names a detector whose weights are on this machine (does not load anything)."""
    cfg = load_models_config(config_path).get("detector", {})
    return cfg.get("backend", "ultralytics") == "ultralytics" and Path(cfg.get("weights", "yolo11n.engine")).exists()


def build_detector(config_path: str | Path = "config/models.yaml", *, track: bool = False) -> YoloDetector | None:
    """The detector models.yaml describes, or None when its weights are not on this machine (laptops, CI)."""
    cfg = load_models_config(config_path).get("detector", {})
    weights = Path(cfg.get("weights", "yolo11n.engine"))
    if cfg.get("backend", "ultralytics") != "ultralytics" or not weights.exists():
        return None
    return YoloDetector(weights, conf=float(cfg.get("conf", 0.35)), imgsz=int(cfg.get("imgsz", 640)), track=track)


def export_engine(weights: str | Path, *, imgsz: int = 640) -> Path:
    from ultralytics import YOLO

    return Path(YOLO(str(weights)).export(format="engine", half=True, imgsz=imgsz, device=0, batch=1, workspace=2))


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "export":
        sys.exit(__doc__)
    print(export_engine(sys.argv[2]))
