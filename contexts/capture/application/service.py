"""Capture and perception on the box (C1, P1, P2, P6): the cameras in config/cameras.yaml, each with its own
reader, tracking detector and blurred keyframe store, started with the gateway and sharing its event bus.

    capture = CaptureService.load("config/cameras.yaml", bus, data_dir="data", policy=policy)
    capture.start(asyncio.get_running_loop())     # when the gateway starts
    capture.stop()                                # when it stops

`load` returns None (and says why in the log) when there is nothing to start: no cameras listed, capture switched
off with ATLAS_CAPTURE=0, or no detector engine on this machine (laptops, CI). A camera that is unplugged is not an
error: its reader keeps retrying, and `stats()` reports it as down.

The Vault policy is read live: faces are blurred before a keyframe is stored unless `policy.blur_faces` is off, and a
paused camera is not looked at.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import yaml

from contexts.capture.infrastructure.camera import CameraReader
from contexts.perception.application.tracking import TrackingPipeline
from contexts.perception.infrastructure.detector import YoloDetector, build_detector
from contexts.perception.infrastructure.face_blur import FaceBlur
from contexts.perception.infrastructure.keyframe_store import KeyframeStore

log = logging.getLogger("atlas.capture")


class CaptureService:
    def __init__(self, cameras: dict[str, dict], bus: Any, detector_factory: Any, data_dir: str | Path = "data",
                 policy: Any = None) -> None:
        self.cameras, self.bus, self.policy = cameras, bus, policy
        self.detector_factory, self.data_dir = detector_factory, Path(data_dir)
        self.readers: dict[str, CameraReader] = {}
        self.pipelines: dict[str, TrackingPipeline] = {}

    @classmethod
    def load(cls, cameras_file: str | Path, bus: Any, *, models_yaml: str | Path = "config/models.yaml",
             data_dir: str | Path = "data", policy: Any = None) -> "CaptureService | None":
        if os.environ.get("ATLAS_CAPTURE", "1") == "0":
            return None
        path = Path(cameras_file)
        cameras = ((yaml.safe_load(path.read_text()) or {}).get("cameras") or {}) if path.exists() else {}
        if not cameras:
            return None
        if build_detector(models_yaml) is None:
            log.warning("cameras are configured but there is no detector engine here (see config/models.yaml): "
                        "capture is off")
            return None
        return cls(cameras, bus, lambda: build_detector(models_yaml, track=True), data_dir, policy)

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        for camera_id, cfg in self.cameras.items():
            source = cfg["source"]
            fps = float(cfg.get("fps", 15))
            reader = CameraReader(camera_id, source, bus=self.bus, loop=loop, fps=fps,
                                  width=cfg.get("width", 1280), height=cfg.get("height", 720))
            detector: YoloDetector = self.detector_factory()  # ByteTrack keeps its state in the model: one per camera
            blur = FaceBlur()
            store = KeyframeStore(self.data_dir, blur=lambda frame, blur=blur: blur(frame) if self._blur_on() else frame)
            pipeline = TrackingPipeline(
                camera_id, reader, detector, self.bus, loop, store=store,
                detect_fps=float(cfg.get("detect_fps", 10)),
                observe_every_s=float(cfg.get("observe_every_s", 1.0)),
                keyframe_every_s=float(cfg.get("keyframe_every_s", 5.0)),
                keepalive_s=float(cfg.get("keepalive_s", 30.0)),
                paused=lambda camera_id=camera_id: bool(self.policy and self.policy.is_paused(camera_id)),
            )
            self.readers[camera_id], self.pipelines[camera_id] = reader, pipeline
            reader.start()
            pipeline.start()
            log.info("camera %s: %s at %s fps", camera_id, source, fps)

    def stop(self) -> None:
        for pipeline in self.pipelines.values():
            pipeline.stop()
        for reader in self.readers.values():
            reader.stop()

    def stats(self) -> dict[str, dict]:
        return {cid: {**r.stats(), "detected": self.pipelines[cid].detected, "errors": self.pipelines[cid].errors}
                for cid, r in self.readers.items()}

    def _blur_on(self) -> bool:
        return self.policy is None or self.policy.blur_faces
