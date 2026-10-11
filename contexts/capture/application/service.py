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

Two blurs (contexts/perception/infrastructure/face_blur.py). A picture that is kept or shared gets the strict one,
repeated until no face is found: every stored keyframe, and `snapshot(camera, blur=True)` for a recorder. The live
picture for a screen without the owner's ticket gets one pass: it runs on every frame shown and nothing of it is kept.
A camera can be set back to one pass for its keyframes with `keyframe_blur: fast` in config/cameras.yaml, if the box
cannot afford the second detector pass (about twice the time of one pass on a keyframe with a face in it, the same
on one without; keyframes are stored at most once per `observe_every_s`, usually every `keyframe_every_s`).

The live picture for the owner's ticket has three settings (`policy.live_faces`): "clear" (every face as it is),
"blurred" (the one-pass blur, the same bytes as without a ticket) and "owner": one pass again, but the faces that
`owner_faces` says are the scanned owner's are left as they are. `owner_faces(camera_id, rows, tracks, width, height)
-> [bool per face]` is handed in by bootstrap (the identity service's person tags; nothing is recognised per frame
here). While it is not set, fails, or answers False (nobody scanned, recognition off, no tag yet), the face is blurred.
"""

import asyncio
import logging
import os
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from contexts.capture.infrastructure.camera import CameraReader
from contracts.capture import CameraHealth
from contexts.perception.application.tracking import TrackingPipeline
from contexts.perception.infrastructure.detector import build_detector, detector_available
from contexts.perception.infrastructure.face_blur import STRICT, FaceBlur
from contexts.perception.infrastructure.face_counter import FaceCounter
from contexts.perception.infrastructure.keyframe_store import KeyframeStore

log = logging.getLogger("atlas.capture")


class LazyDetector:
    """Builds the detector on the first frame. torch and TensorRT cost about 1.2 GB of the Jetson's 8 GB, which a box
    whose camera is unplugged should not spend."""

    def __init__(self, factory: Any) -> None:
        self.factory, self._detector = factory, None

    def __call__(self, frame: Any) -> Any:
        if self._detector is None:
            self._detector = self.factory()
        return self._detector(frame)


class CaptureService:
    def __init__(self, cameras: dict[str, dict], bus: Any, detector_factory: Any, data_dir: str | Path = "data",
                 policy: Any = None) -> None:
        self.cameras, self.bus, self.policy = cameras, bus, policy
        self.detector_factory, self.data_dir = detector_factory, Path(data_dir)
        self.readers: dict[str, CameraReader] = {}
        self._snapshot_blur = FaceBlur()  # the Live screen's own detector: the pipeline's is used on another thread
        self._snapshot_lock = threading.Lock()
        self._shared_blur = FaceBlur(**STRICT)  # for snapshot(blur=True): a frame that will be kept or shared
        self._shared_lock = threading.Lock()  # its own, so a recording does not hold up the Live screen
        self._snapshot_cache: dict[str, tuple[int, bytes]] = {}
        self._health_sent: dict[str, tuple[str, datetime]] = {}  # camera -> (status last published, when)
        self.pipelines: dict[str, TrackingPipeline] = {}
        self.owner_faces: Any = None  # see the module docstring; None: "only me" is the blurred picture

    @classmethod
    def load(cls, cameras_file: str | Path, bus: Any, *, models_yaml: str | Path = "config/models.yaml",
             data_dir: str | Path = "data", policy: Any = None) -> "CaptureService | None":
        if os.environ.get("ATLAS_CAPTURE", "1") == "0":
            return None
        path = Path(cameras_file)
        cameras = ((yaml.safe_load(path.read_text()) or {}).get("cameras") or {}) if path.exists() else {}
        if not cameras:
            return None
        if not detector_available(models_yaml):
            log.warning("cameras are configured but there is no detector engine here (see config/models.yaml): "
                        "capture is off")
            return None
        return cls(cameras, bus, lambda: build_detector(models_yaml, track=True), data_dir, policy)

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        for camera_id, cfg in self.cameras.items():
            source = cfg["source"]
            fps = float(cfg.get("fps", 15))
            reader = CameraReader(camera_id, source, bus=self.bus, loop=loop, fps=fps,
                                  width=cfg.get("width", 1280), height=cfg.get("height", 720),
                                  motion_threshold=cfg.get("motion_threshold"), keepalive_s=float(cfg.get("keepalive_event_s", 5)))
            detector = LazyDetector(self.detector_factory)  # ByteTrack keeps its state in the model: one per camera
            blur = FaceBlur()  # one pass: the face counter's finder, and the keyframe blur only with `keyframe_blur: fast`
            strict = None if cfg.get("keyframe_blur", "strict") == "fast" else FaceBlur(**STRICT)  # anything else: strict
            store = KeyframeStore(self.data_dir, blur=lambda frame, blur=blur, strict=strict: self._saved(frame, blur, strict))
            pipeline = TrackingPipeline(
                camera_id, reader, detector, self.bus, loop, store=store,
                detect_fps=float(cfg.get("detect_fps", 10)),
                observe_every_s=float(cfg.get("observe_every_s", 1.0)),
                keyframe_every_s=float(cfg.get("keyframe_every_s", 5.0)),
                keepalive_s=float(cfg.get("keepalive_s", 30.0)),
                faces=FaceCounter(blur) if cfg.get("faces", True) else None,
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
        out = {}
        for cid, r in self.readers.items():
            p = self.pipelines[cid]
            speed = p.detector_speed() or {}
            out[cid] = {**r.stats(), "detected": p.detected, "errors": p.errors,
                        "detector_fps": speed.get("fps"), "detector_ms": speed.get("ms")}
        return out

    def detector_speed(self) -> dict | None:
        """The detector's speed on the camera timed most recently ({camera_id, fps, ms, samples, ts}), for
        GET /runtime/stats; None before any camera has run its detector."""
        timed = [{"camera_id": cid, **s} for cid, p in self.pipelines.items() if (s := p.detector_speed())]
        return max(timed, key=lambda s: s["ts"]) if timed else None

    def _saved(self, frame: Any, fast: Any, strict: Any) -> Any:
        """The copy of a frame that goes to disk: blurred until no face is found (or once, for a `fast` camera)."""
        if not self._blur_on():
            return frame
        return strict.scrambled_until_clear(frame)[0] if strict is not None else fast(frame)

    def snapshot(self, camera_id: str, clear: bool = False, blur: bool = False) -> bytes | None:
        """The newest frame as a JPEG for the phone's Live screen, faces blurred unless the policy says otherwise.
        None when the camera is unknown, paused, or has not delivered a frame yet. `clear` (the caller holds the
        owner's view ticket) leaves the faces as they are while `policy.live_clear`, leaves only the scanned owner's
        own face as it is while `policy.live_owner_only` (see the module docstring), and changes nothing once the owner
        chose "blurred"; this is the live picture only, a saved keyframe is blurred as before. `blur` is for a picture
        that will be kept or shared (a recorded clip): the strict blur, whatever the policy says, and `clear` is ignored."""
        import cv2

        reader = self.readers.get(camera_id)
        if reader is None or (self.policy and self.policy.is_paused(camera_id)) or reader.latest() is None:
            return None
        seq, _, frame = reader.latest()
        ticket = clear and not blur and self._blur_on()
        clear = ticket and getattr(self.policy, "live_clear", False)
        mine = ticket and not clear and self.owner_faces is not None and getattr(self.policy, "live_owner_only", False)
        # the pictures of one frame are never mixed up
        key = f"{camera_id} shared" if blur else f"{camera_id} clear" if clear else f"{camera_id} owner" if mine else camera_id
        with self._shared_lock if blur else self._snapshot_lock:
            cached = self._snapshot_cache.get(key)
            if cached and cached[0] == seq:
                return cached[1]
            if blur:
                shown = self._shared_blur.scrambled_until_clear(frame)[0]
            elif mine:
                shown = self._snapshot_blur.scrambled(frame, keep=self._owner(camera_id, frame))[0]
            else:
                shown = self._snapshot_blur(frame) if self._blur_on() and not clear else frame
            ok, jpeg = cv2.imencode(".jpg", shown, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                return None
            self._snapshot_cache[key] = (seq, jpeg.tobytes())
            return self._snapshot_cache[key][1]

    def _owner(self, camera_id: str, frame: Any) -> Any:
        """`keep` for the one-pass blur: which of the faces it found are the owner's, by the person tracks of the
        newest detection. The blur itself treats a failure or a wrong answer as "nobody" and blurs every face."""
        pipeline = self.pipelines.get(camera_id)
        latest = pipeline.latest if pipeline is not None else None
        tracks = list(latest[1]) if latest else []
        height, width = frame.shape[:2]

        def keep(rows: list[Any]) -> list[bool]:
            try:
                return list(self.owner_faces(camera_id, rows, tracks, width, height)) if tracks else [False] * len(rows)
            except Exception:  # cannot tell: blur everyone
                return [False] * len(rows)

        return keep

    def health(self, now: datetime, every: timedelta = timedelta(minutes=5)) -> list[CameraHealth]:
        """`capture.camera.health` for each camera whose status changed since the last call, and for every camera at
        least once per `every` (so the log shows a camera that stayed fine, and when the box last looked)."""
        out = []
        for camera_id, reader in self.readers.items():
            stats = reader.stats()
            status = "paused" if (self.policy and self.policy.is_paused(camera_id)) else stats["status"]
            last = self._health_sent.get(camera_id)
            if last and last[0] == status and now - last[1] < every:
                continue
            self._health_sent[camera_id] = (status, now)
            out.append(CameraHealth(ts=now, camera_id=camera_id, status=status, fps_actual=stats["fps_actual"], dropped=stats["dropped"]))
        return out

    def tracks(self, camera_id: str) -> dict | None:
        pipeline = self.pipelines.get(camera_id)
        if pipeline is None:
            return None
        latest = pipeline.latest
        return {"camera_id": camera_id, "ts_ms": latest[0] if latest else None,
                "tracks": [t.model_dump() for t in latest[1]] if latest else []}

    def _blur_on(self) -> bool:
        return self.policy is None or self.policy.blur_faces
