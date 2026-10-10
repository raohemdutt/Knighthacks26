"""Synthetic frames and a box with recognition wired the way the gateway wires it. No camera, no OpenCV."""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from contexts.perception.application.identity import IdentityService
from contexts.perception.infrastructure.identity_store import IdentityStore
from contexts.perception.infrastructure.photo import PhotoReader
from contexts.perception.infrastructure.recognizer import FakeRecognizer
from contexts.vault import KeyRing, Vault
from contracts.types import Box, Track

T0 = datetime(2026, 10, 11, 12, tzinfo=UTC)
W, H = 640, 480


def row(x, y, w=60, h=80, facing=True, score=0.95):
    """A YuNet row for a face at (x, y) in pixels: frontal, or turned to the side."""
    eyes = 0.5 * w if facing else 0.15 * w
    cx = x + w / 2
    return [x, y, w, h, cx - eyes / 2, y + 0.4 * h, cx + eyes / 2, y + 0.4 * h, cx, y + 0.55 * h,
            cx - eyes / 2, y + 0.8 * h, cx + eyes / 2, y + 0.8 * h, score]


class Frame:
    """A picture that knows its faces: each is (who it is as bytes, YuNet row). `pixels` stands in for the image."""

    shape = (H, W, 3)

    def __init__(self, *faces, pixels: bytes = b"\x10\x20\x30" * 64):
        self.faces = list(faces)
        self.pixels = bytearray(pixels)

    def crop(self, face_row):
        return next(who for who, r in self.faces if r[:2] == list(face_row[:2]))


class Finder:
    """Stands in for YuNet (FaceBlur.faces)."""

    def __init__(self):
        self.calls = 0

    def faces(self, frame):
        self.calls += 1
        return [r for _, r in frame.faces]


def person(track_id, x1=0.1, y1=0.1, x2=0.4, y2=0.95):
    return Track(label="person", confidence=0.9, track_id=track_id, box=Box(x1=x1, y1=y1, x2=x2, y2=y2))


def face_in(track: Track, who: bytes, facing=True, score=0.95):
    """`who`'s face where that person's head is."""
    x = (track.box.x1 + track.box.x2) / 2 * W - 30
    return who, row(x, track.box.y1 * H + 10, facing=facing, score=score)


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Rig:
    def __init__(self, data_dir: Path, recognizer=..., **settings):
        self.data_dir = data_dir
        self.clock = Clock()
        self.finder = Finder()
        self.store = IdentityStore(data_dir, Vault(KeyRing(data_dir)))
        self.service = IdentityService(self.store, FakeRecognizer() if recognizer is ... else recognizer,
                                       clock=self.clock, now=lambda: T0 + timedelta(seconds=self.clock.t), **settings)

    def look(self, frame, tracks=(), camera="cam-desk", step=1.0):
        self.clock.t += step
        return self.service.look(camera, frame, list(tracks), self.finder)

    def enroll(self, member_id, name, who: bytes, camera="cam-desk"):
        """Switch on if needed and enrol `who` from five single-face frames. Returns the events it caused."""
        out = [] if self.service.enabled else self.service.set_enabled(True)
        self.service.start_enrollment(member_id, name, consent_ack=True)
        for i in range(self.service.frames):
            out += self.look(Frame((who + b"#enrol%d" % i, row(300, 100))), camera=camera)
        assert self.service.enrollment_status()["state"] == "done"
        return out


# --- the face profile: frames whose faces have a look, and uploaded "photos" ---


class Mood(Frame):
    """A frame that also knows how each face looks: `moods` maps who (the bytes before "#") to a label."""

    def __init__(self, *faces, moods=None, **kw):
        super().__init__(*faces, **kw)
        self.moods = moods or {}

    def expression(self, face_row):
        return self.moods.get(self.crop(face_row).split(b"#", 1)[0], "neutral")


def photo(who: str = "ana", n: int = 0, *, w: int = 240, facing: bool = True, score: float = 0.95, brightness: float = 120,
          sharpness: float = 300, others: tuple[str, ...] = (), yaw: float = 0.0, drop: float = 0.5) -> bytes:
    """Stands in for an uploaded JPEG: a description of the picture, which a FakePhotoReader "decodes". Its
    `secret` is the part of the file that must never be found anywhere afterwards."""
    faces = [{"who": f"{who}#photo{n}", "w": w, "facing": facing, "score": score, "yaw": yaw, "drop": drop}]
    faces += [{"who": f"{o}#photo{n}", "w": w, "facing": True, "score": 0.95} for o in others]
    return json.dumps({"secret": f"PIXELS-OF-{who.upper()}-{n}-" + "x" * 40, "faces": faces, "brightness": brightness,
                       "sharpness": sharpness}).encode()


class FakePhotoReader(PhotoReader):
    """Stands in for OpenCV: the "picture" says what is in it and how bright and sharp the face is."""

    def __init__(self):
        self.decoded = 0

    def decode(self, data):
        try:
            raw = json.loads(data)
        except ValueError:
            return None
        self.decoded += 1
        frame = Frame(*[(f["who"].encode(), row(100 + 300 * i, 100, w=f["w"], h=f["w"] * 1.3, facing=f["facing"], score=f["score"]))
                        for i, f in enumerate(raw["faces"])])
        for f, (_, r) in zip(raw["faces"], frame.faces, strict=True):  # the head's pose: where the nose tip sits
            r[8] += f.get("yaw", 0.0) * (r[6] - r[4])
            r[9] = r[5] + f.get("drop", 0.375) * (r[11] - r[5])
        frame.brightness, frame.sharpness = raw["brightness"], raw["sharpness"]
        return frame

    def measure(self, image, face_row):
        return image.brightness, image.sharpness


def kinds(events):
    return [(e.type.rsplit(".", 1)[-1], getattr(e, "track_id", None), getattr(e, "member_id", None)) for e in events
            if e.type.startswith("perception.person")]
