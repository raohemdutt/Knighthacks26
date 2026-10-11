"""The outbound gate and its bypass guard: a picture that leaves the box has its faces scrambled, or it does not leave.

Four parts:
- the gate itself (contexts/vault/application/outbound.py), with a stand-in scrambler, so it runs without OpenCV;
- the two paths that leave the box, driven through the real wiring (contexts/gateway/bootstrap.py) with the blur
  switch OFF: the approved training export and the phone alert's photo (GET /memory/keyframes/<ref>?view=out);
- the pixels: the same two paths with OpenCV and a fake YuNet that reports one face (skipped without OpenCV);
- a static walk over every non-test module in contexts/, runtime/ and jetson/ that fails when a module both touches
  a frame source and has a way to send, export or serve it, unless it is in ALLOWED below with its reason.
"""

import ast
import asyncio
import hashlib
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from contexts.agent.application.memory_port import FakeMemory
from contexts.gateway.app import create_app
from contexts.gateway.bootstrap import Settings, build
from contexts.learning.application.export import DatasetExporter, teach
from contexts.memory.api import build_router
from contexts.perception.infrastructure.face_blur import MODEL, FaceBlur
from contexts.vault import OutboundGate, Refused, jpeg_scrambler
from contexts.workspace.domain.approvals import ActionRequest
from contracts.learning import TaskSpec
from contracts.memory import Episode
from contracts.vault import PolicyChanged

REPO = Path(__file__).resolve().parents[3]
NOW = datetime(2026, 10, 11, 9, 0, tzinfo=UTC)
REF = "frames:cam-desk:1791724365000"
STORED = b"\xff\xd8 a keyframe saved with the blur switch off: a face in the clear"
FACE = (120, 60, 80, 80)  # x, y, w, h of the one face the fake YuNet reports


# --- the gate, with a stand-in for the whole decode / find faces / scramble / encode step (runs without OpenCV) --------


def scrambled(jpeg: bytes) -> bytes:
    return b"\xff\xd8 face scrambled " + hashlib.sha256(jpeg).hexdigest().encode()


def one_face(jpeg: bytes) -> tuple[bytes, int]:
    return scrambled(jpeg), 1


def no_face(jpeg: bytes) -> tuple[bytes, int]:
    return b"\xff\xd8 encoded again", 0


def broken(jpeg: bytes) -> tuple[bytes, int]:
    raise RuntimeError("the detector is down")


def test_a_face_is_scrambled_and_the_release_is_reported():
    seen = []
    out = OutboundGate(one_face, on_release=lambda *a: seen.append(a)).release(STORED, destination="lab", purpose="training export")
    assert out == scrambled(STORED) and seen == [("lab", "training export", 1, len(out), 1)]


def test_a_picture_with_no_face_comes_back_as_the_same_bytes():
    seen = []
    out = OutboundGate(no_face, on_release=lambda *a: seen.append(a)).release(STORED, destination="phone", purpose="alert photo")
    assert out is STORED and seen == [("phone", "alert photo", 1, len(STORED), 0)]


@pytest.mark.parametrize("scramble", [broken, lambda jpeg: (jpeg, 1), lambda jpeg: (b"", 2), lambda jpeg: None])
def test_anything_that_goes_wrong_refuses_and_reports_no_release(scramble):
    """The detector failing, or a face reported and the picture handed back unchanged: nothing comes out."""
    seen = []
    with pytest.raises(Refused):
        OutboundGate(scramble, on_release=lambda *a: seen.append(a)).release(STORED, destination="phone", purpose="alert photo")
    assert seen == []


def test_the_real_scrambler_refuses_what_it_cannot_check(tmp_path):
    """No fallback on the real path: bytes that are not a picture, a missing YuNet file and a machine without OpenCV
    (where even a good picture cannot be decoded) all end in Refused."""
    for model in (MODEL, tmp_path / "missing.onnx"):
        real = OutboundGate(jpeg_scrambler(FaceBlur(model=model).scrambled))
        with pytest.raises(Refused):
            real.release(STORED, destination="phone", purpose="alert photo")
    pytest.importorskip("cv2")
    with pytest.raises(Refused):
        OutboundGate(jpeg_scrambler(FaceBlur(model=tmp_path / "missing.onnx").scrambled)).release(
            picture(), destination="phone", purpose="alert photo")


# --- the two paths that leave the box, through the real wiring, blur switch OFF ----------------------------------------


@pytest.fixture
def box(tmp_path):
    """The app as bootstrap builds it, with "Blur faces before saving" OFF and one unblurred keyframe on disk."""
    atlas = build(Settings(data_dir=tmp_path, connectors=False, ntfy_url=None, llm_url=None))
    asyncio.run(atlas.bus.publish(PolicyChanged(ts=NOW, blur_faces=False)))
    assert atlas.policy.blur_faces is False
    frame = tmp_path / "frames" / "cam-desk" / "1791724365000.jpg"
    frame.parent.mkdir(parents=True)
    frame.write_bytes(STORED)
    return atlas


def export(atlas, tmp_path) -> tuple[str, list[Path]]:
    """Run the wired `export_training_data` executor on an approved request; the result and the pictures it wrote."""
    exporter = atlas.approvals.executors["export_training_data"]
    assert isinstance(exporter, DatasetExporter)
    episode = Episode(id="e1", camera_id="cam-desk", start=NOW - timedelta(hours=1), captions=["at the desk"], keyframe_refs=[REF])
    exporter.memory = FakeMemory([episode])
    exporter.publish = lambda e: None
    _, proposal = teach(TaskSpec(task="desk-objects", instruction="List the objects.", camera="cam-desk"), NOW)
    result = exporter(ActionRequest(id=proposal.action_id, kind=proposal.kind, target=proposal.target, reason=proposal.reason,
                                    params=proposal.params, expires_at=NOW + timedelta(hours=1)))
    return result, sorted((tmp_path / "exports").glob("*/images/*"))


def test_an_export_with_the_blur_switch_off_still_scrambles_the_face(box, tmp_path):
    box.outbound.scramble = one_face
    result, written = export(box, tmp_path)
    assert result.startswith("exported 1 examples as desk-objects-") and "held back" not in result
    assert [f.read_bytes() for f in written] == [scrambled(STORED)]


def test_an_export_leaves_out_a_picture_the_gate_refuses_and_says_so(box, tmp_path):
    box.outbound.scramble = broken
    result, written = export(box, tmp_path)
    assert written == []
    assert result.startswith("exported 0 examples as desk-objects-")
    assert result.endswith("; 1 picture held back (faces could not be checked)")
    [root] = (tmp_path / "exports").iterdir()
    assert (root / "examples.jsonl").read_text() == "" and '"examples": 0' in (root / "manifest.json").read_text()
    assert STORED not in b"".join(f.read_bytes() for f in root.rglob("*") if f.is_file())


def test_an_export_on_a_box_whose_detector_cannot_run_writes_no_picture(box, tmp_path):
    """Nothing swapped in: the gate bootstrap wired, given bytes it cannot read as a picture, holds them back."""
    result, written = export(box, tmp_path)
    assert written == [] and "1 picture held back" in result


def test_the_phone_copy_of_a_keyframe_is_scrambled_with_the_blur_switch_off(box):
    box.outbound.scramble = one_face
    client = TestClient(create_app(atlas=box))
    out = client.get(f"/memory/keyframes/{REF}", params={"view": "out"})
    assert out.status_code == 200 and out.headers["content-type"] == "image/jpeg" and out.headers["cache-control"] == "no-store"
    assert out.content == scrambled(STORED)
    assert client.get(f"/memory/keyframes/{REF}").content == STORED  # the owner's own view is the stored file, as before
    assert client.get("/memory/keyframes/frames:cam-desk:1", params={"view": "out"}).status_code == 404
    assert client.get(f"/memory/keyframes/{REF}", params={"view": "raw"}).status_code == 422


@pytest.mark.parametrize("scramble", [broken, None])  # None: the gate as wired, which cannot read these bytes as a picture
def test_the_phone_copy_is_an_error_and_no_picture_when_the_gate_refuses(box, scramble):
    if scramble:
        box.outbound.scramble = scramble
    out = TestClient(create_app(atlas=box)).get(f"/memory/keyframes/{REF}", params={"view": "out"})
    assert out.status_code == 503 and out.headers["content-type"] == "application/json"
    assert STORED not in out.content and b"\xff\xd8" not in out.content


def test_the_keyframe_route_without_a_gate_serves_no_phone_copy(tmp_path):
    (tmp_path / "cam-desk").mkdir()
    (tmp_path / "cam-desk" / "1.jpg").write_bytes(STORED)
    app = FastAPI()
    app.include_router(build_router(None, tmp_path))
    client = TestClient(app)
    assert client.get("/memory/keyframes/frames:cam-desk:1", params={"view": "out"}).status_code == 503
    assert client.get("/memory/keyframes/frames:cam-desk:1").content == STORED


def test_the_alert_on_the_phone_links_the_gated_copy():
    from contexts.workspace.infrastructure.ntfy import PhoneNotifier
    from contracts.agent import AlertRaised
    from contracts.types import Citation

    sent = []

    class Recorder:
        async def publish(self, topic, message, **kwargs):
            sent.append(kwargs)

    alert = AlertRaised(alert_id="x", rule_id="r", message="Package", evidence=[Citation(frame_ref=REF, ts=NOW)])
    asyncio.run(PhoneNotifier(Recorder(), gateway_url="http://box:8000").on_alert(alert))
    assert sent[0]["attach"] == f"http://box:8000/memory/keyframes/{REF}?view=out"


# --- the pixels, with OpenCV and a fake YuNet that reports one face (skipped on a machine without OpenCV) ---------------


class OneFace:
    def detect(self, frame):
        import numpy as np

        return 1, np.array([[*FACE, *([0.0] * 10), 0.95]], dtype=np.float32)


class NoFace:
    def detect(self, frame):
        return 1, None


def picture() -> bytes:
    """A 320x240 JPEG of fine noise: any scrambled region shows up as a flat patch."""
    import cv2
    import numpy as np

    noise = np.random.default_rng(7).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    ok, jpeg = cv2.imencode(".jpg", noise, [cv2.IMWRITE_JPEG_QUALITY, 95])
    assert ok
    return jpeg.tobytes()


def detail(jpeg: bytes, region: tuple[int, int, int, int]) -> float:
    """How much fine detail is left in a region: the mean step between neighbouring pixels."""
    import cv2
    import numpy as np

    x, y, w, h = region
    grey = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)[y:y + h, x:x + w].astype(np.int16)
    return float(np.abs(np.diff(grey, axis=1)).mean())


def assert_face_scrambled(out: bytes, original: bytes) -> None:
    assert out != original
    assert detail(original, FACE) > 40  # the noise is there to begin with
    assert detail(out, FACE) < 5  # and gone where the face was
    assert detail(out, (0, 160, 320, 80)) > 40  # the rest of the picture is still the picture


def test_the_face_region_is_scrambled_in_the_export_and_in_the_phone_copy(box, tmp_path):
    pytest.importorskip("cv2")
    (tmp_path / "frames" / "cam-desk" / "1791724365000.jpg").write_bytes(picture())
    box.outbound.scramble = jpeg_scrambler(FaceBlur(detector=OneFace()).scrambled)  # bootstrap's wiring, YuNet faked
    _, [written] = export(box, tmp_path)
    assert_face_scrambled(written.read_bytes(), picture())
    client = TestClient(create_app(atlas=box))
    assert_face_scrambled(client.get(f"/memory/keyframes/{REF}", params={"view": "out"}).content, picture())
    assert client.get(f"/memory/keyframes/{REF}").content == picture()


def test_a_frame_with_no_face_leaves_byte_for_byte_as_it_was_stored():
    pytest.importorskip("cv2")
    original = picture()
    gate = OutboundGate(jpeg_scrambler(FaceBlur(detector=NoFace()).scrambled))
    assert gate.release(original, destination="lab", purpose="training export") is original


def test_the_yunet_model_the_gate_needs_ships_in_the_repo_and_loads():
    assert MODEL.is_file() and MODEL.stat().st_size > 100_000
    assert os.path.relpath(MODEL, REPO) == "contexts/perception/models/face_detection_yunet_2023mar.onnx"
    pytest.importorskip("cv2")
    seen = []
    gate = OutboundGate(jpeg_scrambler(FaceBlur().scrambled), on_release=lambda *a: seen.append(a))  # the real YuNet
    assert gate.release(picture(), destination="lab", purpose="training export") == picture()  # noise has no face
    assert seen[0][4] == 0


# --- the bypass guard: no module sends a picture around the gate --------------------------------------------------------

ROOTS = ("contexts", "runtime", "jetson")
NETWORK_CLIENTS = {"httpx", "requests", "urllib.request", "aiohttp", "http.client", "urllib3", "websockets", "smtplib",
                   "ftplib", "socket"}
FRAME_NAMES = {"load_keyframe", "keyframe_loader", "load_image"}  # load_image is the runtime's name for the same loader
PICTURE_TYPES = ("image/jpeg", "image/png", "multipart/x-mixed-replace")

# Every module that both touches a frame source and can send, export or serve it. One line of reason each: why a
# picture with a face cannot leave the box through it unblurred. A new entry is a privacy decision, not a formality.
ALLOWED = {
    "contexts/gateway/bootstrap.py":
        "the wiring: builds the gate and gives the exporter the gated loader (the export tests above run through it)",
    "contexts/learning/application/export.py":
        "writes only what its injected loader returns; bootstrap injects the gate, and a refused picture is left out",
    "contexts/vault/application/wipe.py":
        "names data/frames and data/exports to delete them; it writes and sends nothing",
    "contexts/workspace/infrastructure/ntfy.py":
        "sends a link, not a picture, and the link is the gated ?view=out copy (checked below)",
    "contexts/memory/api.py":
        "the plain route is the owner's own app reading the stored file; ?view=out, the phone alert's copy, is gated",
    "contexts/capture/api.py":
        "the owner's live view (snapshot and motion-JPEG), blurred by the local switch; no alert or export links it",
    "runtime/llama.py":
        "the on-box model call: llama-server on loopback (config/models.yaml, jetson/atlas-gateway.service)",
    "jetson/record_clip.py":
        "the fixture recorder: reads the box's own snapshot route and refuses to record while blur_faces is off",
}


def touches(path: Path) -> tuple[set[str], set[str]]:
    """(frame sources, ways out) a module mentions, read from its syntax tree: names, imports and short strings."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    sources, ways_out, picture_types, serves = set(), set(), set(), False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import | ast.ImportFrom):
            module = getattr(node, "module", None) or ""
            names = {a.name for a in node.names} | {module} | {f"{module}.{a.name}" for a in node.names}
            ways_out |= {f"imports {n}" for n in names & NETWORK_CLIENTS}
            serves = serves or any(n.split(".")[0] == "fastapi" for n in names)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "snapshot" and node.args:
            sources.add(".snapshot(camera)")  # CaptureService.snapshot(camera_id); Policy.snapshot() takes nothing
        elif isinstance(node, ast.Name | ast.arg | ast.keyword | ast.Attribute):
            name = getattr(node, "id", None) or getattr(node, "arg", None) or getattr(node, "attr", None)
            if name in FRAME_NAMES:
                sources.add(name)
            elif isinstance(node, ast.Attribute) and name == "imencode":
                sources.add("cv2.imencode")
            elif isinstance(node, ast.Attribute) and name == "latest" and "reader" in ast.unparse(node.value):
                sources.add("reader.latest")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and len(node.value) < 200 and "\n" not in node.value:
            text = node.value
            parts = text.replace("\\", "/").split("/")
            if "frames" in parts:
                sources.add("a frames directory")
            if "memory/keyframes" in text or "snapshot.jpg" in text:
                sources.add("a picture URL")
            if "exports" in parts:
                ways_out.add("an exports directory")
            picture_types |= {t for t in PICTURE_TYPES if text.startswith(t)}
    if serves and picture_types:
        ways_out.add("serves " + ", ".join(sorted(picture_types)))
    return sources, ways_out


def modules() -> list[Path]:
    found = [p for root in ROOTS for p in sorted((REPO / root).rglob("*.py"))]
    return [p for p in found if "tests" not in p.relative_to(REPO).parts and not p.name.startswith(("test_", "conftest"))]


def flagged() -> dict[str, str]:
    out = {}
    for path in modules():
        sources, ways_out = touches(path)
        if sources and ways_out:
            out[path.relative_to(REPO).as_posix()] = f"touches {sorted(sources)} and {sorted(ways_out)}"
    return out


def test_no_module_sends_a_picture_around_the_outbound_gate():
    found = flagged()
    new = {rel: why for rel, why in found.items() if rel not in ALLOWED}
    assert not new, (
        "these modules touch a frame source and can send, export or serve it. Pass the picture through "
        "contexts.vault OutboundGate.release, or add the module to ALLOWED in this test with the reason no face can "
        f"leave through it unblurred: {new}")
    stale = sorted(set(ALLOWED) - set(found))
    assert not stale, f"ALLOWED lists modules the walk no longer flags; remove them so the list stays honest: {stale}"


def test_the_walk_sees_every_module_and_catches_a_bypass(tmp_path):
    assert len(modules()) > 200  # the walk really covers the tree
    leak = tmp_path / "leak.py"
    for source in ("jpeg = load_keyframe(ref)", "jpeg = store.keyframe_loader(ref)", "ok, jpeg = cv2.imencode('.jpg', frame)",
                   "jpeg = capture.snapshot(camera)", "jpeg = self.reader.latest", "jpeg = (data / 'frames' / name).read_bytes()",
                   "jpeg = open('data/frames/cam/1.jpg', 'rb').read()"):
        for way_out in ("import httpx", "import requests", "import urllib.request", "from urllib import request", "import aiohttp",
                        "import http.client", "out = data_dir / 'exports'", "out = 'data/exports/x.jpg'"):
            leak.write_text(f"{way_out}\n{source}\n")
            assert all(touches(leak)), (source, way_out)
    leak.write_text("from fastapi import Response\n\ndef photo(ref):\n    return Response(load_keyframe(ref), media_type='image/jpeg')\n")
    assert all(touches(leak))
    leak.write_text("import httpx\nstate = policy.snapshot()\n")  # a network client alone is not a picture leaving
    assert touches(leak)[0] == set()


def test_only_the_gated_keyframe_link_is_ever_sent_to_a_phone():
    """ntfy.py is allowed because its link is the gated copy: every keyframe link it builds must say ?view=out."""
    tree = ast.parse((REPO / "contexts/workspace/infrastructure/ntfy.py").read_text())
    links = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n, ast.JoinedStr) and "memory/keyframes" in ast.unparse(n)]
    assert links and all("?view=out" in link for link in links), links


def test_the_recorder_still_refuses_when_blur_is_off():
    """record_clip.py is allowed because of this refusal; if the line goes, so does the reason."""
    source = (REPO / "jetson/record_clip.py").read_text()
    assert 'policy.get("blur_faces") is not True' in source and "nothing is recorded unblurred" in source
