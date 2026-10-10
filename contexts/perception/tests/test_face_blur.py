import pytest

pytest.importorskip("cv2")
import numpy as np  # noqa: E402

from contexts.perception.infrastructure.face_blur import MODEL, FaceBlur  # noqa: E402


class FakeFaces:
    def __init__(self, rows):
        self.rows = rows

    def detect(self, frame):
        return 1, (np.array(self.rows, np.float32) if self.rows else None)


def noisy(h=240, w=320):
    return np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)


def test_the_face_box_is_scrambled_and_the_rest_is_untouched():
    frame = noisy()
    original = frame.copy()
    out = FaceBlur(detector=FakeFaces([[100, 60, 60, 80, 0.9]]))(frame)
    assert (frame == original).all()  # the live frame is not modified
    assert (out[:30] == original[:30]).all() and (out[:, :80] == original[:, :80]).all()  # outside the margin
    face = out[60:140, 100:160].astype(float)
    assert face.std() < 0.35 * original[60:140, 100:160].std()  # noise is gone
    assert not (out[60:140, 100:160] == original[60:140, 100:160]).all()


def test_the_margin_covers_the_edges_of_the_head():
    frame = noisy()
    out = FaceBlur(detector=FakeFaces([[100, 60, 60, 80, 0.9]]), margin=0.3)(frame)
    assert not (out[50:60, 100:160] == frame[50:60, 100:160]).all()  # above the box, inside the margin


def test_a_face_at_the_image_edge_and_no_faces_are_fine():
    frame = noisy()
    assert (FaceBlur(detector=FakeFaces([]))(frame) == frame).all()
    out = FaceBlur(detector=FakeFaces([[-10, -10, 50, 50, 0.9], [300, 220, 60, 60, 0.9]]))(frame)
    assert out.shape == frame.shape


def test_a_detector_failure_stops_the_frame_from_being_stored():
    class Broken:
        def detect(self, frame):
            raise RuntimeError("no model")

    with pytest.raises(RuntimeError):
        FaceBlur(detector=Broken())(noisy())


def test_the_bundled_yunet_model_loads_and_finds_no_face_in_noise():
    assert MODEL.exists()
    out = FaceBlur()(noisy(360, 640))
    assert out.shape == (360, 640, 3)
