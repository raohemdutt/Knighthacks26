"""A scan frame's face row reaches the recogniser as a plain list (scaled back from the detector's small copy).
OpenCV's alignCrop takes only an array: on the box every scan frame was refused with cv2.error and no scan finished."""

import pytest

np = pytest.importorskip("numpy")

from contexts.perception.infrastructure.recognizer import SFaceRecognizer  # noqa: E402


class Net:
    def __init__(self):
        self.rows = []

    def alignCrop(self, frame, row):  # noqa: N802 (OpenCV's name)
        if not isinstance(row, np.ndarray):
            raise TypeError("face_box is not a numpy array")
        self.rows.append(row)
        return "crop"


@pytest.mark.parametrize("row", [[10.0, 20.0, 80.0, 90.0] + [0.0] * 11, tuple(range(15)), np.arange(15, dtype=np.float64)])
def test_the_crop_hands_alignCrop_a_float32_array_whatever_the_row_came_as(row):  # noqa: N802
    net = Net()
    assert SFaceRecognizer(net=net).crop("frame", row) == "crop"
    assert net.rows[0].dtype == np.float32 and net.rows[0].shape == (15,)
