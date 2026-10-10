"""Uploaded enrolment photos: bytes to a picture in memory, and two numbers about how usable a face is.

    reader = build_photo_reader()                 # None without OpenCV
    image = reader.decode(jpeg_or_png_bytes)      # None when it is not a picture this box can read
    small, scale = reader.small(image)            # a copy for the face detector; `scale` maps its pixels back
    brightness, sharpness = reader.measure(image, yunet_row)

Nothing here writes a file, and nothing here logs: a photo exists only as the bytes the request carried and
the array decoded from them, both dropped when the request is answered.

- Only JPEG and PNG, and only up to MAX_PIXELS: the size is read from the file's header before anything is
  decoded, so a small file that claims to be a huge picture is refused instead of filling the box's memory.
- A decoded picture is shrunk to MAX_SIDE on its long side (a phone photo is 12 megapixels; the face models
  want a face of about 112 pixels). The detector looks at a DETECT_SIDE copy first: YuNet is tuned for faces
  up to a few hundred pixels and can miss one that fills a large picture.
- brightness: the mean grey level of the face box, 0 to 255.
- sharpness: the variance of the Laplacian of the face box scaled to 112x112 grey, so the number does not
  depend on how many pixels the phone has. A blurred or out-of-focus face has few edges and scores low.
"""

import struct
from typing import Any

MAX_PIXELS = 50_000_000
MAX_SIDE = 1600
DETECT_SIDE = 640
FACE_SIDE = 112


def image_size(data: bytes) -> tuple[int, int] | None:
    """(width, height) from a JPEG's or PNG's header, or None for anything else. Pure: no decoding."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR" and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 9 <= len(data):
        if data[i] != 0xFF:
            return None
        marker = data[i + 1]
        if marker == 0xFF:  # padding
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:  # markers without a length
            i += 2
            continue
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):  # a start-of-frame segment
            height, width = struct.unpack(">HH", data[i + 5:i + 9])
            return width, height
        if length < 2:
            return None
        i += 2 + length
    return None


class PhotoReader:
    """The port."""

    def decode(self, data: bytes) -> Any | None:
        raise NotImplementedError

    def small(self, image: Any) -> tuple[Any, float]:
        """(a copy no larger than DETECT_SIDE, its size relative to `image`)."""
        return image, 1.0

    def measure(self, image: Any, face_row: Any) -> tuple[float, float]:
        """(brightness 0..255, sharpness) of the face `face_row` (a YuNet row, in `image`'s pixels)."""
        raise NotImplementedError


class CvPhotoReader(PhotoReader):
    def decode(self, data: bytes) -> Any | None:
        import cv2
        import numpy as np

        size = image_size(data)
        if size is None or size[0] < 1 or size[1] < 1 or size[0] * size[1] > MAX_PIXELS:
            return None
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)  # upright: EXIF rotation is applied
        return None if image is None else self._fit(image, MAX_SIDE)[0]

    def small(self, image: Any) -> tuple[Any, float]:
        return self._fit(image, DETECT_SIDE)

    @staticmethod
    def _fit(image: Any, side: int) -> tuple[Any, float]:
        import cv2

        h, w = image.shape[:2]
        scale = min(1.0, side / max(h, w))
        if scale >= 1.0:
            return image, 1.0
        return cv2.resize(image, (max(round(w * scale), 1), max(round(h * scale), 1)), interpolation=cv2.INTER_AREA), scale

    def measure(self, image: Any, face_row: Any) -> tuple[float, float]:
        import cv2

        height, width = image.shape[:2]
        x, y, w, h = (float(v) for v in face_row[:4])
        x1, y1 = max(int(x), 0), max(int(y), 0)
        x2, y2 = min(int(x + w), width), min(int(y + h), height)
        if x2 - x1 < 2 or y2 - y1 < 2:
            return 0.0, 0.0
        grey = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        grey = cv2.resize(grey, (FACE_SIDE, FACE_SIDE), interpolation=cv2.INTER_AREA)
        return float(grey.mean()), float(cv2.Laplacian(grey, cv2.CV_64F).var())


def build_photo_reader() -> PhotoReader | None:
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        return None
    return CvPhotoReader()
