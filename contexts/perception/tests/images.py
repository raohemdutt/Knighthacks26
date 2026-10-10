"""Real JPEGs for tests, made with whichever image library the box has (Pillow or OpenCV). None with neither."""

import io
import random


def have_image_library() -> bool:
    return make_jpeg(8, 8) is not None


def make_jpeg(w: int, h: int, quality: int = 85) -> bytes | None:
    """A textured `w` x `h` picture (soft blobs with some grain, so it compresses roughly like a camera frame)."""
    rng = random.Random(7)
    try:
        from PIL import Image
    except ImportError:
        Image = None
    if Image is not None:
        small = Image.frombytes("RGB", (max(w // 8, 1), max(h // 8, 1)), rng.randbytes(max(w // 8, 1) * max(h // 8, 1) * 3))
        grain = Image.frombytes("RGB", (w, h), rng.randbytes(w * h * 3))
        out = io.BytesIO()
        Image.blend(small.resize((w, h), Image.BICUBIC), grain, 0.08).save(out, "JPEG", quality=quality)
        return out.getvalue()
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    small = np.frombuffer(rng.randbytes(max(w // 8, 1) * max(h // 8, 1) * 3), np.uint8).reshape(max(h // 8, 1), max(w // 8, 1), 3)
    grain = np.frombuffer(rng.randbytes(w * h * 3), np.uint8).reshape(h, w, 3)
    picture = cv2.addWeighted(cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC), 0.92, grain, 0.08, 0)
    ok, jpeg = cv2.imencode(".jpg", picture, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return jpeg.tobytes() if ok else None


def jpeg_size(jpeg: bytes) -> tuple[int, int]:
    """(width, height), read with whichever library is present."""
    try:
        from PIL import Image
    except ImportError:
        import cv2
        import numpy as np

        h, w = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR).shape[:2]
        return w, h
    with Image.open(io.BytesIO(jpeg)) as im:
        return im.size
