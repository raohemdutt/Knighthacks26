"""S3: how many faces are in view, and how many of them face the camera. Counts only: no pixels leave this function.

    count_faces = FaceCounter()
    count, facing = count_faces(frame)

Uses the same YuNet model as the blur (P6). A face counts as facing when both eyes and the nose were found, the eyes
are far enough apart for the width of the face (a profile squeezes them together) and the nose sits between them.
"""

import math
from typing import Any

from contexts.perception.infrastructure.face_blur import FaceBlur


def is_facing(row: Any, min_score: float = 0.7) -> bool:
    """`row`: YuNet's [x, y, w, h, right eye (x, y), left eye, nose, right mouth corner, left mouth corner, score]."""
    x, y, w, h = (float(v) for v in row[:4])
    rex, rey, lex, ley, nx = (float(v) for v in (row[4], row[5], row[6], row[7], row[8]))
    if float(row[14]) < min_score or w <= 0:
        return False
    eye_dist = math.hypot(rex - lex, rey - ley)
    mid_x = (rex + lex) / 2
    return eye_dist >= 0.38 * w and abs(nx - mid_x) <= 0.4 * eye_dist


class FaceCounter:
    def __init__(self, finder: FaceBlur | None = None, min_facing_score: float = 0.7) -> None:
        self.finder = finder or FaceBlur()
        self.min_facing_score = min_facing_score

    def __call__(self, frame: Any) -> tuple[int, int]:
        rows = self.finder.faces(frame)
        return len(rows), sum(is_facing(r, self.min_facing_score) for r in rows)
