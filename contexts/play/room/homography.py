"""Camera picture -> floor, as a plane homography from four taps. Pure Python: no numpy.

A homography H (3x3) maps a point of the picture to a point of the floor:

    [X, Y, W] = H . [x, y, 1]        floor = (X / W, Y / W)

Four point pairs give eight equations (two per pair) in the eight unknown entries:

    x*h11 + y*h12 + h13 - x*X'*h31 - y*X'*h32 = X'
    x*h21 + y*h22 + h23 - x*Y'*h31 - y*Y'*h32 = Y'

(with h33 = 1, in coordinates centred on the four picture points) solved here by Gaussian elimination with partial
pivoting. It is exact for a flat floor and a camera without lens
distortion; a wide-angle lens bends it a little at the edges of the picture.
"""

Point = tuple[float, float]
Matrix = list[list[float]]

EPS = 1e-9
MIN_TRIANGLE = 1e-6  # twice the area of the smallest triangle three of the four points may make


class Degenerate(ValueError):
    """The four points do not pin a plane down: two coincide or three are in a line."""


def solve(a: list[list[float]], b: list[float]) -> list[float]:
    """x with a.x = b, by Gaussian elimination with partial pivoting. Raises Degenerate when a is singular."""
    n = len(b)
    m = [list(map(float, row)) + [float(b[i])] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < EPS:
            raise Degenerate("the points do not determine a homography")
        m[col], m[pivot] = m[pivot], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            if f:
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))) / m[r][r]
    return x


def _check(points: list[Point], what: str) -> None:
    if len(points) != 4 or any(len(p) != 2 for p in points):
        raise Degenerate(f"{what}: four points of [x, y] are needed")
    for i in range(4):
        (ax, ay), (bx, by), (cx, cy) = (points[j] for j in range(4) if j != i)
        if abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) < MIN_TRIANGLE:
            raise Degenerate(f"{what}: three of the points are in a line (or two are the same)")


def homography(src: list[Point], dst: list[Point]) -> Matrix:
    """The 3x3 matrix taking each of the four `src` points to its `dst` point."""
    _check(src, "picture points")
    _check(dst, "floor points")
    # Solve around the middle of the four picture points. Fixing h33 = 1 assumes the origin does not map to infinity;
    # the picture's own origin can (a camera whose horizon runs through the top-left corner), the middle of four
    # points on the floor cannot.
    sx, sy = sum(p[0] for p in src) / 4, sum(p[1] for p in src) / 4
    a, b = [], []
    for (px, py), (u, v) in zip(src, dst, strict=True):
        x, y = px - sx, py - sy
        a.append([x, y, 1, 0, 0, 0, -x * u, -y * u])
        b.append(u)
        a.append([0, 0, 0, x, y, 1, -x * v, -y * v])
        b.append(v)
    h = solve(a, b)
    # undo the shift: H = H' . T(-sx, -sy)
    rows = [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]]
    out = [[r[0], r[1], r[2] - r[0] * sx - r[1] * sy] for r in rows]
    s = out[2][2]  # the usual form has h33 = 1; when the picture's origin is on the horizon it is 0 and stays so
    return [[v / s for v in row] for row in out] if abs(s) > EPS else out


def apply(h: Matrix, x: float, y: float) -> Point | None:
    """The point (x, y) through `h`; None when it lies on the horizon (it maps to infinity)."""
    w = h[2][0] * x + h[2][1] * y + h[2][2]
    if abs(w) < EPS:
        return None
    return (h[0][0] * x + h[0][1] * y + h[0][2]) / w, (h[1][0] * x + h[1][1] * y + h[1][2]) / w


def invert(h: Matrix) -> Matrix:
    """The inverse map (floor -> picture), by the adjugate; scaled so the last entry is 1 when it can be."""
    (a, b, c), (d, e, f), (g, i, j) = h
    det = a * (e * j - f * i) - b * (d * j - f * g) + c * (d * i - e * g)
    if abs(det) < EPS:
        raise Degenerate("the homography cannot be inverted")
    inv = [[(e * j - f * i) / det, (c * i - b * j) / det, (b * f - c * e) / det],
           [(f * g - d * j) / det, (a * j - c * g) / det, (c * d - a * f) / det],
           [(d * i - e * g) / det, (b * g - a * i) / det, (a * e - b * d) / det]]
    s = inv[2][2]
    return [[v / s for v in row] for row in inv] if abs(s) > EPS else inv
