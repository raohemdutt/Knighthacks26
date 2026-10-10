"""P1 benchmark: how many frames per second does the detector sustain, end to end?

    python jetson/bench_detector.py yolo11n.engine data/clip.mp4          # a video file
    python jetson/bench_detector.py yolo11n.engine yolo-test/bus.jpg      # one image, repeated
    python jetson/bench_detector.py yolo11n.engine 0                      # camera index 0

Times the whole `detector(frame)` call (resize, inference, boxes -> Tracks), not just the GPU.
Run from the repo root so `contexts` imports.
"""

import statistics
import sys
import time

import cv2

from contexts.perception.infrastructure.detector import YoloDetector


def frames(source: str, limit: int):
    if source.lower().endswith((".jpg", ".jpeg", ".png")):
        image = cv2.imread(source)
        for _ in range(limit):
            yield image
        return
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    n = 0
    while n < limit:
        ok, frame = cap.read()
        if not ok:
            if source.isdigit():
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop a short clip
            continue
        yield frame
        n += 1


def main() -> None:
    weights, source = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 300
    detector = YoloDetector(weights)
    times, seen = [], 0
    for i, frame in enumerate(frames(source, n + 20)):
        t = time.perf_counter()
        tracks = detector(frame)
        if i >= 20:  # the first frames warm up the engine
            times.append(time.perf_counter() - t)
            seen += len(tracks)
    ms = sorted(times)
    print(f"{len(times)} frames, {len(times) / sum(times):.1f} FPS sustained")
    print(f"per frame: mean {statistics.mean(times) * 1000:.1f} ms, p50 {ms[len(ms) // 2] * 1000:.1f} ms, "
          f"p95 {ms[int(len(ms) * 0.95)] * 1000:.1f} ms, worst {ms[-1] * 1000:.1f} ms; {seen / len(times):.1f} objects/frame")


if __name__ == "__main__":
    main()
