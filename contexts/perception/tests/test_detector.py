from contexts.perception.infrastructure.detector import YoloDetector, build_detector, tracks_from


class Arr:
    """Just enough of a tensor: ultralytics gives .tolist() on torch and numpy alike."""

    def __init__(self, data):
        self.data = data

    def tolist(self):
        return self.data

    def __len__(self):
        return len(self.data)


class Boxes:
    def __init__(self, xyxyn, conf, cls, ids=None):
        self.xyxyn, self.conf, self.cls = Arr(xyxyn), Arr(conf), Arr(cls)
        self.id = Arr(ids) if ids is not None else None

    def __len__(self):
        return len(self.xyxyn)


class Result:
    names = {0: "person", 39: "bottle"}

    def __init__(self, boxes):
        self.boxes = boxes


class FakeYolo:
    def __init__(self, result):
        self.result, self.calls = result, []

    def predict(self, frame, **kw):
        self.calls.append(("predict", kw))
        return [self.result]

    def track(self, frame, **kw):
        self.calls.append(("track", kw))
        return [self.result]


def test_boxes_become_normalized_tracks_named_by_class():
    result = Result(Boxes([[0.1, 0.2, 0.5, 0.9], [-0.02, 0.0, 1.03, 0.4]], [0.91, 0.5], [0, 39]))
    tracks = tracks_from(result)
    assert [(t.label, t.track_id, t.confidence) for t in tracks] == [("person", 0, 0.91), ("bottle", 1, 0.5)]
    assert (tracks[0].box.x1, tracks[0].box.y2) == (0.1, 0.9)
    assert (tracks[1].box.x1, tracks[1].box.x2) == (0.0, 1.0)  # clamped to the image


def test_nothing_detected_is_an_empty_list():
    assert tracks_from(Result(Boxes([], [], []))) == []
    assert tracks_from(Result(None)) == []


def test_tracking_mode_uses_the_tracker_ids():
    fake = FakeYolo(Result(Boxes([[0, 0, 0.5, 0.5]], [0.8], [0], ids=[7.0])))
    assert YoloDetector(model=fake)("frame")[0].track_id == 7 and fake.calls[0][0] == "predict"  # ids ride along if given
    assert YoloDetector(model=fake, track=True)("frame")[0].track_id == 7
    assert fake.calls[1][0] == "track" and fake.calls[1][1]["persist"] is True


def test_no_engine_means_no_detector(tmp_path):
    cfg = tmp_path / "models.yaml"
    cfg.write_text(f"detector: {{backend: ultralytics, weights: {(tmp_path / 'missing.engine').as_posix()}}}\n")
    assert build_detector(cfg) is None
