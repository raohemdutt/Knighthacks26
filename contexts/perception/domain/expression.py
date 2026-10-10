"""Apparent expression, the pure part: what a look is called, the head's pose from five landmarks, and how
single looks become one fact per stretch of time.

"Apparent expression" is how a face looked to a small classifier, one of LABELS. It is a noisy signal about
a face's shape in one frame. It is not an emotion, a mood, a state of health or a measure of attention, and
nothing here says it is: the words for people are "looked tense / tired / upbeat / neutral" (LOOKS).

    pose = head_pose(yunet_row)             # .away (turned from the camera), .down (head tipped down)
    run = ExpressionRun()                   # one per camera and member (or per camera, for the anonymous room)
    for summary in run.add(at, label, confidence, away=pose.away, down=pose.down):
        ...                                 # a stretch that ended: publish it

What the five landmarks (two eyes, nose tip, two mouth corners) can say honestly:
- turned away: the eyes are close together for the width of the face, or the nose is off to one side
  (the same test the face counter uses for "facing");
- head down: the nose tip sits low between the eye line and the mouth line. Straight on it is about half
  way (0.5); tipping the head down by 25 degrees or so moves it to about 0.7 (the nose sticks out, so it
  projects lower). It is relative to the camera: a camera mounted above a desk sees everyone a little "down".
What they cannot say: whether the eyes are closed, or a yawn. There is no eyelid and no open-mouth landmark,
so neither is reported.

No per-frame facts: a run collects looks into windows of `window_s`. A window with too few readable looks is
dropped. A window whose dominant label differs from the last one reported is reported at once (a change);
windows that repeat it are added up and reported together every `heartbeat_s`. A counted look is in at most
one summary (never two), so summaries can be added up again by whoever reads them.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

LABELS = ("angry", "disgust", "fearful", "happy", "neutral", "sad", "surprised")
STRAINED = ("angry", "disgust", "fearful", "sad")  # what the study coach counts as "looked strained"
LOOKS = {"angry": "tense", "disgust": "tense", "fearful": "tense", "sad": "tired", "happy": "upbeat",
         "surprised": "upbeat", "neutral": "neutral"}  # the words for people: "looked ..."
MIN_CONFIDENCE = 0.4  # a look the classifier is less sure of than this is not counted as any expression
WINDOW_S = 30.0
HEARTBEAT_S = 180.0
MIN_SAMPLES = 5  # counted looks a window needs
IDLE_S = 60.0  # no look for this long: the stretch is over
AWAY_EYE_RATIO, AWAY_NOSE_OFF = 0.38, 0.4  # as contexts.perception.infrastructure.face_counter.is_facing
HEAD_DOWN_RATIO = 0.7


@dataclass(frozen=True)
class HeadPose:
    eye_ratio: float  # distance between the eyes / width of the face box
    nose_off: float  # how far the nose is from the middle of the eyes, in eye distances
    nose_drop: float  # where the nose tip sits between the eye line (0) and the mouth line (1)

    @property
    def away(self) -> bool:
        return self.eye_ratio < AWAY_EYE_RATIO or self.nose_off > AWAY_NOSE_OFF

    @property
    def down(self) -> bool:
        return self.nose_drop >= HEAD_DOWN_RATIO


def head_pose(row: Any) -> HeadPose:
    """`row`: YuNet's [x, y, w, h, right eye, left eye, nose, right mouth corner, left mouth corner, score]."""
    w = float(row[2])
    rex, rey, lex, ley, nx, ny, rmx, rmy, lmx, lmy = (float(v) for v in row[4:14])
    eye_dist = ((rex - lex) ** 2 + (rey - ley) ** 2) ** 0.5
    eye_y, mouth_y = (rey + ley) / 2, (rmy + lmy) / 2
    return HeadPose(eye_ratio=eye_dist / w if w > 0 else 0.0,
                    nose_off=abs(nx - (rex + lex) / 2) / eye_dist if eye_dist > 0 else 9.0,
                    nose_drop=(ny - eye_y) / (mouth_y - eye_y) if mouth_y > eye_y else 0.5)


def top(probabilities: dict[str, float]) -> tuple[str, float]:
    """The likeliest label and how sure the classifier is of it."""
    label = max(LABELS, key=lambda k: probabilities.get(k, 0.0))
    return label, float(probabilities.get(label, 0.0))


@dataclass
class Summary:
    """A stretch of looks at one face (or at the anonymous faces of one room), added up."""

    start: datetime
    end: datetime
    counts: dict[str, int] = field(default_factory=dict)  # label -> counted looks
    confidence_sum: float = 0.0
    looks: int = 0  # every look, also those turned away or too unsure to count
    down: int = 0
    away: int = 0
    people: int = 1  # the most faces in one look (the anonymous room only)

    @property
    def samples(self) -> int:
        return sum(self.counts.values())

    @property
    def dominant(self) -> str | None:
        # the most frequent; a tie goes to neutral, then to the alphabet, so it is the same on every run
        return max(self.counts, key=lambda k: (self.counts[k], k == "neutral", k), default=None)

    @property
    def shares(self) -> dict[str, float]:
        n = self.samples
        return {k: round(self.counts[k] / n, 3) for k in LABELS if self.counts.get(k)} if n else {}

    @property
    def confidence(self) -> float:
        return round(self.confidence_sum / self.samples, 2) if self.samples else 0.0

    def add(self, label: str | None, confidence: float, away: bool, down: bool, people: int = 1) -> None:
        self.looks += 1
        self.away += bool(away)
        self.down += bool(down)
        self.people = max(self.people, people)
        if label is not None and confidence >= MIN_CONFIDENCE:
            self.counts[label] = self.counts.get(label, 0) + 1
            self.confidence_sum += confidence

    def merged(self, other: "Summary") -> "Summary":
        counts = {k: self.counts.get(k, 0) + other.counts.get(k, 0) for k in {*self.counts, *other.counts}}
        return Summary(start=min(self.start, other.start), end=max(self.end, other.end), counts=counts,
                       confidence_sum=self.confidence_sum + other.confidence_sum, looks=self.looks + other.looks,
                       down=self.down + other.down, away=self.away + other.away, people=max(self.people, other.people))


class ExpressionRun:
    """Looks in, summaries out. Times are whatever clock the caller uses for `at` (the box uses its own time)."""

    def __init__(self, *, window_s: float = WINDOW_S, heartbeat_s: float = HEARTBEAT_S, min_samples: int = MIN_SAMPLES,
                 idle_s: float = IDLE_S) -> None:
        self.window_s, self.heartbeat_s, self.min_samples, self.idle_s = window_s, heartbeat_s, min_samples, idle_s
        self.window: Summary | None = None  # the window being filled
        self.pending: Summary | None = None  # closed windows that repeat the last reported label, not yet reported
        self.reported: str | None = None  # the dominant label of the last summary handed out

    def add(self, at: datetime, label: str | None, confidence: float = 0.0, *, away: bool = False, down: bool = False,
            people: int = 1) -> list[Summary]:
        out = self.idle(at)
        if self.window is None:
            self.window = Summary(start=at, end=at)
        self.window.add(label, confidence, away, down, people)
        self.window.end = at
        if (at - self.window.start).total_seconds() >= self.window_s:
            out += self._close()
        return out

    def idle(self, at: datetime) -> list[Summary]:
        """Nothing was seen for `idle_s`: what was collected is handed out, and the next look starts afresh."""
        last = self.window.end if self.window else self.pending.end if self.pending else None
        if last is None or (at - last).total_seconds() < self.idle_s:
            return []
        return self.finish()

    def finish(self) -> list[Summary]:
        out = self._close()
        if self.pending is not None:
            out.append(self.pending)
            self.pending = None
        return out

    def _close(self) -> list[Summary]:
        w, self.window = self.window, None
        if w is None or w.samples < self.min_samples:
            return []  # too few readable looks to say anything: dropped, not guessed at
        if w.dominant != self.reported:  # a change: what was waiting goes out, then the window that changed
            out = [self.pending] if self.pending is not None else []
            self.pending, self.reported = None, w.dominant
            return [*out, w]
        self.pending = w if self.pending is None else self.pending.merged(w)
        if (self.pending.end - self.pending.start).total_seconds() >= self.heartbeat_s:
            out, self.pending = [self.pending], None
            return out
        return []
