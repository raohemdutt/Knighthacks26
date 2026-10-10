"""One realistic sample payload per event type (F2: "every event in the catalog has a
model in contracts/, with a validation test on a sample payload").

Adding an event type without a sample here fails the first test, on purpose.
"""

import json

import pytest

from contracts import REGISTRY, parse_event

TS = "2026-10-11T02:12:00-04:00"
BOX = {"x1": 0.1, "y1": 0.2, "x2": 0.3, "y2": 0.4}
PROPOSAL = {"title": "Study session", "start": "2026-10-11T14:00:00-04:00", "end": "2026-10-11T16:00:00-04:00",
            "invitees": ["hemdutt"]}

SAMPLES: dict[str, dict] = {
    "agent.action.proposed": {"action_id": "a1", "kind": "send_message", "target": "dom", "reason": "Package at the door",
                              "params": {"text": "Package at the door"}},
    "agent.alert.raised": {"alert_id": "al1", "rule_id": "package-at-door", "message": "Package left at the door",
                           "evidence": [{"frame_ref": "frames:cam-door:1728627120000", "ts": TS}]},
    "agent.answer.generated": {"question_id": "q1", "answer": "Next to the coffee mug.", "model_version": "atlas-base",
                               "citations": [{"frame_ref": "frames:cam-desk:1", "url": None}]},
    "agent.note.written": {"note_id": "n1", "hour": TS, "text": "Coding, then pizza.", "episode_ids": ["e1"]},
    "agent.plan.created": {"plan_id": "p1", "subject": "COP 3502 Exam 1", "text": "- Study recursion 3-5 PM",
                           "model_version": "atlas-base", "blocks": [{"title": "Study", "start": TS, "end": TS}]},
    "agent.rule.added": {"rule": {"id": "bag", "when": {"label": ["backpack"], "zone": "desk", "kind": "dwell", "seconds": 60},
                                  "notify": "Bag left", "cooldown_s": 300}, "source_text": "tell me if a bag is left"},
    "capture.camera.health": {"camera_id": "cam-door", "status": "ok", "fps_actual": 14.8},
    "capture.frames.captured": {"camera_id": "cam-door", "seq": 42, "width": 1280, "height": 720, "motion_score": 0.3},
    "circle.event.agreed": {"circle_id": "team", "thread_id": "t1", "event": PROPOSAL, "attendees": ["dom", "hemdutt"]},
    "circle.message.received": {"message_id": "m1", "circle_id": "team", "thread_id": "t1", "from_member": "hemdutt",
                                "kind": "reply", "reply": {"answer": "free"}},
    "circle.place.shared": {"circle_id": "team", "member_id": "dom", "place": "Library", "source": "phone",
                            "expires_at": "2026-10-11T04:12:00-04:00"},
    "circle.share.queued": {"share_id": "s1", "circle_id": "team", "kind": "impact_count", "payload": {"recycled": 3}},
    "circle.share.synced": {"share_id": "s1", "circle_id": "team"},
    "connector.item.synced": {"connector": "canvas", "kind": "assignment", "external_id": "hw4", "title": "HW 4",
                              "when": "2026-10-13T23:59:00-04:00", "url": "https://webcourses.ucf.edu/a/4"},
    "connector.sync.failed": {"connector": "canvas", "error": "401 Unauthorized"},
    "experience.approval.decided": {"action_id": "a1", "decision": "approve"},
    "experience.correction.submitted": {"subject_ref": "q1", "verdict": "down", "note": "wrong mug"},
    "experience.question.asked": {"question_id": "q1", "text": "Where did I leave my keys?"},
    "habits.day.closed": {"habit_id": "sleep", "day": "2026-10-10", "met": False, "value": 4.5, "streak": 0},
    "habits.nudge.raised": {"nudge_id": "nu1", "habit_id": "meals", "message": "Time to eat something."},
    "habits.signal.recorded": {"kind": "meal", "value": 1},
    "impact.count.updated": {"items_seen": 5, "recyclable_seen": 3, "recycled": 2, "contamination": 1, "diversion_rate": 0.667},
    "impact.item.disposed": {"item_id": "cam-door:4", "label": "can", "zone": "recycling", "bin": "recycling", "correct": True},
    "impact.item.seen": {"item_id": "cam-door:4", "label": "can", "camera_id": "cam-door", "recyclable": True},
    "learning.dataset.exported": {"task": "desk-objects", "dataset_id": "desk-objects-1a2b", "examples": 120, "action_id": "a2"},
    "learning.model.deployed": {"model_version": "atlas-v3-desk-objects", "previous_version": "atlas-base"},
    "learning.run.evaluated": {"task": "desk-objects", "run_id": "run-0003",
                               "scores": {"current": 0.41, "candidate": 0.78, "teacher": 0.86},
                               "decision": "deploy", "model_version": "atlas-v3-desk-objects"},
    "learning.run.progress": {"task": "desk-objects", "run_id": "run-0003", "stage": "training", "progress": 0.4, "loss": 1.2},
    "learning.task.created": {"task": "desk-objects", "instruction": "List small objects on the desk as JSON."},
    "memory.episode.closed": {"episode_id": "e1", "camera_id": "cam-desk", "start": TS, "end": TS, "labels": ["person"],
                              "captions": ["A person types at the desk."]},
    "memory.episode.opened": {"episode_id": "e1", "camera_id": "cam-desk"},
    "perception.faces.counted": {"camera_id": "cam-door", "count": 1, "facing": 1},
    "perception.observation.recorded": {"camera_id": "cam-desk",
                                        "tracks": [{"track_id": 1, "label": "person", "confidence": 0.9, "box": BOX}]},
    "perception.scene.described": {"camera_id": "cam-desk", "caption": "Keys next to a mug.", "zone_id": "desk",
                                   "objects": [{"label": "keys", "position": "next to the mug"}],
                                   "frame_ref": "frames:cam-desk:1", "model_version": "atlas-base"},
    "perception.zone.event": {"camera_id": "cam-door", "track_id": 2, "label": "package", "zone_id": "door",
                              "kind": "dwell", "dwell_s": 6},
    "play.round.scored": {"round_id": "r1", "session_id": "s1", "players": 1, "item": "can", "correct": True,
                          "seconds": 4.2, "path_len": 7, "score": 150},
    "play.round.started": {"round_id": "r1", "session_id": "s1", "players": 1, "item": "can", "target_bin": "recycling",
                           "path": [{"row": 1, "col": 2}]},
    "runtime.backend.health": {"backend": "atlas_model", "status": "ok"},
    "runtime.model.changed": {"model": {"version": "atlas-v3", "base": "Qwen/Qwen3.5-4B"}, "previous_version": "atlas-base"},
    "vault.policy.changed": {"blur_faces": True, "paused_cameras": [], "offline": False},
    "vault.retention.changed": {"keep_days": 7},
    "vault.wipe.requested": {"scope": "keyframes"},
    "workspace.action.completed": {"action_id": "a1", "status": "done", "result": "sent to dom"},
    "workspace.approval.requested": {"action_id": "a1", "kind": "send_message", "target": "dom",
                                     "reason": "Package at the door", "expires_at": TS},
}


def test_every_registered_event_has_a_sample():
    assert set(SAMPLES) == set(REGISTRY), f"missing: {set(REGISTRY) - set(SAMPLES)}, stale: {set(SAMPLES) - set(REGISTRY)}"


@pytest.mark.parametrize("type_name", sorted(SAMPLES))
def test_sample_validates_and_round_trips_through_the_log(type_name):
    line = json.dumps({"event_id": "x", "type": type_name, "version": 1, "ts": TS, **SAMPLES[type_name]})
    event = parse_event(line)
    assert isinstance(event, REGISTRY[type_name])
    assert parse_event(event.model_dump_json()) == event
