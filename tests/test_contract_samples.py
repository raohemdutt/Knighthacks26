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
DAY_MOVE = {"id": "m0", "kind": "study", "title": "Study at Desk", "start": TS, "end": TS, "minutes": 90, "place": "apartment.desk",
            "place_name": "Desk", "fixed": False, "deadline": "hw4"}
DAY_MOVE_IN = {"slot": "s1", "kind": "study", "place": "apartment.desk", "minutes": 90, "start": TS, "deadline": "hw4"}
DAY_OUTCOME = {"won": True, "stars": 2, "points": 81, "summary": "Win, 2 stars", "sleep_hours": 7.5, "meals": 2,
               "study_minutes": 180, "energy_min": 31.0, "deadline_risk": 0.3}

COACH_SESSION = {
    "block_id": "m0", "title": "Study at Desk", "start": TS, "end": TS, "place": "apartment.desk", "place_name": "Desk",
    "planned_minutes": 120, "measured": True, "desk_minutes": 70, "late_start_minutes": 6, "ended_early_minutes": 0,
    "longest_stretch_minutes": 42, "longest_stretch_start": TS,
    "breaks": [{"start": TS, "end": TS, "minutes": 31, "trigger": "chat",
                "evidence": [{"kind": "chat", "id": "groupme:friends:1760100009", "at": TS, "label": "a message in the group chat"},
                             {"kind": "phone", "id": "ep-cam-desk-1760150000", "at": TS, "label": "the phone was in view"}]},
               {"start": TS, "end": TS, "minutes": 13, "trigger": "none", "evidence": []}],
    "text": "Study at Desk: you planned 120 minutes and were at the desk for 70."}
COACH_CALIBRATION = {"score": 58, "ratio": 1.7, "blocks": 3, "text": "Your plans run 1.7x what happens (calibration 58/100)."}

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
    "agent.rule.toggled": {"rule_id": "package-at-door", "enabled": False},
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
    "day.plan.committed": {"day": "2026-10-11", "plan_id": "plan-2026-10-11-0815", "action_id": "a3"},
    "day.plan.proposed": {"day": "2026-10-11", "plan_id": "plan-2026-10-11-0815", "start": TS, "end": TS,
                          "moves": [DAY_MOVE], "dayrun_moves": [DAY_MOVE_IN], "outcome": DAY_OUTCOME, "by": "atlas",
                          "sources": [{"kind": "item", "id": "hw4", "text": "HW 4 due Tue 11:59 PM", "origin": "canvas"},
                                      {"kind": "habit", "id": "sleep", "text": "slept 5.4 h last night", "origin": "watch"}],
                          "missing": ["circle: not configured"], "text": "Study at the desk until lunch, then judging.",
                          "action_id": "a3"},
    "day.replan.proposed": {"day": "2026-10-11", "plan_id": "plan-2026-10-11-0912", "replaces": "plan-2026-10-11-0815",
                            "why": "At the desk at 9:12; the 9:20 mentor check-in is a 9 minute walk away.", "start": TS,
                            "moves": [DAY_MOVE], "dayrun_moves": [DAY_MOVE_IN], "outcome": DAY_OUTCOME, "action_id": "a4",
                            "asked": False, "for_event": "outlook:kh-mentor"},
    "day.reviewed": {"day": "2026-10-11", "plan_id": "plan-2026-10-11-0815", "score": 62, "followed": ["Study at Desk"],
                     "slipped": ["Eat at Kitchen"], "text": "You followed 62% of what could be measured.",
                     "change_tomorrow": "Put lunch on the calendar before the afternoon block.", "stars": 1, "points": 64,
                     "plan_stars": 2, "streak": 3, "sessions": [COACH_SESSION],
                     "coach_text": "You planned 120 study minutes at the desk and did 70.",
                     "suggestions": ["Hold chat notifications until the break."], "calibration": COACH_CALIBRATION,
                     "block_minutes": 45},
    "day.coach.noted": {"day": "2026-10-11", "plan_id": "plan-2026-10-11-0815", "block_id": "m0",
                        "text": "You did 70 of 120 minutes. Two breaks; the long one came right after the group chat. "
                                "Next study block is 45 minutes.",
                        "session": COACH_SESSION, "suggestions": ["Hold chat notifications until the break."],
                        "calibration": COACH_CALIBRATION,
                        "next_block": {"id": "m2", "title": "Study at Desk", "start": TS, "end": TS, "place": "apartment.desk",
                                       "place_name": "Desk"}},
    "day.status.changed": {"day": "2026-10-11", "plan_id": "plan-2026-10-11-0815", "now": TS, "block_id": "m0",
                           "block_title": "Study at Desk", "block_end": TS, "should_be": "apartment.desk", "should_be_name": "Desk",
                           "seen_at": "apartment.door", "seen_at_name": "Front door", "seen_how": "door camera", "seen_ts": TS,
                           "drift_minutes": 12,
                           "next_block": {"id": "m1", "title": "Eat at Kitchen", "start": TS, "end": TS, "place": "apartment.kitchen",
                                          "place_name": "Kitchen"},
                           "next_event": {"id": "outlook:kh-judging", "title": "Judging", "start": TS, "end": TS, "place": "ucf.eng2",
                                          "place_name": "Engineering II", "walk_minutes": 9, "leave_by": TS},
                           "risk": "leave_now", "deadline_risk": 0.4,
                           "meters": {"meals": 1, "desk_minutes": 95, "recycled": 2, "sleep_hours": 6.2, "sleep_source": "camera"},
                           "reason": "block started"},
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
    "perception.identity.changed": {"enabled": True, "members": ["m-ana"]},
    "perception.observation.recorded": {"camera_id": "cam-desk",
                                        "tracks": [{"track_id": 1, "label": "person", "confidence": 0.9, "box": BOX}]},
    "perception.person.identified": {"camera_id": "cam-desk", "track_id": 1, "member_id": "m-ana"},
    "perception.person.lost": {"camera_id": "cam-desk", "track_id": 1},
    "perception.scene.described": {"camera_id": "cam-desk", "caption": "Keys next to a mug.", "zone_id": "desk",
                                   "objects": [{"label": "keys", "position": "next to the mug"}],
                                   "frame_ref": "frames:cam-desk:1", "model_version": "atlas-base"},
    "perception.zone.event": {"camera_id": "cam-door", "track_id": 2, "label": "package", "zone_id": "door",
                              "kind": "dwell", "dwell_s": 6},
    "play.league.challenge.closed": {"circle_id": "team", "challenge_id": "ch1", "title": "No energy drinks",
                                     "results": [{"member_id": "maya", "display_name": "Maya", "met_days": 3, "days": 3, "done": True}],
                                     "winners": ["maya"], "summary": "No energy drinks, 3 days: Maya made it."},
    "play.league.challenge.started": {"circle_id": "team", "challenge_id": "ch1", "title": "No energy drinks",
                                      "habit_id": "energy_drinks", "direction": "at_most", "target": 0,
                                      "starts": "2026-10-11", "ends": "2026-10-13", "members": ["hemdutt", "maya"],
                                      "started_by": "hemdutt"},
    "play.league.stake.set": {"circle_id": "team", "week_start": "2026-10-05", "stake": {"kind": "pays_for_food"},
                              "set_by": "hemdutt"},
    "play.league.summary.received": {"circle_id": "team", "member_id": "maya", "kind": "habit_summary",
                                     "payload": {"habit_id": "sleep", "day": "2026-10-10", "met": False, "value": 4.5, "streak": 0}},
    "play.league.week.closed": {"circle_id": "team", "week_start": "2026-10-05", "week_end": "2026-10-11",
                                "standings": [{"member_id": "hemdutt", "display_name": "Hemdutt", "rank": 1, "score": 71.4,
                                               "met": 10, "judged": 14, "reasons": ["hit sleep 6 of 7 nights"]},
                                              {"member_id": "maya", "display_name": "Maya", "rank": 2, "score": 28.6,
                                               "met": 4, "judged": 14, "reasons": ["missed sleep 5 of 7 nights"]}],
                                "worst": "maya", "best": "hemdutt", "stake": {"kind": "pays_for_food"},
                                "summary": "Maya had the worst week: 2 of 7 sleep targets. Dinner is on Maya. Hemdutt picks the place."},
    "play.round.scored": {"round_id": "r1", "session_id": "s1", "players": 1, "item": "can", "correct": True,
                          "seconds": 4.2, "path_len": 7, "score": 150},
    "play.round.started": {"round_id": "r1", "session_id": "s1", "players": 1, "item": "can", "target_bin": "recycling",
                           "path": [{"row": 1, "col": 2}]},
    "runtime.backend.health": {"backend": "atlas_model", "status": "ok"},
    "runtime.model.changed": {"model": {"version": "atlas-v3", "base": "Qwen/Qwen3.5-4B"}, "previous_version": "atlas-base"},
    "telemetry.gpu.power": {"node": "gpu-a", "source": "amd-smi", "period_ms": 1000, "socket_w": 201.0, "gfx_mhz": 2350,
                            "mem_mhz": 1258, "throttled": False, "power_field": "average_socket_power",
                            "temps_c": {"edge": 55.0, "hotspot": 68.0}, "received_ts": TS},
    "telemetry.jetson.power": {"node": "jetson", "source": "tegrastats", "period_ms": 1000, "nvpmodel": "MAXN_SUPER",
                               "rails": {"VDD_IN": 12800, "VDD_CPU_GPU_CV": 6900, "VDD_SOC": 2600},
                               "temps_c": {"gpu": 52.0}, "oc_events": {"oc1": 0}},
    "telemetry.request": {"node": "jetson", "request_id": "r1", "started": "2026-10-11T02:11:55-04:00",
                          "first_token": "2026-10-11T02:11:55.600000-04:00", "ended": TS, "tokens_in": 412,
                          "tokens_out": 61, "target": "student", "model": "atlas-base", "engine": "llama.cpp"},
    "telemetry.train.step": {"node": "gpu-b", "run_id": "run-0003", "step": 120, "loss": 1.02, "step_ms": 2480.0,
                             "socket_w": 245.0, "tokens": 4096, "received_ts": TS},
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
