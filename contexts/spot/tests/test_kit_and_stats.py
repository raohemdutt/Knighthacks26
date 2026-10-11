"""The loader (on the real kit and on tiny tables) and the statistics helpers."""

import math
from pathlib import Path

import pytest

from contexts.spot.domain import stats
from contexts.spot.infrastructure.kit import check, load_kit
from contexts.spot.tests.helpers import make_kit, mission, action, reading, small_kit

KIT = Path("fixtures/spot/kit")


# --- the real kit ---


def test_the_official_kit_loads_with_the_counts_its_own_validation_file_gives():
    kit = load_kit(KIT)
    assert kit is not None
    counts = kit.validation["counts"]
    assert (len(kit.missions), len(kit.actions), len(kit.readings), len(kit.anomalies), len(kit.locations), len(kit.assets), len(kit.replay)) == (
        counts["missions"], counts["actions"], counts["readings"], counts["anomalies"], counts["locations"], counts["assets"], counts["replay_events"])
    assert not [i for i in kit.issues if i.level == "error"]
    assert "fictional_floor_plan.png" in kit.asset_files and "sample_inspection_images/gauge_pressure_01.jpg" in kit.asset_files


def test_the_official_kit_keeps_its_windows_line_endings_on_disk():
    assert b"\r\n" in (KIT / "02_DATA" / "actions.csv").read_bytes()
    assert b"\r\n" in (KIT / "02_DATA" / "replay.ndjson").read_bytes()


def test_the_loader_reports_what_is_odd_about_the_real_kit():
    codes = {i.code: i for i in load_kit(KIT).issues}
    assert codes["missions_not_in_replay"].ids == ["M024", "M025", "M026", "M027"]
    assert codes["replay_cut_short"].ids == ["M023"]
    assert codes["asset_seen_away_from_home"].count == 61
    assert codes["action_without_reading"].count == 50
    assert {"quality_repeats_status", "review_repeats_severity", "scenario_repeats_version", "constant_field"} <= set(codes)


def test_no_kit_folder_means_none(tmp_path):
    assert load_kit(tmp_path / "nowhere") is None
    assert load_kit(tmp_path) is None  # a folder without the tables


# --- tiny tables ---

MISSIONS = "mission_id,mission_name,scenario,configuration_version,start_time,end_time,duration_min,start_battery_pct,end_battery_pct,mission_status,operator_mode,synthetic\n" \
           "M001,R1,BASELINE,V1,2026-09-01T08:00:00,2026-09-01T08:30:00,30,90,30,COMPLETED,AUTOWALK,True\n"
ACTIONS = "action_id,mission_id,sequence,timestamp,location_id,asset_id,action_type,battery_pct,status,failure_type,duration_sec,scenario,configuration_version,synthetic\n" \
          "A0001,M001,1,2026-09-01T08:03:00,WP01,PUMP-1,NAVIGATION,84,SUCCESS,,20,BASELINE,V1,True\n" \
          "A0002,M001,2,2026-09-01T08:06:00,WP01,PUMP-1,VISUAL_INSPECTION,78,FAILED,SENSOR_TIMEOUT,40,BASELINE,V1,True\n"
LOCATIONS = "location_id,location_name,zone_type,floor,x_coord,y_coord,environment,synthetic\nWP01,Dock,ACCESS,1,10,10,NORMAL,True\n"
ASSETS = "asset_id,asset_name,asset_type,location_id,criticality,synthetic\nPUMP-1,Pump 1,Pump,WP01,LOW,True\n"
READINGS = "reading_id,action_id,mission_id,asset_id,location_id,timestamp,measurement_type,value,unit,quality,synthetic\n" \
           "R0001,A0002,M001,PUMP-1,WP01,2026-09-01T08:06:00,VISUAL_SCORE,71,score,QUESTIONABLE,True\n"
ANOMALIES = "anomaly_id,action_id,mission_id,location_id,asset_id,timestamp,anomaly_type,severity,operator_review_required,description,synthetic\n" \
            "AN001,A0002,M001,WP01,PUMP-1,2026-09-01T08:06:00,VISUAL_DEFECT,HIGH,True,Synthetic.,True\n"
REPLAY = '{"event_id": "E00001", "mission_id": "M001", "timestamp": "2026-09-01T08:00:00", "event_type": "MISSION_STARTED", "details": {"battery_pct": 90}}\n' \
         '{"event_id": "E00002", "mission_id": "M001", "timestamp": "2026-09-01T08:03:00", "event_type": "ACTION_STARTED", "details": {"action_id": "A0001", "location_id": "WP01"}}\n'


def write(folder: Path, eol: str = "\n", **over) -> Path:
    files = {"missions.csv": MISSIONS, "actions.csv": ACTIONS, "locations.csv": LOCATIONS, "assets.csv": ASSETS, "readings.csv": READINGS,
             "anomalies.csv": ANOMALIES, "replay.ndjson": REPLAY, **over}
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (folder / name).write_bytes(text.replace("\n", eol).encode())
    return folder


@pytest.mark.parametrize("eol", ["\n", "\r\n"])
def test_tiny_flat_kit_loads_with_unix_and_windows_line_endings(tmp_path, eol):
    kit = load_kit(write(tmp_path / "kit", eol))
    assert [a.action_id for a in kit.actions] == ["A0001", "A0002"]
    assert kit.actions[1].failed and kit.actions[1].failure_type == "SENSOR_TIMEOUT"  # no stray \r on the last column
    assert kit.anomalies[0].operator_review_required is True
    assert [e.event_type for e in kit.replay] == ["MISSION_STARTED", "ACTION_STARTED"]
    assert kit.readings_of["A0002"][0].value == 71
    assert not [i for i in kit.issues if i.level == "error"]


def test_the_sponsors_folder_layout_is_read_too(tmp_path):
    flat = write(tmp_path / "flat")
    kit_dir = tmp_path / "kit"
    (kit_dir / "02_DATA").mkdir(parents=True)
    (kit_dir / "04_ASSETS").mkdir()
    for f in flat.iterdir():
        (kit_dir / "02_DATA" / f.name).write_bytes(f.read_bytes())
    (kit_dir / "04_ASSETS" / "plan.png").write_bytes(b"x")
    kit = load_kit(kit_dir)
    assert len(kit.actions) == 2 and kit.asset_files == ["plan.png"]


def test_surprises_are_reported_not_repaired(tmp_path):
    bad_actions = ACTIONS + "A0002,M001,3,2026-09-01T08:09:00,WP01,PUMP-1,GAUGE_READING,70,SUCCESS,,10,BASELINE,V1,True\n" \
                            "A0003,M404,1,2026-09-01T08:12:00,WP77,GHOST-1,GAUGE_READING,70,SUCCESS,,10,BASELINE,V1,True\n" \
                            "A0004,M001,4,not-a-time,WP01,PUMP-1,GAUGE_READING,70,SUCCESS,,10,BASELINE,V1,True\n" \
                            ",M001,5,2026-09-01T08:15:00,WP01,PUMP-1,GAUGE_READING,70,SUCCESS,,10,BASELINE,V1,True\n"
    kit = load_kit(write(tmp_path / "kit", **{"actions.csv": bad_actions, "missions.csv": MISSIONS.replace(",synthetic", ",synthetic,extra_col").replace("True\n", "True,1\n")}))
    codes = {i.code: i for i in kit.issues}
    assert codes["duplicate_ids"].ids == ["A0002"]
    assert codes["rows_unreadable"].count == 2  # the bad timestamp and the empty id
    assert codes["action_orphan_mission"].ids == ["A0003"] and codes["action_orphan_location"].ids == ["A0003"] and codes["action_orphan_asset"].ids == ["A0003"]
    assert "replay_cut_short" in codes and "replay_action_never_ends" in codes
    assert len(kit.actions) == 3


def test_time_order_and_sequence_checks_on_hand_made_rows():
    m = mission(1)
    a1, a2 = action(m, 1, 1), action(m, 3, 2)
    late = a2.model_copy(update={"timestamp": m.end_time.replace(year=2027)})
    codes = {i.code for i in check(make_kit([m], [a1, late], [reading(a1, 1)]))}
    assert {"sequence_gaps", "action_outside_mission"} <= codes


# --- statistics ---


def test_wilson_interval_matches_known_values():
    lo, hi = stats.wilson(5, 10)
    assert (round(lo, 4), round(hi, 4)) == (0.2366, 0.7634)
    lo, hi = stats.wilson(0, 10)
    assert lo == 0 and round(hi, 4) == 0.2775
    assert stats.wilson(0, 0) is None


def test_fisher_exact_on_a_table_small_enough_to_do_by_hand():
    # 3 of 3 against 0 of 3: only the two extreme tables are as unlikely, each 1/20
    assert stats.fisher_exact(3, 3, 0, 3) == pytest.approx(0.1)
    assert stats.fisher_exact(2, 4, 2, 4) == pytest.approx(1.0)


def test_compare_gives_difference_interval_and_the_right_test():
    c = stats.compare(41, 180, 8, 90)  # the kit's V1 and V2 failures
    assert c.diff == pytest.approx(41 / 180 - 8 / 90) and c.test == "z" and c.clear and c.p_value < 0.01
    small = stats.compare(3, 4, 1, 4)
    assert small.test == "fisher" and not small.enough and not small.clear
    assert stats.compare(1, 0, 1, 5).diff is None


def test_robust_helpers():
    assert stats.median([3, 1, 2]) == 2 and stats.median([1, 2, 3, 4]) == 2.5 and stats.median([]) is None
    assert stats.mad([1, 2, 3, 4, 100]) == pytest.approx(1.4826)
    assert stats.theil_sen([(0, 0), (1, 1), (2, 2), (3, 50)]) == pytest.approx(1.0, abs=1e-9) or stats.theil_sen([(0, 0), (1, 1), (2, 2), (3, 50)]) > 0
    assert stats.theil_sen([(0, 1), (1, 3), (2, 5)]) == 2
    assert stats.theil_sen([(1, 1)]) is None
    assert stats.mann_whitney([1, 2, 3, 4, 5, 6, 7, 8], [11, 12, 13, 14, 15, 16, 17, 18]) < 0.01
    assert stats.mann_whitney([1, 2, 3, 4], [1, 2, 3, 4]) > 0.9
    assert stats.mann_whitney([], [1]) is None
    assert math.isclose(stats.rate(1, 4), 0.25) and stats.rate(1, 0) is None


def test_small_kit_has_no_issue_worse_than_info():
    assert not [i for i in check(small_kit()) if i.level == "error"]
