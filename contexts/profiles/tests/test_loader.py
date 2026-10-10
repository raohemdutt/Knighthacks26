"""The profile loader: layering, defaults, errors, and the student profile being today's box unchanged."""

from datetime import time
from pathlib import Path

import pytest
import yaml

from contexts.habits.domain.config import load_config as load_habits
from contexts.play.dayrun.config import DayRunConfig
from contexts.profiles import ProfileError, Vocabulary, available, load_profile, merge, profile_name

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config"


def write(folder: Path, name: str, data) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(yaml.safe_dump(data), encoding="utf-8")


def config_dir(tmp_path: Path) -> Path:
    """A small config dir: base habits and dayrun files, plus a profile that overrides parts of them."""
    cfg = tmp_path / "config"
    write(cfg, "agent.yaml", {"sleep": {"timezone": "America/Chicago", "bedtime": "22:00", "wake": "06:00"}, "places": ["Lab"]})
    write(cfg, "habits.yaml", {"habits": [
        {"id": "sleep", "name": "Sleep", "signal": "away_window", "target": 6, "direction": "at_least", "nudge": "slept {value} at the {workplace}"},
        {"id": "meals", "name": "Meals", "signal": "meal", "target": 2, "direction": "at_least", "nudge": "ate {value}"},
    ]})
    write(cfg, "dayrun.yaml", {"pack": "ucf", "win": {"sleep_hours": 6, "meals": 2}, "meal_words": ["lunch"]})
    write(cfg, "sorting.yaml", {"bins": {"trash": "trash"}, "items": {"cup": "trash"}})
    prof = cfg / "profiles" / "lab"
    write(prof, "profile.yaml", {"name": "lab", "description": "A shared lab", "audience": "a team",
                                 "vocabulary": {"workplace": "bench", "activity": "work", "event": "run", "deadline": "milestone"},
                                 "meters": {"habits": ["sleep", "tidy"]}})
    write(prof, "habits.yaml", {"habits": [
        {"id": "sleep", "target": 7, "nudge": "bench empty {value}"},  # replaces two keys of the base habit
        {"id": "tidy", "name": "Tidy", "signal": "disposal", "target": 90, "direction": "at_least", "nudge": "tidy {value}"},
    ]})
    write(prof, "dayrun.yaml", {"win": {"meals": 3}, "meal_words": ["lunch", "pizza"]})
    write(prof, "sorting.yaml", {"bins": {"trash": "trash"}, "items": {"pipette": "trash"}})
    write(cfg / "profiles" / "student", "profile.yaml", {"name": "student"})
    return cfg


def test_merge_layers_mappings_lists_by_id_and_replaces_the_rest():
    base = {"a": 1, "nested": {"x": 1, "y": 2}, "items": [{"id": "p", "v": 1}, {"id": "q", "v": 2}], "words": ["a"]}
    over = {"nested": {"y": 3}, "items": [{"id": "q", "v": 9}, {"id": "r", "v": 3}], "words": ["b"], "new": True}
    out = merge(base, over)
    assert out == {"a": 1, "nested": {"x": 1, "y": 3}, "items": [{"id": "p", "v": 1}, {"id": "q", "v": 9}, {"id": "r", "v": 3}],
                   "words": ["b"], "new": True}
    assert base["nested"] == {"x": 1, "y": 2}  # the base is never changed


def test_a_profile_layers_its_files_over_the_base_and_reports_its_overrides(tmp_path):
    p = load_profile(config_dir(tmp_path), "lab")
    assert p.name == "lab" and p.description == "A shared lab" and p.spec.audience == "a team"
    assert p.vocabulary.event == "run" and p.vocabulary.plural("event") == "runs" and p.vocabulary.words()["Activity"] == "Work"
    assert p.overrides == {"habits.yaml": "profiles/lab/habits.yaml", "dayrun.yaml": "profiles/lab/dayrun.yaml",
                           "sorting.yaml": "profiles/lab/sorting.yaml"}
    # whole-file overrides pick the profile's file, or the base one when the profile has none
    assert p.path("sorting.yaml") == p.folder / "sorting.yaml" and p.path("rules.yaml") == p.config_dir / "rules.yaml"
    # layered: the base's keys stay unless the profile sets them
    raw = p.raw("dayrun.yaml")
    assert raw == {"pack": "ucf", "win": {"sleep_hours": 6, "meals": 3}, "meal_words": ["lunch", "pizza"]}
    assert p.agent()["places"] == ["Lab"]  # no agent.yaml in the profile: the base one, untouched


def test_habits_and_dayrun_loaders_take_the_profile(tmp_path):
    cfg = config_dir(tmp_path)
    p = load_profile(cfg, "lab")
    habits = load_habits(cfg / "habits.yaml", cfg / "agent.yaml", profile=p)
    assert [h.id for h in habits.habits] == ["sleep", "tidy"]  # meters.habits dropped "meals" and kept the new one
    sleep = habits.by_id("sleep")
    assert sleep.target == 7 and sleep.nudge == "bench empty {value}" and sleep.signal == "away_window"  # merged, not replaced
    assert habits.vocabulary.workplace == "bench" and habits.schedule.bedtime == time(22)
    dayrun = DayRunConfig.load(cfg / "dayrun.yaml", cfg / "agent.yaml", profile=p)
    assert dayrun.win.meals == 3 and dayrun.win.sleep_hours == 6 and dayrun.meal_words == ["lunch", "pizza"]
    assert dayrun.vocabulary.activity == "work" and str(dayrun.tz) == "America/Chicago"


def test_a_shift_moves_the_sleep_window_in_agent_and_dayrun(tmp_path):
    cfg = config_dir(tmp_path)
    write(cfg / "profiles" / "nights", "profile.yaml", {"name": "nights", "shift": {"start": "22:00", "end": "06:00",
                                                                                     "sleep_after_minutes": 90, "sleep_hours": 7}})
    p = load_profile(cfg, "nights")
    assert p.agent()["sleep"] == {"timezone": "America/Chicago", "bedtime": "07:30", "wake": "14:30"}
    habits = load_habits(cfg / "habits.yaml", cfg / "agent.yaml", profile=p)
    assert (habits.schedule.bedtime, habits.schedule.wake) == (time(7, 30), time(14, 30))
    dayrun = DayRunConfig.load(cfg / "dayrun.yaml", cfg / "agent.yaml", profile=p)
    assert (dayrun.sleep.bedtime, dayrun.sleep.wake) == (time(7, 30), time(14, 30))


def test_selection_argument_then_env_then_agent_yaml_then_student(tmp_path, monkeypatch):
    cfg = config_dir(tmp_path)
    monkeypatch.delenv("ATLAS_PROFILE", raising=False)
    assert profile_name(cfg) == "student"
    write(cfg, "agent.yaml", {"profile": "lab"})
    assert profile_name(cfg) == "lab"
    monkeypatch.setenv("ATLAS_PROFILE", "student")
    assert profile_name(cfg) == "student"
    assert profile_name(cfg, "lab") == "lab"
    assert available(cfg) == ["lab", "student"]


def test_an_unknown_profile_is_a_clear_error_that_lists_what_exists(tmp_path):
    cfg = config_dir(tmp_path)
    with pytest.raises(ProfileError, match=r"no profile 'nurse'.*have: lab, student.*ATLAS_PROFILE"):
        load_profile(cfg, "nurse")
    with pytest.raises(ProfileError, match="must be 1 to 40"):
        load_profile(cfg, "../etc")
    write(cfg / "profiles" / "broken", "profile.yaml", {"name": "broken", "vocabulary": {"colour": "red"}})
    with pytest.raises(ProfileError, match="profile.yaml"):
        load_profile(cfg, "broken")


def test_a_box_with_no_profiles_folder_is_the_student(tmp_path, monkeypatch):
    monkeypatch.delenv("ATLAS_PROFILE", raising=False)
    cfg = tmp_path / "config"
    write(cfg, "agent.yaml", {})
    p = load_profile(cfg)
    assert p.name == "student" and p.folder is None and p.overrides == {} and p.vocabulary == Vocabulary()
    assert p.raw("habits.yaml") == {} and p.path("sorting.yaml") == cfg / "sorting.yaml"
    with pytest.raises(ProfileError, match="have: student"):
        load_profile(cfg, "lab")


def test_the_repo_student_profile_changes_nothing(monkeypatch):
    """The student profile is today's box: the loaders give the same settings with it as without it."""
    monkeypatch.delenv("ATLAS_PROFILE", raising=False)
    p = load_profile(CONFIG)
    assert p.name == "student" and p.overrides == {} and p.vocabulary == Vocabulary()
    plain = load_habits(CONFIG / "habits.yaml", CONFIG / "agent.yaml")
    with_profile = load_habits(CONFIG / "habits.yaml", CONFIG / "agent.yaml", profile=p)
    assert with_profile.model_dump() == plain.model_dump()
    assert DayRunConfig.load(CONFIG / "dayrun.yaml", CONFIG / "agent.yaml", profile=p).model_dump() == \
        DayRunConfig.load(CONFIG / "dayrun.yaml", CONFIG / "agent.yaml").model_dump()
    assert p.agent() == yaml.safe_load((CONFIG / "agent.yaml").read_text())
    assert p.places_dirs() == [CONFIG / "places"] and p.scenarios_dir() is None
    for name in ("sorting.yaml", "rules.yaml"):
        assert p.path(name) == CONFIG / name


@pytest.mark.parametrize("name", ["shift-worker", "family-home"])
def test_the_shipped_profiles_load_and_their_scenarios_parse(name):
    from contexts.play.dayrun.service import Scenario

    p = load_profile(CONFIG, name)
    assert p.description and p.vocabulary != Vocabulary()
    habits = load_habits(CONFIG / "habits.yaml", CONFIG / "agent.yaml", profile=p)
    assert {h.id for h in habits.habits} == set(p.spec.meters.habits)
    for h in habits.habits:  # every nudge text formats with the profile's words and no stray braces
        words = {**habits.vocabulary.words(), "value": 1, "target": 2, "title": "t", "hours": 3, "minutes": 4, "items": "keys", "name": h.name}
        assert "{" not in h.nudge.format_map(words)
    dayrun = DayRunConfig.load(CONFIG / "dayrun.yaml", CONFIG / "agent.yaml", profile=p)
    assert dayrun.vocabulary == p.vocabulary and dayrun.pack != "ucf"
    sc = Scenario.load(p.scenarios_dir() / f"{p.spec.scenario}.yaml")
    assert sc.items and sc.plans.keys() >= {"good", "bad"}
