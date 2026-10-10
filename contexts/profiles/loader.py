"""Profiles: who this box serves, as configuration (config/profiles/<name>/).

    profile = load_profile(config_dir)                 # ATLAS_PROFILE, else agent.yaml `profile:`, else "student"
    profile = load_profile(config_dir, "shift-worker")

A profile is a folder holding `profile.yaml` (name, description, audience, vocabulary, meters) and
optional copies of the base config files, layered over `config/*.yaml`:

- `habits.yaml`, `dayrun.yaml`, `agent.yaml` are merged key by key (`layer`): a mapping overrides the
  base's keys, a list of `{id: ...}` entries replaces entries with the same id and appends the rest,
  any other value replaces the base's.
- `sorting.yaml`, `rules.yaml`, `league.yaml` replace the base file whole (`path`): their loaders
  live in other contexts and take a file.
- `places/*.yaml` are extra place packs, loaded after the base ones.
- `scenarios/*.yaml` are Day Run scenarios, found before `fixtures/play/`.

Everything is read once at boot. The student profile has no overrides, so a box with no profile
folder at all behaves exactly as before: `load_profile` then returns the built-in student profile.
"""

import copy
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

DEFAULT_PROFILE = "student"
PROFILE_ENV = "ATLAS_PROFILE"
PROFILE_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
LAYERED = ("habits.yaml", "dayrun.yaml", "agent.yaml")  # merged over the base file
REPLACED = ("sorting.yaml", "rules.yaml", "league.yaml")  # the profile's file is used instead of the base one
OVERRIDABLE = LAYERED + REPLACED


class ProfileError(ValueError):
    """An unknown profile name, or a profile folder that cannot be read."""


class Vocabulary(BaseModel):
    """The words the UI and the nudges use for this person's day. Defaults are the student's."""

    model_config = ConfigDict(extra="forbid")

    workplace: str = "desk"  # where the main activity happens and the box watches: desk, station, home
    activity: str = "study"  # the main activity, as a verb and a noun: study, work, homework
    event: str = "class"  # a fixed block on the calendar: class, shift, appointment
    deadline: str = "assignment"  # something due: assignment, handover, chore
    plurals: dict[str, str] = {}  # irregular plurals, e.g. {event: classes}

    def plural(self, key: str) -> str:
        word = getattr(self, key)
        if key in self.plurals:
            return self.plurals[key]
        if word.endswith(("s", "x", "z", "ch", "sh")):
            return word + "es"
        if word.endswith("y") and word[-2:-1] not in "aeiou":
            return word[:-1] + "ies"
        return word + "s"

    def words(self) -> dict[str, str]:
        """Every word, with its plural and capitalised forms, for `str.format_map` in nudge texts and titles."""
        out = {}
        for key in ("workplace", "activity", "event", "deadline"):
            word = getattr(self, key)
            out[key] = word
            out[f"{key}_plural"] = self.plural(key)
            out[f"{key.capitalize()}"] = word.capitalize()
            out[f"{key.capitalize()}_plural"] = self.plural(key).capitalize()
        return out


class Meters(BaseModel):
    """Which habits and Day Run meters this profile shows. Empty means all of them."""

    model_config = ConfigDict(extra="forbid")

    habits: list[str] = []  # habit ids from habits.yaml; others are dropped from the layered config
    dayrun: list[str] = []  # meter names the Day Run screen shows: energy, focus, hunger


class Shift(BaseModel):
    """A shift worker's hours. The sleep window follows them: it starts `sleep_after_minutes` after the shift
    ends and lasts `sleep_hours`, so moving the shift moves the night (habits.yaml and dayrun.yaml read it)."""

    model_config = ConfigDict(extra="forbid")

    start: time
    end: time
    sleep_after_minutes: int = 60
    sleep_hours: float = 7

    def sleep_window(self) -> tuple[time, time]:
        anchor = datetime(2000, 1, 1, self.end.hour, self.end.minute)
        bed = anchor + timedelta(minutes=self.sleep_after_minutes)
        wake = bed + timedelta(hours=self.sleep_hours)
        return bed.time().replace(second=0, microsecond=0), wake.time().replace(second=0, microsecond=0)


class ProfileSpec(BaseModel):
    """profile.yaml."""

    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    audience: str = ""
    vocabulary: Vocabulary = Vocabulary()
    meters: Meters = Meters()
    shift: Shift | None = None
    scenario: str | None = None  # the Day Run scenario that demos this profile (a file under scenarios/ or fixtures/play/)


@dataclass(frozen=True)
class Profile:
    name: str
    spec: ProfileSpec
    config_dir: Path
    folder: Path | None  # None: the built-in student profile on a box with no profiles folder
    overrides: dict[str, str] = field(default_factory=dict)  # base file name -> the profile file that overrides it

    # --- what the rest of the box reads ---

    @property
    def vocabulary(self) -> Vocabulary:
        return self.spec.vocabulary

    @property
    def description(self) -> str:
        return self.spec.description

    def path(self, name: str) -> Path:
        """The file to load for `name`: the profile's copy when it has one, else the base `config/<name>`."""
        own = self.folder / name if self.folder else None
        return own if own and own.is_file() else self.config_dir / name

    def layer(self, name: str, base: dict | list | None):
        """`base` (the parsed base file) with the profile's copy of `name` merged over it, then the profile's own
        rules applied: a shift sets agent.yaml's sleep window, `meters.habits` keeps only those habits."""
        out = copy.deepcopy(base) if base is not None else {}
        own = self.folder / name if self.folder else None
        if own and own.is_file():
            out = merge(out, _read(own))
        if name in ("agent.yaml", "dayrun.yaml") and self.spec.shift and isinstance(out, dict):
            # the camera's overnight (agent.yaml) and Day Run's usual night (dayrun.yaml) both follow the shift
            bed, wake = self.spec.shift.sleep_window()
            out["sleep"] = {**(out.get("sleep") or {}), "bedtime": bed.strftime("%H:%M"), "wake": wake.strftime("%H:%M")}
        if name == "habits.yaml" and self.spec.meters.habits and isinstance(out, dict) and isinstance(out.get("habits"), list):
            keep = set(self.spec.meters.habits)
            out["habits"] = [h for h in out["habits"] if not isinstance(h, dict) or h.get("id") in keep]
        return out

    def raw(self, name: str):
        """The base file (empty when missing) with the profile layered over it."""
        base = self.config_dir / name
        return self.layer(name, _read(base) if base.is_file() else None)

    def agent(self) -> dict:
        return self.raw("agent.yaml") or {}

    def places_dirs(self) -> list[Path]:
        """Where place packs come from, base first; a profile pack with the same name wins."""
        dirs = [self.config_dir / "places"]
        if self.folder and (self.folder / "places").is_dir():
            dirs.append(self.folder / "places")
        return dirs

    def scenarios_dir(self) -> Path | None:
        return self.folder / "scenarios" if self.folder and (self.folder / "scenarios").is_dir() else None

    def summary(self) -> dict:
        """GET /api/profile."""
        return {
            "name": self.name,
            "description": self.spec.description,
            "audience": self.spec.audience,
            "vocabulary": self.spec.vocabulary.model_dump(),
            "words": self.spec.vocabulary.words(),
            "meters": self.spec.meters.model_dump(),
            "shift": self.spec.shift.model_dump(mode="json") if self.spec.shift else None,
            "scenario": self.spec.scenario,
            "overrides": dict(self.overrides),
        }


def merge(base, over):
    """`over` on top of `base`. Mappings merge by key; lists of `{id: ...}` merge by id (the profile's entry
    replaces the base's, new ids are appended in order); anything else is replaced by `over`."""
    if isinstance(base, dict) and isinstance(over, dict):
        out = dict(base)
        for k, v in over.items():
            out[k] = merge(base[k], v) if k in base else copy.deepcopy(v)
        return out
    if isinstance(base, list) and isinstance(over, list) and _by_id(base) and _by_id(over):
        ids = {e["id"]: e for e in over}
        out = [merge(e, ids[e["id"]]) if e["id"] in ids else e for e in base]
        seen = {e["id"] for e in base}
        return out + [e for e in over if e["id"] not in seen]
    return copy.deepcopy(over)


def _by_id(items: list) -> bool:
    return bool(items) and all(isinstance(e, dict) and "id" in e for e in items)


def _read(path: Path):
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProfileError(f"cannot read {path}: {exc}") from None


def available(config_dir: str | Path) -> list[str]:
    folder = Path(config_dir) / "profiles"
    if not folder.is_dir():
        return [DEFAULT_PROFILE]
    names = sorted(p.name for p in folder.iterdir() if (p / "profile.yaml").is_file())
    return names or [DEFAULT_PROFILE]


def profile_name(config_dir: str | Path, name: str | None = None) -> str:
    """Which profile to load: the argument, else $ATLAS_PROFILE, else agent.yaml's `profile:`, else student."""
    if name:
        return name
    if env := os.environ.get(PROFILE_ENV, "").strip():
        return env
    agent = Path(config_dir) / "agent.yaml"
    if agent.is_file():
        raw = _read(agent) or {}
        if isinstance(raw, dict) and raw.get("profile"):
            return str(raw["profile"])
    return DEFAULT_PROFILE


def load_profile(config_dir: str | Path = "config", name: str | None = None) -> Profile:
    """Read `config/profiles/<name>/`. Unknown names are a ProfileError that lists what exists."""
    config_dir = Path(config_dir)
    name = profile_name(config_dir, name)
    folder = config_dir / "profiles" / name
    if not PROFILE_NAME.match(name):
        raise ProfileError(f"profile name {name!r} must be 1 to 40 of a-z, 0-9, _ or - (ATLAS_PROFILE or agent.yaml `profile:`)")
    if not (folder / "profile.yaml").is_file():
        if name == DEFAULT_PROFILE and not (config_dir / "profiles").is_dir():
            return Profile(name=name, spec=ProfileSpec(name=name, description="The built-in student profile"), config_dir=config_dir, folder=None)
        have = ", ".join(available(config_dir))
        raise ProfileError(f"no profile {name!r} under {config_dir / 'profiles'} (have: {have}); set ATLAS_PROFILE or agent.yaml `profile:` to one of them")
    try:
        spec = ProfileSpec.model_validate(_read(folder / "profile.yaml") or {})
    except ValueError as exc:  # pydantic's ValidationError is a ValueError
        raise ProfileError(f"{folder / 'profile.yaml'}: {exc}") from None
    overrides = {f: (folder / f).relative_to(config_dir).as_posix() for f in OVERRIDABLE if (folder / f).is_file()}
    if (folder / "places").is_dir() and any((folder / "places").glob("*.yaml")):
        overrides["places/"] = (folder / "places").relative_to(config_dir).as_posix()
    if (folder / "scenarios").is_dir() and any((folder / "scenarios").glob("*.yaml")):
        overrides["scenarios/"] = (folder / "scenarios").relative_to(config_dir).as_posix()
    return Profile(name=name, spec=spec, config_dir=config_dir, folder=folder, overrides=overrides)
