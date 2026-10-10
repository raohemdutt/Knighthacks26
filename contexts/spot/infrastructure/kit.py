"""Read the SPOT challenge kit from a folder and say what is odd about it.

    kit = load_kit()                      # ATLAS_SPOT_KIT, else fixtures/spot/kit; None when the folder has no tables
    kit = load_kit("some/other/folder")

Standard library only (csv, json); Windows line endings are fine. Both layouts are read: the sponsor's numbered
folders (02_DATA/, 03_DATA_DICTIONARY/, 04_ASSETS/) or every file flat in one folder (images in assets/). Every row is parsed into its typed model; a row that does not parse is
skipped and reported. Nothing is repaired silently: `kit.issues` lists every surprise (columns that differ from
the data dictionary, duplicate or orphan ids, times out of order, missions the replay stream never reaches,
fields that carry no information), each with a count and the ids behind it.
"""

import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import ValidationError

from contexts.spot.domain.models import Action, Anomaly, Asset, Issue, Kit, Location, Mission, Reading, ReplayEvent, Row

KIT_ENV = "ATLAS_SPOT_KIT"
DEFAULT_KIT = "fixtures/spot/kit"
TABLES: dict[str, tuple[type[Row], str]] = {
    "missions.csv": (Mission, "mission_id"),
    "actions.csv": (Action, "action_id"),
    "readings.csv": (Reading, "reading_id"),
    "anomalies.csv": (Anomaly, "anomaly_id"),
    "locations.csv": (Location, "location_id"),
    "assets.csv": (Asset, "asset_id"),
}
IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".webp", ".gif")
SUBFOLDERS = ("02_DATA", "03_DATA_DICTIONARY", "00_START_HERE")  # the sponsor's layout
ASSET_FOLDERS = ("04_ASSETS", "assets")
HOW_TO = ("No SPOT kit found. Put the sponsor's kit folder (02_DATA/ with missions.csv, actions.csv, readings.csv, anomalies.csv, "
          "locations.csv, assets.csv and replay.ndjson; 03_DATA_DICTIONARY/; 04_ASSETS/) in fixtures/spot/kit/, or set ATLAS_SPOT_KIT "
          "to the folder that holds it (the files may also sit flat in one folder), and restart the gateway.")
MAX_IDS = 40  # ids kept per issue; the count is always the full number


def kit_path(path: str | Path | None = None) -> Path:
    return Path(path or os.environ.get(KIT_ENV) or DEFAULT_KIT)


def find(folder: Path, name: str) -> Path:
    """A kit file, in the kit's own numbered folders or flat in `folder`. The flat path when it is nowhere."""
    for sub in ("", *SUBFOLDERS):
        if (folder / sub / name).is_file():
            return folder / sub / name
    return folder / name


def assets_folder(folder: Path) -> Path | None:
    """Where the kit's images are: 04_ASSETS/ in the sponsor's layout, assets/ in a flat one."""
    return next((folder / d for d in ASSET_FOLDERS if (folder / d).is_dir()), None)


def load_kit(path: str | Path | None = None) -> Kit | None:
    folder = kit_path(path)
    if not folder.is_dir() or not find(folder, "missions.csv").is_file() or not find(folder, "actions.csv").is_file():
        return None
    issues: list[Issue] = []
    dictionary = _dictionary(find(folder, "data_dictionary.csv"), issues)
    rows: dict[str, list] = {}
    for name, (model, key) in TABLES.items():
        rows[name] = _table(find(folder, name), model, key, dictionary.get(name), issues)
    replay = _replay(find(folder, "replay.ndjson"), issues)
    validation = None
    if find(folder, "validation_results.json").is_file():
        try:
            validation = json.loads(find(folder, "validation_results.json").read_text(encoding="utf-8"))
        except ValueError as exc:
            issues.append(Issue(code="validation_unreadable", level="warn", message=f"validation_results.json could not be read: {exc}"))
    assets_dir = assets_folder(folder)
    files = (sorted(str(p.relative_to(assets_dir)).replace(os.sep, "/") for p in assets_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_TYPES) if assets_dir else [])
    kit = Kit(missions=rows["missions.csv"], actions=rows["actions.csv"], readings=rows["readings.csv"],
              anomalies=rows["anomalies.csv"], locations=rows["locations.csv"], assets=rows["assets.csv"], replay=replay,
              issues=issues, validation=validation, path=str(folder), asset_files=files)
    kit.assets_dir = str(assets_dir) if assets_dir else None
    kit.issues.extend(check(kit))
    return kit


# --- reading ---


def _dictionary(path: Path, issues: list[Issue]) -> dict[str, list[str]]:
    if not path.is_file():
        issues.append(Issue(code="dictionary_missing", level="info",
                            message="data_dictionary.csv is not in the kit folder, so columns were not checked against it."))
        return {}
    fields: dict[str, list[str]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fields[row.get("dataset", "")].append(row.get("field_name", ""))
    return dict(fields)


def _table(path: Path, model: type[Row], key: str, expected: list[str] | None, issues: list[Issue]) -> list:
    if not path.is_file():
        issues.append(Issue(code="table_missing", level="error", message=f"{path.name} is missing from the kit folder.", ids=[path.name]))
        return []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw = list(reader)
        columns = list(reader.fieldnames or [])
    if expected is not None:
        missing = [c for c in expected if c not in columns]
        extra = [c for c in columns if c not in expected]
        if missing:
            issues.append(Issue(code="columns_missing", level="error", count=len(missing), ids=missing,
                                message=f"{path.name} lacks {len(missing)} column(s) the data dictionary describes: {', '.join(missing)}."))
        if extra:
            issues.append(Issue(code="columns_undocumented", level="info", count=len(extra), ids=extra,
                                message=f"{path.name} has {len(extra)} column(s) the data dictionary does not describe: {', '.join(extra)}."))
    out, bad, seen, dup = [], [], set(), []
    for n, row in enumerate(raw, 2):  # line 1 is the header
        try:
            item = model.model_validate({k: (v.strip() if isinstance(v, str) else v) for k, v in row.items() if k})
        except ValidationError as exc:
            bad.append(f"line {n}: {exc.errors()[0]['loc'][0]} {exc.errors()[0]['msg']}")
            continue
        ident = getattr(item, key)
        if not ident:
            bad.append(f"line {n}: empty {key}")
            continue
        if ident in seen:
            dup.append(ident)
            continue
        seen.add(ident)
        out.append(item)
    if bad:
        issues.append(Issue(code="rows_unreadable", level="error", count=len(bad), ids=bad[:MAX_IDS],
                            message=f"{path.name}: {len(bad)} row(s) could not be read and were left out."))
    if dup:
        issues.append(Issue(code="duplicate_ids", level="error", count=len(dup), ids=dup[:MAX_IDS],
                            message=f"{path.name}: {len(dup)} repeated {key} value(s); the first row of each was kept."))
    not_synthetic = [r.get(key, "") for r in raw if "synthetic" in r and str(r["synthetic"]).strip().lower() != "true"]
    if not_synthetic:
        issues.append(Issue(code="not_marked_synthetic", level="warn", count=len(not_synthetic), ids=not_synthetic[:MAX_IDS],
                            message=f"{path.name}: {len(not_synthetic)} row(s) are not marked synthetic."))
    return out


def _replay(path: Path, issues: list[Issue]) -> list[ReplayEvent]:
    if not path.is_file():
        issues.append(Issue(code="replay_missing", level="warn",
                            message="replay.ndjson is missing; the replay is rebuilt from missions.csv and actions.csv."))
        return []
    events, bad = [], []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()  # the official kit has Windows line endings
            if not line:
                continue
            try:
                events.append(ReplayEvent.model_validate_json(line))
            except ValidationError:
                bad.append(f"line {n}")
    if bad:
        issues.append(Issue(code="replay_unreadable", level="error", count=len(bad), ids=bad[:MAX_IDS],
                            message=f"replay.ndjson: {len(bad)} line(s) could not be read and were left out."))
    out_of_order = [events[i + 1].event_id for i in range(len(events) - 1) if events[i + 1].timestamp < events[i].timestamp]
    if out_of_order:
        issues.append(Issue(code="replay_out_of_order", level="warn", count=len(out_of_order), ids=out_of_order[:MAX_IDS],
                            message=f"replay.ndjson: {len(out_of_order)} event(s) come before the event on the line above; "
                                    "the player sorts by event time (file order breaks ties)."))
    order = {e.event_id: i for i, e in enumerate(events)}
    return sorted(events, key=lambda e: (e.timestamp, order[e.event_id]))


# --- checking ---


def check(kit: Kit) -> list[Issue]:
    """Every surprise in a loaded kit. Pure: the same tables always give the same list."""
    out: list[Issue] = []

    def add(code: str, level: str, ids: list[str], message: str) -> None:
        if ids:
            out.append(Issue(code=code, level=level, count=len(ids), ids=sorted(set(ids))[:MAX_IDS], message=message.format(n=len(ids))))

    # references that point nowhere
    add("action_orphan_mission", "error", [a.action_id for a in kit.actions if a.mission_id not in kit.mission],
        "{n} action(s) name a mission that is not in missions.csv.")
    add("action_orphan_location", "error", [a.action_id for a in kit.actions if a.location_id not in kit.location],
        "{n} action(s) name a location that is not in locations.csv.")
    add("action_orphan_asset", "error", [a.action_id for a in kit.actions if a.asset_id not in kit.asset],
        "{n} action(s) name an asset that is not in assets.csv.")
    add("reading_orphan_action", "error", [r.reading_id for r in kit.readings if r.action_id not in kit.action],
        "{n} reading(s) name an action that is not in actions.csv.")
    add("anomaly_orphan_action", "error", [x.anomaly_id for x in kit.anomalies if x.action_id not in kit.action],
        "{n} anomaly row(s) name an action that is not in actions.csv.")
    add("asset_orphan_location", "error", [x.asset_id for x in kit.assets if x.location_id not in kit.location],
        "{n} asset(s) sit at a location that is not in locations.csv.")
    add("mission_without_actions", "warn", [m.mission_id for m in kit.missions if not kit.actions_of.get(m.mission_id)],
        "{n} mission(s) have no actions.")

    # a child row that disagrees with its action
    def disagrees(row) -> bool:
        a = kit.action.get(row.action_id)
        return a is not None and (row.mission_id, row.asset_id, row.location_id) != (a.mission_id, a.asset_id, a.location_id)

    add("reading_disagrees_with_action", "warn", [r.reading_id for r in kit.readings if disagrees(r)],
        "{n} reading(s) carry a mission, asset or location different from their action's.")
    add("anomaly_disagrees_with_action", "warn", [x.anomaly_id for x in kit.anomalies if disagrees(x)],
        "{n} anomaly row(s) carry a mission, asset or location different from their action's.")
    add("child_time_differs", "info",
        [r.reading_id for r in kit.readings if r.action_id in kit.action and r.timestamp != kit.action[r.action_id].timestamp]
        + [x.anomaly_id for x in kit.anomalies if x.action_id in kit.action and x.timestamp != kit.action[x.action_id].timestamp],
        "{n} reading(s) or anomaly row(s) have a timestamp different from their action's.")
    add("action_config_differs", "warn",
        [a.action_id for a in kit.actions if a.mission_id in kit.mission
         and (a.scenario, a.configuration_version) != (kit.mission[a.mission_id].scenario, kit.mission[a.mission_id].configuration_version)],
        "{n} action(s) carry a scenario or configuration version different from their mission's.")

    # time
    add("mission_ends_before_start", "error", [m.mission_id for m in kit.missions if m.end_time < m.start_time],
        "{n} mission(s) end before they start.")
    add("mission_duration_differs", "warn",
        [m.mission_id for m in kit.missions if abs((m.end_time - m.start_time).total_seconds() / 60 - m.duration_min) > 1],
        "{n} mission(s) have a duration_min more than a minute off end_time minus start_time.")
    add("action_outside_mission", "warn",
        [a.action_id for a in kit.actions if a.mission_id in kit.mission
         and not kit.mission[a.mission_id].start_time <= a.timestamp <= kit.mission[a.mission_id].end_time],
        "{n} action(s) are timestamped outside their mission's start and end.")
    unordered, gaps, rising = [], [], []
    for mid, acts in kit.actions_of.items():
        if any(b.timestamp < a.timestamp for a, b in zip(acts, acts[1:])):
            unordered.append(mid)
        if [a.sequence for a in acts] != list(range(1, len(acts) + 1)):
            gaps.append(mid)
        if any(b.battery_pct > a.battery_pct for a, b in zip(acts, acts[1:])):
            rising.append(mid)
    add("actions_out_of_time_order", "warn", unordered, "{n} mission(s) have actions whose time goes backwards as the sequence goes up.")
    add("sequence_gaps", "warn", gaps, "{n} mission(s) have a sequence that does not run 1, 2, 3, ... without gaps.")
    add("battery_rises_in_mission", "info", rising, "{n} mission(s) show battery going up between two actions.")
    overlap = [b.mission_id for a, b in zip(kit.missions, kit.missions[1:]) if b.start_time < a.end_time]
    add("missions_overlap", "warn", overlap, "{n} mission(s) start before the previous one ended (one robot is assumed).")

    # status fields
    add("failed_without_failure_type", "warn", [a.action_id for a in kit.actions if a.failed and not a.failure_type],
        "{n} failed action(s) have no failure_type.")
    add("success_with_failure_type", "warn", [a.action_id for a in kit.actions if not a.failed and a.failure_type],
        "{n} successful action(s) carry a failure_type.")
    unknown_status = [a.action_id for a in kit.actions if a.status not in ("SUCCESS", "FAILED")]
    add("unknown_action_status", "warn", unknown_status, "{n} action(s) have a status other than SUCCESS or FAILED; they are counted as failed.")

    # things that are true of this kit and shape what the analysis may claim (info, not faults)
    away = [a.action_id for a in kit.actions if a.asset_id in kit.asset and kit.asset[a.asset_id].location_id != a.location_id]
    if away:
        out.append(Issue(code="asset_seen_away_from_home", level="info", count=len(away), ids=away[:MAX_IDS],
                         message=f"{len(away)} of {len(kit.actions)} actions name an asset at a waypoint other than the asset's "
                                 "location in assets.csv. Waypoint numbers use the action's location; the map draws an asset at its "
                                 "assets.csv location."))
    no_reading = [a.action_id for a in kit.actions if not kit.readings_of.get(a.action_id)]
    if no_reading:
        failed = sum(1 for i in no_reading if kit.action[i].failed)
        out.append(Issue(code="action_without_reading", level="info", count=len(no_reading), ids=no_reading[:MAX_IDS],
                         message=f"{len(no_reading)} of {len(kit.actions)} actions have no reading row ({failed} of them failed). "
                                 "A successful inspection with no reading row is still counted as a good inspection."))
    multi = [aid for aid, rs in kit.readings_of.items() if len(rs) > 1]
    add("action_with_several_readings", "info", multi, "{n} action(s) have more than one reading.")
    with_reading = [(kit.action[r.action_id], r) for r in kit.readings if r.action_id in kit.action]
    if with_reading:
        agree = sum(1 for a, r in with_reading if a.failed == (not r.good))
        if agree == len(with_reading):
            out.append(Issue(code="quality_repeats_status", level="info", count=agree,
                             message=f"Reading quality carries no information of its own: all {agree} readings are QUESTIONABLE exactly "
                                     "when their action FAILED and GOOD exactly when it succeeded."))
        else:
            out.append(Issue(code="quality_differs_from_status", level="info", count=len(with_reading) - agree,
                             ids=[r.reading_id for a, r in with_reading if a.failed != (not r.good)][:MAX_IDS],
                             message=f"{len(with_reading) - agree} reading(s) have a quality that does not follow the action's status."))
    if kit.anomalies:
        same = sum(1 for x in kit.anomalies if x.operator_review_required == (x.severity in ("HIGH", "CRITICAL")))
        if same == len(kit.anomalies):
            out.append(Issue(code="review_repeats_severity", level="info", count=same,
                             message=f"operator_review_required repeats severity: it is True for every HIGH or CRITICAL anomaly and "
                                     f"False for every LOW or MEDIUM one ({same} of {same})."))
    pairs = Counter((m.scenario, m.configuration_version) for m in kit.missions)
    if pairs and len(pairs) == len({s for s, _ in pairs}) == len({v for _, v in pairs}):
        text = ", ".join(f"{s} is always {v}" for s, v in sorted(pairs))
        out.append(Issue(code="scenario_repeats_version", level="info", count=len(kit.missions),
                         message=f"scenario and configuration_version are the same split ({text}), so they are one comparison, not two."))
    for name, values in (("operator_mode", {m.operator_mode for m in kit.missions}),):
        if len(values) == 1 and kit.missions:
            out.append(Issue(code="constant_field", level="info", count=len(kit.missions), ids=[name],
                             message=f"{name} is {next(iter(values))!r} on every mission, so it explains nothing."))

    out.extend(_check_replay(kit))
    return out


def _check_replay(kit: Kit) -> list[Issue]:
    out: list[Issue] = []
    if not kit.replay:
        return out
    in_stream = {e.mission_id for e in kit.replay}
    absent = [m.mission_id for m in kit.missions if m.mission_id not in in_stream]
    if absent:
        out.append(Issue(code="missions_not_in_replay", level="warn", count=len(absent), ids=absent,
                         message=f"{len(absent)} mission(s) in missions.csv never appear in replay.ndjson ({', '.join(absent)}). "
                                 "The player rebuilds their events from the tables and marks them as rebuilt."))
    unknown = sorted(in_stream - set(kit.mission))
    if unknown:
        out.append(Issue(code="replay_unknown_mission", level="warn", count=len(unknown), ids=unknown,
                         message=f"replay.ndjson names {len(unknown)} mission(s) that are not in missions.csv."))
    started = {e.mission_id for e in kit.replay if e.event_type == "MISSION_STARTED"}
    ended = {e.mission_id for e in kit.replay if e.event_type == "MISSION_COMPLETED"}
    cut = sorted(started - ended)
    if cut:
        last = kit.replay[-1]
        out.append(Issue(code="replay_cut_short", level="warn", count=len(cut), ids=cut,
                         message=f"replay.ndjson stops at {last.event_id} ({len(kit.replay)} events), in the middle of "
                                 f"{', '.join(cut)}: the mission starts but never ends in the stream."))
    acted = {e.details.get("action_id") for e in kit.replay if e.event_type in ("ACTION_COMPLETED", "ACTION_FAILED")}
    begun = {e.details.get("action_id") for e in kit.replay if e.event_type == "ACTION_STARTED"}
    open_actions = sorted(str(a) for a in begun - acted)
    if open_actions:
        out.append(Issue(code="replay_action_never_ends", level="warn", count=len(open_actions), ids=open_actions,
                         message=f"{len(open_actions)} action(s) start in the replay and never complete or fail there."))
    wrong = []
    for e in kit.replay:
        a = kit.action.get(str(e.details.get("action_id", "")))
        if e.event_type in ("ACTION_COMPLETED", "ACTION_FAILED") and a is not None:
            if (e.event_type == "ACTION_FAILED") != a.failed or (e.details.get("failure_type") or "") != a.failure_type or e.timestamp != a.timestamp:
                wrong.append(a.action_id)
        elif e.event_type.startswith("ACTION") and a is None:
            wrong.append(str(e.details.get("action_id")))
    if wrong:
        out.append(Issue(code="replay_disagrees_with_tables", level="warn", count=len(wrong), ids=wrong[:MAX_IDS],
                         message=f"{len(wrong)} replay action event(s) disagree with actions.csv on outcome, failure type or time."))
    counts = Counter(e.event_type for e in kit.replay)
    if "ANOMALY" not in "".join(counts) and kit.anomalies:
        out.append(Issue(code="replay_has_no_anomaly_events", level="info", count=len(kit.anomalies),
                         message="The replay stream has mission and action events only. Readings and anomalies are attached at the "
                                 "moment their action ends in the stream (they share its timestamp in the tables)."))
    return out
