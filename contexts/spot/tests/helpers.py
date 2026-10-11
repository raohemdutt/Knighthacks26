"""Tiny hand-made kits for tests, so every expected number can be worked out on paper."""

from datetime import datetime, timedelta

from contexts.spot.domain.models import Action, Anomaly, Asset, Kit, Location, Mission, Reading

T0 = datetime(2026, 9, 1, 8, 0, 0)


def mission(n: int, status: str = "COMPLETED", version: str = "V1", start: datetime | None = None, steps: int = 10) -> Mission:
    start = start or T0 + timedelta(hours=8 * (n - 1))
    return Mission(mission_id=f"M{n:03d}", mission_name=f"Route {n}", scenario="BASELINE" if version == "V1" else "IMPROVED",
                   configuration_version=version, start_time=start, end_time=start + timedelta(minutes=3 * steps), duration_min=3 * steps,
                   start_battery_pct=90, end_battery_pct=90 - 6 * steps, mission_status=status, operator_mode="AUTOWALK")


def action(m: Mission, seq: int, n: int, failed: bool = False, kind: str = "VISUAL_INSPECTION", loc: str = "WP01", asset: str = "PUMP-1",
           failure: str = "SENSOR_TIMEOUT", battery: float | None = None) -> Action:
    return Action(action_id=f"A{n:04d}", mission_id=m.mission_id, sequence=seq, timestamp=m.start_time + timedelta(minutes=3 * seq),
                  location_id=loc, asset_id=asset, action_type=kind, battery_pct=90 - 6 * seq if battery is None else battery,
                  status="FAILED" if failed else "SUCCESS", failure_type=failure if failed else "", duration_sec=40,
                  scenario=m.scenario, configuration_version=m.configuration_version)


def reading(a: Action, n: int, value: float = 50.0, kind: str = "VISUAL_SCORE") -> Reading:
    return Reading(reading_id=f"R{n:04d}", action_id=a.action_id, mission_id=a.mission_id, asset_id=a.asset_id, location_id=a.location_id,
                   timestamp=a.timestamp, measurement_type=kind, value=value, unit="score", quality="QUESTIONABLE" if a.failed else "GOOD")


def anomaly(a: Action, n: int, severity: str = "HIGH", kind: str = "VISUAL_DEFECT") -> Anomaly:
    return Anomaly(anomaly_id=f"AN{n:03d}", action_id=a.action_id, mission_id=a.mission_id, location_id=a.location_id, asset_id=a.asset_id,
                   timestamp=a.timestamp, anomaly_type=kind, severity=severity, operator_review_required=severity in ("HIGH", "CRITICAL"))


LOCATIONS = [Location(location_id="WP01", location_name="Dock", zone_type="ACCESS", floor=1, x_coord=10, y_coord=10, environment="NORMAL"),
             Location(location_id="WP02", location_name="Pump Bay", zone_type="MECHANICAL", floor=2, x_coord=60, y_coord=40, environment="HIGH_NOISE")]
ASSETS = [Asset(asset_id="PUMP-1", asset_name="Pump 1", asset_type="Pump", location_id="WP01", criticality="LOW"),
          Asset(asset_id="HX-9", asset_name="Exchanger 9", asset_type="Heat Exchanger", location_id="WP02", criticality="CRITICAL")]


def make_kit(missions, actions, readings=(), anomalies=(), replay=(), validation=None) -> Kit:
    return Kit(missions=list(missions), actions=list(actions), readings=list(readings), anomalies=list(anomalies), locations=LOCATIONS,
               assets=ASSETS, replay=list(replay), validation=validation, path="(test)")


def small_kit() -> Kit:
    """Three missions of 10 actions.

    M001 (V1, COMPLETED): steps 9 and 10 fail at WP02 on HX-9; a HIGH anomaly on the failed step 10.
    M002 (V1, FAILED):    step 1 fails at WP01 on PUMP-1.
    M003 (V2, COMPLETED): nothing fails; a LOW anomaly on PUMP-1 from a successful action.
    So: 3 of 30 actions failed; 1 of 2 COMPLETED missions hides failures; WP02 has 2 failures of 10, WP01 1 of 20.
    """
    m1, m2, m3 = mission(1), mission(2, status="FAILED"), mission(3, version="V2")
    acts, n = [], 0
    for m in (m1, m2, m3):
        for seq in range(1, 11):
            n += 1
            at_hx = m is m1 and seq >= 6
            failed = (m is m1 and seq >= 9) or (m is m2 and seq == 1)
            acts.append(action(m, seq, n, failed=failed, loc="WP02" if at_hx else "WP01", asset="HX-9" if at_hx else "PUMP-1"))
    readings = [reading(a, i, value=40 + i) for i, a in enumerate(acts, 1)]
    anomalies = [anomaly(acts[9], 1, "HIGH"), anomaly(acts[25], 2, "LOW")]
    return make_kit([m1, m2, m3], acts, readings, anomalies)
