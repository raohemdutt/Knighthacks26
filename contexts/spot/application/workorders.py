"""Operator actions behind approvals: "Re-inspect <asset>" and "Create work order for <anomaly>".

    orders = WorkOrders(spot, data_dir, publish)
    await bus.publish(orders.propose_reinspect("HX-202"))       # an approval card; nothing else happens
    approvals.executors["connector_write"] = orders.executor(approvals.executors.get("connector_write"))

Both are proposed as `agent.action.proposed`; nothing is written until a person approves the card. The executor
then writes one JSON file under <data dir>/spot/work_orders/ and publishes `spot.work_order.created`. Nothing
leaves the box: there is no network call here at all.

The card's kind is the existing `connector_write` with a target that starts with "spot.": `contracts.agent.ActionKind`
is a closed list and adding a kind to it needs a second person's OK. The executor registered for that kind is
wrapped (the League does the same for `circle_share`): targets starting with "spot." are handled here, everything
else goes on to the connector hub untouched.
"""

import json
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from contexts.spot.application.service import Spot
from contexts.spot.domain.events import WorkOrderCreated
from contracts.agent import ActionProposed

KIND = "connector_write"
TARGET_PREFIX = "spot."
TARGETS = {"reinspect": "spot.reinspect", "work_order": "spot.work_order", "mission_request": "spot.mission_request"}
SAFE = re.compile(r"[^A-Za-z0-9_-]")


class WorkOrders:
    def __init__(self, spot: Spot, data_dir: str | Path, publish: Callable | None = None,
                 clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> None:
        self.spot = spot
        self.folder = Path(data_dir) / "spot" / "work_orders"
        self.publish = publish
        self.clock = clock
        self.orders: list[dict] = []  # rebuilt from the log

    # --- proposing (no side effect: only the event that becomes a card) ---

    def propose_reinspect(self, asset_id: str) -> ActionProposed:
        kit = self.spot.kit
        if kit is None or asset_id not in kit.asset:
            raise ValueError(f"unknown asset {asset_id!r}")
        asset = kit.asset[asset_id]
        item = next((i for i in self.spot.triage if i["id"] == asset_id), None)
        why = item["reason"] if item else "Requested by the operator."
        evidence = (item["evidence"]["anomalies"] + item["evidence"]["actions"])[:12] if item else []
        return ActionProposed(action_id=uuid4().hex, kind=KIND, target=TARGETS["reinspect"],
                              reason=f"Re-inspect {asset_id} {asset.asset_name} at {asset.location_id} on the next mission. {why} "
                                     "Recorded as a local work order on this box; nothing is sent anywhere.",
                              params={"order": "re-inspection", "asset_id": asset_id, "asset": asset.asset_name, "criticality": asset.criticality,
                                      "location_id": asset.location_id, "evidence": ", ".join(evidence) or "none", "stays_on_this_box": True})

    def propose_plan(self, route: list[str], turn_back_at: float, summary: str = "") -> ActionProposed:
        """A planned round from Shift Run, as a mission request. Like the others: a card first, then one local file."""
        kit = self.spot.kit
        route = [a.upper() for a in route]
        if kit is None or not route or any(a not in kit.asset for a in route):
            raise ValueError("a plan needs at least one known asset")
        names = ", ".join(f"{a} {kit.asset[a].asset_name}" for a in route)
        return ActionProposed(action_id=uuid4().hex, kind=KIND, target=TARGETS["mission_request"],
                              reason=f"Request a round for SPOT: {len(route)} stop(s) in this order: {names}. Turn back at {turn_back_at:.0f}% battery. "
                                     f"{summary} Recorded as a local mission request on this box; nothing is sent anywhere and SPOT is not commanded.".replace("  ", " "),
                              params={"order": "mission request", "asset_id": route[0], "route": ", ".join(route), "turn_back_at": turn_back_at,
                                      "stops": len(route), "stays_on_this_box": True})

    def propose_work_order(self, anomaly_id: str) -> ActionProposed:
        kit = self.spot.kit
        if kit is None or anomaly_id not in kit.anomaly:
            raise ValueError(f"unknown anomaly {anomaly_id!r}")
        x = kit.anomaly[anomaly_id]
        asset = kit.asset.get(x.asset_id)
        action = kit.action.get(x.action_id)
        confirmed = bool(action and not action.failed)
        return ActionProposed(action_id=uuid4().hex, kind=KIND, target=TARGETS["work_order"],
                              reason=f"Create a work order for {anomaly_id}: {x.severity} {x.anomaly_type.replace('_', ' ').lower()} on {x.asset_id}"
                                     f"{' ' + asset.asset_name if asset else ''} at {x.location_id} (mission {x.mission_id}), "
                                     + ("from a successful inspection. " if confirmed else "raised by a failed action, so unverified. ")
                                     + "Recorded as a local work order on this box; nothing is sent anywhere.",
                              params={"order": "work order", "anomaly_id": anomaly_id, "asset_id": x.asset_id, "severity": x.severity,
                                      "criticality": asset.criticality if asset else "", "location_id": x.location_id,
                                      "evidence": f"{anomaly_id}, {x.action_id}, {x.mission_id}", "stays_on_this_box": True})

    # --- executing (only ever called by Approvals after an approve) ---

    def executor(self, fallback: Callable | None = None) -> Callable:
        def run(req):
            if not str(req.target).startswith(TARGET_PREFIX):
                if fallback is None:
                    raise RuntimeError(f"no executor for {req.target}")
                return fallback(req)
            return self.write(req)

        return run

    def write(self, req) -> str:
        kind = {v: k for k, v in TARGETS.items()}.get(req.target, "work_order")
        now = self.clock()
        order_id = f"WO-{now.strftime('%Y%m%dT%H%M%S')}-{SAFE.sub('', str(req.id))[:6]}"
        self.folder.mkdir(parents=True, exist_ok=True)
        path = self.folder / f"{order_id}.json"
        record = {"work_order_id": order_id, "kind": kind, "created_at": now.isoformat(), "approved_action_id": req.id, "status": "open",
                  "reason": req.reason, **dict(req.params)}
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        event = WorkOrderCreated(work_order_id=order_id, kind=kind, asset_id=str(req.params.get("asset_id", "")),
                                 anomaly_id=req.params.get("anomaly_id"), reason=req.reason, path=str(path), action_id=req.id)
        if self.publish is not None:
            self.publish(event)
        return f"Work order {order_id} written to {path}. Nothing left this box."

    # --- the list, rebuilt from the log ---

    def on_created(self, e) -> None:
        self.orders.append({"work_order_id": e.work_order_id, "kind": getattr(e, "kind", ""), "asset_id": getattr(e, "asset_id", ""),
                            "anomaly_id": getattr(e, "anomaly_id", None), "reason": getattr(e, "reason", ""), "path": getattr(e, "path", ""),
                            "action_id": getattr(e, "action_id", ""), "created_at": e.ts.isoformat()})

    def view(self) -> list[dict]:
        return list(reversed(self.orders))
