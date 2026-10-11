"""The node's calibrated idle: the figure stored beside every joules-per-answer value (T4b; telemetry rulings of T10, T13).

    figures = idle_figures(os.environ)           # config/telemetry_idle.yaml, then ATLAS_TELEMETRY_IDLE_W on top
    Calibration(idle=figures)                    # JoulesPerAnswerProjection reads it

A pod's idle is the deep-idle reading taken by the sampler alone with the model loaded and no request (10.27 W on
pod C, metric pod.idle_watts, T10); the Jetson's is T13's jetson.idle_watts. Neither is "the latest reading before the
window", which on a pod lands on the power tail of the previous answer and on the Jetson on a captioning stretch.
Each figure names the ledger row it was measured as (metric and timestamp) so the stored value can be traced back.

The setting: ATLAS_TELEMETRY_IDLE_W="pod-a=10.27,pod-b=10.27,jetson=15.76", one watts figure per node, over the
file's figure for that node. A figure set this way names no ledger row (its source is the setting), because the
number is no longer the row's. config/telemetry_idle.yaml carries the measured figures and their source rows.
A node with no figure falls back to the latest quiet reading, and the stored value says so.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from contexts.telemetry.domain.nodes import KNOWN_NODES, canonical

IDLE_ENV = "ATLAS_TELEMETRY_IDLE_W"
CONFIG = Path(__file__).resolve().parents[3] / "config" / "telemetry_idle.yaml"


@dataclass(frozen=True)
class IdleFigure:
    watts: float
    source: str  # where the figure was read from: "config" or the name of the setting
    metric: str | None = None  # the ledger metric it was measured as, e.g. pod.idle_watts
    measured: datetime | None = None  # the timestamp of that ledger row (UTC)
    note: str = ""


def idle_figures(env: Mapping[str, str], path: Path | None = CONFIG) -> dict[str, IdleFigure]:
    """The calibrated idle per node: the file's figures, each overridden by the setting. Raises ValueError, naming
    the setting or the file, on anything it cannot read as a positive number of watts for a known node."""
    figures = _from_file(path) if path is not None else {}
    raw = (env.get(IDLE_ENV) or "").strip()
    seen: set[str] = set()
    for part in (p.strip() for p in raw.split(",") if p.strip()):
        node, eq, value = part.partition("=")
        node = canonical(node.strip())
        if not eq or node not in KNOWN_NODES:
            raise ValueError(f"{IDLE_ENV} must read node=watts for a known node ({', '.join(KNOWN_NODES)}), not {part!r}")
        if node in seen:
            raise ValueError(f"{IDLE_ENV} names {node} twice")
        seen.add(node)
        figures[node] = IdleFigure(_watts(value, f"{IDLE_ENV} for {node}"), source=IDLE_ENV)
    return figures


def _watts(value: object, where: str) -> float:
    try:
        watts = float(str(value).strip())
    except ValueError:
        raise ValueError(f"{where} must be a number of watts, not {value!r}") from None
    if not math.isfinite(watts) or watts <= 0:
        raise ValueError(f"{where} must be a positive number of watts, not {value!r}")
    return watts


def _from_file(path: Path) -> dict[str, IdleFigure]:
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"{path.name} is not valid YAML: {e}") from None
    nodes = data.get("nodes") if isinstance(data, dict) else None
    if not isinstance(nodes, dict):
        raise ValueError(f"{path.name} must hold a `nodes:` mapping of node to its idle figure")
    out: dict[str, IdleFigure] = {}
    for name, row in nodes.items():
        node = canonical(str(name))
        if node not in KNOWN_NODES or not isinstance(row, dict) or "watts" not in row:
            raise ValueError(f"{path.name}: {name} must be a known node with a `watts` figure")
        out[node] = IdleFigure(_watts(row["watts"], f"{path.name} {node} watts"), source="config",
                               metric=row.get("metric"), measured=_when(row.get("measured"), f"{path.name} {node}"),
                               note=str(row.get("note") or ""))
    return out


def _when(value: object, where: str) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"{where} measured must be an ISO timestamp, not {value!r}") from None
