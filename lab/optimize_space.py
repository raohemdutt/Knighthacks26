"""The search spaces and the night configuration the settings optimizer reads, as the research tasks wrote them.

Two search-space formats are on main, and both are read as they are, never converted on disk:

    atlas.searchspace.v1 (R35, lab/refine/research/kernels/data/searchspace-inference-gfx1151.json)
        {"schema": "atlas.searchspace.v1", "parameters": [{"name", "type": "choice" | "range" | "fixed",
         "values" | "bounds" | "value", "value_type", "is_ordered"}], "baseline": {...},
         "fixed_launch": {"args": [...], "env": {...}, "unset_env": [...]},
         "mapping": {parameter: {value as text: {"args": [...], "env": {...}}}}, "interactions": [...]}
        A trial's server is launched with fixed_launch.args followed by every searched parameter's mapped args, in
        parameter order, with fixed_launch.env plus the mapped env, and with every unset_env name removed from the
        environment unless a mapping sets it. "{run_dir}" in a mapped value is the trial's scratch folder.

    Ax-config style (R36, lab/refine/research/kernels/data/searchspace-training-gfx1151.json)
        {"name", "parameters": [{"kind": "choice" | "range" | "derived", "name", "parameter_type", "values" |
         "bounds", "is_ordered", "expression_str", "setting"}], "invalid_combinations": [{"when": {...},
         "unless"?, "why"}], "baseline": {...}, "pinned": {..., "env": {NAME: value | null}},
         "fixed_effective_batch_size": 16}
        A trial's trainer gets a settings file (every parameter, the derived ones computed from expression_str, and
        the pinned block), the pinned env (null means removed), and the NAME=value assignments that a parameter's
        "setting" text names, applied when the parameter is away from its baseline value.

The night configuration (schema lab.optimize.night/1, lab/refine/research/kernels/data/ax-config-night.json, R49)
names the lanes (inference, training), their workers, objective, tracking metrics, constraints, restart classes,
grouping, fidelities, early-stop rules, initial arms (the baseline and the prior arms), budgets and confirmation.
load_night reads it; Lane is one lane of it; initial_arms(lane, space) fills each initial arm out to a full
parameter set from the space's baseline.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import operator
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
DATA_DIR = REPO / "lab" / "refine" / "research" / "kernels" / "data"
NIGHT_CONFIG = DATA_DIR / "ax-config-night.json"
SAMPLE_SPACE = DATA_DIR / "searchspace-sample.json"
R35_SCHEMA = "atlas.searchspace.v1"
AX_CONFIG = "ax-config"
NIGHT_SCHEMA = "lab.optimize.night/1"
KINDS = ("inference", "training")
GOALS = ("minimize", "maximize")
LANES = ("inference", "training")
CONSTRAINT_OPS = (">=", "<=")
# The inference floors from R56 (branch ctx/runtime/energy-model-gfx1151, lab/energy_model.py OBJECTIVE): used when
# the night configuration's inference lane names no outcome_constraints of its own.
DEFAULT_INFERENCE_CONSTRAINTS = ("decode_tok_s.b1 >= 2.88", "ttft_s.b1 <= 2.0")

_METRIC = re.compile(r"[a-z][a-z0-9_]*(\.[a-z0-9_]+)*")
_CONSTRAINT = re.compile(r"\s*([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*)\s*(>=|<=)\s*(-?[0-9]+(?:\.[0-9]+)?)\s*")
_ENV_ASSIGN = re.compile(r"\b([A-Z][A-Z0-9_]{2,})=([A-Za-z0-9_.]+)")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class SpaceError(ValueError):
    """A search space or night configuration that cannot be used; .problems lists every reason."""

    def __init__(self, problems: Sequence[str]) -> None:
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


# --- values -----------------------------------------------------------------------------------------------------------

def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def same(a: Any, b: Any) -> bool:
    """Equal and of the same kind, so True is not 1 and 1 is not "1"."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if is_number(a) and is_number(b):
        return a == b
    return type(a) is type(b) and a == b


def value_key(value: Any) -> str:
    """A value as the R35 mapping keys spell it: true/false for booleans, the plain text otherwise."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _value_type(values: Sequence[Any]) -> str | None:
    if all(isinstance(v, bool) for v in values):
        return "bool"
    if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
        return "int"
    if all(is_number(v) for v in values):
        return "float"
    if all(isinstance(v, str) for v in values):
        return "str"
    return None


def content_hash(data: Any) -> str:
    """The hash of a JSON document's content, blind to whitespace and key order."""
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def shown(path: Path | str) -> str:
    """A path as the files record it: relative to the repo when it is inside it, with forward slashes."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return p.as_posix()


# --- parameters -------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Parameter:
    name: str
    type: str                                  # choice | range
    value_type: str                            # int | float | str | bool
    values: tuple[Any, ...] | None = None
    bounds: tuple[float, float] | None = None
    ordered: bool | None = None
    log_scale: bool = False
    setting: str | None = None                 # the R36 "setting" text, which may name NAME=value assignments

    def contains(self, value: Any) -> bool:
        if self.type == "choice":
            return any(same(value, v) for v in self.values or ())
        if not is_number(value):
            return False
        if self.value_type == "int" and not float(value).is_integer():
            return False
        low, high = self.bounds or (0.0, 0.0)
        return low <= value <= high

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"name": self.name, "type": self.type, "value_type": self.value_type}
        if self.type == "choice":
            out.update(values=list(self.values or ()), ordered=self.ordered)
        else:
            out.update(bounds=list(self.bounds or ()), log_scale=self.log_scale)
        return out


def _choice_or_range(name: str, kind: Any, raw: Mapping[str, Any], where: str, *, type_key: str,
                     ordered_key: str) -> tuple[Parameter | None, list[str]]:
    problems: list[str] = []
    given = raw.get(type_key)
    if kind == "choice":
        values = raw.get("values")
        if not (isinstance(values, list) and len(values) >= 2):
            return None, [f"{where}: a choice has two or more values, not {values!r}"]
        inferred = _value_type(values)
        value_type = given or inferred
        if inferred is None or value_type not in ("int", "float", "str", "bool"):
            return None, [f"{where}: the values are all int, float, str or bool, not {values!r}"]
        if given and given != inferred and not (given == "float" and inferred == "int"):
            problems.append(f"{where}: {type_key} {given!r} does not match the values {values!r}")
        if any(same(a, b) for i, a in enumerate(values) for b in values[i + 1:]):
            problems.append(f"{where}: the values repeat: {values!r}")
        ordered = raw.get(ordered_key)
        if ordered is not None and not isinstance(ordered, bool):
            problems.append(f"{where}: {ordered_key} is true or false")
        if problems:
            return None, problems
        vals = tuple(float(v) for v in values) if value_type == "float" else tuple(values)
        return Parameter(name, "choice", str(value_type), values=vals, ordered=ordered,
                         setting=raw.get("setting")), []
    bounds = raw.get("bounds")
    if not (isinstance(bounds, list) and len(bounds) == 2 and all(is_number(b) for b in bounds)
            and bounds[0] < bounds[1]):
        return None, [f"{where}: a range has bounds [low, high] with low < high, not {bounds!r}"]
    inferred = "int" if all(isinstance(b, int) for b in bounds) else "float"
    value_type = given or inferred
    if value_type not in ("int", "float") or (value_type == "int" and inferred != "int"):
        return None, [f"{where}: a range is int (whole bounds) or float, not {value_type!r} over {bounds!r}"]
    log_scale = raw.get("log_scale", raw.get("scaling") == "log")
    if log_scale and bounds[0] <= 0:
        return None, [f"{where}: a log-scale range needs a positive low bound"]
    pair = (int(bounds[0]), int(bounds[1])) if value_type == "int" else (float(bounds[0]), float(bounds[1]))
    return Parameter(name, "range", value_type, bounds=pair, log_scale=bool(log_scale),
                     setting=raw.get("setting")), []


# --- derived parameters (R36 expression_str) ---------------------------------------------------------------------------

_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv}


def evaluate(expression: str, values: Mapping[str, Any]) -> float:
    """An arithmetic expression over parameter names and numbers (+ - * / //), nothing else."""
    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and is_number(node.value):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in values or not is_number(values[node.id]):
                raise SpaceError([f"{expression!r} names {node.id}, which has no numeric value"])
            return values[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -walk(node.operand)
        raise SpaceError([f"{expression!r} is not plain arithmetic over parameter names"])
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise SpaceError([f"{expression!r} does not parse: {exc}"]) from exc
    return walk(tree)


@dataclass(frozen=True)
class Derived:
    name: str
    expression: str
    value_type: str

    def value(self, params: Mapping[str, Any]) -> Any:
        v = evaluate(self.expression, params)
        if self.value_type == "int":
            if not float(v).is_integer():
                raise SpaceError([f"{self.name} = {self.expression} gives {v}, not a whole number"])
            return int(v)
        return float(v)


@dataclass(frozen=True)
class Invalid:
    when: Mapping[str, Any]
    why: str
    unless: str | None = None

    def matches(self, params: Mapping[str, Any]) -> bool:
        return all(k in params and same(params[k], v) for k, v in self.when.items())


# --- what a process receives -------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Launch:
    """A trial's settings as a process receives them: arguments, environment, removed variables, a settings file."""

    args: list[str]
    env: dict[str, str]
    unset_env: list[str]
    settings: dict[str, Any]

    def describe(self) -> dict[str, Any]:
        return {"args": list(self.args), "env": dict(self.env), "unset_env": list(self.unset_env),
                "settings": dict(self.settings)}

    def environ(self, base: Mapping[str, str]) -> dict[str, str]:
        """base with unset_env removed and env laid over it."""
        out = {k: v for k, v in base.items() if k not in self.unset_env}
        out.update(self.env)
        return out


# --- the search space --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SearchSpace:
    name: str
    kind: str                                  # inference | training
    format: str                                # atlas.searchspace.v1 | ax-config
    parameters: tuple[Parameter, ...]
    baseline: Mapping[str, Any]                # the searched parameters' baseline values
    fixed: Mapping[str, Any]                   # parameters held at one value (R35 type fixed)
    derived: tuple[Derived, ...]
    invalid: tuple[Invalid, ...]
    fixed_args: tuple[str, ...]
    fixed_env: Mapping[str, str]
    unset_env: tuple[str, ...]
    mapping: Mapping[str, Mapping[str, Mapping[str, Any]]]
    pinned: Mapping[str, Any]
    objective: Mapping[str, Any] | None        # what the file itself says; the night configuration's lane wins
    raw: Mapping[str, Any]
    sha256: str
    path: str | None = None
    note: str | None = None

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.parameters]

    def parameter(self, name: str) -> Parameter:
        for p in self.parameters:
            if p.name == name:
                return p
        raise KeyError(name)

    def problems_with(self, params: Mapping[str, Any]) -> list[str]:
        """Why these values are not one point of the space (unknown names, missing or outside values, invalid)."""
        known = set(self.names)
        out = [f"{k} is not a searched parameter of {self.name}" for k in params if k not in known]
        for p in self.parameters:
            if p.name not in params:
                out.append(f"{p.name} has no value")
            elif not p.contains(params[p.name]):
                out.append(f"{p.name}={params[p.name]!r} is outside the space")
        out += [f"invalid combination {dict(c.when)}: {c.why}" for c in self.invalid if c.matches(params)]
        return out

    def full(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """The searched values over the baseline, in parameter order."""
        return {p.name: params.get(p.name, self.baseline[p.name]) for p in self.parameters}

    def launch(self, params: Mapping[str, Any], *, run_dir: str | os.PathLike[str] = ".") -> Launch:
        """How a process receives these values (module docstring), for either format."""
        values = self.full(params)
        folder = str(run_dir)
        if self.format == R35_SCHEMA:
            args = list(self.fixed_args)
            env = dict(self.fixed_env)
            for p in self.parameters:
                entry = self.mapping.get(p.name, {}).get(value_key(values[p.name]), {})
                args += [str(a).replace("{run_dir}", folder) for a in entry.get("args", [])]
                env.update({k: str(v).replace("{run_dir}", folder) for k, v in entry.get("env", {}).items()})
            unset = [n for n in self.unset_env if n not in env]
            settings = {**dict(self.fixed), **values}
            return Launch(args, env, unset, settings)
        settings = dict(values)
        for d in self.derived:
            settings[d.name] = d.value(settings)
        env: dict[str, str] = {}
        unset: list[str] = []
        for k, v in (self.pinned.get("env") or {}).items():
            if v is None:
                unset.append(k)
            else:
                env[k] = str(v)
        for p in self.parameters:
            if p.setting and not same(values[p.name], self.baseline[p.name]):
                env.update(dict(_ENV_ASSIGN.findall(p.setting)))
        unset = [n for n in unset if n not in env]
        pinned = {k: v for k, v in self.pinned.items() if k != "env"}
        return Launch([], env, unset, {**settings, "pinned": pinned})

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "format": self.format, "path": self.path,
                "sha256": self.sha256, "parameters": [p.describe() for p in self.parameters],
                "baseline": dict(self.baseline), "fixed": dict(self.fixed),
                "derived": {d.name: d.expression for d in self.derived},
                "invalid": [{"when": dict(c.when), "why": c.why, **({"unless": c.unless} if c.unless else {})}
                            for c in self.invalid]}


def _parse_r35(data: Mapping[str, Any], problems: list[str]) -> dict[str, Any]:
    params: list[Parameter] = []
    fixed: dict[str, Any] = {}
    for i, raw in enumerate(data.get("parameters") or []):
        where = f"parameters[{i}]"
        if not isinstance(raw, Mapping) or not isinstance(raw.get("name"), str):
            problems.append(f"{where}: an object with a name")
            continue
        name, kind = raw["name"], raw.get("type")
        if kind == "fixed":
            if "value" not in raw:
                problems.append(f"{where} ({name}): a fixed parameter has a value")
            fixed[name] = raw.get("value")
            continue
        if kind not in ("choice", "range"):
            problems.append(f"{where} ({name}): type is choice, range or fixed, not {kind!r}")
            continue
        p, found = _choice_or_range(name, kind, raw, f"{where} ({name})", type_key="value_type",
                                    ordered_key="is_ordered")
        problems.extend(found)
        if p is not None:
            params.append(p)
    launch = data.get("fixed_launch") or {}
    if not isinstance(launch, Mapping):
        problems.append("fixed_launch is an object")
        launch = {}
    args, env, unset = launch.get("args", []), launch.get("env", {}), launch.get("unset_env", [])
    if not (isinstance(args, list) and all(isinstance(a, str) for a in args)):
        problems.append("fixed_launch.args is a list of strings")
        args = []
    if not (isinstance(env, Mapping) and all(_ENV_NAME.fullmatch(k) for k in env)):
        problems.append("fixed_launch.env maps variable names to values")
        env = {}
    if not (isinstance(unset, list) and all(isinstance(n, str) and _ENV_NAME.fullmatch(n) for n in unset)):
        problems.append("fixed_launch.unset_env is a list of variable names")
        unset = []
    mapping = data.get("mapping") or {}
    if not isinstance(mapping, Mapping):
        problems.append("mapping is an object")
        mapping = {}
    for p in params:
        entry = mapping.get(p.name)
        if not isinstance(entry, Mapping):
            problems.append(f"mapping has no entry for {p.name}")
            continue
        if p.type == "choice":
            missing = [value_key(v) for v in p.values or () if value_key(v) not in entry]
            if missing:
                problems.append(f"mapping.{p.name} has no entry for {', '.join(missing)}")
        for key, item in entry.items():
            if not (isinstance(item, Mapping) and isinstance(item.get("args", []), list)
                    and isinstance(item.get("env", {}), Mapping)):
                problems.append(f"mapping.{p.name}.{key} is {{\"args\": [...], \"env\": {{...}}}}")
    target = data.get("target") or {}
    kind = "training" if isinstance(target, Mapping) and target.get("trainer") else "inference"
    objectives = data.get("objectives")
    objective = None
    if isinstance(objectives, list) and objectives and isinstance(objectives[0], Mapping):
        o = objectives[0]
        objective = {"metric": o.get("name"), "goal": "minimize" if o.get("minimize") else "maximize"}
    return {"parameters": params, "fixed": fixed, "derived": (), "invalid": (), "fixed_args": tuple(args),
            "fixed_env": {k: str(v) for k, v in env.items()}, "unset_env": tuple(unset), "mapping": mapping,
            "pinned": {}, "objective": objective, "kind": kind,
            "name": data.get("name") or Path(str(data.get("note") or "searchspace")).stem}


def _parse_ax_config(data: Mapping[str, Any], problems: list[str]) -> dict[str, Any]:
    params: list[Parameter] = []
    derived: list[Derived] = []
    for i, raw in enumerate(data.get("parameters") or []):
        where = f"parameters[{i}]"
        if not isinstance(raw, Mapping) or not isinstance(raw.get("name"), str):
            problems.append(f"{where}: an object with a name")
            continue
        name, kind = raw["name"], raw.get("kind")
        if kind == "derived":
            expr = raw.get("expression_str")
            if not isinstance(expr, str):
                problems.append(f"{where} ({name}): a derived parameter has an expression_str")
                continue
            derived.append(Derived(name, expr, str(raw.get("parameter_type") or "float")))
            continue
        if kind not in ("choice", "range"):
            problems.append(f"{where} ({name}): kind is choice, range or derived, not {kind!r}")
            continue
        p, found = _choice_or_range(name, kind, raw, f"{where} ({name})", type_key="parameter_type",
                                    ordered_key="is_ordered")
        problems.extend(found)
        if p is not None:
            params.append(p)
    invalid: list[Invalid] = []
    for i, raw in enumerate(data.get("invalid_combinations") or []):
        if not (isinstance(raw, Mapping) and isinstance(raw.get("when"), Mapping) and raw["when"]):
            problems.append(f"invalid_combinations[{i}] has a non-empty when")
            continue
        invalid.append(Invalid(dict(raw["when"]), str(raw.get("why") or ""), raw.get("unless")))
    pinned = data.get("pinned") or {}
    if not isinstance(pinned, Mapping) or not isinstance(pinned.get("env", {}), Mapping):
        problems.append("pinned is an object whose env maps names to values or null")
        pinned = {}
    objective = data.get("objective") if isinstance(data.get("objective"), Mapping) else None
    kind = "training" if pinned or "fixed_effective_batch_size" in data else "inference"
    return {"parameters": params, "fixed": {}, "derived": tuple(derived), "invalid": tuple(invalid),
            "fixed_args": (), "fixed_env": {}, "unset_env": (), "mapping": {}, "pinned": dict(pinned),
            "objective": objective, "kind": kind, "name": data.get("name") or "searchspace"}


def parse_space(data: Any, *, path: str | None = None, kind: str | None = None) -> SearchSpace:
    """A search space from either format's JSON content; SpaceError lists every problem at once."""
    if not isinstance(data, Mapping):
        raise SpaceError(["a search space is a JSON object"])
    problems: list[str] = []
    if data.get("schema") == R35_SCHEMA:
        fmt, parts = R35_SCHEMA, _parse_r35(data, problems)
    elif isinstance(data.get("parameters"), list) and all(isinstance(p, Mapping) and "kind" in p
                                                          for p in data["parameters"]):
        fmt, parts = AX_CONFIG, _parse_ax_config(data, problems)
    else:
        raise SpaceError([f"neither format: schema {R35_SCHEMA!r} (R35) nor Ax-config parameters with a kind (R36)"])
    params: list[Parameter] = parts["parameters"]
    if not params:
        problems.append("the space searches no parameter")
    names = [p.name for p in params]
    for dup in sorted({n for n in names if names.count(n) > 1}):
        problems.append(f"two parameters are named {dup!r}")
    raw_base = data.get("baseline")
    baseline: dict[str, Any] = {}
    if not isinstance(raw_base, Mapping):
        problems.append("baseline maps every searched parameter to its pinned value")
    else:
        for p in params:
            if p.name not in raw_base:
                problems.append(f"the baseline has no value for {p.name}")
            elif not p.contains(raw_base[p.name]):
                problems.append(f"the baseline's {p.name}={raw_base[p.name]!r} is outside the space")
            else:
                v = raw_base[p.name]
                baseline[p.name] = float(v) if p.value_type == "float" else v
        for name, value in parts["fixed"].items():
            if name in raw_base and not same(raw_base[name], value):
                problems.append(f"the baseline's {name}={raw_base[name]!r} differs from its fixed value {value!r}")
        for d in parts["derived"]:
            if d.name in raw_base and len(baseline) == len(params):
                try:
                    if not same(d.value(baseline), raw_base[d.name]):
                        problems.append(f"the baseline's {d.name}={raw_base[d.name]!r} is not {d.expression}")
                except SpaceError as exc:
                    problems.extend(exc.problems)
    if kind is not None and kind not in KINDS:
        problems.append(f"kind is {' or '.join(KINDS)}, not {kind!r}")
    if problems:
        raise SpaceError(problems)
    space = SearchSpace(name=str(parts["name"]), kind=kind or parts["kind"], format=fmt, parameters=tuple(params),
                        baseline=baseline, fixed=parts["fixed"], derived=parts["derived"], invalid=parts["invalid"],
                        fixed_args=parts["fixed_args"], fixed_env=parts["fixed_env"], unset_env=parts["unset_env"],
                        mapping=parts["mapping"], pinned=parts["pinned"], objective=parts["objective"], raw=data,
                        sha256=content_hash(data), path=path,
                        note=data.get("note") if isinstance(data.get("note"), str) else None)
    bad = space.problems_with(space.baseline)
    if bad:
        raise SpaceError([f"the baseline is not a valid point: {'; '.join(bad)}"])
    return space


def load_json(path: str | os.PathLike[str]) -> Any:
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SpaceError([f"cannot read {p}: {exc}"]) from exc
    except ValueError as exc:
        raise SpaceError([f"{p.name} is not JSON: {exc}"]) from exc


def load_space(path: str | os.PathLike[str], *, kind: str | None = None) -> SearchSpace:
    return parse_space(load_json(path), path=shown(path), kind=kind)


# --- the night configuration -------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Constraint:
    metric: str
    op: str                                    # >= | <=
    bound: float

    def text(self) -> str:
        return f"{self.metric} {self.op} {self.bound:g}"

    def holds(self, value: float | None) -> bool | None:
        if value is None:
            return None
        return value >= self.bound if self.op == ">=" else value <= self.bound


def parse_constraint(text: Any) -> Constraint | None:
    """'metric >= number' or 'metric <= number'; None for free text such as a rule still waiting on a measurement."""
    if not isinstance(text, str):
        return None
    m = _CONSTRAINT.fullmatch(text)
    return Constraint(m.group(1), m.group(2), float(m.group(3))) if m else None


@dataclass(frozen=True)
class Worker:
    name: str
    account: str | None
    lanes: tuple[str, ...]
    exclusive: bool
    enabled: Any                               # True | False | "if_granted"
    runner: str                                # local | hub
    stagger_s: float
    precondition: str | None
    raw: Mapping[str, Any]

    @property
    def available(self) -> bool:
        """Enabled outright; a worker that waits on a grant or a precondition is not, until the call names it."""
        return self.enabled is True and not self.precondition


@dataclass(frozen=True)
class Lane:
    name: str
    task: str | None
    worker_names: tuple[str, ...]
    search_space: str
    parameters: tuple[str, ...]
    metric: str
    goal: str
    tracking: tuple[str, ...]
    finalist_tracking: tuple[str, ...]
    constraints: tuple[Constraint, ...]
    constraint_notes: tuple[str, ...]          # constraint texts that are rules, not numbers yet
    restart: Mapping[str, Any]
    grouping: Mapping[str, Any]
    fidelity: Mapping[str, Any]
    repeat_policy: Mapping[str, Any]
    early_stop: Mapping[str, Any]
    initial: tuple[Mapping[str, Any], ...]
    budget: Mapping[str, Any]
    confirmation: Mapping[str, Any]
    raw: Mapping[str, Any]

    @property
    def minimize(self) -> bool:
        return self.goal == "minimize"

    @property
    def metrics(self) -> list[str]:
        out = [self.metric]
        for m in [*self.tracking, *self.finalist_tracking, *(c.metric for c in self.constraints)]:
            if m not in out:
                out.append(m)
        return out

    def better(self, a: float, b: float) -> bool:
        return a < b if self.minimize else a > b


@dataclass(frozen=True)
class Night:
    path: str | None
    sha256: str
    generation_strategy: Mapping[str, Any]
    workers: tuple[Worker, ...]
    lanes: Mapping[str, Lane]
    priors: tuple[Mapping[str, Any], ...]
    raw: Mapping[str, Any] = field(repr=False)

    def worker(self, name: str) -> Worker:
        for w in self.workers:
            if w.name == name:
                return w
        raise KeyError(name)


def _lane(name: str, raw: Any, problems: list[str]) -> Lane | None:
    where = f"lanes.{name}"
    if not isinstance(raw, Mapping):
        problems.append(f"{where} is an object")
        return None
    objective = raw.get("objective") or {}
    metric, goal = objective.get("metric"), objective.get("goal")
    if not (isinstance(metric, str) and _METRIC.fullmatch(metric)) or goal not in GOALS:
        problems.append(f"{where}.objective is {{\"metric\": <dotted name>, \"goal\": minimize | maximize}}")
    if not isinstance(raw.get("search_space"), str):
        problems.append(f"{where}.search_space names the search-space file")
    params = raw.get("parameters") or []
    if not (isinstance(params, list) and all(isinstance(p, str) for p in params)):
        problems.append(f"{where}.parameters is a list of names")
        params = []
    texts = raw.get("outcome_constraints")
    if texts is None and name == "inference":
        texts = list(DEFAULT_INFERENCE_CONSTRAINTS)
    constraints, notes = [], []
    for t in texts or []:
        c = parse_constraint(t)
        (constraints.append(c) if c else notes.append(str(t)))
    initial = raw.get("initial_arms") or []
    if not (isinstance(initial, list) and all(isinstance(a, Mapping) and isinstance(a.get("name"), str)
                                              and isinstance(a.get("parameters", {}), Mapping) for a in initial)):
        problems.append(f"{where}.initial_arms is a list of {{\"name\", \"parameters\"}}")
        initial = []
    names = [a["name"] for a in initial]
    if len(set(names)) != len(names):
        problems.append(f"{where}.initial_arms repeats a name")
    if sum(1 for a in initial if a.get("attach_as") == "baseline") > 1:
        problems.append(f"{where}.initial_arms has more than one baseline")
    budget = raw.get("budget") or {}
    for key in ("minutes",):
        if key in budget and not (is_number(budget[key]) and budget[key] > 0):
            problems.append(f"{where}.budget.{key} is a positive number")

    def tup(key: str) -> tuple[str, ...]:
        v = raw.get(key) or []
        return tuple(str(x) for x in v) if isinstance(v, list) else ()

    def obj(key: str) -> Mapping[str, Any]:
        v = raw.get(key)
        return v if isinstance(v, Mapping) else {}

    return Lane(name=name, task=raw.get("task"), worker_names=tup("worker_names"),
                search_space=str(raw.get("search_space") or ""), parameters=tuple(params), metric=str(metric),
                goal=str(goal), tracking=tup("tracking"), finalist_tracking=tup("finalist_tracking"),
                constraints=tuple(constraints), constraint_notes=tuple(notes), restart=obj("restart"),
                grouping=obj("grouping"), fidelity=obj("fidelity"), repeat_policy=obj("repeat_policy"),
                early_stop=obj("early_stop"), initial=tuple(initial), budget=budget, confirmation=obj("confirmation"),
                raw=raw)


def parse_night(data: Any, *, path: str | None = None) -> Night:
    if not isinstance(data, Mapping):
        raise SpaceError(["the night configuration is a JSON object"])
    problems: list[str] = []
    if data.get("schema") != NIGHT_SCHEMA:
        problems.append(f"schema must be {NIGHT_SCHEMA!r}, not {data.get('schema')!r}")
    workers: list[Worker] = []
    for i, w in enumerate(data.get("workers") or []):
        if not (isinstance(w, Mapping) and isinstance(w.get("name"), str)):
            problems.append(f"workers[{i}] is an object with a name")
            continue
        workers.append(Worker(name=w["name"], account=w.get("account"), lanes=tuple(w.get("lanes") or ()),
                              exclusive=bool(w.get("exclusive")), enabled=w.get("enabled", True),
                              runner=str(w.get("runner") or "local"), stagger_s=float(w.get("stagger_s") or 0),
                              precondition=w.get("precondition"), raw=w))
    known = {w.name for w in workers}
    lanes: dict[str, Lane] = {}
    raw_lanes = data.get("lanes") or {}
    for name in LANES:
        if name in raw_lanes:
            lane = _lane(name, raw_lanes[name], problems)
            if lane is not None:
                lanes[name] = lane
                missing = [w for w in lane.worker_names if w not in known]
                if missing:
                    problems.append(f"lanes.{name}.worker_names names unknown workers: {', '.join(missing)}")
    if not lanes:
        problems.append("the configuration names no inference or training lane")
    gs = data.get("generation_strategy") or {}
    if not isinstance(gs, Mapping):
        problems.append("generation_strategy is an object")
        gs = {}
    if problems:
        raise SpaceError(problems)
    priors = tuple(p for p in data.get("priors") or [] if isinstance(p, Mapping))
    return Night(path=path, sha256=content_hash(data), generation_strategy=gs, workers=tuple(workers),
                 lanes=lanes, priors=priors, raw=data)


def load_night(path: str | os.PathLike[str] = NIGHT_CONFIG) -> Night:
    return parse_night(load_json(path), path=shown(path))


def lane_of(night: Night, name: str) -> Lane:
    if name not in night.lanes:
        raise SpaceError([f"the night configuration has no {name!r} lane (it has {', '.join(night.lanes)})"])
    return night.lanes[name]


def check_lane(lane: Lane, space: SearchSpace) -> list[str]:
    """Where a lane and its search space disagree: a lane parameter the space lacks, or the reverse."""
    out = [f"lane {lane.name} names {p}, which {space.name} does not search" for p in lane.parameters
           if p not in space.names]
    if lane.parameters:
        out += [f"{space.name} searches {p}, which lane {lane.name} does not name" for p in space.names
                if p not in lane.parameters]
    return out


def initial_arms(lane: Lane, space: SearchSpace) -> list[dict[str, Any]]:
    """The lane's initial arms as full parameter sets over the space's baseline, in configuration order:
    [{"name", "parameters", "attach_as": baseline | trial, "restart", "priors"}]. The baseline arm comes first even
    when the configuration lists it later; an arm outside the space is a SpaceError."""
    arms: list[dict[str, Any]] = []
    problems: list[str] = []
    for a in lane.initial:
        params = space.full(a.get("parameters") or {})
        bad = [k for k in (a.get("parameters") or {}) if k not in space.names] + space.problems_with(params)
        if bad:
            problems.append(f"initial arm {a['name']}: {'; '.join(str(b) for b in bad)}")
            continue
        arms.append({"name": a["name"], "parameters": params, "attach_as": a.get("attach_as", "trial"),
                     "restart": a.get("restart"), "priors": list(a.get("priors") or [])})
    if problems:
        raise SpaceError(problems)
    if not any(a["attach_as"] == "baseline" for a in arms):
        arms.insert(0, {"name": "baseline", "parameters": space.full({}), "attach_as": "baseline",
                        "restart": None, "priors": []})
    arms.sort(key=lambda a: a["attach_as"] != "baseline")
    return arms
