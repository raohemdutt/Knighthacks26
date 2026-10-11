"""The suggestion strategies behind the settings optimizer: Ax (imported only when used) and a stub for tests.

    from lab.optimize_ax import AxStrategy, StubStrategy, make_strategy
    s = make_strategy("ax", space, lane, generation_strategy=night.generation_strategy, seed=49)
    s.attach(baseline_params, baseline=True)            # the baseline arm, then the prior arms: no Sobol points
    index, params = s.next(fixed={"linear_kernel": "exllama_fp16"})   # one trial per free worker, others pending
    s.complete(index, {"joules_per_token.b1": (41.7, 1.2), "decode_tok_s.b1": (2.9, 0.05)})

Both strategies number trials 0, 1, 2, ... in the order they are attached or asked for, which is the order of the
run directory's trial files, so a resume replays the files into a fresh strategy and gets the same numbering.

AxStrategy is Ax 1.x's ask-tell Client (ax-platform pinned in lab/requirements-optimize.txt; API checked on 1.3.1):
configure_experiment with one ChoiceParameterConfig or RangeParameterConfig per searched parameter,
configure_optimization with the lane's objective (a leading "-" to minimize) and its outcome constraints,
configure_generation_strategy with the night configuration's generation_strategy keys (method fast, initialization
budget 0 and the initial arms counting as initialization), attach_baseline and attach_trial for the initial arms,
get_next_trials(max_trials=1, fixed_parameters=...) per free worker while other trials are still running (Ax holds
them as pending points), complete_trial, mark_trial_failed, attach_data with a progression and
should_stop_trial_early for the training lane's percentile early stopping, mark_trial_early_stopped. Metric names
lose their dots on the way in (Ax parses objective strings as expressions): joules_per_token.b1 is
joules_per_token__b1 to Ax and joules_per_token.b1 everywhere else.

StubStrategy needs nothing installed: it enumerates the space (every choice value, and the bounds and midpoint of a
range) in a seeded order, skips points already attached and invalid combinations, honours fixed parameters, and
never repeats a point. It is the strategy of the tests and of `--strategy stub` dry runs; it models nothing.
"""
from __future__ import annotations

import inspect
import itertools
import logging
import random
import warnings
from collections.abc import Mapping, Sequence
from typing import Any

from lab.optimize_space import Lane, SearchSpace, same

STRATEGIES = ("ax", "stub")
GS_KEYS = ("method", "initialization_budget", "initialization_random_seed", "initialize_with_center",
           "use_existing_trials_for_initialization", "allow_exceeding_initialization_budget",
           "min_observed_initialization_trials")


class StrategyError(RuntimeError):
    """The strategy cannot go on (Ax missing, or Ax and the run directory disagree)."""


def ax_metric(name: str) -> str:
    return name.replace(".", "__")


def plain(value: Any) -> Any:
    """numpy scalars from Ax as plain JSON values."""
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        return value.item()
    return value


def ax_installed() -> bool:
    import importlib.util
    return importlib.util.find_spec("ax") is not None and importlib.util.find_spec("botorch") is not None


class StubStrategy:
    name = "stub"

    def __init__(self, space: SearchSpace, lane: Lane, *, seed: int = 0) -> None:
        self.space = space
        self.lane = lane
        self.trials: list[dict[str, Any]] = []
        grids = []
        for p in space.parameters:
            if p.type == "choice":
                grids.append(list(p.values or ()))
            else:
                low, high = p.bounds or (0, 0)
                mid = (low + high) / 2
                grids.append([low, int(round(mid)) if p.value_type == "int" else mid, high])
        points = [dict(zip(space.names, combo)) for combo in itertools.product(*grids)]
        random.Random(seed).shuffle(points)
        self.points = points

    def _seen(self, params: Mapping[str, Any]) -> bool:
        return any(all(same(t["parameters"][k], params[k]) for k in params) for t in self.trials)

    def attach(self, params: Mapping[str, Any], *, baseline: bool = False) -> int:
        self.trials.append({"parameters": dict(params), "status": "running", "observation": None})
        return len(self.trials) - 1

    def next(self, fixed: Mapping[str, Any] | None = None) -> tuple[int, dict[str, Any]] | None:
        for point in self.points:
            if fixed and not all(same(point[k], v) for k, v in fixed.items()):
                continue
            if self._seen(point) or self.space.problems_with(point):
                continue
            return self.attach(point), dict(point)
        return None

    def complete(self, index: int, observation: Mapping[str, Any]) -> None:
        self.trials[index].update(status="completed", observation=dict(observation))

    def fail(self, index: int) -> None:
        self.trials[index]["status"] = "failed"

    def early_stopped(self, index: int, observation: Mapping[str, Any]) -> None:
        self.trials[index].update(status="early_stopped", observation=dict(observation))

    def progress(self, index: int, progression: int, observation: Mapping[str, Any]) -> None:
        return None

    def should_stop(self, index: int) -> bool:
        return False

    def predict(self, params: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]] | None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "points": len(self.points)}


class AxStrategy:
    name = "ax"

    def __init__(self, space: SearchSpace, lane: Lane, *, generation_strategy: Mapping[str, Any] | None = None,
                 seed: int | None = None) -> None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import ax
                from ax.utils.common.logger import set_stderr_log_level
        except ImportError as exc:
            raise StrategyError(f"Ax is not installed here ({exc}): pip install -r lab/requirements-optimize.txt") \
                from exc
        set_stderr_log_level(logging.WARNING)
        self.space = space
        self.lane = lane
        gs = dict(generation_strategy or {})
        seed = seed if seed is not None else gs.get("initialization_random_seed")
        self.client = ax.Client(random_seed=seed)
        configs = []
        for p in space.parameters:
            if p.type == "range":
                configs.append(ax.RangeParameterConfig(name=p.name, bounds=tuple(p.bounds or ()),
                                                       parameter_type=p.value_type,
                                                       scaling="log" if p.log_scale else None))
            else:
                configs.append(ax.ChoiceParameterConfig(name=p.name, values=list(p.values or ()),
                                                        parameter_type=p.value_type, is_ordered=p.ordered))
        objective = ("-" if lane.minimize else "") + ax_metric(lane.metric)
        constraints = [f"{ax_metric(c.metric)} {c.op} {c.bound:g}" for c in lane.constraints]
        tracking = [ax_metric(m) for m in lane.metrics if m != lane.metric
                    and m not in [c.metric for c in lane.constraints]]
        wanted = {k: gs[k] for k in GS_KEYS if k in gs}
        if seed is not None:
            wanted["initialization_random_seed"] = seed
        accepted = set(inspect.signature(self.client.configure_generation_strategy).parameters)
        self.generation = {k: v for k, v in wanted.items() if k in accepted}
        self.ignored = sorted(set(wanted) - accepted)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.client.configure_experiment(parameters=configs, name=f"{space.name}-{lane.name}")
            self.client.configure_optimization(objective=objective, outcome_constraints=constraints or None)
            if tracking:
                self.client.configure_tracking_metrics(tracking)
            self.client.configure_generation_strategy(**self.generation)
        self.count = 0

    def _check(self, index: int) -> int:
        if index != self.count:
            raise StrategyError(f"Ax numbered a trial {index} where the run directory expects {self.count}")
        self.count += 1
        return index

    def _observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        out = {}
        for k, v in observation.items():
            if isinstance(v, (tuple, list)):
                out[ax_metric(k)] = (float(v[0]), float(v[1])) if v[1] is not None else float(v[0])
            else:
                out[ax_metric(k)] = float(v)
        return out

    def attach(self, params: Mapping[str, Any], *, baseline: bool = False) -> int:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if baseline:
                index = self.client.attach_baseline(parameters=dict(params))
            else:
                index = self.client.attach_trial(parameters=dict(params))
        return self._check(int(index))

    def next(self, fixed: Mapping[str, Any] | None = None) -> tuple[int, dict[str, Any]] | None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                trials = self.client.get_next_trials(max_trials=1, fixed_parameters=dict(fixed) if fixed else None)
            except Exception as exc:
                if type(exc).__name__ in ("GenerationStrategyCompleted", "OptimizationComplete", "SearchSpaceExhausted"):
                    return None
                raise
        if not trials:
            return None
        index, params = next(iter(trials.items()))
        return self._check(int(index)), {k: plain(v) for k, v in params.items()}

    def complete(self, index: int, observation: Mapping[str, Any]) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.client.complete_trial(trial_index=index, raw_data=self._observation(observation))

    def fail(self, index: int) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mark = getattr(self.client, "mark_trial_failed", None) or self.client.mark_trial_abandoned
            mark(trial_index=index)

    def early_stopped(self, index: int, observation: Mapping[str, Any]) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if observation:
                self.client.attach_data(trial_index=index, raw_data=self._observation(observation))
            self.client.mark_trial_early_stopped(trial_index=index)

    def progress(self, index: int, progression: int, observation: Mapping[str, Any]) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.client.attach_data(trial_index=index, raw_data=self._observation(observation),
                                    progression=progression)

    def should_stop(self, index: int) -> bool:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return bool(self.client.should_stop_trial_early(trial_index=index))
            except Exception:
                return False

    def predict(self, params: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]] | None:
        """The model's mean and SEM per metric at these points, or None while the current node cannot predict."""
        names = {ax_metric(m): m for m in self.lane.metrics}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                predicted = self.client.predict([dict(p) for p in params])
            except Exception:
                return None
        return [{names.get(k, k): {"mean": float(v[0]), "sem": float(v[1])} for k, v in row.items() if k in names}
                for row in predicted]

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "generation_strategy": self.generation, "ignored_keys": self.ignored}


def make_strategy(kind: str, space: SearchSpace, lane: Lane, *, generation_strategy: Mapping[str, Any] | None = None,
                  seed: int | None = None) -> StubStrategy | AxStrategy:
    if kind == "stub":
        return StubStrategy(space, lane, seed=seed or 0)
    if kind == "ax":
        return AxStrategy(space, lane, generation_strategy=generation_strategy, seed=seed)
    raise StrategyError(f"strategy is one of {', '.join(STRATEGIES)}, not {kind!r}")
