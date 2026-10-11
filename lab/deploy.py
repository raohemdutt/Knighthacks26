"""Deploy a packaged model to the box: pull it from pod B through the hub proxy, check it, hot-swap, publish (L6).

    python -m lab deploy [VERSION] [--from-pod C|D] [--port 8300] [--dry-run]
                         [--models-dir data/models] [--events-dir data/events] [--part-mib 64]
                         [--llm-url http://127.0.0.1:8080] [--app-url http://127.0.0.1:8000] [--no-app]
                         [--service atlas-llm] [--quiet]

Run it ON THE BOX from the repo root. One call does the whole chain:

1. resolve: VERSION, or the newest complete package under the pod's package root (the server's listing, then each
   version's manifest.json; the newest `created` wins);
2. pull: every file the manifest names, in Range parts through https://<hub>/user/<account>/proxy/<port>/
   (lab/package_pull.py, R10: resumable, a sha256 check per file, manifest.json written last), into
   <models-dir>/<version>/; the hub token and the account's hub address come from the .env search lab/hub.py does;
3. swap: jetson/atlas_swap.py's swap (R7): every sha256 checked again against manifest.json, `current` switched,
   the model server restarted, its health and test question waited for; a swap that fails is put back by the swap;
   `previous` keeps the version that was serving, so `python jetson/atlas_swap.py rollback` goes back to it;
4. check: one request through the box's llama-server the way the gateway sends it (runtime/llama.py: POST
   /v1/chat/completions, messages, max_tokens, temperature 0.2), which must answer with text;
   a failed check switches back to the version that was serving (and restores `previous`);
5. publish: learning.model.deployed (version, previous_version, base, task, package_mb, pulled_seconds,
   checksums_ok) to the running app (POST /api/learning/deployed, which publishes it on the app's bus and so into
   data/events) or, when the app cannot be reached or --no-app is given, straight into the box's event log
   (<events-dir>/YYYY-MM-DD.jsonl). Any failure after the version is known is published as learning.run.progress
   with stage "failed" into the event log, with the reason.

--dry-run resolves the version, reads its manifest and file sizes from the pod and prints the plan; nothing is
written and nothing is switched. --from-pod picks the account (D, the trainer's on the SJC hub, by default) and
--port the package server's port on the pod (8300, lab/package_server.py).

Answer: one JSON object, ok and outcome (deployed, already_serving, dry_run, error; kind says which step failed:
no_credentials, no_package, pull, swap, check, publish) and the pull's mb_s, bytes and wall_seconds.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import httpx

from contracts.learning import ModelDeployed, ModelManifest, RunProgress
from lab import package_pull, package_server
from lab.hub import MissingCredentials, PodRoleError, load_accounts
from lab.package_pull import PackageClient, PullError
from lab.transfer import RESERVED_VERSIONS

CLI_VERB = "deploy"   # registers python -m lab deploy

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ACCOUNT = "D"
CHECK_PROMPT = "Reply with the single word: ok"
CHECK_TIMEOUT_S = 120.0
APP_TIMEOUT_S = 10.0


class DeployError(Exception):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message


# --- the pod side: which version, and what is in it ---------------------------------------------------------------

def list_versions(client: PackageClient, *, sleep: Callable[[float], None] = time.sleep) -> list[str]:
    raw = package_pull.fetch_small(client, client.proxy_path(), sleep=sleep)
    try:
        return [str(name) for name in json.loads(raw)["versions"]]
    except (ValueError, KeyError, TypeError):
        raise DeployError("no_package", "the package server's listing is not {\"versions\": [...]}") from None


def resolve_version(client: PackageClient, version: str | None = None, *,
                    sleep: Callable[[float], None] = time.sleep) -> tuple[str, ModelManifest]:
    """(version, manifest): the one asked for, or the newest complete one (by manifest `created`) on the pod."""
    if version is not None:
        check_version(version)
        try:
            return version, package_pull.fetch_manifest(client, version, sleep=sleep)[1]
        except PullError as err:
            raise DeployError("no_package", f"{version} on the pod: {client.scrub(err.message)}") from None
    found: list[tuple[ModelManifest, str]] = []
    for name in list_versions(client, sleep=sleep):
        if name in RESERVED_VERSIONS:
            continue
        try:
            found.append((package_pull.fetch_manifest(client, name, sleep=sleep)[1], name))
        except PullError:
            continue                       # a package still being written has no manifest yet
    if not found:
        raise DeployError("no_package", "the pod's package root has no complete package (no version with a manifest.json)")
    manifest, name = max(found, key=lambda item: (item[0].created, item[1]))
    return name, manifest


def check_version(version: str) -> str:
    if not package_server.NAME_RE.match(version) or version in RESERVED_VERSIONS:
        raise DeployError("no_package", f"{version!r} cannot be a version")
    return version


def file_sizes(client: PackageClient, version: str, *, sleep: Callable[[float], None] = time.sleep) -> dict[str, int]:
    raw = package_pull.fetch_small(client, client.proxy_path(version), sleep=sleep)
    try:
        return {item["name"]: int(item["size"]) for item in json.loads(raw)["files"]}
    except (ValueError, KeyError, TypeError):
        return {}


# --- the box side: the model store, the swap, the check ------------------------------------------------------------

class Box(Protocol):
    def current(self) -> str | None: ...
    def previous(self) -> str | None: ...
    def verify(self, version: str) -> list[str]: ...
    def swap(self, version: str) -> None: ...                          # raises DeployError("swap", ...) when it failed
    def swap_back(self, version: str, previous: str | None) -> str | None: ...   # None when back and healthy


def _atlas_swap() -> Any:
    """jetson/atlas_swap.py, a script beside the box's files, not a package."""
    spec = importlib.util.spec_from_file_location("atlas_swap", REPO / "jetson" / "atlas_swap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class JetsonBox:
    """The real box: runtime/modelstore.py under models_dir and jetson/atlas_swap.py's swap. The swap restarts the
    model server with `sudo -n systemctl restart <service>`, as atlas_swap.py does; the app is not told by the swap
    (announce=None): deploy publishes the one learning.model.deployed itself, with the package's numbers."""

    def __init__(self, models_dir: str | Path, *, service: str = "atlas-llm", llm_url: str,
                 log: Callable[[str], None] = print, restart: Callable[[], None] | None = None,
                 wait: Callable[[], str | None] | None = None) -> None:
        from runtime.modelstore import ModelStore
        self.swapper = _atlas_swap()
        self.store = ModelStore(models_dir)
        self.service = service
        self.log = log
        self._restart = restart or self._systemctl
        self._wait = wait or (lambda: self.swapper.wait_until_serving(llm_url))

    def _systemctl(self) -> None:
        subprocess.run(["sudo", "-n", "systemctl", "restart", self.service], check=True)

    def current(self) -> str | None:
        return self.store.current()

    def previous(self) -> str | None:
        return self.store.previous()

    def verify(self, version: str) -> list[str]:
        return self.store.verify(version)

    def swap(self, version: str) -> None:
        from runtime.modelstore import ModelStoreError
        try:
            self.swapper.swap(self.store, version, restart=self._restart, check=self._wait, log=self.log, announce=None)
        except (ModelStoreError, subprocess.CalledProcessError, OSError) as exc:
            raise DeployError("swap", str(exc)) from None

    def swap_back(self, version: str, previous: str | None) -> str | None:
        try:
            self.store.activate(version)
            self.store.set_previous(previous)
            self._restart()
        except Exception as exc:      # noqa: BLE001 - whatever went wrong is the answer's text
            return f"{type(exc).__name__}: {exc}"
        return self._wait()


def gateway_check(llm_url: str, *, transport: httpx.BaseTransport | None = None,
                  timeout_s: float = CHECK_TIMEOUT_S) -> str | None:
    """None when llama-server answers one chat request, sent the way runtime/llama.py's _complete sends it, with
    text; otherwise what went wrong."""
    body = {"messages": [{"role": "user", "content": CHECK_PROMPT}], "max_tokens": 16, "temperature": 0.2}
    try:
        with httpx.Client(base_url=llm_url, transport=transport, timeout=timeout_s) as http:
            r = http.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"].get("content") or ""
    except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
        return f"the gateway's request failed: {type(exc).__name__}: {exc}"
    return None if text.strip() else "the model answered the gateway's request with nothing"


# --- publishing ------------------------------------------------------------------------------------------------

class Publisher:
    """learning.model.deployed to the running app, else into the event log; a failure into the event log."""

    def __init__(self, events_dir: str | Path, *, app_url: str | None, token: str | None = None,
                 transport: httpx.BaseTransport | None = None) -> None:
        self.events_dir = Path(events_dir)
        self.app_url = app_url
        self.token = token
        self.transport = transport

    def to_log(self, event: Any) -> None:
        from bus.bus import EventBus
        asyncio.run(EventBus(self.events_dir).publish(event))

    def deployed(self, event: ModelDeployed) -> tuple[str, str | None]:
        """(where it went: app or log, why the app was not used)."""
        why = None
        if self.app_url:
            body = event.model_dump(mode="json", include={"model_version", "previous_version", "base", "task",
                                                          "package_mb", "pulled_seconds", "checksums_ok"})
            try:
                with httpx.Client(transport=self.transport, timeout=APP_TIMEOUT_S) as http:
                    r = http.post(self.app_url.rstrip("/") + "/api/learning/deployed", json=body,
                                  headers={"Authorization": f"Bearer {self.token}"} if self.token else {})
                if r.status_code == 200:
                    return "app", None
                why = f"the app answered HTTP {r.status_code}"
            except httpx.HTTPError as exc:
                why = f"the app could not be reached: {type(exc).__name__}"
        self.to_log(event)
        return "log", why

    def failed(self, event: RunProgress) -> str:
        self.to_log(event)
        return "log"


# --- the deploy ------------------------------------------------------------------------------------------------

@dataclasses.dataclass
class Plan:
    version: str
    manifest: ModelManifest
    sizes: dict[str, int]


def _mb(n: int) -> float:
    return round(n / 1e6, 1)


def plan_view(plan: Plan, client: PackageClient, box: Box, models_dir: Path) -> dict[str, Any]:
    m = plan.manifest
    names = [name for name, _ in package_pull.package_files(m)]
    total = sum(plan.sizes.get(name, 0) for name in names)
    try:
        free = shutil.disk_usage(models_dir if models_dir.exists() else models_dir.parent).free
    except OSError:
        free = None
    return {
        "version": plan.version, "base": m.base, "task": m.task, "scores": m.scores,
        "files": [{"name": name, "mb": _mb(plan.sizes.get(name, 0)), "sha256": m.sha256.get(name)} for name in names],
        "package_mb": _mb(total), "disk_free_mb": None if free is None else _mb(free),
        "from": client.proxy_path(plan.version), "hub": client.host, "account": client.account.letter,
        "into": str(models_dir / plan.version), "current": box.current(), "previous": box.previous(),
        "steps": ["pull every file in Range parts through the hub proxy, sha256 per file, manifest.json last",
                  f"jetson/atlas_swap.py swap {plan.version}: verify the sha256s, switch current, restart, wait",
                  "one chat request through llama-server as the gateway sends it",
                  "publish learning.model.deployed (to the app, else the event log)",
                  f"on a failed check: back to {box.current()} and learning.run.progress failed"],
    }


def deploy(client: PackageClient, box: Box, publisher: Publisher, *, version: str | None = None,
           models_dir: str | Path = "data/models", check: Callable[[], str | None],
           part_bytes: int = package_pull.DEFAULT_PART_BYTES, dry_run: bool = False,
           sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.perf_counter,
           progress: Callable[[str], None] | None = None) -> dict[str, Any]:
    """See the module docstring. A failure is an answer (ok false, kind), never an exception."""
    say = progress or (lambda line: None)
    models_dir = Path(models_dir)
    run_id = time.strftime("deploy-%Y%m%d-%H%M%S", time.gmtime())
    answer: dict[str, Any] = {"command": "deploy", "account": client.account.letter, "hub": client.host,
                              "port": client.port, "run_id": run_id}
    manifest: ModelManifest | None = None
    try:
        name, manifest = resolve_version(client, version, sleep=sleep)
        answer.update({"version": name, "base": manifest.base, "task": manifest.task})
        if dry_run:
            plan = Plan(name, manifest, file_sizes(client, name, sleep=sleep))
            return {**answer, "ok": True, "outcome": "dry_run", "plan": plan_view(plan, client, box, models_dir)}
        old, old_previous = box.current(), box.previous()
        answer["previous_version"] = old
        if old == name:
            return {**answer, "ok": True, "outcome": "already_serving", "previous_version": box.previous()}

        say(f"pulling {name} from pod {client.account.letter} through {client.host}")
        pulled = package_pull.pull_package(client, name, models_dir, part_bytes=part_bytes, sleep=sleep, clock=clock,
                                           progress=progress, run_id=run_id)
        answer["pull"] = {key: pulled.get(key) for key in ("outcome", "bytes", "package_bytes", "seconds",
                                                           "wall_seconds", "mb_s", "parts", "retries")}
        if not pulled["ok"]:
            raise DeployError("pull", f"{pulled.get('kind')}: {pulled.get('error')}")
        problems = box.verify(name)
        if problems:
            raise DeployError("pull", "the installed copy does not match its manifest: " + "; ".join(problems))
        answer["checksums_ok"] = True

        say(f"swapping {old} -> {name}")
        box.swap(name)
        say("asking the model one question the way the gateway does")
        failure = check()
        if failure is not None:
            answer["check"] = failure
            if old:
                again = box.swap_back(old, old_previous)
                answer["rolled_back_to"] = old
                if again is not None:
                    answer["rollback_problem"] = again
            raise DeployError("check", failure)
        answer["check"] = "ok"

        event = ModelDeployed(model_version=name, previous_version=old, base=manifest.base, task=manifest.task,
                              package_mb=_mb(int(pulled.get("package_bytes") or 0)),
                              pulled_seconds=pulled.get("wall_seconds"), checksums_ok=True)
        try:
            where, why = publisher.deployed(event)
        except Exception as exc:     # noqa: BLE001 - the model is serving; say why the event was not written
            raise DeployError("publish", f"{name} is serving but learning.model.deployed was not written: "
                                         f"{type(exc).__name__}: {exc}") from None
        answer.update({"published": where, "event_id": event.event_id})
        if why:
            answer["app_note"] = why
        return {**answer, "ok": True, "outcome": "deployed", "package_mb": event.package_mb,
                "pulled_seconds": event.pulled_seconds, "mb_s": answer["pull"]["mb_s"]}
    except DeployError as err:
        answer.update({"ok": False, "outcome": "error", "kind": err.kind, "error": client.scrub(err.message)})
    if not dry_run and answer.get("version") and answer["kind"] != "publish":
        failed = RunProgress(task=(manifest.task if manifest and manifest.task else answer["version"]), run_id=run_id,
                             stage="failed", progress=1.0,
                             message=f"deploy of {answer['version']} failed at {answer['kind']}: {answer['error']}"[:500])
        try:
            answer["published"] = publisher.failed(failed)
            answer["event_id"] = failed.event_id
        except Exception as exc:     # noqa: BLE001
            answer["publish_error"] = f"{type(exc).__name__}: {exc}"
    return answer


# --- the command line ------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m lab deploy", description=__doc__.splitlines()[0])
    p.add_argument("version", nargs="?", help="the package version (default: the newest complete one on the pod)")
    p.add_argument("--from-pod", dest="account", choices=["C", "D"], default=DEFAULT_ACCOUNT,
                   help="the hub account whose pod serves the package (default D, the trainer's)")
    p.add_argument("--port", type=int, default=package_server.DEFAULT_PORT, help="the package server's port on the pod")
    p.add_argument("--part-mib", type=float, default=64, help="Range part size in MiB (under 96)")
    p.add_argument("--models-dir", default="data/models")
    p.add_argument("--events-dir", default="data/events")
    p.add_argument("--llm-url", default=f"http://127.0.0.1:{os.environ.get('ATLAS_LLM_PORT', '8080')}")
    p.add_argument("--app-url", default=os.environ.get("ATLAS_URL", "http://127.0.0.1:8000"))
    p.add_argument("--no-app", action="store_true", help="write learning.model.deployed straight into the event log")
    p.add_argument("--service", default="atlas-llm", help="the systemd unit the swap restarts")
    p.add_argument("--dry-run", action="store_true", help="print the plan; write and switch nothing")
    p.add_argument("--quiet", action="store_true")
    return p


def run(argv: Sequence[str], *, environ: Mapping[str, str] | None = None, env_files: Sequence[Path] | None = None,
        transport: httpx.BaseTransport | None = None, local_transport: httpx.BaseTransport | None = None,
        box: Box | None = None, sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.perf_counter) -> dict[str, Any]:
    """One deploy. transport replaces the network to the hub, local_transport the box's llama-server and app, box
    the model store and swap: for tests and rehearsals."""
    args = build_parser().parse_args(list(argv))
    environ = os.environ if environ is None else environ
    try:
        accounts = load_accounts([args.account], environ=environ, env_files=env_files, require_role=True)
    except PodRoleError as exc:
        return {"ok": False, "command": "deploy", "outcome": "error", "kind": "no_credentials", "error": str(exc)}
    except MissingCredentials as exc:
        return {"ok": False, "command": "deploy", "outcome": "error", "kind": "no_credentials", "missing": exc.missing,
                "error": "keys missing from the environment and the .env files (values are never printed)"}
    progress = None if args.quiet else (lambda line: print(f"deploy: {line}", file=sys.stderr, flush=True))
    client = PackageClient(accounts[args.account], args.port, transport=transport, sleep=sleep, environ=environ)
    try:
        if box is None:
            box = JetsonBox(args.models_dir, service=args.service, llm_url=args.llm_url,
                            log=progress or (lambda line: None))
        publisher = Publisher(args.events_dir, app_url=None if args.no_app else args.app_url,
                              token=environ.get("ATLAS_TOKEN"), transport=local_transport)
        return deploy(client, box, publisher, version=args.version, models_dir=args.models_dir,
                      check=lambda: gateway_check(args.llm_url, transport=local_transport),
                      part_bytes=package_pull.check_part_bytes(int(args.part_mib * package_pull.MIB)),
                      dry_run=args.dry_run, sleep=sleep, clock=clock, progress=progress)
    finally:
        client.close()


def main(argv: list[str] | None = None) -> dict[str, Any]:
    """Pull a packaged model from pod B through the hub proxy, verify, hot-swap, publish learning.model.deployed."""
    return run(list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    answer = main()
    print(json.dumps(answer))
    sys.exit(0 if answer.get("ok") else 1)
