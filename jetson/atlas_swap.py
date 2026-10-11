"""Switch the on-device model to another installed version, or back (R7).

    python jetson/atlas_swap.py status
    python jetson/atlas_swap.py swap atlas-v2-desk-objects [--trust-verified]
    python jetson/atlas_swap.py rollback [--trust-verified]

Run from the repo root so `runtime` imports (PYTHONPATH=.). A swap:
  1. checks every file of the new version against the sha256s in its manifest (the old model keeps serving meanwhile);
  2. points `data/models/current` at it and restarts the `atlas-llm` service;
  3. waits for llama-server to load it and answer a test question;
  4. if that fails, puts the old version back and restarts again, and exits with an error;
     if it works, tells the app (POST /api/learning/deployed) so the phone's Teach screen shows the new version.
Each step also prints `STAGE <name> <start|done|failed> <unix seconds>` (verify, switch, restart, check, publish,
rollback), which the Learning lab's deploy button (contexts/learning/lab_deploy.py) reads from the log.
The model is about 3.6 GB and the box has 8 GB shared with the OS and the detector, so the new model cannot be
loaded beside the old one: the phone gets no answers from the model for the ~40 s the reload takes.

--trust-verified (off by default; `python -m lab deploy` turns it on): a file that lab/package_pull.py hashed and
recorded in <version>/.verified.json, and whose size and mtime have not moved since, is not hashed again by the
check in step 1 or by the switch in step 2 (runtime/modelstore.py). A file the record does not vouch for is hashed
as before. Without the flag every file is hashed at both points, about 40 s each for the 3.4 GB package on the box.
A check that hashed and passed writes the record itself (same fields and writer marker as the pull's), so a version
never pulled, such as atlas-base, or pulled before the record existed, is hashed once on its first trusting swap and
not at all on the next ones while its files keep their size and mtime.
"""

import argparse
import os
import subprocess
import sys
import time
from collections.abc import Callable

import httpx

from runtime.modelstore import ModelStore, ModelStoreError


def wait_until_serving(url: str, timeout_s: float = 240, sleep: Callable[[float], None] = time.sleep) -> str | None:
    """None when llama-server is up and answers a test question; otherwise what went wrong."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if httpx.get(url + "/health", timeout=3).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        sleep(2)
    else:
        return f"llama-server did not become healthy within {timeout_s:.0f} s"
    try:
        r = httpx.post(url + "/v1/chat/completions", timeout=60, json={
            "messages": [{"role": "user", "content": "Reply with the single word: ok"}], "max_tokens": 8, "temperature": 0})
        text = r.json()["choices"][0]["message"]["content"]
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
        return f"the test question failed: {e}"
    return None if text.strip() else "the model answered with nothing"


def announce_to_app(base_url: str, token: str | None = None) -> Callable[[str, str | None], str | None]:
    """Tell the running app which model is now serving (it publishes learning.model.deployed for the Teach screen).
    Returns None on success or a sentence; never raises: the swap itself already worked."""

    def announce(version: str, previous: str | None) -> str | None:
        try:
            r = httpx.post(base_url + "/api/learning/deployed", json={"model_version": version, "previous_version": previous},
                           headers={"Authorization": f"Bearer {token}"} if token else {}, timeout=10)
            return None if r.status_code == 200 else f"the app answered HTTP {r.status_code}"
        except httpx.HTTPError as e:
            return f"the app could not be reached: {e}"

    return announce


def swap(store: ModelStore, version: str, *, restart: Callable[[], None], check: Callable[[], str | None],
         log: Callable[[str], None] = print, announce: Callable[[str, str | None], str | None] | None = None,
         trust_verified: bool = False, stage: Callable[[str, str], None] | None = None) -> None:
    """stage(name, state) is told as each step starts and ends: verify, switch, restart, check, publish, and
    rollback after a failed check; state is start, done or failed (the Learning lab's deploy button reads them)."""
    mark = stage or (lambda name, state: None)

    def step(name: str, action: Callable[[], object]) -> object:
        mark(name, "start")
        try:
            out = action()
        except BaseException:
            mark(name, "failed")
            raise
        mark(name, "done")
        return out

    log(f"checking {version} ..." + (" (trusting the pull's .verified.json where it still holds)" if trust_verified else ""))
    mark("verify", "start")
    try:  # record=True: a full hash that passes writes .verified.json, so the switch and every later trusting swap skip it
        problems = store.verify(version, trust_verified=trust_verified, record=True)
    except BaseException:
        mark("verify", "failed")
        raise
    if problems:
        mark("verify", "failed")
        raise ModelStoreError(f"{version} is not safe to switch to: " + "; ".join(problems))
    mark("verify", "done")
    old, old_previous = store.current(), store.previous()
    step("switch", lambda: store.activate(version, trust_verified=trust_verified, record=True))
    log(f"switched the link {old} -> {version}; restarting the model server (about 40 s of no answers)")
    step("restart", restart)
    mark("check", "start")
    failure = check()
    if failure is None:
        mark("check", "done")
        log(f"{version} is serving")
        if announce:
            mark("publish", "start")
            problem = announce(version, old)
            mark("publish", "done" if problem is None else "failed")
            log("told the app" if problem is None else f"could not tell the app ({problem}); the model is serving anyway")
        return
    mark("check", "failed")
    log(f"{version} failed: {failure}")
    if old:
        mark("rollback", "start")
        store.activate(old, trust_verified=trust_verified, record=True)
        store.set_previous(old_previous)  # a failed attempt must not become the version `rollback` goes to
        restart()
        again = check()
        mark("rollback", "done" if again is None else "failed")
        log(f"went back to {old}" + ("" if again is None else f", but it is not healthy either: {again}"))
    raise ModelStoreError(f"{version} failed: {failure}")


def print_stage(name: str, state: str) -> None:
    """One machine-readable line per step: `STAGE <name> <start|done|failed> <unix seconds>`."""
    print(f"STAGE {name} {state} {time.time():.3f}", flush=True)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["status", "swap", "rollback"])
    p.add_argument("version", nargs="?")
    p.add_argument("--models-dir", default="data/models")
    p.add_argument("--service", default="atlas-llm")
    p.add_argument("--url", default=f"http://127.0.0.1:{os.environ.get('ATLAS_LLM_PORT', '8080')}")
    p.add_argument("--app-url", default=os.environ.get("ATLAS_URL", "http://127.0.0.1:8000"), help="the app to tell about the new model")
    p.add_argument("--trust-verified", action="store_true",
                   help="skip the sha256 of a file the pull's .verified.json vouches for while its size and mtime hold")
    args = p.parse_args(argv)
    store = ModelStore(args.models_dir)

    if args.command == "status":
        print(f"current:  {store.current()}\nprevious: {store.previous()}")
        for m in store.versions():
            print(f"  {m.version}  base={m.base}  task={m.task}  created={m.created:%Y-%m-%d %H:%M}")
        return 0

    def restart() -> None:
        subprocess.run(["sudo", "-n", "systemctl", "restart", args.service], check=True)

    try:
        if args.command == "rollback":
            target = store.previous()
            if not target:
                raise ModelStoreError("there is no earlier version to go back to")
        else:
            if not args.version:
                p.error("swap needs a version")
            target = args.version
        swap(store, target, restart=restart, check=lambda: wait_until_serving(args.url),
             announce=announce_to_app(args.app_url, os.environ.get("ATLAS_TOKEN")), trust_verified=args.trust_verified,
             stage=print_stage)
    except ModelStoreError as e:
        print(f"not switched: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
