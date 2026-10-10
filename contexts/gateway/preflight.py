"""Demo pre-flight (D9): the Demo Plan's T-15 checklist as checks the box runs on itself.

    GET  /api/preflight         every check: {name, ok, detail, fix}
    POST /api/preflight/reset   clear the stage: alert cooldowns, odd-hours flags, pending approval cards

The operator runs `python -m contexts.demo preflight --gateway http://<jetson>:8000`, which adds what
only a client can see (a warm-up question and its time, camera rates) and prints PASS/FAIL with the fix.
Nothing here touches memory or the event log's history.
"""

import shutil
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

import httpx

from contexts.agent.application.loop import ODD_HOURS_RULE
from contexts.voice.proxy import voice_check
from contracts.experience import ApprovalDecided

PACKAGE_RULE = "package-at-door"
MIN_DISK_GB = 2.0
MIN_MEMORY_MB = 500


def check(name: str, ok: bool, detail: str, fix: str = "") -> dict:
    return {"name": name, "ok": ok, "detail": detail, "fix": "" if ok else fix}


def run_checks(atlas, now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(UTC)
    s = atlas.settings
    out = []

    host = urlparse(s.public_url).hostname
    out.append(check("phone link", host not in ("127.0.0.1", "localhost", "::1"), f"ATLAS_PUBLIC_URL={s.public_url}",
                     "set ATLAS_PUBLIC_URL to the Jetson's LAN address (hostname -I), or Approve buttons on phones fail"))

    out.append(check("on-device model", bool(s.llm_url), f"llm_url={s.llm_url or 'none (fake model)'}",
                     "set ATLAS_LLM_URL=http://127.0.0.1:8080 and check `systemctl status atlas-llm`"))

    if s.ntfy_url:
        try:
            ok = httpx.get(s.ntfy_url.rstrip("/") + "/v1/health", timeout=3).status_code < 500
        except httpx.HTTPError:
            ok = False
        out.append(check("phone alerts (ntfy)", ok, s.ntfy_url, "start ntfy, and check NTFY_URL"))
    else:
        out.append(check("phone alerts (ntfy)", False, "NTFY_URL not set", "set NTFY_URL so alerts buzz the phone"))

    rules = atlas.rules
    if rules is not None:
        package = next((r for r in rules.rules if r.id == PACKAGE_RULE), None)
        on = package is not None and rules.enabled(PACKAGE_RULE)
        out.append(check("package alert on", on, f"{PACKAGE_RULE}: {'on' if on else 'off or missing'}",
                         f"POST /api/rules/{PACKAGE_RULE}/enabled {{\"enabled\": true}}"))
        last = rules._last_fired.get(PACKAGE_RULE)
        cool = package is None or last is None or (now - last).total_seconds() >= package.cooldown_s
        out.append(check("package alert ready", cool, "cooldown clear" if cool else f"fired at {last:%H:%M:%S}, still cooling down",
                         "run preflight --reset, or wait out the cooldown"))
        sleeping = _in_sleep(atlas, now)
        odd_on = rules.enabled(ODD_HOURS_RULE)
        out.append(check("no sleep-hours alerts on stage", not (sleeping and odd_on),
                         f"sleep hours now: {sleeping}; odd-hours rule {'on' if odd_on else 'off'}",
                         f"POST /api/rules/{ODD_HOURS_RULE}/enabled {{\"enabled\": false}}"))

    pending = atlas.approvals.pending()
    out.append(check("no stale approval cards", not pending, f"{len(pending)} pending",
                     "run preflight --reset (denies them), or answer them on the phone"))

    if atlas.policy is not None:
        offline = atlas.policy.state.offline
        out.append(check("online (learning available)", not offline, f"offline={offline}",
                         "switch offline off in Privacy, or `make online`"))

    free_gb = shutil.disk_usage(s.data_dir if Path(s.data_dir).exists() else ".").free / 1e9
    out.append(check("disk", free_gb >= MIN_DISK_GB, f"{free_gb:.1f} GB free", "free space on the SD card (old frames, models)"))

    mem = _available_mb()
    if mem is not None:
        out.append(check("memory", mem >= MIN_MEMORY_MB, f"{mem} MB available",
                         "stop anything else running on the Jetson; the detector needs about 1 GB"))

    out.append(voice_check())

    if atlas.capture is not None:
        for cam, st in atlas.capture.stats().items():
            out.append(check(f"camera {cam}", st.get("status") == "ok",
                             f"{st.get('status')} at {st.get('fps_actual', 0):.1f} fps", "check the USB cable and `ls /dev/video*`"))
    else:
        out.append(check("cameras", False, "no capture service on this box", "run on the Jetson with config/cameras.yaml"))
    return out


def reset(atlas) -> dict:
    """Clear the stage without touching memory: cooldowns and odd-hours flags, and deny leftover cards."""
    cleared = 0
    if atlas.rules is not None:
        cleared = len(atlas.rules._last_fired)
        atlas.rules._last_fired.clear()
    atlas.loop._flagged.clear()
    denied = [ApprovalDecided(action_id=r.id, decision="deny") for r in atlas.approvals.pending()]
    return {"cooldowns_cleared": cleared, "cards_denied": len(denied), "events": denied}


def _in_sleep(atlas, now: datetime) -> bool:
    start, end = atlas.qa.sleep.last_night(now)
    return start <= now < end


def _available_mb() -> int | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    for line in meminfo.read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return None
