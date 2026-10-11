"""Writes the recorded-sample fixtures of this folder by running the pod sampler (lab/pod_power.py) over amd-smi text.

    python fixtures/telemetry/pod_power/make_samples.py          # rewrites samples_gpu-a.jsonl, samples_gpu-b_stale.jsonl

Every sample goes through the real code path: amd-smi text in the CLI's layout (amdsmi_metric_idle.txt with its
numbers replaced) -> parse_amdsmi -> PodSampler.record, with scripted times and event ids so the output is the same
on every run. tests/test_lab_pod_power.py checks that the committed files are exactly what this writes.

samples_gpu-a.jsonl: pod A at idle (the start-up energy check), a teacher load (the load check, after its settle
time), one read that failed, the return to idle, then a sampler restart about 6 s later, which shows as a new
sampler_started, reset counters and a long gap.
samples_gpu-b_stale.jsonl: pod B goes under load and its readings freeze, counter included, so the samples turn
stale after 10 s and the start-up counter check disagrees with integrated power.

Where the tool text comes from. No pod could be reached when these were made (both off, the hub answering 502),
so the three text files are not captures. amdsmi_metric_idle.txt and amdsmi_metric_load.txt were printed by
amd-smi's own human-readable printer (AMDSMILogger._convert_json_to_human_readable and custom_dump in
amdsmi_cli/amdsmi_logger.py, ROCm/rocm-systems tag therock-7.13, the pods' ROCm) from the per-GPU dict that
amdsmi_commands.py metric_gpu builds for --usage --power --clock --temperature --energy. rocmsmi_idle.txt follows
rocm_smi.py's printLog and printLogSpacer at the same tag. The numbers checked on a pod are SOCKET_POWER near 40 W
at idle and the clock ranges (GFX 500 to 2350 MHz, MEM 96 to 1258 MHz); every other number (temperatures,
activity, the energy counter, fclk and socclk, the 201 W load reading) is a stand-in. The first live run's
`bash pod_power.sh check` saves the real outputs under ~/atlas-power/raw-<time>/; add them here beside these.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

from lab.pod_power import PodSampler, Reading, parse_amdsmi  # noqa: E402

AMDSMI = SimpleNamespace(source="amd-smi")  # the reader the samples name; the text below stands in for its reads

TEMPLATE = (HERE / "amdsmi_metric_idle.txt").read_text(encoding="utf-8")


def render(watts: int, activity: int, gfx: int, mem: int, edge: int, hotspot: int, vram: int, energy: float) -> str:
    """amd-smi metric text with these readings, in the idle fixture's layout."""
    text = TEMPLATE
    for pattern, value in ((r"(SOCKET_POWER: )\d+", watts), (r"(GFX_ACTIVITY: )\d+", activity),
                           (r"(GFX_0:\n\s+CLK: )\d+", gfx), (r"(MEM_0:\n\s+CLK: )\d+", mem), (r"(EDGE: )\d+", edge),
                           (r"(HOTSPOT: )\d+", hotspot), (r"(\n\s+MEM: )\d+", vram),
                           (r"(TOTAL_ENERGY_CONSUMPTION: )[\d.]+", f"{energy:.3f}")):
        text, n = re.subn(pattern, lambda m, v=value: f"{m.group(1)}{v}", text)
        assert n == 1, pattern
    return text


def stamp(text: str) -> float:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


class Ids:
    def __init__(self, prefix: str) -> None:
        self.prefix, self.n = prefix, 0

    def __call__(self) -> str:
        self.n += 1
        return f"{self.prefix}-{self.n:03d}"


def run(sampler: PodSampler, script: list[tuple[float, str | None]]) -> list[dict]:
    """One sample per (time, amd-smi text); None is a read that failed."""
    out = []
    for t, text in script:
        if text is None:
            out.append(sampler.record(Reading(), t, error="ReadError: amd-smi gave no answer within 10 s"))
        else:
            out.append(sampler.record(parse_amdsmi(text)[0], t))
    return out


def pod_a() -> list[dict]:
    idle = [40, 40, 39, 40, 40, 40, 39, 39, 40, 40, 41, 40, 40, 40, 39, 40]
    load = [95, 150, 190, 199, 201, 202, 200, 201, 199, 200, 201, 203, 202, 200, 201, 199, 200, 202, 201, 200, 201,
            200, 199, 201, 200]
    cool = [120, 60, 45, 41, 40, 40, 40]
    t0 = stamp("2026-10-10T18:00:00.200Z")
    energy = 1843212.402
    script: list[tuple[float, str | None]] = []
    last_t, last_w = None, None
    for i, (w, busy) in enumerate([(w, False) for w in idle] + [(w, True) for w in load] + [(w, False) for w in cool]):
        t = t0 + i + (i * 37 % 7) * 0.01
        if last_t is not None:
            energy += 1.03 * (t - last_t) * (w + last_w) / 2.0  # a counter reading 3 percent above the average
        last_t, last_w = t, w
        warm = min(i, 40) // 3
        reading = render(w, 99 if busy else 0, 2350 if busy else 500, 1258 if busy else 96, 34 + warm, 37 + warm,
                         44 + warm, energy)
        if i == 5:
            script.append((t, script[-1][1]))  # the same snapshot twice: a poll, not an update
            continue
        script.append((t, None if i == 36 else reading))
    first = PodSampler(AMDSMI, "gpu-a", clock=lambda: t0 - 0.2, new_id=Ids("pod-a-1"))
    out = run(first, script)
    t1 = script[-1][0] + 6.4  # the sampler restarts; its first read comes about 6 s after the last one
    again = PodSampler(AMDSMI, "gpu-a", clock=lambda: t1 - 0.3, new_id=Ids("pod-a-2"))
    second = []
    for j in range(12):
        t = t1 + j + (j * 37 % 7) * 0.01
        energy += 1.03 * (t - last_t) * (40 + last_w) / 2.0
        last_t, last_w = t, 40
        second.append((t, render(40, 0, 500, 96, 46, 49, 56, energy)))
    return out + run(again, second)


def pod_b() -> list[dict]:
    t0 = stamp("2026-10-10T18:05:00.400Z")
    energy = 2210004.118
    script: list[tuple[float, str | None]] = []
    last_t, last_w = None, None
    for i in range(3):  # idle, counter moving
        t = t0 + i
        if last_t is not None:
            energy += (t - last_t) * 41.0
        last_t, last_w = t, 41
        script.append((t, render(41, 0, 500, 96, 36, 39, 47, energy)))
    frozen = render(245, 99, 2300, 1258, 61, 74, 70, energy + 200.0)
    for i in range(3, 25):  # under load, and every reading (counter included) stops changing
        script.append((t0 + i, frozen))
    sampler = PodSampler(AMDSMI, "gpu-b", clock=lambda: t0 - 0.4, new_id=Ids("pod-b-1"))
    return run(sampler, script)


def write(path: Path, events: list[dict]) -> None:
    path.write_text("".join(json.dumps(e, separators=(",", ":")) + "\n" for e in events), encoding="utf-8",
                    newline="\n")


def main() -> None:
    write(HERE / "samples_gpu-a.jsonl", pod_a())
    write(HERE / "samples_gpu-b_stale.jsonl", pod_b())
    print(datetime.now(timezone.utc).isoformat(timespec="seconds"), "wrote", HERE)


if __name__ == "__main__":
    main()
