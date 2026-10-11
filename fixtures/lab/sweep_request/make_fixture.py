"""Records recorded_full.out: the request-side sweep (lab/sweep_request.py) of the committed grid, run on this machine
against lab/bench_fake.py, saved as the pod run prints it (the headline lines, then the marked result block).

    python fixtures/lab/sweep_request/make_fixture.py

The run is a real one of the harness: the fake server is spawned as a process, every cell is measured by
lab/bench.py's Session.measure over a real token stream, and the conditions are read from a fake Strix Halo pod
(lab.bench_fake.FakeHost, FakePower). Only the GPU is pretend, so the rates and watts are the fake's, never a
measurement. It runs the committed grid unchanged (so the result names its sha256) with port 0 and 0.05 s idle
readings in place of the grid's server values. Paths of this machine are replaced before writing (the temporary folder
by /fake, the interpreter by python3, the repo by /repo), and the script refuses to write a file that still names one.
Run it again whenever the grid changes: tests/test_lab_sweep_request.py checks the recorded grid's sha256.
"""
from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from lab import bench_fake  # noqa: E402
from lab import sweep_request as sr  # noqa: E402

OUT = HERE / "recorded_full.out"
LOCAL = re.compile(r"(?<![A-Za-z])[A-Za-z]:\\|[A-Za-z]:/(Users|Program|Python)|\\Users\\|/Users/|hackathon|HACKAT",
                   re.IGNORECASE)


def scrub(value: Any, swaps: list[tuple[str, str]]) -> Any:
    if isinstance(value, str):
        for old, new in swaps:
            value = value.replace(old, new)
        return value.replace("\\", "/") if LOCAL.search(value) is None and ("/fake" in value or "/repo" in value) else value
    if isinstance(value, list):
        return [scrub(v, swaps) for v in value]
    if isinstance(value, dict):
        return {k: scrub(v, swaps) for k, v in value.items()}
    return value


def main() -> int:
    grid = sr.load_grid()
    with tempfile.TemporaryDirectory() as folder:
        tmp = Path(folder)
        root = tmp / "pod"
        root.mkdir()
        host = sr.read_once(bench_fake.FakeHost(root))
        power = bench_fake.FakePower(root)
        launch = sr.Launch("fake", tuple(bench_fake.serve_command(root=str(root), set_env=["GPU_MAX_HW_QUEUES=4"])),
                           {}, ())
        raw = sr.run_sweep(grid, grid.cells(), [launch], host=host, power=power, out_dir=tmp / "runs",
                           checkpoint="/home/jovyan/models/Qwen3.8-27B-INT4", engine_dir="/home/jovyan/vllm",
                           patches=[str(REPO / grid.server["patch"])], port=0, idle_s=0.05,
                           hub="https://sjc.aupcloud.io", account="D")
        files = sr.result_files(raw)
    swaps = []
    for path in (tmp, Path(folder).resolve(), REPO):
        for form in (str(path), str(path).replace("\\", "/"), str(path).replace("\\", "\\\\")):
            swaps.append((form, "/repo" if path == REPO else "/fake"))
    for exe in (bench_fake.spawn_python(), sys.executable):
        swaps += [(exe, "python3"), (exe.replace("\\", "/"), "python3")]
    swaps.sort(key=lambda pair: -len(pair[0]))
    clean = scrub(raw, swaps)
    files["raw.json"] = json.dumps(clean, separators=(",", ":")).encode("utf-8")
    for name in [n for n in files if n != "raw.json"]:
        files[name] = scrub(files[name].decode("utf-8"), swaps).encode("utf-8")
    text = "\n".join(sr.headline_lines(sr.summarize(clean))) + "\n" + sr.bundle_text(sr.make_bundle(files))
    leaks = {name: LOCAL.search(blob.decode("utf-8", "replace")) for name, blob in files.items()}
    leaks = {name: blob.decode("utf-8", "replace")[max(0, m.start() - 60):m.end() + 60]
             for name, blob in files.items() if (m := leaks[name])}
    if leaks:
        print(json.dumps({"ok": False, "error": "a local path is still in the output", "where": leaks}))
        return 1
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps({"ok": True, "written": str(OUT.relative_to(REPO).as_posix()), "rows": len(clean["rows"]),
                      "error": clean["error"], "bytes": len(text)}))
    return 0 if clean["error"] is None else 1


if __name__ == "__main__":
    sys.exit(main())
