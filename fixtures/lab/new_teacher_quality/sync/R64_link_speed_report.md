# R14: push and pull speed of a 1GB package on pod-a

Measured 2026-10-11 06:20 UTC from a windows-amd64 machine against hub sjc.aupcloud.io, account C, node pod-a. Push goes through the Jupyter contents API in base64 chunks; pull comes back through the hub proxy (port 8301) from R10's Range-capable file server in parts, and every pulled copy is checked against the generated file's sha256. Speeds are MB/s of 10^6 bytes, mean over the runs that worked, with the sample standard deviation and the range. A push's seconds cover the whole call (folders, chunks, size read back); a pull's cover its requests only. No MB/s bar applies.

| Package | Direction | Chunk or part | Runs ok | MB/s mean | stdev | min | max | sha256 pass (push: size) |
|---|---|---|---|---|---|---|---|---|
| 1GB | push | 64 MiB | 3 of 3 | 9.24 | 0.14 | 9.08 | 9.35 | size 3 of 3 |
| 1GB | pull | 64 MiB | 3 of 3 | 47.47 | 3.56 | 45.07 | 51.56 | 3 of 3 |

The largest file the contents API accepted whole (in 64 MiB chunks, size read back on the pod) was 1GB (1000000000 bytes).

## Every run

| Package | Direction | Repeat | Bytes | Seconds | MB/s | Result |
|---|---|---|---|---|---|---|
| 1GB | push | 1 | 1000000000 | 107.0 | 9.35 | ok |
| 1GB | pull | 1 | 1000000000 | 22.2 | 45.07 | ok |
| 1GB | push | 2 | 1000000000 | 107.6 | 9.29 | ok |
| 1GB | pull | 2 | 1000000000 | 19.4 | 51.56 | ok |
| 1GB | push | 3 | 1000000000 | 110.1 | 9.08 | ok |
| 1GB | pull | 3 | 1000000000 | 21.8 | 45.77 | ok |

The scratch folder, the file server and the script files were removed from the pod after the run.

Run id r14-20261011-062019-fabc; one ledger row per run (`link push`, `link pull`) and one for the probe (`link part-probe`).
