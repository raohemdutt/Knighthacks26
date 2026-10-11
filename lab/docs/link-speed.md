# Link speed: `python -m lab link-speed` (R14)

Moved here from lab/README.md (O70); the verbs list there links every page. Arguments: `python -m lab link-speed -h`.

Measures how fast a few-GB package crosses the hub: a seeded random file (it does not compress) goes up to the pod
through the contents API in base64 chunks (`lab/transfer.py`), comes back through the proxy from R10's file server in
Range parts (`lab/package_pull.py`), and is checked by sha256, three repeats each way. A short probe first doubles one
single PUT until the contents API refuses it and keeps the last size it accepted.

```
python -m lab link-speed --account D [--sizes-mb 1000,3000] [--repeats 3] [--chunk-mib 64] [--part-mib 64]
                         [--port 8301] [--probe-ceiling-mib 256] [--no-probe] [--scratch r14-link-speed] [--report lab/R14_REPORT.md]
```

- It runs the file server on its own port (8301) from a copy named `r14_file_server.py` in `~/.r14_link_speed`, so a
  standing package server on 8300 is left alone, and it serves the folder the push wrote to. At the end it stops that
  server, deletes `~/.r14_link_speed`, the scratch folder and the controller's script files, so the pod is as found.
  No GPU and no server restart; it needs a ready pod with `jupyter-server-proxy`.
- One ledger row per run (`link push`, `link pull`) with `link.push_mb_s` or `link.pull_mb_s`, bytes, seconds,
  `chunk_bytes` or `part_bytes`, the node, `hub` and `refs.account`; a failed run is an `error` row with no speed. The probe
  is one row (`link part-probe`, metric `link.max_part_mb`). The Markdown table is written to `--report`.
- On SJC the edge caches `GET /user/<account>/api/contents/<path>?content=0` for four hours (`cache-control: public,
  max-age=14400`, `cf-cache-status: HIT`), so a size read back after a rewrite is the old size. This verb adds a
  never-repeated query parameter to every contents read (`FreshLink`); `lab/transfer.py` does not, and a push of a
  different size to a path it has read before can fail its size check. The proxy route answers `no-store`.
- A push's seconds cover the whole call (folders, chunks, size read back); a pull's seconds are its requests only.
  No MB/s bar applies; this measures.
- Rehearsal: `tests/test_lab_link_speed.py` runs it at 1 MB and 3 MB against `lab/proxy_fake.py` and `lab/package_fake.py`.
