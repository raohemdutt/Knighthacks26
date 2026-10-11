# Chip sheet: `lab/kernels/chipsheet/` (R33)

Moved here from lab/README.md (O70); the verbs list there links every page. Arguments: `python -m lab chipsheet -h`.

What the Strix Halo pods' chip (gfx1151) measures about itself, so every kernel note can cite a number taken on it:
device-to-device copy bandwidth from 1 MiB to 4 GiB, fp16 and bf16 matmul at the 27B decode shapes (batch 1, 4, 8,
16) and the 4B LoRA step, a plain Triton int4-to-fp16 dequantize-plus-matmul kernel, the LDS size and the largest
workgroup LDS allocation that launches, the wavefront size the compiler chose for a probe kernel built for wave32 and
for wave64, an occupancy curve (achieved pointer-chase throughput against live VGPRs and LDS per workgroup, beside
the compiler's VGPR count and occupancy remark), the cache sizes rocminfo and the HIP runtime report, the amd-smi
watts (idle before each section, loaded during each measurement) and the ROCm, HIP, torch and triton versions. Each
number carries its exact command, its repeats and their spread. A full run stays inside 840 s of GPU time and
`--quick` inside 110 s; a measurement that would start past the budget is written as skipped.

| File | What it is |
|---|---|
| `chipsheet.py` | the pod runner (`run --node pod-b [--quick]`) and the no-GPU `self-check` |
| `chipsheet_format.py` | the sheet's JSON format, its Markdown summary and the parsers for rocminfo, hipcc, amd-smi, the probe and the compiler's resource remarks; standard library only |
| `probe.hip` | the HIP probes: device properties, wave size, LDS launch limit, occupancy kernels |
| `pod_chipsheet.sh` | what `run-script` runs on the pod: copies the uploaded files to `~/chipsheet` and runs the sheet there |
| `ingest.py` | `python -m lab chipsheet ingest`: files a pulled sheet under `lab/refine/research/kernels/data/` and writes one ledger row per section on the sheet's node |

On pod D, inside one GPU Queue slot (the pod must be up and the slot started first):

```
python -m lab.controller run-script --account D --timeout 1200 --with lab/kernels/chipsheet/chipsheet.py --with lab/kernels/chipsheet/chipsheet_format.py --with lab/kernels/chipsheet/probe.hip --with lab/pod_power.py lab/kernels/chipsheet/pod_chipsheet.sh run --node pod-b
python -m lab.transfer pull-file --account D chipsheet/chipsheet-gfx1151-<date>.json data/chipsheet-gfx1151-<date>.json
python -m lab chipsheet ingest --sheet data/chipsheet-gfx1151-<date>.json
```

The last line writes `lab/refine/research/kernels/data/chipsheet-gfx1151-<date>.json` and `.md` (commit both) and the
`chipsheet` ledger rows on node pod-b, metrics named `chip.<entry id>`. Add `--quick` after `--node pod-b` for the
two-minute version. The runner refuses (exit 2) unless rocminfo names gfx1151.

On a machine with no GPU, `python -m lab chipsheet self-check` (or `python3 lab/kernels/chipsheet/chipsheet.py
self-check`) parses the recorded tool output in `fixtures/lab/chipsheet/`, writes a sheet marked self-check to a
temporary folder, reads it back and exits 0; it measures nothing and `ingest` refuses its sheet.
