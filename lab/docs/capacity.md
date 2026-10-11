# Chip capacity and the student ladder: `python -m lab capacity`, `lab/ladder.py` (R44)

Moved here from lab/README.md (O70); the verbs list there links every page. Arguments: `python -m lab capacity -h`.

`python -m lab capacity` probes one chip in under two minutes through a chip client (R33's quick chip sheet and the
pod's facts) and writes a capacity record: free memory, best copy bandwidth, best bf16 matmul rate, chip name, ROCm
version, node, role and the probe's wall-clock. `lab/ladder.py` holds the committed student ladder (Qwen3.5-4B
measured, 9B and 27B predicted from it) and picks the largest student whose LoRA fits the records, with its
predicted run time and a one-sentence reason; with two chips it uses the placement plan to run one LoRA job per chip
at once. What a friend's chip receives and what the ladder assumes: `lab/CIRCLE_CHIPS.md`. The pod commands, the
one-chip against two-chip evidence and the data-parallel kill criterion: `lab/R44_REAL_RUN.md`.

```
python -m lab capacity --fake --node pod-b                                   # the R33 sample, no chip
python -m lab capacity --sheet <quick sheet>.json --facts pod-b.facts.txt --node pod-b
python -m lab.ladder --capacity <record>.json [--capacity <record>.json] --samples 2000 --epochs 2 [--teacher-share 0]
```

Tests: `tests/test_lab_capacity.py`, `tests/test_lab_ladder.py` (the fake client only; no pod, hub or GPU).
