# Merge log for the four refinement tracks

One entry per merge pass, in the merge-entry format set by `lab/refine/TEMPLATES.md` (section 1.4).
A merge pass runs once after every round of track passes. It reconciles what the passes of the round
did across the telemetry, learning, plumbing and kernels tracks: blockers that cross tracks (M1),
shared contract changes (M2), the demo flow (M3), duplicate tasks across tracks (M4), Tier
consistency (M5) and whether every entry of the round is valid (M6). The rules it works under are
`lab/refine/PASS_RULES.md`, section "The merge pass"; `lab/machine/refine_merge.py` states the checks
an entry here must pass, and `tests/test_lab_refine_merge.py` runs them on this file.

Each pass entry a merge walks is named `<track> pass <N>`. An entry walked `held`, or broken on any
check but M6, counts as reconciled, so the next round starts after the highest such pass of each
track. Merge numbers count the entries in this file; there is no counter file. Every pass reads the
latest entry here before it starts, and a move of a task from one track to another is recorded here
as a numbered change (the latest record wins).

---
