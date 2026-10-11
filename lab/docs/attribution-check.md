# Attribution check: `python -m lab attribution-check` (T8)

Moved here from lab/README.md (O70); the verbs list there links every page. Arguments: `python -m lab attribution-check -h`.

Do the answers' windows hold the energy the teacher's pod spends on them? From the box, against the running teacher
on pod A and the pod sampler already running there (nothing is started, stopped or changed on the pod):

```
python -m lab attribution-check run --account C [--requests 100] [--idle-s 60] [--gap-s 10] [--wait-min 30]
python -m lab attribution-check replay --tag <date> [--record]      # the numbers again from the saved rows
```

It checks that vLLM's `/metrics` shows nothing in flight (and waits, two minutes at a time, up to `--wait-min`),
reads the idle watts for 60 s, then sends bench's text prompts one at a time (greedy, thinking off, 128 tokens,
ignore_eos), each followed by a 10 s gap with nothing in flight. Every number is gross (telemetry ruling 1): each
answer's joules come from T4's attribution (`contexts/telemetry/application/joules.py`) fed the pod's samples, the
windows and the pod's clock offset as the box feeds it; the gaps and the whole run are integrated the same way, and
answers plus gaps must equal the whole run (conservation). The gap energy above the idle watts times the gap time is
what the windows missed (the sensor's averaging lag and the tail after each answer); its share of the answers' energy
is the relative error. The verdict (`lab/machine/verdict.py`) reads each answer's joules plus the gap after it above
idle against its joules alone: no_difference, with at least 10 samples in every window, conservation, a quiet idle
window and no other client's request during the run (vLLM's counters before and after), calls the attribution
trustworthy at the sampler's rate. The raw rows go to `fixtures/telemetry/attribution_check_<tag>_{samples,requests,run}.jsonl`
and one ledger row (verb `attribution-check`, task T8, node pod-a, `low_confidence` while the sensor's own error is
unmeasured) holds the result. Tests: `tests/test_lab_attribution_check.py` (a synthetic trace through the real pod
sampler and a fake teacher, worked out by hand).
