# T8 result for T0: attribution validation on pod A

Experiment result (T8, telemetry track, rubric TT1 and TT2). Run on 2026-10-10 from about 9:31 to 10:09 PM ET on pod A (account C, Radeon 8060S, Strix Halo gfx1151). The teacher was vLLM b1388b1 serving RedHatAI/Qwen3.8-27B-INT4 as "teacher", called through the hub proxy. The pod sampler was lab/pod_power.py 1, reading amd-smi SOCKET_POWER (the whole APU socket) every 1000 ms; the measured period was 1000.012 ms and the longest gap 1163 ms. Nothing was started, stopped or changed on the pod, and no GPU queue slot was used.

The run is 40 sequential requests, not the 100 the Done when asks for. The owner chose 40 to fit the night, because teacher time is the demo night's scarcest resource.

Method: the check confirmed that vLLM's counters showed nothing running or waiting, then measured idle for 60 s with the teacher loaded. It sent 40 text requests one at a time, each followed by a 10 s gap with nothing in flight. The requests used bench's two text prompts in turn, greedy, thinking off, 128 tokens with ignore_eos, and a nonce in front of each. vLLM's counters before and after show exactly 40 finished requests, so no other client's request ran. Every number is gross (telemetry ruling 1). Per-answer joules come from T4's attribution. The gaps and the whole run are integrated the same way: the straight line between samples, with the edges interpolated. Request times and sample times are on one clock, through a pod clock offset of 64.000 s, known within 0.048 s, from the best of 42 exchanges.

| What | Value |
|---|---|
| Idle watts (60 s, teacher loaded, 60 samples, none under load) | 10.33 W (10 to 11 W) |
| Per-request gross joules, summed over the 40 answers | 204,067.6 J (mean 5,101.7 J per answer, sd 62.9; 39.9 J per output token) |
| Gap energy (40 gaps, 411.7 s) | 8,269.4 J |
| Whole-run energy (2,242.6 s) | 212,337.0 J |
| Conservation: answers plus gaps minus the whole run | -0.0002 J (relative -9e-10), holds |
| Gap energy above idle (missed by the windows) | 4,014.7 J: 3,807.8 J in the first half of the gaps (the tail after each answer), 206.9 J in the second half (the ramp up to the next answer) |
| Share of the answers' energy the windows missed (relative error) | 1.93 percent (attribution error -1.93 percent) |
| Samples per window | median 46 (45 to 47), none under 10 |
| Verdict (Welch: each answer's joules plus its gap's excess, against its joules alone) | b_ahead: 5,202 against 5,102 J mean, t = 6.83, df = 77.5, p < 0.0001 |

Reading: at 1000 ms sampling, each answer's window holds about 98 percent of its energy. The windows miss about 100 J per answer, almost all of it in the 3 to 5 s tail after the answer ends, which matches the settle tail T10 measured. The miss is small but systematic: the verdict says it is measurable, well outside the answer-to-answer spread. So the attribution is not trustworthy as is at this sampling rate. It is usable with its error stated as -1.9 percent (the windows read low), or with each window widened by the settle time, which would recover most of the miss.

Every value stays low confidence because the sensor's own error is not measured on our node against a reference (TT1). This check measures what the windows miss of the sensor's own energy, not the sensor's accuracy.

Citation: ledger row verb attribution-check, task T8, node pod-a, written 2026-10-11T02:09:33Z (seq 1 in the T8 clone's lab/state/ledger.jsonl, copied in fixtures/telemetry/attribution_check_2026-10-11-r40_ledger.jsonl). Raw rows: fixtures/telemetry/attribution_check_2026-10-11-r40_{samples,requests,run}.jsonl. Recomputed by tests/test_lab_attribution_check.py::test_the_live_run_recomputes_to_its_ledger_row and by `python -m lab attribution-check replay --tag 2026-10-11-r40`. Code: lab/attribution_check.py, PR 270.
