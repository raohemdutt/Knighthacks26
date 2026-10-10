# Owner rulings for the kernels track

Questions the passes could not decide, because they are policy, not method. Each shows what the
plan does NOW (the default the pass applied so the loop could continue), the recommendation, the
source of the tension, and a blank for the owner's answer. A filled Answer is an owner rule at top
authority: the next pass applies it and cites it in `REFINE_LOG.md`. Passes never edit this file;
the merge pass adds questions from the pass entries' "Rulings raised" sections. The format is
`lab/refine/TEMPLATES.md`, section "The rulings file".

## 1. Do R20 to R26 join the kernels track's walk list? (pass 1)

Now: they are walked for findings only and never edited, so R20 and R21 (both P0) keep three repeats, compiled-graph reference outputs and the same Wave as R11, R23 has no Blocked by, and R24 puts the Conch config where a vLLM diff cannot carry it.
Recommend: yes for R20, R21, R22, R23, R24, R25 and R26, which the track run split created for this track and whose Done whens cite the kernels rubric and research notes; R27 (shared with Xander, on the Jetson) stays out. The merge pass carries it into the rubric's walk list, and pass 2 then fixes those rows.
Source of tension: lab/refine/kernels/RUBRIC.md, "The walk list" (W1 to W6 replace the starting set of PASS_RULES.md, "The four tracks": R11, R12 and T0's lines) against the Task Board rows R20 to R26, created on Sat 7:00 AM ET from the track run split, whose Done whens cite "kernels rubric test 1" (R20) and "kernels rubric test 4" (R21).
Answer: ____

## 2. Is E2 (vLLM against llama.cpp) a lever ranked by expected gain, or a reference measurement kept out of the ranking? (pass 1)

Now: E2 stays unscored in the ledger and on T0 (no confidence), so it ranks after every scored line.
Recommend: a reference measurement, run once beside the stock matrix and again after E13, outside the gain ranking: the Design Doc keeps llama.cpp ROCm as an emergency fallback, so its gain is not one the teacher would take, and its arms differ in engine and weight format at once.
Source of tension: lab/refine/kernels/RUBRIC.md K8 (kernels experiments ordered by expected joules-per-token gain times confidence) and K2 (one lever) against Atlas Design Doc section 10 (llama.cpp ROCm only as an emergency fallback) and T0's E2 line.
Answer: ____

## 3. Should R11 (the benchmark harness) become P0? (pass 1)

Now: P1; a pass changes no Priority.
Recommend: P0, because the P0 tasks R20 and R21, and through them R24 and D8, are blocked by it, so the demo's lab numbers cannot exist until it is built, and a P0-first pick order would leave it waiting.
Source of tension: lab/refine/PASS_RULES.md, "What a pass may never change" (a Priority change takes a filled ruling) against the Task Board's Blocked by links (R20 and R21, P0, blocked by R11, P1) and AGENTS.md section 5 (P0 first).
Answer: ____

## 4. Where does the experiment queue live from now on? (pass 1)

Now: T0 is the record; each pass ran the experiment verb in a scratch state folder, so its rank sees only that pass's lines, and the next E number is read from T0 (one collision so far, E7, renumbered by the plumbing pass).
Recommend: seed the real state folder once with python -m lab experiment seed and add E6 to E15 from T0 in one ordinary task, so the verb hands out numbers under its lock and ranks the whole queue; until then passes keep reading T0 for the next number.
Source of tension: lab/machine/experiments.py (an experiment lives in the ledger and T0 mirrors it through --print) against the O27 page ("Nobody has run python -m lab experiment seed on the real state folder yet; Ilan's call") and the scratch state folder every pass of this round was told to use.
Answer: ____

## 5. Which batch sizes make the stock record? (pass 1)

Now: both sets: E11 runs 1, 4, 8 and 16 and adds 64 when it fits, as the rubric's known open items ask until a ruling picks one.
Recommend: 1, 4, 8 and 16 as the record, with 64 an extra arm only when the KV cache and the float32 GDN state fit at the workload's context, since 64 sequences need about 9.7 GB of GDN state (about 151 MB each) on top of the KV cache.
Source of tension: lab/refine/kernels/RUBRIC.md K1 (batch sizes 1, 4, 8 and 16) and its Known open items for pass 1 (K1) against T0's E3 line (batch 1, 4, 16 and 64).
Answer: ____
