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
Answer: Yes, the walk list is every kernels task in my lane, R11, R12, R20 to R27, under the same rules. (owner, 2026-10-10)

## 2. Is E2 (vLLM against llama.cpp) a lever ranked by expected gain, or a reference measurement kept out of the ranking? (pass 1)

Now: E2 stays unscored in the ledger and on T0 (no confidence), so it ranks after every scored line.
Recommend: a reference measurement, run once beside the stock matrix and again after E13, outside the gain ranking: the Design Doc keeps llama.cpp ROCm as an emergency fallback, so its gain is not one the teacher would take, and its arms differ in engine and weight format at once.
Source of tension: lab/refine/kernels/RUBRIC.md K8 (kernels experiments ordered by expected joules-per-token gain times confidence) and K2 (one lever) against Atlas Design Doc section 10 (llama.cpp ROCm only as an emergency fallback) and T0's E2 line.
Answer: Reference measurement, kept out of the ranking; llama.cpp is the fallback engine, not a lever. (owner, 2026-10-10)

## 3. Should R11 (the benchmark harness) become P0? (pass 1)

Now: P1; a pass changes no Priority.
Recommend: P0, because the P0 tasks R20 and R21, and through them R24 and D8, are blocked by it, so the demo's lab numbers cannot exist until it is built, and a P0-first pick order would leave it waiting.
Source of tension: lab/refine/PASS_RULES.md, "What a pass may never change" (a Priority change takes a filled ruling) against the Task Board's Blocked by links (R20 and R21, P0, blocked by R11, P1) and AGENTS.md section 5 (P0 first).
Answer: Yes, R11 is P0 (done on the board Sat 3:30 PM ET), because R20 and R21 are blocked by it. (owner, 2026-10-10)

## 4. Where does the experiment queue live from now on? (pass 1)

Now: T0 is the record; each pass ran the experiment verb in a scratch state folder, so its rank sees only that pass's lines, and the next E number is read from T0 (one collision so far, E7, renumbered by the plumbing pass).
Recommend: seed the real state folder once with python -m lab experiment seed and add E6 to E15 from T0 in one ordinary task, so the verb hands out numbers under its lock and ranks the whole queue; until then passes keep reading T0 for the next number.
Source of tension: lab/machine/experiments.py (an experiment lives in the ledger and T0 mirrors it through --print) against the O27 page ("Nobody has run python -m lab experiment seed on the real state folder yet; Ilan's call") and the scratch state folder every pass of this round was told to use.
Answer: The real lab ledger once it is seeded; T0 stays the human-readable mirror. The seed command waits for my go. (owner, 2026-10-10)

## 5. Which batch sizes make the stock record? (pass 1)

Now: both sets: E11 runs 1, 4, 8 and 16 and adds 64 when it fits, as the rubric's known open items ask until a ruling picks one.
Recommend: 1, 4, 8 and 16 as the record, with 64 an extra arm only when the KV cache and the float32 GDN state fit at the workload's context, since 64 sequences need about 9.7 GB of GDN state (about 151 MB each) on top of the KV cache.
Source of tension: lab/refine/kernels/RUBRIC.md K1 (batch sizes 1, 4, 8 and 16) and its Known open items for pass 1 (K1) against T0's E3 line (batch 1, 4, 16 and 64).
Answer: 1, 4, 8 and 16, plus 64 when it fits in memory. (owner, 2026-10-10)

## 6. If E18 shows GPU_MAX_HW_QUEUES=1 pins the fast band at no measured cost, does the teacher adopt it and the stock record restart with it set? (pass 2)

Now: the teacher and the stock record run with the variable unset; E18's result changes only its own rows, since this track never restarts or patches the teacher on pod A, and kernels rubric K1 (c) does not list an environment variable among the changes that start a new stock record.
Recommend: yes. Once E18 reads b_ahead with outputs unchanged, an ordinary task sets the variable in the serve step of lab/pod_a_teacher.sh and R20's stock matrix is measured with it set and recorded, because a teacher in the slow band labels at about half the tokens per joule and every reported measurement of the variable found no cost; if E18 shows one band, nothing changes.
Source of tension: lab/refine/kernels/RUBRIC.md, "What a pass may NOT change" (nothing on this track restarts, rebuilds or patches the teacher on pod A) and K1 (c), against E18's expected +70% decode at batch 1 and lab/pod_a_teacher.sh's serve step, which sets no such variable.
Answer: Yes. Once E18 reads b_ahead with outputs unchanged, an ordinary task sets GPU_MAX_HW_QUEUES=1 in the serve step of lab/pod_a_teacher.sh and R20's stock matrix is measured with it set and recorded; if E18 shows one band, nothing changes. (owner, 2026-10-10)

## 7. Which task builds the stock vLLM and the checkpoint on pod B (SJC account D)? (pass 2)

Now: none. R29 builds vLLM b1388b1 with lab/vllm-gfx1201.diff and serves the checkpoint on account C only, while E18, R20, R12 and every kernels line on T0 run on pod B, whose fresh home volume holds neither; plumbing pass 2's E17 line names the same gap.
Recommend: a small task, or a second clause on R29, that runs the same lab/pod_a_teacher.sh build and checkpoint download on account D and records the same hour-0 facts and the hunks that apply, so E18 can run first; it is a build, not a measurement, so its rows are none.
Source of tension: R29's Done when (account C only) and R20's Done when (pod B), against lab/refine/PASS_RULES.md, "What a pass may change" (a pass edits the properties of its own track's tasks, splits and merges them; R29 is on no walk list and no pass creates a free-standing task).
Answer: R30, Hemdutt's, already built vLLM and downloaded the checkpoint on account D (a full restart takes 523 s); R30's page is the hour-0 record and the hunks that apply are R29's nine. No new task. (owner, 2026-10-10)

## 8. How does the kernels rubric change now that the lab is the two SJC Strix Halo pods (gfx1151, RDNA 3.5), when it was written for the R9700 (gfx1201, RDNA 4)? (merge 2)

Now: The rubric stands as written for gfx1201: its goal and K1 name the R9700's serving path and stock record (vLLM b1388b1 plus lab/vllm-gfx1201.diff on an R9700), K5 states the 64 KB LDS limit and the unified-attention overflow of gfx1201, the baseline to beat is the R9700's 7.7 tok/s and about 26 J per token, and the ceiling arithmetic uses the R9700's 640 GB/s; passes read pod B as account D on the SJC Strix Halo pods by Ilan's decision, mark the R9700 readings old, other chip, and edit no rubric; R29 measured the same build on pod C at 2.88 tok/s on text, 2.82 tok/s on an image, 120 W average and 9.2 W idle with the model loaded (the twelve pod-c rows of lab/refine/metrics.jsonl, PR #194), which no rubric clause names yet.
Recommend: Rewrite the gfx1201 clauses for gfx1151 and keep the R9700 record as a dated note: K1's stock record becomes the gfx1151 build R29 settled (vLLM b1388b1 plus lab/vllm-gfx1201.diff, all nine hunks unchanged, with the TRITON_ATTN vision-encoder backend), measured by R20 on pod B with repeats and bands; K5 states the LDS limit of RDNA 3.5 from the ISA guide and the LLVM target for gfx1151 instead of gfx1201's; "The baseline to beat" becomes R29's rows once R20 repeats them, with the R9700 numbers kept beside them as another chip; the ceiling arithmetic uses the Radeon 8060S's memory bandwidth and its shared-memory layout; and the RDNA4-only levers (R12's 64 KB LDS attention tuning, E19's RDNA hybrid W4A16 kernel, R24's gfx1201 Triton sweep) are rewritten for RDNA 3.5 or dropped by the kernels loop (R28), as Ilan's decision says. Until the answer, the kernels pass reads every gfx1201 clause as gfx1151 and cites this ruling.
Source of tension: lab/refine/kernels/RUBRIC.md ("The goal", K1, K5, "The baseline to beat", W7, W10 and "What a pass may NOT change": only the merge pass edits it, and only with a filled Answer in this file) against Ilan's decision of 2026-10-10 on the Ilan1 and Ilan2 dispatcher page (the lab is the two SJC Strix Halo pods; everything that says R9700 or gfx1201 is retargeted as it is touched; the R9700 numbers stay as another chip), R29's task page and the twelve pod-c rows of lab/refine/metrics.jsonl (PR #194, 498e161), and lab/refine/plumbing/REFINE_LOG.md, "## Pass 2, 2026-10-10", "Left for the next pass" (the kernels loop rewrites or drops the R9700-only work).
Answer: Rewrite the gfx1201 clauses for gfx1151 as recommended, with the R9700 record kept as a dated note beside the new baseline; until the merge pass edits the file, every gfx1201 clause reads as gfx1151 and cites this ruling. R12's LDS tuning is rewritten for RDNA 3.5 (its cloud build does that tonight); E19 and R24 are dropped unless the kernels loop revives them. (owner, 2026-10-10)
