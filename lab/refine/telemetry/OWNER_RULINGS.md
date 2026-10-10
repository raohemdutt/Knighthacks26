# Owner rulings for the telemetry track

Questions the passes could not decide, because they are policy, not method. Each shows what the
plan does NOW (the default the pass applied so the loop could continue), the recommendation, the
source of the tension, and a blank for the owner's answer. A filled Answer is an owner rule at top
authority: the next pass applies it and cites it in `REFINE_LOG.md`. Passes never edit this file;
the merge pass adds questions from the pass entries' "Rulings raised" sections. The format is
`lab/refine/TEMPLATES.md`, section "The rulings file".

## 1. Should routing and the slide use gross or marginal joules per answer? (pass 1)

Now: gross, the Telemetry context's definition and the rubric's stated default in TT2, with the node's idle watts and their timestamp stored beside every value (change 1), so marginal can be derived later without re-measuring.
Recommend: gross as the headline and on the slide, with marginal stored beside it under a name that says marginal; pod idle (39.0 W, metrics.jsonl:17) is about a fifth of the teacher's 200 W (metrics.jsonl:7), so a marginal figure would make a pod that stays up anyway look far cheaper than the energy it draws, and T8 should then compare gross with gross.
Source of tension: Context: Telemetry, "Ubiquitous language" (gross by definition) and attribution-and-baselines.md#2-the-idle-baseline, AT-05 and AT-07 (the resident floor as a term inside the total; MLPerf reports gross), against T8's Done when (whole-run energy "minus the idle baseline") and T10's (one idle-baseline policy from a pod B run), with RUBRIC.md TT2 leaving the router's choice open (Known open items 7).
Answer: ____

## 2. Do T6 to T11 belong in the telemetry track's walk list? (pass 1)

Now: no. RUBRIC.md, "The walk list", names T0 to T5 and replaces PASS_RULES.md's starting set, so T6 to T11 are walked for findings only and no pass changes them.
Recommend: yes, add T6 to T11. They produce the track's first measured values (T6 and T8 are P0) and its closed-loop test, and the findings on them (T6's rows name no sampler, period or error; T9 and T10 write into the rubric; T11 asks for 100 golden questions that cannot leave the box) have nobody to act on them otherwise.
Source of tension: PASS_RULES.md, "The four tracks" (the telemetry track covers "the T tasks"), against RUBRIC.md, "The walk list" (T0 to T5, which replaces that starting set).
Answer: ____

## 3. Which node names do telemetry events and metrics rows use: gpu-a and gpu-b, or pod-a and pod-b? (pass 1)

Now: both, with a mapping: events follow Context: Telemetry (jetson, gpu-a, gpu-b), as T1's open PR #112 does, and metrics rows keep pod-a and pod-b, as all 22 rows do; this pass wrote no node name onto the board.
Recommend: jetson, pod-a and pod-b everywhere, with the spec's language line changed to match, because every stored row, lab/machine/metrics.py, lab/README.md and the GPU Queue already say pod A and pod B, and T1 has not merged, so its docstring costs one edit now and every later comparison needs no mapping.
Source of tension: Context: Telemetry, "Ubiquitous language" (Node: jetson, gpu-a, gpu-b) and PR #112's contracts/telemetry.py, against lab/refine/metrics.jsonl lines 1 to 22 (pod-a, pod-b) and T3's scope note from Ilan's split, which makes the pod account the node, a third name.
Answer: ____

## 4. Which task builds the first routing policy that reads a telemetry value? (pass 1)

Now: none. T5's first policy routes by capability, as Context: Inference Runtime's v1 says, and T11 tests "the energy-aware policy" that no task defines, so TT6 cannot be met yet.
Recommend: give T5 a second policy behind a switch, off by default, that sends Learning or Circle work to the teacher only when the snapshot's latest joules per answer for that work favour it, so T11 compares it with v1 on that work rather than on the golden questions, which stay on the box.
Source of tension: Context: Inference Runtime, "Routing hook" (v1 by capability; later versions weigh joules per answer), against RUBRIC.md TT6 (a decision that reads the numbers, with its effect measured) and T11's Done when.
Answer: ____

## 5. Should T2, T3 and T4 become P0, since the P0 tasks T6, T8, R20 and D8 are blocked by them? (merge 1)

Now: T2, T3 and T4 stay P1; T6 (P0) is Blocked by T3, T8 (P0) by T4, D8 (P0, the slide's lab numbers) by T8, and R20 (P0) by T3, so under the P0-first pick order the P0 measurements wait on P1 builds.
Recommend: P0 for T2, T3 and T4. The slide's watts per node and joules per answer and per token (D8) cannot exist without the two samplers and the attribution, and R11, for which kernels ruling 3 asks the same, is blocked by T3 too. Keep T5 and T7 at P1.
Source of tension: The Task Board's Priority values (T2, T3, T4 at P1; T6, T8, R20, D8 at P0) and their Blocked by links, against AGENTS.md section 5 (P0 first) and lab/refine/PASS_RULES.md, "What a pass may never change" (a Priority change takes a filled ruling, which the merge pass applies).
Answer: ____

## 6. What is the per-pass change cap? (merge 1)

Now: 5. lab/refine/TEMPLATES.md section 1.1 says the cap is set in PASS_RULES.md, PASS_RULES.md names none, and all four pass 1 entries used 5 and wrote "of at most 5".
Recommend: 5, written into PASS_RULES.md by an ordinary task (the file changes only through a ruling and a board task): five changes with BEFORE, AFTER, SOURCE and WHY are what one Opus sitting verified on the board in one pass today, and a higher cap leaves more changes unwalked before the next pass.
Source of tension: lab/refine/TEMPLATES.md, section 1.1 ("Changes (<n> of at most <cap>): ... The cap is set in PASS_RULES.md") against lab/refine/PASS_RULES.md, which sets no cap and says no pass edits it; raised by telemetry pass 1 under "Left for the next pass".
Answer: ____

## 7. Does the Jetson get an external meter on its DC input, so its attribution error can be measured? (merge 1)

Now: No meter; every Jetson energy number keeps the low-confidence flag, because TT1 lets only an error measured on our own node against a reference lift it, and published calibrations found Jetson internal sensors under-reading by up to 50 percent.
Recommend: No purchase this weekend: the Jetson's numbers stay flagged, the slide and the stats page show the flag (lab/slide_numbers.py prints it since PR 132) with the VDD_IN rail named as the source, and the meter is revisited after the hackathon if the energy thread continues.
Source of tension: lab/refine/telemetry/RUBRIC.md, TT1 (only a bound measured on our own node against a reference lifts the flag; Shalavi et al. 2023, up to 50 percent under-reading) and "Known open items for pass 1", item 5 (a meter is hardware and money, a question for Ilan), against the open-ticket skill, section 7 (no new paid services or purchases without Ilan).
Answer: ____
