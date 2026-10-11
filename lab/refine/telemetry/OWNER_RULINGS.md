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
Answer: Gross joules per answer, with the node's idle watts stored beside every value so marginal can be derived later. Routing and the slide use gross. (owner, 2026-10-10)

## 2. Do T6 to T11 belong in the telemetry track's walk list? (pass 1)

Now: no. RUBRIC.md, "The walk list", names T0 to T5 and replaces PASS_RULES.md's starting set, so T6 to T11 are walked for findings only and no pass changes them.
Recommend: yes, add T6 to T11. They produce the track's first measured values (T6 and T8 are P0) and its closed-loop test, and the findings on them (T6's rows name no sampler, period or error; T9 and T10 write into the rubric; T11 asks for 100 golden questions that cannot leave the box) have nobody to act on them otherwise.
Source of tension: PASS_RULES.md, "The four tracks" (the telemetry track covers "the T tasks"), against RUBRIC.md, "The walk list" (T0 to T5, which replaces that starting set).
Answer: Yes. The walk list is every telemetry task in my lane, T0 to T11, under the same rules (lane only, 5 changes per pass, no priority or rubric edits by a pass). (owner, 2026-10-10)

## 3. Which node names do telemetry events and metrics rows use: gpu-a and gpu-b, or pod-a and pod-b? (pass 1)

Now: both, with a mapping: events follow Context: Telemetry (jetson, gpu-a, gpu-b), as T1's open PR #112 does, and metrics rows keep pod-a and pod-b, as all 22 rows do; this pass wrote no node name onto the board.
Recommend: jetson, pod-a and pod-b everywhere, with the spec's language line changed to match, because every stored row, lab/machine/metrics.py, lab/README.md and the GPU Queue already say pod A and pod B, and T1 has not merged, so its docstring costs one edit now and every later comparison needs no mapping.
Source of tension: Context: Telemetry, "Ubiquitous language" (Node: jetson, gpu-a, gpu-b) and PR #112's contracts/telemetry.py, against lab/refine/metrics.jsonl lines 1 to 22 (pod-a, pod-b) and T3's scope note from Ilan's split, which makes the pod account the node, a third name.
Answer: Events and metrics rows use pod-a, pod-b and jetson, what every stored row already uses. The spec's gpu-a and gpu-b map to pod-a and pod-b; the mapping is one line in the telemetry context docs. (owner, 2026-10-10)

## 4. Which task builds the first routing policy that reads a telemetry value? (pass 1)

Now: none. T5's first policy routes by capability, as Context: Inference Runtime's v1 says, and T11 tests "the energy-aware policy" that no task defines, so TT6 cannot be met yet.
Recommend: give T5 a second policy behind a switch, off by default, that sends Learning or Circle work to the teacher only when the snapshot's latest joules per answer for that work favour it, so T11 compares it with v1 on that work rather than on the golden questions, which stay on the box.
Source of tension: Context: Inference Runtime, "Routing hook" (v1 by capability; later versions weigh joules per answer), against RUBRIC.md TT6 (a decision that reads the numbers, with its effect measured) and T11's Done when.
Answer: T5's second policy. T5 gets a line: policy one routes by capability, policy two reads the latest telemetry snapshot; both ship behind the same hook. (owner, 2026-10-10)

## 5. Should T2, T3 and T4 become P0, since the P0 tasks T6, T8, R20 and D8 are blocked by them? (merge 1)

Now: T2, T3 and T4 stay P1; T6 (P0) is Blocked by T3, T8 (P0) by T4, D8 (P0, the slide's lab numbers) by T8, and R20 (P0) by T3, so under the P0-first pick order the P0 measurements wait on P1 builds.
Recommend: P0 for T2, T3 and T4. The slide's watts per node and joules per answer and per token (D8) cannot exist without the two samplers and the attribution, and R11, for which kernels ruling 3 asks the same, is blocked by T3 too. Keep T5 and T7 at P1.
Source of tension: The Task Board's Priority values (T2, T3, T4 at P1; T6, T8, R20, D8 at P0) and their Blocked by links, against AGENTS.md section 5 (P0 first) and lab/refine/PASS_RULES.md, "What a pass may never change" (a Priority change takes a filled ruling, which the merge pass applies).
Answer: Yes, T2, T3 and T4 are P0 (done on the board Sat 3:30 PM ET), because the P0 tasks T6, T8 and D8 are blocked by them. (owner, 2026-10-10)

## 6. What is the per-pass change cap? (merge 1)

Now: 5. lab/refine/TEMPLATES.md section 1.1 says the cap is set in PASS_RULES.md, PASS_RULES.md names none, and all four pass 1 entries used 5 and wrote "of at most 5".
Recommend: 5, written into PASS_RULES.md by an ordinary task (the file changes only through a ruling and a board task): five changes with BEFORE, AFTER, SOURCE and WHY are what one Opus sitting verified on the board in one pass today, and a higher cap leaves more changes unwalked before the next pass.
Source of tension: lab/refine/TEMPLATES.md, section 1.1 ("Changes (<n> of at most <cap>): ... The cap is set in PASS_RULES.md") against lab/refine/PASS_RULES.md, which sets no cap and says no pass edits it; raised by telemetry pass 1 under "Left for the next pass".
Answer: 5 changes per pass, as every pass used; write it into PASS_RULES.md through the merge pass. (owner, 2026-10-10)

## 7. Does the Jetson get an external meter on its DC input, so its attribution error can be measured? (merge 1)

Now: No meter; every Jetson energy number keeps the low-confidence flag, because TT1 lets only an error measured on our own node against a reference lift it, and published calibrations found Jetson internal sensors under-reading by up to 50 percent.
Recommend: No purchase this weekend: the Jetson's numbers stay flagged, the slide and the stats page show the flag (lab/slide_numbers.py prints it since PR 132) with the VDD_IN rail named as the source, and the meter is revisited after the hackathon if the energy thread continues.
Source of tension: lab/refine/telemetry/RUBRIC.md, TT1 (only a bound measured on our own node against a reference lifts the flag; Shalavi et al. 2023, up to 50 percent under-reading) and "Known open items for pass 1", item 5 (a meter is hardware and money, a question for Ilan), against the open-ticket skill, section 7 (no new paid services or purchases without Ilan).
Answer: Not now. Every Jetson energy number keeps the low-confidence flag; the question stays open for me. (owner, 2026-10-10)

## 8. Should the Jetson power sampler be switched on in the gateway, and at what publish rate? (pass 2)

Now: it is switched on nowhere: nothing on main sets ATLAS_TELEMETRY_SOURCE=tegrastats, so the box writes no Jetson power row and the Jetson cells of the matrix have no rows; when it is switched on it publishes one telemetry.jetson.power event every 100 ms, T2's default.
Recommend: switch it on at 100 ms, which keeps a warm caption of about 2.1 s above the 10-sample floor, and time the next gateway restart on the box (power-sampling.md PS-35's candidate experiment); move to ATLAS_TELEMETRY_JETSON_INTERVAL_MS=200 only if one day of samples adds more than about 10 s to a start. The volume fits (about 414 MB a day against about 92 GB free on the box's card, R3), and a change-only or batched stream (PS-36) can wait until the log has to outlive the weekend.
Source of tension: lab/refine/telemetry/RUBRIC.md TT1 (fewer than 10 samples is low confidence) and T2's Done when (100 ms by default) against Context: Telemetry (tegrastats described as about 1 Hz; energy per day rebuilt from the log) and power-sampling.md#6-the-jetson-sampler-on-the-box-power-modes-idle-power-cost-and-event-volume, PS-35 (about 414 MB and about 5.4 s of parsing per day of samples here), with the T2 page's merge line leaving retention or a lower rate to Ilan.
Answer: On, at 350 ms: ATLAS_TELEMETRY_SOURCE=tegrastats and ATLAS_TELEMETRY_JETSON_INTERVAL_MS=350, applied at the next gateway restart Xander does. 100 ms costs 3.5 percent of a core (T13), over the 1 percent budget, and 350 ms still keeps 10 samples inside a 3.5 s window. Retention and a batched stream wait until after the demo. (owner, 2026-10-10)

## 9. On the Strix Halo pods, does a pod's joules per answer count the whole APU socket (CPU cores and graphics together) or a GPU-only share, and if a share, how is it split? (pass 2)

Now: the whole socket. The averaged power the samplers read on these APUs is the firmware's SocketPower, the package power (power-sampling.md PS-37), and T4 (change 3) names that boundary and the accelerator on every value; nothing splits it, so a teacher answer's joules include the serving process's CPU work and anything else running on pod C.
Recommend: keep the whole socket as the headline, since it is what the node draws and what ruling 1's gross definition integrates, and store a GPU-only share beside it only when the firmware reports one: if the pods expose the APU metrics table (gpu_metrics v3.0, with average_gfx_power and average_all_core_power), T3's sampler records both with every sample and T4 stores the graphics joules beside each value; never split by a fixed ratio. One read of that table on pod C, idle and under a GPU-only and a CPU-only load (PS-37's candidate experiment), says which case holds, and pass 3 can queue it on T0 as E6's gfx1151 restatement.
Source of tension: lab/refine/telemetry/OWNER_RULINGS.md 1 and Context: Telemetry (joules per answer is the answering node's power, gross) against the plan written for the R9700, which counted the graphics package only (T3's Done when, the rubric's TT4 power boundary), and the dispatcher's note on the retarget that GPU-only attribution needs a ruling or a measured split; the driver carries a firmware GFX power on these APUs (kgd_pp_interface.h, gpu_metrics_v3_0), whose exposure on the pods is unmeasured.
Answer: The whole APU socket is the headline joules per answer, with the accelerator named on every value. A GPU-only share is stored beside it only when the firmware reports one (gpu_metrics v3.0 average_gfx_power); never a fixed ratio. One read of that table on pod C, idle, GPU-only load and CPU-only load, decides which case holds, queued as E6 restated for gfx1151. (owner, 2026-10-10)

## 10. How do the telemetry rubric's R9700 clauses (TT1's AMD SMI note, TT4's graphics-package boundary and known open item 3) change for the Strix Halo pods? (merge 2)

Now: The rubric stands as written: TT1 reasons from the R9700 being a Navi 48 (RDNA 4) card, TT4 says the R9700's reading is the graphics package (socket), and known open item 3 asks for the averaging window and lag of the R9700's socket power; telemetry pass 2 put the APU reading (the whole socket, CPU cores and graphics together, PS-37) into the T4 and T10 Done whens and raised ruling 9 on the power split, E6 is still written for the R9700's field, and passes read every R9700 clause as the Radeon 8060S by Ilan's decision while editing no rubric.
Recommend: Rewrite the three clauses for the Radeon 8060S (gfx1151, an APU under the SMU 14.0.x driver): TT1 names the APU's averaged SocketPower as the field the samplers read and keeps the R9700 sentence as a dated note; TT4's power boundary for a pod becomes the whole APU socket with the accelerator named on every value, as T4 now says, with a GPU-only share stored beside it only as ruling 9 decides; known open item 3 asks for the averaging window and lag of the 8060S's socket power, which telemetry pass 3 restates E6 to measure; nothing else in the rubric changes.
Source of tension: lab/refine/telemetry/RUBRIC.md, TT1, TT4 and "Known open items for pass 1", item 3 (written for the R9700; only the merge pass edits the file, and only with a filled ruling) against Ilan's decision of 2026-10-10 on the Ilan1 and Ilan2 dispatcher page (the lab is the two SJC Strix Halo pods; everything that says R9700 or gfx1201 is retargeted as it is touched), lab/refine/research/telemetry/power-sampling.md#7-the-strix-halo-pods-what-an-apus-power-fields-cover (PS-37) and the T4 and T10 Done whens as telemetry pass 2 left them.
Answer: Rewrite the three clauses for the Radeon 8060S as recommended: TT1 names the APU's averaged SocketPower as the field the samplers read, TT4's pod boundary is the whole socket with the accelerator named, known open item 3 asks for the 8060S's averaging window and lag. The R9700 sentences stay as dated notes. (owner, 2026-10-10)

## 11. Do rows from the SJC pods name their node by role (pod-a and pod-b, as telemetry ruling 3 decides) or by account (pod-c and pod-d, as R29's twelve metrics rows and the controller since O54 do)? (merge 2)

Now: By role for every row written from here on, with the hub and the account letter in its conditions: merge 2 reads ruling 3 and the O56 Done when that way, R13's Done when says so, and R31 carries it into lab/hub.py and lab/controller.py; the twelve rows R29 wrote on node pod-c (lab/refine/metrics.jsonl, PR #194) stay as written and are read as pod-a-role rows of account C, because a stored row is never rewritten.
Recommend: Role, with the hub and account beside it: every comparison the tracks make is between the teacher's pod or the trainer's pod across hubs and chips, which the role names and the conditions tell apart, while an account name changes with every hub move; the metrics tooling may alias node pod-c to role pod-a with account C for the twelve rows, or a later task may append a re-read of those values under the role name, and either way no row is edited.
Source of tension: lab/refine/telemetry/OWNER_RULINGS.md 3 (owner, 2026-10-10: events and metrics rows use pod-a, pod-b and jetson, what every stored row already uses) against R29's Done when (rows on node pod-c with the GPU named) and the twelve pod-c rows of lab/refine/metrics.jsonl (PR #194, 498e161), lab/hub.py since O54 (5b3e9dc: an account beyond A and B is named pod-<letter>), and the O56 Done when (the conflict resolved to roles mapped to accounts).
Answer: Role names: pod-a, pod-b and jetson on every row written from here on, with the hub and the account letter in the conditions. The twelve pod-c rows stay as written and are read as pod-a rows of account C; the metrics tooling may alias them; no stored row is edited. (owner, 2026-10-10)
