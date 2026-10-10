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

## Merge 1, 2026-10-10

Reconciled the first round: pass 1 of each of the four tracks (telemetry, learning, plumbing and kernels; PRs 126, 128, 129 and 130, squash-merged as 144871f, 291cb89, 4e4b08d and 1626dde), all four valid. Ran the checks M1 to M7 over the four pass entries, the 62 open tasks of the Learning, Runtime and Demo contexts (every open task of the four tracks and the tasks they block or consume), T0's experiment queue, the three demo beats Ilan owns ("Where are my keys?", "It learns" and "Under the hood") with the AMD and Hugging Face sponsor cuts, and the commits to shared files since the round started. Nine board changes (one cross-track blocker, the queue's numbering reconciled, four tasks or experiments given a track, two empty Tiers set, one Done when aligned to one reading path), one proposal for two shared tasks (Q16 on Dominick and Team), 19 rulings filed from the passes and 6 raised by the merge. The walk-list questions the four passes raised (T6 to T11, L13 to L20, R13 to R18, R20 to R26) and the duplicates among those tasks (E5 against R23 and R25, E11 against R26) are left as they stand until Ilan answers, so no ruling is decided by a board edit.

Walked (pass entry, task, demo beat or file :: checks :: result):
- telemetry pass 1 :: M6 :: held
- learning pass 1 :: M6 :: held
- plumbing pass 1 :: M6 :: held
- kernels pass 1 :: M6 :: held
- L18 :: M1 :: BROKE M1; FIXED (1)
- T0, the queue's numbering :: M4 :: BROKE M4; FIXED (2)
- E4 on T0 and its task R27 :: M4 :: BROKE M4; FIXED (3)
- E1 on T0 :: M4 :: BROKE M4; FIXED (4)
- B1 :: M4, M5 :: BROKE M4, M5; FIXED (5, 7)
- L10 :: M4, M5 :: BROKE M4, M5; FIXED (6, 8)
- L3 :: M2 :: BROKE M2; FIXED (9)
- L6 :: M1, M3, M5 :: BROKE M1, M5; OPEN, proposal Q16 on Dominick and Team
- R5 and D3 :: M3 :: held, R5's priority in Q16 on Dominick and Team
- L2 and F5 :: M1 :: BROKE M1; OPEN, learning ruling 3
- T6 to T11, L13 to L20, R13 to R18, R20 to R26 :: M4 :: BROKE M4; OPEN, the walk-list rulings
- E5 against R23 and R25, and E11 against R26 :: M4 :: BROKE M4; OPEN, after kernels ruling 1
- L8 and L19 :: M3, M4 :: BROKE M3; OPEN, learning ruling 8
- E4 against R27 :: M4 :: held
- T2, T3, T4 under the P0 measurements T6, T8, R20 and D8 :: M3 :: BROKE M3; OPEN, telemetry ruling 5
- the "It learns" fallback (the recorded training clip) :: M3 :: BROKE M3; OPEN, learning ruling 7
- the Jetson's attribution error (no external meter) :: M3 :: BROKE M3; OPEN, telemetry ruling 7
- the rubric-writing clauses on T9, T10, T11, L14, L20, R14, R15, R16, R18, R20, R21 and R25 :: M2 :: BROKE M2; OPEN, plumbing ruling 5
- the per-pass change cap :: M6 :: held, telemetry ruling 6
- demo beats "Where are my keys?", "It learns" and "Under the hood", and the AMD and Hugging Face cuts :: M3 :: held
- commits to contracts/, runtime/, bus/, contexts/gateway/bootstrap.py, pyproject.toml and AGENTS.md since the round started :: M2 :: held
- shared-file changes the round's Done whens plan (T5's hook, T4's stored fields, L3's telemetry.train.step) :: M2 :: held
- metric and node names across the four rubrics and the metrics file :: M2 :: held, node names in telemetry ruling 3
- O41's Blocked by (O12, O37 to O40, O42, O43) :: M1 :: held
- every other open task of the four tracks :: M1, M4, M5 :: held
- answered rulings in the four OWNER_RULINGS.md files :: M7 :: held

Findings (by dimension):
- M1 L18 :: L18 pulls the package to a non-pod machine through the proxy, and the Range-capable server that serves it is R10's (plumbing), with no Blocked by link; L18 is Ilan's alone, unlocked and Not started, Wave 5 against R10's Wave 1. FIXED (1).
- M1 L6 :: L6 (shared with Xander) pulls the packaged model from pod B through the proxy, the path R10 now serves in Range parts, and is Blocked by R6, not R10; a link to a shared task is a proposal. OPEN, Q16 on Dominick and Team.
- M1 L2, F5 :: L2 is Blocked by F5 (Xander's camera clips, Not started) although its Done when starts on fixture data and L11 builds a fixture with no cameras; F5 is another person's task, and learning pass 1 raised the question. OPEN. RULING (g).
- M1 R11, R20, R22, T9, R15 :: the cross-track consumers already carry their links (R11, R20 and R22 Blocked by T3; T9 by R11; R15 by L3), and no open task of the four tracks sits in a Wave below a blocker's (R20 and R21 share Wave 3 with R11, which the rule allows). HELD.
- M1 O41 :: Blocked by lists O12, O37, O38, O39, O40, O42 and O43, so the four pass tasks are linked and nothing is missing. HELD.
- M2 L3 :: pod B's power would be read three ways (T3's sampler, R11's harness, L3's per-step watts on telemetry.train.step) with nothing saying they read the same field the same way; the kernels rubric takes T3's sampler as the source of every joules-per-token number (W9), and the telemetry rubric (TT4, open item 9) leaves naming the shared path on L3 to the merge. FIXED (9).
- M2 shared files since the round started :: five commits touched shared files, none by a pass: 427d422 (runtime/fake.py, runtime/interface.py, runtime/llama.py, streamed answers, Dominick's), 16591d8 (contracts/telemetry.py new, contracts/__init__.py, contexts/gateway/bootstrap.py: T1, additive, FYI posted, second OK from Hemdutt's agent), 0f42dbb (contracts/day.py new, contracts/__init__.py, bootstrap.py: Hemdutt's), fa40dd5 (bootstrap.py: Hemdutt's), 2f8a38b (contracts/play.py, five additive events, bootstrap.py: Hemdutt's); each additive, each with a Build Log entry, none a rename or removal. HELD.
- M2 planned shared-file changes :: T5 plans the routing hook in runtime/ "agreed with Dominick as an additive change" (consumers runtime/ and contexts/telemetry); T4 stores its attribution fields on the projection rows and plans no change to telemetry.request beyond T1's RequestWindow (consumers contexts/telemetry, R5, D7); L3 publishes telemetry.train.step, which T1 merged; no rename, removal or version bump is planned. HELD.
- M2 metric and node names :: the four metric sets name no quantity twice and no two quantities once (teacher.joules_per_token is one name with two methods, the derived row 12 and K6's integral, told apart by conditions and re-measured by R20); node names are gpu-a and gpu-b in Context: Telemetry and contracts/telemetry.py against pod-a and pod-b in every metrics row, which telemetry pass 1 raised. OPEN. RULING (c).
- M2 the rubric-writing clauses :: twelve measurement tasks (T9, T10, T11, L14, L20, R14, R15, R16, R18, R20, R21, R25) say they write a bound, a policy, a confirmation or a citation into a RUBRIC.md, a file only the merge edits and only with a filled ruling; none is on a walk list yet. OPEN. RULING (y).
- M3 "Where are my keys?" :: needs P4 (Done), M4 (Done), A2 (Blocked, Dominick's, Locked) and L7 (shared, open); L7's chain through L2, L4 and L6 ends on the board (L2 waits on F5 and L11, L4 on L3 and L12, L6 on L5, R7 and R6), every link is P0, and the fallback (replay the keys fixture) exists. HELD.
- M3 "It learns" :: needs L1 (Done), L2 to L7 (open, P0, chains ending on the board), X10, W2 and X9 (Done) and R5 (shared, P1, Not started, Blocked by R3); the before, after and teacher scores come from L4 and L17, the model version badge from R7 and X10, labeling and training on the stats page from R5 and L3; no round change dropped a clause the beat needs. HELD.
- M3 "It learns", the fallback :: "the finished run's numbers and the recorded training clip" has no task that records a clip; L7 is shared and L16 is on no walk list, so the merge may not add the clause. OPEN. RULING (w).
- M3 "It learns", the adapter :: the beat shows the adapter on Hugging Face and the Hugging Face cut shows the published adapter and card, while L8 and L19 are P1; a Priority changes only through a filled ruling. OPEN. RULING (x).
- M3 "Under the hood" :: D3 (shared, In progress) is Blocked by R5 (shared, P1); D8 (shared, P0) by D7 (Done), T8, L17 and R20; D7 now cites each number's row, measured time and low-confidence flag (PR 132, Q15 answered for D7; R5's half waits on Xander). The path holds; R5's priority under two P0 items is in Q16. HELD.
- M3 T2, T3, T4 :: the P0 measurements T6 and R20 are Blocked by T3, T8 (P0) by T4 and D8 (P0) by T8, while T2, T3 and T4 are P1, so the slide's lab numbers wait on P1 builds under a P0-first pick order; kernels pass 1 asked the same of R11. OPEN. RULING (t).
- M3 the Jetson's attribution error :: no external meter sits on the Jetson's DC input, so no Jetson energy number can lose the low-confidence flag TT1 keeps on it, and the slide shows the flag; a meter is hardware and money, which telemetry pass 1 left for the merge to raise. OPEN. RULING (v).
- M4 T0, the queue's numbering :: the learning and plumbing passes both appended an E7 three minutes apart, the plumbing pass restated its line as E8, and nothing said which number comes next, because the experiment verb numbers from each clone's own ledger. FIXED (2).
- M4 E4 :: no track walked E4: the kernels rubric leaves it to telemetry, the telemetry rubric calls E1 to E5 kernels lines, and both passes left it to the merge; R27 (shared with Xander) is E4's task and is Blocked by T2, and the ledger seed tags E4 telemetry. FIXED (3).
- M4 E1 :: no track walked E1: the kernels rubric leaves it to plumbing, the plumbing rubric walks it for context only, and the telemetry rubric calls it a kernels line; its own line says its result decides the controller's stop and start policy. FIXED (4).
- M4 B1 :: a lab task whose only Assignee is Ilan and that no track claims; its text is kernel development and short benchmarks on gfx1201. FIXED (5).
- M4 L10 :: a learning-loop task whose only Assignee is Ilan and that no track claims (the learning lists name L2 to L9, L11 and L12). FIXED (6).
- M4 T6 to T11, L13 to L20, R13 to R18, R20 to R26 :: thirty open tasks with Ilan as their only Assignee that no rubric walk list names, so no pass may edit them; each pass raised the question for its own track, and adding them to the track map here would decide those rulings by a board edit. OPEN. RULING (b, h, m, o).
- M4 E5, R23, R25 :: E5 (the checkpoint shootout) and the board tasks R23 (the Quark loader and comparison) and R25 (FP8 weights or KV cache) measure the same levers; R23 and R25 are on no walk list, so the reconciliation waits on kernels ruling 1: if R20 to R26 join the kernels track, kernels pass 2 folds E5 into R23 and R25 inside the track; if not, merge 2 does it. OPEN. RULING (o).
- M4 E11, R26 :: E11 (batch 1, 4, 8, 16 and 64, five cold spawns per level) and R26 (concurrency 1 to 32, three repeats) run the same lever on the same metric; R26 is on no walk list, so the rule for E5 applies. OPEN. RULING (o).
- M4 L8, L19 :: not duplicates: L8 builds the publisher and the card, L19 runs the publish with the real adapter and its numbers, the build-then-run split of L2 and L13, L3 and L16, L4 and L17, L5 and L18; the card's license, n and noise clauses sit on L19 only, which learning pass 2 can carry onto L8 once learning ruling 4 is answered. HELD.
- M4 E4, R27 :: R27 is E4's own task (its page says so) and is shared with Xander, so the two are not merged; E4's track is settled by change 3 and R27 stays as it is. HELD.
- M5 B1 :: Tier empty on a task whose repair of a ROCm build is a hard build. FIXED (7).
- M5 L10 :: Tier empty on a pooled-export learning run that is a hard build. FIXED (8).
- M5 L6 :: Tier empty on a P0 task shared with Xander, so the merge may not set it. OPEN, Q16 on Dominick and Team.
- M5 every other open task of the four tracks :: Tiers agree across tracks for the same kind of work (measurement and benchmark slots Sonnet: R13, R14, R20, R22, R26, T6, T7, T9, T10; builds Opus: L2 to L5, L16, R11, R12, R15, R23 to R25, T2 to T5, T8, T11); the pass tasks O37 to O40 are Opus and O41 Fable; T0 is Fable as a standing judgment ticket. HELD.
- M6 telemetry pass 1, learning pass 1, plumbing pass 1, kernels pass 1 :: the checks pass on origin/main (162 passed); every board change reads as its AFTER on a rows-mode read of the board at 17:14Z (T2, T3, T4, T5; L2, L3, L4, L12; R10; R11, R12) and on T0 at 16:20Z (E6 to E15); each change has its Plan changes line with a Was value; each PR touched only lab/refine/<track>/ and lab/refine/research/<track>/; no Priority, Locked, Done, shared or unassigned task, rubric, rulings file, code, contract, config, fixture, test or Demo Plan changed; the sources opened here say what the changes claim (Context: Telemetry, Context: Inference Runtime, Context: Learning, contracts/learning.py L74 to L98, lab/README.md L12 to L16, lab/pod_a_teacher.sh lines 7, 53 and 58, lab/machine/verdict.py L45 and L48, lab/machine/experiments.py SEEDS, metrics.jsonl lines 3, 12, 13 and 17, the research-note headings each change cites, and research-check clean on all four folders). HELD.
- M6 the per-pass change cap :: TEMPLATES.md says PASS_RULES.md sets the cap and PASS_RULES.md names none; the four passes used 5, which no rule forbids, so the entries stand. OPEN. RULING (u).
- M7 the four OWNER_RULINGS.md files :: no ruling has an Answer yet (the files held no ruling before this merge). HELD.

Changes (9 of at most 10):
1. L18 Blocked by gains R10 (M1)
   BEFORE: "Blocked by: L5, L16"
   AFTER:  "Blocked by: L5, L16, R10"
   SOURCE: Notion task L18, Done when ("the package was pulled to a non-pod machine through the proxy"); Notion task R10, Done when as plumbing pass 1 left it (Range parts under 100 MiB from the persistent home); lab/refine/PASS_RULES.md, "The merge pass" (a Blocked by link between tasks of different tracks)
   WHY: L18 consumes what R10 produces and nothing ordered them; L18 is Wave 5 and R10 Wave 1, so the order already follows the dependency.
2. T0 queue numbering line appended (M4)
   BEFORE: "two lines numbered E7 (learning, 12:15 PM ET; plumbing, 12:18 PM ET), the plumbing pass's 'E8 supersedes E7 (plumbing)' line after them, and no line naming the next free number"
   AFTER:  "one signed merge 1 line at the end of T0: the queue reads E1 to E5 (E3 superseded by E11), E6 telemetry, E7 learning, E8 to E10 plumbing (the plumbing line first numbered E7 void), E11 to E15 kernels; the next free number is E16; a pass takes its number from a fresh fetch of T0 until the real state folder is seeded; the track map for E1 and E4"
   SOURCE: Notion task T0 as fetched 2026-10-10 16:20Z (the two E7 lines and the E8 line); lab/refine/plumbing/REFINE_LOG.md, "## Pass 1, 2026-10-10", finding "U2 lab ledger"; lab/refine/kernels/REFINE_LOG.md, "## Pass 1, 2026-10-10", ruling (d); lab/machine/experiments.py, next_id (the seeded E1 to E5 only)
   WHY: a queue whose numbers collide is ambiguous for the ledger, the merge and whoever runs an experiment; no line someone wrote was edited.
3. E4 added to the telemetry track (M4)
   BEFORE: "E4 in no track (the kernels rubric's W6 leaves it to telemetry; the telemetry rubric's walk list calls E1 to E5 kernels lines)"
   AFTER:  "E4 in the telemetry track"
   SOURCE: Notion task R27, page line of Sat 7:00 AM ET (created from the track run split for experiment E4 on T0) and its Blocked by (T2); lab/refine/kernels/RUBRIC.md, "The walk list", W6; lab/machine/experiments.py, SEEDS (E4, track telemetry, metric student.joules_per_token); lab/refine/PASS_RULES.md, "The four tracks" (the merge pass records a move and the latest record wins); the dissent, lab/refine/telemetry/RUBRIC.md, "The walk list" (E1 to E5 name tok/s or joules per token first)
   WHY: a line walked by nobody never runs; three sources tie E4 to the Jetson sampler and the routing policy, one sentence in the telemetry rubric says otherwise, and the disagreement is recorded here rather than averaged.
4. E1 added to the plumbing track (M4)
   BEFORE: "E1 in no track (the kernels rubric's W6 leaves it to plumbing; the plumbing rubric walks it for context only; the telemetry rubric calls E1 to E5 kernels lines)"
   AFTER:  "E1 in the plumbing track"
   SOURCE: Notion task T0, the E1 line ("Result decides the controller's stop/start policy"); lab/refine/PASS_RULES.md, "The four tracks" (plumbing: pods, the hub proxy link, the controller, the package server and credits); lab/refine/kernels/RUBRIC.md, "The walk list", W6; lab/machine/experiments.py, SEEDS (E1, track plumbing); the dissent, lab/refine/plumbing/RUBRIC.md, "The walk list" (walked for context, owned elsewhere) and lab/refine/telemetry/RUBRIC.md, "The walk list"
   WHY: the experiment's deliverable is a controller policy, which is plumbing's, and the metrics it reads from the other tracks are read, not moved.
5. B1 added to the kernels track (M4)
   BEFORE: "B1 in no track"
   AFTER:  "B1 in the kernels track"
   SOURCE: Notion task B1, Done when ("used only for kernel development and short benchmarks") and page line of Sat 5:55 AM (same gfx1201 silicon as the R9700, so kernels written here transfer to the pods unchanged); lab/refine/PASS_RULES.md, "The four tracks" (a task belongs to exactly one track, and the merge pass records a move)
   WHY: a task no track claims is never walked, and kernel development on the same silicon is the kernels track's.
6. L10 added to the learning track (M4)
   BEFORE: "L10 in no track"
   AFTER:  "L10 in the learning track"
   SOURCE: Notion task L10, Done when (teacher labels, one LoRA on the pooled set, the eval gate, the packaged group model with rollback, pooled data deleted); Context: Learning, "Group learning on the network GPU (L10)", https://app.notion.com/p/3f596ef5fec5814bb5dfde0052175b1e; lab/refine/PASS_RULES.md, "The four tracks"
   WHY: a task no track claims is never walked, and L10 is the learning loop run on pooled exports.
7. B1 Tier set (M5)
   BEFORE: "Tier: (empty)"
   AFTER:  "Tier: Opus"
   SOURCE: lab/refine/PASS_RULES.md, "What a pass may change" (Opus for hard builds); Notion task B1, page line of Sat 5:55 AM (the venv's torch is built against HIP 7.0 and reports 0 devices; the last serve attempt failed with "Failed to infer device type")
   WHY: an empty Tier stops the open-ticket skill, and repairing a ROCm build is a hard build.
8. L10 Tier set (M5)
   BEFORE: "Tier: (empty)"
   AFTER:  "Tier: Opus"
   SOURCE: lab/refine/PASS_RULES.md, "What a pass may change" (Opus for hard builds); Notion task L10, Done when
   WHY: an empty Tier stops the open-ticket skill, and a pooled labeling, training, gating and packaging run across two members' exports is a hard build.
9. L3 Done when, one reading path for pod B power (M2)
   BEFORE: "PEFT/TRL LoRA SFT of the same base model the Jetson runs, bf16 on one R9700 (pod torch 2.9.1 + ROCm 7.13; weights loaded in bf16, never TRL's float32 default), DataLoader with forkserver workers, no torch.compile; trains only on an approved export or the L11 fixture dataset as formatted by L12, and deletes its copy of the dataset from pod B when the run ends; LoRA on the language model only, with the vision encoder and projector frozen; for image examples, the image-text model class with max_length None or above the longest tokenized example; writes one ledger row per run with run_id, the full LoraConfig and SFTConfig values, the base model's Hub commit hash, dataset_id and the sha256 of examples.jsonl, seed and data_seed, and the torch, ROCm, PEFT, TRL and transformers versions; a few hundred examples train in under ~30 min, with trainer.samples_per_s measured as in metrics.jsonl:13 (examples seen over trainer.train() wall time, batch, max_length, rank and data recorded); logs loss to the stats page and per-step watts to telemetry.train.step"
   AFTER:  "PEFT/TRL LoRA SFT of the same base model the Jetson runs, bf16 on one R9700 (pod torch 2.9.1 + ROCm 7.13; weights loaded in bf16, never TRL's float32 default), DataLoader with forkserver workers, no torch.compile; trains only on an approved export or the L11 fixture dataset as formatted by L12, and deletes its copy of the dataset from pod B when the run ends; LoRA on the language model only, with the vision encoder and projector frozen; for image examples, the image-text model class with max_length None or above the longest tokenized example; writes one ledger row per run with run_id, the full LoraConfig and SFTConfig values, the base model's Hub commit hash, dataset_id and the sha256 of examples.jsonl, seed and data_seed, and the torch, ROCm, PEFT, TRL and transformers versions; a few hundred examples train in under ~30 min, with trainer.samples_per_s measured as in metrics.jsonl:13 (examples seen over trainer.train() wall time, batch, max_length, rank and data recorded); logs loss to the stats page and per-step watts to telemetry.train.step, read the same way as T3's pod sampler reads pod B (amd-smi SOCKET_POWER, with sampler, period_ms and the power field recorded on every row), so pod B's power has one reading path across the trainer, the sampler and the benchmark harness"
   SOURCE: lab/refine/telemetry/RUBRIC.md, TT4 ("One reading path per node") and "Known open items for pass 1", item 9; lab/refine/kernels/RUBRIC.md, "The walk list", W9; Notion task T3, Done when as telemetry pass 1 left it (SOCKET_POWER, the tool, period_ms and the power field on every sample); Context: Telemetry, "Events published" (telemetry.train.step carries socket_w) and "Design" (one sampler process per node), https://app.notion.com/p/3f596ef5fec581ba9b28f19394fa042e
   WHY: three readers of one card's power would drift apart, and the trainer's per-step watts are the telemetry matrix's pod B training cell; the clause names the field and the row keys, nothing else in L3 changes.

Rulings raised (filed into OWNER_RULINGS.md; the arrow names the track and the ruling's number there):
(a) Should routing and the slide use gross or marginal joules per answer? -> telemetry 1
    Now: gross, the Telemetry context's definition and the rubric's stated default in TT2, with the node's idle watts and their timestamp stored beside every value (change 1), so marginal can be derived later without re-measuring.
    Recommend: gross as the headline and on the slide, with marginal stored beside it under a name that says marginal; pod idle (39.0 W, metrics.jsonl:17) is about a fifth of the teacher's 200 W (metrics.jsonl:7), so a marginal figure would make a pod that stays up anyway look far cheaper than the energy it draws, and T8 should then compare gross with gross.
    Source of tension: Context: Telemetry, "Ubiquitous language" (gross by definition) and attribution-and-baselines.md#2-the-idle-baseline, AT-05 and AT-07 (the resident floor as a term inside the total; MLPerf reports gross), against T8's Done when (whole-run energy "minus the idle baseline") and T10's (one idle-baseline policy from a pod B run), with RUBRIC.md TT2 leaving the router's choice open (Known open items 7).
    Answer: ____
(b) Do T6 to T11 belong in the telemetry track's walk list? -> telemetry 2
    Now: no. RUBRIC.md, "The walk list", names T0 to T5 and replaces PASS_RULES.md's starting set, so T6 to T11 are walked for findings only and no pass changes them.
    Recommend: yes, add T6 to T11. They produce the track's first measured values (T6 and T8 are P0) and its closed-loop test, and the findings on them (T6's rows name no sampler, period or error; T9 and T10 write into the rubric; T11 asks for 100 golden questions that cannot leave the box) have nobody to act on them otherwise.
    Source of tension: PASS_RULES.md, "The four tracks" (the telemetry track covers "the T tasks"), against RUBRIC.md, "The walk list" (T0 to T5, which replaces that starting set).
    Answer: ____
(c) Which node names do telemetry events and metrics rows use: gpu-a and gpu-b, or pod-a and pod-b? -> telemetry 3
    Now: both, with a mapping: events follow Context: Telemetry (jetson, gpu-a, gpu-b), as T1's open PR #112 does, and metrics rows keep pod-a and pod-b, as all 22 rows do; this pass wrote no node name onto the board.
    Recommend: jetson, pod-a and pod-b everywhere, with the spec's language line changed to match, because every stored row, lab/machine/metrics.py, lab/README.md and the GPU Queue already say pod A and pod B, and T1 has not merged, so its docstring costs one edit now and every later comparison needs no mapping.
    Source of tension: Context: Telemetry, "Ubiquitous language" (Node: jetson, gpu-a, gpu-b) and PR #112's contracts/telemetry.py, against lab/refine/metrics.jsonl lines 1 to 22 (pod-a, pod-b) and T3's scope note from Ilan's split, which makes the pod account the node, a third name.
    Answer: ____
(d) Which task builds the first routing policy that reads a telemetry value? -> telemetry 4
    Now: none. T5's first policy routes by capability, as Context: Inference Runtime's v1 says, and T11 tests "the energy-aware policy" that no task defines, so TT6 cannot be met yet.
    Recommend: give T5 a second policy behind a switch, off by default, that sends Learning or Circle work to the teacher only when the snapshot's latest joules per answer for that work favour it, so T11 compares it with v1 on that work rather than on the golden questions, which stay on the box.
    Source of tension: Context: Inference Runtime, "Routing hook" (v1 by capability; later versions weigh joules per answer), against RUBRIC.md TT6 (a decision that reads the numbers, with its effect measured) and T11's Done when.
    Answer: ____
(e) How many held-out items does each task need before the gate may deploy? -> learning 1
    Now: L4 (change 3) deploys only past the measured noise with a paired test at p below 0.05; the size follows the Design Doc's hand-checked set of about 30 and L15's floor of 20 percent or 30 examples.
    Recommend: 50 items per task, frozen before training, with the episode count reported for image tasks. At 50 the exact McNemar test finds a 20-point net gain about 70 percent of the time against about 40 percent at 30 (eval_noise.md, appendix), and 50 is the lab's verdict floor for rates. Keep 30 only if labeling 50 does not fit before Saturday night.
    Source of tension: Atlas Design Doc section 13 (about 30 hand-checked examples) against lab/refine/research/learning/eval_noise.md#2-which-test-and-how-many-items (EN-4, 50 items) and lab/machine/verdict.py L48 (MIN_TRIALS_PER_ARM = 50).
    Answer: ____
(f) Which decode setting do the current, candidate and teacher scores share? -> learning 2
    Now: L4 requires one recorded setting for all three with thinking off; the values are unset, and runtime/llama.py sends temperature 0.2 with no seed, top_p or top_k.
    Recommend: score all three at the setting the box serves (temperature 0.2, thinking off, the runtime's max_tokens), sample each item 3 times, and record the seed where the server takes one; do not drop to temperature 0 only to shrink the noise. L17's "different seeds" then means the recorded seeds of those samples.
    Source of tension: E. Miller 2024, arXiv 2411.00640, section 3.3 (do not lower the temperature to cut variance) and the Qwen3.5-4B model card (0.7, 0.8 and 20 for non-thinking use) against lab/refine/research/learning/eval_noise.md#3-run-to-run-noise-the-same-model-can-score-differently-twice (EN-10, temperature 0 with a fixed seed) and runtime/llama.py L161 (0.2, no seed).
    Answer: ____
(g) Should L2, the teacher labeling pipeline, still wait for F5, Xander's camera clips? -> learning 3
    Now: L2 is Blocked by F5, which is not started and waits on Xander at the camera, so the labeling pipeline cannot start, although its Done when starts on fixture data and L11 builds a fixture with no cameras; this pass left the link alone.
    Recommend: drop F5 from L2's Blocked by. The tool-calling labels (L13) need only L11's fixture, and desk-object labels need a real approved export from L1 whatever the clips hold; F5 stays a blocker of C4.
    Source of tension: L2's Blocked by property (F5) against the L11 page ("this unblocks L2 and L3 from the cameras") and the Task Board's lab track note ("It only needs fixture data until L1 exists, so it can start in hour 1").
    Answer: ____
(h) Should L13 to L20 join the learning track's walk list? -> learning 4
    Now: the rubric's walk list names L2 to L9, L11 and L12, so this pass walked L13 to L20 for findings only and changed none of them.
    Recommend: yes. They are the runs that produce every number in the metric set (labels, agreement, split, first fine-tune, gate, package dry run, publish, loop time), and several need fixes no pass can make today: L14 checks agreement against a second teacher pass instead of gold, L16 deletes nothing from pod B, L17 asks for seeds the runtime cannot set, and L14 and L20 say they edit the rubric.
    Source of tension: lab/refine/learning/RUBRIC.md, "The walk list", against lab/refine/PASS_RULES.md, "The four tracks" (learning covers the chain from labeled dataset to published adapter) and the L13 to L20 pages from the 7:00 AM track run split.
    Answer: ____
(i) Does the gate score the candidate before packaging, or as the Q4_K_M GGUF the box will run? -> learning 5
    Now: as the Learning spec draws the loop (fine-tune, eval gate, then merge, GGUF and Q4), with L4 blocked by L3 only, while the current model is scored as the Q4_K_M GGUF the Jetson serves.
    Recommend: score the candidate as the packaged Q4_K_M GGUF under llama-server with the Jetson's flags, keep its bf16 score beside it as a dotted key, and make L4 wait for L5, because the before number is already a Q4_K_M number and quantizing after training cost 2.4 and 2.9 points of a fine-tune's gain in QA-LoRA's Table 1.
    Source of tension: Context: Learning, "The learning loop" (the gate before merge, GGUF and Q4) against lab/refine/research/learning/packaging_model_card.md#3-what-q4_k_m-costs-and-how-to-measure-it (PK-7; QA-LoRA, arXiv 2309.14717, Table 1) and the universal rule U1 (measured the same way).
    Answer: ____
(j) Which task does the "It learns" beat show? -> learning 6
    Now: the Demo Plan's 2:05 beat shows the desk-objects task; this pass changed nothing on the Demo Plan.
    Recommend: plan the beat on the tool-calling task's before and after, which the lab can label from fixtures today and which is the skill the keys beat credits, and show desk objects only if an approved export from L1 and a desk-object scorer both exist by the cut-off.
    Source of tension: Demo Plan, "It learns" (desk objects) against the L11 page (the first learned task can start from text fixtures today) and lab/refine/research/learning/eval_noise.md#7-scoring-the-desk-object-task-learning-pass-1-2026-10-10 (no desk-object scorer exists).
    Answer: ____
(k) During the long sessions Ilan has asked for (both pods relaunched for 40 hours once the hub is back), does the controller ping the pods every 10 minutes whatever they are doing, or only while a job runs or the teacher is in use? -> plumbing 1
    Now: R9's Done when stands, a ping every 10 minutes so the culler never fires, which Design Doc section 13 also asks for; rubric T3 cannot pass under it, because it wants 0 credits per hour when nothing works.
    Recommend: ping unconditionally during the 40-hour sessions and record credits per hour in the ledger. Both pods for 40 hours cost 2 x 2,400 x 4 = 19,200 credits of about 118,900 left (M5, M15), a culled teacher costs a relaunch of about 5 minutes plus a hand step, and the live-teach demo beat needs the teacher warm. The merge pass would then narrow T3's keep-alive clause to "while the session plan says the pod is up", and a pod-side keep-alive (IDL-12) would keep the pods alive through a front-door outage like today's.
    Source of tension: Atlas Design Doc section 13 ("a keep-alive request every 10 min against the idle culler") and R9's Done when, against lab/refine/plumbing/RUBRIC.md, T3 ("keep-alive runs only while something is working") and its known open item 6.
    Answer: ____
(l) What pull and push speed counts as acceptable for test T6? -> plumbing 2
    Now: no bound; passes record MB/s and never mark T6 passed (rubric, known open item 10).
    Recommend: a package of a few GB pulled within 10 minutes (about 5 MB/s for 3 GB) and a dataset push of the same size within 10 minutes, because a trainer run takes tens of minutes (Design Doc section 8) and shipping should not take longer than training.
    Source of tension: lab/refine/plumbing/RUBRIC.md, T6 ("both meet the bound set by ruling (open item 10)") against lab/refine/plumbing/OWNER_RULINGS.md, which holds no ruling.
    Answer: ____
(m) Do R13 to R18 join the plumbing walk list? -> plumbing 3
    Now: they are walked for findings only and never edited (refine-track skill, section 3), because the rubric's walk list names R6, R9 and R10; their findings in this entry wait.
    Recommend: yes, all six. Each was created at 7:00 AM ET from the track run split to carry one plumbing test into a measurement (R13 tests 1, 3 and 8; R14 test 6; R15 test 2; R16 test 4; R17 test 3; R18 test 5), and three of them would produce a wrong or unusable result as written (R17's credits per hour from balances, R16's shape scan, R13's 70-minute window), which pass 2 can fix once they are on the list.
    Source of tension: lab/refine/plumbing/RUBRIC.md, "The walk list" (R6, R9, R10 and the credit rules), against the R13 to R18 task pages, each created "from the track run split; why: plumbing rubric test N".
    Answer: ____
(n) Should the rubric's Jupyter Server citations be re-pinned from 2.21.1 to 2.19.0? -> plumbing 4
    Now: the rubric cites jupyter_server v2.21.1 (serverapp.py body limits, api.yaml terminals) while the research notes cite v2.19.0, the version pod A reports (measurements M6).
    Recommend: yes; the rubric's own known open item 9 says to correct a pinned citation once the pod's version is known, and only the merge pass may edit the rubric, with a filled ruling.
    Source of tension: lab/refine/plumbing/RUBRIC.md, "Primary sources" item 3 and known open item 9, against lab/refine/research/plumbing/measurements-2026-10-10.md M6.
    Answer: ____
(o) Do R20 to R26 join the kernels track's walk list? -> kernels 1
    Now: they are walked for findings only and never edited, so R20 and R21 (both P0) keep three repeats, compiled-graph reference outputs and the same Wave as R11, R23 has no Blocked by, and R24 puts the Conch config where a vLLM diff cannot carry it.
    Recommend: yes for R20, R21, R22, R23, R24, R25 and R26, which the track run split created for this track and whose Done whens cite the kernels rubric and research notes; R27 (shared with Xander, on the Jetson) stays out. The merge pass carries it into the rubric's walk list, and pass 2 then fixes those rows.
    Source of tension: lab/refine/kernels/RUBRIC.md, "The walk list" (W1 to W6 replace the starting set of PASS_RULES.md, "The four tracks": R11, R12 and T0's lines) against the Task Board rows R20 to R26, created on Sat 7:00 AM ET from the track run split, whose Done whens cite "kernels rubric test 1" (R20) and "kernels rubric test 4" (R21).
    Answer: ____
(p) Is E2 (vLLM against llama.cpp) a lever ranked by expected gain, or a reference measurement kept out of the ranking? -> kernels 2
    Now: E2 stays unscored in the ledger and on T0 (no confidence), so it ranks after every scored line.
    Recommend: a reference measurement, run once beside the stock matrix and again after E13, outside the gain ranking: the Design Doc keeps llama.cpp ROCm as an emergency fallback, so its gain is not one the teacher would take, and its arms differ in engine and weight format at once.
    Source of tension: lab/refine/kernels/RUBRIC.md K8 (kernels experiments ordered by expected joules-per-token gain times confidence) and K2 (one lever) against Atlas Design Doc section 10 (llama.cpp ROCm only as an emergency fallback) and T0's E2 line.
    Answer: ____
(q) Should R11 (the benchmark harness) become P0? -> kernels 3
    Now: P1; a pass changes no Priority.
    Recommend: P0, because the P0 tasks R20 and R21, and through them R24 and D8, are blocked by it, so the demo's lab numbers cannot exist until it is built, and a P0-first pick order would leave it waiting.
    Source of tension: lab/refine/PASS_RULES.md, "What a pass may never change" (a Priority change takes a filled ruling) against the Task Board's Blocked by links (R20 and R21, P0, blocked by R11, P1) and AGENTS.md section 5 (P0 first).
    Answer: ____
(r) Where does the experiment queue live from now on? -> kernels 4
    Now: T0 is the record; each pass ran the experiment verb in a scratch state folder, so its rank sees only that pass's lines, and the next E number is read from T0 (one collision so far, E7, renumbered by the plumbing pass).
    Recommend: seed the real state folder once with python -m lab experiment seed and add E6 to E15 from T0 in one ordinary task, so the verb hands out numbers under its lock and ranks the whole queue; until then passes keep reading T0 for the next number.
    Source of tension: lab/machine/experiments.py (an experiment lives in the ledger and T0 mirrors it through --print) against the O27 page ("Nobody has run python -m lab experiment seed on the real state folder yet; Ilan's call") and the scratch state folder every pass of this round was told to use.
    Answer: ____
(s) Which batch sizes make the stock record? -> kernels 5
    Now: both sets: E11 runs 1, 4, 8 and 16 and adds 64 when it fits, as the rubric's known open items ask until a ruling picks one.
    Recommend: 1, 4, 8 and 16 as the record, with 64 an extra arm only when the KV cache and the float32 GDN state fit at the workload's context, since 64 sequences need about 9.7 GB of GDN state (about 151 MB each) on top of the KV cache.
    Source of tension: lab/refine/kernels/RUBRIC.md K1 (batch sizes 1, 4, 8 and 16) and its Known open items for pass 1 (K1) against T0's E3 line (batch 1, 4, 16 and 64).
    Answer: ____
(t) Should T2, T3 and T4 become P0, since the P0 tasks T6, T8, R20 and D8 are blocked by them? -> telemetry 5
    Now: T2, T3 and T4 stay P1; T6 (P0) is Blocked by T3, T8 (P0) by T4, D8 (P0, the slide's lab numbers) by T8, and R20 (P0) by T3, so under the P0-first pick order the P0 measurements wait on P1 builds.
    Recommend: P0 for T2, T3 and T4. The slide's watts per node and joules per answer and per token (D8) cannot exist without the two samplers and the attribution, and R11, for which kernels ruling 3 asks the same, is blocked by T3 too. Keep T5 and T7 at P1.
    Source of tension: The Task Board's Priority values (T2, T3, T4 at P1; T6, T8, R20, D8 at P0) and their Blocked by links, against AGENTS.md section 5 (P0 first) and lab/refine/PASS_RULES.md, "What a pass may never change" (a Priority change takes a filled ruling, which the merge pass applies).
    Answer: ____
(u) What is the per-pass change cap? -> telemetry 6
    Now: 5. lab/refine/TEMPLATES.md section 1.1 says the cap is set in PASS_RULES.md, PASS_RULES.md names none, and all four pass 1 entries used 5 and wrote "of at most 5".
    Recommend: 5, written into PASS_RULES.md by an ordinary task (the file changes only through a ruling and a board task): five changes with BEFORE, AFTER, SOURCE and WHY are what one Opus sitting verified on the board in one pass today, and a higher cap leaves more changes unwalked before the next pass.
    Source of tension: lab/refine/TEMPLATES.md, section 1.1 ("Changes (<n> of at most <cap>): ... The cap is set in PASS_RULES.md") against lab/refine/PASS_RULES.md, which sets no cap and says no pass edits it; raised by telemetry pass 1 under "Left for the next pass".
    Answer: ____
(v) Does the Jetson get an external meter on its DC input, so its attribution error can be measured? -> telemetry 7
    Now: No meter; every Jetson energy number keeps the low-confidence flag, because TT1 lets only an error measured on our own node against a reference lift it, and published calibrations found Jetson internal sensors under-reading by up to 50 percent.
    Recommend: No purchase this weekend: the Jetson's numbers stay flagged, the slide and the stats page show the flag (lab/slide_numbers.py prints it since PR 132) with the VDD_IN rail named as the source, and the meter is revisited after the hackathon if the energy thread continues.
    Source of tension: lab/refine/telemetry/RUBRIC.md, TT1 (only a bound measured on our own node against a reference lifts the flag; Shalavi et al. 2023, up to 50 percent under-reading) and "Known open items for pass 1", item 5 (a meter is hardware and money, a question for Ilan), against the open-ticket skill, section 7 (no new paid services or purchases without Ilan).
    Answer: ____
(w) Which task records the training clip that the "It learns" beat's fallback needs? -> learning 7
    Now: None. The Demo Plan's fallback for the "It learns" beat is the finished run's numbers and the recorded training clip; L2 to L20 produce the numbers, and no task records a clip.
    Recommend: L16, the first real fine-tune run on pod B: its Done when gains a screen recording of the run's progress (the loss curve on the stats page, or the trainer's log scrolling) saved under fixtures/demo/ and named in the run's ledger row, applied by learning pass 2 once L16 is on the walk list (learning ruling 4) or by Ilan by hand; L7 (shared with Dominick) stays as it is.
    Source of tension: Demo Plan, "It learns", column "If it breaks" (the recorded training clip) and lab/refine/learning/RUBRIC.md, LT9 (a rehearsed fallback), against the L2 to L20 Done whens, none of which names a clip, and lab/refine/PASS_RULES.md, "What a pass may change" (only a task's own track rewrites its Done when, and L16 is on no walk list until ruling 4 is answered).
    Answer: ____
(x) Should L8, publishing the adapter to the Hugging Face Hub, become P0? -> learning 8
    Now: P1 (and L19, the publish run with the card's numbers, P1), while the "It learns" beat shows the adapter on Hugging Face and the Hugging Face sponsor cut shows our published LoRA adapter and model card; a Priority changes only through a filled ruling.
    Recommend: P0 for L8 (a Sonnet-sized build: the push and the card template), with L19 kept at P1 unless the cut is rehearsed with the real card; the beat and the cut are the reason the adapter is on the plan at all.
    Source of tension: Demo Plan, "It learns" (adapter on Hugging Face) and "2-minute sponsor-room cuts", Hugging Face (our published LoRA adapter and model card), against the Task Board's Priority on L8 and L19 (P1) and lab/refine/PASS_RULES.md, "What a pass may never change" (any task's Priority).
    Answer: ____
(y) May the measurement tasks write their results into a RUBRIC.md directly, as their Done whens say, or does each such result come to the merge pass as a ruling? -> plumbing 5
    Now: PASS_RULES.md stands: only the merge pass edits a RUBRIC.md, and only to carry out a filled ruling, so the clauses "written into the rubric", "confirmed or corrected" in a rubric and "rubric test N cites it" on T9, T10, T11, L14, L20, R14, R15, R16, R18, R20, R21 and R25 cannot be met as written, and none of those tasks is on a walk list yet (telemetry ruling 2, learning ruling 4, plumbing ruling 3, kernels ruling 1).
    Recommend: The task records its result in the ledger or the metrics file and on its own page, and the rubric edit is raised as a ruling in its track's file through the track's next pass, which the merge pass carries out; the tracks' next passes reword those clauses once the walk-list rulings admit the tasks. Nothing else changes: the rubrics already say their bounds and citations are corrected through a ruling.
    Source of tension: The Done whens of T9, T10, T11, L14, L20, R14, R15, R16, R18, R20, R21 and R25 (created Sat 7:00 AM ET from the track run split, each citing a rubric test), against lab/refine/PASS_RULES.md, "The merge pass" (the RUBRIC.md of any track, only to carry out a filled ruling) and "What a pass may never change" (the rubric).
    Answer: ____

Left for the next pass: Track map beyond the rubric walk lists: telemetry E4; learning L10; plumbing E1; kernels B1. Telemetry pass 2 walks E4 beside T0 to T5, E6 and the matrix (R27, E4's task, is shared with Xander, so findings only there) and reads the merge 1 numbering line on T0; learning pass 2 walks L10, and re-walks L3 as change 9 left it and L18 with its new blocker; plumbing pass 2 walks E1 against T1 to T8 beside R10, E8, E9 and E10; kernels pass 2 walks B1 against K1 to K8 and re-checks E5 against R23 and R25 and E11 against R26 once kernels ruling 1 is answered. Every pass picks a T0 number from a fresh fetch (next free: E16) until kernels ruling 4 seeds the real state folder. Waits on Ilan: telemetry 1 to 7, learning 1 to 8, plumbing 1 to 5, kernels 1 to 5. Waits on Xander (and Dominick for D3): Q16 on Dominick and Team (L6's Tier and its R10 link, R5's priority); waits on Xander: Q15's R5 half.

Checks run (commands and their actual output):
python -m pytest -q tests/test_lab_refine_merge.py tests/test_lab_refine_formats.py on origin/main at 1626dde, before any change: 162 passed in 3.18s.
python -m pytest -q tests/test_lab_refine_merge.py tests/test_lab_refine_formats.py with this entry and the 25 rulings in place: 162 passed in 2.78s.
python -m lab rulings list on origin/main before the merge: {"ok": true, "unanswered": 0, "answered": 0, "rulings": [], "errors": [], "warnings": []}; after the rulings were filed: {"ok": true, "unanswered": 25, "answered": 0, "rulings": [...25 entries...], "errors": [], "warnings": []} (trimmed).
python -m lab research-check on lab/refine/research/telemetry, learning, plumbing and kernels: clean, with 64, 83, 79 and 101 findings.
Board read: notion-query-data-sources, rows mode, Status is not Done and Context in Learning, Runtime or Demo, 2026-10-10 17:14Z: 62 rows, has_more false; every pass 1 change reads as its AFTER (T2, T3, T4, T5; L2, L3, L4, L12; R10; R11, R12). T0 fetched at 16:20Z: E6 to E15 as the passes left them, the two E7 lines and the E8 supersedes line.
Re-fetched after the changes: L18 at 17:25:07Z (Blocked by L5, L16, R10); T0 at 17:25:46Z (the merge 1 numbering line last, once); B1 at 17:27:43Z (Tier Opus); L10 at 17:27:50Z (Tier Opus); L3 at 17:28:10Z (Done when reads as AFTER); Ilan's Team Comms page at 17:28:18Z (the nine merge 1 lines under Plan changes, after the plumbing E10 line); Dominick and Team at 17:28:45Z (Q16 under Open questions with a blank Answer line) and the Team Comms hub (the Q16 line under Open questions at a glance).

---
