# Owner rulings for the learning track

Questions the passes could not decide, because they are policy, not method. Each shows what the
plan does NOW (the default the pass applied so the loop could continue), the recommendation, the
source of the tension, and a blank for the owner's answer. A filled Answer is an owner rule at top
authority: the next pass applies it and cites it in `REFINE_LOG.md`. Passes never edit this file;
the merge pass adds questions from the pass entries' "Rulings raised" sections. The format is
`lab/refine/TEMPLATES.md`, section "The rulings file".

## 1. How many held-out items does each task need before the gate may deploy? (pass 1)

Now: L4 (change 3) deploys only past the measured noise with a paired test at p below 0.05; the size follows the Design Doc's hand-checked set of about 30 and L15's floor of 20 percent or 30 examples.
Recommend: 50 items per task, frozen before training, with the episode count reported for image tasks. At 50 the exact McNemar test finds a 20-point net gain about 70 percent of the time against about 40 percent at 30 (eval_noise.md, appendix), and 50 is the lab's verdict floor for rates. Keep 30 only if labeling 50 does not fit before Saturday night.
Source of tension: Atlas Design Doc section 13 (about 30 hand-checked examples) against lab/refine/research/learning/eval_noise.md#2-which-test-and-how-many-items (EN-4, 50 items) and lab/machine/verdict.py L48 (MIN_TRIALS_PER_ARM = 50).
Answer: At least 50 held-out items per task before the gate may deploy, per the eval-noise research; fewer is reported but never deploys. (owner, 2026-10-10)

## 2. Which decode setting do the current, candidate and teacher scores share? (pass 1)

Now: L4 requires one recorded setting for all three with thinking off; the values are unset, and runtime/llama.py sends temperature 0.2 with no seed, top_p or top_k.
Recommend: score all three at the setting the box serves (temperature 0.2, thinking off, the runtime's max_tokens), sample each item 3 times, and record the seed where the server takes one; do not drop to temperature 0 only to shrink the noise. L17's "different seeds" then means the recorded seeds of those samples.
Source of tension: E. Miller 2024, arXiv 2411.00640, section 3.3 (do not lower the temperature to cut variance) and the Qwen3.5-4B model card (0.7, 0.8 and 20 for non-thinking use) against lab/refine/research/learning/eval_noise.md#3-run-to-run-noise-the-same-model-can-score-differently-twice (EN-10, temperature 0 with a fixed seed) and runtime/llama.py L161 (0.2, no seed).
Answer: One recorded setting for current, candidate and teacher: thinking off, greedy decoding, one fixed maximum output length; the setting is written in the ledger row of every score. (owner, 2026-10-10)

## 3. Should L2, the teacher labeling pipeline, still wait for F5, Xander's camera clips? (pass 1)

Now: L2 is Blocked by F5, which is not started and waits on Xander at the camera, so the labeling pipeline cannot start, although its Done when starts on fixture data and L11 builds a fixture with no cameras; this pass left the link alone.
Recommend: drop F5 from L2's Blocked by. The tool-calling labels (L13) need only L11's fixture, and desk-object labels need a real approved export from L1 whatever the clips hold; F5 stays a blocker of C4.
Source of tension: L2's Blocked by property (F5) against the L11 page ("this unblocks L2 and L3 from the cameras") and the Task Board's lab track note ("It only needs fixture data until L1 exists, so it can start in hour 1").
Answer: No. L2 starts on the no-camera fixture dataset from L11; camera exports are labeled later when clips exist. F5 removed from L2's Blocked by on Sat 3:30 PM ET. (owner, 2026-10-10)

## 4. Should L13 to L20 join the learning track's walk list? (pass 1)

Now: the rubric's walk list names L2 to L9, L11 and L12, so this pass walked L13 to L20 for findings only and changed none of them.
Recommend: yes. They are the runs that produce every number in the metric set (labels, agreement, split, first fine-tune, gate, package dry run, publish, loop time), and several need fixes no pass can make today: L14 checks agreement against a second teacher pass instead of gold, L16 deletes nothing from pod B, L17 asks for seeds the runtime cannot set, and L14 and L20 say they edit the rubric.
Source of tension: lab/refine/learning/RUBRIC.md, "The walk list", against lab/refine/PASS_RULES.md, "The four tracks" (learning covers the chain from labeled dataset to published adapter) and the L13 to L20 pages from the 7:00 AM track run split.
Answer: Yes, the walk list is every learning task in my lane, L2 to L20, under the same rules. (owner, 2026-10-10)

## 5. Does the gate score the candidate before packaging, or as the Q4_K_M GGUF the box will run? (pass 1)

Now: as the Learning spec draws the loop (fine-tune, eval gate, then merge, GGUF and Q4), with L4 blocked by L3 only, while the current model is scored as the Q4_K_M GGUF the Jetson serves.
Recommend: score the candidate as the packaged Q4_K_M GGUF under llama-server with the Jetson's flags, keep its bf16 score beside it as a dotted key, and make L4 wait for L5, because the before number is already a Q4_K_M number and quantizing after training cost 2.4 and 2.9 points of a fine-tune's gain in QA-LoRA's Table 1.
Source of tension: Context: Learning, "The learning loop" (the gate before merge, GGUF and Q4) against lab/refine/research/learning/packaging_model_card.md#3-what-q4_k_m-costs-and-how-to-measure-it (PK-7; QA-LoRA, arXiv 2309.14717, Table 1) and the universal rule U1 (measured the same way).
Answer: The gate scores the packaged Q4_K_M GGUF the box will run, served the way the box serves it; the bf16 candidate may be scored earlier as a preview but only the packaged score can deploy. (owner, 2026-10-10)

## 6. Which task does the "It learns" beat show? (pass 1)

Now: the Demo Plan's 2:05 beat shows the desk-objects task; this pass changed nothing on the Demo Plan.
Recommend: plan the beat on the tool-calling task's before and after, which the lab can label from fixtures today and which is the skill the keys beat credits, and show desk objects only if an approved export from L1 and a desk-object scorer both exist by the cut-off.
Source of tension: Demo Plan, "It learns" (desk objects) against the L11 page (the first learned task can start from text fixtures today) and lab/refine/research/learning/eval_noise.md#7-scoring-the-desk-object-task-learning-pass-1-2026-10-10 (no desk-object scorer exists).
Answer: ____

## 7. Which task records the training clip that the "It learns" beat's fallback needs? (merge 1)

Now: None. The Demo Plan's fallback for the "It learns" beat is the finished run's numbers and the recorded training clip; L2 to L20 produce the numbers, and no task records a clip.
Recommend: L16, the first real fine-tune run on pod B: its Done when gains a screen recording of the run's progress (the loss curve on the stats page, or the trainer's log scrolling) saved under fixtures/demo/ and named in the run's ledger row, applied by learning pass 2 once L16 is on the walk list (learning ruling 4) or by Ilan by hand; L7 (shared with Dominick) stays as it is.
Source of tension: Demo Plan, "It learns", column "If it breaks" (the recorded training clip) and lab/refine/learning/RUBRIC.md, LT9 (a rehearsed fallback), against the L2 to L20 Done whens, none of which names a clip, and lab/refine/PASS_RULES.md, "What a pass may change" (only a task's own track rewrites its Done when, and L16 is on no walk list until ruling 4 is answered).
Answer: ____

## 8. Should L8, publishing the adapter to the Hugging Face Hub, become P0? (merge 1)

Now: P1 (and L19, the publish run with the card's numbers, P1), while the "It learns" beat shows the adapter on Hugging Face and the Hugging Face sponsor cut shows our published LoRA adapter and model card; a Priority changes only through a filled ruling.
Recommend: P0 for L8 (a Sonnet-sized build: the push and the card template), with L19 kept at P1 unless the cut is rehearsed with the real card; the beat and the cut are the reason the adapter is on the plan at all.
Source of tension: Demo Plan, "It learns" (adapter on Hugging Face) and "2-minute sponsor-room cuts", Hugging Face (our published LoRA adapter and model card), against the Task Board's Priority on L8 and L19 (P1) and lab/refine/PASS_RULES.md, "What a pass may never change" (any task's Priority).
Answer: No, L8 and L19 stay P1. (owner, 2026-10-10)

## 9. What does the gate score for the tool-calling task, now that every demo question is answered by the pre-fetch and the current model scores 42 of 42 there, and does tool calling stay the task L7 and the beat lead with? (pass 2)

Now: L4 scores through contexts/agent/eval.py, as Ilan's scope note on L4 and RUBRIC.md LT4 say, which answers every golden wording through QA.ask's pre-fetch; L7 starts with desk objects and tool calling; learning ruling 6 (which task the beat shows) is unanswered; E16 (change 2) measures the current model on both paths.
Recommend: decide after E16, which needs only the Jetson: if the current model also passes most of the 24 tool-answerable wordings through QA.tool_loop, the tool-calling task has no room to show a gain, so L7 and the beat lead with desk objects, which Context: Learning ties to the keys beat; if it fails a good share, keep tool calling, score it through QA.tool_loop with eval.py's checks and keep the QA.ask score as a dotted key, and say on the slide that the gain is on questions answered without the pre-fetch. Either way the keys line holds only for desk objects, because the keys answer comes from the pre-fetch.
Source of tension: Ilan's scope note on L4 (reuse python -m contexts.agent.eval, its score the top-line number) and RUBRIC.md LT4 (the same command), against contexts/agent/eval.py L47 and L71 with qa.py QA.ask (demo questions never take the open tool loop), PR #120, PR #148 (42 of 42 through QA.ask on the Jetson), lab/refine/research/learning/distillation_tool_calls.md#6-which-answer-path-a-tool-calling-score-reads-learning-pass-2-2026-10-10 (DT-15) and Context: Learning, "Tasks we teach at the hackathon" (desk objects powers the keys beat; tool calling makes on-device Q&A reliable).
Answer: ____

## 10. Which tool list does the student train on and get scored with? (pass 2)

Now: L12 (Locked, being built) formats traces with TOOLS, the three memory tools (qa.py L102, kept "for the learning task"), and eval.py's agent offers the same three, while the box's open tool loop offers those plus get_keyframe, and the web tools when the browser connector is on (bootstrap.py L243-L253, qa.py L289).
Recommend: train and score with the list the box sends, because the chat template renders the tool list into the prompt, owner ruling 5 has the gate score the model served the way the box serves it, and DT-14 asks for the training render to equal the served prompt; the labels keep calling only the memory tools. If Dominick prefers the box to offer the student only TOOLS, that is a change in his qa.py and goes to him as a proposal.
Source of tension: contexts/agent/application/qa.py L102 (TOOLS kept as the student's training set) against contexts/gateway/bootstrap.py L243-L253 and qa.py L289 (the box's agent offers get_keyframe and the web tools), lab/refine/learning/OWNER_RULINGS.md 5 (served the way the box serves it) and lab/refine/research/learning/distillation_tool_calls.md#4-tool-call-traces-train-exactly-the-format-llama-server-will-parse (DT-14).
Answer: ____

## 11. Besides ruling 1's 50 items, how many question groups must a held-out set span? (pass 2)

Now: L4 (change 1) tests at the group level and records n as items and as groups; owner ruling 1 counts items (at least 50); L15 (Locked) still says 20 percent or 30 examples; holding out the L11 fixture's 4 golden groups gives 58 items in 4 groups.
Recommend: at least 6 groups, better 8. An exact two-sided sign-flip test over 4 groups cannot go below p = 0.125 and over 5 below 0.0625, so with the 4 golden groups alone no candidate could ever deploy; 6 groups reach 0.031 only when every group favours the candidate. Hold out the 4 golden groups plus 2 to 4 variant groups, and if training then runs short, add variant groups to the fixture rather than more wordings of the same questions.
Source of tension: lab/refine/learning/OWNER_RULINGS.md 1 (50 items) and L15's floor in examples, against lab/refine/research/learning/eval_noise.md#8-few-question-groups-the-paired-test-and-the-held-out-size-learning-pass-2-2026-10-10 (Durkalski et al. 2003; MacKinnon, Nielsen and Webb 2022; MacKinnon and Webb 2018; the fixture's 16 groups).
Answer: ____

## 12. If the teacher is not serving on SJC pod C in time, what labels the tool-calling fixture? (pass 2)

Now: L2 and L13 label with Qwen3.8-27B on pod A, now SJC account C (Strix Halo, gfx1151), where R29 is bringing vLLM up for gfx1151; until it serves, L13, L16, L17 and the beat's numbers wait.
Recommend: keep the teacher as the plan with a cut-off; if it is not serving by then, train the tool-calling task on the reference traces the L11 fixture already holds in expected.jsonl (the calls the agent's own pre-fetch makes, their results and a reference answer), and say on the slide and the card that it learned from the agent's own traces rather than from the 27B. Either way, use expected.jsonl as the gold L14's agreement check needs.
Source of tension: Context: Learning (the 27B teacher labels the data) and L2's and L13's Done whens, against the Ilan1 and Ilan2 page (the R9700s given up; the teacher brought up on pod C through R29, with llama.cpp on Strix Halo raised as the alternative) and PR #117's body (expected.jsonl holds the expected tool calls, their results, the citations and a reference answer for every example; the teacher never sees it).
Answer: ____
