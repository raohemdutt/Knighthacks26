# Owner rulings for the plumbing track

Questions the passes could not decide, because they are policy, not method. Each shows what the
plan does NOW (the default the pass applied so the loop could continue), the recommendation, the
source of the tension, and a blank for the owner's answer. A filled Answer is an owner rule at top
authority: the next pass applies it and cites it in `REFINE_LOG.md`. Passes never edit this file;
the merge pass adds questions from the pass entries' "Rulings raised" sections. The format is
`lab/refine/TEMPLATES.md`, section "The rulings file".

## 1. During the long sessions Ilan has asked for (both pods relaunched for 40 hours once the hub is back), does the controller ping the pods every 10 minutes whatever they are doing, or only while a job runs or the teacher is in use? (pass 1)

Now: R9's Done when stands, a ping every 10 minutes so the culler never fires, which Design Doc section 13 also asks for; rubric T3 cannot pass under it, because it wants 0 credits per hour when nothing works.
Recommend: ping unconditionally during the 40-hour sessions and record credits per hour in the ledger. Both pods for 40 hours cost 2 x 2,400 x 4 = 19,200 credits of about 118,900 left (M5, M15), a culled teacher costs a relaunch of about 5 minutes plus a hand step, and the live-teach demo beat needs the teacher warm. The merge pass would then narrow T3's keep-alive clause to "while the session plan says the pod is up", and a pod-side keep-alive (IDL-12) would keep the pods alive through a front-door outage like today's.
Source of tension: Atlas Design Doc section 13 ("a keep-alive request every 10 min against the idle culler") and R9's Done when, against lab/refine/plumbing/RUBRIC.md, T3 ("keep-alive runs only while something is working") and its known open item 6.
Answer: ____

## 2. What pull and push speed counts as acceptable for test T6? (pass 1)

Now: no bound; passes record MB/s and never mark T6 passed (rubric, known open item 10).
Recommend: a package of a few GB pulled within 10 minutes (about 5 MB/s for 3 GB) and a dataset push of the same size within 10 minutes, because a trainer run takes tens of minutes (Design Doc section 8) and shipping should not take longer than training.
Source of tension: lab/refine/plumbing/RUBRIC.md, T6 ("both meet the bound set by ruling (open item 10)") against lab/refine/plumbing/OWNER_RULINGS.md, which holds no ruling.
Answer: ____

## 3. Do R13 to R18 join the plumbing walk list? (pass 1)

Now: they are walked for findings only and never edited (refine-track skill, section 3), because the rubric's walk list names R6, R9 and R10; their findings in this entry wait.
Recommend: yes, all six. Each was created at 7:00 AM ET from the track run split to carry one plumbing test into a measurement (R13 tests 1, 3 and 8; R14 test 6; R15 test 2; R16 test 4; R17 test 3; R18 test 5), and three of them would produce a wrong or unusable result as written (R17's credits per hour from balances, R16's shape scan, R13's 70-minute window), which pass 2 can fix once they are on the list.
Source of tension: lab/refine/plumbing/RUBRIC.md, "The walk list" (R6, R9, R10 and the credit rules), against the R13 to R18 task pages, each created "from the track run split; why: plumbing rubric test N".
Answer: ____

## 4. Should the rubric's Jupyter Server citations be re-pinned from 2.21.1 to 2.19.0? (pass 1)

Now: the rubric cites jupyter_server v2.21.1 (serverapp.py body limits, api.yaml terminals) while the research notes cite v2.19.0, the version pod A reports (measurements M6).
Recommend: yes; the rubric's own known open item 9 says to correct a pinned citation once the pod's version is known, and only the merge pass may edit the rubric, with a filled ruling.
Source of tension: lab/refine/plumbing/RUBRIC.md, "Primary sources" item 3 and known open item 9, against lab/refine/research/plumbing/measurements-2026-10-10.md M6.
Answer: ____

## 5. May the measurement tasks write their results into a RUBRIC.md directly, as their Done whens say, or does each such result come to the merge pass as a ruling? (merge 1)

Now: PASS_RULES.md stands: only the merge pass edits a RUBRIC.md, and only to carry out a filled ruling, so the clauses "written into the rubric", "confirmed or corrected" in a rubric and "rubric test N cites it" on T9, T10, T11, L14, L20, R14, R15, R16, R18, R20, R21 and R25 cannot be met as written, and none of those tasks is on a walk list yet (telemetry ruling 2, learning ruling 4, plumbing ruling 3, kernels ruling 1).
Recommend: The task records its result in the ledger or the metrics file and on its own page, and the rubric edit is raised as a ruling in its track's file through the track's next pass, which the merge pass carries out; the tracks' next passes reword those clauses once the walk-list rulings admit the tasks. Nothing else changes: the rubrics already say their bounds and citations are corrected through a ruling.
Source of tension: The Done whens of T9, T10, T11, L14, L20, R14, R15, R16, R18, R20, R21 and R25 (created Sat 7:00 AM ET from the track run split, each citing a rubric test), against lab/refine/PASS_RULES.md, "The merge pass" (the RUBRIC.md of any track, only to carry out a filled ruling) and "What a pass may never change" (the rubric).
Answer: ____
