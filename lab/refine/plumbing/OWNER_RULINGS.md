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
Answer: Unconditional, a ping every 10 minutes for the whole session, as R9's Done when says; I accept the credit cost and stop the pods by hand when they are not needed. (owner, 2026-10-10)

## 2. What pull and push speed counts as acceptable for test T6? (pass 1)

Now: no bound; passes record MB/s and never mark T6 passed (rubric, known open item 10).
Recommend: a package of a few GB pulled within 10 minutes (about 5 MB/s for 3 GB) and a dataset push of the same size within 10 minutes, because a trainer run takes tens of minutes (Design Doc section 8) and shipping should not take longer than training.
Source of tension: lab/refine/plumbing/RUBRIC.md, T6 ("both meet the bound set by ruling (open item 10)") against lab/refine/plumbing/OWNER_RULINGS.md, which holds no ruling.
Answer: No bar yet. Passes record MB per second and never mark T6 passed until I set a bar from the first measured run. (owner, 2026-10-10)

## 3. Do R13 to R18 join the plumbing walk list? (pass 1)

Now: they are walked for findings only and never edited (refine-track skill, section 3), because the rubric's walk list names R6, R9 and R10; their findings in this entry wait.
Recommend: yes, all six. Each was created at 7:00 AM ET from the track run split to carry one plumbing test into a measurement (R13 tests 1, 3 and 8; R14 test 6; R15 test 2; R16 test 4; R17 test 3; R18 test 5), and three of them would produce a wrong or unusable result as written (R17's credits per hour from balances, R16's shape scan, R13's 70-minute window), which pass 2 can fix once they are on the list.
Source of tension: lab/refine/plumbing/RUBRIC.md, "The walk list" (R6, R9, R10 and the credit rules), against the R13 to R18 task pages, each created "from the track run split; why: plumbing rubric test N".
Answer: Yes, the walk list is every plumbing task in my lane, R6, R9 to R18, under the same rules. (owner, 2026-10-10)

## 4. Should the rubric's Jupyter Server citations be re-pinned from 2.21.1 to 2.19.0? (pass 1)

Now: the rubric cites jupyter_server v2.21.1 (serverapp.py body limits, api.yaml terminals) while the research notes cite v2.19.0, the version pod A reports (measurements M6).
Recommend: yes; the rubric's own known open item 9 says to correct a pinned citation once the pod's version is known, and only the merge pass may edit the rubric, with a filled ruling.
Source of tension: lab/refine/plumbing/RUBRIC.md, "Primary sources" item 3 and known open item 9, against lab/refine/research/plumbing/measurements-2026-10-10.md M6.
Answer: Re-pin to jupyter_server 2.19.0, the version the pods run; the merge pass makes the rubric edit. (owner, 2026-10-10)

## 5. May the measurement tasks write their results into a RUBRIC.md directly, as their Done whens say, or does each such result come to the merge pass as a ruling? (merge 1)

Now: PASS_RULES.md stands: only the merge pass edits a RUBRIC.md, and only to carry out a filled ruling, so the clauses "written into the rubric", "confirmed or corrected" in a rubric and "rubric test N cites it" on T9, T10, T11, L14, L20, R14, R15, R16, R18, R20, R21 and R25 cannot be met as written, and none of those tasks is on a walk list yet (telemetry ruling 2, learning ruling 4, plumbing ruling 3, kernels ruling 1).
Recommend: The task records its result in the ledger or the metrics file and on its own page, and the rubric edit is raised as a ruling in its track's file through the track's next pass, which the merge pass carries out; the tracks' next passes reword those clauses once the walk-list rulings admit the tasks. Nothing else changes: the rubrics already say their bounds and citations are corrected through a ruling.
Source of tension: The Done whens of T9, T10, T11, L14, L20, R14, R15, R16, R18, R20, R21 and R25 (created Sat 7:00 AM ET from the track run split, each citing a rubric test), against lab/refine/PASS_RULES.md, "The merge pass" (the RUBRIC.md of any track, only to carry out a filled ruling) and "What a pass may never change" (the rubric).
Answer: No task writes a rubric directly; results reach a rubric only through the merge pass, as PASS_RULES.md says. (owner, 2026-10-10)

## 6. Do test T3 and the metric pod.credits_per_hour still bind on a hub whose quota is switched off, and what does the track record there instead? (pass 2)

Now: T3 and the metric set stand as written (credits per hour per account, projected spend against the balance), and ruling 1 already sets the keep-alive (a ping every 10 minutes for the whole session, credit cost accepted). On SJC both accounts read balance 0, unlimited true, enabled false (M19), so credits per hour are 0 by the hub's own setting and R13 and R17 would measure nothing there (R17's builder plans "unmetered: usage N minutes today").
Recommend: Keep T3's credit tracking for a metered hub (DUB, if it returns), and on an unmetered hub judge T3 by running minutes per account from each server's started time, under its own metric name (pod.running_minutes) with the hub and the account in the conditions, so a pod left up for nothing still shows. The merge pass then edits T3 and the metric set, and narrows T3's keep-alive clause to ruling 1's answer.
Source of tension: lab/refine/plumbing/RUBRIC.md, T3 and "The metric set" (credits per hour; 4 credits per minute and about 60,000 per account), against lab/refine/research/plumbing/measurements-2026-10-10.md M19 (SJC quota disabled, 2400-minute sessions), Ilan's Team Comms FYI of Sat 2:47 PM ET (nothing is metered on SJC) and lab/refine/plumbing/OWNER_RULINGS.md 1.
Answer: On an unmetered hub T3 is judged by running minutes per account from each server's started time, as pod.running_minutes with the hub and the account in the conditions; credits per hour stays for a metered hub. The merge pass edits T3 and the metric set and narrows T3's keep-alive clause to ruling 1's answer. (owner, 2026-10-10)

## 7. Should the walk list gain a scenario for a whole hub site lost for hours, with the work moving to another site's accounts? (pass 2)

Now: The fourteen scenarios cover a hub restart while a pod serves (P13) but not a site down for hours. Today's DUB outage (502 from about 15:20Z to at least 19:04Z, which Cloudflare puts on the origin, M21) moved the pods to SJC with empty home volumes, new tokens, a new address and no metering, and no scenario checks what that move needs: a hub address per account (O54), the Jetson's egress allowlist (jetson/egress.py L50 allows only dub.aupcloud.io), the readiness check (lab/machine/ready.py asks DUB only, M23) and a timed cold bring-up from scripts (T8).
Recommend: Add P15, "the hub site is down for hours and the pods move to another site's accounts", walked against T2 (work lost), T5 (the box's path and the egress allowlist), T7 (status per hub) and T8 (a scripted cold bring-up, timed). The merge pass writes it into the rubric's walk list, and convergence then needs all fifteen walked.
Source of tension: lab/refine/plumbing/RUBRIC.md, "The walk list" (P1 to P14; a pass converges only after all fourteen are walked), against measurements-2026-10-10.md M16, M21 and M23, Ilan's FYI of Sat 2:47 PM ET and the O54 and R29 task pages.
Answer: Add P15, the hub site down for hours with the pods moving to another site's accounts, walked against T2, T5, T7 and T8. Convergence needs all fifteen walked. (owner, 2026-10-10)

## 8. Ruling 4 re-pins the rubric's Jupyter Server citations to 2.19.0 as "the version the pods run", but the pods that run now, the SJC Strix Halo pods, report 2.20.0: does the re-pin to 2.19.0 stand? (pass 2)

Now: Ruling 4's answer stands and the merge pass re-pins the rubric to jupyter_server 2.19.0, the version DUB's pod A reported (M6). The Strix Halo pods launched through SJC's web form report 2.20.0 (M24), the first REST-launched SJC pod reported 2.21.1 (M20), the rubric pins 2.21.1, and DUB has answered 502 since about 11:20 AM ET.
Recommend: Pin to the version of the pods that run and say which: 2.20.0 while the lab is the SJC Strix Halo pods, with each research finding citing the version of the pod it was read on (2.19.0 for DUB's records, 2.20.0 for the Strix Halo pods), as hub-api-lifecycle.md section 8 now does; the cited lines (the body limit, the chunked save, the files handler) are re-opened at 2.20.0 before a change rests on them. That keeps the answer's own reason, the version the pods run.
Source of tension: lab/refine/plumbing/OWNER_RULINGS.md 4 (owner, 2026-10-10: re-pin to 2.19.0, "the version the pods run"), against lab/refine/research/plumbing/measurements-2026-10-10.md M20 and M24 (2.21.1 on the first SJC pod, 2.20.0 on the Strix Halo pods) and lab/refine/plumbing/RUBRIC.md, "Primary sources" item 3 (pins 2.21.1).
Answer: Pin to the version the running pods report, 2.20.0 while the lab is the SJC Strix Halo pods, with each finding citing the version of the pod it was read on (2.19.0 for DUB's records). The cited lines are re-opened at 2.20.0 before any change rests on them. (owner, 2026-10-10)

## 9. How does the plumbing rubric change now that the lab is the two SJC Strix Halo pods? (pass 2)

Now: The rubric stands as written for R9700 pods: 4 credits per R9700-minute and about 60,000 per account, a 32 GB card, a launch through the REST API with gpu_selection r9700, the auto accelerator refused "rather than the R9700", a vLLM engine start of about 270 s, and "the Strix Halo node" among its dead paths. Passes read pod A and pod B as accounts C and D on the SJC Strix Halo pods by Ilan's decision, mark the R9700 readings old, other chip, and edit no rubric.
Recommend: The merge pass rewrites those clauses for the pods in use: the strix-halo accelerator on SJC (unmetered, so T3 per candidate (a)), the web spawn form as the launch path where the REST launch is ignored (O55), the pods' 64 GB shared memory in place of the 32 GB card, an engine-start figure to be measured on gfx1151 (T8), and the R9700 readings kept in a dated note as another chip; and it says that "the Strix Halo node" among the dead paths means a Strix Halo machine of our own as an Atlas node, not the AUP-hosted pods Ilan chose, or removes it if Ilan meant both.
Source of tension: lab/refine/plumbing/RUBRIC.md (T3, T8, "The metric set", "What a pass may NOT change": the launch keys and the dead paths), against Ilan's decision on the Ilan1 and Ilan2 dispatcher page (line of Sat 4:27 PM ET by that page's clock), the same page's line of Sat 3:46 PM ET (SJC ignores the REST launch; Strix Halo pods only) and measurements-2026-10-10.md M24.
Answer: Rewrite the R9700 clauses as recommended: the strix-halo accelerator on SJC, unmetered, the web spawn form as the launch path, 64 GB shared memory, an engine-start figure measured on gfx1151 by T8, the R9700 readings in a dated note as another chip. The Strix Halo node among the dead paths means a Strix Halo machine of our own as an Atlas node; the AUP-hosted pods are the lab. (owner, 2026-10-10)
