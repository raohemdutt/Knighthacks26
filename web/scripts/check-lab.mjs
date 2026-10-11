#!/usr/bin/env node
// Self-check for the Learning lab's pure rules in lib/lab.ts: which run is training now (isLiveRun, isLive) and which
// measured trainer rows count as our hardware (onOurHardware). The web app has no test runner, so this compiles the
// file with the TypeScript that is already a dev dependency and runs it in node, like scripts/check-guide.mjs.
//
//   node scripts/check-lab.mjs        (npm run check-lab)

import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import ts from "typescript";

const web = join(dirname(fileURLToPath(import.meta.url)), "..");
const dir = mkdtempSync(join(tmpdir(), "atlas-lab-"));
async function load(name) {
  const js = ts.transpileModule(readFileSync(join(web, "lib", `${name}.ts`), "utf8"), { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
  writeFileSync(join(dir, `${name}.mjs`), js);
  return import(pathToFileURL(join(dir, `${name}.mjs`)).href);
}

let failed = 0;
let passed = 0;
const is = (name, got, want) => {
  const ok = got === want;
  ok ? passed++ : failed++;
  if (!ok) console.log(`  FAIL  ${name}: got ${JSON.stringify(got)}, want ${JSON.stringify(want)}`);
};

try {
  const L = await load("lab");
  const ended = "2026-10-11T09:14:00Z";

  // ---- training now: the gateway's run_status decides (PR 421)
  is("run_status running, training", L.isLiveRun({ status: "training", run_status: "running", ended }), true);
  is("run_status finished", L.isLiveRun({ status: "training", run_status: "finished", ended }), false);
  is("run_status failed", L.isLiveRun({ status: "failed", run_status: "failed", ended }), false);
  // the bug on the box: final2-t12, cleared by the gate at 5:14 AM, the gateway still saying running
  is("gated with run_status running", L.isLiveRun({ status: "gated", run_status: "running", ended }), false);
  is("offered", L.isLiveRun({ status: "offered", run_status: "running", ended: null }), false);
  is("finished status", L.isLiveRun({ status: "finished", run_status: null, ended: null }), false);
  is("deployed", L.isLiveRun({ status: "deployed", run_status: "running", ended }), false);
  is("not deployed", L.isLiveRun({ status: "not deployed" }), false);
  is("rolled back", L.isLiveRun({ status: "rolled back" }), false);
  is("sample run", L.isLiveRun({ sample: true, status: "training", run_status: "running" }), false);

  // ---- an older gateway, no run_status: exactly "running" and no ended time
  is("old gateway, running, no end", L.isLiveRun({ status: "running", ended: null }), true);
  is("old gateway, running, ended", L.isLiveRun({ status: "running", ended }), false);
  is("old gateway, training", L.isLiveRun({ status: "training", ended: null }), false);
  is("old gateway, gated", L.isLiveRun({ status: "gated" }), false);

  // ---- a story: its ended time is marks.end
  const story = (status, run_status, end) => ({ sample: false, status, run_status, marks: { end } });
  is("story running", L.isLive(story("training", "running", ended)), true);
  is("story gated", L.isLive(story("gated", "running", ended)), false);
  is("story finished", L.isLive(story("training", "finished", ended)), false);
  is("story old gateway ended", L.isLive(story("running", undefined, ended)), false);

  // ---- measured on our hardware: the Strix Halo pods or the box, never the R9700s
  const chip = (key, role, current) => ({ key, name: key, where: null, current, role });
  is("strix halo row", L.onOurHardware({ chip: chip("strix-halo", "current", true), other_chip: false }), true);
  is("jetson row", L.onOurHardware({ chip: chip("jetson", "current", true), other_chip: false }), true);
  is("r9700 row", L.onOurHardware({ chip: chip("r9700", "earlier", false), other_chip: true }), false);
  is("reference row", L.onOurHardware({ chip: chip("nvidia-a100", "reference", false), other_chip: true }), false);
  is("unnamed row", L.onOurHardware({ chip: chip("other", "unnamed", false), other_chip: true }), false);
  is("row with no chip", L.onOurHardware({ other_chip: false }), false);

  // ---- the Circle's Shared brain card (G6): two pods, one brain (lib/sharedBrain.ts)
  const S = await load("sharedBrain");
  const member = (name, role, online, watts) => ({ name, role, node: role === "head" ? "pod-a" : "pod-b", layers: 24, online, watts_now: watts, watts_idle: 10.27, vram_gb: online ? 34.5 : null });
  const brain = (configured, a, b, serving = { tok_s: 10, ttft_s: 0.29, requests_running: 0 }) => ({
    configured,
    model: { name: "Qwen3.5-122B-A10B", params_b: 122, active_b: 10, quant: "int4", served: ["teacher", "group-brain"] },
    members: [member("pod C", "head", a, a ? 61.4 : null), member("pod D", "half", b, b ? 48 : null)],
    link: { kind: "tunnel", rtt_ms: 0.36, mbps: 100 },
    serving,
    updated: "2026-10-11T12:00:00Z",
    reason: null,
  });
  is("both online", S.pairState(brain(true, true, true)), "both");
  is("one online", S.pairState(brain(true, true, false)), "one");
  is("both offline is no friends online", S.pairState(brain(true, false, false)), "none");
  is("not configured is no friends online", S.pairState(brain(false, true, true)), "none");
  is("no answer is no friends online", S.pairState(null), "none");
  is("model line", S.modelLine(brain(true, true, true)), "Qwen3.5-122B, split 24 and 24 layers across two friends");
  is("link line", S.linkLine(brain(true, true, true)), "0.36 ms over a tunnel");
  is("serving line", S.servingLine(brain(true, true, true)), "10.0 tokens a second, first token in 0.29 s");
  is("serving line busy", S.servingLine(brain(true, true, true, { tok_s: 9.5, ttft_s: 0.3, requests_running: 2 })), "9.5 tokens a second, first token in 0.30 s, 2 answering now");
  is("not serving", S.servingLine(brain(true, false, false, { tok_s: null, ttft_s: null, requests_running: null })), "not serving right now");
  is("watts", S.wattsText(61.4), "61 W");
  is("no watts", S.wattsText(null), "--");

  // ---- what an answer on the box cost (lib/labDeploy.ts energyLine, the Ask panel)
  const D = await load("labDeploy");
  is("cost line", D.energyLine({ joules: 41.26, idle_w: 15.76, seconds: 2.4, sampler: "tegrastats", samples: 7 }), "this answer cost about 41 J on the box");
  is("cost line small", D.energyLine({ joules: 3.24, idle_w: 15.76, seconds: 0.2, sampler: "tegrastats", samples: 2 }), "this answer cost about 3.2 J on the box");
  is("sampler off", D.energyLine(null), "sampler off");
  is("older gateway", D.energyLine(undefined), null);

  // ---- the Ask panel's comparison (ask-compare): the box and the network brain side by side
  const look = (min) => ({ name: "episodes_between", arguments: { start: "2026-10-11T14:00:00-04:00", end: new Date(Date.parse("2026-10-11T14:00:00-04:00") + min * 60000).toISOString() } });
  const seen = { name: "last_seen", arguments: { label: "keys" } };
  const search = { name: "search_memory", arguments: { q: "keys" } };
  is("both looked back", D.verdictLine({ answer: "a", calls: [look(25)] }, { answer: "b", calls: [look(25)] }), "both looked back over the last 25 minutes");
  is("tools differ", D.verdictLine({ answer: "a", calls: [search] }, { answer: "b", calls: [seen] }), "the box searched memory, the brain checked last seen");
  is("no tool on either", D.verdictLine({ answer: "a", calls: [] }, { answer: "b", calls: [] }), "both answered without a tool");
  is("a side that failed has no verdict", D.verdictLine({ answer: "a", calls: [] }, { error: "no answer in 60 s" }), null);
  is("no brain, no verdict", D.verdictLine({ answer: "a", calls: [] }, null), null);
  is("hours", D.toolWords([look(180)]), "looked back over the last 3 hours");
  is("tool line", D.toolLine([seen]), 'last_seen(label="keys")');
  is("no tool line", D.toolLine([]), "no tool");
  is("cost short", D.costShort({ seconds: 2.43, energy: { joules: 41.3 } }, "sampler off"), "about 41 J in 2.4 s");
  is("cost short brain", D.costShort({ seconds: 9.5, energy: { joules: 1234.5, watts_avg: 130, pods: {} } }, "watts not read"), "about 1,235 J in 9.5 s");
  is("cost short off", D.costShort({ seconds: 2.4, energy: null }, "sampler off"), "sampler off · 2.4 s");
  const oldAnswer = { question: "q", serving: "atlas-base", text: "t", calls: [seen], expected: null, seconds: 1, prompt: "eval", energy: null };
  is("an older gateway's answer is the box side", D.boxSide(oldAnswer).answer, "t");

  // ---- three tabs, as a grouping of the five views (lib/labTabs.ts): nothing is dropped, old links land
  const B = await load("labBench");
  const T = await load("labTabs");
  is("the five views are still the lab's five", B.LAB_TABS.map((x) => `${x.key}=${x.title}`).join("|"), "run=Training run|compare=Alone vs with a friend|speed=Speed and energy|routing=Routing|experiments=Experiments");
  is("three tabs", T.LAB_GROUPS.map((g) => g.title).join("|"), "The model|Speed and energy|Together");
  is("every view is on exactly one tab", T.LAB_GROUPS.flatMap((g) => g.views).sort().join(","), B.LAB_TABS.map((x) => x.key).sort().join(","));
  is("the order inside each tab", T.LAB_GROUPS.map((g) => g.views.join("+")).join("|"), "run+experiments|speed+routing|compare");
  is("an old ?tab= value opens its own view", ["run", "experiments", "speed", "routing", "compare"].map(T.labViewOf).join(","), "run,experiments,speed,routing,compare");
  is("and its tab", ["run", "experiments", "speed", "routing", "compare"].map((v) => T.labGroupOf(v)).join(","), "model,model,speed,speed,together");
  is("the new tab names are values too", [T.labViewOf("model"), T.labViewOf("together"), T.labViewOf("nope"), T.labViewOf(null)].join(","), "run,compare,,");
  is("the address keeps the old values", ["run", "experiments", "speed", "routing", "compare"].map(T.labTabParam).join(","), ",experiments,speed,routing,compare");
  const page = readFileSync(join(web, "app", "lab", "page.tsx"), "utf8");
  for (const view of ["run", "experiments", "speed", "routing", "compare"]) is(`the ${view} view has its anchor`, page.includes(`<section id="${view}"`), true);
  for (const part of ["<LabRun ", "<Experiments />", "<SpeedEnergy ", "<Routing ", "<Compare", "<LoopView ", 'title="No runs yet; start the loop"']) is(`the page still draws ${part}`, page.includes(part), true);
  is("the loop is above the Training run view", page.indexOf("<LoopView ") < page.indexOf('<section id="run"'), true);
  is("the swap buttons are where the lab put them, in LabRun", readFileSync(join(web, "components", "lab", "LabRun.tsx"), "utf8").includes("<DeployButton"), true);
  is("looking again never restarts a request in flight", /setInterval/.test(page) || !page.includes("usePoll([list, story], watching"), false);

  // ---- stand-in gates, honest statuses and the run the tab opens on (lib/lab.ts)
  const fake = { before: { answer: "fake miss" }, after: { answer: "fake answer" } };
  is("a stand-in gate, by its flag", L.gateIsStandIn({ stand_in: true }), true);
  is("a stand-in gate, by the box's explanation", L.gateIsStandIn({ rehearsal: "the test did not ask the model" }), true);
  is("a stand-in gate, by a note that says so", L.gateIsStandIn({ note: "a stand-in test: not proof" }), true);
  is("a stand-in gate, by its literal answers", L.gateIsStandIn({ items: [fake] }), true);
  is("a real gate is not a stand-in", L.gateIsStandIn({ rehearsal: null, verdict: "Better by more than chance", items: [{ before: { answer: "x" }, after: { answer: "y" } }] }), false);
  const loopRow = { run_id: "live-1011-081402-duo", status: "not deployed", run_status: "finished", started: "2026-10-11T12:14:12Z", ended: "2026-10-11T12:17:28Z", model_version: "atlas-group-desk", tested: { decision: "reject", n_items: 6, right_before: 0, right_after: 6, rehearsal: "fake answer" } };
  const realRow = { run_id: "final2-t12", status: "rolled back", run_status: "finished", started: "2026-10-11T08:55:02Z", ended: "2026-10-11T11:16:22Z", model_version: "atlas-tool-calling-v2-a100-f2r4", tested: { decision: "deploy", n_items: 228, right_before: 95, right_after: 179, rehearsal: null } };
  const oldRow = { run_id: "train-7dd098", status: "rolled back", run_status: "finished", started: "2026-10-11T03:23:12Z", ended: "2026-10-11T06:42:14Z", model_version: "atlas-tool-calling-train-7dd098", tested: { decision: "reject", n_items: 24, right_before: 24, right_after: 24, rehearsal: null } };
  is("the tab opens on the newest run with a real gate", L.defaultRunId([loopRow, realRow, oldRow]), "final2-t12");
  is("with no real gate, the newest run", L.defaultRunId([oldRow, { ...loopRow }].map((r) => ({ ...r, tested: null }))), "live-1011-081402-duo");
  const onBox = { current: "atlas-tool-calling-v2-a100-f2r4", swapRolledBack: null };
  is("its package is on the box now", L.runStatus(realRow, onBox).label, "On the box");
  is("cleared by the gate, the box serves another", L.runStatus(realRow, { current: "atlas-base", swapRolledBack: null }).label, "Cleared by the gate");
  is("rolled back only when the swap itself rolled back", L.runStatus(realRow, { current: "atlas-base", swapRolledBack: "atlas-tool-calling-v2-a100-f2r4" }).label, "Rolled back");
  is("a refused run that was later taken off is not 'Rolled back'", L.runStatus(oldRow, onBox).label, "Not cleared by the gate");
  is("a loop run reads Loop run and its time", L.runStatus(loopRow, onBox).label.startsWith("Loop run · "), true);
  is("low confidence: power yes", L.lowConfidence({ low_confidence: true, unit: "W" }), true);
  is("low confidence: a count never", L.lowConfidence({ low_confidence: true, unit: "steps" }), false);
  is("one line for what the log lacks", L.missingLine(["Training speed", "Energy per training sample"]), "the relay did not post training speed or energy per training sample for this run");
  is("a short model name", L.modelShort("atlas-tool-calling-v2-a100-f2r4"), "v2 f2r4");
  is("a short base name", L.modelShort("atlas-base"), "base");

  // ---- the loop's five stages, from one run's own story (lib/labLoop.ts)
  const P = await load("labLoop");
  const when = (iso) => (iso ? "then" : "");
  const num = (value, unit = null) => ({ label: "", value, unit, source: null, low_confidence: false, reason: null });
  is("no run: five stages, all waiting", P.loopStages(null, when).map((x) => `${x.key}:${x.state}`).join(","), "out:waiting,label:waiting,train:waiting,test:waiting,box:waiting");
  const loopStory = {
    run_id: "v2", status: "gated", run_status: "finished", marks: { end: ended, evaluated: ended },
    dataset: { examples: num(1140), train: num(912), held_out: num(228), approval: { approved: null } },
    training: { steps: num(48), of: num(48), points: [{ loss: 0.42 }], loss_last: num(0.42) },
    eval: { decision: "deploy", n_items: num(228), scores: { current: num(0.417), candidate: num(0.785) }, correct: { current: num(95), candidate: num(179) }, rehearsal: null },
    package: { bytes: num(null) }, deploy: { deployed: null, swap_s: num(null), rolled_back: null },
  };
  const stages = P.loopStages(loopStory, when);
  is("examples out, from the dataset card's number", stages[0].value, "1,140 examples");
  is("the trainer's step, from the training card's number", stages[2].value, "step 48 of 48");
  is("the test, in the gate card's counts", stages[3].value, "95 → 179 of 228");
  is("passed, not installed", `${stages[3].state},${stages[4].state},${stages[4].value}`, "done,waiting,not installed yet");
  is("a stage the run did not report says so", `${stages[1].state},${stages[1].note}`, "none,not reported for this run");
  const standInStages = P.loopStages({ ...loopStory, eval: { ...loopStory.eval, decision: "reject", rehearsal: "fake answer" } }, when, { standIn: true });
  is("a stand-in gate's caption shows no count", `${standInStages[3].value},${standInStages[3].note}`, "stand-in gate,no held-out score");
  const onBoxStages = P.loopStages({ ...loopStory, package: { bytes: num(null), model_version: "v2" } }, when, { onBox: "v2" });
  is("the box caption says when the run's model is on the box now", `${onBoxStages[4].state},${onBoxStages[4].value}`, "done,on the box now");
  // ---- answers in plain words (lib/labWords.ts)
  const W = await load("labWords");
  is("JSON tool call: last_seen", W.answerWords('{"tool_calls": [{"name": "last_seen", "arguments": {"label": "backpack"}}]}'), "checked where backpack was last seen");
  is("markup tool call: last_seen", W.answerWords("<tool_call>\n<function=last_seen>\n<parameter=label>\nkeys\n</parameter>\n</function>\n</tool_call>"), "checked where keys was last seen");
  is("search_memory", W.answerWords('{"tool_calls": [{"name": "search_memory", "arguments": {"q": "mug"}}]}'), 'searched memory for "mug"');
  is("another tool, its name in words", W.answerWords('{"tool_calls": [{"name": "set_reminder", "arguments": {}}]}'), "set reminder");
  is("episodes_between says looked back", W.answerWords('{"tool_calls": [{"name": "episodes_between", "arguments": {"start": "2026-10-11T00:44:00-04:00", "end": "2026-10-11T01:09:00-04:00"}}]}').startsWith("looked back from "), true);
  is("a plain sentence stays a sentence", W.answerWords("  The keys are on the desk. "), "The keys are on the desk.");
  // ---- the live power graph (lib/livePower.ts): real readings only, gaps where there are none
  const LP = await load("livePower");
  const lpMember = (node, online, w) => ({ name: node, node, online, watts_now: w, watts_idle: 10.27 });
  const route = (a, b) => ({ configured: true, members: [lpMember("pod-a", ...a), lpMember("pod-b", ...b)] });
  const r0 = LP.readingFrom(route([true, 10], [true, 12]), 0);
  const r1 = LP.readingFrom(route([true, 20], [true, 12]), 2000);
  is("a reading takes each pod's watts", `${r0.w["pod-a"]},${r0.w["pod-b"]}`, "10,12");
  is("an offline pod is a gap, not its last watts", LP.readingFrom(route([false, 30], [true, 12]), 0).w["pod-a"], null);
  is("a failed poll is a gap for both pods", LP.metersOffline(LP.readingFrom(null, 0)), true);
  is("a route not configured is offline", LP.metersOffline(LP.readingFrom({ configured: false, members: [] }, 0)), true);
  is("one pod up is not offline", LP.metersOffline(LP.readingFrom(route([false, null], [true, 12]), 0)), false);
  is("trapezoid, both pods: (10+20)/2*2 + 12*2", LP.trapezoidJoules(r0, r1), 54);
  is("a missed stretch adds nothing", LP.trapezoidJoules(r0, { ...r1, t: 10_000 }), 0);
  is("a pod without a reading at one end adds nothing", LP.trapezoidJoules(r0, LP.readingFrom(route([false, null], [true, 12]), 2000)), 24);
  let buf = [];
  for (let i = 0; i <= 70; i++) buf = LP.appendReading(buf, LP.readingFrom(route([true, 10], [true, 10]), i * 2000));
  is("the buffer keeps two minutes and a poll of margin", buf[0].t >= 140_000 - 125_000, true);
  is("the buffer starts empty and fills, nothing made up", LP.appendReading([], r0).length, 1);
  const line = LP.podLine([r0, r1, { t: 20_000, w: { "pod-a": 15 } }], "pod-a");
  is("the line breaks across a missed stretch", line.map((x) => x[1]).join(","), "10,20,-,15");
  is("kJ text", LP.kjText(54), "0.05 kJ");
  // ---- the live chat beside the power graph (lib/liveChat.ts): SSE parsing and the footer, never a made-up joule
  const LC = await load("liveChat");
  const NL = String.fromCharCode(10);
  const CR = String.fromCharCode(13);
  const lines = (...xs) => xs.join(NL);
  const two = LC.parseSse(lines("event: token", 'data: {"text": "Hel"}', "", "event: token", 'data: {"text": "lo"}', "", "event: do"));
  is("sse: two whole events", two.events.length, 2);
  is("sse: token text", two.events[1].data.text, "lo");
  is("sse: the partial block waits", two.rest, "event: do");
  is("sse: CRLF blocks", LC.parseSse(["event: error", 'data: {"error": "x"}', "", ""].join(CR + NL)).events[0].data.error, "x");
  is("sse: not JSON is skipped", LC.parseSse(lines("event: token", "data: nope", "", "")).events.length, 0);
  const done = { answer: "a", target: "pair", serving: null, tokens: 42, seconds: 3.04, tok_s: 18.2, ttft_s: 0.4, calls: [], energy: { joules: 512.6 } };
  is("footer", LC.footerText(done), "42 tokens in 3.0 s, 15.9 tok/s, about 513 joules");
  is("footer, meters offline", LC.footerText({ ...done, energy: null, tok_s: null, tokens: 1 }), "1 token in 3.0 s, 0.4 tok/s, meters offline");
  // the box's live reading: tokens 51, seconds 12.11, tok_s 28453.0 from the route; the footer ignores that tok_s
  is("footer ignores the event's tok_s", LC.footerText({ ...done, tokens: 51, seconds: 12.11, ttft_s: null, tok_s: 28453.0 }), "51 tokens in 12.1 s, 4.2 tok/s, about 513 joules");
  is("rate: ttft not smaller than seconds uses seconds", LC.rateOf({ tokens: 10, seconds: 2, ttft_s: 2 }), 5);
  is("rate: no tokens", LC.rateOf({ tokens: 0, seconds: 2, ttft_s: null }), null);
  is("keep the last three", LC.keepLast([1, 2, 3, 4]).join(","), "2,3,4");
  const call = (name) => ({ name, arguments: {} });
  const boxD = { ...done, target: "box", calls: [call("last_seen")], energy: { joules: 40 } };
  const pairD = { ...done, calls: [call("last_seen")], energy: { joules: 280 } };
  is("compare: same tool, ratio", LC.compareText(boxD, pairD), "same tool (last_seen), the brain used 7x the joules");
  is("compare: different tools", LC.compareText({ ...boxD, calls: [] }, pairD), "different tools: the box used none, the brain used last_seen, the brain used 7x the joules");
  is("compare: meters offline", LC.compareText(boxD, { ...pairD, energy: null }), "same tool (last_seen), joules not compared, meters offline on the pods");
  is("compare: a fractional ratio", LC.compareText({ ...boxD, energy: { joules: 100 } }, pairD), "same tool (last_seen), the brain used 2.8x the joules");
  is("both asks two", LC.targetsOf("both").join(","), "box,pair");
  // ---- the "Train for a minute" button (lib/trainBurst.ts): real steps and joules, or said missing
  const TB = await load("trainBurst");
  is("burst: the default choice is both pods", TB.BURST_CHOICES[0].pods.join(","), "C,D");
  is("burst: button text, both pods", TB.buttonText(["C", "D"]), "Train for a minute on both pods");
  is("burst: button text, one pod", TB.buttonText(["D"]), "Train for a minute on pod D");
  is("burst: pods in words", TB.podsText(["C", "D"]), "pod C and pod D");
  is("burst: pod C shades the head's line", `${TB.burstNode("C")},${TB.burstNode("D")}`, "pod-a,pod-b");
  const pr = (steps, seconds) => ({ ok: true, steps, seconds, train_s: 62, loss: 0.13, peak_gb: 12.1, error: null });
  const bv = { available: true, reason: null, state: "done", pods: ["C", "D"], seconds_asked: 60, started: null, ended: null,
    started_ms: 0, ended_ms: 88_000, per_pod: { C: pr(1, 73.8), D: pr(1, 71.7) },
    energy: { joules: 15116, watts_avg: 170.8, seconds: 88.5, pods: {} }, message: null };
  is("burst line", TB.burstLine(bv), "training burst: pod C 1 step in 74 s, pod D 1 step in 72 s, about 15,116 joules on the pods (171 W average)");
  is("burst line, meters offline", TB.burstLine({ ...bv, energy: null }).endsWith("meters offline, no joules"), true);
  is("burst line, a pod that failed", TB.burstLine({ ...bv, per_pod: { C: { ...pr(null, null), ok: false, error: "x" }, D: pr(2, 130) } }),
    "training burst: pod C did not train (x), pod D 2 steps in 130 s, about 15,116 joules on the pods (171 W average)");
  is("burst problem: 503 reason", TB.burstProblem({ ...bv, state: "idle", available: false, reason: "missing keys" }, null), "Training is not available from this box: missing keys");
  is("burst problem: none when done", TB.burstProblem(bv, null), null);
  // ---- the "16 people asking" and ramp buttons (lib/loadRun.ts)
  const LR = await load("loadRun");
  is("load: button text", LR.loadButtonText(16), "16 people asking for 30 s");
  is("load: one person", LR.askingText(1), "1 person asking");
  const lv = { available: true, reason: null, state: "done", streams: 16, seconds: 30, started: null, ended: null,
    started_ms: 1000, ended_ms: 31000, requests: 48, errors: 0, tokens: 4512, tok_s_total: 150.4, tok_s_per_stream: 9.4,
    ttft_s: 0.4, energy: { joules: 3000, watts_avg: 100.6, seconds: 30, pods: {} }, j_per_token: 0.665, message: null,
    ramp: null, seconds_per_stage: null, stage: null, stages: [] };
  is("load line", LR.loadLine(lv), "48 requests, 4,512 tokens, 150 tok/s total, watts stayed at 101 W, 0.67 J per token");
  is("load line, meters offline", LR.loadLine({ ...lv, energy: null, j_per_token: null }).includes("meters offline, no watts"), true);
  is("load problem: 503 reason", LR.loadProblem({ ...lv, state: "idle", available: false, reason: "missing keys" }, null), "The pair is not reachable from this box: missing keys");
  const st = (streams, a, b, tok) => ({ streams, requests: 1, tokens: 1, tok_s_total: tok, watts_avg: 100, j_per_token: 0.5, started_ms: a, ended_ms: b });
  const rv = { ...lv, state: "running", ramp: [1, 4, 16, 32], seconds_per_stage: 20, stage: 3, stages: [st(1, 1000, 21000, 10), st(4, 21000, 41000, 38)] };
  is("ramp title", LR.rampTitle(rv), "ramp: 16 of 32 streams");
  is("ramp windows, running", LR.stageWindows(rv, 5000, 60000).map((w) => `${w.label}:${w.start}-${w.end}`).join(","), "1:5000-25000,4:25000-45000,16:45000-60000");
  is("ramp rows", LR.rampRows({ ...rv, state: "done" }).map((r) => `${r.streams}|${r.tokS}|${r.watts}|${r.jpt}`).join(","), "1|10|100 W|0.50,4|38|100 W|0.50");
} finally {
  rmSync(dir, { recursive: true, force: true });
}

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
