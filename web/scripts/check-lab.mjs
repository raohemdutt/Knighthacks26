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

  // ---- the words and the slides (lib/labWords.ts): worked out from what the gateway sends, never typed in
  const W = await load("labWords");
  const B = await load("labBench");
  const when = (iso) => (iso ? "then" : "");
  const num = (value) => ({ label: "", value, unit: null, source: { kind: "event", ref: "learning.run.evaluated e1", ids: [] }, low_confidence: false, reason: null });
  const run = (id, status, tested, more = {}) => ({ run_id: id, task: "tool-calling", status, started: "2026-10-11T03:00:00Z", ended: null, model_version: `m-${id}`, verdict: null, scores: { current: num(tested ? tested.right_before / tested.n_items : null), candidate: num(tested ? tested.right_after / tested.n_items : null), teacher: num(null) }, tested, ...more });
  const test = (decision, b, a, n, rehearsal = null) => ({ decision, n_items: n, right_before: b, right_after: a, rehearsal, source: null });
  const box = { model_version: "atlas-base", previous_version: null, since: "2026-10-11T09:09:11Z", source: null, reason: null };
  // the box's own story on Sunday: a rehearsal on top, the retrain that passed, the first student that was refused
  const runs = [run("duo", "gated", test("deploy", 0, 50, 50, "all 50 questions got the same answer")), run("v2", "gated", test("deploy", 95, 179, 228)), run("v1", "rolled back", test("reject", 24, 24, 24))];
  const hero = W.modelHero(runs, box, when);
  is("the hero is the newest real test, not the rehearsal above it", hero.run.run_id, "v2");
  is("its big line is the counts", JSON.stringify(hero.hero.big), JSON.stringify([{ n: 95 }, "→", { n: 179 }, "of 228"]));
  is("and what the test said", hero.hero.line, "Passed the test. Waiting to go on the box.");
  is("the test said no once and yes once", hero.hero.tiles[0].value, "1 passed, 1 refused");
  is("a refusal with no gain says so", W.modelHero([runs[2]], box, when).hero.line, "Refused: no better than before. Installed, then undone.");
  is("only rehearsals: nothing is claimed", W.modelHero([runs[0]], box, when).hero.big.length, 0);
  is("a row says how a run ended", W.runRow(runs[1]).chip, "Passed · 95 → 179 of 228");
  is("a trained run with no test is not training now", W.runRow(run("x", "training", null, { run_status: "finished" })).chip, "Trained, not tested");
  is("a fifth", W.shareWords(1.88 / 8.82), "about a fifth of");
  is("the lab's note without its function names", W.plainConfidence("Each point is the median of 3 repeats, all from one session (bench-1). Every point is marked low confidence: 3 spawn(s), under the 5 that verdict() needs to call a difference."), "Each point is the median of 3 repeats. Every point is marked low confidence: 3 fresh starts; the lab wants 5 before it calls a difference real.");
  is("a reason without its event names", W.plainReason("the log has no learning.run.evaluated event for this run"), "The box's log has no test result for this run");
  is("old tab links land where their content went", [null, "run", "experiments", "routing", "speed", "compare", "together"].map(B.labTabOf).join(","), "run,run,run,speed,speed,compare,compare");
  // the speed slide: two sweeps of the same pod, read as before and now
  const pt = (x, value, unit) => ({ x, value, unit, label: "", source: { kind: "file", ref: `lab/refine/metrics.jsonl:${x}`, ids: [] }, kind: "measured", formula: null, low_confidence: true, reason: null });
  const sweep = (id, ts, tok, j, w) => ({ id, ts, node: "pod-a", family: "teacher_batching", chip: { key: "strix-halo", name: "Radeon 8060S", where: null, current: true, role: "current" }, other_chip: false, batches: [1, 4, 16], build: "vllm", session: id, model: "m", points: 9, low_confidence: 9, confidence_note: "", comparisons: [],
    metrics: { decode_tok_s: [pt(1, tok, "tok/s"), pt(4, tok * 0.8, "tok/s"), pt(16, tok * 0.3, "tok/s")], decode_tok_s_total: [pt(1, tok, "tok/s"), pt(4, tok * 3, "tok/s"), pt(16, tok * 4, "tok/s")], joules_per_token: [pt(1, j, "J/token"), pt(4, j / 3, "J/token"), pt(16, j / 4.7, "J/token")], watts_avg: [pt(1, w, "W"), pt(4, w, "W"), pt(16, w, "W")] } });
  const two = { sweeps: [{ ...sweep("short", "2026-10-11T07:00:00Z", 12, 9, 98), batches: [1, 4] }, sweep("new", "2026-10-11T05:21:22Z", 12.3264, 8.8161, 96.7243), sweep("old", "2026-10-11T00:23:20Z", 2.7021, 36.6587, 96.0046)], power: { nodes: [{ node: "pod-a", idle: pt(0, 10.44, "W"), serving: null }], absent: [], other_chip: [], other_chips: [] } };
  is("the tab opens on the newest full curve, not the newest two points", W.headlineSweep(two).id, "new");
  const speed = W.speedHero(two, when);
  is("the speed-up is the two sweeps' own ratio", JSON.stringify(speed.big), JSON.stringify([{ n: 4.6, digits: 1 }, "× faster"]));
  is("on the same power", speed.line, "on the same 96 W: 2.7 → 12.3 tokens a second.");
  is("one sweep alone has no before: the slide is the batching saving", W.speedHero({ ...two, sweeps: [two.sweeps[1]] }, when).big[1], "× less energy");

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

} finally {
  rmSync(dir, { recursive: true, force: true });
}

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
