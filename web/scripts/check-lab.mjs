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
} finally {
  rmSync(dir, { recursive: true, force: true });
}

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
