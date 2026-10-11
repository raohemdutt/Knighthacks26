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
} finally {
  rmSync(dir, { recursive: true, force: true });
}

console.log(`${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
