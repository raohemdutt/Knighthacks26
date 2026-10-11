#!/usr/bin/env node
// Self-check for the look guide's pure functions (lib/guide.ts, lib/gaze.ts): the where-question intent, the app
// resolver, the pointer's path, the calibration fit, region rounding and dwell. The web app has no test runner, so
// this compiles the two files with the TypeScript that is already a dev dependency and runs them in node.
//
//   node scripts/check-guide.mjs

import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import ts from "typescript";

const web = join(dirname(fileURLToPath(import.meta.url)), "..");
const dir = mkdtempSync(join(tmpdir(), "atlas-guide-"));
async function load(name) {
  const js = ts.transpileModule(readFileSync(join(web, "lib", `${name}.ts`), "utf8"), { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } }).outputText;
  writeFileSync(join(dir, `${name}.mjs`), js);
  return import(pathToFileURL(join(dir, `${name}.mjs`)).href);
}

let failed = 0;
let passed = 0;
const eq = (name, got, want) => {
  const ok = JSON.stringify(got) === JSON.stringify(want);
  ok ? passed++ : failed++;
  if (!ok) console.log(`  FAIL  ${name}\n        got  ${JSON.stringify(got)}\n        want ${JSON.stringify(want)}`);
};
const yes = (name, cond, detail = "") => {
  cond ? passed++ : failed++;
  if (!cond) console.log(`  FAIL  ${name} ${detail}`);
};

try {
  const G = await load("guide");
  const Z = await load("gaze");

  // ---- the question
  eq("intent: keys", G.whereIntent("Where did I leave my keys?"), "my keys");
  eq("intent: look for", G.whereIntent("Where do I look for the battery chart?"), "the battery chart");
  eq("intent: show me", G.whereIntent("Hey Atlas, show me the League please"), "the league");
  eq("intent: wheres", G.whereIntent("where's my next class"), "my next class");
  eq("intent: not a where-question", G.whereIntent("What is due this week?"), null);
  eq("intent: not a where-question 2", G.whereIntent("Does battery affect reliability?"), null);
  eq("intent: where should I go tonight is a where-question", G.whereIntent("Where should I go tonight?") !== null, true);

  // ---- room thing, app thing, or unsure
  const kind = (q, route = "/") => G.classify(q, route)?.kind ?? null;
  for (const q of ["Where did I leave my keys?", "where is my wallet", "have you seen my charger", "where's the package", "Where are my keys?", "where is my laptop", "find my phone"]) eq(`room: ${q}`, kind(q), "room");
  for (const q of ["Where should I go tonight?", "where is Dominick", "where is the nearest pizza", "where was I at noon"]) eq(`unsure: ${q}`, kind(q), "unsure");
  eq("not a where-question", kind("What should I eat?"), null);

  const APP = [
    ["where's my League?", "circle:league"],
    ["where do I find the league", "circle:league"],
    ["show me the leaderboard", "circle:league"],
    ["where is my next class", "now:next"],
    ["where's my plan", "now:plan"],
    ["where is the check-in", "now:checkin"],
    ["where can I find brief me", "now:brief"],
    ["where do I see what is due", "today:due"],
    ["show me my habits", "today:habits"],
    ["where is waste watch", "today:waste"],
    ["where do I find Day Run", "dayrun:plan"],
    ["show me the campus map", "dayrun:map"],
    ["where are my friends", "circle:people"],
    ["where do I find plans with friends", "circle:plans"],
    ["where is the goal", "impact:goal"],
    ["show me the impact story", "impact:story"],
    ["where is the practice set", "impact:practice"],
    ["where is the room map", "map:room"],
    ["show me which bin", "map:sort"],
    ["where do I find sort it", "play:game"],
    ["where are the approvals", "approvals:pending"],
    ["where is the pending card", "approvals:pending"],
    ["where do I teach atlas", "teach:home"],
    ["where is the learning lab", "lab:home"],
    ["where is privacy", "privacy:home"],
    ["where do I enrol my face", "people:enrol"],
    ["where is forget me", "people:forget"],
    ["where is the timeline", "timeline:home"],
    ["where are the settings", "settings"],
    ["where do I turn on eye tracking", "settings"],
    ["where is the microphone", "ask:mic"],
    ["where do I ask a question", "ask:input"],
    ["where is the inspection screen", "spot:home"],
    ["where is the leage", "circle:league"], // a misspelling
    ["circle:league", "circle:league"], // a guide id, as a link carries it
  ];
  for (const [q, id] of APP) {
    const r = G.resolveApp(q, "/");
    eq(`app: ${q}`, r.target?.id ?? null, id);
    if (q.startsWith("where") || q.startsWith("show")) eq(`app kind: ${q}`, kind(q), "app");
  }
  eq("app: nothing close", G.resolveApp("where is the coffee machine").target, null);
  eq("pin: keys", G.pinnedThing("my keys"), "keys");
  eq("pin: a key", G.pinnedThing("my key"), "keys");
  eq("pin: none for a package", G.pinnedThing("the package"), null);
  eq("pin target", [G.pinTarget("keys").id, G.pinTarget("keys").route, G.pinTarget("keys").at[0].id], ["pin:keys", "/map/", "pin:keys"]);
  const ids = G.APP_TARGETS.map((t) => t.id);
  yes("registry ids are unique", new Set(ids).size === ids.length);
  yes("every row has a screen that exists", G.APP_TARGETS.every((t) => t.route === null || G.SCREENS[t.route]));
  yes("every row has two sentences that end in a full stop", G.APP_TARGETS.every((t) => t.steps.length === 2 && t.steps.every((s) => /[.?]$/.test(s.say) && s.say.length < 140)));

  // ---- the Inspection screen's answer, mapped onto the page
  const spot = G.spotTarget({ configured: true, query: "", target: "round:17", label: "Round 17", confidence: 1, alternatives: [], steps: [{ reveal: "tab:shift", say: "A." }, { reveal: "round:pick:M017", say: "B." }, { say: "C." }, { say: "D." }] });
  eq("spot: a round points at the picker", spot.at, [{ id: "round:pick" }]);
  eq("spot: reveals", spot.reveal, [{ op: "press", at: { id: "tab:shift" }, key: "tab:shift" }, { op: "select", at: { id: "round:pick" }, value: "M017", key: "round:pick:M017" }]);
  eq("spot: a fold", G.spotReveal("open:analysis:findings"), { op: "open", at: { id: "analysis:findings" }, key: "open:analysis:findings" });
  eq("narration: only the reveals that happened", G.narration(spot.steps, ["round:pick:M017"], null, null), { reveal: ["B."], travel: ["C."], arrive: "D." });
  eq("narration: gaze line first", G.narration(spot.steps, [], null, "From where you are, look right.").travel, ["From where you are, look right.", "C."]);

  // ---- the path
  const a = { x: 100, y: 100 };
  const b = { x: 900, y: 500 };
  eq("ease ends", [G.ease(0), G.ease(1), G.ease(0.5)], [0, 1, 0.5]);
  yes("ease is monotonic", Array.from({ length: 50 }, (_, i) => G.ease(i / 49)).every((v, i, all) => i === 0 || v >= all[i - 1]));
  eq("path starts at a", G.pathPoint(a, b, 0), a);
  eq("path ends at b", G.pathPoint(a, b, 1), b);
  const mid = G.pathPoint(a, b, 0.5);
  yes("path bows off the straight line", Math.abs((mid.x - a.x) * (b.y - a.y) - (mid.y - a.y) * (b.x - a.x)) / Math.hypot(b.x - a.x, b.y - a.y) > 20);
  eq("path to the same point stays put", G.pathPoint(a, a, 0.5), a);
  const tr = G.trail(a, b, 0.6);
  yes("trail ends at the pointer", tr.length === 14 && Math.abs(tr[13].x - G.pathPoint(a, b, G.ease(0.6)).x) < 1e-9);
  yes("travel time is bounded", G.travelMs(a, a) === 700 && G.travelMs({ x: 0, y: 0 }, { x: 5000, y: 0 }) === 1800 && G.travelMs(a, b) > 700);

  // ---- words for places and directions (a 1440 x 900 screen)
  eq("region: top left", G.regionWords({ x: 100, y: 100 }, 1440, 900), "the top left");
  eq("region: middle", G.regionWords({ x: 720, y: 450 }, 1440, 900), "the middle");
  eq("region: bottom", G.regionWords({ x: 720, y: 850 }, 1440, 900), "the bottom");
  eq("direction: right", G.directionWords({ x: 200, y: 450 }, { x: 1200, y: 460 }, 1440, 900), "right");
  eq("direction: down and a little left", G.directionWords({ x: 800, y: 100 }, { x: 560, y: 760 }, 1440, 900), "down and a little left");
  eq("direction: up and right", G.directionWords({ x: 300, y: 800 }, { x: 1000, y: 300 }, 1440, 900), "up and right");
  eq("direction: close by", G.directionWords({ x: 700, y: 450 }, { x: 760, y: 470 }, 1440, 900), "");
  eq("gaze line", G.fromGaze({ x: 200, y: 450 }, { x: 1200, y: 460 }, 1440, 900), "From where you are, look right.");
  eq("gaze line: already there", G.fromGaze({ x: 700, y: 450 }, { x: 760, y: 470 }, 1440, 900), "It is right where you are looking.");
  eq("cell", G.cellOf({ x: 1439, y: 0 }, 1440, 900), { col: 2, row: 0 });

  // ---- gaze: features from a synthetic face
  const face = (ix, iy, yaw = 0, pitch = 0) => {
    const lm = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5 }));
    const set = (i, x, y) => (lm[i] = { x, y });
    // eye one: outer 33 at x .38, inner 133 at .46; eye two: inner 362 at .54, outer 263 at .62
    set(33, 0.38, 0.4), set(133, 0.46, 0.4), set(159, 0.42, 0.385), set(145, 0.42, 0.415);
    set(362, 0.54, 0.4), set(263, 0.62, 0.4), set(386, 0.58, 0.385), set(374, 0.58, 0.415);
    set(468, 0.42 + ix, 0.4 + iy), set(473, 0.58 + ix, 0.4 + iy);
    set(1, 0.5 + yaw, 0.5 + pitch);
    return lm;
  };
  const f0 = Z.features(face(0, 0));
  yes("features: centred eyes read zero", f0 && Math.abs(f0[0]) < 1e-9 && Math.abs(f0[1]) < 1e-9, JSON.stringify(f0));
  const fr = Z.features(face(0.01, 0));
  const fd = Z.features(face(0, 0.004));
  yes("features: irises to the right read positive x on both eyes", fr[0] > 0.1 && Math.abs(fr[1]) < 1e-9, JSON.stringify(fr));
  yes("features: irises down read positive y", fd[1] > 0.03 && Math.abs(fd[0]) < 1e-9, JSON.stringify(fd));
  eq("features: no iris points", Z.features(face(0, 0).slice(0, 468)), null);
  const blink = face(0, 0);
  blink[159] = { x: 0.42, y: 0.399 };
  blink[145] = { x: 0.42, y: 0.401 };
  eq("features: a blink is skipped", Z.features(blink), null);

  // ---- gaze: the fit, on a made-up person whose irises move 0.02 of the picture across the screen
  let seed = 7;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff - 0.5) * 2;
  const look = (x, y, noise) => Z.features(face((0.5 - x) * 0.02 + rnd() * noise, (y - 0.5) * 0.008 + rnd() * noise * 0.4, (0.5 - x) * 0.05 + rnd() * noise, (y - 0.5) * 0.04 + rnd() * noise));
  const calibrate = (noise) => Z.fitCalibration(Z.DOTS.flatMap((d) => Array.from({ length: 11 }, () => ({ f: look(d.x, d.y, noise), x: d.x, y: d.y }))), 1.6);
  const clean = calibrate(0.0002);
  yes("fit: a steady person is 'good'", clean.quality === "good", `residual ${clean.residual.toFixed(3)} scatter ${clean.scatter.toFixed(3)}`);
  const probe = Z.predict(clean, look(0.75, 0.25, 0));
  yes("fit: a new point lands in the right third", Math.abs(probe.x - 0.75) < 0.15 && Math.abs(probe.y - 0.25) < 0.15, JSON.stringify(probe));
  const shaky = calibrate(0.015);
  yes("fit: a shaky reading is not 'good'", shaky.quality !== "good", `quality ${shaky.quality} residual ${shaky.residual.toFixed(3)} scatter ${shaky.scatter.toFixed(3)}`);
  yes("fit: a rough fit only earns a 2x2 grid", shaky.quality === "poor" || (shaky.cols === 2 && shaky.rows === 2));
  const noisy = calibrate(0.08);
  eq("fit: noise is 'poor'", noisy.quality, "poor");
  eq("fit: too few dots", Z.fitCalibration([{ f: [0, 0, 0, 0], x: 0.5, y: 0.5 }]), null);
  yes("words: good says it is a grid, not a spot", /grid/.test(Z.qualityWords(clean)) && /not the exact spot/.test(Z.qualityWords(clean)));
  yes("words: rough says what it is used for", /only use it to start the pointer/.test(Z.qualityWords({ ...clean, quality: "rough" })));

  // ---- gaze: smoothing, rounding, dwell
  const sm = new Z.MedianSmoother(5);
  let out;
  for (const x of [0.2, 0.21, 0.9, 0.2, 0.19]) out = sm.push({ x, y: 0.5 });
  eq("median: one wild frame does not move it", out, { x: 0.2, y: 0.5 });
  eq("quantise", Z.quantise({ x: 0.1, y: 0.95 }, 3, 3), { col: 0, row: 2, x: 1 / 6, y: 5 / 6 });
  eq("quantise: outside the screen is clamped", Z.quantise({ x: 1.3, y: -0.2 }, 3, 3), { col: 2, row: 0, x: 5 / 6, y: 1 / 6 });
  const box = { x0: 0.05, y0: 0.05, x1: 0.2, y1: 0.2 };
  eq("on target: same cell", Z.onTarget({ col: 0, row: 0 }, 3, 3, box), true);
  eq("on target: far cell", Z.onTarget({ col: 2, row: 2 }, 3, 3, box), false);
  const d = new Z.Dwell();
  eq("dwell: not yet", [d.update(true, 0), d.update(true, 400), d.update(false, 500), d.update(true, 600), d.update(true, 1200)], [null, null, null, null, null]);
  eq("dwell: arrived after 700 ms on the target", d.update(true, 1300), "arrived");
  eq("dwell: says it once", d.update(true, 5000), null);
  const d2 = new Z.Dwell();
  eq("dwell: away for 3 s repeats the direction once", [d2.update(false, 0), d2.update(false, 2900), d2.update(false, 3000), d2.update(false, 7000)], [null, null, "repeat", null]);
  const d3 = new Z.Dwell();
  eq("dwell: a lost face is neither", [d3.update(false, 0), d3.update(null, 2000), d3.update(false, 2100), d3.update(false, 4000)], [null, null, null, null]);
} finally {
  rmSync(dir, { recursive: true, force: true });
}

console.log(failed ? `\n${failed} failed, ${passed} passed` : `look guide self-check: ${passed} checks passed`);
process.exit(failed ? 1 : 0);
