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
  for (const q of ["Where did I leave my keys?", "where is my wallet", "have you seen my charger", "where's the package", "Where are my keys?", "where is my laptop", "find my phone", "where is my stapler", "where did I leave the blue folder"]) eq(`thing: ${q}`, kind(q), "thing");
  for (const q of ["Where should I go tonight?", "where is Dominick", "where is the nearest pizza", "where was I at noon"]) eq(`a name or a place is a thing to ask the box about too: ${q}`, kind(q), "thing");
  yes("no list of nouns decides what a thing is", !("ROOM_THINGS" in G) && !("PINNED" in G));
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
    ["where is the learning loop", "lab:loop"],
    ["where do I see what an answer costs", "lab:cost"],
    ["show me routing", "lab:routing"],
    ["where is alone or with a friend", "lab:friend"],
    ["where do I put it on the box", "lab:deploy"],
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
  // Today's Plan: asked from that screen, its own parts win; from elsewhere the old names still find it
  for (const [q, id] of [
    ["where's my plan", "dayrun:plan"],
    ["where do I change my plan", "dayrun:plan"], // the agenda: tap a block to change it
    ["where do I tell atlas what to change", "dayrun:tell"],
    ["where's the map", "dayrun:map"],
    ["show me yesterday", "dayrun:dates"],
    ["where do I keep this plan", "dayrun:keep"],
    ["where is what is due", "dayrun:board"],
  ]) eq(`today's plan: ${q}`, G.resolveApp(q, "/dayrun/").target?.id ?? null, id);
  eq("today's plan, from Ask", G.resolveApp("where do I find today's plan", "/").target?.id ?? null, "dayrun:plan");
  // Privacy, People and Teach: every named control can be asked for, from its own screen
  for (const [route, q, id] of [
    ["/privacy/", "where is the live view switch", "privacy:live-view"],
    ["/privacy/", "where do I see faces on my screen", "privacy:live-view"],
    ["/privacy/", "where are saved pictures", "privacy:saved-pictures"],
    ["/privacy/", "where is anything that leaves this box", "privacy:leaves"],
    ["/privacy/", "where do I pause a camera", "privacy:cameras"],
    ["/privacy/", "where is offline mode", "privacy:offline"],
    ["/privacy/", "where do I see what left the box", "privacy:left"],
    ["/privacy/", "where do I delete my data", "privacy:erase"],
    ["/people/", "where is the scan button", "people:scan"],
    ["/people/", "where is the expressions switch", "people:expressions"],
    ["/people/", "where is scan to finish", "people:scan-to-finish"],
    ["/people/", "where is the demo data tag", "people:demo"],
    ["/teach/", "where is the microphone", "teach:mic"],
    ["/teach/", "where do I start teaching", "teach:start"],
    ["/teach/", "where is the progress", "teach:progress"],
    ["/teach/", "where are the examples", "teach:examples"],
  ]) {
    eq(`${route} ${q}`, G.resolveApp(q, route).target?.id ?? null, id);
    eq(`${route} kind: ${q}`, G.classify(q, route)?.kind ?? null, "app");
  }
  // "what is this": only on those three screens, only about something on that screen; everything else is still a question for the box
  eq("what: the phrase", [G.whatIntent("What does the live view switch do?"), G.whatIntent("what is scan to finish"), G.whatIntent("where is it")], ["the live view switch", "scan to finish", null]);
  eq("what: on Privacy", G.classify("what does the live view switch do", "/privacy/")?.target?.id ?? null, "privacy:live-view");
  eq("what: on People", G.classify("what is scan to finish?", "/people/")?.target?.id ?? null, "people:scan-to-finish");
  eq("what: on Teach", G.classify("what does start teaching do", "/teach/")?.target?.id ?? null, "teach:start");
  eq("what: another screen's thing stays a question", G.classify("what is the league", "/teach/"), null);
  eq("what: not a part of the app stays a question", G.classify("what is on my calendar today", "/privacy/"), null);
  eq("what: on Ask nothing changes", G.classify("what does the live view switch do", "/"), null);
  eq("what: with no screen nothing changes", G.classify("what is offline mode"), null);
  eq("app: nothing close", G.resolveApp("where is the coffee machine").target, null);
  eq("labels: keys", G.labelsFor("my keys"), ["keys", "key"]);
  eq("labels: a key", G.labelsFor("my key"), ["key", "keys"]);
  eq("labels: two words", G.labelsFor("the coffee mugs"), ["coffee mugs", "coffee mug", "mugs", "mug"]);
  eq("own thing or not", [G.ownThing("my keys"), G.ownThing("our charger"), G.ownThing("dominick"), G.ownThing("the nearest pizza")], [true, true, false, false]);
  eq("pin target", [G.pinTarget("keys").id, G.pinTarget("keys").route, G.pinTarget("keys").at[0].id], ["pin:keys", "/map/", "pin:keys"]);
  const ids = G.APP_TARGETS.map((t) => t.id);
  yes("registry ids are unique", new Set(ids).size === ids.length);
  yes("every row has a screen that exists", G.APP_TARGETS.every((t) => t.route === null || G.SCREENS[t.route]));
  yes("every row has two sentences that end in a full stop", G.APP_TARGETS.every((t) => t.steps.length === 2 && t.steps.every((s) => /[.?]$/.test(s.say) && s.say.length < 140)));

  // ---- exact or about: the drawing and the words follow one flag, and exact is never claimed from about data
  const TABLE = [
    ["element", "exact", "Here it is"], ["box", "exact", "Here it is"], ["zone", "exact", "Here it is"], ["floor", "exact", "Here it is"], ["route-real", "exact", "Here it is"],
    ["picture", "about", "About here"], ["words", "about", "About here"], ["zone-pin", "about", "About here"], ["route-traced", "about", "About here"], ["unscaled", "about", "About here"],
  ];
  for (const [source, want, label] of TABLE) {
    const p = G.precisionOf(source);
    eq(`precision: ${source}`, [p.precision, G.spotWords(p), p.reason.length > 10], [want, label, true]);
    const said = G.hereSentence(p, "keys", true, "left of the laptop");
    eq(`precision words: ${source}`, [/about/.test(said), said], want === "exact" ? [false, "Your keys are here."] : [true, "Your keys are about here: left of the laptop."]);
  }
  eq("precision: an unknown source is never exact", G.precisionOf("something-new").precision, "about");
  eq("precision: singular", [G.spotWords(G.precisionOf("box")), G.spotWords(G.precisionOf("box"), true), G.hereSentence(G.precisionOf("words"), "wallet", false)], ["Here it is", "Here they are", "Your wallet is about here."]);

  // ---- "where does this go?" (lib/guideBin.ts is not loaded here: it imports the api; its two pure helpers are small and repeated in the page test)

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

  // ---- A* around the cards (lib/guidePath.ts, the room twin's router ported)
  const P = await load("guidePath");
  const open5 = Array.from({ length: 5 }, () => new Array(5).fill(1));
  eq("astar: a straight run on an open grid", P.astar(open5, [0, 2], [4, 2]), [[0, 2], [1, 2], [2, 2], [3, 2], [4, 2]]);
  eq("astar: the diagonal on an open grid", P.astar(open5, [0, 0], [4, 4]).length, 5);
  const wall = open5.map((row, r) => row.map((v, c) => (c === 2 && r < 4 ? Infinity : v)));
  const round = P.astar(wall, [0, 0], [4, 0]);
  yes("astar: goes round a wall through its one gap", round.some(([c, r]) => c === 2 && r === 4) && round.every(([c, r]) => Number.isFinite(wall[r][c])));
  eq("astar: no way through a full wall", P.astar(open5.map((row) => row.map((v, c) => (c === 2 ? Infinity : v))), [0, 0], [4, 0]), null);
  eq("astar: a blocked start", P.astar(wall, [2, 0], [4, 0]), null);
  const costly = open5.map((row, r) => row.map((v, c) => (c === 2 && r < 4 ? 7 : v)));
  yes("astar: a costly card is gone round when the gutter is cheaper", P.astar(costly, [0, 0], [4, 0]).some(([c, r]) => c === 2 && r === 4));
  yes("astar: but crossed when the way round is longer than it is worth", P.astar(open5.map((row, r) => row.map((v, c) => (c === 2 ? 2 : v))), [0, 0], [4, 0]).every(([, r]) => r === 0));
  const grid = P.occupancy(480, 240, [{ left: 200, top: 0, right: 280, bottom: 190 }], [], 24);
  eq("occupancy: size", [grid.cols, grid.rows], [20, 10]);
  eq("occupancy: a card cell and a gutter cell", [grid.cost[2][9], grid.cost[9][9], grid.cost[2][2]], [P.CARD_COST, 1, 1]);
  const card = [{ left: 200, top: 0, right: 280, bottom: 190 }];
  const way = P.pointerRoute({ x: 40, y: 60 }, { x: 440, y: 60 }, card, [], 480, 240);
  yes("route: goes under the card, not through it", way && way.every((q) => !(q.x > 205 && q.x < 275 && q.y < 185)), JSON.stringify(way?.slice(0, 4)));
  yes("route: starts and ends where asked", way && way[0].x === 40 && way[way.length - 1].x === 440 && way[way.length - 1].y === 60);
  yes("route: is a detour, but under 2.5 times the straight way", way && P.length(way) > 400 && P.length(way) < 1000);
  eq("route: nothing in the way means the plain eased path", P.pointerRoute({ x: 40, y: 220 }, { x: 440, y: 220 }, card, [], 480, 240), null);
  eq("route: the target's own card is free", P.pointerRoute({ x: 40, y: 60 }, { x: 240, y: 60 }, card, card, 480, 240), null);
  eq("along: halfway by distance", P.along([{ x: 0, y: 0 }, { x: 10, y: 0 }, { x: 10, y: 30 }], 0.5), { x: 10, y: 10 });
  eq("along: the ends", [P.along([{ x: 1, y: 2 }, { x: 5, y: 2 }], 0), P.along([{ x: 1, y: 2 }, { x: 5, y: 2 }], 1)], [{ x: 1, y: 2 }, { x: 5, y: 2 }]);
  const bent = P.curve([{ x: 0, y: 0 }, { x: 100, y: 0 }, { x: 100, y: 100 }]);
  yes("curve: passes through its points and rounds the corner", bent[0].x === 0 && bent[bent.length - 1].y === 100 && bent.length > 15 && bent.some((q) => q.x === 100 && q.y === 0));
  eq("pull: waypoints in the open are dropped", P.pull(P.occupancy(480, 240, [], [], 24), [{ x: 10, y: 10 }, { x: 100, y: 12 }, { x: 200, y: 10 }, { x: 400, y: 10 }]).length, 2);

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
  yes("features: centred eyes read zero", f0 && f0.length === 6 && Math.abs(f0[0]) < 1e-9 && Math.abs(f0[1]) < 1e-9, JSON.stringify(f0));
  const fr = Z.features(face(0.01, 0));
  const fd = Z.features(face(0, 0.004));
  yes("features: irises to the right read positive x on both eyes", fr[0] > 0.03 && Math.abs(fr[1]) < 1e-9, JSON.stringify(fr));
  yes("features: irises down read positive y", fd[1] > 0.01 && Math.abs(fd[0]) < 1e-9, JSON.stringify(fd));
  yes("features: in units of the distance between the eyes", Math.abs(fr[0] - 0.01 / 0.24) < 1e-9);
  eq("features: no iris points", Z.features(face(0, 0).slice(0, 468)), null);
  const blink = face(0, 0);
  blink[159] = { x: 0.42, y: 0.399 };
  blink[145] = { x: 0.42, y: 0.401 };
  eq("features: a blink is skipped", Z.features(blink), null);
  const m = new Array(16).fill(0);
  (m[8] = 0.2), (m[9] = -0.1), (m[12] = 3), (m[13] = -6), (m[14] = -60);
  eq("features: head terms come from the face transform when it is given", Z.features(face(0, 0), m).slice(2), [0.2, -0.1, 0.05, -0.1]);

  // ---- gaze: calibration on a made-up person. Their irises move 0.02 of the picture across the screen, a little less
  //      towards the edges (so the truth is slightly curved), and their head sways on its own (it carries no hint of where they look). `noise` is the camera's jitter.
  let seed = 11;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff - 0.5) * 2;
  const look = (x, y, noise) => {
    const gx = 0.5 - x;
    const gy = y - 0.5;
    const sway = [rnd() * 0.01, rnd() * 0.01];
    const mm = new Array(16).fill(0);
    (mm[8] = sway[0]), (mm[9] = sway[1]), (mm[12] = sway[0] * 40), (mm[13] = sway[1] * 40), (mm[14] = -55);
    return Z.features(face(gx * 0.02 * (1 - 0.25 * gx * gx) + rnd() * noise, gy * 0.008 * (1 - 0.25 * gy * gy) + rnd() * noise * 0.4), mm);
  };
  const W = 1440;
  const H = 900;
  const looks = (dots, noise, n = 28) => dots.flatMap((d) => Array.from({ length: n }, () => ({ f: look(d.x, d.y, noise), x: d.x, y: d.y })));
  const calibrate = (noise, dots = Z.DOTS_9) => Z.validate(Z.fitCalibration(looks(dots, noise), W / H, 1000), looks(Z.DOTS_CHECK, noise), W, H);
  const truth = (c, noise) => {
    // 300 fresh looks anywhere on the screen: the real mean miss in px, which the validation number should be close to
    let sum = 0;
    for (let i = 0; i < 300; i++) {
      const x = 0.05 + 0.9 * Math.abs(rnd());
      const y = 0.05 + 0.9 * Math.abs(rnd());
      const ps = Array.from({ length: 9 }, () => Z.predict(c, look(x, y, noise)));
      sum += Math.hypot((Z.median(ps.map((q) => q.x)) - x) * W, (Z.median(ps.map((q) => q.y)) - y) * H);
    }
    return sum / 300;
  };
  const rows = [0.0003, 0.001, 0.003, 0.008, 0.02].map((noise) => {
    const c = calibrate(noise);
    return { noise, c, real: truth(c, noise) };
  });
  for (const r of rows) console.log(`        camera jitter ${r.noise}: ${r.c.model} model, left-out ${(r.c.leftOut * 100).toFixed(1)}% of diagonal, miss on fresh dots ${Math.round(r.c.validation.freshPx)} px, counted as ${Math.round(r.c.validation.meanPx)} px (true ${Math.round(r.real)} px) -> ${r.c.quality}, ${r.c.cols}x${r.c.rows}`);
  const steady = rows[0].c;
  yes("fit: a steady person is 'good' with a grid finer than 3x3", steady.quality === "good" && steady.cols >= 4, `${steady.cols}x${steady.rows}`);
  yes("fit: the miss it counts is close to the true miss (within a factor of two, both ways)", rows.every((r) => r.c.validation.meanPx < 2 * r.real + 15 && r.real < 2 * r.c.validation.meanPx + 15));
  yes("fit: the miss it counts is never less than the miss on the fresh dots", rows.every((r) => r.c.validation.meanPx >= r.c.validation.freshPx));
  yes("fit: a cell is always at least twice the measured miss", rows.every((r) => r.c.quality === "poor" || (W / r.c.cols >= 2 * r.c.validation.meanPx && H / r.c.rows >= 2 * r.c.validation.meanPx)));
  yes("fit: more jitter never earns a finer grid", rows.every((r, i) => i === 0 || r.c.cols <= rows[i - 1].c.cols));
  yes("fit: a very noisy camera is not used", rows[4].c.quality === "poor" && rows[4].c.cols === 1, `${rows[4].c.quality} ${rows[4].c.cols}`);
  yes("fit: an unchecked fit is not trusted", Z.fitCalibration(looks(Z.DOTS_9, 0.0003), W / H).quality === "poor" && Z.fitCalibration(looks(Z.DOTS_9, 0.0003), W / H).validation === null);
  eq("fit: five dots only ever use the straight model", calibrate(0.0003, Z.DOTS_5).model, "linear");
  eq("fit: too few dots", Z.fitCalibration([{ f: [0, 0, 0, 0, 0, 0], x: 0.5, y: 0.5 }]), null);
  eq("dots: nine, five and four, none shared between fit and check", [Z.DOTS_9.length, Z.DOTS_5.length, Z.DOTS_CHECK.length, Z.DOTS_CHECK.some((d) => Z.DOTS_9.some((e) => e.x === d.x && e.y === d.y))], [9, 5, 4, false]);
  eq("grid: a 90 px miss on 1440 x 900 earns 6 x 4 at most", Z.gridFor(90, 1440, 900), { cols: 6, rows: 4 });
  eq("grid: a 200 px miss earns 3 x 2", Z.gridFor(200, 1440, 900), { cols: 3, rows: 2 });
  eq("grid: a 500 px miss earns nothing", Z.gridFor(500, 1440, 900), { cols: 1, rows: 1 });
  eq("words: a share of the width", Z.shareWords(180, 1440), "1/8 of the screen width");
  yes("words: good states the measured number and that it is not a spot", /Accurate to about 1\/\d+ of the screen width \(\d+ px on average: \d+ px on 4 fresh dots, \d+ px on calibration dots left out of the fit; the worse one counts\)/.test(Z.qualityWords(steady)) && /Not the exact spot/.test(Z.qualityWords(steady)));
  yes("words: rough says what it is used for", /only use it to start the pointer/.test(Z.qualityWords({ ...steady, quality: "rough" })));
  yes("words: poor says it is not used and what to change", /not good enough to use/.test(Z.qualityWords(rows[4].c)));

  // ---- gaze: outliers, drift
  const dot = Array.from({ length: 20 }, (_, i) => ({ f: [0.01 + i * 1e-5, 0.02, 0, 0, 0, 0], x: 0.5, y: 0.5 }));
  eq("clean: a glance away during a dot is dropped", Z.clean([...dot, { f: [0.2, 0.02, 0, 0, 0, 0], x: 0.5, y: 0.5 }]).length, 20);
  eq("clean: steady looks are all kept", Z.clean(dot).length, 20);
  eq("drift: same pose, fresh", Z.drifted(steady, [0, 0, steady.pose[0], steady.pose[1], steady.pose[2], steady.pose[3]], 2000), null);
  eq("drift: head turned", Z.drifted(steady, [0, 0, steady.pose[0] + 0.3, steady.pose[1], steady.pose[2], steady.pose[3]], 2000), "moved");
  eq("drift: moved sideways", Z.drifted(steady, [0, 0, steady.pose[0], steady.pose[1], steady.pose[2] + 0.2, steady.pose[3]], 2000), "moved");
  eq("drift: twenty minutes on", Z.drifted(steady, [0, 0, steady.pose[0], steady.pose[1], steady.pose[2], steady.pose[3]], 1000 + 21 * 60 * 1000), "old");

  // ---- gaze: the One-Euro filter, fixations, rounding
  const oe = new Z.OneEuro();
  let rest;
  let spread = 0;
  for (let i = 0; i < 60; i++) {
    rest = oe.push({ x: 0.3 + rnd() * 0.02, y: 0.5 + rnd() * 0.02 }, i * 40);
    if (i > 30) spread = Math.max(spread, Math.abs(rest.x - 0.3));
  }
  yes("one-euro: jitter at rest is cut to under a third", spread < 0.02 / 3, String(spread));
  let jump;
  for (let i = 60; i < 68; i++) jump = oe.push({ x: 0.8, y: 0.5 }, i * 40);
  yes("one-euro: a jump is followed within a third of a second", Math.abs(jump.x - 0.8) < 0.05, String(jump.x));
  const fx = new Z.Fixation(0.1);
  let fixed;
  for (let i = 0; i <= 10; i++) fixed = fx.push({ x: 0.4 + (i % 2) * 0.01, y: 0.4 }, i * 50);
  yes("fixation: resting eyes count up", fixed.ms === 500 && Math.abs(fixed.x - 0.4) < 0.02, JSON.stringify(fixed));
  eq("fixation: a jump starts again", fx.push({ x: 0.9, y: 0.4 }, 550).ms, 0);
  eq("quantise", Z.quantise({ x: 0.1, y: 0.95 }, 3, 3), { col: 0, row: 2, x: 1 / 6, y: 5 / 6 });
  eq("quantise: outside the screen is clamped", Z.quantise({ x: 1.3, y: -0.2 }, 3, 3), { col: 2, row: 0, x: 5 / 6, y: 1 / 6 });

  // ---- gaze: snap to the one marked thing within the measured miss; never pick between two
  const thing = (id, left, top) => ({ id, left, top, right: left + 100, bottom: top + 40 });
  const marked = [thing("a", 100, 100), thing("b", 100, 160), thing("c", 600, 100)];
  eq("snap: inside one thing", Z.snap({ x: 650, y: 120 }, 50, marked), { kind: "one", id: "c" });
  eq("snap: near one thing, within the miss", Z.snap({ x: 730, y: 120 }, 50, marked), { kind: "one", id: "c" });
  eq("snap: nothing that close", Z.snap({ x: 400, y: 400 }, 50, marked), { kind: "none" });
  eq("snap: two within the miss is 'one of these', nearest first", Z.snap({ x: 150, y: 145 }, 50, marked), { kind: "several", ids: ["a", "b"] });
  eq("snap: a smaller measured miss separates them", Z.snap({ x: 150, y: 120 }, 15, marked), { kind: "one", id: "a" });
  eq("landed: on it", Z.landed({ kind: "one", id: "a" }, "a"), "on");
  eq("landed: one of a few", Z.landed({ kind: "several", ids: ["b", "a"] }, "a"), "near");
  eq("landed: on something else", [Z.landed({ kind: "one", id: "b" }, "a"), Z.landed({ kind: "none" }, "a")], [null, null]);
  eq("distance to a box", [Z.distanceTo({ x: 150, y: 120 }, marked[0]), Z.distanceTo({ x: 230, y: 180 }, marked[0])], [0, Math.hypot(30, 40)]);

  // ---- gaze: dwell
  const d = new Z.Dwell();
  eq("dwell: not yet", [d.update(true, 0), d.update(true, 400), d.update(false, 500), d.update(true, 600), d.update(true, 1200)], [null, null, null, null, null]);
  eq("dwell: arrived after 700 ms on the target", d.update(true, 1300), "arrived");
  eq("dwell: says it once", d.update(true, 5000), null);
  const d2 = new Z.Dwell();
  eq("dwell: away for 3 s repeats the direction once", [d2.update(false, 0), d2.update(false, 2900), d2.update(false, 3000), d2.update(false, 7000)], [null, null, "repeat", null]);
  const d3 = new Z.Dwell();
  eq("dwell: a lost face is neither", [d3.update(false, 0), d3.update(null, 2000), d3.update(false, 2100), d3.update(false, 4000)], [null, null, null, null]);

  // ---- the attention check (lib/guideAttention.ts): what was looked at, what is missing before an action
  const A = await load("guideAttention");
  const jobs = [{ asset_id: "HX-202", name: "Heat exchanger 2" }, { asset_id: "HX-204", name: "Heat exchanger 4" }, { asset_id: "HX-213", name: "Heat exchanger 13" }];
  const box = { left: 0, top: 0, right: 10, bottom: 10 };
  eq("attention: starts off and empty", [A.attention.get().mode, Object.keys(A.attention.get().seen).length], ["off", 0]);
  eq("attention: nothing looked at yet", A.summary({}, jobs).text, "You've looked at 0 of 3 things that need you.");
  eq("attention: before acting on something not looked at", A.missingForAsset("HX-204", "Heat exchanger 4", {}, jobs), { target: "do-first:2", say: "You haven't looked at Heat exchanger 4's alarm yet." });
  A.attention.note("do-first:2", 400, box);
  A.attention.note("do-first:2", 100, box);
  A.attention.note("asset:HX-202", 400, box);
  A.attention.note("do-first:2", 400, box);
  const seen = A.attention.get().seen;
  eq("attention: dwell adds up and the first look keeps its place", [seen["do-first:2"].ms, seen["do-first:2"].order, seen["asset:HX-202"].order], [900, 1, 2]);
  eq("attention: the scan path keeps returns as new entries", A.attention.get().scan, [{ id: "do-first:2", ms: 500 }, { id: "asset:HX-202", ms: 400 }, { id: "do-first:2", ms: 400 }]);
  eq("attention: a job counts by its card or its mark", A.summary(seen, jobs).text, "You've looked at 2 of 3 things that need you.");
  eq("attention: nothing missing once it was looked at", A.missingForAsset("HX-204", "Heat exchanger 4", seen, jobs), null);
  eq("attention: equipment that is not a job points at its mark", A.missingForAsset("PUMP-205", "Pump 5", seen, jobs).target, "asset:PUMP-205");
  eq("attention: a plan needs the battery turn-back looked at first", A.missingForPlan(jobs, jobs, seen).target, "shiftrun:turn-back");
  A.attention.note("shiftrun:turn-back", 400, box);
  eq("attention: then any stop that needs you and was never looked at", A.missingForPlan(jobs, jobs, A.attention.get().seen), { target: "run-asset:HX-213", say: "You haven't looked at Heat exchanger 13 on the plan yet, and it needs you." });
  const reasons = A.runReasons(A.attention.get(), [jobs[0], jobs[2]], jobs);
  eq("attention: Shift Run reasons", reasons.map((x) => `${x.tone}: ${x.text}`), [
    "good: You looked at the battery turn-back while planning: good.",
    "good: You looked at Heat exchanger 2 before planning it.",
    "bad: You put Heat exchanger 13 in your round without looking at it, and left it for last.",
  ]);
  eq("attention: the handoff sentence", A.handoffLine(A.attention.get().seen, jobs), "Reviewed on screen this shift: 2 of 3 items needing attention (by gaze).");
  eq("attention: the sim script", A.simScript(" do-first:1, trust ,,asset:HX-202"), ["do-first:1", "trust", "asset:HX-202"]);
  A.attention.reset();
  eq("attention: reset forgets everything", [Object.keys(A.attention.get().seen).length, A.attention.get().scan.length], [0, 0]);

  // ---- the walk on the plant floor (lib/guideWalk.ts): the dot and the voice in step, a tap snapped to the floor
  {
  const WK = await load("guideWalk");
  const size = { w: 1000, h: 500 };
  const oneFloor = [{ floor: 1, points: [[0.1, 0.2], [0.3, 0.2], [0.3, 0.6]] }]; // 200 px right, then 200 px down
  const twoSteps = [
    { say: "From the yard, go along the corridor.", floor: 1, length_px: 200, upto: 0.5 },
    { say: "Turn right and go into the hall.", floor: 1, length_px: 200, upto: 1 },
    { say: "Pump 5 is on your left.", floor: 1, length_px: 0, upto: 1, arrive: true },
  ];
  const quiet = WK.planWalk(oneFloor, twoSteps, size, false);
  eq("walk: each point's share of the way", quiet.at, [0, 0.5, 1]);
  eq("walk: the arrival sentence is kept for the end", [quiet.arrive, quiet.steps.length], ["Pump 5 is on your left.", 2]);
  eq("walk: silent, the pace is by length", [quiet.ms, quiet.steps.map((s) => [s.t0, s.t1])], [WK.paceMs(400), [[0, 0.5], [0.5, 1]]]);
  eq("walk: pace is clamped", [WK.paceMs(10), WK.paceMs(400), WK.paceMs(5000)], [2500, 3600, 12000]);
  const spoken = WK.planWalk(oneFloor, twoSteps, size, true);
  eq("walk: spoken, a stretch lasts as long as its sentence", spoken.ms, WK.sayMs(twoSteps[0].say) + WK.sayMs(twoSteps[1].say));
  yes("walk: spoken is slower than silent", spoken.ms > quiet.ms);
  eq("walk: which stretch at which time", [WK.stepAt(quiet, 0), WK.stepAt(quiet, 0.49), WK.stepAt(quiet, 0.5), WK.stepAt(quiet, 1)], [0, 0, 1, 1]);
  eq("walk: share of the way at a share of the time", [WK.walkFraction(quiet, 0), WK.walkFraction(quiet, 0.25), WK.walkFraction(quiet, 0.75), WK.walkFraction(quiet, 1)], [0, 0.25, 0.75, 1]);
  const screen = [{ x: 0, y: 0 }, { x: 100, y: 0 }, { x: 100, y: 100 }];
  eq("walk: the dot along the stops", [WK.alongStops(screen, quiet.at, 0.25), WK.alongStops(screen, quiet.at, 0.75), WK.alongStops(screen, quiet.at, 2)], [{ x: 50, y: 0 }, { x: 100, y: 50 }, { x: 100, y: 100 }]);
  eq("walk: no stops, no crash", WK.alongStops([], [], 0.5), { x: 0, y: 0 });
  const stairs = WK.planWalk(
    [{ floor: 1, points: [[0.1, 0.2], [0.3, 0.2]] }, { floor: 2, points: [[0.7, 0.2], [0.9, 0.2]] }],
    [{ say: "Go along the corridor.", floor: 1, length_px: 200, upto: 0.5 }, { say: "Take the stairs up to floor 2.", floor: 1, length_px: 0, upto: 0.5, stairs: true },
     { say: "On floor 2, go along the landing.", floor: 2, length_px: 200, upto: 1 }, { say: "It is here.", floor: 2, length_px: 0, upto: 1, arrive: true }],
    size, false);
  yes("walk: stairs get a stretch of their own", stairs.steps[1].stairs && stairs.steps[1].upto > stairs.steps[1].from && stairs.steps[1].t1 > stairs.steps[1].t0);
  eq("walk: the jump between floors sits where the stairs stretch is", [stairs.at[1], stairs.at[2]].map((v) => Number(v.toFixed(3))), [stairs.steps[1].from, stairs.steps[1].upto].map((v) => Number(v.toFixed(3))));
  yes("walk: the stretches cover the whole way and the whole time", stairs.steps[0].from === 0 && stairs.steps[2].upto === 1 && stairs.steps[2].t1 === 1 && stairs.at[3] === 1);
  eq("walk: an empty route does not divide by zero", WK.walkFraction(WK.planWalk([], [], size, true), 0.4), 0.4);
  // a 10 x 4 grid of 10 px cells from (0.1, 0.1): row 1 is free in columns 2..5, row 2 in columns 2..3
  const grid = { x0: 0.1, y0: 0.1, cell_x: 0.01, cell_y: 0.02, cols: 10, rows: 4, free: 6, runs: [[1, 2, 5], [2, 2, 3]] };
  eq("floor: free and not", [WK.isFree(grid, 0.125, 0.13), WK.isFree(grid, 0.165, 0.13), WK.isFree(grid, 0.145, 0.15), WK.isFree(grid, 0.125, 0.11)], [true, false, false, false]);
  eq("floor: a tap on the floor stays put", WK.snapToFloor(grid, 0.125, 0.13, size), { x: 0.125, y: 0.13, moved: 0 });
  eq("floor: a tap just off it snaps to the nearest cell", WK.snapToFloor(grid, 0.17, 0.13, size), { x: 0.155, y: 0.13, moved: 15 });
  eq("floor: a tap far from any floor is refused", WK.snapToFloor(grid, 0.6, 0.6, size), null);
  eq("floor: which floor a point is on", [WK.floorAt({ 1: grid, 2: { ...grid, x0: 0.5 } }, 0.15, 0.12), WK.floorAt({ 1: grid, 2: { ...grid, x0: 0.5 } }, 0.55, 0.12), WK.floorAt({ 1: grid }, 0.9, 0.9)], [1, 2, null]);
  const legs = { "WP01>WP02": { floors: [{ floor: 1, points: [[0, 0], [1, 0], [1, 1]] }, { floor: 2, points: [[5, 5], [6, 6]] }], connector: "s", walked: true, times_walked: 2 } };
  eq("legs: forwards as stored", WK.legBetween(legs, "WP01", "WP02").floors[0].points, [[0, 0], [1, 0], [1, 1]]);
  eq("legs: backwards is the same line walked the other way", WK.legBetween(legs, "WP02", "WP01").floors.map((f) => [f.floor, f.points]), [[2, [[6, 6], [5, 5]]], [1, [[1, 1], [1, 0], [0, 0]]]]);
  eq("legs: none, and none to itself", [WK.legBetween(legs, "WP01", "WP09"), WK.legBetween(legs, "WP01", "WP01")], [null, null]);
  }

  // ---- any object: a tight outline only with a box, a soft region from the words, else the whole picture
  {
    const laptop = { label: "laptop", box: { x1: 0.5, y1: 0.3, x2: 0.9, y2: 0.7 } };
    const mug = { label: "coffee mug", box: { x1: 0.1, y1: 0.4, x2: 0.2, y2: 0.55 } };
    const REL = [
      ["left of the closed laptop", "left of", ["closed laptop"]],
      ["to the right of the mug", "right of", ["mug"]],
      ["in front of the monitor", "in front of", ["monitor"]],
      ["behind the monitor stand", "behind", ["monitor stand"]],
      ["next to the coffee mug, left of the closed laptop", "next to", ["coffee mug"]],
      ["on the desk", "on", ["desk"]],
      ["under the lamp", "under", ["lamp"]],
      ["between the mug and the laptop", "between", ["mug", "laptop"]],
    ];
    for (const [text, relation, things] of REL) eq(`relation: ${text}`, G.parseRelation(text), { relation, things });
    eq("relation: none in the words", [G.parseRelation("plugged into the power strip"), G.parseRelation(""), G.parseRelation(null)], [null, null, null]);
    const region = (rel, anchors) => G.aboutRegion(rel, anchors);
    yes("region: left of is left of the anchor", region("left of", [laptop.box]).x2 <= 0.5 + 1e-9 && region("left of", [laptop.box]).x1 < 0.5);
    yes("region: right of is right of the anchor", region("right of", [mug.box]).x1 >= 0.2 - 1e-9);
    yes("region: behind is above it in the picture, in front of is below", region("behind", [laptop.box]).y1 < 0.3 && region("in front of", [laptop.box]).y2 > 0.7);
    yes("region: between lies between the two", region("between", [mug.box, laptop.box]).x1 >= 0.2 - 1e-9 && region("between", [mug.box, laptop.box]).x2 <= 0.5 + 1e-9);
    yes("region: stays inside the picture", Object.values(region("right of", [{ x1: 0.8, y1: 0.0, x2: 0.9, y2: 0.2 }])).every((v) => v >= 0 && v <= 1));
    eq("region: no anchor, no region", region("left of", []), null);

    const boxed = G.spotlightFor({ box: { x1: 0.4, y1: 0.55, x2: 0.48, y2: 0.62 }, position: "left of the laptop", anchors: [laptop] });
    eq("spotlight: a box gives a tight outline and 'Here it is'", [boxed.kind, boxed.precision.precision, boxed.label, boxed.rect], ["box", "exact", "Here it is", { x1: 0.4, y1: 0.55, x2: 0.48, y2: 0.62 }]);
    const worded = G.spotlightFor({ box: null, position: "left of the closed laptop", anchors: [laptop, mug] });
    eq("spotlight: words beside a boxed anchor give a soft region and 'About here'", [worded.kind, worded.precision.precision, worded.label, worded.anchor, worded.words], ["about", "about", "About here", "laptop", "left of the closed laptop"]);
    const alone = G.spotlightFor({ box: null, position: "behind the monitor stand", anchors: [laptop] });
    eq("spotlight: words with no boxed anchor ring the whole picture", [alone.kind, alone.rect, alone.precision.precision, alone.words], ["picture", null, "about", "behind the monitor stand"]);
    eq("spotlight: nothing known rings the whole picture", [G.spotlightFor(null).kind, G.spotlightFor(null).precision.precision], ["picture", "about"]);
    // the rule a test must prove: without box data there is never a tight outline, whatever else is known
    const junk = [null, undefined, {}, { x1: 0.5, y1: 0.5, x2: 0.5, y2: 0.9 }, { x1: 0.6, y1: 0.2, x2: 0.4, y2: 0.9 }, { x1: NaN, y1: 0, x2: 1, y2: 1 }];
    const wordsList = [null, "", "left of the laptop", "on the desk", "between the mug and the laptop", "plugged into the power strip"];
    const anchorSets = [[], [laptop], [laptop, mug]];
    let tight = 0;
    let tried = 0;
    for (const box of junk) for (const position of wordsList) for (const anchors of anchorSets) {
      const sp = G.spotlightFor({ box, position, anchors });
      tried++;
      if (sp.kind === "box" || sp.precision.precision === "exact" || sp.label !== "About here") tight++;
    }
    eq(`spotlight: never tight, exact or 'Here it is' without a box (${tried} cases)`, tight, 0);
    const keyframe = readFileSync(join(web, "components", "Keyframe.tsx"), "utf8");
    yes("the picture draws a solid outline only for kind box with exact precision", keyframe.includes('const exact = spot?.kind === "box" && spot.precision.precision === "exact"') && keyframe.includes('exact ? "3px solid var(--danger)" : "3px dashed var(--danger)"'));

    eq("cover: a 4:3 picture in a 4:3 frame maps straight", G.coverRect({ x1: 0.25, y1: 0.5, x2: 0.5, y2: 1 }, { w: 640, h: 480 }, { w: 4, h: 3 }), { left: 25, top: 50, width: 25, height: 50 });
    const wide = G.coverRect({ x1: 0.5, y1: 0, x2: 1, y2: 1 }, { w: 1600, h: 900 }, { w: 4, h: 3 });
    yes("cover: a 16:9 picture is cropped at the sides, and the box follows", Math.abs(wide.left - 50) < 1e-6 && wide.top === 0 && wide.height === 100 && Math.abs(wide.width - 50) < 1e-6);

    const where = { label: "keys", camera_id: "cam-desk", zone: "desk", position: "left of the closed laptop", frame_ref: "f1", box: null, seen_at: "2026-10-10T12:00:00Z", anchors: [] };
    const found = (o) => G.resolveThing({ thing: "my keys", citationFrame: null, where: null, live: false, ...o });
    const SOURCES = [
      ["the answer's picture, with the thing's box", found({ citationFrame: "f1", where: { ...where, box: { x1: 0.4, y1: 0.5, x2: 0.5, y2: 0.6 } } }), "answer-box", "exact", "Your keys are here."],
      ["the answer's picture, words beside a boxed anchor", found({ citationFrame: "f1", where: { ...where, anchors: [laptop] } }), "answer-words", "about", "Your keys are about here: left of the closed laptop."],
      ["the answer's picture, words only", found({ citationFrame: "f1", where }), "answer-words", "about", "Your keys are about here: left of the closed laptop."],
      ["the answer's picture, and the sighting is of another frame", found({ citationFrame: "f9", where: { ...where, box: { x1: 0.4, y1: 0.5, x2: 0.5, y2: 0.6 } } }), "answer-picture", "about", "Your keys are about here: somewhere in this picture."],
      ["a live detector track", found({ live: true, where }), "live-track", "exact", "Your keys are here. This is the camera's picture, refreshed, not video."],
      ["the room's pin", found({ where }), "room-pin", "about", "Your keys are about here: left of the closed laptop."],
    ];
    for (const [name, got, source, precision, say] of SOURCES) eq(`thing: ${name}`, [got.source, got.precision.precision, got.say], [source, precision, say]);
    eq("thing: never seen", [found({}).source, found({}).precision, found({}).spotlight, found({}).say], ["none", null, null, "I haven't seen that."]);
    yes("thing: a box for another frame is never drawn on this one", found({ citationFrame: "f9", where: { ...where, box: { x1: 0.4, y1: 0.5, x2: 0.5, y2: 0.6 } } }).spotlight.kind === "picture");

    // ---- precision is one contract: every source, the outline and the sentence
    const ROWS = [
      ["element", "exact"], ["box", "exact"], ["zone", "exact"], ["floor", "exact"], ["route-real", "exact"],
      ["picture", "about"], ["words", "about"], ["zone-pin", "about"], ["floor-default", "about"], ["route-traced", "about"], ["unscaled", "about"], ["something-new", "about"],
    ];
    for (const [source, want] of ROWS) {
      const pr = G.precisionOf(source);
      eq(`precision: ${source}`, [pr.precision, pr.reason.length > 8, G.spotWords(pr), G.hereSentence(pr, "keys", true, "left of the laptop")],
        [want, true, want === "exact" ? "Here it is" : "About here", want === "exact" ? "Your keys are here." : "Your keys are about here: left of the laptop."]);
    }
    eq("precision: a room pin", [G.roomPrecision(true, "set").precision, G.roomPrecision(true, undefined).precision, G.roomPrecision(true, "default").precision, G.roomPrecision(false, "set").precision], ["exact", "exact", "about", "about"]);
    eq("precision: an uncalibrated camera says why", G.roomPrecision(true, "default").reason, "this camera is not calibrated for the room yet");

    // ---- the rebuilt Room screen and Home
    const at = (q, route) => G.resolveApp(q, route).target?.id ?? null;
    eq("room: the room", [at("where is the room map", "/"), at("show me the room", "/")], ["map:room", "map:room"]);
    eq("room: which bin opens the fold", [at("show me which bin", "/map/"), G.resolveApp("show me which bin").target.reveal[0].op], ["map:sort", "open"]);
    eq("room: who is in view", at("where do I see who is in view", "/map/"), "map:now");
    eq("home: targets", [at("where is the home screen", "/"), at("show me what atlas does", "/home/"), at("where are the numbers", "/home/"), at("show me how a question travels", "/home/")], ["home:top", "home:what", "home:numbers", "home:flow"]);
    const pin = G.pinTarget("keys");
    eq("room: a thing's pin, then its card", [pin.route, pin.at.slice(0, 2).map((a) => a.id), pin.reveal[0].op], ["/map/", ["pin:keys", "thing:keys"], "press"]);
  }

  // ---- the hooks each screen must keep (lib/guide-hooks.json; tests/test_guide_hooks.py checks the same in pytest)
  {
    const repo = join(web, "..");
    const manifest = JSON.parse(readFileSync(join(web, "lib", "guide-hooks.json"), "utf8"));
    for (const [kind, key, what] of [["attributes", "need", "the hook"], ["headings", "text", "the heading or label"]]) {
      for (const row of manifest[kind]) {
        const sources = row.files.map((f) => {
          try {
            return readFileSync(join(repo, f), "utf8");
          } catch {
            yes(`hooks: ${row.screen}: ${f} exists`, false, "(the file is gone; see web/lib/guide.ts and web/lib/guide-hooks.json)");
            return "";
          }
        });
        for (const needle of row[key]) yes(`hooks: ${row.screen} lost ${what} ${needle}`, sources.some((s) => s.includes(needle)), `(expected in ${row.files.join(" or ")}; the look guide points there: put it back, or change web/lib/guide.ts and web/lib/guide-hooks.json)`);
      }
    }
  }
} finally {
  rmSync(dir, { recursive: true, force: true });
}

console.log(failed ? `\n${failed} failed, ${passed} passed` : `look guide self-check: ${passed} checks passed`);
process.exit(failed ? 1 : 0);
