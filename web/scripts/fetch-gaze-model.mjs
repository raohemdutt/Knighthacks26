#!/usr/bin/env node
// The look guide's eye tracking runs in the browser from files this app serves itself (web/public/gaze/), never a CDN.
//
//   node scripts/fetch-gaze-model.mjs            copy the wasm runtime from node_modules; check the model's checksum
//   node scripts/fetch-gaze-model.mjs --fetch    also download the model if it is missing or does not match (needs the internet once)
//
// `npm run build` runs the first form (prebuild). The model file is committed (3.6 MB), so a box without internet builds as it is.
// Without the model the app still works: the "Use my eyes" switch says the gaze model is not installed.

import { createHash } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const web = join(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(web, "public", "gaze");
export const MODEL = {
  file: "face_landmarker.task",
  url: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
  sha256: "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
  bytes: 3758596,
};
const WASM = ["vision_wasm_internal.js", "vision_wasm_internal.wasm", "vision_wasm_nosimd_internal.js", "vision_wasm_nosimd_internal.wasm"];

const sha = (path) => createHash("sha256").update(readFileSync(path)).digest("hex");

mkdirSync(join(out, "wasm"), { recursive: true });
const pkg = join(web, "node_modules", "@mediapipe", "tasks-vision");
if (existsSync(pkg)) {
  // From 1.0.0 on, the package carries a usage logger that posts to Google (https://odml.pa.googleapis.com/v1/log) every
  // minute a task is alive. This app promises that nothing is sent, so the build stops if such a version is installed.
  // 0.10.35 is the last release without it and is the one pinned in package.json.
  const bundle = readFileSync(join(pkg, "vision_bundle.mjs"), "utf8");
  const phones = bundle.match(/https?:\/\/[a-z0-9.-]+\.[a-z]{2,}[^"'`\s)]*/gi) ?? [];
  if (phones.length > 0) {
    console.error(`gaze: @mediapipe/tasks-vision in node_modules can call out (${[...new Set(phones)].join(", ")}). Install the pinned version: npm ci`);
    process.exit(1);
  }
  for (const f of WASM) {
    const from = join(pkg, "wasm", f);
    const to = join(out, "wasm", f);
    if (!existsSync(to) || statSync(to).size !== statSync(from).size) copyFileSync(from, to);
  }
  const version = JSON.parse(readFileSync(join(pkg, "package.json"), "utf8")).version;
  console.log(`gaze: wasm runtime copied from @mediapipe/tasks-vision ${version} to public/gaze/wasm/`);
} else {
  console.log("gaze: @mediapipe/tasks-vision is not installed (npm install); the eye-tracking switch will say so");
}

const model = join(out, MODEL.file);
let ok = existsSync(model) && sha(model) === MODEL.sha256;
if (!ok && process.argv.includes("--fetch")) {
  console.log(`gaze: downloading ${MODEL.url}`);
  const res = await fetch(MODEL.url);
  if (!res.ok) throw new Error(`HTTP ${res.status} for the gaze model`);
  const body = Buffer.from(await res.arrayBuffer());
  const got = createHash("sha256").update(body).digest("hex");
  if (got !== MODEL.sha256) throw new Error(`the downloaded model's sha256 is ${got}, expected ${MODEL.sha256}: not saved`);
  writeFileSync(model, body);
  console.log(`gaze: saved ${MODEL.file}, ${body.length} bytes, sha256 ${got}`);
  ok = got === MODEL.sha256;
}
console.log(ok ? `gaze: model ${MODEL.file} is in place (sha256 ok)` : `gaze: model ${MODEL.file} is ${existsSync(model) ? "not the expected file" : "missing"}; run: node scripts/fetch-gaze-model.mjs --fetch`);
