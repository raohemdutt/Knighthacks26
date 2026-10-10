#!/usr/bin/env node
// Calls every gateway endpoint the phone app uses and fails loudly when a field
// the app reads is missing or has the wrong type. Keep it in step with lib/api.ts.
//
//   node scripts/check-api.mjs [http://127.0.0.1:8000] [--writes]
//
// ATLAS_URL sets the gateway too; ATLAS_TOKEN is sent as the bearer token on writes.
// Without --writes the only POST is /api/ask (which the gateway logs as a question).
// With --writes it also creates a harmless training-export approval through
// /api/teach and denies it, to check the approval card and the live events, and
// tries POST /api/presence. Use --writes against a throwaway data directory.

import { randomBytes } from "node:crypto";
import http from "node:http";
import https from "node:https";

const args = process.argv.slice(2);
const WRITES = args.includes("--writes");
const BASE = (args.find((a) => !a.startsWith("--")) || process.env.ATLAS_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
const TOKEN = process.env.ATLAS_TOKEN || "";

const DEMO_QUESTIONS = [
  "Where did I leave my keys?",
  "What is due this week?",
  "What happened at our table overnight?",
  "What should I eat?",
  "Where should I go tonight?",
];

let failures = 0;
const ok = (name, note = "") => console.log(`  ok    ${name}${note ? `  (${note})` : ""}`);
const skip = (name, note) => console.log(`  skip  ${name}  (${note})`);
const fail = (name, note) => {
  failures++;
  console.log(`  FAIL  ${name}  ${note}`);
};

// ---------- shape checking ----------
// A spec maps field -> type. Types: string, number, boolean, array, object; a trailing "?" also allows null.
// A function is a custom check that returns an error string or nothing.

function typeOf(v) {
  return v === null ? "null" : Array.isArray(v) ? "array" : typeof v;
}

function problems(value, spec, where) {
  if (typeOf(value) !== "object") return [`${where}: expected an object, got ${typeOf(value)}`];
  const out = [];
  for (const [field, want] of Object.entries(spec)) {
    if (!(field in value)) {
      out.push(`${where}.${field}: missing`);
      continue;
    }
    const got = value[field];
    if (typeof want === "function") {
      const err = want(got);
      if (err) out.push(`${where}.${field}: ${err}`);
      continue;
    }
    const nullable = want.endsWith("?");
    const base = nullable ? want.slice(0, -1) : want;
    if (got === null && nullable) continue;
    if (typeOf(got) !== base) out.push(`${where}.${field}: expected ${want}, got ${typeOf(got)} (${JSON.stringify(got)?.slice(0, 60)})`);
  }
  return out;
}

const listOf = (spec) => (v) => {
  if (!Array.isArray(v)) return `expected an array, got ${typeOf(v)}`;
  const bad = v.flatMap((row, i) => problems(row, spec, `[${i}]`));
  return bad.length ? bad.slice(0, 3).join("; ") : undefined;
};
const oneOf = (...values) => (v) => (values.includes(v) ? undefined : `expected one of ${values.join(", ")}, got ${JSON.stringify(v)}`);
const nullableObject = (spec) => (v) => (v === null ? undefined : problems(v, spec, "").join("; ") || undefined);
const isoDate = (v) => (typeof v === "string" && !Number.isNaN(Date.parse(v)) ? undefined : `expected an ISO datetime, got ${JSON.stringify(v)}`);
const isoDateOrNull = (v) => (v === null ? undefined : isoDate(v));
const stringList = (v) => (Array.isArray(v) && v.every((x) => typeof x === "string") ? undefined : "expected a list of strings");

const ENVELOPE = { event_id: "string", type: "string", version: "number", ts: isoDate };
const CITATION = { episode_id: "string?", frame_ref: "string?", ts: isoDateOrNull, url: "string?", title: "string?" };
const ANSWER = { ...ENVELOPE, question_id: "string", answer: "string", model_version: "string", citations: listOf(CITATION) };
const EPISODE = { id: "string", camera_id: "string", start: isoDate, end: isoDateOrNull, labels: stringList, captions: stringList, summary: "string?", keyframe_refs: stringList };
const STATS = { hours_recorded: "number", episodes: "number", captions: "number", first_event: isoDateOrNull };
const HIT = { episode_id: "string", camera_id: "string", ts: isoDate, text: "string", score: "number", frame_ref: "string?", kind: "string", url: "string?", title: "string?" };
const ITEM = { connector: "string", kind: "string", external_id: "string", title: "string", when: isoDateOrNull, url: "string?", summary: "string?" };
const NUDGE = { message: "string", tone: oneOf("warn", "celebrate"), ts: isoDate };
const HABIT = {
  id: "string", name: "string", unit: "string", direction: oneOf("at_least", "at_most"), target: "number", day: "string",
  value: "number", met: "boolean", applies: "boolean", streak: "number", closed: "boolean", nudge: nullableObject(NUDGE),
};
const HABITS_TODAY = { day: "string?", habits: listOf(HABIT), latest_nudge: nullableObject(NUDGE) };
const COUNTS = { items_seen: "number", recyclable_seen: "number", recycled: "number", contamination: "number", diversion_rate: "number" };
const IMPACT = {
  total: (v) => problems(v, COUNTS, "").join("; ") || undefined,
  before_after_nudges: nullableObject({ before: nullableObject(COUNTS), after: nullableObject(COUNTS) }),
};
const APPROVAL = { id: "string", kind: "string", target: "string", reason: "string", params: "object", status: "string", expires_at: isoDate, result: "string?" };
const PLACE = { ...ENVELOPE, circle_id: "string", member_id: "string", place: "string", source: oneOf("phone", "desk"), expires_at: isoDate };
const LEADERBOARD = {
  circle_id: "string",
  impact: listOf({ member_id: "string", display_name: "string", items_seen: "number", recycled: "number", contamination: "number", diversion_rate: "number" }),
  habits: listOf({ member_id: "string", display_name: "string", habit_id: "string", streak: "number", day: "string" }),
  scores: listOf({ round_id: "string", member_id: "string", display_name: "string", item: "string", players: "number", correct: "boolean", seconds: "number", score: "number", played_at: isoDate }),
};
const ROUTE = { item: "string", bin: "string", rows: "number", cols: "number", path: listOf({ row: "number", col: "number" }) };
const ACCURACY = { session_id: "string", rounds: "number", first_round: "number?", round_5_on: "number?" };

// ---------- transport ----------

async function call(method, path, body) {
  const headers = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (method === "POST" && TOKEN) headers["Authorization"] = `Bearer ${TOKEN}`;
  const res = await fetch(BASE + path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
  const type = res.headers.get("content-type") || "";
  const data = type.includes("json") ? await res.json() : null;
  return { status: res.status, type, data };
}

/** Check one endpoint. `optional` lists statuses that mean "this part of the box is not set up". */
async function check(name, method, path, spec, { body, optional = {}, list = false } = {}) {
  let r;
  try {
    r = await call(method, path, body);
  } catch (err) {
    fail(name, `request failed: ${err.cause?.code || err.message}`);
    return undefined;
  }
  if (r.status in optional) {
    skip(name, `HTTP ${r.status}: ${r.data?.detail ?? optional[r.status]}`);
    return undefined;
  }
  if (r.status !== 200) {
    fail(name, `HTTP ${r.status}: ${JSON.stringify(r.data)?.slice(0, 200)}`);
    return undefined;
  }
  const bad = list ? (listOf(spec)(r.data) ? [listOf(spec)(r.data)] : []) : problems(r.data, spec, "body");
  if (bad.length) fail(name, bad.join("; "));
  else ok(name, list ? `${r.data.length} rows` : "");
  return bad.length ? undefined : r.data;
}

/** A tiny WebSocket client (Node 20 has none built in): collects the JSON text frames the gateway sends. */
function openEvents() {
  const url = new URL(BASE.replace(/^http/, "ws") + "/ws");
  const lib = url.protocol === "wss:" ? https : http;
  const events = [];
  return new Promise((resolve, reject) => {
    const req = lib.request({
      host: url.hostname,
      port: url.port || (url.protocol === "wss:" ? 443 : 80),
      path: url.pathname,
      headers: { Connection: "Upgrade", Upgrade: "websocket", "Sec-WebSocket-Version": "13", "Sec-WebSocket-Key": randomBytes(16).toString("base64") },
    });
    req.on("error", reject);
    req.on("response", (res) => reject(new Error(`/ws answered HTTP ${res.statusCode} instead of upgrading`)));
    req.on("upgrade", (_res, socket, head) => {
      let buf = head;
      const drain = () => {
        for (;;) {
          if (buf.length < 2) return;
          const opcode = buf[0] & 0x0f;
          let len = buf[1] & 0x7f;
          let at = 2;
          if (len === 126) {
            if (buf.length < 4) return;
            len = buf.readUInt16BE(2);
            at = 4;
          } else if (len === 127) {
            if (buf.length < 10) return;
            len = Number(buf.readBigUInt64BE(2));
            at = 10;
          }
          if (buf.length < at + len) return;
          if (opcode === 1) {
            try {
              events.push(JSON.parse(buf.subarray(at, at + len).toString("utf8")));
            } catch {
              events.push({ type: "(not JSON)" });
            }
          }
          buf = buf.subarray(at + len);
        }
      };
      socket.on("data", (d) => {
        buf = Buffer.concat([buf, d]);
        drain();
      });
      socket.on("error", () => undefined);
      drain();
      resolve({ events, close: () => socket.destroy() });
    });
    req.end();
  });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function waitFor(events, type, ms = 3000) {
  const until = Date.now() + ms;
  for (;;) {
    const e = events.find((x) => x.type === type);
    if (e || Date.now() > until) return e;
    await sleep(50);
  }
}

// ---------- the run ----------

console.log(`Atlas API check against ${BASE}${TOKEN ? " (with token)" : ""}${WRITES ? " (with writes)" : ""}\n`);

console.log("Shell");
const health = await check("GET /health", "GET", "/health", { ok: "boolean", model: "string" });
if (health === undefined && failures) {
  console.log(`\nThe gateway at ${BASE} is not answering; nothing else can be checked.`);
  process.exit(1);
}

let ws;
try {
  ws = await openEvents();
  ok("WS /ws", "connected");
} catch (err) {
  fail("WS /ws", err.message);
}

console.log("\nAsk");
let frameRef;
for (const q of DEMO_QUESTIONS) {
  const a = await check(`POST /api/ask "${q}"`, "POST", "/api/ask", ANSWER, { body: { text: q } });
  if (a) {
    frameRef ??= a.citations.find((c) => c.frame_ref)?.frame_ref;
    const kinds = a.citations.map((c) => (c.frame_ref ? "frame" : c.url ? "link" : "titled source"));
    console.log(`        model ${a.model_version}; ${a.citations.length} citations${kinds.length ? `: ${kinds.join(", ")}` : ""}`);
  }
}
if (ws) {
  const asked = await waitFor(ws.events, "experience.question.asked");
  const answered = await waitFor(ws.events, "agent.answer.generated");
  if (!asked || !answered) fail("WS /ws events for a question", `saw only: ${[...new Set(ws.events.map((e) => e.type))].join(", ") || "nothing"}`);
  else {
    const bad = [...problems(asked, { ...ENVELOPE, question_id: "string", text: "string" }, "asked"), ...problems(answered, ANSWER, "answered")];
    if (bad.length) fail("WS /ws events for a question", bad.join("; "));
    else ok("WS /ws events for a question", "experience.question.asked, agent.answer.generated");
  }
}
if (frameRef) {
  const r = await fetch(`${BASE}/memory/keyframes/${encodeURIComponent(frameRef)}`);
  const type = r.headers.get("content-type") || "";
  if (r.status === 200 && type.startsWith("image/")) ok(`GET /memory/keyframes/${frameRef}`, type);
  else if (r.status === 404) ok(`GET /memory/keyframes/${frameRef}`, "404: no image on this box, the app shows its placeholder");
  else fail(`GET /memory/keyframes/${frameRef}`, `HTTP ${r.status} ${type}`);
} else skip("GET /memory/keyframes/{ref}", "no answer cited a frame");

console.log("\nTimeline");
await check("GET /memory/stats", "GET", "/memory/stats", STATS);
const episodes = await check("GET /memory/episodes", "GET", "/memory/episodes", EPISODE, { list: true });
if (episodes?.[0]) await check(`GET /memory/episodes?camera=${episodes[0].camera_id}`, "GET", `/memory/episodes?camera=${encodeURIComponent(episodes[0].camera_id)}`, EPISODE, { list: true });
await check("GET /memory/search?q=keys", "GET", "/memory/search?q=keys&limit=20", HIT, { list: true });
await check("GET /memory/items", "GET", "/memory/items", ITEM, { list: true });

console.log("\nToday");
await check("GET /api/habits/today", "GET", "/api/habits/today", HABITS_TODAY, { optional: { 404: "habits are not configured" } });
await check("GET /memory/items?from=<now>", "GET", `/memory/items?from=${encodeURIComponent(new Date().toISOString())}`, ITEM, { list: true });
await check("GET /api/impact", "GET", "/api/impact", IMPACT, { optional: { 404: "Waste Watch is not configured" } });

console.log("\nApprovals");
await check("GET /api/approvals", "GET", "/api/approvals", APPROVAL, { list: true });
if (WRITES) {
  if (ws) ws.events.length = 0;
  const task = `check-api-${Date.now()}`;
  const taught = await call("POST", "/api/teach", { task, instruction: "check-api.mjs: safe to deny" });
  const id = taught.data?.action_id;
  if (taught.status !== 200 || typeof id !== "string") fail("POST /api/teach (to create a card)", `HTTP ${taught.status}: ${JSON.stringify(taught.data)?.slice(0, 200)}`);
  else {
    const pending = await check("GET /api/approvals with a pending card", "GET", "/api/approvals", APPROVAL, { list: true });
    if (pending && !pending.some((p) => p.id === id)) fail("GET /api/approvals with a pending card", `the new request ${id} is not listed`);
    if (ws) {
      const e = await waitFor(ws.events, "workspace.approval.requested");
      const bad = e ? problems(e, { ...ENVELOPE, action_id: "string", kind: "string", target: "string", reason: "string", expires_at: isoDate }, "event") : ["not received"];
      if (bad.length) fail("WS workspace.approval.requested", bad.join("; "));
      else ok("WS workspace.approval.requested");
    }
    await check(`POST /api/approvals/{id} deny`, "POST", `/api/approvals/${id}`, APPROVAL, { body: { decision: "deny" } });
    if (ws) {
      const e = await waitFor(ws.events, "workspace.action.completed");
      if (!e) fail("WS workspace.action.completed", "not received");
      else ok("WS workspace.action.completed", `status ${e.status}`);
    }
    const after = await call("GET", "/api/approvals");
    if (after.data?.some?.((p) => p.id === id)) fail("GET /api/approvals after deny", "the denied request is still pending");
    else ok("GET /api/approvals after deny", "card gone");
  }
} else skip("approve / deny round trip", "run with --writes");

console.log("\nCircle");
await check("GET /api/presence", "GET", "/api/presence", PLACE, { list: true, optional: { 404: "the Circle is not configured" } });
await check("GET /api/circle/leaderboard", "GET", "/api/circle/leaderboard", LEADERBOARD, { optional: { 404: "the Circle is not connected" } });
if (WRITES) {
  await check("POST /api/presence", "POST", "/api/presence", PLACE, {
    body: { place: "Library", hours: 1 },
    optional: { 404: "the Circle is not configured", 422: "sharing is off or the place is not on the box's list" },
  });
} else skip("POST /api/presence", "run with --writes");

console.log("\nSort It");
for (const item of ["can", "cup"]) {
  const r = await check(`GET /api/play/route?item=${item}`, "GET", `/api/play/route?item=${item}`, ROUTE, { optional: { 404: "Sort It is not configured" } });
  if (r) console.log(`        ${item} -> ${r.bin}; ${r.cols}x${r.rows} grid; ${r.path.length} cells`);
}
await check("GET /api/play/accuracy", "GET", "/api/play/accuracy", ACCURACY, { optional: { 404: "Sort It is not configured" } });

ws?.close();
console.log(failures ? `\nFAILED: ${failures} check${failures === 1 ? "" : "s"} did not match what the app expects.` : "\nAll checks passed.");
process.exit(failures ? 1 : 0);
