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
// tries POST /api/presence and commits Day Run's scripted good plan (which creates
// an approval card when the box can write to a calendar; the card is denied again).
// Use --writes against a throwaway data directory.

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
const SIGHTING = { label: "string", camera_id: "string", zone_id: "string?", ts: isoDate, frame_ref: "string", position: "string?" };

// Place packs and Day Run (contexts/play/dayrun/models.py): the fields lib/api.ts reads.
const inner = (spec) => (v) => problems(v, spec, "").join("; ") || undefined;
const optionalNumber = (v) => (v === undefined || typeof v === "number" ? undefined : `expected a number or nothing, got ${typeOf(v)}`);
const PLACE_PACK = {
  name: "string", kind: "string", entry: "string?",
  places: (v) => {
    if (typeOf(v) !== "object") return `expected an object, got ${typeOf(v)}`;
    // `lat`/`lon` are left out (not null) for rooms, so they are checked only when present.
    const bad = Object.entries(v).flatMap(([id, p]) => [
      ...problems(p, { name: "string", kind: "string", aliases: stringList }, id),
      ...["lat", "lon"].flatMap((k) => (optionalNumber(p?.[k]) ? [`${id}.${k}: ${optionalNumber(p[k])}`] : [])),
    ]);
    return bad.length ? bad.slice(0, 3).join("; ") : undefined;
  },
};
const MOVE_OPTION = {
  kind: oneOf("study", "eat", "sleep", "rest", "event"), place: "string?", place_name: "string?", title: "string", walk_minutes: "number",
  back_minutes: "number", max_minutes: "number", feasible: "boolean", reason: "string?", friends: stringList, event: "string?", route: stringList,
};
const INITIAL = { energy: "number", focus: "number", sleep_hours: "number", meals: "number", sleep_debt: "number", hungry_minutes: "number", source: "string" };
const BOARD = {
  start: isoDate, end: isoDate, tick_minutes: "number", pack: "string", start_place: "string", home: "string", closed: stringList,
  initial: inner(INITIAL),
  blocks: listOf({ id: "string", title: "string", start: isoDate, end: isoDate, place: "string?", place_name: "string?", activity: "string", deadline: "string?", url: "string?" }),
  deadlines: listOf({ id: "string", title: "string", due: isoDate, work_left: "number", in_window: "boolean", risk: "number", url: "string?" }),
  slots: listOf({ id: "string", start: isoDate, end: isoDate, minutes: "number", from_place: "string", to_place: "string?", next_block: "string?", moves: listOf(MOVE_OPTION) }),
  friends: listOf({ member: "string", place: "string", place_name: "string", start: isoDate, end: isoDate }),
  events: listOf({ id: "string", title: "string", start: isoDate, end: isoDate, place: "string?", place_name: "string?", url: "string?" }),
};
const pairOrNull = (v) => (v === null || (Array.isArray(v) && v.length === 2 && v.every((n) => typeof n === "number")) ? undefined : "expected [lat, lon] or null");
const MOVE = { slot: "string", kind: "string", place: "string?", minutes: "number?", start: "string?", with_friend: "string?", deadline: "string?", event: "string?" };
const OUTCOME = {
  won: "boolean", stars: "number", reasons: stringList, summary: "string",
  checks: listOf({ id: "string", ok: "boolean", value: "number", target: "number", text: "string", status: "string?" }),
};
const CHANGE = { move: "number?", kind: oneOf("rerouted", "shortened", "friend_gone", "late", "route_changed", "rejected"), text: "string" };
const TOTALS = { study_minutes: "number", focused_minutes: "number", walk_minutes: "number", sleep_hours: "number", meals: "number", energy_end: "number", energy_min: "number", points: "number" };
const SIM = {
  start: isoDate, end: isoDate, tick_minutes: "number", stars: "number", notes: stringList,
  outcome: inner(OUTCOME),
  timeline: listOf({ t: isoDate, energy: "number", focus: "number", deadline_risk: "number", work_left: "object", activity: "string", place: "string" }),
  legs: listOf({ from: "string", to: "string", from_name: "string", to_name: "string", minutes: "number", from_xy: pairOrNull, to_xy: pairOrNull, route: "number", depart: isoDate }),
  routes: listOf({ index: "number", purpose: "string", move: "number?", start: "string", end: "string", depart: isoDate, arrive: isoDate, minutes: "number", places: stringList }),
  rejected: listOf({ index: "number", reason: "string", minutes: "number?", move: inner(MOVE) }),
  blocks: listOf({ id: "string", kind: "string", title: "string", start: isoDate, end: isoDate, minutes: "number", place: "string", place_name: "string", fixed: "boolean", move: "number?", with_friend: "string?" }),
  changes: listOf(CHANGE),
  totals: inner(TOTALS),
};
const HARD = {
  seed: "number", before: inner(SIM), after: inner(SIM), changes: listOf(CHANGE),
  disruptions: listOf({ kind: oneOf("block_moved", "friend_cancelled", "place_closed"), text: "string" }),
};
const MEASURED = { focus_minutes: "number?", meals: "number?", sleep_hours: "number?", blocks: "object" };
const REVIEW = {
  day: "string", score: "number?", planned: "object", measured: inner(MEASURED), text: "string", measured_by: "string",
  blocks: listOf({ id: "string", title: "string", kind: "string", planned_minutes: "number", measured_minutes: "number?", followed: "number?", note: "string" }),
  biggest_gap: nullableObject({ block: "string?", kind: "string", planned: "number", measured: "number", text: "string" }),
};
const REPLAY = {
  day: "string", actual: inner(MEASURED), baseline: inner(SIM), plan: inner(SIM), points_gained: "number", stars_gained: "number", text: "string",
  baseline_moves: listOf(MOVE), board: inner(BOARD),
};
const CAMPAIGN = { won: "boolean", stars: "number", points: "number", days: listOf({ day: "string", result: inner(SIM), carry_in: inner(INITIAL) }) };
const FORECAST = {
  score: "number", bias: "number", multiplier: "number", unmeasured: stringList, text: "string",
  rows: listOf({ block: "string", estimated: "number", measured: "number", error: "number", off_by: "number" }),
};
const COMMIT = { action_id: "string", day: "string", stars: "number", reason: "string", events: listOf({ title: "string", start: isoDate, end: isoDate, location: "string" }) };

// The scripted plans the app embeds (lib/demoDay.ts, copied from fixtures/play/demo.yaml). Keep the three in step.
const DEMO_BAD_PLAN = [
  { slot: "s1", kind: "rest" },
  { slot: "s2", kind: "eat", place: "ucf.dining_63", minutes: 40 },
  { slot: "s3", kind: "rest", minutes: 300 },
  { slot: "s3", kind: "study", place: "apartment.desk", minutes: 540 },
  { slot: "s3", kind: "sleep" },
];
const DEMO_GOOD_PLAN = [
  { slot: "s1", kind: "eat", place: "ucf.dining_63", minutes: 30 },
  { slot: "s1", kind: "rest" },
  { slot: "s2", kind: "eat", place: "ucf.union", minutes: 25 },
  { slot: "s3", kind: "study", place: "ucf.library", minutes: 170, with_friend: "Maya" },
  { slot: "s3", kind: "eat", place: "ucf.union", minutes: 35 },
  { slot: "s3", kind: "study", place: "ucf.library", minutes: 105 },
  { slot: "s3", kind: "rest", place: "apartment.desk", minutes: 60 },
  { slot: "s3", kind: "sleep", start: "2026-10-11T22:30:00-04:00", minutes: 510 },
];
const DEMO_DISRUPTIONS = [
  { kind: "block_moved", block: "outlook:kh-judging:2026-10-11T12:00:00-04:00", to_place: "ucf.cb1", shift_minutes: -15 },
  { kind: "place_closed", place: "ucf.union" },
];
const DEMO_CAMPAIGN = [
  [DEMO_GOOD_PLAN[0], DEMO_GOOD_PLAN[2], DEMO_GOOD_PLAN[3], DEMO_GOOD_PLAN[4], DEMO_GOOD_PLAN[7]],
  [
    { slot: "s1", kind: "eat", place: "apartment.kitchen", minutes: 20 },
    { slot: "s2", kind: "eat", place: "ucf.union", minutes: 30 },
    { slot: "s2", kind: "study", place: "ucf.library", minutes: 180 },
    { slot: "s2", kind: "eat", place: "ucf.union", minutes: 30 },
    { slot: "s2", kind: "sleep", start: "2026-10-12T23:00:00-04:00", minutes: 480 },
  ],
];

const PLACE = { ...ENVELOPE, circle_id: "string", member_id: "string", place: "string", source: oneOf("phone", "desk"), expires_at: isoDate };
const CIRCLE_EVENTS = {
  configured: "boolean", circle_id: "string?", source: oneOf("neon", "local", "none"), offline: "boolean",
  events: listOf({
    id: "string", thread_id: "string?", title: "string", starts_at: isoDate, ends_at: isoDate, location: "string?", created_by: "string?",
    attendees: listOf({ member_id: "string", display_name: "string", answer: oneOf("free", "busy", "counter") }),
  }),
};
const CIRCLE_SHARING = {
  configured: "boolean", circle_id: "string?", member_id: "string?", share_summaries: "boolean", share_presence: "boolean", sharing_now: "boolean",
  database: oneOf("connected", "offline", "unknown", "not_configured"), database_note: "string?", last_reached_at: isoDateOrNull,
  queued: "number", sent: "number", last_synced_at: isoDateOrNull, oldest_queued_at: isoDateOrNull,
  kinds: listOf({ kind: "string", on: "boolean", queued: "number?", sent: "number?", last_synced_at: isoDateOrNull }),
  place: nullableObject({ place: "string", source: oneOf("phone", "desk"), expires_at: isoDate }),
};
const LEADERBOARD = {
  circle_id: "string",
  impact: listOf({ member_id: "string", display_name: "string", items_seen: "number", recycled: "number", contamination: "number", diversion_rate: "number" }),
  habits: listOf({ member_id: "string", display_name: "string", habit_id: "string", streak: "number", day: "string" }),
  scores: listOf({ round_id: "string", member_id: "string", display_name: "string", item: "string", players: "number", correct: "boolean", seconds: "number", score: "number", played_at: isoDate }),
};
const ROUTE = { item: "string", bin: "string", rows: "number", cols: "number", path: listOf({ row: "number", col: "number" }) };
const ACCURACY = { session_id: "string", rounds: "number", first_round: "number?", round_5_on: "number?" };

// The day in words, the coach's questions, the Circle's conversations, the Impact screen, "when to leave", the room
// twin and the inspection dashboard: the fields lib/api.ts and the screens read, checked against the handlers.
const NARRATION = { ...ENVELOPE, day: "string", moment: "string", plan_id: "string?", ref: "string?", spoken: "string", display: "string", engine: "string", facts_hash: "string", urgent: "boolean" };
const mapOf = (spec) => (v) => {
  if (typeOf(v) !== "object") return `expected an object, got ${typeOf(v)}`;
  const bad = Object.entries(v).flatMap(([k, row]) => problems(row, spec, k));
  return bad.length ? bad.slice(0, 3).join("; ") : undefined;
};
const BRIEF = { mode: oneOf("model", "template", "off"), engine: oneOf("model", "template", "off"), timeout_s: "number", gap_s: "number", latest: mapOf(NARRATION), asks: mapOf(NARRATION), model: "object" };
const FEEDBACK_SESSION = { day: "string", block_id: "string", title: "string", start: isoDate, end: isoDate, planned_minutes: "number", measured: "boolean", breaks: "array", text: "string" };
const COACH_FEEDBACK = {
  now: isoDate, words: inner({ activity: "string", workplace: "string" }), options: listOf({ id: "string", label: "string" }),
  sessions: listOf(FEEDBACK_SESSION), suggestions: stringList, precision: "array", answers: "number", progress: "array", threshold: "object",
  learning: "array", notes: stringList,
};
const WHO = { member_id: "string", display_name: "string", mine: "boolean" };
const CIRCLE_THREADS = {
  configured: "boolean", circle_id: "string?", member_id: "string?", source: "string", offline: "boolean", members: listOf(WHO),
  threads: listOf({
    thread_id: "string", legs: stringList, title: "string", mine: "boolean", starts_at: isoDateOrNull, ends_at: isoDateOrNull, location: "string?",
    state: "string", invitees: listOf(WHO), waiting_on: listOf(WHO), last_activity: isoDateOrNull, pending_action_id: "string?", messages_count: "number",
  }),
};
const CIRCLE_MEMBERS = { configured: "boolean", circle_id: "string?", member_id: "string?", connect: "array", members: listOf({ ...WHO, has_box: "boolean" }) };
const DISPOSAL = { item_id: "string", ts: isoDate, day: "string", label: "string", item: "string", bin: "string", zone: "string", correct: "boolean", belongs_in: "string", rule: "string", frame_ref: "string?", in_practice: "boolean" };
const IMPACT_STORY = {
  now: isoDate, week: inner({ start: "string", end: "string", today: "string" }), empty: "boolean", empty_text: "string?", rules: "object", total: "object",
  this_week: "object", equivalents: "object", contamination: "object", feed: listOf(DISPOSAL), goal: "object",
  days: listOf({ day: "string", disposals: "number", right: "number", wrong: "number", clean: "boolean" }), clean_streak: "number", milestones: "object", practice: "object",
};
const IMPACT_PRACTICE = {
  week_start: "string", member_id: "string?", size: "number", how: "string", rules: inner({ file: "string", name: "string", place: "string?" }),
  items: listOf({ item: "string", mistakes: "number", asked: "boolean", last_wrong_at: isoDateOrNull, went_to: "string?", belongs_in: "string", zone: "string", rule: "string", frame_ref: "string?", members: stringList }),
};
const PLACE_ROUTE = {
  from: "string", to: "string", from_name: "string", to_name: "string", mode: oneOf("arrive_by", "depart"), depart: isoDate, arrive: isoDate, minutes: "number",
  walk_minutes: "number", meters: "number?", path: stringList, polyline: "array", sentence: "string", from_xy: pairOrNull, to_xy: pairOrNull, attribution: "string?",
};
const PLACE_MAP = { attribution: "string", pack: "string", bbox: "array", origin: "array", unit: "number", buildings: "array", areas: "array", roads: "array", walk: "object", places: "object" };
const ROOM = { name: "string", title: "string", size: "array", outline: "array", doors: "array", windows: "array", furniture: listOf({ id: "string", type: "string", at: "array", size: "array" }), calibration: "object", play: "object", zones: "object", rooms: "array", uncalibrated: stringList };
const ROOM_LIVE = { room: "string", ts: isoDateOrNull, tracks: "array", seen: "object", round: "object?", result: "object?" };
const ROOM_ROUTE = { item: "string", bin: "string", bin_id: "string", zone: "string", room: "string", from: inner({ x: "number", y: "number", track: "string?" }), reachable: "boolean", points: "array", distance_m: "number", par_s: "number" };
const NO_ROOM = { 404: "the room twin is not set up (no floor plan)" };

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

console.log("\nNow (the day in words, the coach's questions)");
const NO_DAY = { 404: "the Day loop is not on this box (it needs the place packs)" };
const brief = await check("GET /api/day/brief", "GET", "/api/day/brief", BRIEF, { optional: NO_DAY });
if (brief) console.log(`        ${brief.engine} engine; said so far: ${Object.keys(brief.latest).join(", ") || "nothing"}`);
const coachAsks = await check("GET /api/day/coach/feedback", "GET", "/api/day/coach/feedback", COACH_FEEDBACK, { optional: NO_DAY });
if (coachAsks) console.log(`        ${coachAsks.sessions.length} sessions, ${coachAsks.sessions.reduce((n, x) => n + x.breaks.length, 0)} breaks to ask about, ${coachAsks.answers} answered`);
{
  // GET /api/day/coach carries the same view as `feedback` (null on a box without the feedback loop)
  const r = await call("GET", "/api/day/coach");
  if (r.status === 404) skip("GET /api/day/coach .feedback", `HTTP 404: ${r.data?.detail ?? NO_DAY[404]}`);
  else if (r.status !== 200 || !("feedback" in (r.data ?? {}))) fail("GET /api/day/coach .feedback", `HTTP ${r.status}: no feedback field`);
  else if (r.data.feedback === null) ok("GET /api/day/coach .feedback", "null: the feedback loop is not wired");
  else {
    const bad = problems(r.data.feedback, COACH_FEEDBACK, "feedback");
    if (bad.length) fail("GET /api/day/coach .feedback", bad.join("; "));
    else ok("GET /api/day/coach .feedback");
  }
}

console.log("\nImpact");
const NO_IMPACT = { 404: "Waste Watch is not configured" };
const story = await check("GET /api/impact/story", "GET", "/api/impact/story", IMPACT_STORY, { optional: NO_IMPACT });
if (story) console.log(`        ${story.empty ? "nothing counted yet" : `${story.feed.length} disposals in the feed, ${story.clean_streak} clean days in a row`}`);
const practice = await check("GET /api/impact/practice", "GET", "/api/impact/practice", IMPACT_PRACTICE, { optional: NO_IMPACT });
if (practice) console.log(`        ${practice.size} to practise${practice.items[0] ? `, first: ${practice.items[0].item} -> ${practice.items[0].belongs_in}` : ""}`);

console.log("\nPeople");
const NO_IDENTITY = { 404: "owner recognition is not wired into this gateway" };
await check("GET /identity/settings", "GET", "/identity/settings", { identity_enabled: "boolean", available: "boolean", recognizer: "string?" }, { optional: NO_IDENTITY });
await check("GET /identity/members", "GET", "/identity/members", { member_id: "string", name: "string", enrolled_at: "string" }, { list: true, optional: NO_IDENTITY });
await check("GET /identity/enroll/status", "GET", "/identity/enroll/status", { state: "string", frames_needed: "number" }, { optional: NO_IDENTITY });
await check("GET /identity/settings (face profile)", "GET", "/identity/settings", { everyone: "boolean", expressions_available: "boolean", expression_reader: "string?", photos_available: "boolean" }, { optional: NO_IDENTITY });
await check("GET /identity/present", "GET", "/identity/present", { present: "array" }, { optional: NO_IDENTITY });
await check("GET /identity/expressions", "GET", "/identity/expressions", { member_id: "string?", hours: "number", samples: "number", looked: "object", strained_share: "number?", windows: "array", note: "string" }, { optional: NO_IDENTITY });

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
const calendar = await check("GET /api/circle/events", "GET", "/api/circle/events", CIRCLE_EVENTS);
if (calendar) console.log(`        ${calendar.events.length} upcoming, from ${calendar.source}${calendar.offline ? " (shared calendar offline)" : ""}`);
const sharing = await check("GET /api/circle/sharing", "GET", "/api/circle/sharing", CIRCLE_SHARING);
if (sharing) console.log(`        summaries ${sharing.sharing_now ? "on" : "off"}; ${sharing.queued} queued, ${sharing.sent} sent; database ${sharing.database}`);
const threads = await check("GET /api/circle/threads", "GET", "/api/circle/threads", CIRCLE_THREADS);
if (threads) console.log(`        ${threads.threads.length} conversations (${threads.source}${threads.offline ? ", shared store offline" : ""}): ${threads.threads.map((t) => `${t.title} [${t.state}]`).join(", ") || "none"}`);
const circleMembers = await check("GET /api/circle/members", "GET", "/api/circle/members", CIRCLE_MEMBERS);
if (circleMembers) console.log(`        ${circleMembers.members.length} members${circleMembers.configured ? "" : " (the Circle is not configured)"}`);
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

const roomRoute = await check("GET /api/play/room-route?item=can", "GET", "/api/play/room-route?item=can", ROOM_ROUTE, { optional: NO_ROOM });
if (roomRoute) console.log(`        can -> ${roomRoute.bin} in the ${roomRoute.room} room; ${roomRoute.distance_m} m, par ${roomRoute.par_s} s${roomRoute.reachable ? "" : " (not reachable)"}`);

console.log("\nRoom twin");
const room = await check("GET /api/room", "GET", "/api/room", ROOM, { optional: NO_ROOM });
if (room) console.log(`        ${room.title}: ${room.furniture.length} pieces of furniture, cameras calibrated: ${Object.keys(room.calibration).join(", ") || "none"}`);
await check("GET /api/room/live", "GET", "/api/room/live", ROOM_LIVE, { optional: NO_ROOM });

console.log("\nRoom map");
for (const label of ["keys", "wallet", "charger", "laptop", "phone"]) {
  const s = await check(`GET /memory/objects/${label}/last-seen`, "GET", `/memory/objects/${label}/last-seen`, SIGHTING, { optional: { 404: "never seen" } });
  if (s) console.log(`        ${s.camera_id} / ${s.zone_id ?? "no zone"}: ${s.position ?? "no position text"}`);
}
{
  const r = await call("GET", "/api/zones");
  // camera -> zone -> polygon of [x, y] in 0..1 (lib/useZones.ts reads it; the copy embedded at build time is the fallback)
  const polygon = (p) => Array.isArray(p) && p.length >= 3 && p.every((xy) => Array.isArray(xy) && xy.length === 2 && xy.every((n) => typeof n === "number" && n >= 0 && n <= 1));
  const cameras = typeOf(r.data) === "object" ? Object.entries(r.data) : [];
  if (r.status === 404) ok("GET /api/zones", "404: this box does not serve its zones, so the app uses the copy embedded at build time");
  else if (r.status !== 200 || typeOf(r.data) !== "object") fail("GET /api/zones", `HTTP ${r.status}: ${JSON.stringify(r.data)?.slice(0, 200)}`);
  else if (!cameras.every(([, zones]) => typeOf(zones) === "object" && Object.values(zones).every(polygon))) fail("GET /api/zones", "expected {camera: {zone: [[x, y], ...]}} with every point in 0..1");
  else ok("GET /api/zones", `${cameras.length} cameras, ${cameras.reduce((n, [, z]) => n + Object.keys(z).length, 0)} zones`);
}

console.log("\nPlaces");
const expect = (name, cond, note) => (cond ? ok(name) : fail(name, note));
const NO_DAYRUN = { 404: "Day Run is not configured (it needs the place packs)" };
const packs = await check("GET /api/places", "GET", "/api/places", PLACE_PACK, { list: true, optional: { 404: "no place packs loaded" } });
if (packs) {
  const located = packs.flatMap((p) => Object.values(p.places)).filter((p) => typeof p.lat === "number" && typeof p.lon === "number").length;
  const all = packs.reduce((n, p) => n + Object.keys(p.places).length, 0);
  console.log(`        ${packs.length} packs, ${all} places, ${located} with coordinates (the rest are listed beside the map)`);
}

{
  // "When to leave": the walk between the first two places that have coordinates, and the map it is drawn on.
  const mapped = (packs ?? []).find((p) => typeof p.map === "string");
  const NO_MAP = { 404: "no place pack has a drawn map" };
  if (!mapped) skip("GET /api/places/route, GET /api/places/map/{pack}", NO_MAP[404]);
  else {
    const located = Object.entries(mapped.places).filter(([, p]) => typeof p.lat === "number" && typeof p.lon === "number").map(([id]) => `${mapped.name}.${id}`);
    const [a, b] = located;
    if (a && b) {
      const arriveBy = new Date(Date.now() + 3600_000).toISOString();
      const r = await check(`GET /api/places/route?from=${a}&to=${b}&arrive_by=`, "GET", `/api/places/route?from=${encodeURIComponent(a)}&to=${encodeURIComponent(b)}&arrive_by=${encodeURIComponent(arriveBy)}`, PLACE_ROUTE);
      if (r) {
        expect("  the route is for arriving by the time asked", r.mode === "arrive_by" && Date.parse(r.arrive) === Date.parse(arriveBy) && Date.parse(r.depart) <= Date.parse(r.arrive), `${r.mode}: ${r.depart} to ${r.arrive}`);
        console.log(`        ${r.sentence}`);
      }
    }
    await check(`GET /api/places/map/${mapped.name}`, "GET", `/api/places/map/${mapped.name}`, PLACE_MAP, { optional: NO_MAP });
  }
}

console.log("\nInspection (SPOT)");
// Every GET answers 200 also without a kit ({configured: false, note}); 404 only on a gateway without the context.
const NO_SPOT = { 404: "the inspection context is not in this gateway" };
const spotOverview = await check("GET /api/spot/overview", "GET", "/api/spot/overview", { configured: "boolean", model: "object" }, { optional: NO_SPOT });
const spotTriage = await check("GET /api/spot/triage", "GET", "/api/spot/triage", { configured: "boolean", items: "array", uninspected: "array", next: "array" }, { optional: NO_SPOT });
if (spotTriage) console.log(`        ${spotOverview?.configured ? `${spotTriage.items.length} assets ranked, ${spotTriage.uninspected.length} uninspected` : "no kit on this box: the screen says how to add one"}`);
await check("GET /api/spot/findings", "GET", "/api/spot/findings", { configured: "boolean", findings: "array" }, { optional: NO_SPOT });
await check("GET /api/spot/assets", "GET", "/api/spot/assets", { configured: "boolean", assets: "array" }, { optional: NO_SPOT });
await check("GET /api/spot/locations", "GET", "/api/spot/locations", { configured: "boolean", locations: "array", plan: "object?" }, { optional: NO_SPOT });
await check("GET /api/spot/handoff", "GET", "/api/spot/handoff", { configured: "boolean", lines: "array", text: "string" }, { optional: NO_SPOT });
await check("GET /api/spot/assumptions", "GET", "/api/spot/assumptions", { configured: "boolean", assumptions: "array" }, { optional: NO_SPOT });
await check("GET /api/spot/validation", "GET", "/api/spot/validation", { configured: "boolean", rows: "array", issues: "array" }, { optional: NO_SPOT });
await check("GET /api/spot/work_orders", "GET", "/api/spot/work_orders", { configured: "boolean", work_orders: "array", folder: "string" }, { optional: NO_SPOT });
await check("GET /api/spot/replay/state", "GET", "/api/spot/replay/state", { configured: "boolean", index: "number", total: "number", playing: "boolean" }, { optional: NO_SPOT });

console.log("\nDay Run (scenario demo)");
const board = await check("GET /api/dayrun/board?scenario=demo", "GET", "/api/dayrun/board?scenario=demo", BOARD, { optional: NO_DAYRUN });
if (board) {
  console.log(`        ${board.slots.length} slots, ${board.blocks.length} fixed blocks, ${board.deadlines.length} deadlines, ${board.friends.length} friends, ${board.events.length} optional events`);
  const known = new Set((packs ?? []).flatMap((p) => Object.keys(p.places).map((id) => `${p.name}.${id}`)));
  const refs = [board.start_place, board.home, ...board.slots.flatMap((s) => s.moves.map((m) => m.place)), ...board.friends.map((f) => f.place)].filter(Boolean);
  if (packs) expect("  every place the board names is in GET /api/places", refs.every((r) => known.has(r)), `unknown: ${[...new Set(refs.filter((r) => !known.has(r)))].join(", ")}`);
  const plan = (moves, extra = {}) => ({ scenario: "demo", start: board.start, moves, ...extra });

  const empty = await check("POST /api/dayrun/simulate (no moves)", "POST", "/api/dayrun/simulate", SIM, { body: plan([]) });
  if (empty) expect("  the timeline covers the board", empty.timeline.length > 1 && empty.timeline[0].t === empty.start && empty.timeline.at(-1).t === empty.end, `${empty.timeline.length} ticks, ${empty.timeline[0]?.t} to ${empty.timeline.at(-1)?.t}`);

  const bad = await check("POST /api/dayrun/simulate (bad plan)", "POST", "/api/dayrun/simulate", SIM, { body: plan(DEMO_BAD_PLAN) });
  if (bad) {
    expect("  the bad plan fails with no stars", !bad.outcome.won && bad.stars === 0 && bad.outcome.reasons.length > 0, `won=${bad.outcome.won} stars=${bad.stars}`);
    expect("  the lunch run is rejected with a reason", bad.rejected.length === 1 && bad.rejected[0].index === 1 && /miss the 12:00/.test(bad.rejected[0].reason), JSON.stringify(bad.rejected).slice(0, 200));
    console.log(`        ${bad.outcome.summary}`);
    console.log(`        rejected: ${bad.rejected.map((r) => r.reason).join("; ")}`);
  }

  const good = await check("POST /api/dayrun/simulate (good plan)", "POST", "/api/dayrun/simulate", SIM, { body: plan(DEMO_GOOD_PLAN) });
  if (good) {
    expect("  the good plan wins three stars", good.outcome.won && good.stars === 3 && good.rejected.length === 0, `won=${good.outcome.won} stars=${good.stars} rejected=${good.rejected.length}`);
    expect("  every leg belongs to a route", good.legs.length > 0 && good.legs.every((l) => good.routes.some((r) => r.index === l.route)), "a leg names a route that is not listed");
    expect("  every accepted move has its block", DEMO_GOOD_PLAN.every((_, i) => good.blocks.some((b) => !b.fixed && b.move === i && b.id === `m${i}`)), "blocks are not m<index> per move");
    const drawn = good.legs.filter((l) => l.from_xy && l.to_xy).length;
    console.log(`        ${good.outcome.summary}`);
    console.log(`        ${good.routes.length} routes, ${good.legs.length} legs (${drawn} with coordinates, ${good.legs.length - drawn} indoors), ${good.timeline.length} ticks`);
  }

  const hard = await check("POST /api/dayrun/hard (scripted disruptions)", "POST", "/api/dayrun/hard", HARD, { body: plan(DEMO_GOOD_PLAN, { disruptions: DEMO_DISRUPTIONS }) });
  if (hard) {
    expect("  both disruptions come back with their text", hard.disruptions.length === 2 && hard.disruptions.every((d) => d.text.length > 0), JSON.stringify(hard.disruptions).slice(0, 200));
    expect("  the plan is re-routed and shortened", ["rerouted", "shortened", "route_changed"].every((k) => hard.changes.some((c) => c.kind === k)), hard.changes.map((c) => c.kind).join(", "));
    console.log(`        before: ${hard.before.outcome.summary} (${hard.before.totals.walk_minutes} min walking)`);
    console.log(`        after:  ${hard.after.outcome.summary} (${hard.after.totals.walk_minutes} min walking)`);
    for (const c of hard.changes) console.log(`        [${c.kind}] ${c.text}`);
  }
  const seeded = await check("POST /api/dayrun/hard (seed 7, count 2)", "POST", "/api/dayrun/hard", HARD, { body: plan(DEMO_GOOD_PLAN, { seed: 7, count: 2 }) });
  if (seeded) expect("  the seed is echoed and picks two disruptions", seeded.seed === 7 && seeded.disruptions.length === 2, `seed=${seeded.seed}, ${seeded.disruptions.length} disruptions`);

  // Commit: a failing plan is refused with the reasons; a winning one becomes an approval card, or 409 when no calendar can be written.
  const refused = await call("POST", "/api/dayrun/commit", plan(DEMO_BAD_PLAN));
  const detail = refused.data?.detail;
  if (refused.status === 401) skip("POST /api/dayrun/commit (bad plan)", "the box asks for a token: set ATLAS_TOKEN");
  else expect("POST /api/dayrun/commit (bad plan) -> 422 {message, reasons}", refused.status === 422 && typeof detail?.message === "string" && Array.isArray(detail?.reasons) && detail.reasons.length > 0, `HTTP ${refused.status}: ${JSON.stringify(refused.data)?.slice(0, 200)}`);
  let committed = false;
  if (WRITES) {
    const c = await call("POST", "/api/dayrun/commit", plan(DEMO_GOOD_PLAN));
    if (c.status === 409 && typeof c.data?.detail === "string") ok("POST /api/dayrun/commit (good plan)", `409: ${c.data.detail}; the app shows this message`);
    else if (c.status === 200) {
      const wrong = problems(c.data, COMMIT, "body");
      const pending = (await call("GET", "/api/approvals")).data;
      const card = Array.isArray(pending) ? pending.find((p) => p.id === c.data.action_id) : undefined;
      if (wrong.length) fail("POST /api/dayrun/commit (good plan)", wrong.join("; "));
      else if (!card) fail("POST /api/dayrun/commit (good plan)", `no approval card ${c.data.action_id} in GET /api/approvals`);
      else {
        committed = true;
        ok("POST /api/dayrun/commit (good plan)", `approval card ${card.kind} ${card.target}: ${c.data.reason}`);
        await check("POST /api/approvals/{id} deny (the Day Run card)", "POST", `/api/approvals/${c.data.action_id}`, APPROVAL, { body: { decision: "deny" } });
      }
    } else fail("POST /api/dayrun/commit (good plan)", `HTTP ${c.status}: ${JSON.stringify(c.data)?.slice(0, 200)}`);
  } else skip("POST /api/dayrun/commit (good plan)", "run with --writes");

  const day = board.start.slice(0, 10);
  const review = await call("GET", `/api/dayrun/review?day=${day}&scenario=demo`);
  if (review.status === 404 && typeof review.data?.detail === "string") {
    if (committed) fail("GET /api/dayrun/review", `a plan was just committed for ${day} but the box says: ${review.data.detail}`);
    else ok("GET /api/dayrun/review", `404: ${review.data.detail}; the app shows this message`);
  } else if (review.status === 200) {
    const wrong = problems(review.data, REVIEW, "body");
    if (wrong.length) fail("GET /api/dayrun/review", wrong.join("; "));
    else ok("GET /api/dayrun/review", `score ${review.data.score}: ${review.data.text}`.slice(0, 170));
  } else fail("GET /api/dayrun/review", `HTTP ${review.status}: ${JSON.stringify(review.data)?.slice(0, 200)}`);

  const replay = await check("POST /api/dayrun/replay", "POST", "/api/dayrun/replay", REPLAY, { body: { scenario: "demo", day, moves: DEMO_GOOD_PLAN } });
  if (replay) console.log(`        ${replay.points_gained >= 0 ? "+" : ""}${replay.points_gained} points, ${replay.stars_gained >= 0 ? "+" : ""}${replay.stars_gained} stars`);
  const campaign = await check("POST /api/dayrun/campaign", "POST", "/api/dayrun/campaign", CAMPAIGN, { body: { scenario: "demo", days: DEMO_CAMPAIGN } });
  if (campaign) {
    expect("  two days come back, each with its own result", campaign.days.length === 2, `${campaign.days.length} days`);
    console.log(`        ${campaign.won ? "won" : "lost"}, ${campaign.stars} stars, ${campaign.points} points`);
  }
  const forecast = await check("POST /api/dayrun/forecast (measured given)", "POST", "/api/dayrun/forecast", FORECAST, {
    body: { estimates: [{ block: "m3", minutes: 120 }, { block: "m5", minutes: 60 }], measured: { m3: 170, m5: 90 } },
  });
  if (forecast) console.log(`        ${forecast.text}`);
  const fromReview = await call("POST", "/api/dayrun/forecast", { estimates: [{ block: "m3", minutes: 120 }], day, scenario: "demo" });
  if (fromReview.status === 200) {
    const wrong = problems(fromReview.data, FORECAST, "body");
    if (wrong.length) fail("POST /api/dayrun/forecast (from the review)", wrong.join("; "));
    else ok("POST /api/dayrun/forecast (from the review)", `${fromReview.data.rows.length} scored, ${fromReview.data.unmeasured.length} unmeasured`);
  } else if (fromReview.status === 404 && typeof fromReview.data?.detail === "string") ok("POST /api/dayrun/forecast (from the review)", `404: ${fromReview.data.detail}; the app shows this message`);
  else fail("POST /api/dayrun/forecast (from the review)", `HTTP ${fromReview.status}: ${JSON.stringify(fromReview.data)?.slice(0, 200)}`);

  console.log("\nDay Run (my day: live sources)");
  const mine = await check("GET /api/dayrun/board", "GET", "/api/dayrun/board", BOARD, { optional: NO_DAYRUN });
  if (mine) {
    console.log(`        ${mine.slots.length} slots, ${mine.blocks.length} fixed blocks, ${mine.deadlines.length} deadlines, ${mine.friends.length} friends; starting values from ${mine.initial.source}`);
    const r = await check("POST /api/dayrun/simulate (my day, no moves)", "POST", "/api/dayrun/simulate", SIM, { body: { start: mine.start, moves: [] } });
    if (r) expect("  simulating from the board's own start gives the same window", r.start === mine.start && r.end === mine.end, `${r.start} to ${r.end} vs ${mine.start} to ${mine.end}`);
    await check("POST /api/dayrun/hard (my day, seed 7)", "POST", "/api/dayrun/hard", HARD, { body: { start: mine.start, moves: [], seed: 7 } });
  }
}

ws?.close();
console.log(failures ? `\nFAILED: ${failures} check${failures === 1 ? "" : "s"} did not match what the app expects.` : "\nAll checks passed.");
process.exit(failures ? 1 : 0);
