/**
 * Day Run playback: turns the results the box already returns into something
 * that can be played back minute by minute.
 *
 * Nothing here simulates. Where the character stands, which way it walks, what
 * the meters read and why a move was rejected all come from `timeline`,
 * `blocks`, `routes`, `legs`, `rejected` and (in hard mode) `before`, `after`,
 * `disruptions` and `changes`. This file only interpolates between the rows.
 *
 * Three things the API does not say, and what the playback does instead:
 *  - WHEN a disruption becomes known. It is revealed at the first moment the
 *    calm day and the disrupted day differ (`splitAt`).
 *  - The walkway graph. /api/places returns no edges, so the network drawn on
 *    the map is the set of place-to-place hops seen in the routes the box sent
 *    (the day's legs and every option on the board).
 *  - The low-energy line. `ENERGY_LOW` repeats config/dayrun.yaml stars.energy_floor.
 */

import type { Board, Change, Disruption, FriendAt, HardResult, MoveOption, PlannedBlock, Rejected, RouteTaken, SimResult, Tick } from "./api";
import { clock, minutes, ms, parentPlace, placeName, type PlaceIndex } from "./dayrun";

export const ENERGY_LOW = 25;
const MIN = 60000;

export interface Pt {
  x: number;
  y: number;
}

// ---------- the world: places with coordinates, in one flat plane ----------

export interface World {
  /** Located places by "pack.place", in world units (the longer side of the campus is 1000). */
  at: Map<string, Pt>;
  /** A place's own point, or the point of the campus place that leads into its pack (a room -> its dorm). */
  resolve: (ref: string) => Pt | undefined;
  /** True when the place itself has no coordinates (a room). */
  indoors: (ref: string) => boolean;
}

export function buildWorld(places: PlaceIndex): World {
  const located = [...places.values()].filter((p) => typeof p.lat === "number" && typeof p.lon === "number");
  const at = new Map<string, Pt>();
  if (located.length > 0) {
    // Equirectangular around the middle latitude, as on the campus map: fine at this scale.
    const lats = located.map((p) => p.lat as number);
    const lons = located.map((p) => p.lon as number);
    const latMax = Math.max(...lats);
    const latMin = Math.min(...lats);
    const lonMin = Math.min(...lons);
    const lonMax = Math.max(...lons);
    const k = Math.cos((((latMin + latMax) / 2) * Math.PI) / 180);
    const span = Math.max((lonMax - lonMin) * k, latMax - latMin, 1e-6);
    const scale = 1000 / span;
    for (const p of located) at.set(p.ref, { x: ((p.lon as number) - lonMin) * k * scale, y: (latMax - (p.lat as number)) * scale });
  }
  const resolve = (ref: string) => at.get(ref) ?? at.get(parentPlace(places, ref)?.ref ?? "");
  return { at, resolve, indoors: (ref) => !at.has(ref) };
}

/** Place-to-place hops seen in any route the box returned: the walkways we know exist. */
export function knownEdges(world: World, results: SimResult[], board: Board | undefined): [string, string][] {
  const seen = new Map<string, [string, string]>();
  const add = (a: string | undefined, b: string | undefined) => {
    if (!a || !b || a === b || !world.at.has(a) || !world.at.has(b)) return;
    const key = a < b ? `${a}|${b}` : `${b}|${a}`;
    if (!seen.has(key)) seen.set(key, a < b ? [a, b] : [b, a]);
  };
  for (const r of results) for (const l of r.legs) add(l.from, l.to);
  for (const s of board?.slots ?? []) for (const m of s.moves) for (let i = 1; i < m.route.length; i++) add(m.route[i - 1], m.route[i]);
  return [...seen.values()];
}

// ---------- one result as a script ----------

export interface Hop {
  from: string;
  to: string;
  a: Pt;
  b: Pt;
  minutes: number;
  indoors: boolean;
  m0: number; // minutes walked when this hop starts
}

export interface Walk {
  route: RouteTaken;
  t0: number;
  t1: number;
  hops: Hop[];
  total: number; // minutes, the sum of the hops
  /** The outdoor polyline, in world units, without repeated points. */
  points: Pt[];
  names: string[];
}

export interface Span {
  block: PlannedBlock;
  s: number;
  e: number;
}

export interface Script {
  result: SimResult;
  t0: number;
  t1: number;
  walks: Walk[];
  spans: Span[];
  startPlace: string;
}

export function buildScript(result: SimResult, world: World, places: PlaceIndex): Script {
  const t0 = ms(result.start);
  const t1 = ms(result.end);
  const startPlace = result.timeline[0]?.place ?? result.routes[0]?.places[0] ?? "";
  const walks: Walk[] = [];
  let last: Pt = world.resolve(startPlace) ?? { x: 500, y: 500 };
  for (const route of [...result.routes].sort((a, b) => ms(a.depart) - ms(b.depart))) {
    const hops: Hop[] = [];
    let m0 = 0;
    for (const leg of result.legs) {
      if (leg.route !== route.index) continue;
      const a = world.resolve(leg.from) ?? last;
      const b = world.resolve(leg.to) ?? a;
      hops.push({ from: leg.from, to: leg.to, a, b, minutes: leg.minutes, indoors: leg.from_xy === null || leg.to_xy === null, m0 });
      m0 += leg.minutes;
      last = b;
    }
    const points: Pt[] = [];
    for (const h of hops) for (const p of [h.a, h.b]) if (!points.length || Math.hypot(p.x - (points[points.length - 1] as Pt).x, p.y - (points[points.length - 1] as Pt).y) > 0.5) points.push(p);
    walks.push({ route, t0: ms(route.depart), t1: Math.max(ms(route.arrive), ms(route.depart) + 1), hops, total: m0, points, names: route.places.map((p) => placeName(places, p)) });
  }
  const spans = result.blocks.map((block) => ({ block, s: ms(block.start), e: ms(block.end) })).sort((a, b) => a.s - b.s);
  return { result, t0, t1, walks, spans, startPlace };
}

/** A point `f` (0..1) of the way along a walk, by walking minutes. */
export function alongWalk(walk: Walk, f: number): { pt: Pt; hop: Hop | undefined; index: number; dx: number; dy: number } {
  const first = walk.hops[0];
  if (!first) return { pt: walk.points[0] ?? { x: 0, y: 0 }, hop: undefined, index: 0, dx: 0, dy: 0 };
  const m = Math.max(0, Math.min(1, f)) * walk.total;
  let index = walk.hops.length - 1;
  for (let i = 0; i < walk.hops.length; i++) {
    const h = walk.hops[i] as Hop;
    if (m < h.m0 + h.minutes) {
      index = i;
      break;
    }
  }
  const hop = walk.hops[index] as Hop;
  const g = hop.minutes > 0 ? Math.max(0, Math.min(1, (m - hop.m0) / hop.minutes)) : 1;
  return { pt: { x: hop.a.x + (hop.b.x - hop.a.x) * g, y: hop.a.y + (hop.b.y - hop.a.y) * g }, hop, index, dx: hop.b.x - hop.a.x, dy: hop.b.y - hop.a.y };
}

/** The part of a walk's outdoor polyline covered when `f` of it is done. */
export function walkedPoints(walk: Walk, f: number): Pt[] {
  const here = alongWalk(walk, f);
  const out: Pt[] = [];
  const push = (p: Pt) => {
    const q = out[out.length - 1];
    if (!q || Math.hypot(p.x - q.x, p.y - q.y) > 0.5) out.push(p);
  };
  for (let i = 0; i < here.index; i++) {
    const h = walk.hops[i] as Hop;
    push(h.a);
    push(h.b);
  }
  if (here.hop) push(here.hop.a);
  push(here.pt);
  return out;
}

export type Doing = "walk" | "eat" | "study" | "rest" | "sleep" | "class" | "event" | "idle";

const DOINGS: readonly Doing[] = ["walk", "eat", "study", "rest", "sleep", "class", "event", "idle"];
const asDoing = (kind: string): Doing => (DOINGS as readonly string[]).includes(kind) ? (kind as Doing) : "class";

export interface Meters {
  energy: number;
  focus: number;
  risk: number;
  workLeft: Record<string, number>;
  index: number; // the tick at or before `t`
}

export function metersAt(result: SimResult, t: number): Meters {
  const ticks = result.timeline;
  const first = ticks[0];
  if (!first) return { energy: 0, focus: 0, risk: 0, workLeft: {}, index: 0 };
  const step = result.tick_minutes * MIN || 15 * MIN;
  const raw = (t - ms(first.t)) / step;
  const index = Math.max(0, Math.min(ticks.length - 1, Math.floor(raw)));
  const a = ticks[index] as Tick;
  const b = ticks[Math.min(ticks.length - 1, index + 1)] as Tick;
  const g = Math.max(0, Math.min(1, raw - index));
  const mix = (x: number, y: number) => x + (y - x) * g;
  const workLeft: Record<string, number> = {};
  for (const [id, v] of Object.entries(a.work_left)) workLeft[id] = mix(v, b.work_left[id] ?? v);
  return { energy: mix(a.energy, b.energy), focus: mix(a.focus, b.focus), risk: mix(a.deadline_risk, b.deadline_risk), workLeft, index };
}

export interface Moment {
  t: number;
  script: Script;
  pt: Pt;
  place: string; // where you are, or the place you are walking towards
  doing: Doing;
  walk?: Walk;
  walkF: number;
  hop?: Hop;
  dx: number;
  span?: Span;
  meters: Meters;
}

export function momentAt(script: Script, t: number): Moment {
  const meters = metersAt(script.result, t);
  const span = script.spans.find((s) => s.s <= t && t < s.e);
  let place = script.startPlace;
  let walk: Walk | undefined;
  for (const w of script.walks) {
    if (w.t0 > t) break;
    if (t < w.t1) walk = w;
    else place = w.route.places[w.route.places.length - 1] ?? place;
  }
  if (walk) {
    const f = (t - walk.t0) / (walk.t1 - walk.t0);
    const here = alongWalk(walk, f);
    return { t, script, pt: here.pt, place: here.hop?.to ?? place, doing: "walk", walk, walkF: f, hop: here.hop, dx: here.dx, span, meters };
  }
  if (span) place = span.block.place || place;
  return { t, script, pt: FALLBACK, place, doing: span ? asDoing(span.block.kind) : "idle", walkF: 0, dx: 0, span, meters };
}

const FALLBACK: Pt = { x: 500, y: 500 };

// ---------- the show: one script, or the calm day spliced into the disrupted one ----------

export interface RouteCheck {
  ok: boolean;
  text: string;
}

export interface Hold {
  kind: "replan" | "reject" | "route";
  t: number;
  seconds: number;
  walk?: Walk;
  rejected?: Rejected;
  ghost?: { points: Pt[]; names: string[]; option: MoveOption };
}

export interface FeedItem {
  t: number;
  tone: "ok" | "warn" | "bad" | "info";
  text: string;
  place?: string;
}

export interface Show {
  t0: number;
  t1: number;
  main: Script;
  before: Script | null;
  /** Hard mode: the moment the disruptions arrive. Null on a calm day. */
  splitAt: number | null;
  disruptions: Disruption[];
  changes: Change[];
  hard: HardResult | null;
  holds: Hold[];
  feed: FeedItem[];
  /** Times worth stopping at when stepping: every block start, departure, arrival, the split and the end. */
  stops: number[];
  /** Everything the day touches, for framing the camera. */
  extent: Pt[];
}

const scriptFor = (show: Show, t: number): Script => (show.before && show.splitAt !== null && t < show.splitAt ? show.before : show.main);

export function showMoment(show: Show, world: World, t: number): Moment {
  const m = momentAt(scriptFor(show, t), t);
  return m.walk ? m : { ...m, pt: world.resolve(m.place) ?? m.pt };
}

/** The first moment the two days differ: a block or a walk that starts, ends or happens somewhere else. */
function firstDifference(a: Script, b: Script): number | null {
  const parts = (s: Script) =>
    [...s.spans.map((x) => ({ s: x.s, key: `b|${x.block.id}|${x.block.place}|${x.s}|${x.e}|${x.block.with_friend ?? ""}` })), ...s.walks.map((w) => ({ s: w.t0, key: `w|${w.route.places.join(">")}|${w.t0}|${w.t1}` }))].sort(
      (x, y) => x.s - y.s || (x.key < y.key ? -1 : 1),
    );
  const pa = parts(a);
  const pb = parts(b);
  for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
    const x = pa[i];
    const y = pb[i];
    if (x && y && x.key === y.key) continue;
    return Math.min(x?.s ?? Infinity, y?.s ?? Infinity);
  }
  return null;
}

/** Whether the next fixed block can still be reached once this walk is done, read from the routes that follow. */
export function routeCheck(script: Script, walk: Walk): RouteCheck | null {
  const next = script.spans.find((s) => s.block.fixed && s.s >= walk.t1 - MIN);
  if (!next) return null;
  const title = `the ${clock(next.block.start)} ${next.block.title}`;
  const dest = walk.route.places[walk.route.places.length - 1];
  if (dest === next.block.place && walk.t1 <= next.s + MIN) {
    const late = Math.round((walk.t1 - next.s) / MIN);
    const spare = Math.round((next.s - walk.t1) / MIN);
    return late > 0 ? { ok: false, text: `Late for ${title} by ${minutes(late)}` } : { ok: true, text: spare > 0 ? `Reaches ${title} with ${minutes(spare)} to spare` : `Reaches ${title} exactly on time` };
  }
  const onward = script.walks.find((w) => w.t0 >= walk.t1 && w.t1 <= next.s + MIN && w.route.places[w.route.places.length - 1] === next.block.place);
  if (onward) return { ok: true, text: `${title} stays reachable: leave by ${clock(onward.route.depart)}, ${minutes(onward.route.minutes)} on` };
  return { ok: true, text: `Next fixed: ${title}` };
}

function thresholdFeed(script: Script, from: number, to: number, board: Board | undefined): FeedItem[] {
  const out: FeedItem[] = [];
  const ticks = script.result.timeline;
  const target = script.result.outcome.checks.find((c) => board?.deadlines.some((d) => d.id === c.id))?.target ?? null;
  for (let i = 1; i < ticks.length; i++) {
    const a = ticks[i - 1] as Tick;
    const b = ticks[i] as Tick;
    const t = ms(b.t);
    if (t < from || t >= to) continue;
    const at = clock(b.t);
    if (a.energy > 0.5 && b.energy <= 0.5) out.push({ t, tone: "bad", text: `Energy hits 0 at ${at}: nothing left to work with` });
    else if (a.energy >= ENERGY_LOW && b.energy < ENERGY_LOW) out.push({ t, tone: "warn", text: `Energy drops under ${ENERGY_LOW} at ${at}` });
    if (a.energy < ENERGY_LOW && b.energy >= ENERGY_LOW) out.push({ t, tone: "ok", text: `Energy back over ${ENERGY_LOW} at ${at}` });
    if (target !== null) {
      if (a.deadline_risk > target && b.deadline_risk <= target) out.push({ t, tone: "ok", text: `Deadline back on track at ${at} (risk ${Math.round(b.deadline_risk * 100)}%)` });
      if (a.deadline_risk <= target && b.deadline_risk > target) out.push({ t, tone: "warn", text: `Deadline at risk from ${at} (risk ${Math.round(b.deadline_risk * 100)}%)` });
    }
  }
  for (const s of script.spans) {
    if (s.e >= from && s.e < to && s.block.kind === "eat") out.push({ t: s.e, tone: "ok", text: `Ate at ${s.block.place_name} (${minutes(s.block.minutes)})`, place: s.block.place });
    if (s.s >= from && s.s < to && s.block.kind === "sleep") out.push({ t: s.s, tone: "info", text: `Asleep at ${clock(s.block.start)}`, place: s.block.place });
    if (s.s >= from && s.s < to && s.block.fixed && s.s > script.t0) {
      const arrival = script.walks.find((w) => w.route.places[w.route.places.length - 1] === s.block.place && w.t1 > s.s && w.t0 < s.s);
      if (arrival) out.push({ t: s.s, tone: "bad", text: `Late for ${s.block.title}: arriving ${clock(arrival.route.arrive)}`, place: s.block.place });
      else out.push({ t: s.s, tone: "ok", text: `On time for the ${clock(s.block.start)} ${s.block.title}`, place: s.block.place });
    }
  }
  for (const e of board?.events ?? []) {
    const t = ms(e.start);
    if (t < from || t >= to) continue;
    const went = script.spans.some((s) => s.block.kind === "event" && s.s < ms(e.end) && s.e > t);
    if (!went) out.push({ t, tone: "info", text: `Skipped: ${e.title} (optional)`, place: e.place ?? undefined });
  }
  return out;
}

export function buildShow(args: { result: SimResult; hard: HardResult | null; world: World; places: PlaceIndex; board: Board | undefined }): Show {
  const { hard, world, places, board } = args;
  const main = buildScript(hard ? hard.after : args.result, world, places);
  const before = hard ? buildScript(hard.before, world, places) : null;
  const { t0, t1 } = main;
  const disruptions = hard?.disruptions ?? [];
  const changes = hard?.changes ?? [];
  const splitAt = before ? Math.max(t0, Math.min(t1, firstDifference(before, main) ?? t0)) : null;
  const split = splitAt ?? t0;

  const holds: Hold[] = [];
  const feed: FeedItem[] = [];
  if (before && splitAt !== null) {
    holds.push({ kind: "replan", t: splitAt, seconds: 5 + Math.min(4, changes.length) * 1.1 });
    for (const d of disruptions) feed.push({ t: splitAt, tone: "bad", text: d.text || d.kind.replace("_", " "), place: d.place ?? d.to_place ?? undefined });
    feed.push({ t: splitAt, tone: "warn", text: changes.length === 0 ? "Re-checked every move: nothing had to change" : `Re-planned the rest of the day: ${changes.length} ${changes.length === 1 ? "change" : "changes"}` });
    feed.push(...thresholdFeed(before, t0, splitAt, board));
    for (const w of before.walks) if (w.t0 < splitAt) holds.push(routeHold(w));
  }
  feed.push(...thresholdFeed(main, split, t1 + 1, board));
  for (const w of main.walks) if (w.t0 >= split) holds.push(routeHold(w));

  // A rejected move: shown when its slot opens, with the route the box tried (from the board's options).
  for (const r of main.result.rejected) {
    const slot = board?.slots.find((s) => s.id === r.move.slot);
    const t = Math.max(split, Math.min(t1, r.move.start ? ms(r.move.start) : slot ? ms(slot.start) : t0));
    const option = slot?.moves.find((o) => o.kind === r.move.kind && (o.place ?? null) === (r.move.place ?? null) && o.route.length > 1);
    let ghost: Hold["ghost"];
    if (option) {
      const there = option.route.map((ref) => world.resolve(ref)).filter((p): p is Pt => p !== undefined);
      const onward = slot?.to_place ? world.resolve(slot.to_place) : undefined;
      const round = slot?.to_place === option.route[0];
      const points = round ? [...there, ...there.slice(0, -1).reverse()] : onward ? [...there, onward] : there;
      if (points.length > 1) ghost = { points, names: option.route.map((ref) => placeName(places, ref)), option };
    }
    holds.push({ kind: "reject", t, seconds: 4.2, rejected: r, ghost });
    feed.push({ t, tone: "bad", text: `Rejected: ${r.reason}`, place: r.move.place ?? undefined });
  }

  const order = { replan: 0, reject: 1, route: 2 } as const;
  holds.sort((a, b) => a.t - b.t || order[a.kind] - order[b.kind]);
  feed.sort((a, b) => a.t - b.t);

  const stops = new Set<number>([t0, t1]);
  if (splitAt !== null) stops.add(splitAt);
  const live = (s: Script, from: number, to: number) => {
    for (const x of s.spans) if (x.s >= from && x.s < to) stops.add(x.s);
    for (const w of s.walks) if (w.t0 >= from && w.t0 < to) stops.add(w.t0).add(Math.min(w.t1, t1));
  };
  if (before) live(before, t0, split);
  live(main, split, t1);
  for (const h of holds) stops.add(h.t);

  const extent: Pt[] = [];
  for (const s of before ? [before, main] : [main]) {
    const start = world.resolve(s.startPlace);
    if (start) extent.push(start);
    for (const w of s.walks) extent.push(...w.points);
    for (const x of s.spans) {
      const p = world.resolve(x.block.place);
      if (p) extent.push(p);
    }
  }
  for (const h of holds) if (h.ghost) extent.push(...h.ghost.points);
  if (extent.length === 0) extent.push(...world.at.values());

  return { t0, t1, main, before, splitAt, disruptions, changes, hard: hard ?? null, holds, feed, stops: [...stops].sort((a, b) => a - b), extent };
}

function routeHold(walk: Walk): Hold {
  const outdoor = walk.hops.filter((h) => !h.indoors).length;
  return { kind: "route", t: walk.t0, seconds: outdoor === 0 ? 0.7 : Math.min(2.4, 1.1 + 0.3 * outdoor), walk };
}

/** Every walk of the day as it is actually lived: the calm day up to the split, the disrupted day after it. */
export function livedWalks(show: Show): Walk[] {
  if (!show.before || show.splitAt === null) return show.main.walks;
  const split = show.splitAt;
  return [...show.before.walks.filter((w) => w.t0 < split), ...show.main.walks.filter((w) => w.t0 >= split)];
}

// ---------- pace ----------

export const SPEEDS = [1, 12, 36, 72] as const; // simulated minutes per second
export const DEFAULT_SPEED = 36;

/**
 * Simulated minutes per real second at `t`. Standing still runs at the chosen speed,
 * the night runs three times faster, and a walk is slowed so it can be watched:
 * at the default speed a walk takes two to four seconds whatever its length.
 */
export function paceAt(show: Show, t: number, speed: number): number {
  const s = scriptFor(show, t);
  const walk = s.walks.find((w) => w.t0 <= t && t < w.t1);
  if (walk) {
    const span = (walk.t1 - walk.t0) / MIN;
    const outdoor = walk.hops.some((h) => !h.indoors);
    const seconds = (outdoor ? Math.max(2, Math.min(4, 1.2 + span * 0.16)) : 0.5) * Math.max(0.5, Math.min(1, DEFAULT_SPEED / speed));
    return Math.min(speed, span / seconds);
  }
  const span = s.spans.find((x) => x.s <= t && t < x.e);
  return span?.block.kind === "sleep" ? speed * 3 : speed;
}

// ---------- words ----------

/** "3 h 10 min", "2 d 14 h" */
export function countdown(msLeft: number): string {
  const total = Math.max(0, Math.round(msLeft / MIN));
  if (total >= 48 * 60) return `${Math.floor(total / 1440)} d ${Math.floor((total % 1440) / 60)} h`;
  return minutes(total);
}

/** The moment in the day whose clock reads `hhmm` (the first one at or after the start). */
export function timeOfClock(show: Show, hhmm: string | null | undefined): number | null {
  const m = hhmm ? /^(\d{1,2}):(\d{2})$/.exec(hhmm) : null;
  const start = /^(\d{2}):(\d{2})$/.exec(clock(show.main.result.start));
  if (!m || !start) return null;
  const want = Number(m[1]) * 60 + Number(m[2]);
  const from = Number(start[1]) * 60 + Number(start[2]);
  const t = show.t0 + ((((want - from) % 1440) + 1440) % 1440) * MIN;
  return t <= show.t1 ? t : null;
}

/** "09:30" on the board's own clock for a moment between ticks. */
export function clockAt(show: Show, t: number): { day: string; hhmm: string } {
  const start = /^(\d{2}):(\d{2})$/.exec(clock(show.main.result.start));
  const from = start ? Number(start[1]) * 60 + Number(start[2]) : 0;
  const total = from + Math.floor((t - show.t0) / MIN + 1e-6);
  const mod = ((total % 1440) + 1440) % 1440;
  return { day: total >= 1440 ? "+1" : "", hhmm: `${String(Math.floor(mod / 60)).padStart(2, "0")}:${String(mod % 60).padStart(2, "0")}` };
}

/** Friends on the board who are at their place at `t` (and have not cancelled). */
export function friendsAt(show: Show, friends: FriendAt[], t: number): FriendAt[] {
  const gone = new Set(show.splitAt !== null && t >= show.splitAt ? show.disruptions.filter((d) => d.kind === "friend_cancelled" && d.friend).map((d) => d.friend as string) : []);
  return friends.filter((f) => ms(f.start) <= t && t < ms(f.end) && !gone.has(f.member));
}
