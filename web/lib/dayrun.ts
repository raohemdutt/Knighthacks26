/**
 * Day Run display helpers. No data is made here: these only format and index
 * what the box returned.
 *
 * Day Run times are shown on the board's own clock (the offset the box sent,
 * e.g. -04:00), in 24-hour form, so they read the same as the box's messages
 * ("you miss the 12:00 by 3 minutes") whatever zone the phone is in.
 */

import type { Move, OpenSlot, Place, PlacePack, PlannedBlock, RouteTaken, SimResult } from "./api";

export const ms = (iso: string): number => new Date(iso).getTime();

/** "2026-10-11T09:30:00-04:00" -> "09:30" */
export function clock(iso: string | null | undefined): string {
  const m = iso ? /T(\d{2}:\d{2})/.exec(iso) : null;
  return m?.[1] ?? "";
}

const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

/** "2026-10-11T09:30:00-04:00" or "2026-10-11" -> "Sun Oct 11" (the date as written, no zone shift) */
export function dateLabel(iso: string | null | undefined): string {
  const m = iso ? /^(\d{4})-(\d{2})-(\d{2})/.exec(iso) : null;
  if (!m) return "";
  const d = new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3])));
  return `${WEEKDAYS[d.getUTCDay()]} ${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()}`;
}

/** "2026-10-11T09:30:00-04:00" -> "2026-10-11" */
export const dateOf = (iso: string): string => iso.slice(0, 10);

/** "Sun 09:30" */
export function dayClock(iso: string): string {
  return `${dateLabel(iso).slice(0, 3)} ${clock(iso)}`;
}

/** "09:30–11:00" */
export function clockSpan(start: string, end: string): string {
  return `${clock(start)}–${clock(end)}`;
}

/** 95 -> "1 h 35 min", 60 -> "1 h", 4 -> "4 min", 21.6 -> "22 min" */
export function minutes(n: number): string {
  const total = Math.round(n);
  if (total === 0 && n > 0) return "under 1 min";
  if (total < 60) return `${total} min`;
  const h = Math.floor(total / 60);
  const m = total % 60;
  return m ? `${h} h ${m} min` : `${h} h`;
}

/** 0.642 -> "64%" */
export const riskPercent = (risk: number): string => `${Math.round(risk * 100)}%`;

export function stars(n: number): string {
  return n > 0 ? "★".repeat(n) + "☆".repeat(Math.max(0, 3 - n)) : "☆☆☆";
}

// ---------- places ----------

export type PlaceIndex = Map<string, Place & { ref: string; pack_name: string }>;

/** Every place of every pack by its "pack.place" reference. */
export function indexPlaces(packs: PlacePack[]): PlaceIndex {
  const out: PlaceIndex = new Map();
  for (const pack of packs) for (const [id, place] of Object.entries(pack.places)) out.set(`${pack.name}.${id}`, { ...place, ref: `${pack.name}.${id}`, pack_name: pack.name });
  return out;
}

export function placeName(index: PlaceIndex, ref: string | null | undefined): string {
  if (!ref) return "";
  return index.get(ref)?.name ?? ref;
}

/** The place that leads into a nested pack (the dorm on campus for a room in the apartment), if any. */
export function parentPlace(index: PlaceIndex, ref: string): (Place & { ref: string }) | undefined {
  const pack = ref.split(".", 1)[0];
  for (const p of index.values()) if (p.pack === pack) return p;
  return undefined;
}

// ---------- plans ----------

/** Insert a move at the end of its slot's run, keeping slots in board order. */
export function addMove(moves: Move[], move: Move, slots: OpenSlot[]): Move[] {
  const order = slots.map((s) => s.id);
  const rank = (slot: string) => {
    const i = order.indexOf(slot);
    return i < 0 ? order.length : i;
  };
  let at = moves.length;
  for (let i = 0; i < moves.length; i++) {
    const m = moves[i];
    if (m && rank(m.slot) > rank(move.slot)) {
      at = i;
      break;
    }
  }
  return [...moves.slice(0, at), move, ...moves.slice(at)];
}

/** Swap a move with its neighbour in the same slot. `dir` is -1 (earlier) or 1 (later). */
export function shiftMove(moves: Move[], index: number, dir: -1 | 1): Move[] {
  const other = index + dir;
  const a = moves[index];
  const b = moves[other];
  if (!a || !b || a.slot !== b.slot) return moves;
  const out = moves.slice();
  out[index] = b;
  out[other] = a;
  return out;
}

/** The planned block a move became, when the box accepted it. */
export function blockOfMove(result: SimResult | undefined, index: number): PlannedBlock | undefined {
  return result?.blocks.find((b) => !b.fixed && b.move === index);
}

/** The route being walked at `t`, else the most recent one finished. */
export function routeAt(result: SimResult, t: number): { route: RouteTaken; walking: boolean } | undefined {
  let found: RouteTaken | undefined;
  for (const r of result.routes) if (ms(r.depart) <= t) found = r;
  return found ? { route: found, walking: t < ms(found.arrive) } : undefined;
}

/** What is going on at `t`, in the box's own words where it has them. */
export function activityAt(result: SimResult, index: number, places: PlaceIndex): string {
  const tick = result.timeline[index];
  if (!tick) return "";
  const t = ms(tick.t);
  const block = result.blocks.find((b) => ms(b.start) <= t && t < ms(b.end));
  if (block) return block.fixed ? `${block.title} (fixed)` : block.title;
  const walk = routeAt(result, t);
  if (tick.activity === "walk" || walk?.walking) return walk ? `Walking ${walk.route.purpose}` : "Walking";
  const where = placeName(places, tick.place);
  return tick.activity === "idle" ? `Nothing planned, at ${where}` : `${tick.activity}, at ${where}`;
}

/** The first moment at or after `from` whose clock reads `hhmm`, written with `from`'s own offset. */
export function nextClock(from: string, hhmm: string): string | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})T\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(Z|[+-]\d{2}:\d{2})$/.exec(from);
  if (!m || !/^\d{2}:\d{2}$/.test(hhmm)) return null;
  for (let add = 0; add < 3; add++) {
    const d = new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3]) + add));
    const iso = `${d.toISOString().slice(0, 10)}T${hhmm}:00${m[4]}`;
    if (ms(iso) >= ms(from)) return iso;
  }
  return null;
}
