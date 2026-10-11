/**
 * Today's Plan: words and rows. Nothing here simulates or scores; it lays out what the box returned.
 *
 * Times are written on the plan's own clock (the offset the box sent), in 12-hour form like the Calendar screen,
 * whatever zone the phone is in: a plan for a day at UCF reads the same from anywhere.
 */

import type { Move, OpenSlot, PlanBoard, PlannedBlock, RouteTaken, SimResult, Suggestion } from "./api";
import { ms } from "./dayrun";

const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const WEEKDAYS_LONG = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const MIN = 60_000;

/** "2026-10-11T14:05:00-04:00" -> "2:05 PM" (the time as written, no zone shift). */
export function t12(iso: string | null | undefined): string {
  const m = iso ? /T(\d{2}):(\d{2})/.exec(iso) : null;
  if (!m) return "";
  const h = Number(m[1]);
  return `${h % 12 || 12}:${m[2]} ${h < 12 ? "AM" : "PM"}`;
}

/** The offset the box wrote its times in ("-04:00"), so a moment can be written back the same way. */
export function offsetOf(iso: string): string {
  return /(Z|[+-]\d{2}:\d{2})$/.exec(iso)?.[1] ?? "Z";
}

/** A moment as an ISO string on the plan's own clock. */
export function isoAt(t: number, like: string): string {
  const off = offsetOf(like);
  const minutes = off === "Z" ? 0 : (off.startsWith("-") ? -1 : 1) * (Number(off.slice(1, 3)) * 60 + Number(off.slice(4, 6)));
  const d = new Date(t + minutes * MIN);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getUTCFullYear()}-${p(d.getUTCMonth() + 1)}-${p(d.getUTCDate())}T${p(d.getUTCHours())}:${p(d.getUTCMinutes())}:00${off}`;
}

export const clockAt = (t: number, like: string) => t12(isoAt(t, like));

function parts(day: string): Date {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(day);
  return m ? new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3]))) : new Date(NaN);
}

export const weekday = (day: string) => WEEKDAYS[parts(day).getUTCDay()] ?? "";
export const weekdayLong = (day: string) => WEEKDAYS_LONG[parts(day).getUTCDay()] ?? "";
export const dayNumber = (day: string) => parts(day).getUTCDate();
/** "Mon, Oct 5" */
export const dateWords = (day: string) => `${weekday(day)}, ${MONTHS[parts(day).getUTCMonth()] ?? ""} ${dayNumber(day)}`;

/** 95 -> "1 h 35 min", 30 -> "30 min" (the way Calendar writes a length). */
export function dur(minutes: number): string {
  const total = Math.max(0, Math.round(minutes));
  const h = Math.floor(total / 60);
  const m = total % 60;
  return h === 0 ? `${m} min` : m === 0 ? `${h} h` : `${h} h ${m} min`;
}

/** 7.5 -> "7.5 h", 8 -> "8 h" */
export const hoursShort = (h: number) => `${Number.isInteger(h) ? h : h.toFixed(1)} h`;

export function stars(n: number): string {
  return "★".repeat(Math.max(0, Math.min(3, n))) + "☆".repeat(Math.max(0, 3 - n));
}

/** What a person calls a block: "Lunch", "Study", a fixed block's own title. */
export function blockTitle(b: Pick<PlannedBlock, "kind" | "title" | "start" | "fixed">, activity = "Study"): string {
  if (b.fixed) return b.title;
  if (b.kind === "eat") {
    const m = /T(\d{2}):(\d{2})/.exec(b.start);
    const hour = m ? Number(m[1]) + Number(m[2]) / 60 : 12;
    return hour < 10.5 ? "Breakfast" : hour < 15.5 ? "Lunch" : "Dinner";
  }
  if (b.kind === "study") return /with /.test(b.title) ? `${activity} ${b.title.slice(b.title.indexOf("with "))}` : activity;
  if (b.kind === "sleep") return "Sleep";
  if (b.kind === "rest") return "Rest";
  return b.title;
}

export const placeShort = (name: string | null | undefined) => (name ?? "").split(" (")[0] ?? "";

// ---------- the agenda ----------

export interface BlockRow {
  type: "block";
  key: string;
  block: PlannedBlock;
  /** The walk that brings you to this block, when there is one. */
  walk?: RouteTaken;
  /** The walk that takes you back from the block before it, when that is a different walk. */
  slot?: string;
}

export interface GapRow {
  type: "gap";
  key: string;
  slot: OpenSlot;
  start: number;
  end: number;
  /** A "Near you" event that would fall here. */
  suggestions: Suggestion[];
}

export interface StuckRow {
  type: "stuck";
  key: string;
  index: number;
  move: Move;
  slot: OpenSlot | undefined;
  reason: string;
}

export type Row = BlockRow | GapRow | StuckRow;

export const MIN_GAP = 25; // minutes of open time worth an "Add" row

/** The plan as rows in time order: every block, the open gaps between them, and the moves that did not fit. */
export function rowsOf(board: PlanBoard, result: SimResult, moves: Move[]): Row[] {
  const out: Row[] = [];
  const blocks = [...result.blocks].sort((a, b) => ms(a.start) - ms(b.start));
  const walkTo = (b: PlannedBlock): RouteTaken | undefined =>
    result.routes.find((r) => (b.move !== null ? r.move === b.move : r.move === null && r.end === b.place && Math.abs(ms(r.arrive) - ms(b.start)) <= 30 * MIN && ms(r.arrive) <= ms(b.start) + MIN));
  const taken = new Set(moves.map((m) => m.event).filter(Boolean));
  const offered = (board.suggestions ?? []).filter((s) => !taken.has(s.id));
  const gap = (from: number, to: number) => {
    if (to - from < MIN_GAP * MIN) return;
    const slot = board.slots.find((s) => ms(s.start) < to && ms(s.end) > from);
    if (!slot) return;
    const start = Math.max(from, ms(slot.start));
    const end = Math.min(to, ms(slot.end));
    if (end - start < MIN_GAP * MIN) return;
    out.push({ type: "gap", key: `gap-${start}`, slot, start, end, suggestions: offered.filter((s) => ms(s.start) >= start - 30 * MIN && ms(s.start) < end) });
  };
  let cursor = ms(board.start);
  for (const b of blocks) {
    const walk = walkTo(b);
    gap(cursor, walk ? ms(walk.depart) : ms(b.start));
    out.push({ type: "block", key: b.id + b.start, block: b, walk, slot: b.move !== null ? moves[b.move]?.slot : undefined });
    cursor = Math.max(cursor, ms(b.end));
  }
  gap(cursor, ms(board.end));
  for (const r of result.rejected) {
    const slot = board.slots.find((s) => s.id === r.move.slot);
    out.push({ type: "stuck", key: `stuck-${r.index}`, index: r.index, move: r.move, slot, reason: r.reason });
  }
  // a move that did not fit is shown where its slot begins
  return out.sort((a, b) => at(a) - at(b));
}

function at(r: Row): number {
  if (r.type === "block") return ms(r.block.start);
  if (r.type === "gap") return r.start;
  return (r.slot ? ms(r.slot.start) : 0) - 1;
}

// ---------- edits, as new lists of moves ----------

const rank = (board: PlanBoard, slot: string) => {
  const i = board.slots.findIndex((s) => s.id === slot);
  return i < 0 ? board.slots.length : i;
};

/** `move` put where it belongs: slots in board order, and inside its slot before the first move that starts after `t`. */
export function insertAt(moves: Move[], move: Move, board: PlanBoard, result: SimResult | undefined, t: number | null): { moves: Move[]; index: number } {
  const started = new Map<number, number>();
  for (const b of result?.blocks ?? []) if (b.move !== null) started.set(b.move, ms(b.start));
  let index = moves.length;
  for (let i = 0; i < moves.length; i++) {
    const m = moves[i] as Move;
    const later = rank(board, m.slot) > rank(board, move.slot);
    const after = m.slot === move.slot && t !== null && (started.get(i) ?? (m.start ? ms(m.start) : Infinity)) >= t;
    if (later || after) {
      index = i;
      break;
    }
  }
  return { moves: [...moves.slice(0, index), move, ...moves.slice(index)], index };
}

/** Start a move at another time: it goes to the open slot that holds that time, in order. Null when the time is taken. */
export function moveTo(moves: Move[], index: number, t: number, board: PlanBoard, result: SimResult | undefined): { moves: Move[]; index: number } | null {
  const move = moves[index];
  const slot = board.slots.find((s) => ms(s.start) <= t && t < ms(s.end));
  if (!move || !slot) return null;
  const rest = moves.filter((_, i) => i !== index);
  // the starts of the other moves, re-indexed without this one
  const shifted: SimResult | undefined = result
    ? { ...result, blocks: result.blocks.filter((b) => b.move !== index).map((b) => (b.move !== null && b.move > index ? { ...b, move: b.move - 1 } : b)) }
    : undefined;
  return insertAt(rest, { ...move, slot: slot.id, start: isoAt(t, board.start) }, board, shifted, t);
}

/** The nearest time at or after `t` (or before it, going back) that lies in an open slot, in steps of `step` minutes. */
export function freeTime(board: PlanBoard, t: number, dir: 1 | -1, step = 15): number | null {
  for (let n = 1; n <= 96; n++) {
    const x = t + dir * n * step * MIN;
    if (board.slots.some((s) => ms(s.start) <= x && x < ms(s.end) - 10 * MIN)) return x;
  }
  return null;
}
