/**
 * Attention on the Inspection screen: which marked things the operator has actually LOOKED AT during this visit.
 *
 * "Looked at" is a gaze fixation of at least 400 ms that snaps to exactly one marked element (lib/gaze.ts `snap`), so it
 * is only as good as the measured gaze accuracy, and it means looked at, not understood. Everything here lives in memory
 * for this visit: nothing is stored, nothing is sent, and no per-person record exists beyond one optional handoff sentence.
 *
 * `mode` says where the fixations come from: "gaze" (the camera, calibrated and checked), "sim" (?gazeSim=..., a scripted
 * stand-in for demonstrations, labelled on screen) or "off" (then one line invites the person to turn gaze on and
 * nothing else changes). No imports: scripts/check-guide.mjs runs the pure functions in node.
 */

export interface Look {
  id: string;
  /** Total time rested on it, ms. */
  ms: number;
  /** The order it was first looked at in, from 1. */
  order: number;
  /** Where it was on the page (document coordinates) when last looked at, for the scan-path drawing. */
  box: { left: number; top: number; right: number; bottom: number };
}

export type AttentionMode = "off" | "gaze" | "sim";

export interface AttentionState {
  mode: AttentionMode;
  seen: Record<string, Look>;
  /** The scan path: fixations in order (a return to something earlier is a new entry). */
  scan: { id: string; ms: number }[];
}

export const FIXATION_MS = 400;

let state: AttentionState = { mode: "off", seen: {}, scan: [] };
const listeners = new Set<() => void>();
const emit = () => listeners.forEach((fn) => fn());

export const attention = {
  get: (): AttentionState => state,
  subscribe: (fn: () => void): (() => void) => (listeners.add(fn), () => listeners.delete(fn)),
  setMode(mode: AttentionMode): void {
    if (mode !== state.mode) (state = { ...state, mode }), emit();
  },
  /** The eyes have rested `ms` more on this thing (called once the rest has passed FIXATION_MS, then as it goes on). */
  note(id: string, ms: number, box: Look["box"]): void {
    const before = state.seen[id];
    const last = state.scan[state.scan.length - 1];
    const scan = last && last.id === id ? [...state.scan.slice(0, -1), { id, ms: last.ms + ms }] : [...state.scan, { id, ms }];
    state = { ...state, scan, seen: { ...state.seen, [id]: { id, ms: (before?.ms ?? 0) + ms, order: before?.order ?? Object.keys(state.seen).length + 1, box } } };
    emit();
  },
  reset(): void {
    state = { ...state, seen: {}, scan: [] };
    emit();
  },
};

// ---------- what matters, and what is missing (pure) ----------

export interface Job {
  asset_id: string;
  name: string;
}

/** The ids that count as "looked at this piece of equipment": its do-first item, or its mark on either plan. */
export function idsFor(assetId: string, jobs: Job[]): string[] {
  const n = jobs.findIndex((j) => j.asset_id === assetId);
  return [...(n >= 0 ? [`do-first:${n + 1}`] : []), `asset:${assetId}`, `run-asset:${assetId}`];
}

const looked = (seen: Record<string, Look>, ids: string[]) => ids.some((id) => seen[id]);

/** "You've looked at 3 of 4 things that need you." */
export function summary(seen: Record<string, Look>, jobs: Job[]): { done: number; of: number; text: string } {
  const done = jobs.filter((j) => looked(seen, idsFor(j.asset_id, jobs))).length;
  return { done, of: jobs.length, text: jobs.length === 0 ? "Nothing needs you, so there is nothing to look over." : `You've looked at ${done} of ${jobs.length} thing${jobs.length === 1 ? "" : "s"} that need${jobs.length === 1 ? "s" : ""} you.` };
}

export interface Missing {
  /** Where the pointer should go. */
  target: string;
  say: string;
}

/** Before "Send SPOT back" / "Send a technician" for one piece of equipment: has its alarm been looked at? */
export function missingForAsset(assetId: string, name: string, seen: Record<string, Look>, jobs: Job[]): Missing | null {
  const ids = idsFor(assetId, jobs);
  return looked(seen, ids) ? null : { target: ids[0]!, say: `You haven't looked at ${name}'s alarm yet.` };
}

/** Before "Send this plan": the battery turn-back, then any stop that needs you and was never looked at. */
export function missingForPlan(route: Job[], needing: Job[], seen: Record<string, Look>): Missing | null {
  if (!seen["shiftrun:turn-back"]) return { target: "shiftrun:turn-back", say: "You haven't looked at when SPOT turns back to charge yet." };
  const stop = route.find((s) => needing.some((j) => j.asset_id === s.asset_id) && !looked(seen, idsFor(s.asset_id, needing)));
  return stop ? { target: `run-asset:${stop.asset_id}`, say: `You haven't looked at ${stop.name} on the plan yet, and it needs you.` } : null;
}

/** After a Shift Run: what the trainee's eyes did, as reasons in the same form as the plan's own. */
export function runReasons(state: AttentionState, route: Job[], needing: Job[]): { tone: "good" | "bad"; text: string }[] {
  const out: { tone: "good" | "bad"; text: string }[] = [];
  const order = state.scan.map((s) => s.id);
  const battery = order.indexOf("shiftrun:turn-back");
  out.push(battery >= 0 ? { tone: "good", text: "You looked at the battery turn-back while planning: good." } : { tone: "bad", text: "You never looked at when SPOT turns back to charge." });
  for (const j of needing) {
    const inRoute = route.findIndex((s) => s.asset_id === j.asset_id);
    const saw = looked(state.seen, idsFor(j.asset_id, needing));
    if (inRoute < 0 && !saw) out.push({ tone: "bad", text: `You never looked at ${j.name}, which needs you, and it is not in your round.` });
    else if (inRoute >= 0 && !saw) out.push({ tone: "bad", text: `You put ${j.name} in your round without looking at it${inRoute === route.length - 1 && route.length > 1 ? ", and left it for last" : ""}.` });
    else if (inRoute >= 0) out.push({ tone: "good", text: `You looked at ${j.name} before planning it.` });
  }
  return out.slice(0, 4);
}

/** The one sentence a handoff may carry, only when the operator ticks the box. */
export function handoffLine(seen: Record<string, Look>, jobs: Job[]): string {
  const s = summary(seen, jobs);
  return `Reviewed on screen this shift: ${s.done} of ${s.of} items needing attention (by gaze).`;
}

/** The scripted stand-in: "?gazeSim=do-first:1,trust,asset:HX-202" -> the ids, in order. */
export function simScript(param: string | null): string[] {
  return (param ?? "").split(",").map((s) => s.trim()).filter(Boolean);
}
