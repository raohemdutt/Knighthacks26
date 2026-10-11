/**
 * The look guide on the Inspection screen: the box resolves the question (GET /api/spot/guide), and for a piece of
 * equipment or a place it also plans the walk there (GET /api/spot/route) over the legs SPOT has walked. The plan draws
 * that route, and the pointer follows it waypoint by waypoint before it lands on the mark.
 *
 * `walk` is the route on show and `here` is where the operator said they are; both live in memory for this visit only.
 */

import { api } from "./api";
import { resolveApp, spotTarget, type SpotGuide } from "./guide";
import type { GuideRun } from "./guideBus";

export interface SpotRoute {
  configured: boolean;
  from: string | null;
  to: string | null;
  found: boolean;
  path: string[];
  legs: { from: string; to: string; length: number; floor_change: boolean; times_walked: number }[];
  length: number;
  floor_changes: number;
  say: string[];
  summary: string;
  note: string;
  places: { location_id: string; name: string; floor: number; walked: boolean }[];
}

interface State {
  walk: SpotRoute | null;
  here: string | null;
}

let state: State = { walk: null, here: null };
const listeners = new Set<() => void>();
const set = (patch: Partial<State>) => {
  state = { ...state, ...patch };
  for (const fn of listeners) fn();
};

export const spotWalk = {
  get: (): State => state,
  subscribe: (fn: () => void): (() => void) => (listeners.add(fn), () => listeners.delete(fn)),
  /** "You are here": the next route starts from this waypoint; the one on show is planned again from it. */
  setHere: (here: string | null): void => {
    set({ here });
    const to = state.walk?.to;
    if (to) void api.spotRoute(to, here).then((walk) => set({ walk }), () => undefined);
  },
  clear: (): void => set({ walk: null }),
};

/** A question (or a guide id) on the Inspection screen -> the pointer's run, or the closest places when the box is unsure. */
export async function spotRun(query: string, t: number | null = null): Promise<{ run: GuideRun | null; found: SpotGuide | null }> {
  const found = await api.spotGuide(query);
  const target = spotTarget(found);
  if (!target) {
    const app = resolveApp(query, "/spot/"); // "where is my League?": somewhere else in the app
    return { run: app.target && app.confidence >= 0.8 ? { target: app.target, t } : null, found };
  }
  if (!/^(asset|place):/.test(target.id)) return { run: { target, t }, found };
  // equipment or a place: plan the walk too. The guide still points if the route cannot be had.
  let walk: SpotRoute | null = null;
  try {
    walk = await api.spotRoute(target.id, state.here);
  } catch {
    walk = null;
  }
  set({ walk });
  const shown = walk?.found && walk.path.length > 1 ? walk : null;
  return {
    found,
    run: {
      target,
      t,
      via: shown ? shown.path.map((id) => ({ id: `at:${id}` })) : undefined,
      extra: shown ? [...shown.say, "The green dashes on the plan are that route."] : walk && !walk.found && walk.say[0] ? [walk.say[0]] : undefined,
      note: () => (shown ? `${shown.summary} ${shown.note}` : null),
    },
  };
}

let nextJob = 0;
/** "Next": step through the do-first list by voice, one job at a time, round and round. */
export function nextDoFirst(): string {
  const jobs = new Set([...document.querySelectorAll("[data-guide^='do-first:']")].map((el) => el.getAttribute("data-guide")));
  nextJob = (nextJob % Math.max(1, jobs.size)) + 1;
  return `do-first:${nextJob}`;
}
export const NEXT = /^(?:(?:and |ok |okay )?(?:next|next one|the next one|show me the next one|show me the next|what s next|whats next|go on|and then))$/;
