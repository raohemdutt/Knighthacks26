/**
 * The look guide on the Inspection screen: the box resolves the question (GET /api/spot/guide), and for a piece of
 * equipment or a place it also plans the walk there (GET /api/spot/route) on the floor traced from the kit's plan
 * picture. The plan draws that route, the pointer walks it at a walking pace while one instruction per stretch is said,
 * and then the mark itself is ringed.
 *
 * `walk` is the route on show and `here` is where the operator said they are (a waypoint, or "xy:0.31,0.52" for a spot
 * tapped on the plan); both live in memory for this visit only.
 */

import { useEffect, useState } from "react";
import { api } from "./api";
import { precisionOf, resolveApp, spotTarget, spotWords, type GuideStep, type GuideTarget, type Precision, type SpotGuide } from "./guide";
import type { GuideRun } from "./guideBus";
import { planWalk, type FloorGridData, type TracedLeg, type WalkFloor, type WalkStep } from "./guideWalk";

export interface RoutePoint {
  floor: number;
  x: number;
  y: number;
  snapped_px: number;
  waypoint: string | null;
  name: string;
  pin?: { x: number; y: number };
}

export interface SpotRoute {
  configured: boolean;
  from: string | null;
  to: string | null;
  found: boolean;
  /** The legs SPOT walked (waypoint to waypoint): what the data vouches for. Also under `via_walked_legs`. */
  path: string[];
  legs: { from: string; to: string; length: number; floor_change: boolean; times_walked: number }[];
  length: number;
  floor_changes: number;
  say: string[];
  summary: string;
  note: string;
  places: { location_id: string; name: string; floor: number; walked: boolean }[];
  /** Present when the box has a traced floor: the line to draw and the steps to say. */
  steps?: WalkStep[];
  routed?: { found: boolean; connector: string | null; connector_name?: string; length_px: number; floors: WalkFloor[]; start: RoutePoint; end: RoutePoint } | null;
  walked?: boolean;
  times_walked?: number;
  precision?: "exact" | "about";
  precision_reason?: string;
  via_walked_legs?: { found: boolean; path: string[]; say: string[]; summary: string; note: string } | null;
}

export interface TracedThing {
  rect: [number, number, number, number];
  name?: string;
  label?: string;
  kind?: string;
  seen?: boolean;
}

export interface SpotWalkable {
  configured: boolean;
  available: boolean;
  caption: string;
  source?: string;
  width?: number;
  height?: number;
  floors: Record<string, { rooms: TracedThing[]; doors: TracedThing[]; obstacles: TracedThing[]; connectors: { id: string; name: string; x: number; y: number }[]; grid: FloorGridData }>;
  waypoints: Record<string, { floor: number; pin: { x: number; y: number }; stand: { x: number; y: number } | null; snapped_px: number | null }>;
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
  /** "You are here": the next route starts from this waypoint or tapped spot; the one on show is planned again from it. */
  setHere: (here: string | null): void => {
    set({ here });
    const to = state.walk?.to;
    if (to) void api.spotRoute(to, here).then((walk) => set({ walk }), () => undefined);
  },
  clear: (): void => set({ walk: null }),
};

// the traced floor and the traced legs: fetched once per visit, shared by every plan on the page
let floorOnce: Promise<SpotWalkable | null> | null = null;
let legsOnce: Promise<Record<string, TracedLeg>> | null = null;
export const tracedFloor = (): Promise<SpotWalkable | null> => (floorOnce ??= api.spotWalkable().then((w) => (w.available ? w : null), () => ((floorOnce = null), null)));
export const tracedLegs = (): Promise<Record<string, TracedLeg>> => (legsOnce ??= api.spotWalkableLegs().then((r) => r.legs ?? {}, () => ((legsOnce = null), {})));

export function useTraced(): { floor: SpotWalkable | null; legs: Record<string, TracedLeg> } {
  const [got, setGot] = useState<{ floor: SpotWalkable | null; legs: Record<string, TracedLeg> }>({ floor: null, legs: {} });
  useEffect(() => {
    let on = true;
    void Promise.all([tracedFloor(), tracedLegs()]).then(([floor, legs]) => on && setGot({ floor, legs }));
    return () => {
      on = false;
    };
  }, []);
  return got;
}

/** The precision of a walk the box planned: always about on a traced floor, and the reason is the box's own. */
export function walkPrecision(walk: SpotRoute | null): Precision {
  const base = precisionOf("route-traced");
  return walk?.precision_reason ? { precision: "about", reason: walk.precision_reason } : base;
}

/** The guide ids of the route's points on the plan, floor after floor (FloorMap tags them). */
export const walkIds = (floors: WalkFloor[]): string[] => floors.flatMap((f) => f.points).map((_, i) => `walk:${i}`);

/** A route over two floors needs both on show: the box's "switch to floor N" becomes "show both". */
export function bothFloors(target: GuideTarget): GuideTarget {
  const both: GuideStep = { reveal: "map:floor:both", say: "I am showing both floors: the way changes floor." };
  const steps = target.steps.map((s) => (s.reveal?.startsWith("map:floor:") ? both : s));
  const had = steps.includes(both);
  return {
    ...target,
    steps: had ? steps : [both, ...steps],
    reveal: [...(target.reveal ?? []).filter((r) => !(r.key ?? "").startsWith("map:floor:")), { op: "press", at: { id: "map:floor:both" }, key: "map:floor:both" }],
  };
}

/** A question (or a guide id) on the Inspection screen -> the pointer's run, or the closest places when the box is unsure. */
export async function spotRun(query: string, t: number | null = null, speaking = false): Promise<{ run: GuideRun | null; found: SpotGuide | null }> {
  const found = await api.spotGuide(query);
  const target = spotTarget(found);
  if (!target) {
    const app = resolveApp(query, "/spot/"); // "where is my League?": somewhere else in the app
    return { run: app.target && app.confidence >= 0.8 ? { target: app.target, t } : null, found };
  }
  if (!/^(asset|place):/.test(target.id)) return { run: { target, t, precision: precisionOf("element") }, found };
  // equipment or a place: plan the walk too. The guide still points if the route cannot be had.
  let walk: SpotRoute | null = null;
  try {
    walk = await api.spotRoute(target.id, state.here);
  } catch {
    walk = null;
  }
  set({ walk });
  const mark = precisionOf("element"); // the mark is the app's own, so it is ringed exactly; the way to it is "about" and says so
  const about = walkPrecision(walk);
  const routed = walk?.found && walk.routed && walk.routed.floors.length > 0 ? walk.routed : null;
  if (routed && walk?.steps) {
    const floor = await tracedFloor();
    const pace = planWalk(routed.floors, walk.steps, { w: floor?.width ?? 1095, h: floor?.height ?? 595 }, speaking);
    return {
      found,
      run: {
        target: routed.floors.length > 1 ? bothFloors(target) : target,
        t,
        precision: mark,
        spot: { label: spotWords(mark) },
        via: walkIds(routed.floors).map((id) => ({ id })),
        pace,
        arrive: pace.arrive || undefined,
        note: () => `${walk!.summary} The way is ${about.precision}: ${about.reason}. ${walk!.note}`,
      },
    };
  }
  // no traced floor on this box: the legs SPOT walked, waypoint to waypoint, as before
  const shown = walk?.found && walk.path.length > 1 ? walk : null;
  return {
    found,
    run: {
      target,
      t,
      precision: mark,
      spot: { label: spotWords(mark) },
      via: shown ? shown.path.map((id) => ({ id: `at:${id}` })) : undefined,
      extra: shown ? [...shown.say, shown.floor_changes > 0 ? "It changes floor: choose Both on the plan to see the whole route." : "The dashes on the plan are that route."] : walk && !walk.found && walk.say[0] ? [walk.say[0]] : undefined,
      note: () => (shown ? `${shown.summary} ${shown.note}` : walk?.found ? walk.summary : null),
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
