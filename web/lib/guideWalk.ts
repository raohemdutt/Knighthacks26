/**
 * The walk on the plant floor, as the pointer follows it: pure helpers with no imports, so `node scripts/check-guide.mjs`
 * can run them.
 *
 * The box plans the route on the floor traced from the kit's plan picture (GET /api/spot/route) and gives one instruction
 * per stretch with `upto`, the share of the route walked when that stretch ends. Here that becomes a timeline: each
 * stretch lasts as long as it takes to say its sentence (or a walking pace when the speaker is off), so the dot and the
 * voice stay in step. Stairs have no length on a plan; they get a short stretch of their own so the dot can be seen
 * changing floor.
 */

export interface WalkStep {
  say: string;
  floor: number;
  length_px: number;
  upto: number;
  arrive?: boolean;
  stairs?: boolean;
}

export interface WalkFloor {
  floor: number;
  points: number[][]; // normalised on the plan picture
  walked?: boolean;
}

export interface WalkPlan {
  /** One per point of the route, floor after floor: its share (0..1) of the way, stairs included. */
  at: number[];
  /** The stretches, in order, with the share of the way and of the time each one covers. */
  steps: { say: string; from: number; upto: number; t0: number; t1: number; stairs: boolean }[];
  /** How long the whole walk takes on screen. */
  ms: number;
  /** Said on arrival ("Pump 5 is on your left, by the valve manifold."). */
  arrive: string;
}

const CHAR_MS = 62; // about how long the browser's voice takes per character
export const sayMs = (text: string): number => Math.max(1400, 500 + text.length * CHAR_MS);

/** Walking pace on the plan: about 9 ms per picture pixel, never slower than 2.5 s nor longer than 12 s for the whole way. */
export const paceMs = (lengthPx: number): number => Math.round(Math.max(2500, Math.min(12000, lengthPx * 9)));

export function planWalk(floors: WalkFloor[], steps: WalkStep[], size: { w: number; h: number }, speaking: boolean): WalkPlan {
  const moving = steps.filter((s) => !s.arrive);
  const real = moving.reduce((n, s) => n + s.length_px, 0);
  const jump = Math.max(24, real * 0.08); // what a flight of stairs counts for
  const lens = moving.map((s) => (s.stairs ? jump : s.length_px));
  const total = lens.reduce((a, b) => a + b, 0) || 1;
  // the points, with the jump between floors counted as that stretch
  const at: number[] = [];
  let d = 0;
  floors.forEach((f, i) => {
    f.points.forEach((p, k) => {
      if (k > 0) d += Math.hypot((p[0]! - f.points[k - 1]![0]!) * size.w, (p[1]! - f.points[k - 1]![1]!) * size.h);
      else if (i > 0) d += jump;
      at.push(d);
    });
  });
  const span = d || 1;
  const base = paceMs(real);
  const durs = moving.map((s, i) => Math.max((lens[i]! / total) * base, speaking ? sayMs(s.say) : s.stairs ? 700 : 0));
  const ms = Math.round(durs.reduce((a, b) => a + b, 0)) || base;
  let from = 0;
  let t = 0;
  const out = moving.map((s, i) => {
    const row = { say: s.say, from: from / total, upto: (from + lens[i]!) / total, t0: t / ms, t1: (t + durs[i]!) / ms, stairs: Boolean(s.stairs) };
    from += lens[i]!;
    t += durs[i]!;
    return row;
  });
  if (out.length > 0) (out[out.length - 1]!.upto = 1), (out[out.length - 1]!.t1 = 1);
  return { at: at.map((v) => v / span), steps: out, ms, arrive: steps.find((s) => s.arrive)?.say ?? "" };
}

/** Which stretch is being walked at this share of the time (the last one once the walk is over; -1 when there are none). */
export function stepAt(plan: WalkPlan, tf: number): number {
  for (let i = 0; i < plan.steps.length; i++) if (tf < plan.steps[i]!.t1) return i;
  return plan.steps.length - 1;
}

/** The share of the way walked at this share of the time. */
export function walkFraction(plan: WalkPlan, tf: number): number {
  if (plan.steps.length === 0) return Math.max(0, Math.min(1, tf));
  if (tf <= 0) return 0;
  if (tf >= 1) return 1;
  const s = plan.steps[stepAt(plan, tf)]!;
  const k = s.t1 > s.t0 ? (tf - s.t0) / (s.t1 - s.t0) : 1;
  return s.from + (s.upto - s.from) * Math.max(0, Math.min(1, k));
}

/** The point at share `f` of the way along stops whose shares are `at` (same order, same length). */
export function alongStops(pts: { x: number; y: number }[], at: number[], f: number): { x: number; y: number } {
  if (pts.length === 0) return { x: 0, y: 0 };
  if (f <= at[0]!) return pts[0]!;
  for (let i = 1; i < pts.length; i++) {
    if (f <= at[i]!) {
      const span = at[i]! - at[i - 1]!;
      const k = span > 0 ? (f - at[i - 1]!) / span : 1;
      return { x: pts[i - 1]!.x + (pts[i]!.x - pts[i - 1]!.x) * k, y: pts[i - 1]!.y + (pts[i]!.y - pts[i - 1]!.y) * k };
    }
  }
  return pts[pts.length - 1]!;
}

// ---------- the traced floor ----------

export interface FloorGridData {
  x0: number;
  y0: number;
  cell_x: number;
  cell_y: number;
  cols: number;
  rows: number;
  free: number;
  runs: number[][]; // [row, first col, last col] of walkable cells
}

/** Is this point (normalised on the picture) on walkable floor? */
export function isFree(g: FloorGridData, x: number, y: number): boolean {
  const col = Math.floor((x - g.x0) / g.cell_x);
  const row = Math.floor((y - g.y0) / g.cell_y);
  return g.runs.some((r) => r[0] === row && col >= r[1]! && col <= r[2]!);
}

/** The nearest walkable point to a tap, within `limitPx` picture pixels; null when the tap is nowhere near the floor. */
export function snapToFloor(g: FloorGridData, x: number, y: number, size: { w: number; h: number }, limitPx = 60): { x: number; y: number; moved: number } | null {
  if (isFree(g, x, y)) return { x, y, moved: 0 };
  let best: { x: number; y: number; moved: number } | null = null;
  const py = (y - g.y0) / g.cell_y;
  const px = (x - g.x0) / g.cell_x;
  for (const [row, c0, c1] of g.runs as [number, number, number][]) {
    const col = Math.max(c0, Math.min(c1, Math.floor(px)));
    const cx = g.x0 + (col + 0.5) * g.cell_x;
    const cy = g.y0 + (row + 0.5) * g.cell_y;
    const moved = Math.hypot((cx - x) * size.w, (cy - y) * size.h);
    if (moved <= limitPx && (!best || moved < best.moved)) best = { x: cx, y: cy, moved };
  }
  void py;
  return best ? { x: Number(best.x.toFixed(4)), y: Number(best.y.toFixed(4)), moved: Math.round(best.moved * 10) / 10 } : null;
}

/** Which floor's part of the picture a point is on, by each floor's grid; null when it is on neither. */
export function floorAt(grids: Record<string, FloorGridData>, x: number, y: number): number | null {
  for (const [f, g] of Object.entries(grids)) {
    if (x >= g.x0 && x <= g.x0 + g.cols * g.cell_x && y >= g.y0 && y <= g.y0 + g.rows * g.cell_y) return Number(f);
  }
  return null;
}

// ---------- legs between waypoints, drawn along the traced floor ----------

export interface TracedLeg {
  floors: WalkFloor[];
  connector: string | null;
  walked: boolean;
  times_walked: number;
}

/** The traced line from a to b (kept one way on the box, 'A>B' with A first): the points come back in walking order. */
export function legBetween(legs: Record<string, TracedLeg>, a: string, b: string): TracedLeg | null {
  if (a === b) return null;
  const fwd = legs[`${a}>${b}`];
  if (fwd) return fwd;
  const back = legs[`${b}>${a}`];
  return back ? { ...back, floors: [...back.floors].reverse().map((f) => ({ ...f, points: [...f.points].reverse() })) } : null;
}
