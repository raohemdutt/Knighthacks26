/**
 * The pointer's way across the screen: around the cards, not through their text.
 *
 * The same A* as the room twin's router (contexts/play/room/grid.py: 8 neighbours on a grid, a straight-line
 * heuristic, then the path pulled tight), ported small. The screen is cut into coarse cells; a cell under a card costs
 * more to cross than a gutter, so the cheapest way runs along the gaps and enters only the target's own card. Cards
 * are costly rather than forbidden, so there is always a way.
 *
 * Pure, no imports: scripts/check-guide.mjs runs it in node.
 */

export interface Pt {
  x: number;
  y: number;
}
export interface Box {
  left: number;
  top: number;
  right: number;
  bottom: number;
}
export type Cell = [number, number]; // column, row

export const CARD_COST = 7;

/** Cheapest 8-connected path over a grid of costs (Infinity: blocked). A step costs its length times the two cells' mean cost. */
export function astar(cost: number[][], start: Cell, goal: Cell): Cell[] | null {
  const rows = cost.length;
  const cols = rows ? cost[0]!.length : 0;
  const ok = (c: number, r: number) => c >= 0 && r >= 0 && c < cols && r < rows && Number.isFinite(cost[r]![c]!);
  if (!ok(start[0], start[1]) || !ok(goal[0], goal[1])) return null;
  const id = (c: number, r: number) => r * cols + c;
  const h = (c: number, r: number) => Math.hypot(goal[0] - c, goal[1] - r); // every cell costs at least 1: never an overestimate
  const best = new Map<number, number>([[id(start[0], start[1]), 0]]);
  const back = new Map<number, number>();
  const open: [number, number, number, number][] = [[h(start[0], start[1]), 0, start[0], start[1]]]; // f, g, c, r
  while (open.length > 0) {
    let at = 0;
    for (let i = 1; i < open.length; i++) if (open[i]![0] < open[at]![0]) at = i;
    const [, g, c, r] = open.splice(at, 1)[0]!;
    if (c === goal[0] && r === goal[1]) {
      const path: Cell[] = [[c, r]];
      let k = id(c, r);
      while (back.has(k)) {
        k = back.get(k)!;
        path.push([k % cols, Math.floor(k / cols)]);
      }
      return path.reverse();
    }
    if (g > (best.get(id(c, r)) ?? Infinity)) continue;
    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        if ((dc === 0 && dr === 0) || !ok(c + dc, r + dr)) continue;
        if (dc !== 0 && dr !== 0 && (!ok(c + dc, r) || !ok(c, r + dr))) continue; // no cutting a blocked corner
        const step = g + Math.hypot(dc, dr) * ((cost[r]![c]! + cost[r + dr]![c + dc]!) / 2);
        const k = id(c + dc, r + dr);
        if (step < (best.get(k) ?? Infinity)) {
          best.set(k, step);
          back.set(k, id(c, r));
          open.push([step + h(c + dc, r + dr), step, c + dc, r + dr]);
        }
      }
    }
  }
  return null;
}

export interface Grid {
  cost: number[][];
  cols: number;
  rows: number;
  cell: number;
}

/** The screen as cells: 1 in the open, CARD_COST under an obstacle, 1 again under anything in `free` (the target's own card). */
export function occupancy(w: number, h: number, obstacles: Box[], free: Box[] = [], cell = 24): Grid {
  const cols = Math.max(1, Math.ceil(w / cell));
  const rows = Math.max(1, Math.ceil(h / cell));
  const cost = Array.from({ length: rows }, () => new Array<number>(cols).fill(1));
  const paint = (b: Box, v: number, grow: number) => {
    const c0 = Math.max(0, Math.floor((b.left - grow) / cell));
    const c1 = Math.min(cols - 1, Math.floor((b.right + grow - 1) / cell));
    const r0 = Math.max(0, Math.floor((b.top - grow) / cell));
    const r1 = Math.min(rows - 1, Math.floor((b.bottom + grow - 1) / cell));
    for (let r = r0; r <= r1; r++) for (let c = c0; c <= c1; c++) cost[r]![c] = v;
  };
  for (const b of obstacles) paint(b, CARD_COST, 0);
  for (const b of free) paint(b, 1, 0);
  return { cost, cols, rows, cell };
}

const cellOf = (g: Grid, p: Pt): Cell => [Math.max(0, Math.min(g.cols - 1, Math.floor(p.x / g.cell))), Math.max(0, Math.min(g.rows - 1, Math.floor(p.y / g.cell)))];

/** The total cost of the straight line from a to b, sampled every half cell. */
function lineCost(g: Grid, a: Pt, b: Pt): number {
  const n = Math.max(1, Math.ceil(Math.hypot(b.x - a.x, b.y - a.y) / (g.cell / 2)));
  let sum = 0;
  for (let i = 0; i <= n; i++) {
    const [c, r] = cellOf(g, { x: a.x + ((b.x - a.x) * i) / n, y: a.y + ((b.y - a.y) * i) / n });
    sum += g.cost[r]![c]!;
  }
  return sum / (n + 1);
}

/** String pulling: drop every waypoint whose neighbours can be joined directly without crossing anything costlier. */
export function pull(g: Grid, points: Pt[]): Pt[] {
  if (points.length <= 2) return points;
  const out = [points[0]!];
  let i = 0;
  while (i < points.length - 1) {
    let j = points.length - 1;
    while (j > i + 1 && lineCost(g, points[i]!, points[j]!) > 1.001) j--;
    out.push(points[j]!);
    i = j;
  }
  return out;
}

/** A Catmull-Rom curve through the points, so the corners are rounded. */
export function curve(points: Pt[], per = 10): Pt[] {
  if (points.length < 3) return points;
  const out: Pt[] = [];
  const at = (i: number) => points[Math.max(0, Math.min(points.length - 1, i))]!;
  for (let i = 0; i < points.length - 1; i++) {
    const [p0, p1, p2, p3] = [at(i - 1), at(i), at(i + 1), at(i + 2)];
    for (let k = 0; k < per; k++) {
      const t = k / per;
      const f = (a: number, b: number, c: number, d: number) => 0.5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t * t + (3 * b - a - 3 * c + d) * t * t * t);
      out.push({ x: f(p0.x, p1.x, p2.x, p3.x), y: f(p0.y, p1.y, p2.y, p3.y) });
    }
  }
  out.push(points[points.length - 1]!);
  return out;
}

export function length(points: Pt[]): number {
  let sum = 0;
  for (let i = 1; i < points.length; i++) sum += Math.hypot(points[i]!.x - points[i - 1]!.x, points[i]!.y - points[i - 1]!.y);
  return sum;
}

/** The point a fraction u (0..1) of the way along a polyline, by distance. */
export function along(points: Pt[], u: number): Pt {
  if (points.length === 0) return { x: 0, y: 0 };
  const total = length(points);
  let want = Math.max(0, Math.min(1, u)) * total;
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1]!;
    const b = points[i]!;
    const d = Math.hypot(b.x - a.x, b.y - a.y);
    if (want <= d && d > 0) return { x: a.x + ((b.x - a.x) * want) / d, y: a.y + ((b.y - a.y) * want) / d };
    want -= d;
  }
  return points[points.length - 1]!;
}

export const MAX_DETOUR = 2.5;

/**
 * The pointer's route from `from` to `to` around the obstacles, as a smooth polyline. null when a straight (eased)
 * path is the better choice: nothing in the way, a degenerate grid, or a detour of more than 2.5 times the direct way.
 */
export function pointerRoute(from: Pt, to: Pt, obstacles: Box[], free: Box[], w: number, h: number, cell = 24): Pt[] | null {
  const direct = Math.hypot(to.x - from.x, to.y - from.y);
  if (direct < cell * 3 || w < cell * 4 || h < cell * 4) return null;
  const g = occupancy(w, h, obstacles, free, cell);
  if (lineCost(g, from, to) <= 1.001) return null; // nothing in the way
  const a = cellOf(g, from);
  const b = cellOf(g, to);
  // start and end are where they are, even on a card: stepping off them is free
  g.cost[a[1]]![a[0]] = 1;
  g.cost[b[1]]![b[0]] = 1;
  const cells = astar(g.cost, a, b);
  if (!cells || cells.length < 2) return null;
  const mid = cells.slice(1, -1).map(([c, r]) => ({ x: (c + 0.5) * g.cell, y: (r + 0.5) * g.cell }));
  const pulled = pull(g, [from, ...mid, to]);
  if (pulled.length <= 2) return null;
  const smooth = curve(pulled);
  return length(smooth) > MAX_DETOUR * direct ? null : smooth;
}

/** Several legs joined: the way to the first point, then on through the rest (a walking route drawn on a map). */
export function chain(first: Pt[] | null, from: Pt, stops: Pt[]): Pt[] {
  const lead = first ?? [from, stops[0]!];
  return [...lead, ...stops.slice(1)];
}
