/**
 * Gaze from a laptop webcam, as precise as it honestly gets, with the precision MEASURED rather than assumed.
 *
 * Pure maths, no imports (scripts/check-guide.mjs runs it in node). The camera and the face model are in lib/gazeTracker.ts.
 *
 *   features(landmarks, matrix)   where the irises sit between the eye corners and lids (both eyes, in units of the distance
 *                                 between the eyes) and where the head is and points (from the model's face transform)
 *   clean(samples)                per calibration dot, looks that sit far from that dot's median are dropped (MAD)
 *   fitCalibration(samples)       ridge regression per axis; a straight-line model and a second-order one are both tried and
 *                                 the one that predicts LEFT-OUT dots better is kept (never the one that fits its own dots best)
 *   validate(fit, fresh looks)    four dots the fit never saw: the mean miss in pixels and as a share of the screen. The worse
 *                                 of that and the left-out miss sets everything after it: the grid the estimate is rounded to (a cell is at least
 *                                 twice the mean miss), the snap radius, and whether gaze is used at all
 *   OneEuro                       the smoothing filter: steady when the eyes rest, quick when they jump
 *   Fixation                      "the eyes have rested in one place" by dispersion, for dwell
 *   snap(point, radius, things)   the one registered thing within the measured miss of the estimate; never a guess between two
 *
 * Nothing here stores where anyone looked. A calibration is a few dozen weights, the head pose it was made at and its
 * measured error. Degrees of visual angle are not claimed: the distance to the screen and its physical size are not known.
 */

export interface Landmark {
  x: number;
  y: number;
  z?: number;
}
/** hx, hy: irises in the eyes. rx, ry: where the head points. px, py: where the head is, seen from the camera. */
export type Features = [number, number, number, number, number, number];

// MediaPipe Face Landmarker indices (478 points with the irises refined)
const EYES = [
  { outer: 33, inner: 133, top: 159, bottom: 145, iris: 468 },
  { outer: 263, inner: 362, top: 386, bottom: 374, iris: 473 },
];
const NOSE = 1;
export const BLINK_EAR = 0.14; // lid opening over eye width below which an eye counts as closed

/**
 * One look as six numbers, or null when it cannot be read (no iris points, a blink).
 * `matrix` is the model's 4x4 face transform (column-major, 16 numbers); without it the head terms come from the landmarks.
 */
export function features(lm: Landmark[], matrix?: ArrayLike<number> | null): Features | null {
  if (lm.length < 478) return null;
  const l = lm[EYES[0]!.outer]!;
  const r = lm[EYES[1]!.outer]!;
  const span = Math.hypot(r.x - l.x, r.y - l.y); // between the eyes' outer corners: the unit for everything below
  if (span < 1e-4) return null;
  let hx = 0;
  let hy = 0;
  for (const e of EYES) {
    const a = lm[e.outer]!;
    const b = lm[e.inner]!;
    const iris = lm[e.iris]!;
    const wx = b.x - a.x;
    const wy = b.y - a.y;
    const width = Math.hypot(wx, wy);
    if (width < 1e-5) return null;
    const open = Math.hypot(lm[e.top]!.x - lm[e.bottom]!.x, lm[e.top]!.y - lm[e.bottom]!.y) / width;
    if (open < BLINK_EAR) return null;
    // the iris relative to the eye's own middle (corners and lids), along and across the corner-to-corner line
    const mx = (a.x + b.x + lm[e.top]!.x + lm[e.bottom]!.x) / 4;
    const my = (a.y + b.y + lm[e.top]!.y + lm[e.bottom]!.y) / 4;
    const ux = wx / width;
    const uy = wy / width;
    const right = wx > 0 ? 1 : -1; // the two eyes mirror each other: turn both into "towards the picture's right / bottom"
    hx += (right * ((iris.x - mx) * ux + (iris.y - my) * uy)) / span;
    hy += (-right * ((iris.x - mx) * uy - (iris.y - my) * ux)) / span;
  }
  let rx: number;
  let ry: number;
  let px: number;
  let py: number;
  if (matrix && matrix.length >= 16 && Math.abs(matrix[14]!) > 1e-6) {
    // rotation: the face's forward axis, sideways and up-down; position: where the head is, over its distance
    rx = matrix[8]!;
    ry = matrix[9]!;
    px = matrix[12]! / Math.abs(matrix[14]!);
    py = matrix[13]! / Math.abs(matrix[14]!);
  } else {
    const nose = lm[NOSE]!;
    rx = (nose.x - (l.x + r.x) / 2) / span;
    ry = (nose.y - (l.y + r.y) / 2) / span;
    px = (l.x + r.x) / 2 - 0.5;
    py = (l.y + r.y) / 2 - 0.5;
  }
  return [hx / 2, hy / 2, rx, ry, px, py];
}

export function median(values: number[]): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
}

export interface Sample {
  f: Features;
  /** Where the person was asked to look, as a fraction of the screen (0..1 each way). */
  x: number;
  y: number;
}

const dotKey = (s: { x: number; y: number }) => `${s.x},${s.y}`;

/** Per dot, drop the looks whose iris or head numbers sit more than 3 robust deviations from that dot's median (a glance away, a half blink). */
export function clean(samples: Sample[]): Sample[] {
  const out: Sample[] = [];
  for (const key of new Set(samples.map(dotKey))) {
    const here = samples.filter((s) => dotKey(s) === key);
    const med = [0, 1, 2, 3, 4, 5].map((j) => median(here.map((s) => s.f[j]!)));
    const mad = [0, 1, 2, 3, 4, 5].map((j) => 1.4826 * median(here.map((s) => Math.abs(s.f[j]! - med[j]!))) || 1e-9);
    const kept = here.filter((s) => s.f.every((v, j) => Math.abs(v - med[j]!) <= 3 * mad[j]! + 1e-9));
    out.push(...(kept.length >= Math.max(3, here.length / 2) ? kept : here));
  }
  return out;
}

export type Model = "linear" | "poly";

/** The terms a model uses: the six numbers, and for "poly" the iris terms squared and multiplied as well. */
function terms(f: number[], model: Model): number[] {
  return model === "poly" ? [...f, f[0]! * f[0]!, f[1]! * f[1]!, f[0]! * f[1]!] : f;
}

/** Solve A x = b for a small system (Gaussian elimination with pivoting). */
function solve(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]!]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r]![c]!) > Math.abs(M[p]![c]!)) p = r;
    [M[c], M[p]] = [M[p]!, M[c]!];
    const d = M[c]![c]! || 1e-12;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r]![c]! / d;
      for (let k = c; k <= n; k++) M[r]![k]! -= f * M[c]![k]!;
    }
  }
  return M.map((row, i) => row[n]! / (row[i]! || 1e-12));
}

const LAMBDA = 0.012;

function ridge(rows: number[][], ys: number[]): number[] {
  const d = rows[0]!.length;
  const A = Array.from({ length: d + 1 }, () => new Array<number>(d + 1).fill(0));
  const b = new Array<number>(d + 1).fill(0);
  for (let i = 0; i < rows.length; i++) {
    const r = [...rows[i]!, 1];
    for (let j = 0; j <= d; j++) {
      b[j]! += r[j]! * ys[i]!;
      for (let k = 0; k <= d; k++) A[j]![k]! += r[j]! * r[k]!;
    }
  }
  for (let j = 0; j < d; j++) A[j]![j]! += LAMBDA * rows.length; // not on the constant
  return solve(A, b);
}

export interface Fit {
  model: Model;
  mean: number[];
  scale: number[];
  wx: number[];
  wy: number[];
}

function fit(samples: Sample[], model: Model): Fit {
  const mean = [0, 1, 2, 3, 4, 5].map((j) => samples.reduce((s, v) => s + v.f[j]!, 0) / samples.length);
  const scale = [0, 1, 2, 3, 4, 5].map((j) => Math.sqrt(samples.reduce((s, v) => s + (v.f[j]! - mean[j]!) ** 2, 0) / samples.length) || 1);
  const rows = samples.map((s) => terms(s.f.map((v, j) => (v - mean[j]!) / scale[j]!), model));
  return { model, mean, scale, wx: ridge(rows, samples.map((s) => s.x)), wy: ridge(rows, samples.map((s) => s.y)) };
}

/** Where a fit puts a look, as a fraction of the screen (it may fall a little outside 0..1). */
export function predict(c: Fit, f: Features): { x: number; y: number } {
  const r = terms(f.map((v, j) => (v - c.mean[j]!) / c.scale[j]!), c.model);
  const dot = (w: number[]) => r.reduce((s, v, j) => s + v * w[j]!, w[w.length - 1]!);
  return { x: dot(c.wx), y: dot(c.wy) };
}

export type Quality = "good" | "rough" | "poor";

export interface Validation {
  /** Mean miss at the fresh dots, in CSS pixels of the screen it was measured on. */
  freshPx: number;
  /** Mean miss at the calibration dots when each is left out of the fit, in the same pixels. */
  leftOutPx: number;
  /** The number everything else uses: the worse of the two (the fresh dots sit mid-screen, where it is easiest). */
  meanPx: number;
  /** `meanPx` as a fraction of the screen's diagonal, and of its width. */
  ofDiagonal: number;
  ofWidth: number;
  points: number;
  width: number;
  height: number;
}

export interface Calibration extends Fit {
  /** Mean miss when each calibration dot is predicted from a fit that never saw it, as a fraction of the screen's diagonal. */
  leftOut: number;
  /** How much the estimate jitters while the eyes rest on one dot (median, same unit). */
  scatter: number;
  dots: number;
  /** The head pose it was made at (rx, ry, px, py), to notice when the person has moved. */
  pose: number[];
  /** When it was made (ms since the epoch). */
  at: number;
  /** The measured error on fresh dots. Until this exists the fit is not used. */
  validation: Validation | null;
  quality: Quality;
  /** The grid the estimate is rounded to: each cell at least twice the mean miss across. 1x1 means "not used". */
  cols: number;
  rows: number;
}

/** Mean miss of a fit on dots it never saw: leave each dot out, fit the rest, predict it. */
function leftOutError(samples: Sample[], model: Model, aspect: number): number {
  const dots = [...new Set(samples.map(dotKey))];
  const diag = Math.hypot(aspect, 1);
  let sum = 0;
  for (const key of dots) {
    const here = samples.filter((s) => dotKey(s) === key);
    const without = fit(samples.filter((s) => dotKey(s) !== key), model);
    const ps = here.map((s) => predict(without, s.f));
    sum += Math.hypot((median(ps.map((p) => p.x)) - here[0]!.x) * aspect, median(ps.map((p) => p.y)) - here[0]!.y) / diag;
  }
  return sum / dots.length;
}

/**
 * Fit the calibration dots. Both models are tried; the one with the smaller leave-one-dot-out miss is kept. The result
 * is not yet trusted: `validate` measures it on fresh dots and sets its quality and grid. `aspect` is width over height.
 */
export function fitCalibration(raw: Sample[], aspect = 16 / 10, now = 0): Calibration | null {
  const samples = clean(raw);
  const dots = [...new Set(samples.map(dotKey))];
  if (dots.length < 5 || samples.length < 3 * dots.length) return null;
  const models: Model[] = dots.length >= 9 ? ["linear", "poly"] : ["linear"]; // five dots cannot support the curved model
  const scored = models.map((m) => ({ m, e: leftOutError(samples, m, aspect) })).sort((a, b) => a.e - b.e);
  const best = scored[0]!;
  const all = fit(samples, best.m);
  const diag = Math.hypot(aspect, 1);
  let scatter = 0;
  for (const key of dots) {
    const own = samples.filter((s) => dotKey(s) === key).map((s) => predict(all, s.f));
    const mid = { x: median(own.map((p) => p.x)), y: median(own.map((p) => p.y)) };
    scatter += median(own.map((p) => Math.hypot((p.x - mid.x) * aspect, p.y - mid.y) / diag)) / dots.length;
  }
  const pose = [2, 3, 4, 5].map((j) => median(samples.map((s) => s.f[j]!)));
  return { ...all, leftOut: best.e, scatter, dots: dots.length, pose, at: now, validation: null, quality: "poor", cols: 1, rows: 1 };
}

export const MAX_COLS = 6;
export const MAX_ROWS = 4;

/** The grid a measured miss earns on a screen: a cell at least twice the mean miss across, and never finer than 6 x 4. */
export function gridFor(meanPx: number, width: number, height: number): { cols: number; rows: number } {
  const cell = Math.max(1, 2 * meanPx);
  return { cols: Math.max(1, Math.min(MAX_COLS, Math.floor(width / cell))), rows: Math.max(1, Math.min(MAX_ROWS, Math.floor(height / cell))) };
}

/**
 * Measure a fit on looks at fresh dots (ones it was not fitted on), on a screen of `width` x `height` CSS pixels, and
 * set what it may be used for. Per dot the median prediction is compared with the dot; the mean of those misses is the number.
 */
export function validate(c: Calibration, fresh: Sample[], width: number, height: number): Calibration {
  const dots = [...new Set(fresh.map(dotKey))];
  if (dots.length === 0) return { ...c, validation: null, quality: "poor", cols: 1, rows: 1 };
  let sum = 0;
  for (const key of dots) {
    const here = clean(fresh.filter((s) => dotKey(s) === key));
    const ps = here.map((s) => predict(c, s.f));
    sum += Math.hypot((median(ps.map((p) => p.x)) - here[0]!.x) * width, (median(ps.map((p) => p.y)) - here[0]!.y) * height);
  }
  const freshPx = sum / dots.length;
  const leftOutPx = c.leftOut * Math.hypot(width, height);
  const meanPx = Math.max(freshPx, leftOutPx);
  const v: Validation = { freshPx, leftOutPx, meanPx, ofDiagonal: meanPx / Math.hypot(width, height), ofWidth: meanPx / width, points: dots.length, width, height };
  const { cols, rows } = gridFor(meanPx, width, height);
  // good: at least a 3 x 2 grid was earned. rough: only "which side". poor: not even that; gaze is not used.
  const quality: Quality = cols >= 3 && rows >= 2 ? "good" : cols >= 2 ? "rough" : "poor";
  return { ...c, validation: v, quality, cols: quality === "poor" ? 1 : cols, rows: quality === "poor" ? 1 : rows };
}

/** "about 1/8 of the screen width" for a miss in pixels. */
export function shareWords(px: number, width: number): string {
  const n = Math.round(width / Math.max(px, 1));
  return n >= 2 ? `1/${Math.min(n, 30)} of the screen width` : n === 1 ? "the whole screen width" : "more than the screen width";
}

/** What to tell the person, plainly: the measured accuracy first, then what it will be used for. */
export function qualityWords(c: Calibration | null): string {
  if (!c) return "Not calibrated yet.";
  const v = c.validation;
  if (!v) return "Calibrated, but not checked yet. Run the check so I know how far to trust it.";
  const measured = `Accurate to about ${shareWords(v.meanPx, v.width)} (${Math.round(v.meanPx)} px on average: ${Math.round(v.freshPx)} px on ${v.points} fresh dots, ${Math.round(v.leftOutPx)} px on calibration dots left out of the fit; the worse one counts).`;
  if (c.quality === "good") return `${measured} I can tell which part of the screen you are looking at (a ${c.cols} by ${c.rows} grid) and which marked thing you are on when only one is that close. Not the exact spot.`;
  if (c.quality === "rough") return `${measured} That is rough: I will only use it to start the pointer from your side of the screen.`;
  return `${measured} That is not good enough to use, so the pointer starts from the middle. More light on your face, no glare on glasses, then recalibrate.`;
}

/** The radius, in pixels on the current screen, inside which the estimate is expected to fall: the measured mean miss, scaled to this screen. */
export function errorRadius(c: Calibration, width: number): number {
  return c.validation ? c.validation.ofWidth * width : width;
}

export const DRIFT_ROTATION = 0.22; // about 13 degrees of head turn
export const DRIFT_POSITION = 0.12;
export const STALE_MS = 20 * 60 * 1000;

/** Has the head moved far from where the calibration was made, or has it been a long time? Then a re-check is asked for. */
export function drifted(c: Calibration, f: Features, now: number): "moved" | "old" | null {
  if (Math.hypot(f[2] - c.pose[0]!, f[3] - c.pose[1]!) > DRIFT_ROTATION || Math.hypot(f[4] - c.pose[2]!, f[5] - c.pose[3]!) > DRIFT_POSITION) return "moved";
  return c.at > 0 && now - c.at > STALE_MS ? "old" : null;
}

// ---------- smoothing and fixations ----------

/** The One-Euro filter (Casiez, Roussel and Vogel, 2012) on one number: little lag when it moves fast, little jitter when it rests. */
class OneEuro1 {
  private x: number | null = null;
  private dx = 0;
  private t = 0;
  constructor(private minCutoff: number, private beta: number, private dCutoff = 1) {}
  private static alpha(cutoff: number, dt: number): number {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }
  push(v: number, tMs: number): number {
    if (this.x === null) {
      this.x = v;
      this.t = tMs;
      return v;
    }
    const dt = Math.max(1e-3, (tMs - this.t) / 1000);
    this.t = tMs;
    const d = (v - this.x) / dt;
    this.dx += OneEuro1.alpha(this.dCutoff, dt) * (d - this.dx);
    this.x += OneEuro1.alpha(this.minCutoff + this.beta * Math.abs(this.dx), dt) * (v - this.x);
    return this.x;
  }
  clear(): void {
    this.x = null;
    this.dx = 0;
  }
}

/** One-Euro on a point given in fractions of the screen. */
export class OneEuro {
  private fx: OneEuro1;
  private fy: OneEuro1;
  constructor(minCutoff = 0.8, beta = 2.5) {
    this.fx = new OneEuro1(minCutoff, beta);
    this.fy = new OneEuro1(minCutoff, beta);
  }
  push(p: { x: number; y: number }, tMs: number): { x: number; y: number } {
    return { x: this.fx.push(p.x, tMs), y: this.fy.push(p.y, tMs) };
  }
  clear(): void {
    this.fx.clear();
    this.fy.clear();
  }
}

/**
 * Fixation by dispersion (I-DT): the eyes are resting when the estimates of the last stretch stay inside a small box.
 * `push` returns how long they have rested so far (ms) and where; 0 while they move.
 */
export class Fixation {
  private window: { x: number; y: number; t: number }[] = [];
  /** `dispersion` is the box's width plus height, in fractions of the screen; the measured miss sets it. */
  constructor(private dispersion = 0.12) {}
  push(p: { x: number; y: number }, tMs: number): { ms: number; x: number; y: number } {
    this.window.push({ ...p, t: tMs });
    const spread = () => {
      const xs = this.window.map((w) => w.x);
      const ys = this.window.map((w) => w.y);
      return Math.max(...xs) - Math.min(...xs) + (Math.max(...ys) - Math.min(...ys));
    };
    while (this.window.length > 1 && spread() > this.dispersion) this.window.shift();
    const n = this.window.length;
    return { ms: tMs - this.window[0]!.t, x: this.window.reduce((s, w) => s + w.x, 0) / n, y: this.window.reduce((s, w) => s + w.y, 0) / n };
  }
  clear(): void {
    this.window = [];
  }
}

/** A point (fractions of the screen) rounded to the middle of its cell in a cols x rows grid. */
export function quantise(p: { x: number; y: number }, cols: number, rows: number): { x: number; y: number; col: number; row: number } {
  const col = Math.max(0, Math.min(cols - 1, Math.floor(p.x * cols)));
  const row = Math.max(0, Math.min(rows - 1, Math.floor(p.y * rows)));
  return { col, row, x: (col + 0.5) / cols, y: (row + 0.5) / rows };
}

// ---------- snapping to the things on screen ----------

export interface Thing {
  id: string;
  left: number;
  top: number;
  right: number;
  bottom: number;
}

/** How far a point is from a box, in pixels (0 inside it). */
export function distanceTo(p: { x: number; y: number }, b: Thing): number {
  return Math.hypot(Math.max(b.left - p.x, 0, p.x - b.right), Math.max(b.top - p.y, 0, p.y - b.bottom));
}

export type Snap = { kind: "one"; id: string } | { kind: "several"; ids: string[] } | { kind: "none" };

/**
 * The registered thing the eyes are on: the only one within `radius` (the measured miss) of the estimate. When two or
 * more are that close the answer is "one of these", never a pick. Pass leaf things only (a card and the items in it would always tie).
 */
export function snap(p: { x: number; y: number }, radius: number, things: Thing[]): Snap {
  const near = things.map((t) => ({ t, d: distanceTo(p, t) })).filter((x) => x.d <= radius).sort((a, b) => a.d - b.d);
  if (near.length === 0) return { kind: "none" };
  if (near.length === 1) return { kind: "one", id: near[0]!.t.id };
  return { kind: "several", ids: near.map((x) => x.t.id) };
}

/** Is the target among what the eyes are on? "on": it alone. "near": it and others (region-level arrival). */
export function landed(s: Snap, id: string): "on" | "near" | null {
  if (s.kind === "one") return s.id === id ? "on" : null;
  return s.kind === "several" && s.ids.includes(id) ? "near" : null;
}

export const DWELL_MS = 700;
export const AWAY_MS = 3000;

/** Feed it "on the target?" with a clock; it answers "arrived" once, and "repeat" once if the eyes stay away for 3 s first. */
export class Dwell {
  private since: number | null = null;
  private awaySince: number | null = null;
  private arrived = false;
  private repeated = false;
  constructor(private dwellMs = DWELL_MS, private awayMs = AWAY_MS) {}
  update(on: boolean | null, now: number): "arrived" | "repeat" | null {
    if (this.arrived) return null;
    if (on === null) {
      // no estimate (face lost): neither dwelling nor away
      this.since = null;
      this.awaySince = null;
      return null;
    }
    if (on) {
      this.awaySince = null;
      this.since ??= now;
      if (now - this.since >= this.dwellMs) {
        this.arrived = true;
        return "arrived";
      }
      return null;
    }
    this.since = null;
    this.awaySince ??= now;
    if (!this.repeated && now - this.awaySince >= this.awayMs) {
      this.repeated = true;
      return "repeat";
    }
    return null;
  }
}

// ---------- the dots ----------

const at = (xs: number[], ys: number[]) => ys.flatMap((y) => xs.map((x) => ({ x, y })));
/** Nine dots, three by three, the middle first so the first look is the easy one. */
export const DOTS_9: { x: number; y: number }[] = [{ x: 0.5, y: 0.5 }, ...at([0.08, 0.5, 0.92], [0.1, 0.5, 0.9]).filter((p) => !(p.x === 0.5 && p.y === 0.5))];
/** The quick mode: the middle and the four corners. */
export const DOTS_5: { x: number; y: number }[] = [{ x: 0.5, y: 0.5 }, { x: 0.08, y: 0.1 }, { x: 0.92, y: 0.1 }, { x: 0.92, y: 0.9 }, { x: 0.08, y: 0.9 }];
/** Four fresh dots for the check, between the calibration dots. */
export const DOTS_CHECK: { x: number; y: number }[] = [{ x: 0.29, y: 0.3 }, { x: 0.71, y: 0.3 }, { x: 0.71, y: 0.7 }, { x: 0.29, y: 0.7 }];
export const DOT_MS = 1700;
/** The first part of each dot is the eyes getting there; it is not sampled. */
export const SETTLE_MS = 550;
