/**
 * Gaze, as far as a laptop webcam can honestly give it: which REGION of the screen someone is looking at, not which pixel.
 *
 * Pure maths, no imports (scripts/check-guide.mjs runs it in node). The camera and the face model are in lib/gazeTracker.ts.
 *
 *   features(landmarks)        four numbers from one face: where the irises sit in the eyes (x, y) and where the head points (x, y)
 *   fitCalibration(samples)    a ridge regression from those to screen coordinates, from five dots looked at in turn
 *   quality of the fit         "good": a 3x3 grid (4x3 when the fit is tight); "rough": only "which half"; "poor": not used
 *   MedianSmoother             a moving median, so one bad frame does not move the estimate
 *   Dwell                      "the eyes have rested on the target for 700 ms" and "they have been elsewhere for 3 s"
 *
 * Nothing here stores where anyone looked: a calibration is five weights per axis, and the smoother keeps half a second.
 */

export interface Landmark {
  x: number;
  y: number;
  z?: number;
}
export type Features = [number, number, number, number];

// MediaPipe Face Landmarker indices (478 points with the irises refined)
const EYES = [
  { outer: 33, inner: 133, top: 159, bottom: 145, iris: 468 },
  { outer: 263, inner: 362, top: 386, bottom: 374, iris: 473 },
];
const NOSE = 1;

/** null when the face has no iris points or an eye is closed too far to read. */
export function features(lm: Landmark[]): Features | null {
  if (lm.length < 478) return null;
  let hx = 0;
  let hy = 0;
  for (const e of EYES) {
    const a = lm[e.outer]!;
    const b = lm[e.inner]!;
    const iris = lm[e.iris]!;
    const wx = b.x - a.x;
    const wy = b.y - a.y;
    const width2 = wx * wx + wy * wy;
    if (width2 < 1e-8) return null;
    const open = Math.hypot(lm[e.top]!.x - lm[e.bottom]!.x, lm[e.top]!.y - lm[e.bottom]!.y) / Math.sqrt(width2);
    if (open < 0.12) return null; // a blink
    const t = ((iris.x - a.x) * wx + (iris.y - a.y) * wy) / width2; // 0 at the outer corner, 1 at the inner
    const off = ((iris.x - a.x) * wy - (iris.y - a.y) * wx) / width2; // signed distance from the corner-to-corner line
    // the two eyes mirror each other: turn both into "towards the image's right" and "towards the image's bottom"
    const right = wx > 0 ? 1 : -1;
    hx += right * (t - 0.5);
    hy += -right * off;
  }
  const l = lm[EYES[0]!.outer]!;
  const r = lm[EYES[1]!.outer]!;
  const span = Math.hypot(r.x - l.x, r.y - l.y);
  if (span < 1e-4) return null;
  const nose = lm[NOSE]!;
  const yaw = (nose.x - (l.x + r.x) / 2) / span;
  const pitch = (nose.y - (l.y + r.y) / 2) / span;
  return [hx / 2, hy / 2, yaw, pitch];
}

export function median(values: number[]): number {
  if (values.length === 0) return 0;
  const s = [...values].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m]! : (s[m - 1]! + s[m]!) / 2;
}

/** Solve A x = b for a small symmetric system (Gaussian elimination with pivoting). */
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

export interface Sample {
  f: Features;
  /** Where the person was asked to look, as a fraction of the screen (0..1 each way). */
  x: number;
  y: number;
}

export type Quality = "good" | "rough" | "poor";

export interface Calibration {
  mean: number[];
  scale: number[];
  wx: number[]; // 4 weights + the constant
  wy: number[];
  /** Average miss at the calibration dots, as a fraction of the screen's diagonal (measured leaving each dot out in turn). */
  residual: number;
  /** How much the estimate jitters while the eyes rest on one dot, same unit. */
  scatter: number;
  quality: Quality;
  /** The grid the estimate is rounded to: 4x3 or 3x3 when good, 2x2 when rough. */
  cols: number;
  rows: number;
}

const LAMBDA = 0.6;

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
  for (let j = 0; j < d; j++) A[j]![j]! += LAMBDA * rows.length * 0.02; // not on the constant
  return solve(A, b);
}

function standardise(samples: Sample[]): { mean: number[]; scale: number[] } {
  const mean = [0, 1, 2, 3].map((j) => samples.reduce((s, v) => s + v.f[j]!, 0) / samples.length);
  const scale = [0, 1, 2, 3].map((j) => Math.sqrt(samples.reduce((s, v) => s + (v.f[j]! - mean[j]!) ** 2, 0) / samples.length) || 1);
  return { mean, scale };
}

const z = (f: Features, mean: number[], scale: number[]) => f.map((v, j) => (v - mean[j]!) / scale[j]!);
const dot = (w: number[], r: number[]) => r.reduce((s, v, j) => s + v * w[j]!, w[w.length - 1]!);

function fit(samples: Sample[]): Pick<Calibration, "mean" | "scale" | "wx" | "wy"> {
  const { mean, scale } = standardise(samples);
  const rows = samples.map((s) => z(s.f, mean, scale));
  return { mean, scale, wx: ridge(rows, samples.map((s) => s.x)), wy: ridge(rows, samples.map((s) => s.y)) };
}

/** Where the fit puts a face's features, as a fraction of the screen (it may fall a little outside 0..1). */
export function predict(c: Pick<Calibration, "mean" | "scale" | "wx" | "wy">, f: Features): { x: number; y: number } {
  const r = z(f, c.mean, c.scale);
  return { x: dot(c.wx, r), y: dot(c.wy, r) };
}

/** Thresholds, as fractions of the screen's diagonal. A third of the screen is about 0.33 across. */
export const GOOD_RESIDUAL = 0.13;
export const FINE_RESIDUAL = 0.07;
export const ROUGH_RESIDUAL = 0.26;

/**
 * Fit the five dots and say how far to trust it. The residual is honest: each dot is predicted from a fit that never saw it.
 * `aspect` is the screen's width over its height, so a miss is measured in the same unit both ways.
 */
export function fitCalibration(samples: Sample[], aspect = 16 / 10): Calibration | null {
  const dots = [...new Set(samples.map((s) => `${s.x},${s.y}`))];
  if (dots.length < 5 || samples.length < 15) return null;
  const all = fit(samples);
  const diag = Math.hypot(aspect, 1);
  const miss = (p: { x: number; y: number }, s: { x: number; y: number }) => Math.hypot((p.x - s.x) * aspect, p.y - s.y) / diag;
  let residual = 0;
  let scatter = 0;
  for (const key of dots) {
    const here = samples.filter((s) => `${s.x},${s.y}` === key);
    const without = fit(samples.filter((s) => `${s.x},${s.y}` !== key));
    const ps = here.map((s) => predict(without, s.f));
    const centre = { x: median(ps.map((p) => p.x)), y: median(ps.map((p) => p.y)) };
    residual += miss(centre, here[0]!) / dots.length;
    const own = here.map((s) => predict(all, s.f));
    const mid = { x: median(own.map((p) => p.x)), y: median(own.map((p) => p.y)) };
    scatter += median(own.map((p) => miss(p, mid))) / dots.length;
  }
  const quality: Quality = residual <= GOOD_RESIDUAL && scatter <= 0.08 ? "good" : residual <= ROUGH_RESIDUAL && scatter <= 0.16 ? "rough" : "poor";
  const fine = quality === "good" && residual <= FINE_RESIDUAL && scatter <= 0.04;
  return { ...all, residual, scatter, quality, cols: quality === "good" ? (fine ? 4 : 3) : 2, rows: quality === "good" ? 3 : 2 };
}

/** What to tell the person, in words. */
export function qualityWords(c: Calibration | null): string {
  if (!c) return "Not calibrated yet.";
  if (c.quality === "good") return `Good: I can tell which part of the screen you are looking at (a ${c.cols} by ${c.rows} grid), not the exact spot.`;
  if (c.quality === "rough") return "Rough: I will only use it to start the pointer from your side of the screen.";
  return "Poor: I could not read your eyes well enough, so the pointer starts from the middle. Try more light on your face, then recalibrate.";
}

/** A moving median over the last `n` estimates. */
export class MedianSmoother {
  private xs: number[] = [];
  private ys: number[] = [];
  constructor(private n = 7) {}
  push(p: { x: number; y: number }): { x: number; y: number } {
    this.xs.push(p.x);
    this.ys.push(p.y);
    if (this.xs.length > this.n) {
      this.xs.shift();
      this.ys.shift();
    }
    return { x: median(this.xs), y: median(this.ys) };
  }
  clear(): void {
    this.xs = [];
    this.ys = [];
  }
}

/** A point (fractions of the screen) rounded to the middle of its cell in a cols x rows grid. */
export function quantise(p: { x: number; y: number }, cols: number, rows: number): { x: number; y: number; col: number; row: number } {
  const col = Math.max(0, Math.min(cols - 1, Math.floor(p.x * cols)));
  const row = Math.max(0, Math.min(rows - 1, Math.floor(p.y * rows)));
  return { col, row, x: (col + 0.5) / cols, y: (row + 0.5) / rows };
}

/**
 * Is the looked-at cell on the target? True when the cell and the target's box (all in fractions of the screen) overlap,
 * with half a cell of slack: region-level, as the estimate is.
 */
export function onTarget(cell: { col: number; row: number }, cols: number, rows: number, box: { x0: number; y0: number; x1: number; y1: number }): boolean {
  const x0 = (cell.col - 0.5) / cols;
  const x1 = (cell.col + 1.5) / cols;
  const y0 = (cell.row - 0.5) / rows;
  const y1 = (cell.row + 1.5) / rows;
  const cx = (box.x0 + box.x1) / 2;
  const cy = (box.y0 + box.y1) / 2;
  return cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1;
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

/** The five dots, as fractions of the screen: the middle, then the four corners. */
export const DOTS: { x: number; y: number }[] = [
  { x: 0.5, y: 0.5 },
  { x: 0.08, y: 0.1 },
  { x: 0.92, y: 0.1 },
  { x: 0.92, y: 0.9 },
  { x: 0.08, y: 0.9 },
];
export const DOT_MS = 1200;
/** The first part of each dot is the eyes getting there; it is not sampled. */
export const SETTLE_MS = 450;
