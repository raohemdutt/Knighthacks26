/**
 * The Learning lab's Compare tab (L31): two training runs side by side, GET /api/learning/lab/compare?a=&b=.
 * The gateway's side is contexts/learning/application/lab_compare.py; every ratio (b over a) is computed there once.
 */

export interface CompareStage {
  seconds?: number | null;
  joules?: number | null;
  mean_w?: number | null;
  peak_w?: number | null;
  idle_j?: number | null;
  node?: string | null;
  tokens?: number | null;
  j_per_token?: number | null;
  j_per_example?: number | null;
  steps?: number | null;
  j_per_step?: number | null;
  j_per_example_seen?: number | null;
  j_per_train_row?: number | null;
  gross_window_j?: number | null;
  overlap_s?: number | null;
  loss_first?: number | null;
  loss_last?: number | null;
}

export interface CompareTrace {
  name: string;
  /** [seconds after the run's start, watts] */
  points: [number, number][];
}

export interface CompareRun {
  run_id: string;
  act: string | null;
  members: number | null;
  /** The pods this act ran on, such as ["pod-d", "pod-c"]. Absent on older gateways. */
  pods?: string[] | null;
  /** The pods the teacher labeled on, such as ["pod-c", "pod-d"]. Absent on older gateways. */
  teacher_nodes?: string[] | null;
  /** Each pod that trained this act, by pod; absent or null when one pod trained it. */
  training_pods?: Record<string, CompareStage> | null;
  examples: number | null;
  train_rows: number | null;
  held_out: number | null;
  outcome: string | null;
  seconds: Partial<Record<"labeling" | "waiting" | "training" | "packaging" | "total", number>>;
  labeling: CompareStage | null;
  training: CompareStage | null;
  packaging: CompareStage | null;
  box: CompareStage | null;
  pods_joules: number | null;
  j_per_example: number | null;
  traces: Partial<Record<"pod-c" | "pod-d" | "box", CompareTrace>>;
  source: { kind: "loop" | "log"; ref: string; stamp: string | null; rule: string | null; split: string | null };
  idle: Record<string, { calibrated_w: number | null; quiet_w: number | null; note?: string }> | null;
  /** Why the run has no joules, when it has none. */
  reason: string | null;
}

export interface CompareChoice {
  run_id: string;
  act: string | null;
  members: number | null;
  examples: number | null;
  measured: boolean;
  stamp: string | null;
  source: string;
}

/** One node's watts on the loop's own clock (seconds after the loop's start), with its idle floors. */
export interface SeriesNode {
  name: string;
  idle_w: number | null;
  quiet_w?: number | null;
  points: [number, number][];
}

/** When each act ran on that clock, and its stages as [start, end] seconds. */
export interface SeriesRun {
  run_id: string;
  act: string | null;
  start_s: number;
  end_s: number;
  stages: Partial<Record<"labeling" | "waiting" | "training" | "packaging", [number, number]>>;
}

/** Both acts of one loop on one clock; present when both acts come from the same loop. */
export interface CompareSeries {
  zero: string | null;
  axis: string | null;
  nodes: Partial<Record<"pod-c" | "pod-d" | "box", SeriesNode>>;
  runs: SeriesRun[];
}

export interface LabCompare {
  a: CompareRun | null;
  b: CompareRun | null;
  ratios: Record<string, number | null> | null;
  ratio_keys: Record<string, string[]>;
  runs: CompareChoice[];
  asked: { a: string | null; b: string | null };
  missing: string[];
  /** The loop's clock; absent on older gateways and when the two acts come from different loops. */
  series?: CompareSeries | null;
  same_loop?: boolean | null;
  teacher_nodes?: string[] | null;
}

/** One row of the table: what it is, the ratio key, how to read it from a run, its unit and the idle floor beside. */
export interface CompareRow {
  key: string;
  label: string;
  group: string;
  unit: string;
  value: (r: CompareRun) => number | null | undefined;
  idle?: (r: CompareRun) => number | null | undefined;
}

export const COMPARE_ROWS: CompareRow[] = [
  { key: "examples", label: "Examples", group: "The run", unit: "", value: (r) => r.examples },
  { key: "members", label: "Members pooled", group: "The run", unit: "", value: (r) => r.members },
  { key: "labeling_s", label: "Labeling", group: "Seconds", unit: "s", value: (r) => r.seconds.labeling },
  { key: "training_s", label: "Training", group: "Seconds", unit: "s", value: (r) => r.seconds.training },
  { key: "packaging_s", label: "Packaging", group: "Seconds", unit: "s", value: (r) => r.seconds.packaging },
  { key: "act_seconds", label: "Start to finish", group: "Seconds", unit: "s", value: (r) => r.seconds.total },
  { key: "labeling_j", label: "Labeling (pod C)", group: "Joules", unit: "J", value: (r) => r.labeling?.joules, idle: (r) => r.labeling?.idle_j },
  { key: "labeling_j_per_token", label: "Labeling per token", group: "Joules", unit: "J", value: (r) => r.labeling?.j_per_token },
  { key: "labeling_j_per_example", label: "Labeling per example", group: "Joules", unit: "J", value: (r) => r.labeling?.j_per_example },
  { key: "training_j", label: "Training (pod D)", group: "Joules", unit: "J", value: (r) => r.training?.joules, idle: (r) => r.training?.idle_j },
  { key: "training_j_per_step", label: "Training per step", group: "Joules", unit: "J", value: (r) => r.training?.j_per_step },
  { key: "training_j_per_train_row", label: "Training per example", group: "Joules", unit: "J", value: (r) => r.training?.j_per_train_row },
  { key: "packaging_j", label: "Packaging (pod D)", group: "Joules", unit: "J", value: (r) => r.packaging?.joules, idle: (r) => r.packaging?.idle_j },
  { key: "pods_joules", label: "Both pods, total", group: "Joules", unit: "J", value: (r) => r.pods_joules },
  { key: "j_per_example", label: "Per example, total", group: "Joules", unit: "J", value: (r) => r.j_per_example },
  { key: "box_j", label: "The box (Jetson)", group: "Joules", unit: "J", value: (r) => r.box?.joules, idle: (r) => r.box?.idle_j },
];

/** A run's name on the card: "Alone (100 examples)", "With a friend (200 examples)", or its id. */
export function runName(r: Pick<CompareRun, "act" | "members" | "examples" | "run_id">): string {
  const n = r.examples ? ` (${r.examples} examples)` : "";
  if (r.members === 1) return `Alone${n}`;
  if (r.members && r.members > 1) return `With ${r.members === 2 ? "a friend" : `${r.members - 1} friends`}${n}`;
  return r.run_id;
}

export function fmtValue(v: number | null | undefined, unit: string): string {
  if (v === null || v === undefined) return "not measured";
  const n = Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : Number(v.toPrecision(3)).toLocaleString();
  return unit ? `${n} ${unit}` : n;
}

/** b over a as "×1.82", or "" when there is nothing to divide. */
export function fmtRatio(x: number | null | undefined): string {
  return typeof x === "number" ? `×${x.toFixed(x >= 10 ? 1 : 2)}` : "";
}

/* ---------- only what the box recorded ---------- */

/**
 * The gateway also reads the committed rehearsal under fixtures/learning/ (lab_compare.default_dirs). The screen shows
 * only what a box has really recorded, so a run whose record is a committed fixture is left out, here and nowhere else.
 */
const COMMITTED = "fixtures/";

export function isRecordedRun(r: CompareRun | null | undefined): r is CompareRun {
  return !!r && !(r.source.kind === "loop" && r.source.ref.startsWith(COMMITTED));
}

export function isRecordedChoice(c: CompareChoice): boolean {
  return !c.source.startsWith(COMMITTED);
}

/** The response with every committed fixture taken out: a run that is one becomes absent, and so do the ratios. */
export function recordedOnly(d: LabCompare): LabCompare {
  const a = isRecordedRun(d.a) ? d.a : null;
  const b = isRecordedRun(d.b) ? d.b : null;
  return { ...d, a, b, ratios: a && b ? d.ratios : null, runs: d.runs.filter(isRecordedChoice) };
}

/** Act 1 and act 2 by default: the newest measured run done alone, and the newest measured run pooled with friends. */
export function defaultPair(runs: CompareChoice[]): { a: string | null; b: string | null } {
  const measured = runs.filter((c) => c.measured);
  const alone = measured.find((c) => c.act === "solo" || c.members === 1);
  const pooled = measured.find((c) => (c.members ?? 0) > 1);
  return { a: alone?.run_id ?? null, b: pooled?.run_id ?? null };
}

/** "2.00x", or "" when there is nothing to divide. */
export function timesX(x: number | null | undefined): string {
  return typeof x === "number" ? `${x.toFixed(x >= 10 ? 1 : 2)}x` : "";
}

/** Ratios where act 2 doing better means a number below 1 (energy per unit of work), or above 1 (examples). */
export const LOWER_IS_BETTER = new Set(["labeling_j_per_token", "labeling_j_per_example", "training_j_per_step", "training_j_per_train_row", "j_per_example"]);
export const HIGHER_IS_BETTER = new Set(["examples", "members"]);

/* ---------- the hero numbers ---------- */

const SMALL = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"];

/** A ratio in plain words: 2 is "twice", 3 is "three times", 0.9 is "nine tenths", 0.5 is "half", 1.3 is "1.3 times". */
export function ratioWords(x: number): string {
  const whole = Math.round(x);
  if (x >= 1.95 && Math.abs(x - whole) < 0.05) return whole === 2 ? "twice" : `${SMALL[whole] ?? whole} times`;
  if (Math.abs(x - 1) < 0.05) return "the same";
  if (x > 0 && x < 1) {
    const tenths = Math.round(x * 10);
    if (tenths === 5) return "half";
    if (tenths >= 1 && tenths <= 9) return `${SMALL[tenths]} tenth${tenths === 1 ? "" : "s"}`;
    return `${x.toFixed(2)} times`;
  }
  return `${x.toFixed(1)} times`;
}

/** "2.0x" at or above 1, "0.90x" below it, so a small change still shows. */
export function heroX(x: number): string {
  return `${x.toFixed(x >= 1 ? 1 : 2)}x`;
}

export interface HeroTile {
  key: "examples" | "act_seconds" | "j_per_example";
  x: number;
  /** "2.0x" */
  times: string;
  /** "the examples" */
  noun: string;
  /** "twice the examples" */
  words: string;
  /** Act 1's and act 2's own numbers behind the ratio. */
  from: number | null;
  to: number | null;
  unit: string;
  /** Act 2 did better on this line. */
  good: boolean;
}

/** The three tiles above the acts, read from the comparison's ratios; a ratio the gateway did not compute is left out. */
export function heroTiles(d: Pick<LabCompare, "a" | "b" | "ratios">): HeroTile[] {
  const r = d.ratios ?? {};
  const spec: { key: HeroTile["key"]; noun: string; unit: string; higher: boolean; value: (run: CompareRun) => number | null | undefined }[] = [
    { key: "examples", noun: "the examples", unit: "", higher: true, value: (run) => run.examples },
    { key: "act_seconds", noun: "the time", unit: "s", higher: false, value: (run) => run.seconds.total },
    { key: "j_per_example", noun: "the energy per example", unit: "J", higher: false, value: (run) => run.j_per_example },
  ];
  const out: HeroTile[] = [];
  for (const sp of spec) {
    const x = r[sp.key];
    if (typeof x !== "number" || !Number.isFinite(x) || x <= 0) continue;
    const w = ratioWords(x);
    const of = /tenths?$|^half$/.test(w) ? " of " : " ";
    out.push({
      key: sp.key,
      x,
      times: heroX(x),
      noun: sp.noun,
      words: `${w}${of}${sp.noun}`,
      from: d.a ? (sp.value(d.a) ?? null) : null,
      to: d.b ? (sp.value(d.b) ?? null) : null,
      unit: sp.unit,
      good: sp.higher ? x > 1 : x < 1,
    });
  }
  return out;
}

/** How many pods a list of nodes names, counting only the pods (never the box). */
function podCount(nodes: string[] | null | undefined): number {
  return (nodes ?? []).filter((n) => n.startsWith("pod-")).length;
}

/** "pods C and D" or "pod D", from ["pod-c", "pod-d"]. */
export function podWords(nodes: string[]): string {
  const letters = nodes
    .filter((n) => n.startsWith("pod-"))
    .map((n) => n.slice(4).toUpperCase())
    .sort();
  if (letters.length === 0) return "";
  if (letters.length === 1) return `pod ${letters[0]}`;
  return `pods ${letters.slice(0, -1).join(", ")} and ${letters[letters.length - 1]}`;
}

/** The pods that trained a run: each one in training_pods, else the pods the run names, else none. */
export function trainingPods(r: CompareRun): string[] {
  if (r.training_pods && Object.keys(r.training_pods).length) return Object.keys(r.training_pods);
  return (r.pods ?? []).filter((n) => n.startsWith("pod-"));
}

/** The one line under the hero: who labeled and where act 2 trained, from the runs' own fields. */
export function heroLine(d: Pick<LabCompare, "a" | "b" | "teacher_nodes">): string | null {
  const run = d.b ?? d.a;
  if (!run) return null;
  const teachers = run.teacher_nodes ?? d.teacher_nodes ?? null;
  const parts: string[] = [];
  const nT = podCount(teachers);
  if (nT >= 2) parts.push(`labeled by the 122B across ${nT === 2 ? "both pods" : `${nT} pods`}, with pictures`);
  else if (nT === 1 && teachers) parts.push(`labeled by the 122B on ${podWords(teachers)}, with pictures`);
  if (d.b) {
    const pods = trainingPods(d.b);
    if (pods.length >= 2) parts.push(`act 2 trained on ${pods.length === 2 ? "both pods" : `${pods.length} pods`} at once`);
    else if (pods.length === 1) parts.push(`act 2 trained on ${podWords(pods)}`);
  }
  return parts.length ? parts.join("; ") : null;
}

/* ---------- training now, only while it is really moving ---------- */

/** A run counts as training now only if its newest step or stage event is younger than this. */
export const FRESH_MS = 2 * 60 * 1000;

/** What the run list says about a run's motion: the gateway's run_status and, when it sends one, its newest event. */
export interface RunMotion {
  run_status?: string | null;
  last_event_ts?: string | null;
  ended?: string | null;
  started?: string | null;
}

/**
 * Training now, for the buddies: run_status is "running" (never "stalled", "finished" or "failed") and the run's newest
 * event (last_event_ts when the gateway sends it, else its ended time, else its start) is less than FRESH_MS old. A loop
 * that died mid-run keeps run_status "running" on an older gateway; its events stop, so it falls out here.
 */
export function isFreshRun(r: RunMotion, nowMs: number): boolean {
  if (r.run_status !== "running") return false;
  const ts = r.last_event_ts ?? r.ended ?? r.started ?? null;
  if (!ts) return false;
  const t = Date.parse(ts);
  if (!Number.isFinite(t)) return false;
  return nowMs - t < FRESH_MS;
}
