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

export interface LabCompare {
  a: CompareRun | null;
  b: CompareRun | null;
  ratios: Record<string, number | null> | null;
  ratio_keys: Record<string, string[]>;
  runs: CompareChoice[];
  asked: { a: string | null; b: string | null };
  missing: string[];
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
