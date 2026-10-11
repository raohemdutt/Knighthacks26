/**
 * GET /runtime/stats (R5): the box's and the lab's numbers, assembled on the gateway from what is already recorded
 * (contexts/telemetry/application/stats.py). Every figure names its record, or is null with the reason.
 */

export interface StatFigure {
  value: number | null;
  unit: string;
  node: string | null;
  ts: string | null;
  source: string | null;
  low_confidence: boolean;
  reason: string | null;
}

export interface EvalRow {
  model_version: string;
  deployed_ts: string | null;
  previous_version: string | null;
  task: string | null;
  run_id: string | null;
  decision: string | null;
  before: number | null;
  after: number | null;
  teacher: number | null;
  ts: string | null;
  source: string | null;
  reason: string | null;
  low_confidence: boolean;
}

export type FigureKey = "detector_fps" | "student_tok_s" | "caption_latency_s" | "trainer_samples_per_s" | "train_minutes" | "train_loss" | "teacher_tok_s";

export type RuntimeStats = Record<FigureKey, StatFigure> & { eval_scores: { models: EvalRow[]; reason: string | null } };

/** The figures in the order the page shows them, grouped by the machine that does the work. */
export const STAT_GROUPS: { title: string; where: string; figures: { key: FigureKey; label: string }[] }[] = [
  {
    title: "The box",
    where: "Jetson Orin Nano, on the desk",
    figures: [
      { key: "detector_fps", label: "Detector speed" },
      { key: "student_tok_s", label: "On-device model" },
      { key: "caption_latency_s", label: "One caption" },
    ],
  },
  {
    title: "The lab",
    where: "AMD Strix Halo pods",
    figures: [
      { key: "teacher_tok_s", label: "Teacher model" },
      { key: "trainer_samples_per_s", label: "Trainer speed" },
      { key: "train_minutes", label: "Last training run" },
      { key: "train_loss", label: "Training loss" },
    ],
  },
];

export function statValue(f: StatFigure): string {
  if (f.value === null) return "not recorded";
  const v = f.value;
  const n = Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : Number(v.toPrecision(3)).toLocaleString();
  if (f.unit === "loss") return n;
  if (f.unit === "frames/s") return `${n} fps`;
  return `${n} ${f.unit}`;
}

export function pct(x: number | null): string {
  return x === null ? "–" : `${Math.round(x * 100)}%`;
}
