/**
 * The Learning lab's other tabs: speed and energy (GET /api/learning/lab/bench), the routing what-if
 * (POST /api/learning/lab/routing/simulate) and the experiment timeline (GET /api/learning/lab/experiments).
 * The gateway's read models are contexts/learning/application/lab_bench.py, lab_routing.py and lab_experiments.py.
 *
 * A number is a LabNum with a `kind`: "measured" is a row's own value, "derived" is worked out from rows with the
 * formula shown. The screen adds two more labels of its own: "modelled" (the what-if) and "sample" (the sample run).
 */
import type { LabNum, LabSource } from "./lab";

export type LabTab = "run" | "speed" | "routing" | "experiments";
export const LAB_TABS: { key: LabTab; title: string }[] = [
  { key: "run", title: "Training run" },
  { key: "speed", title: "Speed and energy" },
  { key: "routing", title: "Routing" },
  { key: "experiments", title: "Experiments" },
];

export interface BenchPoint extends Omit<LabNum, "value" | "source"> {
  value: number;
  source: LabSource;
  kind: "measured" | "derived";
  formula: string | null;
  metric?: string;
  node?: string | null;
  ts?: string | null;
  line?: number;
  recorded_by?: string | null;
  sampler?: { sampler: string | null; tool: string | null; period_ms: number | null; samples: number | null; error: string | null } | null;
  x?: number | null;
  runs?: number | null;
  statistic?: string | null;
  spread?: { min?: number; max?: number; sd?: number } | null;
  model?: string | null;
  build?: string | null;
  session?: string | null;
  state?: string | null;
  inputs?: string[];
}

export interface BenchSweep {
  id: string;
  family: string;
  node: string | null;
  session: string | null;
  model: string | null;
  build: string | null;
  chip: string | null;
  mode: string | null;
  max_tokens: number | null;
  ts: string | null;
  recorded_by: string | null;
  batches: number[];
  metrics: Record<string, BenchPoint[]>;
  other_metrics: string[];
  points: number;
  low_confidence: number;
  confidence_note: string;
}

export interface BenchStudent {
  node: string;
  watts: BenchPoint;
  tok_s: BenchPoint | null;
  workload: string | null;
  j_per_token: BenchPoint | null;
  request_s: BenchPoint | null;
  j_per_answer: BenchPoint | null;
  answer_tokens: BenchPoint | null;
  j_per_token_request: BenchPoint | null;
  first_token_s: BenchPoint | null;
}

export interface BenchPowerNode {
  node: string;
  idle: BenchPoint | null;
  serving: BenchPoint | null;
  peak: BenchPoint | null;
  idle_rows: number;
  serving_rows: number;
}

export interface LabBench {
  file: string;
  rows: number;
  skipped: { line: number; problem: string }[];
  low_confidence_rows: number;
  newest: string | null;
  families: { family: string; metrics: string[]; nodes: string[]; rows: number; charted: number; low_confidence: number }[];
  sweeps: BenchSweep[];
  student: BenchStudent | null;
  power: { nodes: BenchPowerNode[]; absent: string[] };
  series: { metric: string; family: string; node: string | null; points: BenchPoint[] }[];
  charted_rows: number;
  not_charted: { metric: string; node: string | null; rows: number; of: number; unit: string | null; latest: BenchPoint; lines: number[] }[];
  empty_reason: string | null;
}

/* ---------- routing what-if ---------- */

export type PolicyName = "capability" | "energy";
export const POLICY_TITLE: Record<PolicyName, string> = { capability: "Capability only", energy: "Energy aware" };

export interface SimKind {
  kind: string;
  title: string;
  source: string;
  own_space: boolean;
  student_can: boolean;
  tokens: number;
  locked: boolean;
  share: number;
  n: number;
}

export interface SimRun {
  policy: PolicyName;
  targets: string; // one letter per request: p (kept on the box by the privacy line), s (student), t (teacher)
  by_kind: Record<string, { target: "student" | "teacher"; decided_by: string; reason: string; n: number; joules: number; latency_s: number; energy_j: number }>;
  to_teacher: number;
  to_student: number;
  locked: number;
  energy_j: number;
  mean_latency_s: number;
  teacher_batch: number | null;
  teacher_rpm: number;
  teacher_j_per_token: number | null;
  saturated: boolean;
  cold_start_differs: boolean;
  cold_start_note: string | null;
}

export interface SimParams {
  preset: string | null;
  mix: Record<string, number>;
  rpm: number;
  minutes: number;
  tokens: Record<string, number>;
  policy: PolicyName;
  student_basis: "request" | "decode";
  seed: number;
}

export interface LabSim {
  available: boolean;
  reason: string | null;
  label: "simulation" | "what-if";
  model_statement: string;
  calls: string[];
  assumptions: string[];
  measured: { rows: { metric: string; node: string | null; unit: string | null; capability: BenchPoint; energy: BenchPoint; change_pct: number | null; low_confidence: boolean }[]; min_requests: number } | null;
  measured_reason: string | null;
  presets: { id: string; title: string; rpm: number; mix: Record<string, number> }[];
  params: SimParams;
  privacy: { rule: string; locked: { kind: string; reason: string }[]; lab_sources: string[]; source: string };
  kinds?: SimKind[];
  inputs?: {
    teacher: { node: string; model: string | null; session: string | null; chip: string | null; confidence_note: string; batches: number[]; j_per_token: BenchPoint[]; tok_s: BenchPoint[]; ttft_s: BenchPoint[] };
    student: BenchStudent;
    student_basis: "request" | "decode";
    student_j_per_token: BenchPoint;
    student_bases: { request: BenchPoint | null; decode: BenchPoint | null };
  };
  requests?: { n: number; window_s: number; t: number[]; kind: number[] };
  policies?: Record<PolicyName, SimRun>;
  compare?: { moved: number; saved_j: number; saved_pct: number; slower_s: number; sentence: string };
  break_even?: {
    student_j_per_token: number;
    curve: { batch: number; teacher_j_per_token: number; recorded: boolean }[];
    batch: number | null;
    rpm_to_teacher: number | null;
    tokens: number;
    reason: string | null;
    now: Record<PolicyName, number | null>;
  };
}

/* ---------- experiments ---------- */

export type CardStatus = "queued" | "running" | "won" | "lost" | "inconclusive" | "dropped" | "measured" | "converged" | "open";
export const CARD_STATUS: Record<CardStatus, string> = {
  queued: "Queued",
  running: "Running",
  won: "Won",
  lost: "Lost",
  inconclusive: "Inconclusive",
  dropped: "Dropped",
  measured: "Measured",
  converged: "Converged",
  open: "Not converged",
};

export interface LabCard {
  id: string;
  kind: "experiment" | "pass" | "merge" | "bench";
  title: string;
  status: CardStatus;
  ts: string | null;
  day_only: boolean;
  track: string | null;
  node?: string | null;
  lever?: string | null;
  arms?: Record<string, string>;
  metric?: string;
  better?: "higher" | "lower";
  hypothesis?: string;
  rank?: number | null;
  verdict: string | null;
  detail: string | null;
  numbers: LabNum[];
  source: { kind: string; ref: string };
  citation?: string | null;
}

export interface LabTimeline {
  cards: LabCard[];
  counts: Partial<Record<CardStatus, number>>;
  experiments: { total: number; queued: number; running: number; run: number; dropped: number; source: { kind: string; ref: string | null; note: string | null } };
  passes: number;
  sessions: number;
  summary: string;
}

/* ---------- words and numbers ---------- */

/** A value with as many digits as a reader needs: 36.7, 2.70, 0.89, 1,234. */
export function fmt(v: number, digits?: number): string {
  if (digits !== undefined) return v.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
  const a = Math.abs(v);
  const d = a >= 100 ? 0 : a >= 10 ? 1 : 2;
  return v.toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
}

export function joules(j: number): string {
  return j >= 10_000 ? `${fmt(j / 1000, 1)} kJ` : `${fmt(j, 0)} J`;
}

/** The metrics-file line of a source ref, as a short tag: "lab/refine/metrics.jsonl:67" -> "metrics.jsonl:67". */
export function shortRef(ref: string | undefined | null): string {
  return (ref ?? "").replace("lab/refine/", "");
}

export function when(iso: string | null | undefined, dayOnly = false): string {
  if (!iso) return "";
  const d = new Date(dayOnly ? `${iso}T12:00:00` : iso);
  if (Number.isNaN(d.getTime())) return iso;
  return dayOnly ? d.toLocaleDateString([], { month: "short", day: "numeric" }) : d.toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

/** The plain-language readout under the batching chart, from the sweep's own numbers at one batch size. */
export function batchReadout(sweep: BenchSweep, batch: number): string | null {
  const at = (name: string, b: number) => sweep.metrics[name]?.find((p) => p.x === b)?.value;
  const first = sweep.batches[0];
  if (first === undefined) return null;
  const tok = at("decode_tok_s", batch);
  const j = at("joules_per_token", batch);
  const j1 = at("joules_per_token", first);
  const tok1 = at("decode_tok_s", first);
  if (tok === undefined || j === undefined) return null;
  if (batch === first || j1 === undefined)
    return `At batch ${batch} the teacher answers ${batch === 1 ? "one person at a time" : `${batch} people at once`}: ${fmt(tok)} tokens a second, and each token costs ${fmt(j)} J.`;
  const pace = tok1 !== undefined && tok < tok1 ? (tok > tok1 * 0.75 ? "each a bit slower" : tok > tok1 * 0.5 ? "each slower" : "each much slower") : "each";
  return `At batch ${batch} the teacher answers ${batch} people at once, ${pace} (${fmt(tok)} tok/s instead of ${tok1 !== undefined ? fmt(tok1) : "?"}), and each token costs ${fmt(j)} J instead of ${fmt(j1)}: ${fmt(j1 / j, 1)} times less energy.`;
}
