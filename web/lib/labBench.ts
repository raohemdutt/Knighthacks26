/**
 * The Learning lab's other tabs: speed and energy (GET /api/learning/lab/bench), the routing what-if
 * (POST /api/learning/lab/routing/simulate) and the experiment timeline (GET /api/learning/lab/experiments).
 * The gateway's read models are contexts/learning/application/lab_bench.py, lab_routing.py and lab_experiments.py.
 *
 * A number is a LabNum with a `kind`: "measured" is a row's own value, "derived" is worked out from rows with the
 * formula shown. The screen adds two more labels of its own: "modelled" (the what-if) and "sample" (the sample run).
 */
import type { LabChip, LabNum, LabSource } from "./lab";

export type LabTab = "run" | "compare" | "speed" | "routing" | "experiments";
export const LAB_TABS: { key: LabTab; title: string }[] = [
  { key: "run", title: "Training run" },
  { key: "compare", title: "Alone vs with a friend" },
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
  other_chip?: boolean; // not measured on the current lab's chip (kept for older screens; `chip` says which chip)
  chip?: LabChip | null;
}

/** Rows of one chip: how many, and their lines in the metrics file. */
export interface BenchChipRows extends LabChip {
  rows: number;
  lines: number[];
}

/** Another chip's sweep that recorded the same metrics at the same batch sizes as this one. */
export interface BenchComparison {
  id: string;
  chip: LabChip;
  metrics: string[];
  batches: number[];
  same_model: boolean;
}

export interface BenchSweep {
  id: string;
  other_chip: boolean;
  chip: LabChip;
  chip_note: string | null; // the rows' own words about a chip that is not the lab's
  power_source: string | null;
  gpu: string | null; // the chip as the rows spell it
  comparisons: BenchComparison[];
  family: string;
  node: string | null;
  session: string | null;
  model: string | null;
  build: string | null;
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
  power: { nodes: BenchPowerNode[]; absent: string[]; other_chip: BenchPoint[]; other_chips: BenchChipRows[] };
  other_chip_rows: number;
  other_chip_label: string;
  chips: BenchChipRows[];
  series: { metric: string; family: string; node: string | null; points: BenchPoint[] }[];
  charted_rows: number;
  not_charted: { metric: string; node: string | null; rows: number; of: number; other_chip: number; other_chips: BenchChipRows[]; unit: string | null; latest: BenchPoint; lines: number[] }[];
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
  experiments: {
    total: number;
    queued: number;
    running: number;
    run: number;
    dropped: number;
    /** `from`: which ledger was read (LAB_STATE_DIR, the copy the lab pushed, the checkout); `received_at` and `rows` describe a pushed copy. */
    source: { kind: string; ref: string | null; note: string | null; from?: "LAB_STATE_DIR" | "imported" | "checkout"; received_at?: string | null; rows?: number };
  };
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

/* ---------- the speed and energy story (GET /api/learning/lab/bench, summary shape) ---------- */

/**
 * One measurement session, as the summary route returns it. `chip` and `engine` are the rows' own words (such as
 * "strix-halo" and "vllm"); the screen matches them loosely, so "Radeon 8060S (Strix Halo)" and "llamacpp@23b0202"
 * read the same. `node` and `build` are only present when the session was derived from the full payload.
 */
export interface BenchSession {
  id: string;
  chip: string;
  engine: string;
  model: string | null;
  date: string | null;
  lever: string | null;
  rows: number;
  node?: string | null;
  build?: string | null;
  families?: string[];
  source?: string; // the session id the route knows, when one route session was split by node here
  rows_url?: string | null;
}

/** One headline point of a session at one batch size (streams answered at once). */
export interface BenchHeadline {
  session: string;
  batch: number;
  tok_s_stream: number | null;
  tok_s_total: number | null;
  ttft_s: number | null;
  watts: number | null;
  j_per_token: number | null;
  confidence: string | number | boolean | null;
}

export interface BenchSummary {
  sessions: BenchSession[];
  points: BenchHeadline[];
}

/** One number of a session's headline as the route sends it: the row's value with its line and confidence. */
export interface BenchCell {
  value: number;
  unit?: string | null;
  line?: number | null;
  low_confidence?: boolean;
  reason?: string | null;
  runs?: number | null;
  statistic?: string | null;
}

/** A session as GET /api/learning/lab/bench returns it (summary: true): chip, engine, date and headline per node and batch. */
export interface BenchApiSession {
  id: string;
  chip: { key?: string; name?: string; current?: boolean; role?: string; where?: string | null } | string | null;
  engine: string | null;
  model: string | null;
  date: string | null;
  ts?: string | null;
  first_ts?: string | null;
  lever: string | null;
  nodes?: string[];
  families?: string[];
  rows: number;
  batches?: number[];
  rows_url?: string | null;
  headline?: {
    node: string | null;
    batch: number | null;
    chip?: string | null;
    tok_s_stream: BenchCell | number | null;
    tok_s_total: BenchCell | number | null;
    ttft_s: BenchCell | number | null;
    watts: BenchCell | number | null;
    j_per_token: BenchCell | number | null;
    confidence: { level?: string; low_confidence?: number; of?: number } | string | null;
  }[];
}

/** What GET /api/learning/lab/bench returns: the summary (sessions with their headline), or today's full payload. */
export type BenchPayload = Omit<Partial<LabBench>, never> & { summary?: boolean; sessions?: (BenchApiSession | BenchSession)[]; points?: BenchHeadline[] };

const cellValue = (c: BenchCell | number | null | undefined): number | null => (c === null || c === undefined ? null : typeof c === "number" ? c : typeof c.value === "number" ? c.value : null);

/** The summary route's sessions as the screen's sessions and points: one screen session per route session and node. */
export function summaryFromSessions(list: BenchApiSession[]): BenchSummary {
  const sessions: BenchSession[] = [];
  const points: BenchHeadline[] = [];
  for (const x of list) {
    const chip = typeof x.chip === "string" ? x.chip : (x.chip?.key ?? x.chip?.name ?? "");
    const byNode = new Map<string, NonNullable<BenchApiSession["headline"]>>();
    for (const h of x.headline ?? []) {
      const key = h.node ?? "";
      byNode.set(key, [...(byNode.get(key) ?? []), h]);
    }
    if (byNode.size === 0) byNode.set(x.nodes?.join("+") ?? "", []);
    for (const [node, cells] of byNode) {
      const id = byNode.size > 1 ? `${x.id} @ ${node}` : x.id;
      sessions.push({ id, chip, engine: x.engine ?? "", model: x.model, date: x.ts ?? x.date, lever: x.lever, rows: x.rows, node: node || null, build: x.engine, families: x.families, source: x.id, rows_url: x.rows_url ?? null });
      for (const h of cells) {
        if (h.batch === null || h.batch === undefined) continue;
        const conf = h.confidence;
        points.push({
          session: id,
          batch: h.batch,
          tok_s_stream: cellValue(h.tok_s_stream),
          tok_s_total: cellValue(h.tok_s_total),
          ttft_s: cellValue(h.ttft_s),
          watts: cellValue(h.watts),
          j_per_token: cellValue(h.j_per_token),
          confidence: conf === null || typeof conf === "string" ? conf : conf.level ?? ((conf.low_confidence ?? 0) > 0 ? "low" : null),
        });
      }
    }
  }
  return { sessions, points };
}

/** True when a point's confidence says it is low: the word "low", false, or a number under one half. */
export function isLowConfidence(c: BenchHeadline["confidence"]): boolean {
  if (c === null || c === undefined) return false;
  if (typeof c === "boolean") return !c;
  if (typeof c === "number") return c < 0.5;
  return /low/i.test(c);
}

export type Engine = "vllm" | "llamacpp" | "other";
export function engineOf(s: string | null | undefined): Engine {
  if (!s) return "other";
  if (/llama/i.test(s)) return "llamacpp";
  if (/vllm/i.test(s)) return "vllm";
  return "other";
}
export const isStrixHalo = (chip: string | null | undefined) => /strix|8060s|gfx1151/i.test(chip ?? "");
export const isA100 = (chip: string | null | undefined) => /a100/i.test(chip ?? "");
/** A session of the 122B pair across two pods: its rows are the pair_pp122b family, or its model is the 122B. */
export const isPair = (s: BenchSession) => /pair_pp122b/i.test(`${s.id} ${s.lever ?? ""} ${(s.families ?? []).join(" ")}`) || /122b/i.test(s.model ?? "");

/** The summary shape, worked out from today's full payload: one session per sweep, one point per batch size. */
export function summaryFromFull(full: Partial<LabBench>): BenchSummary {
  const sessions: BenchSession[] = [];
  const points: BenchHeadline[] = [];
  for (const sw of full.sweeps ?? []) {
    sessions.push({
      id: sw.id,
      chip: sw.chip?.key ?? sw.gpu ?? "",
      engine: engineOf(sw.build) !== "other" ? engineOf(sw.build) : engineOf(sw.family),
      model: sw.model,
      date: sw.ts,
      lever: sw.family,
      rows: sw.points,
      node: sw.node,
      build: sw.build,
      families: [sw.family],
      source: sw.session ?? sw.id,
    });
    const at = (name: string, b: number) => sw.metrics[name]?.find((p) => p.x === b) ?? null;
    for (const batch of sw.batches) {
      const picked = ["decode_tok_s", "decode_tok_s_total", "ttft_s", "watts_avg", "joules_per_token"].map((m) => at(m, batch));
      const [stream, total, ttft, watts, j] = picked;
      points.push({
        session: sw.id,
        batch,
        tok_s_stream: stream?.value ?? null,
        tok_s_total: total?.value ?? null,
        ttft_s: ttft?.value ?? null,
        watts: watts?.value ?? null,
        j_per_token: j?.value ?? null,
        confidence: picked.some((p) => p?.low_confidence) ? "low" : "ok",
      });
    }
  }
  return { sessions, points };
}

/** The summary from whatever the route returned: its sessions and headlines when present, else derived from the full payload. */
export function benchSummary(p: BenchPayload | null | undefined): BenchSummary {
  if (p && Array.isArray(p.sessions) && Array.isArray(p.points)) return { sessions: p.sessions as BenchSession[], points: p.points };
  if (p && Array.isArray(p.sessions) && p.sessions.length) return summaryFromSessions(p.sessions as BenchApiSession[]);
  return summaryFromFull(p ?? {});
}

export interface StoryLine {
  session: BenchSession;
  points: BenchHeadline[]; // by batch, ascending
}

export interface SpeedStory {
  after: StoryLine | null; // vLLM on a Strix Halo pod, newest session: after the kernel fix
  before: StoryLine | null; // the newest older vLLM Strix Halo session of the same model that was slower at one stream
  llama: StoryLine | null; // llama.cpp on a Strix Halo pod, newest session with two or more batch sizes
  a100: StoryLine | null; // the NVIDIA A100 reference, newest
  pair: StoryLine | null; // the 122B pair across two pods, newest
  base: StoryLine | null; // the 27B on one pod at the pair's batch sizes (from `after`)
}

const newest = (a: BenchSession, b: BenchSession) => (b.date ?? "").localeCompare(a.date ?? "");

/** Picks the sessions the story is told from, by engine, chip and date, never by a hard-coded number. */
export function speedStory(sum: BenchSummary): SpeedStory {
  const byS = new Map<string, BenchHeadline[]>();
  for (const p of sum.points) {
    const list = byS.get(p.session) ?? [];
    list.push(p);
    byS.set(p.session, list);
  }
  const line = (s: BenchSession | undefined): StoryLine | null => (s ? { session: s, points: (byS.get(s.id) ?? []).slice().sort((a, b) => a.batch - b.batch) } : null);
  const has = (s: BenchSession) => (byS.get(s.id)?.length ?? 0) > 0;
  const sorted = sum.sessions.filter(has).slice().sort(newest);
  const single = sorted.filter((s) => !isPair(s));
  const vllm = single.filter((s) => engineOf(s.engine) === "vllm" && isStrixHalo(s.chip));
  const after = line(vllm[0]);
  const b1 = (l: StoryLine | null) => l?.points.find((p) => p.batch === 1)?.tok_s_stream ?? null;
  const afterB1 = b1(after);
  const before = line(
    vllm.slice(1).find((s) => (after?.session.model ? s.model === after.session.model : true) && afterB1 !== null && (b1(line(s)) ?? Infinity) < afterB1),
  );
  const llamas = single.filter((s) => engineOf(s.engine) === "llamacpp" && isStrixHalo(s.chip));
  const llama = line(llamas.find((s) => (byS.get(s.id)?.length ?? 0) >= 2) ?? llamas[0]);
  const a100 = line(single.find((s) => isA100(s.chip)));
  const pair = line(sorted.find(isPair));
  const base = pair && after ? { session: after.session, points: after.points.filter((p) => pair.points.some((q) => q.batch === p.batch)) } : null;
  return { after, before, llama, a100, pair, base };
}

/** The model size in billions from its name: "Qwen3.8-27B-INT4" -> 27. */
export function modelBillions(model: string | null | undefined): number | null {
  const m = /(\d+(?:\.\d+)?)\s*b(?![a-z])/i.exec(model ?? "");
  return m ? Number(m[1]) : null;
}

/** "pod-a" -> "Pod A", "pod-a+pod-b" -> "Pods A and B together", "jetson" -> "The box (Jetson)". */
export function machineName(node: string | null | undefined): string {
  if (!node) return "unknown machine";
  if (node === "jetson") return "The box (Jetson)";
  const pair = /^pod-([a-z])\+pod-([a-z])$/i.exec(node);
  if (pair) return `Pods ${pair[1]!.toUpperCase()} and ${pair[2]!.toUpperCase()} together`;
  const pod = /^pod-([a-z])$/i.exec(node);
  return pod ? `Pod ${pod[1]!.toUpperCase()}` : node;
}
