/**
 * The Learning lab screen's data (GET /api/learning/lab, /api/learning/lab/{run_id}) and the small pure
 * functions its player uses. The gateway's read model is contexts/learning/application/lab_view.py.
 *
 * Every number arrives as a LabNum: its value, where it came from, and, when the log does not hold it,
 * `value: null` with the reason. The screen never fills a gap with a guess.
 */

export interface LabSource {
  kind: "event" | "file" | "derived";
  ref: string;
  ids: string[];
  events?: number;
}

export interface LabNum {
  label: string;
  value: number | null;
  unit: string | null;
  source: LabSource | null;
  low_confidence: boolean;
  reason: string | null;
}

export type LabStatus = "training" | "gated" | "deployed" | "not deployed" | "rolled back" | "failed";

/** Whether the run is still moving, as the gateway says it (lab_view._run_state, PR 421). Absent on older gateways. */
export type RunStatus = "running" | "finished" | "failed";

export interface LabRunSummary {
  run_id: string;
  task: string | null;
  status: LabStatus;
  started: string | null;
  ended: string | null;
  model_version: string | null;
  verdict: string | null;
  scores: { current: LabNum; candidate: LabNum; teacher: LabNum };
  run_status?: RunStatus | null;
  /** What the test said, in counts when the box holds them. null until the run is tested; absent on an older box. */
  tested?: { decision: "deploy" | "reject"; n_items: number | null; right_before: number | null; right_after: number | null; rehearsal: string | null; source: LabSource | null } | null;
  deployed?: string | null;
  rolled_back?: { ts: string; to: string | null; reason?: string | null } | null;
}

/** The newest install the box's log holds (whichever run it belongs to), or `reason` when it holds none. */
export interface LabBoxNow {
  model_version: string | null;
  previous_version: string | null;
  since: string | null;
  source: LabSource | null;
  reason: string | null;
  /** In a run's story: whether that install is this run's model. */
  this_run?: boolean | null;
}

/**
 * The chip a lab row was measured on, worked out by the gateway from the row itself (lab_bench.chip_of): the lab's
 * Strix Halo pods ("current"), the R9700s it had before ("earlier"), any other named chip ("reference", such as the
 * A100 on Colab) or a row that only says it is another chip ("unnamed"). `reason` is how that was decided.
 */
export interface LabChip {
  key: string;
  name: string;
  where: string | null;
  current: boolean;
  role: "current" | "earlier" | "reference" | "unnamed";
  reason?: string;
}

/** "NVIDIA A100 (Colab)": a reference chip with where it ran; the lab's own cards by name alone. */
export function chipShort(chip: Pick<LabChip, "name" | "where" | "role">): string {
  return chip.role === "reference" && chip.where ? `${chip.name} (${chip.where})` : chip.name;
}

/** The heading for a chip that is not the current lab: "Reference: NVIDIA A100 (Colab)", "Earlier lab cards: Radeon R9700". */
export function chipLabel(chip: Pick<LabChip, "name" | "where" | "role">): string {
  if (chip.role === "earlier") return `Earlier lab cards: ${chip.name}`;
  if (chip.role === "reference") return `Reference: ${chipShort(chip)}`;
  if (chip.role === "unnamed") return `Another chip, not named${chip.where ? ` (${chip.where})` : ""}`;
  return chip.name;
}

/** What a chip is to the lab, in a few words. */
export function chipRole(chip: Pick<LabChip, "role">): string {
  if (chip.role === "earlier") return "the lab's earlier cards, or untagged rows from before the lab moved";
  if (chip.role === "reference") return "a reference, not the lab's hardware";
  if (chip.role === "unnamed") return "rows that mark themselves as another chip without naming it";
  return "the lab now";
}

export interface LabMeasured extends LabNum {
  metric: string;
  node: string | null;
  measured_at: string | null;
  other_chip?: boolean; // not measured on the current lab's chip
  chip?: LabChip; // which chip it was
}

export interface LabList {
  runs: LabRunSummary[];
  sample: boolean;
  sample_reason: string | null;
  real_runs: number;
  current_model: { version: string; source: string } | null;
  box_now?: LabBoxNow;
  measured: LabMeasured[];
}

export interface LabRow {
  id: string;
  frame_ref: string;
  context: string | null;
  target: string | null;
  verdict: "up" | "down" | null;
  side: "train" | "held_out" | null;
}

export type LabChange = "fixed" | "broke" | "still_right" | "still_wrong";

export interface LabItem {
  id: string;
  group: string | null;
  question: string;
  expected: string | null;
  before: { answer: string; correct: boolean };
  after: { answer: string; correct: boolean };
  change: LabChange;
}

export interface LabBoxSide {
  version: string | null;
  j_per_answer: LabNum;
  tok_s: LabNum;
  answers: number;
}

export interface LabStory {
  run_id: string;
  sample: boolean;
  sample_reason: string | null;
  status: LabStatus;
  run_status?: RunStatus | null;
  phase: string | null;
  task: { slug: string | null; instruction: string | null; instruction_reason: string | null };
  marks: Record<"start" | "run_start" | "taught" | "exported" | "training_start" | "training_end" | "evaluated" | "packaged" | "deployed" | "end", string | null>;
  wall_s: LabNum;
  /** The run's learning.run.progress events in time order (labeling, training, eval, packaging, failed, ...). */
  stages?: { stage: string; progress: number | null; ts: string; message: string | null; event_id: string }[];
  dataset: {
    dataset_id: string | null;
    examples: LabNum;
    train: LabNum;
    held_out: LabNum;
    held_out_groups: LabNum;
    rows: LabRow[] | null;
    rows_reason: string | null;
    approval: { asked: string | null; approved: string | null; reason: string | null; approved_reason?: string | null; approved_by_reason: string | null };
  };
  training: {
    node: string | null;
    base: string | null;
    lora_rank: number | null;
    epochs: number | null;
    batch: number | null;
    steps: LabNum;
    of: LabNum;
    wall_s: LabNum;
    samples: LabNum;
    samples_per_s: LabNum;
    loss_first: LabNum;
    loss_last: LabNum;
    points: { t: number; step: number | null; loss: number | null }[];
    points_reason: string | null;
  };
  power: {
    node: string | null;
    points: { t: number; w: number; dt?: number }[];
    points_reason: string | null;
    energy_j: LabNum;
    mean_w: LabNum;
    peak_w: LabNum;
    j_per_sample: LabNum;
  };
  eval: {
    scores: { current: LabNum; candidate: LabNum; teacher: LabNum };
    decision: "deploy" | "reject" | null;
    model_version: string | null;
    n_items: LabNum;
    n_groups: LabNum;
    noise: LabNum;
    p_value: LabNum;
    alpha: number | null;
    test: string | null;
    decode: string | null;
    noise_runs?: number[];
    counts: Record<"n" | "before_right" | "after_right" | "fixed" | "broke" | "still_right" | "still_wrong", LabNum> | null;
    items: LabItem[] | null;
    items_reason: string | null;
    groups: { group: string; n: number; before: number; after: number; diff: number }[] | null;
    verdict: string | null;
    verdict_detail: string[];
    /** Why the per-question answers cannot be the model's own (a rehearsal of the loop), or null. Absent on an older box. */
    rehearsal?: string | null;
    correct?: Record<string, LabNum>;
    source?: LabSource | null;
  };
  package: { model_version: string | null; gguf: string | null; base: string | null; quantization: string | null; sha256: string | null; reason: string | null; bytes: LabNum };
  deploy: {
    model_version: string | null;
    previous_version: string | null;
    deployed: string | null;
    deployed_reason: string | null;
    swap_s: LabNum;
    rolled_back: { ts: string; to: string | null; reason?: string | null } | null;
  };
  box: { before: LabBoxSide | null; after: LabBoxSide | null; reason: string | null };
  box_now?: LabBoxNow;
  /** The live loop's stages as the gateway lit them (lab_view._strip): labeling, training, packaging, offered. */
  stage_strip?: { stage: string; state: "done" | "now" | "waiting" | "failed" | "not reported"; started: string | null; ended: string | null; seconds: number | null; message: string | null }[];
  steps_so_far?: number;
  gaps: { label: string; reason: string }[];
  sources: LabNum[];
  measured: LabMeasured[];
}

/* ---------- the player: one position 0..1 drives every section ---------- */

export type ChapterKey = "taught" | "training" | "answers" | "gate" | "box";

export interface Chapter {
  key: ChapterKey;
  n: number;
  title: string;
  from: number;
  to: number;
}

/** Shares of the bar, not of the wall clock: training would otherwise be a sliver beside the waiting. */
export const CHAPTERS: Chapter[] = [
  { key: "taught", n: 1, title: "Taught from", from: 0, to: 0.1 },
  { key: "training", n: 2, title: "Training", from: 0.1, to: 0.55 },
  { key: "answers", n: 3, title: "Before / after", from: 0.55, to: 0.8 },
  { key: "gate", n: 4, title: "The test", from: 0.8, to: 0.88 },
  { key: "box", n: 5, title: "On the box", from: 0.88, to: 1 },
];

export const PLAY_SECONDS = 26;

export function clamp01(x: number): number {
  return x < 0 ? 0 : x > 1 ? 1 : x;
}

/** How far through a chapter the playhead is, 0 before it and 1 after it. */
export function within(u: number, key: ChapterKey): number {
  const c = CHAPTERS.find((x) => x.key === key)!;
  return clamp01((u - c.from) / (c.to - c.from));
}

export function chapterAt(u: number): Chapter {
  return CHAPTERS.find((c) => u < c.to) ?? CHAPTERS[CHAPTERS.length - 1]!;
}

/* ---------- words and numbers ---------- */

export function isNum(n: LabNum | null | undefined): n is LabNum & { value: number } {
  return !!n && typeof n.value === "number";
}

export function pct(fraction: number, digits = 0): string {
  return `${(fraction * 100).toFixed(digits)}%`;
}

export function duration(seconds: number): string {
  const s = Math.round(seconds);
  if (s < 90) return `${s} s`;
  const m = Math.floor(s / 60);
  if (m < 90) return `${m} min ${s % 60} s`;
  return `${Math.floor(m / 60)} h ${m % 60} min`;
}

export function fileSize(bytes: number): string {
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(2)} GB` : `${(bytes / 1e6).toFixed(0)} MB`;
}

export function clock(iso: string | null | undefined, seconds = false): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleTimeString([], seconds ? { hour: "numeric", minute: "2-digit", second: "2-digit" } : { hour: "numeric", minute: "2-digit" });
}

/** A number as the screen prints it, with its unit. */
export function show(n: LabNum): string {
  if (n.value === null) return "not in the log";
  const v = n.value;
  switch (n.unit) {
    case "fraction right":
    case "fraction":
      return pct(v, v < 0.1 ? 1 : 0);
    case "s":
      return duration(v);
    case "bytes":
      return fileSize(v);
    case "J":
      return v >= 3600 ? `${(v / 3600).toFixed(1)} Wh (${Math.round(v).toLocaleString()} J)` : v >= 100 ? `${Math.round(v).toLocaleString()} J` : `${v.toFixed(1)} J`;
    case "samples/s":
      return `${Number(v.toPrecision(2))} examples a second`;
    case null:
      return String(Number(v.toPrecision(3)));
    default:
      return `${Number.isInteger(v) ? v.toLocaleString() : Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : Number(v.toPrecision(3))} ${n.unit}`;
  }
}

/**
 * One everyday comparison for the training energy: full charges of a phone.
 * Assumption, stated on the screen: one charge is 15 Wh (a 4,000 mAh battery at 3.85 V holds 15.4 Wh;
 * charger losses are left out, so a real charge draws a little more from the wall).
 */
export const PHONE_CHARGE_WH = 15;
export const PHONE_ASSUMPTION = "Assumes one full phone charge is 15 Wh: a 4,000 mAh battery at 3.85 V holds 15.4 Wh. Charger losses are left out.";

export function phoneCharges(wh: number): string {
  const n = wh / PHONE_CHARGE_WH;
  if (n < 0.05) return "less than a twentieth of a phone charge";
  if (n < 0.95) return `${Math.round(n * 100)}% of one phone charge`;
  return `${n.toFixed(1)} phone charges`;
}

/** Energy in joules up to `t` seconds into training: each reading holds for its step (or until the next sample). */
export function energyUpTo(points: { t: number; w: number; dt?: number }[], t: number): number {
  let j = 0;
  for (let i = 0; i < points.length; i++) {
    const p = points[i]!;
    if (p.dt !== undefined) {
      const start = p.t - p.dt;
      if (t <= start) break;
      j += p.w * Math.min(p.dt, t - start);
    } else if (i > 0) {
      const a = points[i - 1]!;
      if (t <= a.t) break;
      const end = Math.min(p.t, t);
      const wEnd = a.w + ((p.w - a.w) * (end - a.t)) / (p.t - a.t || 1);
      j += ((a.w + wEnd) / 2) * (end - a.t);
    }
  }
  return j;
}

export const CHANGE_LABEL: Record<LabChange, string> = {
  fixed: "Fixed",
  broke: "Got worse",
  still_right: "Still right",
  still_wrong: "Still wrong",
};

/* ---------- a run that is still going (L30) ---------- */

/** A run that has stopped training, whatever else is still to report: never "training now". */
const NOT_LIVE = new Set(["gated", "finished", "failed", "offered", "deployed", "not deployed", "rolled back"]);

/**
 * Training now, and nothing else: the gateway's `run_status` is "running" (PR 421) and the run's own status is not
 * one that has stopped (gated, finished, failed, offered, deployed, ...). A gateway older than `run_status` counts a
 * run as live only when its status is exactly "running" and it has no ended time. Anything else is resting.
 */
export function isLiveRun(r: { sample?: boolean; status: string; run_status?: string | null; ended?: string | null }): boolean {
  if (r.sample) return false;
  if (NOT_LIVE.has(r.status)) return false;
  if (r.run_status !== undefined && r.run_status !== null) return r.run_status === "running";
  return r.status === "running" && !r.ended;
}

/** isLiveRun for a run's story: its ended time is the story's last mark. */
export function isLive(story: Pick<LabStory, "sample" | "status" | "run_status" | "marks">): boolean {
  return isLiveRun({ sample: story.sample, status: story.status, run_status: story.run_status, ended: story.marks.end });
}

/**
 * A measured trainer row from the lab's own hardware: the Strix Halo pods (gfx1151) or the box (Jetson), by the chip the
 * gateway worked out from the row (`chip.key`, lab_bench.chip_of). A row from another chip (the R9700s the lab had
 * before, a reference machine) or with no chip at all is not "measured on our hardware".
 */
export const OUR_CHIPS = new Set(["strix-halo", "jetson"]);
export function onOurHardware(m: Pick<LabMeasured, "chip" | "other_chip">): boolean {
  return !!m.chip && OUR_CHIPS.has(m.chip.key) && m.other_chip !== true;
}

export type StageKey = "taught" | "training" | "packaging" | "box";
export type StageState = "done" | "now" | "waiting" | "failed";

export const LIVE_STAGES: { key: StageKey; title: string }[] = [
  { key: "taught", title: "Taught from" },
  { key: "training", title: "Training" },
  { key: "packaging", title: "Packaging" },
  { key: "box", title: "On the box" },
];

/**
 * Where each stage of a live run stands, from what the run has reported so far: its progress events, its marks and
 * its training rows. A later stage reporting means the ones before it are done.
 */
export function stageStates(story: LabStory): Record<StageKey, StageState> {
  const stages = story.stages ?? [];
  const m = story.marks;
  const has = (name: string, full = false) => stages.some((s) => s.stage === name && (!full || (s.progress ?? 0) >= 1));
  const failed = story.status === "failed" || has("failed");
  const box = !!m.deployed;
  const packaged = box || !!m.packaged || has("packaging", true);
  const packaging = packaged || has("packaging");
  const trained = packaging || !!m.evaluated || has("eval") || has("training", true);
  const training = trained || !!m.training_start || story.training.points.length > 0 || has("training");
  const taught = training || !!m.taught || !!m.exported || has("labeling") || isNum(story.dataset.examples);
  const state = (reported: boolean, finished: boolean): StageState => (finished ? "done" : reported ? "now" : "waiting");
  return {
    taught: state(taught, training || has("labeling", true)),
    training: failed && !trained ? "failed" : state(training, trained),
    packaging: state(packaging, packaged),
    box: state(box, box),
  };
}

/** Seconds from the run's first event to `now`. */
export function elapsedSeconds(story: Pick<LabStory, "marks">, now: number): number | null {
  const from = story.marks.run_start ?? story.marks.start;
  if (!from) return null;
  const t = Date.parse(from);
  return Number.isNaN(t) ? null : Math.max(0, (now - t) / 1000);
}

export const STATUS_LABEL: Record<LabStatus, string> = {
  training: "Training",
  gated: "Cleared by the gate",
  deployed: "Deployed",
  "not deployed": "Not deployed",
  "rolled back": "Rolled back",
  failed: "Failed",
};
