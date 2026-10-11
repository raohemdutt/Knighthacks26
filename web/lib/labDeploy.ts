/**
 * The On the box card's deploy button (L35): types for GET/POST /api/learning/lab/deploy, /deploy/rollback and
 * /deploy/ask (contexts/learning/lab_deploy.py). The swap itself runs on the box (jetson/atlas_swap.py).
 */

export type DeployState = "idle" | "running" | "done" | "failed";
export type DeployStageState = "waiting" | "running" | "done" | "failed" | "skipped";
export type DeployStageName = "verify" | "switch" | "restart" | "check" | "publish";

export interface DeployVersion {
  version: string;
  base: string;
  task: string | null;
  created: string;
  current: boolean;
  previous: boolean;
  /** manifest.json reads, every file is there, and no stale .verified.json says otherwise */
  verified: boolean;
  record: "fresh" | "stale" | "none";
}

export interface DeployStage {
  name: DeployStageName;
  state: DeployStageState;
  seconds: number | null;
}

export interface DeployResult {
  ok: boolean;
  returncode: number | null;
  message: string | null;
  rolled_back_to: string | null;
}

export interface DeployRun {
  state: DeployState;
  action: "deploy" | "rollback" | null;
  version: string | null;
  from: string | null;
  started: string | null;
  ended: string | null;
  seconds: number | null;
  stages: DeployStage[];
  log_tail: string[];
  result: DeployResult | null;
}

export interface DeployView {
  versions: DeployVersion[];
  current: string | null;
  previous: string | null;
  base: string | null;
  /** the newest verified version that is not current (and not the base) */
  suggested: string | null;
  /** what "Put v2 on the box" deploys: the demo's package when it is on disk and verified, else the newest; may be current */
  featured: string | null;
  deploy: DeployRun;
}

export interface DeployAnswer {
  question: string;
  serving: string | null;
  text: string;
  calls: { name: string; arguments: Record<string, unknown> }[];
  expected: { name: string; arguments: Record<string, unknown> }[] | null;
  seconds: number;
  prompt: "eval" | "eval template";
  /** What the answer cost on the box (G6): null while the Jetson's power sampler is off; absent on an older gateway. */
  energy?: AnswerEnergy | null;
  /** The comparison (ask-compare): the box's answer and the 122B pair's; absent on an older gateway. */
  box?: AskSide;
  brain?: AskSide | null;
}

export type ToolCallShape = { name: string; arguments: Record<string, unknown> };

/** Both pods' power over the brain's request window, gross, each pod's idle beside it. */
export interface BrainEnergy {
  joules: number;
  watts_avg: number;
  seconds?: number;
  pods: Record<string, { joules: number; watts_avg: number; idle_w: number | null; samples: number } | null>;
}

/** One side of the comparison: an answer, or why there is none. */
export interface AskSide {
  answer?: string;
  calls?: ToolCallShape[];
  serving?: string | null;
  seconds?: number;
  energy?: AnswerEnergy | BrainEnergy | null;
  error?: string;
}

/** The box's side from an answer, also from a gateway older than the comparison (its top-level fields). */
export function boxSide(a: DeployAnswer): AskSide {
  return a.box ?? { answer: a.text, calls: a.calls, serving: a.serving, seconds: a.seconds, energy: a.energy ?? null };
}

/** "about 41 J in 2.4 s"; without a reading, what is missing and the seconds. */
export function costShort(side: AskSide, missing: string): string {
  const secs = side.seconds !== undefined ? `${side.seconds.toFixed(1)} s` : "";
  const e = side.energy;
  if (!e) return secs ? `${missing} · ${secs}` : missing;
  const j = e.joules >= 10 ? Math.round(e.joules).toLocaleString("en-US") : e.joules.toFixed(1);
  return `about ${j} J${secs ? ` in ${secs}` : ""}`;
}

/** The tool a side used, in one line: "last_seen(label=\"keys\")", or "no tool". */
export function toolLine(calls: ToolCallShape[] | undefined): string {
  return calls && calls.length ? calls.map(callText).join(", ") : "no tool";
}

/** What a side's tools did, in plain words: "checked last seen", "looked back over the last 25 minutes". */
export function toolWords(calls: ToolCallShape[] | undefined): string {
  if (!calls || !calls.length) return "answered without a tool";
  const c = calls[0]!;
  if (c.name === "last_seen") return "checked last seen";
  if (c.name === "search_memory") return "searched memory";
  if (c.name === "episodes_between") {
    const a = Date.parse(String(c.arguments.start ?? ""));
    const b = Date.parse(String(c.arguments.end ?? ""));
    if (Number.isFinite(a) && Number.isFinite(b) && b > a) {
      const min = Math.round((b - a) / 60000);
      return min >= 120 && min % 60 === 0 ? `looked back over the last ${min / 60} hours` : `looked back over the last ${min} minutes`;
    }
    return "looked back over a stretch of time";
  }
  return `used ${c.name}`;
}

/** One line under both answers: "both checked last seen", or "the box searched memory, the brain checked last seen". */
export function verdictLine(box: AskSide, brain: AskSide | null | undefined): string | null {
  if (!brain || box.error || brain.error) return null;
  const a = toolWords(box.calls);
  const b = toolWords(brain.calls);
  if (a === b) return a === "answered without a tool" ? "both answered without a tool" : `both ${a}`;
  return `the box ${a}, the brain ${b}`;
}

/** contexts/learning/answer_energy.py: the box's power over the answer's window, gross, with its idle beside it. */
export interface AnswerEnergy {
  joules: number;
  idle_w: number | null;
  seconds: number;
  sampler: string;
  samples?: number;
  above_idle_j?: number | null;
  low_confidence?: boolean;
}

/** "this answer cost about 41 J on the box"; "sampler off" when it was not measured; null from an older gateway. */
export function energyLine(energy: AnswerEnergy | null | undefined): string | null {
  if (energy === undefined) return null;
  if (energy === null) return "sampler off";
  const j = energy.joules >= 10 ? Math.round(energy.joules).toString() : energy.joules.toFixed(1);
  return `this answer cost about ${j} J on the box`;
}

export const DEPLOY_STAGES: { key: DeployStageName; title: string }[] = [
  { key: "verify", title: "verify" },
  { key: "switch", title: "switch" },
  { key: "restart", title: "restart" },
  { key: "check", title: "check" },
  { key: "publish", title: "published" },
];

/** "v2" for atlas-tool-calling-v2-a100-f2r4, else the version itself. */
export function shortVersion(version: string): string {
  const m = /(?:^|-)(v\d+)(?:-|$)/.exec(version);
  return m ? m[1]! : version;
}

/** last_seen(label="water bottle") */
export function callText(call: { name: string; arguments: Record<string, unknown> }): string {
  const args = Object.entries(call.arguments ?? {})
    .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
    .join(", ");
  return `${call.name}(${args})`;
}
