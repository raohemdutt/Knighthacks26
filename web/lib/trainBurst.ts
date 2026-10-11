/**
 * The "Train for a minute" button on the live power graph (components/lab/TrainBurst.tsx): a real LoRA burst on the
 * pods from POST /api/learning/lab/train-burst, its state from GET (contexts/learning/train_burst.py). Pure rules,
 * free of imports, so scripts/check-lab.mjs can run them in node. Nothing here makes up a number: a missing step count
 * or joule figure is said as missing.
 */

export type BurstPod = "C" | "D";
export type BurstState = "idle" | "running" | "done" | "failed";

export interface BurstPodResult {
  ok: boolean;
  steps: number | null;
  seconds: number | null;
  train_s: number | null;
  loss: number | null;
  peak_gb: number | null;
  error: string | null;
}

export interface BurstEnergy {
  joules: number;
  watts_avg: number;
  seconds: number;
  pods: Record<string, { joules: number; watts_avg: number; idle_w: number | null; samples: number } | null>;
}

export interface BurstView {
  available: boolean;
  reason: string | null;
  state: BurstState;
  pods: BurstPod[];
  seconds_asked: number | null;
  started: string | null;
  ended: string | null;
  started_ms: number | null;
  ended_ms: number | null;
  per_pod: Record<string, BurstPodResult | null>;
  energy: BurstEnergy | null;
  message: string | null;
}

/** The picker: both pods first, the default. */
export const BURST_CHOICES: { id: string; label: string; pods: BurstPod[] }[] = [
  { id: "both", label: "both pods", pods: ["C", "D"] },
  { id: "D", label: "pod D", pods: ["D"] },
];

/** "pod D", "pod C and pod D". */
export function podsText(pods: string[]): string {
  return pods.map((p) => `pod ${p}`).join(" and ");
}

/** The button's words for a choice. */
export function buttonText(pods: string[]): string {
  return pods.length > 1 ? "Train for a minute on both pods" : `Train for a minute on ${podsText(pods)}`;
}

/** The pod line on the graph a burst pod shades: pod C is the head (pod-a), pod D the half (pod-b). */
export function burstNode(pod: string): string {
  return pod === "C" ? "pod-a" : "pod-b";
}

const num = (x: unknown): x is number => typeof x === "number" && Number.isFinite(x);

/**
 * The line under the graph once a burst ended: each pod's steps and seconds, then the joules for every pod's
 * samplers over the window. "training burst: pod C 1 step in 74 s, pod D 1 step in 72 s, about 15,116 joules on the
 * pods (171 W average)". A pod that failed says so; meters that did not cover the window say "meters offline".
 */
export function burstLine(v: BurstView): string {
  const parts = v.pods.map((p) => {
    const r = v.per_pod[p];
    if (!r || !r.ok) return `pod ${p} did not train${r?.error ? ` (${r.error})` : ""}`;
    const steps = num(r.steps) ? `${r.steps} step${r.steps === 1 ? "" : "s"}` : "steps unknown";
    return `pod ${p} ${steps}${num(r.seconds) ? ` in ${Math.round(r.seconds)} s` : ""}`;
  });
  const e = v.energy;
  const joules = e && num(e.joules)
    ? `about ${Math.round(e.joules).toLocaleString("en-US")} joules on the pods${num(e.watts_avg) ? ` (${Math.round(e.watts_avg)} W average)` : ""}`
    : "meters offline, no joules";
  return `training burst: ${parts.join(", ")}, ${joules}`;
}

/** What the button area says when the burst cannot start or did not work, in plain words; null when all is well. */
export function burstProblem(v: BurstView | null, postError: string | null): string | null {
  if (postError) return postError;
  if (!v) return null;
  if (v.state === "failed") return `The burst did not train: ${v.message ?? "no reason given"}`;
  if (v.state !== "running" && !v.available) return `Training is not available from this box: ${v.reason ?? "no reason given"}`;
  return null;
}
