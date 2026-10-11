/**
 * The "16 people asking" button on the live power graph (components/lab/LoadButton.tsx): many streams on the 122B
 * pair at once from POST /api/learning/lab/load, its state from GET (contexts/learning/load_run.py). Pure rules, free
 * of imports, so scripts/check-lab.mjs can run them in node. A missing number is said as missing, never made up.
 */

export type LoadState = "idle" | "running" | "done" | "failed";

export interface LoadEnergy {
  joules: number;
  watts_avg: number;
  seconds: number;
  pods: Record<string, { joules: number; watts_avg: number; idle_w: number | null; samples: number } | null>;
}

export interface LoadView {
  available: boolean;
  reason: string | null;
  state: LoadState;
  streams: number | null;
  seconds: number | null;
  started: string | null;
  ended: string | null;
  started_ms: number | null;
  ended_ms: number | null;
  requests: number;
  errors: number;
  tokens: number;
  tok_s_total: number | null;
  tok_s_per_stream: number | null;
  ttft_s: number | null;
  energy: LoadEnergy | null;
  j_per_token: number | null;
  message: string | null;
}

/** The streams picker, 16 first: the default. */
export const LOAD_STREAMS = [16, 4, 32] as const;
export const LOAD_SECONDS = 30;

/** "16 people asking", "1 person asking". */
export function askingText(streams: number): string {
  return `${streams} ${streams === 1 ? "person" : "people"} asking`;
}

/** The button's words. */
export function loadButtonText(streams: number, seconds = LOAD_SECONDS): string {
  return `${askingText(streams)} for ${seconds} s`;
}

const num = (x: unknown): x is number => typeof x === "number" && Number.isFinite(x);
const whole = (x: number) => Math.round(x).toLocaleString("en-US");

/**
 * The line under the graph once a run ended: "48 requests, 4,512 tokens, 150 tok/s total, watts stayed at 101 W,
 * 0.66 J per token". Meters that did not cover the window say so.
 */
export function loadLine(v: LoadView): string {
  const rate = num(v.tok_s_total) ? `${whole(v.tok_s_total)} tok/s total` : "tok/s unknown";
  const e = v.energy;
  const watts = e && num(e.watts_avg) ? `watts stayed at ${whole(e.watts_avg)} W` : "meters offline, no watts";
  const jpt = num(v.j_per_token) ? `${v.j_per_token.toFixed(2)} J per token` : "no J per token";
  const failed = v.errors ? `, ${v.errors} failed` : "";
  return `${whole(v.requests)} requests, ${whole(v.tokens)} tokens, ${rate}, ${watts}, ${jpt}${failed}`;
}

/** What the button area says when the run cannot start or did not work, in plain words; null when all is well. */
export function loadProblem(v: LoadView | null, postError: string | null): string | null {
  if (postError) return postError;
  if (!v) return null;
  if (v.state === "failed") return `The load run got no answers: ${v.message ?? "no reason given"}`;
  if (v.state !== "running" && !v.available) return `The pair is not reachable from this box: ${v.reason ?? "no reason given"}`;
  return null;
}

/** One ramp stage's numbers as GET reports them, filled the moment the stage ends (watts once the run is over). */
export interface LoadStage {
  streams: number;
  requests: number;
  tokens: number;
  tok_s_total: number | null;
  watts_avg: number | null;
  j_per_token: number | null;
  started_ms: number;
  ended_ms: number;
}

export interface RampView extends LoadView {
  ramp: number[] | null;
  seconds_per_stage: number | null;
  stage: number | null;
  stages: LoadStage[];
}

export const RAMP = [1, 4, 16, 32];
/** The seconds-per-stage picker, 20 first: the default. */
export const RAMP_STAGE_SECONDS = [20, 10, 30] as const;

/** The title while a ramp runs: "ramp: 16 of 32 streams". */
export function rampTitle(v: RampView): string | null {
  if (v.state !== "running" || !v.ramp || !v.stage) return null;
  return `ramp: ${v.ramp[v.stage - 1]} of ${v.ramp[v.ramp.length - 1]} streams`;
}

/**
 * The stages on the graph's clock (browser ms): each finished stage from its reported offsets against the run's
 * start, the running one from where the last finished one ended until `now`. `pressed` is when the press was
 * answered, on the browser clock.
 */
export function stageWindows(v: RampView, pressed: number, now: number): { start: number; end: number; label: string }[] {
  if (!v.ramp || !v.seconds_per_stage) return [];
  const t0 = v.started_ms ?? 0;
  const out = v.stages.map((s) => ({ start: pressed + (s.started_ms - t0), end: pressed + (s.ended_ms - t0), label: String(s.streams) }));
  if (v.state === "running" && v.stage && out.length < v.stage) {
    const start = out.length ? out[out.length - 1]!.end : pressed;
    out.push({ start, end: now, label: String(v.ramp[v.stage - 1]) });
  }
  return out;
}

/** The ramp's table once done: one row per stage, numbers said as missing when missing. */
export function rampRows(v: RampView): { streams: string; tokS: string; watts: string; jpt: string }[] {
  return v.stages.map((s) => ({
    streams: String(s.streams),
    tokS: num(s.tok_s_total) ? whole(s.tok_s_total) : "-",
    watts: num(s.watts_avg) ? `${whole(s.watts_avg)} W` : "-",
    jpt: num(s.j_per_token) ? s.j_per_token.toFixed(2) : "-",
  }));
}
