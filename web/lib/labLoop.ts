/**
 * The learning loop, stage by stage, from one run's real events: what the 3D scene at the top of the Learning lab's
 * "The model" tab lights (components/lab/loopScene.ts) and what the strip of five stages under it says in words
 * (components/lab/LoopView.tsx). Everything here is worked out from the same run story the Training run view below it
 * draws (GET /api/learning/lab/{run_id}) and the same run list; nothing is typed in.
 *
 * No imports at run time (the types are erased), so `node scripts/check-lab.mjs` can run this file as it is.
 *
 * The fields the gateway added for this (PR 433) are optional and declared here, not in lib/lab.ts, so that file
 * stays exactly as the lab lane wrote it. An older gateway that does not send them still draws: a stage with no
 * report says "not reported".
 */
import type { LabNum, LabRunSummary, LabSource, LabStatus, LabStory } from "./lab";

export type Tone = "good" | "bad" | "watch" | "plain";

/** How a date is written: the caller passes the app's own formatter (lib/format.ts dayTime). */
export type When = (iso: string | null | undefined) => string;

/** The newest install the box's log holds (whichever run it belongs to), or `reason` when it holds none. */
export interface LabBoxNow {
  model_version: string | null;
  previous_version: string | null;
  since: string | null;
  source: LabSource | null;
  reason: string | null;
  /** Whether that install is this run's model. */
  this_run?: boolean | null;
}

/** A run's story with the optional fields the loop reads. */
export type LoopStory = Omit<LabStory, "eval"> & {
  eval: LabStory["eval"] & {
    /** Why the per-question answers cannot be the model's own (a rehearsal of the loop), or null. */
    rehearsal?: string | null;
    correct?: Record<string, LabNum>;
  };
  box_now?: LabBoxNow;
  /** The live loop's stages as the gateway lit them (lab_view._strip): labeling, training, packaging, offered. */
  stage_strip?: { stage: string; state: "done" | "now" | "waiting" | "failed" | "not reported"; started: string | null; ended: string | null; seconds: number | null; message: string | null }[];
  steps_so_far?: number;
};

/** A row of the run list with the optional test counts the gateway adds. */
export type LoopRun = LabRunSummary & {
  tested?: { decision: "deploy" | "reject"; n_items: number | null; right_before: number | null; right_after: number | null; rehearsal: string | null; source: LabSource | null } | null;
};

/* ---------- numbers ---------- */

/** A number as a sentence says it: 96, 51, 8.8, 0.33. Never 10.9589. */
export function said(v: number): string {
  const a = Math.abs(v);
  if (a >= 20) return Math.round(v).toLocaleString();
  if (a >= 1) return String(Math.round(v * 10) / 10);
  return String(Number(v.toPrecision(2)));
}

/** A loss as the screen prints it: two decimals, or two figures when it is small (0.0095, never 0.01 for it). */
export function lossWords(v: number): string {
  return Math.abs(v) >= 0.1 ? v.toFixed(2) : String(Number(v.toPrecision(2)));
}

/** A share of one, as a whole percent. */
export function percentWord(fraction: number): string {
  return `${Math.round(fraction * 100)}%`;
}

/* ---------- a run's state, in one sentence ---------- */

const STALE_MS = 10 * 60 * 1000; // a training run that has sent nothing for this long is not "running"

export interface RunState {
  /** The pill beside the title. */
  pill: string;
  tone: Tone;
  /** The one sentence. */
  line: string;
  /** Training, and a step arrived recently. */
  running: boolean;
  /** Training by the log, but silent for a while. */
  stale: boolean;
}

function span(ms: number): string {
  const m = Math.round(ms / 60000);
  if (m < 90) return `${m} min`;
  const h = Math.floor(m / 60);
  return h < 48 ? `${h} h ${m % 60} min` : `${Math.round(h / 24)} days`;
}

/** What a run is, said once: running, passed, refused, installed, undone, failed, or a rehearsal. `now` is the phone's clock. */
export function runState(story: Pick<LoopStory, "status" | "run_status" | "marks" | "eval" | "deploy" | "training" | "box_now" | "stages">, now: number | null, when: When): RunState {
  const st: LabStatus = story.status;
  const rehearsal = story.eval.rehearsal ?? null;
  const rejected = story.eval.decision === "reject";
  const last = story.marks.end ? Date.parse(story.marks.end) : NaN;
  const quiet = now !== null && !Number.isNaN(last) ? now - last : 0;
  const base = { running: false, stale: false };
  if (st === "training") {
    const steps = story.training.steps.value;
    const of = story.training.of.value;
    const at = typeof steps === "number" ? `step ${steps}${typeof of === "number" ? ` of ${of}` : ""}` : "no step yet";
    if (story.run_status && story.run_status !== "running")
      return { ...base, tone: "watch", pill: "Trained, not tested", line: `Training ended ${when(story.marks.end)} (${at}). No test result for it has reached the box.` };
    if (quiet > STALE_MS)
      return { ...base, stale: true, tone: "watch", pill: "No word since training", line: `The last word from this run came ${when(story.marks.end)} (${at}), ${span(quiet)} ago. Its test result has not reached the box, so this screen cannot say how it ended.` };
    return { ...base, running: true, tone: "plain", pill: "Training now", line: `Training now: ${at}. It is tested when training ends.` };
  }
  if (st === "failed") {
    const why = [...(story.stages ?? [])].reverse().find((s) => s.stage === "failed")?.message;
    return { ...base, tone: "bad", pill: "Failed", line: `This run failed${why ? `: ${why}` : " before it was tested"}. The box kept the model it had.` };
  }
  if (rehearsal) {
    const where = st === "deployed" ? " Its model was installed on the box all the same." : "";
    return { ...base, tone: "watch", pill: "Rehearsal", line: `A rehearsal of the loop, not proof: the test did not ask the model anything, so the score below says nothing about it.${where}` };
  }
  if (st === "gated") return { ...base, tone: "good", pill: "Passed the test", line: `Passed the test${story.marks.evaluated ? ` ${when(story.marks.evaluated)}` : ""}. Not on the box yet: the box's log holds no install of this model.` };
  if (st === "deployed") {
    const still = story.box_now?.this_run;
    const tail = still === false ? ` The box has since moved to another model (${when(story.box_now?.since)}).` : "";
    return rejected
      ? { ...base, tone: "watch", pill: "Installed without passing", line: `The test did not show this model is better, but it was installed on the box ${when(story.deploy.deployed)} anyway.${tail}` }
      : { ...base, tone: "good", pill: "On the box", line: `Passed the test and was installed on the box ${when(story.deploy.deployed)}.${tail}` };
  }
  if (st === "rolled back") {
    const back = story.deploy.rolled_back;
    return { ...base, tone: "bad", pill: "Installed, then undone", line: `Installed ${when(story.deploy.deployed)}, then the box went back to ${back?.to ?? "its earlier model"}${back?.ts ? ` ${when(back.ts)}` : ""}${back?.reason ? `: ${back.reason}` : ""}.` };
  }
  return { ...base, tone: "bad", pill: "Refused by the test", line: "Refused by the test: it was not shown to be better than the model on the box, so the box kept the model it had." };
}

/* ---------- the loop, stage by stage, from a run's real events ---------- */

export type LoopState = "done" | "now" | "waiting" | "failed" | "refused" | "none";
export type LoopKey = "out" | "label" | "train" | "test" | "box";

export interface LoopStage {
  key: LoopKey;
  title: string;
  state: LoopState;
  /** The run's own number for this stage, or null when the box holds none. */
  value: string | null;
  note: string | null;
}

/**
 * The five stages the 3D scene and the strip under it light, each from what the run really reported: the gateway's
 * stage strip (labeling, training, packaging), the steps, the test's result and the install. A stage the run sent
 * nothing about is "none" and says so; nothing is lit because the stage after it is.
 */
export function loopStages(story: LoopStory | null, when: When, opts: { standIn?: boolean; onBox?: string | null } = {}): LoopStage[] {
  const titles: Record<LoopKey, string> = { out: "Examples go out", label: "The teacher labels", train: "The trainer trains", test: "The test decides", box: "Back on the box" };
  const blank = (key: LoopKey): LoopStage => ({ key, title: titles[key], state: "waiting", value: null, note: null });
  if (!story) return (["out", "label", "train", "test", "box"] as LoopKey[]).map(blank);
  const strip = Object.fromEntries((story.stage_strip ?? []).map((x) => [x.stage, x]));
  const map = (s: string | undefined): LoopState => (s === "done" ? "done" : s === "now" ? "now" : s === "failed" ? "failed" : s === "waiting" ? "waiting" : "none");
  const secs = (x: { seconds: number | null } | undefined) => (x && typeof x.seconds === "number" ? `${said(x.seconds)} s` : null);
  const d = story.dataset;
  const n = typeof d.examples.value === "number" ? d.examples.value : typeof d.train.value === "number" && typeof d.held_out.value === "number" ? d.train.value + d.held_out.value : null;
  const t = story.training;
  const running = story.run_status ? story.run_status === "running" : story.status === "training";
  const steps = typeof t.steps.value === "number" ? t.steps.value : (story.steps_so_far ?? null);
  const loss = t.points.length ? t.points[t.points.length - 1]!.loss : t.loss_last.value;
  const trainState: LoopState = strip.training ? map(strip.training.state) : story.status === "failed" ? "failed" : steps ? (running && story.status === "training" ? "now" : "done") : "waiting";
  const ev = story.eval;
  const c = ev.correct ?? {};
  const before = c.current?.value;
  const after = c.candidate?.value;
  const nItems = ev.n_items.value;
  // A stand-in gate never asked the model: its counts are not a result, so no count is shown for it.
  const standIn = opts.standIn ?? !!ev.rehearsal;
  const score = standIn
    ? "stand-in gate"
    : typeof before === "number" && typeof after === "number" && typeof nItems === "number"
      ? `${before} → ${after} of ${nItems}`
      : typeof ev.scores.current.value === "number" && typeof ev.scores.candidate.value === "number"
        ? `${percentWord(ev.scores.current.value)} → ${percentWord(ev.scores.candidate.value)}`
        : null;
  const testState: LoopState = standIn ? "none" : !ev.decision ? (running && trainState === "done" ? "now" : "waiting") : ev.decision === "reject" ? "refused" : "done";
  const dep = story.deploy;
  // what the box serves now (the deploy record's `current`), when the caller knows it
  const version = dep.model_version ?? ev.model_version ?? story.package.model_version;
  const onBoxNow = !!opts.onBox && !!version && opts.onBox === version;
  const boxState: LoopState = onBoxNow ? "done" : story.status === "rolled back" ? "done" : dep.deployed ? "done" : strip.packaging?.state === "now" ? "now" : ev.decision === "reject" ? "none" : "waiting";
  const size = typeof story.package.bytes.value === "number" ? (story.package.bytes.value >= 1e9 ? `${(story.package.bytes.value / 1e9).toFixed(2)} GB` : `${Math.round(story.package.bytes.value / 1e6)} MB`) : null;
  const swap = typeof dep.swap_s.value === "number" ? `swapped in ${said(dep.swap_s.value)} s` : null;
  return [
    { key: "out", title: titles.out, state: n !== null || steps || strip.labeling ? "done" : "waiting", value: n !== null ? `${n.toLocaleString()} examples` : null, note: d.approval.approved ? `approved ${when(d.approval.approved)}` : null },
    { key: "label", title: titles.label, state: strip.labeling ? map(strip.labeling.state) : "none", value: secs(strip.labeling), note: !strip.labeling || strip.labeling.state === "not reported" ? "not reported for this run" : null },
    { key: "train", title: titles.train, state: trainState, value: steps ? `step ${steps}${typeof t.of.value === "number" ? ` of ${t.of.value}` : ""}` : null, note: [typeof loss === "number" ? `loss ${lossWords(loss)}` : null, secs(strip.training)].filter(Boolean).join(" · ") || null },
    { key: "test", title: titles.test, state: testState, value: score, note: standIn ? "no held-out score" : ev.decision === "reject" ? "refused" : ev.decision === "deploy" ? "passed" : null },
    {
      key: "box",
      title: titles.box,
      state: boxState,
      value: onBoxNow ? "on the box now" : story.status === "rolled back" ? "installed, later taken off" : dep.deployed ? `installed ${when(dep.deployed)}` : ev.decision === "reject" ? "kept its model" : ev.decision === "deploy" ? "not installed yet" : null,
      note: [size ? `${size} file` : null, swap, secs(strip.packaging) ? `packed in ${secs(strip.packaging)}` : null].filter(Boolean).join(" · ") || null,
    },
  ];
}

/** A run in a few words: what it was and how it ended. */
export function runRow(r: LoopRun): { what: string; chip: string; tone: Tone | "live" } {
  const what = (r.task ?? "A training run").replace(/[-_]/g, " ");
  const t = r.tested;
  const score = t && t.n_items && typeof t.right_before === "number" && typeof t.right_after === "number" ? `${t.right_before} → ${t.right_after} of ${t.n_items}` : typeof r.scores.current.value === "number" && typeof r.scores.candidate.value === "number" ? `${percentWord(r.scores.current.value)} → ${percentWord(r.scores.candidate.value)}` : null;
  const running = r.run_status ? r.run_status === "running" : r.status === "training";
  if (r.status === "training") return running ? { what, chip: "Training now", tone: "live" } : { what, chip: "Trained, not tested", tone: "watch" };
  if (r.status === "failed") return { what, chip: "Failed", tone: "bad" };
  if (t?.rehearsal) return { what, chip: "Rehearsal of the loop", tone: "watch" };
  const tail = score ? ` · ${score}` : "";
  if (r.status === "rolled back") return { what, chip: `Installed, then undone${tail}`, tone: "bad" };
  if (r.status === "deployed") return { what, chip: `On the box${tail}`, tone: "good" };
  if (r.status === "gated") return { what, chip: `Passed${tail}`, tone: "good" };
  return { what, chip: `Refused${tail}`, tone: "bad" };
}
