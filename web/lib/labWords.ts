/**
 * The Learning lab in plain words: the one-line takeaway under each tab's title, the one sentence that says what state
 * a run is in, and the screen's glossary. Everything here is worked out from what the gateway returned and nothing
 * else: a takeaway carries the numbers it stands on (`sources`), so each is one tap from the row or event it came from.
 *
 * No imports at run time (the types are erased), so `node scripts/check-lab.mjs` can run this file as it is.
 *
 * The order of the tabs is the order of the questions a person asks:
 *   1. is the box's model getting better, and what was the last run        (run)
 *   2. what does it cost in time and energy, and how does batching change it (speed)
 *   3. alone, or with a friend                                              (compare)
 *   4. what would routing do                                                (routing)
 *   5. what the lab has tried                                               (experiments)
 */
import type { LabBoxNow, LabNum, LabRunSummary, LabSource, LabStatus, LabStory } from "./lab";
import type { BenchPoint, BenchSweep, LabBench, LabCard, LabSim, LabTimeline } from "./labBench";
import type { CompareRun } from "./labCompare";

export type Tone = "good" | "bad" | "watch" | "plain";

/** One number a sentence stands on, and where it was read. `from` is the raw reference: an event, or a file and line. */
export interface Source {
  label: string;
  value?: string;
  from: string;
}

export interface Takeaway {
  /** The sentence under the title. */
  line: string;
  /** One more sentence, when the first needs it. */
  sub?: string | null;
  tone: Tone;
  sources: Source[];
}

/** How a date is written: the caller passes the app's own formatter (lib/format.ts dayTime). */
export type When = (iso: string | null | undefined) => string;

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

/** 0.21 -> "about a fifth of": a ratio below one, in the words a person would use. null when it is not a round share. */
export function shareWords(ratio: number): string | null {
  const named: [number, string][] = [
    [1 / 2, "about half"],
    [1 / 3, "about a third of"],
    [1 / 4, "about a quarter of"],
    [1 / 5, "about a fifth of"],
    [1 / 6, "about a sixth of"],
    [1 / 8, "about an eighth of"],
    [1 / 10, "about a tenth of"],
  ];
  for (const [x, words] of named) if (Math.abs(ratio - x) / x <= 0.11) return words;
  return null;
}

function ref(source: LabSource | null | undefined): string {
  return (source?.ref ?? "").replace("lab/refine/", "");
}

/* ---------- 1. is it getting better? ---------- */

function scoreWords(run: LabRunSummary): { before: string; after: string; n: number | null; b: number; a: number } | null {
  const t = run.tested;
  const sb = run.scores.current.value;
  const sa = run.scores.candidate.value;
  if (t && t.n_items && typeof t.right_before === "number" && typeof t.right_after === "number")
    return { before: `${t.right_before}`, after: `${t.right_after} of ${t.n_items}`, n: t.n_items, b: t.right_before / t.n_items, a: t.right_after / t.n_items };
  if (typeof sb === "number" && typeof sa === "number") return { before: percentWord(sb), after: percentWord(sa), n: null, b: sb, a: sa };
  return null;
}

/** A run tested for real: it has a decision and its test asked the model (not a rehearsal of the loop). */
export function isProof(run: LabRunSummary): boolean {
  if (run.tested === undefined) return run.status !== "training" && run.status !== "failed" && run.scores.candidate.value !== null; // an older box: no counts, no rehearsal flag
  return !!run.tested && !run.tested.rehearsal;
}

/** Where a tested model stands on the box, in one sentence. */
function whereWords(run: LabRunSummary, box: LabBoxNow | undefined, when: When): string {
  const rejected = run.tested?.decision === "reject";
  if (run.status === "rolled back") return `It was installed, then the box went back to its earlier model${run.rolled_back?.ts ? ` (${when(run.rolled_back.ts)})` : ""}.`;
  if (run.status === "deployed") {
    const still = box?.model_version && run.model_version ? box.model_version === run.model_version : null;
    return `${rejected ? "It was installed on the box anyway" : "It was installed on the box"}${run.deployed ? `, ${when(run.deployed)}` : ""}${still === false ? ", and has since been replaced" : ""}.`;
  }
  if (run.status === "gated") return "It is not on the box yet: the box's log holds no install of it.";
  return "The box kept the model it had.";
}

/**
 * The answer to "is the box's model getting better?", from the run list alone. The newest run whose test really asked
 * the model decides; a rehearsal of the loop and a run still training never do.
 */
export function betterTakeaway(runs: LabRunSummary[], box: LabBoxNow | undefined, when: When): { takeaway: Takeaway; run: LabRunSummary | null } | null {
  if (runs.length === 0) return null;
  const proof = runs.find(isProof) ?? null;
  const rehearsals = runs.filter((r) => r.tested?.rehearsal).length;
  if (!proof) {
    const training = runs.some((r) => r.status === "training");
    return {
      run: null,
      takeaway: {
        tone: "plain",
        line: "Not known yet: no run on this box has been tested for real.",
        sub: rehearsals
          ? `${rehearsals === 1 ? "The one tested run here is a rehearsal" : `The ${rehearsals} tested runs here are rehearsals`} of the loop: the test did not ask the model anything, so the score proves nothing.`
          : training
            ? "A run has trained; its test result has not reached the box."
            : null,
        sources: runs.filter((r) => r.tested?.rehearsal).map((r) => ({ label: "Why it is a rehearsal", value: r.tested!.rehearsal!, from: `${ref(r.tested!.source)}; run ${r.run_id}` })),
      },
    };
  }
  const s = scoreWords(proof);
  const t = proof.tested;
  const rejected = t ? t.decision === "reject" : proof.status === "not deployed";
  const questions = s?.n ? "held-back questions" : "of the held-back questions";
  let line: string;
  let tone: Tone;
  if (!s) {
    line = rejected ? "Not proven: the last tested model was refused by the test." : "The last tested model passed its test.";
    tone = rejected ? "watch" : "good";
  } else if (!rejected && s.a > s.b) {
    line = `Yes. The newest tested model got ${s.after} ${questions} right; the model before it got ${s.before}.`;
    tone = "good";
  } else if (!rejected) {
    line = `The newest tested model passed its test: ${s.after} ${questions} right, against ${s.before} before.`;
    tone = "good";
  } else if (s.a < s.b) {
    line = `No. The last tested model got worse: ${s.after} ${questions} right, down from ${s.before}, so the test refused it.`;
    tone = "bad";
  } else if (s.a === s.b) {
    line = `Not yet. The last tested model was no better than the one before: ${s.after} ${questions} right both times, so the test refused it.`;
    tone = "watch";
  } else {
    line = `Not proven. The last tested model got ${s.after} ${questions} right against ${s.before} before: too small a gain to tell from chance, so the test refused it.`;
    tone = "watch";
  }
  const newer = runs.indexOf(proof);
  const sources: Source[] = [];
  if (s && t) {
    sources.push({ label: "Right before training", value: s.n ? `${t.right_before} of ${s.n} (${percentWord(s.b)})` : s.before, from: ref(t.source) || ref(proof.scores.current.source) });
    sources.push({ label: "Right after training", value: s.n ? `${t.right_after} of ${s.n} (${percentWord(s.a)})` : s.after.replace(/ of .*/, ""), from: ref(t.source) || ref(proof.scores.candidate.source) });
  } else if (s) {
    sources.push({ label: "Score before", value: s.before, from: ref(proof.scores.current.source) });
    sources.push({ label: "Score after", value: s.after, from: ref(proof.scores.candidate.source) });
  }
  sources.push({ label: "The run", value: proof.model_version ?? undefined, from: `run ${proof.run_id}` });
  if (box?.model_version) sources.push({ label: "On the box now, by its log", value: `${box.model_version}, since ${when(box.since)}`, from: ref(box.source) });
  else if (box?.reason) sources.push({ label: "On the box now", value: "not known", from: box.reason });
  return {
    run: proof,
    takeaway: {
      tone,
      line,
      sub: `${whereWords(proof, box, when)}${newer > 0 ? ` ${newer === 1 ? "One newer run is not counted: it is" : `${newer} newer runs are not counted: they are`} ${rehearsals >= newer ? (newer === 1 ? "a rehearsal of the loop" : "rehearsals of the loop") : "still training, or rehearsals of the loop"}.` : ""}`,
      sources,
    },
  };
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
export function runState(story: Pick<LabStory, "status" | "run_status" | "marks" | "eval" | "deploy" | "training" | "box_now" | "stages">, now: number | null, when: When): RunState {
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

/** A run's status in the picker and the pill, when there is no story to read a fuller state from. */
export const STATUS_WORDS: Record<LabStatus, string> = {
  training: "Training",
  gated: "Passed the test",
  deployed: "On the box",
  "not deployed": "Refused by the test",
  "rolled back": "Installed, then undone",
  failed: "Failed",
};

/* ---------- reasons, without the event names ---------- */

const EVENT_WORDS: [RegExp, string][] = [
  [/learning\.task\.created/g, "task record"],
  [/learning\.dataset\.exported/g, "export record"],
  [/learning\.training\.started(\.\w+)?/g, "trainer's settings"],
  [/learning\.run\.evaluated/g, "test result"],
  [/learning\.eval\.reported/g, "per-question report"],
  [/learning\.package\.built/g, "package record"],
  [/learning\.swap\.completed/g, "swap record"],
  [/learning\.model\.deployed/g, "install record"],
  [/learning\.run\.progress/g, "progress record"],
  [/telemetry\.gpu\.power/g, "power"],
  [/telemetry\.train\.step/g, "training step"],
  [/telemetry\.joules_per_answer/g, "energy-per-answer record"],
  [/telemetry\.request/g, "request record"],
  [/split\.json/g, "list of held-back questions"],
  [/\bexamples\.jsonl\b/g, "examples file"],
];

/**
 * Why a number is missing, for a person: the gateway's reason with its event and file names swapped for what they are.
 * The reason as the gateway wrote it stays in the Sources sheet.
 */
export function plainReason(reason: string | null | undefined): string {
  if (!reason) return "";
  let out = reason
    .replace(/\s*\((?:learning|telemetry)\.[\w.]+\)/g, "")
    .replace(/no event says how many samples a training step holds/g, "the trainer's settings did not reach the box, so how many examples a step holds is not known");
  for (const [re, words] of EVENT_WORDS) out = out.replace(re, words);
  out = out
    .replace(/\bthe log has no (\w[\w' -]*?) event\b/g, "the box's log has no $1")
    .replace(/\bno (\w[\w' -]*?) event\b/g, "no $1")
    .replace(/\bthe (eval|package|export) event\b/g, (_m, w: string) => `the ${w === "eval" ? "test result" : w + " record"}`)
    .replace(/\s+/g, " ")
    .trim();
  return out.charAt(0).toUpperCase() + out.slice(1);
}

/* ---------- 2. speed and energy ---------- */

/** "vLLM", "llama.cpp": the engine a sweep ran on, from its build line, else nothing. */
export function engineName(sweep: Pick<BenchSweep, "build" | "family">): string | null {
  const b = `${sweep.build ?? ""} ${sweep.family}`.toLowerCase();
  if (b.includes("llamacpp") || b.includes("llama.cpp") || b.includes("llama-server")) return "llama.cpp";
  if (b.includes("vllm")) return "vLLM";
  return null;
}

/**
 * The sweep the tab opens on and the takeaway is read from: the newest one on the lab's own chip that covers at least
 * three load levels with its energy measured. A newer run of only two levels is still listed, but two points are not a
 * curve. With no such sweep, the newest of the lab's; with none of the lab's, the newest there is.
 */
export function headlineSweep(bench: Pick<LabBench, "sweeps">): BenchSweep | undefined {
  const own = bench.sweeps.filter((s) => s.chip.current && !s.other_chip);
  const byNewest = (a: BenchSweep, b: BenchSweep) => (b.ts ?? "").localeCompare(a.ts ?? "");
  const curves = own.filter((s) => s.batches.length >= 3 && (s.metrics.joules_per_token?.length ?? 0) >= 3).sort(byNewest);
  return curves[0] ?? [...own].sort(byNewest)[0] ?? bench.sweeps[0];
}

function at(sweep: BenchSweep, name: string, batch: number): BenchPoint | undefined {
  return sweep.metrics[name]?.find((p) => p.x === batch);
}

function pointSource(label: string, p: BenchPoint | undefined): Source[] {
  return p ? [{ label, value: `${said(p.value)} ${p.unit ?? ""}`.trim(), from: ref(p.source) }] : [];
}

/** What batching does to the cost of an answer, from one sweep's own first and last load level. */
export function batchTakeaway(sweep: BenchSweep | undefined, when: When): Takeaway | null {
  if (!sweep || sweep.batches.length < 2) return null;
  const first = sweep.batches[0]!;
  const last = sweep.batches[sweep.batches.length - 1]!;
  const j1 = at(sweep, "joules_per_token", first);
  const jN = at(sweep, "joules_per_token", last);
  const t1 = at(sweep, "decode_tok_s_total", first);
  const tN = at(sweep, "decode_tok_s_total", last);
  const w1 = at(sweep, "watts_avg", first);
  const wN = at(sweep, "watts_avg", last);
  if (!j1 || !jN) return null;
  const one = first === 1 ? "one at a time" : `${first} at once`;
  const ratio = jN.value / j1.value;
  let line: string;
  if (ratio >= 0.95) line = `Answering ${last} requests at once did not make a token cheaper here: ${said(jN.value)} J against ${said(j1.value)} J ${one}.`;
  else {
    const share = shareWords(ratio);
    const how = share ? `${share} the energy` : ratio < 0.15 ? `${said(1 / ratio)} times less energy` : `${percentWord(ratio)} of the energy`;
    line = `Answering ${last} requests at once makes each token cost ${how}: ${said(j1.value)} J ${one}, down to ${said(jN.value)} J.`;
  }
  const flat = w1 && wN && Math.abs(wN.value - w1.value) / w1.value <= 0.08;
  const power = w1 && wN ? (flat ? `The teacher draws about ${said((w1.value + wN.value) / 2)} W either way` : `The teacher draws ${said(w1.value)} W ${one} and ${said(wN.value)} W at ${last}`) : null;
  const work = t1 && tN ? `writes ${said(tN.value)} tokens a second in total instead of ${said(t1.value)}` : null;
  const sub = power && work ? `${power} and ${work}.` : power ? `${power}.` : work ? `In total it ${work}.` : null;
  const engine = engineName(sweep);
  return {
    tone: ratio < 0.95 ? "good" : "plain",
    line,
    sub,
    sources: [
      ...pointSource(`Energy per token, ${first === 1 ? "1 request" : `${first} requests`}`, j1),
      ...pointSource(`Energy per token, ${last} requests`, jN),
      ...pointSource(`Power, ${first === 1 ? "1 request" : `${first} requests`}`, w1),
      ...pointSource(`Power, ${last} requests`, wN),
      ...pointSource(`Tokens a second in total, ${first === 1 ? "1 request" : `${first} requests`}`, t1),
      ...pointSource(`Tokens a second in total, ${last} requests`, tN),
      { label: "Measured", value: `${sweep.model ?? "the teacher"}${engine ? ` on ${engine}` : ""}, ${sweep.node ?? "the lab"}, ${when(sweep.ts)}`, from: sweep.session ? `session ${sweep.session}` : sweep.id },
      ...(sweep.low_confidence > 0 ? [{ label: "How sure", value: `${sweep.low_confidence} of ${sweep.points} points are low confidence`, from: plainConfidence(sweep.confidence_note) }] : []),
    ],
  };
}

/** A sweep in the picker: "llama.cpp · 4 loads (1 to 16) · pod-b · Oct 11, 1:10 AM". */
export function sweepName(sweep: BenchSweep, when: When): string {
  const b = sweep.batches;
  return [engineName(sweep) ?? sweep.family.replace(/_/g, " "), `${b.length} loads (${b[0]} to ${b[b.length - 1]})`, sweep.node, when(sweep.ts)].filter(Boolean).join(" · ");
}

/** The lab's own note on how sure a sweep is, without its function names: "run(s)", "verdict()", session ids. */
export function plainConfidence(note: string | null | undefined): string {
  if (!note) return "";
  return note
    .replace(/,? all from one session \([^)]*\)/g, "")
    .replace(/\b(\d+) (run|spawn|repeat)\(s\)/g, (_m, n: string, w: string) => `${n} ${w === "spawn" ? "fresh start" : w}${n === "1" ? "" : "s"}`)
    .replace(/\bspawns\b/g, "fresh starts")
    .replace(/\bspawn\b/g, "fresh start")
    .replace(/,? ?(and )?under the 5 that verdict\(\) needs( to call a difference)?/g, "; the lab wants 5 before it calls a difference real")
    .replace(/: under 3 repeats;/g, ", under 3 repeats;")
    .replace(/ \(tap a point\)/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

/* ---------- 3. alone, or with a friend ---------- */

export function compareTakeaway(a: CompareRun | null, b: CompareRun | null, ratios: Record<string, number | null> | null): Takeaway | null {
  if (!a && !b) return null;
  const only = a ?? b!;
  if (!a || !b)
    return {
      tone: "plain",
      line: `Only one of the two runs is recorded so far: ${only.members === 1 ? "alone" : "with a friend"}, ${only.examples ?? "?"} examples${typeof only.j_per_example === "number" ? `, ${said(only.j_per_example)} J for each` : ""}.`,
      sub: "The comparison appears when the other run finishes.",
      sources: [{ label: "The run", from: `${only.run_id}: ${only.source.ref}` }],
    };
  const ja = a.j_per_example;
  const jb = b.j_per_example;
  const friends = b.members && b.members > 2 ? `${b.members - 1} friends` : "a friend";
  const sources: Source[] = [
    ...(typeof ja === "number" ? [{ label: "Energy per example, alone", value: `${said(ja)} J`, from: `${a.run_id}: ${a.source.ref}` }] : []),
    ...(typeof jb === "number" ? [{ label: `Energy per example, with ${friends}`, value: `${said(jb)} J`, from: `${b.run_id}: ${b.source.ref}` }] : []),
    ...(typeof a.pods_joules === "number" && typeof b.pods_joules === "number" ? [{ label: "Both pods, whole run", value: `${said(a.pods_joules)} J alone, ${said(b.pods_joules)} J with ${friends}`, from: a.source.rule ?? "the two runs' power traces" }] : []),
    { label: "Examples", value: `${a.examples ?? "?"} alone, ${b.examples ?? "?"} with ${friends}`, from: `${a.run_id}, ${b.run_id}` },
  ];
  if (typeof ja !== "number" || typeof jb !== "number")
    return { tone: "plain", line: `Both runs are recorded, but ${typeof ja !== "number" ? "the run alone" : "the pooled run"} has no power trace, so the energy cannot be compared.`, sub: a.reason ?? b.reason, sources };
  const r = jb / ja;
  const less = 1 - r;
  const total = ratios?.pods_joules;
  const ex = ratios?.examples;
  const whole = typeof total === "number" && typeof ex === "number" ? `The whole run used ${said(total)} times the energy for ${said(ex)} times the examples.` : null;
  if (r < 0.97) {
    const how = Math.abs(less - 0.25) < 0.035 ? "about a quarter less" : Math.abs(less - 1 / 3) < 0.04 ? "about a third less" : Math.abs(less - 0.5) < 0.05 ? "about half" : `${percentWord(less)} less`;
    return { tone: "good", line: `Pooling examples with ${friends} made each one cheaper to learn from: ${said(jb)} J an example instead of ${said(ja)} J, ${how}.`, sub: whole, sources };
  }
  if (r > 1.03) return { tone: "watch", line: `Pooling with ${friends} did not make an example cheaper here: ${said(jb)} J each, against ${said(ja)} J alone.`, sub: whole, sources };
  return { tone: "plain", line: `Pooling with ${friends} cost about the same for each example: ${said(jb)} J against ${said(ja)} J alone.`, sub: whole, sources };
}

/* ---------- 4. routing ---------- */

export function routingTakeaway(sim: LabSim): Takeaway | null {
  const cap = sim.policies?.capability;
  const eng = sim.policies?.energy;
  const n = sim.requests?.n;
  if (!sim.available || !cap || !eng || !n || !sim.compare) return null;
  const kj = (j: number) => (j >= 10_000 ? `${said(j / 1000)} kJ` : `${said(j)} J`);
  const same = sim.compare.moved === 0;
  const line = same
    ? `For this mix, weighing energy changes nothing: either way ${eng.to_teacher} of ${n} requests go to the lab's teacher and the rest stay on the box, for ${kj(eng.energy_j)}.`
    : `For this mix, weighing energy sends ${eng.to_teacher} of ${n} requests to the lab's teacher instead of ${cap.to_teacher}, and uses ${kj(eng.energy_j)} instead of ${kj(cap.energy_j)}${sim.compare.saved_pct ? ` (${said(Math.abs(sim.compare.saved_pct))}% ${sim.compare.saved_j >= 0 ? "less" : "more"})` : ""}${Math.abs(sim.compare.slower_s) >= 0.05 ? `, with answers about ${said(Math.abs(sim.compare.slower_s))} s ${sim.compare.slower_s > 0 ? "slower" : "faster"} on average` : ""}.`;
  const tj = sim.inputs?.teacher.j_per_token ?? [];
  return {
    tone: "plain",
    line,
    sub: `${eng.locked} of them never leave the box whatever the rule: they are about your own space. This is a what-if worked out from recorded numbers, not a live measurement.`,
    sources: [
      { label: "Requests to the teacher, energy weighed", value: `${eng.to_teacher} of ${n}`, from: "worked out by the box's own routing code on the mix below" },
      { label: "Requests to the teacher, by ability only", value: `${cap.to_teacher} of ${n}`, from: "the same code, with the energy rule off" },
      { label: "Energy, energy weighed", value: kj(eng.energy_j), from: "modelled: recorded energy per token times assumed answer lengths" },
      { label: "Energy, by ability only", value: kj(cap.energy_j), from: "modelled the same way" },
      ...(sim.inputs ? pointSource("The box's energy per token", sim.inputs.student_j_per_token) : []),
      ...(tj.length ? pointSource(`The teacher's energy per token, ${tj[tj.length - 1]!.x} at once`, tj[tj.length - 1]) : []),
    ],
  };
}

/** The two routing rules, named once in words: the code's names stay in the precise term. */
export const POLICY_WORDS: Record<"capability" | "energy", { title: string; means: string }> = {
  capability: { title: "By ability only", means: "The box answers whatever its small model can do; only the rest goes to the lab's teacher. (The code calls this rule capability.)" },
  energy: { title: "Energy weighed", means: "As above, and work the box could do still goes to the teacher when that costs less energy for the answer. (The code calls this rule energy.)" },
};

/** A measured metric's name in the routing table, for a person. The raw name stays behind the number's source. */
export function metricWords(metric: string): string {
  const known: Record<string, string> = {
    "label.seconds": "Time to label one example",
    "learning.label_agreement": "Labels that agree with the checked ones",
    "telemetry.joules_per_answer": "Energy for one answer",
  };
  return known[metric] ?? metric.replace(/^[a-z_]+\./, "").replace(/_/g, " ");
}

/* ---------- 5. what the lab has tried ---------- */

export function labTakeaway(t: LabTimeline): Takeaway {
  const e = t.experiments;
  const withVerdict = Math.max(0, e.run);
  const exp =
    e.total === 0
      ? "No experiment is planned"
      : withVerdict > 0
        ? `${withVerdict} of the lab's ${e.total} experiments ${withVerdict === 1 ? "has" : "have"} a result`
        : `None of the lab's ${e.total} planned experiments has a result yet${e.running ? ` (${e.running} running)` : ""}`;
  const tests = `${t.sessions} speed test${t.sessions === 1 ? "" : "s"} ${t.sessions === 1 ? "is" : "are"} measured`;
  const passes = t.passes ? ` and ${t.passes} review pass${t.passes === 1 ? "" : "es"} of its own work ${t.passes === 1 ? "is" : "are"} on record` : "";
  return {
    tone: "plain",
    line: `${exp}; ${tests}${passes}.`,
    sub: e.source.kind === "ledger" ? null : "This box does not hold the lab's own record of its experiments, so the planned ones are shown as the lab's code lists them.",
    sources: [
      { label: "Experiments", value: `${e.total} planned, ${e.queued} waiting, ${e.running} running, ${e.run} with a result, ${e.dropped} dropped`, from: e.source.ref ?? e.source.note ?? "the queue in the lab's code" },
      { label: "Speed tests", value: String(t.sessions), from: "sessions in metrics.jsonl" },
      { label: "Review passes", value: String(t.passes), from: "the REFINE_LOG.md files" },
    ],
  };
}

/** A card's verdict line without the lab's function names. */
export function plainVerdict(verdict: string | null | undefined): string {
  if (!verdict) return "";
  const m = /^Measured, no verdict: (.*)$/.exec(verdict);
  if (m) return `Measured, too few times to call a difference: ${plainConfidence(m[1]).replace(/; the lab wants 5 before it calls a difference real/g, "")}. The lab wants 5.`.replace(/\s+\./g, ".");
  if (verdict.startsWith("Not converged")) return "Not finished: the review found more to change.";
  if (verdict.startsWith("Converged")) return "Finished: the review found nothing left to change.";
  return verdict;
}

/** A card's title: "Benchmark: teacher_llamacpp on pod-a" -> "Speed test: the teacher on llama.cpp, pod-a". */
export function plainCardTitle(card: Pick<LabCard, "title" | "kind">): string {
  const m = /^Benchmark: (\S+) on (\S+)$/.exec(card.title);
  if (!m) return card.title;
  const family = m[1]!;
  const engine = family.includes("llamacpp") ? " on llama.cpp" : family.includes("batching") ? ", many at once" : "";
  return `Speed test: the ${family.startsWith("teacher") ? "teacher" : family.replace(/_/g, " ")}${engine}, ${m[2] === "a100" ? "NVIDIA A100 (not our hardware)" : m[2]}`;
}

export const CARD_WORDS: Record<string, string> = {
  queued: "Planned",
  running: "Running",
  won: "Better",
  lost: "Not better",
  inconclusive: "No clear difference",
  dropped: "Dropped",
  measured: "Measured",
  converged: "Finished",
  open: "Not finished",
};

/* ---------- the glossary ---------- */

export interface Word {
  /** What the screen says. */
  plain: string;
  /** The lab's own term, kept one tap away. */
  term: string;
  means: string;
}

export const GLOSSARY: Record<string, Word> = {
  gate: { plain: "the test", term: "the gate", means: "The test a new model must pass before it may replace the one on the box: it has to do better on held-back questions, by more than chance." },
  heldout: { plain: "held-back questions", term: "held-out set", means: "Questions kept out of training. The model never sees them while it learns, so they test what it learned and not what it memorised." },
  loss: { plain: "loss", term: "training loss", means: "How wrong the model is on its own training examples. Lower is better. It falling does not prove the model got better: only the test does." },
  teacher: { plain: "the teacher", term: "teacher model", means: "The large model on the lab's two pods. It labels the examples and answers what the small model cannot." },
  student: { plain: "the box's model", term: "student model", means: "The small model that runs on the box itself. Training teaches it to do one job the way the teacher would." },
  batch: { plain: "at once", term: "batch size", means: "How many requests the teacher answers at the same moment. More at once is slower for each person and cheaper for each answer." },
  tokens: { plain: "tokens a second", term: "tok/s", means: "A token is a piece of a word, a little under a word on average. Tokens a second is how fast a model writes." },
  joules: { plain: "energy per token", term: "J/token", means: "The energy to write one token. A joule is one watt for one second." },
  low: { plain: "low confidence", term: "low confidence", means: "Measured too few times to call a difference real. The lab wants 5 repeats; these have fewer." },
  sweep: { plain: "a run of the same test at different loads", term: "sweep", means: "The same speed test repeated with 1, 4, 8 and more requests at once, so the points can be read as one curve." },
  lora: { plain: "a small add-on", term: "LoRA adapter", means: "A small set of extra weights trained on top of the model. The model itself is not changed, so the add-on can be dropped again." },
  pvalue: { plain: "how likely by chance", term: "p-value", means: "How often a gain this large would show up by luck alone if the model were no better. The test wants it below 0.05, one time in twenty." },
  noise: { plain: "run-to-run wobble", term: "run-to-run noise", means: "Score the same model twice and the score moves a little. A gain has to be larger than that wobble to count." },
  measured: { plain: "measured", term: "measured", means: "A row the lab recorded on a machine, with the file and line it sits on." },
  derived: { plain: "worked out", term: "derived", means: "Not measured directly: worked out from measured rows, with the formula shown." },
  modelled: { plain: "what-if", term: "modelled", means: "What a model of the recorded rows predicts. Not a measurement." },
};

/* ---------- one slide per tab: the hero ---------- */

/** A hero's big line: numbers count up, strings sit between them. */
export type BigPart = { n: number; digits?: number } | string;

export interface HeroData {
  big: BigPart[];
  /** At most about eight words. */
  line: string;
  tiles: { value: string; label: string }[];
  tone: Tone;
  sources: Source[];
}

/**
 * The model tab's slide: the newest real test, before and after, and what the test said. Counts when the box holds
 * them ("95 → 179 of 228"), percentages when it holds only the scores. A rehearsal of the loop never leads.
 */
export function modelHero(runs: LabRunSummary[], box: LabBoxNow | undefined, when: When): { hero: HeroData; run: LabRunSummary | null } | null {
  const found = betterTakeaway(runs, box, when);
  if (!found) return null;
  const proof = found.run;
  const real = runs.filter(isProof);
  const said_no = real.filter((r) => (r.tested ? r.tested.decision === "reject" : r.status === "not deployed")).length;
  const said_yes = real.length - said_no;
  const tiles = [
    ...(real.length ? [{ value: `${said_yes} passed, ${said_no} refused`, label: "what the test has said so far" }] : []),
    { value: box?.model_version ?? "not recorded", label: box?.model_version ? `on the box since ${when(box.since)}` : "which model is on the box" },
    { value: String(runs.length), label: runs.length === 1 ? "training run on this box" : "training runs on this box" },
  ];
  if (!proof) return { run: null, hero: { big: [], line: "No run has been tested for real yet.", tiles, tone: "plain", sources: found.takeaway.sources } };
  const s = scoreWords(proof);
  const t = proof.tested;
  const rejected = t ? t.decision === "reject" : proof.status === "not deployed";
  const where = proof.status === "rolled back" ? "Installed, then undone." : proof.status === "deployed" ? "On the box." : proof.status === "gated" ? "Waiting to go on the box." : "The box kept its model.";
  const verdict = !s ? (rejected ? "Refused by the test." : "Passed the test.") : rejected ? (s.a < s.b ? "Refused: it got worse." : s.a === s.b ? "Refused: no better than before." : "Refused: too small a gain to trust.") : "Passed the test.";
  const big: BigPart[] = !s
    ? []
    : s.n && t && typeof t.right_before === "number" && typeof t.right_after === "number"
      ? [{ n: t.right_before }, "→", { n: t.right_after }, `of ${s.n}`]
      : [{ n: Math.round(s.b * 100) }, "%", "→", { n: Math.round(s.a * 100) }, "%"];
  return { run: proof, hero: { big, line: `${verdict} ${where}`, tiles, tone: rejected ? (s && s.a < s.b ? "bad" : "watch") : "good", sources: [{ label: "What is counted", value: "held-back questions answered right, before and after training", from: "the test's own result" }, ...found.takeaway.sources] } };
}

/**
 * The speed tab's slide: how much faster the teacher writes now than on the lab's first full measurement of the same
 * pod, at one request at a time, and that the power did not move. Both ends are sweeps the gateway serves; with only
 * one sweep there is no "before", and the slide leads with the batching saving instead.
 */
export function speedHero(bench: Pick<LabBench, "sweeps" | "power">, when: When): HeroData | null {
  const lead = headlineSweep(bench);
  const take = batchTakeaway(lead, when);
  if (!lead || !take) return null;
  const first = lead.batches[0]!;
  const last = lead.batches[lead.batches.length - 1]!;
  const older = bench.sweeps
    .filter((s) => s.id !== lead.id && s.chip.current && !s.other_chip && s.node === lead.node && s.family === lead.family && s.batches[0] === first && (s.ts ?? "") < (lead.ts ?? ""))
    .sort((a, b) => (a.ts ?? "").localeCompare(b.ts ?? ""))[0];
  const tokNew = at(lead, "decode_tok_s", first);
  const tokOld = older ? at(older, "decode_tok_s", first) : undefined;
  const jNew = at(lead, "joules_per_token", first);
  const jOld = older ? at(older, "joules_per_token", first) : undefined;
  const jMany = at(lead, "joules_per_token", last);
  const wNew = at(lead, "watts_avg", first);
  const wOld = older ? at(older, "watts_avg", first) : undefined;
  const idle = bench.power.nodes.find((n) => n.node === lead.node)?.idle ?? null;
  const tiles = [
    ...(jOld && jNew ? [{ value: `${said(jOld.value)} → ${said(jNew.value)} J`, label: "energy for one token, one request at a time" }] : []),
    ...(jMany ? [{ value: `${said(jMany.value)} J`, label: `for one token with ${last} requests at once` }] : []),
    ...(idle ? [{ value: `${said(idle.value)} W`, label: "waiting, with the whole model loaded" }] : []),
  ];
  const sources = [
    ...pointSource("Tokens a second, before", tokOld),
    ...pointSource("Tokens a second, now", tokNew),
    ...pointSource("Power, before", wOld),
    ...pointSource("Power, now", wNew),
    ...pointSource("Energy per token, before", jOld),
    ...pointSource("Energy per token, now", jNew),
    ...pointSource(`Energy per token, ${last} at once`, jMany),
    ...pointSource("Idle power", idle ?? undefined),
    ...(older ? [{ label: "Before", value: `${engineName(older) ?? "the teacher"}, ${older.node}, ${when(older.ts)}`, from: older.session ? `session ${older.session}` : older.id }] : []),
    { label: "Now", value: `${engineName(lead) ?? "the teacher"}, ${lead.node}, ${when(lead.ts)}`, from: lead.session ? `session ${lead.session}` : lead.id },
    ...(lead.low_confidence ? [{ label: "How sure", value: "every point is low confidence", from: plainConfidence(lead.confidence_note) }] : []),
  ];
  if (tokOld && tokNew && wOld && wNew && tokNew.value > tokOld.value * 1.2) {
    const same = Math.abs(wNew.value - wOld.value) / wOld.value <= 0.08;
    return {
      big: [{ n: Math.round((tokNew.value / tokOld.value) * 10) / 10, digits: 1 }, "× faster"],
      line: same ? `on the same ${said((wNew.value + wOld.value) / 2)} W: ${said(tokOld.value)} → ${said(tokNew.value)} tokens a second.` : `${said(tokOld.value)} → ${said(tokNew.value)} tokens a second, at ${said(wNew.value)} W.`,
      tiles,
      tone: "good",
      sources,
    };
  }
  if (!jNew || !jMany) return null;
  const ratio = jNew.value / jMany.value;
  return { big: [{ n: Math.round(ratio * 10) / 10, digits: 1 }, "× less energy"], line: `for each token, answering ${last} requests at once.`, tiles, tone: "good", sources: take.sources };
}

/** The Together tab's slide: what pooling a friend's examples did to the cost of learning from one. */
export function togetherHero(a: CompareRun | null, b: CompareRun | null, ratios: Record<string, number | null> | null): HeroData | null {
  const take = compareTakeaway(a, b, ratios);
  if (!take || !a || !b) return take ? { big: [], line: take.line, tiles: [], tone: "plain", sources: take.sources } : null;
  const ex = ratios?.examples;
  const en = ratios?.pods_joules;
  const secs = (r: CompareRun) => (typeof r.seconds.total === "number" ? `${said(r.seconds.total)} s` : "not timed");
  const stage = (r: CompareRun) => (["labeling", "training", "packaging"] as const).map((k) => (typeof r.seconds[k] === "number" ? `${k === "labeling" ? "label" : k === "training" ? "train" : "package"} ${said(r.seconds[k]!)} s` : null)).filter(Boolean).join(" · ");
  const tiles = [
    ...(typeof a.j_per_example === "number" && typeof b.j_per_example === "number" ? [{ value: `${said(a.j_per_example)} → ${said(b.j_per_example)} J`, label: "energy to learn from one example" }] : []),
    { value: `${a.examples ?? "?"} → ${b.examples ?? "?"}`, label: "examples, alone and pooled" },
    { value: `${secs(a)} → ${secs(b)}`, label: "start to finish" },
  ];
  const sources = [...take.sources, ...(stage(a) ? [{ label: "The loop alone, stage by stage", value: stage(a), from: `${a.run_id}: ${a.source.ref}` }] : [])];
  if (typeof ex === "number" && typeof en === "number")
    return { big: [{ n: Math.round(ex * 10) / 10, digits: ex % 1 ? 1 : 0 }, "× the examples"], line: `for ${said(en)}× the energy.`, tiles, tone: en < ex ? "good" : "plain", sources };
  return { big: [], line: take.line, tiles, tone: take.tone, sources };
}

/** Routing in one sentence, from one run of the box's own routing code on recorded numbers. */
export function routingLine(sim: LabSim): { line: string; sources: Source[] } | null {
  const cap = sim.policies?.capability;
  const eng = sim.policies?.energy;
  const n = sim.requests?.n;
  if (!sim.available || !cap || !eng || !n || !sim.compare) return null;
  const kj = (j: number) => (j >= 10_000 ? `${said(j / 1000)} kJ` : `${said(j)} J`);
  const slower = eng.mean_latency_s - cap.mean_latency_s;
  const pct = cap.mean_latency_s ? Math.abs(slower / cap.mean_latency_s) * 100 : 0;
  const pace = Math.abs(slower) < 0.05 ? "no slower" : `${said(Math.abs(slower))} s ${slower > 0 ? "slower" : "faster"} an answer${pct >= 0.5 ? ` (${said(pct)}%)` : ""}`;
  const line =
    sim.compare.moved === 0
      ? `Weighing energy changes nothing here: ${eng.to_teacher} of ${n} requests go to the teacher either way, ${kj(eng.energy_j)}.`
      : `Sending ${eng.to_teacher} of ${n} requests to the teacher spends ${kj(eng.energy_j)} instead of ${kj(cap.energy_j)} and is ${pace}.`;
  return { line, sources: routingTakeaway(sim)?.sources ?? [] };
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
export function loopStages(story: LabStory | null, when: When): LoopStage[] {
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
  const score =
    typeof before === "number" && typeof after === "number" && typeof nItems === "number"
      ? `${before} → ${after} of ${nItems}`
      : typeof ev.scores.current.value === "number" && typeof ev.scores.candidate.value === "number"
        ? `${percentWord(ev.scores.current.value)} → ${percentWord(ev.scores.candidate.value)}`
        : null;
  const testState: LoopState = !ev.decision ? (running && trainState === "done" ? "now" : "waiting") : ev.decision === "reject" ? "refused" : "done";
  const dep = story.deploy;
  const boxState: LoopState = story.status === "rolled back" ? "failed" : dep.deployed ? "done" : strip.packaging?.state === "now" ? "now" : ev.decision === "reject" ? "none" : "waiting";
  const size = typeof story.package.bytes.value === "number" ? (story.package.bytes.value >= 1e9 ? `${(story.package.bytes.value / 1e9).toFixed(2)} GB` : `${Math.round(story.package.bytes.value / 1e6)} MB`) : null;
  const swap = typeof dep.swap_s.value === "number" ? `swapped in ${said(dep.swap_s.value)} s` : null;
  return [
    { key: "out", title: titles.out, state: n !== null || steps || strip.labeling ? "done" : "waiting", value: n !== null ? `${n.toLocaleString()} examples` : null, note: d.approval.approved ? `approved ${when(d.approval.approved)}` : null },
    { key: "label", title: titles.label, state: strip.labeling ? map(strip.labeling.state) : "none", value: secs(strip.labeling), note: !strip.labeling || strip.labeling.state === "not reported" ? "not reported for this run" : null },
    { key: "train", title: titles.train, state: trainState, value: steps ? `step ${steps}${typeof t.of.value === "number" ? ` of ${t.of.value}` : ""}` : null, note: [typeof loss === "number" ? `loss ${lossWords(loss)}` : null, secs(strip.training)].filter(Boolean).join(" · ") || null },
    { key: "test", title: titles.test, state: testState, value: score, note: ev.rehearsal ? "a stand-in test: not proof" : ev.decision === "reject" ? "refused" : ev.decision === "deploy" ? "passed" : null },
    {
      key: "box",
      title: titles.box,
      state: boxState,
      value: story.status === "rolled back" ? "installed, then undone" : dep.deployed ? `installed ${when(dep.deployed)}` : ev.decision === "reject" ? "kept its model" : ev.decision === "deploy" ? "not installed yet" : null,
      note: [size ? `${size} file` : null, swap, secs(strip.packaging) ? `packed in ${secs(strip.packaging)}` : null].filter(Boolean).join(" · ") || null,
    },
  ];
}

/** A run in one compact row of the timeline: what it was and how it ended. */
export function runRow(r: LabRunSummary): { what: string; chip: string; tone: Tone | "live" } {
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

/** One number the gateway sent, as a source line. */
export function numSource(label: string, n: LabNum | null | undefined, value?: string): Source[] {
  if (!n || n.value === null) return [];
  return [{ label, value: value ?? `${said(n.value)}${n.unit ? ` ${n.unit}` : ""}`, from: ref(n.source) }];
}
