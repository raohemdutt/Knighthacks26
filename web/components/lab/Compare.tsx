"use client";

import dynamic from "next/dynamic";
import { useMemo, useState } from "react";
import { Empty } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api } from "@/lib/api";
import { isLiveRun, isNum, type LabStory } from "@/lib/lab";
import {
  COMPARE_ROWS,
  type CompareRun,
  type CompareTrace,
  HIGHER_IS_BETTER,
  type HeroTile,
  type LabCompare,
  LOWER_IS_BETTER,
  type RunMotion,
  defaultPair,
  fmtValue,
  heroLine,
  heroTiles,
  isFreshRun,
  podWords,
  recordedOnly,
  runName,
  timesX,
  trainingPods,
} from "@/lib/labCompare";
import { useFetch } from "@/lib/useFetch";
import { usePoll } from "@/lib/usePoll";
import b from "./bench.module.css";
import c from "./compare.module.css";
import { type BuddiesState, LearningBuddies } from "./LearningBuddies";

/** The power chart draws with ECharts in the browser only; the page is a static export. */
const TogetherTraces = dynamic(() => import("./TogetherTraces"), { ssr: false, loading: () => <div className={c.chart} style={{ minHeight: 480 }} /> });

/**
 * Alone vs with a friend (L31): act 1 trained on one person's examples, act 2 on those pooled with a friend's. Read top
 * down, it says the result at a glance: the learning buddies in the matching state, three hero numbers (act 2 over act
 * 1: the examples, the time, the energy per example), which two acts are compared, the three power traces of the loop on
 * one clock (TogetherTraces.tsx, ECharts) and each act in brief. Every other number (seconds per stage, the joules of
 * labeling, training, packaging and the box with each idle floor beside, T4: gross, never subtracted; act 2 over act 1
 * line by line; where the numbers come from) sits under "Every number", closed until asked for.
 *
 * The buddies follow a run as "training now" only while its run_status is "running" and its newest event is less than
 * two minutes old (lib/labCompare.ts, isFreshRun); a loop that died mid-run is never shown as training. Otherwise they
 * show the comparison's finished pair, together, with that pair's own numbers.
 *
 * Data, all from the gateway and all live: GET /api/learning/lab/compare?a=&b= (the two acts), GET /api/learning/lab
 * (a run still going on the box) and GET /api/learning/lab/{run_id} (its loss, step by step). A run whose record is the
 * committed rehearsal is never shown (lib/labCompare.ts, recordedOnly). Until both acts are recorded the tab looks again
 * every 3 s, and with nothing recorded it says so.
 *
 *   /lab/?tab=compare&a=<run_id>&b=<run_id>
 */

const POLL_MS = 3000;
const NODES = ["pod-c", "pod-d", "box"] as const;
const NODE_TITLE: Record<(typeof NODES)[number], string> = { "pod-c": "Pod C, the teacher (labeling)", "pod-d": "Pod D, the trainer", box: "The box (Jetson)" };

type Loaded = { data: LabCompare | null; why: string | null };

/** The comparison, with committed fixtures taken out; act 1 and act 2 picked from what the box recorded. */
async function loadCompare(a: string | null, bId: string | null, signal: AbortSignal): Promise<Loaded> {
  let raw: LabCompare;
  try {
    raw = await api.labCompare(a, bId, signal);
  } catch (err) {
    if (signal.aborted) throw err;
    const status = err instanceof ApiError ? err.status : 0;
    return { data: null, why: status === 404 ? "The gateway does not serve the comparison yet." : status === 0 ? "The gateway could not be reached." : `The gateway answered ${status}.` };
  }
  let d = recordedOnly(raw);
  if (!a && !bId) {
    const pick = defaultPair(d.runs);
    if ((pick.a || pick.b) && (pick.a !== raw.asked.a || pick.b !== raw.asked.b)) d = recordedOnly(await api.labCompare(pick.a, pick.b, signal));
    else if (!pick.a && !pick.b) d = { ...d, a: null, b: null, ratios: null };
  }
  return { data: d, why: null };
}

/** The loss at each reported step, from the run's story. */
function lossOf(story: LabStory | null | undefined): number[] {
  return (story?.training.points ?? []).map((p) => p.loss).filter((v): v is number => typeof v === "number");
}

/* ---------- the two acts ---------- */

const ROW = Object.fromEntries(COMPARE_ROWS.map((r) => [r.key, r]));
const ACT_GROUPS: { title: string; rows: [string, string][] }[] = [
  { title: "Seconds per stage", rows: [["labeling_s", "Labeling"], ["training_s", "Training"], ["packaging_s", "Packaging"], ["act_seconds", "Start to finish"]] },
  { title: "Labeling, pod C", rows: [["labeling_j", "Joules"], ["labeling_j_per_token", "Per token"], ["labeling_j_per_example", "Per example"]] },
  { title: "Training, pod D", rows: [["training_j", "Joules"], ["training_j_per_step", "Per step"], ["training_j_per_train_row", "Per example"]] },
  { title: "Packaging, pod D", rows: [["packaging_j", "Joules"]] },
  { title: "Totals", rows: [["pods_joules", "Both pods"], ["j_per_example", "Per example, all stages"], ["box_j", "The box, during the act"]] },
];

/** Where an act ran, in words: "pod D" alone, "pods C and D" with a friend. */
function whereOf(run: CompareRun): string {
  const pods = trainingPods(run);
  return pods.length ? `trained on ${podWords(pods)}` : "";
}

function Act({ n, run, title, detail = false }: { n: 1 | 2; run: CompareRun | null; title: string; detail?: boolean }) {
  if (!run)
    return (
      <article className={`${c.act} ${c.actWaiting} ${n === 1 ? c.actA : c.actB}`}>
        <span className={c.eyebrow}>Act {n}</span>
        <h2>{title}</h2>
        <p className={b.note}>Not recorded yet. It appears here on its own once the loop finishes this act.</p>
      </article>
    );
  const total = run.seconds.total ?? 0;
  const parts = (["labeling", "training", "packaging"] as const).map((k) => ({ k, s: run.seconds[k] ?? 0 }));
  return (
    <article className={`${c.act} ${n === 1 ? c.actA : c.actB}`}>
      <span className={c.eyebrow}>
        Act {n}
        {whereOf(run) ? <span className={c.where}>, {whereOf(run)}</span> : null}
      </span>
      <h2>{title}</h2>
      {detail ? null : (
      <div className={c.big}>
        <div>
          <span className={c.bigNum}>{run.examples ?? "?"}</span>
          <span className={c.bigLabel}>examples</span>
        </div>
        <div>
          <span className={c.bigNum}>{run.members ?? "?"}</span>
          <span className={c.bigLabel}>{run.members === 1 ? "member" : "members"}</span>
        </div>
        <div>
          <span className={c.bigNum}>{typeof run.j_per_example === "number" ? Math.round(run.j_per_example) : "?"}</span>
          <span className={c.bigLabel}>J per example</span>
        </div>
      </div>
      )}
      {total > 0 ? (
        <div className={c.split} role="img" aria-label={parts.map((p) => `${p.k} ${p.s} s`).join(", ")}>
          {parts.map((p) => (p.s > 0 ? <span key={p.k} className={c[`split_${p.k}`]} style={{ flexGrow: p.s }} title={`${p.k}: ${p.s} s`} /> : null))}
        </div>
      ) : null}
      {detail ? (
      <dl className={c.rows}>
        {ACT_GROUPS.map((g) => (
          <div key={g.title} className={c.group}>
            <dt className={c.groupTitle}>{g.title}</dt>
            {g.rows.map(([key, label]) => {
              const row = ROW[key]!;
              const v = row.value(run);
              const idle = row.idle?.(run);
              return (
                <dd key={key} className={c.row}>
                  <span>{label}</span>
                  <span className={v === null || v === undefined ? c.none : c.num}>
                    {fmtValue(v, row.unit)}
                    {typeof idle === "number" ? <span className={c.idle}>idle {fmtValue(idle, "J")}</span> : null}
                  </span>
                </dd>
              );
            })}
          </div>
        ))}
      </dl>
      ) : null}
      {run.reason ? <p className={b.note}>{run.reason}.</p> : null}
      {detail ? <p className={`${b.note} ${b.code}`}>{run.run_id}</p> : null}
    </article>
  );
}

/* ---------- act 2 over act 1 ---------- */

function Ratios({ ratios }: { ratios: Record<string, number | null> }) {
  const keys = COMPARE_ROWS.filter((r) => typeof ratios[r.key] === "number");
  return (
    <div className={b.card}>
      <div className={b.cardHead}>
        <h3>Act 2 over act 1</h3>
        <span className={b.note}>1x means the same; below 1x on a per-example line means the friend made it cheaper.</span>
      </div>
      <ul className={c.ratios}>
        {keys.map((r) => {
          const x = ratios[r.key]!;
          const good = (LOWER_IS_BETTER.has(r.key) && x < 1) || (HIGHER_IS_BETTER.has(r.key) && x > 1);
          return (
            <li key={r.key} className={c.ratioRow}>
              <span className={c.ratioLabel}>{r.group === "Seconds" ? `${r.label}, seconds` : r.label}</span>
              <span className={c.ratioTrack} aria-hidden="true">
                <span className={c.ratioOne} />
                <span className={`${c.ratioBar} ${good ? c.ratioBarGood : ""}`} style={{ width: `${Math.min(100, (x / 2.5) * 100)}%` }} />
              </span>
              <span className={`${c.ratio} ${good ? c.good : ""}`}>{timesX(x)}</span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/* ---------- the three power traces on one time axis ---------- */

function Panel({ node, ra, rb, maxT, idle, note, last }: { node: (typeof NODES)[number]; ra: CompareTrace | undefined; rb: CompareTrace | undefined; maxT: number; idle: number | null; note: string | null; last: boolean }) {
  const W = 640;
  const H = last ? 128 : 108;
  const pad = { l: 44, r: 10, t: 10, b: last ? 26 : 8 };
  const all = [...(ra?.points ?? []), ...(rb?.points ?? [])];
  const maxW = Math.max(20, idle ?? 0, ...all.map((p) => p[1])) * 1.1;
  const x = (t: number) => pad.l + (Math.max(0, Math.min(t, maxT)) / maxT) * (W - pad.l - pad.r);
  const y = (w: number) => pad.t + (1 - w / maxW) * (H - pad.t - pad.b);
  const line = (tr?: CompareTrace) =>
    (tr?.points ?? [])
      .filter((p) => p[0] >= 0 && p[0] <= maxT)
      .map((p) => `${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`)
      .join(" ");
  const step = maxT > 240 ? 60 : 30;
  const ticks: number[] = [];
  for (let t = 0; t <= maxT; t += step) ticks.push(t);
  const peak = (tr?: CompareTrace) => (tr?.points.length ? Math.max(...tr.points.map((p) => p[1])) : null);
  const pa = peak(ra);
  const pb = peak(rb);
  return (
    <div className={c.panel}>
      <div className={c.panelHead}>
        <span className={c.panelTitle}>{NODE_TITLE[node]}</span>
        <span className={c.panelPeak}>
          {pa !== null ? `peak ${Math.round(pa)} W` : "no samples"}
          {" / "}
          {pb !== null ? `${Math.round(pb)} W` : "no samples"}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className={c.svg} role="img" aria-label={`${NODE_TITLE[node]}: watts over each act, act 1 dashed, act 2 solid${idle !== null ? `, idle floor ${idle} W` : ""}`}>
        {[0, 0.5, 1].map((f) => (
          <g key={f}>
            <line x1={pad.l} x2={W - pad.r} y1={y(f * maxW)} y2={y(f * maxW)} className={c.grid} />
            <text x={pad.l - 6} y={y(f * maxW) + 4} className={c.axis} textAnchor="end">
              {Math.round(f * maxW)} W
            </text>
          </g>
        ))}
        {ticks.map((t) => (
          <line key={t} x1={x(t)} x2={x(t)} y1={pad.t} y2={H - pad.b} className={c.gridV} />
        ))}
        {last
          ? ticks.map((t) => (
              <text key={t} x={x(t)} y={H - 6} className={c.axis} textAnchor="middle">
                {t} s
              </text>
            ))
          : null}
        {idle !== null ? (
          <g>
            <line x1={pad.l} x2={W - pad.r} y1={y(idle)} y2={y(idle)} className={c.idleLine} />
            <text x={W - pad.r} y={y(idle) - 5} className={c.idleText} textAnchor="end">
              idle {idle} W
            </text>
          </g>
        ) : null}
        <polyline points={line(ra)} className={`${c.line} ${c.lineA}`} />
        <polyline points={line(rb)} className={`${c.line} ${c.lineB}`} />
      </svg>
      {note ? <span className={b.note}>{note}</span> : null}
    </div>
  );
}

/** The fallback when the gateway sends no loop clock (an older gateway, or two acts from different loops). */
function Traces({ ra, rb }: { ra: CompareRun; rb: CompareRun | null }) {
  const maxT = Math.max(ra.seconds.total ?? 0, rb?.seconds.total ?? 0, 30);
  const idleOf = (n: string) => {
    const row = (rb?.idle ?? ra.idle)?.[n];
    return row ? (row.calibrated_w ?? row.quiet_w ?? null) : null;
  };
  const noteOf = (n: string) => (rb?.idle ?? ra.idle)?.[n]?.note ?? null;
  return (
    <div className={b.card}>
      <div className={b.cardHead}>
        <h3>Power, on one clock</h3>
        <span className={c.legend}>
          <span className={c.keyA}>Act 1, alone</span>
          <span className={c.keyB}>Act 2, with a friend</span>
          <span className={c.keyIdle}>idle floor</span>
        </span>
      </div>
      <span className={b.note}>Seconds from each act&apos;s own start, so the two acts line up. Watts as each meter read them.</span>
      <div className={c.stack}>
        {NODES.map((n, i) => (
          <Panel key={n} node={n} ra={ra.traces[n]} rb={rb?.traces[n]} maxT={maxT} idle={idleOf(n)} note={noteOf(n)} last={i === NODES.length - 1} />
        ))}
      </div>
    </div>
  );
}

/* ---------- the hero ---------- */

function fmtPlain(v: number | null, unit: string): string {
  if (v === null) return "?";
  const n = Math.abs(v) >= 100 ? Math.round(v).toLocaleString("en-US") : Number(v.toPrecision(3)).toLocaleString("en-US");
  return unit ? `${n} ${unit}` : n;
}

function Hero({ tiles, line }: { tiles: HeroTile[]; line: string | null }) {
  if (!tiles.length) return null;
  return (
    <section className={c.hero} aria-label="Act 2 over act 1">
      <ul className={c.heroTiles}>
        {tiles.map((t, i) => (
          <li key={t.key} className={`${c.heroTile} ${t.good ? c.heroGood : ""}`} style={{ "--i": i } as React.CSSProperties}>
            <span className={c.heroNum}>{t.times}</span>
            <span className={c.heroNoun}>{t.noun}</span>
            <span className={c.heroWords}>{t.words}</span>
            {t.from !== null && t.to !== null ? (
              <span className={c.heroFrom}>
                {fmtPlain(t.from, t.unit)} alone, {fmtPlain(t.to, t.unit)} with a friend
              </span>
            ) : null}
          </li>
        ))}
      </ul>
      {line ? <p className={c.heroLine}>{line}</p> : null}
      {tiles.some((t) => t.key === "j_per_example" && t.good) ? <p className={c.whyLine}>Friends make learning cheaper per example.</p> : null}
    </section>
  );
}

function PowerChart({ data, ra, rb, forceReduced }: { data: LabCompare; ra: CompareRun | null; rb: CompareRun | null; forceReduced?: boolean }) {
  const series = data.series ?? null;
  const ids = new Set([ra?.run_id, rb?.run_id].filter(Boolean));
  const usable = !!series && series.runs.some((r) => ids.has(r.run_id)) && Object.values(series.nodes).some((nd) => nd && nd.points.length > 0);
  if (!usable) return ra ? <Traces ra={ra} rb={rb} /> : rb ? <Traces ra={rb} rb={null} /> : null;
  const runs = series!.runs.filter((r) => ids.has(r.run_id));
  const together = runs.length === 2 && runs[0]!.start_s < runs[1]!.end_s && runs[1]!.start_s < runs[0]!.end_s;
  return (
    <div className={b.card}>
      <div className={b.cardHead}>
        <h3>Power, on one clock</h3>
        <span className={c.legend}>
          <span className={c.keyPodC}>Pod C</span>
          <span className={c.keyPodD}>Pod D</span>
          <span className={c.keyBox}>The box</span>
          <span className={c.keyStage}>act 2&apos;s stages</span>
          <span className={c.keyIdle}>idle floor</span>
        </span>
      </div>
      <span className={b.note}>
        Seconds after the loop started, every meter moved onto one clock. Watts as each meter read them.
        {together ? " Both acts ran in the same loop at the same time, so each meter reads both." : ""}
      </span>
      <TogetherTraces series={series!} actB={rb?.run_id ?? null} forceReduced={forceReduced} />
      <p className={c.whyLine}>Three meters, one clock, gross joules.</p>
    </div>
  );
}

/* ---------- the tab ---------- */

function Picker({ label, value, runs, onPick }: { label: string; value: string | null; runs: { run_id: string; label: string }[]; onPick: (id: string) => void }) {
  return (
    <label className={c.pick}>
      <span>{label}</span>
      <select className={ui.select} value={value ?? ""} onChange={(e) => onPick(e.target.value)}>
        {value ? null : <option value="">Not recorded yet</option>}
        {runs.map((r) => (
          <option key={r.run_id} value={r.run_id}>
            {r.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function Compare({ a, b: bId, onPick, forceReduced }: { a: string | null; b: string | null; onPick: (which: "a" | "b", id: string) => void; forceReduced?: boolean }) {
  const res = useFetch((signal) => loadCompare(a, bId, signal), `compare-${a ?? ""}-${bId ?? ""}`);
  const list = useFetch((signal) => api.lab(false, signal).catch(() => null), "list");
  const data = res.data?.data ?? null;
  const ra = data?.a ?? null;
  const rb = data?.b ?? null;
  const complete = !!ra && !!rb;

  // A run training now on the box (the loop posts each act there as it trains): the buddies follow it live. Only the
  // gateway's run_status "running" counts (lib/lab.ts, isLiveRun), and only while its newest event is under two minutes
  // old (lib/labCompare.ts, isFreshRun): a loop that died mid-run keeps saying "running" but its events stop.
  const now = Date.now();
  const live =
    list.data && !list.data.sample ? (list.data.runs.find((r) => isLiveRun({ ...r, sample: false }) && isFreshRun(r as RunMotion, now)) ?? null) : null;
  const [picking, setPicking] = useState(false);
  const focusId = live?.run_id ?? rb?.run_id ?? ra?.run_id ?? null;
  const story = useFetch((signal) => (focusId ? api.labRun(focusId, false, signal).catch(() => null) : Promise.resolve(null)), `story-${focusId ?? ""}`);
  const focus = story.data && !story.data.sample ? story.data : null;

  // Look again every 3 s until both acts are recorded, and while a run is still going.
  // A request still on its way is left to finish, never restarted (lib/usePoll.ts): a box slower than the beat answers.
  const watching = !complete || !!live;
  usePoll([res, list, story], watching, POLL_MS);

  const buddies = useMemo(() => {
    const loss = lossOf(focus);
    const stepsOf = (r: CompareRun | null) => (typeof r?.training?.steps === "number" ? r.training.steps : 0);
    const storySteps = focus ? (isNum(focus.training.steps) ? focus.training.steps.value : focus.training.points.length) : 0;
    let state: BuddiesState = "idle";
    if (live) state = live.run_id.endsWith("-duo") && ra ? "with-friend" : "alone"; // the loop names act 2's run <stamp>-duo
    else if (complete) state = "with-friend"; // both acts recorded: the comparison's result, the two boxes together
    // only act 1 recorded and nothing training now: the box rests
    const run = live ? null : state === "with-friend" ? rb : ra;
    const liveJpe = focus && isNum(focus.power.j_per_sample) ? focus.power.j_per_sample.value : null;
    return {
      state,
      steps: Math.max(storySteps, stepsOf(run)),
      loss,
      joulesPerExample: live ? liveJpe : (run?.j_per_example ?? null),
      joulesPerExampleAlone: state === "with-friend" ? (ra?.j_per_example ?? null) : null,
      examplesRatio: state === "with-friend" ? (data?.ratios?.examples ?? null) : null,
      friends: state === "with-friend" ? Math.max(1, (rb?.members ?? 2) - 1) : 0,
      lossReason: loss.length ? null : run ? "The per-step loss of this run is not in the box's log." : null,
      caption: live ? `Training now on the box: ${live.run_id}` : run ? `Finished: ${runName(run).toLowerCase()}${whereOf(run) ? `, ${whereOf(run)}` : ""}. ${run.run_id}` : null,
    };
  }, [focus, live, complete, ra, rb, data]);

  if (res.loading && !res.data)
    return (
      <div className={b.page}>
        <Head />
        <LearningBuddies state="idle" steps={0} loss={[]} joulesPerExample={null} friends={0} forceReduced={forceReduced} />
      </div>
    );

  const why = res.data?.why ?? null;
  const choices = (data?.runs ?? []).map((r) => ({ run_id: r.run_id, label: `${runName(r)}${r.measured ? "" : ", no power trace"}, ${r.run_id}` }));

  return (
    <div className={b.page}>
      <Head />
      <LearningBuddies
        state={buddies.state}
        steps={buddies.steps}
        loss={buddies.loss}
        joulesPerExample={buddies.joulesPerExample}
        joulesPerExampleAlone={buddies.joulesPerExampleAlone}
        examplesRatio={buddies.examplesRatio}
        friends={buddies.friends}
        lossReason={buddies.lossReason}
        caption={buddies.caption}
        forceReduced={forceReduced}
      />

      {!ra && !rb ? (
        <>
          <Empty title="No runs yet; start the loop">
            This tab shows only what the box has recorded. Act 1 (alone) and act 2 (with a friend) appear here on their own as the live loop finishes them; the
            page looks again every 3 seconds.
          </Empty>
          {why ? <p className={b.note}>{why} Looking again every 3 seconds.</p> : null}
          {data?.missing.length ? <p className={b.note}>No recorded run with the id {data.missing.join(" or ")}.</p> : null}
        </>
      ) : (
        <>
          {complete && data ? <Hero tiles={heroTiles(data)} line={heroLine(data)} /> : null}
          <p className={c.comparing}>
            Comparing <b>{ra ? runName(ra).toLowerCase() : "act 1, not recorded yet"}</b> with <b>{rb ? runName(rb).toLowerCase() : "act 2, not recorded yet"}</b>
            {choices.length > 1 ? (
              <button type="button" className={c.change} aria-expanded={picking} onClick={() => setPicking((v) => !v)}>
                {picking ? "done" : "change"}
              </button>
            ) : null}
          </p>
          {picking && choices.length > 1 ? (
            <div className={c.pickers}>
              <Picker label="Act 1" value={ra?.run_id ?? null} runs={choices} onPick={(id) => onPick("a", id)} />
              <Picker label="Act 2" value={rb?.run_id ?? null} runs={choices} onPick={(id) => onPick("b", id)} />
            </div>
          ) : null}
          {data ? <PowerChart data={data} ra={ra} rb={rb} forceReduced={forceReduced} /> : null}
          <div className={c.acts}>
            <Act n={1} run={ra} title={ra ? runName(ra).replace(/ \(.*\)$/, "") : "Alone"} />
            <Act n={2} run={rb} title={rb ? runName(rb).replace(/ \(.*\)$/, "") : "With a friend"} />
          </div>
          <details className={c.every}>
            <summary className={c.everySummary}>Every number</summary>
            <div className={c.everyBody}>
          <div className={c.acts}>
            <Act n={1} run={ra} title={ra ? runName(ra).replace(/ \(.*\)$/, "") : "Alone"} detail />
            <Act n={2} run={rb} title={rb ? runName(rb).replace(/ \(.*\)$/, "") : "With a friend"} detail />
          </div>
          {complete && data?.ratios ? <Ratios ratios={data.ratios} /> : null}
          <div className={b.card}>
            <h3>Where these numbers come from</h3>
            {[ra, rb].map((r) =>
              r ? (
                <span key={r.run_id} className={`${b.note} ${b.code}`}>
                  {r.run_id}: {r.source.ref}
                </span>
              ) : null,
            )}
            {(ra ?? rb)?.source.rule ? <span className={b.note}>Rule: {(ra ?? rb)!.source.rule}.</span> : null}
            {(ra ?? rb)?.source.split ? <span className={b.note}>{(ra ?? rb)!.source.split}.</span> : null}
            {!complete ? <span className={b.note}>Waiting for the other act; looking again every 3 seconds.</span> : null}
          </div>
            </div>
          </details>
        </>
      )}
    </div>
  );
}

function Head() {
  return (
    <div className={ui.pageHead}>
      <h1>Alone vs with a friend</h1>
      <p className={b.sub}>
        The same task trained twice: once on one person&apos;s examples, then pooled with a friend&apos;s. Joules are gross, as the meters read them, with the
        idle floor beside each one, never subtracted.
      </p>
    </div>
  );
}
