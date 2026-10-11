"use client";

import { useEffect, useMemo } from "react";
import { Empty } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api } from "@/lib/api";
import { isLiveRun, isNum, type LabStory } from "@/lib/lab";
import { dayTime } from "@/lib/format";
import {
  COMPARE_ROWS,
  type CompareRun,
  type CompareTrace,
  HIGHER_IS_BETTER,
  type LabCompare,
  LOWER_IS_BETTER,
  defaultPair,
  fmtValue,
  recordedOnly,
  runName,
  timesX,
} from "@/lib/labCompare";
import { said, togetherHero } from "@/lib/labWords";
import { useFetch } from "@/lib/useFetch";
import { usePoll } from "@/lib/usePoll";
import b from "./bench.module.css";
import c from "./compare.module.css";
import { useWidth } from "./Labels";
import { type BuddiesState, LearningBuddies } from "./LearningBuddies";
import { ChartWords, Fold, Hero, Words } from "./Story";
import s from "./story.module.css";

/**
 * Alone vs with a friend (L31): act 1 trained on one person's examples, act 2 on those pooled with a friend's, side by
 * side. Per act: examples, members, seconds per stage, the joules of labeling (pod C), training and packaging (pod D)
 * and the box, each with its idle floor beside (T4: gross joules, the idle shown, never subtracted); then act 2 over
 * act 1, and the three power traces stacked on one time axis. The learning buddies sit above in the matching state.
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

/** The box's own comparison, with committed fixtures taken out; act 1 and act 2 picked from what the box recorded. */
async function loadOwn(a: string | null, bId: string | null, signal: AbortSignal): Promise<Loaded> {
  let raw: LabCompare;
  try {
    raw = await api.labCompare(a, bId, signal);
  } catch (err) {
    if (signal.aborted) throw err;
    const status = err instanceof ApiError ? err.status : 0;
    return { data: null, why: status === 404 ? "This box has not recorded a run alone and a run with a friend yet." : status === 0 ? "The box could not be reached." : `The box answered ${status}.` };
  }
  let d = recordedOnly(raw);
  if (!a && !bId) {
    const pick = defaultPair(d.runs);
    if ((pick.a || pick.b) && (pick.a !== raw.asked.a || pick.b !== raw.asked.b)) d = recordedOnly(await api.labCompare(pick.a, pick.b, signal));
    else if (!pick.a && !pick.b) d = { ...d, a: null, b: null, ratios: null };
  }
  return { data: d, why: null };
}

/**
 * What the Together tab shows: the box's own comparison when it has one, else the lab's recorded rehearsal, which is a
 * second source with a name of its own (GET /api/learning/lab/rehearsal) and always carries its label (`record`). The
 * two are never mixed. With neither, `data` is null and the tab is not shown at all (app/lab/page.tsx).
 */
export async function loadTogether(a: string | null, bId: string | null, signal: AbortSignal): Promise<Loaded> {
  const own = await loadOwn(a, bId, signal);
  if (own.data?.a || own.data?.b) return own;
  try {
    return { data: await api.labRehearsal(signal), why: own.why };
  } catch (err) {
    if (signal.aborted) throw err;
    return { data: null, why: own.why };
  }
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

function Act({ n, run, title }: { n: 1 | 2; run: CompareRun | null; title: string }) {
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
      <span className={c.eyebrow}>Act {n}</span>
      <h2>{title}</h2>
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
      {total > 0 ? (
        <div className={c.split} role="img" aria-label={parts.map((p) => `${p.k} ${p.s} s`).join(", ")}>
          {parts.map((p) => (p.s > 0 ? <span key={p.k} className={c[`split_${p.k}`]} style={{ flexGrow: p.s }} title={`${p.k}: ${p.s} s`} /> : null))}
        </div>
      ) : null}
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
      {run.reason ? <p className={b.note}>{run.reason}.</p> : null}
    </article>
  );
}

/* ---------- act 2 over act 1 ---------- */

function Ratios({ ratios }: { ratios: Record<string, number | null> }) {
  const keys = COMPARE_ROWS.filter((r) => typeof ratios[r.key] === "number");
  return (
    <div className={b.card}>
      <div className={b.cardHead}>
        <h3>With a friend, against alone</h3>
        <span className={b.note}>Each bar is the pooled run divided by the run alone: 1x is the same, 2x is double. Green is where pooling did better.</span>
      </div>
      <ul className={c.ratios}>
        {keys.map((r) => {
          const x = ratios[r.key]!;
          const good = (LOWER_IS_BETTER.has(r.key) && x < 1) || (HIGHER_IS_BETTER.has(r.key) && x > 1);
          return (
            <li key={r.key} className={c.ratioRow}>
              <span className={c.ratioLabel}>{r.group === "Seconds" ? `${r.label}, seconds` : r.group === "Joules" ? `${r.label}, joules` : r.label}</span>
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

function Panel({ node, ra, rb, maxT, idle, note, last, width }: { node: (typeof NODES)[number]; ra: CompareTrace | undefined; rb: CompareTrace | undefined; maxT: number; idle: number | null; note: string | null; last: boolean; width: number }) {
  // drawn at the card's real width, so the axis labels stay 12 px on a phone instead of shrinking with the picture
  const W = Math.max(width, 260);
  const H = last ? 136 : 112;
  const pad = { l: 46, r: 10, t: 10, b: last ? 30 : 8 };
  const all = [...(ra?.points ?? []), ...(rb?.points ?? [])];
  const maxW = Math.max(20, idle ?? 0, ...all.map((p) => p[1])) * 1.1;
  const x = (t: number) => pad.l + (Math.max(0, Math.min(t, maxT)) / maxT) * (W - pad.l - pad.r);
  const y = (w: number) => pad.t + (1 - w / maxW) * (H - pad.t - pad.b);
  const line = (tr?: CompareTrace) =>
    (tr?.points ?? [])
      .filter((p) => p[0] >= 0 && p[0] <= maxT)
      .map((p) => `${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`)
      .join(" ");
  const step = (maxT > 240 ? 60 : 30) * (W < 420 ? 2 : 1); // fewer ticks on a phone, so "150 s" and "180 s" do not touch
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
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} className={c.svg} role="img" aria-label={`${NODE_TITLE[node]}: power in watts over each run, second by second; alone dashed, with a friend solid${idle !== null ? `; idle it draws ${idle} W` : ""}`}>
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
              <text key={t} x={x(t)} y={H - 10} className={c.axis} textAnchor={t === 0 ? "start" : "middle"}>
                {t === 0 ? (W < 620 ? "0 s" : "0 s from the start") : `${t} s`}
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

function Traces({ ra, rb }: { ra: CompareRun; rb: CompareRun | null }) {
  const maxT = Math.max(ra.seconds.total ?? 0, rb?.seconds.total ?? 0, 30);
  const idleOf = (n: string) => {
    const row = (rb?.idle ?? ra.idle)?.[n];
    return row ? (row.calibrated_w ?? row.quiet_w ?? null) : null;
  };
  const noteOf = (n: string) => (rb?.idle ?? ra.idle)?.[n]?.note ?? null;
  const [ref, width] = useWidth();
  const peak = (tr?: CompareTrace) => (tr?.points.length ? Math.round(Math.max(...tr.points.map((p) => p[1]))) : null);
  return (
    <div className={b.card} data-guide="lab:compare-power">
      <div className={b.cardHead}>
        <h3>Power while each run worked, second by second</h3>
        <span className={c.legend}>
          <span className={c.keyA}>Alone</span>
          <span className={c.keyB}>With a friend</span>
          <span className={c.keyIdle}>idle</span>
        </span>
      </div>
      <span className={b.note}>Watts over each run, counted from its own start. Dotted: what the machine draws doing nothing.</span>
      <div className={c.stack} ref={ref}>
        {width > 0 ? NODES.map((n, i) => <Panel key={n} node={n} ra={ra.traces[n]} rb={rb?.traces[n]} maxT={maxT} idle={idleOf(n)} note={noteOf(n)} last={i === NODES.length - 1} width={width} />) : null}
      </div>
      <ChartWords>
        <b>The chart in words.</b>{" "}
        {NODES.map((n) => {
          const a = peak(ra.traces[n]);
          const z = peak(rb?.traces[n]);
          const idle = idleOf(n);
          return `${NODE_TITLE[n]}: ${a !== null ? `peaked at ${a} W alone` : "no samples alone"}${rb ? (z !== null ? `, ${z} W with a friend` : ", no samples with a friend") : ""}${idle !== null ? `; idle ${said(idle)} W` : ""}.`;
        }).join(" ")}
      </ChartWords>
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
  const res = useFetch((signal) => loadTogether(a, bId, signal), `compare-${a ?? ""}-${bId ?? ""}`);
  const list = useFetch((signal) => api.lab(false, signal).catch(() => null), "list");
  const data = res.data?.data ?? null;
  const ra = data?.a ?? null;
  const rb = data?.b ?? null;
  const complete = !!ra && !!rb;

  // A run training now on the box (the loop posts each act there as it trains): the buddies follow it live. Only the
  // gateway's run_status "running" counts (lib/lab.ts, isLiveRun); a gated or finished run is not training now.
  const live = list.data && !list.data.sample ? (list.data.runs.find((r) => isLiveRun({ ...r, sample: false })) ?? null) : null;
  const focusId = live?.run_id ?? rb?.run_id ?? ra?.run_id ?? null;
  const story = useFetch((signal) => (focusId ? api.labRun(focusId, false, signal).catch(() => null) : Promise.resolve(null)), `story-${focusId ?? ""}`);
  const focus = story.data && !story.data.sample ? story.data : null;

  // Look again every 3 s until both acts are recorded, and while a run is still going; while the lab's rehearsal
  // stands in for a box with no record of its own, every 15 s. Never while the last ask is still out (lib/usePoll.ts).
  const rehearsal = data?.record ?? null;
  usePoll([res, list, story], !rehearsal && (!complete || !!live), POLL_MS);
  usePoll([res, list], !!rehearsal, 15_000);

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
    // A run that has finished training but is not installed yet is still "live" for the page (it looks again every 3 s),
    // but it is not training: once its record is in, the card shows the recorded joules for each example, and the
    // caption never says "training now" about a run whose training is over.
    const recorded = live ? ([ra, rb].find((r) => r?.run_id === live.run_id) ?? null) : null;
    const trainingNow = live?.status === "training";
    return {
      state,
      steps: Math.max(storySteps, stepsOf(run ?? recorded)),
      loss,
      joulesPerExample: live ? (liveJpe ?? recorded?.j_per_example ?? null) : (run?.j_per_example ?? null),
      joulesPerExampleAlone: state === "with-friend" ? (ra?.j_per_example ?? null) : null,
      examplesRatio: state === "with-friend" ? (data?.ratios?.examples ?? null) : null,
      friends: state === "with-friend" ? Math.max(1, (rb?.members ?? 2) - 1) : 0,
      lossReason: loss.length ? null : run ? "How the loss fell, step by step, has not reached the box for this run." : null,
      caption: live ? (trainingNow ? "Training now on the box." : "The newest run has finished training. It is not installed on the box yet.") : run ? `The run shown: ${runName(run).toLowerCase()}.` : null,
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
  // A run in the picker by what it is and when it started, not by its id. The relay stamps a run "MMDD-HHMMSS" on its
  // own clock; a run read from the box's log carries a full time.
  const stampWords = (stamp: string | null): string => {
    const m = /^(\d{2})(\d{2})-(\d{2})(\d{2})\d{2}$/.exec(stamp ?? "");
    if (m) {
      const d = new Date(new Date().getFullYear(), Number(m[1]) - 1, Number(m[2]), Number(m[3]), Number(m[4]));
      return d.toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
    }
    return stamp && !Number.isNaN(Date.parse(stamp)) ? dayTime(stamp) : "";
  };
  const choices = (data?.runs ?? []).map((r) => {
    const name = runName(r);
    return { run_id: r.run_id, label: `${name === r.run_id ? "A run with no alone or pooled label" : name}${r.measured ? "" : ", no power measured"}${stampWords(r.stamp) ? ` · ${stampWords(r.stamp)}` : ""}` };
  });
  const hero = togetherHero(ra, rb, data?.ratios ?? null);

  return (
    <div className={b.page}>
      <Head />
      {rehearsal ? (
        <span className={s.recordTag} data-guide="lab:rehearsal-tag" title={rehearsal.file}>
          {rehearsal.label}
        </span>
      ) : null}
      {hero ? <Hero label="Alone, or with a friend?" hero={hero} guide="lab:friend" /> : null}
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
          <Empty title="Nothing to compare yet">{why ?? "No run alone and no run with a friend is recorded."} This page looks again every 3 seconds.</Empty>
        </>
      ) : (
        <>
          {ra ? <Traces ra={ra} rb={rb} /> : rb ? <Traces ra={rb} rb={null} /> : null}
          {!complete ? <p className={b.note}>Waiting for the other run; looking again every 3 seconds.</p> : null}
          <Fold summary="Every number, stage by stage" guide="lab:compare-detail">
            {choices.length > 1 ? (
              <div className={c.pickers}>
                <Picker label="Alone" value={ra?.run_id ?? null} runs={choices} onPick={(id) => onPick("a", id)} />
                <Picker label="With a friend" value={rb?.run_id ?? null} runs={choices} onPick={(id) => onPick("b", id)} />
              </div>
            ) : null}
            <div className={c.acts}>
              <Act n={1} run={ra} title={ra ? runName(ra).replace(/ \(.*\)$/, "") : "Alone"} />
              <Act n={2} run={rb} title={rb ? runName(rb).replace(/ \(.*\)$/, "") : "With a friend"} />
            </div>
            {complete && data?.ratios ? <Ratios ratios={data.ratios} /> : null}
          </Fold>
          <Fold summary="Where these numbers come from" guide="lab:compare-sources">
            {rehearsal ? <span className={b.note}>This is the lab&apos;s own recording of the loop on its two pods, not a run of this box. This box has no comparison of its own yet; when it records one, that is shown instead.</span> : null}
            <span className={b.note}>
              Every joule here is everything the meter read while that stage ran. What a machine draws doing nothing (&ldquo;idle&rdquo;) is shown beside it and never taken off.
            </span>
            {[ra, rb].map((r) =>
              r ? (
                <span key={r.run_id} className={`${b.note} ${b.code}`}>
                  {r.run_id}: {r.source.ref}
                </span>
              ) : null,
            )}
            {(ra ?? rb)?.source.rule ? <span className={b.note}>The lab&apos;s rule, in its own words: {(ra ?? rb)!.source.rule}.</span> : null}
            {(ra ?? rb)?.source.split ? <span className={b.note}>{(ra ?? rb)!.source.split}.</span> : null}
          </Fold>
          <Words keys={["teacher", "student", "joules", "loss"]} />
        </>
      )}
    </div>
  );
}

function Head() {
  return (
    <div className={ui.pageHead}>
      <h1>Together</h1>
      <p className={b.sub}>The same task trained twice: alone, then with a friend&apos;s examples pooled in.</p>
    </div>
  );
}
