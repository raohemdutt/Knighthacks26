"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { dayTime } from "@/lib/format";
import { type LabStory, isLiveRun } from "@/lib/lab";
import { type LabCard, when } from "@/lib/labBench";
import { CARD_WORDS, type LoopStage, type Tone, loopStages, modelHero, plainCardTitle, runRow, runState } from "@/lib/labWords";
import { useTheme } from "@/lib/theme";
import { useFetch } from "@/lib/useFetch";
import { usePoll } from "@/lib/usePoll";
import { DeployButton } from "./DeployButton";
import { CardBody } from "./Experiments";
import { LabRun } from "./LabRun";
import lab from "./lab.module.css";
import type { LoopScene } from "./loopScene";
import { Hero, Words } from "./Story";
import s from "./story.module.css";

/**
 * The model: the Learning lab's first tab. One slide, top to bottom:
 *
 *   1. the hero: the newest real test, before and after, and what the test said (lib/labWords.ts modelHero);
 *   2. the loop in 3D (loopScene.ts), each stage lit from the shown run's real events with its own number beside it,
 *      and the same five stages as a strip of words under it. The run on show is the one training now, else the last;
 *   3. one timeline: every training run and every record of the lab's own work as a compact row. A tap opens the
 *      row's detail in place (a run: its loss, its test, the questions it fixed or broke, its settings and power).
 *
 * Data: GET /api/learning/lab, /api/learning/lab/{run_id}, /api/learning/lab/experiments. Only what the box recorded.
 *
 * Looking again. While the box has no run, and while one is training, the list and the shown run are asked for again
 * every 3 s (PR 399), and at once on a learning event. A request still in flight is never restarted: the page used to
 * abort and re-send it on every beat, so a box that took longer than 3 s to answer never got to answer at all and the
 * tab sat on "Loading the learning lab" (lib/usePoll.ts).
 */
const FIRST_ROWS = 7;

/* ---------- the loop ---------- */

const STATE_WORD: Record<LoopStage["state"], string> = { done: "done", now: "now", waiting: "not yet", failed: "failed", refused: "refused", none: "not reported" };

/** The 3D loop with the strip of stages under it. The strip is the picture in words, and all there is without WebGL. */
function LoopView({ stages, runKey, forceReduced }: { stages: LoopStage[]; runKey: string; forceReduced: boolean }) {
  const host = useRef<HTMLDivElement>(null);
  const overlay = useRef<HTMLDivElement>(null);
  const scene = useRef<LoopScene | null>(null);
  const shown = useRef<string | null>(null);
  const latest = useRef(stages);
  latest.current = stages;
  const [failed, setFailed] = useState(false);
  const [theme] = useTheme();

  // three.js and the scene arrive after the page has painted, and only this tab asks for them
  useEffect(() => {
    const h = host.current;
    const o = overlay.current;
    if (!h || !o) return;
    let gone = false;
    let watcher: IntersectionObserver | null = null;
    const calm = forceReduced || (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false);
    import("./loopScene")
      .then((m) => {
        if (gone) return;
        try {
          scene.current = new m.LoopScene(h, o, { calm, dark: document.documentElement.dataset.theme !== "light", tagClass: s.loopTag ?? "" });
        } catch {
          setFailed(true); // no WebGL: the strip below still tells the run
          return;
        }
        scene.current.setStages(latest.current, true);
        watcher = new IntersectionObserver((entries) => scene.current?.setActive(entries.some((e) => e.isIntersecting)), { threshold: 0 });
        watcher.observe(h);
      })
      .catch(() => setFailed(true));
    return () => {
      gone = true;
      watcher?.disconnect();
      scene.current?.dispose();
      scene.current = null;
    };
  }, [forceReduced]);

  // a different run lights its stages in order once; the same run, updated, just changes what is lit
  const sig = stages.map((x) => `${x.state}|${x.value ?? ""}`).join(",");
  useEffect(() => {
    scene.current?.setStages(latest.current, shown.current !== runKey);
    shown.current = runKey;
  }, [sig, runKey]);
  useEffect(() => scene.current?.setTheme(theme === "dark"), [theme]);

  return (
    <>
      {failed ? null : (
        <div className={s.loopStage} aria-hidden="true">
          <div ref={host} className={s.loopHost} />
          <div ref={overlay} className={s.loopHost} />
        </div>
      )}
      <ol className={s.strip} aria-label="The loop, stage by stage" data-guide="lab:stages">
        {stages.map((x) => (
          <li key={x.key} className={s.step} data-state={x.state}>
            <span className={s.stepTitle}>{x.title}</span>
            <span className={s.stepValue}>{x.value ?? STATE_WORD[x.state]}</span>
            {x.note ? <span className={s.stepNote}>{x.note}</span> : null}
            <span className={s.srOnly}>: {STATE_WORD[x.state]}</span>
          </li>
        ))}
      </ol>
    </>
  );
}

/* ---------- the timeline ---------- */

interface Row {
  key: string;
  ts: string | null;
  when: string;
  what: string;
  kind: string;
  chip: string;
  tone: Tone | "live";
  run?: string;
  card?: LabCard;
}

function RunDetail({ runId, forceReduced, initial }: { runId: string; forceReduced: boolean; initial: number | null }) {
  const story = useFetch((signal) => api.labRun(runId, false, signal), `run-${runId}`);
  usePoll([story], !!story.data && isLiveRun({ status: story.data.status, run_status: story.data.run_status, ended: story.data.marks.end }));
  if (story.error && !story.data) return <ErrorBox what="this training run" error={story.error} onRetry={story.reload} />;
  if (!story.data) return <Loading what="this training run" />;
  return <LabRun key={runId} story={story.data} initial={initial} forceReduced={forceReduced} />;
}

function Timeline({ rows, open, setOpen, forceReduced, initial }: { rows: Row[]; open: string | null; setOpen: (k: string | null) => void; forceReduced: boolean; initial: number | null }) {
  const [all, setAll] = useState(false);
  const shown = all ? rows : rows.filter((r, i) => i < FIRST_ROWS || r.key === open);
  return (
    <section className={ui.card} aria-labelledby="lab-timeline" data-guide="lab:timeline">
      <h2 id="lab-timeline">Everything so far</h2>
      <div className={s.rows}>
        {shown.map((r) => (
          <div key={r.key} className={s.rowWrap}>
            <button type="button" className={s.row} aria-expanded={open === r.key} onClick={() => setOpen(open === r.key ? null : r.key)} data-guide={r.run ? `lab:row:${r.run}` : undefined}>
              <span className={s.rowWhen}>{r.when}</span>
              <span className={s.rowWhat}>
                {r.what} <span>· {r.kind}</span>
              </span>
              <span className={s.rowChip} data-tone={r.tone}>
                {r.chip}
              </span>
              <span className={s.rowMark} aria-hidden="true">
                {open === r.key ? "–" : "+"}
              </span>
            </button>
            {open === r.key ? <div className={s.rowBody}>{r.run ? <RunDetail runId={r.run} forceReduced={forceReduced} initial={initial} /> : r.card ? <CardBody c={r.card} /> : null}</div> : null}
          </div>
        ))}
      </div>
      {rows.length > FIRST_ROWS ? (
        <button type="button" className={ui.btn} style={{ alignSelf: "flex-start" }} onClick={() => setAll((v) => !v)}>
          {all ? "Show fewer" : `Show all ${rows.length}`}
        </button>
      ) : null}
    </section>
  );
}

/* ---------- the tab ---------- */

export function Model({ initialRun, initialT, forceReduced }: { initialRun: string | null; initialT: number | null; forceReduced: boolean }) {
  const list = useFetch((signal) => api.lab(false, signal), "list");
  const timeline = useFetch((signal) => api.labExperiments(signal).catch(() => null), "experiments");
  const runs = useMemo(() => (list.data && !list.data.sample ? list.data.runs : []), [list.data]);
  const live = runs.find((r) => isLiveRun({ ...r, sample: false })) ?? null;
  const onShow = live ?? runs[0] ?? null; // the run the loop is drawn for: the one training now, else the last
  const story = useFetch((signal) => (onShow ? api.labRun(onShow.run_id, false, signal) : Promise.resolve(null)), `scene-${onShow?.run_id ?? ""}`);
  const shown: LabStory | null = story.data && story.data.run_id === onShow?.run_id ? story.data : null;
  const [open, setOpen] = useState<string | null>(initialRun ? `run:${initialRun}` : null);

  useAtlasEvents(["learning.", "telemetry.train.step"], () => {
    if (!list.loading) list.reload();
    if (!story.loading) story.reload();
  });
  usePoll([list, story], !!list.data && (runs.length === 0 || !!live));
  usePoll([list], !list.data && !!list.error); // the box could not be reached: try again, one request at a time

  const [now, setNow] = useState<number | null>(null);
  useEffect(() => {
    setNow(Date.now());
    const id = window.setInterval(() => setNow(Date.now()), 30_000);
    return () => window.clearInterval(id);
  }, []);

  const hero = useMemo(() => modelHero(runs, list.data?.box_now, dayTime), [runs, list.data?.box_now]);
  const stages = useMemo(() => loopStages(shown, dayTime), [shown]);
  const rows = useMemo<Row[]>(() => {
    const out: Row[] = runs.map((r) => ({ key: `run:${r.run_id}`, ts: r.started, when: dayTime(r.started) || "no time recorded", kind: "training run", run: r.run_id, ...runRow(r) }));
    for (const c of timeline.data?.cards ?? []) {
      const tone: Tone = c.status === "won" || c.status === "converged" ? "good" : c.status === "lost" ? "bad" : "plain";
      out.push({ key: `${c.kind}:${c.id}`, ts: c.ts, when: c.ts ? (c.day_only ? when(c.ts, true) : dayTime(c.ts)) : "planned", what: plainCardTitle(c), kind: c.kind === "bench" ? "speed test" : c.kind === "experiment" ? "experiment" : "review of the lab's own work", chip: CARD_WORDS[c.status] ?? c.status, tone, card: c });
    }
    // newest first; what has no time yet (planned experiments) comes last, in the lab's own order
    return out.sort((a, b) => (a.ts && b.ts ? b.ts.localeCompare(a.ts) : a.ts ? -1 : b.ts ? 1 : 0));
  }, [runs, timeline.data]);

  if (list.loading && !list.data) return <Loading what="the learning lab" />;
  if (!list.data)
    return (
      <>
        <div className={ui.pageHead}>
          <h1>The model</h1>
        </div>
        <Empty title="The box is not answering">The learning lab could not be reached{list.error ? ` (${list.error.message})` : ""}. This page tries again every 3 seconds.</Empty>
      </>
    );

  const state = shown ? runState(shown, now, dayTime) : null;
  return (
    <div className={lab.page}>
      <div className={ui.pageHead}>
        <h1>The model</h1>
      </div>
      {runs.length === 0 ? (
        <>
          {/* the honest empty state (PR 415): nothing recorded, no sample run, and the page keeps looking */}
          <Empty title="No runs yet; start the loop">
            Atlas runs {list.data.current_model?.version ?? "its model"} on the box and has not trained anything yet. A run appears here on its own as soon as one starts.
          </Empty>
          <Link href="/teach" className={`${ui.btn} ${ui.btnPrimary} ${lab.cta}`}>
            Teach Atlas something
          </Link>
        </>
      ) : hero ? (
        <Hero label="Is Atlas getting better?" hero={hero.hero} guide="lab:better" />
      ) : null}

      <section className={s.loop} aria-label="The learning loop" data-guide="lab:loop">
        <LoopView stages={stages} runKey={onShow?.run_id ?? "none"} forceReduced={forceReduced} />
        <p className={s.loopLine} role="status" data-guide="lab:now">
          {runs.length === 0 ? (
            "Nothing has run yet. This is the path a run takes: out to the lab, through the test, and back."
          ) : live && state ? (
            <b>{state.line}</b>
          ) : (
            <>
              Nothing is training right now. <b>Last run: {dayTime(onShow?.started) || "no time recorded"}</b>
              {onShow ? `, ${runRow(onShow).chip.toLowerCase()}.` : "."}
            </>
          )}
        </p>
      </section>

      {/* The last step of the story, and the one place the box itself is changed from: Ilan's swap buttons (L35),
          exactly as built. What they call, when they are enabled and what they confirm are DeployButton.tsx's. */}
      <section className={ui.card} aria-labelledby="lab-on-box" data-guide="lab:deploy">
        <h2 id="lab-on-box">On the box</h2>
        <DeployButton
          onBoxChanged={() => {
            if (!list.loading) list.reload();
            if (!story.loading) story.reload();
          }}
        />
      </section>

      {rows.length ? <Timeline rows={rows} open={open} setOpen={setOpen} forceReduced={forceReduced} initial={initialT} /> : null}
      <Words keys={["gate", "heldout", "loss", "lora", "noise", "pvalue", "teacher", "student"]} />
    </div>
  );
}
