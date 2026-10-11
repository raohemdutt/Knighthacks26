"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import bench from "@/components/lab/bench.module.css";
import { Compare } from "@/components/lab/Compare";
import { Experiments } from "@/components/lab/Experiments";
import { LabRun } from "@/components/lab/LabRun";
import lab from "@/components/lab/lab.module.css";
import { Routing } from "@/components/lab/Routing";
import { SpeedEnergy } from "@/components/lab/SpeedEnergy";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { STATUS_LABEL, clamp01, clock, isLive } from "@/lib/lab";
import { LAB_TABS, type LabTab } from "@/lib/labBench";
import { useFetch } from "@/lib/useFetch";

/**
 * Learning lab: one training run, replayed. What Atlas was taught from, how training went (loss and power on
 * one timeline), what changed on questions it never trained on, why the gate let the model through or did
 * not, and what runs on the box now.
 *
 * Data: GET /api/learning/lab (the runs) and /api/learning/lab/{run_id} (one run's story), both rebuilt from
 * the event log. Only what the box has recorded is shown: with no real run yet the tab says so ("No runs yet; start
 * the loop") and looks again every 3 s. The gateway's committed sample run is never asked for, and if it is served
 * anyway (a box with no run of its own) the screen treats it as no run at all.
 *
 *   /lab/?run=<run_id>    open one run
 *   /lab/?t=0.4           start at a position on the bar (0 to 1); without it the finished run is shown
 *   /lab/?motion=reduce   as if the system asked for reduced motion
 *
 * More tabs sit beside the run, each on the lab's own records: "Alone vs with a friend" (two runs compared, L31),
 * "Speed and energy" (the benchmark rows), "Routing" (a what-if that runs the real routing hook) and "Experiments"
 * (the optimisation loop's timeline).
 *
 *   /lab/?tab=compare|speed|routing|experiments     open a tab ("run" is the default)
 *   /lab/?tab=compare&a=<run_id>&b=<run_id> see components/lab/Compare.tsx
 *   /lab/?tab=speed&batch=16&ttft=1         see components/lab/SpeedEnergy.tsx
 *   /lab/?tab=routing&preset=burst&t=0.5    see components/lab/Routing.tsx
 */
const LIVE_POLL_MS = 3000;

export default function LabPage() {
  const [params, setParams] = useState<{ run: string | null; t: number | null; reduce: boolean; q: URLSearchParams } | null>(null);
  const [picked, setPicked] = useState<string | null>(null);
  const [tab, setTab] = useState<LabTab>("run");
  const [compareIds, setCompareIds] = useState<{ a: string | null; b: string | null }>({ a: null, b: null });

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const t = q.get("t");
    setParams({ run: q.get("run"), t: t !== null && !Number.isNaN(Number(t)) ? clamp01(Number(t)) : null, reduce: q.get("motion") === "reduce", q });
    setCompareIds({ a: q.get("a"), b: q.get("b") });
    const asked = q.get("tab");
    if (LAB_TABS.some((x) => x.key === asked)) setTab(asked as LabTab);
  }, []);

  const list = useFetch((signal) => (params ? api.lab(false, signal) : new Promise<never>(() => {})), params ? "list" : "wait");
  // A list the gateway filled from its committed sample is no run at all: the screen shows only what the box recorded.
  const runs = list.data && !list.data.sample ? list.data.runs : [];
  const runId = picked ?? (params?.run && runs.some((r) => r.run_id === params.run) ? params.run : (runs[0]?.run_id ?? null));
  const story = useFetch((signal) => (runId ? api.labRun(runId, false, signal) : new Promise<never>(() => {})), runId ? `run-${runId}` : "none");
  const shown = story.data && !story.data.sample ? story.data : null;
  const reloadList = list.reload;
  const reloadStory = story.reload;
  useAtlasEvents(["learning.", "telemetry.train.step"], () => {
    reloadList();
    reloadStory();
  });
  // A run read from the trainer's own files sends no event: until the box has a run, and while one is still going
  // (training, or cleared by the gate with packaging and the swap to come), look again every 3 s so it arrives
  // without a reload. A gateway that cannot be reached is looked at again on the same beat.
  const watching = tab === "run" && (runs.length === 0 || (!!shown && isLive(shown)));
  useEffect(() => {
    if (!watching) return;
    const id = window.setInterval(() => {
      reloadList();
      reloadStory();
    }, LIVE_POLL_MS);
    return () => window.clearInterval(id);
  }, [watching, reloadList, reloadStory]);

  const pickTab = (key: LabTab) => {
    setTab(key);
    const url = new URL(window.location.href);
    if (key === "run") url.searchParams.delete("tab");
    else url.searchParams.set("tab", key);
    window.history.replaceState(null, "", url);
  };
  const tabs = (
    <div className={bench.tabs} role="tablist" aria-label="Learning lab">
      {LAB_TABS.map((x) => (
        <button key={x.key} type="button" role="tab" aria-selected={tab === x.key} className={`${bench.tab} ${tab === x.key ? bench.tabOn : ""}`} onClick={() => pickTab(x.key)}>
          {x.title}
        </button>
      ))}
    </div>
  );

  if (!params) return <Loading what="the learning lab" />;
  const q = params.q;
  const number = (key: string) => (q.get(key) !== null && !Number.isNaN(Number(q.get(key))) ? Number(q.get(key)) : null);
  if (tab === "speed")
    return (
      <>
        {tabs}
        <SpeedEnergy initialBatch={number("batch")} initialTtft={q.get("ttft") === "1"} initialPoint={q.get("point")} initialDrawer={q.get("drawer")} initialCompare={q.get("compare")} initialSweep={q.get("sweep")} forceReduced={params.reduce} />
      </>
    );
  if (tab === "routing")
    return (
      <>
        {tabs}
        <Routing initialPreset={q.get("preset")} initialT={params.t} initialPolicy={q.get("policy")} initialBasis={q.get("basis")} initialDrawer={q.get("drawer")} forceReduced={params.reduce} />
      </>
    );
  if (tab === "compare")
    return (
      <>
        {tabs}
        <Compare
          a={compareIds.a}
          b={compareIds.b}
          onPick={(which, id) => {
            setCompareIds((ids) => ({ ...ids, [which]: id }));
            const url = new URL(window.location.href);
            url.searchParams.set(which, id);
            window.history.replaceState(null, "", url);
          }}
          forceReduced={params.reduce}
        />
      </>
    );
  if (tab === "experiments")
    return (
      <>
        {tabs}
        <Experiments />
      </>
    );

  if (list.loading && !list.data)
    return (
      <>
        {tabs}
        <Loading what="the learning lab" />
      </>
    );
  if (runs.length === 0) {
    return (
      <>
        {tabs}
        <div className={ui.pageHead}>
          <h1>What Atlas learned</h1>
          <p className={ui.muted}>
            Atlas runs {list.data?.current_model?.version ?? "its model"} on the box. It has not trained anything yet.
          </p>
        </div>
        <Empty title="No runs yet; start the loop">
          This tab shows only what the box has recorded. A run appears here on its own, step by step, as soon as one starts; the page looks again every 3
          seconds.
        </Empty>
        {list.error && !list.data ? <p className={`${ui.muted} ${ui.small}`}>Could not reach the learning lab ({list.error.message}). Looking again every 3 seconds.</p> : null}
        <div className={ui.card}>
          <h2>How a run starts</h2>
          <p className={lab.flow}>
            <span>Teach a task</span>→<span>Approve the export</span>→<span>The lab trains</span>→<span>The gate tests it</span>→<span>The box swaps models</span>
          </p>
          <p className={ui.muted}>
            Nothing leaves the box until you approve it. A model is installed only when it beats the old one on questions it never trained on, by more than chance.
          </p>
          <Link href="/teach" className={`${ui.btn} ${ui.btnPrimary} ${lab.cta}`}>
            Teach Atlas something
          </Link>
        </div>
      </>
    );
  }

  const picker =
    runs.length > 1 ? (
      <div className={ui.segmented} role="group" aria-label="Which run">
        {runs.map((r) => (
          <button key={r.run_id} type="button" className={`${ui.seg} ${r.run_id === runId ? ui.segOn : ""}`} aria-pressed={r.run_id === runId} onClick={() => setPicked(r.run_id)}>
            {clock(r.started)} · {STATUS_LABEL[r.status]}
          </button>
        ))}
      </div>
    ) : null;

  return (
    <>
      {tabs}
      {story.error && !story.data ? (
        <ErrorBox what="this training run" error={story.error} onRetry={story.reload} />
      ) : !shown ? (
        <Loading what="this training run" />
      ) : (
        <LabRun key={shown.run_id} story={shown} initial={params.t} forceReduced={params.reduce} picker={picker} onBoxChanged={story.reload} />
      )}
    </>
  );
}
