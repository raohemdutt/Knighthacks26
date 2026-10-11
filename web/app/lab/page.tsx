"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import bench from "@/components/lab/bench.module.css";
import { Compare } from "@/components/lab/Compare";
import { Experiments } from "@/components/lab/Experiments";
import { LabRun } from "@/components/lab/LabRun";
import lab from "@/components/lab/lab.module.css";
import loop from "@/components/lab/loop.module.css";
import { LoopView } from "@/components/lab/LoopView";
import { Routing } from "@/components/lab/Routing";
import { SpeedEnergy } from "@/components/lab/SpeedEnergy";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { type BoxNow, clamp01, clock, defaultRunId, hasRealGate, isLive, isLoopRun, modelShort, runStatus, storyIsStandIn } from "@/lib/lab";
import { LAB_TABS, type LabTab } from "@/lib/labBench";
import { LAB_GROUPS, type LabGroup, labGroupOf, labTabParam, labViewOf } from "@/lib/labTabs";
import { useFetch } from "@/lib/useFetch";
import { usePoll } from "@/lib/usePoll";

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
 *
 * Three tabs, as a grouping only (lib/labTabs.ts). Each of the five views above is shown whole and unchanged; the strip
 * at the top holds three tabs, and a tab that holds two views shows one after the other:
 *
 *   The model          the loop in 3D (components/lab/LoopView.tsx, lit from the shown run's own events), then the
 *                      Training run view (#run), one stage card at a time (?stage=1..5). The Experiments view
 *                      (#experiments) is not on the tab's bar; /lab/?tab=experiments opens it alone.
 *   Speed and energy   the Speed and energy view (#speed), then the Routing view (#routing)
 *   Together           the Alone vs with a friend view (#compare), with its own empty state when the box has no pair
 *
 * Every ?tab= value above still works: it opens the tab that holds the view and scrolls to the view. ?tab=model and
 * ?tab=together name the new tabs. Plain /lab/ opens The model at its top.
 *
 * Looking again every 3 s never restarts a request that is still on its way (lib/usePoll.ts): a box slower than the
 * beat used to be cut off and asked again on every beat, and never got to answer.
 */
const LIVE_POLL_MS = 3000;

export default function LabPage() {
  const [params, setParams] = useState<{ run: string | null; t: number | null; reduce: boolean; q: URLSearchParams } | null>(null);
  const [picked, setPicked] = useState<string | null>(null);
  const [tab, setTab] = useState<LabTab>("run");
  const [compareIds, setCompareIds] = useState<{ a: string | null; b: string | null }>({ a: null, b: null });
  const [jump, setJump] = useState<{ view: LabTab; n: number } | null>(null);
  const group: LabGroup = labGroupOf(tab);

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const t = q.get("t");
    setParams({ run: q.get("run"), t: t !== null && !Number.isNaN(Number(t)) ? clamp01(Number(t)) : null, reduce: q.get("motion") === "reduce", q });
    setCompareIds({ a: q.get("a"), b: q.get("b") });
    const asked = q.get("tab");
    if (LAB_TABS.some((x) => x.key === asked)) setTab(asked as LabTab);
    // the two new tab names, and an anchor (/lab/#routing), ask for a view the same way
    const named = labViewOf(asked) ?? labViewOf(window.location.hash.slice(1));
    if (named) {
      setTab(named);
      setJump({ view: named, n: 0 });
    }
  }, []);

  // A view that was asked for by name is scrolled to, and kept in place while what is above it loads (it grows as its
  // data arrives), until the reader scrolls or presses something. A view at the top of its tab needs no scroll.
  useEffect(() => {
    if (!params || !jump || LAB_GROUPS.some((g) => g.views[0] === jump.view && g.key !== "model")) return;
    const el = document.getElementById(jump.view);
    if (!el) return;
    const go = () => el.scrollIntoView({ block: "start" });
    go();
    const watcher = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(go);
    watcher?.observe(el.closest("main") ?? document.body);
    const names = ["wheel", "touchstart", "keydown", "pointerdown"];
    let timer = 0;
    const stop = () => {
      watcher?.disconnect();
      for (const name of names) window.removeEventListener(name, stop);
      window.clearTimeout(timer);
    };
    timer = window.setTimeout(stop, 30_000);
    for (const name of names) window.addEventListener(name, stop, { passive: true });
    return stop;
  }, [params, jump]);

  const list = useFetch((signal) => (params ? api.lab(false, signal) : new Promise<never>(() => {})), params ? "list" : "wait");
  // A list the gateway filled from its committed sample is no run at all: the screen shows only what the box recorded.
  const runs = list.data && !list.data.sample ? list.data.runs : [];
  // The tab opens on the newest run with a real gate result; the loop's runs (a stand-in gate) stay one press away.
  const runId = picked ?? (params?.run && runs.some((r) => r.run_id === params.run) ? params.run : defaultRunId(runs));
  // What the box serves now, from the deploy record (the run list's own word as a fallback): "On the box" means this.
  const deployView = useFetch((signal) => (params ? api.labDeploy(signal) : new Promise<never>(() => {})), params ? "deploy" : "wait");
  const dv = deployView.data;
  const box: BoxNow = {
    current: dv?.current ?? list.data?.current_model?.version ?? null,
    swapRolledBack: dv && dv.deploy.state === "failed" && dv.deploy.result?.rolled_back_to ? dv.deploy.version : null,
  };
  const realRun = [...runs].filter(hasRealGate).sort((a, b) => Date.parse(b.started ?? "") - Date.parse(a.started ?? ""))[0] ?? null;
  const pickRun = (id: string) => {
    setPicked(id);
    const url = new URL(window.location.href);
    url.searchParams.set("run", id);
    window.history.replaceState(null, "", url);
  };
  const story = useFetch((signal) => (runId ? api.labRun(runId, false, signal) : new Promise<never>(() => {})), runId ? `run-${runId}` : "none");
  const shown = story.data && !story.data.sample ? story.data : null;
  // a request still on its way is left to finish, never restarted (lib/usePoll.ts)
  useAtlasEvents(["learning.", "telemetry.train.step"], () => {
    if (!list.loading) list.reload();
    if (!story.loading) story.reload();
    if (!deployView.loading) deployView.reload();
  });
  // A run read from the trainer's own files sends no event: until the box has a run, and while one is still going
  // (training, or cleared by the gate with packaging and the swap to come), look again every 3 s so it arrives
  // without a reload. A gateway that cannot be reached is looked at again on the same beat.
  const watching = group === "model" && (runs.length === 0 || (!!shown && isLive(shown)));
  usePoll([list, story], watching, LIVE_POLL_MS);

  const pickTab = (key: LabTab, scroll = false) => {
    setTab(key);
    setJump(scroll ? { view: key, n: Date.now() } : null);
    const url = new URL(window.location.href);
    const value = labTabParam(key);
    if (value === null) url.searchParams.delete("tab");
    else url.searchParams.set("tab", value);
    url.hash = "";
    window.history.replaceState(null, "", url);
  };
  // Experiments is not on the model tab's bar: it opens only from its own link (/lab/?tab=experiments), so with one view
  // left the model tab has no sub-tab bar and the Training run view sits right under the loop.
  const views = (LAB_GROUPS.find((g) => g.key === group)?.views ?? []).filter((v) => !(group === "model" && v === "experiments"));
  // The lab's own name for a view, kept in sight above it, so "the Training run tab" in a run sheet is still on screen.
  // Speed and energy and Alone vs with a friend open on those very words as their own heading, so they need no label.
  const named = (v: LabTab) => <p className={loop.viewName}>{LAB_TABS.find((x) => x.key === v)?.title ?? v}</p>;
  const tabs = (
    <>
      <div className={bench.tabs} role="tablist" aria-label="Learning lab">
        {LAB_GROUPS.map((x) => (
          <button key={x.key} type="button" role="tab" aria-selected={group === x.key} className={`${bench.tab} ${group === x.key ? bench.tabOn : ""}`} onClick={() => pickTab(x.views[0] ?? "run")}>
            {x.title}
          </button>
        ))}
      </div>
      {views.length > 1 ? (
        <nav className={ui.segmented} aria-label="On this tab" data-guide="lab:views">
          {views.map((v) => (
            <button key={v} type="button" className={`${ui.seg} ${tab === v ? ui.segOn : ""}`} aria-pressed={tab === v} onClick={() => pickTab(v, group !== "model")}>
              {LAB_TABS.find((x) => x.key === v)?.title ?? v}
            </button>
          ))}
        </nav>
      ) : null}
    </>
  );

  if (!params) return <Loading what="the learning lab" />;
  const q = params.q;
  const asked = labViewOf(q.get("tab")) ?? labViewOf(window.location.hash.slice(1));
  const stageParam = q.get("stage") !== null && /^[1-5]$/.test(q.get("stage") ?? "") ? Number(q.get("stage")) : null;
  const number = (key: string) => (q.get(key) !== null && !Number.isNaN(Number(q.get(key))) ? Number(q.get(key)) : null);
  if (group === "speed")
    return (
      <>
        {tabs}
        <section id="speed" className={loop.view} aria-label="Speed and energy" data-guide="lab:speed">
          <SpeedEnergy initialBatch={number("batch")} initialTtft={q.get("ttft") === "1"} initialPoint={q.get("point")} initialDrawer={asked === "routing" ? null : q.get("drawer")} initialCompare={q.get("compare")} initialSweep={q.get("sweep")} forceReduced={params.reduce} />
        </section>
        <section id="routing" className={loop.view} aria-label="Routing" data-guide="lab:routing">
          {named("routing")}
          <Routing initialPreset={q.get("preset")} initialT={params.t} initialPolicy={q.get("policy")} initialBasis={q.get("basis")} initialDrawer={asked === "routing" ? q.get("drawer") : null} forceReduced={params.reduce} />
        </section>
      </>
    );
  if (group === "together")
    return (
      <>
        {tabs}
        <section id="compare" className={loop.view} aria-label="Alone vs with a friend" data-guide="lab:compare">
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
        </section>
      </>
    );

  // The model: the loop and the Training run view, or the Experiments view alone when its sub-tab is picked.
  const shownRow = runs.find((r) => r.run_id === runId) ?? null;
  const model = (run: React.ReactNode) =>
    tab === "experiments" ? (
      <>
        {tabs}
        <section id="experiments" className={loop.view} aria-label="Experiments" data-guide="lab:experiments">
          <button type="button" className={ui.btn} onClick={() => pickTab("run")}>
            Back to the model
          </button>
          <Experiments />
        </section>
      </>
    ) : (
    <>
      {tabs}
      <LoopView story={shown} run={shownRow} label={shownRow ? runStatus(shownRow, box).label : null} standIn={!!shown && storyIsStandIn(shown)} onBox={box.current} empty={!!list.data && runs.length === 0} forceReduced={params.reduce} />
      <section id="run" className={loop.view} aria-label="Training run" data-guide="lab:run">
        {named("run")}
        {run}
      </section>
    </>
  );

  if (list.loading && !list.data)
    return model(<Loading what="the learning lab" />);
  if (runs.length === 0) {
    return model(
      <>
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
      </>,
    );
  }

  const picker =
    runs.length > 1 ? (
      <div className={ui.segmented} role="group" aria-label="Which run">
        {runs.map((r) => (
          <button key={r.run_id} type="button" className={`${ui.seg} ${r.run_id === runId ? ui.segOn : ""}`} aria-pressed={r.run_id === runId} onClick={() => pickRun(r.run_id)} title={r.model_version ?? r.run_id}>
            {isLoopRun(r) ? runStatus(r, box).label : `${clock(r.started)} · ${runStatus(r, box).label}`}
          </button>
        ))}
      </div>
    ) : null;

  return model(
    <>
      {story.error && !story.data ? (
        <ErrorBox what="this training run" error={story.error} onRetry={story.reload} />
      ) : !shown ? (
        <Loading what="this training run" />
      ) : (
        <LabRun key={shown.run_id}
          story={shown}
          initial={params.t}
          forceReduced={params.reduce}
          picker={picker}
          onBoxChanged={() => {
            story.reload();
            deployView.reload();
          }}
          status={shownRow ? runStatus(shownRow, box) : null}
          boxNow={box.current}
          realRun={realRun && realRun.run_id !== shown.run_id ? { id: realRun.run_id, label: `the ${clock(realRun.started)} run${realRun.model_version ? ` (${modelShort(realRun.model_version)})` : ""}` } : null}
          onPickRun={pickRun}
          initialStage={stageParam}
        />
      )}
    </>,
  );
}
