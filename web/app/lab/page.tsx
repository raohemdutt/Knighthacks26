"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { LabRun } from "@/components/lab/LabRun";
import lab from "@/components/lab/lab.module.css";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { STATUS_LABEL, clamp01, clock } from "@/lib/lab";
import { useFetch } from "@/lib/useFetch";

/**
 * Learning lab: one training run, replayed. What Atlas was taught from, how training went (loss and power on
 * one timeline), what changed on questions it never trained on, why the gate let the model through or did
 * not, and what runs on the box now.
 *
 * Data: GET /api/learning/lab (the runs) and /api/learning/lab/{run_id} (one run's story), both rebuilt from
 * the event log. With no real run the box serves the committed SAMPLE run and the screen says so on a banner
 * that never scrolls away.
 *
 *   /lab/?sample=1        show the sample run even when real runs exist
 *   /lab/?run=<run_id>    open one run
 *   /lab/?t=0.4           start at a position on the bar (0 to 1); without it the finished run is shown
 *   /lab/?motion=reduce   as if the system asked for reduced motion
 */
export default function LabPage() {
  const [params, setParams] = useState<{ sample: boolean; run: string | null; t: number | null; reduce: boolean } | null>(null);
  const [picked, setPicked] = useState<string | null>(null);

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const t = q.get("t");
    setParams({ sample: q.get("sample") === "1", run: q.get("run"), t: t !== null && !Number.isNaN(Number(t)) ? clamp01(Number(t)) : null, reduce: q.get("motion") === "reduce" });
  }, []);

  const sample = params?.sample ?? false;
  const list = useFetch((signal) => (params ? api.lab(sample, signal) : new Promise<never>(() => {})), params ? `list-${sample}` : "wait");
  const runs = list.data?.runs ?? [];
  const runId = picked ?? (params?.run && runs.some((r) => r.run_id === params.run) ? params.run : (runs[0]?.run_id ?? null));
  const story = useFetch((signal) => (runId ? api.labRun(runId, sample, signal) : new Promise<never>(() => {})), runId ? `run-${runId}-${sample}` : "none");
  const reloadList = list.reload;
  const reloadStory = story.reload;
  useAtlasEvents(["learning.", "telemetry.train.step"], () => {
    reloadList();
    reloadStory();
  });

  if (!params || (list.loading && !list.data)) return <Loading what="the learning lab" />;
  if (list.error && !list.data) return <ErrorBox what="the learning lab" error={list.error} onRetry={list.reload} />;

  if (runs.length === 0) {
    return (
      <>
        <div className={ui.pageHead}>
          <h1>What Atlas learned</h1>
          <p className={ui.muted}>
            Atlas runs {list.data?.current_model?.version ?? "its model"} on the box. It has not trained anything yet.
          </p>
        </div>
        <Empty title="No training run yet">A run shows up here, step by step, as soon as one starts.</Empty>
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

  if (story.error && !story.data) return <ErrorBox what="this training run" error={story.error} onRetry={story.reload} />;
  if (!story.data) return <Loading what="this training run" />;
  return <LabRun key={story.data.run_id} story={story.data} initial={params.t} forceReduced={params.reduce} picker={picker} />;
}
