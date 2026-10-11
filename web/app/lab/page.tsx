"use client";

import { useEffect, useState } from "react";
import bench from "@/components/lab/bench.module.css";
import { Compare, loadTogether } from "@/components/lab/Compare";
import { Model } from "@/components/lab/Model";
import { SpeedEnergy } from "@/components/lab/SpeedEnergy";
import { Loading } from "@/components/States";
import { clamp01 } from "@/lib/lab";
import { LAB_TABS, type LabTab, labTabOf } from "@/lib/labBench";
import { useFetch } from "@/lib/useFetch";

/**
 * Learning lab: three tabs, each one slide (a hero statement, two or three tiles, the picture, detail on a tap).
 *
 *   The model          is it getting better; the loop in 3D lit from the run's real events; every run and every
 *                      record of the lab's work in one timeline            (components/lab/Model.tsx)
 *   Speed and energy   what an answer costs, the batching charts, and routing as one card   (SpeedEnergy.tsx)
 *   Together           alone against with a friend: the box's own comparison, else the lab's recorded rehearsal
 *                      under its label. Not shown at all when there is neither                (Compare.tsx)
 *
 * Only what the box has recorded is shown; the one exception is that rehearsal, which says what it is.
 *
 *   /lab/?run=<run_id>          open one run's row in the timeline
 *   /lab/?t=0.4                 that run's replay starts at a position on its bar (0 to 1)
 *   /lab/?motion=reduce         as if the system asked for reduced motion
 *   /lab/?tab=speed|together    open a tab ("model" is the default). The links of the five tabs this screen used to
 *                               have still land: run and experiments on the model, routing on speed (with its
 *                               working shown), compare on together.
 *   /lab/?tab=speed&batch=16&ttft=1&compare=1&sweep=<chip>&preset=burst    see SpeedEnergy.tsx
 *   /lab/?tab=together&a=<run_id>&b=<run_id>                               see Compare.tsx
 */
export default function LabPage() {
  const [params, setParams] = useState<{ run: string | null; t: number | null; reduce: boolean; q: URLSearchParams; asked: string | null } | null>(null);
  const [tab, setTab] = useState<LabTab>("run");
  const [compareIds, setCompareIds] = useState<{ a: string | null; b: string | null }>({ a: null, b: null });

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const t = q.get("t");
    setParams({ run: q.get("run"), t: t !== null && !Number.isNaN(Number(t)) ? clamp01(Number(t)) : null, reduce: q.get("motion") === "reduce", q, asked: q.get("tab") });
    setCompareIds({ a: q.get("a"), b: q.get("b") });
    setTab(labTabOf(q.get("tab")));
  }, []);

  // Together is a tab only when there is something to put on it: the box's own comparison, or the lab's rehearsal.
  const together = useFetch((signal) => (params ? loadTogether(null, null, signal) : new Promise<never>(() => {})), params ? "together" : "wait");
  const hasTogether = !!(together.data?.data?.a || together.data?.data?.b);
  const tabs = LAB_TABS.filter((x) => x.key !== "compare" || hasTogether);
  const shownTab: LabTab = tab === "compare" && !hasTogether && !together.loading ? "run" : tab;

  const pickTab = (key: LabTab) => {
    setTab(key);
    const url = new URL(window.location.href);
    if (key === "run") url.searchParams.delete("tab");
    else url.searchParams.set("tab", key === "compare" ? "together" : key);
    window.history.replaceState(null, "", url);
  };

  if (!params) return <Loading what="the learning lab" />;
  const q = params.q;
  const number = (key: string) => (q.get(key) !== null && !Number.isNaN(Number(q.get(key))) ? Number(q.get(key)) : null);
  return (
    <>
      <div className={bench.tabs} role="tablist" aria-label="Learning lab">
        {tabs.map((x) => (
          <button key={x.key} type="button" role="tab" aria-selected={shownTab === x.key} className={`${bench.tab} ${shownTab === x.key ? bench.tabOn : ""}`} onClick={() => pickTab(x.key)}>
            {x.title}
          </button>
        ))}
      </div>
      {shownTab === "speed" ? (
        <SpeedEnergy
          initialBatch={number("batch")}
          initialTtft={q.get("ttft") === "1"}
          initialPoint={q.get("point")}
          initialDrawer={q.get("drawer")}
          initialCompare={q.get("compare")}
          initialSweep={q.get("sweep")}
          initialPreset={q.get("preset")}
          initialRouting={params.asked === "routing"}
          forceReduced={params.reduce}
        />
      ) : shownTab === "compare" ? (
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
      ) : (
        <Model initialRun={params.run} initialT={params.t} forceReduced={params.reduce} />
      )}
    </>
  );
}
