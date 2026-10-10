"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AnalysisLayer } from "@/components/spot/AnalysisLayer";
import { FloorMap } from "@/components/spot/FloorMap";
import o from "@/components/spot/op.module.css";
import { OperatorAsk } from "@/components/spot/OperatorAsk";
import { Round } from "@/components/spot/Round";
import { AllSheet, AssetSheet, DoFirst, LooksFine, StatusLine, TrustLine, WatchOut } from "@/components/spot/Shift";
import type { Replay, Target } from "@/components/spot/types";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { useFetch } from "@/lib/useFetch";

/**
 * Inspection (SPOT), for the person working the shift. Three layers:
 *
 *  1. "Your shift": one status line, three cards (do this first, watch out, looks fine) and the floor plan with one mark per
 *     piece of equipment. Everything is a sentence from GET /api/spot/shift; no statistics.
 *  2. "A round, replayed": one mission as a row of steps with SPOT on the plan, after the round or as it happened.
 *  3. "Analysis": everything the analyst screen showed, behind the Analysis tab and every "Why?".
 *
 * Only layers 1 and 2 load with the page (one request, plus the live picture). The analysis payloads are fetched when the
 * Analysis tab or a "Why?" is first opened.
 *
 * Deep links: ?asset=HX-202 opens that equipment's panel, ?round=M017&step=6 puts that round, at that step, at the top of the page, ?why=battery opens a
 * finding, ?tab=analysis opens the analysis, ?floor=2 picks the floor, ?all=1 lists all equipment.
 */
type Tab = "shift" | "analysis";

export default function SpotPage() {
  const shift = useFetch((sig) => api.spotShift(sig));
  const orders = useFetch((sig) => api.spotWorkOrders(sig));
  const [live, setLive] = useState<Replay | null>(null);
  const [tab, setTab] = useState<Tab>("shift");
  const [asset, setAsset] = useState<string | null>(null);
  const [all, setAll] = useState(false);
  const [target, setTarget] = useState<Target | null>(null);
  const [evidence, setEvidence] = useState(false); // becomes true the first time the analysis is needed, and stays
  const [link, setLink] = useState<{ round: string | null; step: number | null; floor: number | null } | null>(null);

  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const up = (k: string) => q.get(k)?.toUpperCase() ?? null;
    if (up("asset")) setAsset(up("asset"));
    if (q.get("why")) (setTarget({ kind: "finding", id: q.get("why")! }), setEvidence(true));
    if (q.get("tab") === "analysis") (setTab("analysis"), setEvidence(true));
    if (q.get("all")) setAll(true);
    setLink({ round: up("round"), step: q.get("step") ? Number(q.get("step")) : null, floor: q.get("floor") ? Number(q.get("floor")) : null });
  }, []);

  // The live picture: one reader, at most four reads a second.
  const reading = useRef(false);
  const again = useRef(false);
  const readLive = useCallback(() => {
    if (reading.current) {
      again.current = true;
      return;
    }
    reading.current = true;
    api
      .spotReplay()
      .then(setLive)
      .catch(() => undefined)
      .finally(() => {
        window.setTimeout(() => {
          reading.current = false;
          if (again.current) {
            again.current = false;
            readLive();
          }
        }, 250);
      });
  }, []);
  useEffect(readLive, [readLive]);
  useAtlasEvents(["spot."], (e) => (e.type === "spot.work_order.created" ? orders.reload() : readLive()));
  const playing = Boolean(live?.playing);
  useEffect(() => {
    if (!playing) return;
    const t = window.setInterval(readLive, 1000);
    return () => window.clearInterval(t);
  }, [playing, readLive]);

  const why = useCallback((t: Target | null) => {
    if (t) setEvidence(true);
    setTarget(t);
  }, []);
  const pick = (t: Tab) => {
    if (t === "analysis") setEvidence(true);
    setTab(t);
  };

  if (shift.error) return <ErrorBox what="the inspection data" error={shift.error} onRetry={shift.reload} />;
  if (!shift.data || link === null) return <Loading what="your shift" />;
  const d = shift.data;
  if (!d.configured || !d.map) {
    return (
      <div className={o.page}>
        <div className={ui.pageHead}>
          <h1>Inspection</h1>
        </div>
        <Empty title="No inspection data on this box yet">{d.note}</Empty>
      </div>
    );
  }
  const written = orders.data?.work_orders ?? [];
  const open = asset ? d.assets.find((a) => a.asset_id === asset) : undefined;
  const robotAt = playing ? live?.robot?.location_id ?? null : null;

  return (
    <div className={o.page}>
      <div className={o.top}>
        <StatusLine shift={d} />
        <div className={`${ui.segmented} ${o.tabs}`} role="tablist" aria-label="View">
          {(
            [
              ["shift", "Your shift"],
              ["analysis", "Analysis"],
            ] as [Tab, string][]
          ).map(([k, label]) => (
            <button type="button" key={k} role="tab" aria-selected={tab === k} className={`${ui.seg} ${tab === k ? ui.segOn : ""}`} onClick={() => pick(k)}>
              {label}
            </button>
          ))}
        </div>
      </div>

      {tab === "shift" ? (
        <>
          {d.trust ? <TrustLine trust={d.trust} onWhy={why} /> : null}

          <div className={o.layout}>
          <div className={o.first}>
            <div className={o.mapCol}>
              <FloorMap
                map={d.map}
                assets={d.assets}
                title="The plant floor"
                route={d.map.route}
                robotAt={robotAt}
                selected={asset}
                onPick={setAsset}
                initialFloor={link.floor}
                note={<p className={o.note}>Tap a mark to see what SPOT found there. The picture is the kit's sample floor plan; it is practice data, not a real plant.</p>}
              />
            </div>
            <div className={o.sideCol}>
              <DoFirst shift={d} orders={written} onAsset={setAsset} onWhy={why} onAll={() => setAll(true)} />
              <div className={o.onlyWide}>
                <WatchOut items={d.watch} onWhy={why} />
              </div>
            </div>
          </div>

          <div className={o.second}>
            <div className={o.onlyNarrow}>
              <WatchOut items={d.watch} onWhy={why} />
            </div>
            <LooksFine shift={d} onAll={() => setAll(true)} />
            <OperatorAsk chips={d.ask ?? []} onWhy={why} />
          </div>

          </div>

          <div id="round" className={link.round ? o.roundFirst : undefined}>
            <Round shift={d} live={live} onLive={setLive} onWhy={why} startRound={link.round} startStep={link.step} />
          </div>

          <button type="button" className={o.more} onClick={() => pick("analysis")}>
            <span>
              <b>The analysis behind this</b>
              How sure each sentence is, the tables, the method and what the data cannot say.
            </span>
            <span aria-hidden="true">›</span>
          </button>
        </>
      ) : null}

      {evidence ? <AnalysisLayer shift={d} show={tab === "analysis"} target={target} setTarget={why} live={live} onLive={setLive} orders={written} /> : null}

      {all ? <AllSheet shift={d} onClose={() => setAll(false)} onPick={(id) => (setAll(false), setAsset(id))} /> : null}
      {open ? <AssetSheet asset={open} orders={written} onClose={() => setAsset(null)} onWhy={why} /> : null}
    </div>
  );
}
