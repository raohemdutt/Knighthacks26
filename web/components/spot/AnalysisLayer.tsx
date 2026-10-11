"use client";

import { useMemo, useState } from "react";
import { EvidenceLine, OpenCtx, pct, range, type Opener } from "@/components/spot/bits";
import { Loading } from "@/components/States";
import { api } from "@/lib/api";
import { useFetch } from "@/lib/useFetch";
import { AskBox, HandoffCard } from "./AskBox";
import { Detail } from "./Detail";
import { BatteryCard, ModelCard, OverTimeCard, RateCard, VersionCard } from "./EvidencePanels";
import { ExplainSheet, ExplainerGrid, type ExplainId, type Force } from "./Explainers";
import { FacilityMap } from "./FacilityMap";
import { LevelIcon } from "./FloorMap";
import o from "./op.module.css";
import { MissionTimeline, ReplayBar } from "./ReplayPanel";
import { Why } from "./Shift";
import s from "./spot.module.css";
import { TriageQueue, UninspectedRisk } from "./Triage";
import type { Replay, Shift, Target, WorkOrder } from "./types";

/**
 * Layer 3, "The evidence": everything the analyst screen showed (findings with intervals and p-values, V1 against V2 with
 * the battery caveat, the tables by type, waypoint and asset, the trained scorer, assumptions, data check, sources).
 *
 * It is mounted only when someone opens the Analysis tab or taps a "Why?", so the first screen never waits for these
 * payloads. `show` renders the whole section; `target` opens one evidence panel over whatever is on screen.
 */
export function AnalysisLayer({
  shift,
  show,
  target,
  setTarget,
  live,
  onLive,
  orders,
  explain,
  setExplain,
  force,
}: {
  shift: Shift;
  show: boolean;
  target: Target | null;
  setTarget: (t: Target | null) => void;
  live: Replay | null;
  onLive: (r: Replay) => void;
  orders: WorkOrder[];
  /** The animated explainer a "Why?" asked for, shown over the page. */
  explain: ExplainId | null;
  setExplain: (id: ExplainId | null) => void;
  /** From a link: draw the explainers at their last (or a middle) step instead of playing them. */
  force: Force;
}) {
  const overview = useFetch((sig) => api.spotOverview(sig));
  const findings = useFetch((sig) => api.spotFindings(sig));
  const locations = useFetch((sig) => api.spotLocations(sig));
  const [tables, setTables] = useState(false);
  const wanted = show && tables;
  const heavy = wanted ? "on" : "off"; // the tables load only when their fold is opened
  const triage = useFetch((sig) => (wanted ? api.spotTriage(sig) : Promise.resolve(null)), heavy);
  const assets = useFetch((sig) => (wanted ? api.spotAssets(sig) : Promise.resolve(null)), heavy);
  const scorer = useFetch((sig) => (wanted ? api.spotModel(sig) : Promise.resolve(null)), heavy);
  const handoff = useFetch((sig) => (wanted ? api.spotHandoff(sig) : Promise.resolve(null)), heavy);
  const latestId = shift.latest_round ?? "";
  const latest = useFetch((sig) => (wanted && latestId ? api.spotMission(latestId, sig) : Promise.resolve(null)), `${heavy}:${latestId}`);

  const index = overview.data?.index;
  const opener = useMemo<Opener>(() => {
    const locs = new Set((shift.map?.places ?? []).map((l) => l.location_id));
    const as = new Set(shift.assets.map((a) => a.asset_id));
    return {
      open: setTarget,
      resolve: (id) => {
        if (/^M\d{3}$/.test(id)) return { kind: "mission", id };
        if (locs.has(id)) return { kind: "location", id };
        if (as.has(id)) return { kind: "asset", id };
        const mission = index?.action_mission[id];
        if (mission) return { kind: "mission", id: mission };
        const asset = index?.anomaly_asset[id];
        if (asset) return { kind: "asset", id: asset };
        return null;
      },
    };
  }, [shift, index, setTarget]);

  const o_ = overview.data;
  const byId = Object.fromEntries((findings.data?.findings ?? []).map((f) => [f.id, f]));
  const overall = locations.data?.overall ? locations.data.overall.k / Math.max(locations.data.overall.n, 1) : 0.18;
  const control = (call: () => Promise<Replay>) => () => void call().then(onLive).catch(() => undefined);

  return (
    <OpenCtx.Provider value={opener}>
      {show ? (
        <div className={s.page}>
          <div className={o.analysisHead}>
            <div>
              <h2 className={o.layerTitle}>The analysis behind this</h2>
              <p className={o.layerSub}>
                Every sentence on the shift screen, with the numbers under it: how many rows, how sure, and what cannot be concluded.
                {o_?.counts ? ` ${o_.counts.missions} missions, ${o_.counts.actions} actions, ${o_.counts.anomalies} anomalies; synthetic challenge data.` : ""}
              </p>
            </div>
            <div className={s.headRight}>
              <button type="button" className={s.linkBtn} data-guide="analysis:assumptions" onClick={() => setTarget({ kind: "assumptions", id: "assumptions" })}>
                Assumptions and method
              </button>
              <button type="button" className={s.linkBtn} data-guide="analysis:data-check" onClick={() => setTarget({ kind: "validation", id: "validation" })} title={o_?.validation?.summary}>
                Data check {o_?.validation ? `${o_.validation.agree}/${o_.validation.compared}` : ""}
              </button>
            </div>
          </div>

          {findings.data ? (
            <ExplainerGrid shift={shift} findings={findings.data.findings} live={live} force={force} onNumbers={setTarget} />
          ) : (
            <Loading what="the analysis" />
          )}

          <details className={o.fold} data-guide="analysis:findings">
            <summary>Every finding in one sentence ({shift.findings.length})</summary>
            <ul className={o.claims}>
              {shift.findings.map((f) => (
                <li key={f.id} data-guide={`finding:${f.id}`}>
                  <LevelIcon level={f.level} size={18} />
                  <span>
                    <b>{f.headline}.</b> {f.sentence} {f.note ? <i>{f.note}</i> : null}
                  </span>
                  <Why to={{ kind: "finding", id: f.finding_id }} onWhy={setTarget}>
                    The numbers
                  </Why>
                </li>
              ))}
            </ul>
          </details>

          <details className={o.fold} data-guide="analysis:tables" open={tables} onToggle={(e) => setTables((e.target as HTMLDetailsElement).open)}>
            <summary>The full tables: ranked queue, intervals, the trained scorer, replay of every round</summary>
            {tables ? (
              <div className={o.foldBody}>
          <section aria-label="Key numbers" className={s.kpis}>
            {(o_?.kpis ?? []).map((k) => {
              const warn = k.good === "low" ? (k.rate ?? 0) > 0.25 : (k.rate ?? 1) < 0.85;
              const to: Target = k.key === "hidden_failures" ? { kind: "finding", id: "hidden_failures" } : k.key === "anomalies_for_review" ? { kind: "finding", id: "anomaly_matrix" } : { kind: "finding", id: "version_comparison" };
              return (
                <button type="button" key={k.key} className={`${s.kpi} ${warn ? s.kpiWarn : ""}`} onClick={() => setTarget(to)} title={`95% interval ${range(k.lo, k.hi)}`}>
                  <span className={s.kpiNum}>
                    {pct(k.rate)}
                    <span className={s.kpiN}>
                      {k.k} of {k.n}
                    </span>
                  </span>
                  <span className={s.kpiLabel}>
                    {k.label}
                    {k.note ? ` (${k.note})` : ""}
                  </span>
                </button>
              );
            })}
          </section>

          <div className={s.grid}>
            <div className={s.col}>
              {triage.data ? <TriageQueue items={triage.data.items} scoreText={triage.data.score} orders={orders} limit={3} /> : <Loading what="the triage queue" />}
              {triage.data ? <UninspectedRisk rows={triage.data.uninspected} /> : null}
            </div>
            <div className={s.col}>
              {locations.data ? <FacilityMap data={locations.data} assets={assets.data?.assets ?? []} live={live} /> : <Loading what="the map" />}
              {live?.configured ? (
                <ReplayBar
                  live={live}
                  onStart={control(api.spotReplayStart)}
                  onPause={control(api.spotReplayPause)}
                  onSeek={(i) => void api.spotReplaySeek({ index: i }).then(onLive).catch(() => undefined)}
                  onSpeed={(v) => void api.spotReplaySpeed(v).then(onLive).catch(() => undefined)}
                />
              ) : null}
            </div>
          </div>

          <h2 className={s.sub} style={{ fontSize: 14 }}>
            Evidence: each panel says what was found, on how many rows, and what cannot be concluded
          </h2>
          {findings.data ? (
            <div className={s.evidence}>
              {byId.version_comparison ? <VersionCard f={byId.version_comparison} /> : null}
              {byId.battery ? <BatteryCard f={byId.battery} overall={overall} /> : null}
              {byId.reliability_by_type ? <RateCard f={byId.reliability_by_type} overall={overall} /> : null}
              {byId.reliability_by_location ? <RateCard f={byId.reliability_by_location} overall={overall} limit={8} onRow={(b) => setTarget({ kind: "location", id: b.key })} /> : null}
              {byId.reliability_by_asset ? <RateCard f={byId.reliability_by_asset} overall={overall} limit={8} onRow={(b) => setTarget({ kind: "asset", id: b.key })} /> : null}
              {byId.over_time ? <OverTimeCard f={byId.over_time} /> : null}
              {scorer.data?.configured ? <ModelCard m={scorer.data} /> : null}
              {byId.hidden_failures ? <RateCard f={byId.hidden_failures} overall={overall} limit={8} onRow={(b) => setTarget({ kind: "mission", id: b.key })} /> : null}
              {byId.navigation_hotspot ? <RateCard f={byId.navigation_hotspot} overall={overall} limit={5} onRow={(b) => setTarget({ kind: "location", id: b.key })} /> : null}
              {byId.anomalies_hidden ? <RateCard f={{ ...byId.anomalies_hidden, rows: [] }} overall={overall} /> : null}
              {byId.reading_trends ? <RateCard f={{ ...byId.reading_trends, rows: [] }} overall={overall} /> : null}
            </div>
          ) : (
            <Loading what="the findings" />
          )}

          <div className={s.grid}>
            <div className={s.col}>
              <MissionTimeline live={live} latest={latest.data ?? null} alerts={live?.alerts ?? []} />
            </div>
            <div className={s.col}>
              <AskBox modelLine={o_?.model.role} />
              {handoff.data?.configured ? <HandoffCard handoff={handoff.data} /> : null}
              {o_?.next?.length ? (
                <div className={s.card}>
                  <div className={s.cardHead}>
                    <h2>Investigate next, as the analysis words it</h2>
                  </div>
                  {o_.next.map((n) => (
                    <div key={n.key}>
                      <p className={s.sentence}>
                        <b>{n.title}.</b> {n.sentence}
                      </p>
                      <EvidenceLine evidence={n.evidence} max={8} />
                    </div>
                  ))}
                </div>
              ) : null}
            </div>
          </div>
              </div>
            ) : null}
          </details>
        </div>
      ) : null}

      {explain && (findings.data || explain === "hidden" || explain === "alarms") ? (
        <ExplainSheet id={explain} shift={shift} findings={findings.data?.findings ?? []} live={live} force={force} onNumbers={setTarget} onClose={() => setExplain(null)} />
      ) : null}

      {target && ((target.kind !== "finding" && target.kind !== "metric") || findings.data) && (target.kind !== "location" || locations.data) ? (
        <div className={s.page} style={{ display: "contents" }}>
          <Detail target={target} onClose={() => setTarget(null)} findings={findings.data?.findings ?? []} locations={locations.data} overview={o_} orders={orders} />
        </div>
      ) : null}
    </OpenCtx.Provider>
  );
}
