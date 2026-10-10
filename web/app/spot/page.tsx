"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AskBox, HandoffCard } from "@/components/spot/AskBox";
import { OpenCtx, pct, range, type Opener } from "@/components/spot/bits";
import { Detail } from "@/components/spot/Detail";
import { BatteryCard, ModelCard, OverTimeCard, RateCard, VersionCard } from "@/components/spot/EvidencePanels";
import { FacilityMap } from "@/components/spot/FacilityMap";
import { MissionTimeline, ReplayBar } from "@/components/spot/ReplayPanel";
import s from "@/components/spot/spot.module.css";
import { TriageQueue, UninspectedRisk } from "@/components/spot/Triage";
import type { Replay, Target } from "@/components/spot/types";
import { Empty, ErrorBox, Loading } from "@/components/States";
import { api } from "@/lib/api";
import { useAtlas, useAtlasEvents } from "@/lib/atlas";
import { useFetch } from "@/lib/useFetch";

/**
 * Inspection (SPOT): Atlas pointed at a robot's missions. One page for a laptop or wall display (two or three columns)
 * and for a phone (one column).
 *
 *  - Top: the three things to investigate next, then KPIs with their denominators.
 *  - Facility map with waypoint health, assets and the robot; the replay bar under it.
 *  - The mission in the replay as a timeline, with alerts as they arrive.
 *  - The triage queue with one-tap Re-inspect / Work order (each an approval card), and uninspected risk.
 *  - Evidence panels: V1 against V2, by inspection type, by waypoint, battery, over time.
 *  - Ask with cited answers, the shift handoff, and the Assumptions and method drawer.
 *
 * The live picture comes from GET /api/spot/replay/state, re-read when a `spot.` event arrives on /ws (and once a second
 * while playing, in case the socket is down). Everything else is computed once by the box.
 */
export default function SpotPage() {
  const { model, link } = useAtlas();
  const overview = useFetch((sig) => api.spotOverview(sig));
  const findings = useFetch((sig) => api.spotFindings(sig));
  const triage = useFetch((sig) => api.spotTriage(sig));
  const locations = useFetch((sig) => api.spotLocations(sig));
  const assets = useFetch((sig) => api.spotAssets(sig));
  const handoff = useFetch((sig) => api.spotHandoff(sig));
  const orders = useFetch((sig) => api.spotWorkOrders(sig));
  const scorer = useFetch((sig) => api.spotModel(sig));
  const latestId = overview.data?.latest_mission?.mission_id;
  const latest = useFetch((sig) => (latestId ? api.spotMission(latestId, sig) : Promise.resolve(null)), latestId ?? "");

  const [live, setLive] = useState<Replay | null>(null);
  const [target, setTarget] = useState<Target | null>(null);

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

  const control = (call: () => Promise<Replay>) => () => void call().then(setLive).catch(() => undefined);

  const index = overview.data?.index;
  const known = useMemo(() => {
    const locs = new Set((locations.data?.locations ?? []).map((l) => l.location_id));
    const as = new Set((assets.data?.assets ?? []).map((a) => a.asset_id));
    return { locs, as };
  }, [locations.data, assets.data]);
  const opener = useMemo<Opener>(
    () => ({
      open: setTarget,
      resolve: (id) => {
        if (/^M\d{3}$/.test(id)) return { kind: "mission", id };
        if (known.locs.has(id)) return { kind: "location", id };
        if (known.as.has(id)) return { kind: "asset", id };
        const mission = index?.action_mission[id];
        if (mission) return { kind: "mission", id: mission };
        const asset = index?.anomaly_asset[id];
        if (asset) return { kind: "asset", id: asset };
        return null;
      },
    }),
    [known, index],
  );

  if (overview.error) return <ErrorBox what="the inspection data" error={overview.error} onRetry={overview.reload} />;
  if (!overview.data) return <Loading what="the inspection data" />;
  const o = overview.data;
  if (!o.configured) {
    return (
      <div className={s.page}>
        <div className={s.head}>
          <h1>Inspection (SPOT)</h1>
        </div>
        <Empty title="No inspection data on this box yet">{o.note}</Empty>
      </div>
    );
  }

  const byId = Object.fromEntries((findings.data?.findings ?? []).map((f) => [f.id, f]));
  const overall = locations.data?.overall ? locations.data.overall.k / Math.max(locations.data.overall.n, 1) : 0.18;

  return (
    <OpenCtx.Provider value={opener}>
      <div className={s.page}>
        <div className={s.head}>
          <div>
            <h1>Inspection (SPOT)</h1>
            <p className={`${s.small} ${s.muted}`}>
              {o.counts?.missions} missions, {o.counts?.actions} actions, {o.counts?.anomalies} anomalies · data to {o.now?.replace("T", " ").slice(0, 16)} · synthetic challenge data
            </p>
          </div>
          <div className={s.headRight}>
            <span className={s.offline} title={o.model.role}>
              <i className={s.dotOk} /> Runs on this box, offline · model <b>{model ?? o.model.version ?? "none"}</b>
              {link !== "live" ? " · reconnecting" : ""}
            </span>
            <button type="button" className={s.linkBtn} onClick={() => setTarget({ kind: "assumptions", id: "assumptions" })}>
              Assumptions and method
            </button>
            <button type="button" className={s.linkBtn} onClick={() => setTarget({ kind: "validation", id: "validation" })} title={o.validation?.summary}>
              Data check {o.validation ? `${o.validation.agree}/${o.validation.compared}` : ""}
            </button>
          </div>
        </div>

        <section aria-label="Investigate next" className={s.next}>
          {(o.next ?? []).map((n, i) => (
            <button type="button" key={n.key} className={s.nextCard} onClick={() => setTarget(n.target)}>
              <span className={s.nextNum}>{i + 1}</span>
              <span className={s.nextTitle}>{n.title}</span>
              <span className={s.nextWhy}>{n.sentence}</span>
            </button>
          ))}
        </section>

        <section aria-label="Key numbers" className={s.kpis}>
          {(o.kpis ?? []).map((k) => {
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
            {locations.data ? <FacilityMap data={locations.data} assets={assets.data?.assets ?? []} live={live} /> : <Loading what="the map" />}
            {live?.configured ? (
              <ReplayBar
                live={live}
                onStart={control(api.spotReplayStart)}
                onPause={control(api.spotReplayPause)}
                onSeek={(i) => void api.spotReplaySeek({ index: i }).then(setLive).catch(() => undefined)}
                onSpeed={(v) => void api.spotReplaySpeed(v).then(setLive).catch(() => undefined)}
              />
            ) : null}
            {triage.data ? <UninspectedRisk rows={triage.data.uninspected} /> : null}
            <AskBox modelLine={o.model.role} />
            {handoff.data?.configured ? <HandoffCard handoff={handoff.data} /> : null}
          </div>
          <div className={s.col}>
            {live && live.index > 0 ? <MissionTimeline live={live} latest={latest.data ?? null} alerts={live.alerts ?? []} /> : null}
            {triage.data ? <TriageQueue items={triage.data.items} scoreText={triage.data.score} orders={orders.data?.work_orders ?? []} limit={3} /> : <Loading what="the triage queue" />}
            {live && live.index > 0 ? null : <MissionTimeline live={live} latest={latest.data ?? null} alerts={[]} />}
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

        {target ? <Detail target={target} onClose={() => setTarget(null)} findings={findings.data?.findings ?? []} locations={locations.data} overview={o} orders={orders.data?.work_orders ?? []} /> : null}
      </div>
    </OpenCtx.Provider>
  );
}
