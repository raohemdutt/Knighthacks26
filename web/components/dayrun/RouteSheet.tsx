"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { api, type PlaceRoute } from "@/lib/api";
import { distance, pinKind, toScreen, type Pt } from "@/lib/campusmap";
import { clock, indexPlaces, minutes, ms } from "@/lib/dayrun";
import { useFetch } from "@/lib/useFetch";
import { ErrorBox, Loading } from "../States";
import ui from "../ui.module.css";
import cm from "./campusmap.module.css";
import { CampusLayers, MapButtons, MapFurniture, PlacePin, useCampusMap, useMapView } from "./CampusLayers";
import { linePath, midpoint, pinName } from "./CampusMap";

/**
 * "Get me there": a Route button that opens the campus map with the walk from
 * one place to another along the real footpaths, and when to leave.
 *
 *   <RouteSheet from={status.seen_at ?? "here"} to={next.place} arriveBy={next.start} now={data.now} />
 *
 * `from` and `to` are what GET /api/places/route takes: a pack id ("ucf.eng2"),
 * a place's name, "friend:<member>", or "here" (where the box last saw you).
 * With `arriveBy` the sheet counts down to the time to leave; without it, it
 * says how long the walk is from now. `now` is the box's own time when the
 * screen was loaded (a replayed or scripted day is not on the phone's clock).
 */

const W = 360;
const H = 320;

export function RouteSheet({
  from = "here",
  to,
  arriveBy,
  now,
  label = "Route",
  what,
}: {
  from?: string;
  to: string;
  arriveBy?: string | null;
  now?: string | null;
  label?: string;
  /** What is at the other end, for the heading: "Knight Hacks: judging and demos". */
  what?: string;
}) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <button type="button" className={cm.routeBtn} aria-haspopup="dialog" onClick={() => setOpen(true)}>
        {label}
      </button>
      {open ? <Sheet from={from} to={to} arriveBy={arriveBy ?? null} now={now ?? null} what={what} onClose={() => setOpen(false)} /> : null}
    </>
  );
}

/** "leave in 6 min", "leave now", "you should have left 3 min ago". */
export function leaveText(route: PlaceRoute, at: number): { text: string; tone: "ok" | "soon" | "late" } {
  if (route.walk_minutes === 0) return { text: "You are already there", tone: "ok" };
  if (route.mode !== "arrive_by") return { text: `${minutes(route.walk_minutes)} on foot`, tone: "ok" };
  const left = (ms(route.depart) - at) / 60000;
  if (left >= 1) return { text: `Leave in ${minutes(Math.floor(left))}`, tone: left <= 5 ? "soon" : "ok" };
  if (left > -1) return { text: "Leave now", tone: "soon" };
  if (at < ms(route.arrive)) return { text: `Leave now: ${minutes(Math.floor(-left))} behind`, tone: "late" };
  return { text: `It started ${minutes(Math.floor((at - ms(route.arrive)) / 60000))} ago`, tone: "late" };
}

function Sheet({ from, to, arriveBy, now, what, onClose }: { from: string; to: string; arriveBy: string | null; now: string | null; what?: string; onClose: () => void }) {
  const route = useFetch((s) => api.placeRoute({ from, to, arrive_by: arriveBy ?? undefined, depart: arriveBy ? undefined : (now ?? undefined) }, s), `${from}|${to}|${arriveBy}|${now}`);
  const packs = useFetch((s) => api.places(s));
  const places = useMemo(() => indexPlaces(packs.data ?? []), [packs.data]);
  const r = route.data;
  const wanted = useMemo(() => new Set(r ? r.path.map((p) => p.split(".")[0] ?? "") : []), [r]);
  const { map } = useCampusMap(r ? packs.data : undefined, wanted);

  // The box's clock, carried forward by the time this sheet has been open.
  const opened = useRef({ wall: Date.now(), box: now ? ms(now) : Date.now() });
  const [at, setAt] = useState(opened.current.box);
  useEffect(() => {
    const id = window.setInterval(() => setAt(opened.current.box + (Date.now() - opened.current.wall)), 5000);
    return () => window.clearInterval(id);
  }, []);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const line: Pt[] = useMemo(() => (map && r ? r.polyline.map(([lat, lon]) => map.project(lat, lon)) : []), [map, r]);
  const controls = useMapView(line.length > 1 ? map : null, W, H, line, `${r?.from}|${r?.to}|${line.length}`);
  const view = controls.view;
  const leave = r ? leaveText(r, at) : null;
  const ends = r
    ? [r.path.find((p) => typeof places.get(p)?.lat === "number"), [...r.path].reverse().find((p) => typeof places.get(p)?.lat === "number")].filter((p): p is string => Boolean(p))
    : [];
  const on = new Set<number>();
  if (map) for (const ref of ends) {
    const b = map.places.get(ref.split(".").slice(1).join("."))?.building;
    if (b !== undefined) on.add(b);
  }
  const indoors = r ? r.path.filter((p) => typeof places.get(p)?.lat !== "number").length : 0;
  const badge = midpoint(line);

  // Through a portal to <body>: the page's entrance animation leaves a transform on every section, and a
  // transformed ancestor makes `position: fixed` resolve against that section instead of the viewport.
  return createPortal(
    <div className={ui.overlay} onClick={onClose}>
      <div className={ui.sheet} role="dialog" aria-modal="true" aria-label={r ? `Route to ${r.to_name}` : "Route"} onClick={(e) => e.stopPropagation()}>
        <div className={`${ui.row} ${ui.between}`}>
          <h2 className={ui.breakWords}>{r ? `To ${r.to_name}` : "Route"}</h2>
          <button type="button" className={ui.btn} onClick={onClose}>
            Close
          </button>
        </div>
        {route.error ? (
          <ErrorBox what="the route" error={route.error} onRetry={route.reload} />
        ) : !r || !leave ? (
          <Loading what="the route" />
        ) : (
          <>
            <div>
              <p className={`${cm.leave} ${leave.tone === "soon" ? cm.leaveSoon : leave.tone === "late" ? cm.leaveLate : ""}`} role="status">
                {leave.text}
              </p>
              <p className={ui.breakWords}>
                {what ? `${what}: ` : ""}
                {r.sentence}.
              </p>
            </div>
            <p className={cm.facts} hidden={r.walk_minutes === 0}>
              {r.mode === "arrive_by" ? (
                <span>
                  leave by <strong>{clock(r.depart)}</strong>
                </span>
              ) : (
                <span>
                  there by <strong>{clock(r.arrive)}</strong>
                </span>
              )}
              <span>
                <strong>{minutes(r.walk_minutes)}</strong> on foot
              </span>
              {typeof r.meters === "number" ? (
                <span>
                  <strong>{distance(r.meters)}</strong>
                </span>
              ) : null}
              {r.mode === "arrive_by" ? (
                <span>
                  be there <strong>{clock(r.arrive)}</strong>
                </span>
              ) : null}
            </p>

            {map && line.length > 1 ? (
              <div className={cm.frame}>
                <svg {...controls.bind} className={cm.svg} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Map of the walk from ${r.from_name} to ${r.to_name}: ${r.sentence}. Drag to move, pinch to zoom.`}>
                  <CampusLayers map={map} view={view} width={W} height={H} highlight={{ on }} quiet={on} />
                  <path className={cm.routeCase} d={linePath(view, W, H, line)} />
                  <path className={cm.route} d={linePath(view, W, H, line)} />
                  {ends.map((ref, i) => {
                    const p = places.get(ref);
                    if (!p || typeof p.lat !== "number" || typeof p.lon !== "number") return null;
                    const s = toScreen(view, W, H, map.project(p.lat, p.lon));
                    if (i === 0 && ends.length > 1)
                      return (
                        <g key={ref} transform={`translate(${s.x.toFixed(1)} ${s.y.toFixed(1)})`}>
                          <circle className={cm.youHalo} r={15} />
                          <circle className={cm.you} r={6} />
                          <text className={cm.pinLabel} x={0} y={-12} textAnchor="middle">
                            {pinName(p.name, p.aliases)}
                          </text>
                        </g>
                      );
                    return <PlacePin key={ref} x={s.x} y={s.y} kind={pinKind(p.kind)} label={pinName(p.name, p.aliases)} />;
                  })}
                  {badge ? (
                    (() => {
                      const s = toScreen(view, W, H, badge);
                      const text = `${minutes(r.walk_minutes)}${typeof r.meters === "number" ? ` · ${distance(r.meters)}` : ""}`;
                      const w = text.length * 5.9 + 12;
                      return (
                        <g transform={`translate(${s.x.toFixed(1)} ${s.y.toFixed(1)})`}>
                          <rect className={cm.badge} x={-w / 2} y={-9.5} width={w} height={19} rx={9.5} />
                          <text className={cm.badgeText} x={0} y={3.8} textAnchor="middle">
                            {text}
                          </text>
                        </g>
                      );
                    })()
                  ) : null}
                  <MapFurniture map={map} view={view} width={W} height={H} />
                </svg>
                <MapButtons controls={controls} onFit={() => controls.fit(line)} />
              </div>
            ) : packs.data && r.polyline.length < 2 ? (
              <p className={ui.muted}>{r.walk_minutes === 0 ? "No walk to draw." : "These places are not on a drawn map, so the walk is listed, not drawn."}</p>
            ) : (
              <Loading what="the map" />
            )}

            <p className={`${ui.muted} ${ui.small} ${ui.breakWords}`}>
              {r.path.map((p) => places.get(p)?.name ?? p).join(" → ")}.
              {indoors > 0 && r.polyline.length > 1 ? " The part inside the building is counted in the minutes, not drawn." : ""}
              {r.attribution ? ` Map data ${r.attribution}.` : ""}
            </p>
          </>
        )}
      </div>
    </div>,
    document.body,
  );
}
