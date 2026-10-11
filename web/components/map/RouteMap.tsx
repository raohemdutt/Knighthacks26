"use client";

import { useMemo } from "react";
import type { PlacePack, PlaceRoute } from "@/lib/api";
import { distance, pinKind, toScreen, type Pt } from "@/lib/campusmap";
import { indexPlaces, minutes } from "@/lib/dayrun";
import { t12 as clock } from "@/lib/plan";
import { CampusLayers, MapButtons, MapFurniture, PlacePin, useCampusMap, useMapView } from "../dayrun/CampusLayers";
import cm from "../dayrun/campusmap.module.css";
import { linePath, midpoint, pinName } from "../dayrun/geometry";
import { Loading } from "../States";
import ui from "../ui.module.css";

/**
 * One walk on the campus map: the shared drawing of a route from GET /api/places/route.
 *
 * Every screen that shows a walk between two places draws it with this: the route sheet behind "Route" and
 * "Directions" (Today's Plan, Now, Calendar) and the Circle's walk to a friend. Same map (the campus layers of
 * config/places/<pack>.map.json), same line, same pins and "you" dot, same minutes badge, same buttons, so a walk
 * looks the same wherever it is asked for. The layers and their palette stay where they were written
 * (components/dayrun/CampusLayers.tsx, campusmap.module.css); this is the one place that puts a route on them.
 *
 * It only draws what the answer holds. Whether a friend's place may be shown at all is decided on the box before
 * there is an answer (a place is routed to only while it is shared and has not run out).
 */

const W = 360;
const H = 320;

/** The map with the walk on it; a plain line of words when the places are not on a drawn map. */
export function RouteMap({ route: r, packs, guide }: { route: PlaceRoute; packs: PlacePack[] | undefined; guide?: string }) {
  const places = useMemo(() => indexPlaces(packs ?? []), [packs]);
  const wanted = useMemo(() => new Set(r.path.map((p) => p.split(".")[0] ?? "")), [r]);
  const { map } = useCampusMap(packs, wanted);
  const line: Pt[] = useMemo(() => (map ? r.polyline.map(([lat, lon]) => map.project(lat, lon)) : []), [map, r]);
  const controls = useMapView(line.length > 1 ? map : null, W, H, line, `${r.from}|${r.to}|${line.length}`);
  const view = controls.view;
  const ends = [r.path.find((p) => typeof places.get(p)?.lat === "number"), [...r.path].reverse().find((p) => typeof places.get(p)?.lat === "number")].filter((p): p is string => Boolean(p));
  const on = new Set<number>();
  if (map)
    for (const ref of ends) {
      const b = map.places.get(ref.split(".").slice(1).join("."))?.building;
      if (b !== undefined) on.add(b);
    }
  const badge = midpoint(line);

  if (map && line.length > 1)
    return (
      <div className={cm.frame} data-guide={guide}>
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
          {badge
            ? (() => {
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
            : null}
          <MapFurniture map={map} view={view} width={W} height={H} />
        </svg>
        <MapButtons controls={controls} onFit={() => controls.fit(line)} />
      </div>
    );
  if (packs && r.polyline.length < 2) return <p className={ui.muted}>{r.walk_minutes === 0 ? "No walk to draw." : "These places are not on a drawn map, so the walk is listed, not drawn."}</p>;
  return <Loading what="the map" />;
}

/** When to leave or arrive, how long on foot, how far: the same row of facts under every walk. */
export function RouteFacts({ route: r }: { route: PlaceRoute }) {
  return (
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
  );
}

/** The places the walk passes, in order, and whose map data it is drawn on. */
export function RoutePlaces({ route: r, packs }: { route: PlaceRoute; packs: PlacePack[] | undefined }) {
  const places = useMemo(() => indexPlaces(packs ?? []), [packs]);
  const indoors = r.path.filter((p) => typeof places.get(p)?.lat !== "number").length;
  return (
    <p className={`${ui.muted} ${ui.small} ${ui.breakWords}`}>
      {r.path.map((p) => places.get(p)?.name ?? p).join(" → ")}.
      {indoors > 0 && r.polyline.length > 1 ? " The part inside the building is counted in the minutes, not drawn." : ""}
      {r.attribution ? ` Map data ${r.attribution}.` : ""}
    </p>
  );
}
