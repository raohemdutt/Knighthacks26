"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { SimResult } from "@/lib/api";
import { pinKind, toScreen, type CampusMap, type Pt } from "@/lib/campusmap";
import type { Moment, World } from "@/lib/dayplay";
import { minutes, parentPlace, type PlaceIndex } from "@/lib/dayrun";
import cm from "./campusmap.module.css";
import { CampusLayers, PlacePin, layoutPins, lineBoxes, useMapView, type ScreenBox } from "./CampusLayers";
import { Figure } from "./Figure";
import { legLine, linePath, midpoint, pinName } from "./geometry";
import pl from "./plan.module.css";

/**
 * The map of the plan: a modest well inside the hero that the agenda drives.
 *
 * It shows only what today's plan touches: the buildings you go to (named, at a fixed size), and the walks between
 * them. Select a row of the agenda and its walk and its building are lit and the rest dims; tap a building and its
 * row is selected. While the day plays, the figure walks the same map. Drag to move, pinch or use the buttons to zoom.
 *
 * The svg is drawn at the well's own pixel size (no scaling viewBox), so text and pins stay the size they are set.
 */
export function PlanMap({
  map,
  places,
  world,
  result,
  focusRoute,
  focusPlace,
  moment,
  playing,
  phase,
  still,
  closed,
  pills,
  onPlace,
}: {
  map: CampusMap;
  places: PlaceIndex;
  world: World;
  result: SimResult;
  /** The walk of the selected row (an index into `result.routes`), lit; null lights none. */
  focusRoute: number | null;
  /** Where the selected row happens. */
  focusPlace: string | null;
  /** Where you are at the cursor, when the day is being played or scrubbed. */
  moment: Moment | null;
  playing: boolean;
  phase: number;
  still: boolean;
  /** Places shut today (hard mode). */
  closed: string[];
  /** What went wrong, in a few words each, shown over the map. */
  pills: string[];
  onPlace: (ref: string) => void;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ w: 340, h: 212 });
  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const measure = () => {
      const r = el.getBoundingClientRect();
      if (r.width > 40 && r.height > 40) setSize((s) => (Math.abs(s.w - r.width) < 1 && Math.abs(s.h - r.height) < 1 ? s : { w: Math.round(r.width), h: Math.round(r.height) }));
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const { w, h } = size;

  /** A place's own spot on the map, or the building that leads into it (a room is shown at its dorm). */
  const outdoor = (ref: string | null | undefined): string | undefined => (ref ? (world.at.has(ref) ? ref : parentPlace(places, ref)?.ref) : undefined);

  const lines = useMemo(
    () =>
      result.routes.map((r) => {
        const points: Pt[] = [];
        for (const leg of result.legs) {
          if (leg.route !== r.index) continue;
          for (const p of legLine(map, leg)) {
            const q = points[points.length - 1];
            if (!q || Math.hypot(p.x - q.x, p.y - q.y) > 0.3) points.push(p);
          }
        }
        return { route: r, points };
      }),
    [result, map],
  );

  const touched = useMemo(() => {
    const out = new Set<string>();
    const add = (ref: string | null | undefined) => {
      const o = outdoor(ref);
      if (o && world.at.has(o)) out.add(o);
    };
    add(result.timeline[0]?.place);
    for (const b of result.blocks) add(b.place);
    for (const c of closed) add(c);
    return out;
    // `outdoor` reads `world` and `places` only
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result, closed, world, places]);

  const walking = moment?.walk ? lines.find((l) => l.route.index === moment.walk?.route.index) : undefined;
  const lit = walking ?? (focusRoute !== null ? lines.find((l) => l.route.index === focusRoute && l.points.length > 1) : undefined);
  const litPlace = outdoor(focusPlace);
  const dayPoints = useMemo(() => [...lines.flatMap((l) => l.points), ...[...touched].map((ref) => world.at.get(ref)).filter((p): p is Pt => p !== undefined)], [lines, touched, world]);
  const litPt = litPlace ? world.at.get(litPlace) : undefined;
  const focus = lit ? lit.points : litPt && !playing ? [litPt, ...dayPoints.filter((p) => Math.hypot(p.x - litPt.x, p.y - litPt.y) < 260)] : dayPoints;
  const focusKey = `${lit ? `r${lit.route.index}` : litPlace && !playing ? `p${litPlace}` : "day"}|${result.start}|${result.routes.length}|${w}x${h}`;
  const controls = useMapView(map, w, h, focus, focusKey);
  const view = controls.view;
  const S = (p: Pt) => toScreen(view, w, h, p);

  const buildingOf = (ref: string) => map.places.get(ref.split(".").slice(1).join("."))?.building;
  const on = new Set<number>();
  const shut = new Set<number>();
  for (const ref of touched) {
    const b = buildingOf(ref);
    if (b === undefined) continue;
    on.add(b);
    if (closed.includes(ref)) shut.add(b);
  }
  const quiet = useMemo(() => new Set(map.buildings.keys()), [map]); // only the plan's places are named

  const dimOthers = Boolean(lit || litPlace) && !playing;
  const spots = [...touched]
    .map((ref) => ({ ref, p: places.get(ref), s: S(world.at.get(ref) as Pt) }))
    .filter((x) => x.p && x.s.x > -20 && x.s.x < w + 20 && x.s.y > -20 && x.s.y < h + 20);
  const taken: ScreenBox[] = [{ x0: w - 92, y0: h - 46, x1: w, y1: h }];
  if (pills.length > 0) taken.push({ x0: 0, y0: 0, x1: w, y1: 34 });
  const badge = lit && !playing ? midpoint(lit.points) : undefined;
  const badgeText = lit ? minutes(lit.route.minutes) : "";
  const badgeW = badgeText.length * 6.4 + 14;
  if (badge) taken.push({ x0: S(badge).x - badgeW / 2 - 2, y0: S(badge).y - 12, x1: S(badge).x + badgeW / 2 + 2, y1: S(badge).y + 12 });
  const you = moment ? S(moment.pt) : null;
  if (you) taken.push({ x0: you.x - 16, y0: you.y - 34, x1: you.x + 16, y1: you.y + 6 });
  if (lit) taken.push(...lineBoxes(view, w, h, lit.points, 4));
  const where = layoutPins(
    spots.map(({ ref, p, s }) => ({ key: ref, x: s.x, y: s.y, big: true, text: `${pinName(p?.name ?? "", p?.aliases ?? [])}${closed.includes(ref) ? " · closed" : ""}`, named: true })),
    taken,
  );

  const label = `Map of today's plan: ${spots.length} places and ${result.routes.length} walks.${lit ? ` Showing the walk ${lit.route.purpose}.` : ""} Drag to move, pinch to zoom.`;
  return (
    <div ref={wrap} className={`${cm.frame} ${pl.mapWell}`} data-guide="plan:map">
      <svg {...controls.bind} className={pl.mapSvg} width={w} height={h} viewBox={`0 0 ${w} ${h}`} role="group" aria-label={label}>
        <CampusLayers map={map} view={view} width={w} height={h} highlight={{ on, closed: shut }} quiet={quiet} taken={taken} />

        {lines.map((l) => (l.points.length > 1 && l !== lit ? <path key={`walk-${l.route.index}`} className={`${cm.routeDim} ${dimOthers ? pl.dim : ""}`} d={linePath(view, w, h, l.points)} /> : null))}
        {lit ? (
          <>
            <path className={cm.routeCase} d={linePath(view, w, h, lit.points)} />
            <path className={cm.route} d={linePath(view, w, h, lit.points)} />
          </>
        ) : null}

        {spots.map(({ ref, p, s }) => {
          const isLit = ref === litPlace || (lit && (outdoor(lit.route.end) === ref || outdoor(lit.route.start) === ref));
          const below = where.get(ref);
          const name = `${pinName(p?.name ?? "", p?.aliases ?? [])}${closed.includes(ref) ? " · closed" : ""}`;
          const r = 9;
          return (
            <a
              key={ref}
              className={`${pl.spot} ${dimOthers && !isLit ? pl.dim : ""}`}
              role="button"
              tabIndex={0}
              aria-label={`${p?.name ?? ref}: show it in the plan`}
              onClick={() => onPlace(ref)}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  onPlace(ref);
                }
              }}
            >
              <circle cx={s.x} cy={s.y} r={18} fill="transparent" />
              <PlacePin x={s.x} y={s.y} kind={pinKind(p?.kind)} closed={closed.includes(ref)} />
              {below === null || below === undefined ? null : (
                <text className={`${pl.place} ${closed.includes(ref) ? pl.placeShut : ""}`} x={s.x} y={below ? s.y + r + 13 : s.y - r - 6} textAnchor="middle">
                  {name}
                </text>
              )}
            </a>
          );
        })}

        {badge && badgeText ? (
          <g transform={`translate(${S(badge).x.toFixed(1)} ${S(badge).y.toFixed(1)})`}>
            <rect className={cm.badge} x={-badgeW / 2} y={-10} width={badgeW} height={20} rx={10} />
            <text className={cm.badgeText} x={0} y={3.8} textAnchor="middle">
              {badgeText}
            </text>
          </g>
        ) : null}

        {you && moment ? <Figure x={you.x} y={you.y} dir={moment.dx < 0 ? -1 : 1} doing={moment.doing} phase={phase} still={still} scale={1.05} /> : null}
      </svg>

      {pills.length > 0 ? (
        <div className={pl.mapPills}>
          {pills.map((text) => (
            <span key={text} className={pl.mapPill}>
              {text}
            </span>
          ))}
        </div>
      ) : null}
      <a className={pl.mapCredit} href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer noopener">
        {map.attribution}
      </a>
      <div className={pl.mapTools}>
        <button type="button" className={pl.mapTool} onClick={() => controls.fit(focus)} aria-label="Fit the plan in view">
          ⤢
        </button>
        <button type="button" className={pl.mapTool} onClick={() => controls.zoomBy(1.6)} disabled={controls.atMax} aria-label="Zoom in">
          +
        </button>
        <button type="button" className={pl.mapTool} onClick={() => controls.zoomBy(1 / 1.6)} disabled={controls.atMin} aria-label="Zoom out">
          −
        </button>
      </div>
    </div>
  );
}
