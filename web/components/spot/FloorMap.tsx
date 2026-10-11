"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { spotUrl } from "@/lib/api";
import { useTraced, type SpotRoute } from "@/lib/guideSpot";
import { floorAt, legBetween, snapToFloor, type WalkFloor } from "@/lib/guideWalk";
import o from "./op.module.css";
import type { AssetState, Level, Shift } from "./types";

/**
 * The floor plan for the person on shift: one mark per piece of equipment, in three levels that differ by shape as well as
 * colour (a round "!" needs you, a triangle is one to keep an eye on, a small tick is fine), SPOT's route drawn lightly, and
 * SPOT itself when a round is being replayed.
 *
 * Routes are drawn along the floor traced from the plan picture (GET /api/spot/walkable/legs) when the box has that
 * trace: the round's legs, a planned round's legs, and the look guide's walk (`walk`: solid where SPOT has walked that
 * leg, dashed where the line is from the trace only). Lines end at a pin's tip, where the kit's pin points; the marks sit
 * on the pin's head. `showFloor` lays the traced walkable area faintly over the picture.
 *
 * Backdrop and pin positions are the same as the analyst map (FacilityMap): the kit's plan image with marks on its pins
 * (fixtures/spot/waypoint_pixels.json), or a schematic from locations.csv x/y when the box has no plan image.
 */

export type FloorView = "both" | number;
interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}
export interface Geometry {
  w: number;
  h: number;
  panels: Record<number, Rect>;
  pos: Record<string, { x: number; y: number; floor: number }>;
  image: string | null;
}

const PIN_HEAD_PX = 20; // the kit's pins are drawn about this far above their tip; our marks sit on the pin's head

export function geometry(map: NonNullable<Shift["map"]>): Geometry {
  const plan = map.plan;
  const pos: Geometry["pos"] = {};
  const panels: Geometry["panels"] = {};
  if (plan) {
    for (const [f, p] of Object.entries(plan.panels)) panels[Number(f)] = { x: p.x0 * plan.width, y: p.y0 * plan.height, w: (p.x1 - p.x0) * plan.width, h: (p.y1 - p.y0) * plan.height };
    for (const l of map.places) {
      const p = plan.waypoints[l.location_id];
      if (p) pos[l.location_id] = { x: p.x * plan.width, y: p.y * plan.height - PIN_HEAD_PX, floor: l.floor };
    }
    return { w: plan.width, h: plan.height, panels, pos, image: spotUrl(plan.image_url) };
  }
  const floors = [...new Set(map.places.map((l) => l.floor))].sort();
  const size = 560;
  const gap = 28;
  floors.forEach((f, i) => (panels[f] = { x: i * (size + gap), y: 0, w: size, h: size }));
  for (const l of map.places) {
    const p = panels[l.floor];
    if (p) pos[l.location_id] = { x: p.x + 50 + (l.x / 100) * (size - 100), y: p.y + 60 + (l.y / 100) * (size - 110), floor: l.floor };
  }
  return { w: Math.max(1, floors.length) * size + Math.max(0, floors.length - 1) * gap, h: size, panels, pos, image: null };
}

const FILL: Record<Level, string> = { act: "var(--bad)", watch: "var(--watch)", ok: "var(--ok)" };

/** The mark for a level, centred on 0,0. `u` is viewBox units per CSS pixel, so marks keep their size on any screen. */
export function LevelMark({ level, u, small }: { level: Level; u: number; small?: boolean }) {
  const k = small ? 0.8 : 1;
  if (level === "act") {
    const r = 13 * u * k;
    return (
      <>
        <circle r={r} fill={FILL.act} stroke="var(--surface)" strokeWidth={2.5 * u} />
        <rect x={-1.7 * u * k} y={-7.5 * u * k} width={3.4 * u * k} height={9 * u * k} rx={1.5 * u * k} fill="var(--on-media)" />
        <circle cy={5.6 * u * k} r={2 * u * k} fill="var(--on-media)" />
      </>
    );
  }
  if (level === "watch") {
    const r = 13.5 * u * k;
    return (
      <>
        <path d={`M 0 ${-r} L ${r} ${r * 0.82} L ${-r} ${r * 0.82} Z`} fill={FILL.watch} stroke="var(--surface)" strokeWidth={2.5 * u} strokeLinejoin="round" />
        <circle cy={r * 0.2} r={2.6 * u * k} fill="var(--on-media)" />
      </>
    );
  }
  const r = 9.5 * u * k;
  return (
    <>
      <circle r={r} fill={FILL.ok} stroke="var(--surface)" strokeWidth={2.5 * u} />
      <path d={`M ${-4 * u * k} ${0.3 * u * k} L ${-1.2 * u * k} ${3.2 * u * k} L ${4.3 * u * k} ${-3 * u * k}`} fill="none" stroke="var(--on-media)" strokeWidth={2 * u * k} strokeLinecap="round" strokeLinejoin="round" />
    </>
  );
}

/** The same three marks as a small inline icon, for cards, lists and the legend. */
export function LevelIcon({ level, size = 22 }: { level: Level; size?: number }) {
  return (
    <svg className={o.levelIcon} width={size} height={size} viewBox="-16 -16 32 32" aria-hidden="true">
      <LevelMark level={level} u={1} />
    </svg>
  );
}

export function FloorMap({
  map,
  assets,
  onPick,
  selected,
  route,
  done,
  robotAt,
  robotMs,
  dog,
  quiet,
  title,
  note,
  initialFloor,
  order,
  route2,
  routeLabel,
  guide,
  walk,
  hereAt,
  onTap,
  showFloor,
  children,
}: {
  map: NonNullable<Shift["map"]>;
  assets: AssetState[];
  onPick?: (assetId: string) => void;
  selected?: string | null;
  /** Waypoints in the order SPOT visits them; drawn lightly. */
  route?: string[];
  /** How many points of `route` SPOT has already passed (drawn stronger). */
  done?: number;
  robotAt?: string | null;
  /** How long SPOT takes to move to the next point (Shift Run plays slower); the plan's own 700 ms when not given. */
  robotMs?: number;
  /** Shift Run: SPOT is drawn as a small walking robot dog, facing the way it goes and standing still at each stop. */
  dog?: boolean;
  /** Round view: the equipment marks step back so the route reads first. */
  quiet?: boolean;
  title: string;
  note?: ReactNode;
  initialFloor?: FloorView | null;
  /** Planning: the stop number to draw on a piece of equipment. */
  order?: Record<string, number>;
  /** A second route to compare with (drawn as a thin dashed line). */
  route2?: string[];
  routeLabel?: string;
  /** The look guide's name for this plan ("map"): its floor buttons, marks and places are tagged under it. */
  guide?: string;
  /** The look guide's walk on the traced floor: drawn and kept, with the start marked "You are here". */
  walk?: SpotRoute | null;
  /** Where the operator said they are, before any walk is planned (normalised on the picture). */
  hereAt?: { floor: number; x: number; y: number } | null;
  /** The next tap on the plan sets "You are here": it is snapped to walkable floor, or null comes back when it is nowhere near any. */
  onTap?: ((at: { floor: number; x: number; y: number; moved: number } | null) => void) | null;
  /** Lay the traced walkable area over the picture (faint). */
  showFloor?: boolean;
  children?: ReactNode;
}) {
  const traced = useTraced();
  const geo = useMemo(() => geometry(map), [map]);
  const floors = Object.keys(geo.panels).map(Number).sort();
  const [view, setView] = useState<FloorView>(initialFloor ?? floors[0] ?? 1);
  const [width, setWidth] = useState(700);
  const wrap = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const measure = () => setWidth(el.clientWidth || 700);
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // follow SPOT to the other floor while a round plays
  const robotFloor = robotAt ? geo.pos[robotAt]?.floor ?? null : null;
  useEffect(() => {
    if (robotFloor && view !== "both" && robotFloor !== view) setView(robotFloor);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [robotFloor]);

  const pad = 14;
  const panel = view === "both" ? null : geo.panels[view as number];
  const box: Rect = panel ? { x: panel.x - pad, y: panel.y - pad, w: panel.w + 2 * pad, h: panel.h + 2 * pad } : { x: 0, y: 0, w: geo.w, h: geo.h };
  const u = box.w / Math.max(width, 240);
  const roomy = width >= 620;
  const visible = (loc: string) => Boolean(geo.pos[loc]) && (view === "both" || geo.pos[loc]?.floor === view);

  const byHome = useMemo(() => {
    const m: Record<string, AssetState[]> = {};
    for (const a of assets) (m[a.location_id] ??= []).push(a);
    return m;
  }, [assets]);
  const count = (f: number) => assets.filter((a) => a.floor === f && a.level === "act").length;

  // each piece of equipment gets its own spot around its waypoint's pin
  const spots = useMemo(() => {
    const out: { a: AssetState; x: number; y: number; side: number }[] = [];
    const gap = (roomy ? 34 : 30) * u;
    for (const [loc, list] of Object.entries(byHome)) {
      const p = geo.pos[loc];
      if (!p) continue;
      const perRow = list.length > 3 ? 2 : list.length;
      list.forEach((a, i) => {
        const row = Math.floor(i / perRow);
        const inRow = Math.min(perRow, list.length - row * perRow);
        const col = i - row * perRow;
        out.push({ a, x: p.x + (col - (inRow - 1) / 2) * gap, y: p.y + row * gap, side: Math.sign(col - (inRow - 1) / 2) });
      });
    }
    return out;
  }, [byHome, geo, u, roomy]);

  // a leg between two waypoints: along the traced floor when the box has the trace (the part on the floor in view), else a straight line
  const onTrace = Boolean(geo.image) && Object.keys(traced.legs).length > 0;
  const shown = (f: number) => view === "both" || f === view;
  const line = (points: number[][]) => points.map((p) => `${(p[0]! * geo.w).toFixed(1)},${(p[1]! * geo.h).toFixed(1)}`).join(" ");
  const legLines = (a: string, b: string): string[] | null => {
    const leg = onTrace ? legBetween(traced.legs, a, b) : null;
    return leg ? leg.floors.filter((f) => shown(f.floor) && f.points.length > 1).map((f) => line(f.points)) : null;
  };
  const pts = (route ?? []).filter((id) => geo.pos[id]);
  const segs: { a: string; b: string; passed: boolean; lines: string[] | null }[] = [];
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1]!;
    const b = pts[i]!;
    const lines = a === b ? null : legLines(a, b);
    if (lines ? lines.length > 0 : visible(a) && visible(b)) segs.push({ a, b, passed: done === undefined ? false : i < done, lines });
  }
  const pts2 = (route2 ?? []).filter((id) => geo.pos[id]);
  const segs2: { a: string; b: string; lines: string[] | null }[] = [];
  for (let i = 1; i < pts2.length; i++) {
    const a = pts2[i - 1]!;
    const b = pts2[i]!;
    if (a === b) continue;
    const lines = legLines(a, b);
    if (lines ? lines.length > 0 : visible(a) && visible(b)) segs2.push({ a, b, lines });
  }
  const robot = robotAt && visible(robotAt) ? geo.pos[robotAt] : undefined;
  // the dog walks while it moves to the next stop and stands while that stop's check is shown; it faces the way it goes
  const [gait, setGait] = useState<{ walking: boolean; left: boolean }>({ walking: false, left: false });
  const lastAt = useRef<{ x: number; y: number } | null>(null);
  const rx = robot?.x;
  const ry = robot?.y;
  useEffect(() => {
    if (!dog || rx === undefined || ry === undefined) return; // off the floor in view for a moment: where it last stood is kept
    const from = lastAt.current;
    lastAt.current = { x: rx, y: ry };
    if (!from || (Math.abs(from.x - rx) < 0.5 && Math.abs(from.y - ry) < 0.5)) return;
    setGait((g) => ({ walking: true, left: Math.abs(from.x - rx) < 0.5 ? g.left : rx < from.x }));
    const t = window.setTimeout(() => setGait((g) => ({ ...g, walking: false })), robotMs ?? 700);
    return () => window.clearTimeout(t);
  }, [dog, rx, ry, robotMs]);

  // the look guide's walk: its parts on the floors in view, every point tagged for the pointer, and where it starts
  const routed = geo.image && walk?.found && walk.routed ? walk.routed : null;
  let nth = 0;
  const walkParts = (routed?.floors ?? []).map((f: WalkFloor) => ({ ...f, first: (nth += f.points.length) - f.points.length }));
  const startAt = routed ? routed.start : geo.image && hereAt ? hereAt : null;
  const floorData = geo.image && showFloor && traced.floor ? Object.entries(traced.floor.floors).filter(([f]) => shown(Number(f))) : [];

  const svg = useRef<SVGSVGElement>(null);
  const tap = (e: React.MouseEvent) => {
    if (!onTap || !svg.current || !traced.floor) return;
    const m = svg.current.getScreenCTM();
    if (!m) return;
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse());
    const x = p.x / geo.w;
    const y = p.y / geo.h;
    const grids = Object.fromEntries(Object.entries(traced.floor.floors).map(([f, v]) => [f, v.grid]));
    const f = floorAt(grids, x, y);
    const at = f !== null && shown(f) ? snapToFloor(grids[String(f)]!, x, y, { w: geo.w, h: geo.h }) : null;
    onTap(at && f !== null ? { floor: f, ...at } : null);
  };

  return (
    <section className={`${o.card} ${o.mapCard}`} aria-label={title} data-guide={guide}>
      <div className={o.cardHead}>
        <h2>{title}</h2>
        {floors.length > 1 ? (
          <div className={o.floors} role="group" aria-label="Floor">
            {floors.map((f) => (
              <button type="button" key={f} data-guide={guide ? `${guide}:floor:${f}` : undefined} className={`${o.floorBtn} ${view === f ? o.floorOn : ""}`} aria-pressed={view === f} onClick={() => setView(f)}>
                Floor {f}
                {!quiet && count(f) > 0 ? <span className={o.floorCount} aria-label={`${count(f)} need you`}>{count(f)}</span> : null}
              </button>
            ))}
            <button type="button" data-guide={guide ? `${guide}:floor:both` : undefined} className={`${o.floorBtn} ${view === "both" ? o.floorOn : ""}`} aria-pressed={view === "both"} onClick={() => setView("both")}>
              Both
            </button>
          </div>
        ) : null}
      </div>

      <div className={o.mapWrap} ref={wrap}>
        <svg ref={svg} viewBox={`${box.x} ${box.y} ${box.w} ${box.h}`} role="group" aria-label="Floor plan with one mark per piece of equipment" onClick={onTap ? tap : undefined} style={onTap ? { cursor: "crosshair" } : undefined} data-tap={onTap ? "on" : undefined}>
          {geo.image ? (
            <>
              <image className={o.planImage} href={geo.image} x="0" y="0" width={geo.w} height={geo.h} preserveAspectRatio="none" />
              <rect x="0" y="0" width={geo.w} height={geo.h} fill="var(--surface)" opacity="0.66" />
            </>
          ) : (
            Object.entries(geo.panels).map(([f, p]) => (
              <g key={f}>
                <rect x={p.x + 4} y={p.y + 4} width={p.w - 8} height={p.h - 8} rx="14" fill="var(--surface-2)" stroke="var(--line)" strokeWidth={1.5 * u} />
                <text x={p.x + 20} y={p.y + 34} fontSize={15 * u} fontWeight="700" fill="var(--muted)">
                  FLOOR {f}
                </text>
              </g>
            ))
          )}

          {floorData.map(([f, fl]) => (
            <g key={`floor${f}`} data-walkable={f} pointerEvents="none">
              {fl.grid.runs.map((r, i) => (
                <rect key={i} x={(fl.grid.x0 + r[1]! * fl.grid.cell_x) * geo.w} y={(fl.grid.y0 + r[0]! * fl.grid.cell_y) * geo.h} width={(r[2]! - r[1]! + 1) * fl.grid.cell_x * geo.w} height={fl.grid.cell_y * geo.h + 0.4} fill="var(--ok)" opacity="0.2" />
              ))}
              {fl.obstacles.map((ob, i) => (
                <rect key={`o${i}`} x={ob.rect[0] * geo.w} y={ob.rect[1] * geo.h} width={(ob.rect[2] - ob.rect[0]) * geo.w} height={(ob.rect[3] - ob.rect[1]) * geo.h} fill="var(--muted)" fillOpacity="0.16" stroke="var(--muted)" strokeWidth={1 * u} />
              ))}
              {fl.doors.map((d, i) => (
                <rect key={`d${i}`} x={d.rect[0] * geo.w} y={d.rect[1] * geo.h} width={(d.rect[2] - d.rect[0]) * geo.w} height={(d.rect[3] - d.rect[1]) * geo.h} fill="none" stroke="var(--watch)" strokeWidth={1.6 * u} strokeDasharray={d.seen ? undefined : `${3 * u} ${3 * u}`} />
              ))}
              {fl.connectors.map((c) => (
                <rect key={c.id} x={c.x * geo.w - 5 * u} y={c.y * geo.h - 5 * u} width={10 * u} height={10 * u} rx={2 * u} fill="var(--accent-text)" stroke="var(--surface)" strokeWidth={1.5 * u} />
              ))}
            </g>
          ))}

          {segs.map((sg, i) =>
            sg.lines ? (
              sg.lines.map((points, k) => (
                <polyline key={`${i}-${k}`} points={points} fill="none" stroke="var(--accent-text)" strokeWidth={(sg.passed ? 3.5 : 2.5) * u} strokeDasharray={sg.passed ? undefined : `${2 * u} ${7 * u}`} strokeLinecap="round" strokeLinejoin="round" opacity={sg.passed ? 0.85 : 0.6} />
              ))
            ) : (
              <line
                key={i}
                x1={geo.pos[sg.a]!.x}
                y1={geo.pos[sg.a]!.y}
                x2={geo.pos[sg.b]!.x}
                y2={geo.pos[sg.b]!.y}
                stroke="var(--accent-text)"
                strokeWidth={(sg.passed ? 3.5 : 2.5) * u}
                strokeDasharray={sg.passed ? undefined : `${2 * u} ${7 * u}`}
                strokeLinecap="round"
                opacity={sg.passed ? 0.85 : 0.6}
              />
            ),
          )}

          {segs2.map((sg, i) =>
            sg.lines ? (
              sg.lines.map((points, k) => <polyline key={`b${i}-${k}`} points={points} transform={`translate(${3 * u} ${3 * u})`} fill="none" stroke="var(--ok)" strokeWidth={2.5 * u} strokeDasharray={`${8 * u} ${6 * u}`} strokeLinecap="round" strokeLinejoin="round" opacity="0.8" />)
            ) : (
              <line key={`b${i}`} x1={geo.pos[sg.a]!.x + 5 * u} y1={geo.pos[sg.a]!.y + 5 * u} x2={geo.pos[sg.b]!.x + 5 * u} y2={geo.pos[sg.b]!.y + 5 * u} stroke="var(--ok)" strokeWidth={2.5 * u} strokeDasharray={`${8 * u} ${6 * u}`} strokeLinecap="round" opacity="0.8" />
            ),
          )}

          {walkParts.filter((f) => shown(f.floor)).map((f) => (
            <g key={`walk${f.floor}`} data-walk-floor={f.floor} data-walked={f.walked ? "yes" : "no"} pointerEvents="none">
              <polyline points={line(f.points)} fill="none" stroke="var(--surface-solid)" strokeWidth={7 * u} strokeLinecap="round" strokeLinejoin="round" opacity="0.9" />
              <polyline points={line(f.points)} fill="none" stroke="var(--danger)" strokeWidth={3.5 * u} strokeDasharray={f.walked ? undefined : `${9 * u} ${7 * u}`} strokeLinecap="round" strokeLinejoin="round" />
              {f.points.map((p, i) => (
                <circle key={i} data-guide={`walk:${f.first + i}`} cx={p[0]! * geo.w} cy={p[1]! * geo.h} r={2 * u} fill="none" />
              ))}
            </g>
          ))}
          {routed && routed.floors.length > 1
            ? routed.floors.filter((f) => shown(f.floor)).map((f, i) => {
                const p = routed.floors[0] === f ? f.points[f.points.length - 1]! : f.points[0]!;
                return (
                  <g key={`st${i}`} transform={`translate(${p[0]! * geo.w} ${p[1]! * geo.h})`} pointerEvents="none">
                    <rect x={-7 * u} y={-7 * u} width={14 * u} height={14 * u} rx={3 * u} fill="var(--danger)" stroke="var(--surface)" strokeWidth={2 * u} />
                    <text className={o.placeLabel} y={-11 * u} textAnchor="middle" fontSize={11.5 * u} strokeWidth={3.5 * u}>
                      {routed.connector_name?.replace(/^the /, "") ?? "stairs"}
                    </text>
                  </g>
                );
              })
            : null}
          {startAt && shown(startAt.floor) ? (
            <g transform={`translate(${startAt.x * geo.w} ${startAt.y * geo.h})`} data-here pointerEvents="none">
              <circle r={9 * u} fill="var(--accent-text)" stroke="var(--surface)" strokeWidth={2.5 * u} />
              <circle r={3 * u} fill="var(--on-media)" />
              <text className={o.placeLabel} y={22 * u} textAnchor="middle" fontSize={12 * u} strokeWidth={3.5 * u} style={{ fill: "var(--accent-text)", fontWeight: 800 }}>
                You are here
              </text>
            </g>
          ) : null}

          {map.places.filter((l) => visible(l.location_id)).map((l) => {
            const p = geo.pos[l.location_id]!;
            const n = byHome[l.location_id]?.length ?? 0;
            const rows = n > 3 ? Math.ceil(n / 2) : n ? 1 : 0;
            const empty = n === 0;
            return (
              <g key={l.location_id} data-guide={guide === "map" ? `place:${l.location_id}` : undefined}>
                {guide === "map" ? <circle data-guide={`at:${l.location_id}`} cx={p.x} cy={p.y} r={2 * u} fill="none" /> : null}
                {empty ? <circle cx={p.x} cy={p.y} r={4 * u} fill="var(--surface)" stroke="var(--muted)" strokeWidth={1.5 * u} /> : null}
                {roomy ? (
                  <text className={o.placeLabel} x={p.x} y={p.y + ((rows > 0 ? (rows - 1) * 34 + 16 : 6) + 14) * u} textAnchor="middle" fontSize={11.5 * u} strokeWidth={3.5 * u}>
                    {l.name}
                  </text>
                ) : null}
              </g>
            );
          })}

          {spots
            .filter((sp) => visible(sp.a.location_id))
            .sort((a, b) => (a.a.level === "act" ? 1 : 0) - (b.a.level === "act" ? 1 : 0))
            .map(({ a, x, y }) => {
              const on = selected === a.asset_id;
              const pick = () => onPick?.(a.asset_id);
              return (
                <g
                  key={a.asset_id}
                  data-guide={guide === "map" ? `asset:${a.asset_id}` : guide === "shiftrun:map" ? `run-asset:${a.asset_id}` : undefined}
                  className={`${o.dot} ${quiet ? o.dotQuiet : ""}`}
                  transform={`translate(${x} ${y})`}
                  tabIndex={onPick ? 0 : undefined}
                  role={onPick ? "button" : "img"}
                  aria-label={`${a.label}: ${a.level_words}`}
                  onClick={pick}
                  onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), pick())}
                >
                  <title>{`${a.label}: ${a.level_words}`}</title>
                  <circle r={22 * u} fill="transparent" />
                  {a.level === "act" && !quiet ? <circle className={o.ring} r={17 * u} fill="none" stroke="var(--bad)" strokeWidth={2 * u} /> : null}
                  {on ? <circle r={19 * u} fill="none" stroke="var(--accent-text)" strokeWidth={3 * u} /> : null}
                  <LevelMark level={a.level} u={u} small={quiet} />
                  {order?.[a.asset_id] ? (
                    <g transform={`translate(${12 * u} ${-12 * u})`}>
                      <circle r={9.5 * u} fill="var(--accent-text)" stroke="var(--surface)" strokeWidth={2 * u} />
                      <text y={4 * u} textAnchor="middle" fontSize={11.5 * u} fontWeight="800" fill="var(--on-media)">
                        {order[a.asset_id]}
                      </text>
                    </g>
                  ) : null}
                </g>
              );
            })}

          {!quiet && !order
            ? spots
                .filter((sp) => sp.a.level === "act" && visible(sp.a.location_id))
                .map(({ a, x, y, side }) => (
                  <text key={a.asset_id} className={o.actLabel} x={x + side * 12 * u} y={y - 20 * u} textAnchor={side < 0 ? "end" : side > 0 ? "start" : "middle"} fontSize={12.5 * u} strokeWidth={4 * u}>
                    {a.asset_id}
                  </text>
                ))
            : null}

          {robot ? (
            <g className={o.robot} style={{ transform: `translate(${robot.x}px, ${robot.y - 26 * u}px)`, transitionDuration: robotMs ? `${robotMs}ms` : undefined }} aria-label="SPOT is here">
              <path d={`M ${-5 * u} ${11 * u} L 0 ${19 * u} L ${5 * u} ${11 * u} Z`} fill="var(--accent-text)" />
              {dog ? (
                <g className={`${o.dog} ${gait.walking ? o.dogWalk : ""}`} transform={`scale(${(gait.left ? -1 : 1) * u} ${u})`} data-dog={gait.walking ? "walking" : "standing"}>
                  <g stroke="var(--accent-text)" strokeWidth="3.4" strokeLinecap="round">
                    <line className={o.dogLegA} x1="-11" y1="-3" x2="-11" y2="9" />
                    <line className={o.dogLegB} x1="-7" y1="-3" x2="-7" y2="9" />
                    <line className={o.dogLegB} x1="7" y1="-3" x2="7" y2="9" />
                    <line className={o.dogLegA} x1="11" y1="-3" x2="11" y2="9" />
                  </g>
                  <rect x="-15" y="-11" width="30" height="10" rx="4" fill="var(--accent-text)" stroke="var(--surface)" strokeWidth="1.5" />
                  <rect x="12" y="-18" width="11" height="8" rx="3" fill="var(--accent-text)" stroke="var(--surface)" strokeWidth="1.5" />
                  <circle cx="19.5" cy="-14.5" r="1.3" fill="var(--on-media)" />
                </g>
              ) : (
                <>
              <rect x={-22 * u} y={-11 * u} width={44 * u} height={22 * u} rx={11 * u} fill="var(--accent-text)" stroke="var(--surface)" strokeWidth={2 * u} />
              <text x="0" y={4.2 * u} textAnchor="middle" fontSize={12 * u} fontWeight="800" fill="var(--on-media)">
                SPOT
              </text>
                </>
              )}
            </g>
          ) : null}
        </svg>
      </div>

      <div className={o.legend}>
        {(quiet ? [] : (["act", "watch", "ok"] as Level[])).map((lv) => (
          <span key={lv}>
            <LevelIcon level={lv} size={20} />
            {lv === "act" ? "Needs you" : lv === "watch" ? "Keep an eye" : "Looks fine"}
            <b>{assets.filter((a) => a.level === lv).length}</b>
          </span>
        ))}
        {pts.length > 1 ? (
          <span>
            <i className={o.routeKey} /> {routeLabel ?? (done === undefined ? "SPOT's last route" : "SPOT's route")}
          </span>
        ) : null}
        {routed && routed.floors.length > 0 ? (
          <>
            <span data-legend="walked">
              <i className={o.routeKey} style={{ borderTop: "3px solid var(--danger)" }} /> The way there: SPOT has walked this leg
            </span>
            <span data-legend="traced">
              <i className={o.routeKey} style={{ borderTop: "3px dashed var(--danger)" }} /> from the traced plan only
            </span>
          </>
        ) : null}
      </div>
      {note}
      {children}
    </section>
  );
}
