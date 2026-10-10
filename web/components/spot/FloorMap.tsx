"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { spotUrl } from "@/lib/api";
import o from "./op.module.css";
import type { AssetState, Level, Shift } from "./types";

/**
 * The floor plan for the person on shift: one mark per piece of equipment, in three levels that differ by shape as well as
 * colour (a round "!" needs you, a triangle is one to keep an eye on, a small tick is fine), SPOT's route drawn lightly, and
 * SPOT itself when a round is being replayed.
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
  quiet,
  title,
  note,
  initialFloor,
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
  /** Round view: the equipment marks step back so the route reads first. */
  quiet?: boolean;
  title: string;
  note?: ReactNode;
  initialFloor?: FloorView | null;
  children?: ReactNode;
}) {
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

  const pts = (route ?? []).filter((id) => geo.pos[id]);
  const segs: { a: string; b: string; passed: boolean }[] = [];
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1]!;
    const b = pts[i]!;
    if (visible(a) && visible(b)) segs.push({ a, b, passed: done === undefined ? false : i < done });
  }
  const robot = robotAt && visible(robotAt) ? geo.pos[robotAt] : undefined;

  return (
    <section className={`${o.card} ${o.mapCard}`} aria-label={title}>
      <div className={o.cardHead}>
        <h2>{title}</h2>
        {floors.length > 1 ? (
          <div className={o.floors} role="group" aria-label="Floor">
            {floors.map((f) => (
              <button type="button" key={f} className={`${o.floorBtn} ${view === f ? o.floorOn : ""}`} aria-pressed={view === f} onClick={() => setView(f)}>
                Floor {f}
                {!quiet && count(f) > 0 ? <span className={o.floorCount} aria-label={`${count(f)} need you`}>{count(f)}</span> : null}
              </button>
            ))}
            <button type="button" className={`${o.floorBtn} ${view === "both" ? o.floorOn : ""}`} aria-pressed={view === "both"} onClick={() => setView("both")}>
              Both
            </button>
          </div>
        ) : null}
      </div>

      <div className={o.mapWrap} ref={wrap}>
        <svg viewBox={`${box.x} ${box.y} ${box.w} ${box.h}`} role="group" aria-label="Floor plan with one mark per piece of equipment">
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

          {segs.map((sg, i) => (
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
          ))}

          {map.places.filter((l) => visible(l.location_id)).map((l) => {
            const p = geo.pos[l.location_id]!;
            const n = byHome[l.location_id]?.length ?? 0;
            const rows = n > 3 ? Math.ceil(n / 2) : n ? 1 : 0;
            const empty = n === 0;
            return (
              <g key={l.location_id}>
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
                </g>
              );
            })}

          {!quiet
            ? spots
                .filter((sp) => sp.a.level === "act" && visible(sp.a.location_id))
                .map(({ a, x, y, side }) => (
                  <text key={a.asset_id} className={o.actLabel} x={x + side * 12 * u} y={y - 20 * u} textAnchor={side < 0 ? "end" : side > 0 ? "start" : "middle"} fontSize={12.5 * u} strokeWidth={4 * u}>
                    {a.asset_id}
                  </text>
                ))
            : null}

          {robot ? (
            <g className={o.robot} style={{ transform: `translate(${robot.x}px, ${robot.y - 26 * u}px)` }} aria-label="SPOT is here">
              <path d={`M ${-5 * u} ${11 * u} L 0 ${19 * u} L ${5 * u} ${11 * u} Z`} fill="var(--accent-text)" />
              <rect x={-22 * u} y={-11 * u} width={44 * u} height={22 * u} rx={11 * u} fill="var(--accent-text)" stroke="var(--surface)" strokeWidth={2 * u} />
              <text x="0" y={4.2 * u} textAnchor="middle" fontSize={12 * u} fontWeight="800" fill="var(--on-media)">
                SPOT
              </text>
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
            <i className={o.routeKey} /> {done === undefined ? "SPOT's last route" : "SPOT's route"}
          </span>
        ) : null}
      </div>
      {note}
      {children}
    </section>
  );
}
