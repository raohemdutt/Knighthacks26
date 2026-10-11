"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { spotUrl } from "@/lib/api";
import { useOpen } from "./bits";
import s from "./spot.module.css";
import type { AssetRow, Locations, Replay } from "./types";

/**
 * The facility: both floors with a marker per waypoint, coloured by health, the assets that live there (ring = criticality,
 * fill = worst anomaly), an anomaly count, and the robot with its trail during replay.
 *
 * Backdrop: the kit's floor-plan image when the box has it, with markers on the pins (positions read off the image, see
 * fixtures/spot/waypoint_pixels.json). Otherwise, or on request, a schematic drawn from locations.csv x/y. The two do not
 * agree (the CSV coordinates are not the picture's), which is why both are offered.
 */

type View = "both" | 1 | 2;
interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}
interface Geometry {
  w: number;
  h: number;
  panels: Record<number, Rect>;
  pos: Record<string, { x: number; y: number; floor: number }>;
  image: string | null;
}

const SEV_COLOR: Record<string, string> = { CRITICAL: "var(--sev-critical)", HIGH: "var(--sev-high)", MEDIUM: "var(--sev-medium)", LOW: "var(--sev-low)" };
const HEALTH_COLOR: Record<string, string> = { ok: "var(--ok)", watch: "var(--watch)", bad: "var(--bad)", unknown: "var(--unknown)" };
const SEV_ORDER = ["LOW", "MEDIUM", "HIGH", "CRITICAL"];
const PIN_HEAD_PX = 20; // the kit's pins are drawn about this far above their tip; our marker sits on the pin's head

function geometry(data: Locations, usePlan: boolean): Geometry {
  const plan = usePlan ? data.plan : null;
  const pos: Geometry["pos"] = {};
  const panels: Geometry["panels"] = {};
  if (plan) {
    for (const [f, p] of Object.entries(plan.panels)) panels[Number(f)] = { x: p.x0 * plan.width, y: p.y0 * plan.height, w: (p.x1 - p.x0) * plan.width, h: (p.y1 - p.y0) * plan.height };
    for (const l of data.locations) {
      const p = plan.waypoints[l.location_id];
      if (p) pos[l.location_id] = { x: p.x * plan.width, y: p.y * plan.height - PIN_HEAD_PX, floor: l.floor };
    }
    return { w: plan.width, h: plan.height, panels, pos, image: spotUrl(plan.image_url) };
  }
  const floors = [...new Set(data.locations.map((l) => l.floor))].sort();
  const size = 560;
  const gap = 28;
  floors.forEach((f, i) => (panels[f] = { x: i * (size + gap), y: 0, w: size, h: size }));
  for (const l of data.locations) {
    const p = panels[l.floor];
    if (p) pos[l.location_id] = { x: p.x + 50 + (l.x / 100) * (size - 100), y: p.y + 60 + (l.y / 100) * (size - 110), floor: l.floor };
  }
  return { w: floors.length * size + (floors.length - 1) * gap, h: size, panels, pos, image: null };
}

export function FacilityMap({ data, assets, live }: { data: Locations; assets: AssetRow[]; live: Replay | null }) {
  const { open } = useOpen();
  const hasPlan = Boolean(data.plan);
  const [backdrop, setBackdrop] = useState<"plan" | "schematic">(hasPlan ? "plan" : "schematic");
  const [view, setView] = useState<View>("both");
  const [follow, setFollow] = useState(true);
  const [width, setWidth] = useState(900);
  const wrap = useRef<HTMLDivElement>(null);
  const geo = useMemo(() => geometry(data, backdrop === "plan" && hasPlan), [data, backdrop, hasPlan]);

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const measure = () => setWidth(el.clientWidth || 900);
    measure();
    if (window.innerWidth < 700) setView(1); // a phone cannot show both floors at a readable size
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const replaying = Boolean(live && live.index > 0);
  const robotAt = replaying ? live?.robot?.location_id ?? null : null;
  const robotFloor = robotAt ? geo.pos[robotAt]?.floor ?? null : null;
  useEffect(() => {
    if (follow && view !== "both" && robotFloor && robotFloor !== view && live?.playing) setView(robotFloor as View);
  }, [follow, view, robotFloor, live?.playing]);

  const box: Rect = view === "both" ? { x: 0, y: 0, w: geo.w, h: geo.h } : geo.panels[view] ?? { x: 0, y: 0, w: geo.w, h: geo.h };
  const u = box.w / Math.max(width, 200); // viewBox units per CSS pixel
  const r = (width < 500 ? 13 : 15) * u;
  const visible = (id: string) => view === "both" || geo.pos[id]?.floor === view;
  const byHome = useMemo(() => {
    const m: Record<string, AssetRow[]> = {};
    for (const a of assets) (m[a.location_id] ??= []).push(a);
    return m;
  }, [assets]);

  const trail = (replaying ? live?.robot?.trail ?? [] : []).filter((id) => geo.pos[id] && visible(id)).slice(-5);
  const robot = robotAt ? geo.pos[robotAt] : undefined;

  return (
    <div className={s.card}>
      <div className={s.cardHead}>
        <h2>Facility</h2>
        <span className={`${s.small} ${s.muted}`}>{replaying ? "as of the replay position (counts so far)" : "all missions in the data"}</span>
        {hasPlan ? (
          <span className={s.seg} role="group" aria-label="Backdrop">
            <button type="button" aria-pressed={backdrop === "plan"} onClick={() => setBackdrop("plan")}>
              Plan
            </button>
            <button type="button" aria-pressed={backdrop === "schematic"} onClick={() => setBackdrop("schematic")}>
              CSV x/y
            </button>
          </span>
        ) : null}
        <span className={s.seg} role="group" aria-label="Floor">
          <button type="button" aria-pressed={view === "both"} onClick={() => setView("both")}>
            Both
          </button>
          {Object.keys(geo.panels).map((f) => (
            <button type="button" key={f} aria-pressed={view === Number(f)} onClick={() => (setView(Number(f) as View), setFollow(false))}>
              Floor {f}
              {robotFloor === Number(f) ? " •" : ""}
            </button>
          ))}
        </span>
      </div>

      <div className={s.mapWrap} ref={wrap}>
        <svg viewBox={`${box.x} ${box.y} ${box.w} ${box.h}`} role="group" aria-label="Facility map with waypoint health">
          {geo.image ? (
            <>
              <image href={geo.image} x="0" y="0" width={geo.w} height={geo.h} preserveAspectRatio="none" />
              <rect className={s.wash} x="0" y="0" width={geo.w} height={geo.h} fill="#eaf1fc" opacity="0.35" />
            </>
          ) : (
            Object.entries(geo.panels).map(([f, p]) => (
              <g key={f}>
                <rect x={p.x + 4} y={p.y + 4} width={p.w - 8} height={p.h - 8} rx="14" fill="var(--surface-2)" stroke="var(--line)" strokeWidth={1.5 * u} />
                {[0.25, 0.5, 0.75].map((t) => (
                  <g key={t} stroke="var(--line)" strokeWidth={u} opacity="0.5">
                    <line x1={p.x + t * p.w} x2={p.x + t * p.w} y1={p.y + 12} y2={p.y + p.h - 12} />
                    <line x1={p.x + 12} x2={p.x + p.w - 12} y1={p.y + t * p.h} y2={p.y + t * p.h} />
                  </g>
                ))}
                <text x={p.x + 20} y={p.y + 34} fontSize={15 * u} fontWeight="700" fill="var(--muted)">
                  FLOOR {f}
                </text>
              </g>
            ))
          )}

          {trail.length > 1 ? (
            <polyline
              points={trail.map((id) => `${geo.pos[id]!.x},${geo.pos[id]!.y}`).join(" ")}
              fill="none"
              stroke="var(--text)"
              strokeWidth={2.5 * u}
              strokeDasharray={`${6 * u} ${5 * u}`}
              strokeLinecap="round"
              strokeLinejoin="round"
              opacity="0.65"
            />
          ) : null}

          {data.locations.filter((l) => geo.pos[l.location_id] && visible(l.location_id)).map((l) => {
            const p = geo.pos[l.location_id]!;
            const so = live?.by_location?.[l.location_id];
            let color = HEALTH_COLOR[l.health] ?? "var(--unknown)";
            let label = l.enough && l.rate !== null ? `${Math.round(100 * l.rate)}%` : `${l.k}/${l.n}`;
            if (replaying) {
              const k = so?.failed ?? 0;
              const n = k + (so?.ok ?? 0);
              color = n === 0 ? "var(--unknown)" : k === 0 ? "var(--ok)" : k >= 2 && k / n >= 0.34 ? "var(--bad)" : "var(--watch)";
              label = n === 0 ? "" : `${k}/${n}`;
            }
            const home = byHome[l.location_id] ?? [];
            const anomalyCount = replaying ? home.reduce((t, a) => t + (live?.by_asset?.[a.asset_id]?.anomalies ?? 0), 0) : l.anomalies.length;
            const worst = replaying
              ? home.map((a) => live?.by_asset?.[a.asset_id]?.worst).filter((x): x is string => Boolean(x)).sort((a, b) => SEV_ORDER.indexOf(b) - SEV_ORDER.indexOf(a))[0]
              : l.anomalies.map((a) => a.severity).sort((a, b) => SEV_ORDER.indexOf(b) - SEV_ORDER.indexOf(a))[0];
            const here = robotAt === l.location_id;
            return (
              <g
                key={l.location_id}
                className={s.wp}
                tabIndex={0}
                role="button"
                aria-label={`${l.location_id} ${l.location_name}: ${l.k} of ${l.n} actions failed, ${l.anomalies.length} anomalies`}
                onClick={() => open({ kind: "location", id: l.location_id })}
                onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && open({ kind: "location", id: l.location_id })}
              >
                <title>{`${l.location_id} ${l.location_name}: ${l.k} of ${l.n} actions failed${l.enough ? "" : " (not enough data)"}; ${l.anomalies.length} anomaly rows`}</title>
                <circle cx={p.x} cy={p.y} r={r * 1.7} fill="transparent" />
                {home.map((a, i) => {
                  const spread = Math.min(0.62, 1.9 / Math.max(home.length, 1));
                  const angle = -Math.PI / 2 + (i - (home.length - 1) / 2) * spread;
                  const ax = p.x + Math.cos(angle) * (r + 8.5 * u);
                  const ay = p.y + Math.sin(angle) * (r + 8.5 * u);
                  const sev = replaying ? live?.by_asset?.[a.asset_id]?.worst ?? null : a.worst_severity;
                  const ring = { CRITICAL: 2.4, HIGH: 1.6, MEDIUM: 0.8, LOW: 0 }[a.criticality] ?? 0;
                  return (
                    <g key={a.asset_id} onClick={(e) => (e.stopPropagation(), open({ kind: "asset", id: a.asset_id }))}>
                      <title>{`${a.asset_id} ${a.asset_name}: ${a.criticality} criticality, ${a.anomalies} anomaly rows`}</title>
                      <circle className={a.criticality === "CRITICAL" ? s.critRing : s.assetRing} cx={ax} cy={ay} r={5 * u} fill={sev ? SEV_COLOR[sev] : "var(--surface)"} stroke={a.criticality === "CRITICAL" ? "#1b2433" : "#8e9bb0"} strokeWidth={ring * u} />
                      {a.criticality === "CRITICAL" ? <circle className={s.critRing} cx={ax} cy={ay} r={8 * u} fill="none" stroke="#1b2433" strokeWidth={1 * u} /> : null}
                    </g>
                  );
                })}
                <circle className={s.plate} cx={p.x} cy={p.y} r={r + 2.5 * u} fill="#ffffff" opacity="0.9" />
                <circle cx={p.x} cy={p.y} r={r} fill={color} stroke={here ? "#1b2433" : "transparent"} strokeWidth={2.5 * u} />
                <text x={p.x} y={p.y + 3.8 * u} textAnchor="middle" fontSize={10.5 * u} fontWeight="800" fill="#ffffff">
                  {label}
                </text>
                {anomalyCount > 0 ? (
                  <g>
                    <circle cx={p.x + r * 0.95} cy={p.y + r * 0.8} r={7.5 * u} fill={SEV_COLOR[worst ?? "LOW"]} stroke="#ffffff" strokeWidth={1.5 * u} />
                    <text x={p.x + r * 0.95} y={p.y + r * 0.8 + 3.3 * u} textAnchor="middle" fontSize={9 * u} fontWeight="800" fill="#ffffff">
                      {anomalyCount}
                    </text>
                  </g>
                ) : null}
                {!geo.image ? (
                  <text className={s.halo} x={p.x} y={p.y + r + 13 * u} textAnchor="middle" fontSize={11 * u} fontWeight="700" fill="var(--text)" stroke="#eaf1fc" strokeWidth={3 * u} paintOrder="stroke">
                    {l.location_id}
                  </text>
                ) : null}
              </g>
            );
          })}

          {robot && visible(robotAt!) ? (
            <g className={s.robot} style={{ transform: `translate(${robot.x}px, ${robot.y}px)` }}>
              <circle className={s.pulse} r={r + 5 * u} fill="none" stroke="var(--accent-bright)" strokeWidth={2.5 * u} />
              <g transform={`translate(${-r - 4 * u}, ${-r - 12 * u})`}>
                <rect x={-17 * u} y={-9 * u} width={34 * u} height={16 * u} rx={4 * u} fill="var(--accent)" />
                <text x="0" y={3 * u} textAnchor="middle" fontSize={10 * u} fontWeight="800" fill="#ffffff">
                  SPOT
                </text>
              </g>
            </g>
          ) : null}
        </svg>
      </div>

      <div className={s.legend}>
        <span>
          <i className={s.swatch} style={{ background: "var(--ok)" }} /> {replaying ? "no failure so far" : "at or below site failure rate"}
        </span>
        <span>
          <i className={s.swatch} style={{ background: "var(--watch)" }} /> {replaying ? "a failure so far" : "above it"}
        </span>
        <span>
          <i className={s.swatch} style={{ background: "var(--bad)" }} /> {replaying ? "2+ failures, a third or more" : "clearly above (interval excludes it)"}
        </span>
        <span>
          <i className={s.swatch} style={{ background: "var(--unknown)" }} /> {replaying ? "not visited yet" : "under 8 actions"}
        </span>
        <span>
          <i className={s.swatch} style={{ background: "var(--surface)", border: "2px solid #1b2433", outline: "1px solid #1b2433", outlineOffset: 2 }} /> asset (ring = criticality, fill = worst anomaly)
        </span>
        <span>
          <i className={s.swatch} style={{ background: "var(--sev-high)" }} /> anomaly count
        </span>
      </div>
      <p className={`${s.small} ${s.muted}`}>
        {geo.image
          ? "Backdrop: the kit's fictional floor plan; markers sit on its pins. Number in a marker: share of actions failed there. Tap a waypoint for its evidence."
          : hasPlan
            ? "Schematic from locations.csv x and y, as given. These coordinates do not match the plan image (see Assumptions and method)."
            : "Schematic from locations.csv x and y (the kit's floor-plan image is not on this box). Tap a waypoint for its evidence."}
      </p>
    </div>
  );
}
