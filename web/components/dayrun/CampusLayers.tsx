"use client";

import { memo, useCallback, useEffect, useRef, useState, type CSSProperties, type PointerEvent, type ReactNode } from "react";
import type { PlacePack } from "@/lib/api";
import { LAYERS, clampView, fitView, loadCampusMap, minZoom, scaleBar, toScreen, type CampusMap, type Layer, type MapLabel, type MapTile, type PinKind, type Pt, type View } from "@/lib/campusmap";
import cm from "./campusmap.module.css";

/**
 * The drawn campus: greens, water, roads, footpaths and building footprints
 * from the map the box serves (© OpenStreetMap contributors), under whatever a
 * screen wants to put on top (routes, pins, the Day Run figure).
 *
 * The map is in metres. It is cut into tiles whose path strings never change,
 * so panning and zooming only changes one transform and which tiles are
 * mounted; footpaths and service roads are left out while the whole campus is
 * in view, where they would only be noise.
 */

const LAYER_CLASS: Record<Layer, string | undefined> = {
  wood: cm.wood,
  green: cm.green,
  parking: cm.parking,
  pitch: cm.pitch,
  water: cm.water,
  pool: cm.pool,
  plaza: cm.plaza,
  service: cm.service,
  minor: cm.minor,
  major: cm.major,
  foot: cm.foot,
  steps: cm.steps,
  building: cm.building,
};
const FINE: readonly Layer[] = ["service", "foot", "steps"];
const DETAIL_ZOOM = 0.24; // pixels per metre from which footpaths and service roads are drawn

const Tile = memo(function Tile({ tile, detail }: { tile: MapTile; detail: boolean }) {
  return (
    <>
      {LAYERS.map((layer) => (tile.d[layer] && (detail || !FINE.includes(layer)) ? <path key={layer} className={LAYER_CLASS[layer]} d={tile.d[layer]} /> : null))}
    </>
  );
});

/** Find the map of the first pack that has one, and load it (once per page load). */
export function useCampusMap(packs: PlacePack[] | undefined, only?: Set<string>): { map: CampusMap | null; pack: string | null; failed: boolean } {
  const pack = packs?.find((p) => p.map && (!only || only.has(p.name)));
  const path = pack?.map;
  const [state, setState] = useState<{ path: string; map: CampusMap | null; failed: boolean }>({ path: "", map: null, failed: false });
  useEffect(() => {
    if (!path) return;
    let live = true;
    loadCampusMap(path).then(
      (map) => live && setState({ path, map, failed: false }),
      () => live && setState({ path, map: null, failed: true }),
    );
    return () => {
      live = false;
    };
  }, [path]);
  const fresh = path !== undefined && state.path === path;
  return { map: fresh ? state.map : null, pack: pack?.name ?? null, failed: fresh && state.failed };
}

export interface Highlight {
  /** Building ids that are part of the plan: filled in the accent colour. */
  on: Set<number>;
  /** Building ids drawn as closed. */
  closed?: Set<number>;
}

const CHAR = 5.1; // rough width of one 9.5px character, for keeping labels apart

export interface ScreenBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

const overlap = (a: ScreenBox, b: ScreenBox) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;

/** Small boxes along a line on screen, so names can stay off a route. */
export function lineBoxes(view: View, width: number, height: number, points: Pt[], half = 5): ScreenBox[] {
  const out: ScreenBox[] = [];
  for (let i = 1; i < points.length; i++) {
    const a = toScreen(view, width, height, points[i - 1] as Pt);
    const b = toScreen(view, width, height, points[i] as Pt);
    const steps = Math.max(1, Math.ceil(Math.hypot(b.x - a.x, b.y - a.y) / 8));
    for (let n = 0; n <= steps; n++) {
      const x = a.x + ((b.x - a.x) * n) / steps;
      const y = a.y + ((b.y - a.y) * n) / steps;
      if (x > -10 && x < width + 10 && y > -10 && y < height + 10) out.push({ x0: x - half, y0: y - half, x1: x + half, y1: y + half });
    }
  }
  return out;
}

export interface PinSpot {
  key: string;
  x: number;
  y: number;
  big: boolean;
  text: string;
  /** False for a pin that is drawn without a name at this zoom. */
  named: boolean;
}

/**
 * Where each pin's name goes so that no two sit on each other: under the pin (true), above it (false), or
 * left out (null). Pins the plan touches choose first. Every pin and every placed name is added to `taken`.
 */
export function layoutPins(spots: PinSpot[], taken: ScreenBox[]): Map<string, boolean | null> {
  const out = new Map<string, boolean | null>();
  const order = [...spots].sort((a, b) => Number(b.big) - Number(a.big));
  for (const s of order) {
    const r = s.big ? 9 : 6.2;
    taken.push({ x0: s.x - r, y0: s.y - r, x1: s.x + r, y1: s.y + r });
  }
  for (const s of order) {
    const r = s.big ? 9 : 6.2;
    const w = s.text.length * (s.big ? 6.1 : 5.2);
    let below: boolean | null = null;
    if (s.named) {
      const under = { x0: s.x - w / 2, y0: s.y + r + 2, x1: s.x + w / 2, y1: s.y + r + 14 };
      const over = { x0: s.x - w / 2, y0: s.y - r - 15, x1: s.x + w / 2, y1: s.y - r - 3 };
      if (!taken.some((o) => overlap(under, o))) {
        below = true;
        taken.push(under);
      } else if (!taken.some((o) => overlap(over, o))) {
        below = false;
        taken.push(over);
      }
    }
    out.set(s.key, below);
  }
  return out;
}

function labelText(l: MapLabel, z: number): string {
  const full = l.name.length > 26 ? `${l.name.slice(0, 25).trimEnd()}…` : l.name;
  if (l.short && (z < 0.75 || l.name.length * CHAR > l.size * z * 1.6)) return l.short;
  return full;
}

/**
 * Everything of the base map, for an <svg> of `width` by `height` user units.
 * `quiet` are building ids whose label a pin already carries.
 */
export function CampusLayers({
  map,
  view,
  width,
  height,
  highlight,
  quiet,
  taken,
}: {
  map: CampusMap;
  view: View;
  width: number;
  height: number;
  highlight?: Highlight;
  quiet?: Set<number>;
  /** Screen boxes a label must stay clear of (pins and their names). */
  taken?: ScreenBox[];
}) {
  const { cx, cy, z } = view;
  const halfW = width / 2 / z;
  const halfH = height / 2 / z;
  const seen = { x0: cx - halfW, y0: cy - halfH, x1: cx + halfW, y1: cy + halfH };
  const detail = z >= DETAIL_ZOOM;
  const tiles = map.tiles.filter((t) => t.box.x1 >= seen.x0 && t.box.x0 <= seen.x1 && t.box.y1 >= seen.y0 && t.box.y0 <= seen.y1);
  // Stroke widths are set in pixels (the paths do not scale their strokes): a road keeps a believable width at any zoom.
  const widths = {
    "--w-major": `${Math.max(1.6, Math.min(16, 11 * z)).toFixed(2)}px`,
    "--w-minor": `${Math.max(1.1, Math.min(11, 7 * z)).toFixed(2)}px`,
    "--w-service": `${Math.max(0.6, Math.min(7, 4.5 * z)).toFixed(2)}px`,
    "--w-foot": `${Math.max(0.7, Math.min(4, 2.4 * z)).toFixed(2)}px`,
  } as CSSProperties;

  // Names: buildings big enough on screen, largest first, none on top of another or of a pin.
  // The compass, the scale bar and the credit are never written over either.
  const boxes: ScreenBox[] = [
    ...(taken ?? []),
    { x0: width - 36, y0: 4, x1: width, y1: 38 },
    { x0: 0, y0: height - 26, x1: 112, y1: height },
    { x0: width - 136, y0: height - 23, x1: width, y1: height },
  ];
  const names: { l: MapLabel; x: number; y: number; text: string }[] = [];
  for (const l of map.labels) {
    if (l.at.x < seen.x0 || l.at.x > seen.x1 || l.at.y < seen.y0 || l.at.y > seen.y1) continue;
    if (l.building !== undefined && quiet?.has(l.building)) continue;
    const px = l.size * z;
    if (l.kind === "water" ? px < 46 : px < (l.short ? 30 : 44)) continue;
    const text = labelText(l, z);
    const s = toScreen(view, width, height, l.at);
    const w = text.length * CHAR;
    const box = { x0: s.x - w / 2 - 2, y0: s.y - 7, x1: s.x + w / 2 + 2, y1: s.y + 5 };
    if (box.x0 < 2 || box.x1 > width - 2 || box.y0 < 2 || box.y1 > height - 2) continue;
    if (boxes.some((b) => overlap(box, b))) continue;
    boxes.push(box);
    names.push({ l, x: s.x, y: s.y, text });
    if (names.length >= 28) break;
  }

  return (
    <g>
      <rect className={cm.land} x={0} y={0} width={width} height={height} />
      <g transform={`translate(${(width / 2 - cx * z).toFixed(2)} ${(height / 2 - cy * z).toFixed(2)}) scale(${z.toFixed(4)})`} style={widths}>
        <rect className={cm.ground} x={0} y={0} width={map.width} height={map.height} />
        {tiles.map((t) => (
          <Tile key={t.key} tile={t} detail={detail} />
        ))}
        {highlight
          ? [...highlight.on].map((id) => {
              const b = map.buildings.get(id);
              return b ? <path key={id} className={highlight.closed?.has(id) ? cm.buildingClosed : cm.buildingOn} d={b.d} /> : null;
            })
          : null}
      </g>
      {names.map(({ l, x, y, text }) => (
        <text key={l.key} className={l.kind === "water" ? cm.waterName : cm.buildingName} x={x} y={y} textAnchor="middle">
          {text}
        </text>
      ))}
    </g>
  );
}

/** North arrow, scale bar and the map's credit, in the corners of a `width` by `height` map. */
export function MapFurniture({ map, view, width, height, bottom = 0 }: { map: CampusMap; view: View; width: number; height: number; bottom?: number }) {
  const bar = scaleBar(view.z);
  const y = height - 9 - bottom;
  return (
    <g className={cm.furniture}>
      <g transform={`translate(${width - 19} 21)`} aria-label="North is up">
        <circle className={cm.compass} r={12} />
        <path className={cm.compassN} d="M0 -8.5 L4.2 3.5 L0 1 L-4.2 3.5 Z" />
        <text className={cm.compassText} x={0} y={9.6} textAnchor="middle">
          N
        </text>
      </g>
      <g transform={`translate(8 ${y})`} aria-label={`Scale: ${bar.label}`}>
        <rect className={cm.plate} x={-4} y={-13} width={bar.px + 8} height={17} rx={4} />
        <path className={cm.scale} d={`M0 -3 V1 H${bar.px.toFixed(1)} V-3`} />
        <text className={cm.scaleText} x={bar.px / 2} y={-3.5} textAnchor="middle">
          {bar.label}
        </text>
      </g>
      <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer noopener">
        <rect className={cm.plate} x={width - 131} y={y - 10} width={127} height={14} rx={4} />
        <text className={cm.credit} x={width - 8} y={y} textAnchor="end">
          {map.attribution}
        </text>
      </a>
    </g>
  );
}

const GLYPH: Record<PinKind, string> = {
  // drawn in a 12 by 12 box centred on 0,0
  eat: "M-3.2 -4.6 V-1.2 M-1.6 -4.6 V-1.2 M-4.8 -4.6 V-1.6 Q-4.8 0 -3.2 0 Q-1.6 0 -1.6 -1.6 M-3.2 0 V4.8 M3.4 4.8 V-4.6 Q1 -3.4 1.2 0.6 H3.4",
  study: "M0 -3 V4.2 M0 -3 Q-2.6 -4.6 -5 -3.6 V3.6 Q-2.6 2.8 0 4.2 Q2.6 2.8 5 3.6 V-3.6 Q2.6 -4.6 0 -3",
  class: "M-5.4 -1 L0 -3.8 L5.4 -1 L0 1.8 Z M-3 0.6 V3 Q0 5 3 3 V0.6 M5.4 -1 V2.6",
  home: "M-5 0.2 L0 -4.6 L5 0.2 M-3.6 -0.8 V4.4 H3.6 V-0.8 M-1 4.4 V1.4 H1 V4.4",
  gym: "M-5.2 -2.2 V2.2 M-3.4 -3.4 V3.4 M3.4 -3.4 V3.4 M5.2 -2.2 V2.2 M-3.4 0 H3.4",
  park: "M0 4.8 V1.2 M0 1.6 Q-4.6 1.6 -4.2 -1.6 Q-4 -4.8 0 -4.8 Q4 -4.8 4.2 -1.6 Q4.6 1.6 0 1.6",
  place: "M0 -1.6 A1.6 1.6 0 1 0 0.01 -1.6",
};

const PIN_CLASS: Record<PinKind, string | undefined> = { eat: cm.pinEat, study: cm.pinStudy, class: cm.pinClass, home: cm.pinHome, gym: cm.pinGym, park: cm.pinPark, place: cm.pinPlace };

/** A place marker with an icon for what the place is for. `small` is a place the plan does not touch. */
export function PlacePin({ x, y, kind, small = false, closed = false, label, below = true }: { x: number; y: number; kind: PinKind; small?: boolean; closed?: boolean; label?: string; below?: boolean }) {
  const r = small ? 6.2 : 9;
  return (
    <g transform={`translate(${x.toFixed(1)} ${y.toFixed(1)})`}>
      <circle className={`${cm.pin} ${closed ? cm.pinClosed : PIN_CLASS[kind]} ${small ? cm.pinSmall : ""}`} r={r} />
      <path className={cm.pinGlyph} d={GLYPH[kind]} transform={small ? "scale(0.66)" : undefined} />
      {closed ? <path className={cm.pinShut} d={`M${-r - 2} ${-r - 2} L${r + 2} ${r + 2} M${r + 2} ${-r - 2} L${-r - 2} ${r + 2}`} /> : null}
      {label ? (
        <text className={`${cm.pinLabel} ${small ? cm.pinLabelSmall : ""} ${closed ? cm.pinLabelClosed : ""}`} x={0} y={below ? r + 11 : -r - 5} textAnchor="middle">
          {label}
        </text>
      ) : null}
    </g>
  );
}

/** The key under a map: which colour is which kind of place. */
export function PinKey({ kinds }: { kinds: PinKind[] }) {
  const words: Record<PinKind, string> = { eat: "eat", study: "study", class: "class", home: "home", gym: "gym", park: "outdoors", place: "other" };
  return (
    <>
      {kinds.map((k) => (
        <span key={k} className={cm.keyItem}>
          <svg className={cm.keyPin} viewBox="-10 -10 20 20" aria-hidden="true">
            <circle className={`${cm.pin} ${PIN_CLASS[k]}`} r={9} />
            <path className={cm.pinGlyph} d={GLYPH[k]} />
          </svg>
          {words[k]}
        </span>
      ))}
    </>
  );
}

// ---------- a map you can drag and pinch ----------

export interface MapControls {
  view: View;
  /** Spread over the <svg>: pointer handlers and the ref the wheel listener needs. */
  bind: {
    ref: (el: SVGSVGElement | null) => void;
    onPointerDown: (e: PointerEvent<SVGSVGElement>) => void;
    onPointerMove: (e: PointerEvent<SVGSVGElement>) => void;
    onPointerUp: (e: PointerEvent<SVGSVGElement>) => void;
    onPointerCancel: (e: PointerEvent<SVGSVGElement>) => void;
  };
  zoomBy: (factor: number) => void;
  fit: (points: Pt[]) => void;
  /** True once the person has moved the map themselves: the screen stops re-framing it for them. */
  moved: boolean;
  atMin: boolean;
  atMax: boolean;
}

/**
 * Pan with one finger (or the mouse), pinch with two, ctrl+wheel or the buttons to zoom.
 * The view never leaves the campus and never zooms out past all of it. `focus` is framed
 * at first and again whenever `focusKey` changes, until the person moves the map.
 */
export function useMapView(map: CampusMap | null, width: number, height: number, focus: Pt[], focusKey: string): MapControls {
  const [view, setView] = useState<View>({ cx: 0, cy: 0, z: 1 });
  const [moved, setMoved] = useState(false);
  const viewRef = useRef(view);
  viewRef.current = view;
  const el = useRef<SVGSVGElement | null>(null);
  const fingers = useRef(new Map<number, Pt>());
  const focusRef = useRef(focus);
  focusRef.current = focus;

  const put = useCallback(
    (v: View) => {
      if (map) setView(clampView(map, v, width, height));
    },
    [map, width, height],
  );

  useEffect(() => {
    if (!map) return;
    setMoved(false);
    setView(fitView(map, focusRef.current, width, height));
  }, [map, width, height, focusKey]);

  // Client pixels -> the svg's own units (the svg is scaled to the width of its card).
  const local = (e: { clientX: number; clientY: number }): Pt => {
    const box = el.current?.getBoundingClientRect();
    if (!box || !box.width) return { x: width / 2, y: height / 2 };
    return { x: ((e.clientX - box.left) / box.width) * width, y: ((e.clientY - box.top) / box.height) * height };
  };
  const zoomAt = useCallback(
    (p: Pt, factor: number) => {
      const v = viewRef.current;
      if (!map) return;
      const z = Math.max(minZoom(map, width, height), Math.min(4, v.z * factor));
      // keep the point under the finger where it is
      const mx = v.cx + (p.x - width / 2) / v.z;
      const my = v.cy + (p.y - height / 2) / v.z;
      put({ cx: mx - (p.x - width / 2) / z, cy: my - (p.y - height / 2) / z, z });
      setMoved(true);
    },
    [map, width, height, put],
  );

  const ref = useCallback(
    (node: SVGSVGElement | null) => {
      el.current = node;
    },
    [],
  );
  useEffect(() => {
    const node = el.current;
    if (!node) return;
    const onWheel = (e: WheelEvent) => {
      if (!e.ctrlKey && !e.metaKey) return; // a plain wheel scrolls the page
      e.preventDefault();
      zoomAt(local(e), Math.exp(-e.deltaY * 0.01));
    };
    node.addEventListener("wheel", onWheel, { passive: false });
    return () => node.removeEventListener("wheel", onWheel);
    // `local` reads refs only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [zoomAt, map]);

  const onPointerDown = (e: PointerEvent<SVGSVGElement>) => {
    if ((e.target as Element).closest("a, button")) return;
    fingers.current.set(e.pointerId, local(e));
    e.currentTarget.setPointerCapture?.(e.pointerId);
  };
  const onPointerMove = (e: PointerEvent<SVGSVGElement>) => {
    const was = fingers.current.get(e.pointerId);
    if (!was) return;
    const now = local(e);
    const v = viewRef.current;
    if (fingers.current.size === 1) {
      if (Math.abs(now.x - was.x) + Math.abs(now.y - was.y) < 0.5) return;
      put({ ...v, cx: v.cx - (now.x - was.x) / v.z, cy: v.cy - (now.y - was.y) / v.z });
      setMoved(true);
    } else if (fingers.current.size === 2) {
      const other = [...fingers.current.entries()].find(([id]) => id !== e.pointerId)?.[1];
      if (other) {
        const before = Math.hypot(was.x - other.x, was.y - other.y) || 1;
        const after = Math.hypot(now.x - other.x, now.y - other.y) || 1;
        zoomAt({ x: (now.x + other.x) / 2, y: (now.y + other.y) / 2 }, after / before);
      }
    }
    fingers.current.set(e.pointerId, now);
  };
  const onPointerUp = (e: PointerEvent<SVGSVGElement>) => {
    fingers.current.delete(e.pointerId);
  };

  const lo = map ? minZoom(map, width, height) : 0;
  return {
    view,
    bind: { ref, onPointerDown, onPointerMove, onPointerUp, onPointerCancel: onPointerUp },
    zoomBy: (factor) => zoomAt({ x: width / 2, y: height / 2 }, factor),
    fit: (points) => {
      if (!map) return;
      setView(fitView(map, points, width, height));
      setMoved(false);
    },
    moved,
    atMin: view.z <= lo * 1.001,
    atMax: view.z >= 4 * 0.999,
  };
}

/** The buttons over a map: zoom in, zoom out, and back to the route. */
export function MapButtons({ controls, onFit, fitLabel = "Fit route", extra }: { controls: MapControls; onFit: () => void; fitLabel?: string; extra?: ReactNode }) {
  return (
    <div className={cm.buttons}>
      <button type="button" className={cm.mapBtn} onClick={onFit} aria-label={fitLabel}>
        {fitLabel}
      </button>
      {extra}
      <span className={cm.zoomPair}>
        <button type="button" className={cm.mapBtn} onClick={() => controls.zoomBy(1.6)} disabled={controls.atMax} aria-label="Zoom in">
          +
        </button>
        <button type="button" className={cm.mapBtn} onClick={() => controls.zoomBy(1 / 1.6)} disabled={controls.atMin} aria-label="Zoom out">
          −
        </button>
      </span>
    </div>
  );
}
