/**
 * The campus map the box serves (GET /api/places/map/{pack}; built from
 * OpenStreetMap by scripts/build_campus_pack.py), decoded once into things an
 * SVG can draw: path strings in metres, cut into tiles so only what is on
 * screen is drawn, plus the building labels and where each pack place stands.
 *
 * Map data © OpenStreetMap contributors (ODbL). The credit is drawn on every map.
 */

import { api } from "./api";

export interface Pt {
  x: number;
  y: number;
}

export interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

/** The file as the box sends it. Coordinates are integers: origin + n * unit degrees, flat [lat, lon, ...]. */
export interface CampusMapData {
  attribution: string;
  bbox: [number, number, number, number]; // south, west, north, east
  origin: [number, number];
  unit: number;
  buildings: { id: number; name?: string; short?: string; kind: string; c: [number, number]; p: number[] }[];
  areas: { k: string; name?: string; p: number[] }[];
  roads: { k: string; name?: string; p: number[] }[];
  walk: { nodes: number[]; edges: [number, number, number, number[]?][] };
  places: Record<string, { osm: string; name: string; c: [number, number]; doors: number[] }>;
}

export const LAYERS = ["wood", "green", "parking", "pitch", "water", "pool", "plaza", "service", "minor", "major", "foot", "steps", "building"] as const;
export type Layer = (typeof LAYERS)[number];

export interface MapTile {
  key: string;
  box: Box; // what is actually in the tile, which can reach past its cell
  d: Partial<Record<Layer, string>>;
}

export interface MapBuilding {
  id: number;
  name?: string;
  short?: string;
  d: string;
  at: Pt; // a point inside it, for the label
  box: Box;
  ring: Pt[];
}

export interface MapLabel {
  key: string;
  kind: "building" | "water";
  at: Pt;
  name: string;
  short?: string;
  size: number; // metres across, for deciding when it is worth a label
  building?: number;
}

export interface CampusMap {
  attribution: string;
  /** Metres: x east from the west edge, y south from the north edge. */
  width: number;
  height: number;
  project: (lat: number, lon: number) => Pt;
  tiles: MapTile[];
  buildings: Map<number, MapBuilding>;
  labels: MapLabel[];
  /** Pack place id ("union") -> where it is and which building it is (in). */
  places: Map<string, { at: Pt; building?: number; name: string }>;
}

const M_PER_DEG = 111_194.93;
const COLS = 5;
const ROWS = 6;

const n1 = (v: number) => (Math.round(v * 10) / 10).toString();

export function decodeCampusMap(data: CampusMapData): CampusMap {
  const [south, west, north, east] = data.bbox;
  const kx = M_PER_DEG * Math.cos((((south + north) / 2) * Math.PI) / 180);
  const project = (lat: number, lon: number): Pt => ({ x: (lon - west) * kx, y: (north - lat) * M_PER_DEG });
  const width = (east - west) * kx;
  const height = (north - south) * M_PER_DEG;
  const [lat0, lon0] = data.origin;
  const pts = (flat: number[]): Pt[] => {
    const out: Pt[] = [];
    for (let i = 0; i + 1 < flat.length; i += 2) out.push(project(lat0 + (flat[i] as number) * data.unit, lon0 + (flat[i + 1] as number) * data.unit));
    return out;
  };
  const boxOf = (points: Pt[]): Box => {
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    for (const p of points) {
      if (p.x < x0) x0 = p.x;
      if (p.x > x1) x1 = p.x;
      if (p.y < y0) y0 = p.y;
      if (p.y > y1) y1 = p.y;
    }
    return { x0, y0, x1, y1 };
  };
  const pathOf = (points: Pt[], closed: boolean) => (points.length < 2 ? "" : `M${points.map((p) => `${n1(p.x)} ${n1(p.y)}`).join("L")}${closed ? "Z" : ""}`);

  const cells = new Map<string, { box: Box; parts: Partial<Record<Layer, string[]>> }>();
  const add = (layer: Layer, points: Pt[], closed: boolean) => {
    if (points.length < 2) return;
    const b = boxOf(points);
    const col = Math.max(0, Math.min(COLS - 1, Math.floor(((b.x0 + b.x1) / 2 / width) * COLS)));
    const row = Math.max(0, Math.min(ROWS - 1, Math.floor(((b.y0 + b.y1) / 2 / height) * ROWS)));
    const key = `${col}-${row}`;
    let cell = cells.get(key);
    if (!cell) cells.set(key, (cell = { box: { ...b }, parts: {} }));
    cell.box = { x0: Math.min(cell.box.x0, b.x0), y0: Math.min(cell.box.y0, b.y0), x1: Math.max(cell.box.x1, b.x1), y1: Math.max(cell.box.y1, b.y1) };
    (cell.parts[layer] ??= []).push(pathOf(points, closed));
  };

  const labels: MapLabel[] = [];
  for (const a of data.areas) {
    const ring = pts(a.p);
    if ((LAYERS as readonly string[]).includes(a.k)) add(a.k as Layer, ring, true);
    if (a.k === "water" && a.name && /[a-z]{3}/i.test(a.name)) {
      const b = boxOf(ring);
      labels.push({ key: `w-${a.name}`, kind: "water", at: { x: (b.x0 + b.x1) / 2, y: (b.y0 + b.y1) / 2 }, name: a.name, size: Math.min(b.x1 - b.x0, b.y1 - b.y0) });
    }
  }
  for (const r of data.roads) add(r.k === "major" ? "major" : r.k === "minor" ? "minor" : "service", pts(r.p), false);
  const nodes = pts(data.walk.nodes);
  for (const e of data.walk.edges) {
    const a = nodes[e[0]];
    const b = nodes[e[1]];
    if (!a || !b || e[2] === 3) continue; // roads are drawn as roads
    add(e[2] === 1 ? "steps" : "foot", [a, ...pts(e[3] ?? []), b], false);
  }
  const buildings = new Map<number, MapBuilding>();
  for (const b of data.buildings) {
    const ring = pts(b.p);
    const box = boxOf(ring);
    const at = project(lat0 + b.c[0] * data.unit, lon0 + b.c[1] * data.unit);
    add("building", ring, true);
    buildings.set(b.id, { id: b.id, name: b.name, short: b.short, d: pathOf(ring, true), at, box, ring });
    if (b.name) labels.push({ key: `b-${b.id}`, kind: "building", at, name: b.name, short: b.short, size: Math.max(box.x1 - box.x0, box.y1 - box.y0), building: b.id });
  }
  labels.sort((p, q) => q.size - p.size);

  const inside = (ring: Pt[], p: Pt) => {
    let hit = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const a = ring[i] as Pt;
      const b = ring[j] as Pt;
      if (a.y > p.y !== b.y > p.y && p.x < a.x + ((p.y - a.y) * (b.x - a.x)) / (b.y - a.y)) hit = !hit;
    }
    return hit;
  };
  const places = new Map<string, { at: Pt; building?: number; name: string }>();
  for (const [id, p] of Object.entries(data.places ?? {})) {
    const at = project(lat0 + p.c[0] * data.unit, lon0 + p.c[1] * data.unit);
    const osm = /^way\/(\d+)$/.exec(p.osm);
    let building = osm && buildings.has(Number(osm[1])) ? Number(osm[1]) : undefined;
    if (building === undefined) for (const b of buildings.values()) if (at.x >= b.box.x0 && at.x <= b.box.x1 && at.y >= b.box.y0 && at.y <= b.box.y1 && inside(b.ring, at)) building = b.id;
    places.set(id, { at, building, name: p.name });
  }

  const tiles: MapTile[] = [...cells.entries()].map(([key, cell]) => {
    const d: Partial<Record<Layer, string>> = {};
    for (const layer of LAYERS) if (cell.parts[layer]) d[layer] = (cell.parts[layer] as string[]).join("");
    return { key, box: cell.box, d };
  });
  return { attribution: data.attribution || "© OpenStreetMap contributors", width, height, project, tiles, buildings, labels, places };
}

const loaded = new Map<string, Promise<CampusMap>>();

/** Fetch and decode a map once per page load; every screen that draws it shares the result. */
export function loadCampusMap(path: string): Promise<CampusMap> {
  let p = loaded.get(path);
  if (!p) {
    p = api.placeMap(path).then(decodeCampusMap);
    p.catch(() => loaded.delete(path)); // a failed fetch may be tried again
    loaded.set(path, p);
  }
  return p;
}

// ---------- the view: a centre in metres and a zoom in pixels per metre ----------

export interface View {
  cx: number;
  cy: number;
  z: number;
}

export const MAX_ZOOM = 4;

export function minZoom(map: CampusMap, w: number, h: number): number {
  return Math.min(w / map.width, h / map.height);
}

/** Keep the zoom between "the whole campus" and "one doorway", and the centre on the map. */
export function clampView(map: CampusMap, v: View, w: number, h: number): View {
  const z = Math.max(minZoom(map, w, h), Math.min(MAX_ZOOM, v.z));
  const halfW = w / 2 / z;
  const halfH = h / 2 / z;
  // When the map is narrower than the view at this zoom it is centred; otherwise its edge stops at the view's edge.
  const cx = map.width <= halfW * 2 ? map.width / 2 : Math.max(halfW, Math.min(map.width - halfW, v.cx));
  const cy = map.height <= halfH * 2 ? map.height / 2 : Math.max(halfH, Math.min(map.height - halfH, v.cy));
  return { cx, cy, z };
}

/** The view that shows all of `points` with room around them for pins and labels. */
export function fitView(map: CampusMap, points: Pt[], w: number, h: number, pad = 46): View {
  if (points.length === 0) return clampView(map, { cx: map.width / 2, cy: map.height / 2, z: 0 }, w, h);
  let x0 = Infinity;
  let y0 = Infinity;
  let x1 = -Infinity;
  let y1 = -Infinity;
  for (const p of points) {
    if (p.x < x0) x0 = p.x;
    if (p.x > x1) x1 = p.x;
    if (p.y < y0) y0 = p.y;
    if (p.y > y1) y1 = p.y;
  }
  const z = Math.min((w - 2 * pad) / Math.max(x1 - x0, 120), (h - 2 * pad) / Math.max(y1 - y0, 120));
  return clampView(map, { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, z }, w, h);
}

export const toScreen = (v: View, w: number, h: number, p: Pt): Pt => ({ x: (p.x - v.cx) * v.z + w / 2, y: (p.y - v.cy) * v.z + h / 2 });

/** A round length for the scale bar that comes out between about 40 and 100 pixels wide. */
export function scaleBar(z: number): { meters: number; px: number; label: string } {
  const steps = [10, 20, 50, 100, 200, 500, 1000, 2000];
  const meters = steps.find((m) => m * z >= 40) ?? 2000;
  return { meters, px: meters * z, label: meters >= 1000 ? `${meters / 1000} km` : `${meters} m` };
}

/** 566 -> "570 m", 1344 -> "1.3 km" */
export function distance(meters: number): string {
  if (meters >= 1000) return `${(meters / 1000).toFixed(1)} km`;
  return `${meters >= 100 ? Math.round(meters / 10) * 10 : Math.round(meters)} m`;
}

// ---------- pins ----------

export type PinKind = "eat" | "study" | "class" | "home" | "gym" | "park" | "place";

/** A pack place's `kind` (dining, library, class, dorm, gym, ...) as one of the map's pin kinds. */
export function pinKind(kind: string | undefined): PinKind {
  const k = (kind ?? "").toLowerCase();
  if (/dining|eat|food|cafe|kitchen|canteen|restaurant/.test(k)) return "eat";
  if (/library|study|desk/.test(k)) return "study";
  if (/class|lecture|school|work|office|lab/.test(k)) return "class";
  if (/dorm|home|house|flat|apartment|bed|door/.test(k)) return "home";
  if (/gym|sport|rec/.test(k)) return "gym";
  if (/outdoor|park|lawn|garden/.test(k)) return "park";
  return "place";
}

export const PIN_WORD: Record<PinKind, string> = { eat: "eat", study: "study", class: "class", home: "home", gym: "gym", park: "outdoors", place: "place" };
