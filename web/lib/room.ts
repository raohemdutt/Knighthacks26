/**
 * Geometry for the Room map, shared by the 2D and 3D views.
 *
 * What the box knows: zone polygons per camera, in that camera's image (0..1),
 * and for a watch item its camera, zone and a text position. It knows no floor
 * plan and no item coordinates. So each camera's view is laid out as its own
 * panel, and an item's pin goes inside its zone, on a grid when several items
 * share one. Every pin has a number; the list under the map says which is which.
 */

import type { Cell, PlayRoute, Sighting } from "./api";
import { ZONES, type ZonePolygon } from "./zones.generated";

/** The items the box knows how to sort (config/sorting.yaml). The gateway has no endpoint that lists them. */
export const SORT_ITEMS = ["can", "bottle", "paper", "cardboard", "cup", "napkin", "chip bag", "food container", "banana", "wine glass"];

/** A short list of things worth finding. Each is asked for by name; most boxes have seen only some. */
export const WATCH_ITEMS = ["phone", "laptop", "water bottle"];

export type Point = [number, number];

export interface RoomZone {
  id: string;
  polygon: ZonePolygon;
  center: Point;
}

export interface RoomCamera {
  id: string;
  zones: RoomZone[];
}

export interface RoomPin {
  label: string;
  n: number; // the number on the pin and in the list under the map, from 1
  camera: string;
  zone: string;
  at: Point; // inside the zone; not a measured position
  named: boolean; // there is room to write the name beside the pin; otherwise only its number is drawn
  lift: number; // rows of pins in front of this one in its zone (the 3D view stands those behind taller)
  sighting: Sighting;
}

/** Area centroid of a polygon (falls back to the mean of its points for a degenerate one). */
export function centroid(poly: ZonePolygon): Point {
  let a = 0;
  let cx = 0;
  let cy = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x0, y0] = poly[i] as Point;
    const [x1, y1] = poly[(i + 1) % poly.length] as Point;
    const f = x0 * y1 - x1 * y0;
    a += f;
    cx += (x0 + x1) * f;
    cy += (y0 + y1) * f;
  }
  if (Math.abs(a) < 1e-9) return [poly.reduce((s, p) => s + p[0], 0) / poly.length, poly.reduce((s, p) => s + p[1], 0) / poly.length];
  return [cx / (3 * a), cy / (3 * a)];
}

export const CAMERAS: RoomCamera[] = Object.entries(ZONES).map(([id, zones]) => ({
  id,
  zones: Object.entries(zones).map(([zone, polygon]) => ({ id: zone, polygon, center: centroid(polygon) })),
}));

// Both views draw a camera's panel 4:3. The layout below works in panel heights (the panel is ASPECT wide, 1 tall).
export const ASPECT = 4 / 3;
const PITCH_X = 0.2; // the widest gap between two pins side by side
const PITCH_Y = 0.14; // ... and between two rows of them
const NAME_BAND = 0.1; // the top of a zone, kept clear for its name
const CHAR = 7 / 270; // about one letter of a pin's name in the 2D view (12 px type on a 270 px tall panel)

/**
 * Where `count` pins go inside a zone: a grid in the zone's box, under its name. The grid takes as many
 * columns as keep the pins furthest apart, so a wide desk gets one row and a narrow bin gets a few short ones.
 * `slot` is the width each pin has to itself for a name, 0 when the pins are stacked in rows.
 */
function grid(zone: RoomZone, count: number): { at: Point[]; lift: number[]; slot: number } {
  const xs = zone.polygon.map((p) => p[0]);
  const ys = zone.polygon.map((p) => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const top = Math.min(y0 + NAME_BAND, (y0 + y1) / 2);
  const width = (x1 - x0) * ASPECT * 0.84;
  const height = (y1 - top) * 0.8;
  let best = { cols: count, rows: 1, px: 0, py: 0 };
  for (let cols = count; cols >= 1; cols--) {
    const rows = Math.ceil(count / cols);
    const px = Math.min(PITCH_X, width / cols);
    const py = Math.min(PITCH_Y, height / rows);
    if (Math.min(px, py) > Math.min(best.px, best.py)) best = { cols, rows, px, py };
  }
  const { cols, rows, px, py } = best;
  const at: Point[] = [];
  const lift: number[] = [];
  for (let i = 0; i < count; i++) {
    const row = Math.floor(i / cols);
    const inRow = row === rows - 1 ? count - row * cols : cols; // a short last row is centred too
    const col = i - row * cols;
    at.push([(x0 + x1) / 2 + ((col - (inRow - 1) / 2) * px) / ASPECT, (top + y1) / 2 + (row - (rows - 1) / 2) * py]);
    lift.push(rows - 1 - row);
  }
  return { at, lift, slot: rows > 1 ? 0 : count === 1 ? (x1 - x0) * ASPECT : px };
}

/**
 * Pins for the sightings that name a zone this build knows, numbered in the order given. Items sharing a
 * zone are spread on a grid inside it, and are named on the map only when every name there has room.
 */
export function placePins(sightings: Sighting[]): { pins: RoomPin[]; unplaced: Sighting[] } {
  const pins: RoomPin[] = [];
  const unplaced: Sighting[] = [];
  const groups = new Map<RoomZone, { camera: string; list: Sighting[] }>();
  for (const s of sightings) {
    const zone = s.zone_id ? CAMERAS.find((c) => c.id === s.camera_id)?.zones.find((z) => z.id === s.zone_id) : undefined;
    if (!zone) unplaced.push(s);
    else groups.set(zone, { camera: s.camera_id, list: [...(groups.get(zone)?.list ?? []), s] });
  }
  const order = sightings.filter((s) => !unplaced.includes(s));
  for (const [zone, { camera, list }] of groups) {
    const { at, lift, slot } = grid(zone, list.length);
    const named = list.every((s) => s.label.length * CHAR + 0.02 <= slot);
    list.forEach((s, i) => {
      pins.push({ label: s.label, n: order.indexOf(s) + 1, camera, zone: zone.id, at: at[i] as Point, named, lift: lift[i] as number, sighting: s });
    });
  }
  return { pins: pins.sort((a, b) => a.n - b.n), unplaced };
}

/** The camera whose view a Sort It route is drawn in: the one that has the bin's zone. */
export function cameraOfZone(zone: string): string | undefined {
  return CAMERAS.find((c) => c.zones.some((z) => z.id === zone))?.id;
}

/** Route cells to points in the camera's image: the middle of each cell, as the box's grid defines it. */
export function routePoints(route: PlayRoute): Point[] {
  return route.path.map((c: Cell) => [(c.col + 0.5) / route.cols, (c.row + 0.5) / route.rows]);
}
