/**
 * Geometry for the Room map, shared by the 2D and 3D views.
 *
 * What the box knows: zone polygons per camera, in that camera's image (0..1),
 * and for a watch item its camera, zone and a text position. It knows no floor
 * plan and no item coordinates. So each camera's view is laid out as its own
 * panel, and an item's pin goes to the middle of its zone, spread out when
 * several items share one.
 */

import type { Cell, PlayRoute, Sighting } from "./api";
import { ZONES, type ZonePolygon } from "./zones.generated";

/** The items the box knows how to sort (config/sorting.yaml). The gateway has no endpoint that lists them. */
export const SORT_ITEMS = ["can", "bottle", "paper", "cardboard", "cup", "napkin", "chip bag", "food container", "banana", "wine glass"];

/** A short list of things worth finding. Each is asked for by name; most boxes have seen only some. */
export const WATCH_ITEMS = ["keys", "wallet", "charger", "laptop", "phone"];

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
  camera: string;
  zone: string;
  at: Point; // inside the zone; not a measured position
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

/** Pins for the sightings that name a zone this build knows. Items sharing a zone are spread along its width. */
export function placePins(sightings: Sighting[]): { pins: RoomPin[]; unplaced: Sighting[] } {
  const pins: RoomPin[] = [];
  const unplaced: Sighting[] = [];
  const groups = new Map<string, Sighting[]>();
  for (const s of sightings) {
    const zone = s.zone_id ? CAMERAS.find((c) => c.id === s.camera_id)?.zones.find((z) => z.id === s.zone_id) : undefined;
    if (!zone) unplaced.push(s);
    else groups.set(`${s.camera_id}/${zone.id}`, [...(groups.get(`${s.camera_id}/${zone.id}`) ?? []), s]);
  }
  for (const [key, list] of groups) {
    const [camera, zoneId] = key.split("/") as [string, string];
    const zone = CAMERAS.find((c) => c.id === camera)?.zones.find((z) => z.id === zoneId);
    if (!zone) continue;
    const xs = zone.polygon.map((p) => p[0]);
    const width = Math.max(...xs) - Math.min(...xs);
    const step = Math.min(0.2, (width * 0.7) / Math.max(1, list.length));
    list.forEach((s, i) => {
      const dx = (i - (list.length - 1) / 2) * step;
      pins.push({ label: s.label, camera, zone: zone.id, at: [zone.center[0] + dx, zone.center[1]], sighting: s });
    });
  }
  return { pins, unplaced };
}

/** The camera whose view a Sort It route is drawn in: the one that has the bin's zone. */
export function cameraOfZone(zone: string): string | undefined {
  return CAMERAS.find((c) => c.zones.some((z) => z.id === zone))?.id;
}

/** Route cells to points in the camera's image: the middle of each cell, as the box's grid defines it. */
export function routePoints(route: PlayRoute): Point[] {
  return route.path.map((c: Cell) => [(c.col + 0.5) / route.cols, (c.row + 0.5) / route.rows]);
}
