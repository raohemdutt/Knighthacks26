import type { RouteLeg } from "@/lib/api";
import { toScreen, type CampusMap, type Pt, type View } from "@/lib/campusmap";

/** A name short enough to sit by a pin: the name without its note in brackets, else the longest alias that fits. */
export function pinName(name: string, aliases: string[]): string {
  const base = name.split(" (")[0] ?? name;
  if (base.length <= 18) return base;
  const fits = aliases.filter((a) => a.length <= 18).sort((a, b) => b.length - a.length);
  return fits[0] ?? `${base.slice(0, 17).trimEnd()}…`;
}

/** The line a leg was walked along, in map metres: its own path when the box sent one, else straight between its ends. */
export function legLine(map: CampusMap, leg: Pick<RouteLeg, "path" | "from_xy" | "to_xy">): Pt[] {
  if (leg.path && leg.path.length > 1) return leg.path.map(([lat, lon]) => map.project(lat, lon));
  if (leg.from_xy && leg.to_xy) return [map.project(leg.from_xy[0], leg.from_xy[1]), map.project(leg.to_xy[0], leg.to_xy[1])];
  return [];
}

export const linePath = (view: View, w: number, h: number, points: Pt[]) =>
  points
    .map((p, i) => {
      const s = toScreen(view, w, h, p);
      return `${i ? "L" : "M"}${s.x.toFixed(1)} ${s.y.toFixed(1)}`;
    })
    .join("");

/** The point half way along a line, for a route's badge. */
export function midpoint(points: Pt[]): Pt | undefined {
  if (points.length < 2) return points[0];
  let total = 0;
  for (let i = 1; i < points.length; i++) total += Math.hypot((points[i] as Pt).x - (points[i - 1] as Pt).x, (points[i] as Pt).y - (points[i - 1] as Pt).y);
  let left = total / 2;
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1] as Pt;
    const b = points[i] as Pt;
    const seg = Math.hypot(b.x - a.x, b.y - a.y);
    if (left <= seg && seg > 0) return { x: a.x + ((b.x - a.x) * left) / seg, y: a.y + ((b.y - a.y) * left) / seg };
    left -= seg;
  }
  return points[points.length - 1];
}
