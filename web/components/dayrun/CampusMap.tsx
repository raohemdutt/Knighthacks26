"use client";

import type { Disruption, FriendAt, PlacePack, RouteLeg, SimResult } from "@/lib/api";
import { clock, clockSpan, minutes, ms, parentPlace, placeName, routeAt, type PlaceIndex } from "@/lib/dayrun";
import ui from "../ui.module.css";
import dr from "./dayrun.module.css";

/**
 * The places of every pack that have coordinates (GET /api/places), projected
 * from latitude and longitude, with the day's walks drawn from `legs` in the
 * order of `routes`. Places without coordinates (rooms inside the apartment)
 * are listed under the map; nothing is drawn at a made-up position.
 */

const W = 360;
const H = 470;
const PAD_Y = 26;
const SIDE = 96; // room either side of the dots for labels
const CHAR = 6.3; // rough width of one 11.5px character, for keeping labels apart

interface Dot {
  ref: string;
  name: string;
  label: string;
  x: number;
  y: number;
}

interface Box {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

const overlaps = (a: Box, b: Box) => a.x0 < b.x1 && b.x0 < a.x1 && a.y0 < b.y1 && b.y0 < a.y1;

/** A name that fits the map: the full name when short, else the longest alias that is. The key under the map has the full names. */
function shortName(name: string, aliases: string[]): string {
  if (name.length <= 15) return name;
  const fits = aliases.filter((a) => a.length <= 15).sort((a, b) => b.length - a.length);
  return fits[0] ?? name;
}

export function CampusMap({
  packs,
  places,
  result,
  friends,
  disruptions,
  scrub,
  onScrubTo,
}: {
  packs: PlacePack[];
  places: PlaceIndex;
  result: SimResult;
  friends: FriendAt[];
  disruptions: Disruption[];
  scrub: number;
  onScrubTo: (iso: string) => void;
}) {
  const located = [...places.values()].filter((p) => typeof p.lat === "number" && typeof p.lon === "number");
  const rooms = packs
    .map((pack) => ({
      pack,
      refs: Object.entries(pack.places)
        .filter(([, p]) => typeof p.lat !== "number" || typeof p.lon !== "number")
        .map(([id]) => `${pack.name}.${id}`),
    }))
    .filter((g) => g.refs.length > 0);

  const tick = result.timeline[Math.max(0, Math.min(result.timeline.length - 1, scrub))];
  const t = tick ? ms(tick.t) : ms(result.start);
  const current = routeAt(result, t);
  const here = tick?.place;
  const closed = new Set(disruptions.filter((d) => d.kind === "place_closed" && d.place).map((d) => d.place as string));
  const friendsAt = new Map<string, FriendAt[]>();
  for (const f of friends) friendsAt.set(f.place, [...(friendsAt.get(f.place) ?? []), f]);

  if (located.length === 0) {
    return <p className={ui.muted}>None of the box&apos;s places has coordinates, so there is no map to draw. The walks are listed below.</p>;
  }

  // Equirectangular projection around the middle latitude: fine at campus scale.
  const lats = located.map((p) => p.lat as number);
  const lons = located.map((p) => p.lon as number);
  const latMin = Math.min(...lats);
  const latMax = Math.max(...lats);
  const lonMin = Math.min(...lons);
  const lonMax = Math.max(...lons);
  const k = Math.cos((((latMin + latMax) / 2) * Math.PI) / 180);
  const spanX = (lonMax - lonMin) * k || 1e-6;
  const spanY = latMax - latMin || 1e-6;
  const scale = Math.min((W - 2 * SIDE) / spanX, (H - 2 * PAD_Y) / spanY);
  const offX = (W - spanX * scale) / 2;
  const offY = (H - spanY * scale) / 2;
  const project = (lat: number, lon: number): [number, number] => [offX + (lon - lonMin) * k * scale, offY + (latMax - lat) * scale];

  const dots: Dot[] = located
    .map((p) => {
      const [x, y] = project(p.lat as number, p.lon as number);
      const who = (friendsAt.get(p.ref) ?? []).map((f) => f.member).join(", ");
      const label = `${shortName(p.name, p.aliases)}${who ? ` · ${who}` : ""}${closed.has(p.ref) ? " (closed)" : ""}`;
      return { ref: p.ref, name: p.name, label, x, y };
    })
    .sort((a, b) => a.y - b.y);
  const dotAt = new Map(dots.map((d) => [d.ref, d]));

  // Walks that share an edge are drawn side by side, in the order they were walked.
  const drawable = result.legs.filter((l): l is RouteLeg & { from_xy: [number, number]; to_xy: [number, number] } => l.from_xy !== null && l.to_xy !== null);

  // Label placement. Each label tries eight spots around its dot and takes the one that costs least:
  // leaving the map costs most, then covering a dot or another label, then covering a stretch of a walk.
  const hard: Box[] = dots.map((d) => ({ x0: d.x - 7, y0: d.y - 7, x1: d.x + 7, y1: d.y + 7 }));
  const soft: Box[] = [];
  for (const l of drawable) {
    const [ax, ay] = project(...l.from_xy);
    const [bx, by] = project(...l.to_xy);
    const steps = Math.max(1, Math.ceil(Math.hypot(bx - ax, by - ay) / 5));
    for (let n = 0; n <= steps; n++) {
      const x = ax + ((bx - ax) * n) / steps;
      const y = ay + ((by - ay) * n) / steps;
      soft.push({ x0: x - 3, y0: y - 3, x1: x + 3, y1: y + 3 });
    }
  }
  const labels = dots.map((d) => {
    const w = d.label.length * CHAR;
    const side = (dx: number, dy: number, anchor: "start" | "end" | "middle") => {
      const x = d.x + dx;
      const y = d.y + dy;
      const x0 = anchor === "start" ? x : anchor === "end" ? x - w : x - w / 2;
      return { x, y, anchor, box: { x0: x0 - 1, y0: y - 11, x1: x0 + w + 1, y1: y + 3 } };
    };
    const options = [side(10, 4, "start"), side(-10, 4, "end"), side(0, -11, "middle"), side(0, 20, "middle"), side(8, -8, "start"), side(-8, -8, "end"), side(8, 17, "start"), side(-8, 17, "end")];
    const cost = (box: Box) =>
      (box.x0 < 2 || box.x1 > W - 2 || box.y0 < 2 || box.y1 > H - 2 ? 1000 : 0) + hard.filter((h) => overlaps(box, h)).length * 100 + soft.filter((h) => overlaps(box, h)).length * 4;
    let pick = options[0]!;
    let best = Infinity;
    for (const o of options) {
      const c = cost(o.box);
      if (c < best) {
        best = c;
        pick = o;
      }
    }
    hard.push(pick.box);
    return { dot: d, ...pick };
  });

  const edgeKey = (l: RouteLeg) => [l.from, l.to].sort().join("|");
  const lanes = new Map<string, number[]>();
  for (const l of drawable) {
    const list = lanes.get(edgeKey(l)) ?? [];
    if (!list.includes(l.route)) list.push(l.route);
    lanes.set(edgeKey(l), list);
  }
  const segments = drawable.map((l, i) => {
    const flip = l.from > l.to;
    const [ax, ay] = project(...(flip ? l.to_xy : l.from_xy));
    const [bx, by] = project(...(flip ? l.from_xy : l.to_xy));
    const len = Math.hypot(bx - ax, by - ay) || 1;
    const list = lanes.get(edgeKey(l)) ?? [l.route];
    const lane = (list.indexOf(l.route) - (list.length - 1) / 2) * 3.2;
    const nx = (-(by - ay) / len) * lane;
    const ny = ((bx - ax) / len) * lane;
    const [x1, y1] = project(...l.from_xy);
    const [x2, y2] = project(...l.to_xy);
    return { key: `${l.route}-${i}`, leg: l, x1: x1 + nx, y1: y1 + ny, x2: x2 + nx, y2: y2 + ny, on: current?.route.index === l.route };
  });

  // Where you are at the scrubbed time: the place itself, or the campus place that leads into its pack.
  const hereDot = here ? (dotAt.get(here) ?? dotAt.get(parentPlace(places, here)?.ref ?? "")) : undefined;
  const hereIndoors = here && !dotAt.has(here) ? here : null;

  return (
    <div className={dr.chartWrap}>
      <svg className={dr.map} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Campus map with ${dots.length} places and ${result.routes.length} walks.`}>
        <defs>
          <marker id="dayrun-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
            <path className={dr.arrow} d="M0 1.5 L8.5 5 L0 8.5 z" />
          </marker>
        </defs>

        {segments
          .filter((s) => !s.on)
          .map((s) => (
            <line key={s.key} className={dr.leg} x1={s.x1} y1={s.y1} x2={s.x2} y2={s.y2} />
          ))}
        {segments
          .filter((s) => s.on)
          .map((s) => (
            <line key={s.key} className={dr.legOn} x1={s.x1} y1={s.y1} x2={s.x2} y2={s.y2} markerEnd="url(#dayrun-arrow)" />
          ))}

        {dots.map((d) => {
          const friendHere = friendsAt.has(d.ref);
          return (
            <g key={d.ref}>
              {friendHere ? <circle className={dr.friendRing} cx={d.x} cy={d.y} r={9} /> : null}
              <circle className={closed.has(d.ref) ? dr.placeClosed : dr.place} cx={d.x} cy={d.y} r={4.5} />
            </g>
          );
        })}
        {hereDot ? <circle className={dr.you} cx={hereDot.x} cy={hereDot.y} r={6.5} /> : null}

        {labels.map((l) => (
          <text key={l.dot.ref} className={dr.placeLabel} x={l.x} y={l.y} textAnchor={l.anchor}>
            {l.dot.label}
          </text>
        ))}
      </svg>

      <p className={dr.legend}>
        <span>
          <span className={`${dr.dotKey} ${dr.dotYou}`} aria-hidden="true" />
          you at {tick ? clock(tick.t) : ""}
          {hereIndoors ? ` (indoors: ${placeName(places, hereIndoors)})` : ""}
        </span>
        <span>
          <span className={`${dr.lineKey} ${dr.lineKeyOn}`} aria-hidden="true" />
          {current ? (current.walking ? "the walk under way" : "the last walk") : "no walk yet"}
        </span>
        <span>
          <span className={dr.lineKey} aria-hidden="true" />
          other walks
        </span>
        {friends.length > 0 ? (
          <span>
            <span className={`${dr.dotKey} ${dr.dotFriend}`} aria-hidden="true" />
            a friend shared this place
          </span>
        ) : null}
      </p>

      {friends.length > 0 ? (
        <p className={`${ui.muted} ${ui.small}`}>{friends.map((f) => `${f.member} is at ${f.place_name || placeName(places, f.place)} ${clockSpan(f.start, f.end)}`).join(". ")}.</p>
      ) : null}

      {result.routes.length === 0 ? (
        <p className={ui.muted}>This plan has no walks: every move stays where the day leaves you.</p>
      ) : (
        <ol className={dr.routes}>
          {result.routes.map((r) => {
            const on = current?.route.index === r.index;
            const indoors = result.legs.filter((l) => l.route === r.index && (l.from_xy === null || l.to_xy === null));
            const steps = r.places.map((p) => placeName(places, p)).join(" → ");
            return (
              <li key={r.index}>
                <button type="button" className={`${dr.routeRow} ${on ? dr.routeRowOn : ""}`} aria-pressed={on} onClick={() => onScrubTo(r.depart)}>
                  <span className={dr.routeNum}>{r.index + 1}</span>
                  <span className={ui.grow}>
                    <span className={dr.routeTitle}>
                      {clock(r.depart)} {r.purpose} · {minutes(r.minutes)}
                    </span>
                    {on ? <span className={dr.routeSteps}>{steps}</span> : null}
                    {on && indoors.length > 0 ? <span className={dr.routeSteps}>{indoors.length === 1 ? "1 leg" : `${indoors.length} legs`} of it indoors, not drawn.</span> : null}
                  </span>
                </button>
              </li>
            );
          })}
        </ol>
      )}

      <div className={dr.mapKey}>
        <p className={ui.sectionTitle}>Places on the map</p>
        <ul className={dr.keyList}>
          {dots.map((d) => {
            const short = shortName(d.name, places.get(d.ref)?.aliases ?? []);
            return <li key={d.ref}>{short === d.name ? d.name : `${short}: ${d.name}`}</li>;
          })}
        </ul>
        {rooms.map(({ pack, refs }) => {
          const parent = refs[0] ? parentPlace(places, refs[0]) : undefined;
          return (
            <div key={pack.name}>
              <p className={ui.sectionTitle}>Not drawn: {parent ? `inside ${parent.name}` : `the ${pack.name} pack`}</p>
              <p className={`${ui.muted} ${ui.small}`}>
                These places have no coordinates on the box, so they are listed, not placed
                {parent ? `. On the map you are shown at ${parent.name} while you are in one of them` : ""}.
              </p>
              <ul className={dr.keyList}>
                {refs.map((ref) => (
                  <li key={ref} className={ref === here ? dr.keyHere : undefined}>
                    {placeName(places, ref)}
                    {ref === here ? " (you are here)" : ""}
                  </li>
                ))}
              </ul>
            </div>
          );
        })}
      </div>
    </div>
  );
}
