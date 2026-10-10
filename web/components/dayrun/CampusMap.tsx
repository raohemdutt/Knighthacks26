"use client";

import type { Disruption, FriendAt, PlacePack, RouteLeg, SimResult } from "@/lib/api";
import { distance, pinKind, toScreen, type CampusMap as DrawnMap, type PinKind, type Pt, type View } from "@/lib/campusmap";
import { clock, clockSpan, minutes, ms, parentPlace, placeName, routeAt, type PlaceIndex } from "@/lib/dayrun";
import ui from "../ui.module.css";
import cm from "./campusmap.module.css";
import { CampusLayers, MapButtons, MapFurniture, PinKey, PlacePin, layoutPins, lineBoxes, useCampusMap, useMapView, type ScreenBox } from "./CampusLayers";
import dr from "./dayrun.module.css";
import { PlainCampusMap } from "./PlainCampusMap";

/**
 * Where the day takes you, on the real campus: building footprints, greens,
 * water, roads and footpaths (© OpenStreetMap contributors), the pack's places
 * as pins by what they are for, friends where they shared, and each walk drawn
 * along the paths it actually takes, with its minutes and metres.
 *
 * A pack with no drawn map (a hand-made one) keeps the plain dot map.
 */

const W = 360;
const H = 440;

interface Props {
  packs: PlacePack[];
  places: PlaceIndex;
  result: SimResult;
  friends: FriendAt[];
  disruptions: Disruption[];
  scrub: number;
  onScrubTo: (iso: string) => void;
}

export function CampusMap(props: Props) {
  const visited = new Set<string>();
  for (const t of props.result.timeline) visited.add(t.place.split(".")[0] ?? "");
  for (const b of props.result.blocks) visited.add(b.place.split(".")[0] ?? "");
  for (const f of props.friends) visited.add(f.place.split(".")[0] ?? "");
  // A room's pack is drawn on the map of the pack its building is in.
  for (const p of props.places.values()) if (p.pack && visited.has(p.pack)) visited.add(p.pack_name);
  const { map, pack, failed } = useCampusMap(props.packs, visited);
  if (map && pack) return <DrawnCampusMap {...props} map={map} mapPack={pack} />;
  if (pack && !failed) return <p className={ui.muted}>Loading the campus map…</p>;
  return <PlainCampusMap {...props} />;
}

/** A name short enough to sit under a pin: the name without its note in brackets, else the longest alias that fits. */
export function pinName(name: string, aliases: string[]): string {
  const base = name.split(" (")[0] ?? name;
  if (base.length <= 18) return base;
  const fits = aliases.filter((a) => a.length <= 18).sort((a, b) => b.length - a.length);
  return fits[0] ?? `${base.slice(0, 17).trimEnd()}…`;
}

/** The line a leg was walked along, in map metres: its own path when the box sent one, else straight between its ends. */
export function legLine(map: DrawnMap, leg: Pick<RouteLeg, "path" | "from_xy" | "to_xy">): Pt[] {
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

function DrawnCampusMap({ packs, places, result, friends, disruptions, scrub, onScrubTo, map, mapPack }: Props & { map: DrawnMap; mapPack: string }) {
  const located = [...places.values()].filter((p) => p.pack_name === mapPack && typeof p.lat === "number" && typeof p.lon === "number");
  const at = new Map<string, Pt>(located.map((p) => [p.ref, map.project(p.lat as number, p.lon as number)]));
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
  const resolve = (ref: string | undefined | null): Pt | undefined => (ref ? (at.get(ref) ?? at.get(parentPlace(places, ref)?.ref ?? "")) : undefined);
  const outdoorRef = (ref: string): string | undefined => (at.has(ref) ? ref : parentPlace(places, ref)?.ref);
  const closed = new Set(disruptions.filter((d) => d.kind === "place_closed" && d.place).map((d) => d.place as string));
  const friendsAt = new Map<string, FriendAt[]>();
  for (const f of friends) friendsAt.set(f.place, [...(friendsAt.get(f.place) ?? []), f]);

  // Every walk as the line it follows.
  const lines = result.routes.map((r) => {
    const points: Pt[] = [];
    for (const leg of result.legs) if (leg.route === r.index) for (const p of legLine(map, leg)) if (!points.length || Math.hypot(p.x - (points[points.length - 1] as Pt).x, p.y - (points[points.length - 1] as Pt).y) > 0.3) points.push(p);
    return { route: r, points };
  });
  const currentLine = lines.find((l) => l.route.index === current?.route.index && l.points.length > 1);

  // Places the day touches get a full pin and a name; the rest of the pack is shown small.
  const touched = new Set<string>();
  const touch = (ref: string | null | undefined) => {
    const out = ref ? outdoorRef(ref) : undefined;
    if (out) touched.add(out);
  };
  for (const x of result.timeline) touch(x.place);
  for (const b of result.blocks) touch(b.place);
  for (const r of result.routes) for (const p of r.places) touch(p);
  for (const f of friends) touch(f.place);

  const dayPoints = [...lines.flatMap((l) => l.points), ...[...touched].map((ref) => at.get(ref)).filter((p): p is Pt => p !== undefined)];
  const focus = currentLine ? currentLine.points : dayPoints;
  const controls = useMapView(map, W, H, focus, `${currentLine ? currentLine.route.index : "day"}|${result.routes.length}|${result.start}`);
  const view = controls.view;
  const S = (p: Pt) => toScreen(view, W, H, p);

  const buildingOf = (ref: string) => map.places.get(ref.split(".").slice(1).join("."))?.building;
  const on = new Set<number>();
  const shut = new Set<number>();
  for (const ref of touched) {
    const b = buildingOf(ref);
    if (b !== undefined) {
      on.add(b);
      if (closed.has(ref)) shut.add(b);
    }
  }
  for (const ref of closed) {
    const b = buildingOf(ref);
    if (b !== undefined) {
      on.add(b);
      shut.add(b);
    }
  }

  const hereRef = here ? outdoorRef(here) : undefined;
  const herePt = hereRef ? at.get(hereRef) : undefined;
  const hereIndoors = here && !at.has(here) ? here : null;
  const kinds = [...new Set(located.map((p) => pinKind(p.kind)))] as PinKind[];
  const badge = currentLine ? midpoint(currentLine.points) : undefined;
  const badgeText = currentLine ? `${minutes(currentLine.route.minutes)}${typeof currentLine.route.meters === "number" ? ` · ${distance(currentLine.route.meters)}` : ""}` : "";
  const badgeW = badgeText.length * 5.9 + 12;

  // Pins, and names that do not sit on each other, on the route's badge or on the buttons: under the pin, else above it, else left out.
  const taken: ScreenBox[] = [{ x0: 0, y0: 0, x1: 250, y1: 44 }];
  if (badge) taken.push({ x0: S(badge).x - badgeW / 2 - 2, y0: S(badge).y - 11, x1: S(badge).x + badgeW / 2 + 2, y1: S(badge).y + 11 });
  const quiet = new Set<number>();
  const shown = located
    .map((p) => ({ p, s: S(at.get(p.ref) as Pt), big: touched.has(p.ref) || closed.has(p.ref), text: `${pinName(p.name, p.aliases)}${closed.has(p.ref) ? " · closed" : ""}` }))
    .filter(({ s }) => s.x > -12 && s.x < W + 12 && s.y > -12 && s.y < H + 12);
  const where = layoutPins(
    shown.map(({ p, s, big, text }) => ({ key: p.ref, x: s.x, y: s.y, big, text, named: big || view.z > 0.45 })),
    taken,
  );
  const pins = shown.map((x) => ({ ...x, below: where.get(x.p.ref) ?? null }));
  for (const p of located) {
    const b = buildingOf(p.ref);
    if (b !== undefined) quiet.add(b);
  }
  // The other buildings' names also keep off the walk being shown.
  if (currentLine) taken.push(...lineBoxes(view, W, H, currentLine.points));

  return (
    <div className={dr.chartWrap}>
      <div className={cm.frame}>
        <svg
          {...controls.bind}
          className={cm.svg}
          viewBox={`0 0 ${W} ${H}`}
          role="img"
          aria-label={`Campus map with ${located.length} places and ${result.routes.length} walks. ${current ? `Showing the walk ${current.route.purpose}.` : ""} Drag to move, pinch to zoom.`}
        >
          <CampusLayers map={map} view={view} width={W} height={H} highlight={{ on, closed: shut }} quiet={quiet} taken={taken} />

          {lines.map((l) => (l.points.length > 1 && l !== currentLine ? <path key={`dim-${l.route.index}`} className={cm.routeDim} d={linePath(view, W, H, l.points)} /> : null))}
          {currentLine ? (
            <>
              <path className={cm.routeCase} d={linePath(view, W, H, currentLine.points)} />
              <path className={cm.route} d={linePath(view, W, H, currentLine.points)} />
              <circle className={cm.routeDot} cx={S(currentLine.points[0] as Pt).x} cy={S(currentLine.points[0] as Pt).y} r={4} />
            </>
          ) : null}

          {pins.map(({ p, s, big, text, below }) => (
            <PlacePin key={p.ref} x={s.x} y={s.y} kind={pinKind(p.kind)} small={!big} closed={closed.has(p.ref)} label={below === null ? undefined : text} below={below ?? true} />
          ))}

          {[...friendsAt.entries()].map(([ref, list]) => {
            const p = resolve(ref);
            if (!p) return null;
            const s = S(p);
            return list.map((f, i) => (
              <g key={`${f.member}-${f.start}`} transform={`translate(${(s.x + 11 + i * 13).toFixed(1)} ${(s.y - 11).toFixed(1)})`}>
                <title>{`${f.member} is here ${clockSpan(f.start, f.end)}`}</title>
                <circle className={cm.friend} r={7} />
                <text className={cm.friendText} x={0} y={3.2} textAnchor="middle">
                  {f.member.slice(0, 1).toUpperCase()}
                </text>
                {i === list.length - 1 ? (
                  <text className={cm.friendName} x={10} y={3.2}>
                    {list.map((x) => x.member).join(", ")}
                  </text>
                ) : null}
              </g>
            ));
          })}

          {herePt ? (
            <g transform={`translate(${S(herePt).x.toFixed(1)} ${S(herePt).y.toFixed(1)})`}>
              <circle className={cm.youHalo} r={15} />
              <circle className={cm.you} r={6} />
            </g>
          ) : null}

          {badge && badgeText ? (
            <g transform={`translate(${S(badge).x.toFixed(1)} ${S(badge).y.toFixed(1)})`}>
              <rect className={cm.badge} x={-badgeW / 2} y={-9.5} width={badgeW} height={19} rx={9.5} />
              <text className={cm.badgeText} x={0} y={3.8} textAnchor="middle">
                {badgeText}
              </text>
            </g>
          ) : null}

          <MapFurniture map={map} view={view} width={W} height={H} />
        </svg>
        <MapButtons
          controls={controls}
          onFit={() => controls.fit(focus)}
          fitLabel={currentLine ? "Fit route" : "Fit day"}
          extra={
            <button type="button" className={cm.mapBtn} onClick={() => controls.fit(currentLine ? dayPoints : [])} aria-label={currentLine ? "Show the whole day" : "Show the whole campus"}>
              {currentLine ? "Whole day" : "Campus"}
            </button>
          }
        />
      </div>

      <p className={cm.key}>
        <span className={cm.keyItem}>
          <span className={cm.keyYou} aria-hidden="true" />
          you at {tick ? clock(tick.t) : ""}
          {hereIndoors ? ` (indoors: ${placeName(places, hereIndoors)})` : ""}
        </span>
        <span className={cm.keyItem}>
          <span className={cm.keyLine} aria-hidden="true" />
          {current ? (current.walking ? "the walk under way" : "the last walk") : "no walk yet"}
        </span>
        <span className={cm.keyItem}>
          <span className={`${cm.keyLine} ${cm.keyLineDim}`} aria-hidden="true" />
          other walks
        </span>
        {friends.length > 0 ? (
          <span className={cm.keyItem}>
            <span className={cm.keyFriend} aria-hidden="true" />
            a friend shared this place
          </span>
        ) : null}
        <PinKey kinds={kinds} />
      </p>

      {friends.length > 0 ? (
        <p className={`${ui.muted} ${ui.small}`}>{friends.map((f) => `${f.member} is at ${f.place_name || placeName(places, f.place)} ${clockSpan(f.start, f.end)}`).join(". ")}.</p>
      ) : null}

      {result.routes.length === 0 ? (
        <p className={ui.muted}>This plan has no walks: every move stays where the day leaves you.</p>
      ) : (
        <ol className={dr.routes}>
          {result.routes.map((r) => {
            const isOn = current?.route.index === r.index;
            const indoors = result.legs.filter((l) => l.route === r.index && (l.from_xy === null || l.to_xy === null));
            const steps = r.places.map((p) => placeName(places, p)).join(" → ");
            return (
              <li key={r.index}>
                <button type="button" className={`${dr.routeRow} ${isOn ? dr.routeRowOn : ""}`} aria-pressed={isOn} onClick={() => onScrubTo(r.depart)}>
                  <span className={dr.routeNum}>{r.index + 1}</span>
                  <span className={ui.grow}>
                    <span className={dr.routeTitle}>
                      {clock(r.depart)} {r.purpose} · {minutes(r.minutes)}
                      {typeof r.meters === "number" ? ` · ${distance(r.meters)}` : ""}
                    </span>
                    {isOn ? <span className={dr.routeSteps}>{steps}</span> : null}
                    {isOn && indoors.length > 0 ? <span className={dr.routeSteps}>{indoors.length === 1 ? "1 leg" : `${indoors.length} legs`} of it indoors, not drawn.</span> : null}
                  </span>
                </button>
              </li>
            );
          })}
        </ol>
      )}

      <div className={dr.mapKey}>
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
        <p className={`${ui.muted} ${ui.small}`}>
          Map data{" "}
          <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer noopener">
            {map.attribution}
          </a>
          . Walks follow the mapped footpaths at a walking pace of 1.35 m/s.
        </p>
      </div>
    </div>
  );
}
