"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import ui from "@/components/ui.module.css";
import { api, type RoomPlan, type RoomRouteAnswer, type RoundPlans, type RouteAround, type RouteFound, type RouteObstacle, type RouteSearch } from "@/lib/api";
import { humanize } from "@/lib/format";
import type { RoomTwinData } from "@/lib/roomTwin";
import css from "./play.module.css";

/**
 * "How the route was found": the room from above on the planner's own 10 cm grid, with the cells it could not
 * use, the cells the A* search looked at (in the order it looked at them), the path it settled on, and what the
 * walk bends around. One plain line says it in words.
 *
 * Nothing here is made up for the picture. The grid, the blocked cells, the explored cells and the path are the
 * planner's record of the run that produced the route on screen (contexts/play/room/grid.py: `astar`'s trace, sent
 * by GET /api/play/room-route?search=1 and GET /api/play/route-search). The animation only plays that record back:
 * explored cells appear in the order the search expanded them, then the path's cells, then the walk.
 *
 * Three sources, in this order:
 *   a round that is on   its newest plan; a re-plan draws the old way dashed and says what moved into the way
 *   the last round       on request ("Show the last round's plans"): step through its first plan and its re-plans
 *   no round             the walk for the picked item from wherever a camera sees someone, else the room's start
 *                        point, asked again every two seconds; it plays again when what is in the way changes
 *
 * Reduced motion (the system setting, or ?motion=reduce): the finished picture at once, and the last round's
 * plans are stepped by hand.
 */

const PREVIEW_EVERY_MS = 2000;
const U = 10; // drawing units per grid cell
const EXPLORE_MS = 1100; // the whole search plays back in this long, however many cells it looked at
const STEP_MS = 14; // then each cell of the path
const HOLD_MS = 4200; // a plan of the last round stays this long before the next one plays

type Pt = [number, number];

/** One route to draw, with how it was found. `key` changes when the drawing should play again. */
export interface Found {
  key: string;
  source: "round" | "last" | "preview";
  item: string;
  binId: string | null;
  reachable: boolean;
  points: Pt[];
  distance_m: number;
  from: { x: number; y: number; track: string | null };
  search: RouteSearch | null;
  around: RouteAround[];
  obstacles: RouteObstacle[];
  rerouted: { why: "blocked" | "off_route" | "opened"; label: string | null; kind: string | null } | null;
  was: { points: Pt[]; distance_m: number } | null;
  step: { n: number; of: number } | null;
}

/** True when the system, or `?motion=reduce`, asks for no animation (the same switch Today's Plan uses). */
export function useCalm(): boolean {
  const [calm, setCalm] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    const asked = new URLSearchParams(window.location.search).get("motion") === "reduce";
    const read = () => setCalm(asked || Boolean(mq?.matches));
    read();
    mq?.addEventListener?.("change", read);
    return () => mq?.removeEventListener?.("change", read);
  }, []);
  return calm;
}

type Preview = RoomRouteAnswer & Partial<RouteFound>;

function signature(p: Preview): string {
  // what can re-route a walk: a person or a thing it now bends around (or no longer does), or the way closing or opening.
  // Furniture comes and goes from this list as the walker moves, and that is the same route, followed.
  return JSON.stringify([(p.around ?? []).filter((a) => a.kind === "person" || a.kind === "thing").map((a) => a.id), p.reachable, p.from.track]);
}

/**
 * The walk for `item` while no round is on, with how it was found. The route follows the person a camera sees, so
 * it is refreshed quietly; it is only played again (a new `key`) when the item changes or when something came into
 * or left the way, and then it remembers the way it replaced.
 */
function usePreview(item: string, on: boolean): Found | null {
  const [found, setFound] = useState<Found | null>(null);
  const turn = useRef(0);
  useEffect(() => {
    if (!on) {
      setFound(null);
      return;
    }
    let live = true;
    const load = () =>
      api
        .roomRouteFound(item)
        .then((r: Preview) => {
          if (!live) return;
          setFound((old) => {
            const sig = signature(r);
            const same = old !== null && old.item === r.item;
            const base = {
              source: "preview" as const,
              item: r.item,
              binId: r.bin_id,
              reachable: r.reachable,
              points: r.points,
              distance_m: r.distance_m,
              from: r.from,
              search: r.search ?? null,
              around: r.around ?? [],
              obstacles: r.obstacles ?? [],
              step: null,
            };
            if (same && old.key.endsWith(sig)) return JSON.stringify(old.points) === JSON.stringify(r.points) ? old : { ...old, ...base };
            turn.current += 1;
            const key = `preview:${turn.current}:${r.item}:${sig}`;
            // a different item, or a different walker (someone stepped into view, or out of it): a new walk, not a re-route
            if (!same || old.from.track !== r.from.track) return { ...base, key, rerouted: null, was: null };
            if (!old.reachable) return { ...base, key, rerouted: r.reachable ? { why: "opened", label: null, kind: null } : null, was: null };
            const added = base.around.find((a) => (a.kind === "person" || a.kind === "thing") && !old.around.some((o) => o.id === a.id)) ?? null;
            return {
              ...base,
              key,
              rerouted: added ? { why: "blocked", label: added.label, kind: added.kind } : { why: "opened", label: null, kind: null },
              was: { points: old.points, distance_m: old.distance_m },
            };
          });
        })
        .catch(() => live && setFound(null));
    load();
    const id = window.setInterval(load, PREVIEW_EVERY_MS);
    return () => {
      live = false;
      window.clearInterval(id);
    };
  }, [item, on]);
  return found;
}

function ofPlan(plans: RoundPlans, index: number, source: "round" | "last"): Found | null {
  const p = plans.plans[index];
  if (!p) return null;
  const before = index > 0 ? plans.plans[index - 1] : undefined;
  return {
    key: `${source}:${plans.round_id}:${p.seq}`,
    source,
    item: plans.item ?? "",
    binId: p.bin_id,
    reachable: p.reachable,
    points: p.points,
    distance_m: p.distance_m,
    from: p.from,
    search: p.search,
    around: p.around,
    obstacles: p.obstacles,
    rerouted: p.reason === "replan" ? (p.rerouted ?? { why: "off_route", label: null, kind: null }) : null,
    was: p.reason === "replan" && before?.reachable ? { points: before.points, distance_m: before.distance_m } : null,
    step: { n: index + 1, of: plans.plans.length },
  };
}

export interface Pathfinding {
  found: Found | null;
  /** The last round's plans, when there is one to look back at and no round is on. */
  last: RoundPlans | null;
  /** Which of the last round's plans is shown (0-based), or null for the walk as it is now. */
  shown: number | null;
  show: (index: number | null) => void;
}

/** The route to draw on the Sort It screen and how it was found. */
export function usePathfinding(item: string, twin: RoomTwinData, calm: boolean): Pathfinding {
  const round = twin.live?.round ?? null;
  const resultId = twin.live?.result?.round_id ?? null;
  const [plans, setPlans] = useState<RoundPlans | null>(null);
  const [shown, setShown] = useState<number | null>(null);
  const asked = useRef<{ plan: number | null; replay: boolean } | null>(null);
  const hasPlan = Boolean(twin.plan);

  useEffect(() => {
    if (!hasPlan) return;
    let live = true;
    api
      .roundPlans()
      .then((p) => {
        if (!live) return;
        setPlans(p);
        if (asked.current === null) {
          // ?plan=2 opens the last round's second plan; ?replay=1 plays them through from the first
          const q = new URLSearchParams(window.location.search);
          const n = Number(q.get("plan"));
          asked.current = { plan: Number.isInteger(n) && n >= 1 ? n - 1 : null, replay: q.get("replay") === "1" };
          if (!p.live && p.plans.length > 0) {
            if (asked.current.plan !== null) setShown(Math.min(asked.current.plan, p.plans.length - 1));
            else if (asked.current.replay) setShown(0);
          }
        }
      })
      .catch(() => live && setPlans(null));
    return () => {
      live = false;
    };
  }, [hasPlan, round?.round_id, round?.replans, resultId]);

  // a round that starts takes the picture back
  useEffect(() => {
    if (round) setShown(null);
  }, [round?.round_id]); // eslint-disable-line react-hooks/exhaustive-deps

  const last = !round && plans && !plans.live && plans.plans.length > 0 ? plans : null;
  const stepping = last !== null && shown !== null;

  // the last round plays through by itself; with reduced motion, or when one plan was asked for by link, it waits for a tap
  useEffect(() => {
    if (!stepping || calm || asked.current?.plan != null || shown === null || !last || shown >= last.plans.length - 1) return;
    const id = window.setTimeout(() => setShown(shown + 1), HOLD_MS);
    return () => window.clearTimeout(id);
  }, [stepping, calm, shown, last]);

  const preview = usePreview(item, hasPlan && !round && !stepping);
  const found = useMemo(() => {
    if (round && plans?.live && plans.round_id === round.round_id) return ofPlan(plans, plans.plans.length - 1, "round");
    if (stepping && last && shown !== null) return ofPlan(last, shown, "last");
    return round ? null : preview;
  }, [round, plans, stepping, last, shown, preview]);

  return {
    found,
    last,
    shown: stepping ? shown : null,
    show: (index) => {
      if (asked.current) asked.current.plan = null; // a tap takes over from the link
      setShown(index);
    },
  };
}

// ---------- words ----------

const LIST = typeof Intl !== "undefined" && "ListFormat" in Intl ? new Intl.ListFormat("en", { style: "long", type: "conjunction" }) : null;

function named(kind: string | null, label: string | null): string {
  if (kind === "person") return "a person";
  if (kind === "wall") return "the wall";
  if (kind === "player") return "the other player's line";
  return label ? `the ${label.replace(/-/g, " ")}` : "something";
}

function metres(m: number): string {
  return `${m.toFixed(1)} m`;
}

/** The one line: "Route found with A* on a 10 cm grid: 4.2 m, around the chair." */
export function routeLine(f: Found): { kind: "found" | "rerouted" | "none"; text: string } {
  const cm = Math.round((f.search?.cell_m ?? 0.1) * 100);
  if (!f.reachable) return { kind: "none", text: `No way to the bin right now: A* searched the ${cm} cm grid and every path is blocked. It searches again when something moves.` };
  if (!f.rerouted && f.distance_m < 0.25) return { kind: "found", text: "You are at the right bin: there is no walk to find from here." };
  const things = f.around.map((a) => named(a.kind, a.label));
  const around = things.length ? `around ${LIST ? LIST.format(things) : things.join(", ")}` : "a clear walk";
  if (f.rerouted) {
    const was = f.was ? ` (it was ${metres(f.was.distance_m)})` : "";
    if (f.rerouted.why === "blocked") return { kind: "rerouted", text: `Re-routed around ${named(f.rerouted.kind, f.rerouted.label)}: ${metres(f.distance_m)} now${was}.` };
    if (f.rerouted.why === "off_route") return { kind: "rerouted", text: `Re-routed from where you are now: ${metres(f.distance_m)} to go, ${around}.` };
    return { kind: "rerouted", text: `Re-routed, the way changed: ${metres(f.distance_m)} now${was}, ${around}.` };
  }
  return { kind: "found", text: `Route found with A* on a ${cm} cm grid: ${metres(f.distance_m)}, ${around}.` };
}

function evidence(f: Found): string | null {
  const s = f.search;
  if (!s) return null;
  const open = s.blocked.reduce((n, row) => n + (row.match(/\./g)?.length ?? 0), 0);
  const again = f.rerouted ? "A* ran again on the same grid: it" : "The search";
  return `${again} looked at ${s.explored_total.toLocaleString("en-US")} of ${open.toLocaleString("en-US")} open cells before it reached the bin, and kept the ${s.path.length} that make the shortest way.`;
}

// ---------- the drawing ----------

/** Runs of one mark along a row of the blocked map, as [col, length]. */
function runs(row: string, mark: string): [number, number][] {
  const out: [number, number][] = [];
  let start = -1;
  for (let i = 0; i <= row.length; i++) {
    if (row[i] === mark) {
      if (start < 0) start = i;
    } else if (start >= 0) {
      out.push([start, i - start]);
      start = -1;
    }
  }
  return out;
}

function furnitureName(f: RoomPlan["furniture"][number]): string {
  return f.label ?? (f.bin ? humanize(f.bin) : humanize(f.type));
}

function FoundMap({ plan, found, calm }: { plan: RoomPlan; found: Found; calm: boolean }) {
  const uid = useId().replace(/[^a-zA-Z0-9]/g, "");
  const s = found.search;
  const cell = s?.cell_m ?? 0.1;
  const cols = s?.cols ?? Math.round(plan.size[0] / cell);
  const rows = s?.rows ?? Math.round(plan.size[1] / cell);
  const m = (v: number) => (v / cell) * U; // metres to drawing units
  const W = cols * U;
  const H = rows * U;
  const aroundIds = new Set(found.around.map((a) => a.id));
  const n = s?.explored.length ?? 0;
  const exploreMs = n > 0 ? EXPLORE_MS : 0;
  const pathMs = (s?.path.length ?? 0) * STEP_MS;
  const line = (pts: Pt[]) => pts.map((p) => `${m(p[0]).toFixed(1)},${m(p[1]).toFixed(1)}`).join(" ");
  const start = found.points[0] ?? ([found.from.x, found.from.y] as Pt);
  const anchor = (x: number): "start" | "middle" | "end" => (x < 26 ? "start" : x > W - 26 ? "end" : "middle");
  const words = routeLine(found).text;

  return (
    <svg className={css.map} data-calm={calm ? "true" : "false"} data-route={found.key} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`The room from above on a ${cols} by ${rows} grid of ${Math.round(cell * 100)} cm cells. ${words}`}>
      <defs>
        <pattern id={`${uid}-hatch`} patternUnits="userSpaceOnUse" width={7} height={7} patternTransform="rotate(45)">
          <line className={css.hatchLine} x1={0} y1={0} x2={0} y2={7} />
        </pattern>
        <pattern id={`${uid}-way`} patternUnits="userSpaceOnUse" width={6} height={6} patternTransform="rotate(45)">
          <line className={css.hatchLineWay} x1={0} y1={0} x2={0} y2={6} />
        </pattern>
      </defs>

      {/* the walkable grid, faint */}
      {Array.from({ length: cols - 1 }, (_, i) => (
        <line key={`v${i}`} className={css.gridLine} x1={(i + 1) * U} y1={0} x2={(i + 1) * U} y2={H} />
      ))}
      {Array.from({ length: rows - 1 }, (_, i) => (
        <line key={`h${i}`} className={css.gridLine} x1={0} y1={(i + 1) * U} x2={W} y2={(i + 1) * U} />
      ))}

      {/* cells nobody can stand in: furniture and the walls, grown by a body's width; then whatever is in the way right now */}
      {(s?.blocked ?? []).map((row, r) => runs(row, "#").map(([c, len]) => <rect key={`b${r}-${c}`} x={c * U} y={r * U} width={len * U} height={U} fill={`url(#${uid}-hatch)`} />))}
      {(s?.blocked ?? []).map((row, r) => runs(row, "o").map(([c, len]) => <rect key={`o${r}-${c}`} x={c * U} y={r * U} width={len * U} height={U} fill={`url(#${uid}-way)`} />))}

      {/* the search, the path's cells and the walk: remounted (so replayed) whenever the route's key changes */}
      <g key={found.key}>
        {(s?.explored ?? []).map((cellNo, i) => (
          <rect key={`e${i}`} className={css.seen} x={(cellNo % cols) * U + 0.5} y={Math.floor(cellNo / cols) * U + 0.5} width={U - 1} height={U - 1} style={calm ? undefined : { animationDelay: `${Math.round((i / n) * exploreMs)}ms` }} />
        ))}
        {found.was ? <polyline className={css.oldLine} points={line(found.was.points)} /> : null}
        {(s?.path ?? []).map((cellNo, i) => (
          <rect key={`p${i}`} className={css.pathCell} x={(cellNo % cols) * U + 0.5} y={Math.floor(cellNo / cols) * U + 0.5} width={U - 1} height={U - 1} rx={2} style={calm ? undefined : { animationDelay: `${exploreMs + i * STEP_MS}ms` }} />
        ))}
        {found.reachable && found.points.length > 1 ? <polyline className={css.routeLine} pathLength={1} points={line(found.points)} style={calm ? undefined : { animationDelay: `${exploreMs + pathMs}ms` }} /> : null}
      </g>

      {/* the furniture: what the walk went around is outlined and named in the warm colour */}
      {plan.furniture.map((f) => {
        const isBin = f.type === "bin";
        const cls = isBin ? (f.id === found.binId ? css.binTarget : css.binOther) : `${css.furniture} ${f.type === "rug" ? css.furnitureRug : ""} ${aroundIds.has(f.id) ? css.furnitureAround : ""}`;
        return (
          <g key={f.id} transform={`translate(${m(f.at[0])} ${m(f.at[1])}) rotate(${f.rot})`}>
            <rect className={cls} x={-m(f.size[0]) / 2} y={-m(f.size[1]) / 2} width={m(f.size[0])} height={m(f.size[1])} rx={4} />
          </g>
        );
      })}
      {plan.furniture.map((f) => {
        if (f.type === "rug") return null;
        const isBin = f.type === "bin";
        const went = aroundIds.has(f.id);
        const x = Math.min(Math.max(m(f.at[0]), 4), W - 4);
        const y = isBin ? m(f.at[1]) + m(Math.max(...f.size)) / 2 + 15 : m(f.at[1]) + 5;
        return (
          <text key={f.id} className={`${css.label} ${went ? css.labelAround : isBin && f.id === found.binId ? css.labelStrong : ""}`} x={x} y={y} textAnchor={anchor(x)}>
            {furnitureName(f)}
          </text>
        );
      })}

      {/* people and things in the way for this plan: the space kept clear around them, and where they are */}
      {found.obstacles.map((o, i) => {
        const x = m(o.x);
        const y = m(o.y);
        return (
          <g key={`${o.key ?? o.label}-${i}`}>
            <circle className={css.keepOut} cx={x} cy={y} r={m(o.r)} />
            <circle className={css.body} cx={x} cy={y} r={o.kind === "person" ? 9 : 6} />
            <text className={`${css.label} ${css.labelAround}`} x={x} y={y - (o.kind === "person" ? 14 : 11)} textAnchor={anchor(x)}>
              {o.kind === "person" ? "Person" : humanize(o.label)}
            </text>
          </g>
        );
      })}

      <circle className={css.you} cx={m(start[0])} cy={m(start[1])} r={8} />
      <text className={`${css.label} ${css.labelStrong}`} x={m(start[0])} y={m(start[1]) + (m(start[1]) > H - 30 ? -14 : 24)} textAnchor={anchor(m(start[0]))}>
        You
      </text>
    </svg>
  );
}

function Legend({ found }: { found: Found }) {
  const way = found.obstacles.length > 0;
  return (
    <ul className={css.legend} aria-label="What the map shows">
      <li>
        <span className={`${css.key} ${css.keyYou}`} aria-hidden="true" />
        where the walk starts
      </li>
      <li>
        <span className={`${css.key} ${css.keySeen}`} aria-hidden="true" />
        cells A* looked at
      </li>
      <li>
        <span className={`${css.key} ${css.keyRoute}`} aria-hidden="true" />
        the route it found
      </li>
      <li>
        <span className={`${css.key} ${css.keyBlocked}`} aria-hidden="true" />
        blocked: furniture and walls, with room for a body
      </li>
      {way ? (
        <li>
          <span className={`${css.key} ${css.keyWay}`} aria-hidden="true" />
          in the way when this was planned
        </li>
      ) : null}
      {found.around.some((a) => a.kind === "furniture") ? (
        <li>
          <span className={`${css.key} ${css.keyAround}`} aria-hidden="true" />
          what the route went around
        </li>
      ) : null}
      {found.was ? (
        <li>
          <span className={`${css.key} ${css.keyOld}`} aria-hidden="true" />
          the way before it was re-routed
        </li>
      ) : null}
    </ul>
  );
}

/** The card: the map, the line, the evidence, the legend, and the last round's plans to step through. */
export function PathfinderCard({ plan, path, calm }: { plan: RoomPlan; path: Pathfinding; calm: boolean }) {
  const { found, last, shown, show } = path;
  const line = found ? routeLine(found) : null;
  const cm = Math.round((found?.search?.cell_m ?? 0.1) * 100);
  const proof = found ? evidence(found) : null;
  return (
    <section className={ui.card} data-guide="play:path" aria-label="How the route was found">
      <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
        <h2>How the route was found</h2>
        <span className={ui.tag}>A* pathfinding · {cm} cm grid</span>
      </div>
      {!found ? (
        <p className={`${ui.muted} ${ui.small}`}>Working out the way to the bin.</p>
      ) : (
        <div className={css.finder}>
          <div className={css.mapWrap}>
            <FoundMap plan={plan} found={found} calm={calm} />
          </div>
          <div className={css.words}>
            {found.step && found.source === "last" ? (
              <p className={`${ui.muted} ${ui.small}`}>
                The last round, plan {found.step.n} of {found.step.of}: {found.rerouted ? "planned again" : "the first plan"}.
              </p>
            ) : null}
            <p className={css.line} data-kind={line?.kind} data-guide="play:route-line" role="status">
              {line?.text}
            </p>
            {found.source === "preview" ? (
              <p className={`${ui.muted} ${ui.small}`}>
                {humanize(found.item)} · {found.from.track ? "from where the camera sees you" : "from the room's start point: no camera sees anyone"}.
              </p>
            ) : null}
            {proof ? <p className={css.evidence}>{proof}</p> : null}
            <Legend found={found} />
            {calm ? <p className={`${ui.muted} ${ui.small}`}>Reduced motion is on: the search and the route are shown finished, nothing plays.</p> : null}
          </div>
        </div>
      )}
      {last ? (
        <div className={css.replay}>
          <p className={css.replayNote}>
            The last round ({humanize(last.item ?? "")}) was planned {last.plans.length === 1 ? "once" : `${last.plans.length} times`}
            {last.plans.some((p) => p.rerouted?.why === "blocked") ? ": something moved into the way." : "."}
          </p>
          {shown === null ? (
            <button type="button" className={css.replayBtn} onClick={() => show(0)}>
              Show the last round&apos;s plans
            </button>
          ) : (
            <>
              {last.plans.map((p, i) => (
                <button key={p.seq} type="button" className={css.replayBtn} aria-pressed={shown === i} onClick={() => show(i)}>
                  {i === 0 ? "First plan" : `Re-plan ${i}`}
                </button>
              ))}
              <button type="button" className={css.replayBtn} onClick={() => show(null)}>
                Back to now
              </button>
            </>
          )}
        </div>
      ) : null}
    </section>
  );
}
