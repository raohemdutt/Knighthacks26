"use client";

import { useCallback, useEffect, useMemo, useRef, useState, type PointerEvent } from "react";
import { api, type Board, type HardResult, type PlacePack, type SimResult, type Tick } from "@/lib/api";
import { pinKind, type CampusMap } from "@/lib/campusmap";
import {
  DEFAULT_SPEED,
  ENERGY_LOW,
  SPEEDS,
  alongWalk,
  buildShow,
  buildWorld,
  clockAt,
  countdown,
  friendsAt,
  knownEdges,
  livedWalks,
  paceAt,
  routeCheck,
  showMoment,
  timeOfClock,
  walkedPoints,
  type Hold,
  type Moment,
  type Pt,
  type Show,
  type Span,
  type Walk,
  type World,
} from "@/lib/dayplay";
import { clock, clockSpan, dateLabel, minutes, ms, parentPlace, placeName, riskPercent, stars, type PlaceIndex } from "@/lib/dayrun";
import ui from "../ui.module.css";
import cm from "./campusmap.module.css";
import dr from "./dayrun.module.css";
import { CampusLayers, MapFurniture, PlacePin, layoutPins, lineBoxes, type ScreenBox } from "./CampusLayers";
import { pinName } from "./CampusMap";
import { Figure } from "./Figure";

/**
 * Play mode: the simulated day, watched. A small figure lives the plan on the
 * campus map in accelerated time while the clock, the meters and the feed run
 * with it. Every position and number is read from the result the box returned
 * (see lib/dayplay.ts); this component only draws the moment it is given.
 * On a pack with a drawn map the figure walks the real footpaths over the
 * campus's buildings, greens and roads (© OpenStreetMap contributors).
 *
 * In hard mode the calm day plays until the disruptions arrive, the remaining
 * route is re-planned on screen (old path out, new path in, with the box's own
 * change texts), and the disrupted day plays from there.
 */

const W = 360;
const H = 340;
const CW = 360;
const CH = 80;
const CTOP = 5;
const CBOT = 60;
const MIN = 60000;

interface Cam {
  cx: number;
  cy: number;
  z: number;
}

interface Frame {
  t: number;
  hold: number | null; // index into show.holds while the clock is held
  holdF: number; // 0..1 through the hold
  next: number; // the next hold not yet played
  cam: Cam | null; // null: snap to the target
  phase: number; // real seconds, for the walk cycle
  seen: number; // feed items at or before t
  flash: number; // feed item being flashed, or -1
  flashAge: number;
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
const ease = (v: number) => {
  const x = clamp(v, 0, 1);
  return x * x * (3 - 2 * x);
};

function fit(points: Pt[], drawn: boolean): Cam {
  if (points.length === 0) return { cx: 500, cy: 500, z: 0.3 };
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const y0 = Math.min(...ys);
  const y1 = Math.max(...ys);
  // Room above for the figure and its cue, below for the banner, and either side for the labels.
  // On a drawn map the names sit under their pins, so the route can use more of the width.
  const z = drawn
    ? Math.min((W - 96) / Math.max(x1 - x0, 170), (H - 128) / Math.max(y1 - y0, 170))
    : Math.min((W - 200) / Math.max(x1 - x0, 260), (H - 132) / Math.max(y1 - y0, 260));
  return { cx: (x0 + x1) / 2, cy: (y0 + y1) / 2 - 12 / z, z };
}

/**
 * What the camera frames: the route being chosen or walked; while standing, the walk
 * just finished and the one coming next; the whole day while it is being re-planned.
 */
function cameraTarget(show: Show, walks: Walk[], frame: Frame, moment: Moment, drawn: boolean): Cam {
  const hold = frame.hold !== null ? show.holds[frame.hold] : undefined;
  if (hold?.kind === "reject" && hold.ghost) return fit([...hold.ghost.points, moment.pt], drawn);
  if (hold?.kind === "replan") return fit(show.extent, drawn);
  const walk = hold?.kind === "route" ? hold.walk : moment.walk;
  if (walk && walk.points.length > 1) return fit(walk.points, drawn);
  if (frame.t >= show.t1) return fit(show.extent, drawn);
  let prev: Walk | undefined;
  let next: Walk | undefined;
  for (const w of walks) {
    if (w.points.length < 2) continue; // a walk between rooms: nothing to frame
    if (w.t1 <= frame.t) prev = w;
    else if (!next && w.t0 > frame.t) next = w;
  }
  return fit([...(prev?.points ?? []), ...(next?.points ?? []), moment.pt], drawn);
}

const feedSeen = (show: Show, t: number) => {
  let n = 0;
  while (n < show.feed.length && (show.feed[n]?.t ?? Infinity) <= t) n++;
  return n;
};

/** A still frame at `t`. With `freeze`, a hold that starts exactly there is shown complete (its last frame). */
function frameAt(show: Show, at: number, freeze: boolean): Frame {
  const t = clamp(at, show.t0, show.t1);
  const first = show.holds.findIndex((h) => h.t >= t);
  const next = first < 0 ? show.holds.length : first;
  const base = { t, cam: null, phase: 0, seen: feedSeen(show, t), flash: -1, flashAge: 0 };
  if (freeze && show.holds[next]?.t === t) return { ...base, hold: next, holdF: 1, next: next + 1 };
  return { ...base, hold: null, holdF: 0, next };
}

export function DayPlayer({
  packs,
  places,
  map,
  board,
  result,
  hard,
  autoPlay,
  startClock,
  seek,
  onTick,
  onClose,
}: {
  packs: PlacePack[];
  places: PlaceIndex;
  /** The pack's drawn map, when it has one. Pass it loaded: a map arriving later starts the day again. */
  map: CampusMap | null;
  board: Board;
  result: SimResult;
  hard: HardResult | null;
  autoPlay: boolean;
  /** "14:05": open on that minute of the day. */
  startClock: string | null;
  /** A scrub made elsewhere on the page (the meter chart, a route row): `n` changes with each one. */
  seek: { index: number; n: number } | null;
  onTick: (index: number) => void;
  onClose: () => void;
}) {
  const world = useMemo(() => buildWorld(places, map), [places, map]);
  const drawn = map !== null;
  const show = useMemo(() => buildShow({ result, hard, world, places, board }), [result, hard, world, places, board]);
  const edges = useMemo(() => knownEdges(world, hard ? [hard.before, hard.after] : [result], board), [world, hard, result, board]);
  const walks = useMemo(() => livedWalks(show), [show]);

  const [frame, setFrame] = useState<Frame>(() => frameAt(show, show.t0, false));
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState<number>(DEFAULT_SPEED);
  const [reduced, setReduced] = useState(false);
  const frameRef = useRef(frame);
  const firstShow = useRef(show);
  const chart = useRef<SVGSVGElement>(null);

  const put = useCallback((f: Frame) => {
    frameRef.current = f;
    setFrame(f);
  }, []);

  // Reduced motion: the system setting, or ?motion=reduce on the link.
  useEffect(() => {
    const forced = new URLSearchParams(window.location.search).get("motion") === "reduce";
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const set = () => setReduced(forced || mq.matches);
    set();
    mq.addEventListener("change", set);
    return () => mq.removeEventListener("change", set);
  }, []);

  // A new day to play (first mount, a changed plan, hard mode switched on or off): back to the start.
  useEffect(() => {
    const first = firstShow.current === show;
    const at = first ? timeOfClock(show, startClock) : null;
    put(frameAt(show, at ?? show.t0, !(first && autoPlay)));
    setPlaying(first && autoPlay);
    // startClock and autoPlay are read once, for the first day only.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [show, put]);

  const advance = useCallback(
    (f: Frame, dt: number): Frame => {
      let { t, hold, holdF, next } = f;
      const holdScale = clamp(DEFAULT_SPEED / speed, 0.6, 1);
      let rem = dt;
      while (rem > 1e-6 && t < show.t1) {
        const step = Math.min(rem, 0.03);
        rem -= step;
        if (hold !== null) {
          holdF += step / ((show.holds[hold]?.seconds ?? 1) * holdScale);
          if (holdF >= 1) {
            hold = null;
            holdF = 0;
          }
          continue;
        }
        const h = show.holds[next];
        const tn = t + step * paceAt(show, t, speed) * MIN;
        if (h && h.t <= tn) {
          t = h.t;
          hold = next;
          holdF = 0;
          next += 1;
        } else t = Math.min(tn, show.t1);
      }
      const seen = feedSeen(show, t);
      let { flash, flashAge } = f;
      flashAge += dt;
      if (seen > f.seen) {
        const fresh = show.feed.slice(f.seen, seen);
        const bad = fresh.findIndex((x) => x.tone === "bad");
        flash = f.seen + (bad >= 0 ? bad : fresh.length - 1);
        flashAge = 0;
      }
      const out: Frame = { t, hold, holdF, next, cam: f.cam, phase: f.phase + dt, seen, flash, flashAge };
      const target = cameraTarget(show, walks, out, showMoment(show, world, t), drawn);
      const k = 1 - Math.exp(-dt * 3.2);
      out.cam = f.cam ? { cx: f.cam.cx + (target.cx - f.cam.cx) * k, cy: f.cam.cy + (target.cy - f.cam.cy) * k, z: f.cam.z + (target.z - f.cam.z) * k } : target;
      return out;
    },
    [show, speed, world, walks, drawn],
  );

  const step = useCallback(
    (f: Frame, dir: 1 | -1): Frame => {
      if (dir > 0) {
        if (f.hold !== null && show.holds[f.hold + 1]?.t === f.t) return { ...f, hold: f.hold + 1, holdF: 1, next: f.hold + 2 };
        return frameAt(show, show.stops.find((s) => s > f.t + 1) ?? show.t1, true);
      }
      let prev = show.t0;
      for (const s of show.stops) if (s < f.t - 1) prev = s;
      return frameAt(show, prev, true);
    },
    [show],
  );

  // The clock. One animation frame at a time; stepping block by block when motion is reduced.
  useEffect(() => {
    if (!playing) return;
    if (reduced) {
      const id = window.setInterval(() => {
        const nf = step(frameRef.current, 1);
        put(nf);
        if (nf.t >= show.t1) setPlaying(false);
      }, 1700);
      return () => window.clearInterval(id);
    }
    let raf = 0;
    let last = performance.now();
    const loop = (now: number) => {
      const dt = clamp((now - last) / 1000, 0, 0.25);
      last = now;
      const nf = advance(frameRef.current, dt);
      put(nf);
      if (nf.t >= show.t1) {
        setPlaying(false);
        return;
      }
      raf = window.requestAnimationFrame(loop);
    };
    raf = window.requestAnimationFrame(loop);
    return () => window.cancelAnimationFrame(raf);
  }, [playing, reduced, advance, step, put, show.t1]);

  const moment = useMemo(() => showMoment(show, world, frame.t), [show, world, frame.t]);
  const tickIndex = moment.meters.index;
  useEffect(() => {
    onTick(tickIndex);
  }, [tickIndex, onTick]);

  // A scrub from the rest of the page.
  const lastSeek = useRef(seek?.n ?? 0);
  useEffect(() => {
    if (!seek || seek.n === lastSeek.current) return;
    lastSeek.current = seek.n;
    const tick = show.main.result.timeline[seek.index];
    if (tick) put(frameAt(show, ms(tick.t), false));
  }, [seek, show, put]);

  const ended = frame.t >= show.t1;
  const hold: Hold | undefined = frame.hold !== null ? show.holds[frame.hold] : undefined;
  const script = moment.script;
  const cam = frame.cam ?? cameraTarget(show, walks, frame, moment, drawn);
  const S = (p: Pt): Pt => ({ x: (p.x - cam.cx) * cam.z + W / 2, y: (p.y - cam.cy) * cam.z + H / 2 });
  const path = (points: Pt[]) =>
    points
      .map((p, i) => {
        const s = S(p);
        return `${i ? "L" : "M"}${s.x.toFixed(1)} ${s.y.toFixed(1)}`;
      })
      .join(" ");
  const still = !playing || reduced;
  const split = show.splitAt;
  const disrupted = split !== null && frame.t >= split;
  const replan = hold?.kind === "replan" ? frame.holdF : null;

  const toggle = () => {
    if (ended) put(frameAt(show, show.t0, false));
    setPlaying((p) => !p || ended);
  };
  const restart = () => {
    put(frameAt(show, show.t0, false));
    setPlaying(true);
  };
  const stepBy = (dir: 1 | -1) => {
    setPlaying(false);
    put(step(frameRef.current, dir));
  };
  const pick = (e: PointerEvent<SVGSVGElement>) => {
    const box = chart.current?.getBoundingClientRect();
    if (!box || !box.width) return;
    const f = clamp((((e.clientX - box.left) / box.width) * CW - 6) / (CW - 12), 0, 1);
    put(frameAt(show, show.t0 + f * (show.t1 - show.t0), false));
  };

  // ---------- what the map shows ----------
  const touched = useMemo(() => {
    const refs = new Set<string>();
    const add = (ref: string | null | undefined) => {
      if (!ref) return;
      if (world.at.has(ref)) refs.add(ref);
      else {
        const parent = parentPlace(places, ref)?.ref;
        if (parent && world.at.has(parent)) refs.add(parent);
      }
    };
    for (const s of show.before ? [show.before, show.main] : [show.main]) {
      add(s.startPlace);
      for (const w of s.walks) for (const p of w.route.places) add(p);
      for (const x of s.spans) add(x.block.place);
    }
    for (const d of show.disruptions) {
      add(d.place);
      add(d.to_place);
    }
    for (const h of show.holds) for (const ref of h.ghost?.option.route ?? []) add(ref);
    for (const f of board.friends) add(f.place);
    return refs;
  }, [show, world, places, board.friends]);
  // On a drawn map: the buildings of the places the day touches are picked out, and their pins carry the names.
  const planBuildings = useMemo(() => {
    const of = (ref: string) => map?.places.get(ref.split(".").slice(1).join("."))?.building;
    const on = new Set<number>();
    const all = new Set<number>();
    for (const ref of world.at.keys()) {
      const b = of(ref);
      if (b === undefined) continue;
      all.add(b);
      if (touched.has(ref)) on.add(b);
    }
    return {
      on,
      all,
      shut: (refs: Set<string>) => {
        const out = new Set<number>();
        for (const ref of refs) {
          const b = of(ref);
          if (b !== undefined) out.add(b);
        }
        return out;
      },
    };
  }, [map, world, touched]);
  const midX = useMemo(() => {
    const xs = [...touched].map((ref) => world.at.get(ref)?.x ?? 0).sort((a, b) => a - b);
    return xs.length ? ((xs[Math.floor((xs.length - 1) / 2)] ?? 0) + (xs[Math.ceil((xs.length - 1) / 2)] ?? 0)) / 2 : 0;
  }, [touched, world]);

  // A rejected move's route along the real paths: asked from the box once per pair of places, straight lines until it answers.
  const [realLines, setRealLines] = useState<Map<string, Pt[]>>(() => new Map());
  useEffect(() => {
    const project = world.project;
    if (!project) return;
    const ctl = new AbortController();
    const pairs = new Set<string>();
    for (const h of show.holds) for (let i = 1; i < (h.ghost?.refs.length ?? 0); i++) pairs.add(`${h.ghost?.refs[i - 1]}>${h.ghost?.refs[i]}`);
    for (const key of pairs) {
      const [from, to] = key.split(">") as [string, string];
      if (from === to || !world.at.has(from) || !world.at.has(to)) continue;
      api
        .placeRoute({ from, to }, ctl.signal)
        .then((r) => {
          if (r.polyline.length > 1) setRealLines((m) => new Map(m).set(key, r.polyline.map(([lat, lon]) => project(lat, lon))));
        })
        .catch(() => undefined); // the straight line stays
    }
    return () => ctl.abort();
  }, [show, world]);
  const ghostPoints = useMemo(() => {
    const g = hold?.ghost;
    if (!g) return [] as Pt[];
    const out: Pt[] = [];
    for (let i = 1; i < g.refs.length; i++) {
      const a = world.resolve(g.refs[i - 1] as string);
      const b = world.resolve(g.refs[i] as string);
      const line = realLines.get(`${g.refs[i - 1]}>${g.refs[i]}`) ?? (a && b ? [a, b] : []);
      for (const p of line) if (!out.length || Math.hypot(p.x - (out[out.length - 1] as Pt).x, p.y - (out[out.length - 1] as Pt).y) > 0.5) out.push(p);
    }
    return out.length > 1 ? out : g.points;
  }, [hold, realLines, world]);

  const closedShown = split !== null && (frame.t > split || (frame.t === split && (replan === null || replan > 0.12)));
  const closed = new Set(closedShown ? show.disruptions.filter((d) => d.kind === "place_closed" && d.place).map((d) => d.place as string) : []);

  const current = hold?.kind === "route" ? hold.walk : moment.walk;
  const previewF = hold?.kind === "route" ? clamp(frame.holdF / 0.75, 0, 1) : 1;
  const walkNumber = current ? walks.indexOf(current) + 1 : 0;
  const check = current ? routeCheck(script, current) : null;

  // The next fixed block: the flag the route has to reach.
  const fixedNow = script.spans.find((s) => s.block.fixed && s.s <= frame.t && frame.t < s.e);
  const fixedNext = script.spans.find((s) => s.block.fixed && s.s > frame.t);
  const flagSpan = fixedNow ?? fixedNext;
  const moved =
    replan !== null && show.before
      ? show.disruptions
          .filter((d) => d.kind === "block_moved" && d.block)
          .map((d) => ({ from: show.before?.spans.find((s) => s.block.id === d.block), to: show.main.spans.find((s) => s.block.id === d.block) }))
          .filter((m): m is { from: Span; to: Span } => Boolean(m.from && m.to))
      : [];

  const deadline = [...board.deadlines].sort((a, b) => ms(a.due) - ms(b.due)).find((d) => ms(d.due) > frame.t);
  const friends = friendsAt(show, board.friends, frame.t);
  const now = clockAt(show, frame.t);
  const dayName = dateLabel(now.day ? show.main.result.end : show.main.result.start).slice(0, 3);
  const pace = paceAt(show, frame.t, speed);
  const paceText = ended
    ? "day over"
    : hold?.kind === "route"
      ? "choosing a route"
      : hold?.kind === "replan"
        ? "re-planning"
        : hold?.kind === "reject"
          ? "checking a move"
          : !playing
            ? "paused"
            : reduced
              ? "stepping block by block"
              : `${pace >= 10 ? Math.round(pace) : pace.toFixed(1)} min/s${moment.walk ? " · walking pace" : moment.doing === "sleep" ? " · night" : ""}`;

  const hereIndoors = world.indoors(moment.place);
  const whereText = moment.walk
    ? moment.hop
      ? `${placeName(places, moment.hop.from)} → ${placeName(places, moment.hop.to)}${moment.hop.indoors ? " (indoors)" : ""} · leg ${moment.walk.hops.indexOf(moment.hop) + 1} of ${moment.walk.hops.length}`
      : "Walking"
    : hereIndoors
      ? `Inside ${parentPlace(places, moment.place)?.name ?? "a building"}: ${placeName(places, moment.place)}`
      : `At ${placeName(places, moment.place)}`;
  const nowText = moment.walk ? `Walking ${moment.walk.route.purpose}` : moment.span ? moment.span.block.title : `Free time at ${placeName(places, moment.place)}`;
  const nextText = fixedNow
    ? `${fixedNow.block.title} ends in ${countdown(fixedNow.e - frame.t)}`
    : fixedNext
      ? `${fixedNext.block.title} in ${countdown(fixedNext.s - frame.t)}`
      : "No fixed block left today";
  const dueText = deadline ? `${deadline.title} due in ${countdown(ms(deadline.due) - frame.t)} · ${minutes(moment.meters.workLeft[deadline.id] ?? deadline.work_left)} of work left` : "No deadline in reach";

  const flashItem = !ended && frame.flash >= 0 && frame.flashAge < 3.2 && hold?.kind !== "replan" && hold?.kind !== "reject" ? show.feed[frame.flash] : undefined;
  const flashPt = flashItem ? (flashItem.place ? world.resolve(flashItem.place) : undefined) ?? moment.pt : undefined;
  const recent = show.feed.slice(0, frame.seen).slice(-3).reverse();

  const me = S(moment.pt);
  const facing: 1 | -1 = moment.dx < -0.01 ? -1 : 1;

  // On a drawn map: pins with names that keep off each other, the figure and the route; the other buildings' names keep off all of those.
  const pinsOnMap = map
    ? (() => {
        const taken: ScreenBox[] = [{ x0: me.x - 11, y0: me.y - 34, x1: me.x + 11, y1: me.y + 4 }];
        const spots = [...world.at.entries()]
          .map(([ref, p]) => {
            const s = S(p);
            const place = places.get(ref);
            const big = touched.has(ref) || closed.has(ref);
            return { key: ref, x: s.x, y: s.y, big, text: `${place ? pinName(place.name, place.aliases) : ref}${closed.has(ref) ? " · closed" : ""}`, named: big || cam.z > 0.55, kind: pinKind(place?.kind) };
          })
          .filter((s) => s.x > -20 && s.x < W + 20 && s.y > -20 && s.y < H + 20 && (s.big || cam.z > 0.3));
        const where = layoutPins(spots, taken);
        if (current) taken.push(...lineBoxes(cam, W, H, current.points));
        return { spots, where, taken };
      })()
    : null;
  const ghostF = hold?.kind === "reject" ? frame.holdF : 0;

  // ---------- the strip chart ----------
  const cx = (t: number) => 6 + ((t - show.t0) / (show.t1 - show.t0 || 1)) * (CW - 12);
  const cy = (v: number) => CTOP + (1 - clamp(v, 0, 100) / 100) * (CBOT - CTOP);
  const lived = useMemo(() => {
    const cut = show.splitAt ?? show.t0;
    const ticks: Tick[] = show.before ? [...show.before.result.timeline.filter((x) => ms(x.t) < cut), ...show.main.result.timeline.filter((x) => ms(x.t) >= cut)] : show.main.result.timeline;
    const spans: Span[] = show.before ? [...show.before.spans.filter((x) => x.s < cut), ...show.main.spans.filter((x) => x.s >= cut)] : show.main.spans;
    return { ticks, spans };
  }, [show]);
  const series = [
    { id: "energy", name: "Energy", cls: dr.sEnergy, value: (t: Tick) => t.energy, now: moment.meters.energy, text: String(Math.round(moment.meters.energy)), low: moment.meters.energy < ENERGY_LOW },
    { id: "focus", name: "Focus", cls: dr.sFocus, value: (t: Tick) => t.focus, now: moment.meters.focus, text: String(Math.round(moment.meters.focus)), low: false },
    { id: "risk", name: "Deadline risk", cls: dr.sRisk, value: (t: Tick) => t.deadline_risk * 100, now: moment.meters.risk * 100, text: riskPercent(moment.meters.risk), low: false },
  ];
  const hours = useMemo(() => {
    const out: { t: number; label: string }[] = [];
    for (let t = show.t0; t <= show.t1; t += 15 * MIN) {
      const c = clockAt(show, t).hhmm;
      if (c.endsWith(":00") && Number(c.slice(0, 2)) % 6 === 0) out.push({ t, label: c });
    }
    return out;
  }, [show]);

  const finalResult = show.main.result;

  return (
    <div className={`${ui.card} ${dr.plStage}`} data-playing={playing ? "1" : "0"}>
      <div className={dr.plHead}>
        <p className={dr.plClock} aria-label={`Simulated time ${dayName} ${now.hhmm}`}>
          <span className={dr.plDay}>{dayName}</span> {now.hhmm}
        </p>
        <p className={`${dr.plPace} ${hold ? dr.plPaceHold : ""}`}>{paceText}</p>
        <button type="button" className={dr.pill} onClick={onClose} aria-label="Close the playback">
          Close
        </button>
      </div>
      <p className={dr.plLine}>
        <strong>Now</strong> {nowText}
      </p>
      <p className={dr.plLine}>
        <strong>Next</strong> {nextText}
      </p>
      <p className={`${dr.plLine} ${ui.muted}`}>{dueText}</p>

      <div className={dr.plCols}>
        <div className={dr.plCol}>
      <div className={`${dr.plMapWrap} ${cm.palette}`}>
        <svg className={dr.plMap} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Campus map. ${dayName} ${now.hhmm}: ${nowText}. ${whereText}.`}>
          {/* the campus itself, or on a hand-made pack the walkways the box has routed over */}
          {map ? (
            <CampusLayers map={map} view={cam} width={W} height={H} highlight={{ on: planBuildings.on, closed: planBuildings.shut(closed) }} quiet={planBuildings.all} taken={pinsOnMap?.taken} />
          ) : (
            edges.map(([a, b]) => {
              const p = S(world.at.get(a) as Pt);
              const q = S(world.at.get(b) as Pt);
              return <line key={`${a}|${b}`} className={dr.plEdge} x1={p.x} y1={p.y} x2={q.x} y2={q.y} />;
            })
          )}

          {/* the plan ahead, the trail behind */}
          {walks.map((w) =>
            w.points.length < 2 || w === current ? null : w.t1 <= frame.t ? (
              <path key={`trail-${w.t0}`} className={dr.plTrail} d={path(w.points)} />
            ) : replan !== null ? null : (
              <path key={`ahead-${w.t0}`} className={dr.plAhead} d={path(w.points)} />
            ),
          )}

          {/* hard mode: the old remaining route goes, the new one is drawn */}
          {replan !== null && show.before && split !== null ? (
            <>
              {show.before.walks
                .filter((w) => w.t0 >= split && w.points.length > 1)
                .map((w) => (
                  <path key={`old-${w.t0}`} className={dr.plOld} d={path(w.points)} opacity={1 - ease((replan - 0.35) / 0.35)} />
                ))}
              {replan > 0.4
                ? show.main.walks
                    .filter((w) => w.t0 >= split && w.points.length > 1)
                    .map((w) => <path key={`new-${w.t0}`} className={dr.plNew} d={path(w.points)} pathLength={1} strokeDasharray={`${ease((replan - 0.4) / 0.45).toFixed(3)} 1`} />)
                : null}
            </>
          ) : null}

          {/* the walk being chosen or walked */}
          {current && current.points.length > 1 ? (
            <>
              <path className={dr.plRoute} d={path(current.points)} pathLength={1} strokeDasharray={`${previewF.toFixed(3)} 1`} />
              {moment.walk === current && hold?.kind !== "route" ? <path className={dr.plWalked} d={path(walkedPoints(current, moment.walkF))} /> : null}
            </>
          ) : null}

          {/* a rejected move: the route the box tried */}
          {hold?.kind === "reject" && hold.ghost ? <path className={dr.plGhost} d={path(ghostPoints)} pathLength={1} strokeDasharray={`${clamp(ghostF / 0.6, 0, 1).toFixed(3)} 1`} /> : null}

          {pinsOnMap
            ? pinsOnMap.spots.map((s) => {
                const below = pinsOnMap.where.get(s.key) ?? null;
                return <PlacePin key={s.key} x={s.x} y={s.y} kind={s.kind} small={!s.big} closed={closed.has(s.key)} label={below === null ? undefined : s.text} below={below ?? true} />;
              })
            : null}
          {(map ? [] : [...world.at.entries()]).map(([ref, p]) => {
            const s = S(p);
            if (s.x < -40 || s.x > W + 40 || s.y < -20 || s.y > H + 20) return null;
            const place = places.get(ref);
            const on = touched.has(ref);
            const shut = closed.has(ref);
            const left = p.x < midX;
            const full = place ? ((place.name.length <= 15 ? place.name : [...place.aliases].filter((a) => a.length <= 15).sort((a, b) => b.length - a.length)[0]) ?? place.name) : ref;
            const name = full.length > 16 ? `${full.slice(0, 15).trimEnd()}…` : full;
            return (
              <g key={ref}>
                <circle className={shut ? dr.placeClosed : on ? dr.place : dr.plPlaceOff} cx={s.x} cy={s.y} r={on ? 4.5 : 3} />
                {shut ? <path className={dr.plShut} d={`M${s.x - 7} ${s.y - 7} L${s.x + 7} ${s.y + 7} M${s.x + 7} ${s.y - 7} L${s.x - 7} ${s.y + 7}`} /> : null}
                {on || cam.z > 0.55 ? (
                  <text className={`${dr.placeLabel} ${shut ? dr.plLabelShut : on ? "" : dr.plLabelOff}`} x={s.x + (left ? -9 : 9)} y={s.y + 4} textAnchor={left ? "end" : "start"}>
                    {name}
                    {shut ? " · closed" : ""}
                  </text>
                ) : null}
              </g>
            );
          })}

          {/* minutes so far at each stop of the route being chosen */}
          {hold?.kind === "route" && current
            ? current.hops
                .filter((h) => !h.indoors && current.total > 0 && previewF >= (h.m0 + h.minutes) / current.total - 0.001)
                .map((h) => {
                  const s = S(h.b);
                  return (
                    <g key={`${h.from}>${h.to}`}>
                      <circle className={dr.plNode} cx={s.x} cy={s.y} r={7} />
                      <text className={dr.plNodeText} x={s.x} y={s.y - 11} textAnchor="middle">
                        {minutes(h.m0 + h.minutes)}
                      </text>
                    </g>
                  );
                })
            : null}

          {/* the next fixed block: where the route has to end up, and when */}
          {moved.length > 0
            ? moved.map(({ from, to }) => {
                const a = world.resolve(from.block.place);
                const b = world.resolve(to.block.place);
                if (!a || !b || replan === null) return null;
                const g = ease(replan / 0.3);
                const s = S({ x: a.x + (b.x - a.x) * g, y: a.y + (b.y - a.y) * g });
                return <Flag key={from.block.id} x={s.x} y={s.y} text={clock(g < 0.5 ? from.block.start : to.block.start)} hot />;
              })
            : flagSpan && world.resolve(flagSpan.block.place)
              ? (() => {
                  const s = S(world.resolve(flagSpan.block.place) as Pt);
                  return <Flag x={s.x} y={s.y} text={clock(flagSpan.block.start)} hot={false} />;
                })()
              : null}

          {flashPt && flashItem ? <circle key={`pulse-${frame.flash}`} className={`${dr.plPulse} ${flashItem.tone === "bad" ? dr.plPulseBad : flashItem.tone === "warn" ? dr.plPulseWarn : ""}`} cx={S(flashPt).x} cy={S(flashPt).y - 10} r={16} /> : null}

          {friends.map((f, i) => {
            const p = world.resolve(f.place);
            if (!p) return null;
            const s = S(p);
            const beside = !moment.walk && (moment.place === f.place || world.resolve(moment.place) === p);
            return <Figure key={f.member} x={s.x + (beside ? -15 : 0) - i * 13} y={s.y + (beside ? 3 : 0)} dir={1} phase={frame.phase + 0.4 + i} still={still} tone="friend" label={f.member} />;
          })}

          {hold?.kind === "reject" && hold.ghost && ghostF > 0.02 && ghostF < 0.62
            ? (() => {
                // by distance along the line, so the ghost keeps one pace on a path with many bends
                const pts = ghostPoints;
                const cum = [0];
                for (let n = 1; n < pts.length; n++) cum.push((cum[n - 1] as number) + Math.hypot((pts[n] as Pt).x - (pts[n - 1] as Pt).x, (pts[n] as Pt).y - (pts[n - 1] as Pt).y));
                const d = clamp(ghostF / 0.6, 0, 1) * (cum[cum.length - 1] as number);
                let i = 0;
                while (i < pts.length - 2 && (cum[i + 1] as number) < d) i++;
                const a = pts[i] as Pt;
                const b = pts[i + 1] as Pt;
                const g = (cum[i + 1] as number) > (cum[i] as number) ? (d - (cum[i] as number)) / ((cum[i + 1] as number) - (cum[i] as number)) : 0;
                const s = S({ x: a.x + (b.x - a.x) * g, y: a.y + (b.y - a.y) * g });
                return <Figure x={s.x} y={s.y} dir={b.x < a.x ? -1 : 1} doing="walk" phase={frame.phase} still={still} tone="ghost" />;
              })()
            : null}

          <Figure x={me.x} y={me.y} dir={facing} doing={moment.doing} phase={frame.phase} still={still} thinking={hold !== undefined} scale={1.15} />
          {map ? <MapFurniture map={map} view={cam} width={W} height={H} /> : null}
        </svg>

        {flashItem ? (
          <p key={`flash-${frame.flash}`} className={`${dr.plFlash} ${flashItem.tone === "bad" ? dr.plFlashBad : flashItem.tone === "warn" ? dr.plFlashWarn : flashItem.tone === "ok" ? dr.plFlashOk : ""}`} role="status">
            {flashItem.text}
          </p>
        ) : null}

        {ended ? (
          <div className={`${dr.plEnd} ${finalResult.outcome.won ? dr.plEndWin : dr.plEndFail}`} role="status" aria-live="polite">
            <div className={`${ui.row} ${ui.between}`}>
              <p className={dr.verdictWord}>
                <span aria-hidden="true">{finalResult.outcome.won ? "✓ " : "✗ "}</span>
                {finalResult.outcome.won ? "Win" : "Fail"}
              </p>
              <p className={`${dr.stars} ${finalResult.stars === 0 ? dr.starsNone : ""}`} aria-label={`${finalResult.stars} of 3 stars`}>
                {stars(finalResult.stars)}
              </p>
            </div>
            <div className={`${ui.row} ${ui.between}`}>
              <p className={dr.plEndScore}>
                {finalResult.totals.points} points of 100 · {minutes(finalResult.totals.walk_minutes)} walked
              </p>
              <button type="button" className={`${dr.pill} ${dr.pillOn}`} onClick={restart}>
                Watch again
              </button>
            </div>
            {finalResult.outcome.won ? (
              <p className={ui.breakWords}>{finalResult.outcome.summary}</p>
            ) : (
              <ul className={dr.bullets}>
                {finalResult.outcome.reasons.map((r, i) => (
                  <li key={i} className={ui.breakWords}>
                    {r}
                  </li>
                ))}
              </ul>
            )}
            {finalResult.rejected.map((r) => (
              <p key={r.index} className={`${ui.breakWords} ${ui.small}`}>
                Move {r.index + 1} was rejected: {r.reason}
              </p>
            ))}
            {show.hard ? (
              <p className={`${ui.breakWords} ${ui.small}`}>
                Hard mode: {show.disruptions.length} {show.disruptions.length === 1 ? "thing" : "things"} went wrong{split !== null ? ` at ${clockAt(show, split).hhmm}` : ""}, the box made {show.changes.length}{" "}
                {show.changes.length === 1 ? "change" : "changes"}, and the day {show.hard.after.outcome.won ? (show.hard.before.outcome.won ? "still wins" : "now wins") : "fails"} ({show.hard.before.totals.points} → {show.hard.after.totals.points} points).
              </p>
            ) : null}
          </div>
        ) : null}
      </div>

      <p className={`${dr.plLine} ${dr.plWhere}`}>{whereText}</p>
      {map ? <p className={`${ui.muted} ${ui.small}`}>Map data {map.attribution}. The figure follows the mapped footpaths.</p> : null}
      {hereIndoors && !moment.walk ? <Rooms packs={packs} places={places} here={moment.place} /> : null}
        </div>
        <div className={dr.plCol}>
      <div className={dr.plCaption} aria-live="off">
        <Caption show={show} hold={hold} holdF={frame.holdF} moment={moment} places={places} current={current} walkNumber={walkNumber} walkCount={walks.length} check={check} disrupted={disrupted} />
      </div>

      <ul className={dr.plBars}>
        {series.map((s) => (
          <li key={s.id} className={s.cls}>
            <span className={dr.plBarName}>{s.name}</span>
            <span className={`${dr.plBar} ${s.low ? dr.plBarLow : ""}`} role="img" aria-label={`${s.name} ${s.text}`}>
              <span className={dr.plBarFill} style={{ width: `${clamp(s.now, 0, 100).toFixed(1)}%` }} />
            </span>
            <strong className={dr.plBarValue}>{s.text}</strong>
          </li>
        ))}
      </ul>

      <svg
        ref={chart}
        className={dr.plChart}
        viewBox={`0 0 ${CW} ${CH}`}
        role="slider"
        tabIndex={0}
        aria-label="Time of day in the playback"
        aria-valuemin={0}
        aria-valuemax={lived.ticks.length - 1}
        aria-valuenow={tickIndex}
        aria-valuetext={`${dayName} ${now.hhmm}: ${nowText}`}
        onKeyDown={(e) => {
          if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
            e.preventDefault();
            stepBy(e.key === "ArrowRight" ? 1 : -1);
          }
        }}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          pick(e);
        }}
        onPointerMove={(e) => {
          if (e.buttons || e.pointerType === "touch") pick(e);
        }}
      >
        {lived.spans.map((s) => {
          const x0 = cx(Math.max(show.t0, s.s));
          const x1 = cx(Math.min(show.t1, s.e));
          return x1 > x0 ? <rect key={`${s.block.id}-${s.s}`} className={s.block.fixed ? dr.bandFixed : dr.bandPlanned} x={x0} y={CTOP} width={Math.max(1, x1 - x0 - 0.6)} height={CBOT - CTOP} /> : null;
        })}
        <line className={dr.axis} x1={6} x2={CW - 6} y1={CBOT} y2={CBOT} />
        {hours.map((h) => (
          <text key={h.t} className={dr.axisText} x={cx(h.t)} y={CH - 5} textAnchor="middle">
            {h.label}
          </text>
        ))}
        {series.map((s) => (
          <polyline key={s.id} className={`${dr.line} ${s.cls}`} points={lived.ticks.map((t) => `${cx(ms(t.t)).toFixed(1)},${cy(s.value(t)).toFixed(1)}`).join(" ")} />
        ))}
        {split !== null ? (
          <g>
            <line className={dr.plSplit} x1={cx(split)} x2={cx(split)} y1={CTOP} y2={CBOT} />
            <path className={dr.plSplitMark} d={`M${cx(split) - 4.5} ${CTOP} H${cx(split) + 4.5} L${cx(split)} ${CTOP + 7} Z`} />
          </g>
        ) : null}
        <line className={dr.cursor} x1={cx(frame.t)} x2={cx(frame.t)} y1={CTOP - 3} y2={CBOT + 3} />
        {series.map((s) => (
          <circle key={s.id} className={`${dr.cursorDot} ${s.cls}`} cx={cx(frame.t)} cy={cy(s.now)} r={4} />
        ))}
      </svg>

      <div className={dr.plControls}>
        <button type="button" className={`${ui.btn} ${dr.plIcon}`} onClick={restart} aria-label="Restart from the start of the day">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M5 5v5h5M5.6 10A7 7 0 1 1 5 13.5" />
          </svg>
        </button>
        <button type="button" className={`${ui.btn} ${dr.plIcon}`} onClick={() => stepBy(-1)} aria-label="Step back one block">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M7 5v14M19 6l-9 6 9 6z" />
          </svg>
        </button>
        <button type="button" className={`${ui.btn} ${ui.btnPrimary} ${dr.plPlay}`} onClick={toggle}>
          <svg viewBox="0 0 24 24" aria-hidden="true">
            {playing ? <path d="M8 5v14M16 5v14" /> : <path className={dr.plSolid} d="M8 5l11 7-11 7z" />}
          </svg>
          {playing ? "Pause" : ended ? "Replay" : frame.t > show.t0 ? "Resume" : "Play"}
        </button>
        <button type="button" className={`${ui.btn} ${dr.plIcon}`} onClick={() => stepBy(1)} aria-label="Step forward one block">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M17 5v14M5 6l9 6-9 6z" />
          </svg>
        </button>
      </div>
      {reduced ? null : (
        <div className={ui.segmented} role="group" aria-label="Playback speed, in simulated minutes per second">
          {SPEEDS.map((s) => (
            <button key={s} type="button" className={`${ui.seg} ${speed === s ? ui.segOn : ""}`} aria-pressed={speed === s} onClick={() => setSpeed(s)}>
              {s} min/s
            </button>
          ))}
        </div>
      )}
      <p className={`${ui.muted} ${ui.small}`}>
        {reduced
          ? "Reduced motion is on: the day steps block by block instead of animating."
          : "Walks are slowed so you can watch them, and the night runs three times faster. Drag the strip above, or the chart below, to move through the day."}
      </p>

      <ol className={dr.plFeed} aria-label="What has happened so far">
        {recent.length === 0 ? <li className={ui.muted}>Nothing has happened yet.</li> : null}
        {recent.map((item, i) => (
          <li key={`${item.t}-${item.text}`} className={`${item.tone === "bad" ? dr.plFeedBad : item.tone === "warn" ? dr.plFeedWarn : item.tone === "ok" ? dr.plFeedOk : ""} ${i === 0 ? dr.plFeedNew : ""}`}>
            <span className={dr.when}>{clockAt(show, item.t).hhmm}</span>
            <span className={ui.breakWords}>{item.text}</span>
          </li>
        ))}
      </ol>
        </div>
      </div>
    </div>
  );
}

function Flag({ x, y, text, hot }: { x: number; y: number; text: string; hot: boolean }) {
  return (
    <g transform={`translate(${x.toFixed(1)} ${y.toFixed(1)})`} className={hot ? dr.plFlagHot : dr.plFlag}>
      <line x1={0} y1={-4} x2={0} y2={-30} />
      <path d="M0 -31 H39 L35 -25 L39 -19 H0 Z" />
      <text x={3.5} y={-21.5}>
        {text}
      </text>
    </g>
  );
}

/** The rooms of the pack you are inside: listed, since rooms have no coordinates to draw. */
function Rooms({ packs, places, here }: { packs: PlacePack[]; places: PlaceIndex; here: string }) {
  const pack = packs.find((p) => p.name === here.split(".", 1)[0]);
  if (!pack) return null;
  return (
    <ul className={dr.plRooms} aria-label={`Rooms inside ${parentPlace(places, here)?.name ?? pack.name}`}>
      {Object.entries(pack.places).map(([id, p]) => (
        <li key={id} className={`${pack.name}.${id}` === here ? dr.plRoomOn : undefined}>
          {p.name}
        </li>
      ))}
    </ul>
  );
}

function Caption({
  show,
  hold,
  holdF,
  moment,
  places,
  current,
  walkNumber,
  walkCount,
  check,
  disrupted,
}: {
  show: Show;
  hold: Hold | undefined;
  holdF: number;
  moment: Moment;
  places: PlaceIndex;
  current: Walk | undefined;
  walkNumber: number;
  walkCount: number;
  check: ReturnType<typeof routeCheck>;
  disrupted: boolean;
}) {
  if (hold?.kind === "replan") {
    const n = show.changes.length;
    const at = n > 0 && holdF > 0.3 ? Math.min(n - 1, Math.floor(((holdF - 0.3) / 0.7) * n)) : -1;
    const change = at >= 0 ? show.changes[at] : undefined;
    return (
      <>
        <p>
          <span className={`${ui.tag} ${dr.tagDanger}`}>Plans change</span>
        </p>
        {!change ? (
          <ul className={dr.bullets}>
            {show.disruptions.map((d, i) => (
              <li key={i} className={ui.breakWords}>
                {d.text || d.kind.replace("_", " ")}
              </li>
            ))}
          </ul>
        ) : (
          <p className={ui.breakWords}>
            <strong>
              Re-planning, change {at + 1} of {n} ({change.kind.replace("_", " ")}):
            </strong>{" "}
            {change.move !== null ? `Move ${change.move + 1}: ` : ""}
            {change.text}
          </p>
        )}
        {n === 0 && holdF > 0.3 ? <p className={ui.muted}>The box checked every move and had nothing to change.</p> : null}
      </>
    );
  }
  if (hold?.kind === "reject" && hold.rejected) {
    const g = hold.ghost;
    return (
      <>
        <p>
          <span className={`${ui.tag} ${dr.tagDanger}`}>Move {hold.rejected.index + 1} rejected</span> <strong>{g ? g.option.title : hold.rejected.move.kind}</strong>
        </p>
        <p className={ui.breakWords}>{hold.rejected.reason}</p>
        {g ? (
          <p className={`${ui.muted} ${ui.small} ${ui.breakWords}`}>
            The box tried {g.names.join(" → ")}: {minutes(g.option.walk_minutes)} there, {minutes(g.option.back_minutes)} on to the next fixed block.
          </p>
        ) : null}
      </>
    );
  }
  if (current) {
    const choosing = hold?.kind === "route";
    return (
      <>
        <p>
          <span className={`${ui.tag} ${ui.tagAccent}`}>
            {choosing ? "Choosing route" : "Route"} {walkNumber} of {walkCount}
          </span>{" "}
          <strong>{current.route.purpose}</strong>
        </p>
        <p className={ui.breakWords}>{current.names.join(" → ")}</p>
        <p className={dr.plFacts}>
          {minutes(current.route.minutes)} · leave by {clock(current.route.depart)} · arrive {clock(current.route.arrive)}
        </p>
        {check ? (
          <p className={`${ui.breakWords} ${check.ok ? dr.plOk : dr.plBad}`}>
            <span aria-hidden="true">{check.ok ? "✓ " : "✗ "}</span>
            {check.text}
          </p>
        ) : null}
      </>
    );
  }
  const b = moment.span?.block;
  if (b) {
    return (
      <>
        <p>
          <span className={ui.tag}>{b.fixed ? "Fixed block" : `Move ${(b.move ?? 0) + 1}`}</span> <strong className={ui.breakWords}>{b.title}</strong>
        </p>
        <p className={dr.plFacts}>
          {clockSpan(b.start, b.end)} · {minutes(b.minutes)} · {b.place_name || placeName(places, b.place)}
        </p>
        {b.with_friend ? <p className={ui.muted}>{b.with_friend} is here with you.</p> : null}
        {disrupted && show.changes.some((c) => c.move !== null && c.move === b.move) ? (
          <p className={`${ui.small} ${ui.breakWords} ${dr.plWarn}`}>
            Changed by hard mode:{" "}
            {show.changes
              .filter((c) => c.move === b.move)
              .map((c) => c.text)
              .join("; ")}
          </p>
        ) : null}
      </>
    );
  }
  return (
    <>
      <p>
        <span className={ui.tag}>Free time</span> <strong>{placeName(places, moment.place)}</strong>
      </p>
      <p className={ui.muted}>Nothing is planned for this stretch.</p>
    </>
  );
}
