"use client";

import Link from "next/link";
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useCampusMap } from "@/components/dayrun/CampusLayers";
import { DateStrip } from "@/components/dayrun/DateStrip";
import { DayStrip } from "@/components/dayrun/DayStrip";
import { PlanAgenda, rowKey } from "@/components/dayrun/PlanAgenda";
import { DueSoon, NextDays, PlanDetails } from "@/components/dayrun/PlanDetails";
import { PlanMap } from "@/components/dayrun/PlanMap";
import pl from "@/components/dayrun/plan.module.css";
import { TellAtlas } from "@/components/dayrun/TellAtlas";
import { Empty, ErrorBox, Loading, WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type CommitResult, type Explained, type HardResult, type Move, type SimResult } from "@/lib/api";
import { useAtlas } from "@/lib/atlas";
import { buildShow, buildWorld, paceAt, showMoment, type Moment } from "@/lib/dayplay";
import { indexPlaces, ms, placeName } from "@/lib/dayrun";
import { DEMO_BAD_PLAN, DEMO_DISRUPTIONS, DEMO_GOOD_PLAN, DEMO_SCENARIO } from "@/lib/demoDay";
import { blockTitle, clockAt, dateWords, hoursShort, placeShort, stars, weekdayLong } from "@/lib/plan";
import { useProfileState } from "@/lib/profile";
import { useFetch } from "@/lib/useFetch";
import { setScreenIntent } from "@/lib/voice";

/**
 * Today's Plan: try your day before you live it.
 *
 * One day is on screen: yours. The date strip opens the days around it: a past day against what happened, today's
 * plan (the one Atlas made this morning, yours to change), or a draft for a day ahead. Change the plan three ways:
 * tap a block (length, earlier, later, remove) or "+ Add" in a gap; type or say a change ("move lunch to one"); or
 * tap "Fix it" on something that needs fixing. Every change is run on the box at once.
 *
 * Presenter links (all optional; none of these is a control on the screen):
 *   /dayrun/?day=2026-10-05   open that day (a day of the strip)
 *   /dayrun/?plan=good        the scripted day with its plan that works   (?plan=bad for the one that needs fixing)
 *   /dayrun/?hard=1           throw problems at the plan: with no ?day= this is the scripted day, its plan that
 *                             works and its two scripted disruptions; &seed=7 picks seeded ones (&count=2 for two)
 *   /dayrun/?play=1           start watching the day (?play=0 opens it paused); &t=11:15 opens on that minute;
 *                             &motion=reduce steps block by block, as with the system's reduced motion
 *   /dayrun/?t=14:05          put the marker at that time of day
 *   /dayrun/?more=1           open Details
 *   /dayrun/?plan=day and ?day=mine are accepted and mean what the screen now does by default: your own day.
 */

const SIM_DELAY_MS = 220;
const SPEEDS = [18, 36, 72]; // simulated minutes per second: 1×, 2×, 4×
const MIN = 60_000;
const ICON = { fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round", strokeLinejoin: "round" } as const;

interface Query {
  plan: "good" | "bad" | null;
  hard: boolean;
  seed: number | null;
  count: number;
  t: string | null;
  play: "auto" | "paused" | null;
  calm: boolean;
  day: string | null;
  more: boolean;
}

const asApiError = (err: unknown) => (err instanceof ApiError ? err : new ApiError(0, err instanceof Error ? err.message : String(err)));
const Agenda = memo(PlanAgenda);

export default function TodaysPlanPage() {
  const { approvals } = useAtlas();
  const { profile, ready: profileReady } = useProfileState();
  const activity = profile.words.Activity;

  // ---------- presenter links: read once, after the first render (the page is a static export) ----------
  const [link, setLink] = useState<Query | null>(null);
  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const plan = q.get("plan");
    const seed = Number(q.get("seed"));
    const count = Number(q.get("count"));
    const day = q.get("day");
    setLink({
      plan: plan === "good" || plan === "bad" ? plan : null,
      hard: q.has("hard") && q.get("hard") !== "0",
      seed: q.has("seed") && Number.isInteger(seed) ? seed : null,
      count: Number.isInteger(count) && count > 0 ? count : 1,
      t: q.get("t"),
      play: q.has("play") ? (q.get("play") === "0" ? "paused" : "auto") : null,
      calm: q.get("motion") === "reduce",
      day: day && /^\d{4}-\d{2}-\d{2}$/.test(day) ? day : day === "mine" || plan === "day" ? "mine" : null,
      more: q.has("more"),
    });
  }, []);
  // The scripted day is reached only by a presenter link, never from a control on this screen.
  const scripted = Boolean(link && (link.plan || (link.hard && link.day === null)));
  const scenario = scripted ? (profile.scenario ?? DEMO_SCENARIO) : undefined;

  // ---------- which day ----------
  const strip = useFetch((s) => api.planDays(s));
  const [picked, setPicked] = useState<string | null>(null);
  const today = strip.data?.today;
  const day = picked ?? (link?.day && link.day !== "mine" ? link.day : null) ?? today ?? null;
  const info = useFetch((s) => (day && !scripted ? api.planDay(day, s) : Promise.resolve(null)), `${day ?? ""}|${scripted}`);
  const when = scripted ? "today" : (info.data?.when ?? "today");
  const past = when === "past";
  const start = scripted ? undefined : (info.data?.start ?? undefined);
  const ready = link !== null && (scripted ? profileReady : Boolean(day) && (info.data !== undefined || info.error !== undefined));

  const board = useFetch((s) => (ready ? api.planBoard({ start, scenario }, s) : Promise.resolve(null)), `${ready}|${start ?? ""}|${scenario ?? ""}`);
  const script = useFetch((s) => (scenario ? api.dayrunScenario(scenario, s) : Promise.resolve(null)), scenario ?? "");
  const packs = useFetch((s) => api.places(s));
  const places = useMemo(() => indexPlaces(packs.data ?? []), [packs.data]);
  const boardPack = board.data?.pack;
  const campus = useCampusMap(boardPack ? packs.data : undefined, useMemo(() => new Set(boardPack ? [boardPack] : []), [boardPack]));
  const world = useMemo(() => buildWorld(places, campus.map), [places, campus.map]);

  // ---------- the plan: the moves, and the box's run of them ----------
  const [moves, setMovesState] = useState<Move[]>([]);
  const drafts = useRef(new Map<string, Move[]>());
  const [workLeft, setWorkLeft] = useState<Record<string, number>>({});
  const [sim, setSim] = useState<{ busy: boolean; result?: SimResult; explained?: Explained; error?: ApiError; key?: string }>({ busy: false });
  const [hard, setHard] = useState<{ busy: boolean; result?: HardResult; error?: ApiError } | null>(null);
  const [commit, setCommit] = useState<{ busy: boolean; done?: CommitResult; error?: ApiError }>({ busy: false });
  const [flash, setFlash] = useState<{ index: number; n: number } | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [hover, setHover] = useState<string | null>(null);
  const [t, setT] = useState<number | null>(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [phase, setPhase] = useState(0);
  const pending = useRef<{ hard: boolean; play: Query["play"]; t: string | null }>({ hard: false, play: null, t: null });
  const linkUsed = useRef(false);
  const dayKey = scripted ? `scripted:${link?.plan ?? "good"}` : (day ?? "");

  // a day's moves: the draft kept for it this visit, else what the box has (the Day loop's plan, a past day's, none)
  useEffect(() => {
    if (!ready) return;
    const kept = drafts.current.get(dayKey);
    if (kept) setMovesState(kept);
    else if (scripted) {
      if (script.loading) return;
      const name = link?.plan ?? "good";
      const student = (profile.scenario ?? DEMO_SCENARIO) === DEMO_SCENARIO;
      setMovesState(script.data?.plans[name] ?? (student ? (name === "bad" ? DEMO_BAD_PLAN : DEMO_GOOD_PLAN) : []));
    } else setMovesState(info.data?.moves ?? []);
    setHard(null);
    setCommit({ busy: false });
    setSelected(null);
    setT(null);
    setPlaying(false);
    setWorkLeft({});
    // the link's ?hard=, ?play= and ?t= are for the day the link opened, once
    pending.current = linkUsed.current ? { hard: false, play: null, t: null } : { hard: Boolean(link?.hard), play: link?.play ?? null, t: link?.t ?? null };
    linkUsed.current = true;
    // a new day, or its plan arriving: not when the link object itself is rebuilt
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dayKey, ready, info.data, script.data, script.loading]);

  const applyMoves = useCallback(
    (next: Move[], changed: number | null) => {
      drafts.current.set(dayKey, next);
      setMovesState(next);
      setHard(null);
      setCommit({ busy: false });
      setPlaying(false);
      if (changed !== null) setFlash((f) => ({ index: changed, n: (f?.n ?? 0) + 1 }));
    },
    [dayKey],
  );

  const boardStart = board.data?.start;
  const initial = useMemo(() => (Object.keys(workLeft).length > 0 ? { work_left: workLeft } : undefined), [workLeft]);
  const simKey = `${dayKey}|${boardStart ?? ""}`;
  useEffect(() => {
    if (!boardStart) return;
    const ctl = new AbortController();
    setSim((s) => (s.key === simKey ? { ...s, busy: true } : { busy: true, key: simKey }));
    const body = { moves, start: boardStart, scenario, initial };
    const timer = window.setTimeout(() => {
      Promise.all([api.dayrunSimulate(body, ctl.signal), api.planExplain(body, ctl.signal).catch(() => undefined)])
        .then(([result, explained]) => setSim({ busy: false, result, explained, key: simKey }))
        .catch((err: unknown) => {
          if (!ctl.signal.aborted) setSim((s) => ({ ...s, busy: false, error: asApiError(err) }));
        });
    }, SIM_DELAY_MS);
    return () => {
      window.clearTimeout(timer);
      ctl.abort();
    };
  }, [moves, boardStart, scenario, initial, simKey]);

  const runHard = useCallback(
    (seed: number | null, count: number) => {
      if (!boardStart) return;
      setHard({ busy: true });
      const base = { moves, start: boardStart, scenario, initial };
      const disruptions = script.data?.disruptions ?? ((profile.scenario ?? DEMO_SCENARIO) === DEMO_SCENARIO ? DEMO_DISRUPTIONS : []);
      api
        .dayrunHard(scripted && seed === null && disruptions.length > 0 ? { ...base, disruptions } : { ...base, seed: seed ?? 7, count: seed === null ? 2 : count })
        .then((result) => setHard({ busy: false, result }))
        .catch((err: unknown) => setHard({ busy: false, error: asApiError(err) }));
    },
    [moves, boardStart, scenario, initial, scripted, script.data, profile.scenario],
  );

  const b = board.data ?? undefined;
  const result = sim.key === simKey ? sim.result : undefined;
  const explained = sim.key === simKey ? sim.explained : undefined;
  const shown = hard?.result ? hard.result.after : result;
  const show = useMemo(() => (result && b ? buildShow({ result, hard: hard?.result ?? null, world, places, board: b }) : null), [result, hard?.result, world, places, b]);

  // presenter links that wait for the first run of the linked plan: ?hard=1, ?play=, ?t=
  useEffect(() => {
    if (!result || sim.busy || !show) return;
    const p = pending.current;
    if (p.hard) {
      p.hard = false;
      runHard(link?.seed ?? null, link?.count ?? 1);
      return;
    }
    if (hard?.busy) return;
    if (p.t) {
      const m = /^(\d{1,2}):(\d{2})$/.exec(p.t);
      p.t = null;
      const from = /T(\d{2}):(\d{2})/.exec(show.main.result.start);
      if (m && from) {
        const add = (((Number(m[1]) * 60 + Number(m[2]) - (Number(from[1]) * 60 + Number(from[2]))) % 1440) + 1440) % 1440;
        setT(Math.min(show.t1, show.t0 + add * MIN));
      }
    }
    if (p.play) {
      const auto = p.play === "auto";
      p.play = null;
      setT((x) => x ?? show.t0);
      setPlaying(auto);
    }
  }, [result, sim.busy, show, hard?.busy, runHard, link?.seed, link?.count]);

  // ---------- watching the day ----------
  const calm = useRef(false);
  useEffect(() => {
    calm.current = Boolean(link?.calm) || window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  }, [link?.calm]);
  useEffect(() => {
    if (!playing || !show) return;
    if (calm.current) {
      // reduced motion: no walking, one stop of the day at a time
      const id = window.setInterval(() => {
        setT((x) => {
          const next = show.stops.find((s) => s > (x ?? show.t0) + 1);
          if (next === undefined) {
            setPlaying(false);
            return show.t1;
          }
          return next;
        });
      }, 1200);
      return () => window.clearInterval(id);
    }
    let frame = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = Math.min(0.1, (now - last) / 1000);
      last = now;
      setPhase((p) => p + dt);
      setT((x) => {
        const at = x ?? show.t0;
        const next = at + paceAt(show, at, SPEEDS[speed] ?? 36) * dt * MIN;
        if (next >= show.t1) {
          setPlaying(false);
          return show.t1;
        }
        return next;
      });
      frame = window.requestAnimationFrame(tick);
    };
    frame = window.requestAnimationFrame(tick);
    return () => window.cancelAnimationFrame(frame);
  }, [playing, show, speed]);

  const moment: Moment | null = useMemo(() => (show && t !== null ? showMoment(show, world, t) : null), [show, world, t]);
  const after = Boolean(show && show.splitAt !== null && t !== null && t >= show.splitAt) || (Boolean(hard?.result) && t === null);
  const onScreen = show && show.before && !after ? show.before.result : shown;

  // ---------- what the selected (or hovered) row lights on the map ----------
  const focusKey = hover ?? selected;
  const focus = useMemo(() => {
    if (!onScreen || !focusKey) return { route: null as number | null, place: null as string | null };
    const block = onScreen.blocks.find((x) => rowKey(x) === focusKey);
    if (!block) return { route: null, place: null };
    const walk = onScreen.routes.find((r) => (block.move !== null ? r.move === block.move : r.move === null && r.end === block.place && Math.abs(ms(r.arrive) - ms(block.start)) <= 30 * MIN));
    return { route: walk ? walk.index : null, place: block.place };
  }, [onScreen, focusKey]);
  const pickPlace = useCallback(
    (ref: string) => {
      if (!onScreen) return;
      const near = (x: { place: string }) => x.place === ref || placeName(places, x.place) === placeName(places, ref) || world.resolve(x.place) === world.at.get(ref);
      const order = [...onScreen.blocks].sort((x, y) => ms(x.start) - ms(y.start));
      const at = order.findIndex((x) => rowKey(x) === selected);
      const hit = order.slice(at + 1).find(near) ?? order.find(near);
      if (hit) {
        setSelected(rowKey(hit));
        document.querySelector(`[data-row="${CSS.escape(rowKey(hit))}"]`)?.scrollIntoView({ block: "nearest", behavior: calm.current ? "auto" : "smooth" });
      }
    },
    [onScreen, places, world, selected],
  );

  // ---------- say a change ----------
  const ask = useCallback((text: string) => api.planEdit({ text, moves, start: boardStart, scenario, initial }), [moves, boardStart, scenario, initial]);
  useEffect(() => {
    if (past || !boardStart) return;
    return setScreenIntent("/dayrun/", async (text) => {
      const out = await ask(text);
      if (out.ok) applyMoves(out.moves, out.changed);
      return out.warning ? `${out.understood} ${out.warning}.` : out.understood;
    });
  }, [ask, applyMoves, past, boardStart]);

  const doCommit = () => {
    if (!boardStart) return;
    setCommit({ busy: true });
    api
      .dayrunCommit({ moves, start: boardStart, scenario, initial })
      .then((done) => {
        setCommit({ busy: false, done });
        approvals.reload();
      })
      .catch((err: unknown) => setCommit({ busy: false, error: asApiError(err) }));
  };

  // ---------- words ----------
  const fixes = explained?.fixes ?? [];
  const works = Boolean(result?.outcome.won) && fixes.length === 0;
  const stripDay = strip.data?.days.find((d) => d.day === day);
  const label = scripted ? "A scripted day" : !day ? "" : day === today ? "Today" : dateWords(day);
  let title = explained?.headline ?? (result ? (result.outcome.won ? "This plan works" : "This plan needs fixing") : "");
  let lead = "";
  if (past && info.data) {
    title = `${info.data.label.charAt(0).toUpperCase()}${info.data.label.slice(1)}`;
    lead = info.data.summary;
  } else if (hard?.result) {
    const n = hard.result.changes.length;
    title = hard.result.after.outcome.won ? "It still works" : "It needs fixing after that";
    lead = n === 0 ? "Nothing in your plan had to change." : `Atlas made ${n} ${n === 1 ? "change" : "changes"} to keep the day standing.`;
  } else if (result && moves.length === 0 && !works) {
    title = "Nothing planned yet";
    lead = when === "future" && stripDay?.summary ? stripDay.summary : "Add what you want to do, or start with one of these.";
  } else if (result && works) {
    const due = explained ? Object.values(explained.deadlines) : [];
    const safe = due.length === 0 ? null : due.length === 1 ? `${due[0]!.title} ${due[0]!.status === "done" ? "done" : "on track"}` : "every deadline on track";
    lead = [`${hoursShort(result.totals.sleep_hours)} sleep`, `${result.totals.meals} meals`, safe].filter(Boolean).join(" · ");
  }
  const starCount = past ? (info.data?.review?.stars ?? 0) : hard?.result ? hard.result.after.stars : works ? (result?.stars ?? 0) : 0;
  // the danger ring is for a plan that needs fixing; an empty draft is only empty
  const needsFixing = !past && !hard?.result && Boolean(result) && !works;
  const alarm = needsFixing && moves.length > 0;

  const caption = (() => {
    if (!onScreen || !b) return null;
    if (moment && t !== null) {
      const clock = clockAt(t, onScreen.start);
      if (show && show.splitAt !== null && t >= show.splitAt && t < show.splitAt + 45 * MIN && show.disruptions[0]) return { main: show.disruptions.map((d) => d.text).filter(Boolean).join("; "), side: clock };
      if (moment.walk) {
        const to = placeShort(placeName(places, moment.walk.route.end));
        return { main: `Walking to ${to}`, side: clock };
      }
      if (moment.span) return { main: `${blockTitle(moment.span.block, activity)}${moment.span.block.place_name ? ` at ${placeShort(moment.span.block.place_name)}` : ""}`, side: clock };
      return { main: `Free time at ${placeShort(placeName(places, moment.place))}`, side: clock };
    }
    if (focusKey) {
      const block = onScreen.blocks.find((x) => rowKey(x) === focusKey);
      const walk = focus.route !== null ? onScreen.routes.find((r) => r.index === focus.route) : undefined;
      if (block) return { main: walk ? `${Math.max(1, Math.round(walk.minutes))} min walk · leave by ${clockAt(ms(walk.depart), onScreen.start)}` : `${blockTitle(block, activity)}: no walk, you are already there`, side: "" };
    }
    return { main: "", side: "Tap a block or a building to see its walk." };
  })();

  // what went wrong, in a few words each: "Judging and demos → CB1, 11:45 AM", "Student Union closed"
  const pills =
    hard?.result && after && b
      ? hard.result.disruptions.map((d) => {
          if (d.kind === "place_closed") return `${placeShort(placeName(places, d.place))} closed`;
          if (d.kind === "friend_cancelled") return `${d.friend ?? "A friend"} can't come`;
          const block = b.blocks.find((x) => x.id === d.block);
          const name = (block?.title ?? "A block").replace(/^[^:]{1,14}: /, "");
          const to = d.to_place ? ` → ${places.get(d.to_place)?.aliases[0] ?? placeShort(placeName(places, d.to_place))}` : "";
          const at = block && d.shift_minutes ? `, ${clockAt(ms(block.start) + d.shift_minutes * MIN, block.start)}` : "";
          return `${name.charAt(0).toUpperCase()}${name.slice(1)}${to}${at}`;
        })
      : [];
  const closed = hard?.result && after ? hard.result.disruptions.filter((d) => d.kind === "place_closed" && d.place).map((d) => d.place as string) : [];
  const movedMoves = useMemo(() => new Set(hard?.result ? hard.result.changes.filter((c) => c.move !== null && c.kind !== "rejected").map((c) => c.move as number) : []), [hard?.result]);
  const noHover = useCallback((key: string | null) => setHover(key), []);
  const select = useCallback((key: string | null) => {
    setSelected(key);
    setPlaying(false);
    setT(null);
  }, []);
  const setWork = (id: string, minutes: number) => setWorkLeft((w) => ({ ...w, [id]: minutes }));

  const header = (
    <div className={ui.pageHead}>
      <h1>Today&apos;s Plan</h1>
      <p className={ui.muted}>Try your day before you live it.</p>
    </div>
  );

  if (board.error || (info.error && !scripted)) {
    const error = (board.error ?? info.error) as ApiError;
    return (
      <>
        {header}
        {strip.data && today && day ? <DateStrip days={strip.data.days} today={today} selected={day} onSelect={setPicked} /> : null}
        {error.notConfigured ? <Empty title="Today's Plan is not set up on this box">It needs the map of your places before it can plan a day.</Empty> : <ErrorBox what="this day" error={error} onRetry={() => (info.reload(), board.reload())} />}
      </>
    );
  }

  return (
    <>
      {header}
      {scripted ? (
        <p className={pl.dateNote}>
          A scripted day, for presenting. <Link href="/dayrun/">Back to my day</Link>
        </p>
      ) : strip.data && today && day ? (
        <DateStrip days={strip.data.days} today={today} selected={day} onSelect={setPicked} />
      ) : null}

      {!b || !shown || !onScreen ? (
        sim.error ? (
          <ErrorBox what="the plan" error={sim.error} />
        ) : (
          <Loading what="the plan" />
        )
      ) : (
        <div className={pl.split}>
          <div className={`${pl.pane} ${pl.left}`}>
            <section className={`${pl.hero} ${alarm ? pl.heroFix : ""}`} aria-live="polite" aria-label="How this plan does">
              <div className={pl.heroTop}>
                <span className={pl.heroLabel}>{label}</span>
                {starCount > 0 ? (
                  <span className={pl.stars} aria-label={`${starCount} of 3 stars`}>
                    {stars(starCount)}
                  </span>
                ) : null}
              </div>
              <h2 className={pl.heroTitle}>{title}</h2>
              {lead ? <p className={pl.lead}>{lead}</p> : null}

              {needsFixing && fixes.length > 0 ? (
                <div className={pl.fixes}>
                  {fixes.slice(0, 3).map((f) => (
                    <div key={f.id} className={pl.fix}>
                      <p className={pl.fixText}>
                        <b>{f.problem}</b>
                        {f.label ? <span>{f.label}</span> : null}
                      </p>
                      {f.moves && f.label ? (
                        <button type="button" className={pl.fixBtn} onClick={() => applyMoves(f.moves as Move[], f.changed)}>
                          Fix it
                        </button>
                      ) : null}
                    </div>
                  ))}
                  {fixes.length > 3 ? <p className={pl.legendLine}>And {fixes.length - 3} more after these.</p> : null}
                </div>
              ) : null}

              {campus.map ? (
                <PlanMap
                  map={campus.map}
                  places={places}
                  world={world}
                  result={onScreen}
                  focusRoute={focus.route}
                  focusPlace={focus.place}
                  moment={moment}
                  playing={playing}
                  phase={phase}
                  still={!playing || calm.current}
                  closed={closed}
                  pills={pills}
                  onPlace={pickPlace}
                />
              ) : (
                <div className={pl.mapWell}>
                  <p className={pl.mapEmpty}>{campus.pack && !campus.failed ? "Loading the map…" : "These places are not on a drawn map, so the plan is listed below."}</p>
                </div>
              )}
              {caption ? (
                <p className={pl.caption}>
                  {caption.main ? (
                    <span>
                      <b>{caption.main}</b>
                    </span>
                  ) : null}
                  {caption.side ? <span>{caption.side}</span> : null}
                </p>
              ) : null}

              <DayStrip
                result={onScreen}
                t={t}
                selected={selected}
                onSeek={(x) => {
                  setPlaying(false);
                  setT(x);
                }}
              />

              <div className={pl.heroRow}>
                <button
                  type="button"
                  className={`${pl.pill} ${pl.pillMain}`}
                  disabled={!show || Boolean(hard?.busy)}
                  onClick={() => {
                    if (!show) return;
                    if (!playing && (t === null || t >= show.t1 - 1)) setT(show.t0);
                    setPlaying(!playing);
                  }}
                >
                  <svg viewBox="0 0 24 24" {...ICON} aria-hidden="true">
                    {playing ? <path d="M8.5 6v12M15.5 6v12" /> : <path d="M8 5.5v13l10.5-6.5z" />}
                  </svg>
                  {playing ? "Pause" : past ? "Watch this day" : "Watch the day"}
                </button>
                {playing || t !== null ? (
                  <button type="button" className={`${pl.pill} ${pl.pillQuiet}`} onClick={() => setSpeed((s) => (s + 1) % SPEEDS.length)} aria-label={`Speed ${[1, 2, 4][speed]} times; tap to change`}>
                    {[1, 2, 4][speed]}× speed
                  </button>
                ) : null}
                {!past ? (
                  <button
                    type="button"
                    className={`${pl.pill} ${pl.pillQuiet} ${hard?.result ? pl.pillOn : ""}`}
                    disabled={Boolean(hard?.busy) || moves.length === 0 || !result}
                    aria-pressed={Boolean(hard?.result)}
                    onClick={() => (hard?.result ? setHard(null) : runHard(null, 2))}
                  >
                    {hard?.busy ? "Shaking the day…" : hard?.result ? "Back to the calm day" : "Throw problems at it"}
                  </button>
                ) : null}
              </div>
              {hard?.error ? <WriteError error={hard.error} /> : null}

              {past ? null : <TellAtlas ask={ask} onApply={applyMoves} />}
            </section>
          </div>

          <div className={pl.pane}>
            <section className={pl.pane} aria-label="Your plan">
              <h2 className={pl.paneTitle}>{past ? `${weekdayLong(day ?? "")}'s plan` : "Your plan"}</h2>
              {b.slots.length === 0 && shown.blocks.length === 0 ? (
                <Empty title="Nothing to plan here">Your calendar fills this day, or the box has nothing for it yet.</Empty>
              ) : (
                <Agenda
                  board={b}
                  result={onScreen}
                  moves={moves}
                  activity={activity}
                  past={past ? info.data?.blocks : undefined}
                  readOnly={past}
                  selected={selected}
                  onSelect={select}
                  onHover={noHover}
                  flash={flash}
                  moved={movedMoves}
                  fixes={fixes}
                  explained={explained}
                  onChange={applyMoves}
                />
              )}
            </section>

            <DueSoon board={b} explained={explained} workLeft={workLeft} onWorkLeft={setWork} readOnly={past} />

            {past ? null : (
              <section className={pl.action} aria-label="Keep this plan">
                {commit.done ? (
                  <>
                    <Link href="/approvals/" className={`${ui.btn} ${ui.btnPrimary} ${pl.primary}`}>
                      Sent for approval: open it
                    </Link>
                    <p className={pl.hint}>
                      Nothing is on your calendar until you approve. <Link href="/now/">Back to Calendar</Link>
                    </p>
                  </>
                ) : (
                  <>
                    <button type="button" className={`${ui.btn} ${ui.btnPrimary} ${pl.primary}`} disabled={!works || commit.busy || sim.busy} onClick={doCommit}>
                      {commit.busy ? "Sending…" : "Keep this plan"}
                    </button>
                    <p className={pl.hint}>{works ? "Atlas asks before anything goes on your calendar." : "Fix what is left and you can keep it."}</p>
                  </>
                )}
                {commit.error ? <CommitError error={commit.error} /> : null}
              </section>
            )}

            <PlanDetails result={onScreen} explained={hard?.result ? undefined : explained} t={t} tomorrow={past ? info.data?.review?.change_tomorrow : null} open={link?.more} />
            {!scripted && !past && strip.data && day ? <NextDays days={strip.data.days} selected={day} onSelect={setPicked} /> : null}
          </div>
        </div>
      )}
    </>
  );
}

/** 409: this box cannot write to a calendar. 422: the plan stopped working between the last run and the tap. */
function CommitError({ error }: { error: ApiError }) {
  if (error.needsToken) return <WriteError error={error} />;
  const text =
    error.status === 409
      ? "This box has no calendar it can write to yet, so the plan stays here."
      : error.status === 422
        ? "The plan no longer works as it is. Fix what is left and try again."
        : `Couldn't send it: ${error.message}`;
  return (
    <p className={`${ui.state} ${ui.stateError}`} role="alert">
      {text}
    </p>
  );
}
