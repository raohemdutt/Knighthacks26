"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useCampusMap } from "@/components/dayrun/CampusLayers";
import { CampusMap } from "@/components/dayrun/CampusMap";
import { DayPlayer } from "@/components/dayrun/DayPlayer";
import dr from "@/components/dayrun/dayrun.module.css";
import { Meters } from "@/components/dayrun/Meters";
import { More } from "@/components/dayrun/More";
import { Planner } from "@/components/dayrun/Planner";
import { RouteSheet } from "@/components/dayrun/RouteSheet";
import { Empty, ErrorBox, Loading, WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type Board, type CommitResult, type HardResult, type Move, type SimResult } from "@/lib/api";
import { useAtlas } from "@/lib/atlas";
import { clock, clockSpan, dateLabel, dateOf, indexPlaces, minutes, ms, placeName, riskPercent, stars } from "@/lib/dayrun";
import { DEMO_BAD_PLAN, DEMO_CAMPAIGN, DEMO_DISRUPTIONS, DEMO_GOOD_PLAN, DEMO_SCENARIO } from "@/lib/demoDay";
import { amount, safeUrl } from "@/lib/format";
import { useProfileState, useWords } from "@/lib/profile";
import { useFetch } from "@/lib/useFetch";

/**
 * Day Run: play the next 24 hours before living them.
 *
 * Presenter links (all optional, combine freely):
 *   /dayrun/?plan=good        load the scripted good plan     (?plan=bad for the bad one)
 *   /dayrun/?hard=1           run hard mode on the plan (the good plan if none is named);
 *                             on the demo day this uses the scripted disruptions
 *   /dayrun/?hard=1&seed=7    seeded disruptions instead (&count=2 for two of them)
 *   /dayrun/?t=14:05          put the scrubber at that time of day
 *   /dayrun/?play=1           open Play mode and start it: the day is watched, with a figure walking the
 *                             routes (with ?plan=good|bad and ?hard=1; the good plan if none is named).
 *                             ?play=0 opens it paused; with ?t=11:15 it opens on exactly that minute.
 *                             &motion=reduce steps block by block, as with the system's reduced motion.
 *   /dayrun/?day=mine         start on "My day" (live sources) instead of the demo day
 *   /dayrun/?plan=day         My day with the Day loop's plan for today preloaded (GET /api/day/plan)
 *   /dayrun/?more=1           open the review, replay, campaign and forecast forms
 */

type Mode = "demo" | "mine";
const SIM_DELAY_MS = 250;

interface Sim {
  busy: boolean;
  result?: SimResult;
  error?: ApiError;
}

interface HardRun {
  busy: boolean;
  result?: HardResult;
  error?: ApiError;
  scripted?: boolean;
}

const asApiError = (err: unknown) => (err instanceof ApiError ? err : new ApiError(0, err instanceof Error ? err.message : String(err)));

function line(r: SimResult): string {
  return `${r.outcome.won ? "Win" : "Fail"} ${stars(r.stars)}, ${minutes(r.totals.walk_minutes)} walking, ${r.totals.meals} ${r.totals.meals === 1 ? "meal" : "meals"}, ${amount(r.totals.sleep_hours)} h sleep, ${r.totals.points} points`;
}

export default function DayRunPage() {
  const { approvals } = useAtlas();
  const { profile, ready: profileReady } = useProfileState(); // the box's words, and which scripted day its demo mode plays
  const w = profile.words;
  const scenarioName = profile.scenario ?? DEMO_SCENARIO;
  const [mode, setMode] = useState<Mode>("demo");
  const scenario = mode === "demo" ? scenarioName : undefined;
  const board = useFetch((s) => api.dayrunBoard(scenario, s), `${mode}:${scenario ?? ""}`);
  // The scripted day's own one-tap plans; the student's are embedded (web/lib/demoDay.ts) so the demo works before the box answers.
  const script = useFetch((s) => (scenario ? api.dayrunScenario(scenario, s) : Promise.resolve(null)), scenario ?? "");
  const student = scenarioName === DEMO_SCENARIO;
  const presets = useMemo(
    () => ({
      bad: script.data?.plans.bad ?? (student ? DEMO_BAD_PLAN : []),
      good: script.data?.plans.good ?? (student ? DEMO_GOOD_PLAN : []),
      disruptions: script.data?.disruptions ?? (student ? DEMO_DISRUPTIONS : []),
      campaign: script.data?.campaign ?? (student ? DEMO_CAMPAIGN : []),
    }),
    [script.data, student],
  );
  const [urlPlan, setUrlPlan] = useState<"good" | "bad" | null>(null); // ?plan= from the presenter link, applied once a preset exists
  const packs = useFetch((s) => api.places(s));
  const places = useMemo(() => indexPlaces(packs.data ?? []), [packs.data]);
  // The drawn campus of the board's own pack, if it has one: fetched once, shared by Play mode, the map and the Route sheets.
  const boardPack = board.data?.pack;
  const campus = useCampusMap(boardPack ? packs.data : undefined, useMemo(() => new Set(boardPack ? [boardPack] : []), [boardPack]));

  const [moves, setMovesState] = useState<Move[]>([]);
  const [preset, setPreset] = useState<"good" | "bad" | null>(null);
  const [sim, setSim] = useState<Sim>({ busy: false });
  const [hard, setHard] = useState<HardRun | null>(null);
  const [view, setView] = useState<"after" | "before">("after");
  const [seed, setSeed] = useState(7);
  const [scrub, setScrub] = useState(0);
  const [commit, setCommit] = useState<{ busy: boolean; done?: CommitResult; error?: ApiError }>({ busy: false });
  const [openMore, setOpenMore] = useState(false);
  const auto = useRef<{ hard: boolean; seed: number | null; count: number; t: string | null }>({ hard: false, seed: null, count: 1, t: null });
  // Play mode. `auto` starts the clock as soon as the linked day is in; `at` is the ?t= minute to open on.
  const [play, setPlay] = useState<{ open: boolean; auto: boolean; at: string | null }>({ open: false, auto: false, at: null });
  const [linkHard, setLinkHard] = useState(false); // ?hard=1 was asked for and has not been run yet
  const [seek, setSeek] = useState<{ index: number; n: number } | null>(null);
  const stage = useRef<HTMLDivElement>(null);

  const setMoves = useCallback((next: Move[], from: "good" | "bad" | null = null) => {
    setMovesState(next);
    setPreset(from);
    setHard(null);
    setView("after");
    setCommit({ busy: false });
  }, []);

  // Presenter links: read once, after the first render (the page is a static export).
  useEffect(() => {
    const q = new URLSearchParams(window.location.search);
    const mine = q.get("day") === "mine";
    if (mine) setMode("mine");
    const plan = q.get("plan");
    const wantHard = q.has("hard") && q.get("hard") !== "0";
    const wantPlay = q.has("play");
    if (plan === "day") {
      // the Day loop's plan for today (Now screen "Change it in Day Run"): live sources, moves preloaded
      setMode("mine");
      api.dayPlan().then((p) => setMoves(p.moves)).catch(() => undefined);
    } else if (!mine) {
      if (plan === "bad") setUrlPlan("bad");
      else if (plan === "good" || wantHard || wantPlay) setUrlPlan("good");
    }
    if (wantPlay) setPlay({ open: true, auto: q.get("play") !== "0", at: q.get("t") });
    setLinkHard(wantHard);
    const s = Number(q.get("seed"));
    const c = Number(q.get("count"));
    auto.current = { hard: wantHard, seed: q.has("seed") && Number.isInteger(s) ? s : null, count: Number.isInteger(c) && c > 0 ? c : 1, t: q.get("t") };
    if (auto.current.seed !== null) setSeed(auto.current.seed);
    if (q.has("more")) setOpenMore(true);
  }, [setMoves]);

  useEffect(() => {
    if (urlPlan && profileReady && !script.loading && presets[urlPlan].length > 0) {
      setMoves(presets[urlPlan], urlPlan);
      setUrlPlan(null);
    }
  }, [urlPlan, profileReady, script.loading, presets, setMoves]);

  const start = board.data?.start;

  // Run it: every change to the plan is simulated on the box, a moment after the last tap.
  useEffect(() => {
    if (!start) return;
    const ctl = new AbortController();
    setSim((s) => ({ ...s, busy: true }));
    const timer = window.setTimeout(() => {
      api
        .dayrunSimulate({ moves, start, scenario }, ctl.signal)
        .then((result) => setSim({ busy: false, result }))
        .catch((err: unknown) => {
          if (!ctl.signal.aborted) setSim({ busy: false, error: asApiError(err) });
        });
    }, SIM_DELAY_MS);
    return () => {
      window.clearTimeout(timer);
      ctl.abort();
    };
  }, [moves, start, scenario]);

  const runHard = useCallback(
    (scripted: boolean, withSeed: number, count = 1) => {
      if (!start) return;
      setHard({ busy: true, scripted });
      setView("after");
      api
        .dayrunHard(scripted ? { moves, start, scenario, disruptions: presets.disruptions } : { moves, start, scenario, seed: withSeed, count })
        .then((result) => setHard({ busy: false, result, scripted }))
        .catch((err: unknown) => setHard({ busy: false, error: asApiError(err), scripted }));
    },
    [moves, start, scenario],
  );

  // ?hard=1 and ?t=: applied once, when the first simulation of the linked plan is in.
  useEffect(() => {
    if (!sim.result || sim.busy) return;
    const a = auto.current;
    if (a.hard) {
      a.hard = false;
      setLinkHard(false);
      runHard(mode === "demo" && a.seed === null, a.seed ?? 0, a.count);
    }
    if (a.t) {
      const want = a.t;
      a.t = null;
      // The first tick in that hour at or after the minute asked for (ticks are a quarter of an hour apart).
      const i = sim.result.timeline.findIndex((tick) => clock(tick.t).slice(0, 2) === want.slice(0, 2) && clock(tick.t) >= want);
      if (i >= 0) setScrub(i);
    }
  }, [sim.result, sim.busy, runHard, mode]);

  const switchMode = (next: Mode) => {
    if (next === mode) return;
    setMode(next);
    setMoves([]);
    setSim({ busy: false });
    setScrub(0);
  };

  const doCommit = () => {
    if (!start) return;
    setCommit({ busy: true });
    api
      .dayrunCommit({ moves, start, scenario })
      .then((done) => {
        setCommit({ busy: false, done });
        approvals.reload();
      })
      .catch((err: unknown) => setCommit({ busy: false, error: asApiError(err) }));
  };

  // A scrub made by hand also moves the playback, when it is open.
  const userScrub = useCallback((index: number) => {
    setScrub(index);
    setSeek((s) => ({ index, n: (s?.n ?? 0) + 1 }));
  }, []);

  const scrubTo = (iso: string) => {
    const shown = hard?.result ? (view === "before" ? hard.result.before : hard.result.after) : sim.result;
    if (!shown) return;
    const t = ms(iso);
    const i = shown.timeline.findIndex((tick) => ms(tick.t) >= t);
    userScrub(i < 0 ? shown.timeline.length - 1 : i);
  };

  const watch = () => {
    setPlay({ open: true, auto: true, at: null });
    window.requestAnimationFrame(() => stage.current?.scrollIntoView({ block: "start", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" }));
  };

  const b = board.data;
  const shown = hard?.result ? (view === "before" ? hard.result.before : hard.result.after) : sim.result;
  const disruptions = hard?.result && view === "after" ? hard.result.disruptions : [];
  const changes = hard?.result && view === "after" ? hard.result.changes : [];

  return (
    <>
      <div className={ui.pageHead}>
        <h1>Day Run</h1>
        <p className={ui.muted}>Try out your next 24 hours, then keep the plan that works best.</p>
      </div>

      <div className={ui.segmented} role="group" aria-label="Which day">
        <button type="button" className={`${ui.seg} ${mode === "demo" ? ui.segOn : ""}`} aria-pressed={mode === "demo"} onClick={() => switchMode("demo")}>
          Demo day
        </button>
        <button type="button" className={`${ui.seg} ${mode === "mine" ? ui.segOn : ""}`} aria-pressed={mode === "mine"} onClick={() => switchMode("mine")}>
          My day
        </button>
      </div>

      {board.error ? (
        board.error.notConfigured ? (
          <Empty title="Day Run is not set up on this box">{board.error.message}. It needs the place packs in config/places.</Empty>
        ) : (
          <ErrorBox what="the board" error={board.error} onRetry={board.reload} />
        )
      ) : !b ? (
        <Loading what="the board" />
      ) : (
        <>
          {mode === "demo" ? (
            <div className={ui.section}>
              <p className={`${ui.muted} ${ui.small}`}>{student ? "A scripted Sunday at Knight Hacks." : `A scripted day for the ${profile.name} profile (${scenarioName}).`} Load a plan in one tap, then change it.</p>
              <div className={dr.presets}>
                <button type="button" className={`${ui.btn} ${preset === "bad" ? dr.presetOn : ""}`} aria-pressed={preset === "bad"} disabled={presets.bad.length === 0} onClick={() => setMoves(presets.bad, "bad")}>
                  Bad plan
                </button>
                <button type="button" className={`${ui.btn} ${preset === "good" ? dr.presetOn : ""}`} aria-pressed={preset === "good"} disabled={presets.good.length === 0} onClick={() => setMoves(presets.good, "good")}>
                  Good plan
                </button>
                <button type="button" className={ui.btn} disabled={moves.length === 0} onClick={() => setMoves([])}>
                  Clear
                </button>
              </div>
            </div>
          ) : (
            <p className={`${ui.muted} ${ui.small}`}>{`Your next 24 hours, from your calendar, your ${w.deadline_plural}, Habits and the friends who shared a place.`}</p>
          )}

          <div ref={stage} className={dr.anchor}>
            {!play.open ? (
              <button type="button" className={`${ui.btn} ${ui.btnPrimary} ${dr.plWatch}`} disabled={!sim.result || !packs.data || Boolean(hard?.busy)} onClick={watch}>
                <span aria-hidden="true">▶</span> Watch the day{hard?.result ? " go wrong" : ""}
              </button>
            ) : !sim.result || !packs.data || linkHard || hard?.busy || (campus.pack && !campus.map && !campus.failed) ? (
              <Loading what="the day to play" />
            ) : (
              <DayPlayer
                packs={packs.data}
                places={places}
                map={campus.map}
                board={b}
                result={sim.result}
                hard={hard?.result ?? null}
                autoPlay={play.auto}
                startClock={play.at}
                seek={seek}
                onTick={setScrub}
                onClose={() => setPlay({ open: false, auto: false, at: null })}
              />
            )}
          </div>

          <Verdict sim={sim} shown={shown} moves={moves.length} hard={Boolean(hard?.result)} view={view} />

          {hard?.result ? (
            <div className={ui.card}>
              <h3>Hard mode: what went wrong</h3>
              <ul className={dr.bullets}>
                {hard.result.disruptions.map((d, i) => (
                  <li key={i} className={ui.breakWords}>
                    {d.text || d.kind.replace("_", " ")}
                  </li>
                ))}
              </ul>
              {hard.result.note ? <p className={ui.nudge}>{hard.result.note}</p> : null}
              <dl className={ui.kv}>
                <dt>Before</dt>
                <dd>{line(hard.result.before)}</dd>
                <dt>After</dt>
                <dd>{line(hard.result.after)}</dd>
              </dl>
              <p className={`${ui.muted} ${ui.small}`}>
                {hard.result.changes.length === 0 ? "The box changed nothing." : `The box made ${hard.result.changes.length} ${hard.result.changes.length === 1 ? "change" : "changes"} to keep the plan standing.`}{" "}
                <a href="#hard">See each change</a>
              </p>
            </div>
          ) : (
            <button type="button" className={ui.btn} disabled={hard?.busy || moves.length === 0 || !sim.result} onClick={() => runHard(mode === "demo", seed)}>
              {hard?.busy ? "Disrupting the day…" : "Hard mode: throw problems at this plan"}
            </button>
          )}

          <StartTiles board={b} places={places} />

          <section className={ui.section}>
            <div className={`${ui.row} ${ui.between}`}>
              <h2 className={ui.sectionTitle}>The day, hour by hour</h2>
              {sim.busy ? <span className={`${ui.muted} ${ui.small}`}>Running…</span> : null}
            </div>
            {sim.error && !shown ? (
              <ErrorBox what="the simulation" error={sim.error} />
            ) : !shown ? (
              <Loading what="the simulation" />
            ) : (
              <>
                {sim.error ? <WriteError error={sim.error} /> : null}
                <div className={ui.card}>
                  <Meters result={shown} places={places} scrub={scrub} onScrub={userScrub} />
                </div>
              </>
            )}
          </section>

          {shown ? (
            <section className={ui.section}>
              <h2 className={ui.sectionTitle}>Where the day takes you</h2>
              <div className={ui.card}>
                {packs.error ? (
                  <ErrorBox what="the places" error={packs.error} onRetry={packs.reload} />
                ) : !packs.data ? (
                  <Loading what="the places" />
                ) : (
                  <CampusMap packs={packs.data} places={places} result={shown} friends={b.friends} disruptions={disruptions} scrub={scrub} onScrubTo={scrubTo} />
                )}
              </div>
            </section>
          ) : null}

          <BoardCard board={b} places={places} result={shown} />

          <section className={ui.section}>
            <h2 className={ui.sectionTitle}>Your plan</h2>
            {b.slots.length === 0 ? (
              <Empty title="No open slots in the next 24 hours">
                {b.blocks.length > 0 ? "Fixed blocks fill the whole day, so there is nothing to plan." : "The box has nothing on record for this day yet. Once your calendar syncs, the slots appear here."}
              </Empty>
            ) : (
              <Planner board={b} moves={moves} onChange={(next) => setMoves(next)} result={shown} changes={changes} places={places} />
            )}
          </section>

          {shown ? <OutcomeCard result={shown} /> : null}

          {shown ? (
            <section className={`${ui.section} ${dr.anchor}`} id="hard">
              <h2 className={ui.sectionTitle}>Hard mode</h2>
              <div className={ui.card}>
                <p className={ui.muted}>Things go wrong: a block moves, a place closes, a friend cancels. The box re-checks every move and re-routes.</p>
                <div className={`${ui.row} ${ui.wrap}`}>
                  {mode === "demo" ? (
                    <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} disabled={hard?.busy || moves.length === 0} onClick={() => runHard(true, 0)}>
                      Hard mode
                    </button>
                  ) : null}
                  <button type="button" className={`${ui.btn} ${mode === "demo" ? "" : ui.btnPrimary}`} disabled={hard?.busy || moves.length === 0} onClick={() => runHard(false, seed)}>
                    {mode === "demo" ? "Random" : "Hard mode"}
                  </button>
                  <label className={dr.seed}>
                    <span>Seed</span>
                    <input className={`${ui.input} ${dr.numInput}`} type="number" inputMode="numeric" value={seed} onChange={(e) => setSeed(Number.isFinite(e.target.valueAsNumber) ? Math.trunc(e.target.valueAsNumber) : 0)} />
                  </label>
                  {hard ? (
                    <button type="button" className={dr.pill} onClick={() => setHard(null)}>
                      Back to the calm day
                    </button>
                  ) : null}
                </div>
                {moves.length === 0 ? <p className={`${ui.muted} ${ui.small}`}>Plan at least one move first.</p> : null}
                {hard?.busy ? <Loading what="the disrupted day" /> : null}
                {hard?.error ? <WriteError error={hard.error} /> : null}
                {hard?.result ? <HardCard run={hard.result} scripted={Boolean(hard.scripted)} view={view} onView={setView} /> : null}
              </div>
            </section>
          ) : null}

          {sim.result ? (
            <section className={ui.section}>
              <h2 className={ui.sectionTitle}>Keep this plan</h2>
              <div className={ui.card}>
                {commit.done ? (
                  <>
                    <p className={dr.big}>An approval card is waiting</p>
                    <p className={ui.breakWords}>{commit.done.reason}.</p>
                    <p className={ui.muted}>Nothing is on your calendar yet: {commit.done.events.length === 1 ? "the block is" : `the ${commit.done.events.length} blocks are`} added only after you approve.</p>
                    <Link href="/approvals/" className={`${ui.btn} ${ui.btnPrimary} ${dr.linkBtn}`}>
                      Open Approvals
                    </Link>
                  </>
                ) : sim.result.outcome.won ? (
                  <>
                    <p>
                      This plan wins. Commit it and Atlas asks to add its {sim.result.blocks.filter((x) => !x.fixed).length} blocks to your calendar.
                      {hard?.result ? " The calm plan is committed, not the disrupted one." : ""}
                    </p>
                    <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} disabled={commit.busy || sim.busy} onClick={doCommit}>
                      {commit.busy ? "Committing…" : "Commit to my calendar"}
                    </button>
                  </>
                ) : (
                  <p className={ui.muted}>Only a winning plan can be committed. Fix what fails above and the button appears here.</p>
                )}
                {commit.error ? <CommitError error={commit.error} /> : null}
              </div>
            </section>
          ) : null}

          <section className={ui.section}>
            <h2 className={ui.sectionTitle}>More</h2>
            <More scenario={scenario} day={dateOf(b.start)} moves={moves} result={sim.result} campaignPreset={mode === "demo" && presets.campaign.length > 0 ? presets.campaign : null} openAll={openMore} />
          </section>
        </>
      )}
    </>
  );
}

/** The one-line answer, right under the presets: did this plan win? */
function Verdict({ sim, shown, moves, hard, view }: { sim: Sim; shown: SimResult | undefined; moves: number; hard: boolean; view: "after" | "before" }) {
  if (!shown) return sim.error ? null : <Loading what="the simulation" />;
  const o = shown.outcome;
  return (
    <div className={`${dr.verdict} ${o.won ? dr.verdictWin : dr.verdictFail}`} role="status" aria-live="polite">
      <div className={`${ui.row} ${ui.between}`}>
        <p className={dr.verdictWord}>
          <span aria-hidden="true">{o.won ? "✓ " : "✗ "}</span>
          {o.won ? "Win" : "Fail"}
          {hard ? <span className={dr.verdictNote}> · hard mode, {view === "after" ? "after" : "before"}</span> : null}
        </p>
        <p className={dr.stars} aria-label={`${shown.stars} of 3 stars`}>
          {stars(shown.stars)}
        </p>
      </div>
      <p className={ui.breakWords}>{moves === 0 ? `With nothing planned: ${o.summary}` : o.summary}</p>
      {shown.rejected.length > 0 ? (
        <p className={ui.breakWords}>
          <strong>
            {shown.rejected.length} {shown.rejected.length === 1 ? "move was" : "moves were"} rejected:
          </strong>{" "}
          {shown.rejected.map((r) => r.reason).join("; ")}
        </p>
      ) : null}
    </div>
  );
}

/** Where the meters start: energy, focus, last night and meals so far. */
function StartTiles({ board, places }: { board: Board; places: ReturnType<typeof indexPlaces> }) {
  const i = board.initial;
  return (
    <section className={ui.section}>
      <h2 className={ui.sectionTitle}>You start with</h2>
      <p className={ui.muted}>
        {dateLabel(board.start)} {clock(board.start)} to {dateLabel(board.end)} {clock(board.end)}, starting at {placeName(places, board.start_place)}.
      </p>
      <div className={dr.tiles4}>
        <div className={ui.stat}>
          <p className={ui.statNum}>{Math.round(i.energy)}</p>
          <p className={ui.statLabel}>energy, of 100</p>
        </div>
        <div className={ui.stat}>
          <p className={ui.statNum}>{Math.round(i.focus)}</p>
          <p className={ui.statLabel}>focus, of 100</p>
        </div>
        <div className={ui.stat}>
          <p className={ui.statNum}>{amount(i.sleep_hours)} h</p>
          <p className={ui.statLabel}>sleep last night</p>
        </div>
        <div className={ui.stat}>
          <p className={ui.statNum}>{i.meals}</p>
          <p className={ui.statLabel}>{i.meals === 1 ? "meal" : "meals"} so far</p>
        </div>
      </div>
      <p className={`${ui.muted} ${ui.small}`}>
        {i.source === "defaults" ? "Habits has no numbers for today, so these are the box's default starting values." : `Starting values from: ${i.source}.`}
        {i.sleep_debt > 0 ? ` Sleep debt ${amount(i.sleep_debt)} h.` : ""}
      </p>

    </section>
  );
}

function BoardCard({ board, places, result }: { board: Board; places: ReturnType<typeof indexPlaces>; result?: SimResult }) {
  // Where the plan has you just before a fixed block: the walk the simulation takes to it, else where the day starts.
  const cameFrom = (place: string, start: string): string => {
    const walk = [...(result?.routes ?? [])].reverse().find((r) => r.end === place && ms(r.arrive) <= ms(start) && ms(start) - ms(r.arrive) <= 30 * 60000);
    return walk?.start ?? board.start_place;
  };
  return (
    <section className={ui.section}>
      <h2 className={ui.sectionTitle}>The board</h2>
      <div className={ui.card}>
        <h3>Deadlines</h3>
        {board.deadlines.length === 0 ? (
          <p className={ui.muted}>No deadline in reach of this day.</p>
        ) : (
          <ul className={ui.list}>
            {board.deadlines.map((d) => {
              const url = safeUrl(d.url);
              return (
                <li key={d.id}>
                  <p className={ui.breakWords}>
                    {url ? (
                      <a href={url} target="_blank" rel="noreferrer noopener">
                        {d.title}
                      </a>
                    ) : (
                      d.title
                    )}
                  </p>
                  <p className={`${ui.muted} ${ui.small}`}>
                    Due {dateLabel(d.due)} {clock(d.due)} · {minutes(d.work_left)} of focused work left{d.in_window ? " · due inside this day" : ""}
                  </p>
                  <div className={`${ui.row}`}>
                    <div className={`${ui.meter} ${ui.grow}`} role="img" aria-label={`Risk ${riskPercent(d.risk)}`}>
                      <div className={`${ui.meterFill} ${dr.riskFill}`} style={{ width: `${Math.round(Math.max(0, Math.min(1, d.risk)) * 100)}%` }} />
                    </div>
                    <span className={dr.riskText}>risk {riskPercent(d.risk)}</span>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      <div className={ui.card}>
        <h3>Fixed blocks</h3>
        {board.blocks.length === 0 ? (
          <p className={ui.muted}>Nothing fixed on the calendar: the whole day is open.</p>
        ) : (
          <ul className={dr.rows}>
            {board.blocks.map((f) => (
              <li key={f.id}>
                <span className={dr.when}>{clockSpan(f.start, f.end)}</span>
                <span className={`${ui.row} ${ui.between}`}>
                  <span className={ui.breakWords}>
                    {f.title}
                    <span className={ui.muted}> · {f.place_name ?? "anywhere"}</span>
                  </span>
                  {f.place ? <RouteSheet from={cameFrom(f.place, f.start)} to={f.place} arriveBy={f.start} now={board.start} what={f.title} /> : null}
                </span>
              </li>
            ))}
          </ul>
        )}
        {board.events.length > 0 ? (
          <>
            <h3>Optional</h3>
            <ul className={dr.rows}>
              {board.events.map((e) => (
                <li key={e.id}>
                  <span className={dr.when}>{clockSpan(e.start, e.end)}</span>
                  <span className={ui.breakWords}>
                    {e.title}
                    <span className={ui.muted}> · {e.place_name ?? "anywhere"}</span>
                  </span>
                </li>
              ))}
            </ul>
          </>
        ) : null}
        <h3>Friends</h3>
        {board.friends.length === 0 ? (
          <p className={ui.muted}>No friend has shared a place for this day.</p>
        ) : (
          <ul className={dr.rows}>
            {board.friends.map((f) => (
              <li key={`${f.member}-${f.start}`}>
                <span className={dr.when}>{clockSpan(f.start, f.end)}</span>
                <span className={`${ui.row} ${ui.between}`}>
                  <span className={ui.breakWords}>
                    {f.member}
                    <span className={ui.muted}> · {f.place_name || placeName(places, f.place)}</span>
                  </span>
                  <RouteSheet from={board.start_place} to={f.place} now={board.start} what={`${f.member} is here ${clockSpan(f.start, f.end)}`} />
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}

function OutcomeCard({ result }: { result: SimResult }) {
  const w = useWords();
  const o = result.outcome;
  const t = result.totals;
  return (
    <section className={ui.section}>
      <h2 className={ui.sectionTitle}>Outcome</h2>
      <div className={ui.card}>
        <div className={`${ui.row} ${ui.between}`}>
          <p className={dr.big}>{o.won ? "Win" : "Fail"}</p>
          <p className={`${dr.stars} ${result.stars === 0 ? dr.starsNone : ""}`} aria-label={`${result.stars} of 3 stars`}>
            {stars(result.stars)}
          </p>
        </div>
        <ul className={dr.checks}>
          {o.checks.map((c) => (
            <li key={c.id} className={c.ok ? dr.checkOk : dr.checkFail}>
              <span className={dr.checkMark} aria-hidden="true">
                {c.ok ? "✓" : "✗"}
              </span>
              <span className={ui.breakWords}>
                <span className={dr.srOnly}>{c.ok ? "Passed: " : "Failed: "}</span>
                {c.text}
              </span>
            </li>
          ))}
        </ul>
        {result.notes.map((n, i) => (
          <p key={i} className={`${ui.muted} ${ui.small}`}>
            {n}
          </p>
        ))}
        <div className={dr.tiles}>
          <div className={ui.stat}>
            <p className={ui.statNum}>{t.points}</p>
            <p className={ui.statLabel}>points, of 100</p>
          </div>
          <div className={ui.stat}>
            <p className={ui.statNum}>{minutes(t.focused_minutes)}</p>
            <p className={ui.statLabel}>of real work, from {minutes(t.study_minutes)} of {w.activity}</p>
          </div>
          <div className={ui.stat}>
            <p className={ui.statNum}>{amount(t.sleep_hours)} h</p>
            <p className={ui.statLabel}>sleep</p>
          </div>
          <div className={ui.stat}>
            <p className={ui.statNum}>{t.meals}</p>
            <p className={ui.statLabel}>
              {t.meals === 1 ? "meal" : "meals"}
              {t.meals_before ? `, ${t.meals_before} of them before this board starts` : ""}
            </p>
          </div>
          <div className={ui.stat}>
            <p className={ui.statNum}>{minutes(t.walk_minutes)}</p>
            <p className={ui.statLabel}>walking</p>
          </div>
          <div className={ui.stat}>
            <p className={ui.statNum}>
              {Math.round(t.energy_min)} → {Math.round(t.energy_end)}
            </p>
            <p className={ui.statLabel}>energy: lowest, then at the end</p>
          </div>
        </div>
      </div>
    </section>
  );
}

function HardCard({ run, scripted, view, onView }: { run: HardResult; scripted: boolean; view: "after" | "before"; onView: (v: "after" | "before") => void }) {
  return (
    <>
      <p className={`${ui.muted} ${ui.small}`}>{scripted ? "The demo's scripted disruptions." : `Seed ${run.seed}: the same disruptions every time for this seed.`}</p>
      <h3>What went wrong</h3>
      {run.disruptions.length === 0 ? (
        <p className={ui.muted}>The box found nothing to disrupt in this plan.</p>
      ) : (
        <ul className={dr.bullets}>
          {run.disruptions.map((d, i) => (
            <li key={i} className={ui.breakWords}>
              {d.text || d.kind.replace("_", " ")}
            </li>
          ))}
        </ul>
      )}
      {run.note ? <p className={ui.nudge}>{run.note}</p> : null}
      <h3>What the box changed</h3>
      {run.changes.length === 0 ? (
        <p className={ui.muted}>Nothing: the plan stands as it was.</p>
      ) : (
        <ul className={dr.changes}>
          {run.changes.map((c, i) => (
            <li key={i}>
              <span className={`${ui.tag} ${c.kind === "rejected" ? dr.tagDanger : ui.tagAccent}`}>{c.kind.replace("_", " ")}</span>
              <span className={ui.breakWords}>
                {c.move !== null ? `Move ${c.move + 1}: ` : ""}
                {c.text}
              </span>
            </li>
          ))}
        </ul>
      )}
      <dl className={ui.kv}>
        <dt>Before</dt>
        <dd>{line(run.before)}</dd>
        <dt>After</dt>
        <dd>{line(run.after)}</dd>
      </dl>
      <div className={ui.segmented} role="group" aria-label="Which day the meters and map show">
        <button type="button" className={`${ui.seg} ${view === "after" ? ui.segOn : ""}`} aria-pressed={view === "after"} onClick={() => onView("after")}>
          Show after
        </button>
        <button type="button" className={`${ui.seg} ${view === "before" ? ui.segOn : ""}`} aria-pressed={view === "before"} onClick={() => onView("before")}>
          Show before
        </button>
      </div>
      <p className={`${ui.muted} ${ui.small}`}>The meters, the map and the outcome above now show the {view === "after" ? "disrupted" : "calm"} day.</p>
    </>
  );
}

/** 422: the plan does not win (with the box's reasons). 409: this box cannot write to the calendar. */
function CommitError({ error }: { error: ApiError }) {
  if (error.needsToken) return <WriteError error={error} />;
  return (
    <div className={`${ui.state} ${ui.stateError}`} role="alert">
      <p className={ui.stateTitle}>{error.status === 409 ? "The calendar can't be written from this box" : error.status === 422 ? "Not committed" : "Couldn't commit"}</p>
      <p className={ui.breakWords}>
        {error.message}
        {error.status ? ` (HTTP ${error.status})` : ""}
      </p>
      {error.reasons.length > 0 ? (
        <ul className={dr.bullets}>
          {error.reasons.map((r, i) => (
            <li key={i} className={ui.breakWords}>
              {r}
            </li>
          ))}
        </ul>
      ) : null}
      {error.status === 409 ? <p>Connect a calendar that can create events and try again. The plan above is unchanged.</p> : null}
    </div>
  );
}
