"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import ui from "@/components/ui.module.css";
import { ApiError, api } from "@/lib/api";
import { useAtlas } from "@/lib/atlas";
import { useFetch } from "@/lib/useFetch";
import { FloorMap, LevelIcon } from "./FloorMap";
import o from "./op.module.css";
import { Battery, CheckIcon } from "./Round";
import r from "./run.module.css";
import type { RunResult, RunStep, Shift, WorkOrder } from "./types";

/**
 * Shift Run: plan SPOT's next round, run it, and see why it scored as it did, next to the plan Atlas would make.
 * For someone learning the job. It mirrors the app's Day Run: plan, simulate, a scored result with reasons, compare.
 *
 * The box does all the working out (POST /api/spot/shiftrun/simulate, a toy model built from the recorded rounds; see
 * contexts/spot/domain/shiftrun.py). This file keeps the plan, plays the result as an animation and shows the words.
 * "Send this plan" raises the usual approval card; a file is written on this box after Approve and nothing else happens.
 *
 * Links: ?tab=run&route=last|atlas|HX-202,HX-204&turn=40&show=result|mid&hard=1
 */

export interface RunLink {
  route: string | null;
  turn: number | null;
  show: string | null;
  hard: boolean;
}

function Stars({ n }: { n: number }) {
  return (
    <span className={r.stars} role="img" aria-label={`${n} of 3 stars`}>
      {[1, 2, 3].map((i) => (
        <svg key={i} viewBox="0 0 24 24" width="34" height="34" className={i <= n ? r.starOn : r.starOff} aria-hidden="true">
          <path d="M12 2.6l2.9 5.9 6.5.9-4.7 4.6 1.1 6.5L12 17.4l-5.8 3.1 1.1-6.5L2.6 9.4l6.5-.9z" />
        </svg>
      ))}
    </span>
  );
}

function SendPlan({ route, turnBack, orders }: { route: string[]; turnBack: number; orders: WorkOrder[] }) {
  const { approvals } = useAtlas();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const key = route.join(", ");
  const mine = approvals.items.filter((a) => a.target === "spot.mission_request" && a.params.route === key);
  const written = orders.filter((w) => w.kind === "mission_request");
  const run = (fn: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    fn()
      .then(() => approvals.reload())
      .catch((e: unknown) => setError(e instanceof ApiError ? (e.needsToken ? "This box asks for a token first (Settings)." : e.message) : String(e)))
      .finally(() => setBusy(false));
  };
  if (mine.length > 0) {
    const card = mine[0]!;
    return (
      <div className={o.confirm} role="group" aria-label="Confirm">
        <p>
          <b>Ready to send.</b> A request for this round will be written on this box. SPOT is not commanded and nothing leaves the building.
        </p>
        <div className={o.confirmRow}>
          <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} disabled={busy} onClick={() => run(() => api.decide(card.id, "approve"))}>
            Yes, write it
          </button>
          <button type="button" className={ui.btn} disabled={busy} onClick={() => run(() => api.decide(card.id, "deny"))}>
            Not now
          </button>
        </div>
      </div>
    );
  }
  return (
    <>
      <button type="button" className={`${ui.btn} ${o.act}`} disabled={busy || route.length === 0} onClick={() => run(() => api.spotPlan({ route, turn_back_at: turnBack }))}>
        Send this plan
      </button>
      {written.length > 0 ? (
        <p className={o.done}>
          <LevelIcon level="ok" size={18} /> Done: {written[0]!.work_order_id} is written on this box.
        </p>
      ) : null}
      {error ? <p className={o.error}>{error}</p> : null}
    </>
  );
}

function StepRow({ s, state, onUp, onDown, onDrop }: { s: RunStep; state: "plan" | "done" | "now" | "later"; onUp?: () => void; onDown?: () => void; onDrop?: () => void }) {
  const shown = state === "done" || state === "now";
  const tone = !shown ? "" : s.played === "ok" ? r.rowOk : s.played === "failed" ? r.rowBad : r.rowSkip;
  return (
    <li className={`${r.row} ${tone} ${state === "now" ? r.rowNow : ""} ${state === "later" ? r.rowLater : ""}`}>
      <span className={r.rowN}>{s.n}</span>
      <span className={r.rowIcon}>
        <CheckIcon kind={s.check} />
      </span>
      <span className={r.rowText}>
        <b>{s.name}</b>
        <span>
          {s.outcome === "checked"
            ? `${Math.round(s.battery_pct)}% battery · ${s.p_fail !== null && s.p_fail >= 0.5 ? "likely to fail" : `1 in ${Math.round(1 / Math.max(s.p_fail ?? 0.01, 0.01))} fail`}`
            : s.outcome === "blocked"
              ? "the way is blocked"
              : s.outcome === "stranded"
                ? "battery ran out"
                : "not reached: SPOT turned back"}
        </span>
      </span>
      {shown ? (
        <span className={r.rowResult}>
          <i className={`${o.badge} ${s.played === "ok" ? o.badgeOk : s.played === "failed" ? o.badgeBad : o.badgeLater}`}>{s.played === "ok" ? "✓" : s.played === "failed" ? "✕" : "–"}</i> {s.played_words}
        </span>
      ) : state === "plan" ? (
        <span className={r.rowBtns}>
          <button type="button" onClick={onUp} disabled={!onUp} aria-label={`Move ${s.name} earlier`}>
            ↑
          </button>
          <button type="button" onClick={onDown} disabled={!onDown} aria-label={`Move ${s.name} later`}>
            ↓
          </button>
          <button type="button" onClick={onDrop} aria-label={`Remove ${s.name}`}>
            ✕
          </button>
        </span>
      ) : null}
    </li>
  );
}

export function ShiftRun({ shift, orders, link }: { shift: Shift; orders: WorkOrder[]; link: RunLink }) {
  const board = useFetch((sig) => api.spotRunBoard(sig));
  const [route, setRoute] = useState<string[]>([]);
  const [turnBack, setTurnBack] = useState(link.turn ?? 40);
  const [hard, setHard] = useState(link.hard);
  const [result, setResult] = useState<RunResult | null>(null);
  const [phase, setPhase] = useState<"plan" | "play" | "result">("plan");
  const [at, setAt] = useState(0); // stops played so far
  const [atlas, setAtlas] = useState(false); // Atlas's route drawn on the plan
  const seeded = useRef(false);
  const turn = useRef(0);

  // every change of plan is run through the model on the box; the newest answer wins
  useEffect(() => {
    const mine = ++turn.current;
    const t = window.setTimeout(() => {
      api
        .spotRunSimulate({ route, turn_back_at: turnBack, hard })
        .then((res) => {
          if (mine !== turn.current) return;
          setResult(res);
          if (!seeded.current && link.route === "atlas") {
            seeded.current = true;
            setRoute(res.recommended.route);
            if (link.turn === null) setTurnBack(res.recommended.turn_back_at);
          }
        })
        .catch(() => undefined);
    }, 120);
    return () => window.clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [route, turnBack, hard]);

  // a link can start from a plan
  useEffect(() => {
    if (!board.data?.configured || seeded.current || !link.route || link.route === "atlas") return;
    seeded.current = true;
    setRoute(link.route === "last" ? board.data.last_route ?? [] : link.route.toUpperCase().split(",").filter((id) => board.data!.candidates.some((c) => c.asset_id === id)));
  }, [board.data, link.route]);
  useEffect(() => {
    if (!result || !link.show || phase !== "plan" || result.route.length === 0 || result.route.join() !== route.join()) return;
    if (link.show === "result") (setAt(result.steps.length), setPhase("result"), setAtlas(true));
    if (link.show === "mid") (setAt(Math.ceil(result.steps.length / 2)), setPhase("play"));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [result]);

  // the play: one stop at a time, then the result
  const still = typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
  useEffect(() => {
    if (phase !== "play" || !result || link.show === "mid") return;
    if (at >= result.steps.length) {
      const t = window.setTimeout(() => setPhase("result"), still ? 0 : 700);
      return () => window.clearTimeout(t);
    }
    const t = window.setTimeout(() => setAt((v) => v + 1), still ? 0 : 850);
    return () => window.clearTimeout(t);
  }, [phase, at, result, still, link.show]);

  const order = useMemo(() => Object.fromEntries(route.map((id, i) => [id, i + 1])), [route]);
  if (board.error) return <p className={o.error}>Shift Run could not be loaded.</p>;
  if (!board.data?.configured || !shift.map) return <p className={o.note}>Loading Shift Run…</p>;
  const b = board.data;
  const max = b.max_stops ?? 12;
  const planning = phase === "plan";
  const edit = (next: string[]) => (setPhase("plan"), setAt(0), setRoute(next));
  const toggle = (id: string) => (route.includes(id) ? edit(route.filter((v) => v !== id)) : route.length < max ? edit([...route, id]) : undefined);
  const swap = (i: number, j: number) => {
    const next = [...route];
    [next[i], next[j]] = [next[j]!, next[i]!];
    edit(next);
  };
  const steps = result && result.route.join() === route.join() ? result.steps : [];
  const current = !planning && at > 0 ? steps[Math.min(at, steps.length) - 1] : undefined;
  const battery = current ? current.battery_pct : planning || !result ? b.start_battery_pct ?? null : at >= steps.length ? result.end_battery_pct : b.start_battery_pct ?? null;
  const left = b.candidates.filter((c) => !route.includes(c.asset_id));
  const turnAt = b.turn_back ?? { choices: [40], default: 40, low: 40, danger: 20 };

  return (
    <section className={r.wrap} aria-label="Shift Run">
      <div className={o.roundHead}>
        <div>
          <h2 className={o.layerTitle}>Shift Run: plan SPOT's next round</h2>
          <p className={o.layerSub}>Pick what SPOT checks and in what order, choose when it turns back to charge, then run it and see how your plan holds up.</p>
        </div>
        <span className={o.mode}>
          <i /> Training · simulated
        </span>
      </div>

      <div className={o.roundGrid}>
        <FloorMap
          map={shift.map}
          assets={shift.assets}
          title={planning ? "Tap equipment to add a stop" : "Your round"}
          order={order}
          route={steps.length ? steps.map((s) => s.location_id) : route.map((id) => b.candidates.find((c) => c.asset_id === id)?.location_id ?? "")}
          done={planning ? undefined : at}
          robotAt={current?.location_id ?? null}
          route2={atlas && result ? result.recommended.route_places : undefined}
          routeLabel="Your route"
          guide="shiftrun:map"
          onPick={planning ? toggle : undefined}
          note={
            atlas ? (
              <p className={o.note}>
                <i className={r.atlasKey} /> Atlas's route, for comparison.
              </p>
            ) : (
              <p className={o.note}>A number on a mark is its place in your round. Tap a mark again to take it out.</p>
            )
          }
        />

        <div className={`${o.card} ${r.panel}`}>
          <div className={r.turn} data-guide="shiftrun:turn-back">
            <label htmlFor="turnback">
              Turn back to charge at <b>{turnBack}%</b> battery
            </label>
            <input
              id="turnback"
              className={o.scrub}
              type="range"
              min={Math.min(...turnAt.choices)}
              max={Math.max(...turnAt.choices)}
              step={5}
              value={turnBack}
              onChange={(e) => (setPhase("plan"), setAt(0), setTurnBack(Number(e.target.value)))}
            />
            <Battery pct={battery} />
          </div>

          <p className={r.estimate} aria-live="polite">
            {route.length === 0 ? `No stops yet. SPOT starts with ${Math.round(b.start_battery_pct ?? 0)}% battery; each stop takes about ${b.per_stop?.minutes} minutes and ${Math.round(b.per_stop?.battery_pct ?? 0)}% of the battery.` : result ? `Estimate: ${result.summary}` : "Working it out…"}
          </p>

          {steps.length ? (
            <ol className={r.rows}>
              {steps.map((s, i) => (
                <StepRow
                  key={`${s.asset_id}-${i}`}
                  s={s}
                  state={planning ? "plan" : i < at - 1 || phase === "result" ? "done" : i === at - 1 ? "now" : "later"}
                  onUp={i > 0 ? () => swap(i, i - 1) : undefined}
                  onDown={i < steps.length - 1 ? () => swap(i, i + 1) : undefined}
                  onDrop={() => edit(route.filter((_, j) => j !== i))}
                />
              ))}
            </ol>
          ) : null}

          <div className={o.controls}>
            <button type="button" className={`${ui.btn} ${ui.btnPrimary} ${o.play}`} data-guide="shiftrun:run" disabled={!steps.length || phase === "play"} onClick={() => (setAt(0), setPhase("play"))}>
              {phase === "result" ? "Run it again" : "Run my plan"}
            </button>
            <button type="button" className={ui.btn} data-guide="shiftrun:last" onClick={() => edit(b.last_route ?? [])}>
              Start from last round
            </button>
            <button type="button" className={ui.btn} onClick={() => edit([])} disabled={route.length === 0}>
              Clear
            </button>
            {b.hard?.blocked ? (
              <button type="button" className={`${ui.chip} ${hard ? ui.chipOn : ""}`} aria-pressed={hard} onClick={() => (setPhase("plan"), setAt(0), setHard((v) => !v))}>
                Hard mode
              </button>
            ) : null}
          </div>
          {hard && b.hard?.words ? <p className={o.note}>{b.hard.words}</p> : null}

          {planning ? (
            <div className={r.pool} data-guide="shiftrun:pool">
              <h3 className={r.poolTitle}>
                Add a stop <span>{route.length} of {max} chosen</span>
              </h3>
              <ul>
                {left.map((c) => (
                  <li key={c.asset_id}>
                    <button type="button" className={r.cand} onClick={() => toggle(c.asset_id)} disabled={route.length >= max}>
                      <LevelIcon level={c.level} size={20} />
                      <b>{c.name}</b>
                      <span>{c.why}</span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      </div>

      {phase === "result" && result ? (
        <div className={r.result}>
          <div className={`${o.card} ${r.mine}`}>
            <div className={r.scoreRow}>
              <Stars n={result.stars} />
              <div>
                <h3 className={r.verdict}>{result.verdict}</h3>
                <p className={o.note}>
                  Your plan scores {Math.round(result.score)}; Atlas's scores {Math.round(result.best_score)}. {result.summary}
                </p>
              </div>
            </div>
            <ul className={r.reasons}>
              {result.reasons.map((x, i) => (
                <li key={i}>
                  <LevelIcon level={x.tone === "good" ? "ok" : "watch"} size={20} />
                  <span>{x.text}</span>
                </li>
              ))}
            </ul>
            <div className={o.itemActs}>
              <SendPlan route={route} turnBack={turnBack} orders={orders} />
              <button type="button" className={o.why} onClick={() => setPhase("plan")}>
                Change my plan
              </button>
            </div>
          </div>

          <div className={`${o.card} ${r.atlas}`}>
            <div className={o.cardHead}>
              <h2>Atlas's plan</h2>
              <span className={o.hint}>from the same model</span>
            </div>
            <p className={o.sentence}>{result.recommended.summary}</p>
            <ol className={r.atlasRoute}>
              {result.recommended.steps.map((s) => (
                <li key={s.n} className={route.includes(s.asset_id) ? undefined : r.atlasNew}>
                  {s.name}
                </li>
              ))}
            </ol>
            <h3 className={r.poolTitle}>What made the difference</h3>
            <ul className={r.reasons}>
              {result.differences.map((x, i) => (
                <li key={i}>
                  <span className={r.bullet} aria-hidden="true" />
                  <span>{x}</span>
                </li>
              ))}
            </ul>
            <div className={o.itemActs}>
              <button type="button" className={`${ui.btn} ${o.act}`} onClick={() => (setTurnBack(result.recommended.turn_back_at), edit(result.recommended.route))}>
                Try Atlas's plan
              </button>
              <button type="button" className={`${ui.chip} ${atlas ? ui.chipOn : ""}`} aria-pressed={atlas} onClick={() => setAtlas((v) => !v)}>
                Show it on the plan
              </button>
            </div>
          </div>
        </div>
      ) : null}

      <p className={o.note}>
        <b>{b.footnote}</b> {b.how}
      </p>
    </section>
  );
}
