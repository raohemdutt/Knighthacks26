"use client";

import { useEffect, useMemo, useState } from "react";
import ui from "@/components/ui.module.css";
import { api } from "@/lib/api";
import { useFetch } from "@/lib/useFetch";
import { FloorMap } from "./FloorMap";
import o from "./op.module.css";
import { Why } from "./Shift";
import type { CheckKind, Replay, RoundStep, Shift, Target } from "./types";

/**
 * Layer 2, "A round, replayed": one mission as a row of steps. After a round, step through it or drag the slider. Press
 * Play and the box replays the recorded events in order (the existing replay: /api/spot/replay/*), with SPOT moving on
 * the plan and each check arriving as it happened; the label says "Live" while it follows the newest event.
 *
 * Every step's sentence comes from GET /api/spot/shift/rounds/{id}; the live picture only says how far SPOT has got.
 */

const ICON: Record<CheckKind, React.ReactNode> = {
  heat: (
    <>
      <path d="M10 4a2 2 0 0 1 4 0v9.5a4 4 0 1 1-4 0Z" />
      <path d="M12 9v7" />
    </>
  ),
  sound: (
    <>
      <path d="M4 10v4h3l4 3V7l-4 3Z" />
      <path d="M15 9.5a3.5 3.5 0 0 1 0 5" />
      <path d="M17.5 7a7 7 0 0 1 0 10" />
    </>
  ),
  camera: (
    <>
      <path d="M4 8h3l1.5-2h7L17 8h3v10H4Z" />
      <circle cx="12" cy="13" r="3" />
    </>
  ),
  gauge: (
    <>
      <path d="M4.5 16a8 8 0 1 1 15 0" />
      <path d="M12 14l3.5-4.5" />
      <circle cx="12" cy="14" r="1" />
    </>
  ),
  walk: (
    <>
      <path d="M5 18c3-1 4-4 6-6s4-2 8-6" />
      <path d="M15 6h4v4" />
    </>
  ),
};
const KIND_WORDS: Record<CheckKind, string> = { heat: "Heat check", sound: "Listening check", camera: "Camera check", gauge: "Gauge reading", walk: "Walk" };

export function CheckIcon({ kind }: { kind: CheckKind }) {
  return (
    <svg viewBox="0 0 24 24" width="24" height="24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {ICON[kind] ?? ICON.camera}
    </svg>
  );
}

function Mark({ step, state }: { step: RoundStep; state: "done" | "now" | "later" }) {
  if (state === "later") return <i className={`${o.badge} ${o.badgeLater}`} aria-hidden="true" />;
  if (state === "now") return <i className={`${o.badge} ${o.badgeNow}`} aria-hidden="true">…</i>;
  return (
    <i className={`${o.badge} ${!step.worked ? o.badgeBad : step.alarm ? o.badgeWatch : o.badgeOk}`} aria-hidden="true">
      {!step.worked ? "✕" : step.alarm ? "!" : "✓"}
      {!step.worked && step.alarm ? <b className={o.badgeAlarm}>!</b> : null}
    </i>
  );
}

export function Battery({ pct }: { pct: number | null }) {
  const v = pct === null ? 0 : Math.max(0, Math.min(100, pct));
  const tone = v < 20 ? o.battBad : v < 40 ? o.battWatch : o.battOk;
  return (
    <div className={o.battery} role="img" aria-label={pct === null ? "Battery unknown" : `Battery ${Math.round(v)} percent`}>
      <span className={o.battBody}>
        <i className={tone} style={{ width: `${v}%` }} />
        <em style={{ left: "40%" }} />
      </span>
      <span className={o.battCap} />
      <b>{pct === null ? "–" : `${Math.round(v)}%`}</b>
      <span className={o.battNote}>{v < 20 ? "very low" : v < 40 ? "low: checks fail more often" : "battery"}</span>
    </div>
  );
}

export function Round({
  shift,
  live,
  onLive,
  onWhy,
  startRound,
  startStep,
}: {
  shift: Shift;
  live: Replay | null;
  onLive: (r: Replay) => void;
  onWhy: (t: Target) => void;
  startRound?: string | null;
  startStep?: number | null;
}) {
  const [picked, setPicked] = useState<string>(startRound || shift.latest_round || "");
  const [cursor, setCursor] = useState<number | null>(startStep ?? null); // steps shown so far; null = the whole round
  const round = useFetch((sig) => (picked ? api.spotRound(picked, sig) : Promise.reject(new Error("no round"))), picked);
  const playing = Boolean(live?.playing);
  const liveId = live?.mission?.mission_id ?? null;
  const liveSteps = live?.mission?.actions.length ?? 0;

  // while the box replays, follow it: the round it is in, and how far SPOT has got
  useEffect(() => {
    if (!playing || !liveId) return;
    setPicked(liveId);
    setCursor(liveSteps);
  }, [playing, liveId, liveSteps]);

  const steps = round.data?.steps ?? [];
  const shown = cursor === null ? steps.length : Math.max(0, Math.min(cursor, steps.length));
  const current = shown > 0 ? steps[shown - 1] : undefined;
  const running = playing && liveId === picked && live?.mission?.actions[shown - 1]?.status === "RUNNING";
  const route = useMemo(() => steps.map((s) => s.location_id), [steps]);
  const head = shift.rounds.find((r) => r.mission_id === picked);
  const battery = current ? current.battery_pct : head ? head.start_battery_pct : null;

  const call = (fn: () => Promise<Replay>) => void fn().then(onLive).catch(() => undefined);
  const play = () => {
    if (playing) return call(api.spotReplayPause);
    const midway = liveId === picked && live?.mission?.status === "RUNNING" && shown === liveSteps;
    if (midway) return call(api.spotReplayStart);
    void api
      .spotReplaySeek({ mission_id: picked })
      .then(() => api.spotReplayStart())
      .then(onLive)
      .catch(() => undefined);
  };
  const move = (to: number) => {
    if (playing) call(api.spotReplayPause);
    setCursor(Math.max(0, Math.min(to, steps.length)));
  };
  const choose = (id: string) => {
    if (playing) call(api.spotReplayPause);
    setPicked(id);
    setCursor(null);
  };

  return (
    <section className={o.round} aria-label="A round, replayed">
      <div className={o.roundHead}>
        <div>
          <h2 className={o.layerTitle}>A round, replayed</h2>
          <p className={o.layerSub}>Step through any round after it ran, or press Play to watch it as it happened.</p>
        </div>
        <span className={`${o.mode} ${playing ? o.modeLive : ""}`} aria-live="polite">
          <i /> {playing ? "Live · following SPOT" : cursor === null ? "After the round" : "Paused"}
        </span>
      </div>

      <div className={o.roundGrid}>
        <FloorMap
          map={shift.map!}
          assets={shift.assets}
          quiet
          title="Where SPOT went"
          route={route}
          done={shown}
          robotAt={current?.location_id ?? null}
          onPick={(id) => onWhy({ kind: "asset", id })}
        />

        <div className={`${o.card} ${o.roundCard}`}>
          <label className={o.pickLabel}>
            <span>Round</span>
            <select className={`${ui.input} ${o.pick}`} value={picked} onChange={(e) => choose(e.target.value)}>
              {[...shift.rounds].reverse().map((r) => (
                <option key={r.mission_id} value={r.mission_id}>
                  {r.label}
                </option>
              ))}
            </select>
          </label>

          {round.data ? <p className={o.sentence}>{round.data.summary}</p> : <p className={o.note}>{round.error ? "This round could not be loaded." : "Loading the round…"}</p>}

          <ol className={o.steps} aria-label="Steps of the round">
            {steps.map((st, i) => {
              const state = i < shown - 1 || (i === shown - 1 && !running) ? "done" : i === shown - 1 ? "now" : "later";
              return (
                <li key={st.action_id}>
                  <button
                    type="button"
                    className={`${o.step} ${i === shown - 1 ? o.stepOn : ""} ${state === "later" ? o.stepLater : ""}`}
                    aria-current={i === shown - 1 ? "step" : undefined}
                    aria-label={`Step ${st.n}: ${state === "later" ? `${KIND_WORDS[st.check]}, not reached yet` : st.sentence}`}
                    onClick={() => move(i + 1)}
                  >
                    <CheckIcon kind={st.check} />
                    <Mark step={st} state={state} />
                    <span className={o.stepN}>{st.n}</span>
                  </button>
                </li>
              );
            })}
          </ol>
          <p className={o.stepKey}>
            <span><i className={`${o.badge} ${o.badgeOk}`}>✓</i> worked</span>
            <span><i className={`${o.badge} ${o.badgeBad}`}>✕</i> failed</span>
            <span><i className={`${o.badge} ${o.badgeWatch}`}>!</i> alarm</span>
          </p>

          <div className={o.nowBox} aria-live="polite">
            {current ? (
              <>
                <p className={o.nowLine}>
                  <span className={o.nowIcon}>
                    <CheckIcon kind={current.check} />
                  </span>
                  {running ? `${current.time} · ${current.title} · in progress` : current.sentence}
                </p>
                <p className={o.note}>
                  Step {current.n} of {steps.length} · at {current.place} · {current.asset}{" "}
                  <Why to={current.why_ref} onWhy={onWhy} />
                </p>
              </>
            ) : (
              <p className={o.nowLine}>{steps.length ? "SPOT is setting off. Press Play, or tap a step." : "…"}</p>
            )}
            <Battery pct={battery} />
          </div>

          <div className={o.controls}>
            <button type="button" className={`${ui.btn} ${ui.btnPrimary} ${o.play}`} onClick={play} disabled={!steps.length}>
              {playing ? "Pause" : "Play"}
            </button>
            <button type="button" className={ui.btn} onClick={() => move(shown - 1)} disabled={shown <= 0} aria-label="Previous step">
              ‹ Back
            </button>
            <button type="button" className={ui.btn} onClick={() => move(shown + 1)} disabled={shown >= steps.length} aria-label="Next step">
              Next ›
            </button>
            <div className={ui.segmented} role="group" aria-label="Replay speed">
              {[
                [60, "Normal"],
                [240, "Fast"],
              ].map(([v, label]) => (
                <button type="button" key={v} className={`${ui.seg} ${o.speed} ${live?.speed === v ? ui.segOn : ""}`} aria-pressed={live?.speed === v} onClick={() => call(() => api.spotReplaySpeed(v as number))}>
                  {label}
                </button>
              ))}
            </div>
          </div>
          <input className={o.scrub} type="range" min={0} max={Math.max(1, steps.length)} value={shown} aria-label="Scrub through the round" onChange={(e) => move(Number(e.target.value))} />
          <p className={o.note}>
            This replays recorded data in the order it happened. There is no live robot connected to this box.{" "}
            {round.data ? <Why to={round.data.why_ref} onWhy={onWhy}>See the raw rows</Why> : null}
          </p>
        </div>
      </div>
    </section>
  );
}
