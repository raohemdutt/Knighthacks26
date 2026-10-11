"use client";

import { useEffect, useId, useRef, useState } from "react";
import css from "./buddies.module.css";

/**
 * Learning buddies: the box learning, drawn as a cartoon. The same white block as the Home screen's device (its
 * speaker grille, the thin line of light along the lid and the small black webcam on top), flattened to SVG.
 *
 *   idle         the box breathes slowly
 *   alone        it hops in place while it trains, its loss curve growing beside it, a small burst of sparks per step
 *   with-friend  a second box slides in from the right, the two bump with a squash and stretch, share a glow and hop
 *                in sync; the counters show the pooled examples and the joules per example dropping
 *
 * Everything it shows comes in through props: the caller passes what the gateway reported, nothing is made up here.
 * With reduced motion (the system's setting, or `forceReduced`) nothing moves: the states crossfade instead.
 */

export type BuddiesState = "idle" | "alone" | "with-friend";

export interface LearningBuddiesProps {
  state: BuddiesState;
  /** Training steps reported so far. */
  steps: number;
  /** The loss at each reported step, in order. */
  loss: number[];
  /** Joules per example for the run on screen, or null when no meter measured it. */
  joulesPerExample: number | null;
  /** Friends whose examples are pooled with this box's: 0 alone, 1 with a friend. */
  friends: number;
  /** Act 1's joules per example, so `with-friend` can count down from it. */
  joulesPerExampleAlone?: number | null;
  /** Examples with friends over examples alone (2 for one friend's equal share). */
  examplesRatio?: number | null;
  /** Why the loss curve is empty although steps were trained, such as a run whose per-step loss was not logged. */
  lossReason?: string | null;
  forceReduced?: boolean;
  /** A line under the stage, such as which runs these are. */
  caption?: React.ReactNode;
}

const fmt = (n: number, d = 1) => n.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

/** The system's reduced-motion setting, or the caller's override. */
function useReduced(force: boolean | undefined): boolean {
  const [reduced, setReduced] = useState(!!force);
  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    const apply = () => setReduced(!!force || !!mq?.matches);
    apply();
    mq?.addEventListener("change", apply);
    return () => mq?.removeEventListener("change", apply);
  }, [force]);
  return reduced;
}

/** A number that eases from `from` to `to` after `delay` ms; with reduced motion it is simply `to`. */
function useTween(from: number | null, to: number | null, delay: number, duration: number, reduced: boolean): number | null {
  const [value, setValue] = useState<number | null>(to);
  useEffect(() => {
    if (to === null) return setValue(null);
    if (reduced || from === null || from === to) return setValue(to);
    setValue(from);
    let frame = 0;
    const t0 = performance.now() + delay;
    const tick = (now: number) => {
      const k = Math.min(1, Math.max(0, (now - t0) / duration));
      const eased = 1 - Math.pow(1 - k, 3);
      setValue(from + (to - from) * eased);
      if (k < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [from, to, delay, duration, reduced]);
  return value;
}

/* ---------- the box ---------- */

const GRILLE: [number, number][] = [];
for (let row = 0; row < 5; row++) for (let col = 0; col < 14; col++) GRILLE.push([25 + col * 5.2 + (row % 2) * 2.6, 52 + row * 5.6]);

/** The Atlas box, front three-quarter: lid, grille, the light line along the lid's front edge, the webcam on top. */
function BoxArt({ bar }: { bar: "a" | "b" }) {
  const id = useId().replace(/:/g, "");
  return (
    <svg className={css.boxSvg} viewBox="0 0 132 100" aria-hidden="true">
      <defs>
        <linearGradient id={`front-${id}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#f7f6f3" />
          <stop offset="1" stopColor="#e2e0dc" />
        </linearGradient>
        <linearGradient id={`top-${id}`} x1="0" y1="0" x2="1" y2="0">
          <stop offset="0" stopColor="#ffffff" />
          <stop offset="1" stopColor="#ecebe8" />
        </linearGradient>
        <filter id={`glow-${id}`} x="-20%" y="-300%" width="140%" height="700%">
          <feGaussianBlur stdDeviation="1.8" />
        </filter>
      </defs>
      {/* the side and the top, for depth */}
      <path d="M108 30 L121 19 L121 77 Q121 80 118.5 82 L108 90 Z" fill="#cfd1d4" />
      <path d="M8 30 L21 19 L121 19 L108 30 Z" fill={`url(#top-${id})`} />
      {/* the webcam, small and black, on the top to the right and front */}
      <rect x="72" y="16.5" width="27" height="9" rx="3" fill="#1a1c20" />
      <circle cx="85.5" cy="21" r="2.7" fill="#070a10" />
      <circle cx="84.6" cy="20.1" r="0.8" fill="#9bbcf2" opacity="0.8" />
      {/* the front: lid over the lower shell */}
      <rect x="8" y="30" width="100" height="60" rx="5" fill={`url(#front-${id})`} />
      <rect x="8" y="30" width="100" height="60" rx="5" fill="none" stroke="rgba(20, 24, 32, 0.16)" />
      <line x1="9" y1="42" x2="107" y2="42" stroke="#8a8d92" strokeWidth="0.9" opacity="0.7" />
      {/* the line of light: Atlas is listening */}
      <rect className={css.bar} data-bar={bar} x="20" y="37.2" width="76" height="2.4" rx="1.2" filter={`url(#glow-${id})`} />
      <rect className={css.bar} data-bar={bar} x="20" y="37.5" width="76" height="1.8" rx="0.9" />
      {/* the speaker grille */}
      <rect x="20" y="48" width="76" height="31" rx="4" fill="#e9e8e5" />
      {GRILLE.map(([x, y]) => (x < 94 ? <circle key={`${x}-${y}`} cx={x} cy={y} r="1.25" fill="#2a2d32" /> : null))}
      {/* rubber feet */}
      <rect x="16" y="89" width="12" height="3.2" rx="1.4" fill="#b8babd" />
      <rect x="88" y="89" width="12" height="3.2" rx="1.4" fill="#b8babd" />
    </svg>
  );
}

/** One burst of sparks, placed by the step number so the same step always looks the same. */
function Sparks({ step }: { step: number }) {
  const rays = 6;
  return (
    <span className={css.sparks} aria-hidden="true">
      {Array.from({ length: rays }, (_, i) => {
        const angle = ((i / rays) * 360 + ((step * 47) % 60) - 30) * (Math.PI / 180);
        const reach = 54 + ((step * 13 + i * 29) % 24);
        return (
          <i
            key={i}
            className={css.spark}
            style={{ "--dx": `${(Math.cos(angle) * reach).toFixed(1)}px`, "--dy": `${(Math.sin(angle) * reach * 0.8 - 16).toFixed(1)}px`, "--d": `${i * 25}ms` } as React.CSSProperties}
          />
        );
      })}
    </span>
  );
}

/* ---------- the stage ---------- */

/** The two boxes, the floor, the shared glow and the contact flash; the root's data-state moves them. */
function Stage({ sparks = null }: { sparks?: React.ReactNode }) {
  return (
    <div className={css.stage} aria-hidden="true">
      <span className={css.glow} />
      <span className={css.floor} />
      <div className={`${css.slot} ${css.slotMe}`}>
        <span className={css.shadow} />
        <div className={css.bump}>
          <div className={css.mover}>
            <BoxArt bar="a" />
          </div>
        </div>
        {sparks}
      </div>
      <div className={`${css.slot} ${css.slotFriend}`}>
        <span className={css.shadow} />
        <div className={css.bump}>
          <div className={css.mover}>
            <BoxArt bar="b" />
          </div>
        </div>
      </div>
      <span className={css.contact} />
    </div>
  );
}

export interface BuddiesStageProps {
  state: BuddiesState;
  /** What a screen reader hears for the picture. */
  label: string;
  /** Both boxes side by side without meeting (outside `with-friend`). */
  apart?: boolean;
  /** Boxes shown dimmed and still: `me` is the left box, `friend` the right one. */
  off?: { me?: boolean; friend?: boolean };
  forceReduced?: boolean;
}

/**
 * The stage on its own, for another card (the Circle's Shared brain, G6): the same boxes and motion with no counters
 * or loss curve. Reduced motion is respected as above.
 */
export function BuddiesStage({ state, label, apart = false, off = {}, forceReduced }: BuddiesStageProps) {
  const reduced = useReduced(forceReduced);
  const cls = [css.root, css.bare, reduced ? css.calm : "", apart ? css.apart : "", off.me ? css.offMe : "", off.friend ? css.offFriend : ""];
  return (
    <div className={cls.filter(Boolean).join(" ")} data-state={state} role="img" aria-label={label}>
      <Stage />
    </div>
  );
}

/* ---------- the loss curve ---------- */

function LossCurve({ loss, steps, growKey, reason }: { loss: number[]; steps: number; growKey: string; reason: string | null }) {
  const W = 240;
  const H = 120;
  const pad = { l: 6, r: 10, t: 12, b: 18 };
  const top = Math.max(...loss, 0.0001) * 1.08;
  const n = Math.max(loss.length - 1, 1);
  const x = (i: number) => pad.l + (i / n) * (W - pad.l - pad.r);
  const y = (v: number) => pad.t + (1 - v / top) * (H - pad.t - pad.b);
  const pts = loss.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`);
  const last = loss.length ? loss[loss.length - 1]! : null;
  return (
    <figure className={css.curve}>
      <figcaption className={css.curveHead}>
        <span>Loss</span>
        <b>{last !== null ? fmt(last, 2) : reason ? "not logged" : "waiting"}</b>
      </figcaption>
      {!loss.length && reason ? <span className={css.curveNote}>{reason}</span> : null}
      <svg viewBox={`0 0 ${W} ${H}`} className={css.curveSvg} role="img" aria-label={loss.length ? `Loss over ${loss.length} steps, from ${fmt(loss[0]!, 2)} to ${fmt(last!, 2)}` : "No training step reported yet"}>
        <line x1={pad.l} x2={W - pad.r} y1={H - pad.b} y2={H - pad.b} className={css.baseline} />
        {loss.length > 1 ? (
          <>
            <path key={`a-${growKey}`} className={css.area} d={`M${pts[0]} L${pts.join(" L")} L${x(loss.length - 1).toFixed(1)},${H - pad.b} L${x(0).toFixed(1)},${H - pad.b} Z`} />
            <polyline key={`l-${growKey}`} className={css.lossLine} points={pts.join(" ")} pathLength={1} />
          </>
        ) : null}
        {last !== null ? <circle key={`d-${loss.length}`} className={css.head} cx={x(loss.length - 1)} cy={y(last)} r="4.5" /> : null}
        <text x={pad.l} y={H - 3} className={css.axis}>
          step 1
        </text>
        <text x={W - pad.r} y={H - 3} className={css.axis} textAnchor="end">
          {steps ? `step ${steps}` : ""}
        </text>
      </svg>
    </figure>
  );
}

/* ---------- the whole stage ---------- */

export function LearningBuddies({ state, steps, loss, joulesPerExample, friends, joulesPerExampleAlone = null, examplesRatio = null, lossReason = null, forceReduced, caption }: LearningBuddiesProps) {
  const reduced = useReduced(forceReduced);
  const together = state === "with-friend";
  // The pooled count shows once the friend has arrived; the joules start dropping just after the bump.
  const jpe = useTween(together ? (joulesPerExampleAlone ?? null) : null, joulesPerExample, 1500, 1600, reduced);
  const [lastStep, setLastStep] = useState(steps);
  const seen = useRef(steps);
  useEffect(() => {
    if (steps > seen.current) setLastStep(steps);
    seen.current = steps;
  }, [steps]);

  const said =
    state === "idle"
      ? "The box is resting. Nothing is training."
      : `The box is training ${together ? `with ${friends === 1 ? "a friend" : `${friends} friends`}` : "alone"}: ${steps} step${steps === 1 ? "" : "s"}${loss.length ? `, loss ${fmt(loss[loss.length - 1]!, 2)}` : ""}${joulesPerExample !== null ? `, ${fmt(joulesPerExample)} J per example` : ""}.`;

  const ratio = examplesRatio;
  return (
    <section className={`${css.root} ${reduced ? css.calm : ""}`} data-state={state} aria-label="The box learning">
      <p className={css.srOnly} aria-live="polite">
        {said}
      </p>
      <Stage sparks={state !== "idle" && lastStep > 0 && !reduced ? <Sparks key={`s-${lastStep}`} step={lastStep} /> : null} />

      <div className={css.side} key={state === "idle" ? "rest" : "work"}>
        {state === "idle" ? (
          <>
            <p className={css.resting}>Resting</p>
            <p className={css.counterLabel}>Nothing is training. The box hops as soon as a run reports its first step.</p>
          </>
        ) : (
        <div className={css.counters}>
          <div className={css.counter}>
            <span className={css.counterNum} key={`n-${steps}`}>
              {steps}
            </span>
            <span className={css.counterLabel}>{steps === 1 ? "step trained" : "steps trained"}</span>
          </div>
          {ratio !== null ? (
            <div className={`${css.counter} ${css.counterFriend}`}>
              <span className={`${css.counterNum} ${css.good}`}>{ratio.toFixed(ratio >= 1 ? 1 : 2)}x</span>
              <span className={css.counterLabel}>examples, pooled with {friends === 1 ? "a friend" : `${friends} friends`}</span>
            </div>
          ) : null}
          <div className={css.counter}>
            <span className={`${css.counterNum} ${jpe === null ? css.counterNone : ""} ${together && jpe !== null && joulesPerExampleAlone !== null && jpe < joulesPerExampleAlone ? css.good : ""}`}>
              {jpe !== null ? fmt(jpe) : "not measured yet"}
              {jpe !== null ? <small> J</small> : null}
            </span>
            <span className={css.counterLabel}>per example</span>
            {together && joulesPerExampleAlone !== null ? <span className={css.was}>alone it was {fmt(joulesPerExampleAlone)} J</span> : null}
          </div>
        </div>
        )}
        {state === "idle" ? null : <LossCurve loss={loss} steps={steps} growKey={state} reason={lossReason} />}
        {caption ? <p className={css.caption}>{caption}</p> : null}
      </div>
    </section>
  );
}
