"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, type PlayedRound, type PlayState } from "@/lib/api";
import { useAtlas } from "@/lib/atlas";
import { humanize } from "@/lib/format";
import css from "./play.module.css";

/**
 * Sort It's face loop, made visible: the chip in the round header, "How this round works", and the first round
 * against the latest one.
 *
 * Everything shown is the game's own state (GET /api/play/state, contexts/play/domain/game.py: `state`), read again
 * on every `perception.faces.counted` and `play.round.*` event on /ws, and every two seconds as a safety net. The
 * game runs on the face counts' own times, so between two counts the seconds here tick on from the last answer and
 * stop at the game's own limits (2 s to start, 10 s to end): the chip never claims a round the box has not started.
 *
 * Counts only. No face picture is fetched, kept or drawn anywhere on this screen.
 */

const POLL_MS = 2000;
const COALESCE_MS = 60;
const QUIET_S = 6; // a face camera that has said nothing for this long is not ticking the loop

export interface FaceLoop {
  /** undefined while loading; null when Sort It is not set up on this box. */
  state: PlayState | null | undefined;
  /** Seconds since the answer above was read, for ticking between two face counts. */
  since: number;
}

export function useFaceLoop(): FaceLoop {
  const { subscribe } = useAtlas();
  const [state, setState] = useState<PlayState | null | undefined>(undefined);
  const [since, setSince] = useState(0);
  const at = useRef(0);
  const text = useRef("");

  const load = useCallback(() => {
    api
      .playState()
      .then((s) => {
        const next = JSON.stringify(s);
        if (next !== text.current) {
          text.current = next;
          at.current = performance.now();
          setSince(0);
          setState(s);
        }
      })
      .catch((err: unknown) => {
        if (err && typeof err === "object" && "status" in err && (err as { status: number }).status === 404) setState(null);
      });
  }, []);

  useEffect(() => {
    let timer = 0;
    load();
    const off = subscribe((e) => {
      if (!(e.type.startsWith("perception.faces.") || e.type.startsWith("play.round.")) || timer) return;
      timer = window.setTimeout(() => {
        timer = 0;
        load();
      }, COALESCE_MS);
    });
    const poll = window.setInterval(load, POLL_MS);
    return () => {
      off();
      window.clearTimeout(timer);
      window.clearInterval(poll);
    };
  }, [subscribe, load]);

  // tick only while a number on the chip is moving
  const moving = state?.phase === "holding" || state?.phase === "ending";
  useEffect(() => {
    if (!moving) return;
    const id = window.setInterval(() => setSince((performance.now() - at.current) / 1000), 100);
    return () => window.clearInterval(id);
  }, [moving, state]);

  return { state, since };
}

const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n === 1 ? one : many}`;

/** What the chip says, from the game's state. Exported for the page's own use and for reading in one place. */
export function faceWords(state: PlayState, since: number, now = Date.now()): { text: string; sub: string | null; fill: number } {
  const quiet = state.heard === null || (now - Date.parse(state.heard)) / 1000 > QUIET_S;
  if (state.phase === "holding") {
    const held = Math.min(state.hold_to_start_s, state.held_s + (quiet ? 0 : since));
    return { text: `Face held ${held.toFixed(1)} s`, sub: `a round starts at ${state.hold_to_start_s.toFixed(0)} s · ${plural(state.faces, "face")} facing the camera`, fill: held / state.hold_to_start_s };
  }
  if (state.phase === "round" && state.round) {
    const n = state.round.players;
    return { text: `Round started · ${plural(n, "player")} (${plural(n, "face")})`, sub: `${humanize(state.round.item)} · ${plural(state.faces, "face")} facing the camera now`, fill: 1 };
  }
  if (state.phase === "ending" && state.round) {
    const idle = Math.min(state.idle_to_end_s, state.idle_s + (quiet ? 0 : since));
    return { text: `No face for ${idle.toFixed(0)} s: ending soon`, sub: `the round ends at ${state.idle_to_end_s.toFixed(0)} s with nobody facing the camera`, fill: 1 - idle / state.idle_to_end_s };
  }
  const sub =
    state.heard === null
      ? "no face count from the camera yet"
      : quiet
        ? "the face camera is quiet"
        : state.count > 0
          ? `${plural(state.count, "face")} in view, none facing the camera`
          : "nobody in view · face the bin camera for 2 s to start";
  return { text: "Looking for a player", sub, fill: 0 };
}

const R = 10;
const C = 2 * Math.PI * R;

/** The state chip for the round header. A short ring fills to the 2 s that start a round, and drains over the 10 s that end one. */
export function FaceChip({ loop }: { loop: FaceLoop }) {
  const { state, since } = loop;
  if (!state) return null;
  const words = faceWords(state, since);
  return (
    <span className={css.face} data-phase={state.phase} data-guide="play:face" role="status" aria-label={`Face detection: ${words.text}`}>
      <svg className={css.ring} viewBox="0 0 28 28" aria-hidden="true">
        <circle className={css.ringTrack} cx={14} cy={14} r={R} />
        <circle className={css.ringFill} cx={14} cy={14} r={R} strokeDasharray={C} strokeDashoffset={C * (1 - Math.max(0, Math.min(1, words.fill)))} />
        <circle className={css.ringDot} cx={14} cy={14} r={3.5} />
      </svg>
      <span className={css.faceText} aria-hidden="true">
        {words.text}
        {words.sub ? <span className={css.faceSub}>{words.sub}</span> : null}
      </span>
    </span>
  );
}

/** Three short lines for someone seeing the game for the first time. */
export function HowItWorks() {
  return (
    <details className={css.how} data-guide="play:how">
      <summary>
        <span>How this round works</span>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M6 9l6 6 6-6" />
        </svg>
      </summary>
      <ol className={css.howList}>
        <li>
          <span>
            <strong>A face starts it.</strong> Face detection counts the faces looking at the bin camera. One held for 2 seconds starts the round, each face is a player, and 10 seconds with no face ends it. Counts only: no face picture is kept or shown.
          </span>
        </li>
        <li>
          <span>
            <strong>The camera sees the item.</strong> Whatever is in your hand is the round&apos;s item, and the sorting rules name its bin.
          </span>
        </li>
        <li>
          <span>
            <strong>A* draws the walk.</strong> The floor is a grid of 10 cm cells. A* searches it for the shortest way to the bin around furniture and people, and searches again when something moves into the way.
          </span>
        </li>
      </ol>
    </details>
  );
}

function RoundRow({ r }: { r: PlayedRound }) {
  const over = r.seconds > r.estimate_s;
  return (
    <div className={css.proofRow}>
      <dt>
        Round {r.round} · {humanize(r.item)}
      </dt>
      <dd>
        {r.seconds.toFixed(1)} s against an estimate of {r.estimate_s.toFixed(1)} s · {Math.round(r.score)} points · {r.correct ? "right bin" : "wrong bin"}
        <div className={css.proofBar} aria-hidden="true">
          <div className={`${css.proofBarFill} ${over ? css.proofBarOver : ""}`} style={{ width: `${Math.min(100, (r.seconds / (r.estimate_s * 2)) * 100)}%` }} />
        </div>
      </dd>
    </div>
  );
}

/**
 * The first round of this sitting against the latest, from the box's own scored rounds. Rendered only when two
 * or more rounds were played; with fewer there is nothing to compare, and nothing is shown.
 */
export function RoundProof({ state }: { state: PlayState | null | undefined }) {
  const first = state?.rounds.first;
  const latest = state?.rounds.latest;
  if (!first || !latest) return null;
  const faster = first.seconds - latest.seconds;
  const points = latest.score - first.score;
  const time = Math.abs(faster) < 0.05 ? `took the same time as round ${first.round}` : `was ${Math.abs(faster).toFixed(1)} s ${faster > 0 ? "faster" : "slower"} than round ${first.round}`;
  const score = Math.abs(points) < 0.5 ? "the same score" : `${Math.abs(Math.round(points))} ${Math.abs(Math.round(points)) === 1 ? "point" : "points"} ${points > 0 ? "more" : "fewer"}`;
  return (
    <div className={css.proof} data-guide="play:proof">
      <p className={css.proofLine}>
        Round {latest.round} {time}, with {score}.
      </p>
      <dl className={css.proofRows}>
        <RoundRow r={first} />
        <RoundRow r={latest} />
      </dl>
    </div>
  );
}
