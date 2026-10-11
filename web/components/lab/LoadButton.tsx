"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api } from "@/lib/api";
import {
  LOAD_SECONDS,
  LOAD_STREAMS,
  RAMP,
  RAMP_STAGE_SECONDS,
  type RampView,
  askingText,
  loadButtonText,
  loadLine,
  loadProblem,
  rampRows,
} from "@/lib/loadRun";
import b from "./trainBurst.module.css";

/**
 * The "16 people asking" and "Ramp 1 to 32" buttons beside the live power graph: POST /api/learning/lab/load puts
 * that many streams on the 122B pair at once (a ramp adds streams stage by stage, never stopping one), GET is polled
 * every 2 s while it runs. `pressed` is when the press was answered on the browser clock, the graph's clock.
 */

const POLL_MS = 2000;

export function useLoadRun(enabled = true) {
  const [view, setView] = useState<RampView | null>(null);
  const [postError, setPostError] = useState<string | null>(null);
  const [pressed, setPressed] = useState<number | null>(null);
  const [ended, setEnded] = useState<number | null>(null);
  const [starting, setStarting] = useState(false);
  const running = view?.state === "running";
  const at = useRef<{ pressed: number | null; ended: number | null }>({ pressed: null, ended: null });
  at.current = { pressed, ended };

  const read = useCallback(async () => {
    try {
      const v = await api.loadRun();
      setView(v);
      if (v.state === "running" && at.current.pressed === null) setPressed(Date.now());
      else if (v.state !== "running" && at.current.pressed !== null && at.current.ended === null) setEnded(Date.now());
    } catch {
      /* the next poll tries again */
    }
  }, []);

  useEffect(() => {
    if (enabled) void read();
  }, [enabled, read]);

  useEffect(() => {
    if (!enabled || !running) return;
    const id = window.setInterval(() => void read(), POLL_MS);
    return () => window.clearInterval(id);
  }, [enabled, running, read]);

  const go = async (call: () => Promise<RampView>) => {
    if (running || starting) return;
    setStarting(true);
    setPostError(null);
    try {
      const v = await call();
      setView(v);
      setPressed(Date.now());
      setEnded(null);
    } catch (err) {
      if (err instanceof ApiError && err.status === 503) setPostError(`The pair is not reachable from this box: ${err.message}`);
      else if (err instanceof ApiError && err.status === 409) {
        setPostError("A load run is already going.");
        void read();
      } else if (err instanceof ApiError && err.status === 401) setPostError("Log in to start a load run.");
      else setPostError(err instanceof Error ? `The load run did not start: ${err.message}` : "The load run did not start.");
    } finally {
      setStarting(false);
    }
  };

  return {
    view,
    pressed,
    ended,
    running,
    starting,
    postError,
    start: (streams: number) => go(() => api.loadRunStart(streams, LOAD_SECONDS)),
    ramp: (perStage: number) => go(() => api.loadRampStart(RAMP, perStage)),
  };
}

function Picker<T extends number>({ label, values, value, onPick, disabled, unit = "" }: { label: string; values: readonly T[]; value: T; onPick: (v: T) => void; disabled: boolean; unit?: string }) {
  return (
    <div className={b.pick} role="radiogroup" aria-label={label}>
      {[...values].sort((x, y) => x - y).map((v) => (
        <button key={v} type="button" role="radio" aria-checked={value === v} className={`${b.opt} ${value === v ? b.on : ""}`} onClick={() => onPick(v)} disabled={disabled}>
          {v}
          {unit}
        </button>
      ))}
    </div>
  );
}

export function LoadControl({ load }: { load: ReturnType<typeof useLoadRun> }) {
  const [streams, setStreams] = useState<number>(LOAD_STREAMS[0]);
  const [perStage, setPerStage] = useState<number>(RAMP_STAGE_SECONDS[0]);
  const { view, running, starting, postError } = load;
  const busy = running || starting;
  const problem = loadProblem(view, postError);
  const done = view && view.state === "done";
  const isRamp = !!view?.ramp;
  return (
    <div className={b.burst}>
      <div className={b.row}>
        <button type="button" className={b.go} onClick={() => void load.start(streams)} disabled={busy} aria-busy={busy}>
          {running && !isRamp ? "Asking..." : starting ? "Starting..." : loadButtonText(streams)}
        </button>
        <Picker label="How many people ask at once" values={LOAD_STREAMS} value={streams as (typeof LOAD_STREAMS)[number]} onPick={setStreams} disabled={busy} />
      </div>
      <div className={b.row}>
        <button type="button" className={b.go} onClick={() => void load.ramp(perStage)} disabled={busy} aria-busy={busy}>
          {running && isRamp ? "Ramping..." : `Ramp 1 to ${RAMP[RAMP.length - 1]}`}
        </button>
        <Picker label="Seconds per ramp stage" values={RAMP_STAGE_SECONDS} value={perStage as (typeof RAMP_STAGE_SECONDS)[number]} onPick={setPerStage} disabled={busy} unit=" s" />
      </div>
      <p className={b.note} aria-live="polite">
        {running && view
          ? isRamp
            ? "Streams are added stage by stage and none stops, so the load only climbs. Watch the watts."
            : `${askingText(view.streams ?? streams)} the 122B pair at once. Watch the watts.`
          : problem ?? (done && !isRamp ? loadLine(view) : done ? null : "Many people asking the 122B pair at once, with short questions about a desk.")}
      </p>
      {done && isRamp && view.stages.length ? (
        <table className={b.note}>
          <thead>
            <tr>
              <th scope="col">streams</th>
              <th scope="col">tokens a second, total</th>
              <th scope="col">watts</th>
              <th scope="col">J per token</th>
            </tr>
          </thead>
          <tbody>
            {rampRows(view).map((r) => (
              <tr key={r.streams}>
                <td>{r.streams}</td>
                <td>{r.tokS}</td>
                <td>{r.watts}</td>
                <td>{r.jpt}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </div>
  );
}
