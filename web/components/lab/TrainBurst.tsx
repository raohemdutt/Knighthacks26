"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, api } from "@/lib/api";
import { BURST_CHOICES, type BurstPod, type BurstView, burstLine, burstProblem, buttonText } from "@/lib/trainBurst";
import b from "./trainBurst.module.css";

/**
 * The "Train for a minute" button beside the live power graph: POST /api/learning/lab/train-burst starts a real LoRA
 * burst on the pods (both by default), GET is polled every 2 s while it runs. The graph shades the burst's window from
 * `window` (browser clock: when the press was answered until the burst was seen to end) and keeps the last one.
 */

export interface BurstWindow {
  start: number;
  end: number | null;
  pods: BurstPod[];
}

const POLL_MS = 2000;

export function useTrainBurst(enabled = true) {
  const [view, setView] = useState<BurstView | null>(null);
  const [postError, setPostError] = useState<string | null>(null);
  const [window_, setWindow] = useState<BurstWindow | null>(null);
  const [starting, setStarting] = useState(false);
  const running = view?.state === "running";
  const win = useRef<BurstWindow | null>(null);
  win.current = window_;

  const read = useCallback(async () => {
    try {
      const v = await api.trainBurst();
      setView(v);
      const w = win.current;
      if (v.state === "running" && !w) setWindow({ start: v.started_ms ?? Date.now(), end: null, pods: v.pods });
      else if (v.state !== "running" && w && w.end === null) setWindow({ ...w, end: Date.now() });
    } catch {
      /* the next poll tries again; the button keeps its last known state */
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

  const start = async (pods: BurstPod[]) => {
    if (running || starting) return;
    setStarting(true);
    setPostError(null);
    try {
      const v = await api.trainBurstStart(pods, 60);
      setView(v);
      setWindow({ start: Date.now(), end: null, pods: v.pods });
    } catch (err) {
      if (err instanceof ApiError && err.status === 503) setPostError(`Training is not available from this box: ${err.message}`);
      else if (err instanceof ApiError && err.status === 409) {
        setPostError("A training burst is already running.");
        void read();
      } else if (err instanceof ApiError && err.status === 401) setPostError("Log in to start a training burst.");
      else setPostError(err instanceof Error ? `The burst did not start: ${err.message}` : "The burst did not start.");
    } finally {
      setStarting(false);
    }
  };

  return { view, window: window_, running, starting, postError, start };
}

export function TrainBurstControl({ burst }: { burst: ReturnType<typeof useTrainBurst> }) {
  const [choice, setChoice] = useState(BURST_CHOICES[0]!.id);
  const pods = BURST_CHOICES.find((c) => c.id === choice)!.pods;
  const { view, running, starting, postError, start } = burst;
  const busy = running || starting;
  const problem = burstProblem(view, postError);
  const finished = view && view.state === "done" ? burstLine(view) : null;
  return (
    <div className={b.burst}>
      <div className={b.row}>
        <button type="button" className={b.go} onClick={() => void start(pods)} disabled={busy} aria-busy={busy}>
          {running ? "Training..." : starting ? "Starting..." : buttonText(pods)}
        </button>
        <div className={b.pick} role="radiogroup" aria-label="Which pods train">
          {BURST_CHOICES.map((c) => (
            <button
              key={c.id}
              type="button"
              role="radio"
              aria-checked={choice === c.id}
              className={`${b.opt} ${choice === c.id ? b.on : ""}`}
              onClick={() => setChoice(c.id)}
              disabled={busy}
            >
              {c.label}
            </button>
          ))}
        </div>
      </div>
      <p className={b.note} aria-live="polite">
        {running
          ? "A real LoRA burst on the 4B student, beside the 122B pair. The amber stretch on the graph is the training."
          : problem ?? finished ?? "About a minute of real LoRA training on the pods, beside the 122B pair that keeps answering."}
      </p>
    </div>
  );
}
