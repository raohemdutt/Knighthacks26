"use client";

import { useEffect, useState, useSyncExternalStore } from "react";
import ui from "@/components/ui.module.css";
import v from "@/components/voice/voice.module.css";
import { qualityWords } from "@/lib/gaze";
import { gaze, type GazeState } from "@/lib/gazeTracker";
import { calibrateGaze } from "@/lib/guideBus";

const SERVER: GazeState = { status: "off", wanted: false, calibration: null, face: "none", drift: null, detail: null };

/** The one place eye tracking is switched on (Settings, under Voice): opt-in, with what it does and does not do. */
export function GazeSwitch() {
  const state = useSyncExternalStore(gaze.subscribe, gaze.get, () => SERVER);
  const [blocked, setBlocked] = useState<string | null>(null);
  useEffect(() => {
    gaze.boot();
    setBlocked(gaze.blocked());
  }, []);

  const on = state.wanted && !blocked;
  const busy = state.status === "starting";
  const toggle = () => {
    if (on) return gaze.disable();
    void gaze.enable().then(() => {
      if (gaze.get().status === "on" && !gaze.get().calibration) calibrateGaze();
    });
  };
  const line =
    blocked ??
    (state.status === "starting"
      ? "Starting the camera…"
      : state.status === "on"
        ? state.face === "many"
          ? "I see more than one face, so I am not following anyone's eyes right now."
          : state.face === "none"
            ? "The camera is on, but I do not see your face. The pointer starts from the middle until I do."
            : state.calibration && state.drift
              ? `${state.drift === "moved" ? "You have moved since this was calibrated" : "This calibration is more than 20 minutes old"}, so it may be off. Recalibrate to be sure. ${qualityWords(state.calibration)}`
              : qualityWords(state.calibration)
        : state.detail);

  return (
    <section className={ui.section} aria-label="Look guide">
      <h3>Look guide</h3>
      <button type="button" className={v.switchRow} role="switch" aria-checked={on} disabled={blocked !== null || busy} onClick={toggle} data-guide="gaze:switch">
        <span className={v.switchText}>
          <span className={v.switchLabel}>Use my eyes to guide me</span>
          <span className={v.switchHint}>The camera picture stays in this browser. Nothing is stored or sent.</span>
        </span>
        <span className={`${v.knob} ${on ? v.knobOn : ""}`} aria-hidden="true" />
      </button>
      {line ? (
        <p className={`${ui.muted} ${ui.small}`} role="status" data-gaze-status={blocked ? "blocked" : state.status}>
          {line}
        </p>
      ) : null}
      {state.status === "on" ? (
        <button type="button" className={ui.btn} onClick={calibrateGaze}>
          {state.calibration ? "Recalibrate" : "Calibrate"}
        </button>
      ) : null}
      <p className={`${ui.muted} ${ui.small}`}>
        Ask “where is…?” or “where do I find…?” anywhere and a red pointer shows you. With this on, it starts from where you are looking and notices when your eyes get there. How
        accurate it is on your screen is measured when you calibrate and shown here; on a laptop camera expect a few centimetres, so it tells which part of the screen and which
        marked thing you are looking at, not the exact spot.
      </p>
    </section>
  );
}
