"use client";

import { useEffect, useRef, useState } from "react";
import ui from "@/components/ui.module.css";
import { DOTS, DOT_MS, SETTLE_MS, fitCalibration, qualityWords, type Calibration, type Sample } from "@/lib/gaze";
import { gaze } from "@/lib/gazeTracker";
import g from "./guide.module.css";

/**
 * Five dots, the middle and the four corners, each for 1.2 s with a ring that fills. The person looks at each; the
 * iris-and-head numbers from those looks are fitted to the dots' places. Then the result is said in words: good, rough or
 * poor, and what Atlas will use it for. Nothing from the camera is kept beyond the fit.
 */
export function Calibrate({ onDone }: { onDone: () => void }) {
  const [step, setStep] = useState(-1); // -1: the explanation; 0..4: a dot; 5: the result
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<Calibration | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const samples = useRef<Sample[]>([]);
  const began = useRef(0);
  const stepRef = useRef(step);
  stepRef.current = step;

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onDone();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onDone]);

  // collect the looks while a dot is up
  useEffect(
    () =>
      gaze.onFeatures((f) => {
        const i = stepRef.current;
        if (!f || i < 0 || i >= DOTS.length || performance.now() - began.current < SETTLE_MS) return;
        samples.current.push({ f, x: DOTS[i]!.x, y: DOTS[i]!.y });
      }),
    [],
  );

  useEffect(() => {
    if (step < 0 || step >= DOTS.length) return;
    began.current = performance.now();
    let raf = 0;
    const frame = () => {
      const p = (performance.now() - began.current) / DOT_MS;
      setProgress(Math.min(1, p));
      if (p < 1) raf = window.requestAnimationFrame(frame);
      else if (step + 1 < DOTS.length) setStep(step + 1);
      else {
        const fit = fitCalibration(samples.current, window.innerWidth / Math.max(window.innerHeight, 1));
        samples.current = []; // the looks are not kept
        if (fit) gaze.setCalibration(fit);
        else setFailed(gaze.get().face === "many" ? "I saw more than one face, so I could not tell whose eyes to follow." : "I could not see your face well enough. Face the camera, with light on your face, and try again.");
        setResult(fit);
        setStep(DOTS.length);
      }
    };
    raf = window.requestAnimationFrame(frame);
    return () => window.cancelAnimationFrame(raf);
  }, [step]);

  const begin = () => {
    samples.current = [];
    setFailed(null);
    setResult(null);
    setStep(0);
  };
  const dot = step >= 0 && step < DOTS.length ? DOTS[step]! : null;
  const R = 24;
  const C = 2 * Math.PI * R;

  return (
    <div className={g.veil} role="dialog" aria-modal="true" aria-label="Eye tracking set-up" data-gaze-calibrate={step}>
      {dot ? (
        <svg className={g.calDot} style={{ left: `${dot.x * 100}%`, top: `${dot.y * 100}%` }} viewBox="0 0 56 56" aria-hidden="true">
          <circle className={g.calTrack} cx="28" cy="28" r={R} />
          <circle className={g.calProgress} cx="28" cy="28" r={R} strokeDasharray={C} strokeDashoffset={C * (1 - progress)} />
          <circle className={g.calCore} cx="28" cy="28" r="8" />
        </svg>
      ) : null}
      <div className={g.veilText} aria-live="polite">
        {step < 0 ? (
          <>
            <h2>Look at five dots</h2>
            <p>A red dot will show in the middle and in each corner, for about a second each. Look at it until its ring is full. Keep your head roughly where it is now.</p>
            <div className={`${ui.row} ${ui.wrap}`} style={{ justifyContent: "center" }}>
              <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} onClick={begin}>
                Start
              </button>
              <button type="button" className={ui.btn} onClick={onDone}>
                Not now
              </button>
            </div>
          </>
        ) : dot ? (
          step === 0 ? null : (
            <p>
              Dot {step + 1} of {DOTS.length}
            </p>
          )
        ) : (
          <>
            <h2>{result ? (result.quality === "good" ? "Good" : result.quality === "rough" ? "Rough" : "Not good enough") : "That did not work"}</h2>
            <p>{failed ?? qualityWords(result)}</p>
            <div className={`${ui.row} ${ui.wrap}`} style={{ justifyContent: "center" }}>
              <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} onClick={onDone}>
                Done
              </button>
              <button type="button" className={ui.btn} onClick={begin}>
                Recalibrate
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
