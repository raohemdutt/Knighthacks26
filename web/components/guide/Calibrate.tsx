"use client";

import { useEffect, useRef, useState } from "react";
import ui from "@/components/ui.module.css";
import { DOTS_5, DOTS_9, DOTS_CHECK, DOT_MS, SETTLE_MS, fitCalibration, qualityWords, validate, type Calibration, type Sample } from "@/lib/gaze";
import { gaze } from "@/lib/gazeTracker";
import g from "./guide.module.css";

/**
 * Calibration, then a check of it.
 *
 *  1. Nine dots (three by three; "Quick" uses five), each for 1.7 s with a target that shrinks. The first half second of
 *     each is the eyes getting there and is not used; of the rest, looks far from that dot's median are dropped.
 *  2. The fit (lib/gaze.ts): ridge regression per axis, a straight and a second-order model, the one that predicts
 *     left-out dots better.
 *  3. Four fresh dots the fit never saw. The mean miss on those, in pixels and as a share of the screen, is shown as it
 *     is and decides what gaze is used for: a grid whose cells are at least twice that miss, or nothing.
 *
 * The looks are thrown away as soon as the numbers are worked out.
 */
type Stage = "intro" | "fit" | "check" | "done";

export function Calibrate({ onDone }: { onDone: () => void }) {
  const [stage, setStage] = useState<Stage>("intro");
  const [dots, setDots] = useState(DOTS_9);
  const [step, setStep] = useState(0);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState<Calibration | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  const samples = useRef<Sample[]>([]);
  const fitted = useRef<Calibration | null>(null);
  const began = useRef(0);
  const now = useRef<{ stage: Stage; dot: { x: number; y: number } | null }>({ stage, dot: null });
  const shown = stage === "fit" ? dots : stage === "check" ? DOTS_CHECK : [];
  const dot = shown[step] ?? null;
  now.current = { stage, dot };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onDone();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onDone]);

  // collect the looks while a dot is up
  useEffect(
    () =>
      gaze.onFeatures((f) => {
        const d = now.current.dot;
        if (!f || !d || performance.now() - began.current < SETTLE_MS) return;
        samples.current.push({ f, x: d.x, y: d.y });
      }),
    [],
  );

  useEffect(() => {
    if (!dot) return;
    began.current = performance.now();
    let raf = 0;
    const frame = () => {
      const p = (performance.now() - began.current) / DOT_MS;
      setProgress(Math.min(1, p));
      if (p < 1) return void (raf = window.requestAnimationFrame(frame));
      if (step + 1 < shown.length) return setStep(step + 1);
      const looks = samples.current;
      samples.current = []; // the looks are not kept
      const giveUp = () => {
        setFailed(gaze.get().face === "many" ? "I saw more than one face, so I could not tell whose eyes to follow." : "I could not see your eyes well enough. Face the camera, with light on your face and no glare on glasses, and try again.");
        setResult(null);
        setStage("done");
      };
      if (stage === "fit") {
        fitted.current = fitCalibration(looks, window.innerWidth / Math.max(window.innerHeight, 1), Date.now());
        if (!fitted.current) return giveUp();
        setStep(0);
        setStage("check");
      } else {
        if (!fitted.current || looks.length < 8) return giveUp();
        const checked = validate(fitted.current, looks, window.innerWidth, window.innerHeight);
        gaze.setCalibration(checked);
        setResult(checked);
        setStage("done");
      }
    };
    raf = window.requestAnimationFrame(frame);
    return () => window.cancelAnimationFrame(raf);
    // one run per dot
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stage, step]);

  const begin = (which: typeof DOTS_9) => {
    samples.current = [];
    fitted.current = null;
    setFailed(null);
    setResult(null);
    setDots(which);
    setStep(0);
    setStage("fit");
  };
  const R = 24;
  const C = 2 * Math.PI * R;

  return (
    <div className={g.veil} role="dialog" aria-modal="true" aria-label="Eye tracking set-up" data-gaze-calibrate={stage} data-gaze-quality={result?.quality}>
      {dot ? (
        <svg className={g.calDot} style={{ left: `${dot.x * 100}%`, top: `${dot.y * 100}%` }} viewBox="0 0 56 56" aria-hidden="true">
          <circle className={g.calTrack} cx="28" cy="28" r={R} />
          <circle className={g.calProgress} cx="28" cy="28" r={R} strokeDasharray={C} strokeDashoffset={C * (1 - progress)} />
          <circle className={g.calCore} cx="28" cy="28" r={15 - 10 * progress} />
        </svg>
      ) : null}
      <div className={g.veilText} aria-live="polite">
        {stage === "intro" ? (
          <>
            <h2>Look at the red dots</h2>
            <p>
              A dot shows in nine places, then in four more to check the result. Look at each until its ring is full, and keep your head roughly where it is now. It takes about 25
              seconds. Sit as you normally do, with light on your face.
            </p>
            <div className={`${ui.row} ${ui.wrap}`} style={{ justifyContent: "center" }}>
              <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} onClick={() => begin(DOTS_9)}>
                Start
              </button>
              <button type="button" className={ui.btn} onClick={() => begin(DOTS_5)}>
                Quick (5 dots)
              </button>
              <button type="button" className={ui.btn} onClick={onDone}>
                Not now
              </button>
            </div>
          </>
        ) : dot ? (
          dot.x === 0.5 && dot.y === 0.5 ? null : (
            <p>{stage === "check" ? `Checking: dot ${step + 1} of ${shown.length}` : `Dot ${step + 1} of ${shown.length}`}</p>
          )
        ) : (
          <>
            <h2>{result ? (result.quality === "good" ? "Good" : result.quality === "rough" ? "Rough" : "Not good enough to use") : "That did not work"}</h2>
            <p data-gaze-result>{failed ?? qualityWords(result)}</p>
            <div className={`${ui.row} ${ui.wrap}`} style={{ justifyContent: "center" }}>
              <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} onClick={onDone}>
                Done
              </button>
              <button type="button" className={ui.btn} onClick={() => setStage("intro")}>
                Recalibrate
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
