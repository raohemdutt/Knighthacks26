"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { dayTime } from "@/lib/format";
import { type LabRunSummary, type LabStory, isLive } from "@/lib/lab";
import { type LoopRun, type LoopStage, type LoopStory, loopStages, runRow, runState } from "@/lib/labLoop";
import { useTheme } from "@/lib/theme";
import s from "./loop.module.css";
import type { LoopScene } from "./loopScene";

/**
 * The learning loop in 3D (loopScene.ts), at the top of the Learning lab's "The model" tab, above the Training run view.
 * It is an addition: it replaces none of that view's cards. Each stage is lit from the events of the very run the
 * Training run view under it is showing (the same story object, passed in), and the strip of five stages under the
 * picture says the same thing in words. The strip is all there is when the browser has no WebGL.
 *
 * three.js and the scene are fetched after the page has painted, and drawing stops while the picture is off screen.
 */
const STATE_WORD: Record<LoopStage["state"], string> = { done: "done", now: "now", waiting: "not yet", failed: "failed", refused: "refused", none: "not reported" };

function Scene({ stages, runKey, forceReduced }: { stages: LoopStage[]; runKey: string; forceReduced: boolean }) {
  const host = useRef<HTMLDivElement>(null);
  const overlay = useRef<HTMLDivElement>(null);
  const scene = useRef<LoopScene | null>(null);
  const shown = useRef<string | null>(null);
  const latest = useRef(stages);
  latest.current = stages;
  const [failed, setFailed] = useState(false);
  const [theme] = useTheme();

  useEffect(() => {
    const h = host.current;
    const o = overlay.current;
    if (!h || !o) return;
    let gone = false;
    let watcher: IntersectionObserver | null = null;
    const calm = forceReduced || (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false);
    import("./loopScene")
      .then((m) => {
        if (gone) return;
        try {
          scene.current = new m.LoopScene(h, o, { calm, dark: document.documentElement.dataset.theme !== "light", tagClass: s.loopTag ?? "" });
        } catch {
          setFailed(true); // no WebGL: the strip below still tells the run
          return;
        }
        scene.current.setStages(latest.current, true);
        watcher = new IntersectionObserver((entries) => scene.current?.setActive(entries.some((e) => e.isIntersecting)), { threshold: 0 });
        watcher.observe(h);
      })
      .catch(() => setFailed(true));
    return () => {
      gone = true;
      watcher?.disconnect();
      scene.current?.dispose();
      scene.current = null;
    };
  }, [forceReduced]);

  // a different run lights its stages in order once; the same run, updated, just changes what is lit
  const sig = stages.map((x) => `${x.state}|${x.value ?? ""}`).join(",");
  useEffect(() => {
    scene.current?.setStages(latest.current, shown.current !== runKey);
    shown.current = runKey;
  }, [sig, runKey]);
  useEffect(() => scene.current?.setTheme(theme === "dark"), [theme]);

  return (
    <>
      {failed ? null : (
        <div className={s.loopStage} aria-hidden="true">
          <div ref={host} className={s.loopHost} />
          <div ref={overlay} className={s.loopHost} />
        </div>
      )}
      <ol className={s.strip} aria-label="The loop, stage by stage" data-guide="lab:stages">
        {stages.map((x) => (
          <li key={x.key} className={s.step} data-state={x.state}>
            <span className={s.stepTitle}>{x.title}</span>
            <span className={s.stepValue}>{x.value ?? STATE_WORD[x.state]}</span>
            {x.note ? <span className={s.stepNote}>{x.note}</span> : null}
            <span className={s.srOnly}>: {STATE_WORD[x.state]}</span>
          </li>
        ))}
      </ol>
    </>
  );
}

/**
 * `story` is the run the Training run view shows (null while it loads, or when the box has no run), `run` its row in
 * the run list, `empty` true when the box has recorded no run at all.
 */
export function LoopView({ story, run, empty, forceReduced }: { story: LabStory | null; run: LabRunSummary | null; empty: boolean; forceReduced: boolean }) {
  const [now, setNow] = useState<number | null>(null);
  useEffect(() => {
    setNow(Date.now());
    const id = window.setInterval(() => setNow(Date.now()), 30_000);
    return () => window.clearInterval(id);
  }, []);
  const loop = story as LoopStory | null;
  const stages = useMemo(() => loopStages(loop, dayTime), [loop]);
  const live = !!story && isLive(story);
  return (
    <section className={s.loop} aria-label="The learning loop" data-guide="lab:loop">
      <Scene stages={stages} runKey={story?.run_id ?? "none"} forceReduced={forceReduced} />
      <p className={s.loopLine} role="status" data-guide="lab:now">
        {empty ? (
          "Nothing has run yet. This is the path a run takes: out to the lab, through the test, and back."
        ) : live && loop ? (
          <b>{runState(loop, now, dayTime).line}</b>
        ) : run ? (
          <>
            The run shown below: <b>{dayTime(run.started) || "no time recorded"}</b>, {runRow(run as LoopRun).chip.toLowerCase()}.
          </>
        ) : (
          "Reading the run shown below."
        )}
      </p>
    </section>
  );
}
