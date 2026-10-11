"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { useTheme } from "@/lib/theme";
import type { DeviceScene } from "./deviceScene";
import css from "./home.module.css";

/**
 * The stage at the top of Home: a tall section whose sticky viewport holds the device. Scrolling through the
 * section pulls the device apart (deviceScene.ts owns the model and its callouts); the slider under it does the
 * same by scrolling the page, so the two never disagree. The hero copy is rendered by the caller inside the stage's
 * hero box, which the scene frames the sealed box clear of; it fades out as the pull-apart begins (the stylesheet
 * reads --t on the section).
 */
export function DeviceStage({ children }: { children: ReactNode }) {
  const section = useRef<HTMLElement>(null);
  const host = useRef<HTMLDivElement>(null);
  const overlay = useRef<HTMLDivElement>(null);
  const hero = useRef<HTMLDivElement>(null);
  const footer = useRef<HTMLDivElement>(null);
  const scene = useRef<DeviceScene | null>(null);
  const [t, setT] = useState(0);
  const [step, setStep] = useState<{ index: number; total: number; name: string }>({ index: -1, total: 0, name: "" });
  const [failed, setFailed] = useState<string | null>(null);
  const [theme] = useTheme();

  // How far through the section the page has scrolled: 0 at the top, 1 when the sticky viewport is about to leave.
  useEffect(() => {
    const el = section.current;
    if (!el) return;
    let frame = 0;
    const measure = () => {
      frame = 0;
      const k = progress(el);
      setT((old) => (Math.abs(old - k) < 0.002 ? old : k));
      el.style.setProperty("--t", k.toFixed(3));
      // Once the copy has faded it must not catch clicks meant for the stage.
      el.dataset.open = k > 0.12 ? "true" : "false";
      scene.current?.setExplode(k);
    };
    const onScroll = () => {
      if (!frame) frame = requestAnimationFrame(measure);
    };
    measure();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      if (frame) cancelAnimationFrame(frame);
    };
  }, []);

  // The scene itself, loaded on demand so no other screen pays for three.js.
  useEffect(() => {
    const h = host.current;
    const o = overlay.current;
    if (!h || !o) return;
    let gone = false;
    let watcher: IntersectionObserver | null = null;
    const calm = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
    import("./deviceScene")
      .then((m) => {
        if (gone) return;
        try {
          scene.current = new m.DeviceScene(h, o, {
            calm,
            dark: document.documentElement.dataset.theme !== "light",
            classes: { callout: css.callout ?? "", calloutOn: css.calloutOn ?? "", calloutActive: css.calloutActive ?? "", line: css.line ?? "" },
            hero: hero.current,
            footer: footer.current,
            onStep: (index, total, name) => setStep({ index, total, name }),
          });
        } catch (err) {
          setFailed(err instanceof Error ? err.message : String(err));
          return;
        }
        if (section.current) scene.current.setExplode(progress(section.current));
        // Only draw while the stage is on screen.
        watcher = new IntersectionObserver((entries) => scene.current?.setActive(entries.some((e) => e.isIntersecting)), { threshold: 0 });
        watcher.observe(h);
      })
      .catch((err) => setFailed(err instanceof Error ? err.message : String(err)));
    return () => {
      gone = true;
      watcher?.disconnect();
      scene.current?.dispose();
      scene.current = null;
    };
  }, []);

  useEffect(() => scene.current?.setTheme(theme === "dark"), [theme]);

  // A pointer over the stage tilts the device towards it.
  const onMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (e.pointerType !== "mouse") return;
    const r = e.currentTarget.getBoundingClientRect();
    scene.current?.setPointer(((e.clientX - r.left) / r.width) * 2 - 1, ((e.clientY - r.top) / r.height) * 2 - 1);
  };
  const onLeave = () => scene.current?.setPointer(0, 0);

  // The slider scrolls the page to the matching spot, so the device, the copy and the scrollbar all agree.
  const scrub = (v: number) => {
    const el = section.current;
    if (!el) return;
    const start = el.getBoundingClientRect().top + window.scrollY - stageTop(el);
    window.scrollTo({ top: start + v * travelOf(el), behavior: "auto" });
  };

  return (
    <section ref={section} className={css.stage} aria-label="The Atlas device">
      <div className={css.sticky}>
        <div ref={host} className={css.host} onPointerMove={onMove} onPointerLeave={onLeave} aria-hidden="true">
          {failed ? <p className={css.fallback}>The 3D device needs WebGL, which this browser could not start. Everything else on this page still works.</p> : null}
        </div>
        <div ref={overlay} className={css.overlay} aria-hidden="true" />
        <div ref={hero} className={css.hero}>
          {children}
        </div>
        <p className={css.hint} aria-hidden="true">
          Scroll to open it
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M6 9l6 6 6-6" />
          </svg>
        </p>
        <div ref={footer} className={css.scrub} style={{ "--t": t } as React.CSSProperties}>
          <span className={css.step} aria-live="polite">
            {step.index < 0 ? (
              <span className={css.stepName}>Sealed</span>
            ) : (
              <>
                <span className={css.stepCount}>
                  {String(step.index + 1).padStart(2, "0")}
                  <span>/{String(step.total).padStart(2, "0")}</span>
                </span>
                <span key={step.name} className={css.stepName}>
                  {step.name}
                </span>
              </>
            )}
          </span>
          <input className={css.range} type="range" min={0} max={1000} value={Math.round(t * 1000)} aria-label="Pull the device apart" onChange={(e) => scrub(Number(e.target.value) / 1000)} />
          <span className={`${css.scrubLabel} ${css.scrubLabelEnd}`}>Apart</span>
        </div>
      </div>
    </section>
  );
}

function stageTop(el: HTMLElement): number {
  return parseFloat(getComputedStyle(el).getPropertyValue("--stage-top")) || 0;
}

/** How far the section scrolls while its sticky viewport is held: the section's height less the viewport's. */
function travelOf(el: HTMLElement): number {
  const sticky = el.firstElementChild as HTMLElement | null;
  return el.offsetHeight - (sticky?.offsetHeight ?? window.innerHeight - stageTop(el));
}

function progress(el: HTMLElement): number {
  const travel = travelOf(el);
  return travel > 0 ? Math.min(1, Math.max(0, (stageTop(el) - el.getBoundingClientRect().top) / travel)) : 0;
}
