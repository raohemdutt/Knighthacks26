"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import { useTheme } from "@/lib/theme";
import type { DeviceScene } from "./deviceScene";
import css from "./home.module.css";

/**
 * The stage at the top of Home: a tall section whose sticky viewport holds the device. Scrolling through the
 * section pulls the device apart (deviceScene.ts owns the model and its callouts); the slider under it does the
 * same by scrolling the page, so the two never disagree. The hero copy is rendered by the caller and fades out as
 * the pull-apart begins (the stylesheet reads --t on the section).
 */
export function DeviceStage({ children }: { children: ReactNode }) {
  const section = useRef<HTMLElement>(null);
  const host = useRef<HTMLDivElement>(null);
  const overlay = useRef<HTMLDivElement>(null);
  const scene = useRef<DeviceScene | null>(null);
  const [t, setT] = useState(0);
  const [failed, setFailed] = useState<string | null>(null);
  const [theme] = useTheme();

  // How far through the section the page has scrolled: 0 at the top, 1 when the sticky viewport is about to leave.
  useEffect(() => {
    const el = section.current;
    if (!el) return;
    let frame = 0;
    const measure = () => {
      frame = 0;
      const r = el.getBoundingClientRect();
      const top = parseFloat(getComputedStyle(el).getPropertyValue("--stage-top")) || 0;
      const travel = r.height - (window.innerHeight - top);
      const k = travel > 0 ? Math.min(1, Math.max(0, (top - r.top) / travel)) : 0;
      setT((old) => (Math.abs(old - k) < 0.002 ? old : k));
      el.style.setProperty("--t", k.toFixed(3));
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
            classes: { callout: css.callout ?? "", calloutOn: css.calloutOn ?? "", line: css.line ?? "" },
          });
        } catch (err) {
          setFailed(err instanceof Error ? err.message : String(err));
          return;
        }
        const el = section.current;
        if (el) {
          const r = el.getBoundingClientRect();
          const top = parseFloat(getComputedStyle(el).getPropertyValue("--stage-top")) || 0;
          const travel = r.height - (window.innerHeight - top);
          scene.current.setExplode(travel > 0 ? Math.min(1, Math.max(0, (top - r.top) / travel)) : 0);
        }
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
    const top = parseFloat(getComputedStyle(el).getPropertyValue("--stage-top")) || 0;
    const travel = el.offsetHeight - (window.innerHeight - top);
    const start = el.getBoundingClientRect().top + window.scrollY - top;
    window.scrollTo({ top: start + v * travel, behavior: "auto" });
  };

  return (
    <section ref={section} className={css.stage} aria-label="The Atlas device">
      <div className={css.sticky}>
        <div ref={host} className={css.host} onPointerMove={onMove} onPointerLeave={onLeave} aria-hidden="true">
          {failed ? <p className={css.fallback}>The 3D device needs WebGL, which this browser could not start. Everything else on this page still works.</p> : null}
        </div>
        <div ref={overlay} className={css.overlay} aria-hidden="true" />
        {children}
        <p className={css.hint} aria-hidden="true">
          Scroll to pull it apart
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M6 9l6 6 6-6" />
          </svg>
        </p>
        <div className={css.scrub} style={{ "--t": t } as React.CSSProperties}>
          <span className={css.scrubLabel}>Sealed</span>
          <input className={css.range} type="range" min={0} max={1000} value={Math.round(t * 1000)} aria-label="Pull the device apart" onChange={(e) => scrub(Number(e.target.value) / 1000)} />
          <span className={`${css.scrubLabel} ${css.scrubLabelEnd}`}>Apart</span>
        </div>
      </div>
    </section>
  );
}
