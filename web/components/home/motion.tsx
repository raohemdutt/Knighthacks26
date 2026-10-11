"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import css from "./home.module.css";

/** Marks its element `data-in` once it scrolls into view, so the stylesheet can reveal it. Fires once. */
export function Reveal({ children, as: Tag = "div", className, index = 0 }: { children: ReactNode; as?: "div" | "section" | "li" | "article"; className?: string; index?: number }) {
  const ref = useRef<HTMLElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (!("IntersectionObserver" in window)) {
      el.dataset.in = "true";
      return;
    }
    const watcher = new IntersectionObserver(
      (entries) => {
        for (const e of entries)
          if (e.isIntersecting) {
            el.dataset.in = "true";
            watcher.disconnect();
          }
      },
      { rootMargin: "0px 0px -12% 0px", threshold: 0.1 },
    );
    watcher.observe(el);
    return () => watcher.disconnect();
  }, []);
  // The element type is decided by the caller; the ref is typed loosely for that reason.
  const T = Tag as "div";
  return (
    <T ref={ref as React.RefObject<HTMLDivElement>} className={`${css.reveal} ${className ?? ""}`} style={{ "--i": index } as React.CSSProperties}>
      {children}
    </T>
  );
}

/** A number that counts up from zero the first time it is seen. `decimals` keeps that many places; `suffix` sits after it. */
export function CountUp({ to, decimals = 0, suffix = "", duration = 1400 }: { to: number; decimals?: number; suffix?: string; duration?: number }) {
  const ref = useRef<HTMLSpanElement>(null);
  const [value, setValue] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const calm = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
    let frame = 0;
    const run = () => {
      if (calm) {
        setValue(to);
        return;
      }
      const t0 = performance.now();
      const tick = (now: number) => {
        const k = Math.min(1, (now - t0) / duration);
        const eased = 1 - Math.pow(1 - k, 4);
        setValue(to * eased);
        if (k < 1) frame = requestAnimationFrame(tick);
      };
      frame = requestAnimationFrame(tick);
    };
    if (!("IntersectionObserver" in window)) {
      run();
      return;
    }
    const watcher = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          run();
          watcher.disconnect();
        }
      },
      { threshold: 0.4 },
    );
    watcher.observe(el);
    return () => {
      watcher.disconnect();
      cancelAnimationFrame(frame);
    };
  }, [to, duration]);
  return (
    <span ref={ref}>
      {value.toFixed(decimals)}
      {suffix}
    </span>
  );
}
