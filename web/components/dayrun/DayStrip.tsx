"use client";

import { useRef, type CSSProperties, type PointerEvent } from "react";
import { KIND, OTHER_KIND } from "@/components/calendar/DayBar";
import type { SimResult } from "@/lib/api";
import { ms } from "@/lib/dayrun";
import { clockAt, t12 } from "@/lib/plan";
import pl from "./plan.module.css";

/**
 * The plan as one bar, in the Calendar screen's style: a pastel stretch per block (your calendar's solid, the plan's
 * own paler), a marker for the moment being looked at, and above it one thin line for energy through the day.
 * Tap or drag along the bar to move the marker; the map shows where you are then.
 */
export function DayStrip({ result, t, selected, onSeek }: { result: SimResult; t: number | null; selected: string | null; onSeek: (t: number) => void }) {
  const bar = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const t0 = ms(result.start);
  const t1 = ms(result.end);
  const length = Math.max(1, t1 - t0);
  const pct = (x: number) => (Math.min(Math.max(x - t0, 0), length) / length) * 100;
  const kinds = [...new Set(result.blocks.map((b) => b.kind))];

  const seek = (e: PointerEvent<HTMLDivElement>) => {
    const box = bar.current?.getBoundingClientRect();
    if (!box || box.width <= 0) return;
    const f = Math.min(1, Math.max(0, (e.clientX - box.left) / box.width));
    onSeek(t0 + f * length);
  };

  // energy, 0..100, as one line across the same span
  const ticks = result.timeline;
  const line = ticks.map((k, i) => `${i ? "L" : "M"}${pct(ms(k.t)).toFixed(2)} ${(100 - k.energy).toFixed(1)}`).join("");
  const at = t === null ? null : pct(t);

  return (
    <section className={pl.strip} aria-label="The day at a glance" data-guide="plan:strip">
      <svg className={pl.energy} viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
        <path d={line} />
      </svg>
      <div
        ref={bar}
        className={pl.bar}
        role="slider"
        tabIndex={0}
        aria-label="Time of day"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Math.round(at ?? 0)}
        aria-valuetext={t === null ? "not set" : clockAt(t, result.start)}
        onPointerDown={(e) => {
          dragging.current = true;
          e.currentTarget.setPointerCapture?.(e.pointerId);
          seek(e);
        }}
        onPointerMove={(e) => dragging.current && seek(e)}
        onPointerUp={() => (dragging.current = false)}
        onPointerCancel={() => (dragging.current = false)}
        onKeyDown={(e) => {
          const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
          if (!step) return;
          e.preventDefault();
          onSeek(Math.min(t1, Math.max(t0, (t ?? t0) + step * 15 * 60_000)));
        }}
      >
        {result.blocks.map((b) => {
          const k = KIND[b.kind] ?? OTHER_KIND;
          const left = pct(ms(b.start));
          return (
            <span
              key={b.id + b.start}
              className={`${pl.seg} ${b.fixed ? "" : pl.segMove} ${selected === b.id + b.start ? pl.segOn : ""}`}
              style={{ left: `${left}%`, width: `${Math.max(0.6, pct(ms(b.end)) - left)}%`, "--c": k.color } as CSSProperties}
            />
          );
        })}
        {at !== null && t !== null ? (
          <span className={`${pl.now} ${at < 12 ? pl.nowStart : at > 88 ? pl.nowEnd : ""}`} style={{ left: `${at}%` }}>
            <i>{clockAt(t, result.start)}</i>
          </span>
        ) : null}
      </div>
      <div className={pl.ends}>
        <span>{t12(result.start)}</span>
        <span>{t12(result.end)}</span>
      </div>
      <ul className={pl.key}>
        {kinds.map((k) => {
          const x = KIND[k] ?? OTHER_KIND;
          return (
            <li key={k}>
              <i style={{ background: x.color }} />
              {x.label}
            </li>
          );
        })}
        <li>
          <em />
          Energy
        </li>
      </ul>
    </section>
  );
}
