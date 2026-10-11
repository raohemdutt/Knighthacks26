"use client";

import { useEffect, useRef, useState } from "react";
import type { StripDay } from "@/lib/api";
import { dateWords, dayNumber, weekday, weekdayLong } from "@/lib/plan";
import pl from "./plan.module.css";

/**
 * The days around today: two weeks behind, today, the next few. One tap opens a day: a past one against what
 * happened, today's plan, or a draft for a day ahead. A day the box has nothing for cannot be opened.
 * The small bar under each number is how the day went (accent: it worked; red: it needed fixing).
 */
export function DateStrip({ days, today, selected, onSelect }: { days: StripDay[]; today: string; selected: string; onSelect: (day: string) => void }) {
  const scroller = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<string | null>(null);

  // the selected day stays in view (and today is where the strip opens)
  useEffect(() => {
    const el = scroller.current?.querySelector<HTMLElement>(`[data-day="${selected}"]`);
    const box = scroller.current;
    if (el && box) box.scrollTo({ left: el.offsetLeft - box.clientWidth / 2 + el.clientWidth / 2, behavior: "auto" });
  }, [selected, days.length]);

  const shown = days.find((d) => d.day === (hover ?? selected));
  const note = !shown
    ? ""
    : shown.day === today
      ? `Today, ${dateWords(shown.day)}`
      : !shown.has_data
        ? `${weekdayLong(shown.day)}: ${shown.when === "past" ? "nothing recorded" : "nothing on it yet"}`
        : `${weekdayLong(shown.day)}: ${shown.label}`;
  const step = (dir: 1 | -1) => scroller.current?.scrollBy({ left: dir * 232, behavior: "smooth" });

  return (
    <nav className={pl.dates} aria-label="Days">
      <div className={pl.dateRow}>
        <button type="button" className={pl.dateNav} onClick={() => step(-1)} aria-label="Earlier days">
          ‹
        </button>
        <div ref={scroller} className={pl.dateScroll}>
          {days.map((d) => {
            const mark = d.when === "past" && d.has_data ? (d.won === false ? pl.dateOff : pl.dateGood) : "";
            return (
              <button
                key={d.day}
                type="button"
                data-day={d.day}
                className={`${pl.date} ${mark} ${d.day === selected ? pl.dateOn : ""}`}
                disabled={!d.has_data}
                aria-pressed={d.day === selected}
                aria-label={`${dateWords(d.day)}: ${d.day === today ? "today" : d.has_data ? d.label : "nothing to show"}`}
                onClick={() => onSelect(d.day)}
                onPointerEnter={() => setHover(d.day)}
                onPointerLeave={() => setHover(null)}
                onFocus={() => setHover(d.day)}
                onBlur={() => setHover(null)}
              >
                <span>{d.day === today ? "Today" : weekday(d.day)}</span>
                <b>{dayNumber(d.day)}</b>
                <i />
              </button>
            );
          })}
        </div>
        <button type="button" className={pl.dateNav} onClick={() => step(1)} aria-label="Later days">
          ›
        </button>
      </div>
      <p className={pl.dateNote}>{note}</p>
    </nav>
  );
}
