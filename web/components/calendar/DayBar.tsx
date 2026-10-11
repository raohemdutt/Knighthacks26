import type { CSSProperties } from "react";
import type { DayNow, PlanMove } from "@/lib/api";
import { hoursMinutes, span as timeSpan, time } from "@/lib/format";
import s from "./calendar.module.css";

export const KIND: Record<string, { label: string; color: string }> = {
  study: { label: "Focus", color: "#9bbcf2" },
  eat: { label: "Meals", color: "#f3cf93" },
  sleep: { label: "Sleep", color: "#bdb9ec" },
  rest: { label: "Rest", color: "#bfe3d1" },
  event: { label: "Events", color: "#f2b0bb" },
  class: { label: "Class", color: "#f2b0bb" },
};
export const OTHER_KIND = { label: "Other", color: "#cfd6e6" };
const MINUTE = 60_000;
const ms = (iso: string) => Date.parse(iso);
const iso = (t: number) => new Date(t).toISOString();

/** One stretch of the bar: a block of the plan, or the free time between two blocks. */
type Part = { key: string; start: number; end: number; title: string; kind: string | null; lines: string[] };

function parts(day: PlanMove[]): Part[] {
  const out: Part[] = [];
  let cursor = ms(day[0]?.start ?? "");
  for (const m of day) {
    const start = ms(m.start);
    const end = ms(m.end);
    if (start - cursor >= 5 * MINUTE) out.push({ key: `free-${cursor}`, start: cursor, end: start, title: "Free time", kind: null, lines: [timeSpan(iso(cursor), iso(start)), hoursMinutes((start - cursor) / 3600_000)] });
    const where = m.place_name ? [m.place_name] : [];
    const who = m.with_friend ? [`with ${m.with_friend}`] : [];
    out.push({
      key: m.id,
      start,
      end,
      title: m.title,
      kind: m.kind,
      lines: [timeSpan(m.start, m.end), ...where, ...who, m.fixed ? "On your calendar" : "Suggested by Atlas"],
    });
    cursor = Math.max(cursor, end);
  }
  return out;
}

/** The whole day as one bar: a coloured stretch per block, and a marker for where you are right now. Hover or focus any part to see what it is. */
export function DayBar({ d }: { d: DayNow }) {
  const moves: PlanMove[] = [...(d.plan?.moves ?? [])].sort((a, b) => ms(a.start) - ms(b.start));
  const head = moves[0];
  if (!head) return null;
  const t0 = ms(head.start);
  const day = moves.filter((m) => ms(m.start) < t0 + 24 * 3600_000);
  const t1 = Math.max(...day.map((m) => ms(m.end)));
  const length = t1 - t0;
  const now = ms(d.now);
  const pct = (t: number) => (Math.min(Math.max(t - t0, 0), length) / length) * 100;
  const kinds = [...new Set(day.map((m) => m.kind))];
  const first = time(head.start);
  const last = time(iso(t1));
  const tipSide = (mid: number) => (mid < 22 ? s.tipStart : mid > 78 ? s.tipEnd : "");
  return (
    <section className={s.day} aria-label="Your day at a glance">
      <div className={s.bar}>
        {parts(day).map((p) => {
          const k = p.kind ? (KIND[p.kind] ?? OTHER_KIND) : null;
          const left = pct(p.start);
          const width = pct(p.end) - left;
          return (
            <span
              key={p.key}
              tabIndex={0}
              className={`${s.seg} ${k ? "" : s.segFree} ${p.lines.includes("Suggested by Atlas") ? s.segSuggested : ""}`}
              style={{ left: `${left}%`, width: `${width}%`, ...(k ? ({ "--c": k.color } as CSSProperties) : {}) }}
            >
              <span className={`${s.tip} ${tipSide(left + width / 2)}`} role="tooltip">
                <b>{p.title}</b>
                {p.lines.map((line) => (
                  <em key={line}>{line}</em>
                ))}
              </span>
            </span>
          );
        })}
        {now >= t0 && now <= t1 ? (
          <span className={s.now} style={{ left: `${pct(now)}%` }} tabIndex={0}>
            <i>Now</i>
            <span className={`${s.tip} ${tipSide(pct(now))}`} role="tooltip">
              <b>Now</b>
              <em>{time(d.now)}</em>
            </span>
          </span>
        ) : null}
      </div>
      <div className={s.ends}>
        <span>{first}</span>
        <span>{last}</span>
      </div>
      <ul className={s.key}>
        {kinds.map((k) => {
          const x = KIND[k] ?? OTHER_KIND;
          return (
            <li key={k}>
              <i style={{ background: x.color }} />
              {x.label}
            </li>
          );
        })}
      </ul>
    </section>
  );
}
