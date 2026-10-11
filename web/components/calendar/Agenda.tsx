"use client";

import { useMemo, useState, type CSSProperties, type ReactNode, type RefObject } from "react";
import type { DayNow, Item, PlacePack, PlanMove } from "@/lib/api";
import { dayLabel, hoursMinutes, relative, time } from "@/lib/format";
import { parseNearby, useGoing, type Nearby } from "@/lib/nearby";
import { KIND, OTHER_KIND } from "./DayBar";
import s from "./screen.module.css";

/**
 * Your schedule: what you have to do, in the order you will do it, day by day.
 *
 * Today and the morning after come from the day plan (your calendar's blocks, solid, and Atlas's suggestions,
 * dashed). Further days come from the calendar items the box synced. Events you said yes to under Near you are
 * woven in by time. Deadlines sit on the day they fall due. What already happened today folds away.
 */

const HORIZON_DAYS = 5;
const NIGHT_MIN = 180;
const ms = (iso: string | null | undefined) => (iso ? Date.parse(iso) : NaN);
const iso = (t: number) => new Date(t).toISOString();

type Entry = {
  key: string;
  start: number;
  end: number;
  title: string;
  kind: string | null; // a plan kind, "going", "due" or null
  place: string | null;
  fixed: boolean;
  tag: string | null;
  move?: PlanMove;
  nearby?: Nearby;
  item?: Item;
};

const startOfDay = (t: number) => {
  const d = new Date(t);
  d.setHours(0, 0, 0, 0);
  return d.getTime();
};

function heading(t: number, now: number): string {
  const label = dayLabel(iso(t), new Date(now));
  return label === "Today" ? "Today" : label;
}

/** The part of an item's summary after "until 6:06 PM · ", which is where it is. */
function whereOf(i: Item): string | null {
  const parts = (i.summary ?? "").split(" · ").map((p) => p.trim()).filter(Boolean);
  return (parts[0]?.toLowerCase().startsWith("until") ? parts[1] : parts[0]) ?? null;
}

export function Agenda({
  d,
  items,
  packs,
  reason,
  coachNote,
  closing,
  nextRef,
  nextRisk,
}: {
  d: DayNow;
  items: Item[];
  packs?: PlacePack[];
  /** Why Atlas put a block there: shown when the row is opened. */
  reason: (m: PlanMove) => string;
  /** The coach's note for a block that ended, if there is one. */
  coachNote: (blockId: string) => ReactNode;
  /** The evening review, at the end of today. */
  closing: ReactNode;
  nextRef: RefObject<HTMLLIElement | null>;
  nextRisk: { id: string; text: string; bad: boolean } | null;
}) {
  const now = ms(d.now);
  const { answers, answer } = useGoing();
  const [open, setOpen] = useState<string | null>(null);
  const today0 = startOfDay(now);
  const horizon = today0 + (HORIZON_DAYS + 1) * 86_400_000;

  const entries = useMemo(() => {
    const out: Entry[] = [];
    const seen = new Set<string>();
    const moves = [...(d.plan?.moves ?? [])].sort((a, b) => ms(a.start) - ms(b.start));
    const planEnd = moves.reduce((t, m) => Math.max(t, ms(m.end)), 0);
    for (const m of moves) {
      seen.add(m.id);
      const start = ms(m.start);
      const end = ms(m.end);
      if (m.kind === "sleep" && m.minutes < NIGHT_MIN && end <= now) continue; // a nap that is over is not worth a row
      out.push({
        key: `${m.id}:${m.start}`,
        start,
        end,
        title: m.with_friend ? `${m.title} with ${m.with_friend}` : m.title,
        kind: m.kind,
        place: m.place_name,
        fixed: m.fixed,
        tag: m.fixed ? "On your calendar" : "Suggested by Atlas",
        move: m,
      });
    }
    for (const i of items) {
      const t = ms(i.when);
      if (Number.isNaN(t) || seen.has(i.external_id)) continue;
      if (i.kind === "calendar_event" && !i.external_id.startsWith("nearby:") && t >= Math.max(planEnd, now) && t < horizon) {
        const end = ((): number => {
          const m = /^until\s+(\d{1,2}):(\d{2})\s*(AM|PM)/i.exec(i.summary ?? "");
          if (!m) return t + 3600_000;
          const x = new Date(t);
          x.setHours((Number(m[1]) % 12) + (m[3]!.toUpperCase() === "PM" ? 12 : 0), Number(m[2]), 0, 0);
          return x.getTime() > t ? x.getTime() : t + 3600_000;
        })();
        out.push({ key: i.external_id, start: t, end, title: i.title, kind: "event", place: whereOf(i), fixed: true, tag: "On your calendar", item: i });
      } else if (i.kind === "assignment" && t >= now && t < horizon + 2 * 86_400_000) {
        out.push({ key: i.external_id, start: t, end: t, title: i.title.replace(/\s*\[[^\]]+\]\s*$/, ""), kind: "due", place: null, fixed: true, tag: "Due", item: i });
      }
    }
    for (const n of parseNearby(items, packs)) {
      if (answers[n.id] !== "going" || n.end <= now) continue;
      out.push({ key: n.id, start: n.start, end: n.end, title: n.title, kind: "going", place: n.place, fixed: false, tag: "You said yes", nearby: n });
    }
    return out.sort((a, b) => a.start - b.start || a.end - b.end);
  }, [d.plan, items, packs, answers, now, horizon]);

  // what is over is folded into "Earlier today"; anything from a previous day is not shown at all
  const over = entries.filter((e) => e.kind !== "due" && e.end <= now);
  const earlier = over.filter((e) => startOfDay(e.start) >= today0);
  const rest = entries.filter((e) => !over.includes(e));
  const days: { label: string; day: number; entries: Entry[] }[] = [];
  for (const e of rest) {
    const day = startOfDay(Math.max(e.start, today0));
    const last = days[days.length - 1];
    if (last?.day === day) last.entries.push(e);
    else days.push({ label: heading(day + 3600_000, now), day, entries: [e] });
  }
  if (days.length === 0 || days[0]!.day !== today0) days.unshift({ label: "Today", day: today0, entries: [] });

  const row = (e: Entry): ReactNode => {
    const past = e.kind !== "due" && e.end <= now;
    const isNow = e.start <= now && now < e.end && e.kind !== "due";
    const k = e.kind === "going" ? KIND.event! : e.kind === "due" ? null : (KIND[e.kind ?? ""] ?? OTHER_KIND);
    const nextId = nextRisk && e.move?.id === nextRisk.id;
    const pct = isNow ? Math.min(100, Math.max(0, ((now - e.start) / Math.max(1, e.end - e.start)) * 100)) : 0;
    const mins = Math.round((e.end - e.start) / 60000);
    const meta = [e.place, mins > 0 && e.kind !== "due" ? hoursMinutes(mins / 60) : null].filter(Boolean).join(" · ");
    const canOpen = Boolean(e.move || e.nearby);
    const body = (
      <>
        <span className={s.rowTop}>
          <span className={s.rowTitle}>{e.title}</span>
          <span className={s.rowTags}>
            {isNow ? <span className={`${s.tagPill} ${s.tagNow}`}>Now</span> : null}
            {nextId ? <span className={`${s.tagPill} ${nextRisk!.bad ? s.tagWarn : s.tagNext}`}>{nextRisk!.bad ? nextRisk!.text : "Next"}</span> : null}
            {e.kind === "going" ? <span className={`${s.tagPill} ${s.tagGoing}`}>You&apos;re going</span> : null}
            {e.kind === "due" ? <span className={`${s.tagPill} ${s.tagDue}`}>Due {time(iso(e.start))}</span> : null}
            {e.kind !== "going" && e.kind !== "due" && !e.fixed ? <span className={s.tagPill}>Suggested</span> : null}
          </span>
        </span>
        {meta ? <span className={s.rowMeta}>{meta}</span> : null}
        {e.nearby?.host ? <span className={s.rowMeta}>Hosted by {e.nearby.host}</span> : null}
        {e.kind === "due" ? <span className={s.rowMeta}>{relative(iso(e.start), new Date(now))}</span> : null}
        {isNow ? (
          <span className={s.progress} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(pct)} aria-label="How far through">
            <span className={s.progressFill} style={{ width: `${pct}%` }} />
          </span>
        ) : null}
        {open === e.key ? (
          <span className={s.rowReason}>
            {e.move ? reason(e.move) : e.nearby ? `You said yes to this one. It is not on your calendar yet: it only shows here, on this phone.` : ""}
          </span>
        ) : null}
      </>
    );
    return (
      <li key={e.key} className={`${s.row} ${past ? s.rowPast : ""}`} ref={nextId ? nextRef : undefined}>
        <div className={s.rowTime}>
          {e.kind === "due" ? <b>Due</b> : <b>{time(iso(e.start))}</b>}
          {e.kind !== "due" && mins > 0 ? <span>{time(iso(e.end))}</span> : null}
        </div>
        <div className={s.rowRail} aria-hidden="true">
          <i className={`${s.rowDot} ${e.fixed ? s.rowDotFixed : ""} ${isNow ? s.rowDotNow : ""}`} style={k ? ({ "--c": k.color } as CSSProperties) : undefined} />
        </div>
        <div className={s.rowCell}>
          {canOpen ? (
            <button
              type="button"
              className={`${s.rowCard} ${e.fixed ? s.rowFixed : ""} ${e.kind === "going" ? s.rowGoing : ""} ${isNow ? s.rowNow : ""} ${nextId && nextRisk!.bad ? s.rowWarn : ""}`}
              style={k ? ({ "--c": k.color } as CSSProperties) : undefined}
              aria-expanded={open === e.key}
              onClick={() => setOpen(open === e.key ? null : e.key)}
            >
              {body}
            </button>
          ) : (
            <div className={`${s.rowCard} ${s.rowStatic} ${e.kind === "due" ? s.rowDue : ""} ${e.fixed ? s.rowFixed : ""}`} style={k ? ({ "--c": k.color } as CSSProperties) : undefined}>
              {body}
            </div>
          )}
          {open === e.key && e.nearby ? (
            <button type="button" className={s.undoBtn} onClick={() => answer(e.nearby!.id, null)}>
              Not going after all
            </button>
          ) : null}
          {e.move ? coachNote(e.move.id) : null}
        </div>
      </li>
    );
  };

  return (
    <div className={s.agenda}>
      {days.map((g, gi) => {
        const isToday = g.day === today0;
        const isLast = gi === days.length - 1;
        return (
          <section key={g.day} className={s.agendaDay}>
            <h3 className={s.agendaHead}>
              {g.label}
              {g.label === "Today" || g.label === "Tomorrow" ? <span>{new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(new Date(g.day + 3600_000))}</span> : null}
            </h3>
            {isToday && earlier.length > 0 ? (
              <details className={s.earlier}>
                <summary>Earlier today ({earlier.length})</summary>
                <ul className={s.rows}>{earlier.map(row)}</ul>
              </details>
            ) : null}
            {g.entries.length === 0 ? (
              <p className={s.nothing}>{isToday ? "Nothing else on today." : "Nothing planned."}</p>
            ) : (
              <ul className={s.rows}>{g.entries.map(row)}</ul>
            )}
            {isToday ? closing : null}
            {isLast ? null : <div className={s.dayGap} />}
          </section>
        );
      })}
    </div>
  );
}
