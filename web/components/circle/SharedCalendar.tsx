"use client";

import { useEffect } from "react";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api, type CircleAttendee, type CircleEvent, type CircleEvents, type RsvpAnswer } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { dayLabel, humanize, relative, span, time } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";

/**
 * The Circle's shared calendar: events everyone invited agreed to, soonest first.
 *
 * GET /api/circle/events answers in every state. With the shared database it is the Circle's own
 * calendar; without it (or while it is offline) it is only what this box itself agreed. It reloads
 * when an agreement or a friend's reply arrives on /ws, and once a minute so a friend's box agreeing
 * something (which sends this box no event) and events that have ended are picked up too.
 */

const REFRESH_MS = 60_000;

const ANSWERS: { answer: RsvpAnswer; label: string }[] = [
  { answer: "free", label: "In" },
  { answer: "counter", label: "Asked for another time" },
  { answer: "busy", label: "Can't make it" },
];

/** Call `reload` every `ms` while the screen is showing. */
export function useRefresh(reload: () => void, ms: number): void {
  useEffect(() => {
    const id = window.setInterval(() => {
      if (document.visibilityState === "visible") reload();
    }, ms);
    return () => window.clearInterval(id);
  }, [reload, ms]);
}

/** A member without a display name comes back as their id; show it the way "Where friends are" does. */
function name(a: CircleAttendee): string {
  return !a.display_name || a.display_name === a.member_id ? humanize(a.member_id) : a.display_name;
}

/** "Tomorrow, 12:00 PM to 2:00 PM", or both days when it runs past midnight. */
function when(e: CircleEvent): string {
  const first = dayLabel(e.starts_at);
  const last = dayLabel(e.ends_at);
  if (!first) return "Time not set";
  if (first === last || !last) return `${first}, ${span(e.starts_at, e.ends_at)}`;
  return `${first}, ${time(e.starts_at)} to ${last}, ${time(e.ends_at)}`;
}

function started(e: CircleEvent): boolean {
  const t = Date.parse(e.starts_at);
  return !Number.isNaN(t) && t <= Date.now();
}

export function SharedCalendar() {
  const calendar = useFetch((s) => api.circleEvents(s));
  useAtlasEvents(["circle.event.agreed", "circle.message.received"], () => calendar.reload());
  useRefresh(calendar.reload, REFRESH_MS);

  return (
    <section className={ui.section}>
      <h2 className={ui.sectionTitle}>Shared calendar</h2>
      {calendar.error ? (
        calendar.error.notConfigured ? (
          <Empty title="No shared calendar on this box yet">
            This box&apos;s software is older than the shared calendar. After its next update, plans everyone agreed to appear here.
          </Empty>
        ) : (
          <ErrorBox what="the shared calendar" error={calendar.error} onRetry={calendar.reload} />
        )
      ) : !calendar.data ? (
        <Loading what="the shared calendar" />
      ) : (
        <Calendar calendar={calendar.data} />
      )}
    </section>
  );
}

function Calendar({ calendar }: { calendar: CircleEvents }) {
  const events = Array.isArray(calendar.events) ? calendar.events : [];
  if (!calendar.configured)
    return (
      <Empty title="No shared calendar yet">
        This box has not joined a Circle. Once it has, plans that everyone invited said yes to appear here with the day, time and place.
      </Empty>
    );

  const note = calendar.offline
    ? "This box can't reach the Circle's shared calendar right now, so this shows only plans it agreed itself. Friends' plans come back when it is online."
    : calendar.source !== "neon"
      ? "This box is not connected to the Circle's shared database, so this shows only plans it agreed itself."
      : null;

  return (
    <>
      {note ? <p className={`${ui.muted} ${ui.small}`}>{note}</p> : null}
      {events.length === 0 ? (
        <Empty title="Nothing planned yet">
          When everyone invited says yes to a plan, it appears here with the day, the time on your phone&apos;s clock, the place and who is in.
        </Empty>
      ) : (
        <ul className={ui.list}>
          {events.map((e) => (
            <EventCard key={e.id} event={e} />
          ))}
        </ul>
      )}
    </>
  );
}

function EventCard({ event: e }: { event: CircleEvent }) {
  const attendees = Array.isArray(e.attendees) ? e.attendees : [];
  const groups = ANSWERS.map((a) => ({ ...a, names: attendees.filter((x) => x.answer === a.answer).map(name) })).filter((g) => g.names.length > 0);
  const now = started(e);
  return (
    <li className={ui.card}>
      <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
        <h3 className={ui.breakWords}>{e.title}</h3>
        <span className={`${ui.tag} ${now ? ui.tagAccent : ""}`}>{now ? "Happening now" : relative(e.starts_at)}</span>
      </div>
      <p className={ui.reason}>{when(e)}</p>
      <dl className={ui.kv}>
        <dt>Where</dt>
        <dd>{e.location || "No place set"}</dd>
        {groups.map((g) => (
          <FragmentRow key={g.answer} label={g.label} value={g.names.join(", ")} />
        ))}
        {groups.length === 0 ? <FragmentRow label="Who" value="No answers recorded yet" /> : null}
      </dl>
    </li>
  );
}

function FragmentRow({ label, value }: { label: string; value: string }) {
  return (
    <>
      <dt>{label}</dt>
      <dd>{value}</dd>
    </>
  );
}
