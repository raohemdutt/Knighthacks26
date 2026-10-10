"use client";

import { useMemo, useState } from "react";
import { ItemRow } from "@/components/ItemRow";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api, type Habit, type ImpactCounts, type Item } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { amount, dayLabel, percent, time } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";

const soonestFirst = (a: Item, b: Item) => (a.when ?? "").localeCompare(b.when ?? "");

export default function TodayPage() {
  const [since] = useState(() => new Date().toISOString()); // "upcoming" means from the moment the screen opened
  const habits = useFetch((s) => api.habitsToday(s));
  const items = useFetch((s) => api.items({ from: since }, s));
  const impact = useFetch((s) => api.impact(s));

  useAtlasEvents(["habits."], () => habits.reload());
  useAtlasEvents(["impact."], () => impact.reload());
  useAtlasEvents(["connector.item.synced"], () => items.reload());

  const due = useMemo(() => (items.data ?? []).filter((i) => i.kind === "assignment").sort(soonestFirst), [items.data]);
  const events = useMemo(() => (items.data ?? []).filter((i) => i.kind === "calendar_event").sort(soonestFirst), [items.data]);

  return (
    <>
      <div className={ui.pageHead}>
        <h1>Today</h1>
        {habits.data?.day ? <p className={ui.muted}>{dayLabel(`${habits.data.day}T12:00:00`)}, as Atlas has seen it so far</p> : null}
      </div>

      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>Habits</h2>
        {habits.error ? (
          habits.error.notConfigured ? (
            <Empty title="Habits are not set up on this box">Once habits are configured, each one shows today&apos;s count against its target here.</Empty>
          ) : (
            <ErrorBox what="today's habits" error={habits.error} onRetry={habits.reload} />
          )
        ) : !habits.data ? (
          <Loading what="habits" />
        ) : habits.data.habits.length === 0 ? (
          <Empty title="No habits tracked yet">
            Sleep, meals, focus time and the rest appear here with today&apos;s value, the target and your streak once Atlas has watched for a
            day.
          </Empty>
        ) : (
          <>
            {habits.data.latest_nudge ? (
              <p className={ui.nudge}>
                {habits.data.latest_nudge.message}
                <span className={`${ui.muted} ${ui.small}`}> · {time(habits.data.latest_nudge.ts)}</span>
              </p>
            ) : null}
            <ul className={ui.list}>
              {habits.data.habits.map((h) => (
                <HabitCard key={h.id} habit={h} />
              ))}
            </ul>
          </>
        )}
      </section>

      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>Due next</h2>
        {items.error ? (
          <ErrorBox what="what is due" error={items.error} onRetry={items.reload} />
        ) : !items.data ? (
          <Loading what="assignments" />
        ) : due.length === 0 ? (
          <Empty title="Nothing due">Assignments from Canvas with a due date ahead of now are listed here, soonest first.</Empty>
        ) : (
          <ul className={ui.list}>
            {due.map((i) => (
              <ItemRow key={`${i.connector}:${i.external_id}`} item={i} countdown />
            ))}
          </ul>
        )}
      </section>

      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>Coming up</h2>
        {items.error ? null : !items.data ? (
          <Loading what="calendar events" />
        ) : events.length === 0 ? (
          <Empty title="No upcoming events">Calendar events that have not started yet are listed here, soonest first.</Empty>
        ) : (
          <ul className={ui.list}>
            {events.map((i) => (
              <ItemRow key={`${i.connector}:${i.external_id}`} item={i} countdown />
            ))}
          </ul>
        )}
      </section>

      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>Waste Watch</h2>
        {impact.error ? (
          impact.error.notConfigured ? (
            <Empty title="Waste Watch is not set up on this box">Once it is, this shows how many items the cameras saw and how many were recycled.</Empty>
          ) : (
            <ErrorBox what="Waste Watch" error={impact.error} onRetry={impact.reload} />
          )
        ) : !impact.data ? (
          <Loading what="Waste Watch" />
        ) : (
          <>
            {impact.data.total.items_seen === 0 ? (
              <Empty title="No items counted yet">
                When a camera sees a can, bottle or cup and where it ends up, the counts below start moving. These are measured counts only.
              </Empty>
            ) : null}
            <ImpactStats counts={impact.data.total} />
            {impact.data.before_after_nudges ? (
              <div className={ui.card}>
                <h3>Before and after nudges</h3>
                <dl className={ui.kv}>
                  <dt>Before</dt>
                  <dd>
                    {percent(impact.data.before_after_nudges.before.diversion_rate)} recycled ({impact.data.before_after_nudges.before.recycled} of{" "}
                    {impact.data.before_after_nudges.before.recyclable_seen})
                  </dd>
                  <dt>After</dt>
                  <dd>
                    {percent(impact.data.before_after_nudges.after.diversion_rate)} recycled ({impact.data.before_after_nudges.after.recycled} of{" "}
                    {impact.data.before_after_nudges.after.recyclable_seen})
                  </dd>
                </dl>
              </div>
            ) : (
              <p className={`${ui.muted} ${ui.small}`}>A before and after comparison appears here once nudges are switched on for this box.</p>
            )}
          </>
        )}
      </section>
    </>
  );
}

function ImpactStats({ counts }: { counts: ImpactCounts }) {
  return (
    <div className={ui.stats}>
      <div className={ui.stat}>
        <p className={ui.statNum}>{counts.items_seen}</p>
        <p className={ui.statLabel}>items seen</p>
      </div>
      <div className={ui.stat}>
        <p className={ui.statNum}>
          {counts.recycled}
          <span className={ui.statLabel}> / {counts.recyclable_seen}</span>
        </p>
        <p className={ui.statLabel}>recyclables recycled</p>
      </div>
      <div className={ui.stat}>
        <p className={ui.statNum}>{counts.recyclable_seen > 0 ? percent(counts.diversion_rate) : "–"}</p>
        <p className={ui.statLabel}>diversion rate</p>
      </div>
      <div className={ui.stat}>
        <p className={ui.statNum}>{counts.contamination}</p>
        <p className={ui.statLabel}>wrong items in recycling</p>
      </div>
    </div>
  );
}

function HabitCard({ habit: h }: { habit: Habit }) {
  const atLeast = h.direction === "at_least";
  const fill = h.target > 0 ? Math.min(h.value / h.target, 1) : h.value > 0 ? 1 : 0;
  const good = h.applies && h.met;
  const overLimit = h.applies && !h.closed && !atLeast && !h.met;
  const status = !h.applies
    ? "Nothing to measure yet today"
    : h.closed
      ? h.met
        ? "Day closed: met"
        : "Day closed: missed"
      : h.met
        ? atLeast
          ? "Target met"
          : "Within the limit so far"
        : atLeast
          ? "Not there yet"
          : "Over the limit";
  return (
    <li className={ui.card}>
      <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
        <h3>{h.name}</h3>
        {h.streak > 0 && overLimit ? (
          // The box's streak counts closed days only; a limit already passed today cannot be un-passed.
          <span className={ui.tag}>{h.streak}-day streak ends today</span>
        ) : h.streak > 0 ? (
          <span className={`${ui.tag} ${ui.tagAccent}`}>
            {h.streak}-day streak
          </span>
        ) : (
          <span className={ui.tag}>no streak yet</span>
        )}
      </div>
      <p>
        <span className={ui.habitValue}>{amount(h.value)}</span>{" "}
        <span className={ui.muted}>
          of {atLeast ? "at least" : "at most"} {amount(h.target)} {h.unit}
        </span>
      </p>
      <div className={ui.meter} role="img" aria-label={`${amount(h.value)} of ${amount(h.target)} ${h.unit}`}>
        <div className={`${ui.meterFill} ${good ? ui.meterMet : ""}`} style={{ width: `${Math.round(fill * 100)}%` }} />
      </div>
      <p className={good ? ui.small : `${ui.small} ${ui.muted}`}>
        {good ? <strong>{status}</strong> : status}
      </p>
      {h.nudge ? (
        <p className={ui.nudge}>
          {h.nudge.message}
          <span className={`${ui.muted} ${ui.small}`}> · {time(h.nudge.ts)}</span>
        </p>
      ) : null}
    </li>
  );
}
