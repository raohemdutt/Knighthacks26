"use client";

import { useMemo, useState } from "react";
import { RouteSheet } from "@/components/dayrun/RouteSheet";
import { Empty } from "@/components/States";
import { api, type DayNow, type Item, type PlacePack } from "@/lib/api";
import { dayLabel, relative, time } from "@/lib/format";
import { parseNearby, useGoing, type Answer, type Nearby } from "@/lib/nearby";
import { useFetch } from "@/lib/useFetch";
import s from "./screen.module.css";

/**
 * Near you: optional events around campus that you can say yes to. Yes puts the event in your schedule beside
 * the rest of your day, with a warning when it overlaps something already there; No tucks it away (and can be
 * undone). The events are plain connector items, see lib/nearby.ts.
 */

const SOURCES: Record<string, { logo: string; name: string }> = {
  groupme: { logo: "/apps/groupme.png", name: "GroupMe" },
  "ucf events": { logo: "/apps/ucf.jpg", name: "UCF Events" },
  ucf: { logo: "/apps/ucf.jpg", name: "UCF Events" },
};

type Filter = "all" | "friends" | "free" | "study";
const FILTERS: { key: Filter; label: string }[] = [
  { key: "all", label: "All" },
  { key: "friends", label: "With friends" },
  { key: "free", label: "Free" },
  { key: "study", label: "Study" },
];

const STUDY = /review|study|tutor|workshop|office hours|exam|quiz|lecture/i;
const isStudy = (e: Nearby) => Boolean(e.helpsWith) || STUDY.test(e.title);
const matches = (e: Nearby, f: Filter) =>
  f === "all" || (f === "friends" && (Boolean(e.host) || e.friends.length > 0)) || (f === "free" && e.free) || (f === "study" && isStudy(e));

const PALETTE = ["#9bbcf2", "#f3cf93", "#bdb9ec", "#bfe3d1", "#f2b0bb", "#c9d7f5"];
const colorOf = (name: string) => PALETTE[[...name].reduce((n, c) => n + c.charCodeAt(0), 0) % PALETTE.length] as string;

function Person({ name, host = false }: { name: string; host?: boolean }) {
  return (
    <span className={s.who} title={host ? `${name} is hosting` : `${name} is going`}>
      <i className={s.avatar} style={{ background: colorOf(name) }} aria-hidden="true">
        {name.charAt(0).toUpperCase()}
      </i>
      {name}
      {host ? <em>hosts</em> : null}
    </span>
  );
}

/** "12 min walk" from where the box last saw you, when the place is on the campus map. */
function Walk({ from, to }: { from: string; to: string }) {
  const route = useFetch((sig) => api.placeRoute({ from, to }, sig), `${from}>${to}`);
  const w = route.data?.walk_minutes;
  if (w === undefined) return null;
  return <span className={s.walk}>{w === 0 ? "you are there" : `${w} min walk`}</span>;
}

function dayHeading(t: number): string {
  const iso = new Date(t).toISOString();
  const label = dayLabel(iso);
  if (label !== "Today") return label;
  return new Date(t).getHours() >= 17 ? "Tonight" : "Today";
}

export function overlaps(a: { start: number; end: number }, b: { start: number; end: number }): boolean {
  return a.start < b.end && b.start < a.end;
}

export function NearbyPanel({ d, items, packs }: { d: DayNow; items: Item[]; packs?: PlacePack[] }) {
  const { answers, answer } = useGoing();
  const [filter, setFilter] = useState<Filter>("all");
  const now = Date.parse(d.now);
  const all = useMemo(() => parseNearby(items, packs).filter((e) => e.end > now), [items, packs, now]);
  const fixed = useMemo(
    () => (d.plan?.moves ?? []).filter((m) => m.fixed).map((m) => ({ title: m.title, start: Date.parse(m.start), end: Date.parse(m.end) })),
    [d.plan],
  );
  const here = d.status?.seen_at ?? "here";

  const open = all.filter((e) => !answers[e.id]);
  const going = all.filter((e) => answers[e.id] === "going");
  const skipped = all.filter((e) => answers[e.id] === "skipped");
  const shown = open.filter((e) => matches(e, filter));

  // an event that overlaps something already on the calendar or one you said yes to
  const clash = (e: Nearby): string | null => {
    const hit = fixed.find((f) => overlaps(e, f)) ?? going.find((g) => g.id !== e.id && overlaps(e, g));
    return hit ? hit.title : null;
  };

  const groups: { day: string; events: Nearby[] }[] = [];
  for (const e of shown) {
    const day = dayHeading(e.start);
    const last = groups[groups.length - 1];
    if (last?.day === day) last.events.push(e);
    else groups.push({ day, events: [e] });
  }

  return (
    <section className={s.nearby} aria-label="Near you">
      <header className={s.paneHead}>
        <div>
          <h2 className={s.paneTitle}>Near you</h2>
          <p className={s.paneSub}>Optional, from your friends and campus. Say yes and it joins your schedule.</p>
        </div>
        {going.length > 0 ? <span className={s.countPill}>{going.length} going</span> : null}
      </header>

      {all.length > 0 ? (
        <div className={s.filters} role="group" aria-label="Filter events">
          {FILTERS.map((f) => (
            <button key={f.key} type="button" className={`${s.filter} ${filter === f.key ? s.filterOn : ""}`} aria-pressed={filter === f.key} onClick={() => setFilter(f.key)}>
              {f.label}
            </button>
          ))}
        </div>
      ) : null}

      {all.length === 0 ? (
        <Empty title="Nothing near you right now">Events from your group chats and campus that you can join show up here, soonest first.</Empty>
      ) : shown.length === 0 ? (
        <Empty title={open.length === 0 ? "You have answered everything" : "Nothing matches that filter"}>
          {open.length === 0 ? "New events show up as friends and campus post them." : "Try All to see every event again."}
        </Empty>
      ) : (
        groups.map((g) => (
          <div key={g.day} className={s.dayGroup}>
            <h3 className={s.dayHead}>{g.day}</h3>
            <ul className={s.cards}>
              {g.events.map((e) => (
                <EventCard key={e.id} e={e} here={here} now={d.now} clash={clash(e)} answer={answer} />
              ))}
            </ul>
          </div>
        ))
      )}

      {skipped.length > 0 ? (
        <details className={s.skipped}>
          <summary>Not for me ({skipped.length})</summary>
          <ul className={s.skippedList}>
            {skipped.map((e) => (
              <li key={e.id}>
                <span>
                  {e.title} <em>{dayHeading(e.start)}, {time(new Date(e.start).toISOString())}</em>
                </span>
                <button type="button" className={s.linkBtn} onClick={() => answer(e.id, null)}>
                  Undo
                </button>
              </li>
            ))}
          </ul>
        </details>
      ) : null}
    </section>
  );
}

function EventCard({ e, here, now, clash, answer }: { e: Nearby; here: string; now: string; clash: string | null; answer: (id: string, a: Answer | null) => void }) {
  const src = e.source ? SOURCES[e.source.toLowerCase()] : undefined;
  const iso = new Date(e.start).toISOString();
  const soon = e.start - Date.parse(now) < 3 * 3600_000;
  const people = [...(e.host ? [e.host] : []), ...e.friends.filter((f) => f !== e.host)];
  return (
    <li className={`${s.event} ${e.host ? s.eventHero : ""}`}>
      <div className={s.eventTop}>
        <div className={s.eventWhen}>
          <b>{time(iso)}</b>
          <span>{e.until ? e.until.replace(/^until /i, "to ") : ""}</span>
        </div>
        <div className={s.eventMain}>
          <h4 className={s.eventTitle}>{e.title}</h4>
          <p className={s.eventMeta}>
            {e.place ?? "Place not set"}
            {e.placeId ? <> · <Walk from={here} to={e.placeId} /></> : null}
          </p>
        </div>
        {src ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img className={s.srcLogo} src={src.logo} alt={`from ${src.name}`} title={`From ${src.name}`} width={24} height={24} />
        ) : null}
      </div>

      <div className={s.chipsRow}>
        {soon ? <span className={`${s.chip} ${s.chipAccent}`}>{relative(iso, new Date(now))}</span> : null}
        {e.free ? <span className={`${s.chip} ${s.chipGood}`}>Free</span> : null}
        {e.optional ? <span className={s.chip}>Optional</span> : null}
        {e.extra.map((x) => (
          <span key={x} className={s.chip}>
            {x}
          </span>
        ))}
      </div>

      {people.length > 0 ? (
        <p className={s.people}>
          {people.map((p) => (
            <Person key={p} name={p} host={p === e.host} />
          ))}
          {e.friends.length > 0 && e.host ? <span className={s.going}>{e.friends.length === 1 ? "is going" : "are going"}</span> : null}
        </p>
      ) : null}

      {e.said ? (
        <blockquote className={s.quote}>
          <p>“{e.said.text}”</p>
          <cite>
            {e.host} in {e.said.where}
          </cite>
        </blockquote>
      ) : null}

      {e.helpsWith ? (
        <p className={s.helps}>
          <b>Helps with</b> {e.helpsWith.title.replace(/\s*\[[^\]]+\]\s*$/, "")}, due {dayLabel(e.helpsWith.when)} {time(e.helpsWith.when)}
        </p>
      ) : null}

      {clash ? <p className={s.clash}>Overlaps {clash}</p> : null}

      <div className={s.eventActions}>
        {e.placeId ? <RouteSheet from={here} to={e.placeId} arriveBy={iso} now={now} what={e.title} label="Directions" /> : null}
        <button type="button" className={s.skipBtn} onClick={() => answer(e.id, "skipped")}>
          Not for me
        </button>
        <button type="button" className={s.joinBtn} onClick={() => answer(e.id, "going")}>
          I&apos;m in
        </button>
      </div>
    </li>
  );
}
