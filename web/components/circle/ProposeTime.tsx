"use client";

import Link from "next/link";
import { useEffect, useMemo, useState, type FormEvent } from "react";
import { WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type CircleWho, type PlacePack, type ProposalBody } from "@/lib/api";
import { dayLabel, span } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";
import c from "./circle.module.css";
import { colourOf } from "./shared";

/**
 * "Find a time for all of us": start a plan with the Circle from the phone.
 *
 * Who (the Circle's members, from GET /api/circle/threads), what, when (three presets or any time) and
 * where (the place packs, GET /api/places). "Ask their Atlases" POSTs /api/circle/proposals, which only
 * raises an approval card on this box. Nothing is sent to anyone until that card is approved; after that
 * each friend's Atlas answers free, busy or another time from its own calendar.
 */

const HOURS = [1, 2, 3];
const TITLES = ["Study session", "Dinner", "Gym", "Coffee"];
const ELSEWHERE = "__other__";

function pad(n: number): string {
  return String(n).padStart(2, "0");
}

/** A Date as a datetime-local value, in the phone's own time zone. */
function localInput(d: Date): string {
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function at(now: Date, days: number, hour: number): Date {
  return new Date(now.getFullYear(), now.getMonth(), now.getDate() + days, hour, 0, 0, 0);
}

/** The three "find a time" presets, always in the future. */
export function presets(now: Date = new Date()): { label: string; start: Date; hours: number }[] {
  const evening = now.getHours() < 17 ? at(now, 0, 18) : now.getHours() < 21 ? at(now, 0, now.getHours() + 1) : at(now, 1, 18);
  const dow = now.getDay(); // 0 Sunday .. 6 Saturday
  const toSaturday = dow === 6 ? (now.getHours() < 10 ? 0 : 1) : dow === 0 ? 6 : 6 - dow;
  return [
    { label: now.getHours() < 21 ? "This evening" : "Tomorrow evening", start: evening, hours: 1 },
    { label: "Tomorrow afternoon", start: at(now, 1, 14), hours: 2 },
    { label: "The weekend", start: at(now, toSaturday, 11), hours: 2 },
  ];
}

/** Place names from the packs, campus first, without the "(dorm)" style notes. */
function placeNames(packs: PlacePack[] | undefined): string[] {
  const out: string[] = [];
  for (const pack of [...(packs ?? [])].sort((a, b) => Number(a.kind !== "campus") - Number(b.kind !== "campus"))) {
    if (pack.kind === "home") continue; // rooms of the apartment are not places to meet friends
    for (const p of Object.values(pack.places)) {
      const name = (p.name.split(" (")[0] ?? p.name).trim();
      if (name && !out.includes(name)) out.push(name);
    }
  }
  return out;
}

export function ProposeTime({ members, configured, initial, onRaised }: {
  /** The Circle's members; the one that is `mine` is left out of the people to ask. */
  members: CircleWho[];
  configured: boolean;
  /** A plan to start from (asking again about a plan someone could not make). */
  initial?: ProposalBody | null;
  onRaised?: (threadId: string) => void;
}) {
  const friends = useMemo(() => members.filter((m) => !m.mine), [members]);
  const packs = useFetch((s) => api.places(s));
  const places = useMemo(() => placeNames(packs.data), [packs.data]);
  const quick = useMemo(() => presets(), []);

  const [who, setWho] = useState<string[] | null>(null); // null: everyone
  const [title, setTitle] = useState("");
  const [start, setStart] = useState("");
  const [hours, setHours] = useState(1);
  const [place, setPlace] = useState("");
  const [other, setOther] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [raised, setRaised] = useState<{ title: string; thread: string } | null>(null);

  useEffect(() => {
    if (!initial) return;
    setTitle(initial.title);
    if (initial.invitees?.length) setWho(initial.invitees);
    const from = new Date(initial.start);
    const to = new Date(initial.end);
    if (!Number.isNaN(from.getTime()) && !Number.isNaN(to.getTime())) setHours(Math.max(1, Math.round((to.getTime() - from.getTime()) / 3_600_000)));
    if (initial.location) {
      setPlace(ELSEWHERE);
      setOther(initial.location);
    }
    setStart(""); // the time is the part that did not work: pick a new one
  }, [initial]);

  const asked = who ?? friends.map((m) => m.member_id);
  const begins = start ? new Date(start) : null; // datetime-local: the phone's own time zone
  const valid = begins !== null && !Number.isNaN(begins.getTime());
  const ends = valid && begins ? new Date(begins.getTime() + hours * 3_600_000) : null;
  const ready = configured && Boolean(title.trim()) && valid && asked.length > 0;
  const location = place === ELSEWHERE ? other.trim() : place;

  const toggle = (id: string) => {
    const now = asked.includes(id) ? asked.filter((m) => m !== id) : [...asked, id];
    setWho(now.length === friends.length ? null : now);
  };

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!ready || !begins || !ends || busy) return;
    setBusy(true);
    setError(null);
    setRaised(null);
    try {
      const out = await api.proposeTime({ title: title.trim(), start: begins.toISOString(), end: ends.toISOString(), location: location || undefined, invitees: asked });
      setRaised({ title: title.trim(), thread: out.thread_id });
      onRaised?.(out.thread_id);
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, String(err)));
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className={ui.card} onSubmit={submit}>
      <h3>Find a time for all of us</h3>
      <p className={`${ui.muted} ${ui.small}`}>
        Your Atlas asks theirs. Each one checks its own person&apos;s calendar and answers free, busy or another time. <strong>Nothing is sent until you approve the card.</strong>
      </p>

      <span className={ui.sectionTitle}>Who</span>
      {friends.length === 0 ? (
        <p className={`${ui.muted} ${ui.small}`}>
          {configured ? "No one else is named in this Circle yet (config/agent.yaml circle.names)." : "This box has not joined a Circle, so there is no one to ask yet."}
        </p>
      ) : (
        <div className={ui.chips} role="group" aria-label="Who to ask">
          {friends.map((m) => {
            const on = asked.includes(m.member_id);
            return (
              <button key={m.member_id} type="button" className={`${ui.chip} ${on ? ui.chipOn : ""} ${colourOf(members, m.member_id)}`} aria-pressed={on} onClick={() => toggle(m.member_id)}>
                {m.display_name}
              </button>
            );
          })}
        </div>
      )}

      <span className={ui.sectionTitle}>What</span>
      <input className={ui.input} value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g. COP4533 study session" aria-label="What" required maxLength={80} />
      <div className={ui.chips} role="group" aria-label="Suggestions">
        {TITLES.map((t) => (
          <button key={t} type="button" className={`${ui.chip} ${title === t ? ui.chipOn : ""}`} onClick={() => setTitle(t)}>
            {t}
          </button>
        ))}
      </div>

      <span className={ui.sectionTitle}>When</span>
      <div className={ui.chips} role="group" aria-label="Quick times">
        {quick.map((q) => {
          const value = localInput(q.start);
          return (
            <button
              key={q.label}
              type="button"
              className={`${ui.chip} ${start === value ? ui.chipOn : ""}`}
              aria-pressed={start === value}
              onClick={() => {
                setStart(value);
                setHours(q.hours);
              }}
            >
              {q.label}
            </button>
          );
        })}
      </div>
      <input className={ui.input} type="datetime-local" value={start} onChange={(e) => setStart(e.target.value)} aria-label="Starts" required />
      <div className={ui.chips} role="group" aria-label="For how long">
        {HOURS.map((h) => (
          <button key={h} type="button" className={`${ui.chip} ${hours === h ? ui.chipOn : ""}`} aria-pressed={hours === h} onClick={() => setHours(h)}>
            {h} {h === 1 ? "hour" : "hours"}
          </button>
        ))}
      </div>
      {valid && begins && ends ? (
        <p className={c.where}>
          {dayLabel(begins.toISOString())}, {span(begins.toISOString(), ends.toISOString())}
        </p>
      ) : null}
      <p className={`${ui.muted} ${ui.small}`}>If someone is busy then, their Atlas suggests the next hour that works for them, and you decide whether to ask everyone again.</p>

      <span className={ui.sectionTitle}>Where</span>
      {places.length > 0 ? (
        <select className={ui.select} value={place} onChange={(e) => setPlace(e.target.value)} aria-label="Where">
          <option value="">No place yet</option>
          {places.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
          <option value={ELSEWHERE}>Somewhere else…</option>
        </select>
      ) : null}
      {places.length === 0 || place === ELSEWHERE ? (
        <input
          className={ui.input}
          value={other}
          onChange={(e) => {
            setOther(e.target.value);
            setPlace(ELSEWHERE);
          }}
          placeholder="Where (optional)"
          aria-label="Where, in your own words"
          maxLength={80}
        />
      ) : null}

      {error ? <WriteError error={error} /> : null}
      {raised ? (
        <p className={ui.nudge} role="status">
          <strong>{raised.title}</strong> is waiting for your OK. Nothing has been sent: open <Link href="/approvals/">Approvals</Link> to send it to their Atlases.
        </p>
      ) : null}
      <button type="submit" className={`${ui.btn} ${ui.btnPrimary}`} disabled={busy || !ready}>
        {busy ? "Raising the card…" : "Ask their Atlases"}
      </button>
      {!configured ? <p className={`${ui.muted} ${ui.small}`}>Asking needs a Circle. The two lines that connect this box are at the top of this screen.</p> : null}
    </form>
  );
}
