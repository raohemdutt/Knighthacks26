"use client";

import Link from "next/link";
import { useState, type FormEvent } from "react";
import { Empty, ErrorBox, Loading, WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type LeagueChallenge, type LeagueStanding, type LeagueView, type StakeKind } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { humanize, percent } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";

/**
 * The League: the friends' weekly game on top of Habits, Day Run and Waste Watch.
 *
 * GET /api/league answers in every state. Each member's week is the weighted share of judged
 * target-days they hit; the lowest is the worst week and owes the stake, the highest is the best.
 * It reloads on every play.league.* event (a friend's summary, the stake, a dare, the week closing).
 * "Set the stake" and "Start a dare" only raise approval cards: the user confirms on Approvals.
 */

const STAKES: { kind: StakeKind; label: string; what: string }[] = [
  { kind: "pays_for_food", label: "Pays for food", what: "The worst week buys the next group meal; the best week picks the place." },
  { kind: "picks_the_event", label: "Picks the event", what: "The best week picks the next outing." },
  { kind: "custom", label: "Custom", what: "Your own words." },
];
const DAYS = [3, 5, 7];

export function League() {
  const league = useFetch((s) => api.league(s));
  useAtlasEvents(["play.league.", "habits.day.closed", "impact.count.updated"], () => league.reload());

  return (
    <section className={ui.section}>
      <h2 className={ui.sectionTitle}>The League</h2>
      {league.error ? (
        <ErrorBox what="the League" error={league.error} onRetry={league.reload} />
      ) : !league.data ? (
        <Loading what="the League" />
      ) : !league.data.configured ? (
        <Empty title="No League on this box">{league.data.note}</Empty>
      ) : (
        <Table view={league.data} />
      )}
      {league.data?.configured ? (
        <>
          <SetStake view={league.data} />
          <StartChallenge view={league.data} />
        </>
      ) : null}
    </section>
  );
}

function closesIn(seconds: number): string {
  if (seconds <= 0) return "closing now";
  const h = Math.floor(seconds / 3600);
  if (h >= 48) return `closes in ${Math.round(h / 24)} days`;
  if (h >= 1) return `closes in ${h} h`;
  return `closes in ${Math.max(1, Math.round(seconds / 60))} min`;
}

function weekLabel(iso: string): string {
  const d = new Date(`${iso}T12:00:00`);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function Table({ view }: { view: LeagueView }) {
  const judged = view.standings.filter((s) => s.score !== null);
  return (
    <>
      <div className={ui.card}>
        <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
          <h3>Week of {view.week ? weekLabel(view.week.start) : "…"}</h3>
          {view.week ? <span className={ui.tag}>{closesIn(view.week.closes_in_s)}</span> : null}
        </div>
        {view.local ? (
          <p className={`${ui.muted} ${ui.small}`}>
            This box plays alone. Set <code>circle.member_id</code> and <code>circle.circle_id</code> in config/agent.yaml to play with friends.
          </p>
        ) : null}
        {judged.length === 0 ? (
          <p className={ui.muted}>Nothing has been judged yet this week. The first closed day puts someone on the table.</p>
        ) : (
          <table className={ui.table}>
            <thead>
              <tr>
                <th scope="col">#</th>
                <th scope="col">Who</th>
                <th scope="col" className={ui.num}>Week</th>
                <th scope="col" className={ui.num} style={{ paddingLeft: 14 }}>Hit</th>
              </tr>
            </thead>
            <tbody>
              {view.standings.map((s) => (
                <Row key={s.member_id} s={s} me={s.member_id === view.member_id} worst={view.worst?.member_id === s.member_id} best={view.best?.member_id === s.member_id} />
              ))}
            </tbody>
          </table>
        )}
        {view.summary ? (
          <p className={ui.nudge} role="status">
            {view.summary}
          </p>
        ) : null}
        {view.stake ? (
          <p className={`${ui.muted} ${ui.small}`}>
            Stake: {view.stake.text_for_week}
            {view.stake.set_for_week ? ` (set from the week of ${weekLabel(view.stake.set_for_week)})` : " (the default; change it below)"}
          </p>
        ) : null}
        {view.last_closed ? (
          <p className={`${ui.muted} ${ui.small}`}>
            Last week ({weekLabel(view.last_closed.week_start)}): {view.last_closed.summary}
          </p>
        ) : null}
      </div>

      {view.team_waste ? <TeamWaste view={view} /> : null}

      <div className={ui.card}>
        <h3>Dares</h3>
        {view.challenges.length === 0 ? (
          <p className={ui.muted}>No dares yet. Start one below: &quot;no energy drinks for 3 days&quot;, &quot;three meals a day for 5 days&quot;.</p>
        ) : (
          <ul className={ui.list}>
            {view.challenges.map((c) => (
              <Challenge key={c.challenge_id} c={c} />
            ))}
          </ul>
        )}
      </div>
    </>
  );
}

function Row({ s, me, worst, best }: { s: LeagueStanding; me: boolean; worst: boolean; best: boolean }) {
  const name = s.display_name === s.member_id ? humanize(s.member_id) : s.display_name;
  return (
    <tr>
      <td>{s.rank}</td>
      <td>
        <strong>{name}</strong>
        {me && name !== "You" ? <span className={`${ui.muted} ${ui.small}`}> (you)</span> : null}
        {best ? <span className={`${ui.tag} ${ui.tagAccent}`}> best week</span> : worst ? <span className={ui.tag}> worst week</span> : null}
        {s.reasons.length > 0 ? <p className={`${ui.muted} ${ui.small}`}>{s.reasons.slice(0, 3).join(" · ")}</p> : null}
      </td>
      <td className={ui.num}>{s.score === null ? "—" : `${Math.round(s.score)}%`}</td>
      <td className={ui.num} style={{ paddingLeft: 14 }}>
        {s.met}/{s.judged}
      </td>
    </tr>
  );
}

function TeamWaste({ view }: { view: LeagueView }) {
  const t = view.team_waste!;
  const fill = t.rate === null ? 0 : Math.min(100, Math.round((t.rate / t.target) * 100));
  return (
    <div className={ui.card}>
      <h3>Table vs table</h3>
      <p className={ui.small}>Waste Watch, everyone&apos;s bins together this week. {t.line}.</p>
      <div className={ui.meter} aria-label={t.line}>
        <div className={`${ui.meterFill} ${t.rate !== null && t.rate >= t.target ? ui.meterMet : ""}`} style={{ width: `${fill}%` }} />
      </div>
      {t.members.length > 0 ? (
        <table className={ui.table}>
          <thead>
            <tr>
              <th scope="col">Who</th>
              <th scope="col" className={ui.num}>Recycled</th>
              <th scope="col" className={ui.num}>Rate</th>
            </tr>
          </thead>
          <tbody>
            {t.members.map((m) => (
              <tr key={m.member_id}>
                <td>{m.display_name === m.member_id ? humanize(m.member_id) : m.display_name}</td>
                <td className={ui.num}>
                  {m.recycled} of {m.recyclable}
                </td>
                <td className={ui.num}>{m.diversion_rate === null ? "—" : percent(m.diversion_rate)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
      {t.contamination_of_the_week ? <p className={`${ui.muted} ${ui.small}`}>{t.contamination_of_the_week}</p> : null}
    </div>
  );
}

function Challenge({ c }: { c: LeagueChallenge }) {
  const status = c.status === "closed" ? "over" : c.status === "upcoming" ? `starts ${weekLabel(c.starts)}` : `until ${weekLabel(c.ends)}`;
  return (
    <li>
      <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
        <strong>
          {c.title}, {c.days} {c.days === 1 ? "day" : "days"}
        </strong>
        <span className={c.status === "running" ? `${ui.tag} ${ui.tagAccent}` : ui.tag}>{status}</span>
      </div>
      <p className={ui.small}>
        {c.members
          .map((m) => `${m.display_name === m.member_id ? humanize(m.member_id) : m.display_name} ${m.met_days}/${m.days}`)
          .join(" · ")}
      </p>
      {c.summary ? <p className={`${ui.muted} ${ui.small}`}>{c.summary}</p> : null}
    </li>
  );
}

function SetStake({ view }: { view: LeagueView }) {
  const [kind, setKind] = useState<StakeKind>(view.stake?.kind ?? "pays_for_food");
  const [text, setText] = useState(view.stake?.text ?? "");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [done, setDone] = useState<string | null>(null);
  const ready = kind !== "custom" || text.trim().length > 0;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!ready || busy) return;
    setBusy(true);
    setError(null);
    setDone(null);
    try {
      const r = await api.setStake(kind, kind === "custom" ? text.trim() : undefined);
      setDone(r.reason);
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, String(err)));
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className={ui.card} onSubmit={submit}>
      <h3>Set the stake</h3>
      <div className={ui.chips} role="group" aria-label="Stake">
        {STAKES.map((s) => (
          <button key={s.kind} type="button" className={`${ui.chip} ${kind === s.kind ? ui.chipOn : ""}`} aria-pressed={kind === s.kind} onClick={() => setKind(s.kind)}>
            {s.label}
          </button>
        ))}
      </div>
      <p className={`${ui.muted} ${ui.small}`}>{STAKES.find((s) => s.kind === kind)?.what}</p>
      {kind === "custom" ? (
        <input className={ui.input} value={text} onChange={(e) => setText(e.target.value)} placeholder="The loser does the dishes all week" aria-label="Custom stake" maxLength={200} />
      ) : null}
      {error ? <WriteError error={error} /> : null}
      {done ? (
        <p className={ui.nudge} role="status">
          Card raised: {done} Confirm it on <Link href="/approvals">Approvals</Link>; the stake changes once you approve.
        </p>
      ) : null}
      <button type="submit" className={`${ui.btn} ${ui.btnPrimary}`} disabled={busy || !ready}>
        {busy ? "Raising the card…" : "Set the stake"}
      </button>
    </form>
  );
}

function StartChallenge({ view }: { view: LeagueView }) {
  const presets = view.presets;
  const [kind, setKind] = useState(presets[0]?.kind ?? "");
  const [days, setDays] = useState(3);
  const [invitees, setInvitees] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [done, setDone] = useState<string | null>(null);
  const friends = view.members.filter((m) => m.member_id !== view.member_id);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!kind || busy) return;
    setBusy(true);
    setError(null);
    setDone(null);
    try {
      const r = await api.startChallenge({ kind, days, invitees });
      setDone(r.reason);
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, String(err)));
    } finally {
      setBusy(false);
    }
  };

  return (
    <form className={ui.card} onSubmit={submit}>
      <h3>Start a dare</h3>
      {presets.length === 0 ? <p className={ui.muted}>config/league.yaml lists no dares.</p> : null}
      <div className={ui.chips} role="group" aria-label="Dare">
        {presets.map((p) => (
          <button key={p.kind} type="button" className={`${ui.chip} ${kind === p.kind ? ui.chipOn : ""}`} aria-pressed={kind === p.kind} onClick={() => setKind(p.kind)}>
            {p.title}
          </button>
        ))}
      </div>
      <div className={ui.chips} role="group" aria-label="For how many days">
        {DAYS.map((d) => (
          <button key={d} type="button" className={`${ui.chip} ${days === d ? ui.chipOn : ""}`} aria-pressed={days === d} onClick={() => setDays(d)}>
            {d} days
          </button>
        ))}
      </div>
      {friends.length > 0 ? (
        <div className={ui.chips} role="group" aria-label="Invite">
          {friends.map((f) => {
            const on = invitees.includes(f.member_id);
            return (
              <button
                key={f.member_id}
                type="button"
                className={`${ui.chip} ${on ? ui.chipOn : ""}`}
                aria-pressed={on}
                onClick={() => setInvitees(on ? invitees.filter((m) => m !== f.member_id) : [...invitees, f.member_id])}
              >
                {f.display_name === f.member_id ? humanize(f.member_id) : f.display_name}
              </button>
            );
          })}
        </div>
      ) : (
        <p className={`${ui.muted} ${ui.small}`}>No friends to invite yet; the dare is yours alone. Starts tomorrow, judged from the same daily summaries.</p>
      )}
      {error ? <WriteError error={error} /> : null}
      {done ? (
        <p className={ui.nudge} role="status">
          Card raised: {done}. Confirm it on <Link href="/approvals">Approvals</Link>; friends are invited only once you approve.
        </p>
      ) : null}
      <button type="submit" className={`${ui.btn} ${ui.btnPrimary}`} disabled={busy || !kind}>
        {busy ? "Raising the card…" : "Start the dare"}
      </button>
    </form>
  );
}
