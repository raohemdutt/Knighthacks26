"use client";

import Link from "next/link";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { ErrorBox, Loading, WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type CircleMessage, type CircleThreadDetail, type CircleWho, type ProposalBody } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { dayLabel, span, time } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";
import c from "./circle.module.css";
import { colourOf, STATE } from "./shared";
import { useRefresh } from "./SharedCalendar";

/**
 * One plan's conversation between the members' Atlases, as chat bubbles.
 *
 * GET /api/circle/threads/{id} gives the ordered messages; each `line` was written on the box from the
 * message's own fields (free, busy, another time), so nothing here adds a reason nobody gave. It reloads
 * when a message, an agreement or an approval arrives on /ws, and every 10 s while answers are outstanding
 * (this box's own reply reaches the screen through the Circle's store, which sends no event).
 *
 * Between a proposal and its replies each member still being waited on shows "…'s Atlas is checking the
 * calendar". A box answers on its next timer tick (about half a minute); after CHECKING_MS without an answer
 * the line changes to "no answer yet" instead of pretending something is still happening.
 */

const CHECKING_MS = 90_000;
const REFRESH_MS = 10_000;
const EVENTS = ["circle.message.received", "circle.event.agreed", "agent.action.proposed", "workspace.action.completed"];

export function Conversation({ threadId, members, onBack, onProposeAgain }: {
  threadId: string;
  members: CircleWho[];
  onBack: () => void;
  /** Open the Propose form filled with this plan (a declined plan the proposer wants to move). */
  onProposeAgain: (plan: ProposalBody) => void;
}) {
  const thread = useFetch((s) => api.circleThread(threadId, s), threadId);
  useAtlasEvents(EVENTS, () => thread.reload());
  const open = thread.data ? thread.data.state !== "agreed" : true;
  useRefresh(thread.reload, open ? REFRESH_MS : 60_000);

  return (
    <section className={ui.section}>
      <button type="button" className={c.back} onClick={onBack}>
        ‹ All plans
      </button>
      {thread.error ? (
        thread.error.notConfigured ? (
          <div className={ui.state}>
            <p className={ui.stateTitle}>This conversation is not on this box</p>
            <p>It may belong to another Circle, or this box&apos;s log no longer has it.</p>
          </div>
        ) : (
          <ErrorBox what="the conversation" error={thread.error} onRetry={thread.reload} />
        )
      ) : !thread.data ? (
        <Loading what="the conversation" />
      ) : (
        <Thread thread={thread.data} members={members} onProposeAgain={onProposeAgain} onChanged={thread.reload} />
      )}
    </section>
  );
}

function Thread({ thread: t, members, onProposeAgain, onChanged }: {
  thread: CircleThreadDetail;
  members: CircleWho[];
  onProposeAgain: (plan: ProposalBody) => void;
  onChanged: () => void;
}) {
  const state = STATE[t.state];
  const end = useRef<HTMLDivElement>(null);
  const count = t.messages.length;
  useEffect(() => {
    end.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [count]);

  let day = "";
  return (
    <>
      <div className={ui.card}>
        <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
          <h2 className={ui.breakWords}>{t.title}</h2>
          <span className={`${c.chip} ${state.className}`}>{state.label}</span>
        </div>
        <p className={c.where}>
          {dayLabel(t.starts_at)}, {span(t.starts_at, t.ends_at)}
          {t.location ? ` · ${t.location}` : ""}
        </p>
        <p className={`${ui.muted} ${ui.small}`}>
          Proposed by {t.proposer.mine ? "you" : t.proposer.display_name}. Each Atlas checks its own person&apos;s calendar on its own box and
          answers free, busy or another time. Nothing else about the calendar is shared.
        </p>
      </div>

      <div className={c.chat} role="log" aria-label={`Conversation about ${t.title}`}>
        {t.messages.map((m) => {
          const label = dayLabel(m.at);
          const first = label !== day;
          day = label;
          return (
            <FragmentWithDay key={m.id} day={first ? label : null}>
              <Bubble message={m} members={members} />
            </FragmentWithDay>
          );
        })}
        {t.state === "waiting" && t.asked_at ? t.waiting_on.map((w) => <Checking key={w.member_id} who={w} members={members} since={t.asked_at as string} />) : null}
        <div ref={end} />
      </div>

      <Action thread={t} onProposeAgain={onProposeAgain} onChanged={onChanged} />
      {t.offline ? (
        <p className={`${ui.muted} ${ui.small}`}>
          This box can&apos;t reach the Circle right now, so this is what its own log holds. Its own answers to friends&apos; plans appear when it is back online.
        </p>
      ) : null}
    </>
  );
}

function FragmentWithDay({ day, children }: { day: string | null; children: ReactNode }) {
  return (
    <>
      {day ? <div className={c.day}>{day}</div> : null}
      {children}
    </>
  );
}

function Bubble({ message: m, members }: { message: CircleMessage; members: CircleWho[] }) {
  if (m.kind === "agreed" || m.kind === "note" || !m.from_member)
    return (
      <div className={`${c.system} ${m.kind === "agreed" ? c.systemAgreed : ""}`}>
        {m.line}
        <div className={c.stamp}>{time(m.at)}</div>
      </div>
    );
  // "Maya's Atlas: Busy then." -> who said it goes in the bubble's header, the rest is the message
  const cut = m.line.indexOf(": ");
  const speaker = cut > 0 ? m.line.slice(0, cut) : `${m.display_name}'s Atlas`;
  const text = cut > 0 ? m.line.slice(cut + 2) : m.line;
  return (
    <div className={`${c.msg} ${m.mine ? c.msgMine : ""} ${colourOf(members, m.from_member)}`}>
      <span className={`${c.avatar} ${c.avatarSmall}`} aria-hidden="true">
        {(m.display_name ?? "?").slice(0, 1).toUpperCase()}
      </span>
      <div className={c.bubble}>
        <span className={c.from}>{m.mine ? `${speaker} (yours)` : speaker}</span>
        <span className={c.text}>{text}</span>
        <span className={c.stamp}>
          {time(m.at)}
          {m.source === "agreement" ? " · everyone agreed, so this box answered free" : ""}
        </span>
      </div>
    </div>
  );
}

/** "Maya's Atlas is checking the calendar…" while an answer can still be on its way; then "no answer yet". */
function Checking({ who, members, since }: { who: CircleWho; members: CircleWho[]; since: string }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 5000);
    return () => window.clearInterval(id);
  }, []);
  const asked = Date.parse(since);
  const fresh = !Number.isNaN(asked) && now - asked < CHECKING_MS;
  const whose = who.mine ? "Your Atlas" : `${who.display_name}'s Atlas`;
  if (fresh)
    return (
      <div className={`${c.checking} ${colourOf(members, who.member_id)}`} role="status">
        {whose} is checking {who.mine ? "your" : `${who.display_name}'s`} calendar
        <span className={c.dots} aria-hidden="true">
          <i />
          <i />
          <i />
        </span>
      </div>
    );
  return (
    <div className={`${c.checking} ${c.quiet}`} role="status">
      No answer yet from {whose}. It answers when {who.mine ? "this box" : `${who.display_name}'s box`} is next online.
    </div>
  );
}

function Action({ thread: t, onProposeAgain, onChanged }: { thread: CircleThreadDetail; onProposeAgain: (plan: ProposalBody) => void; onChanged: () => void }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [raised, setRaised] = useState(false);
  const a = t.action;
  const plan = a.proposal;

  const ask = async () => {
    if (!plan || busy) return;
    setBusy(true);
    setError(null);
    try {
      await api.proposeTime({ title: plan.title, start: plan.start, end: plan.end, location: plan.location ?? undefined, invitees: plan.invitees });
      setRaised(true);
      onChanged();
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, String(err)));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className={ui.card}>
      <p>{a.text}</p>
      {error ? <WriteError error={error} /> : null}
      {a.kind === "approve" || raised ? (
        <Link href="/approvals/" className={`${ui.btn} ${ui.btnPrimary}`} style={{ textAlign: "center", textDecoration: "none", display: "grid", placeItems: "center" }}>
          Open the card on Approvals
        </Link>
      ) : a.kind === "propose" && plan ? (
        t.state === "countered" ? (
          <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} onClick={ask} disabled={busy}>
            {busy ? "Raising the card…" : `Ask their Atlases about ${time(plan.start)}`}
          </button>
        ) : (
          <button
            type="button"
            className={`${ui.btn} ${ui.btnPrimary}`}
            onClick={() => onProposeAgain({ title: plan.title, start: plan.start, end: plan.end, location: plan.location ?? undefined, invitees: plan.invitees })}
          >
            Propose another time
          </button>
        )
      ) : null}
      {a.kind === "propose" || raised ? <p className={`${ui.muted} ${ui.small}`}>Nothing is sent to anyone until you approve the card.</p> : null}
    </div>
  );
}
