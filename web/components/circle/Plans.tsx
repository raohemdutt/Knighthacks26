"use client";

import { useEffect, useState } from "react";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import type { CircleThread, CircleThreads, CircleWho, ProposalBody } from "@/lib/api";
import { dayLabel, relative, span } from "@/lib/format";
import type { Loadable } from "@/lib/useFetch";
import c from "./circle.module.css";
import { Conversation } from "./Conversation";
import { ProposeTime } from "./ProposeTime";
import { SharedCalendar } from "./SharedCalendar";
import { colourOf, STATE } from "./shared";

/**
 * Plans: what is coming up (the shared calendar), every conversation between the Atlases, and the form
 * that starts one. Tapping a conversation opens it as chat bubbles; the open thread is in the URL
 * (`/circle/?tab=plans&thread=<id>`) so it can be linked to and survives a reload.
 */
export function Plans({ threads, threadId, onOpen }: {
  threads: Loadable<CircleThreads>;
  threadId: string | null;
  onOpen: (threadId: string | null) => void;
}) {
  const [again, setAgain] = useState<ProposalBody | null>(null);
  const [proposing, setProposing] = useState(false);
  const members = threads.data?.members ?? [];
  const configured = threads.data?.configured ?? false;

  useEffect(() => {
    if (new URLSearchParams(window.location.search).get("propose")) setProposing(true);
  }, []);

  if (threadId)
    return (
      <Conversation
        threadId={threadId}
        members={members}
        onBack={() => onOpen(null)}
        onProposeAgain={(plan) => {
          setAgain(plan);
          setProposing(true);
          onOpen(null);
        }}
      />
    );

  const all = threads.data?.threads ?? [];
  const open = all.filter((t) => t.state !== "agreed");
  const done = all.filter((t) => t.state === "agreed");

  return (
    <>
      <section className={ui.section}>
        {proposing ? (
          <ProposeTime members={members} configured={configured} initial={again} onRaised={() => threads.reload()} />
        ) : (
          <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} onClick={() => setProposing(true)}>
            Find a time for all of us
          </button>
        )}
      </section>

      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>Being worked out</h2>
        {threads.error ? (
          threads.error.notConfigured ? (
            <Empty title="No conversations on this box yet">This box&apos;s software is older than this screen. After its next update, plans being worked out appear here.</Empty>
          ) : (
            <ErrorBox what="the conversations" error={threads.error} onRetry={threads.reload} />
          )
        ) : !threads.data ? (
          <Loading what="the conversations" />
        ) : !configured ? (
          <Empty title="No conversations yet">
            Once this box joins a Circle, every plan appears here as a conversation: your Atlas asks, each friend&apos;s Atlas answers free, busy or another time.
          </Empty>
        ) : open.length === 0 ? (
          <Empty title="Nothing open">Every plan so far is settled. Start one above; its conversation appears here while the Atlases work it out.</Empty>
        ) : (
          <ul className={ui.list}>
            {open.map((t) => (
              <ThreadCard key={t.thread_id} thread={t} members={members} onOpen={onOpen} />
            ))}
          </ul>
        )}
        {threads.data?.offline ? (
          <p className={`${ui.muted} ${ui.small}`}>This box can&apos;t reach the Circle right now, so these are the conversations its own log holds.</p>
        ) : null}
      </section>

      <SharedCalendar title="Coming up" onOpenThread={(id) => onOpen(id)} />

      {done.length > 0 ? (
        <details className={c.fold}>
          <summary>
            Settled conversations ({done.length})
          </summary>
          <ul className={ui.list}>
            {done.map((t) => (
              <ThreadCard key={t.thread_id} thread={t} members={members} onOpen={onOpen} />
            ))}
          </ul>
        </details>
      ) : null}
    </>
  );
}

const PIP: Record<string, string | undefined> = { free: c.pipFree, counter: c.pipCounter, busy: c.pipBusy };
const SAID: Record<string, string> = { free: "free", counter: "another time", busy: "busy" };

function ThreadCard({ thread: t, members, onOpen }: { thread: CircleThread; members: CircleWho[]; onOpen: (id: string) => void }) {
  const state = STATE[t.state];
  return (
    <li className={`${ui.card} ${c.thread}`} onClick={() => onOpen(t.thread_id)}>
        <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
          <h3 className={ui.breakWords}>{t.title}</h3>
          <span className={`${c.chip} ${state.className}`}>{state.label}</span>
        </div>
        <p>
          {dayLabel(t.starts_at)}, {span(t.starts_at, t.ends_at)}
          {t.location ? ` · ${t.location}` : ""}
        </p>
        <p className={c.threadLine}>{t.last_line}</p>
        <div className={c.answers}>
          <span className={`${c.answer} ${colourOf(members, t.proposer.member_id)}`}>
            <span className={`${c.avatar} ${c.avatarSmall}`} aria-hidden="true">
              {t.proposer.display_name.slice(0, 1).toUpperCase()}
            </span>
            {t.proposer.mine ? "You asked" : `${t.proposer.display_name} asked`}
          </span>
          {t.invitees.map((i) => (
            <span key={i.member_id} className={c.answer}>
              <span className={`${c.pip} ${i.answer ? (PIP[i.answer] ?? "") : ""}`} aria-hidden="true" />
              {i.mine ? "You" : i.display_name}: {i.answer ? (SAID[i.answer] ?? i.answer) : "no answer yet"}
            </span>
          ))}
        </div>
        <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
          <span className={`${ui.muted} ${ui.small}`}>
            {t.messages_count} {t.messages_count === 1 ? "message" : "messages"} · last {relative(t.last_activity)}
          </span>
          <button
            type="button"
            className={c.link}
            onClick={(e) => {
              e.stopPropagation();
              onOpen(t.thread_id);
            }}
            aria-label={`Read the conversation about ${t.title}`}
          >
            Read the conversation ›
          </button>
        </div>
    </li>
  );
}
