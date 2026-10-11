"use client";

import { useCallback, useEffect, useState } from "react";
import { Leaderboards } from "@/components/circle/Leaderboards";
import { League } from "@/components/circle/League";
import { People } from "@/components/circle/People";
import { Plans } from "@/components/circle/Plans";
import { SharedBrain } from "@/components/circle/SharedBrain";
import { WhereToGo } from "@/components/circle/WhereToGo";
import { useRefresh } from "@/components/circle/SharedCalendar";
import { TABS, type CircleTab } from "@/components/circle/shared";
import { YouAreSharing } from "@/components/circle/YouAreSharing";
import c from "@/components/circle/circle.module.css";
import ui from "@/components/ui.module.css";
import { api } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { useFetch } from "@/lib/useFetch";

/**
 * The Circle: the friends' network as one place, in four parts.
 *
 *   People   who is in it, where they are (if they shared), their League week, the walk to them
 *   Plans    what is coming up, every conversation between the Atlases, and "find a time for all of us"
 *   League   the weekly game: standings, the stake, dares, the table's recycling
 *   Sharing  exactly what leaves this box, and what never does
 *
 * The part and the open conversation live in the URL (`/circle/?tab=plans&thread=<id>`), read once on
 * load and kept in step with history.replaceState, so a link opens the right thing and a reload stays put.
 * Every part renders when the box has not joined a Circle, and the top of the screen says how to join.
 */

const REFRESH_MS = 30_000;

function readUrl(): { tab: CircleTab; thread: string | null } {
  const q = new URLSearchParams(window.location.search);
  const thread = q.get("thread");
  const tab = TABS.find((t) => t.id === q.get("tab"))?.id ?? (thread ? "plans" : "people");
  return { tab, thread: tab === "plans" ? thread : null };
}

export default function CirclePage() {
  const [tab, setTab] = useState<CircleTab>("people");
  const [thread, setThread] = useState<string | null>(null);

  const members = useFetch((s) => api.circleMembers(s));
  const threads = useFetch((s) => api.circleThreads(s));
  const league = useFetch((s) => api.league(s));
  const packs = useFetch((s) => api.places(s));
  const faceShare = useFetch((s) => api.faceShare(s)); // null while face sharing is off on this box (the default)

  useAtlasEvents(["circle.place.shared", "circle.message.received", "play.league."], () => members.reload());
  useAtlasEvents(["circle.message.received", "circle.event.agreed", "agent.action.proposed", "workspace.action.completed"], () => threads.reload());
  useAtlasEvents(["play.league.", "habits.day.closed"], () => league.reload());
  useAtlasEvents(["circle.face_share", "perception.identity"], () => faceShare.reload());
  const refresh = useCallback(() => {
    members.reload(); // a shared place runs out without an event
    threads.reload(); // and an approval card does too
  }, [members.reload, threads.reload]);
  useRefresh(refresh, REFRESH_MS);

  useEffect(() => {
    const sync = () => {
      const at = readUrl();
      setTab(at.tab);
      setThread(at.thread);
    };
    sync();
    window.addEventListener("popstate", sync);
    return () => window.removeEventListener("popstate", sync);
  }, []);

  const go = useCallback((next: CircleTab, nextThread: string | null = null) => {
    setTab(next);
    setThread(nextThread);
    const q = new URLSearchParams();
    if (next !== "people") q.set("tab", next);
    if (nextThread) q.set("thread", nextThread);
    const url = `${window.location.pathname}${q.toString() ? `?${q}` : ""}`;
    // opening a conversation is a step back can undo; switching parts is not
    if (nextThread) window.history.pushState(null, "", url);
    else window.history.replaceState(null, "", url);
    if (nextThread) window.scrollTo({ top: 0 });
  }, []);

  const off = members.data ? !members.data.configured : false;
  const open = (threads.data?.threads ?? []).filter((t) => t.state !== "agreed").length;

  return (
    <>
      <div className={ui.pageHead}>
        <h1>Circle</h1>
        <p className={ui.muted}>
          Friends who also run Atlas. You share only place names, one-word answers, counts and scores, never video.
          {faceShare.data ? " Face sharing is on: your own face template goes only to the friends you choose, encrypted for their box." : ""}
        </p>
      </div>

      {off ? <NotConnected lines={members.data?.connect ?? []} /> : null}

      <div className={`${ui.segmented} ${c.tabs}`} role="tablist" aria-label="Parts of the Circle">
        {TABS.map((t) => (
          <button key={t.id} type="button" role="tab" aria-selected={tab === t.id} className={`${ui.seg} ${c.segSmall} ${tab === t.id ? ui.segOn : ""}`} onClick={() => go(t.id)}>
            {t.label}
            {t.id === "plans" && open > 0 ? ` (${open})` : ""}
          </button>
        ))}
      </div>

      {tab === "people" ? (
        <>
          <SharedBrain members={members.data} />
          <WhereToGo friendsSharing={(members.data?.members ?? []).filter((m) => !m.mine && m.place).length} />
          <People members={members} standings={league.data?.standings ?? []} packs={packs.data} faceShare={faceShare.data} onChanged={members.reload} />
        </>
      ) : tab === "plans" ? (
        <Plans threads={threads} threadId={thread} onOpen={(id) => go("plans", id)} />
      ) : tab === "league" ? (
        <>
          <League />
          <Leaderboards circleOff={off} />
        </>
      ) : (
        <YouAreSharing faceShare={faceShare.data} />
      )}

      {tab !== "sharing" ? (
        <button type="button" className={c.link} onClick={() => go("sharing")}>
          What leaves this box, and what never does ›
        </button>
      ) : null}
    </>
  );
}

/** The honest not-connected state: what is missing, and the two lines that fix it. */
function NotConnected({ lines }: { lines: string[] }) {
  const config = ["circle:", ...(lines.length ? lines : ["circle.member_id: hemdutt", "circle.circle_id: team"]).map((l) => `  ${l.replace(/^circle\./, "")}`)].join("\n");
  return (
    <div className={ui.card} role="status">
      <h3>This box has not joined a Circle</h3>
      <p>
        So there are no friends to show, nobody to plan with, and nothing is being shared. Everything below still shows what it will hold. The League already runs
        for this box alone.
      </p>
      <p className={`${ui.muted} ${ui.small}`}>
        To join, set these two lines in <code>config/agent.yaml</code> on the box (your own id, and the circle&apos;s name) and restart it:
      </p>
      <pre className={c.config}>{config}</pre>
      <p className={`${ui.muted} ${ui.small}`}>
        Friends on other boxes use the same <code>circle_id</code> and the same shared database (<code>NEON_DATABASE_URL</code> in the box&apos;s <code>.env</code>). Without
        the database this box still plans, but only with itself.
      </p>
    </div>
  );
}
