"use client";

import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api, type CircleSharing, type FaceShareStatus, type ShareKindStatus } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { dayLabel, dayTime, humanize, relative, time } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";
import { useRefresh } from "./SharedCalendar";

/**
 * "You are sharing": what leaves this box for the Circle, said plainly.
 *
 * GET /api/circle/sharing answers in every state: the two settings (config/agent.yaml
 * circle.share_summaries and circle.share_presence), each kind of thing that can go and whether
 * it is on, how many shares wait on the box and how many were sent (the local outbox), when the
 * last one went, and whether the box can reach the Circle's shared database. It reloads when a
 * share is queued or sent, and every half minute, because going offline sends no event.
 */

const REFRESH_MS = 30_000;

const KINDS: Record<string, { label: string; what: string }> = {
  habit_summary: {
    label: "Habit daily summaries",
    what: "One line per habit per day: whether you met it, the day's number and your streak.",
  },
  impact_count: {
    label: "Recycling counts",
    what: "Waste Watch running totals: items seen, how many were recycled or put in the wrong bin, and the rate.",
  },
  game_score: {
    label: "Sort It scores",
    what: "Each round's item, whether it went in the right bin, the time and the score.",
  },
  presence: {
    label: "A named place",
    what: "The name of a place you pick under \"Share where you are\", or \"Our table\" when the desk camera sees someone sit down. The name only, and it expires.",
  },
  event_plan: {
    label: "Plans with friends",
    what: "A plan you approve on its card, and a one-word answer (free, busy or another time) to friends' plans. Your calendar itself stays on the box.",
  },
};

const FACES = "Face templates of enrolled people";
const NEVER = [
  "Pictures and video from the cameras",
  "Raw signals: what the cameras and the habit tracker noticed, moment by moment",
  FACES,
  "Your calendar, messages and anything else from connected apps",
];

/** `faceShare`: GET /api/circle/face-share, or null/undefined while face sharing is off on this box (the default). */
export function YouAreSharing({ faceShare }: { faceShare?: FaceShareStatus | null }) {
  const shared = (faceShare?.friends ?? []).filter((f) => f.outgoing.state === "shared");
  const sharing = useFetch((s) => api.circleSharing(s));
  useAtlasEvents(["circle.share.", "circle.place.shared"], () => sharing.reload());
  useRefresh(sharing.reload, REFRESH_MS);

  return (
    <section className={ui.section}>
      <h2 className={ui.sectionTitle}>You are sharing</h2>
      {sharing.error ? (
        sharing.error.notConfigured ? (
          <Empty title="This box can't list what it shares yet">
            Its software is older than this screen. After its next update, what leaves the box for the Circle is listed here.
          </Empty>
        ) : (
          <ErrorBox what="what this box shares" error={sharing.error} onRetry={sharing.reload} />
        )
      ) : !sharing.data ? (
        <Loading what="what this box shares" />
      ) : (
        <Status status={sharing.data} />
      )}
      {faceShare ? (
        <div className={ui.card}>
          <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
            <h3>Your face template</h3>
            <span className={shared.length ? `${ui.tag} ${ui.tagAccent}` : ui.tag}>{shared.length ? "Shared" : "Not shared"}</span>
          </div>
          <p>{faceShare.explanation}</p>
          <p className={`${ui.muted} ${ui.small}`}>
            {shared.length
              ? `Right now: ${shared.map((f) => `${f.display_name} (until ${dayLabel(f.outgoing.expires_at)})`).join(", ")}. `
              : "Right now nobody has it. "}
            It is numbers, never a photo, and only ever your own. Share, renew or take it back on the People screen. The Circle&apos;s database holds it only as
            ciphertext, and can see who shared with whom and when.
          </p>
        </div>
      ) : null}
      <div className={ui.card}>
        <h3>Never leaves this box for the Circle</h3>
        <ul className={ui.list} style={{ gap: 6 }}>
          {NEVER.map((n) => (
            <li key={n}>{faceShare && n === FACES ? "Face templates of anyone else enrolled here, and every photo a template was made from" : n}</li>
          ))}
        </ul>
        <p className={`${ui.muted} ${ui.small}`}>
          {faceShare
            ? "The Circle only ever gets the short lines listed above, and your own face template when you share it with a friend."
            : "The Circle only ever gets the short lines listed above. That holds whatever the settings are."}
        </p>
      </div>
    </section>
  );
}

/** The one-line answer to "is anything leaving?", then the reason. */
function headline(s: CircleSharing): { title: string; on: boolean; why: string } {
  const anything = Array.isArray(s.kinds) && s.kinds.some((k) => k.on);
  if (!s.configured)
    return { title: "Nothing is shared", on: false, why: "This box has not joined a Circle, so nothing leaves it for friends." };
  if (s.database === "not_configured")
    return {
      title: "Nothing is shared",
      on: false,
      why: `This box is set up as ${humanize(s.member_id ?? "a member")} in circle ${s.circle_id ?? ""}, but it is not connected to the Circle's shared database, so nothing leaves it.${
        s.share_summaries ? " Summaries are switched on and would start going once it is connected." : ""
      }`,
    };
  if (s.sharing_now)
    return {
      title: "Summaries are shared",
      on: true,
      why: `Summaries are switched on for ${humanize(s.member_id ?? "this member")} in circle ${s.circle_id ?? ""}. Each one is written on the box first, then sent.`,
    };
  return {
    title: "Summaries are not shared",
    on: false,
    why: anything
      ? "Summaries are switched off on this box (circle.share_summaries), so no habit, recycling or game lines leave it. The rows marked On below still go."
      : "Summaries are switched off on this box (circle.share_summaries), so no habit, recycling or game lines leave it.",
  };
}

function database(s: CircleSharing): string {
  switch (s.database) {
    case "connected":
      return s.last_reached_at ? `Connected (last reached ${relative(s.last_reached_at)})` : "Connected";
    case "offline":
      return s.last_reached_at ? `Offline (last reached ${dayTime(s.last_reached_at)})` : "Offline";
    case "unknown":
      return "Not tried yet since the box started";
    default:
      return "Not connected";
  }
}

function Status({ status: s }: { status: CircleSharing }) {
  const kinds = Array.isArray(s.kinds) ? s.kinds : [];
  const head = headline(s);
  const queued = Number(s.queued) || 0;
  const sent = Number(s.sent) || 0;
  const connectedSetup = s.configured && s.database !== "not_configured";

  return (
    <>
      <div className={ui.card}>
        <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
          <h3>{head.title}</h3>
          <span className={head.on ? `${ui.tag} ${ui.tagAccent}` : ui.tag}>{head.on ? "On" : "Off"}</span>
        </div>
        <p>{head.why}</p>
        {s.database === "offline" ? (
          <p className={ui.nudge} role="status">
            This box can&apos;t reach the Circle&apos;s shared database right now.{" "}
            {queued > 0
              ? `${queued} ${queued === 1 ? "share waits" : "shares wait"} on the box and will go out when it is back online.`
              : "Anything it shares meanwhile waits on the box and goes out when it is back online."}
          </p>
        ) : null}
      </div>

      {connectedSetup ? (
        <>
          <div className={ui.stats}>
            <div className={ui.stat}>
              <p className={ui.statNum}>{queued}</p>
              <p className={ui.statLabel}>waiting on this box</p>
            </div>
            <div className={ui.stat}>
              <p className={ui.statNum}>{sent}</p>
              <p className={ui.statLabel}>sent to the Circle</p>
            </div>
          </div>
          <div className={ui.card}>
            <dl className={ui.kv}>
              <dt>Last sent</dt>
              <dd>{s.last_synced_at ? `${dayTime(s.last_synced_at)} (${relative(s.last_synced_at)})` : "Nothing has been sent yet"}</dd>
              {queued > 0 && s.oldest_queued_at ? (
                <>
                  <dt>Waiting since</dt>
                  <dd>{dayTime(s.oldest_queued_at)}</dd>
                </>
              ) : null}
              <dt>Shared database</dt>
              <dd>{database(s)}</dd>
              {s.place ? (
                <>
                  <dt>Your place</dt>
                  <dd>
                    {s.place.place}, until {time(s.place.expires_at)}
                  </dd>
                </>
              ) : null}
            </dl>
          </div>
        </>
      ) : null}

      <div className={ui.card}>
        <h3>What can go</h3>
        <ul className={ui.list}>
          {kinds.map((k) => (
            <Kind key={k.kind} kind={k} />
          ))}
        </ul>
      </div>
    </>
  );
}

function Kind({ kind: k }: { kind: ShareKindStatus }) {
  const text = KINDS[k.kind] ?? { label: humanize(k.kind), what: "" };
  const counted = k.queued !== null && k.sent !== null;
  return (
    <li>
      <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
        <strong>{text.label}</strong>
        <span className={k.on ? `${ui.tag} ${ui.tagAccent}` : ui.tag}>{k.on ? "On" : "Off"}</span>
      </div>
      {text.what ? <p className={`${ui.muted} ${ui.small}`}>{text.what}</p> : null}
      {counted && (k.on || Number(k.queued) + Number(k.sent) > 0) ? (
        <p className={ui.small}>
          {k.sent} sent, {k.queued} waiting
          {k.last_synced_at ? `; last ${relative(k.last_synced_at)}` : ""}
        </p>
      ) : null}
    </li>
  );
}
