"use client";

import Link from "next/link";
import { useState } from "react";
import { Empty, ErrorBox, Loading, WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type DayNow, type DayPlan, type DayReplan, type DayReview, type DayStatus, type PlanMove, type PlanSource } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { cameraName, dayLabel, humanize, relative, time, timeOrDay } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";

/**
 * Now: the day Atlas runs unasked (contexts/day). What the cameras see, where the plan has you against where
 * the box thinks you are, the next thing to leave for, today's meters, the plan with its reasons, a pending
 * re-plan, and the evening review. Live from the day.* events on /ws.
 */

const RISK_TEXT: Record<DayStatus["risk"], string> = {
  ok: "On time",
  leave_now: "Leave now",
  late: "Late",
  infeasible: "You will miss it",
};

const ORIGIN_TEXT: Record<string, string> = {
  canvas: "Canvas",
  calendar: "Outlook calendar",
  campus: "campus events",
  groupme: "GroupMe",
  "desk camera": "desk camera",
  watch: "your watch",
  circle: "Circle",
  review: "yesterday's review",
};

const asApiError = (err: unknown) => (err instanceof ApiError ? err : new ApiError(0, err instanceof Error ? err.message : String(err)));

export default function NowPage() {
  const now = useFetch((s) => api.dayNow(s));
  const patterns = useFetch((s) => api.dayPatterns(s));
  useAtlasEvents(["day.", "workspace.action.completed", "experience.approval.decided"], () => now.reload());
  useAtlasEvents(["habits.day.closed"], () => patterns.reload());

  return (
    <>
      <div className={ui.pageHead}>
        <h1>Now</h1>
        {now.data ? (
          <p className={ui.muted}>
            {dayLabel(`${now.data.day}T12:00:00`)}, as of {timeOrDay(now.data.now)}. Atlas plans the day, watches it and reviews it without being asked.
          </p>
        ) : null}
      </div>

      {now.error ? (
        now.error.notConfigured ? (
          <Empty title="The Day loop is not on this box">It needs the place packs (config/places) so it can plan where to be and how long the walks are.</Empty>
        ) : (
          <ErrorBox what="the day" error={now.error} onRetry={now.reload} />
        )
      ) : !now.data ? (
        <Loading what="the day" />
      ) : (
        <DayView d={now.data} reload={now.reload} />
      )}

      {patterns.data && patterns.data.patterns.length > 0 ? (
        <section className={ui.section}>
          <h2 className={ui.sectionTitle}>Patterns</h2>
          <p className={`${ui.muted} ${ui.small}`}>Counted from the habit history on this box, nothing fitted.</p>
          <ul className={ui.list}>
            {patterns.data.patterns.map((p) => (
              <li key={p.text} className={ui.card}>
                <p>{p.text}</p>
              </li>
            ))}
          </ul>
        </section>
      ) : null}
    </>
  );
}

function DayView({ d, reload }: { d: DayNow; reload: () => void }) {
  const s = d.status;
  return (
    <>
      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>What Atlas sees</h2>
        {d.last_seen ? (
          <div className={ui.card}>
            <p>{d.last_seen.caption ?? `${d.last_seen.labels.join(", ") || "Activity"} on the ${cameraName(d.last_seen.camera_id).toLowerCase()}`}</p>
            <p className={`${ui.muted} ${ui.small}`}>
              {cameraName(d.last_seen.camera_id)}, {timeOrDay(d.last_seen.start)}
              {d.last_seen.end ? ` to ${time(d.last_seen.end)}` : ", still going"}
            </p>
          </div>
        ) : (
          <Empty title="No camera activity yet">The last thing a camera saw appears here.</Empty>
        )}
      </section>

      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>Where you should be</h2>
        {!s ? (
          <Empty title={d.plan ? "Waiting for the first status" : "No plan yet"}>
            {d.plan ? "The status updates as the cameras and connectors report." : "Atlas plans the day at wake time, or as soon as it starts after. You can also make one now, below."}
          </Empty>
        ) : (
          <StatusCard s={s} />
        )}
      </section>

      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>Today so far</h2>
        {s ? <Meters s={s} /> : <p className={`${ui.muted} ${ui.small}`}>Meals, desk time, recycling and last night&apos;s sleep appear once the day has a status.</p>}
      </section>

      {d.replan ? (
        <section className={ui.section}>
          <h2 className={ui.sectionTitle}>Suggested re-plan</h2>
          <ReplanCard r={d.replan} />
        </section>
      ) : null}

      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>{d.plan?.type === "day.replan.proposed" ? "Today's plan (re-planned)" : "Today's plan"}</h2>
        {d.plan ? <PlanCard p={d.plan} committed={d.committed} reload={reload} /> : <MakePlan reload={reload} />}
      </section>

      {d.review ? (
        <section className={ui.section}>
          <h2 className={ui.sectionTitle}>The evening review</h2>
          <ReviewCard r={d.review} />
        </section>
      ) : d.yesterday ? (
        <section className={ui.section}>
          <h2 className={ui.sectionTitle}>Yesterday&apos;s review</h2>
          <ReviewCard r={d.yesterday} />
        </section>
      ) : null}

      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>Friends</h2>
        {d.friends.length === 0 ? (
          <p className={`${ui.muted} ${ui.small}`}>Nobody in your Circle is sharing a place right now.</p>
        ) : (
          <ul className={ui.list}>
            {d.friends.map((f) => (
              <li key={f.member_id} className={ui.card}>
                <p>
                  <strong>{humanize(f.member_id)}</strong> is at {f.place}
                </p>
                <p className={`${ui.muted} ${ui.small}`}>shared {timeOrDay(f.ts)}, until {time(f.expires_at)}</p>
              </li>
            ))}
          </ul>
        )}
      </section>
    </>
  );
}

function StatusCard({ s }: { s: DayStatus }) {
  const behind = s.drift_minutes > 0;
  const ahead = s.drift_minutes < 0;
  const warn = s.risk === "late" || s.risk === "infeasible";
  return (
    <div className={ui.card}>
      <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
        <h3>{s.block_title ? s.block_title : s.next_block ? `Free until ${time(s.next_block.start)}` : "Nothing planned right now"}</h3>
        {behind ? <span className={ui.tag}>{s.drift_minutes} min behind</span> : ahead ? <span className={`${ui.tag} ${ui.tagAccent}`}>{-s.drift_minutes} min ahead</span> : <span className={`${ui.tag} ${ui.tagAccent}`}>on plan</span>}
      </div>
      <dl className={ui.kv}>
        <dt>Should be</dt>
        <dd>{s.should_be_name ?? (s.block_id ? "wherever you are (online)" : s.next_block?.place_name ? `${s.next_block.place_name} by ${time(s.next_block.start)}` : "anywhere")}</dd>
        <dt>Seen</dt>
        <dd>
          {s.seen_at_name ? (
            <>
              {s.seen_at_name} <span className={ui.muted}>({s.seen_how}, {time(s.seen_ts)})</span>
            </>
          ) : (
            <span className={ui.muted}>not seen yet today</span>
          )}
        </dd>
        {s.block_end ? (
          <>
            <dt>Until</dt>
            <dd>{time(s.block_end)}</dd>
          </>
        ) : null}
        {s.next_block && s.block_id ? (
          <>
            <dt>Then</dt>
            <dd>
              {s.next_block.title} at {time(s.next_block.start)}
            </dd>
          </>
        ) : null}
      </dl>
      {s.next_event ? (
        <p className={warn ? `${ui.nudge}` : ""}>
          <strong>{RISK_TEXT[s.risk]}.</strong> Next: {s.next_event.title} at {time(s.next_event.start)}
          {s.next_event.place_name ? ` at ${s.next_event.place_name}` : ""}
          {s.next_event.walk_minutes > 0 && s.next_event.leave_by ? ` (${s.next_event.walk_minutes} min walk, leave by ${time(s.next_event.leave_by)})` : ""}.
        </p>
      ) : null}
      <p className={`${ui.muted} ${ui.small}`}>
        {s.reason ? `${s.reason.charAt(0).toUpperCase()}${s.reason.slice(1)}, ` : ""}
        {relative(s.now)}. Deadline risk {Math.round(s.deadline_risk * 100)}%.
      </p>
    </div>
  );
}

function Meters({ s }: { s: DayStatus }) {
  const m = s.meters;
  return (
    <div className={ui.stats}>
      <div className={ui.stat}>
        <p className={ui.statNum}>{m.meals}</p>
        <p className={ui.statLabel}>meals seen</p>
      </div>
      <div className={ui.stat}>
        <p className={ui.statNum}>{m.desk_minutes}</p>
        <p className={ui.statLabel}>minutes at the desk</p>
      </div>
      <div className={ui.stat}>
        <p className={ui.statNum}>{m.recycled}</p>
        <p className={ui.statLabel}>recycled today</p>
      </div>
      <div className={ui.stat}>
        <p className={ui.statNum}>{m.sleep_hours === null ? "–" : `${m.sleep_hours.toFixed(1)} h`}</p>
        <p className={ui.statLabel}>{m.sleep_hours === null ? "sleep not measured" : m.sleep_source === "watch" ? "slept (watch)" : "slept (desk camera)"}</p>
      </div>
    </div>
  );
}

function Blocks({ moves }: { moves: PlanMove[] }) {
  return (
    <ul className={ui.list}>
      {moves.map((m) => (
        <li key={m.id} className={`${ui.row} ${ui.between} ${ui.wrap}`}>
          <span>
            {m.fixed ? <span className={ui.tag}>on the calendar</span> : null} {m.title}
            {m.with_friend ? ` with ${m.with_friend}` : ""}
          </span>
          <span className={ui.muted}>
            {time(m.start)} to {time(m.end)}
          </span>
        </li>
      ))}
    </ul>
  );
}

function Sources({ sources, missing }: { sources: PlanSource[]; missing: string[] }) {
  const named = sources.filter((s) => s.kind !== "config" && s.kind !== "place");
  const pack = sources.find((s) => s.kind === "place");
  return (
    <div className={ui.sources}>
      {named.map((s) => (
        <div key={`${s.kind}:${s.id}`} className={ui.source}>
          <span>{s.text}</span>
          <span className={`${ui.muted} ${ui.small}`}>
            {s.origin ? ORIGIN_TEXT[s.origin] ?? s.origin : s.kind} · {s.id}
          </span>
        </div>
      ))}
      {pack ? <p className={`${ui.muted} ${ui.small}`}>{pack.text}.</p> : null}
      {missing.length > 0 ? <p className={`${ui.muted} ${ui.small}`}>Not available: {missing.join("; ")}.</p> : null}
    </div>
  );
}

function PlanCard({ p, committed, reload }: { p: DayPlan | DayReplan; committed: boolean; reload: () => void }) {
  const [open, setOpen] = useState(false);
  const chosen = p.moves.filter((m) => !m.fixed);
  const isPlan = p.type === "day.plan.proposed";
  return (
    <div className={ui.card}>
      <p className={ui.breakWords}>{isPlan ? p.text : p.why}</p>
      <p className={`${ui.muted} ${ui.small}`}>
        {p.outcome.summary} {p.outcome.points} points. {isPlan && p.by === "person" ? "Your own moves." : "Chosen by Atlas"}
        {committed ? ", approved and on your calendar." : p.action_id ? ", waiting for your OK." : "."}
      </p>
      <Blocks moves={open ? p.moves : chosen.slice(0, 4)} />
      {!open && p.moves.length > 4 ? (
        <button type="button" className={ui.btn} onClick={() => setOpen(true)}>
          Show the whole day ({p.moves.length} blocks)
        </button>
      ) : null}
      {isPlan ? (
        <>
          <h3>Because</h3>
          <Sources sources={p.sources} missing={p.missing} />
        </>
      ) : null}
      <div className={ui.actions}>
        {p.action_id && !committed ? (
          <Link href="/approvals/" className={`${ui.btn} ${ui.btnPrimary}`}>
            Approve in Approvals
          </Link>
        ) : null}
        <Link href="/dayrun/?plan=day" className={ui.btn}>
          Change it in Day Run
        </Link>
      </div>
      <MakePlan reload={reload} small />
    </div>
  );
}

function ReplanCard({ r }: { r: DayReplan }) {
  const chosen = r.moves.filter((m) => !m.fixed);
  return (
    <div className={ui.card}>
      <p className={ui.nudge}>{r.why}</p>
      <p className={`${ui.muted} ${ui.small}`}>
        {r.asked ? "You asked for it." : "Drift triggered it."} {r.outcome.summary} {r.outcome.points} points.
      </p>
      <Blocks moves={chosen.slice(0, 5)} />
      {r.action_id ? (
        <Link href="/approvals/" className={`${ui.btn} ${ui.btnPrimary}`}>
          Approve in Approvals
        </Link>
      ) : null}
    </div>
  );
}

function ReviewCard({ r }: { r: DayReview }) {
  return (
    <div className={ui.card}>
      <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
        <h3>{r.score === null ? "Nothing could be measured" : `You followed ${r.score}%`}</h3>
        <span className={`${ui.tag} ${r.stars > 0 ? ui.tagAccent : ""}`}>
          {"★".repeat(r.stars) || "no stars"} · {r.points} points
        </span>
      </div>
      <p className={ui.breakWords}>{r.text}</p>
      {r.followed.length > 0 ? <p className={ui.small}>Happened: {r.followed.join(", ")}.</p> : null}
      {r.slipped.length > 0 ? <p className={ui.small}>Slipped: {r.slipped.join(", ")}.</p> : null}
      <p className={ui.nudge}>Tomorrow: {r.change_tomorrow}</p>
      <p className={`${ui.muted} ${ui.small}`}>
        Streak {r.streak} {r.streak === 1 ? "day" : "days"} with a star. The plan would have earned {r.plan_stars}.
      </p>
    </div>
  );
}

function MakePlan({ reload, small }: { reload: () => void; small?: boolean }) {
  const [busy, setBusy] = useState<"plan" | "here" | null>(null);
  const [error, setError] = useState<ApiError | undefined>();
  const run = (body: { from_here?: boolean }, what: "plan" | "here") => {
    setBusy(what);
    setError(undefined);
    api
      .dayMakePlan(body)
      .then(() => reload())
      .catch((err: unknown) => setError(asApiError(err)))
      .finally(() => setBusy(null));
  };
  return (
    <div className={ui.section}>
      {!small ? <p className={ui.muted}>Atlas plans at wake time from what it already knows. It is a starting point: remake it, re-plan from where you are, or edit it in Day Run.</p> : null}
      <div className={ui.actions}>
        <button type="button" className={`${ui.btn} ${small ? "" : ui.btnPrimary}`} disabled={busy !== null} onClick={() => run({}, "plan")}>
          {busy === "plan" ? "Planning…" : small ? "Plan again" : "Plan my day now"}
        </button>
        <button type="button" className={ui.btn} disabled={busy !== null} onClick={() => run({ from_here: true }, "here")}>
          {busy === "here" ? "Planning…" : "Re-plan from here"}
        </button>
      </div>
      {error ? <WriteError error={error} /> : null}
    </div>
  );
}
