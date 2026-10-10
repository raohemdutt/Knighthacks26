"use client";

import Link from "next/link";
import { SessionBreaks } from "@/components/coach/CoachFeedback";
import cf from "@/components/coach/coach.module.css";
import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import type { CoachFeedbackView } from "@/lib/coachFeedback";
import { dayLabel, span } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";

/**
 * Coach: the recent study blocks with their breaks, and "Was that right?" under each one.
 *
 * The coach guesses why a break began from what the box recorded in the two minutes before. Your answer
 * is stored on the box (`day.coach.corrected`), changes that session's review, and counts towards how often
 * the coach is right about each kind of break. After enough answers the box asks, with an approval card,
 * to learn to see that thing from your own blurred pictures.
 */
export default function CoachPage() {
  const view = useFetch((s) => api.coachFeedback(7, s));
  useAtlasEvents(["day.coach.", "learning.", "workspace.action.completed"], () => view.reload());

  return (
    <>
      <div className={ui.pageHead}>
        <h1>Coach</h1>
        <p className={ui.muted}>Tell the coach when it guessed a break wrong. That is how it gets better.</p>
      </div>
      {view.error ? (
        view.error.notConfigured ? (
          <Empty title="The coach is not running on this box">It needs the Day loop: place packs and a plan for the day.</Empty>
        ) : (
          <ErrorBox what="the coach" error={view.error} onRetry={view.reload} />
        )
      ) : !view.data ? (
        <Loading what="recent sessions" />
      ) : (
        <Body v={view.data} reload={view.reload} />
      )}
    </>
  );
}

function Body({ v, reload }: { v: CoachFeedbackView; reload: () => void }) {
  const withBreaks = v.sessions.filter((s) => s.breaks.length > 0);
  const quiet = v.sessions.length - withBreaks.length;
  const pending = v.progress.filter((p) => !p.proposed);

  return (
    <>
      {v.notes.map((n) => (
        <p key={n} className={cf.learning}>
          <strong>{n}</strong>
        </p>
      ))}

      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>Recent breaks</h2>
        {withBreaks.length === 0 ? (
          <Empty title="No breaks to ask about">
            When you leave the {v.words.workplace} in the middle of a planned {v.words.activity} block, the break shows up here with the coach&apos;s guess
            at why, and you can say whether it was right.
          </Empty>
        ) : (
          <ul className={ui.list}>
            {withBreaks.map((s) => (
              <li key={`${s.day}/${s.block_id}`} className={ui.card}>
                <h3>{s.title}</h3>
                <p className={`${ui.small} ${ui.muted}`}>
                  {dayLabel(s.start)} {span(s.start, s.end)}
                  {s.desk_minutes !== null ? ` · ${s.desk_minutes} of ${s.planned_minutes} min at the ${v.words.workplace}` : ""}
                </p>
                <SessionBreaks session={s} options={v.options} onAnswered={reload} />
                <p className={ui.small}>{s.text}</p>
              </li>
            ))}
          </ul>
        )}
        {quiet > 0 ? (
          <p className={`${ui.small} ${ui.muted}`}>
            {quiet} other {quiet === 1 ? "block" : "blocks"} in the last days had no breaks, or {quiet === 1 ? "was" : "were"} not measured.
          </p>
        ) : null}
        {v.suggestions.length > 0 && v.answers > 0 ? (
          <div className={ui.card}>
            <h3>What the coach suggests now</h3>
            {v.suggestions.map((s) => (
              <p key={s}>{s}</p>
            ))}
            <p className={`${ui.small} ${ui.muted}`}>Drawn again from the reasons you gave.</p>
          </div>
        ) : null}
      </section>

      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>How often the coach is right</h2>
        {v.precision.length === 0 ? (
          <Empty title="No answers yet">Answer a break above and the count starts: right guesses out of all the guesses you checked, per kind of break.</Empty>
        ) : (
          <div className={ui.card}>
            {v.precision.map((p) => (
              <div key={p.trigger} className={cf.precision}>
                <span>{p.text}</span>
                <strong>
                  {p.right}/{p.total}
                </strong>
                <div className={cf.track} role="img" aria-label={`${p.right} of ${p.total}`}>
                  <div className={cf.fill} style={{ width: `${Math.round(p.precision * 100)}%` }} />
                </div>
              </div>
            ))}
            <p className={`${ui.small} ${ui.muted}`}>Counted from your own answers ({v.answers} so far), on this box.</p>
          </div>
        )}
      </section>

      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>What Atlas is learning from this</h2>
        {v.learning.length === 0 && pending.length === 0 ? (
          <Empty title="Nothing yet">
            After {v.threshold.min_corrections} answers about one kind of break, with at least {v.threshold.min_negatives} where the coach was wrong, Atlas
            asks to learn to see it from your own blurred pictures. You approve or deny that on the Approvals screen.
          </Empty>
        ) : (
          <ul className={ui.list}>
            {v.learning.map((t) => (
              <li key={t.task} className={ui.card}>
                <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
                  <h3>{t.name}</h3>
                  <span className={`${ui.tag} ${t.learned ? ui.tagAccent : ""}`}>{t.learned ? "learned" : t.stage}</span>
                </div>
                <p>{t.text}</p>
                {t.stage === "waiting for approval" ? (
                  <Link href="/approvals/" className={`${ui.btn} ${ui.btnPrimary}`}>
                    Open the card
                  </Link>
                ) : (
                  <Link href="/teach/" className={ui.btn}>
                    See it on Teach
                  </Link>
                )}
              </li>
            ))}
            {pending.map((p) => (
              <li key={p.trigger} className={ui.card}>
                <h3>{p.name}</h3>
                <p className={ui.muted}>
                  {p.answers} of {p.needed} answers, {p.negatives} of {p.negatives_needed} where the coach was wrong. Then Atlas asks to learn it.
                </p>
              </li>
            ))}
          </ul>
        )}
      </section>
    </>
  );
}
