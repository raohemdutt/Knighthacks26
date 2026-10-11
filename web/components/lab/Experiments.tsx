"use client";

import { type CardStatus, type LabCard, fmt, shortRef } from "@/lib/labBench";
import { CARD_WORDS, plainVerdict } from "@/lib/labWords";
import b from "./bench.module.css";
import { Fold } from "./Story";

/**
 * One record of the lab's own work, opened in the timeline of "The model" tab (components/lab/Model.tsx): an
 * experiment, a review pass of its own code, or a speed test (GET /api/learning/lab/experiments). The row in the
 * timeline says when, what and how it ended; this is what a tap on it shows. The record it was read from sits in a
 * fold, raw.
 */

/** The status as a small shape as well as a word, so it is not told by colour alone. */
export function StatusIcon({ status }: { status: CardStatus }) {
  const p: Record<CardStatus, React.ReactNode> = {
    queued: <circle cx="12" cy="12" r="7" strokeDasharray="3 4" />,
    running: <path d="M8 5l11 7-11 7z" />,
    won: <path d="M5 12.5l4.5 4.5L19 7.5" />,
    converged: <path d="M5 12.5l4.5 4.5L19 7.5" />,
    lost: <path d="M6 6l12 12M18 6L6 18" />,
    inconclusive: <path d="M5 12h14" />,
    dropped: <path d="M5 5l14 14M12 5a7 7 0 1 0 0 14 7 7 0 0 0 0-14" />,
    measured: <path d="M5 19V9M12 19V5M19 19v-7" />,
    open: <path d="M4 12a8 8 0 1 0 3-6.2M4 4v4h4" />,
  };
  return (
    <svg viewBox="0 0 24 24" width={16} height={16} fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {p[status]}
    </svg>
  );
}

export function CardBody({ c }: { c: LabCard }) {
  return (
    <>
      {c.verdict ? <b>{plainVerdict(c.verdict)}</b> : <b>{CARD_WORDS[c.status] ?? c.status}: no result yet.</b>}
      {c.hypothesis ? <span className={b.note}>The idea: {c.hypothesis}</span> : null}
      {c.lever ? (
        <span className={b.note}>
          What was changed: {c.lever.replace(/^batch size /, "requests at once: ")}
          {c.metric ? `. Watching ${c.metric.replace(/[._]/g, " ")} (${c.better} is better)` : ""}
          {c.node ? `, on ${c.node}` : ""}.
        </span>
      ) : null}
      {c.arms ? (
        <span className={b.note}>
          {Object.entries(c.arms)
            .map(([name, what]) => `${name}: ${what}`)
            .join(" · ")}
        </span>
      ) : null}
      {c.numbers.length ? (
        <div className={b.expNums}>
          {c.numbers.map((n) => (
            <span key={n.label}>
              {n.label}: <b>{typeof n.value === "number" ? fmt(n.value, Number.isInteger(n.value) ? 0 : undefined) : "none"}</b> {n.unit && !["changes", "findings", "questions", "rows"].includes(n.unit) ? n.unit : ""}
              {n.low_confidence ? <span className={b.low}> low confidence</span> : null}
            </span>
          ))}
        </div>
      ) : null}
      <Fold summary="Where this comes from">
        {c.detail ? <span className={b.note}>{c.detail}</span> : null}
        <span className={`${b.note} ${b.code}`}>{shortRef(c.source.ref)}</span>
        {c.citation ? <span className={b.note}>{c.citation}</span> : null}
      </Fold>
    </>
  );
}
