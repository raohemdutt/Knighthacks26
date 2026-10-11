"use client";

import { useState } from "react";
import { ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api } from "@/lib/api";
import { CARD_STATUS, type CardStatus, type LabCard, fmt, shortRef, when } from "@/lib/labBench";
import { useFetch } from "@/lib/useFetch";
import { Tag } from "./Labels";
import b from "./bench.module.css";

/**
 * Experiments: the lab's optimisation loop as a timeline (GET /api/learning/lab/experiments). One card per experiment,
 * refinement pass and benchmark session the repo records, newest first, each with its verdict and the record it came
 * from. Experiments that were queued but never run sit under the timeline, in queue order.
 */

const TONE: Partial<Record<CardStatus, string>> = { won: b.expWon!, lost: b.expLost!, running: b.expRunning!, measured: b.expMeasured!, converged: b.expWon! };

/** The status as a small shape as well as a word, so it is not told by colour alone. */
function StatusIcon({ status }: { status: CardStatus }) {
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
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      {p[status]}
    </svg>
  );
}

function Card({ c }: { c: LabCard }) {
  const [open, setOpen] = useState(false);
  const long = (c.detail?.length ?? 0) > 180;
  return (
    <div className={`${b.exp} ${TONE[c.status] ?? ""}`}>
      <div className={b.expHead}>
        <span className={b.expTitle}>
          {c.kind === "experiment" ? <span className={b.expId}>{c.id}</span> : null}
          {c.title}
        </span>
        <span className={`${b.pill} ${c.status === "won" || c.status === "converged" ? b.pillWon : c.status === "lost" ? b.pillLost : ""}`}>
          <StatusIcon status={c.status} />
          {CARD_STATUS[c.status]}
        </span>
      </div>
      {c.lever ? (
        <span className={b.note}>
          Lever: {c.lever}
          {c.metric ? ` · watching ${c.metric} (${c.better} is better)` : ""}
          {c.node ? ` · ${c.node}` : ""}
        </span>
      ) : null}
      {c.arms ? (
        <span className={b.note}>
          {Object.entries(c.arms)
            .map(([name, what]) => `${name}: ${what}`)
            .join(" · ")}
        </span>
      ) : null}
      {c.verdict ? <b>{c.verdict}</b> : null}
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
      {c.detail ? (
        <span className={b.note}>
          {open || !long ? c.detail : `${c.detail.slice(0, 180)}…`}{" "}
          {long ? (
            <button type="button" className={b.numBtn} onClick={() => setOpen((v) => !v)}>
              {open ? "less" : "more"}
            </button>
          ) : null}
        </span>
      ) : null}
      <span className={`${b.note} ${b.code}`}>
        <Tag kind="measured">record</Tag> {shortRef(c.source.ref)}
      </span>
    </div>
  );
}

export function Experiments() {
  const data = useFetch((signal) => api.labExperiments(signal), "experiments");
  const [filter, setFilter] = useState<"all" | "experiment" | "bench" | "pass">("all");
  if (data.loading && !data.data) return <Loading what="the lab's experiments" />;
  if (data.error && !data.data) return <ErrorBox what="the lab's experiments" error={data.error} onRetry={data.reload} />;
  const t = data.data!;
  const match = (c: LabCard) => filter === "all" || c.kind === filter || (filter === "pass" && c.kind === "merge");
  const dated = t.cards.filter((c) => c.ts && match(c));
  const queued = t.cards.filter((c) => !c.ts && match(c));
  const days: [string, LabCard[]][] = [];
  for (const c of dated) {
    const day = String(c.ts).slice(0, 10);
    const last = days[days.length - 1];
    if (last && last[0] === day) last[1].push(c);
    else days.push([day, [c]]);
  }
  const filters: ["all" | "experiment" | "bench" | "pass", string][] = [
    ["all", `All (${t.cards.length})`],
    ["experiment", `Experiments (${t.experiments.total})`],
    ["bench", `Benchmark sessions (${t.sessions})`],
    ["pass", `Refinement passes (${t.passes})`],
  ];
  return (
    <div className={b.page}>
      <div className={ui.pageHead}>
        <h1>What the lab has tried</h1>
        <p className={b.sub}>The lab improves the teacher in a loop: change one thing, measure it, let the statistics give the verdict. These are its own records, newest first.</p>
      </div>
      <p className={`${b.banner} ${t.experiments.source.kind === "ledger" ? b.bannerOk : b.bannerWatch}`}>
        <span>
          <b>{t.summary}</b> {t.experiments.source.note ?? ""}
        </span>
      </p>
      <div className={ui.chips} role="group" aria-label="Which records to show">
        {filters.map(([key, label]) => (
          <button key={key} type="button" className={`${ui.chip} ${filter === key ? ui.chipOn : ""}`} aria-pressed={filter === key} onClick={() => setFilter(key)}>
            {label}
          </button>
        ))}
      </div>
      <div className={`${b.two} ${b.twoEven}`}>
        <div className={b.card}>
          <h2>On the timeline</h2>
          {days.length === 0 ? <p className={b.note}>No dated record in this group.</p> : null}
          <div className={b.timeline}>
            {days.map(([day, cards]) => (
              <div key={day} style={{ display: "contents" }}>
                <span className={b.day}>{when(day, true)}</span>
                {cards.map((c) => (
                  <Card key={`${c.kind}-${c.id}`} c={c} />
                ))}
              </div>
            ))}
          </div>
        </div>
        <div className={b.card}>
          <h2>Queued, not run yet</h2>
          {queued.length === 0 ? (
            <p className={b.note}>{filter === "all" || filter === "experiment" ? "Nothing is waiting without a date." : "Only experiments wait in the queue."}</p>
          ) : (
            <>
              <p className={b.note}>In queue order. No time is recorded for these, so they are not placed on the timeline.</p>
              <div className={b.list} style={{ gap: 14 }}>
                {queued.map((c) => (
                  <Card key={c.id} c={c} />
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
