"use client";

import { useState } from "react";
import { ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api } from "@/lib/api";
import { CARD_STATUS, type CardStatus, type LabCard, fmt, shortRef, when } from "@/lib/labBench";
import { useFetch } from "@/lib/useFetch";
import b from "./bench.module.css";
import x from "./experiments.module.css";

/**
 * Experiments: the lab's optimisation loop as one compact timeline (GET /api/learning/lab/experiments). One line per
 * experiment, refinement pass and benchmark session the repo records, grouped by day, newest first; a tap opens the
 * full card with its numbers and the record it came from. Every word on a line comes from the card's own fields.
 */

type Filter = "all" | "experiment" | "bench" | "pass";

const TONE: Partial<Record<CardStatus, string>> = { won: x.won!, converged: x.won!, lost: x.lost!, running: x.running!, measured: x.measured! };

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

const KIND_WORD: Record<LabCard["kind"], string> = { experiment: "Experiment", bench: "Benchmark session", pass: "Refinement pass", merge: "Merge pass" };

/** What kind of record a line is, as a shape: a flask, bars, a loop, two paths joining. */
function KindIcon({ kind }: { kind: LabCard["kind"] }) {
  const p: Record<LabCard["kind"], React.ReactNode> = {
    experiment: <path d="M9.5 3.5h5M10.5 3.5v6L5 19.5h14l-5.5-10v-6M7.5 15h9" />,
    bench: <path d="M5 19V10M12 19V5M19 19v-6M3.5 19.5h17" />,
    pass: <path d="M4 12a8 8 0 1 0 3-6.2M4 4v4h4" />,
    merge: <path d="M6 4v5c0 3 6 4 6 7v4M18 4v5c0 3-6 4-6 7" />,
  };
  return (
    <span className={x.kind} title={KIND_WORD[kind]}>
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        {p[kind]}
      </svg>
      <span className={x.srOnly}>{KIND_WORD[kind]}</span>
    </span>
  );
}

/* ---------- words from the card's own fields ---------- */

const times = (n: number) => (n === 1 ? "once" : n === 2 ? "twice" : `${n} times`);

/**
 * The gateway's "Measured, no verdict: ..." line in plain words. The counts and the repeats needed are read from the
 * line itself: "1 run(s): under 3 repeats, and under the 5 that verdict() needs" becomes "measured once; the lab needs
 * 3 repeats before it calls a verdict". A line in any other shape is shown as it came.
 */
export function plainVerdict(v: string | null | undefined): string | null {
  if (!v) return null;
  const m = /^Measured, no verdict:\s*(.*)$/i.exec(v);
  if (!m) return v;
  const rest = m[1]!;
  const counts = [...rest.matchAll(/(\d+)\s+(?:run|spawn)s?(?:\(s\))?/gi)].map((g) => Number(g[1]));
  const need = /under (\d+) repeats/i.exec(rest)?.[1] ?? /under the (\d+) that verdict\(\) needs/i.exec(rest)?.[1];
  if (!counts.length || !need) return v;
  const lo = Math.min(...counts);
  const hi = Math.max(...counts);
  const how = lo === hi ? times(lo) : `${lo} to ${hi} times`;
  const per = /per batch size/i.test(rest) ? " per batch size" : "";
  return `measured ${how}${per}; the lab needs ${need} repeats before it calls a verdict`;
}

/** "teacher.joules_per_token" -> "J/token", "teacher.decode_tok_s" -> "decode tok/s", "trainer.samples_per_s" -> "samples/s". */
export function metricShort(metric: string | null | undefined): string {
  if (!metric) return "";
  const last = metric.split(".").pop() ?? metric;
  return last
    .replace(/^joules(?=_|$)/, "J")
    .replace(/_per_/g, "/")
    .replace(/_s$/, "/s")
    .replace(/_/g, " ");
}

const num = (v: number) => fmt(v, Number.isInteger(v) ? 0 : undefined);
const COUNT_UNITS = ["changes", "findings", "questions", "rows"];

/** The one phrase after the title on a timeline line. */
export function cardPhrase(c: LabCard): string {
  const status = CARD_STATUS[c.status].toLowerCase();
  if (c.kind === "bench") {
    const rows = c.numbers.find((n) => n.unit === "rows" && typeof n.value === "number");
    const parts = [status];
    if (rows && typeof rows.value === "number") parts.push(`${num(rows.value)} row${rows.value === 1 ? "" : "s"}`);
    if (c.status === "measured" && /no verdict/i.test(c.verdict ?? "")) parts.push("no verdict yet");
    return parts.join(", ");
  }
  if (c.kind === "experiment") {
    const exp = c.numbers.find((n) => n.unit === "%" && typeof n.value === "number" && /^expected/i.test(n.label));
    if (exp && typeof exp.value === "number") return `${status}, expected ${exp.value > 0 ? "+" : ""}${num(exp.value)}% ${metricShort(c.metric)}`.trim();
    if (c.verdict) return `${status}: ${c.verdict}`;
    return c.metric ? `${status}, watching ${metricShort(c.metric)}` : status;
  }
  const counts = c.numbers
    .filter((n) => typeof n.value === "number" && n.unit && COUNT_UNITS.includes(n.unit))
    .map((n) => `${num(n.value as number)} ${n.value === 1 ? n.unit!.replace(/s$/, "") : n.unit}`);
  const state = c.status === "open" ? "not converged yet" : status;
  return counts.length ? `${counts.join(", ")}; ${state}` : state;
}

/** The day a card belongs to, in the reader's own time zone: a timed card by its local date, a day-only card as given. */
function dayKey(c: LabCard): string {
  const ts = String(c.ts);
  if (c.day_only || /^\d{4}-\d{2}-\d{2}$/.test(ts)) return ts.slice(0, 10);
  const d = new Date(ts);
  if (Number.isNaN(d.getTime())) return ts.slice(0, 10);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

function clockOf(c: LabCard): string {
  if (!c.ts || c.day_only || /^\d{4}-\d{2}-\d{2}$/.test(c.ts)) return "";
  const d = new Date(c.ts);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

const stamp = (c: LabCard) => {
  const t = c.ts ? new Date(c.day_only ? `${c.ts}T00:00:00` : c.ts).getTime() : Number.NaN;
  return Number.isNaN(t) ? 0 : t;
};

/* ---------- one line, and the card behind it ---------- */

function Full({ c }: { c: LabCard }) {
  const [more, setMore] = useState(false);
  const long = (c.detail?.length ?? 0) > 280;
  const verdict = plainVerdict(c.verdict);
  return (
    <div className={x.full}>
      <div className={x.fullHead}>
        <span className={`${b.pill} ${c.status === "won" || c.status === "converged" ? b.pillWon : c.status === "lost" ? b.pillLost : ""}`}>
          <StatusIcon status={c.status} />
          {CARD_STATUS[c.status]}
        </span>
        {verdict ? <span className={x.verdict}>{verdict}</span> : null}
      </div>
      {c.hypothesis ? <p className={x.line}>{c.hypothesis}</p> : null}
      {c.lever ? (
        <p className={x.muted}>
          Lever: {c.lever}
          {c.metric ? ` · watching ${c.metric} (${c.better} is better)` : ""}
          {c.node ? ` · ${c.node}` : ""}
        </p>
      ) : null}
      {c.arms ? (
        <p className={x.muted}>
          {Object.entries(c.arms)
            .map(([name, what]) => `${name}: ${what}`)
            .join(" · ")}
        </p>
      ) : null}
      {c.numbers.length ? (
        <div className={b.expNums}>
          {c.numbers.map((n) => (
            <span key={n.label}>
              {n.label}: <b>{typeof n.value === "number" ? num(n.value) : "none"}</b> {n.unit && !COUNT_UNITS.includes(n.unit) ? n.unit : ""}
              {n.low_confidence ? <span className={b.low}> low confidence</span> : null}
            </span>
          ))}
        </div>
      ) : null}
      {c.detail ? (
        <p className={x.muted}>
          {more || !long ? c.detail : `${c.detail.slice(0, 280)}…`}{" "}
          {long ? (
            <button type="button" className={b.numBtn} onClick={() => setMore((v) => !v)}>
              {more ? "less" : "more"}
            </button>
          ) : null}
        </p>
      ) : null}
      {c.source.ref ? (
        <span className={x.recordRow}>
          <span className={x.muted}>From</span>
          <code className={x.record} title={c.source.ref}>
            {shortRef(c.source.ref)}
          </code>
        </span>
      ) : null}
    </div>
  );
}

function Row({ c }: { c: LabCard }) {
  const [open, setOpen] = useState(false);
  const clock = clockOf(c);
  return (
    <li className={`${x.item} ${TONE[c.status] ?? ""} ${open ? x.itemOpen : ""}`}>
      <button type="button" className={x.row} aria-expanded={open} onClick={() => setOpen((v) => !v)}>
        <span className={x.time}>{clock}</span>
        <KindIcon kind={c.kind} />
        <span className={x.text}>
          {clock ? <span className={x.timeInline}>{clock}</span> : null}
          <span className={x.title}>
            {c.kind === "experiment" ? <span className={x.id}>{c.id}</span> : null}
            {c.title}
          </span>
          <span className={x.phrase}>{cardPhrase(c)}</span>
        </span>
        <svg className={x.chev} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M9 6l6 6-6 6" />
        </svg>
      </button>
      {open ? <Full c={c} /> : null}
    </li>
  );
}

export function Experiments() {
  const data = useFetch((signal) => api.labExperiments(signal), "experiments");
  const [filter, setFilter] = useState<Filter>("all");
  if (data.loading && !data.data) return <Loading what="the lab's experiments" />;
  if (data.error && !data.data) return <ErrorBox what="the lab's experiments" error={data.error} onRetry={data.reload} />;
  const t = data.data!;
  const match = (c: LabCard) => filter === "all" || c.kind === filter || (filter === "pass" && c.kind === "merge");
  // Newest first by the card's own time; a day-only card sorts at the start of its day, after the timed ones.
  const dated = t.cards
    .map((c, i) => ({ c, i }))
    .filter(({ c }) => c.ts && match(c))
    .sort((p, q) => stamp(q.c) - stamp(p.c) || p.i - q.i)
    .map(({ c }) => c);
  const undated = t.cards.filter((c) => !c.ts && match(c));
  const days: [string, LabCard[]][] = [];
  for (const c of dated) {
    const day = dayKey(c);
    const last = days[days.length - 1];
    if (last && last[0] === day) last[1].push(c);
    else days.push([day, [c]]);
  }
  const src = t.experiments.source;
  const received = src.from === "imported" && src.received_at ? new Date(src.received_at) : null;
  const sameDay = received ? received.toDateString() === new Date().toDateString() : false;
  const filters: [Filter, string][] = [
    ["all", `All (${t.cards.length})`],
    ["experiment", `Experiments (${t.experiments.total})`],
    ["bench", `Benchmark sessions (${t.sessions})`],
    ["pass", `Refinement passes (${t.passes})`],
  ];
  return (
    <div className={b.page}>
      <div className={ui.pageHead}>
        <h1>What the lab has tried</h1>
        <p className={b.sub}>The lab improves the teacher in a loop: change one thing, measure it, let the statistics give the verdict. These are its own records, newest first. Tap a line for the full record.</p>
      </div>
      <p className={x.summary}>
        {t.summary} {t.experiments.source.note ?? ""}
        {received && !Number.isNaN(received.getTime()) ? (
          <>
            {" "}
            Ledger received from the lab {sameDay ? "at" : "on"} {sameDay ? received.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }) : when(src.received_at)}
            {typeof src.rows === "number" ? `, ${fmt(src.rows, 0)} row${src.rows === 1 ? "" : "s"}` : ""}.
          </>
        ) : null}
      </p>
      <div className={ui.chips} role="group" aria-label="Which records to show">
        {filters.map(([key, label]) => (
          <button key={key} type="button" className={`${ui.chip} ${filter === key ? ui.chipOn : ""}`} aria-pressed={filter === key} onClick={() => setFilter(key)}>
            {label}
          </button>
        ))}
      </div>
      <div className={x.panel}>
        {days.length === 0 && undated.length === 0 ? <p className={x.muted}>No record in this group.</p> : null}
        {days.map(([day, cards]) => (
          <section key={day} className={x.day} aria-label={when(day, true)}>
            <h3 className={x.dayHead}>
              {when(day, true)} <span>{cards.length}</span>
            </h3>
            <ul className={x.list}>
              {cards.map((c) => (
                <Row key={`${c.kind}-${c.id}`} c={c} />
              ))}
            </ul>
          </section>
        ))}
        {undated.length ? (
          <section className={x.day} aria-label="Queued, no time recorded">
            <h3 className={x.dayHead}>
              Queued, no time recorded <span>{undated.length}</span>
            </h3>
            <ul className={x.list}>
              {undated.map((c) => (
                <Row key={`${c.kind}-${c.id}`} c={c} />
              ))}
            </ul>
          </section>
        ) : null}
      </div>
    </div>
  );
}
