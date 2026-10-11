"use client";

import { useEffect, useState } from "react";
import { ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { STAT_GROUPS, type StatFigure, pct, statValue } from "@/lib/stats";
import { useFetch } from "@/lib/useFetch";
import s from "./stats.module.css";

const POLL_MS = 10_000;

function when(iso: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "" : d.toLocaleString([], { weekday: "short", hour: "numeric", minute: "2-digit" });
}

/** One number, and on a tap the record it came from (or why there is none). */
function Tile({ label, f }: { label: string; f: StatFigure }) {
  const [open, setOpen] = useState(false);
  const none = f.value === null;
  return (
    <button type="button" className={`${ui.stat} ${s.tile} ${none ? s.none : ""}`} aria-expanded={open} onClick={() => setOpen((v) => !v)}>
      <span className={ui.statLabel}>{label}</span>
      <span className={none ? s.noneNum : ui.statNum}>{statValue(f)}</span>
      {f.low_confidence ? <span className={s.low}>low confidence</span> : null}
      {open ? (
        <span className={s.source}>
          {none ? f.reason : `${f.source}${f.ts ? ` · measured ${when(f.ts)}` : ""}${f.node ? ` · ${f.node}` : ""}`}
        </span>
      ) : (
        <span className={s.hint}>{none ? "Why not?" : "Where from?"}</span>
      )}
    </button>
  );
}

/**
 * Box stats (R5): how fast the box and the lab are, from GET /runtime/stats. Every number is a record the box or the
 * lab already holds (a lab metrics line, a ledger row, telemetry events, the detector's own timing); tap one to see
 * which. A number nothing has recorded says so instead of showing a guess. Refreshes every 10 s and on new
 * telemetry or learning events.
 */
export default function StatsPage() {
  const stats = useFetch((signal) => api.runtimeStats(signal));
  const reload = stats.reload;
  useAtlasEvents(["telemetry.", "learning."], reload);
  useEffect(() => {
    const id = window.setInterval(reload, POLL_MS);
    return () => window.clearInterval(id);
  }, [reload]);

  if (stats.loading && !stats.data) return <Loading what="the box's numbers" />;
  if (stats.error && !stats.data) return <ErrorBox what="the box's numbers" error={stats.error} onRetry={reload} />;
  const data = stats.data!;
  const evals = data.eval_scores;

  return (
    <>
      <div className={ui.pageHead}>
        <h1>Box stats</h1>
        <p className={ui.muted}>How fast Atlas sees, answers and learns. Every number is one the box or the lab recorded; tap it to see which.</p>
      </div>

      {STAT_GROUPS.map((g) => (
        <section key={g.title} className={s.group}>
          <h2 className={ui.sectionTitle}>
            {g.title} <span className={s.where}>{g.where}</span>
          </h2>
          <div className={ui.stats}>
            {g.figures.map((x) => (
              <Tile key={x.key} label={x.label} f={data[x.key]} />
            ))}
          </div>
        </section>
      ))}

      <section className={s.group}>
        <h2 className={ui.sectionTitle}>Before and after, per model on the box</h2>
        <div className={ui.card}>
          {evals.models.length === 0 ? (
            <p className={ui.muted}>{evals.reason ?? "No model has been deployed from the lab yet."}</p>
          ) : (
            <div className={s.tableWrap}>
              <table className={ui.table}>
                <thead>
                  <tr>
                    <th scope="col">Model</th>
                    <th scope="col" className={ui.num}>
                      Before
                    </th>
                    <th scope="col" className={ui.num}>
                      After
                    </th>
                    <th scope="col" className={ui.num}>
                      Teacher
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {evals.models.map((m) => (
                    <tr key={m.model_version}>
                      <td>
                        <b>{m.model_version}</b>
                        <span className={s.sub}>{m.reason ?? [m.task, m.decision === "deploy" ? "cleared by the gate" : m.decision, when(m.deployed_ts)].filter(Boolean).join(" · ")}</span>
                      </td>
                      <td className={ui.num}>{pct(m.before)}</td>
                      <td className={ui.num}>{pct(m.after)}</td>
                      <td className={ui.num}>{pct(m.teacher)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </section>
    </>
  );
}
