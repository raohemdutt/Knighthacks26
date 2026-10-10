"use client";

import { useState } from "react";
import { ApiError, api } from "@/lib/api";
import { useAtlas } from "@/lib/atlas";
import { EvidenceLine, Id, Pill, Sev, useOpen } from "./bits";
import s from "./spot.module.css";
import type { Risk, TriageItem, WorkOrder } from "./types";

const PART_COLOR: Record<string, string> = {
  severity: "var(--sev-critical)",
  criticality: "var(--sev-high)",
  repeats: "var(--sev-medium)",
  "not inspected": "#c792ea",
  stale: "var(--sev-low)",
  "failure rate": "var(--bad)",
  "seen since": "var(--ok)",
};

/**
 * One-tap operator actions. Each raises an approval card on the box (agent.action.proposed); the card can be approved
 * right here or on the Approvals screen. Only then is a work-order file written, on this box and nowhere else.
 */
export function OrderActions({ assetId, anomalyId, orders, compact }: { assetId: string | null; anomalyId: string | null; orders: WorkOrder[]; compact?: boolean }) {
  const { approvals } = useAtlas();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const mine = approvals.items.filter(
    (r) => r.target.startsWith("spot.") && ((assetId && r.params.asset_id === assetId) || (anomalyId && r.params.anomaly_id === anomalyId)),
  );
  const written = orders.filter((o) => (assetId && o.asset_id === assetId) || (anomalyId && o.anomaly_id === anomalyId));

  const run = (fn: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    fn()
      .then(() => approvals.reload())
      .catch((e: unknown) => setError(e instanceof ApiError ? (e.needsToken ? "This box asks for a token (Settings, top right)." : e.message) : String(e)))
      .finally(() => setBusy(false));
  };

  return (
    <>
      <div className={s.acts}>
        {assetId ? (
          <button type="button" className={`${s.act} ${compact ? "" : s.actPrimary}`} disabled={busy} onClick={() => run(() => api.spotReinspect(assetId))}>
            Re-inspect
          </button>
        ) : null}
        {anomalyId ? (
          <button type="button" className={s.act} disabled={busy} onClick={() => run(() => api.spotWorkOrder(anomalyId))}>
            Work order{compact ? "" : ` for ${anomalyId}`}
          </button>
        ) : null}
        {!compact ? <span className={`${s.small} ${s.muted}`}>raises an approval card; nothing is written until you approve</span> : null}
      </div>
      {error ? <p className={`${s.small} ${s.toneBad}`}>{error}</p> : null}
      {mine.map((r) => (
        <div key={r.id} className={s.pending}>
          <span style={{ flex: "1 1 220px" }}>
            <b>Waiting for your OK:</b> {r.target === "spot.reinspect" ? "re-inspection" : "work order"} ({String(r.params.asset_id ?? "")}
            {r.params.anomaly_id ? `, ${String(r.params.anomaly_id)}` : ""}). Stays on this box.
          </span>
          <button type="button" className={`${s.act} ${s.actPrimary}`} disabled={busy} onClick={() => run(() => api.decide(r.id, "approve"))}>
            Approve
          </button>
          <button type="button" className={s.act} disabled={busy} onClick={() => run(() => api.decide(r.id, "deny"))}>
            Deny
          </button>
        </div>
      ))}
      {written.slice(0, 2).map((o) => (
        <p key={o.work_order_id} className={s.done}>
          ✓ {o.work_order_id} written on this box ({o.kind === "reinspect" ? "re-inspection" : "work order"}).
        </p>
      ))}
    </>
  );
}

export function TriageQueue({ items, scoreText, orders, limit = 6 }: { items: TriageItem[]; scoreText?: string; orders: WorkOrder[]; limit?: number }) {
  const { open } = useOpen();
  const [all, setAll] = useState(false);
  const shown = all ? items : items.slice(0, limit);
  const top = Math.max(1, ...items.map((i) => i.score));
  return (
    <div className={s.card}>
      <div className={s.cardHead}>
        <h2>Needs attention now</h2>
        <span className={`${s.small} ${s.muted}`}>{items.length} ranked</span>
      </div>
      <div className={s.queue}>
        {shown.map((it) => (
          <div key={it.id} className={s.item}>
            <div className={s.itemHead}>
              <span className={s.rank}>#{it.rank}</span>
              <button type="button" className={s.itemTitle} onClick={() => open({ kind: it.kind === "asset" ? "asset" : "location", id: it.id })}>
                {it.title}
              </button>
              {it.criticality ? <Pill tone="muted">{it.criticality} asset</Pill> : <Pill tone="muted">waypoint</Pill>}
              {it.severity ? <Sev level={it.severity} /> : null}
              {it.severity ? <Pill tone={it.confirmed ? "bad" : "watch"}>{it.confirmed ? "confirmed" : "unverified"}</Pill> : null}
              <span className={s.score} title={scoreText}>
                score {it.score}
              </span>
            </div>
            <div className={s.scoreBar} title={it.parts.map((p) => `${p.label} ${p.points}`).join(" + ")}>
              {it.parts
                .filter((p) => p.points > 0)
                .map((p) => (
                  <span key={p.label} style={{ width: `${(100 * p.points) / top}%`, background: PART_COLOR[p.label] ?? "var(--muted)" }} />
                ))}
            </div>
            <div className={s.parts}>
              {it.parts.map((p) => (
                <span key={p.label} title={p.detail}>
                  <i style={{ background: PART_COLOR[p.label] ?? "var(--muted)" }} />
                  {p.label} {p.points > 0 ? "+" : ""}
                  {p.points}
                </span>
              ))}
            </div>
            <p className={s.reason}>{it.reason}</p>
            <p className={s.stepText}>{it.next_step.text}</p>
            {it.model ? (
              <p className={`${s.small} ${s.muted}`} title="The scorer trained on the kit. It orders ties and explains; it never changes a score.">
                Model on its latest action <Id id={it.model.action_id} />: <b>{it.model.score.toFixed(2)}</b>
                {it.model.flagged ? " (flagged)" : " (not flagged)"}
                {it.model.why.length ? `, mainly because ${it.model.why[0]?.why}` : ""}
              </p>
            ) : null}
            <EvidenceLine evidence={it.evidence} max={8} />
            {it.kind === "asset" ? <OrderActions assetId={it.asset_id} anomalyId={it.anomaly_id} orders={orders} compact /> : null}
          </div>
        ))}
      </div>
      {items.length > limit ? (
        <button type="button" className={s.linkBtn} onClick={() => setAll((v) => !v)}>
          {all ? "Show the top only" : `Show all ${items.length}`}
        </button>
      ) : null}
      {scoreText ? (
        <details className={`${s.small} ${s.muted}`}>
          <summary>How the score is made</summary>
          {scoreText}
        </details>
      ) : null}
    </div>
  );
}

export function UninspectedRisk({ rows }: { rows: Risk[] }) {
  const { open } = useOpen();
  const risky = rows.filter((r) => r.at_risk);
  return (
    <div className={s.card}>
      <div className={s.cardHead}>
        <h2>Uninspected risk</h2>
        <span className={`${s.small} ${s.muted}`}>HIGH and CRITICAL assets with no good inspection for 24 h, or a failed attempt since</span>
      </div>
      {risky.length === 0 ? <p className={s.sentence}>Every HIGH and CRITICAL asset has a successful inspection in the last 24 hours of data.</p> : null}
      <table className={s.tbl}>
        <tbody>
          {risky.map((r) => (
            <tr key={r.asset_id}>
              <td>
                <Id id={r.asset_id} />
              </td>
              <td>
                <Sev level={r.criticality} />
              </td>
              <td>
                <b>{r.hours_since_good === null ? "never" : `${Math.round(r.hours_since_good)} h`}</b> since a good inspection
                {r.failed_since.length ? (
                  <>
                    ; failed since: {r.failed_since.map((a) => <Id key={a} id={a} />)}
                  </>
                ) : null}
              </td>
              <td>
                <button type="button" className={s.linkBtn} onClick={() => open({ kind: "asset", id: r.asset_id })}>
                  Open
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
