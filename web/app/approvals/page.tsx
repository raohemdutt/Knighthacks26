"use client";

import { useEffect, useState } from "react";
import { Empty, ErrorBox, Loading, WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type ApprovalRequest, type Decision } from "@/lib/api";
import { useAtlas } from "@/lib/atlas";
import { dayTime, time } from "@/lib/format";

const KIND_TITLE: Record<string, string> = {
  export_training_data: "Export training data",
  send_message: "Send a message",
  browse: "Open a web page",
  connector_write: "Change something in a connected app",
  circle_share: "Share with your Circle",
};

const TARGET_LABEL: Record<string, string> = {
  export_training_data: "Task",
  send_message: "To",
  browse: "Page",
  connector_write: "Tool",
  circle_share: "Circle",
};

export default function ApprovalsPage() {
  // The pending list lives in the app-wide provider: it reloads itself when /ws reports a new request or a decision.
  const { approvals, link } = useAtlas();
  const { items, loading, error, reload } = approvals;

  return (
    <>
      <div className={ui.pageHead}>
        <h1>Approvals</h1>
        <p className={ui.muted}>Atlas never sends, writes or exports anything until you approve it here.</p>
      </div>

      {error && items.length === 0 ? (
        <ErrorBox what="pending approvals" error={error} onRetry={reload} />
      ) : loading && items.length === 0 ? (
        <Loading what="pending approvals" />
      ) : items.length === 0 ? (
        <Empty title="Nothing waiting for you">
          When Atlas wants to message someone, add a calendar event, share with your Circle or export training data, a card appears here
          with Approve and Deny.{" "}
          {link === "live" ? "New requests show up on their own." : "The live connection is down, so pull up this screen again to check."}
        </Empty>
      ) : (
        <ul className={ui.list}>
          {items.map((r) => (
            <ApprovalCard key={r.id} request={r} onDecided={reload} />
          ))}
        </ul>
      )}
    </>
  );
}

function ApprovalCard({ request: r, onDecided }: { request: ApprovalRequest; onDecided: () => void }) {
  const [busy, setBusy] = useState<Decision | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const left = useSecondsLeft(r.expires_at);
  const params = Object.entries(r.params ?? {});

  const decide = async (decision: Decision) => {
    setBusy(decision);
    setError(null);
    try {
      await api.decide(r.id, decision);
      onDecided(); // /ws also triggers a reload; this covers a dropped socket
    } catch (err) {
      const e = err instanceof ApiError ? err : new ApiError(0, String(err));
      if (e.status === 404) onDecided(); // already decided or expired elsewhere: drop the stale card
      else setError(e);
      setBusy(null);
    }
  };

  return (
    <li className={ui.card}>
      <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
        <h2>{KIND_TITLE[r.kind] ?? r.kind}</h2>
        {/* The box decides when a request has expired; the phone's clock may be off, so the buttons stay usable. */}
        <span className={ui.tag}>{left > 0 ? `expires in ${formatLeft(left)}` : `expires ${time(r.expires_at)}`}</span>
      </div>
      <p className={ui.reason}>{r.reason}</p>
      <dl className={ui.kv}>
        <dt>{TARGET_LABEL[r.kind] ?? "Target"}</dt>
        <dd>{r.target}</dd>
        {params.map(([k, v]) => (
          <ParamRow key={k} name={k} value={v} />
        ))}
      </dl>
      {error ? <WriteError error={error} /> : null}
      <div className={ui.actions}>
        <button type="button" className={`${ui.btn} ${ui.btnDeny}`} disabled={busy !== null} onClick={() => void decide("deny")}>
          {busy === "deny" ? "Denying…" : "Deny"}
        </button>
        <button type="button" className={`${ui.btn} ${ui.btnPrimary}`} disabled={busy !== null} onClick={() => void decide("approve")}>
          {busy === "approve" ? "Approving…" : "Approve"}
        </button>
      </div>
    </li>
  );
}

function ParamRow({ name, value }: { name: string; value: unknown }) {
  const text = readable(value);
  if (!text) return null;
  return (
    <>
      <dt>{name.replace(/_/g, " ")}</dt>
      <dd>{text.length > 280 ? `${text.slice(0, 280)}…` : text}</dd>
    </>
  );
}

/** Parameters as a person would read them: times as times, nested settings as "name: value", empty ones left out. */
function readable(value: unknown): string {
  if (value === null || value === undefined || value === "") return "";
  if (typeof value === "string") return /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/.test(value) ? dayTime(value) || value : value;
  if (typeof value !== "object") return String(value);
  if (Array.isArray(value)) return value.map(readable).filter(Boolean).join(", ");
  return Object.entries(value as Record<string, unknown>)
    .map(([k, v]) => [k.replace(/_/g, " "), readable(v)] as const)
    .filter(([, v]) => v)
    .map(([k, v]) => `${k}: ${v}`)
    .join(" · ");
}

function useSecondsLeft(iso: string): number {
  const until = new Date(iso).getTime();
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const t = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(t);
  }, []);
  return Number.isNaN(until) ? 0 : Math.max(0, Math.round((until - now) / 1000));
}

function formatLeft(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return m > 0 ? `${m}:${String(s).padStart(2, "0")}` : `${s}s`;
}
