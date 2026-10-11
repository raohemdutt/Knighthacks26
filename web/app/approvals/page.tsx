"use client";

import { useEffect, useState } from "react";
import { Empty, ErrorBox, Loading, WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type ApprovalRequest, type Decision } from "@/lib/api";
import { useAtlas } from "@/lib/atlas";
import { useBrief } from "@/lib/brief";
import { cameraName, dayTime, humanize, time } from "@/lib/format";

const KIND_TITLE: Record<string, string> = {
  export_training_data: "Export training data",
  send_message: "Send a message",
  browse: "Open a web page",
  connector_write: "Change something in a connected app",
  circle_share: "Share with your Circle",
  device_control: "Change the box itself",
};

const TARGET_LABEL: Record<string, string> = {
  export_training_data: "Task",
  send_message: "To",
  browse: "Page",
  connector_write: "Tool",
  circle_share: "Circle",
  device_control: "Device",
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
  const rows = paramRows(r);
  const ask = useBrief().asks[r.id]; // the narrator's plain question for this card, when it has one

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
      {ask ? (
        <>
          <p className={ui.reason}>{ask.spoken}</p>
          <p className={`${ui.muted} ${ui.small} ${ui.breakWords}`}>{r.reason}</p>
        </>
      ) : (
        <p className={ui.reason}>{r.reason}</p>
      )}
      <dl className={ui.kv}>
        <dt>{TARGET_LABEL[r.kind] ?? "Target"}</dt>
        <dd>{r.target}</dd>
        {rows.map((row, i) => (
          <ParamRow key={`${row.label}-${i}`} row={row} />
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

/**
 * Parameters as a person would read them. The box sends whatever the action needs, so nothing here is
 * printed as it came: pictures are counted instead of listed by their frame refs, times and durations are
 * in the phone's own words, nested settings become short "name: value" lines, and the ids the box uses to
 * find things (action ids, external ids, hashes) are left out.
 */
interface Row {
  label: string;
  lines: string[];
}

const KEY_LABEL: Record<string, string> = {
  instruction: "Instruction",
  output_schema: "Answer format",
  examples: "Your examples",
  camera: "Camera",
  camera_id: "Camera",
  label: "Only when it sees",
  text: "Message",
  frame_ref: "Picture",
  frame_refs: "Pictures",
  keyframe_refs: "Pictures",
};

const MAX_LINES = 6;
const ISO_TIME = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/;
const FRAME_REF = /^(frames?:\S+|[a-z0-9_-]+:[a-z0-9_-]+:\d{10,})$/i;
const OPAQUE = /^([0-9a-f]{24,}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$/i;
const ID_KEY = /(^|_)(id|ids|uuid|token|ref|refs|hash|digest)$/i;
const ISO_DURATION = /^P(?=\d|T\d)(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)(?:\.\d+)?S)?)?$/;
const CLOCK_DURATION = /^(?:(\d+) days?, )?(\d+):(\d{2}):(\d{2})(?:\.\d+)?$/; // Python's timedelta: "1 day, 0:00:00"

type Picture = { verdict?: unknown };

const isObject = (v: unknown): v is Record<string, unknown> => typeof v === "object" && v !== null && !Array.isArray(v);
const isTime = (v: unknown): v is string => typeof v === "string" && ISO_TIME.test(v) && !Number.isNaN(new Date(v).getTime());
const plural = (n: number, one: string) => `${n} ${one}${n === 1 ? "" : "s"}`;
const labelOf = (key: string) => KEY_LABEL[key] ?? humanize(key);

/** 86400 -> "1 day", 9000 -> "2 h 30 min", 45 -> "45 s": the two largest units that are not zero. */
function duration(seconds: number): string {
  const s = Math.round(Math.abs(seconds));
  const parts: [number, string][] = [
    [Math.floor(s / 86400), "day"],
    [Math.floor((s % 86400) / 3600), "h"],
    [Math.floor((s % 3600) / 60), "min"],
    [s % 60, "s"],
  ];
  const shown = parts.filter(([n]) => n > 0).slice(0, 2);
  if (shown.length === 0) return "0 min";
  return shown.map(([n, unit]) => (unit === "day" ? plural(n, "day") : `${n} ${unit}`)).join(" ");
}

function durationOf(text: string): string | null {
  const iso = ISO_DURATION.exec(text);
  if (iso) return duration(Number(iso[1] ?? 0) * 86400 + Number(iso[2] ?? 0) * 3600 + Number(iso[3] ?? 0) * 60 + Number(iso[4] ?? 0));
  const c = CLOCK_DURATION.exec(text);
  if (c) return duration(Number(c[1] ?? 0) * 86400 + Number(c[2]) * 3600 + Number(c[3]) * 60 + Number(c[4]));
  return null;
}

/** A value that stands for a picture: a frame ref, or an example that carries one (with the person's thumb). */
function pictureOf(v: unknown): Picture | null {
  if (typeof v === "string") return FRAME_REF.test(v) ? {} : null;
  if (!isObject(v)) return null;
  const ref = v.frame_ref ?? v.frame ?? v.keyframe_ref;
  return typeof ref === "string" && FRAME_REF.test(ref) ? { verdict: v.verdict } : null;
}

function pictures(found: Picture[]): string {
  const up = found.filter((p) => p.verdict === "up").length;
  const down = found.filter((p) => p.verdict === "down").length;
  const marks = [up ? `${up} marked “like this”` : "", down ? `${down} marked “not this”` : ""].filter(Boolean).join(", ");
  return `${plural(found.length, "picture")}${marks ? ` (${marks})` : ""}`;
}

/** A value as the lines to print for it. An empty list means the row is left out. */
function describe(key: string, value: unknown): string[] {
  if (value === null || value === undefined || value === "") return [];
  if (typeof value === "boolean") return [value ? "yes" : "no"];
  if (typeof value === "number") {
    const unit = /_(seconds|secs|minutes|mins|hours)$/.exec(key)?.[1];
    if (unit) return [duration(value * (unit.startsWith("s") ? 1 : unit.startsWith("m") ? 60 : 3600))];
    return ID_KEY.test(key) ? [] : [String(value)];
  }
  if (typeof value === "string") {
    if (FRAME_REF.test(value)) return [pictures([{}])];
    if (ID_KEY.test(key) || OPAQUE.test(value)) return [];
    if (isTime(value)) return [dayTime(value)];
    if (key === "camera" || key === "camera_id") return [cameraName(value)];
    return [durationOf(value) ?? value];
  }
  if (Array.isArray(value)) {
    const found = value.map(pictureOf);
    const shots = found.filter((p): p is Picture => p !== null);
    const lines = shots.length > 0 ? [pictures(shots)] : [];
    const rest = value.filter((_, i) => found[i] === null);
    const each = rest.map((v) => describe(key, v).join("; ")).filter(Boolean);
    if (each.length === 0) return lines;
    if (rest.every((v) => typeof v !== "object" || v === null)) return [...lines, each.join(", ")];
    return [...lines, ...each.slice(0, MAX_LINES), ...(each.length > MAX_LINES ? [`and ${each.length - MAX_LINES} more`] : [])];
  }
  if (!isObject(value)) return [String(value)];
  if (key === "output_schema") {
    const fields = Object.keys(isObject(value.properties) ? value.properties : value);
    return [fields.length > 0 ? `Structured: ${fields.map((f) => humanize(f).toLowerCase()).join(", ")}` : "Free text"];
  }
  const range = timeRange(value);
  const lines = range ? [range] : [];
  for (const [k, v] of Object.entries(value)) {
    if (range && (k === "start" || k === "end")) continue;
    const text = describe(k, v).join("; ");
    if (text) lines.push(`${labelOf(k)}: ${text}`);
  }
  return lines;
}

/** "Yesterday 3:40 PM to Today 3:40 PM (1 day)" for a start and an end, the pair most actions carry. */
function timeRange(o: Record<string, unknown>): string | null {
  if (!isTime(o.start) || !isTime(o.end)) return null;
  const seconds = (new Date(o.end).getTime() - new Date(o.start).getTime()) / 1000;
  return `${dayTime(o.start)} to ${dayTime(o.end)}${seconds > 0 ? ` (${duration(seconds)})` : ""}`;
}

/** The rows of one card. A nested group of settings (a task's `spec`) is lifted, so each setting has its own row. */
function paramRows(r: ApprovalRequest): Row[] {
  const params = r.params ?? {};
  const rows: Row[] = [];
  const add = (key: string, value: unknown) => {
    if (key === "task" && value === r.target) return; // already the card's first row
    const lines = describe(key, value).map((l) => (l.length > 280 ? `${l.slice(0, 280)}…` : l));
    if (lines.length > 0) rows.push({ label: labelOf(key), lines });
  };
  const range = timeRange(params);
  for (const [k, v] of Object.entries(params)) {
    if (range && (k === "start" || k === "end")) continue;
    if (isObject(v) && k !== "output_schema") for (const [ik, iv] of Object.entries(v)) add(ik, iv);
    else add(k, v);
  }
  if (range) rows.push({ label: r.kind === "export_training_data" ? "Pictures from" : "When", lines: [range] });
  return rows;
}

function ParamRow({ row }: { row: Row }) {
  return (
    <>
      <dt>{row.label}</dt>
      <dd>
        {row.lines.map((line, i) => (
          <div key={i}>{line}</div>
        ))}
      </dd>
    </>
  );
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
