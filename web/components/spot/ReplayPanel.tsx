"use client";

import { useEffect, useState } from "react";
import { Id, Pill, Sev, stamp, useOpen, words } from "./bits";
import s from "./spot.module.css";
import type { Alert, LiveAction, MissionDetail, Replay } from "./types";

/** Play, pause, speed and a scrubber over the whole replay: one segment per mission, a tick per failed action. */
export function ReplayBar({
  live,
  onStart,
  onPause,
  onSeek,
  onSpeed,
}: {
  live: Replay;
  onStart: () => void;
  onPause: () => void;
  onSeek: (index: number) => void;
  onSpeed: (speed: number) => void;
}) {
  const total = Math.max(live.total, 1);
  const [drag, setDrag] = useState<number | null>(null);
  useEffect(() => setDrag(null), [live.index]);
  const at = drag ?? live.index;
  const marks = live.marks ?? [];
  const c = live.counts ?? {};
  const x = (i: number) => (100 * i) / total;
  const current = marks.find((m) => at > m.start && at <= m.end + 1) ?? null;

  return (
    <div className={s.card}>
      <div className={s.cardHead}>
        <h2>Mission replay</h2>
        <span className={s.clock}>
          {live.index === 0 ? (
            "at rest: press Play to run the missions as they happened"
          ) : (
            <>
              <b>{stamp(live.data_ts)}</b> · event {live.index} of {live.total}
              {current ? ` · ${current.mission_id}` : ""}
            </>
          )}
        </span>
      </div>
      <div className={s.controls}>
        <button type="button" className={s.play} onClick={live.playing ? onPause : onStart}>
          {live.playing ? "Pause" : live.at_end ? "Replay" : "Play"}
        </button>
        <button type="button" className={s.linkBtn} onClick={() => onSeek(0)} disabled={live.index === 0}>
          Rewind
        </button>
        <span className={s.seg} role="group" aria-label="Replay speed">
          {(live.speeds ?? [60]).map((v) => (
            <button type="button" key={v} aria-pressed={live.speed === v} onClick={() => onSpeed(v)} title={`${v} seconds of data per second`}>
              {v}×
            </button>
          ))}
        </span>
        <span className={s.counts}>
          <span>
            missions <b>{(c.missions_completed ?? 0) + (c.missions_failed ?? 0)}</b>
          </span>
          <span>
            hiding failures <b style={{ color: (c.missions_hiding_failures ?? 0) > 0 ? "var(--watch)" : undefined }}>{c.missions_hiding_failures ?? 0}</b>
          </span>
          <span>
            actions ok <b>{c.actions_ok ?? 0}</b>
          </span>
          <span>
            failed <b style={{ color: (c.actions_failed ?? 0) > 0 ? "var(--bad)" : undefined }}>{c.actions_failed ?? 0}</b>
          </span>
          <span>
            anomalies <b>{c.anomalies ?? 0}</b>
          </span>
        </span>
      </div>

      <div className={s.scrub}>
        <svg viewBox="0 0 100 34" preserveAspectRatio="none" aria-hidden="true">
          {marks.map((m) => {
            const fill = m.status !== "COMPLETED" ? "var(--bad)" : m.failed > 0 ? "var(--watch)" : "var(--ok)";
            return (
              <g key={m.mission_id}>
                <rect x={x(m.start) + 0.12} y="9" width={Math.max(0.3, x(m.end + 1) - x(m.start) - 0.24)} height="16" rx="0.5" fill={fill} opacity={m.source === "tables" ? 0.3 : 0.5} />
                {m.failures.map((i) => (
                  <rect key={i} x={x(i)} y="5" width="0.28" height="24" fill="var(--bad)" />
                ))}
              </g>
            );
          })}
          <rect className={s.played} x="0" y="9" width={x(at)} height="16" fill="#1b2433" opacity="0.12" />
          <rect x={Math.max(0, x(at) - 0.2)} y="1" width="0.45" height="32" fill="var(--text)" />
        </svg>
        <input
          type="range"
          min={0}
          max={live.total}
          value={at}
          aria-label="Replay position"
          onChange={(e) => setDrag(Number(e.target.value))}
          onPointerUp={(e) => onSeek(Number((e.target as HTMLInputElement).value))}
          onKeyUp={(e) => onSeek(Number((e.target as HTMLInputElement).value))}
        />
      </div>
      <div className={s.scrubLabels}>
        <span>{marks[0]?.mission_id ?? ""}</span>
        <span>one block per mission</span>
        <span>{marks[marks.length - 1]?.mission_id ?? ""}</span>
      </div>
      <p className={`${s.small} ${s.muted}`}>
        Green: no failed action. Amber: marked COMPLETED with failed actions inside. Red: marked FAILED. Red ticks: failed actions.
        {live.rebuilt_from !== null && live.rebuilt_from !== undefined ? ` Faded: rebuilt from the tables, because replay.ndjson stops at event ${live.rebuilt_from}.` : ""}
      </p>
    </div>
  );
}

interface Row {
  action_id: string;
  sequence: number | null;
  action_type: string;
  location_id: string;
  asset_id: string;
  status: string;
  failure_type: string;
  battery_pct: number | null;
  ts: string | null;
  reading: LiveAction["reading"];
  anomalies: { anomaly_id: string; anomaly_type: string; severity: string; confirmed: boolean }[];
}

/** The current mission's actions as a vertical timeline. At rest: the latest mission in the data. */
export function MissionTimeline({ live, latest, alerts }: { live: Replay | null; latest: MissionDetail | null; alerts: Alert[] }) {
  const { open } = useOpen();
  const running = live && live.index > 0 ? live.mission ?? null : null;
  let head: { id: string; version: string; status: string; started: string | null; battery: number | null; failed: number; source: string } | null = null;
  let rows: Row[] = [];
  if (running) {
    head = { id: running.mission_id, version: running.configuration_version, status: running.status, started: running.started, battery: running.battery_pct, failed: running.failed, source: running.source };
    rows = running.actions;
  } else if (latest) {
    head = { id: latest.mission_id, version: latest.configuration_version, status: latest.mission_status, started: latest.start_time, battery: latest.end_battery_pct, failed: latest.failed, source: "tables" };
    rows = latest.timeline.map((a) => ({ ...a, ts: a.timestamp, reading: a.readings[0] ?? null }));
  }
  if (!head) return null;
  const done = head.status !== "RUNNING";
  const hidden = head.status === "COMPLETED" && head.failed > 0;

  return (
    <div className={s.card}>
      <div className={s.missionHead}>
        <h3>
          <Id id={head.id} /> {running ? "in the replay" : "latest mission"}
        </h3>
        {head.version ? <Pill tone="muted">{head.version}</Pill> : null}
        {!done ? (
          <Pill tone="muted">running</Pill>
        ) : hidden ? (
          <span className={s.hiddenBadge} title="The mission status says COMPLETED; the actions say otherwise.">
            ⚠ completed, {head.failed} action{head.failed === 1 ? "" : "s"} failed
          </span>
        ) : head.status === "COMPLETED" ? (
          <Pill tone="ok">completed, no failed action</Pill>
        ) : (
          <Pill tone="bad">
            {head.status.toLowerCase()}, {head.failed} failed
          </Pill>
        )}
        <span className={`${s.small} ${s.muted}`}>
          {stamp(head.started)}
          {head.battery !== null ? ` · battery ${Math.round(head.battery)}%` : ""}
        </span>
      </div>

      <div className={s.timeline}>
        {rows.map((a) => {
          const failed = a.status === "FAILED";
          const runningNow = a.status === "RUNNING";
          return (
            <button type="button" key={a.action_id} className={s.step} onClick={() => open({ kind: "asset", id: a.asset_id })}>
              <span className={`${s.mark} ${failed ? s.markFailed : runningNow ? s.markRunning : ""}`} aria-hidden="true">
                {failed ? "✕" : runningNow ? "…" : "✓"}
              </span>
              <span className={s.stepMain}>
                <span className={s.stepWhat}>{words(a.action_type)}</span>
                <span className={s.muted}>
                  {a.asset_id} at {a.location_id}
                </span>
                {failed ? <span className={s.failText}>failed: {words(a.failure_type) || "no type given"}</span> : null}
              </span>
              <span className={s.stepSub}>
                <span>{stamp(a.ts, false)}</span>
                {a.battery_pct !== null ? <span>battery {Math.round(a.battery_pct)}%</span> : null}
                {a.reading ? (
                  <span style={{ color: a.reading.quality === "GOOD" ? undefined : "var(--watch)" }}>
                    {words(a.reading.measurement_type)} {a.reading.value} {a.reading.unit} ({a.reading.quality.toLowerCase()})
                  </span>
                ) : null}
                {a.anomalies.map((x) => (
                  <span key={x.anomaly_id}>
                    <Sev level={x.severity} /> {words(x.anomaly_type)} {x.confirmed ? "" : "(unverified)"}
                  </span>
                ))}
              </span>
            </button>
          );
        })}
        {rows.length === 0 ? <p className={`${s.small} ${s.muted}`}>Waiting for the first action.</p> : null}
      </div>

      {running ? (
        <>
          <div className={s.sub}>Alerts as they arrive</div>
          <div className={s.alerts} aria-live="polite">
            {alerts.length === 0 ? <p className={`${s.small} ${s.muted}`}>No failure or anomaly yet.</p> : null}
            {alerts.slice(0, 12).map((a) => (
              <button
                type="button"
                key={`${a.alert_id}-${a.seq}`}
                className={`${s.alert} ${s[`alert_${a.level}`] ?? ""}`}
                onClick={() => (a.asset_id ? open({ kind: "asset", id: a.asset_id }) : open({ kind: "mission", id: a.mission_id }))}
              >
                <span className={s.muted}>{stamp(a.data_ts)} </span>
                {a.message}
              </button>
            ))}
          </div>
        </>
      ) : (
        <p className={`${s.small} ${s.muted}`}>Press Play to watch the missions unfold; failures and anomalies raise alerts here as they arrive.</p>
      )}
    </div>
  );
}
