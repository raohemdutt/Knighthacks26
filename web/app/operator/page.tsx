"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { OperatorRoom } from "@/components/map/TwinStage";
import { OperatorCamera } from "@/components/operator/Camera";
import { VoicePanel } from "@/components/operator/Voice";
import op from "@/components/operator/operator.module.css";
import { api, type AtlasEvent, type LearningTask } from "@/lib/api";
import { useAtlas } from "@/lib/atlas";
import { humanize, percent, time } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";

/**
 * Operator: one page for the big display, no phone needed. Open /operator/ on the laptop or TV.
 *
 *  - Cameras: the blurred picture with boxes, track ids and zone outlines (same sources as Live).
 *  - Ticker: events as they arrive on /ws, in plain words. Per-frame noise is left out.
 *  - Model badge: the version llama-server really serves (/runtime/models) and its speed. The speed is read off the
 *    answer stream (agent.answer.delta pieces per second, about one piece per token), or from telemetry.request
 *    when the telemetry context is on.
 *  - Lab: the newest taught task from /api/learning, kept fresh by learning.* events. The page cannot see whether
 *    the teacher is up, only what a run reports.
 *  - Offline badge: /vault/policy, refreshed when vault.policy.changed arrives.
 *  - Voice: push to talk on the box's own microphone and Bluetooth speaker (X16), state from /api/voice/status.
 */

const MAX_TICKS = 14;
const MIN_PIECES = 6; // pieces needed before a speed means anything
const SLOW_POLL_MS = 5000;

/** Events that fire many times a second, or that only matter to other screens. */
const NOISE = new Set([
  "perception.observation.recorded",
  "perception.faces.counted",
  "capture.frames.captured",
  "capture.camera.health",
  "agent.answer.delta",
]);

interface Tick {
  id: string;
  ts: string;
  text: string;
  tone: "plain" | "alert" | "good";
}

function str(v: unknown): string {
  return typeof v === "string" ? v : "";
}

function clip(s: string, n: number): string {
  return s.length > n ? `${s.slice(0, n - 1)}…` : s;
}

function tickFor(e: AtlasEvent): Tick | null {
  if (NOISE.has(e.type) || e.type.startsWith("telemetry.")) return null;
  let text: string;
  let tone: Tick["tone"] = "plain";
  switch (e.type) {
    case "experience.question.asked":
      text = `Asked: ${clip(str(e.text) || str(e.question), 90)}`;
      break;
    case "agent.answer.generated":
      text = `Answered: ${clip(str(e.answer), 110)}`;
      tone = "good";
      break;
    case "agent.alert.raised":
      text = `Alert: ${clip(str(e.message), 110)}`;
      tone = "alert";
      break;
    case "workspace.approval.requested":
      text = `Needs an OK: ${clip(str(e.reason), 100)}`;
      break;
    case "learning.run.progress":
      text = `Lab: ${str(e.task)} is ${str(e.stage)}${typeof e.progress === "number" ? ` (${percent(e.progress)})` : ""}`;
      break;
    case "learning.model.deployed":
      text = `New model on the box: ${str(e.model_version)}`;
      tone = "good";
      break;
    default:
      text = humanize(e.type);
  }
  return { id: e.event_id, ts: e.ts, text, tone };
}

function useClock(ms: number): Date {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), ms);
    return () => window.clearInterval(id);
  }, [ms]);
  return now;
}

/** Tokens per second of the on-device model, or null until an answer has streamed. */
function useTokensPerSecond(subscribe: (fn: (e: AtlasEvent) => void) => () => void): number | null {
  const [rate, setRate] = useState<number | null>(null);
  const streams = useRef(new Map<string, { first: number; last: number; n: number }>());
  useEffect(
    () =>
      subscribe((e) => {
        if (e.type === "agent.answer.delta") {
          const id = str(e.question_id);
          const now = performance.now();
          const s = streams.current.get(id) ?? { first: now, last: now, n: 0 };
          s.n += 1;
          s.last = now;
          streams.current.set(id, s);
          if (streams.current.size > 20) streams.current.delete(streams.current.keys().next().value as string);
          if (s.n >= MIN_PIECES && s.last > s.first) setRate(((s.n - 1) * 1000) / (s.last - s.first));
        } else if (e.type === "telemetry.request" && e.node === "jetson" && typeof e.tokens_out === "number") {
          const from = Date.parse(str(e.first_token) || str(e.started));
          const to = Date.parse(str(e.ended));
          if (to > from && e.tokens_out > 1) setRate((e.tokens_out * 1000) / (to - from));
        }
      }),
    [subscribe],
  );
  return rate;
}

function Lab({ task }: { task: LearningTask | undefined }) {
  if (!task) {
    return (
      <div className={op.panel}>
        <h2>Lab</h2>
        <p className={op.big}>Idle</p>
        <p className={op.muted}>Nothing is being taught right now.</p>
      </div>
    );
  }
  const scores = Object.entries(task.scores ?? {});
  return (
    <div className={op.panel}>
      <h2>Lab</h2>
      <p className={op.big}>{humanize(task.task)}</p>
      <p className={op.muted}>
        {humanize(task.stage)}
        {task.loss != null ? ` · loss ${task.loss.toFixed(3)}` : ""}
        {task.examples ? ` · ${task.examples} examples` : ""}
      </p>
      <div className={op.bar} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(task.progress * 100)}>
        <div style={{ width: `${Math.round(task.progress * 100)}%` }} />
      </div>
      {scores.length ? (
        <p className={op.scores}>
          {scores.map(([k, v]) => (
            <span key={k}>
              {humanize(k)} <b>{v.toFixed(2)}</b>
            </span>
          ))}
        </p>
      ) : null}
    </div>
  );
}

export default function OperatorPage() {
  const { link, model, subscribe } = useAtlas();
  const now = useClock(1000);
  const cameras = useFetch((signal) => api.captureStatus(signal));
  const models = useFetch((signal) => api.runtimeModels(signal));
  const policy = useFetch((signal) => api.vaultPolicy(signal));
  const board = useFetch((signal) => api.learning(signal));
  const [ticks, setTicks] = useState<Tick[]>([]);
  const tokS = useTokensPerSecond(subscribe);

  const { reload: reloadCameras } = cameras;
  const { reload: reloadModels } = models;
  const { reload: reloadPolicy } = policy;
  const { reload: reloadBoard } = board;

  // The slow numbers are polled; the events below only make them quicker when something changes.
  useEffect(() => {
    const id = window.setInterval(() => {
      reloadCameras();
      reloadModels();
      reloadPolicy();
      reloadBoard();
    }, SLOW_POLL_MS);
    return () => window.clearInterval(id);
  }, [reloadCameras, reloadModels, reloadPolicy, reloadBoard]);

  const onEvent = useCallback(
    (e: AtlasEvent) => {
      if (e.type === "vault.policy.changed") reloadPolicy();
      if (e.type.startsWith("learning.") || e.type === "telemetry.train.step") reloadBoard();
      if (e.type === "learning.model.deployed") reloadModels();
      const tick = tickFor(e);
      if (tick) setTicks((t) => [tick, ...t.filter((x) => x.id !== tick.id)].slice(0, MAX_TICKS));
    },
    [reloadPolicy, reloadBoard, reloadModels],
  );
  useEffect(() => subscribe(onEvent), [subscribe, onEvent]);

  const entries = Object.entries(cameras.data?.cameras ?? {});
  const offline = policy.data?.offline === true;
  const served = models.data?.serving ?? models.data?.current ?? model;
  const mismatch = models.data?.in_sync === false;
  // The newest task that is still moving, else the newest one.
  const tasks = board.data?.tasks ?? [];
  const labTask = tasks.find((t) => !["done", "failed", "denied", "expired"].includes(t.stage)) ?? tasks[tasks.length - 1];

  return (
    <main className={op.page}>
      <header className={op.top}>
        <span className={op.brand}>Atlas</span>
        <span className={`${op.badge} ${mismatch ? op.badgeBad : ""}`} title="The model that answers, running on this box">
          {served ? `On-device · ${served}` : "Model unknown"}
          {tokS != null ? (
            <>
              <span aria-hidden="true">·</span>
              <b>{tokS.toFixed(1)} tok/s</b>
            </>
          ) : null}
        </span>
        {offline ? <span className={`${op.badge} ${op.badgeWarn}`}>Offline · LAN only</span> : null}
        <span className={op.spacer} />
        <a className={`${op.badge} ${op.badgeLink}`} href="/zones/" title="Draw where things count, from this screen or a phone">
          Set zones
        </a>
        <span className={`${op.badge} ${link === "live" ? "" : op.badgeBad}`}>{link === "live" ? "Live" : link === "connecting" ? "Connecting…" : "Box unreachable, retrying"}</span>
        <span className={op.clock}>{time(now.toISOString())}</span>
      </header>

      <section className={op.cameras}>
        {entries.map(([id, status]) => (
          <OperatorCamera key={id} camera={id} status={status} />
        ))}
        {entries.length === 0 ? (
          <div className={op.panel}>
            <p className={op.big}>{cameras.loading ? "Looking for cameras…" : "No cameras on this box"}</p>
          </div>
        ) : null}
        <OperatorRoom />
      </section>

      <aside className={op.side}>
        <VoicePanel />
        <Lab task={labTask} />
        <div className={`${op.panel} ${op.tickerPanel}`}>
          <h2>Happening now</h2>
          {ticks.length === 0 ? <p className={op.muted}>Waiting for something to happen…</p> : null}
          <ul className={op.ticker}>
            {ticks.map((t) => (
              <li key={t.id} className={t.tone === "alert" ? op.tickAlert : t.tone === "good" ? op.tickGood : ""}>
                <time>{time(t.ts)}</time> {t.text}
              </li>
            ))}
          </ul>
        </div>
      </aside>
    </main>
  );
}
