"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { EChartsOption } from "echarts";
import { api } from "@/lib/api";
import {
  POWER_PODS,
  POWER_SPAN_MS,
  type PowerReading,
  appendReading,
  kjText,
  metersOffline,
  podLine,
  readingFrom,
  trapezoidJoules,
} from "@/lib/livePower";
import type { SharedBrainPair } from "@/lib/sharedBrain";
import { LabEChart, chartAxis, chartBase, useChartTokens } from "./Charts";
import { useReduced } from "./Labels";
import p from "./livePower.module.css";

/**
 * Power, right now: a strip chart that scrolls like a power monitor. The x axis is the last two minutes, one line per
 * pod (pod C, pod D) from GET /api/circle/shared-brain polled every 2 s into a rolling buffer in the browser. The
 * buffer starts empty and fills as readings arrive; nothing before the first reading is drawn. Each pod's idle floor
 * (watts_idle from the same route) is a dashed line, its newest watts a big number at the line's right end, and under
 * the chart a running count of the energy both pods drew while the page was open (trapezoid between readings, never
 * across a missed poll).
 *
 * The box is not drawn: no route serves its newest power reading live (the telemetry context's only HTTP route is
 * GET /runtime/stats, which has no watts), so the graph says so in one line instead.
 */

export const LIVE_MS = 2000;

/** One completed poll: its number, when it arrived, and the route's answer (null when the poll failed). */
export interface SharedBrainPoll {
  seq: number;
  t: number;
  route: SharedBrainPair | null;
}

/**
 * GET /api/circle/shared-brain every 2 s while mounted. `brain` is the newest good answer (null until the first),
 * `fresh` whether it is recent, and `poll` every completed poll, failed ones included, for the live power graph.
 */
export function useSharedBrain(): { brain: SharedBrainPair | null; fresh: boolean; poll: SharedBrainPoll | null } {
  const [brain, setBrain] = useState<SharedBrainPair | null>(null);
  const [poll, setPoll] = useState<SharedBrainPoll | null>(null);
  const [at, setAt] = useState(0);
  const [now, setNow] = useState(0);
  useEffect(() => {
    let alive = true;
    let seq = 0;
    let ctl: AbortController | null = null;
    const tick = () => {
      ctl?.abort();
      const mine = new AbortController();
      ctl = mine;
      api
        .circleSharedBrain(mine.signal)
        .then((d) => {
          if (!alive) return;
          const t = Date.now();
          setBrain(d);
          setAt(t);
          setPoll({ seq: ++seq, t, route: d });
        })
        .catch(() => {
          // a poll the next tick replaced was aborted on purpose; only a real failure is a gap in the lines
          if (!alive || mine.signal.aborted) return;
          setPoll({ seq: ++seq, t: Date.now(), route: null });
        });
      setNow(Date.now());
    };
    tick();
    const id = window.setInterval(tick, LIVE_MS);
    return () => {
      alive = false;
      ctl?.abort();
      window.clearInterval(id);
    };
  }, []);
  return { brain, fresh: at > 0 && now - at < LIVE_MS * 3, poll };
}

const clock = (t: number) => new Date(t).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", second: "2-digit" });

/** The graph, fed by polls someone else makes (the Speed and energy tab already polls the route for its live strip). */
export function LivePower({ poll, reduced, compact = false }: { poll: SharedBrainPoll | null; reduced: boolean; compact?: boolean }) {
  const t = useChartTokens();
  const [buf, setBuf] = useState<PowerReading[]>([]);
  const [joules, setJoules] = useState(0);
  const [seconds, setSeconds] = useState(0);
  const [idle, setIdle] = useState<Record<string, number | null>>({});
  const last = useRef<PowerReading | null>(null);
  const seen = useRef(0);

  useEffect(() => {
    if (!poll || poll.seq === seen.current) return;
    seen.current = poll.seq;
    const r = readingFrom(poll.route, poll.t);
    const prev = last.current;
    if (prev) {
      const j = trapezoidJoules(prev, r);
      if (j > 0) {
        setJoules((x) => x + j);
        setSeconds((x) => x + (r.t - prev.t) / 1000);
      }
    }
    last.current = r;
    setBuf((b) => appendReading(b, r));
    if (poll.route?.configured) {
      const next: Record<string, number | null> = {};
      for (const pod of POWER_PODS) next[pod.node] = poll.route.members.find((m) => m.node === pod.node)?.watts_idle ?? null;
      setIdle(next);
    }
  }, [poll]);

  const newest = buf.length ? buf[buf.length - 1]! : null;
  const offline = newest !== null && metersOffline(newest);
  const colors = [t.accent, t.accent3];
  const Title = compact ? "h3" : "h2";

  const option = useMemo<EChartsOption | null>(() => {
    if (!newest) return null;
    const axis = chartAxis(t);
    const now = newest.t;
    const shown = buf.filter((r) => r.t >= now - POWER_SPAN_MS);
    const ws = shown.flatMap((r) => POWER_PODS.map((pod) => r.w[pod.node])).filter((w): w is number => typeof w === "number");
    const idles = POWER_PODS.map((pod) => idle[pod.node]).filter((w): w is number => typeof w === "number");
    const top = Math.max(20, Math.ceil((Math.max(0, ...ws, ...idles) * 1.25) / 10) * 10);
    const sameIdle = idles.length === 2 && Math.abs(idles[0]! - idles[1]!) < 0.05;
    const base = chartBase(t, reduced);
    return {
      ...base,
      // one poll's move eased over one poll, so the lines glide left like a monitor's trace
      animationDurationUpdate: reduced ? 0 : LIVE_MS - 100,
      animationEasingUpdate: "linear",
      grid: { left: 44, right: compact ? 78 : 104, top: 14, bottom: compact ? 24 : 30 },
      tooltip: { show: false },
      xAxis: {
        ...axis,
        type: "value",
        min: now - POWER_SPAN_MS,
        max: now,
        interval: compact ? 60_000 : 30_000,
        splitLine: { show: false },
        axisLabel: { ...axis.axisLabel, fontSize: compact ? 11 : 12, formatter: (v: number) => clock(v), hideOverlap: true },
      },
      yAxis: { ...axis, type: "value", min: 0, max: top, splitNumber: compact ? 2 : 4, axisLabel: { ...axis.axisLabel, fontSize: compact ? 11 : 12, formatter: "{value} W" } },
      series: POWER_PODS.map((pod, i) => {
        const data = podLine(shown, pod.node);
        const tail = data.length ? data[data.length - 1]![1] : "-";
        const floor = idle[pod.node];
        const color = colors[i]!;
        return {
          id: pod.node,
          name: pod.name,
          type: "line" as const,
          showSymbol: false,
          smooth: 0.2,
          connectNulls: false,
          color,
          lineStyle: { width: compact ? 2 : 2.5, color },
          areaStyle: { color: { type: "linear" as const, x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: `${color}2e` }, { offset: 1, color: `${color}00` }] } },
          data,
          endLabel: {
            show: typeof tail === "number",
            color,
            fontSize: compact ? 15 : 22,
            fontWeight: 700,
            distance: 8,
            formatter: (x: { value?: unknown }) => {
              const w = Array.isArray(x.value) ? x.value[1] : null;
              return typeof w === "number" ? `${w.toFixed(1)} W` : "";
            },
          },
          markLine:
            typeof floor === "number"
              ? {
                  silent: true,
                  symbol: "none",
                  animation: false,
                  lineStyle: { type: "dashed" as const, width: 1.5, color, opacity: 0.75 },
                  label: {
                    show: !sameIdle || i === 0,
                    position: "insideStartTop" as const,
                    color: t.muted,
                    fontSize: 11,
                    formatter: sameIdle ? `idle floor ${floor.toFixed(1)} W, both pods` : `${pod.name} idle ${floor.toFixed(1)} W`,
                  },
                  data: [{ yAxis: floor }],
                }
              : { data: [] },
        };
      }),
    };
  }, [t, reduced, compact, buf, idle, newest]);

  const label = newest
    ? `Watts over the last two minutes. ${POWER_PODS.map((pod) => {
        const w = newest.w[pod.node];
        return `${pod.name} ${typeof w === "number" ? `${w.toFixed(1)} watts now` : "no reading"}`;
      }).join(", ")}.`
    : "Watts over the last two minutes, waiting for the first reading.";
  const reason = poll?.route?.reason ?? (poll && !poll.route ? "The shared-brain route did not answer." : null);

  return (
    <div className={`${p.card} ${compact ? p.compact : ""}`}>
      <div className={p.head}>
        <Title className={p.title} aria-live="polite">
          {offline ? (
            <>
              Power, <span className={p.off}>meters offline</span>
            </>
          ) : (
            "Power, right now"
          )}
        </Title>
        <div className={p.legend} aria-hidden="true">
          {POWER_PODS.map((pod, i) => (
            <span key={pod.node}>
              <span className={p.swatch} style={{ background: colors[i] }} />
              {pod.name}
            </span>
          ))}
        </div>
      </div>
      <p className={p.caption}>Watts as the meters read them, this second.</p>
      {option ? <LabEChart option={option} height={compact ? 150 : 260} label={label} /> : <p className={p.empty}>Waiting for the first reading from the pods&apos; meters.</p>}
      {offline && reason ? <p className={p.note}>{reason}</p> : null}
      <p className={p.energy}>
        energy drawn while you watched: <b>{kjText(joules)}</b>
      </p>
      <p className={p.note}>
        Both pods, added up between readings{seconds > 0 ? ` over ${Math.round(seconds)} s` : ""}; a missed reading adds nothing. The box is not on this graph: no
        route serves its own watts live yet.
      </p>
    </div>
  );
}

/** The graph with its own poll, for a screen that does not already poll the route (the Circle's Shared brain card). */
export function LivePowerSelf({ compact = false }: { compact?: boolean }) {
  const { poll } = useSharedBrain();
  const reduced = useReduced(false);
  return <LivePower poll={poll} reduced={reduced} compact={compact} />;
}
