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
import { LiveChat } from "./LiveChat";
import { TrainBurstControl, useTrainBurst } from "./TrainBurst";
import { burstLine, burstNode, podsText } from "@/lib/trainBurst";
import { LoadControl, useLoadRun } from "./LoadButton";
import { askingText, loadLine, rampTitle, stageWindows } from "@/lib/loadRun";
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

/** The third shade on the graph: people asking the pair at once (load_run.py). */
const LOAD_COLOR = "#ff7a90";

const clock = (t: number) => new Date(t).toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit", second: "2-digit" });

/** The graph, fed by polls someone else makes (the Speed and energy tab already polls the route for its live strip). */
export function LivePower({ poll, reduced, compact = false, chat = false }: { poll: SharedBrainPoll | null; reduced: boolean; compact?: boolean; chat?: boolean }) {
  // the live chat's answers, as time windows on this graph's clock: end null while the answer is still streaming
  const [windows, setWindows] = useState<{ start: number; end: number | null }[]>([]);
  const onWindow = (start: number, end: number | null) =>
    setWindows((w) => [...w.filter((x) => x.start !== start && x.start > Date.now() - POWER_SPAN_MS - 10_000), { start, end }]);
  // the "Train for a minute" button's burst (TrainBurst.tsx): its window shades the pods that trained, kept after it ends
  const burst = useTrainBurst(chat && !compact);
  const bw = burst.window;
  // the "16 people asking" and ramp buttons (LoadButton.tsx): their window, or each ramp stage, shaded in a third colour
  const load = useLoadRun(chat && !compact);
  const lv = load.view;
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
    const loadWins = lv && load.pressed !== null
      ? lv.ramp
        ? stageWindows(lv, load.pressed, load.ended ?? now)
        : [{ start: load.pressed, end: load.ended ?? now, label: askingText(lv.streams ?? 16) }]
      : [];
    const shadeBoth = !!bw && bw.pods.length > 1; // two pods' identical stretches: one "training" label, not two
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
        const shade = i === 0 ? windows.filter((w) => (w.end ?? now) >= now - POWER_SPAN_MS) : [];
        const train = bw && bw.pods.some((x) => burstNode(x) === pod.node) && (bw.end ?? now) >= now - POWER_SPAN_MS ? [bw] : [];
        const label = (text: string, color: string) => ({ show: true, position: "insideTop" as const, color, fontSize: 12, fontWeight: 600, formatter: text });
        const areas = [
          ...shade.map((w) => [
            { xAxis: Math.max(w.start, now - POWER_SPAN_MS), itemStyle: { color: `${t.accent2}26`, borderColor: `${t.accent2}88`, borderWidth: 1 }, label: label("answering", t.text) },
            { xAxis: Math.min(w.end ?? now, now) },
          ]),
          ...(i === 0 ? loadWins.filter((w) => w.end >= now - POWER_SPAN_MS) : []).map((w, k) => [
            { xAxis: Math.max(w.start, now - POWER_SPAN_MS), itemStyle: { color: `${LOAD_COLOR}${k % 2 ? "2e" : "1f"}`, borderColor: `${LOAD_COLOR}99`, borderWidth: 1 }, label: label(w.label, LOAD_COLOR) },
            { xAxis: Math.min(w.end, now) },
          ]),
          ...train.map((w) => [
            { xAxis: Math.max(w.start, now - POWER_SPAN_MS), itemStyle: { color: `${t.watch}22`, borderColor: `${t.watch}99`, borderWidth: 1 }, label: label(i === 0 || !shadeBoth ? "training" : "", t.watch) },
            { xAxis: Math.min(w.end ?? now, now) },
          ]),
        ];
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
          markArea: areas.length ? { silent: true, animation: false, data: areas as never } : { data: [] },
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
  }, [t, reduced, compact, buf, idle, newest, windows, bw, lv, load.pressed, load.ended]);

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
          ) : load.running && lv ? (
            <>
              Power, right now, <span style={{ color: LOAD_COLOR }}>{rampTitle(lv) ?? askingText(lv.streams ?? 16)}</span>
            </>
          ) : burst.running && burst.view ? (
            <>
              Power, right now, <span style={{ color: t.watch }}>training on {podsText(burst.view.pods)}</span>
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
      <div className={chat && !compact ? p.split : undefined}>
        <div className={p.graph}>
          {option ? <LabEChart option={option} height={compact ? 150 : 260} label={label} /> : <p className={p.empty}>Waiting for the first reading from the pods&apos; meters.</p>}
          {windows.some((w) => w.end === null) ? <p className={p.note}>The shaded stretch is the answer streaming in beside the graph.</p> : null}
          {burst.view?.state === "done" && bw ? <p className={p.energy}>{burstLine(burst.view)}</p> : null}
          {lv?.state === "done" && !lv.ramp && load.pressed !== null ? <p className={p.energy}>{loadLine(lv)}</p> : null}
        </div>
        {chat && !compact ? (
          <div className={p.side}>
            <TrainBurstControl burst={burst} />
            <LoadControl load={load} />
            <LiveChat reduced={reduced} onWindow={onWindow} />
          </div>
        ) : null}
      </div>
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
