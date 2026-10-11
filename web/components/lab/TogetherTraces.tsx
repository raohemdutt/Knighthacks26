"use client";

import { LineChart } from "echarts/charts";
import { AxisPointerComponent, GridComponent, MarkAreaComponent, MarkLineComponent, TitleComponent, TooltipComponent } from "echarts/components";
import * as echarts from "echarts/core";
import { SVGRenderer } from "echarts/renderers";
import ReactEChartsCore from "echarts-for-react/lib/core";
import { useEffect, useMemo, useState } from "react";
import type { CompareSeries, SeriesRun } from "@/lib/labCompare";
import c from "./compare.module.css";

echarts.use([LineChart, GridComponent, TooltipComponent, MarkAreaComponent, MarkLineComponent, TitleComponent, AxisPointerComponent, SVGRenderer]);

/**
 * The three power traces of one loop (pod C, pod D, the box) on the loop's own clock, drawn with ECharts: one panel per
 * meter, stacked on one shared time axis, act 2's stages shaded behind them, each meter's calibrated idle floor dashed,
 * and where each act finished. Colours come from the product tokens (globals.css) at draw time, so the chart follows the
 * Night and Day themes. With reduced motion nothing animates. Loaded only in the browser (Compare.tsx, next/dynamic).
 */

const NODES = ["pod-c", "pod-d", "box"] as const;
const TITLE: Record<(typeof NODES)[number], string> = { "pod-c": "Pod C, the teacher", "pod-d": "Pod D, the trainer", box: "The box (Jetson)" };
const COLOR_VAR: Record<(typeof NODES)[number], string> = { "pod-c": "--accent", "pod-d": "--accent-2", box: "--accent-3" };
const STAGES = ["labeling", "training", "packaging"] as const;
const STAGE_VAR: Record<(typeof STAGES)[number], string> = { labeling: "--accent-3", training: "--accent", packaging: "--accent-2" };

type Tokens = Record<string, string>;

/** The tokens the chart draws with, read from the page; read again when the theme flips. */
function useTokens(): Tokens | null {
  const [tokens, setTokens] = useState<Tokens | null>(null);
  useEffect(() => {
    const read = () => {
      const cs = getComputedStyle(document.documentElement);
      const names = ["--accent", "--accent-2", "--accent-3", "--muted", "--text", "--line", "--surface", "--ok"];
      setTokens(Object.fromEntries(names.map((n) => [n, cs.getPropertyValue(n).trim() || "#888"])));
    };
    read();
    const mo = new MutationObserver(read);
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme", "class"] });
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    mq?.addEventListener("change", read);
    return () => {
      mo.disconnect();
      mq?.removeEventListener("change", read);
    };
  }, []);
  return tokens;
}

function useReducedMotion(force: boolean | undefined): boolean {
  const [reduced, setReduced] = useState(!!force);
  useEffect(() => {
    const mq = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    const apply = () => setReduced(!!force || !!mq?.matches);
    apply();
    mq?.addEventListener("change", apply);
    return () => mq?.removeEventListener("change", apply);
  }, [force]);
  return reduced;
}

/** A hex colour with an alpha, for the shading and the area under each line. */
function alpha(color: string, a: number): string {
  const m = /^#([0-9a-f]{6})$/i.exec(color);
  if (!m) return color;
  const n = parseInt(m[1]!, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

const actName = (r: SeriesRun) => (r.act === "solo" ? "act 1" : r.act === "duo" ? "act 2" : r.run_id);

export default function TogetherTraces({ series, actB, forceReduced }: { series: CompareSeries; actB: string | null; forceReduced?: boolean }) {
  const tokens = useTokens();
  const reduced = useReducedMotion(forceReduced);
  const nodes = NODES.filter((n) => series.nodes[n]?.points.length);

  const option = useMemo(() => {
    if (!tokens) return null;
    const t = (v: string) => tokens[v] ?? "#888";
    const shaded = series.runs.find((r) => r.run_id === actB) ?? series.runs.find((r) => r.act === "duo") ?? series.runs[0] ?? null;
    const allT = nodes.flatMap((n) => series.nodes[n]!.points.map((p) => p[0]));
    const xMin = Math.min(0, ...allT);
    const xMax = Math.max(...allT, ...series.runs.map((r) => r.end_s));
    const n = nodes.length;
    const top = 30;
    const gap = 44;
    const bottom = 34;
    const font = "inherit";
    // One panel per meter at a fixed height, so the chart's own height follows from how many meters reported.
    const panelH = 120;
    const H = panelH * n + top + bottom + (n - 1) * gap;
    const grid = nodes.map((_, i) => ({ left: 52, right: 62, top: top + i * (panelH + gap), height: panelH }));
    const stageAreas = shaded
      ? STAGES.flatMap((s) => {
          const span = shaded.stages[s];
          return span ? [[{ xAxis: span[0], name: s, itemStyle: { color: alpha(t(STAGE_VAR[s]), 0.13) } }, { xAxis: span[1] }]] : [];
        })
      : [];
    const ends = series.runs.map((r) => ({ xAxis: r.end_s, name: `${actName(r)} done`, act: r.act }));
    return {
      height: H,
      option: {
        animation: !reduced,
        animationDuration: 1400,
        animationEasing: "cubicOut",
        textStyle: { fontFamily: font, color: t("--muted") },
        axisPointer: { link: [{ xAxisIndex: "all" }], lineStyle: { color: t("--muted") } },
        tooltip: {
          trigger: "axis",
          backgroundColor: t("--surface"),
          borderColor: t("--line"),
          textStyle: { color: t("--text"), fontFamily: font },
          valueFormatter: (v: unknown) => (typeof v === "number" ? `${Math.round(v)} W` : ""),
        },
        title: nodes.map((node, i) => ({
          text: TITLE[node],
          left: 52,
          top: grid[i]!.top - 24,
          textStyle: { fontSize: 13, fontWeight: 700, color: t("--text"), fontFamily: font },
        })),
        grid,
        xAxis: nodes.map((_, i) => ({
          type: "value",
          gridIndex: i,
          min: Math.floor(xMin),
          max: Math.ceil(xMax / 10) * 10,
          axisLabel: { show: i === n - 1, showMaxLabel: false, formatter: "{value} s", color: t("--muted"), fontFamily: font },
          axisLine: { lineStyle: { color: t("--line") } },
          axisTick: { show: i === n - 1 },
          splitLine: { show: false },
        })),
        yAxis: nodes.map((_, i) => ({
          type: "value",
          gridIndex: i,
          min: 0,
          splitNumber: 2,
          axisLabel: { formatter: "{value} W", color: t("--muted"), fontFamily: font, fontSize: 11 },
          splitLine: { lineStyle: { color: t("--line"), opacity: 0.6 } },
        })),
        series: nodes.map((node, i) => {
          const s = series.nodes[node]!;
          const col = t(COLOR_VAR[node]);
          const idle = s.idle_w;
          return {
            type: "line",
            name: TITLE[node],
            xAxisIndex: i,
            yAxisIndex: i,
            data: s.points,
            showSymbol: false,
            smooth: 0.25,
            sampling: "lttb",
            lineStyle: { width: 2, color: col },
            itemStyle: { color: col },
            areaStyle: { color: { type: "linear", x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: alpha(col, 0.32) }, { offset: 1, color: alpha(col, 0) }] } },
            markArea: {
              silent: true,
              label: { show: i === 0, position: "insideTop", color: t("--muted"), fontSize: 11, fontFamily: font },
              data: stageAreas,
            },
            markLine: {
              silent: true,
              symbol: "none",
              data: [
                ...(typeof idle === "number"
                  ? [{ yAxis: idle, name: "idle", lineStyle: { type: "dashed", color: t("--muted"), width: 1.2 }, label: { formatter: `idle ${Math.round(idle)} W`, position: "insideEndTop", color: t("--muted"), fontSize: 11 } }]
                  : []),
                ...ends.map((e) => ({
                  xAxis: e.xAxis,
                  name: e.name,
                  lineStyle: { type: "dotted", color: e.act === "duo" ? t("--ok") : t("--muted"), width: 1.4 },
                  // act 2 finishes first: its label sits left of its line and act 1's right of its own, so they never collide
                  label: { show: i === 0, formatter: e.name, position: "end", align: e.act === "duo" ? "right" : "left", padding: e.act === "duo" ? [0, 4, 0, 0] : [0, 0, 0, 4], color: e.act === "duo" ? t("--ok") : t("--muted"), fontSize: 11 },
                })),
              ],
            },
          };
        }),
      },
    };
  }, [tokens, reduced, series, nodes.join(","), actB]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!option) return <div className={c.chart} style={{ height: 120 * Math.max(1, nodes.length) + 34 + 30 }} />;
  return (
    <div className={c.chart} role="img" aria-label={`Watts on ${nodes.map((n) => TITLE[n]).join(", ")} over the loop, on one clock, with act 2's stages shaded and each idle floor dashed`}>
      <ReactEChartsCore echarts={echarts} option={option.option} style={{ height: option.height, width: "100%" }} opts={{ renderer: "svg" }} notMerge lazyUpdate />
    </div>
  );
}
