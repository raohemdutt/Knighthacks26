"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import type { EChartsOption } from "echarts";
import { ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api, spotUrl } from "@/lib/api";
import { chipShort } from "@/lib/lab";
import {
  type BenchHeadline,
  type BenchPayload,
  type BenchPoint,
  type BenchPowerNode,
  type BenchSession,
  type LabBench,
  type StoryLine,
  benchSummary,
  engineOf,
  fmt,
  isLowConfidence,
  machineName,
  modelBillions,
  speedStory,
  when,
} from "@/lib/labBench";
import { useFetch } from "@/lib/useFetch";
import { type ChartTokens, LabEChart, chartAxis, chartBase, useChartTokens } from "./Charts";
import { PointDetail, useReduced } from "./Labels";
import { Sheet } from "./LabRun";
import b from "./bench.module.css";
import s from "./speed.module.css";

/**
 * Speed and energy, told as one story from the lab's bench rows (GET /api/learning/lab/bench):
 *
 *   1. The chip fix: energy per token against streams for the teacher on one Strix Halo pod, vLLM after the kernel
 *      fix beside llama.cpp, with the pod's watts on a second axis. The tiles beside it are computed from the rows.
 *   2. The bigger brain: the 122B pair across two pods against the 27B on one pod, at the pair's batch sizes.
 *   3. What it costs to be on: idle and answering watts per machine.
 *   4. Sessions and rows: every session, its numbers and their source rows, in a drawer at the bottom.
 *
 * The route returns the summary shape (sessions and headline points); until it does, the same summary is derived
 * from today's full payload (lib/labBench.ts summaryFromFull). Sessions are picked by engine, chip and date, never
 * by a number written here.
 *
 *   /lab/?tab=speed&compare=1          the hero chart with the NVIDIA A100 reference on
 *   /lab/?tab=speed&sweep=<session>    the drawer open at that session
 *   /lab/?tab=speed&drawer=what|rows   with that sheet open
 */

const ENGINE_NAME: Record<string, string> = { vllm: "vLLM", llamacpp: "llama.cpp", other: "other engine" };
const engineName = (e: string) => ENGINE_NAME[engineOf(e)] ?? e;

function sessionTitle(x: BenchSession): string {
  const where = x.node ? machineName(x.node) : chipShortName(x.chip);
  return `${engineName(x.engine)} · ${where} · ${x.model ?? "model not named"}${x.date ? ` · ${when(x.date)}` : ""}`;
}

function chipShortName(chip: string): string {
  if (/strix|8060s|gfx1151/i.test(chip)) return "Strix Halo";
  if (/a100/i.test(chip)) return "NVIDIA A100";
  return chip || "chip not named";
}

const at = (l: StoryLine | null, batch: number) => l?.points.find((p) => p.batch === batch) ?? null;
const first = (l: StoryLine | null) => l?.points[0] ?? null;
const last = (l: StoryLine | null) => (l?.points.length ? l.points[l.points.length - 1]! : null);
const lineData = (l: StoryLine | null, pick: (p: BenchHeadline) => number | null) =>
  (l?.points ?? []).flatMap((p) => {
    const v = pick(p);
    return v === null || v <= 0 ? [] : [{ value: [p.batch, v] as [number, number], low: isLowConfidence(p.confidence) }];
  });

/** A line series in the product's look: low-confidence points hollow, the rest filled. */
function series(t: ChartTokens, id: string, name: string, color: string, data: { value: [number, number]; low: boolean }[], extra: Record<string, unknown> = {}) {
  return {
    id,
    name,
    type: "line" as const,
    smooth: 0.25,
    symbol: "circle",
    symbolSize: 10,
    color,
    lineStyle: { width: 3, color },
    emphasis: { focus: "series" as const, scale: 1.4 },
    data: data.map((d) => ({ value: d.value, itemStyle: d.low ? { color: t.surface, borderColor: color, borderWidth: 2.5 } : { color, borderColor: color } })),
    ...extra,
  };
}

const streams = (n: number) => `${n} stream${n === 1 ? "" : "s"}`;

/** "?" with the reason, for a tile whose row is missing. */
function Tile({ value, label, sub, missing }: { value: string | null; label: string; sub: string; missing?: string }) {
  return (
    <div className={`${s.tile} ${value === null ? s.tileMissing : ""}`}>
      <span className={s.tileValue}>{value ?? "?"}</span>
      <span className={s.tileLabel}>{label}</span>
      <span className={s.tileSub}>{value === null ? missing : sub}</span>
    </div>
  );
}

/* ---------- 1. the chip fix ---------- */

function ChipFix({ story, reduced, a100, setA100, onPick }: { story: ReturnType<typeof speedStory>; reduced: boolean; a100: boolean; setA100: (v: boolean) => void; onPick: (session: string, metric: string, batch: number) => void }) {
  const t = useChartTokens();
  const { after, before, llama } = story;
  const ref = a100 ? story.a100 : null;

  const a1 = at(after, 1);
  const b1 = at(before, 1);
  const speedup = a1?.tok_s_stream && b1?.tok_s_stream ? a1.tok_s_stream / b1.tok_s_stream : null;
  const watts = (after?.points ?? []).map((p) => p.watts).filter((w): w is number => w !== null);
  const wMin = watts.length ? Math.min(...watts) : null;
  const wMax = watts.length ? Math.max(...watts) : null;
  const wMid = watts.length ? watts.slice().sort((x, y) => x - y)[Math.floor(watts.length / 2)]! : null;
  const jFirst = first(after);
  const jLast = last(after);
  const l1 = at(llama, 1);
  const shared = (llama?.points ?? []).filter((p) => at(after, p.batch)).map((p) => p.batch);
  const top = shared.length ? shared[shared.length - 1]! : null;

  const option = useMemo<EChartsOption>(() => {
    const axis = chartAxis(t);
    const lines = [after, llama, ref].filter(Boolean) as StoryLine[];
    const maxB = Math.max(2, ...lines.flatMap((l) => l.points.map((p) => p.batch)));
    const wTop = Math.max(10, ...lines.flatMap((l) => l.points.map((p) => p.watts ?? 0)));
    return {
      ...chartBase(t, reduced),
      grid: { left: 56, right: 60, top: 84, bottom: 52 },
      legend: { top: 0, left: 0, icon: "roundRect", itemWidth: 14, itemHeight: 4, itemGap: 18, textStyle: { color: t.text, fontSize: 13 } },
      tooltip: {
        ...(chartBase(t, reduced).tooltip as object),
        trigger: "axis",
        axisPointer: { type: "line", lineStyle: { color: t.line, width: 1 } },
        formatter: (raw: unknown) => {
          const ps = (Array.isArray(raw) ? raw : [raw]) as { seriesName: string; value: [number, number]; color: string; seriesId: string }[];
          if (!ps.length) return "";
          const head = `<b>${streams(ps[0]!.value[0])}</b>`;
          const rows = ps.map((p) => `<span style="color:${p.color}">&#9679;</span> ${p.seriesName}: <b>${fmt(p.value[1])}</b> ${p.seriesId.endsWith(":w") ? "W" : "J per token"}`);
          return [head, ...rows].join("<br/>");
        },
      },
      xAxis: { ...axis, type: "log", logBase: 2, min: 1, max: maxB, name: "streams answered at once", nameLocation: "middle", nameGap: 32, splitLine: { show: false }, axisLabel: { ...axis.axisLabel, formatter: (v: number) => String(Math.round(v)) } },
      yAxis: [
        { ...axis, type: "log", logBase: 10, name: "joules per token (log)", nameLocation: "end", nameGap: 14, axisLabel: { ...axis.axisLabel, formatter: (v: number) => String(Number(v.toPrecision(2))) } },
        { ...axis, type: "value", min: 0, max: Math.ceil((wTop * 1.6) / 50) * 50, name: "watts", nameLocation: "end", nameGap: 14, splitLine: { show: false }, position: "right" },
      ],
      series: [
        ...(after
          ? [
              series(t, `${after.session.id}:j`, "vLLM after the kernel fix", t.accent3, lineData(after, (p) => p.j_per_token), {
                areaStyle: { color: { type: "linear", x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: `${t.accent3}33` }, { offset: 1, color: `${t.accent3}00` }] } },
              }),
              series(t, `${after.session.id}:w`, "Pod watts (vLLM)", t.watch, lineData(after, (p) => p.watts), { yAxisIndex: 1, smooth: false, symbolSize: 6, lineStyle: { width: 2, type: "dashed", color: t.watch } }),
            ]
          : []),
        ...(llama ? [series(t, `${llama.session.id}:j`, "llama.cpp Vulkan", t.accent2, lineData(llama, (p) => p.j_per_token))] : []),
        ...(ref ? [series(t, `${ref.session.id}:j`, "NVIDIA A100 (reference)", t.unknown, lineData(ref, (p) => p.j_per_token), { symbol: "triangle", lineStyle: { width: 2, type: "dotted", color: t.unknown } })] : []),
      ],
    };
  }, [t, reduced, after, llama, ref]);

  return (
    <section className={s.section} aria-labelledby="speed-fix">
      <div className={s.kicker}>
        <span className={s.step}>1</span> The chip fix
      </div>
      <h2 id="speed-fix" className={s.h2}>
        Same Strix Halo chip, same power, {speedup ? `${fmt(speedup, 1)} times` : "more"} the speed
      </h2>
      <div className={s.hero}>
        <div className={s.chartCard}>
          {after ? (
            <LabEChart
              option={option}
              height={390}
              label={`Energy per token against streams answered at once, ${after ? "vLLM after the kernel fix" : ""}${llama ? " and llama.cpp Vulkan" : ""}, with the pod's watts on a second axis`}
              onClick={(p) => {
                const id = p.seriesId ?? "";
                const session = id.replace(/:[jw]$/, "");
                const l = [after, llama, ref].find((x) => x?.session.id === session);
                const pt = l?.points.filter((q) => (id.endsWith(":w") ? q.watts : q.j_per_token) !== null)[p.dataIndex];
                if (pt) onPick(session, id.endsWith(":w") ? "watts_avg" : "joules_per_token", pt.batch);
              }}
            />
          ) : (
            <p className={s.empty}>No vLLM session on a Strix Halo pod in these rows yet, so there is nothing to draw.</p>
          )}
          <div className={s.chartFoot}>
            <label className={s.toggle}>
              <input type="checkbox" checked={a100} disabled={!story.a100} onChange={(e) => setA100(e.target.checked)} />
              <span>{story.a100 ? "Show the NVIDIA A100 reference" : "No A100 reference session in these rows"}</span>
            </label>
            <span className={s.foot}>Hollow points are low confidence. Tap a point for its row.</span>
          </div>
        </div>
        <div className={s.tiles}>
          <Tile
            value={speedup ? `${fmt(speedup, 1)}x` : null}
            label="faster for one person"
            sub={`${fmt(b1?.tok_s_stream ?? 0)} to ${fmt(a1?.tok_s_stream ?? 0)} tokens a second at one stream, vLLM before and after the kernel fix`}
            missing={after ? "No slower vLLM session before the fix in these rows, so no ratio." : "No vLLM session on a Strix Halo pod in these rows."}
          />
          <Tile
            value={wMid !== null ? `${fmt(wMid, 0)} W` : null}
            label="flat, however many streams"
            sub={`${fmt(wMin ?? 0, 1)} to ${fmt(wMax ?? 0, 1)} W from ${streams(jFirst?.batch ?? 1)} to ${streams(jLast?.batch ?? 1)}`}
            missing="No watts row for the vLLM session."
          />
          <Tile
            value={jFirst?.j_per_token && jLast?.j_per_token && jLast !== jFirst ? `${fmt(jFirst.j_per_token, 1)} to ${fmt(jLast.j_per_token, 1)} J` : null}
            label="per token"
            sub={`from ${streams(jFirst?.batch ?? 1)} to ${streams(jLast?.batch ?? 1)}: the same watts spread over more tokens`}
            missing="The vLLM session needs two batch sizes with joules per token."
          />
        </div>
      </div>
      <p className={s.story}>
        {l1?.tok_s_stream && a1?.tok_s_stream && l1.j_per_token && a1.j_per_token ? (
          <>
            llama.cpp on Vulkan ties at one stream ({fmt(l1.tok_s_stream)} against {fmt(a1.tok_s_stream)} tokens a second) but each token costs {fmt((l1.j_per_token / a1.j_per_token - 1) * 100, 0)} percent more
            energy there
            {top && top > 1 && at(llama, top)?.tok_s_stream && at(after, top)?.tok_s_stream
              ? `, and it cannot batch: at ${streams(top)} each person gets ${fmt(at(llama, top)!.tok_s_stream!)} tokens a second against vLLM's ${fmt(at(after, top)!.tok_s_stream!)}.`
              : "."}
          </>
        ) : (
          "No llama.cpp session on a Strix Halo pod in these rows, so there is no second engine to compare."
        )}
      </p>
    </section>
  );
}

/* ---------- 2. the bigger brain ---------- */

function BiggerBrain({ story, reduced }: { story: ReturnType<typeof speedStory>; reduced: boolean }) {
  const t = useChartTokens();
  const { pair, base } = story;
  const option = useMemo<EChartsOption | null>(() => {
    if (!pair) return null;
    const axis = chartAxis(t);
    const maxB = Math.max(2, ...pair.points.map((p) => p.batch));
    return {
      ...chartBase(t, reduced),
      grid: { left: 56, right: 24, top: 84, bottom: 52 },
      legend: { top: 0, left: 0, icon: "roundRect", itemWidth: 14, itemHeight: 4, itemGap: 18, textStyle: { color: t.text, fontSize: 13 } },
      tooltip: {
        ...(chartBase(t, reduced).tooltip as object),
        trigger: "axis",
        axisPointer: { type: "line", lineStyle: { color: t.line, width: 1 } },
        formatter: (raw: unknown) => {
          const ps = (Array.isArray(raw) ? raw : [raw]) as { seriesName: string; value: [number, number]; color: string; seriesId: string }[];
          if (!ps.length) return "";
          const batch = ps[0]!.value[0];
          const rows = ps.map((p) => {
            const l = p.seriesId === "pair" ? pair : base;
            const q = at(l, batch);
            return `<span style="color:${p.color}">&#9679;</span> ${p.seriesName}: <b>${fmt(p.value[1])}</b> J per token${q?.tok_s_total ? `, ${fmt(q.tok_s_total)} tok/s in total` : ""}${q?.watts ? `, ${fmt(q.watts, 0)} W` : ""}`;
          });
          return [`<b>${streams(batch)}</b>`, ...rows].join("<br/>");
        },
      },
      xAxis: { ...axis, type: "log", logBase: 2, min: 1, max: maxB, name: "streams answered at once", nameLocation: "middle", nameGap: 32, splitLine: { show: false }, axisLabel: { ...axis.axisLabel, formatter: (v: number) => String(Math.round(v)) } },
      yAxis: { ...axis, type: "log", logBase: 10, name: "joules per token (log)", nameLocation: "end", nameGap: 14, axisLabel: { ...axis.axisLabel, formatter: (v: number) => String(Number(v.toPrecision(2))) } },
      series: [
        series(t, "pair", `${modelBillions(pair.session.model) ?? "Big"}B across two pods`, t.accent, lineData(pair, (p) => p.j_per_token)),
        ...(base ? [series(t, "base", `${modelBillions(base.session.model) ?? "Small"}B on one pod`, t.accent3, lineData(base, (p) => p.j_per_token))] : []),
      ],
    };
  }, [t, reduced, pair, base]);

  const shared = pair && base ? pair.points.filter((p) => at(base, p.batch)?.j_per_token && p.j_per_token).map((p) => p.batch) : [];
  const top = shared.length ? shared[shared.length - 1]! : null;
  const pj = top !== null ? at(pair, top)?.j_per_token : null;
  const bj = top !== null ? at(base, top)?.j_per_token : null;
  const more = pj && bj ? (pj / bj - 1) * 100 : null;
  const pb = modelBillions(pair?.session.model);
  const bb = modelBillions(base?.session.model);
  const times = pb && bb ? pb / bb : null;

  return (
    <section className={s.section} aria-labelledby="speed-brain">
      <div className={s.kicker}>
        <span className={s.step}>2</span> The bigger brain
      </div>
      <h2 id="speed-brain" className={s.h2}>
        Two pods together run a model {times ? `${fmt(times, 1)} times` : "several times"} the size
      </h2>
      {pair && option ? (
        <div className={s.brain}>
          <div className={s.chartCard}>
            <LabEChart option={option} height={330} label="Energy per token against streams answered at once, the big model across two pods and the 27B on one pod" />
          </div>
          <table className={s.table}>
            <thead>
              <tr>
                <th scope="col">Streams</th>
                <th scope="col">Tokens a second</th>
                <th scope="col">Watts</th>
                <th scope="col">J per token</th>
              </tr>
            </thead>
            <tbody>
              {pair.points.map((p) => (
                <tr key={p.batch}>
                  <th scope="row">{p.batch}</th>
                  <td>{p.tok_s_total !== null ? fmt(p.tok_s_total) : "no row"}</td>
                  <td>{p.watts !== null ? fmt(p.watts, 0) : "no row"}</td>
                  <td>
                    {p.j_per_token !== null ? fmt(p.j_per_token) : "no row"}
                    {isLowConfidence(p.confidence) ? <span className={s.lowDot} title="low confidence" aria-label="low confidence" /> : null}
                  </td>
                </tr>
              ))}
            </tbody>
            <caption className={s.caption}>{pb ? `${pb}B` : "The big model"} across two pods, every number from its own row.</caption>
          </table>
        </div>
      ) : (
        <p className={s.empty}>The pair rows are not on this box yet. This section draws the big model across two pods as soon as its bench rows arrive; nothing is drawn in their place.</p>
      )}
      {more !== null && top !== null ? (
        <p className={s.story}>
          A bigger brain costs more joules per token; under load ({streams(top)}), {fmt(more, 0)} percent more for {times ? `${fmt(times, 1)} times` : "a much bigger"} the model.
        </p>
      ) : null}
    </section>
  );
}

/* ---------- 3. what it costs to be on ---------- */

function PowerBars({ nodes, reduced, onPick }: { nodes: BenchPowerNode[]; reduced: boolean; onPick: (p: BenchPoint) => void }) {
  const t = useChartTokens();
  const shown = nodes.filter((n) => n.idle || n.serving);
  const option = useMemo<EChartsOption>(() => {
    const axis = chartAxis(t);
    const label = (p: BenchPoint | null) => ({
      show: !!p,
      position: "right" as const,
      color: t.text,
      fontSize: 13,
      formatter: () => (p ? `${fmt(p.value, p.value >= 100 ? 0 : 1)} W${p.low_confidence ? " {dot|}" : ""}` : ""),
      rich: { dot: { width: 6, height: 6, borderRadius: 3, backgroundColor: t.watch, padding: 0 } },
    });
    const bar = (which: "idle" | "serving", name: string, color: string) => ({
      id: which,
      name,
      type: "bar" as const,
      barWidth: 12,
      barGap: "45%",
      itemStyle: { color, borderRadius: [0, 6, 6, 0] },
      emphasis: { focus: "series" as const },
      data: shown.map((n) => ({ value: n[which]?.value ?? 0, label: label(n[which]) })),
    });
    return {
      ...chartBase(t, reduced),
      grid: { left: 130, right: 70, top: 36, bottom: 24 },
      legend: { top: 0, left: 0, icon: "roundRect", itemWidth: 14, itemHeight: 8, itemGap: 18, textStyle: { color: t.text, fontSize: 13 } },
      tooltip: {
        ...(chartBase(t, reduced).tooltip as object),
        trigger: "item",
        formatter: (raw: unknown) => {
          const p = raw as { seriesId: string; dataIndex: number };
          const n = shown[p.dataIndex];
          const pt = n?.[p.seriesId as "idle" | "serving"];
          if (!n || !pt) return "";
          return `<b>${machineName(n.node)}</b>, ${p.seriesId === "idle" ? "on and idle" : "answering"}: <b>${fmt(pt.value)} W</b>${pt.state ? `<br/><span style="color:${t.muted}">${pt.state.length > 90 ? `${pt.state.slice(0, 90)}...` : pt.state}</span>` : ""}${pt.low_confidence ? `<br/><span style="color:${t.watch}">low confidence</span>` : ""}`;
        },
      },
      xAxis: { ...axis, type: "value", name: "watts", nameLocation: "end", nameGap: 8, splitNumber: 3 },
      yAxis: { ...axis, type: "category", inverse: true, data: shown.map((n) => machineName(n.node)), axisLabel: { ...axis.axisLabel, color: t.text, fontSize: 13 }, splitLine: { show: false } },
      series: [bar("idle", "On, idle", t.unknown), bar("serving", "Answering", t.accent)],
    };
  }, [t, reduced, shown]);
  if (!shown.length) return <p className={s.empty}>No idle or answering power row in this response yet.</p>;
  return (
    <LabEChart
      option={option}
      height={70 + shown.length * 52}
      label={`Idle and answering watts per machine: ${shown.map((n) => `${machineName(n.node)} ${n.idle ? `${fmt(n.idle.value)} W idle` : "no idle row"}, ${n.serving ? `${fmt(n.serving.value)} W answering` : "no answering row"}`).join("; ")}`}
      onClick={(p) => {
        const n = shown[p.dataIndex];
        const pt = n?.[p.seriesId as "idle" | "serving"];
        if (pt) onPick(pt);
      }}
    />
  );
}

/* ---------- 4. sessions and rows ---------- */

function What({ story }: { story: ReturnType<typeof speedStory> }) {
  const from = (l: StoryLine | null) => (l ? `${sessionTitle(l.session)} (${l.session.id})` : "no session in these rows");
  const lines: [string, string][] = [
    ["Streams are how many people the teacher model answers at the same moment (the batch size of each row).", "conditions.batch_size of each row"],
    ["The chip fix: vLLM on one Strix Halo pod after the kernel fix, the newest vLLM session on that chip.", from(story.after)],
    ["The 'before' in the first tile: the newest older vLLM session of the same model that was slower at one stream.", from(story.before)],
    ["llama.cpp Vulkan: the newest llama.cpp session on a Strix Halo pod with two or more batch sizes.", from(story.llama)],
    ["The bigger brain: the newest session of the big model across two pods, against the vLLM session above at the same batch sizes.", from(story.pair)],
    ["What it costs to be on: the newest idle and answering power row of each machine. A small amber dot marks a low-confidence row.", "power rows (jetson.idle_watts, pod idle and serving watts)"],
    ["Hollow points are low confidence: fewer repeats than a verdict needs. Tap a point or a number for its row.", "each row's own confidence and reason"],
  ];
  return (
    <ol className={b.howto}>
      {lines.map(([text, src]) => (
        <li key={text}>
          {text}
          <span className={b.note} style={{ display: "block" }}>
            From: <span className={b.code}>{src}</span>
          </span>
        </li>
      ))}
    </ol>
  );
}

function AllRows({ bench, sessions }: { bench: BenchPayload; sessions: BenchSession[] }) {
  return (
    <>
      <p className={ui.muted}>
        {bench.rows ?? sessions.reduce((n, x) => n + x.rows, 0)} rows{bench.file ? ` in ${bench.file}` : ""}
        {bench.charted_rows !== undefined ? `; ${bench.charted_rows} are drawn on this screen or on the Training tab` : ""}. The rest are listed here so none is dropped silently.
      </p>
      <h3>Sessions</h3>
      <div className={b.list}>
        {sessions.map((x) => (
          <div key={x.id} className={b.listRow}>
            <div className={b.listTop}>
              <span>{sessionTitle(x)}</span>
              <b>{x.rows} rows</b>
            </div>
            <span className={b.note}>
              <span className={b.code}>{x.id}</span>
              {x.lever ? ` · ${x.lever.replace(/_/g, " ")}` : ""}
            </span>
          </div>
        ))}
      </div>
      {bench.not_charted?.length ? (
        <>
          <h3>Present but not charted</h3>
          <div className={b.list}>
            {bench.not_charted.map((x) => (
              <div key={`${x.metric}|${x.node}`} className={b.listRow}>
                <div className={b.listTop}>
                  <span>
                    {x.metric} <span className={b.note}>on {x.node}</span>
                  </span>
                  <b>
                    {fmt(x.latest.value)} {x.unit}
                  </b>
                </div>
                <span className={b.note}>
                  {x.rows} row{x.rows === 1 ? "" : "s"}: <span className={b.code}>metrics.jsonl:{x.lines.join(", ")}</span>
                  {x.other_chip ? ` · not the current lab: ${x.other_chips.map((g) => `${g.rows === x.rows ? "all" : g.rows} from ${chipShort(g)}`).join(", ")}` : ""}
                </span>
              </div>
            ))}
          </div>
        </>
      ) : null}
      {bench.skipped?.length ? <p className={b.note}>Lines that are not rows: {bench.skipped.map((k) => `${k.line} (${k.problem})`).join(", ")}.</p> : null}
    </>
  );
}

const METRIC_OF: Record<string, string> = { tok_s_stream: "decode_tok_s", tok_s_total: "decode_tok_s_total", ttft_s: "ttft_s", watts: "watts_avg", j_per_token: "joules_per_token" };

function Sessions({
  sessions,
  points,
  selected,
  setSelected,
  findPoint,
  picked,
  setPicked,
  rows,
  openSheet,
  rowsState,
}: {
  sessions: BenchSession[];
  points: BenchHeadline[];
  selected: string | null;
  setSelected: (id: string) => void;
  findPoint: (session: string, metric: string, batch: number) => BenchPoint | null;
  picked: BenchPoint | null;
  setPicked: (p: BenchPoint | null) => void;
  rows: number;
  openSheet: (which: "what" | "rows") => void;
  rowsState: "loading" | "error" | "ready" | null;
}) {
  const current = sessions.find((x) => x.id === selected) ?? sessions[0];
  const mine = points.filter((p) => p.session === current?.id).sort((x, y) => x.batch - y.batch);
  const cell = (p: BenchHeadline, key: keyof typeof METRIC_OF, digits?: number) => {
    const v = p[key as keyof BenchHeadline] as number | null;
    if (v === null) return <span className={s.muted}>no row</span>;
    const src = current ? findPoint(current.id, METRIC_OF[key]!, p.batch) : null;
    const text = fmt(v, digits);
    return src ? (
      <button type="button" className={s.num} onClick={() => setPicked(picked === src ? null : src)} aria-label={`${key.replace(/_/g, " ")} at ${streams(p.batch)}: ${text}. Show its row.`}>
        {text}
      </button>
    ) : (
      text
    );
  };
  return (
    <div className={s.drawerBody}>
      <div className={s.drawerButtons}>
        <button type="button" className={ui.btn} onClick={() => openSheet("what")}>
          What am I looking at?
        </button>
        <button type="button" className={ui.btn} onClick={() => openSheet("rows")}>
          All {rows} rows
        </button>
      </div>
      <div className={ui.chips} role="group" aria-label="Measurement sessions, newest first">
        {sessions.map((x) => (
          <button key={x.id} type="button" className={`${ui.chip} ${x.id === current?.id ? ui.chipOn : ""}`} aria-pressed={x.id === current?.id} onClick={() => setSelected(x.id)}>
            {sessionTitle(x)}
          </button>
        ))}
      </div>
      {current ? (
        <div className={s.tableWrap}>
          <table className={s.table}>
            <thead>
              <tr>
                <th scope="col">Streams</th>
                <th scope="col">Tok/s each</th>
                <th scope="col">Tok/s total</th>
                <th scope="col">First token (s)</th>
                <th scope="col">Watts</th>
                <th scope="col">J per token</th>
              </tr>
            </thead>
            <tbody>
              {mine.map((p) => (
                <tr key={p.batch}>
                  <th scope="row">
                    {p.batch}
                    {isLowConfidence(p.confidence) ? <span className={s.lowDot} title="low confidence" aria-label="low confidence" /> : null}
                  </th>
                  <td>{cell(p, "tok_s_stream")}</td>
                  <td>{cell(p, "tok_s_total")}</td>
                  <td>{cell(p, "ttft_s")}</td>
                  <td>{cell(p, "watts", 1)}</td>
                  <td>{cell(p, "j_per_token")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
      {picked ? (
        <PointDetail p={picked} onClose={() => setPicked(null)} />
      ) : (
        <p className={s.foot}>
          {rowsState === "loading" ? "Fetching this session's rows..." : rowsState === "error" ? "This session's rows did not load; the numbers above are its headline." : "Tap a number to see the row it came from."}
        </p>
      )}
    </div>
  );
}

export function SpeedEnergy({
  initialDrawer,
  initialCompare = null,
  initialSweep = null,
  forceReduced,
}: {
  initialBatch: number | null;
  initialTtft: boolean;
  initialPoint: string | null;
  initialDrawer: string | null;
  initialCompare?: string | null;
  initialSweep?: string | null;
  forceReduced: boolean;
}) {
  const bench = useFetch((signal) => api.labBench(signal), "bench");
  const reduced = useReduced(forceReduced);
  const [a100, setA100] = useState(initialCompare === "1" || /a100/i.test(initialCompare ?? ""));
  const [sheet, setSheet] = useState<"what" | "rows" | null>(initialDrawer === "what" || initialDrawer === "rows" ? initialDrawer : null);
  const [open, setOpen] = useState(initialSweep !== null || initialDrawer !== null);
  const [selected, setSelected] = useState<string | null>(initialSweep);
  const [pickedPoint, setPickedPoint] = useState<BenchPoint | null>(null);
  const [pickedKey, setPickedKey] = useState<{ session: string; metric: string; batch: number } | null>(null);
  // a session's rows (GET .../bench/rows?session=) and the whole payload (?all=1), fetched only when the drawer needs them
  const [rowsOf, setRowsOf] = useState<Record<string, Partial<LabBench> | "loading" | "error">>({});
  const [all, setAll] = useState<Partial<LabBench> | "loading" | "error" | null>(null);
  const close = useCallback(() => setSheet(null), []);
  const payload = bench.data as BenchPayload | undefined;
  const summary = useMemo(() => benchSummary(payload), [payload]);
  const story = useMemo(() => speedStory(summary), [summary]);
  // the drawer lists the sessions that have headline numbers, newest first; the rest are counted in "All rows"
  const sessions = useMemo(() => {
    const withNumbers = new Set(summary.points.map((p) => p.session));
    return summary.sessions.filter((x) => withNumbers.has(x.id)).sort((x, y) => (y.date ?? "").localeCompare(x.date ?? ""));
  }, [summary]);
  const current = sessions.find((x) => x.id === selected) ?? sessions.find((x) => x.id === story.after?.session.id) ?? sessions[0];
  const fullHere = (payload?.sweeps?.length ?? 0) > 0;

  useEffect(() => {
    if (!open || !current || fullHere) return;
    const key = current.source ?? current.id;
    if (rowsOf[key]) return;
    setRowsOf((m) => ({ ...m, [key]: "loading" }));
    fetch(spotUrl(current.rows_url ?? `/api/learning/lab/bench/rows?session=${encodeURIComponent(key)}`))
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((d: Partial<LabBench>) => setRowsOf((m) => ({ ...m, [key]: d })))
      .catch(() => setRowsOf((m) => ({ ...m, [key]: "error" })));
  }, [open, current, fullHere, rowsOf]);

  useEffect(() => {
    if (sheet !== "rows" || all !== null || fullHere) return;
    setAll("loading");
    fetch(spotUrl("/api/learning/lab/bench?all=1"))
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then((d: Partial<LabBench>) => setAll(d))
      .catch(() => setAll("error"));
  }, [sheet, all, fullHere]);

  const findPoint = useCallback(
    (session: string, metric: string, batch: number): BenchPoint | null => {
      const x = sessions.find((q) => q.id === session);
      const from = fullHere ? payload : rowsOf[x?.source ?? session];
      if (!from || typeof from === "string") return null;
      const sweeps = (from.sweeps ?? []).filter((w) => w.id === session || !fullHere);
      for (const w of sweeps) {
        if (x?.node && w.node && w.node !== x.node) continue;
        const p = w.metrics[metric]?.find((q) => q.x === batch);
        if (p) return p;
      }
      return null;
    },
    [sessions, fullHere, payload, rowsOf],
  );
  const picked = pickedPoint ?? (pickedKey ? findPoint(pickedKey.session, pickedKey.metric, pickedKey.batch) : null);
  const setPicked = useCallback((p: BenchPoint | null) => {
    setPickedPoint(p);
    setPickedKey(null);
  }, []);
  const pickFromChart = useCallback(
    (session: string, metric: string, batch: number) => {
      setSelected(session);
      setOpen(true);
      setPickedPoint(null);
      setPickedKey({ session, metric, batch });
      if (typeof document !== "undefined") document.getElementById("speed-drawer")?.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
    },
    [reduced],
  );
  const pickPower = useCallback(
    (p: BenchPoint) => {
      setPicked(p);
      setOpen(true);
      if (typeof document !== "undefined") document.getElementById("speed-drawer")?.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
    },
    [reduced, setPicked],
  );

  if (bench.loading && !bench.data) return <Loading what="the lab's measurements" />;
  if (bench.error && !bench.data) return <ErrorBox what="the lab's measurements" error={bench.error} onRetry={bench.reload} />;
  const data = payload!;
  const rows = data.rows ?? summary.sessions.reduce((n, x) => n + x.rows, 0);
  const nodes = data.power?.nodes ?? [];

  return (
    <div className={`${b.page} ${s.page}`}>
      <div className={ui.pageHead}>
        <h1>Speed and energy</h1>
        <p className={b.sub}>What the lab measured on AMD Strix Halo: the fix that made the teacher fast, the bigger model two pods can run together, and what each machine draws.</p>
      </div>
      {sessions.length === 0 ? <p className={b.banner}>{data.empty_reason ?? "No bench session in these rows yet."}</p> : null}

      <ChipFix story={story} reduced={reduced} a100={a100} setA100={setA100} onPick={pickFromChart} />
      <BiggerBrain story={story} reduced={reduced} />

      <section className={s.section} aria-labelledby="speed-power">
        <div className={s.kicker}>
          <span className={s.step}>3</span> What it costs to be on
        </div>
        <h2 id="speed-power" className={s.h2}>
          Idle and answering watts, one row per machine
        </h2>
        <div className={s.chartCard}>
          <PowerBars nodes={nodes} reduced={reduced} onPick={pickPower} />
          <div className={s.chartFoot}>
            <span className={s.foot}>
              <span className={s.lowDot} aria-hidden="true" /> marks a low-confidence row. Tap a bar for its row.
            </span>
          </div>
        </div>
        {data.power?.absent?.length ? <p className={s.foot}>No power row yet for {data.power.absent.join(", ")}; nothing is drawn for a machine the rows do not cover.</p> : null}
      </section>

      <details id="speed-drawer" className={s.drawer} open={open} onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}>
        <summary className={s.drawerHead}>
          <span>Sessions and rows</span>
          <span className={s.foot}>
            {sessions.length} sessions with numbers, {rows} rows
          </span>
        </summary>
        {open ? (
          <Sessions sessions={sessions} points={summary.points} selected={current?.id ?? null} setSelected={setSelected} findPoint={findPoint} picked={picked} setPicked={setPicked} rows={rows} openSheet={setSheet} rowsState={fullHere || !current ? null : rowsOf[current.source ?? current.id] === undefined ? "loading" : typeof rowsOf[current.source ?? current.id] === "string" ? (rowsOf[current.source ?? current.id] as "loading" | "error") : "ready"} />
        ) : null}
      </details>

      {sheet === "what" ? (
        <Sheet title="What am I looking at?" onClose={close}>
          <What story={story} />
        </Sheet>
      ) : null}
      {sheet === "rows" ? (
        <Sheet title="Every row" onClose={close}>
          <AllRows bench={fullHere ? data : all && typeof all !== "string" ? { ...data, ...all } : data} sessions={summary.sessions} />
          {all === "loading" ? <p className={s.foot}>Fetching every row (a few megabytes on the box)...</p> : all === "error" ? <p className={s.foot}>The whole payload did not load; the sessions above are from the summary.</p> : null}
        </Sheet>
      ) : null}
      {picked && !open ? <PointDetail p={picked} onClose={() => setPicked(null)} /> : null}
    </div>
  );
}
