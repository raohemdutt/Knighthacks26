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
  isPair,
  MACHINES,
  machineName,
  modelBillions,
  speedStory,
  when,
} from "@/lib/labBench";
import type { SharedBrainPair } from "@/lib/sharedBrain";
import { useFetch } from "@/lib/useFetch";
import { type ChartTokens, LabEChart, chartAxis, chartBase, useChartTokens } from "./Charts";
import { PointDetail, useReduced } from "./Labels";
import { LivePower, useSharedBrain } from "./LivePower";
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
function Tile({ value, label, sub, missing, why }: { value: string | null; label: string; sub: string; missing?: string; why?: string }) {
  return (
    <div className={`${s.tile} ${value === null ? s.tileMissing : ""}`}>
      <span className={s.tileValue}>{value ?? "?"}</span>
      <span className={s.tileLabel}>{label}</span>
      {why ? <span className={s.why}>{why}</span> : null}
      <span className={s.tileSub}>{value === null ? missing : sub}</span>
    </div>
  );
}

/* ---------- 0. live meters ---------- */

function LiveTile({ title, value, sub, offline, fresh, reduced }: { title: string; value: string | null; sub?: string | null; offline: boolean; fresh: boolean; reduced: boolean }) {
  return (
    <div className={`${s.live} ${offline ? s.liveOff : ""}`}>
      <span className={s.liveTitle}>
        <span className={`${s.pulse} ${fresh && !offline && !reduced ? s.pulseOn : ""} ${offline ? s.pulseOff : ""}`} aria-hidden="true" />
        {title}
      </span>
      <span className={s.liveValue}>{offline ? "offline" : (value ?? "no reading")}</span>
      {sub ? <span className={s.liveSub}>{sub}</span> : null}
    </div>
  );
}

function LiveStrip({ brain, fresh, box, reduced }: { brain: SharedBrainPair | null; fresh: boolean; box: BenchPoint | null; reduced: boolean }) {
  const member = (node: string) => brain?.members.find((m) => m.node === node) ?? null;
  const pod = (node: string, title: string) => {
    const m = member(node);
    const off = !brain || !brain.configured || !m || !m.online;
    return <LiveTile title={title} value={m?.watts_now !== null && m?.watts_now !== undefined ? `${fmt(m.watts_now, 1)} W` : null} sub={m?.watts_idle ? `idle is ${fmt(m.watts_idle, 1)} W` : null} offline={off} fresh={fresh} reduced={reduced} />;
  };
  const sv = brain?.serving;
  const pairOff = !brain || !brain.configured || !brain.members.some((m) => m.online);
  const busy = (sv?.requests_running ?? 0) > 0;
  return (
    <section className={s.section} aria-label="Live meters">
      <div className={s.kicker}>
        <span className={`${s.pulse} ${fresh && !reduced ? s.pulseOn : ""}`} aria-hidden="true" /> Live
      </div>
      <p className={s.why}>Live meters, right now.</p>
      <div className={s.liveRow} aria-live="polite">
        {pod("pod-a", "Pod C now")}
        {pod("pod-b", "Pod D now")}
        <LiveTile title="The box" value={box ? `${fmt(box.value, 1)} W` : null} sub={box ? `last reading, ${when(box.ts)}` : "no box power row yet"} offline={false} fresh={false} reduced={reduced} />
        <LiveTile
          title={busy ? "The pair is answering" : "The pair, last answers"}
          value={sv?.tok_s !== null && sv?.tok_s !== undefined ? `${fmt(sv.tok_s, 1)} tok/s` : null}
          sub={sv?.ttft_s !== null && sv?.ttft_s !== undefined ? `first token ${fmt(sv.ttft_s, 1)} s${busy ? `, ${sv.requests_running} running` : ""}` : null}
          offline={pairOff}
          fresh={fresh}
          reduced={reduced}
        />
      </div>
      {!brain ? <p className={s.foot}>Waiting for the pods&apos; meters (GET /api/circle/shared-brain).</p> : brain.reason ? <p className={s.foot}>{brain.reason}</p> : null}
    </section>
  );
}

/* ---------- 1. the chip fix ---------- */

function ChipFix({ story, reduced, onPick }: { story: ReturnType<typeof speedStory>; reduced: boolean; onPick: (session: string, metric: string, batch: number) => void }) {
  const t = useChartTokens();
  const { after, before, llama } = story;

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
    const lines = [after, llama, before].filter(Boolean) as StoryLine[];
    const maxB = Math.max(2, ...lines.flatMap((l) => l.points.map((p) => p.batch)));
    const wTop = Math.max(10, ...(after?.points ?? []).map((p) => p.watts ?? 0)); // the watts axis is the vLLM pod's own
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
          const sessionOf = (p: { seriesId: string; value: [number, number] }) => {
            const l = [after, llama, before].find((x) => x && p.seriesId.startsWith(`${x.session.id}:`));
            return l?.points.find((q) => q.batch === p.value[0])?.session ?? null;
          };
          const rows = ps.map((p) => {
            const sid = sessionOf(p);
            return `<span style="color:${p.color}">&#9679;</span> ${p.seriesName}: <b>${fmt(p.value[1])}</b> ${p.seriesId.endsWith(":w") ? "W" : "J per token"}${sid && !p.seriesId.endsWith(":w") ? `<br/><span style="color:${t.muted};font-size:12px">session ${sid}</span>` : ""}`;
          });
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
              series(t, `${after.session.id}:w`, wMid !== null ? `Power drawn, stays at ${Math.round(wMid)} W` : "Power drawn", t.watch, lineData(after, (p) => p.watts), { yAxisIndex: 1, smooth: false, symbolSize: 6, lineStyle: { width: 2, type: "dashed", color: t.watch } }),
            ]
          : []),
        ...(llama ? [series(t, `${llama.session.id}:j`, "llama.cpp Vulkan, measured", t.accent2, lineData(llama, (p) => p.j_per_token))] : []),
        // the pre-fix session the 4.5x tile divides by, grey, above the green line so the gap is the speedup
        ...(before && before.session.id !== after?.session.id
          ? [
              series(t, `${before.session.id}:j`, "vLLM before the kernel fix", t.muted, lineData(before, (p) => p.j_per_token), {
                lineStyle: { width: 2.5, color: t.muted },
                label: { show: before.points.length === 1, position: "top", color: t.muted, fontSize: 12, formatter: "vLLM before the kernel fix" },
              }),
            ]
          : []),
      ],
    };
  }, [t, reduced, after, llama, before, wMid]);

  return (
    <section className={s.section} aria-labelledby="speed-fix">
      <div className={s.kicker}>
        <span className={s.step}>1</span> The chip fix
      </div>
      <h2 id="speed-fix" className={s.h2}>
        Same Strix Halo chip, same power, {speedup ? `${fmt(speedup, 1)} times` : "more"} the speed
      </h2>
      <p className={s.why}>Same chip, same power, 4.5 times the work.</p>
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
                const l = [after, llama, before].find((x) => x?.session.id === session);
                const pt = l?.points.filter((q) => (id.endsWith(":w") ? q.watts : q.j_per_token) !== null)[p.dataIndex];
                if (pt) onPick(pt.session, id.endsWith(":w") ? "watts_avg" : "joules_per_token", pt.batch);
              }}
            />
          ) : (
            <p className={s.empty}>No vLLM session on a Strix Halo pod in these rows yet, so there is nothing to draw.</p>
          )}
          <div className={s.chartFoot}>
            <span className={s.foot}>Hollow points are low confidence. Tap a point for its row.</span>
          </div>
        </div>
        <div className={s.tiles}>
          <Tile
            value={speedup ? `${fmt(speedup, 1)}x` : null}
            label="faster for one person"
            why="One person gets answers 4.5 times faster."
            sub={`${fmt(b1?.tok_s_stream ?? 0)} to ${fmt(a1?.tok_s_stream ?? 0)} tokens a second at one stream, vLLM before and after the kernel fix`}
            missing={after ? "No slower vLLM session before the fix in these rows, so no ratio." : "No vLLM session on a Strix Halo pod in these rows."}
          />
          <Tile
            value={wMid !== null ? `${fmt(wMid, 0)} W` : null}
            label="flat, however many streams"
            why="More people cost no extra power."
            sub={`${fmt(wMin ?? 0, 1)} to ${fmt(wMax ?? 0, 1)} W from ${streams(jFirst?.batch ?? 1)} to ${streams(jLast?.batch ?? 1)}`}
            missing="No watts row for the vLLM session."
          />
          <Tile
            value={jFirst?.j_per_token && jLast?.j_per_token && jLast !== jFirst ? `${fmt(jFirst.j_per_token, 1)} to ${fmt(jLast.j_per_token, 1)} J` : null}
            label="per token"
            why="Each token gets cheaper as more ask."
            sub={`from ${streams(jFirst?.batch ?? 1)} to ${streams(jLast?.batch ?? 1)}: the same watts spread over more tokens`}
            missing="The vLLM session needs two batch sizes with joules per token."
          />
        </div>
      </div>
      <p className={s.story}>
        {l1?.tok_s_stream && a1?.tok_s_stream && l1.j_per_token && a1.j_per_token ? (
          <>
            llama.cpp on Vulkan ties at one stream ({fmt(l1.tok_s_stream)} against {fmt(a1.tok_s_stream)} tokens a second)
            {top && top > 1 && at(llama, top)?.j_per_token && at(after, top)?.j_per_token
              ? `, but at ${streams(top)} it spends ${fmt(at(llama, top)!.j_per_token!, 1)} J a token against vLLM's ${fmt(at(after, top)!.j_per_token!, 1)}: it cannot batch, so the chip's flat watts never pay off for it.`
              : `, and each token costs ${fmt((l1.j_per_token / a1.j_per_token - 1) * 100, 0)} percent more energy there.`}
          </>
        ) : (
          "No llama.cpp session on a Strix Halo pod in these rows, so there is no second engine to compare."
        )}
      </p>
    </section>
  );
}

/* ---------- 2. the bigger brain ---------- */

/** Both pods' watts and joules per token for the pair, by batch size: the `_both` rows, from a session's rows or the full payload. */
type Both = Map<number, { watts: number | null; j: number | null }>;
const PAIR_NODES = MACHINES.find((m) => m.key === "pair")?.nodes ?? [];

function bothFrom(payload: Partial<LabBench> | null | undefined, into: Both) {
  const batchOf = (p: BenchPoint) => p.x ?? (Number(/batch (\d+)/.exec(p.state ?? "")?.[1]) || null);
  const put = (metric: string, p: BenchPoint) => {
    const batch = batchOf(p);
    if (!batch) return;
    const cur = into.get(batch) ?? { watts: null, j: null };
    if (/joules_per_token_both$/.test(metric) && cur.j === null) cur.j = p.value;
    if (/watts_avg_both$/.test(metric) && cur.watts === null) cur.watts = p.value;
    into.set(batch, cur);
  };
  for (const w of payload?.sweeps ?? []) if (/pair_pp122b/.test(w.family)) for (const [m, pts] of Object.entries(w.metrics)) for (const p of pts) put(m, p);
  for (const sr of payload?.series ?? []) if (/^pair_pp122b\./.test(sr.metric)) for (const p of sr.points) put(sr.metric, p);
}

/** The pair's `_both` rows: read from the full payload when the route sent it, else from each pair session's rows. */
function usePairBoth(pairSessions: BenchSession[], payload: BenchPayload | undefined): Both {
  const [fetched, setFetched] = useState<Record<string, Partial<LabBench>>>({});
  const ids = pairSessions.map((x) => x.source ?? x.id);
  const full = (payload?.sweeps?.length ?? 0) > 0;
  const key = ids.join("|");
  useEffect(() => {
    if (full) return;
    for (const x of pairSessions) {
      const id = x.source ?? x.id;
      if (fetched[id]) continue;
      fetch(spotUrl(x.rows_url ?? `/api/learning/lab/bench/rows?session=${encodeURIComponent(id)}`))
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
        .then((d: Partial<LabBench>) => setFetched((m) => ({ ...m, [id]: d })))
        .catch(() => undefined);
    }
  }, [key, full]);
  return useMemo(() => {
    const out: Both = new Map();
    if (full) bothFrom(payload as Partial<LabBench>, out);
    else for (const id of ids) bothFrom(fetched[id], out);
    for (const [b, v] of [...out]) if (v.j === null && v.watts === null) out.delete(b);
    return out;
  }, [fetched, full, payload, key]);
}

function BiggerBrain({ story, reduced, both }: { story: ReturnType<typeof speedStory>; reduced: boolean; both: Both }) {
  const t = useChartTokens();
  const { pair, base } = story;
  const hasBoth = !!pair && pair.points.some((p) => both.get(p.batch)?.j);
  // with both pods' rows the pair is the whole machine; without them it is pod C's meter only and is not set against the 27B
  const pj = (p: BenchHeadline) => (hasBoth ? (both.get(p.batch)?.j ?? null) : p.j_per_token);
  const pw = (p: BenchHeadline) => (hasBoth ? (both.get(p.batch)?.watts ?? null) : p.watts);
  const pb = modelBillions(pair?.session.model);
  const bb = modelBillions(base?.session.model);
  const pairName = `${pb ?? "Big"}B across two pods, ${hasBoth ? "both pods" : "pod C's meter only (the head)"}`;
  const option = useMemo<EChartsOption | null>(() => {
    if (!pair) return null;
    const axis = chartAxis(t);
    const maxB = Math.max(2, ...pair.points.map((p) => p.batch));
    const pairData = lineData(pair, pj);
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
            const q = at(p.seriesId === "pair" ? pair : base, batch);
            const w = q ? (p.seriesId === "pair" ? pw(q) : q.watts) : null;
            return `<span style="color:${p.color}">&#9679;</span> ${p.seriesName}: <b>${fmt(p.value[1])}</b> J per token${q?.tok_s_total ? `, ${fmt(q.tok_s_total)} tok/s in total` : ""}${w ? `, ${fmt(w, 0)} W` : ""}`;
          });
          return [`<b>${streams(batch)}</b>`, ...rows].join("<br/>");
        },
      },
      xAxis: { ...axis, type: "log", logBase: 2, min: 1, max: maxB, name: "streams answered at once", nameLocation: "middle", nameGap: 32, splitLine: { show: false }, axisLabel: { ...axis.axisLabel, formatter: (v: number) => String(Math.round(v)) } },
      yAxis: { ...axis, type: "log", logBase: 10, name: "joules per token (log)", nameLocation: "end", nameGap: 14, axisLabel: { ...axis.axisLabel, formatter: (v: number) => String(Number(v.toPrecision(2))) } },
      series: [
        series(t, "pair", pairName, t.accent, pairData, { connectNulls: true }),
        ...(base && hasBoth ? [series(t, "base", `${bb ?? "Small"}B on one pod`, t.accent3, lineData(base, (p) => p.j_per_token))] : []),
      ],
    };
  }, [t, reduced, pair, base, hasBoth, both]);

  const shared = pair && base && hasBoth ? pair.points.filter((p) => at(base, p.batch)?.j_per_token && pj(p)).map((p) => p.batch) : [];
  const top = shared.length ? shared[shared.length - 1]! : null;
  const pairJ = top !== null ? pj(at(pair, top)!) : null;
  const baseJ = top !== null ? at(base, top)?.j_per_token : null;
  const change = pairJ && baseJ ? (pairJ / baseJ - 1) * 100 : null;
  const times = pb && bb ? pb / bb : null;
  const missing = pair ? [1, 4, 16].filter((b) => !pair.points.some((p) => p.batch === b)) : [];

  return (
    <section className={s.section} aria-labelledby="speed-brain">
      <div className={s.kicker}>
        <span className={s.step}>2</span> The bigger brain
      </div>
      <h2 id="speed-brain" className={s.h2}>
        Two pods together run a model {times ? `${fmt(times, 1)} times` : "several times"} the size
      </h2>
      <p className={s.why}>Two chips hold what one cannot.</p>
      {pair && option ? (
        <div className={s.brain}>
          <div className={s.chartCard}>
            <LabEChart option={option} height={330} label={`Energy per token against streams answered at once: ${pairName}${base && hasBoth ? ", and the 27B on one pod" : ""}`} />
          </div>
          <table className={s.table}>
            <thead>
              <tr>
                <th scope="col">Streams</th>
                <th scope="col">Tokens a second</th>
                <th scope="col">{hasBoth ? "Watts, both pods" : "Watts, pod C only"}</th>
                <th scope="col">{hasBoth ? "J per token, both pods" : "J per token, pod C only"}</th>
              </tr>
            </thead>
            <tbody>
              {pair.points.map((p) => (
                <tr key={p.batch}>
                  <th scope="row">{p.batch}</th>
                  <td>{p.tok_s_total !== null ? fmt(p.tok_s_total) : "no row"}</td>
                  <td>{pw(p) !== null ? fmt(pw(p)!, 0) : "no row"}</td>
                  <td>
                    {pj(p) !== null ? fmt(pj(p)!) : "no row"}
                    {isLowConfidence(p.confidence) ? <span className={s.lowDot} title="low confidence" aria-label="low confidence" /> : null}
                  </td>
                </tr>
              ))}
            </tbody>
            <caption className={s.caption}>
              {pb ? `${pb}B` : "The big model"} split across pod C and pod D, every number from its own row.
              {missing.length ? ` No row yet at ${missing.map(streams).join(", ")}.` : ""}
            </caption>
          </table>
        </div>
      ) : (
        <p className={s.empty}>The pair rows are not on this box yet. This section draws the big model across two pods as soon as its bench rows arrive; nothing is drawn in their place.</p>
      )}
      {pair && !hasBoth ? (
        <p className={s.foot}>These watts and joules are pod C&apos;s meter only (the head); pod D&apos;s share is not in these rows yet, so they are not set against the 27B on one pod.</p>
      ) : null}
      {change !== null && top !== null ? (
        <p className={s.story}>
          A bigger brain costs {change >= 0 ? "more" : "fewer"} joules per token; under load ({streams(top)}), {fmt(Math.abs(change), 0)} percent {change >= 0 ? "more" : "fewer"} for {times ? `${fmt(times, 1)} times` : "a much bigger"} the model
          {pairJ && baseJ ? ` (${fmt(pairJ)} against ${fmt(baseJ)} J per token)` : ""}.
        </p>
      ) : null}
    </section>
  );
}

/* ---------- 3. what it costs to be on ---------- */

function PowerBars({ nodes, reduced, onPick, both, brain }: { nodes: BenchPowerNode[]; reduced: boolean; onPick: (p: BenchPoint) => void; both: Both; brain: SharedBrainPair | null }) {
  const t = useChartTokens();
  // one row per real machine: rows under an older node name of the same machine merge, the newest row winning
  const shown = useMemo(() => {
    const newest = (list: (BenchPoint | null)[]) => list.filter((x): x is BenchPoint => !!x).sort((a, c) => (c.ts ?? "").localeCompare(a.ts ?? ""))[0] ?? null;
    const known = MACHINES.map((m) => {
      const ns = nodes.filter((n) => m.nodes.includes(n.node));
      return { ...(ns[0] ?? { idle_rows: 0, serving_rows: 0, peak: null }), node: m.nodes[0]!, idle: newest(ns.map((n) => n.idle)), serving: newest(ns.map((n) => n.serving)) } as BenchPowerNode;
    });
    const rest = nodes.filter((n) => !MACHINES.some((m) => m.nodes.includes(n.node)));
    return [...known, ...rest]
      .map((n) => {
        if (!PAIR_NODES.includes(n.node) || !n.serving) return n;
        // the pair's power rows are pod C's meter only; draw the pair with both pods' watts at the same batch, or not at all
        const batch = Number(/batch (\d+)/.exec(n.serving.state ?? "")?.[1]) || [...both.keys()].sort((a, c) => c - a)[0];
        const w = batch ? both.get(batch)?.watts : null;
        return w ? { ...n, serving: { ...n.serving, value: w, label: "both pods' watts", state: `batch ${batch}, both pods summed`, formula: "pair_pp122b.watts_avg_both" } } : { ...n, serving: null };
      })
      .filter((n) => n.idle || n.serving);
  }, [nodes, both]);
  const nowTicks = useMemo(
    () =>
      brain?.configured
        ? shown.flatMap((n, i) => {
            const m = brain.members.find((x) => x.node === n.node);
            return m && m.online && m.watts_now !== null ? [{ i, w: m.watts_now }] : [];
          })
        : [],
    [brain, shown],
  );
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
          const p = raw as { seriesId: string; dataIndex: number; value: unknown };
          if (p.seriesId === "now") {
            const v = p.value as [number, number];
            return `<b>${machineName(shown[v[1]]?.node)}</b> right now: <b>${fmt(v[0], 1)} W</b>`;
          }
          const n = shown[p.dataIndex];
          const pt = n?.[p.seriesId as "idle" | "serving"];
          if (!n || !pt) return "";
          return `<b>${machineName(n.node)}</b>, ${p.seriesId === "idle" ? "on and idle" : "answering"}: <b>${fmt(pt.value)} W</b>${pt.state ? `<br/><span style="color:${t.muted}">${pt.state.length > 90 ? `${pt.state.slice(0, 90)}...` : pt.state}</span>` : ""}${pt.low_confidence ? `<br/><span style="color:${t.watch}">low confidence</span>` : ""}`;
        },
      },
      xAxis: { ...axis, type: "value", name: "watts", nameLocation: "end", nameGap: 8, splitNumber: 3 },
      yAxis: { ...axis, type: "category", inverse: true, data: shown.map((n) => machineName(n.node)), axisLabel: { ...axis.axisLabel, color: t.text, fontSize: 13 }, splitLine: { show: false } },
      series: [
        bar("idle", "On, idle", t.unknown),
        bar("serving", "Answering", t.accent),
        // a bright tick at each pod's watts right now, only while the shared-brain route answers with a reading
        ...(nowTicks.length
          ? [
              {
                id: "now",
                name: "now",
                type: "scatter" as const,
                symbol: "rect",
                symbolSize: [3, 30],
                color: t.watch,
                itemStyle: { color: t.watch, shadowBlur: 8, shadowColor: t.watch },
                z: 5,
                data: nowTicks.map((n) => ({ value: [n.w, n.i] })),
              },
            ]
          : []),
      ],
    };
  }, [t, reduced, shown, nowTicks]);
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

/* ---------- 4. how the two pods talk ---------- */

/** The link and process-group rows (families link and dist): from the full payload, else from their sessions' rows. */
function useTalkRows(sessions: BenchSession[], payload: BenchPayload | undefined): Map<string, BenchPoint[]> {
  const [fetched, setFetched] = useState<Record<string, Partial<LabBench>>>({});
  const full = (payload?.series?.length ?? 0) > 0;
  const mine = sessions.filter((x) => (x.families ?? []).some((f) => f === "link" || f === "dist"));
  const key = mine.map((x) => x.source ?? x.id).join("|");
  useEffect(() => {
    if (full) return;
    for (const x of mine) {
      const id = x.source ?? x.id;
      if (fetched[id]) continue;
      fetch(spotUrl(x.rows_url ?? `/api/learning/lab/bench/rows?session=${encodeURIComponent(id)}`))
        .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
        .then((d: Partial<LabBench>) => setFetched((m) => ({ ...m, [id]: d })))
        .catch(() => undefined);
    }
  }, [key, full]);
  return useMemo(() => {
    const out = new Map<string, BenchPoint[]>();
    const take = (p: Partial<LabBench> | undefined) => {
      for (const sr of p?.series ?? []) {
        if (!/^(link|dist)\./.test(sr.metric)) continue;
        const list = out.get(sr.metric) ?? [];
        for (const q of sr.points) if (!list.some((x) => x.line === q.line)) list.push(q);
        out.set(sr.metric, list.sort((a, c) => (a.line ?? 0) - (c.line ?? 0)));
      }
    };
    if (full) take(payload as Partial<LabBench>);
    else for (const id of Object.keys(fetched)) take(fetched[id]);
    return out;
  }, [fetched, full, payload]);
}

const median = (xs: number[]) => {
  const v = xs.slice().sort((a, c) => a - c);
  return v.length ? (v.length % 2 ? v[(v.length - 1) / 2]! : (v[v.length / 2 - 1]! + v[v.length / 2]!) / 2) : null;
};

function PodsTalk({ rows }: { rows: Map<string, BenchPoint[]> }) {
  const get = (m: string) => rows.get(m) ?? [];
  // the tunnel's throughput, both directions (R66 tunnel-chisel-probe)
  const send = get("link.pair.tunnel_send_mb_s").map((p) => p.value);
  const sendMb = send.length ? send.reduce((a, c) => a + c, 0) / send.length : null;
  // R66 measured each setting once without and then with the quick-ACK relay; the rows are in that order in the file,
  // and the gloo rows also name their before or after fixture
  const two = get("link.pair.two_write_rtt_ms");
  const rttBefore = two.length >= 2 ? two[0]!.value : null;
  const rttAfter = two.length ? two[two.length - 1]!.value : null;
  const gloo = get("dist.pair.gloo_small_allreduce_ms");
  const gBefore = (gloo.find((p) => /before/i.test(p.recorded_by ?? "")) ?? (gloo.length >= 2 ? gloo[0] : undefined))?.value ?? null;
  const gAfter = (gloo.find((p) => /after/i.test(p.recorded_by ?? "")) ?? (gloo.length >= 2 ? gloo[gloo.length - 1] : undefined))?.value ?? null;
  const bigMb = get("dist.pair.gloo_allreduce_mb_s").at(-1)?.value ?? null;
  const init = get("dist.pair.gloo_init_s").at(-1)?.value ?? null;
  // R65: each training run measured one delta exchange over the hub proxy and then over the Tailscale relay, in that order
  const ex = get("link.exchange_s");
  const byRun = new Map<string, BenchPoint[]>();
  for (const p of ex) byRun.set(p.recorded_by ?? "", [...(byRun.get(p.recorded_by ?? "") ?? []), p]);
  const pairs = [...byRun.values()].filter((l) => l.length >= 2);
  const hub = median(pairs.map((l) => l[0]!.value));
  const relay = pairs.map((l) => l[l.length - 1]!.value);
  const relayMed = median(relay);
  const bytes = get("link.exchange_bytes").at(-1)?.value ?? null;
  const ratio = hub && relayMed ? relayMed / hub : null;
  const stallMax = Math.max(gBefore ?? 0, gAfter ?? 0, 1);
  return (
    <section className={s.section} aria-labelledby="speed-talk">
      <div className={s.kicker}>
        <span className={s.step}>4</span> How the two pods talk
      </div>
      <h2 id="speed-talk" className={s.h2}>
        A network built where none was allowed
      </h2>
      <p className={s.why}>No network between the pods, so we built one.</p>
      <div className={s.talk}>
        <div className={s.talkCol}>
          <h3 className={s.talkHead}>For inference</h3>
          <div className={`${s.tiles} ${s.talkTiles}`}>
            <Tile value={sendMb !== null ? `${fmt(sendMb, 0)} MB/s` : null} label="through the cloud's own proxy" sub={`each way, ${send.length} directions measured`} why="A tunnel where no network was allowed." missing="No tunnel throughput row yet." />
            <Tile value={rttAfter !== null ? `${fmt(rttAfter)} ms` : null} label="per message" sub={rttBefore !== null ? `${fmt(rttBefore)} ms before the quick-ACK relay` : "one setting measured"} why="A 40 ms stall found and removed." missing="No tunnel round-trip row yet." />
            <Tile
              value={gAfter !== null ? `${fmt(gAfter)} ms` : null}
              label="to agree across two pods"
              sub={[gBefore !== null ? `${fmt(gBefore)} ms before the relay` : null, bigMb !== null ? `64 MB all-reduce at ${fmt(bigMb)} MB/s` : null, init !== null ? `group up in ${fmt(init)} s` : null].filter(Boolean).join("; ")}
              why="Two chips act as one group."
              missing="No all-reduce row yet."
            />
          </div>
        </div>
        <div className={s.talkCol}>
          <h3 className={s.talkHead}>For training</h3>
          <div className={`${s.tiles} ${s.talkTiles}`}>
            <Tile
              value={bytes !== null && hub !== null ? `${fmt(bytes / 1e6, 0)} MB in ${fmt(hub, 1)} s` : null}
              label="of deltas per round"
              sub={`over the hub proxy, median of ${pairs.length} runs`}
              why="Only the change moves, in 8 bits."
              missing="No delta exchange row yet."
            />
            <Tile
              value={ratio !== null ? `${fmt(ratio, 0)}x` : null}
              label="faster than the relay path"
              sub={relay.length ? `${fmt(Math.min(...relay), 1)} to ${fmt(Math.max(...relay), 1)} s over the Tailscale relay` : ""}
              why="Same pods, better road."
              missing="Needs both paths measured in one run."
            />
          </div>
        </div>
      </div>
      {gBefore !== null && gAfter !== null ? (
        <div className={s.stall} role="img" aria-label={`Small all-reduce median: ${fmt(gBefore)} ms before the quick-ACK relay, ${fmt(gAfter)} ms after`}>
          <span className={s.foot}>Before the relay</span>
          <span className={s.stallBar} style={{ width: `${(gBefore / stallMax) * 100}%` }} />
          <b>{fmt(gBefore)} ms</b>
          <span className={s.foot}>After</span>
          <span className={`${s.stallBar} ${s.stallAfter}`} style={{ width: `${(gAfter / stallMax) * 100}%` }} />
          <b>{fmt(gAfter)} ms</b>
        </div>
      ) : null}
    </section>
  );
}

/* ---------- 5. sessions and rows ---------- */

function What({ story }: { story: ReturnType<typeof speedStory> }) {
  const from = (l: StoryLine | null) => (l ? `${sessionTitle(l.session)} (${l.session.id})` : "no session in these rows");
  const lines: [string, string][] = [
    ["Streams are how many people the teacher model answers at the same moment (the batch size of each row).", "conditions.batch_size of each row"],
    ["The chip fix: vLLM on one Strix Halo pod after the kernel fix, the newest vLLM session on that chip.", from(story.after)],
    ["The 'before' in the first tile: the newest older vLLM session of the same model that was slower at one stream.", from(story.before)],
    ["llama.cpp Vulkan: its sessions on one Strix Halo pod and model joined, the newest row at each batch size, every point measured.", story.llamaSessions.map((x) => x.id).join(", ") || "no session in these rows"],
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
  llamaNote,
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
  llamaNote: string | null;
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
      {llamaNote ? <p className={s.foot}>{llamaNote}</p> : null}
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
  initialSweep = null,
  forceReduced,
}: {
  initialBatch: number | null;
  initialTtft: boolean;
  initialPoint: string | null;
  initialDrawer: string | null;
  initialSweep?: string | null;
  forceReduced: boolean;
}) {
  const bench = useFetch((signal) => api.labBench(signal), "bench");
  const reduced = useReduced(forceReduced);
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
  const pairSessions = useMemo(() => summary.sessions.filter(isPair), [summary]);
  const live = useSharedBrain();
  const talkRows = useTalkRows(summary.sessions, payload);
  // no live route for the box's own watts: its newest power row, shown as "last reading"
  const boxReading = useMemo(() => {
    const j = (payload?.power?.nodes ?? []).find((n) => n.node === "jetson");
    return [j?.idle ?? null, j?.serving ?? null].filter((x): x is BenchPoint => !!x).sort((a, c) => (c.ts ?? "").localeCompare(a.ts ?? ""))[0] ?? null;
  }, [payload]);
  const llamaNote = useMemo(() => {
    const l = story.llama;
    if (!l || story.llamaSessions.length < 2) return null;
    const parts = story.llamaSessions.map((x) => `${l.points.filter((p) => p.session === x.id).map((p) => p.batch).join(" and ")} from ${x.id}`);
    const engines = [...new Set(story.llamaSessions.map((x) => x.engine))];
    const models = [...new Set(story.llamaSessions.map((x) => x.model ?? "model not named"))];
    const differs = [engines.length > 1 ? `engines differ (${engines.join(", ")})` : null, models.length > 1 ? `models differ (${models.join(", ")})` : null].filter(Boolean);
    return `llama.cpp's line on the first chart joins ${story.llamaSessions.length} sessions on ${machineName(l.session.node)}, streams ${parts.join("; ")}. ${differs.length ? `Note: ${differs.join("; ")}.` : `Same engine (${engines[0]}) and model (${models[0]}) in each.`} The chip is the same.`;
  }, [story]);
  const both = usePairBoth(pairSessions, payload);
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

      <LiveStrip brain={live.brain} fresh={live.fresh} box={boxReading} reduced={reduced} />
      <LivePower poll={live.poll} reduced={reduced} chat />
      <ChipFix story={story} reduced={reduced} onPick={pickFromChart} />
      <BiggerBrain story={story} reduced={reduced} both={both} />

      <section className={s.section} aria-labelledby="speed-power">
        <div className={s.kicker}>
          <span className={s.step}>3</span> What it costs to be on
        </div>
        <h2 id="speed-power" className={s.h2}>
          Idle and answering watts, one row per machine
        </h2>
        <p className={s.why}>Idle is cheap; answering is the cost.</p>
        <div className={s.chartCard}>
          <PowerBars nodes={nodes} reduced={reduced} onPick={pickPower} both={both} brain={live.brain} />
          <div className={s.chartFoot}>
            <span className={s.foot}>
              <span className={s.lowDot} aria-hidden="true" /> marks a low-confidence row. Tap a bar for its row.
            </span>
          </div>
        </div>
        {nodes.some((n) => PAIR_NODES.includes(n.node) && n.serving) && ![...both.values()].some((v) => v.watts) ? (
          <p className={s.foot}>Pods C and D together: their power rows are pod C&apos;s meter only, so the pair is not drawn here until both pods&apos; rows arrive.</p>
        ) : null}
        {nodes.some((n) => n.node === "pod-c") ? (
          <p className={s.foot}>Rows recorded under node pod-c (R29, 2026-10-10) are pod C under its lab name; they count as pod C here, and the newest row of each kind wins.</p>
        ) : null}
        {data.power?.absent?.length ? <p className={s.foot}>No power row yet for {data.power.absent.join(", ")}; nothing is drawn for a machine the rows do not cover.</p> : null}
      </section>

      <PodsTalk rows={talkRows} />

      <details id="speed-drawer" className={s.drawer} open={open} onToggle={(e) => setOpen((e.currentTarget as HTMLDetailsElement).open)}>
        <summary className={s.drawerHead}>
          <span>Sessions and rows</span>
          <span className={s.why}>Every row, with its source.</span>
          <span className={s.foot}>
            {sessions.length} sessions with numbers, {rows} rows
          </span>
        </summary>
        {open ? (
          <Sessions sessions={sessions} points={summary.points} selected={current?.id ?? null} setSelected={setSelected} findPoint={findPoint} picked={picked} setPicked={setPicked} rows={rows} openSheet={setSheet} llamaNote={llamaNote} rowsState={fullHere || !current ? null : rowsOf[current.source ?? current.id] === undefined ? "loading" : typeof rowsOf[current.source ?? current.id] === "string" ? (rowsOf[current.source ?? current.id] as "loading" | "error") : "ready"} />
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
