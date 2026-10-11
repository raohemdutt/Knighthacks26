"use client";

import { ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { api } from "@/lib/api";
import { COMPARE_ROWS, type CompareRun, type CompareTrace, fmtRatio, fmtValue, runName } from "@/lib/labCompare";
import { useFetch } from "@/lib/useFetch";
import b from "./bench.module.css";
import c from "./compare.module.css";

/**
 * Alone vs together (L31): two training runs side by side, by default the latest solo run against the latest pooled
 * one (GET /api/learning/lab/compare). Per run and as the second over the first: examples, members, seconds per
 * stage, the joules of labeling (pod C), training and packaging (pod D) and the box, each with its idle floor beside
 * (T4: gross joules, the idle shown, never subtracted), then the three power traces on one time axis.
 *
 *   /lab/?tab=compare&a=<run_id>&b=<run_id>
 */

const NODES = ["pod-c", "pod-d", "box"] as const;

function Panel({ title, a, bTrace, maxT }: { title: string; a?: CompareTrace; bTrace?: CompareTrace; maxT: number }) {
  const W = 600;
  const H = 120;
  const pad = { l: 40, r: 8, t: 10, b: 20 };
  const all = [...(a?.points ?? []), ...(bTrace?.points ?? [])];
  if (!all.length) {
    return (
      <div className={c.panel}>
        <span className={c.panelTitle}>{title}</span>
        <span className={b.note}>No power trace for either run.</span>
      </div>
    );
  }
  const maxW = Math.max(10, ...all.map((p) => p[1])) * 1.08;
  const x = (t: number) => pad.l + (Math.max(0, Math.min(t, maxT)) / maxT) * (W - pad.l - pad.r);
  const y = (w: number) => pad.t + (1 - w / maxW) * (H - pad.t - pad.b);
  const line = (t?: CompareTrace) => (t?.points ?? []).filter((p) => p[0] >= 0 && p[0] <= maxT).map((p) => `${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join(" ");
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => Math.round(f * maxT));
  return (
    <div className={c.panel}>
      <span className={c.panelTitle}>{title}</span>
      <svg viewBox={`0 0 ${W} ${H}`} className={c.svg} role="img" aria-label={`${title}: watts over the run, both runs`}>
        {[0, 0.5, 1].map((f) => (
          <g key={f}>
            <line x1={pad.l} x2={W - pad.r} y1={y(f * maxW)} y2={y(f * maxW)} className={c.grid} />
            <text x={pad.l - 6} y={y(f * maxW) + 4} className={c.axis} textAnchor="end">
              {Math.round(f * maxW)}
            </text>
          </g>
        ))}
        {ticks.map((t) => (
          <text key={t} x={x(t)} y={H - 4} className={c.axis} textAnchor="middle">
            {t} s
          </text>
        ))}
        <polyline points={line(a)} className={`${c.line} ${c.lineA}`} />
        <polyline points={line(bTrace)} className={`${c.line} ${c.lineB}`} />
      </svg>
    </div>
  );
}

function Traces({ ra, rb }: { ra: CompareRun; rb: CompareRun }) {
  const maxT = Math.max(ra.seconds.total ?? 0, rb.seconds.total ?? 0, 10);
  return (
    <div className={b.card}>
      <div className={b.cardHead}>
        <h3>Power, both runs on one clock</h3>
        <span className={c.legend}>
          <span className={c.keyA}>{runName(ra)}</span>
          <span className={c.keyB}>{runName(rb)}</span>
        </span>
      </div>
      <span className={b.note}>Watts from each run&apos;s own start. Seconds across, watts up.</span>
      {NODES.map((n) => (
        <Panel key={n} title={ra.traces[n]?.name ?? rb.traces[n]?.name ?? n} a={ra.traces[n]} bTrace={rb.traces[n]} maxT={maxT} />
      ))}
    </div>
  );
}

function Picker({ label, value, runs, onPick }: { label: string; value: string | null; runs: { run_id: string; label: string }[]; onPick: (id: string) => void }) {
  return (
    <label className={c.pick}>
      <span>{label}</span>
      <select className={ui.select} value={value ?? ""} onChange={(e) => onPick(e.target.value)}>
        {runs.map((r) => (
          <option key={r.run_id} value={r.run_id}>
            {r.label}
          </option>
        ))}
      </select>
    </label>
  );
}

export function Compare({ a, b: bId, onPick }: { a: string | null; b: string | null; onPick: (which: "a" | "b", id: string) => void }) {
  const res = useFetch((signal) => api.labCompare(a, bId, signal), `compare-${a ?? ""}-${bId ?? ""}`);
  if (res.loading && !res.data) return <Loading what="the comparison" />;
  if (res.error && !res.data) return <ErrorBox what="the comparison" error={res.error} onRetry={res.reload} />;
  const data = res.data!;
  const { a: ra, b: rb, ratios } = data;
  const choices = data.runs.map((r) => ({
    run_id: r.run_id,
    label: `${runName(r)}${r.measured ? "" : " · no power trace"} · ${r.run_id}`,
  }));

  return (
    <div className={b.page}>
      <div className={ui.pageHead}>
        <h1>Alone vs together</h1>
        <p className={b.sub}>
          The same task trained twice: once on one person&apos;s examples, once pooled with a friend&apos;s. Joules are gross, as the meters read them, with
          the idle floor beside each one, never subtracted.
        </p>
      </div>

      {choices.length ? (
        <div className={c.pickers}>
          <Picker label="First run" value={data.asked.a} runs={choices} onPick={(id) => onPick("a", id)} />
          <Picker label="Second run" value={data.asked.b} runs={choices} onPick={(id) => onPick("b", id)} />
        </div>
      ) : null}
      {data.missing.length ? <p className={b.note}>No run with the id {data.missing.join(" or ")} on this box or in the lab&apos;s records.</p> : null}

      {!ra || !rb ? (
        <div className={b.card}>
          <h3>Nothing to compare yet</h3>
          <p className={b.note}>A comparison needs two runs. The live loop (L29) leaves a record for each act; runs on this box show here too.</p>
        </div>
      ) : (
        <>
          <div className={b.card}>
            <div className={b.tableWrap}>
              <table className={`${ui.table} ${c.table}`}>
                <thead>
                  <tr>
                    <th scope="col">Measure</th>
                    <th scope="col">{runName(ra)}</th>
                    <th scope="col">{runName(rb)}</th>
                    <th scope="col">Second ÷ first</th>
                  </tr>
                </thead>
                <tbody>
                  {COMPARE_ROWS.map((row, i) => {
                    const head = i === 0 || COMPARE_ROWS[i - 1]!.group !== row.group;
                    const cell = (r: CompareRun) => {
                      const v = row.value(r);
                      const idle = row.idle?.(r);
                      return (
                        <td className={v === null || v === undefined ? c.none : c.num}>
                          {fmtValue(v, row.unit)}
                          {typeof idle === "number" ? <span className={c.idle}>idle {fmtValue(idle, "J")}</span> : null}
                        </td>
                      );
                    };
                    return [
                      head ? (
                        <tr key={`${row.group}-h`} className={c.group}>
                          <th colSpan={4} scope="colgroup">
                            {row.group}
                          </th>
                        </tr>
                      ) : null,
                      <tr key={row.key}>
                        <th scope="row">{row.label}</th>
                        {cell(ra)}
                        {cell(rb)}
                        <td className={c.ratio}>{fmtRatio(ratios?.[row.key])}</td>
                      </tr>,
                    ];
                  })}
                </tbody>
              </table>
            </div>
            {[ra, rb].map((r) =>
              r.reason ? (
                <span key={r.run_id} className={b.note}>
                  {runName(r)}: {r.reason}.
                </span>
              ) : null,
            )}
          </div>

          <Traces ra={ra} rb={rb} />

          <div className={b.card}>
            <h3>Where these numbers come from</h3>
            {[ra, rb].map((r) => (
              <span key={r.run_id} className={`${b.note} ${b.code}`}>
                {r.run_id}: {r.source.ref}
              </span>
            ))}
            {ra.source.rule ? <span className={b.note}>Rule: {ra.source.rule}.</span> : null}
            {ra.source.split ? <span className={b.note}>{ra.source.split}.</span> : null}
          </div>
        </>
      )}
    </div>
  );
}
