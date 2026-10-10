"use client";

import { useRef, type PointerEvent } from "react";
import type { SimResult, Tick } from "@/lib/api";
import { activityAt, clock, dayClock, ms, riskPercent, type PlaceIndex } from "@/lib/dayrun";
import ui from "../ui.module.css";
import dr from "./dayrun.module.css";

/**
 * The three meters across the 24 hours, from `timeline`: energy and focus
 * (0 to 100) and the worst deadline's risk (0 to 1, drawn as a percentage so
 * all three share one axis). Fixed and planned blocks are shaded behind the
 * lines. Dragging on the chart, or the slider under it, moves the scrubber.
 */

const W = 360;
const H = 214;
const LEFT = 30;
const RIGHT = 62;
const TOP = 10;
const BOTTOM = 24;
const PW = W - LEFT - RIGHT;
const PH = H - TOP - BOTTOM;

const SERIES = [
  { id: "energy", name: "Energy", cls: dr.sEnergy, value: (t: Tick) => t.energy, show: (t: Tick) => String(Math.round(t.energy)) },
  { id: "focus", name: "Focus", cls: dr.sFocus, value: (t: Tick) => t.focus, show: (t: Tick) => String(Math.round(t.focus)) },
  { id: "risk", name: "Deadline risk", cls: dr.sRisk, value: (t: Tick) => t.deadline_risk * 100, show: (t: Tick) => riskPercent(t.deadline_risk) },
] as const;

const END_LABEL = { energy: "Energy", focus: "Focus", risk: "Risk" } as const;

export function Meters({ result, places, scrub, onScrub }: { result: SimResult; places: PlaceIndex; scrub: number; onScrub: (index: number) => void }) {
  const svg = useRef<SVGSVGElement>(null);
  const ticks = result.timeline;
  const t0 = ms(result.start);
  const t1 = ms(result.end);
  const x = (t: number) => LEFT + ((t - t0) / (t1 - t0 || 1)) * PW;
  const y = (v: number) => TOP + (1 - Math.max(0, Math.min(100, v)) / 100) * PH;

  if (ticks.length < 2) return <p className={ui.muted}>The box returned no timeline for this plan.</p>;

  const at = Math.max(0, Math.min(ticks.length - 1, scrub));
  const now = ticks[at] as Tick;
  const last = ticks[ticks.length - 1] as Tick;

  // Hour marks on the board's clock: a tick every 3 hours, a label every 6.
  const marks: { t: number; label: string | null }[] = [];
  const startClock = clock(result.start);
  const startMin = Number(startClock.slice(0, 2)) * 60 + Number(startClock.slice(3, 5));
  for (let m = Math.ceil(startMin / 180) * 180; ; m += 180) {
    const t = t0 + (m - startMin) * 60000;
    if (t > t1) break;
    const hour = (m / 60) % 24;
    marks.push({ t, label: hour % 6 === 0 ? `${String(hour).padStart(2, "0")}:00` : null });
  }

  // Direct labels at the right end, nudged apart when two lines finish close together.
  const ends = SERIES.map((s) => ({ s, y: y(s.value(last)) })).sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) {
    const prev = ends[i - 1];
    const cur = ends[i];
    if (prev && cur && cur.y - prev.y < 13) cur.y = prev.y + 13;
  }
  const overflow = (ends[ends.length - 1]?.y ?? 0) - (TOP + PH);
  if (overflow > 0) for (const e of ends) e.y -= overflow;

  const pick = (e: PointerEvent<SVGSVGElement>) => {
    const box = svg.current?.getBoundingClientRect();
    if (!box || !box.width) return;
    const px = ((e.clientX - box.left) / box.width) * W;
    const f = Math.max(0, Math.min(1, (px - LEFT) / PW));
    onScrub(Math.round(f * (ticks.length - 1)));
  };

  const hourly = ticks.filter((t) => clock(t.t).endsWith(":00") || t === ticks[0]);

  return (
    <div className={dr.chartWrap}>
      <div className={dr.readout} aria-live="off">
        <p className={dr.readoutTime}>
          <strong>{dayClock(now.t)}</strong> <span className={ui.breakWords}>{activityAt(result, at, places)}</span>
        </p>
        <ul className={dr.legend}>
          {SERIES.map((s) => (
            <li key={s.id}>
              <span className={`${dr.key} ${s.cls}`} aria-hidden="true" />
              {s.name} <strong className={dr.readoutValue}>{s.show(now)}</strong>
            </li>
          ))}
        </ul>
      </div>

      <svg
        ref={svg}
        className={dr.chart}
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Energy, focus and deadline risk from ${dayClock(result.start)} to ${dayClock(result.end)}. Energy ends at ${Math.round(last.energy)}, focus at ${Math.round(last.focus)}, risk at ${riskPercent(last.deadline_risk)}.`}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          pick(e);
        }}
        onPointerMove={(e) => {
          if (e.buttons || e.pointerType === "touch") pick(e);
        }}
      >
        {result.blocks.map((b) => {
          const x0 = Math.max(LEFT, x(ms(b.start)));
          const x1 = Math.min(LEFT + PW, x(ms(b.end)));
          if (x1 <= x0) return null;
          return <rect key={b.id} className={b.fixed ? dr.bandFixed : dr.bandPlanned} x={x0} y={TOP} width={Math.max(1, x1 - x0 - 0.6)} height={PH} />;
        })}

        {[0, 50, 100].map((v) => (
          <g key={v}>
            <line className={v === 0 ? dr.axis : dr.grid} x1={LEFT} x2={LEFT + PW} y1={y(v)} y2={y(v)} />
            <text className={dr.axisText} x={LEFT - 6} y={y(v) + 4} textAnchor="end">
              {v}
            </text>
          </g>
        ))}
        {marks.map((m) => (
          <g key={m.t}>
            <line className={dr.axis} x1={x(m.t)} x2={x(m.t)} y1={TOP + PH} y2={TOP + PH + (m.label ? 5 : 3)} />
            {m.label ? (
              <text className={dr.axisText} x={x(m.t)} y={H - 6} textAnchor="middle">
                {m.label}
              </text>
            ) : null}
          </g>
        ))}

        {SERIES.map((s) => (
          <polyline key={s.id} className={`${dr.line} ${s.cls}`} points={ticks.map((t) => `${x(ms(t.t)).toFixed(1)},${y(s.value(t)).toFixed(1)}`).join(" ")} />
        ))}

        {ends.map(({ s, y: ly }) => (
          <g key={s.id}>
            <line className={`${dr.line} ${s.cls}`} x1={LEFT + PW + 5} x2={LEFT + PW + 15} y1={ly} y2={ly} />
            <text className={dr.endLabel} x={LEFT + PW + 19} y={ly + 4}>
              {END_LABEL[s.id]}
            </text>
          </g>
        ))}

        <line className={dr.cursor} x1={x(ms(now.t))} x2={x(ms(now.t))} y1={TOP} y2={TOP + PH} />
        {SERIES.map((s) => (
          <circle key={s.id} className={`${dr.cursorDot} ${s.cls}`} cx={x(ms(now.t))} cy={y(s.value(now))} r={4.5} />
        ))}
      </svg>

      <input
        className={dr.scrubber}
        type="range"
        min={0}
        max={ticks.length - 1}
        step={1}
        value={at}
        aria-label="Time of day"
        aria-valuetext={`${dayClock(now.t)}: ${activityAt(result, at, places)}`}
        onChange={(e) => onScrub(Number(e.target.value))}
      />

      <p className={dr.legend}>
        <span>
          <span className={`${dr.swatch} ${dr.swatchFixed}`} aria-hidden="true" />
          fixed blocks
        </span>
        <span>
          <span className={`${dr.swatch} ${dr.swatchPlanned}`} aria-hidden="true" />
          your moves
        </span>
        <span className={ui.muted}>One scale, 0 to 100. Risk is the worst deadline, as a percentage.</span>
      </p>

      <details className={dr.details}>
        <summary>Show the meters as a table</summary>
        <table className={`${ui.table} ${dr.spaced}`}>
          <thead>
            <tr>
              <th>Time</th>
              <th className={ui.num}>Energy</th>
              <th className={ui.num}>Focus</th>
              <th className={ui.num}>Risk</th>
              <th>Doing</th>
            </tr>
          </thead>
          <tbody>
            {hourly.map((t) => (
              <tr key={t.t}>
                <td>{clock(t.t)}</td>
                <td className={ui.num}>{Math.round(t.energy)}</td>
                <td className={ui.num}>{Math.round(t.focus)}</td>
                <td className={ui.num}>{riskPercent(t.deadline_risk)}</td>
                <td>{t.activity}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </details>
    </div>
  );
}
