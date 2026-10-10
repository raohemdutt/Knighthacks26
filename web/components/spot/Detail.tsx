"use client";

import { useEffect } from "react";
import { api, spotUrl } from "@/lib/api";
import { useFetch } from "@/lib/useFetch";
import { EvidenceLine, Id, Ids, Pill, RateBars, Sev, pValue, pct, range, stamp, words } from "./bits";
import s from "./spot.module.css";
import { OrderActions } from "./Triage";
import type { ActionRow, Assumption, Finding, Locations, Metric, Overview, Sample, Series, Target, Validation, WorkOrder } from "./types";

/** The evidence panel: a waypoint, an asset, a mission, a finding's rows, the assumptions, or the data check. */
export function Detail({
  target,
  onClose,
  findings,
  locations,
  overview,
  orders,
}: {
  target: Target;
  onClose: () => void;
  findings: Finding[];
  locations: Locations | undefined;
  overview: Overview | undefined;
  orders: WorkOrder[];
}) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  const overall = locations?.overall ? locations.overall.k / Math.max(locations.overall.n, 1) : 0.18;
  let body: React.ReactNode = null;
  let title = target.id;
  if (target.kind === "asset") body = <AssetPanel id={target.id} orders={orders} />;
  else if (target.kind === "mission") body = <MissionPanel id={target.id} samples={overview?.samples} />;
  else if (target.kind === "location") {
    const loc = locations?.locations.find((l) => l.location_id === target.id);
    title = loc ? `${loc.location_id} ${loc.location_name}` : target.id;
    body = loc ? <LocationPanel loc={loc} rule={locations?.health_rule} source={locations?.zone_text_source} /> : <p>No such waypoint.</p>;
  } else if (target.kind === "finding") {
    const f = findings.find((x) => x.id === target.id);
    title = f?.title ?? target.id;
    body = f ? <FindingPanel f={f} overall={overall} /> : <p>No such finding.</p>;
  } else if (target.kind === "metric") {
    const vc = findings.find((x) => x.id === "version_comparison");
    const m = ((vc?.extra.metrics as Metric[] | undefined) ?? []).find((x) => x.key === target.id);
    title = m?.label ?? target.id;
    body = m && vc ? <MetricPanel m={m} f={vc} /> : <p>No such line.</p>;
  } else if (target.kind === "assumptions") {
    title = "Assumptions and method";
    body = <AssumptionsPanel plan={locations?.plan?.fit.note} />;
  } else if (target.kind === "validation") {
    title = "Data check";
    body = <ValidationPanel />;
  }
  return (
    <div className={s.scrim} onClick={onClose}>
      <aside className={s.drawer} role="dialog" aria-modal="true" aria-label={title} onClick={(e) => e.stopPropagation()}>
        <div className={s.drawerHead}>
          <h2>{title}</h2>
          <button type="button" className={s.close} onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>
        {body}
      </aside>
    </div>
  );
}

function Samples({ items }: { items: Sample[] }) {
  if (items.length === 0) return null;
  return (
    <div className={s.samples}>
      {items.map((x) => (
        <figure key={x.name} className={s.sample}>
          <img src={spotUrl(x.url)} alt="" loading="lazy" />
          <figcaption>{x.label}</figcaption>
        </figure>
      ))}
    </div>
  );
}

function ActionTable({ rows, showMission }: { rows: ActionRow[]; showMission?: boolean }) {
  return (
    <table className={s.tbl}>
      <thead>
        <tr>
          <th>Action</th>
          <th>What</th>
          <th>Result</th>
          <th>Reading / anomaly</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((a) => (
          <tr key={a.action_id}>
            <td>
              <Id id={a.action_id} />
              <div className={`${s.small} ${s.muted}`}>
                {showMission ? `${a.mission_id} · ` : `#${a.sequence} · `}
                {stamp(a.timestamp, Boolean(showMission))}
              </div>
            </td>
            <td>
              {words(a.action_type)}
              <div className={`${s.small} ${s.muted}`}>
                <Id id={a.asset_id} /> at <Id id={a.location_id} /> · {Math.round(a.battery_pct)}%
              </div>
            </td>
            <td>{a.status === "SUCCESS" ? <Pill tone="ok">ok</Pill> : <span className={s.failText}>failed: {words(a.failure_type)}</span>}</td>
            <td>
              {a.readings.map((r) => (
                <div key={r.reading_id} style={{ color: r.quality === "GOOD" ? undefined : "var(--watch)" }}>
                  {words(r.measurement_type)} {r.value} {r.unit} ({r.quality.toLowerCase()})
                </div>
              ))}
              {a.anomalies.map((x) => (
                <div key={x.anomaly_id}>
                  <Sev level={x.severity} /> {words(x.anomaly_type)} <span className={s.muted}>{x.anomaly_id}, {x.confirmed ? "confirmed" : "unverified"}</span>
                </div>
              ))}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function MissionPanel({ id, samples }: { id: string; samples?: Overview["samples"] }) {
  const m = useFetch((sig) => api.spotMission(id, sig), id);
  if (!m.data) return <p className={s.muted}>{m.error ? m.error.message : "Loading…"}</p>;
  const d = m.data;
  const types = [...new Set(d.timeline.filter((a) => a.status !== "SUCCESS").map((a) => a.action_type))];
  return (
    <>
      <div className={s.missionHead}>
        <Pill tone="muted">
          {d.configuration_version} · {d.scenario}
        </Pill>
        {d.hidden_failure ? <span className={s.hiddenBadge}>⚠ {d.badge}</span> : <Pill tone={d.mission_status === "COMPLETED" ? "ok" : "bad"}>{d.badge}</Pill>}
      </div>
      <dl className={s.kv}>
        <dt>Ran</dt>
        <dd>
          {stamp(d.start_time)} for {d.duration_min} min
        </dd>
        <dt>Battery</dt>
        <dd>
          {d.start_battery_pct}% to {d.end_battery_pct}%
        </dd>
        <dt>Actions</dt>
        <dd>
          {d.failed} of {d.actions} failed · {d.anomalies} anomaly rows
        </dd>
      </dl>
      <div className={s.sub}>Timeline</div>
      <ActionTable rows={d.timeline} />
      <Samples items={types.map((t) => samples?.[t]).filter((x): x is Sample => Boolean(x)).slice(0, 2)} />
    </>
  );
}

function LocationPanel({ loc, rule, source }: { loc: Locations["locations"][number]; rule?: string; source?: string }) {
  return (
    <>
      <dl className={s.kv}>
        <dt>Where</dt>
        <dd>
          Floor {loc.floor} · {words(loc.zone_type)} zone · {words(loc.environment)}
        </dd>
        {loc.zone_text ? (
          <>
            <dt>Zone</dt>
            <dd>
              {loc.zone_text} <span className={s.muted}>({source})</span>
            </dd>
          </>
        ) : null}
        <dt>Failures</dt>
        <dd>
          <b>
            {loc.k} of {loc.n}
          </b>{" "}
          actions here failed{loc.enough ? ` (${pct(loc.rate, 1)}, 95% interval ${range(loc.lo, loc.hi)})` : ": not enough data for a rate"}
        </dd>
        <dt>Route</dt>
        <dd>{loc.first_stop_of ? `First stop of ${loc.first_stop_of} missions` : "Never a mission's first stop"}</dd>
        <dt>Assets here</dt>
        <dd>{loc.assets.length ? <Ids ids={loc.assets} /> : "none in assets.csv"}</dd>
      </dl>
      <div className={s.sub}>By action type (failed / n)</div>
      <table className={s.tbl}>
        <tbody>
          {Object.entries(loc.by_type)
            .sort((a, b) => b[1].n - a[1].n)
            .map(([t, v]) => (
              <tr key={t}>
                <td>{words(t)}</td>
                <td>
                  {v.k} / {v.n}
                </td>
                <td className={s.muted}>{v.n < 8 ? "not enough data for a rate" : pct(v.k / v.n)}</td>
              </tr>
            ))}
        </tbody>
      </table>
      <div className={s.sub}>Failed actions ({loc.failed_actions.length})</div>
      {loc.failed_actions.length ? <ActionTable rows={loc.failed_actions} showMission /> : <p className={s.muted}>None.</p>}
      <div className={s.sub}>Anomalies raised here ({loc.anomalies.length})</div>
      <table className={s.tbl}>
        <tbody>
          {loc.anomalies.map((x) => (
            <tr key={x.anomaly_id}>
              <td>{x.anomaly_id}</td>
              <td>
                <Sev level={x.severity} /> {words(x.anomaly_type)}
              </td>
              <td>
                <Id id={x.asset_id} /> <Id id={x.mission_id} />
              </td>
              <td className={s.muted}>{x.confirmed ? "confirmed" : "unverified"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {rule ? <p className={s.notice}>Colour rule: {rule}</p> : null}
    </>
  );
}

/** Readings over time for one asset and one measurement, with the fleet band. Hollow points are QUESTIONABLE. */
function TrendChart({ sr }: { sr: Series }) {
  const W = 500;
  const H = 130;
  const L = 40;
  const R = 8;
  const T = 10;
  const B = 22;
  const ts = sr.points.map((p) => new Date(`${p.t}Z`).getTime());
  const vals = sr.points.map((p) => p.value);
  const lo = Math.min(...vals, sr.band?.lo ?? Infinity);
  const hi = Math.max(...vals, sr.band?.hi ?? -Infinity);
  const pad = (hi - lo || 1) * 0.08;
  const t0 = Math.min(...ts);
  const t1 = Math.max(...ts);
  const x = (t: number) => (t1 === t0 ? (L + W - R) / 2 : L + ((t - t0) / (t1 - t0)) * (W - L - R));
  const y = (v: number) => T + (1 - (v - (lo - pad)) / (hi - lo + 2 * pad)) * (H - T - B);
  const good = sr.points.filter((p) => p.good);
  return (
    <div>
      <div className={s.small}>
        <b>{words(sr.measurement_type)}</b> ({sr.unit}) · {sr.n_good} good of {sr.n} readings ·{" "}
        {sr.enough && sr.slope_per_day !== null ? `trend ${sr.slope_per_day > 0 ? "+" : ""}${sr.slope_per_day.toFixed(2)} ${sr.unit} per day` : (sr.note ?? "no trend")}
        {sr.toward_edge ? " · heading for the edge of the band" : ""}
      </div>
      <svg className={s.chart} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${sr.measurement_type} readings over time`}>
        {sr.band ? (
          <>
            <rect x={L} y={y(sr.band.hi)} width={W - L - R} height={Math.max(0, y(sr.band.lo) - y(sr.band.hi))} fill="var(--accent-soft)" />
            <line x1={L} x2={W - R} y1={y(sr.band.median)} y2={y(sr.band.median)} stroke="var(--accent)" strokeDasharray="4 4" opacity="0.6" />
            <text x={W - R} y={y(sr.band.hi) + 11} textAnchor="end">
              fleet band (ours, not a limit)
            </text>
          </>
        ) : null}
        {[lo, hi].map((v) => (
          <text key={v} x={L - 5} y={y(v) + 4} textAnchor="end">
            {v.toFixed(0)}
          </text>
        ))}
        <line x1={L} x2={L} y1={T} y2={H - B} stroke="var(--line)" />
        <line x1={L} x2={W - R} y1={H - B} y2={H - B} stroke="var(--line)" />
        {good.length > 1 ? <polyline points={good.map((p) => `${x(new Date(`${p.t}Z`).getTime())},${y(p.value)}`).join(" ")} fill="none" stroke="var(--text)" strokeWidth="1.5" opacity="0.6" /> : null}
        {sr.points.map((p) => (
          <circle key={p.reading_id} cx={x(new Date(`${p.t}Z`).getTime())} cy={y(p.value)} r="4" fill={p.good ? "var(--text)" : "var(--surface)"} stroke={p.good ? "none" : "var(--watch)"} strokeWidth="2">
            <title>{`${p.reading_id}: ${p.value} ${sr.unit}, ${p.good ? "GOOD" : "QUESTIONABLE"}, ${p.mission_id} ${stamp(p.t)}`}</title>
          </circle>
        ))}
        <text x={L} y={H - 6}>
          {stamp(sr.points[0]?.t)}
        </text>
        <text x={W - R} y={H - 6} textAnchor="end">
          {stamp(sr.points[sr.points.length - 1]?.t)}
        </text>
      </svg>
    </div>
  );
}

function AssetPanel({ id, orders }: { id: string; orders: WorkOrder[] }) {
  const a = useFetch((sig) => api.spotAsset(id, sig), id);
  if (!a.data) return <p className={s.muted}>{a.error ? a.error.message : "Loading…"}</p>;
  const d = a.data;
  const i = d.inspection;
  const series = [...d.series].sort((x, y) => y.n_good - x.n_good);
  return (
    <>
      <div className={s.missionHead}>
        <b>{d.asset_name}</b>
        <Pill tone="muted">{d.asset_type}</Pill>
        <Sev level={d.criticality} />
        <span className={s.muted}>criticality</span>
        {d.triage ? <Pill tone="watch">triage #{d.triage.rank}, score {d.triage.score}</Pill> : <Pill tone="ok">not in the queue</Pill>}
      </div>
      {d.triage ? (
        <>
          <p className={s.reason}>{d.triage.reason}</p>
          <p className={s.stepText}>{d.triage.next_step.text}</p>
          <table className={s.tbl}>
            <tbody>
              {d.triage.parts.map((p) => (
                <tr key={p.label}>
                  <td>{p.label}</td>
                  <td>
                    <b>
                      {p.points > 0 ? "+" : ""}
                      {p.points}
                    </b>
                  </td>
                  <td className={s.muted}>{p.detail}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      ) : null}
      <OrderActions assetId={d.asset_id} anomalyId={d.triage?.anomaly_id ?? d.anomalies_list[d.anomalies_list.length - 1]?.anomaly_id ?? null} orders={orders} />
      <dl className={s.kv}>
        <dt>Home</dt>
        <dd>{d.location ? <><Id id={d.location.location_id} /> {d.location.location_name}, floor {d.location.floor}</> : d.location_id}</dd>
        <dt>Seen at</dt>
        <dd>
          <Ids ids={d.seen_at} />
        </dd>
        <dt>Last good inspection</dt>
        <dd>
          {i.last_good_action ? (
            <>
              <Id id={i.last_good_action} /> in <Id id={i.last_good_mission ?? ""} />, <b>{Math.round(i.hours_since_good ?? 0)} h</b> before the end of the data
            </>
          ) : (
            "none in the data"
          )}
        </dd>
        <dt>Failed since</dt>
        <dd>{i.failed_since.length ? <Ids ids={i.failed_since} /> : "nothing"}</dd>
        <dt>All actions</dt>
        <dd>
          {d.k} of {d.n} failed{d.enough ? ` (${pct(d.rate, 1)}, 95% interval ${range(d.lo, d.hi)})` : ": not enough data for a rate"}
        </dd>
      </dl>
      <div className={s.sub}>Anomalies ({d.anomalies_list.length})</div>
      {d.anomalies_list.length === 0 ? <p className={s.muted}>None on record.</p> : null}
      <table className={s.tbl}>
        <tbody>
          {d.anomalies_list.map((x) => (
            <tr key={x.anomaly_id}>
              <td>{x.anomaly_id}</td>
              <td>
                <Sev level={x.severity} /> {words(x.anomaly_type)}
              </td>
              <td>
                <Id id={x.mission_id} /> <Id id={x.action_id} />
                <div className={`${s.small} ${s.muted}`}>{stamp(x.timestamp)}</div>
              </td>
              <td className={s.muted}>{x.confirmed ? "confirmed: from a successful inspection" : "unverified: raised by a failed action"}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className={s.sub}>Readings over time</div>
      {series.length === 0 ? <p className={s.muted}>No readings for this asset.</p> : null}
      {series.map((sr) => (
        <TrendChart key={sr.measurement_type} sr={sr} />
      ))}
      <Samples items={d.samples} />
      <div className={s.sub}>Every action on this asset ({d.actions.length})</div>
      <ActionTable rows={d.actions} showMission />
    </>
  );
}

function FindingPanel({ f, overall }: { f: Finding; overall: number }) {
  const rateRows = f.rows.length > 0 && !["anomalies_hidden", "anomaly_repeats", "failure_types", "durations", "hidden_failures", "anomaly_matrix"].includes(f.id);
  return (
    <>
      <p className={`${s.small} ${s.muted}`}>{f.question}</p>
      <p className={s.sentence}>{f.sentence}</p>
      {f.caveat ? <p className={s.caveat}>{f.caveat}</p> : null}
      <dl className={s.kv}>
        <dt>n</dt>
        <dd>{f.n}</dd>
        {f.interval ? (
          <>
            <dt>95% interval</dt>
            <dd>{f.unit.includes("points") ? `${(100 * f.interval[0]).toFixed(1)} to ${(100 * f.interval[1]).toFixed(1)} points` : range(f.interval[0], f.interval[1])}</dd>
          </>
        ) : null}
        {f.p_value !== null ? (
          <>
            <dt>Test</dt>
            <dd>{pValue(f.p_value)}</dd>
          </>
        ) : null}
      </dl>
      {rateRows ? <RateBars rows={f.rows} overall={overall} /> : null}
      <div className={s.sub}>Rows</div>
      <table className={s.tbl}>
        <tbody>
          {f.rows.map((b) => (
            <tr key={b.key}>
              <td>{b.label}</td>
              <td>
                {b.k}/{b.n}
              </td>
              <td>
                {b.note ? <div className={s.muted}>{b.note}</div> : null}
                <Ids ids={[...b.evidence.missions, ...b.evidence.anomalies, ...b.evidence.actions].filter((x) => x !== b.key)} max={10} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <EvidenceLine evidence={f.evidence} max={60} />
    </>
  );
}

function MetricPanel({ m, f }: { m: Metric; f: Finding }) {
  const a = f.extra.a as { label: string };
  const b = f.extra.b as { label: string };
  const show = (side: Metric["a"]) => (m.kind === "rate" ? `${pct(side.value, 1)} (${side.k}/${side.n}), 95% interval ${range(side.lo, side.hi)}` : `${side.value?.toFixed(1)} ${m.unit ?? ""} mean, n=${side.n}, ${side.lo} to ${side.hi}`);
  return (
    <>
      <Pill tone={m.verdict === "changed" ? "ok" : m.verdict === "not enough data" ? "watch" : "muted"}>{m.verdict}</Pill>
      <dl className={s.kv}>
        <dt>{a.label}</dt>
        <dd>{show(m.a)}</dd>
        <dt>{b.label}</dt>
        <dd>{show(m.b)}</dd>
        <dt>Difference</dt>
        <dd>
          {m.diff === null ? "n/a" : m.kind === "rate" ? `${(100 * m.diff).toFixed(1)} points` : `${m.diff.toFixed(1)} ${m.unit ?? ""}`}
          {m.diff_lo !== null && m.diff_hi !== null ? `, 95% interval ${(100 * m.diff_lo).toFixed(1)} to ${(100 * m.diff_hi).toFixed(1)}` : ""} · {pValue(m.p_value)} ({m.test})
        </dd>
      </dl>
      <div className={s.sub}>{m.kind === "rate" ? "The rows that count against each side" : "Rows"}</div>
      <Ids ids={m.evidence.a} max={80} label={`${a.label}:`} />
      <Ids ids={m.evidence.b} max={80} label={`${b.label}:`} />
      {f.caveat ? <p className={s.caveat}>{f.caveat}</p> : null}
    </>
  );
}

function AssumptionsPanel({ plan }: { plan?: string }) {
  const a = useFetch((sig) => api.spotAssumptions(sig));
  const items: Assumption[] = a.data?.assumptions ?? [];
  return (
    <>
      <p className={s.sentence}>Every assumption the analysis makes. Each number on this screen carries its n; under 8 rows we say "not enough data".</p>
      <ol className={s.assume}>
        {items.map((x) => (
          <li key={x.key}>
            {x.assumption}
            <small>{x.why}</small>
          </li>
        ))}
        {plan ? (
          <li>
            The plan image and the CSV coordinates are two different layouts.<small>{plan}</small>
          </li>
        ) : null}
      </ol>
      <ValidationPanel />
    </>
  );
}

function ValidationPanel() {
  const v = useFetch((sig) => api.spotValidation(sig));
  const d: Validation | undefined = v.data;
  if (!d) return <p className={s.muted}>Loading…</p>;
  return (
    <>
      <div className={s.sub}>Our numbers against the kit's validation_results.json</div>
      <p className={s.sentence}>{d.summary}</p>
      <table className={s.tbl}>
        <thead>
          <tr>
            <th>Number</th>
            <th>Ours</th>
            <th>Kit</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {d.rows.map((r) => (
            <tr key={r.group + r.name} title={r.how}>
              <td>{words(r.name)}</td>
              <td>{r.ours ?? "n/a"}</td>
              <td>{r.theirs ?? "n/a"}</td>
              <td>
                <Pill tone={r.status === "agree" ? "ok" : r.status === "differ" ? "bad" : "muted"}>{r.status}</Pill>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className={s.sub}>What is odd about the files ({d.issues.length})</div>
      <ol className={s.assume}>
        {d.issues.map((i) => (
          <li key={i.code} style={{ borderLeftColor: i.level === "info" ? undefined : "var(--watch)" }}>
            {i.message}
          </li>
        ))}
      </ol>
    </>
  );
}
