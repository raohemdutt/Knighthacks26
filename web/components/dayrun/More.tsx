"use client";

import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import { ApiError, api, type CampaignResult, type ForecastResult, type Move, type ReplayResult, type Review, type SimResult } from "@/lib/api";
import { dateLabel, minutes, stars } from "@/lib/dayrun";
import { amount, percent } from "@/lib/format";
import ui from "../ui.module.css";
import dr from "./dayrun.module.css";

/**
 * Review, replay, campaign and forecast: one plain form each, showing what the
 * box answers. Every form sends the scenario the screen is on ("demo" or none).
 */

type Call<T> = { busy: boolean; data?: T; error?: ApiError };

function useCall<T>(): [Call<T>, (run: () => Promise<T>) => void] {
  const [state, setState] = useState<Call<T>>({ busy: false });
  const go = (run: () => Promise<T>) => {
    setState({ busy: true });
    run()
      .then((data) => setState({ busy: false, data }))
      .catch((err: unknown) => setState({ busy: false, error: err instanceof ApiError ? err : new ApiError(0, err instanceof Error ? err.message : String(err)) }));
  };
  return [state, go];
}

/** A box answer that is "no" rather than "broken" (nothing committed for that day, a plan that does not parse) is shown as its message. */
function Problem({ error }: { error: ApiError }) {
  return (
    <p className={`${ui.state} ${error.status === 404 ? "" : ui.stateError} ${ui.breakWords}`} role="alert">
      {error.needsToken ? "This box asks for a token. Add it under Settings, top right." : `${error.message}${error.status ? ` (HTTP ${error.status})` : ""}`}
    </p>
  );
}

function Panel({ id, title, hint, open, children }: { id: string; title: string; hint: string; open: boolean; children: ReactNode }) {
  return (
    <details id={id} className={`${ui.card} ${dr.panel}`} open={open || undefined}>
      <summary>
        <span className={dr.panelTitle}>{title}</span>
        <span className={`${ui.muted} ${ui.small}`}>{hint}</span>
      </summary>
      {children}
    </details>
  );
}

function verdict(r: SimResult): string {
  return `${r.outcome.won ? "Win" : "Fail"} ${stars(r.stars)} · ${r.totals.points} points`;
}

export function More({
  scenario,
  day,
  moves,
  result,
  campaignPreset,
  openAll,
}: {
  scenario: string | undefined;
  day: string; // the board's first day, YYYY-MM-DD
  moves: Move[];
  result: SimResult | undefined;
  campaignPreset: Move[][] | null;
  openAll: boolean;
}) {
  return (
    <div className={ui.list}>
      <ReviewForm scenario={scenario} day={day} open={openAll} />
      <ReplayForm scenario={scenario} day={day} moves={moves} open={openAll} />
      <CampaignForm scenario={scenario} moves={moves} preset={campaignPreset} open={openAll} />
      <ForecastForm scenario={scenario} day={day} result={result} open={openAll} />
    </div>
  );
}

function ReviewForm({ scenario, day, open }: { scenario: string | undefined; day: string; open: boolean }) {
  const [date, setDate] = useState(day);
  useEffect(() => setDate(day), [day]);
  const [call, go] = useCall<Review>();
  const submit = (e: FormEvent) => {
    e.preventDefault();
    go(() => api.dayrunReview(date, scenario));
  };
  const r = call.data;
  return (
    <Panel id="review" title="Review" hint="A committed plan against what Habits measured that day." open={open}>
      <form className={dr.form} onSubmit={submit}>
        <label className={dr.field}>
          <span>Day</span>
          <input className={ui.input} type="date" required value={date} onChange={(e) => setDate(e.target.value)} />
        </label>
        <button type="submit" className={ui.btn} disabled={call.busy}>
          {call.busy ? "Reviewing…" : "Review that day"}
        </button>
      </form>
      {call.error ? <Problem error={call.error} /> : null}
      {r ? (
        <div className={ui.section}>
          <p className={dr.big}>{r.score === null ? "No score" : `${r.score}/100`}</p>
          <p className={ui.breakWords}>{r.text}</p>
          {r.biggest_gap ? <p className={ui.nudge}>Biggest gap: {r.biggest_gap.text}</p> : null}
          <table className={`${ui.table} ${dr.spaced}`}>
            <thead>
              <tr>
                <th>Block</th>
                <th className={ui.num}>Planned</th>
                <th className={ui.num}>Measured</th>
                <th className={ui.num}>Followed</th>
              </tr>
            </thead>
            <tbody>
              {r.blocks.map((b) => (
                <tr key={b.id}>
                  <td>
                    {b.title}
                    {b.note ? <span className={`${ui.muted} ${ui.small}`}> · {b.note}</span> : null}
                  </td>
                  <td className={ui.num}>{minutes(b.planned_minutes)}</td>
                  <td className={ui.num}>{b.measured_minutes === null ? "–" : minutes(b.measured_minutes)}</td>
                  <td className={ui.num}>{b.followed === null ? "–" : percent(b.followed)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <dl className={ui.kv}>
            <dt>Measured</dt>
            <dd>{measuredText(r.measured)}</dd>
            <dt>Measured by</dt>
            <dd>{r.measured_by}</dd>
          </dl>
        </div>
      ) : null}
    </Panel>
  );
}

function measuredText(m: { focus_minutes: number | null; meals: number | null; sleep_hours: number | null }): string {
  const parts = [
    m.focus_minutes === null ? null : `${minutes(m.focus_minutes)} at the desk`,
    m.meals === null ? null : `${m.meals} ${m.meals === 1 ? "meal" : "meals"}`,
    m.sleep_hours === null ? null : `${amount(m.sleep_hours)} h of sleep`,
  ].filter(Boolean);
  return parts.length ? parts.join(", ") : "nothing";
}

function ReplayForm({ scenario, day, moves, open }: { scenario: string | undefined; day: string; moves: Move[]; open: boolean }) {
  const [date, setDate] = useState(day);
  useEffect(() => setDate(day), [day]);
  const [call, go] = useCall<ReplayResult>();
  const submit = (e: FormEvent) => {
    e.preventDefault();
    go(() => api.dayrunReplay({ day: date, moves, scenario }));
  };
  const r = call.data;
  return (
    <Panel id="replay" title="Replay" hint="Play a past day again: what happened against the plan above." open={open}>
      <form className={dr.form} onSubmit={submit}>
        <label className={dr.field}>
          <span>Day</span>
          <input className={ui.input} type="date" required value={date} onChange={(e) => setDate(e.target.value)} />
        </label>
        <button type="submit" className={ui.btn} disabled={call.busy}>
          {call.busy ? "Replaying…" : `Replay with the ${moves.length} ${moves.length === 1 ? "move" : "moves"} above`}
        </button>
      </form>
      <p className={`${ui.muted} ${ui.small}`}>The replay starts at wake-up time, so its slots can differ from the board above; moves for slots it does not have are rejected.</p>
      {call.error ? <Problem error={call.error} /> : null}
      {r ? (
        <div className={ui.section}>
          <p className={dr.big}>
            {r.points_gained > 0 ? "+" : ""}
            {r.points_gained} points, {r.stars_gained > 0 ? "+" : ""}
            {r.stars_gained} stars
          </p>
          <p className={ui.breakWords}>{r.text}</p>
          <dl className={ui.kv}>
            <dt>Measured</dt>
            <dd>{measuredText(r.actual)}</dd>
            <dt>What happened</dt>
            <dd>
              {verdict(r.baseline)}. {r.baseline.outcome.summary}
            </dd>
            <dt>Your plan</dt>
            <dd>
              {verdict(r.plan)}. {r.plan.outcome.summary}
              {r.plan.rejected.length > 0 ? ` ${r.plan.rejected.length} rejected: ${r.plan.rejected.map((x) => x.reason).join("; ")}` : ""}
            </dd>
          </dl>
        </div>
      ) : null}
    </Panel>
  );
}

function CampaignForm({ scenario, moves, preset, open }: { scenario: string | undefined; moves: Move[]; preset: Move[][] | null; open: boolean }) {
  const [text, setText] = useState("");
  const [bad, setBad] = useState<string | null>(null);
  const [call, go] = useCall<CampaignResult>();
  // Start from the scripted two days when the scenario has them, else from the plan above as day one.
  useEffect(() => setText(JSON.stringify(preset ?? [moves], null, 1)), [preset]); // eslint-disable-line react-hooks/exhaustive-deps
  const submit = (e: FormEvent) => {
    e.preventDefault();
    let days: unknown;
    try {
      days = JSON.parse(text);
    } catch (err) {
      setBad(`That is not JSON: ${err instanceof Error ? err.message : String(err)}`);
      return;
    }
    if (!Array.isArray(days) || !days.every((d) => Array.isArray(d))) {
      setBad("Give a list of days, each a list of moves: [[{…}, {…}], [{…}]].");
      return;
    }
    setBad(null);
    go(() => api.dayrunCampaign({ days: days as Move[][], scenario }));
  };
  const r = call.data;
  return (
    <Panel id="campaign" title="Campaign" hint="Consecutive days: each one starts with what the last left you." open={open}>
      <form className={dr.form} onSubmit={submit}>
        <label className={dr.field}>
          <span>Days, as a list of move lists (JSON)</span>
          <textarea className={`${ui.input} ${dr.textarea}`} spellCheck={false} value={text} onChange={(e) => setText(e.target.value)} />
        </label>
        <div className={`${ui.row} ${ui.wrap}`}>
          <button type="submit" className={ui.btn} disabled={call.busy}>
            {call.busy ? "Running…" : "Run the campaign"}
          </button>
          <button type="button" className={dr.pill} onClick={() => setText(JSON.stringify([moves], null, 1))}>
            Use the plan above as day 1
          </button>
          {preset ? (
            <button type="button" className={dr.pill} onClick={() => setText(JSON.stringify(preset, null, 1))}>
              Scripted two days
            </button>
          ) : null}
        </div>
      </form>
      <p className={`${ui.muted} ${ui.small}`}>Each day has its own slots (s1, s2, …) from that day&apos;s calendar, so a later day&apos;s moves must name that day&apos;s slots.</p>
      {bad ? (
        <p className={`${ui.state} ${ui.stateError}`} role="alert">
          {bad}
        </p>
      ) : null}
      {call.error ? <Problem error={call.error} /> : null}
      {r ? (
        <div className={ui.section}>
          <p className={dr.big}>
            {r.won ? "Campaign won" : "Campaign lost"} · {r.stars} {r.stars === 1 ? "star" : "stars"} · {r.points} points
          </p>
          <ul className={ui.list}>
            {r.days.map((d) => (
              <li key={d.day} className={dr.dayRow}>
                <p>
                  <strong>{dateLabel(d.day)}</strong> · {verdict(d.result)}
                </p>
                <p className={`${ui.muted} ${ui.small}`}>
                  Starts with energy {Math.round(d.carry_in.energy)}, focus {Math.round(d.carry_in.focus)}, {amount(d.carry_in.sleep_hours)} h of sleep the night before ({d.carry_in.source}).
                </p>
                <p className={ui.breakWords}>{d.result.outcome.summary}</p>
                {d.result.rejected.map((x) => (
                  <p key={x.index} className={dr.rejected}>
                    <strong>Rejected move {x.index + 1}:</strong> {x.reason}
                  </p>
                ))}
              </li>
            ))}
          </ul>
        </div>
      ) : null}
    </Panel>
  );
}

function ForecastForm({ scenario, day, result, open }: { scenario: string | undefined; day: string; result: SimResult | undefined; open: boolean }) {
  const blocks = (result?.blocks ?? []).filter((b) => !b.fixed);
  const [estimates, setEstimates] = useState<Record<string, string>>({});
  const [measured, setMeasured] = useState<Record<string, string>>({});
  const [call, go] = useCall<ForecastResult>();
  const num = (s: string | undefined) => (s !== undefined && s.trim() !== "" && Number.isFinite(Number(s)) ? Number(s) : null);
  const measuredRows = Object.fromEntries(blocks.map((b) => [b.id, num(measured[b.id])] as const).filter((x): x is [string, number] => x[1] !== null));
  const useReview = Object.keys(measuredRows).length === 0;

  const submit = (e: FormEvent) => {
    e.preventDefault();
    const rows = blocks.map((b) => ({ block: b.id, minutes: num(estimates[b.id]) ?? b.minutes }));
    go(() => api.dayrunForecast(useReview ? { estimates: rows, day, scenario } : { estimates: rows, measured: measuredRows }));
  };
  const r = call.data;
  const title = (id: string) => blocks.find((b) => b.id === id)?.title ?? id;

  return (
    <Panel id="forecast" title="Forecast" hint="How well you estimate your own time, block by block." open={open}>
      {blocks.length === 0 ? (
        <p className={ui.muted}>Plan at least one move above: the forecast scores your estimate for each planned block.</p>
      ) : (
        <form className={dr.form} onSubmit={submit}>
          <table className={`${ui.table} ${dr.spaced}`}>
            <thead>
              <tr>
                <th>Block</th>
                <th className={ui.num}>Estimate, min</th>
                <th className={ui.num}>Took, min</th>
              </tr>
            </thead>
            <tbody>
              {blocks.map((b) => (
                <tr key={b.id}>
                  <td>
                    {b.title} <span className={`${ui.muted} ${ui.small}`}>({b.id})</span>
                  </td>
                  <td className={ui.num}>
                    <input
                      className={`${ui.input} ${dr.numInput}`}
                      type="number"
                      min={0}
                      inputMode="numeric"
                      aria-label={`Estimate for ${b.title}`}
                      value={estimates[b.id] ?? String(b.minutes)}
                      onChange={(e) => setEstimates((s) => ({ ...s, [b.id]: e.target.value }))}
                    />
                  </td>
                  <td className={ui.num}>
                    <input
                      className={`${ui.input} ${dr.numInput}`}
                      type="number"
                      min={0}
                      inputMode="numeric"
                      placeholder="–"
                      aria-label={`Measured minutes for ${b.title}`}
                      value={measured[b.id] ?? ""}
                      onChange={(e) => setMeasured((s) => ({ ...s, [b.id]: e.target.value }))}
                    />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <button type="submit" className={ui.btn} disabled={call.busy}>
            {call.busy ? "Scoring…" : useReview ? `Score against the review of ${dateLabel(day)}` : "Score my estimates"}
          </button>
          <p className={`${ui.muted} ${ui.small}`}>
            Leave &quot;Took&quot; empty to use what the box measured for the plan committed on {dateLabel(day)}. Blocks it could not measure are listed, not guessed.
          </p>
        </form>
      )}
      {call.error ? <Problem error={call.error} /> : null}
      {r ? (
        <div className={ui.section}>
          <p className={dr.big}>Calibration {r.score}/100</p>
          <p className={ui.breakWords}>{r.text}</p>
          <dl className={ui.kv}>
            <dt>Bias</dt>
            <dd>{r.bias === 0 ? "none" : `${percent(Math.abs(r.bias))} ${r.bias < 0 ? "under" : "over"}`}</dd>
            <dt>Multiply by</dt>
            <dd>{r.multiplier}</dd>
          </dl>
          {r.rows.length > 0 ? (
            <table className={`${ui.table} ${dr.spaced}`}>
              <thead>
                <tr>
                  <th>Block</th>
                  <th className={ui.num}>Estimated</th>
                  <th className={ui.num}>Measured</th>
                  <th className={ui.num}>Off by</th>
                </tr>
              </thead>
              <tbody>
                {r.rows.map((row) => (
                  <tr key={row.block}>
                    <td>{title(row.block)}</td>
                    <td className={ui.num}>{amount(row.estimated)}</td>
                    <td className={ui.num}>{amount(row.measured)}</td>
                    <td className={ui.num}>
                      {row.off_by > 0 ? "+" : ""}
                      {amount(row.off_by)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
          {r.unmeasured.length > 0 ? <p className={`${ui.muted} ${ui.small}`}>Not measured, so not scored: {r.unmeasured.map(title).join(", ")}.</p> : null}
        </div>
      ) : null}
    </Panel>
  );
}
