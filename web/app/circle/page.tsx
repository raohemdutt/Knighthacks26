"use client";

import { useState, type FormEvent } from "react";
import { ProposeTime } from "@/components/circle/ProposeTime";
import { SharedCalendar } from "@/components/circle/SharedCalendar";
import { YouAreSharing } from "@/components/circle/YouAreSharing";
import { Empty, ErrorBox, Loading, WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type Leaderboard, type PlaceShared } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { amount, humanize, percent, relative, time } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";

/**
 * The household's named places (config/agent.yaml `places`). The gateway has no
 * endpoint that lists them, so these are suggestions; the box still has the
 * last word and answers 422 with its own list when a place is not on it.
 */
const SUGGESTED_PLACES = ["Our table", "Library", "Dorm", "Student Union", "Gym"];
const HOURS = [1, 2, 4];

export default function CirclePage() {
  const presence = useFetch((s) => api.presence(s));
  const board = useFetch((s) => api.leaderboard(s));

  useAtlasEvents(["circle.place.shared"], () => presence.reload());
  useAtlasEvents(["circle.share.synced"], () => board.reload());

  const circleOff = presence.error?.notConfigured ?? false;

  return (
    <>
      <div className={ui.pageHead}>
        <h1>Circle</h1>
        <p className={ui.muted}>Friends who also run Atlas. Only named places, counts and scores are shared; never video.</p>
      </div>

      {circleOff ? (
        <Empty title="Not connected to the Circle">
          This box has not joined a Circle, so there are no friends&apos; places to show and nothing is being shared. Once a member and circle
          are set on the box, friends&apos; places and the leaderboards appear here.
        </Empty>
      ) : (
        <>
          <section className={ui.section}>
            <h2 className={ui.sectionTitle}>Where friends are</h2>
            {presence.error ? (
              <ErrorBox what="friends' places" error={presence.error} onRetry={presence.reload} />
            ) : !presence.data ? (
              <Loading what="friends' places" />
            ) : presence.data.length === 0 ? (
              <Empty title="No one is sharing a place right now">
                When a friend shares a spot like the Library or your table, their name and the place show here until the share expires.
              </Empty>
            ) : (
              <ul className={ui.list}>
                {presence.data.map((p) => (
                  <FriendPlace key={p.member_id} place={p} />
                ))}
              </ul>
            )}
          </section>

          <ShareMyPlace onShared={presence.reload} />
          <ProposeTime />
        </>
      )}

      <SharedCalendar />

      <section className={ui.section}>
        <h2 className={ui.sectionTitle}>Leaderboards</h2>
        {board.error ? (
          board.error.notConfigured ? (
            <Empty title="Leaderboards are offline">
              {circleOff
                ? "They need a Circle too. Recycling rates, habit streaks and Sort It scores for you and your friends appear here once this box joins one."
                : "This box is not connected to the Circle's shared database, so there are no recycling rates, habit streaks or Sort It scores to compare yet."}
            </Empty>
          ) : (
            <ErrorBox what="the leaderboards" error={boardError(board.error)} onRetry={board.reload} />
          )
        ) : !board.data ? (
          <Loading what="leaderboards" />
        ) : (
          <Boards board={board.data} />
        )}
      </section>

      <YouAreSharing />
    </>
  );
}

/**
 * The box answers 500 when it cannot reach the Circle's shared database (no network, database asleep).
 * Say that instead of "Internal Server Error"; what was shared meanwhile is queued on the box.
 */
function boardError(error: ApiError): ApiError {
  if (error.status < 500) return error;
  return new ApiError(
    error.status,
    "This box can't reach the Circle's shared database right now. What it shares in the meantime waits on the box and goes out when it is back online.",
  );
}

/** A member without a display name comes back as their id; show it the way "Where friends are" does. */
function who(r: { member_id: string; display_name: string }): string {
  return r.display_name === r.member_id ? humanize(r.member_id) : r.display_name;
}

function FriendPlace({ place: p }: { place: PlaceShared }) {
  return (
    <li className={`${ui.card}`}>
      <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
        <h3>{humanize(p.member_id)}</h3>
        <span className={ui.tag}>{p.source === "desk" ? "seen at the desk" : "shared from their phone"}</span>
      </div>
      <p className={ui.reason}>{p.place}</p>
      <p className={`${ui.muted} ${ui.small}`}>
        Shared at {time(p.ts)} · until {time(p.expires_at)} ({relative(p.expires_at)})
      </p>
    </li>
  );
}

function ShareMyPlace({ onShared }: { onShared: () => void }) {
  const [place, setPlace] = useState(SUGGESTED_PLACES[0] ?? "");
  const [hours, setHours] = useState(2);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [shared, setShared] = useState<PlaceShared | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    if (!place.trim() || busy) return;
    setBusy(true);
    setError(null);
    setShared(null);
    try {
      setShared(await api.sharePlace(place.trim(), hours));
      onShared();
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, String(err)));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className={ui.section}>
      <h2 className={ui.sectionTitle}>Share where you are</h2>
      <form className={ui.card} onSubmit={submit}>
        <p className={`${ui.muted} ${ui.small}`}>Pick a named place. Friends see the name only, and it disappears when the time is up.</p>
        <div className={ui.chips} role="group" aria-label="Place">
          {SUGGESTED_PLACES.map((p) => (
            <button key={p} type="button" className={`${ui.chip} ${place === p ? ui.chipOn : ""}`} aria-pressed={place === p} onClick={() => setPlace(p)}>
              {p}
            </button>
          ))}
        </div>
        <input className={ui.input} value={place} onChange={(e) => setPlace(e.target.value)} placeholder="Place name" aria-label="Place name" />
        <div className={ui.chips} role="group" aria-label="For how long">
          {HOURS.map((h) => (
            <button key={h} type="button" className={`${ui.chip} ${hours === h ? ui.chipOn : ""}`} aria-pressed={hours === h} onClick={() => setHours(h)}>
              {h} {h === 1 ? "hour" : "hours"}
            </button>
          ))}
        </div>
        {error ? <WriteError error={error} /> : null}
        {shared ? (
          <p className={ui.nudge} role="status">
            Sharing <strong>{shared.place}</strong> until {time(shared.expires_at)}.
          </p>
        ) : null}
        <button type="submit" className={`${ui.btn} ${ui.btnPrimary}`} disabled={busy || !place.trim()}>
          {busy ? "Sharing…" : "Share this place"}
        </button>
      </form>
    </section>
  );
}

function Boards({ board }: { board: Leaderboard }) {
  const nothing = board.impact.length === 0 && board.habits.length === 0 && board.scores.length === 0;
  if (nothing)
    return (
      <Empty title="No scores shared yet">
        Circle {board.circle_id} is connected, but nobody has shared recycling counts, habit streaks or Sort It rounds so far.
      </Empty>
    );
  return (
    <>
      <div className={ui.card}>
        <h3>Recycling</h3>
        {board.impact.length === 0 ? (
          <p className={ui.muted}>No one has shared Waste Watch counts yet.</p>
        ) : (
          <table className={ui.table}>
            <thead>
              <tr>
                <th scope="col">Who</th>
                <th scope="col" className={ui.num}>Recycled</th>
                <th scope="col" className={ui.num}>Rate</th>
              </tr>
            </thead>
            <tbody>
              {board.impact.map((r) => (
                <tr key={r.member_id}>
                  <td>{who(r)}</td>
                  <td className={ui.num}>
                    {r.recycled} of {r.items_seen}
                  </td>
                  <td className={ui.num}>{percent(Number(r.diversion_rate))}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className={ui.card}>
        <h3>Habit streaks</h3>
        {board.habits.length === 0 ? (
          <p className={ui.muted}>No one has shared a habit streak yet.</p>
        ) : (
          <table className={ui.table}>
            <thead>
              <tr>
                <th scope="col">Who</th>
                <th scope="col">Habit</th>
                <th scope="col" className={ui.num}>Days</th>
              </tr>
            </thead>
            <tbody>
              {board.habits.map((r) => (
                <tr key={`${r.member_id}-${r.habit_id}`}>
                  <td>{who(r)}</td>
                  <td>{humanize(r.habit_id)}</td>
                  <td className={ui.num}>{r.streak}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className={ui.card}>
        <h3>Sort It, best rounds</h3>
        {board.scores.length === 0 ? (
          <p className={ui.muted}>No Sort It rounds have been shared yet.</p>
        ) : (
          <table className={ui.table}>
            <thead>
              <tr>
                <th scope="col">Who</th>
                <th scope="col">Item</th>
                <th scope="col" className={ui.num}>Time</th>
                <th scope="col" className={ui.num}>Score</th>
              </tr>
            </thead>
            <tbody>
              {board.scores.map((r) => (
                <tr key={r.round_id}>
                  <td>{who(r)}</td>
                  <td>
                    {r.item}
                    {r.correct ? "" : " (wrong bin)"}
                  </td>
                  <td className={ui.num}>{amount(Number(r.seconds))} s</td>
                  <td className={ui.num}>{amount(Number(r.score))}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  );
}
