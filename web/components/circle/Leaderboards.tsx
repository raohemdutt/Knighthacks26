"use client";

import { Empty, ErrorBox, Loading } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type Leaderboard } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { amount, humanize, percent } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";

/**
 * The Circle's leaderboards from its shared database: recycling rates, habit streaks, Sort It rounds.
 * GET /api/circle/leaderboard is 404 without that database; the League above works without it.
 */
export function Leaderboards({ circleOff }: { circleOff: boolean }) {
  const board = useFetch((s) => api.leaderboard(s));
  useAtlasEvents(["circle.share.synced"], () => board.reload());

  return (
    <section className={ui.section}>
      <h2 className={ui.sectionTitle}>Leaderboards</h2>
      {board.error ? (
        board.error.notConfigured ? (
          <Empty title="Leaderboards are offline">
            {circleOff
              ? "They need a Circle too. Recycling rates, habit streaks and Sort It scores for you and your friends appear here once this box joins one."
              : "This box is not connected to the Circle's shared database, so there are no recycling rates, habit streaks or Sort It scores to compare yet. The League above runs from this box's own log."}
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
