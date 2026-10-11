"use client";

import { useState } from "react";
import { WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { useFetch } from "@/lib/useFetch";
import l from "./looks.module.css";

const asApiError = (err: unknown) => (err instanceof ApiError ? err : new ApiError(0, err instanceof Error ? err.message : String(err)));

/**
 * "Looks Atlas remembers" (the look memory, contexts/day/application/looks.py): the most recent five times a look held,
 * each with when it was, what was going on and what helped. Words and numbers only, read from this box. One can be
 * forgotten, or all of them in one tap; a forgotten look stays forgotten after a restart. Shown under the expressions
 * switch of the one person whose looks are read, and nothing is shown while there is nothing to show.
 *
 * Under it, "What Atlas has learned about your looks" (contexts/day/domain/look_model.py): two or three findings as
 * sentences with their counts, how wide the guess still is, the evidence line, and "not enough yet" when that is the
 * truth. It is a count worked out on the box from the looks above, stored nowhere: so "Forget what it learned" forgets
 * all of the looks, and says so before it does.
 */
export function LooksRemembered({ memberId }: { memberId: string }) {
  const looks = useFetch((s) => api.dayLooks(memberId, s), `looks:${memberId}`);
  const [busy, setBusy] = useState<string | null>(null);
  const [asking, setAsking] = useState(false);
  const [unlearning, setUnlearning] = useState(false);
  const [error, setError] = useState<ApiError | undefined>(undefined);
  useAtlasEvents(["day.checkin", "day.look", "perception.identity"], looks.reload);

  const data = looks.data;
  if (!data || !data.on) return null;
  const forget = (id: string | null) => {
    setBusy(id ?? "all");
    setError(undefined);
    (id ? api.dayForgetLook(id) : api.dayForgetLooks(memberId))
      .then(() => {
        setAsking(false);
        setUnlearning(false);
        looks.reload();
      })
      .catch((err: unknown) => setError(asApiError(err)))
      .finally(() => setBusy(null));
  };

  return (
    <div className={l.block} data-guide="people:looks">
      <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
        <div className={ui.grow}>
          <h3>Looks Atlas remembers</h3>
          <p className={`${ui.muted} ${ui.small}`}>
            When a look holds for a while, Atlas keeps what was going on and what helped, so it can say so next time. Words and numbers, kept on
            this box; never a picture. Switching expressions off clears them.
          </p>
        </div>
        {data.moments.length === 0 ? null : asking ? (
          <div className={ui.row}>
            <button type="button" className={ui.btn} disabled={busy !== null} onClick={() => setAsking(false)}>
              Keep
            </button>
            <button type="button" className={`${ui.btn} ${ui.btnDeny}`} disabled={busy !== null} onClick={() => forget(null)}>
              {busy === "all" ? "Forgetting…" : `Forget all ${data.count}`}
            </button>
          </div>
        ) : (
          <button type="button" className={ui.btn} disabled={busy !== null} onClick={() => setAsking(true)}>
            Forget all
          </button>
        )}
      </div>
      {data.moments.length === 0 ? (
        <p className={`${ui.muted} ${ui.small}`}>Nothing remembered yet. A look has to hold for about twenty minutes before Atlas keeps it.</p>
      ) : (
        <ul className={l.list}>
          {data.moments.map((m) => (
            <li key={m.id} className={l.item} data-look={m.day}>
              <div className={l.text}>
                <p className={l.when}>
                  <strong>{m.when}</strong>
                  <span className={ui.muted}>
                    around {m.time} · looked {m.look}
                  </span>
                </p>
                <p className={ui.small}>{m.going_on}</p>
                <p className={`${ui.muted} ${ui.small}`}>{m.helped}</p>
              </div>
              <button type="button" className={ui.btn} disabled={busy !== null} onClick={() => forget(m.id)} aria-label={`Forget the look from ${m.when}`}>
                {busy === m.id ? "Forgetting…" : "Forget"}
              </button>
            </li>
          ))}
        </ul>
      )}
      {data.count > data.moments.length ? (
        <p className={`${ui.muted} ${ui.small}`}>
          The most recent {data.moments.length} of {data.count}. A look is kept for {data.kept_days} days.
        </p>
      ) : null}
      {data.learned && data.moments.length > 0 ? (
        <div className={l.learned} data-guide="people:looks-learned">
          <h3>What Atlas has learned about your looks</h3>
          {data.learned.enough ? (
            <ul className={l.list}>
              {data.learned.findings.map((f) => (
                <li key={f.kind} className={l.finding} data-finding={f.kind}>
                  <p>{f.text}</p>
                  <p className={`${ui.muted} ${ui.small}`}>{f.band}</p>
                </li>
              ))}
            </ul>
          ) : (
            <p>{data.learned.not_enough}</p>
          )}
          <p className={ui.small} data-evidence>
            {data.learned.evidence}
          </p>
          <p className={`${ui.muted} ${ui.small}`}>
            {data.learned.shared_note ? `${data.learned.shared_note} ` : ""}
            {data.learned.note}
          </p>
          {unlearning ? (
            <div className={l.confirm}>
              <p className={ui.small}>
                What Atlas learned is only a count of the looks above, kept nowhere else. Forgetting it forgets all {data.count} of them.
              </p>
              <div className={ui.row}>
                <button type="button" className={ui.btn} disabled={busy !== null} onClick={() => setUnlearning(false)}>
                  Keep
                </button>
                <button type="button" className={`${ui.btn} ${ui.btnDeny}`} disabled={busy !== null} onClick={() => forget(null)}>
                  {busy === "all" ? "Forgetting…" : `Forget all ${data.count} looks`}
                </button>
              </div>
            </div>
          ) : (
            <div>
              <button type="button" className={ui.btn} disabled={busy !== null} onClick={() => setUnlearning(true)}>
                Forget what it learned
              </button>
            </div>
          )}
        </div>
      ) : (
        <div className={l.learned} data-guide="people:looks-learned">
          <h3>What Atlas has learned about your looks</h3>
          <p className={`${ui.muted} ${ui.small}`}>Nothing yet. Atlas learns only from the looks it remembers, and it remembers none.</p>
        </div>
      )}
      {error ? <WriteError error={error} /> : null}
    </div>
  );
}
