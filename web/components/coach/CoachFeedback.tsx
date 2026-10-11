"use client";

import { useState } from "react";
import { Keyframe } from "@/components/Keyframe";
import { WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import type { CoachAnswer, CoachCorrectionResult, CoachFeedbackView, FeedbackBreak, FeedbackSession } from "@/lib/coachFeedback";
import { time } from "@/lib/format";
import { useFetch } from "@/lib/useFetch";
import cf from "./coach.module.css";

/**
 * "Was that right?" under each break the study coach found.
 *
 * Standalone, so any screen can drop it in:
 *
 *     import { CoachFeedback } from "@/components/coach/CoachFeedback";
 *     <CoachFeedback day={note.day} blockId={note.block_id} />
 *
 * It loads GET /api/day/coach/feedback itself, shows that block's breaks with the quick answers, and posts
 * each answer to POST /api/day/coach/corrections. An answer is stored on the box as `day.coach.corrected`;
 * nothing leaves the box. After enough answers about one kind of break the box raises an approval card to
 * learn to see it, and this says so.
 */
export function CoachFeedback({ day, blockId }: { day: string; blockId: string }) {
  const view = useFetch((s) => api.coachFeedback(7, s), `${day}/${blockId}`);
  useAtlasEvents(["day.coach.", "learning."], () => view.reload());
  const session = view.data?.sessions.find((s) => s.day === day && s.block_id === blockId);
  if (!view.data || !session || session.breaks.length === 0) return null; // nothing to ask about
  return <SessionBreaks session={session} options={view.data.options} onAnswered={view.reload} />;
}

/** The breaks of one session, each with its question. `onAnswered` runs after an answer is stored. */
export function SessionBreaks({
  session,
  options,
  onAnswered,
}: {
  session: FeedbackSession;
  options: CoachFeedbackView["options"];
  onAnswered?: (result: CoachCorrectionResult) => void;
}) {
  return (
    <ul className={ui.list}>
      {session.breaks.map((b) => (
        <BreakQuestion key={b.start} day={session.day} blockId={session.block_id} b={b} options={options} onAnswered={onAnswered} />
      ))}
    </ul>
  );
}

function BreakQuestion({
  day,
  blockId,
  b,
  options,
  onAnswered,
}: {
  day: string;
  blockId: string;
  b: FeedbackBreak;
  options: CoachFeedbackView["options"];
  onAnswered?: (result: CoachCorrectionResult) => void;
}) {
  const [choosing, setChoosing] = useState(false);
  const [other, setOther] = useState(false);
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | undefined>(undefined);
  const [proposed, setProposed] = useState(false);
  const noGuess = b.guess === "none";

  const send = async (right: boolean, answer?: CoachAnswer, text?: string) => {
    setBusy(true);
    setError(undefined);
    try {
      const result = await api.coachCorrect({ day, block_id: blockId, break_index: b.index, break_start: b.start, right, answer, note: text || undefined });
      setChoosing(false);
      setOther(false);
      setNote("");
      setProposed(Boolean(result.proposed));
      onAnswered?.(result);
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, String(err)));
    } finally {
      setBusy(false);
    }
  };

  const c = b.correction;
  return (
    <li className={cf.break}>
      <div className={cf.breakHead}>
        {b.frame_ref ? (
          <div className={cf.thumb}>
            <Keyframe frameRef={b.frame_ref} compact />
          </div>
        ) : null}
        <div className={ui.grow}>
          <p>
            <strong>
              {time(b.start)}, away {b.minutes} min
            </strong>
          </p>
          <p className={ui.muted}>The coach&apos;s guess: {b.guess_text}.</p>
        </div>
      </div>

      {c && !choosing ? (
        <div className={`${ui.row} ${ui.between} ${ui.wrap}`}>
          <p className={cf.answered}>
            {c.right ? (noGuess ? "You said: right, no particular reason." : "You said: yes, that was it.") : `You said: ${c.answer_text}.`}
            {c.note ? ` "${c.note}"` : ""}
          </p>
          <button type="button" className={ui.btn} onClick={() => setChoosing(true)}>
            Change
          </button>
        </div>
      ) : (
        <>
          <p className={cf.question}>{noGuess ? "Was there a reason?" : "Was that right?"}</p>
          <div className={ui.chips} role="group" aria-label="Your answer">
            <button type="button" className={`${ui.chip} ${ui.chipOn}`} disabled={busy} onClick={() => send(true)}>
              {noGuess ? "No reason" : "Yes"}
            </button>
            {options
              .filter((o) => o.id !== b.guess)
              .map((o) =>
                o.id === "other" ? (
                  <button key={o.id} type="button" className={ui.chip} disabled={busy} aria-expanded={other} onClick={() => setOther((x) => !x)}>
                    {o.label}
                  </button>
                ) : (
                  <button key={o.id} type="button" className={ui.chip} disabled={busy} onClick={() => send(false, o.id)}>
                    {noGuess ? o.label : `No: ${o.label.startsWith("I ") ? o.label : o.label.charAt(0).toLowerCase() + o.label.slice(1)}`}
                  </button>
                ),
              )}
          </div>
          {other ? (
            <form
              className={cf.other}
              onSubmit={(e) => {
                e.preventDefault();
                void send(false, "other", note.trim());
              }}
            >
              <input
                className={ui.input}
                value={note}
                maxLength={280}
                placeholder="What was it? (optional)"
                aria-label="What was it?"
                onChange={(e) => setNote(e.target.value)}
              />
              <button type="submit" className={`${ui.btn} ${ui.btnPrimary}`} disabled={busy}>
                Save
              </button>
            </form>
          ) : null}
        </>
      )}
      {proposed ? (
        <p className={cf.learning}>
          That was enough answers for Atlas to ask to learn this. The card is on the Approvals screen; nothing leaves the box unless you approve it.
        </p>
      ) : null}
      {error ? <WriteError error={error} /> : null}
    </li>
  );
}
