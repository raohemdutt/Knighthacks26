"use client";

import { useState } from "react";
import { WriteError } from "@/components/States";
import ui from "@/components/ui.module.css";
import { ApiError, api, type Answer } from "@/lib/api";
import { dayTime, safeUrl } from "@/lib/format";
import c from "./circle.module.css";

/**
 * "Where should I go?" (G5 on the Circle screen, X12). Asks this box the question through the same path as the Ask
 * screen (POST /api/ask). The box's planner answers from what is on tonight and where friends say they are (only the
 * place names they chose to share, until they expire), and cites each one. Nothing new leaves the box: the question
 * runs on the box's own model.
 */

const QUESTIONS = ["Where should I go tonight?", "Where is everyone?"];

export function WhereToGo({ friendsSharing }: { friendsSharing: number }) {
  const [asked, setAsked] = useState<string | null>(null);
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);

  const ask = async (q: string) => {
    if (busy) return;
    setBusy(true);
    setAsked(q);
    setError(null);
    setAnswer(null);
    try {
      setAnswer(await api.ask(q));
    } catch (err) {
      setError(err instanceof ApiError ? err : new ApiError(0, String(err)));
    } finally {
      setBusy(false);
    }
  };

  const sources = (answer?.citations ?? []).filter((s) => s.title || s.url);
  return (
    <section className={ui.section} aria-labelledby="where-to-go">
      <h2 className={ui.sectionTitle} id="where-to-go">
        Where should I go?
      </h2>
      <div className={ui.card}>
        <p className={ui.muted}>
          Atlas weighs what is on tonight against where friends are
          {friendsSharing ? ` (${friendsSharing} sharing a place now)` : " (nobody is sharing a place right now)"}, and says why.
        </p>
        <div className={ui.chips} role="group" aria-label="Ask">
          {QUESTIONS.map((q) => (
            <button key={q} type="button" className={`${ui.chip} ${asked === q ? ui.chipOn : ""}`} aria-pressed={asked === q} disabled={busy} onClick={() => void ask(q)}>
              {busy && asked === q ? "Thinking…" : q}
            </button>
          ))}
        </div>
        {answer ? (
          <div className={c.brainAnswer} aria-live="polite">
            <p className={c.whereAnswer}>{answer.answer}</p>
            {sources.length ? (
              <ul className={c.whereSources}>
                {sources.map((s, i) => (
                  <li key={`${s.title ?? s.url}-${i}`}>
                    {s.url && safeUrl(s.url) ? (
                      <a href={safeUrl(s.url)!} target="_blank" rel="noreferrer">
                        {s.title ?? s.url}
                      </a>
                    ) : (
                      s.title
                    )}
                    {s.ts ? <span className={ui.muted}> · {dayTime(s.ts)}</span> : null}
                  </li>
                ))}
              </ul>
            ) : null}
          </div>
        ) : null}
        {error ? <WriteError error={error} /> : null}
      </div>
    </section>
  );
}
