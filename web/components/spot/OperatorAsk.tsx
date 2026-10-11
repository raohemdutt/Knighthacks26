"use client";

import { useState } from "react";
import ui from "@/components/ui.module.css";
import { ApiError, api } from "@/lib/api";
import o from "./op.module.css";
import { Why } from "./Shift";
import type { Answer, Handoff, Target } from "./types";

/**
 * Ask and handoff, simplified: one input, four questions in operator language (from the API), and one button that writes
 * the shift handoff as a note to copy. Answers are built by code from the tables on this box; `plain` words them for an
 * operator and "exact wording" shows the same answer with its numbers and row ids.
 */
export function OperatorAsk({ chips, onWhy }: { chips: string[]; onWhy: (t: Target) => void }) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [answer, setAnswer] = useState<(Answer & { plain?: boolean }) | null>(null);
  const [exact, setExact] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<Handoff | null>(null);
  const [copied, setCopied] = useState(false);

  const fail = (e: unknown) => setError(e instanceof ApiError ? (e.needsToken ? "This box asks for a token first (Settings)." : e.message) : String(e));
  const ask = (q: string) => {
    const question = q.trim();
    if (!question || busy) return;
    setBusy(true);
    setError(null);
    setExact(false);
    setText(question);
    api.spotAskPlain(question).then(setAnswer).catch(fail).finally(() => setBusy(false));
  };
  const handoff = () => {
    setBusy(true);
    setError(null);
    api.spotHandoff().then(setNote).catch(fail).finally(() => setBusy(false));
  };
  const copy = () => {
    if (!note || !navigator.clipboard?.writeText) return;
    navigator.clipboard.writeText(note.text).then(() => (setCopied(true), window.setTimeout(() => setCopied(false), 1800)), () => undefined);
  };

  return (
    <section className={o.card} aria-label="Ask and handoff">
      <div className={o.cardHead}>
        <h2>Ask SPOT's data</h2>
        <span className={o.hint}>answers come from this box</span>
      </div>
      <form
        className={o.askRow}
        onSubmit={(e) => {
          e.preventDefault();
          ask(text);
        }}
      >
        <input className={ui.input} value={text} onChange={(e) => setText(e.target.value)} placeholder="Ask in your own words" aria-label="Question about the inspections" maxLength={500} />
        <button type="submit" className={`${ui.btn} ${ui.btnPrimary}`} disabled={busy || !text.trim()}>
          Ask
        </button>
      </form>
      <div className={ui.chips}>
        {chips.map((q) => (
          <button type="button" key={q} className={ui.chip} onClick={() => ask(q)} disabled={busy}>
            {q}
          </button>
        ))}
      </div>
      {error ? <p className={o.error}>{error}</p> : null}
      {answer ? (
        <div className={o.answer} aria-live="polite">
          <p className={o.answerQ}>{answer.question}</p>
          <p className={o.answerText}>{exact ? answer.draft : answer.text}</p>
          <div className={o.itemActs}>
            {answer.target ? <Why to={answer.target} onWhy={onWhy}>Why? See the evidence</Why> : null}
            {answer.plain ? (
              <button type="button" className={o.why} onClick={() => setExact((v) => !v)}>
                {exact ? "Plain words" : "Exact wording, with the numbers"}
              </button>
            ) : null}
          </div>
        </div>
      ) : null}

      <button type="button" className={`${ui.btn} ${o.act}`} onClick={handoff} disabled={busy}>
        Write my shift handoff
      </button>
      {note ? (
        <div className={o.handoff}>
          <div className={`${ui.row} ${ui.between}`}>
            <b>Shift handoff</b>
            <button type="button" className={ui.btn} onClick={copy}>
              {copied ? "Copied" : "Copy"}
            </button>
          </div>
          <ol>
            {note.lines.map((ln, i) => (
              <li key={i}>
                {ln.text}{" "}
                {ln.target ? <Why to={ln.target} onWhy={onWhy} /> : null}
              </li>
            ))}
          </ol>
        </div>
      ) : null}
    </section>
  );
}
