"use client";

import { useState } from "react";
import { ApiError, api } from "@/lib/api";
import { EvidenceLine, evidenceIds, useOpen } from "./bits";
import s from "./spot.module.css";
import type { Answer, Handoff } from "./types";

const SUGGESTED = ["What should I check first this shift?", "Is V2 better than V1?", "What happened at WP05?", "Which inspection types are least reliable?", "Show hidden failures", "Status of HX-202"];

/** Ask, with evidence: the answer is built by code from the tables; every id in it opens its panel. */
export function AskBox({ modelLine }: { modelLine?: string }) {
  const { open } = useOpen();
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [error, setError] = useState<string | null>(null);

  const ask = (q: string) => {
    const question = q.trim();
    if (!question || busy) return;
    setBusy(true);
    setError(null);
    setText(question);
    api
      .spotAsk(question)
      .then(setAnswer)
      .catch((e: unknown) => setError(e instanceof ApiError ? (e.needsToken ? "This box asks for a token (Settings, top right)." : e.message) : String(e)))
      .finally(() => setBusy(false));
  };

  return (
    <div className={s.card}>
      <div className={s.cardHead}>
        <h2>Ask</h2>
        <span className={`${s.small} ${s.muted}`}>answers come from the tables on this box, with the rows cited</span>
      </div>
      <form
        className={s.askRow}
        onSubmit={(e) => {
          e.preventDefault();
          ask(text);
        }}
      >
        <input value={text} onChange={(e) => setText(e.target.value)} placeholder="What should I check first this shift?" aria-label="Question about the inspections" maxLength={500} />
        <button type="submit" className={s.play} disabled={busy || !text.trim()}>
          {busy ? "…" : "Ask"}
        </button>
      </form>
      <div className={s.chips}>
        {SUGGESTED.map((q) => (
          <button type="button" key={q} className={s.chip} onClick={() => ask(q)} disabled={busy}>
            {q}
          </button>
        ))}
      </div>
      {error ? <p className={`${s.small} ${s.toneBad}`}>{error}</p> : null}
      {answer ? (
        <div className={s.answer} aria-live="polite">
          <span className={s.q}>{answer.question}</span>
          <p style={{ whiteSpace: "pre-line" }}>{answer.text}</p>
          <EvidenceLine evidence={answer.citations} max={18} />
          <span className={`${s.small} ${s.muted}`}>
            {answer.rephrased ? `Worded by ${answer.model}; every number and id checked against the draft.` : "Built from the tables by code; no model wrote this."}
            {answer.check && !answer.rephrased && answer.check !== "ok" ? ` (model wording dropped: ${answer.check})` : ""}
          </span>
          {answer.target && evidenceIds(answer.citations).length > 0 ? (
            <button type="button" className={s.linkBtn} style={{ alignSelf: "flex-start" }} onClick={() => answer.target && open(answer.target)}>
              Open the evidence
            </button>
          ) : null}
        </div>
      ) : null}
      {modelLine ? <p className={`${s.small} ${s.muted}`}>{modelLine}</p> : null}
    </div>
  );
}

export function HandoffCard({ handoff }: { handoff: Handoff }) {
  const { open } = useOpen();
  const [copied, setCopied] = useState(false);
  const copy = () => {
    const done = () => (setCopied(true), window.setTimeout(() => setCopied(false), 1800));
    if (navigator.clipboard?.writeText) navigator.clipboard.writeText(handoff.text).then(done, () => undefined);
  };
  return (
    <div className={s.card}>
      <div className={s.cardHead}>
        <h2>Shift handoff</h2>
        <button type="button" className={s.linkBtn} onClick={copy}>
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <ol className={s.handoff}>
        {handoff.lines.map((ln, i) => (
          <li key={i}>
            {ln.target ? (
              <button type="button" onClick={() => ln.target && open(ln.target)} title="Open the evidence">
                {ln.text}
              </button>
            ) : (
              <span>{ln.text}</span>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}
