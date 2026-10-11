"use client";

import { useState } from "react";
import { useAttention } from "@/components/guide/Attention";
import ui from "@/components/ui.module.css";
import { ApiError, api } from "@/lib/api";
import { plainText, whereIntent } from "@/lib/guide";
import { handoffLine, type Job } from "@/lib/guideAttention";
import { pointAt } from "@/lib/guideBus";
import { NEXT, nextDoFirst, spotRun } from "@/lib/guideSpot";
import { useVoice } from "@/lib/voice";
import o from "./op.module.css";
import { Why } from "./Shift";
import type { Answer, Handoff, Target } from "./types";

/**
 * Ask and handoff, simplified: one input, four questions in operator language (from the API), and one button that writes
 * the shift handoff as a note to copy. Answers are built by code from the tables on this box; `plain` words them for an
 * operator and "exact wording" shows the same answer with its numbers and row ids.
 *
 * "Where do I look for X?", typed or spoken, is not answered in words: the look guide points at X on this screen and talks
 * the person there (GET /api/spot/guide, no model call). When the box is not sure what X is, the question is answered as
 * before and the closest places are offered under it.
 */
export function OperatorAsk({ chips, onWhy, guideChips = [], jobs = [] }: { chips: string[]; onWhy: (t: Target) => void; guideChips?: string[]; jobs?: Job[] }) {
  const voice = useVoice();
  const looks = useAttention();
  const [reviewed, setReviewed] = useState(false); // "include what I reviewed": one sentence, only when ticked
  const extra = reviewed && looks.mode !== "off" ? handoffLine(looks.seen, jobs) : null;
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [answer, setAnswer] = useState<(Answer & { plain?: boolean }) | null>(null);
  const [exact, setExact] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [note, setNote] = useState<Handoff | null>(null);
  const [copied, setCopied] = useState(false);
  const [meant, setMeant] = useState<{ target: string; label: string }[]>([]);

  const fail = (e: unknown) => setError(e instanceof ApiError ? (e.needsToken ? "This box asks for a token first (Settings)." : e.message) : String(e));
  /** True when the pointer took the question. */
  const guide = async (question: string): Promise<boolean> => {
    try {
      const { run, found } = await spotRun(question);
      if (run) return pointAt(run), true;
      setMeant(found?.alternatives ?? []);
    } catch {
      /* the guide is an extra: the question is still answered */
    }
    return false;
  };
  const ask = (q: string) => {
    const question = q.trim();
    if (!question || busy) return;
    setBusy(true);
    setError(null);
    setExact(false);
    setMeant([]);
    setText(question);
    void (async () => {
      // hands busy: "next" steps through the do-first list with the pointer
      if (NEXT.test(plainText(question)) && (await guide(nextDoFirst()))) return setBusy(false), setText("");
      if (whereIntent(question) !== null && (await guide(question))) return setBusy(false);
      api.spotAskPlain(question).then(setAnswer).catch(fail).finally(() => setBusy(false));
    })();
  };
  const handoff = () => {
    setBusy(true);
    setError(null);
    api.spotHandoff().then(setNote).catch(fail).finally(() => setBusy(false));
  };
  const copy = () => {
    if (!note || !navigator.clipboard?.writeText) return;
    navigator.clipboard.writeText(extra ? `${note.text}\n${extra}` : note.text).then(() => (setCopied(true), window.setTimeout(() => setCopied(false), 1800)), () => undefined);
  };
  const listening = voice.status === "listening";
  const talk = () => {
    if (listening) return voice.stopListening(true);
    voice.listen({ onInterim: setText, onFinal: ask, onEnd: () => setText("") });
  };

  return (
    <section className={o.card} aria-label="Ask and handoff" data-guide="ask">
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
        <input className={ui.input} value={text} onChange={(e) => setText(e.target.value)} placeholder={listening ? "Listening…" : "Ask in your own words"} aria-label="Question about the inspections" maxLength={500} />
        {voice.listenSupport?.ok ? (
          <button type="button" className={ui.btn} onClick={talk} disabled={busy} aria-pressed={listening} aria-label={listening ? "Done talking" : "Ask out loud"}>
            {listening ? "Done" : "Speak"}
          </button>
        ) : null}
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
      {guideChips.length > 0 ? (
        <div className={ui.chips} role="group" aria-label="Guide me" data-guide="guide-me">
          <span className={o.hint}>Guide me:</span>
          {guideChips.map((q) => (
            <button type="button" key={q} className={ui.chip} onClick={() => ask(q)} disabled={busy}>
              {q}
            </button>
          ))}
        </div>
      ) : null}
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
      {meant.length > 0 ? (
        <div className={ui.chips} role="group" aria-label="Did you mean">
          <span className={o.hint}>Point me to:</span>
          {meant.map((m) => (
            <button type="button" key={m.target} className={ui.chip} onClick={() => void guide(m.target)}>
              {m.label}
            </button>
          ))}
        </div>
      ) : null}

      <button type="button" className={`${ui.btn} ${o.act}`} data-guide="handoff" onClick={handoff} disabled={busy}>
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
            {extra ? <li data-handoff-reviewed>{extra}</li> : null}
          </ol>
          {looks.mode !== "off" ? (
            <label className={o.note}>
              <input type="checkbox" checked={reviewed} onChange={(e) => setReviewed(e.target.checked)} /> Include what I reviewed (one line, by gaze{looks.mode === "sim" ? ", simulated" : ""}). Never added
              unless you tick this.
            </label>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
