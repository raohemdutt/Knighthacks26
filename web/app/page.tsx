"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import { Evidence } from "@/components/Evidence";
import ui from "@/components/ui.module.css";
import { ApiError, api } from "@/lib/api";
import { useAtlas } from "@/lib/atlas";
import { time } from "@/lib/format";

const DEMO_QUESTIONS = [
  "Where did I leave my keys?",
  "What is due this week?",
  "What happened at our table overnight?",
  "What should I eat?",
  "Where should I go tonight?",
];

let turnSeq = 0;
const nextId = () => `t${Date.now()}-${turnSeq++}`;

export default function AskPage() {
  const { chat, setChat } = useAtlas();
  const [text, setText] = useState("");
  const [waiting, setWaiting] = useState<string | null>(null); // the question being answered
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (chat.length > 0 || waiting) endRef.current?.scrollIntoView({ block: "end", behavior: "smooth" });
  }, [chat.length, waiting]);

  const ask = async (question: string) => {
    const q = question.trim();
    if (!q || waiting) return;
    setText("");
    setWaiting(q);
    setChat((turns) => [...turns, { id: nextId(), role: "you", text: q }]);
    try {
      const answer = await api.ask(q);
      setChat((turns) => [...turns, { id: nextId(), role: "atlas", answer }]);
    } catch (err) {
      const e = err instanceof ApiError ? err : new ApiError(0, String(err));
      const message =
        e.status === 504
          ? "The model took too long to answer. Try again in a moment."
          : e.needsToken
            ? "This box asks for a token before it answers. Add it under Settings, top right."
            : `${e.message}${e.status ? ` (HTTP ${e.status})` : ""}`;
      setChat((turns) => [...turns, { id: nextId(), role: "error", text: message, needsToken: e.needsToken }]);
    } finally {
      setWaiting(null);
    }
  };

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void ask(text);
  };

  return (
    <>
      {chat.length === 0 ? (
        <div className={ui.pageHead}>
          <h1>Ask Atlas</h1>
          <p className={ui.muted}>
            Ask about your desk, your classes or your evening. Every answer comes with the camera frames and links it is based on, and the
            model that wrote it.
          </p>
        </div>
      ) : null}

      <div className={ui.turns} aria-live="polite">
        {chat.map((turn) => {
          if (turn.role === "you")
            return (
              <p key={turn.id} className={ui.you}>
                {turn.text}
              </p>
            );
          if (turn.role === "error")
            return (
              <p key={turn.id} className={`${ui.state} ${ui.stateError} ${ui.breakWords}`} role="alert">
                {turn.text}
              </p>
            );
          const a = turn.answer;
          return (
            <article key={turn.id} className={`${ui.card} ${ui.answer}`}>
              <p className={ui.answerText}>{a.answer}</p>
              <Evidence citations={a.citations} />
              <p className={ui.modelLine}>
                Answered by <strong>{a.model_version}</strong> at {time(a.ts)}
              </p>
            </article>
          );
        })}
        {waiting ? (
          <div className={ui.loading} role="status">
            <span className={ui.spinner} aria-hidden="true" />
            <span>Atlas is looking through its memory…</span>
          </div>
        ) : null}
        <div ref={endRef} />
      </div>

      <section className={ui.section} aria-label="Suggested questions">
        {chat.length === 0 ? <p className={ui.sectionTitle}>Try one</p> : null}
        <div className={ui.chips}>
          {DEMO_QUESTIONS.map((q) => (
            <button key={q} type="button" className={ui.chip} disabled={waiting !== null} onClick={() => void ask(q)}>
              {q}
            </button>
          ))}
        </div>
      </section>

      <form className={ui.composer} onSubmit={onSubmit}>
        <input
          className={ui.input}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Ask anything"
          aria-label="Your question"
          enterKeyHint="send"
          autoComplete="off"
        />
        <button type="submit" className={`${ui.btn} ${ui.btnPrimary}`} disabled={waiting !== null || text.trim() === ""}>
          Ask
        </button>
      </form>
    </>
  );
}
