"use client";

import { useRef, useState } from "react";
import { liveChatStream } from "@/lib/api";
import { CHAT_TARGETS, type ChatDone, type ChatTarget, WATER_BOTTLE, footerText, keepLast } from "@/lib/liveChat";
import c from "./liveChat.module.css";

/**
 * The live chat beside the power graph: ask the pair or the box, watch the answer's tokens arrive from
 * POST /api/learning/lab/live-chat while the graph beside it shades the same seconds, then read what the answer cost.
 * `onWindow` tells the graph when an answer started (end null while streaming) and when it ended.
 */

interface Exchange {
  id: number;
  question: string;
  target: ChatTarget;
  text: string;
  tools: string[];
  done: ChatDone | null;
  error: string | null;
  streaming: boolean;
}

export function LiveChat({ reduced, onWindow }: { reduced: boolean; onWindow: (start: number, end: number | null) => void }) {
  const [target, setTarget] = useState<ChatTarget>("pair");
  const [question, setQuestion] = useState(WATER_BOTTLE);
  const [log, setLog] = useState<Exchange[]>([]);
  const busy = log.some((x) => x.streaming);
  const nextId = useRef(1);

  const update = (id: number, fn: (x: Exchange) => Exchange) => setLog((l) => l.map((x) => (x.id === id ? fn(x) : x)));

  const send = async () => {
    const q = question.trim();
    if (!q || busy) return;
    const id = nextId.current++;
    setLog((l) => keepLast([...l, { id, question: q, target, text: "", tools: [], done: null, error: null, streaming: true }], 3));
    const start = Date.now();
    onWindow(start, null);
    try {
      await liveChatStream(q, target, (e) => {
        if (e.event === "token") update(id, (x) => ({ ...x, text: x.text + e.data.text }));
        else if (e.event === "tool") update(id, (x) => ({ ...x, tools: [...x.tools, e.data.name] }));
        else if (e.event === "done") update(id, (x) => ({ ...x, done: e.data, text: x.text.trim() ? x.text : e.data.answer }));
        else if (e.event === "error") update(id, (x) => ({ ...x, error: e.data.error }));
      });
    } catch (err) {
      update(id, (x) => ({ ...x, error: err instanceof Error ? err.message : "the answer did not arrive" }));
    } finally {
      update(id, (x) => ({ ...x, streaming: false }));
      onWindow(start, Date.now());
    }
  };

  return (
    <div className={c.chat}>
      <div className={c.toggle} role="radiogroup" aria-label="Who answers">
        {CHAT_TARGETS.map((o) => (
          <button
            key={o.id}
            type="button"
            role="radio"
            aria-checked={target === o.id}
            className={`${c.opt} ${target === o.id ? c.on : ""}`}
            onClick={() => setTarget(o.id)}
            disabled={busy}
          >
            {o.label}
          </button>
        ))}
      </div>
      <form
        className={c.ask}
        onSubmit={(e) => {
          e.preventDefault();
          void send();
        }}
      >
        <input className={c.input} value={question} onChange={(e) => setQuestion(e.target.value)} aria-label="Your question" maxLength={300} />
        <button className={c.send} type="submit" disabled={busy || !question.trim()}>
          Send
        </button>
      </form>
      <div className={c.log} aria-live="polite">
        {log.length === 0 ? <p className={c.hint}>Send it and watch the watts beside this move while the words arrive.</p> : null}
        {log.map((x) => (
          <div key={x.id} className={c.item}>
            <p className={c.q}>
              {x.question} <span className={c.who}>{x.target === "pair" ? "the pair" : "the box"}</span>
            </p>
            {x.tools.length ? <p className={c.tool}>looked in the box&apos;s memory: {x.tools.join(", ")}</p> : null}
            <p className={c.a}>
              {x.text}
              {x.streaming ? <span className={`${c.caret} ${reduced ? "" : c.blink}`} aria-hidden="true" /> : null}
            </p>
            {x.error ? <p className={c.err}>{x.error}</p> : null}
            {x.done ? <p className={c.foot}>{footerText(x.done)}</p> : null}
          </div>
        ))}
      </div>
    </div>
  );
}
