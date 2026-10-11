"use client";

import { useRef, useState } from "react";
import { liveChatStream } from "@/lib/api";
import {
  CHAT_MODES,
  type ChatDone,
  type ChatMode,
  type ChatTarget,
  SIDE_NAMES,
  WATER_BOTTLE,
  compareText,
  footerText,
  keepLast,
  targetsOf,
} from "@/lib/liveChat";
import c from "./liveChat.module.css";

/**
 * The live chat beside the power graph: one question, sent to the box and the network brain at once (or to one of
 * them), each answer streaming from POST /api/learning/lab/live-chat into its own transcript while the graph beside
 * it shades the same seconds, then what each answer cost and one line comparing them. `onWindow` tells the graph when
 * the first token of either side arrived (end null while streaming) and when the last side finished.
 */

interface Side {
  text: string;
  tools: string[];
  done: ChatDone | null;
  error: string | null;
  streaming: boolean;
}

interface Exchange {
  id: number;
  question: string;
  sides: Partial<Record<ChatTarget, Side>>;
}

const fresh = (): Side => ({ text: "", tools: [], done: null, error: null, streaming: true });

export function LiveChat({ reduced, onWindow }: { reduced: boolean; onWindow: (start: number, end: number | null) => void }) {
  const [mode, setMode] = useState<ChatMode>("both");
  const [question, setQuestion] = useState(WATER_BOTTLE);
  const [log, setLog] = useState<Exchange[]>([]);
  const busy = log.some((x) => Object.values(x.sides).some((s) => s?.streaming));
  const nextId = useRef(1);

  const update = (id: number, target: ChatTarget, fn: (s: Side) => Side) =>
    setLog((l) => l.map((x) => (x.id === id && x.sides[target] ? { ...x, sides: { ...x.sides, [target]: fn(x.sides[target]!) } } : x)));

  const send = async () => {
    const q = question.trim();
    if (!q || busy) return;
    const id = nextId.current++;
    const targets = targetsOf(mode);
    const sides: Partial<Record<ChatTarget, Side>> = {};
    for (const t of targets) sides[t] = fresh();
    setLog((l) => keepLast([...l, { id, question: q, sides }], 3));
    let first: number | null = null; // the first token of either side: where the graph's shading starts
    const ask = async (target: ChatTarget) => {
      try {
        await liveChatStream(q, target, (e) => {
          if (e.event === "token") {
            if (first === null) {
              first = Date.now();
              onWindow(first, null);
            }
            update(id, target, (s) => ({ ...s, text: s.text + e.data.text }));
          } else if (e.event === "tool") update(id, target, (s) => ({ ...s, tools: [...s.tools, e.data.name] }));
          else if (e.event === "done") update(id, target, (s) => ({ ...s, done: e.data, text: s.text.trim() ? s.text : e.data.answer }));
          else if (e.event === "error") update(id, target, (s) => ({ ...s, error: e.data.error }));
        });
      } catch (err) {
        update(id, target, (s) => ({ ...s, error: err instanceof Error ? err.message : "the answer did not arrive" }));
      } finally {
        update(id, target, (s) => ({ ...s, streaming: false }));
      }
    };
    await Promise.all(targets.map(ask));
    if (first !== null) onWindow(first, Date.now()); // the last done
  };

  return (
    <div className={c.chat}>
      <div className={c.toggle} role="radiogroup" aria-label="Who answers">
        {CHAT_MODES.map((o) => (
          <button
            key={o.id}
            type="button"
            role="radio"
            aria-checked={mode === o.id}
            className={`${c.opt} ${mode === o.id ? c.on : ""}`}
            onClick={() => setMode(o.id)}
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
        {log.map((x) => {
          const box = x.sides.box?.done;
          const pair = x.sides.pair?.done;
          return (
            <div key={x.id} className={c.item}>
              <p className={c.q}>{x.question}</p>
              {(["box", "pair"] as ChatTarget[]).map((target) => {
                const s = x.sides[target];
                if (!s) return null;
                return (
                  <div key={target} className={c.side}>
                    <p className={c.who}>{SIDE_NAMES[target]}</p>
                    {s.tools.length ? <p className={c.tool}>looked in the box&apos;s memory: {s.tools.join(", ")}</p> : null}
                    <p className={c.a}>
                      {s.text}
                      {s.streaming ? <span className={`${c.caret} ${reduced ? "" : c.blink}`} aria-hidden="true" /> : null}
                    </p>
                    {s.error ? <p className={c.err}>{s.error}</p> : null}
                    {s.done ? <p className={c.foot}>{footerText(s.done)}</p> : null}
                  </div>
                );
              })}
              {box && pair ? <p className={c.compare}>{compareText(box, pair)}</p> : null}
            </div>
          );
        })}
      </div>
    </div>
  );
}
