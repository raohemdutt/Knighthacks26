"use client";

import { useState } from "react";
import type { EditResult, Move } from "@/lib/api";
import ui from "../ui.module.css";
import { useVoice } from "@/lib/voice";
import pl from "./plan.module.css";

/**
 * "Tell Atlas what to change": a change to the plan in plain words, typed or said ("move lunch to one", "drop the
 * gym", "study two hours before judging"). The box reads it with a small grammar, no model
 * (contexts/play/dayrun/talk.py), and answers with what it did. When it cannot tell what was meant it changes
 * nothing, says what it understood and offers the nearest edits as buttons.
 */
export function TellAtlas({ ask, onApply }: { ask: (text: string) => Promise<EditResult>; onApply: (moves: Move[], changed: number | null) => void }) {
  const voice = useVoice();
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [listening, setListening] = useState(false);
  const [said, setSaid] = useState<EditResult | { error: string } | null>(null);

  const send = async (words: string, spoken = false) => {
    const clean = words.trim();
    if (!clean || busy) return;
    setBusy(true);
    try {
      const out = await ask(clean);
      setSaid(out);
      if (out.ok) {
        setText("");
        onApply(out.moves, out.changed);
      }
      if (spoken) voice.say(out.warning ? `${out.understood} ${out.warning}.` : out.understood, { tag: "answer", replace: true });
    } catch (err) {
      setSaid({ error: err instanceof Error ? err.message : "The box did not answer." });
    } finally {
      setBusy(false);
    }
  };

  const talk = () => {
    if (listening) {
      voice.stopListening(true);
      return;
    }
    const started = voice.listen({
      onInterim: setText,
      onFinal: (heard) => {
        setListening(false);
        setText(heard);
        void send(heard, true);
      },
      onEnd: () => setListening(false),
    });
    setListening(started);
  };

  const canListen = voice.listenSupport?.ok === true;
  return (
    <div className={pl.tell}>
      <form
        className={pl.tellRow}
        onSubmit={(e) => {
          e.preventDefault();
          void send(text);
        }}
      >
        <input
          className={pl.tellInput}
          aria-label="Tell Atlas what to change"
          placeholder="Tell Atlas what to change…"
          value={text}
          onChange={(e) => setText(e.target.value)}
          enterKeyHint="send"
          autoComplete="off"
          maxLength={300}
        />
        {canListen ? (
          <button type="button" className={`${pl.tellBtn} ${listening ? pl.tellBtnOn : ""}`} onClick={talk} aria-pressed={listening} aria-label={listening ? "Done talking" : "Say the change"}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <rect x="9" y="3.5" width="6" height="11" rx="3" />
              <path d="M5.5 11.5a6.5 6.5 0 0013 0M12 18v2.5" />
            </svg>
          </button>
        ) : null}
        <button type="submit" className={pl.tellBtn} disabled={busy || !text.trim()} aria-label="Change the plan">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M5 12h13M12.5 6.5L18 12l-5.5 5.5" />
          </svg>
        </button>
      </form>
      {said ? (
        <div role="status" aria-live="polite">
          {"error" in said ? (
            <p className={`${pl.heard} ${pl.heardWarn}`}>{said.error}</p>
          ) : (
            <>
              <p className={pl.heard}>{said.understood}</p>
              {said.warning ? <p className={`${pl.heard} ${pl.heardWarn}`}>{said.warning}.</p> : null}
              {!said.ok && said.options.length > 0 ? (
                <div className={ui.chips}>
                  {said.options.map((o) => (
                    <button
                      key={o.label}
                      type="button"
                      className={ui.chip}
                      onClick={() => {
                        onApply(o.moves, o.changed);
                        setSaid(null);
                        setText("");
                      }}
                    >
                      {o.label}
                    </button>
                  ))}
                </div>
              ) : null}
            </>
          )}
        </div>
      ) : null}
    </div>
  );
}
