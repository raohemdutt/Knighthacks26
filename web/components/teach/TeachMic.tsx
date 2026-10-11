"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { VoiceBar } from "@/components/voice/VoiceBar";
import v from "@/components/voice/voice.module.css";
import { ApiError, api } from "@/lib/api";
import { askGuide } from "@/lib/guideBus";
import { speechForAnswer, voiceCommand } from "@/lib/speechText";
import { screenIntent, useVoice } from "@/lib/voice";
import teach from "./teach.module.css";

type Note = { text: string; tone: "info" | "error" };

const RELISTEN_MS = 350;

/**
 * The Atlas mic on Teach: the same bar as on Ask (components/voice/VoiceBar.tsx), docked at the foot of the screen.
 *
 * What is said goes, in this order: to the look guide when it asks where something in the app is; to this screen's
 * own commands (the page registers them with `setScreenIntent("/teach/", ...)`: start teaching, how the run is
 * going, open the Learning lab); and otherwise to the box as a question, POST /api/ask, exactly as on Ask. The
 * answer is shown above the bar and said aloud when "Speak answers" is on.
 *
 * Nothing here approves anything: "start teaching" presses the screen's own button, which raises the approval card.
 *
 * `heard` is what one of the screen's commands answered when the words came from somewhere else in the app (the
 * command palette): it is shown here as if it had been said here.
 */
export function TeachMic({ heard: fromScreen }: { heard?: { asked: string; answer: string } | null }) {
  const voice = useVoice();
  const [thinking, setThinking] = useState(false);
  const [note, setNote] = useState<Note | null>(null);
  const [turn, setTurn] = useState<{ asked: string; answer: string } | null>(null);
  const busy = useRef(false);
  const alive = useRef(true);
  const conversing = useRef(false);
  const listenRef = useRef<() => void>(() => undefined);
  useEffect(() => {
    if (fromScreen) setTurn(fromScreen);
  }, [fromScreen]);

  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      conversing.current = false;
    };
  }, []);

  /** Say the reply when the speaker is on; with hands-free, open the microphone again after it. */
  const respond = useCallback(
    (reply: string) => {
      const again = () => {
        if (!alive.current || !conversing.current || !voice.getSettings().handsFree || document.visibilityState !== "visible") {
          conversing.current = false;
          return;
        }
        window.setTimeout(() => {
          if (alive.current && conversing.current && !busy.current) listenRef.current();
        }, RELISTEN_MS);
      };
      if (!voice.getSettings().speak || !voice.canSpeak) return again();
      voice.say(reply, { tag: "answer", replace: true, raw: true, onDone: (finished) => (finished ? again() : void (conversing.current = false)) });
    },
    [voice],
  );

  const heard = useCallback(
    async (said: string) => {
      const q = said.trim();
      if (!q || busy.current) return;
      if (voiceCommand(q) === "stop") {
        conversing.current = false;
        setNote({ text: "Stopped listening. Tap the mic to talk again.", tone: "info" });
        return;
      }
      if (askGuide(q, "/teach/")) return; // "where do I start teaching?": the look guide points, nothing is asked of the box
      busy.current = true;
      setNote(null);
      setThinking(true);
      try {
        const own = await screenIntent("/teach/", q); // this screen's own commands first
        if (own !== null) {
          setTurn({ asked: q, answer: own });
          respond(own);
          return;
        }
        const answer = await api.ask(q);
        setTurn({ asked: q, answer: answer.answer });
        respond(speechForAnswer(answer.answer, answer.citations.filter((c) => c.frame_ref || c.episode_id).length));
      } catch (err) {
        const e = err instanceof ApiError ? err : new ApiError(0, String(err));
        conversing.current = false;
        setNote({
          text:
            e.status === 504
              ? "The model took too long to answer. Try again in a moment."
              : e.needsToken
                ? "This box asks for a token before it answers. Add it under Settings."
                : `Atlas could not answer: ${e.message}`,
          tone: "error",
        });
      } finally {
        busy.current = false;
        if (alive.current) setThinking(false);
      }
    },
    [respond],
  );

  const listen = useCallback(() => {
    const started = voice.listen({
      onFinal: (text) => void heard(text),
      onEnd: (why, message) => {
        if (why === "stopped") return;
        conversing.current = false;
        setNote(why === "error" ? { text: message ?? "Listening failed. Tap the mic to try again.", tone: "error" } : { text: "I did not hear anything, so I stopped listening.", tone: "info" });
      },
    });
    if (!started) {
      conversing.current = false;
      setNote({ text: "The microphone could not be started.", tone: "error" });
    }
  }, [voice, heard]);
  listenRef.current = listen;

  return (
    <div className={v.dock} data-guide="teach:mic">
      {turn ? (
        <div className={teach.said} role="status" aria-live="polite">
          <p className={teach.saidAsked}>You: {turn.asked}</p>
          <p className={teach.saidAnswer}>{turn.answer}</p>
        </div>
      ) : null}
      <VoiceBar
        phase={thinking ? "thinking" : voice.status}
        note={note}
        onTalk={() => {
          setNote(null);
          conversing.current = true;
          listen();
        }}
        onFinish={() => voice.stopListening(true)}
        onStop={() => {
          conversing.current = false;
          voice.stop();
        }}
      />
    </div>
  );
}
