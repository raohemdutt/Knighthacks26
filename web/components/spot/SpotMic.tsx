"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useAttention } from "@/components/guide/Attention";
import ui from "@/components/ui.module.css";
import { VoiceBar } from "@/components/voice/VoiceBar";
import v from "@/components/voice/voice.module.css";
import { ApiError, api, type Answer as AtlasAnswer } from "@/lib/api";
import { speakOnBox, stopBoxSpeech } from "@/lib/boxSpeech";
import { time } from "@/lib/format";
import { plainText, whereIntent } from "@/lib/guide";
import { handoffLine, type Job } from "@/lib/guideAttention";
import { cancelGuide, onGuide, pointAt } from "@/lib/guideBus";
import { NEXT, nextDoFirst, spotRun } from "@/lib/guideSpot";
import { speechForAnswer, voiceCommand } from "@/lib/speechText";
import { useVoice } from "@/lib/voice";
import o from "./op.module.css";
import { Why } from "./Shift";
import m from "./spotmic.module.css";
import type { Answer, Handoff, Target } from "./types";

type Note = { text: string; tone: "info" | "error" };
type Turn = { asked: string; spot?: Answer & { plain?: boolean; matched?: boolean }; atlas?: AtlasAnswer };

const RELISTEN_MS = 350;

/**
 * The Atlas mic on Inspection: the same bar as on Ask and Teach (components/voice/VoiceBar.tsx), docked at the foot of
 * every Inspection view, with a line to type on. It is the one place to ask on this screen.
 *
 * What is said or typed goes, in this order: to the look guide when it asks where something is (GET /api/spot/guide: the
 * pointer goes there and talks the person through it); to the Inspection assistant (POST /api/spot/ask), whose answers are
 * built by code from the tables on this box; and, when that does not recognise the question (`matched: false`), to the box
 * as a question, POST /api/ask, exactly as on Ask. The answer is shown above the bar and said aloud when "Speak answers"
 * is on: on the box's own speaker when that works, else in this browser.
 *
 * One thing at a time: a new question first stops everything in flight (the guide's run and its walk, the lines waiting
 * for the box, what the box is saying for us, this browser's voice), and an answer still on its way is dropped.
 *
 * The suggested questions, the "Guide me" chips and the shift handoff open in a panel above the bar.
 */
export function SpotMic({ chips, onWhy, guideChips = [], jobs = [] }: { chips: string[]; onWhy: (t: Target) => void; guideChips?: string[]; jobs?: Job[] }) {
  const voice = useVoice();
  const looks = useAttention();
  const [text, setText] = useState("");
  const [thinking, setThinking] = useState(false);
  const [note, setNote] = useState<Note | null>(null);
  const [turn, setTurn] = useState<Turn | null>(null);
  const [exact, setExact] = useState(false);
  const [meant, setMeant] = useState<{ target: string; label: string }[]>([]);
  const [open, setOpen] = useState(false); // the panel with the chips and the handoff
  const [handoffNote, setHandoffNote] = useState<Handoff | null>(null);
  const [reviewed, setReviewed] = useState(false); // "include what I reviewed": one sentence, only when ticked
  const [copied, setCopied] = useState(false);
  const extra = reviewed && looks.mode !== "off" ? handoffLine(looks.seen, jobs) : null;
  const seq = useRef(0); // bumped by every new question: an older one's answer is dropped
  const alive = useRef(true);
  const conversing = useRef(false);
  const listenRef = useRef<() => void>(() => undefined);

  /** Stop everything in flight, and take the next turn. */
  const fresh = useCallback(async (): Promise<number> => {
    const mine = ++seq.current;
    cancelGuide();
    voice.stop();
    await stopBoxSpeech();
    return mine;
  }, [voice]);

  useEffect(() => {
    alive.current = true;
    return () => {
      // leaving Inspection stops the walk and the speech too
      alive.current = false;
      conversing.current = false;
      seq.current++;
      cancelGuide();
      void stopBoxSpeech();
    };
  }, []);

  // the guide is about to point at something in the panel: open it first
  useEffect(() => onGuide((g) => (g.kind === "run" && /^(guide-me|handoff)$/.test(g.run.target.id) ? setOpen(true) : undefined)), []);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => (e.key === "Escape" ? setOpen(false) : undefined);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  /** Say the reply when the speaker is on (the box's speaker first); with hands-free, open the microphone again after it. */
  const respond = useCallback(
    (reply: string, mine: number) => {
      const again = () => {
        if (!alive.current || mine !== seq.current || !conversing.current || !voice.getSettings().handsFree || document.visibilityState !== "visible") {
          conversing.current = false;
          return;
        }
        window.setTimeout(() => {
          if (alive.current && conversing.current && mine === seq.current) listenRef.current();
        }, RELISTEN_MS);
      };
      if (!reply || !voice.getSettings().speak) return again();
      void speakOnBox(reply).then((onBox) => {
        if (!alive.current || mine !== seq.current) return;
        if (onBox) conversing.current = false; // the box is talking and this screen cannot tell when it stops: the microphone is not reopened
        else if (voice.canSpeak) voice.say(reply, { tag: "answer", replace: true, raw: true, onDone: (finished) => (finished ? again() : void (conversing.current = false)) });
        else again();
      });
    },
    [voice],
  );

  /** True when the pointer took the question (or a newer question has taken over). */
  const guide = useCallback(
    async (question: string, mine: number): Promise<boolean> => {
      try {
        const { run, found } = await spotRun(question, null, Boolean(voice.canSpeak && voice.getSettings().speak));
        if (mine !== seq.current) return true;
        if (run) return pointAt(run), true;
        setMeant(found?.alternatives ?? []);
      } catch {
        /* the guide is an extra: the question is still answered */
      }
      return false;
    },
    [voice],
  );

  const heard = useCallback(
    async (said: string) => {
      const q = said.trim();
      if (!q) return;
      if (voiceCommand(q) === "stop") {
        conversing.current = false;
        setNote({ text: "Stopped listening. Tap the mic to talk again.", tone: "info" });
        return;
      }
      setNote(null);
      setMeant([]);
      setExact(false);
      setThinking(true);
      const mine = await fresh();
      const live = () => alive.current && mine === seq.current;
      try {
        // hands busy: "next" steps through the do-first list with the pointer
        if (NEXT.test(plainText(q)) && (await guide(nextDoFirst(), mine))) return void (conversing.current = false);
        if (whereIntent(q) !== null && (await guide(q, mine))) return void (conversing.current = false);
        const spot = await api.spotAskPlain(q); // the Inspection assistant: built by code from the tables on this box
        if (!live()) return;
        if (spot.matched !== false) {
          setTurn({ asked: q, spot });
          respond(speechForAnswer(spot.text), mine);
          return;
        }
        const atlas = await api.ask(q); // not an Inspection question: the box answers it, exactly as on Ask
        if (!live()) return;
        setTurn({ asked: q, atlas });
        respond(speechForAnswer(atlas.answer, atlas.citations.filter((c) => c.frame_ref || c.episode_id).length), mine);
      } catch (err) {
        if (!live()) return;
        const e = err instanceof ApiError ? err : new ApiError(0, String(err));
        conversing.current = false;
        setNote({
          text: e.status === 504 ? "The model took too long to answer. Try again in a moment." : e.needsToken ? "This box asks for a token before it answers. Add it under Settings." : `Atlas could not answer: ${e.message}`,
          tone: "error",
        });
      } finally {
        if (live()) setThinking(false);
      }
    },
    [fresh, guide, respond],
  );

  const ask = (q: string) => {
    setText("");
    setOpen(false);
    void heard(q);
  };

  const listen = useCallback(() => {
    const started = voice.listen({
      onInterim: setText,
      onFinal: (said) => (setText(""), void heard(said)),
      onEnd: (why, message) => {
        setText("");
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

  const handoff = () => {
    setNote(null);
    setThinking(true);
    void (async () => {
      const mine = await fresh();
      api
        .spotHandoff()
        .then((n) => (mine === seq.current ? setHandoffNote(n) : undefined))
        .catch((e: unknown) => (mine === seq.current ? setNote({ text: e instanceof ApiError ? (e.needsToken ? "This box asks for a token first (Settings)." : e.message) : String(e), tone: "error" }) : undefined))
        .finally(() => (mine === seq.current ? setThinking(false) : undefined));
    })();
  };
  const copy = () => {
    if (!handoffNote || !navigator.clipboard?.writeText) return;
    navigator.clipboard.writeText(extra ? `${handoffNote.text}\n${extra}` : handoffNote.text).then(() => (setCopied(true), window.setTimeout(() => setCopied(false), 1800)), () => undefined);
  };

  const spot = turn?.spot;
  const atlas = turn?.atlas;
  return (
    <div className={`${v.dock} ${m.dock}`} data-guide="ask" role="region" aria-label="Ask and handoff">
      {turn || meant.length > 0 ? (
        <div className={m.said} role="status" aria-live="polite">
          <div className={m.saidHead}>
            <p className={m.asked}>{turn ? `You: ${turn.asked}` : null}</p>
            <button type="button" className={o.why} onClick={() => (setTurn(null), setMeant([]))} aria-label="Close the answer">
              Close
            </button>
          </div>
          {spot ? (
            <>
              <p className={m.answer}>{exact ? spot.draft : spot.text}</p>
              <div className={o.itemActs}>
                {spot.target ? <Why to={spot.target} onWhy={onWhy}>Why? See the evidence</Why> : null}
                {spot.plain ? (
                  <button type="button" className={o.why} onClick={() => setExact((x) => !x)}>
                    {exact ? "Plain words" : "Exact wording, with the numbers"}
                  </button>
                ) : null}
              </div>
            </>
          ) : null}
          {atlas ? (
            <>
              <p className={m.answer}>{atlas.answer}</p>
              <p className={ui.modelLine}>
                {atlas.answered_by ? <>Answered by {atlas.answered_by} </> : <>Answered by </>}
                <strong>{atlas.model_version}</strong> at {time(atlas.ts)}
              </p>
            </>
          ) : null}
          {meant.length > 0 ? (
            <div className={ui.chips} role="group" aria-label="Did you mean">
              <span className={o.hint}>Point me to:</span>
              {meant.map((x) => (
                <button type="button" key={x.target} className={ui.chip} onClick={() => void fresh().then((mine) => guide(x.target, mine))}>
                  {x.label}
                </button>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}

      {open ? (
        <div className={m.panel} id="spot-mic-panel">
          <div className={ui.chips}>
            {chips.map((q) => (
              <button type="button" key={q} className={ui.chip} onClick={() => ask(q)}>
                {q}
              </button>
            ))}
          </div>
          {guideChips.length > 0 ? (
            <div className={ui.chips} role="group" aria-label="Guide me" data-guide="guide-me">
              <span className={o.hint}>Guide me:</span>
              {guideChips.map((q) => (
                <button type="button" key={q} className={ui.chip} onClick={() => ask(q)}>
                  {q}
                </button>
              ))}
            </div>
          ) : null}
          <div>
            <button type="button" className={ui.btn} data-guide="handoff" onClick={handoff}>
              Write my shift handoff
            </button>
          </div>
          {handoffNote ? (
            <div className={o.handoff}>
              <div className={`${ui.row} ${ui.between}`}>
                <b>Shift handoff</b>
                <button type="button" className={ui.btn} onClick={copy}>
                  {copied ? "Copied" : "Copy"}
                </button>
              </div>
              <ol>
                {handoffNote.lines.map((ln, i) => (
                  <li key={i}>
                    {ln.text} {ln.target ? <Why to={ln.target} onWhy={onWhy} /> : null}
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
        </div>
      ) : null}

      <VoiceBar
        phase={thinking ? "thinking" : voice.status}
        note={note}
        onTalk={() => {
          setNote(null);
          conversing.current = true;
          void fresh().then(() => listen());
        }}
        onFinish={() => voice.stopListening(true)}
        onStop={() => {
          conversing.current = false;
          void fresh();
        }}
      />
      <form
        className={m.row}
        onSubmit={(e) => {
          e.preventDefault();
          if (text.trim()) ask(text);
        }}
      >
        <button type="button" className={`${ui.btn} ${m.more}`} aria-expanded={open} aria-controls="spot-mic-panel" aria-label={open ? "Hide the suggestions" : "Show the suggestions"} onClick={() => setOpen((x) => !x)}>
          <span aria-hidden="true">{open ? "▾" : "▴"}</span>
        </button>
        <input
          className={ui.input}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onFocus={() => (turn || text ? undefined : setOpen(true))}
          placeholder={voice.status === "listening" ? "Listening…" : "Ask in your own words"}
          aria-label="Question about the inspections"
          enterKeyHint="send"
          autoComplete="off"
          maxLength={500}
        />
        <button type="submit" className={`${ui.btn} ${ui.btnPrimary}`} disabled={!text.trim()}>
          Ask
        </button>
      </form>
    </div>
  );
}
