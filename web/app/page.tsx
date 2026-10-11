"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import { AskTool, OPENS_SCREEN } from "@/components/AskTool";
import { ASK_EVENT } from "@/components/Command";
import { Evidence } from "@/components/Evidence";
import { Logo } from "@/components/Logo";
import ui from "@/components/ui.module.css";
import { VoiceBar, type VoicePhase } from "@/components/voice/VoiceBar";
import v from "@/components/voice/voice.module.css";
import { ApiError, api, type ApprovalRequest } from "@/lib/api";
import { useAnswerStream, useAtlas } from "@/lib/atlas";
import { time } from "@/lib/format";
import { useProfile } from "@/lib/profile";
import { useBrief } from "@/lib/brief";
import { speakOnBox } from "@/lib/boxSpeech";
import { askGuide } from "@/lib/guideBus";
import { approvalSentence, narratedAsk, speechForAnswer, voiceCommand } from "@/lib/speechText";
import { useVoice } from "@/lib/voice";

/** What the box says when it looked and found nothing on record: the one answer with no sources that is not talk. */
const NO_EVIDENCE = "I didn't see that.";

let turnSeq = 0;
const nextId = () => `t${Date.now()}-${turnSeq++}`;

/** Approval requests Atlas has already asked about aloud, so coming back to this screen does not ask again. */
const askedAloud = new Set<string>();
/** The pause between Atlas's last word and the microphone opening, so it does not hear its own echo. */
const RELISTEN_MS = 350;

type Note = { text: string; tone: "info" | "error" };

/** An answer's words, finished or still being written: one place, so streamed text looks exactly like the final text. */
function AnswerText({ text, writing = false }: { text: string; writing?: boolean }) {
  return (
    <p className={ui.answerText}>
      {text}
      {writing ? <span className={v.caret} aria-hidden="true" /> : null}
    </p>
  );
}

export default function AskPage() {
  const { chat, setChat, approvals } = useAtlas();
  // The profile's words, and its suggested questions: things to ask the box, and a few ways in to just talking with it
  // (contexts/agent/application/talk.py). The student's until the box answers.
  const { words: w, suggestions } = useProfile();
  const voice = useVoice();
  const stream = useAnswerStream();
  const streamed = stream.text; // the answer so far, while it is being written
  const router = useRouter();
  const [text, setText] = useState("");
  const [waiting, setWaiting] = useState<string | null>(null); // the question being answered
  const [note, setNote] = useState<Note | null>(null); // what the voice last did, shown above the mic
  const [boxMic, setBoxMic] = useState(false); // the box has a working microphone and speech-to-text: the mic button uses it, not this computer's
  const [boxPhase, setBoxPhase] = useState<VoicePhase | null>(null); // what the box's microphone is doing while a turn is under way
  const endRef = useRef<HTMLDivElement>(null);
  const busy = useRef(false); // `waiting`, for callbacks that outlive a render
  const alive = useRef(true);
  const conversing = useRef(false); // a spoken exchange is under way; hands-free keeps it going
  const pendingApproval = useRef<ApprovalRequest | null>(null); // Atlas just asked about this one and is waiting for yes or no
  const startListeningRef = useRef<() => void>(() => undefined);

  useEffect(() => {
    if (chat.length > 0 || waiting) endRef.current?.scrollIntoView({ block: "end", behavior: streamed ? "auto" : "smooth" });
  }, [chat.length, waiting, streamed]);

  /** Open the microphone again once Atlas has finished, if the conversation is hands-free and still wanted. */
  const relisten = useCallback(() => {
    const wanted = () => alive.current && conversing.current && voice.getSettings().handsFree && document.visibilityState === "visible";
    if (!wanted()) {
      conversing.current = false;
      pendingApproval.current = null;
      return;
    }
    window.setTimeout(() => {
      if (wanted() && !busy.current) startListeningRef.current();
    }, RELISTEN_MS);
  }, [voice]);

  /** Say `reply` if the speaker is on, then carry the conversation on. It comes out of the box's speaker when that works, and out of this browser when it does not. */
  const respond = (reply: string | null) => {
    if (!alive.current) return; // the answer is in the conversation; nobody is on this screen to hear it
    if (!reply || !voice.getSettings().speak) {
      relisten();
      return;
    }
    void speakOnBox(reply).then((onBox) => {
      if (!alive.current) return;
      if (onBox) {
        conversing.current = false; // the box is talking and this screen cannot tell when it stops: the microphone is not reopened
      } else if (voice.canSpeak) {
        voice.say(reply, {
          tag: "answer",
          replace: true,
          raw: true,
          onDone: (finished) => {
            if (finished) relisten();
            else conversing.current = false; // stopped by hand or cut off: do not open the microphone uninvited
          },
        });
      } else relisten();
    });
  };

  const ask = async (question: string, spoken = false) => {
    const q = question.trim();
    if (!q || busy.current) return;
    if (askGuide(q, "/")) return setText(""); // "where do I find the League?": the look guide points, no question to the box
    busy.current = true;
    let reply: string | null = null;
    setNote(null);
    setText("");
    setWaiting(q);
    setChat((turns) => [...turns, { id: nextId(), role: "you", text: q }]);
    stream.begin(q); // pieces of the answer show as they are written; the reply below replaces them
    try {
      const answer = await api.ask(q);
      setChat((turns) => [...turns, { id: nextId(), role: "atlas", answer }]);
      // "Show me the room map", "open approvals": the tool named a screen, so the Ask tab opens it after the answer shows
      const open = answer.tool && answer.link && OPENS_SCREEN.has(answer.tool) ? answer.link : null;
      if (open) window.setTimeout(() => router.push(open), 1500);
      // as the box's voice counts them (contexts/voice/service.py evidence_count): only what the cameras saw is announced as
      // evidence; a calendar entry or a link behind "You've got your lecture at 10:40" is not
      reply = speechForAnswer(answer.answer, answer.citations.filter((c) => c.frame_ref || c.episode_id).length);
    } catch (err) {
      const e = err instanceof ApiError ? err : new ApiError(0, String(err));
      const message =
        e.status === 504
          ? "The model took too long to answer. Try again in a moment."
          : e.needsToken
            ? "This box asks for a token before it answers. Add it under Settings, top right."
            : `${e.message}${e.status ? ` (HTTP ${e.status})` : ""}`;
      setChat((turns) => [...turns, { id: nextId(), role: "error", text: message, needsToken: e.needsToken }]);
      if (spoken) reply = e.status === 504 ? "That took too long. Try again in a moment." : "Sorry, I could not get an answer. The reason is on screen.";
    } finally {
      stream.end();
      busy.current = false;
      setWaiting(null);
    }
    respond(reply);
  };

  /** "No" to the request Atlas just asked about. Denying is the only decision the voice can make. */
  const denyByVoice = async (request: ApprovalRequest) => {
    try {
      await api.decide(request.id, "deny");
      approvals.reload();
      setNote({ text: `Denied by voice: ${request.reason}`, tone: "info" });
      respond("Denied.");
    } catch (err) {
      const e = err instanceof ApiError ? err : new ApiError(0, String(err));
      if (e.status === 404) approvals.reload(); // decided or expired elsewhere
      setNote({ text: e.needsToken ? "This box asks for a token before it accepts a decision." : `Could not deny it: ${e.message}`, tone: "error" });
      respond("I could not deny it. It is still waiting on the Approvals screen.");
    }
  };

  const heard = (said: string) => {
    const request = pendingApproval.current;
    pendingApproval.current = null;
    const command = voiceCommand(said, request ? "yes-no" : "question");
    if (request && command === "yes") {
      // Never approved by voice: a stray "yes" in the room must not send, write or export anything.
      conversing.current = false;
      setText("");
      router.push("/approvals/");
      voice.say("Opening Approvals. Tap Approve to confirm.", { tag: "answer", replace: true, raw: true, keepOnNavigate: true });
      return;
    }
    if (request && command === "no") {
      setText("");
      void denyByVoice(request);
      return;
    }
    if (command === "stop") {
      conversing.current = false;
      setText("");
      setNote({ text: "Stopped listening. Tap the mic to talk again.", tone: "info" });
      return;
    }
    // "brief me", "what's my day", "what's next", "how's it going": questions like any other. The box answers them
    // (/api/ask), so the phone and the box's voice say the same thing.
    void ask(said, true);
  };

  const startListening = () => {
    const started = voice.listen({
      onInterim: setText,
      onFinal: heard,
      onEnd: (why, message) => {
        pendingApproval.current = null;
        setText("");
        if (why === "stopped") return; // closed on purpose: the screen changed, the tab was hidden, or a new turn began
        conversing.current = false;
        setNote(
          why === "error"
            ? { text: message ?? "Listening failed. Tap the mic to try again.", tone: "error" }
            : { text: "I did not hear anything, so I stopped listening.", tone: "info" },
        );
      },
    });
    if (!started) {
      conversing.current = false;
      setNote({ text: "The microphone could not be started. Type your question instead.", tone: "error" });
    }
  };
  startListeningRef.current = startListening;

  // The box's own microphone (the webcam on the Jetson) when it works; this computer's only when it does not.
  useEffect(() => {
    const check = () =>
      api
        .voiceStatus()
        .then((b) => setBoxMic(b.running && b.state !== "off" && b.mic?.ok === true && b.speech_to_text?.ok === true))
        .catch(() => setBoxMic(false));
    void check();
    const t = window.setInterval(check, 15000);
    return () => window.clearInterval(t);
  }, []);

  /** One question through the box's microphone: it listens and writes down what it heard, and this screen asks and shows the answer. */
  const listenOnBox = async () => {
    setNote(null);
    try {
      let status = await api.voiceListen();
      if (status.state !== "listening") {
        setNote({ text: status.problem || "The box is busy. Tap the mic again in a moment.", tone: "info" });
        return;
      }
      for (let i = 0; i < 150 && alive.current; i++) {
        setBoxPhase(status.state === "thinking" ? "thinking" : "listening");
        await new Promise((r) => window.setTimeout(r, 600));
        status = await api.voiceStatus();
        if (status.state !== "listening" && status.state !== "thinking") break;
      }
      setBoxPhase(null);
      if (!alive.current) return;
      if (status.heard) heard(status.heard);
      else setNote({ text: status.problem || "I did not hear anything.", tone: "info" });
    } catch (err) {
      setBoxPhase(null);
      const e = err instanceof ApiError ? err : new ApiError(0, String(err));
      setNote({ text: e.needsToken ? "This box asks for a token before it listens. Add it under Settings." : `The box's microphone could not be used: ${e.message}`, tone: "error" });
    }
  };

  const onTalk = () => {
    if (boxMic) {
      void listenOnBox();
      return;
    }
    setNote(null);
    conversing.current = true;
    startListening();
  };

  const onStopSpeaking = () => {
    conversing.current = false;
    pendingApproval.current = null;
    voice.stop();
  };

  // Leaving the screen or hiding the tab ends the conversation.
  useEffect(() => {
    alive.current = true;
    const onVisibility = () => {
      if (document.visibilityState === "hidden") conversing.current = false; // the provider closes the microphone itself
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      alive.current = false;
      conversing.current = false;
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, []);

  // A question handed over by the palette: in the link (/?q=...) when it came from another screen, or as an event
  // when the palette was opened here. Either way it is asked at once, as if it had been typed.
  const askRef = useRef(ask);
  askRef.current = ask;
  useEffect(() => {
    const q = new URLSearchParams(window.location.search).get("q");
    if (q) {
      window.history.replaceState(null, "", window.location.pathname);
      void askRef.current(q);
    }
    const onAsk = (e: Event) => void askRef.current(String((e as CustomEvent<string>).detail ?? ""));
    window.addEventListener(ASK_EVENT, onAsk);
    return () => window.removeEventListener(ASK_EVENT, onAsk);
  }, []);

  // So does switching hands-free off.
  const handsFree = voice.settings.handsFree;
  useEffect(() => {
    if (!handsFree) {
      conversing.current = false;
      pendingApproval.current = null;
    }
  }, [handsFree]);

  // One request waiting and hands-free on: say what Atlas wants and ask. Several waiting are left to the Approvals screen.
  const only = approvals.items.length === 1 && !approvals.loading ? approvals.items[0] : undefined;
  const canAskAloud =
    voice.ready &&
    handsFree &&
    voice.settings.speak &&
    voice.canSpeak &&
    voice.unlocked &&
    voice.listenSupport?.ok === true &&
    voice.status === "idle" &&
    waiting === null;
  const { say } = voice;
  const { asks } = useBrief();
  useEffect(() => {
    if (!only || !canAskAloud || askedAloud.has(only.id) || document.visibilityState !== "visible") return;
    askedAloud.add(only.id);
    pendingApproval.current = only;
    conversing.current = true;
    // the narrator's plain ask when it has one ("I'd like to add eight blocks to your calendar for tomorrow. Okay?")
    const asking = narratedAsk(asks[only.id]) ?? approvalSentence(only.reason);
    say(`${asking} Say no to deny it, or yes to open the Approvals screen.`, {
      tag: "answer",
      raw: true,
      onDone: (finished) => {
        if (finished) relisten();
        else pendingApproval.current = null;
      },
    });
  }, [only, canAskAloud, say, relisten, asks]);

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    void ask(text);
  };

  return (
    <>
      <div className={ui.watermark} aria-hidden="true">
        <Logo size={420} />
      </div>
      {chat.length === 0 ? (
        <div className={ui.pageHead}>
          <h1>Ask Atlas</h1>
          <p className={ui.muted}>
            Ask about your {w.workplace}, your {w.event_plural} or your evening. Answers show the camera frames and links they come from.
          </p>
        </div>
      ) : null}

      <div className={ui.turns} aria-live="polite" aria-busy={waiting !== null}>
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
          // Conversation ("how's your day?") cites nothing because nothing was looked up: no empty evidence block for it,
          // only a quiet note. An answer that looked and found nothing still says so.
          const talk = a.citations.length === 0 && a.answer.trim() !== NO_EVIDENCE && !a.tool;
          return (
            <article key={turn.id} className={`${ui.card} ${ui.answer}`} data-guide="answer">
              <AnswerText text={a.answer} />
              <AskTool answer={a} />
              {talk || (a.tool && a.citations.length === 0) ? null : <Evidence citations={a.citations} />}
              <p className={ui.modelLine}>
                {a.answered_by ? <>Answered by {a.answered_by} </> : <>Answered by </>}
                <strong>{a.model_version}</strong> at {time(a.ts)}
                {talk ? " · no sources attached" : null}
              </p>
            </article>
          );
        })}
        {waiting && streamed ? (
          <article className={`${ui.card} ${ui.answer}`} data-writing="true">
            <AnswerText text={streamed} writing />
          </article>
        ) : waiting ? (
          <div className={ui.loading} role="status">
            <span className={ui.spinner} aria-hidden="true" />
            <span>Atlas is thinking…</span>
          </div>
        ) : null}
        <div ref={endRef} />
      </div>

      <section className={ui.section} aria-label="Suggested questions">
        {chat.length === 0 ? <p className={ui.sectionTitle}>Try one</p> : null}
        <div className={ui.chips}>
          {suggestions.map((q) => (
            <button key={q} type="button" className={ui.chip} disabled={waiting !== null} onClick={() => void ask(q)}>
              {q}
            </button>
          ))}
        </div>
      </section>

      <div className={v.dock}>
        <VoiceBar
          phase={waiting ? "thinking" : (boxPhase ?? voice.status)}
          note={note}
          onTalk={onTalk}
          onFinish={() => (boxPhase === "listening" ? void api.voiceTalk() : voice.stopListening(true))}
          onStop={onStopSpeaking}
        />
        <form className={v.typeRow} onSubmit={onSubmit}>
          <input
            className={ui.input}
            data-guide="ask:input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder={voice.status === "listening" ? "Listening…" : "Ask anything"}
            aria-label="Your question"
            enterKeyHint="send"
            autoComplete="off"
          />
          <button type="submit" className={`${ui.btn} ${ui.btnPrimary}`} disabled={waiting !== null || text.trim() === ""}>
            Ask
          </button>
        </form>
      </div>
    </>
  );
}
