"use client";

import { useEffect, useState } from "react";
import { useVoice } from "@/lib/voice";
import v from "./voice.module.css";

export type VoicePhase = "idle" | "listening" | "thinking" | "speaking";

/** The one-line rule for approvals by voice. Shown wherever hands-free can be switched on. */
export const VOICE_APPROVAL_RULE = "By voice you can deny a request but never approve one: “yes” only opens Approvals, where you tap Approve.";

const WHY_SEEN_KEY = "atlas.voice.whySeen";

const ICON = { fill: "none", stroke: "currentColor", strokeWidth: 1.8, strokeLinecap: "round", strokeLinejoin: "round" } as const;

function MicIcon({ off = false }: { off?: boolean }) {
  return (
    <svg viewBox="0 0 24 24" {...ICON} aria-hidden="true">
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5.5 11.5a6.5 6.5 0 0013 0M12 18v3M8.5 21h7" />
      {off ? <path d="M4 3.5l16 17" /> : null}
    </svg>
  );
}

function StopIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      <rect x="6.5" y="6.5" width="11" height="11" rx="2" />
    </svg>
  );
}

function SpeakerIcon({ on }: { on: boolean }) {
  return (
    <svg viewBox="0 0 24 24" {...ICON} aria-hidden="true">
      <path d="M4 9.5h3.5L12 5.5v13l-4.5-4H4z" />
      {on ? <path d="M15.5 9a4.2 4.2 0 010 6M18 6.5a7.8 7.8 0 010 11" /> : <path d="M16 9.5l5 5M21 9.5l-5 5" />}
    </svg>
  );
}

function LoopIcon() {
  return (
    <svg viewBox="0 0 24 24" {...ICON} aria-hidden="true">
      <path d="M4.5 11a7.5 7.5 0 0113-4.5L19.5 8.5M19.5 4v4.5H15M19.5 13a7.5 7.5 0 01-13 4.5L4.5 15.5M4.5 20v-4.5H9" />
    </svg>
  );
}

interface Props {
  phase: VoicePhase;
  /** A line about the last thing that happened ("I did not hear anything"), shown while idle. */
  note: { text: string; tone: "info" | "error" } | null;
  /** The mic was tapped while idle and listening is possible. */
  onTalk: () => void;
  /** The mic was tapped while listening: send what was heard. */
  onFinish: () => void;
  /** The stop button was tapped while Atlas was speaking. */
  onStop: () => void;
}

/** The mic, the speaker and hands-free toggles, and one line saying what Atlas is doing. */
export function VoiceBar({ phase, note, onTalk, onFinish, onStop }: Props) {
  const voice = useVoice();
  const { settings, listenSupport, canSpeak, ready } = voice;
  const blocked = listenSupport && !listenSupport.ok ? listenSupport : null;
  const [why, setWhy] = useState(false);

  // The explanation shows itself once; after "Got it" it stays out of the way until the crossed mic is tapped.
  useEffect(() => {
    if (!blocked) return;
    try {
      setWhy(window.localStorage.getItem(WHY_SEEN_KEY) !== blocked.reason);
    } catch {
      setWhy(true);
    }
  }, [blocked]);

  const closeWhy = () => {
    setWhy(false);
    try {
      if (blocked) window.localStorage.setItem(WHY_SEEN_KEY, blocked.reason);
    } catch {
      /* nothing to do */
    }
  };

  const handsFree = settings.handsFree && !blocked;

  let status: string;
  let statusClass = "";
  if (phase === "listening") {
    status = handsFree ? "Listening… say “stop” to end" : "Listening… tap when you are done";
    statusClass = v.statusOn ?? "";
  } else if (phase === "thinking") {
    status = "Thinking…";
  } else if (phase === "speaking") {
    status = "Speaking… tap to stop";
    statusClass = v.statusOn ?? "";
  } else if (note) {
    status = note.text;
    statusClass = note.tone === "error" ? (v.statusError ?? "") : "";
  } else if (!ready) {
    status = "";
  } else if (blocked) {
    status = blocked.reason === "insecure" ? "Microphone off on this address. Type below." : "This browser cannot listen. Type below.";
  } else {
    status = handsFree ? "Hands-free: tap the mic once, then just talk" : "Tap the mic and ask out loud";
  }

  const mic =
    phase === "speaking"
      ? { label: "Stop speaking", cls: v.micSpeaking, icon: <StopIcon />, act: onStop, disabled: false }
      : phase === "listening"
        ? { label: "Done talking", cls: v.micListening, icon: <MicIcon />, act: onFinish, disabled: false }
        : phase === "thinking"
          ? { label: "Atlas is thinking", cls: v.micThinking, icon: <MicIcon />, act: () => undefined, disabled: true }
          : blocked
            ? { label: "Why the microphone is off", cls: v.micOff, icon: <MicIcon off />, act: () => setWhy((w) => !w), disabled: false }
            : { label: "Talk to Atlas", cls: "", icon: <MicIcon />, act: onTalk, disabled: !ready };

  return (
    <div className={v.bar}>
      {blocked && why ? (
        <div className={v.why} role="note">
          <p className={v.whyTitle}>{blocked.title}</p>
          <p>{blocked.detail}</p>
          <button type="button" className={v.whyClose} onClick={closeWhy}>
            Got it
          </button>
        </div>
      ) : null}

      <p className={`${v.status} ${statusClass}`} role="status" aria-live="polite">
        {status}
      </p>

      <div className={v.controls}>
        <button
          type="button"
          className={`${v.toggle} ${settings.speak && canSpeak ? v.toggleOn : ""}`}
          role="switch"
          aria-checked={settings.speak && canSpeak}
          disabled={ready && !canSpeak}
          title={ready && !canSpeak ? "This browser cannot speak" : "Read answers aloud"}
          onClick={() => voice.setSetting("speak", !settings.speak)}
        >
          <SpeakerIcon on={settings.speak && canSpeak} />
          <span>
            Speak answers
            <span className={v.toggleState}>{ready && !canSpeak ? "not available" : settings.speak ? "on" : "off"}</span>
          </span>
        </button>

        <button type="button" className={`${v.mic} ${mic.cls ?? ""}`} aria-label={mic.label} title={mic.label} disabled={mic.disabled} onClick={mic.act}>
          {mic.icon}
        </button>

        <button
          type="button"
          className={`${v.toggle} ${v.toggleRight} ${handsFree ? v.toggleOn : ""}`}
          role="switch"
          aria-checked={handsFree}
          disabled={blocked !== null}
          title={blocked ? "Needs the microphone" : "Keep listening after each answer"}
          onClick={() => voice.setSetting("handsFree", !settings.handsFree)}
        >
          <LoopIcon />
          <span>
            Hands-free
            <span className={v.toggleState}>{blocked ? "needs the mic" : handsFree ? "on" : "off"}</span>
          </span>
        </button>
      </div>

      {handsFree ? <p className={v.rule}>{VOICE_APPROVAL_RULE}</p> : null}
    </div>
  );
}
