"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError, type VoiceStatus } from "@/lib/api";
import op from "./operator.module.css";

/**
 * Voice on the box (X16): the Jetson's own microphone (the webcam's) and speaker (Bluetooth). Push to talk:
 * tap, ask, and the answer is spoken on the box. The state is read from /api/voice/status about once a second;
 * a box without the voice service reads "Off". The audio never reaches this page.
 */

const POLL_MS = 800;

const WORDS: Record<VoiceStatus["state"], string> = {
  off: "Off",
  idle: "Ready",
  listening: "Listening…",
  thinking: "Thinking…",
  speaking: "Speaking…",
};

const OFF: VoiceStatus = { state: "off", running: false, problem: "", heard: "", answer: "", seconds_in_state: 0 };

export function VoicePanel() {
  const [status, setStatus] = useState<VoiceStatus>(OFF);
  const [error, setError] = useState("");
  const busy = useRef(false);

  const refresh = useCallback(async () => {
    try {
      setStatus(await api.voiceStatus());
      setError("");
    } catch (e) {
      setStatus(OFF);
      setError(e instanceof ApiError ? e.message : "Can't reach the box.");
    }
  }, []);

  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => void refresh(), POLL_MS);
    return () => window.clearInterval(id);
  }, [refresh]);

  const press = useCallback(
    async (action: () => Promise<VoiceStatus>) => {
      if (busy.current) return;
      busy.current = true;
      try {
        setStatus(await action());
        setError("");
      } catch (e) {
        setError(e instanceof ApiError ? e.message : "Couldn't reach the voice service.");
      } finally {
        busy.current = false;
      }
    },
    [],
  );

  const { state } = status;
  const trouble = [status.mic, status.speaker, status.speech_to_text, status.text_to_speech].find((d) => d && !d.ok);
  const note = error || status.problem || (trouble ? trouble.detail : "");

  return (
    <div className={`${op.panel} ${op.voice} ${op[`voice_${state}`]}`}>
      <h2>Voice on the box</h2>
      <p className={op.big} role="status" aria-live="polite">
        <span className={op.voiceDot} aria-hidden="true" />
        {WORDS[state]}
      </p>
      <div className={op.voiceButtons}>
        {state === "idle" ? (
          <button type="button" className={op.voicePrimary} onClick={() => void press(api.voiceTalk)}>
            Tap to talk
          </button>
        ) : null}
        {state === "listening" ? (
          <>
            <button type="button" className={op.voicePrimary} onClick={() => void press(api.voiceTalk)}>
              Done talking
            </button>
            <button type="button" className={op.voiceSecondary} onClick={() => void press(api.voiceStop)}>
              Cancel
            </button>
          </>
        ) : null}
        {state === "thinking" ? (
          <button type="button" className={op.voiceSecondary} onClick={() => void press(api.voiceStop)}>
            Cancel
          </button>
        ) : null}
        {state === "speaking" ? (
          <button type="button" className={op.voicePrimary} onClick={() => void press(api.voiceTalk)}>
            Stop speaking
          </button>
        ) : null}
        {state === "off" && status.running ? (
          <button type="button" className={op.voicePrimary} onClick={() => void press(() => api.voiceEnabled(true))}>
            Turn voice on
          </button>
        ) : null}
        {state === "idle" ? (
          <button type="button" className={op.voiceSecondary} onClick={() => void press(() => api.voiceEnabled(false))}>
            Turn off
          </button>
        ) : null}
      </div>
      {status.heard ? <p className={op.muted}>Heard: “{status.heard}”</p> : null}
      {status.answer ? <p className={op.voiceAnswer}>{status.answer.length > 160 ? `${status.answer.slice(0, 159)}…` : status.answer}</p> : null}
      {note ? <p className={op.voiceNote}>{note}</p> : null}
    </div>
  );
}
