"use client";

import ui from "@/components/ui.module.css";
import { VOICE_RATE, useVoice, type VoiceSettings as Flags } from "@/lib/voice";
import { VOICE_APPROVAL_RULE } from "./VoiceBar";
import v from "./voice.module.css";

function Switch({ label, hint, on, disabled, onChange }: { label: string; hint: string; on: boolean; disabled?: boolean; onChange: (on: boolean) => void }) {
  return (
    <button type="button" className={v.switchRow} role="switch" aria-checked={on} disabled={disabled} onClick={() => onChange(!on)}>
      <span className={v.switchText}>
        <span className={v.switchLabel}>{label}</span>
        <span className={v.switchHint}>{hint}</span>
      </span>
      <span className={`${v.knob} ${on ? v.knobOn : ""}`} aria-hidden="true" />
    </button>
  );
}

/** The Voice part of the Settings sheet: the three switches, and what they do. */
export function VoiceSettings() {
  const voice = useVoice();
  const { settings, canSpeak, listenSupport, ready } = voice;
  const blocked = listenSupport && !listenSupport.ok ? listenSupport : null;
  const set = (key: keyof Flags) => (on: boolean) => voice.setSetting(key, on);
  const noVoice = ready && !canSpeak;

  return (
    <section className={ui.section} aria-label="Voice">
      <h3>Voice</h3>
      <Switch
        label="Speak answers"
        hint={noVoice ? "This browser cannot speak." : "Atlas reads each answer aloud."}
        on={settings.speak && canSpeak}
        disabled={noVoice}
        onChange={set("speak")}
      />
      <Switch
        label="Hands-free conversation"
        hint={blocked ? blocked.title + "." : "After it answers, Atlas listens again. Say “stop” to end."}
        on={settings.handsFree && !blocked}
        disabled={blocked !== null}
        onChange={set("handsFree")}
      />
      <Switch
        label="Speak up on its own"
        hint={noVoice ? "This browser cannot speak." : "Says alerts, nudges, approval requests, the leaving checklist and your day (the morning brief, a re-plan, the coach, when to leave, the evening wrap) while this app is open."}
        on={settings.proactive && canSpeak}
        disabled={noVoice}
        onChange={set("proactive")}
      />
      <p className={`${ui.muted} ${ui.small}`}>{VOICE_APPROVAL_RULE}</p>
      {blocked ? <p className={`${ui.muted} ${ui.small} ${ui.breakWords}`}>{blocked.detail}</p> : null}
      <p className={`${ui.muted} ${ui.small}`}>
        {canSpeak ? `Voice: ${voice.voiceName ?? "the browser's default"}, at ${VOICE_RATE}× speed. ` : ""}
        Listening uses the browser&apos;s own speech recognition, which in Chrome and Safari can send the audio to Google or Apple to turn it
        into text. Some browser voices are made online too.
      </p>
    </section>
  );
}
