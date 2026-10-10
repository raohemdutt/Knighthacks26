"use client";

/**
 * Atlas's voice, for the whole app: speaking (speechSynthesis) and listening
 * (SpeechRecognition), both the browser's own. Any screen can do
 *
 *   const { say } = useVoice();
 *   say("Your keys are on the desk.");
 *
 * What is guaranteed here, so no screen has to think about it:
 *  - nothing throws where a browser has no speech or no microphone;
 *  - Atlas never listens while it is speaking (listening cancels speech, and
 *    queued speech waits for listening to end), so it cannot hear itself;
 *  - speech is queued, never overlapped, and fed to the browser a sentence or
 *    two at a time because Chrome cuts long utterances off;
 *  - speech stops when the screen changes, and listening stops when the tab is hidden.
 *
 * The words themselves (what an answer or an alert sounds like) are in speechText.ts.
 */

import { usePathname } from "next/navigation";
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { useAtlasEvents } from "./atlas";
import { SPEAK_COOLDOWN_MS, chunkForSpeech, cleanForSpeech, sentenceForEvent, type SpokenKind } from "./speechText";

// ---------- the knobs, in one place ----------

/** How fast Atlas talks. 1 is the voice's normal speed. */
export const VOICE_RATE = 1.05;
export const VOICE_PITCH = 1;
/** Listening gives up after this long with nothing heard. */
export const SILENCE_MS = 8000;
/** At most this many unprompted sentences wait their turn; older ones are dropped. */
const MAX_QUEUED_EVENTS = 3;
const SETTINGS_KEY = "atlas.voice";
const SPOKEN_EVENTS = ["agent.alert.raised", "habits.nudge.raised", "workspace.approval.requested", "agent.plan.created", "day.coach.noted"];

export interface VoiceSettings {
  /** Read answers aloud. */
  speak: boolean;
  /** After Atlas finishes speaking, listen again without a tap. */
  handsFree: boolean;
  /** Say alerts, nudges, approval requests and the leaving checklist as they arrive. */
  proactive: boolean;
}

const DEFAULT_SETTINGS: VoiceSettings = { speak: true, handsFree: false, proactive: false };

export type VoiceStatus = "idle" | "listening" | "speaking";

export type ListenSupport =
  | { ok: true }
  | {
      ok: false;
      /** insecure: the page is on plain http. unsupported: the browser has no speech recognition. */
      reason: "insecure" | "unsupported";
      title: string;
      detail: string;
    };

export interface SayOptions {
  /** Called once: true when the whole text was spoken, false when it was stopped, replaced or could not be spoken. */
  onDone?: (finished: boolean) => void;
  /** Speech with the same tag can be replaced as a group. */
  tag?: string;
  /** Drop anything already queued or playing with this tag first. */
  replace?: boolean;
  /** Keep talking when the screen changes (by default speech stops on navigation). */
  keepOnNavigate?: boolean;
  /** The text is already fit to be spoken: skip the markdown/link/time cleaning. */
  raw?: boolean;
}

export interface ListenHandlers {
  /** Everything heard so far, as it changes. */
  onInterim?: (text: string) => void;
  /** The finished utterance. */
  onFinal: (text: string) => void;
  /** Listening ended with nothing to act on. */
  onEnd?: (why: "silence" | "stopped" | "error", message?: string) => void;
}

export interface Voice {
  /** False until the browser has been examined (always false during the static render). */
  ready: boolean;
  canSpeak: boolean;
  /** null until `ready`. */
  listenSupport: ListenSupport | null;
  status: VoiceStatus;
  /** Browsers only let a page speak after a tap or key press. True once that has happened. */
  unlocked: boolean;
  /** The voice in use, e.g. "Samantha (en-US)". */
  voiceName: string | null;
  settings: VoiceSettings;
  /** The settings as they are right now, for callbacks that outlive a render. */
  getSettings: () => VoiceSettings;
  setSetting: (key: keyof VoiceSettings, value: boolean) => void;
  /** Queue `text` to be spoken. Safe to call anywhere; does nothing where the browser cannot speak. */
  say: (text: string, options?: SayOptions) => void;
  /** Stop talking and forget everything queued. */
  stop: () => void;
  /** Start listening for one utterance. Stops any speech first. False when it could not start. */
  listen: (handlers: ListenHandlers) => boolean;
  /** Stop listening. `submit` hands what was heard so far to onFinal; otherwise it is thrown away. */
  stopListening: (submit?: boolean) => void;
}

// ---------- the browser's speech recognition, typed as far as it is used ----------

interface RecAlternative {
  transcript: string;
}
interface RecResult {
  isFinal: boolean;
  length: number;
  [index: number]: RecAlternative | undefined;
}
interface RecEvent {
  results: { length: number; [index: number]: RecResult | undefined };
}
interface Recognizer {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  onresult: ((e: RecEvent) => void) | null;
  onerror: ((e: { error?: string }) => void) | null;
  onend: (() => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
}
type RecognizerCtor = new () => Recognizer;

function recognizerCtor(): RecognizerCtor | null {
  const w = window as unknown as { SpeechRecognition?: RecognizerCtor; webkitSpeechRecognition?: RecognizerCtor };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

function synth(): SpeechSynthesis | null {
  try {
    return typeof window !== "undefined" && "speechSynthesis" in window && typeof SpeechSynthesisUtterance !== "undefined" ? window.speechSynthesis : null;
  } catch {
    return null;
  }
}

function detectListen(canSpeak: boolean): ListenSupport {
  const tail = canSpeak ? "Typing still works, and Atlas can still read answers aloud." : "Typing still works.";
  if (!window.isSecureContext) {
    const port = window.location.port ? `:${window.location.port}` : "";
    return {
      ok: false,
      reason: "insecure",
      title: "The microphone is off at this address",
      detail:
        `Browsers only allow the microphone on https:// pages and on localhost, and this page is ${window.location.protocol}//${window.location.host}. ` +
        `To talk to Atlas, open http://localhost${port} on the computer that runs the box, or serve the box over HTTPS. ${tail}`,
    };
  }
  if (!recognizerCtor()) {
    return {
      ok: false,
      reason: "unsupported",
      title: "This browser cannot listen",
      detail: `It has no speech recognition (Firefox does not). Open Atlas in Chrome, Edge or Safari to talk to it. ${tail}`,
    };
  }
  return { ok: true };
}

function listenError(code: string): string {
  switch (code) {
    case "not-allowed":
    case "service-not-allowed":
      return "The microphone is blocked for this page. Allow it in the browser's site settings (in Safari, also turn on Dictation), then tap the mic again.";
    case "audio-capture":
      return "No microphone was found. Plug one in or pick one in the browser's settings.";
    case "network":
      return "The browser could not reach its speech service. Chrome needs an internet connection to turn speech into text.";
    case "language-not-supported":
      return "The browser cannot recognise speech in this device's language.";
    default:
      return `Listening stopped unexpectedly (${code}). Tap the mic to try again.`;
  }
}

// ---------- choosing a voice ----------

/** macOS and iOS list joke and legacy voices under the same language as the good ones. */
const NOVELTY =
  /\b(Albert|Bad News|Bahh|Bells|Boing|Bubbles|Cellos|Deranged|Good News|Hysterical|Jester|Organ|Superstar|Trinoids|Whisper|Wobble|Zarvox|Fred|Junior|Kathy|Ralph|Princess|Eddy|Flo|Grandma|Grandpa|Reed|Rocko|Sandy|Shelley)\b/i;
const HIGH_QUALITY = /natural|neural|enhanced|premium|siri/i;
const KNOWN_GOOD = /^(Google (US|UK) English|Samantha|Ava|Allison|Susan|Karen|Daniel|Moira|Tessa|Microsoft (Aria|Jenny|Guy|Zira|Sonia))/i;

/** The best installed voice for `locale`, or null to leave the choice to the browser. */
export function pickVoice(voices: SpeechSynthesisVoice[], locale: string): SpeechSynthesisVoice | null {
  const want = locale.replace("_", "-").toLowerCase();
  const language = want.split("-")[0] ?? want;
  let best: SpeechSynthesisVoice | null = null;
  let bestScore = 0;
  for (const v of voices) {
    const lang = (v.lang || "").replace("_", "-").toLowerCase();
    if (lang.split("-")[0] !== language) continue;
    let score = lang === want ? 40 : 20;
    if (HIGH_QUALITY.test(v.name)) score += 12;
    if (KNOWN_GOOD.test(v.name)) score += 10;
    if (v.default) score += 5;
    if (v.localService) score += 2; // works with no network
    if (NOVELTY.test(v.name)) score -= 50;
    if (score > bestScore) {
      best = v;
      bestScore = score;
    }
  }
  return best;
}

// ---------- settings ----------

function loadSettings(): VoiceSettings {
  try {
    const raw: unknown = JSON.parse(window.localStorage.getItem(SETTINGS_KEY) ?? "{}");
    const o = raw && typeof raw === "object" ? (raw as Record<string, unknown>) : {};
    const flag = (k: keyof VoiceSettings) => (typeof o[k] === "boolean" ? (o[k] as boolean) : DEFAULT_SETTINGS[k]);
    return { speak: flag("speak"), handsFree: flag("handsFree"), proactive: flag("proactive") };
  } catch {
    return DEFAULT_SETTINGS; // private mode or blocked storage: the defaults, not remembered
  }
}

function saveSettings(s: VoiceSettings): void {
  try {
    window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(s));
  } catch {
    /* nothing to do */
  }
}

// ---------- the provider ----------

interface Job {
  chunks: string[];
  next: number;
  options: SayOptions;
}

const Ctx = createContext<Voice | null>(null);

function normalize(path: string): string {
  return path.length > 1 ? path.replace(/\/+$/, "") : path;
}

export function VoiceProvider({ children }: { children: ReactNode }) {
  const path = normalize(usePathname() || "/");
  const [ready, setReady] = useState(false);
  const [canSpeak, setCanSpeak] = useState(false);
  const [listenSupport, setListenSupport] = useState<ListenSupport | null>(null);
  const [status, setStatus] = useState<VoiceStatus>("idle");
  const [unlocked, setUnlocked] = useState(false);
  const [voiceName, setVoiceName] = useState<string | null>(null);
  const [settings, setSettings] = useState<VoiceSettings>(DEFAULT_SETTINGS);

  const settingsRef = useRef(settings);
  const pathRef = useRef(path);
  pathRef.current = path;
  const voiceRef = useRef<SpeechSynthesisVoice | null>(null);
  const queue = useRef<Job[]>([]);
  const current = useRef<Job | null>(null);
  const turn = useRef(0); // bumped whenever the utterance in the browser stops being ours to follow
  const utterance = useRef<SpeechSynthesisUtterance | null>(null); // held so Chrome does not drop its end event
  const watchdog = useRef<number | undefined>(undefined);
  const cancelledAt = useRef(0);
  const retry = useRef<number | undefined>(undefined);
  const pumpRef = useRef<() => void>(() => undefined);
  const recognizer = useRef<{ rec: Recognizer; discard: boolean } | null>(null);
  const silence = useRef<number | undefined>(undefined);
  const lastSpoken = useRef<Partial<Record<SpokenKind, number>>>({});
  const spokenIds = useRef(new Set<string>());

  // --- speaking ---

  const pump = useCallback(() => {
    if (current.current || recognizer.current) return; // one thing at a time, and never over the microphone
    const sinceCancel = Date.now() - cancelledAt.current;
    if (sinceCancel < 120 && queue.current.length > 0) {
      // Chrome and Safari drop an utterance handed over in the same breath as cancel()
      window.clearTimeout(retry.current);
      retry.current = window.setTimeout(() => pumpRef.current(), 130 - sinceCancel);
      return;
    }
    const job = queue.current.shift();
    if (!job) {
      setStatus((s) => (s === "speaking" ? "idle" : s));
      return;
    }
    current.current = job;
    setStatus("speaking");
    speakNext(job);
    // speakNext and finish only touch refs and state setters, so the first render's copies stay correct
  }, []);
  pumpRef.current = pump;

  const finish = useCallback(
    (job: Job, finished: boolean) => {
      window.clearTimeout(watchdog.current);
      utterance.current = null;
      if (current.current === job) current.current = null;
      try {
        job.options.onDone?.(finished);
      } catch {
        /* a caller's callback must not break the queue */
      }
      pump();
    },
    [pump],
  );

  function speakNext(job: Job) {
    const engine = synth();
    const text = job.chunks[job.next];
    if (!engine || text === undefined) {
      finish(job, engine !== null);
      return;
    }
    const mine = ++turn.current;
    const advance = () => {
      if (turn.current !== mine) return;
      window.clearTimeout(watchdog.current);
      job.next += 1;
      speakNext(job);
    };
    try {
      const u = new SpeechSynthesisUtterance(text);
      const voice = voiceRef.current;
      if (voice) {
        u.voice = voice;
        u.lang = voice.lang;
      } else {
        u.lang = navigator.language || "en-US";
      }
      u.rate = VOICE_RATE;
      u.pitch = VOICE_PITCH;
      u.onend = advance;
      u.onerror = (e) => {
        if (turn.current !== mine) return;
        if (e.error === "not-allowed") setUnlocked(false); // no tap yet: the browser refused
        if (e.error === "not-allowed" || e.error === "interrupted" || e.error === "canceled") finish(job, false);
        else advance(); // one bad sentence should not lose the rest
      };
      utterance.current = u;
      // Chrome sometimes never reports the end of an utterance: move on once it must be over.
      watchdog.current = window.setTimeout(advance, 6000 + text.length * 120);
      if (engine.paused) engine.resume();
      engine.speak(u);
    } catch {
      finish(job, false);
    }
  }

  /** Drop the queued and playing speech that matches. Their callers hear `false`. */
  const cancelWhere = useCallback(
    (match: (job: Job) => boolean) => {
      const dropped = queue.current.filter(match);
      queue.current = queue.current.filter((j) => !match(j));
      const playing = current.current;
      if (playing && match(playing)) {
        turn.current += 1;
        window.clearTimeout(watchdog.current);
        current.current = null;
        utterance.current = null;
        cancelledAt.current = Date.now();
        try {
          synth()?.cancel();
        } catch {
          /* nothing to do */
        }
        dropped.unshift(playing);
      }
      for (const job of dropped) {
        try {
          job.options.onDone?.(false);
        } catch {
          /* as above */
        }
      }
      pump();
    },
    [pump],
  );

  const stop = useCallback(() => cancelWhere(() => true), [cancelWhere]);

  const say = useCallback(
    (text: string, options: SayOptions = {}) => {
      const chunks = chunkForSpeech(options.raw ? text : cleanForSpeech(text));
      if (!synth() || chunks.length === 0) {
        if (options.onDone) window.setTimeout(() => options.onDone?.(false), 0);
        return;
      }
      if (options.replace && options.tag) cancelWhere((j) => j.options.tag === options.tag);
      queue.current.push({ chunks, next: 0, options });
      pump();
    },
    [cancelWhere, pump],
  );

  // --- listening ---

  const stopListening = useCallback((submit = false) => {
    const active = recognizer.current;
    if (!active) return;
    try {
      if (submit) active.rec.stop();
      else {
        active.discard = true;
        active.rec.abort();
      }
    } catch {
      /* already stopped */
    }
  }, []);

  const listen = useCallback(
    (handlers: ListenHandlers): boolean => {
      const Ctor = typeof window !== "undefined" && window.isSecureContext ? recognizerCtor() : null;
      if (!Ctor) return false;
      stopListening();
      let rec: Recognizer;
      try {
        rec = new Ctor();
      } catch {
        return false;
      }
      const active = { rec, discard: false };
      recognizer.current = active; // from here on nothing new starts speaking
      // Never listen while speaking: what is playing stops, and so does any answer still queued.
      // Unprompted sentences that have not started yet wait until the microphone closes.
      const playing = current.current;
      cancelWhere((j) => j === playing || j.options.tag !== "event");
      let finalText = "";
      let interim = "";
      let error: string | null = null;
      const quietTimer = () => {
        window.clearTimeout(silence.current);
        silence.current = window.setTimeout(() => {
          try {
            rec.stop(); // ends with whatever was heard, or with nothing: the caller is told "silence"
          } catch {
            /* already stopped */
          }
        }, SILENCE_MS);
      };
      rec.lang = navigator.language || "en-US";
      rec.continuous = false;
      rec.interimResults = true;
      rec.maxAlternatives = 1;
      rec.onresult = (e) => {
        let f = "";
        let i = "";
        for (let k = 0; k < e.results.length; k++) {
          const r = e.results[k];
          const said = r?.[0]?.transcript ?? "";
          if (r?.isFinal) f += said;
          else i += said;
        }
        finalText = f.trim();
        interim = i.trim();
        if (!active.discard) handlers.onInterim?.(`${finalText} ${interim}`.trim());
        quietTimer();
      };
      rec.onerror = (e) => {
        error = e.error ?? "unknown";
      };
      rec.onend = () => {
        rec.onresult = rec.onerror = rec.onend = null;
        if (recognizer.current === active) {
          recognizer.current = null;
          window.clearTimeout(silence.current);
          setStatus((s) => (s === "listening" ? "idle" : s));
        }
        const heard = finalText || interim; // Safari can end without marking the last words final
        try {
          if (active.discard) handlers.onEnd?.("stopped");
          else if (heard) handlers.onFinal(heard);
          else if (error && error !== "no-speech" && error !== "aborted") handlers.onEnd?.("error", listenError(error));
          else handlers.onEnd?.(error === "aborted" ? "stopped" : "silence");
        } catch {
          /* a caller's callback must not break listening */
        }
        pump(); // anything that waited for the microphone to close
      };
      try {
        rec.start();
      } catch {
        rec.onresult = rec.onerror = rec.onend = null;
        if (recognizer.current === active) recognizer.current = null;
        pump();
        return false;
      }
      setStatus("listening");
      quietTimer();
      return true;
    },
    [pump, cancelWhere, stopListening],
  );

  // --- set-up: what this browser can do, the remembered toggles, the voices ---

  useEffect(() => {
    const engine = synth();
    setCanSpeak(engine !== null);
    setListenSupport(detectListen(engine !== null));
    const loaded = loadSettings();
    settingsRef.current = loaded;
    setSettings(loaded);
    setReady(true);
    if (!engine) return;

    // Voices arrive late (and on Chrome only after this event), so choose again whenever the list changes.
    const choose = () => {
      try {
        const v = pickVoice(engine.getVoices(), navigator.language || "en-US");
        voiceRef.current = v;
        setVoiceName(v ? `${v.name} (${v.lang})` : null);
      } catch {
        /* keep the browser's default voice */
      }
    };
    choose();
    const canListen = typeof engine.addEventListener === "function";
    if (canListen) engine.addEventListener("voiceschanged", choose);
    else engine.onvoiceschanged = choose;

    // The first tap or key press anywhere lets the page speak from then on (iOS and Chrome both insist on it).
    const unlock = () => {
      window.removeEventListener("click", unlock, true);
      window.removeEventListener("touchend", unlock, true);
      window.removeEventListener("keydown", unlock, true);
      setUnlocked(true);
      try {
        if (!engine.speaking && !engine.pending) {
          const silent = new SpeechSynthesisUtterance(" ");
          silent.volume = 0;
          engine.speak(silent);
        }
      } catch {
        /* nothing to do */
      }
    };
    window.addEventListener("click", unlock, true);
    window.addEventListener("touchend", unlock, true);
    window.addEventListener("keydown", unlock, true);

    const onLeave = () => {
      try {
        engine.cancel(); // Chrome otherwise keeps talking over the next page
      } catch {
        /* nothing to do */
      }
    };
    window.addEventListener("pagehide", onLeave);
    return () => {
      if (canListen) engine.removeEventListener("voiceschanged", choose);
      else engine.onvoiceschanged = null;
      window.removeEventListener("click", unlock, true);
      window.removeEventListener("touchend", unlock, true);
      window.removeEventListener("keydown", unlock, true);
      window.removeEventListener("pagehide", onLeave);
      onLeave();
    };
  }, []);

  // A hidden tab must not keep the microphone open.
  useEffect(() => {
    const onVisibility = () => {
      if (document.visibilityState === "hidden") stopListening();
    };
    document.addEventListener("visibilitychange", onVisibility);
    return () => document.removeEventListener("visibilitychange", onVisibility);
  }, [stopListening]);

  // Changing screens ends what was being said about the old one, and closes the microphone.
  const firstPath = useRef(true);
  useEffect(() => {
    if (firstPath.current) {
      firstPath.current = false;
      return;
    }
    stopListening();
    cancelWhere((j) => !j.options.keepOnNavigate);
  }, [path, cancelWhere, stopListening]);

  // --- speaking up on its own ---

  useAtlasEvents(SPOKEN_EVENTS, (e) => {
    const s = settingsRef.current;
    if (!s.proactive || !synth()) return;
    const sentence = sentenceForEvent(e);
    if (!sentence) return;
    // On the Ask screen in hands-free mode the request is asked as a question instead (see app/page.tsx).
    if (sentence.kind === "approval" && s.handsFree && s.speak && pathRef.current === "/") return;
    if (spokenIds.current.has(e.event_id)) return;
    const now = Date.now();
    const last = lastSpoken.current[sentence.kind];
    if (last !== undefined && now - last < SPEAK_COOLDOWN_MS[sentence.kind]) return; // shown as a toast as usual, just not said
    lastSpoken.current[sentence.kind] = now;
    spokenIds.current.add(e.event_id);
    const waiting = queue.current.filter((j) => j.options.tag === "event");
    if (waiting.length >= MAX_QUEUED_EVENTS) cancelWhere((j) => j === waiting[0]);
    say(sentence.text, { tag: "event", keepOnNavigate: true, raw: true });
  });

  const getSettings = useCallback(() => settingsRef.current, []);

  const setSetting = useCallback(
    (key: keyof VoiceSettings, value: boolean) => {
      const next = { ...settingsRef.current, [key]: value };
      settingsRef.current = next;
      setSettings(next);
      saveSettings(next);
      if (key === "speak" && !value) cancelWhere((j) => j.options.tag !== "event");
      if (key === "proactive" && !value) cancelWhere((j) => j.options.tag === "event");
    },
    [cancelWhere],
  );

  const value = useMemo<Voice>(
    () => ({ ready, canSpeak, listenSupport, status, unlocked, voiceName, settings, getSettings, setSetting, say, stop, listen, stopListening }),
    [ready, canSpeak, listenSupport, status, unlocked, voiceName, settings, getSettings, setSetting, say, stop, listen, stopListening],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useVoice(): Voice {
  const v = useContext(Ctx);
  if (!v) throw new Error("useVoice must be used inside <VoiceProvider>");
  return v;
}
