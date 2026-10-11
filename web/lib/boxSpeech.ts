/**
 * Saying something on the box's own speaker (POST /api/voice/say), with the browser's voice as the fallback.
 *
 *   speakOnBox(text)            one sentence or answer: true when the box took it, false when the caller should say it itself
 *   boxLines.say(text, else)    several lines in order (the look guide on the Inspection screen)
 *   boxLines.cancel()           drop the lines still waiting, and stop the one being spoken
 *
 * The box speaks one thing at a time and only while it is idle: a line sent while it is busy is dropped, and the call
 * returns before the speech ends. So `boxLines` sends one line, waits until the box is idle again, then sends the next.
 * A line the box cannot take (no voice service, no speaker, switched off, never idle) goes to `otherwise`: never both.
 */

import { api, type VoiceStatus } from "./api";

const canSay = (box: VoiceStatus): boolean => Boolean(box.running && box.state !== "off" && box.speaker?.ok && box.text_to_speech?.ok);

/** Say it on the box's own speaker, if the box has one working. False means nothing was said there. */
export async function speakOnBox(text: string): Promise<boolean> {
  try {
    const box = await api.voiceStatus();
    if (!canSay(box)) return false;
    await api.voiceSay(text);
    saidAt = Date.now();
    return true;
  } catch {
    return false;
  }
}

let saidAt = 0; // when the box last took an answer of ours (speakOnBox): it may still be saying it

/**
 * Stop what the box is saying for us: the guide's waiting lines, the line on its speaker, and an answer sent with speakOnBox
 * in the last minute and a half that it is still speaking. The box's own conversations are left alone when nothing of ours is out.
 */
export async function stopBoxSpeech(): Promise<void> {
  const line = onSpeaker;
  const answer = Date.now() - saidAt < 90_000;
  saidAt = 0;
  boxLines.cancel(); // stops the box itself when one of the guide's lines is on its speaker
  if (line || !answer) return;
  try {
    const box = await api.voiceStatus();
    if (box.running && (box.state === "speaking" || box.state === "thinking")) await api.voiceStop();
  } catch {
    /* the box is not answering: nothing of ours is being said there */
  }
}

const sleep = (ms: number) => new Promise<void>((r) => window.setTimeout(r, ms));
const POLL_MS = 80;
const IDLE_WAIT_MS = 12_000; // how long a line waits for the box to finish what it is doing
const LINE_MAX_MS = 30_000; // no single line is longer than this on the speaker

let waiting: { text: string; otherwise: (text: string) => void }[] = [];
let turn = 0; // bumped by every cancel: lines from before it are never said
let running = false;
let onSpeaker = false; // a line of ours is being spoken by the box right now
let idleUntil = 0; // the box just finished a line of ours: until then it is taken to be able to speak and idle, with no status call first
const TRUST_MS = 6000;

async function drain(): Promise<void> {
  if (running) return;
  running = true;
  try {
    while (waiting.length > 0) {
      const mine = turn;
      const line = waiting.shift()!;
      let taken = false;
      try {
        // the box drops what it is sent while busy: wait for it to be idle first, unless it has only just finished a line of ours
        for (let attempt = 0; attempt < 2 && !taken && mine === turn; attempt++) {
          const trusted = attempt === 0 && Date.now() < idleUntil;
          let ready = trusted;
          if (!trusted) {
            let box = await api.voiceStatus();
            for (let waited = 0; canSay(box) && box.state !== "idle" && waited < IDLE_WAIT_MS && mine === turn; waited += POLL_MS) {
              await sleep(POLL_MS);
              box = await api.voiceStatus();
            }
            ready = canSay(box) && box.state === "idle";
          }
          if (mine !== turn || !ready) break;
          onSpeaker = true;
          const started = await api.voiceSay(line.text);
          taken = canSay(started) && started.state !== "idle"; // still idle: the box did not take it
          idleUntil = 0;
          if (!trusted) break; // it was asked first and still did not take the line: the browser says it
        }
        for (let waited = 0; taken && waited < LINE_MAX_MS && mine === turn; waited += POLL_MS) {
          await sleep(POLL_MS);
          if ((await api.voiceStatus()).state === "idle") {
            idleUntil = Date.now() + TRUST_MS;
            break;
          }
        }
      } catch {
        /* the box is not answering: the browser says the line unless the box had already taken it */
      }
      onSpeaker = false;
      if (!taken && mine === turn) line.otherwise(line.text);
    }
  } finally {
    running = false;
  }
}

export const boxLines = {
  say(text: string, otherwise: (text: string) => void): void {
    waiting.push({ text, otherwise });
    void drain();
  },
  /** True while a line is waiting or being said on the box (the pointer waits for it). */
  busy: (): boolean => running || waiting.length > 0,
  cancel(): void {
    turn++;
    waiting = [];
    idleUntil = 0;
    if (onSpeaker) {
      onSpeaker = false;
      void api.voiceStop().catch(() => undefined);
    }
  },
};
