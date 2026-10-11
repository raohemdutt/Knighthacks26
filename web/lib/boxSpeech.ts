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
    return true;
  } catch {
    return false;
  }
}

const sleep = (ms: number) => new Promise<void>((r) => window.setTimeout(r, ms));
const POLL_MS = 200;
const IDLE_WAIT_MS = 12_000; // how long a line waits for the box to finish what it is doing
const LINE_MAX_MS = 30_000; // no single line is longer than this on the speaker

let waiting: { text: string; otherwise: (text: string) => void }[] = [];
let turn = 0; // bumped by every cancel: lines from before it are never said
let running = false;
let onSpeaker = false; // a line of ours is being spoken by the box right now

async function drain(): Promise<void> {
  if (running) return;
  running = true;
  try {
    while (waiting.length > 0) {
      const mine = turn;
      const line = waiting.shift()!;
      let taken = false;
      try {
        // the box drops what it is sent while busy: wait for it to be idle first
        let box = await api.voiceStatus();
        for (let waited = 0; canSay(box) && box.state !== "idle" && waited < IDLE_WAIT_MS && mine === turn; waited += POLL_MS) {
          await sleep(POLL_MS);
          box = await api.voiceStatus();
        }
        if (mine !== turn) continue;
        if (canSay(box) && box.state === "idle") {
          onSpeaker = true;
          const started = await api.voiceSay(line.text);
          taken = started.state !== "idle" && started.state !== "off"; // still idle: the box did not take it
          for (let waited = 0; taken && waited < LINE_MAX_MS && mine === turn; waited += POLL_MS) {
            await sleep(POLL_MS);
            if ((await api.voiceStatus()).state === "idle") break;
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
  cancel(): void {
    turn++;
    waiting = [];
    if (onSpeaker) {
      onSpeaker = false;
      void api.voiceStop().catch(() => undefined);
    }
  },
};
