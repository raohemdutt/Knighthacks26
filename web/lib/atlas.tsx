"use client";

/**
 * App-wide live state: the single WebSocket to /ws, toasts, the pending
 * approvals (so the tab badge and the Approvals screen agree), the model
 * version from /health, and the Ask conversation (so it survives tab changes).
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from "react";
import { ApiError, api, wsUrl, type Answer, type AnswerDelta, type ApprovalRequest, type AtlasEvent } from "./api";
import { briefStore } from "./brief";

export type LinkState = "connecting" | "live" | "offline";

export interface Toast {
  id: string;
  title: string;
  message: string;
  tone: "alert" | "warn" | "celebrate" | "info";
  href?: string;
}

export type ChatTurn =
  | { id: string; role: "you"; text: string }
  | { id: string; role: "atlas"; answer: Answer }
  | { id: string; role: "error"; text: string; needsToken: boolean };

type Listener = (e: AtlasEvent) => void;

const DELTA_TYPE = "agent.answer.delta";
/** How many finished or foreign question ids are remembered, so their late pieces are never shown. */
const MAX_SETTLED = 200;

/**
 * The answer being written for the question this phone just asked, put together from the
 * `agent.answer.delta` pieces on /ws.
 *
 * The box makes the question id and only returns it with the finished answer, and every phone gets every
 * phone's pieces, so which pieces are ours is worked out here:
 *  - an id first seen while we had no question in flight is somebody else's, for good;
 *  - `experience.question.asked` carries the id and the text: our text claims the id, any other text rules it out
 *    (it normally arrives just before the first piece);
 *  - if that event was missed, pieces are shown while exactly one unclaimed question is being written during ours.
 *    Two at once show nothing until one of them is ruled out.
 * Pieces carry no position, so a gap cannot be mended: when the socket drops, or a piece arrives on a different
 * connection than the one before it, that question's text is withdrawn and the screen waits for the final
 * answer, which always replaces what was streamed. Nothing is ever appended twice.
 */
class AnswerStreams {
  private question: string | null = null; // the text of our question in flight
  private mine: string | null = null; // its id, once the box has said so
  private buffers = new Map<string, { text: string; conn: number; broken: boolean }>();
  private settled = new Set<string>(); // ids that are not, or are no longer, ours
  private shown: string | null = null;
  private subs = new Set<() => void>();

  subscribe = (fn: () => void) => {
    this.subs.add(fn);
    return () => {
      this.subs.delete(fn);
    };
  };

  /** The text to show so far, or null when there is none to show. */
  snapshot = () => this.shown;

  /** We just asked `question`. */
  begin(question: string): void {
    this.end();
    this.question = question.trim();
  }

  /** Our question was answered or failed; anything still arriving for it is ignored. */
  end(): void {
    for (const id of this.buffers.keys()) this.settle(id);
    if (this.mine) this.settle(this.mine);
    this.buffers.clear();
    this.question = null;
    this.mine = null;
    this.publish();
  }

  piece(d: AnswerDelta, conn: number): void {
    if (this.settled.has(d.question_id)) return;
    if (this.question === null) {
      this.settle(d.question_id); // being written while we were not asking: not ours, however long it runs
      return;
    }
    const b = this.buffers.get(d.question_id);
    if (!b) this.buffers.set(d.question_id, { text: d.text, conn, broken: false });
    else if (b.broken) return;
    else if (b.conn !== conn) b.broken = true; // pieces were missed in between
    else b.text += d.text;
    this.publish();
  }

  /** The socket closed: whatever was being written now has a hole in it. */
  linkLost(): void {
    if (this.buffers.size === 0) return;
    for (const b of this.buffers.values()) b.broken = true;
    this.publish();
  }

  event(e: AtlasEvent): void {
    const id = e.question_id;
    if (typeof id !== "string" || !id) return;
    if (e.type === "experience.question.asked") {
      const ours = this.question !== null && this.mine === null && !this.settled.has(id) && str(e.text).trim() === this.question;
      if (ours) this.mine = id;
      else if (id !== this.mine) this.drop(id);
    } else if (e.type === "agent.answer.generated" && id !== this.mine) {
      this.drop(id); // ours stays up until the screen has the answer itself and calls end()
    } else return;
    this.publish();
  }

  private drop(id: string): void {
    this.settle(id);
    this.buffers.delete(id);
  }

  private settle(id: string): void {
    this.settled.add(id);
    if (this.settled.size > MAX_SETTLED) this.settled.delete(this.settled.values().next().value as string);
  }

  private publish(): void {
    let b = this.mine ? this.buffers.get(this.mine) : undefined;
    if (!this.mine && this.buffers.size === 1) b = this.buffers.values().next().value;
    const next = b && !b.broken && b.text.trim() !== "" ? b.text : null;
    if (next === this.shown) return;
    this.shown = next;
    this.subs.forEach((fn) => fn());
  }
}

interface AtlasState {
  link: LinkState;
  model: string | null;
  toasts: Toast[];
  dismissToast: (id: string) => void;
  /** Call `fn` for every event on /ws while the calling component is mounted. */
  subscribe: (fn: Listener) => () => void;
  approvals: { items: ApprovalRequest[]; loading: boolean; error: ApiError | undefined; reload: () => void };
  chat: ChatTurn[];
  setChat: (update: (turns: ChatTurn[]) => ChatTurn[]) => void;
  /** The answer being written for this phone's question; read it with useAnswerStream(). */
  answerStream: AnswerStreams;
}

const Ctx = createContext<AtlasState | null>(null);

const BACKOFF_START_MS = 1000;
const BACKOFF_MAX_MS = 30000;
const TOAST_MS = 9000;
const APPROVAL_EVENTS = new Set(["workspace.approval.requested", "workspace.action.completed", "experience.approval.decided"]);

function str(v: unknown, fallback = ""): string {
  return typeof v === "string" ? v : fallback;
}

const NARRATED_TOAST: Record<string, { title: string; tone: Toast["tone"] }> = {
  morning: { title: "Your day", tone: "info" },
  replan: { title: "Re-plan", tone: "warn" },
  coach: { title: "Coach", tone: "info" },
  evening: { title: "Your day, reviewed", tone: "info" },
};
const RAW_DAY_TOASTS = new Set(["day.plan.proposed", "day.replan.proposed", "day.coach.noted", "day.reviewed"]);

function toastFor(e: AtlasEvent): Toast | null {
  // With the narrator on, the day's moments are toasted once, in its words (`day.narrated`), not as the raw event too.
  if (RAW_DAY_TOASTS.has(e.type) && briefStore.on()) return null;
  if (e.type === "day.narrated") {
    const moment = str(e.moment);
    if (moment === "now" && e.urgent === true) return { id: e.event_id, title: "Time to move", message: str(e.display), tone: "warn", href: "/now/" };
    const kind = NARRATED_TOAST[moment];
    return kind && str(e.display) ? { id: e.event_id, title: kind.title, message: str(e.display), tone: kind.tone, href: "/now/" } : null;
  }
  switch (e.type) {
    case "agent.alert.raised":
      return { id: e.event_id, title: "Alert", message: str(e.message, "Atlas raised an alert."), tone: "alert" };
    case "habits.nudge.raised":
      return {
        id: e.event_id,
        title: e.tone === "celebrate" ? "Nice" : "Nudge",
        message: str(e.message, "Atlas has a nudge for you."),
        tone: e.tone === "celebrate" ? "celebrate" : "warn",
        href: "/today/",
      };
    case "day.plan.proposed":
      return { id: e.event_id, title: "Your day", message: str(e.text, "Atlas planned your day."), tone: "info", href: "/now/" };
    case "day.replan.proposed":
      return { id: e.event_id, title: "Re-plan", message: str(e.why, "Atlas has a new plan for the rest of the day."), tone: "warn", href: "/now/" };
    case "day.coach.noted":
      return { id: e.event_id, title: "Coach", message: str(e.text, "A block just ended."), tone: "info", href: "/now/" };
    case "day.reviewed":
      return { id: e.event_id, title: "Your day, reviewed", message: str(e.change_tomorrow, "The day's review is in."), tone: "info", href: "/now/" };
    case "workspace.approval.requested":
      if (window.location.pathname.startsWith("/approvals")) return null; // the card itself is appearing on screen
      return { id: e.event_id, title: "Needs your OK", message: str(e.reason, "Atlas wants to do something."), tone: "info", href: "/approvals/" };
    default:
      return null;
  }
}

export function AtlasProvider({ children }: { children: ReactNode }) {
  const [link, setLink] = useState<LinkState>("connecting");
  const [model, setModel] = useState<string | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [chat, setChatState] = useState<ChatTurn[]>([]);
  const [approvals, setApprovals] = useState<{ items: ApprovalRequest[]; loading: boolean; error: ApiError | undefined }>({
    items: [],
    loading: true,
    error: undefined,
  });
  const listeners = useRef(new Set<Listener>());
  const [answerStream] = useState(() => new AnswerStreams());

  const dismissToast = useCallback((id: string) => setToasts((t) => t.filter((x) => x.id !== id)), []);

  const pushToast = useCallback(
    (toast: Toast) => {
      setToasts((t) => (t.some((x) => x.id === toast.id) ? t : [...t.slice(-2), toast]));
      window.setTimeout(() => dismissToast(toast.id), TOAST_MS);
    },
    [dismissToast],
  );

  const reloadApprovals = useCallback(() => {
    setApprovals((a) => ({ ...a, loading: true }));
    api
      .approvals()
      .then((items) => setApprovals({ items, loading: false, error: undefined }))
      .catch((err: unknown) =>
        setApprovals((a) => ({ ...a, loading: false, error: err instanceof ApiError ? err : new ApiError(0, String(err)) })),
      );
  }, []);

  const refreshHealth = useCallback(() => {
    api
      .health()
      .then((h) => setModel(h.model))
      .catch(() => undefined); // the link badge already says the box is unreachable
  }, []);

  const subscribe = useCallback((fn: Listener) => {
    listeners.current.add(fn);
    return () => {
      listeners.current.delete(fn);
    };
  }, []);

  // The one socket: reconnects with exponential backoff, and straight away when the phone wakes up.
  useEffect(() => {
    let ws: WebSocket | null = null;
    let timer: number | undefined;
    let delay = BACKOFF_START_MS;
    let stopped = false;
    let conn = 0; // which connection a piece of an answer came in on

    const connect = () => {
      window.clearTimeout(timer);
      if (stopped) return;
      setLink((l) => (l === "live" ? l : "connecting"));
      let socket: WebSocket;
      try {
        socket = new WebSocket(wsUrl());
      } catch {
        retry();
        return;
      }
      ws = socket;
      const id = ++conn;
      socket.onopen = () => {
        delay = BACKOFF_START_MS;
        setLink("live");
        reloadApprovals(); // catch up on anything missed while disconnected
        refreshHealth();
        void briefStore.load();
      };
      socket.onmessage = (msg) => {
        let e: AtlasEvent;
        try {
          e = JSON.parse(String(msg.data)) as AtlasEvent;
        } catch {
          return;
        }
        if (!e || typeof e.type !== "string") return;
        if (e.type === DELTA_TYPE) {
          // A piece of an answer, not an event: it has no event_id or ts, so it goes nowhere near toasts or listeners.
          const d = e as unknown as AnswerDelta;
          if (ws === socket && typeof d.question_id === "string" && d.question_id && typeof d.text === "string") answerStream.piece(d, id);
          return;
        }
        if (typeof e.event_id !== "string") return; // every real event has one; toasts and handlers key on it
        answerStream.event(e);
        briefStore.apply(e); // the day in words, before any listener or toast reads it
        const toast = toastFor(e);
        if (toast) pushToast(toast);
        if (APPROVAL_EVENTS.has(e.type)) reloadApprovals();
        listeners.current.forEach((fn) => fn(e));
      };
      socket.onclose = () => {
        if (ws !== socket) return; // an older socket closing after we already moved on
        ws = null;
        answerStream.linkLost();
        retry();
      };
      socket.onerror = () => socket.close();
    };

    const retry = () => {
      if (stopped) return;
      setLink("offline");
      const wait = delay * (0.75 + Math.random() * 0.5); // jitter, so phones do not reconnect in step
      delay = Math.min(delay * 2, BACKOFF_MAX_MS);
      timer = window.setTimeout(connect, wait);
    };

    const onVisible = () => {
      if (document.visibilityState !== "visible") return;
      if (!ws || ws.readyState === WebSocket.CLOSED) {
        delay = BACKOFF_START_MS;
        connect();
      } else {
        reloadApprovals();
      }
    };

    reloadApprovals();
    refreshHealth();
    void briefStore.load();
    connect();
    document.addEventListener("visibilitychange", onVisible);
    window.addEventListener("online", onVisible);
    return () => {
      stopped = true;
      window.clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisible);
      window.removeEventListener("online", onVisible);
      const socket = ws;
      ws = null;
      answerStream.linkLost();
      socket?.close();
    };
  }, [pushToast, reloadApprovals, refreshHealth, answerStream]);

  const setChat = useCallback((update: (turns: ChatTurn[]) => ChatTurn[]) => setChatState(update), []);

  const value = useMemo<AtlasState>(
    () => ({
      link,
      model,
      toasts,
      dismissToast,
      subscribe,
      approvals: { ...approvals, reload: reloadApprovals },
      chat,
      setChat,
      answerStream,
    }),
    [link, model, toasts, dismissToast, subscribe, approvals, reloadApprovals, chat, setChat, answerStream],
  );

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useAtlas(): AtlasState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useAtlas must be used inside <AtlasProvider>");
  return v;
}

/** Run `fn` for every /ws event whose type starts with one of `prefixes`. */
export function useAtlasEvents(prefixes: string[], fn: Listener): void {
  const { subscribe } = useAtlas();
  const ref = useRef(fn);
  ref.current = fn;
  const key = prefixes.join("|");
  useEffect(() => {
    const wanted = key.split("|");
    return subscribe((e) => {
      if (wanted.some((p) => e.type.startsWith(p))) ref.current(e);
    });
  }, [subscribe, key]);
}

/**
 * The answer to this phone's question as it is being written. `begin(question)` just before asking, `end()` once
 * the answer (or an error) is in; `text` is what has been written so far, or null when there is nothing to show
 * (no pieces yet, the socket is down, or the answer came without any).
 */
export function useAnswerStream(): { text: string | null; begin: (question: string) => void; end: () => void } {
  const { answerStream } = useAtlas();
  const text = useSyncExternalStore(answerStream.subscribe, answerStream.snapshot, () => null);
  return useMemo(() => ({ text, begin: (q: string) => answerStream.begin(q), end: () => answerStream.end() }), [text, answerStream]);
}
