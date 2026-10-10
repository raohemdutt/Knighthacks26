"use client";

/**
 * App-wide live state: the single WebSocket to /ws, toasts, the pending
 * approvals (so the tab badge and the Approvals screen agree), the model
 * version from /health, and the Ask conversation (so it survives tab changes).
 */

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { ApiError, api, wsUrl, type Answer, type ApprovalRequest, type AtlasEvent } from "./api";

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
}

const Ctx = createContext<AtlasState | null>(null);

const BACKOFF_START_MS = 1000;
const BACKOFF_MAX_MS = 30000;
const TOAST_MS = 9000;
const APPROVAL_EVENTS = new Set(["workspace.approval.requested", "workspace.action.completed", "experience.approval.decided"]);

function str(v: unknown, fallback = ""): string {
  return typeof v === "string" ? v : fallback;
}

function toastFor(e: AtlasEvent): Toast | null {
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
      socket.onopen = () => {
        delay = BACKOFF_START_MS;
        setLink("live");
        reloadApprovals(); // catch up on anything missed while disconnected
        refreshHealth();
      };
      socket.onmessage = (msg) => {
        let e: AtlasEvent;
        try {
          e = JSON.parse(String(msg.data)) as AtlasEvent;
        } catch {
          return;
        }
        if (!e || typeof e.type !== "string") return;
        const toast = toastFor(e);
        if (toast) pushToast(toast);
        if (APPROVAL_EVENTS.has(e.type)) reloadApprovals();
        listeners.current.forEach((fn) => fn(e));
      };
      socket.onclose = () => {
        if (ws !== socket) return; // an older socket closing after we already moved on
        ws = null;
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
      socket?.close();
    };
  }, [pushToast, reloadApprovals, refreshHealth]);

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
    }),
    [link, model, toasts, dismissToast, subscribe, approvals, reloadApprovals, chat, setChat],
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
