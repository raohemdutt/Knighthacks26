"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import { getToken, setToken } from "@/lib/api";
import { useAtlas, type LinkState } from "@/lib/atlas";
import ui from "./ui.module.css";

const TABS: { href: string; label: string; icon: ReactNode }[] = [
  { href: "/", label: "Ask", icon: <path d="M4 5.5h16v10.5H10l-4.5 3.5V16H4z" /> },
  {
    href: "/timeline",
    label: "Timeline",
    icon: (
      <>
        <circle cx="12" cy="12" r="8.5" />
        <path d="M12 7.5V12l3 2" />
      </>
    ),
  },
  {
    href: "/today",
    label: "Today",
    icon: (
      <>
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2.5v2.5M12 19v2.5M2.5 12H5M19 12h2.5M5.3 5.3l1.8 1.8M16.9 16.9l1.8 1.8M5.3 18.7l1.8-1.8M16.9 7.1l1.8-1.8" />
      </>
    ),
  },
  {
    href: "/approvals",
    label: "Approvals",
    icon: (
      <>
        <path d="M12 3l7.5 3v5.5c0 4.5-3.2 8-7.5 9.5-4.3-1.5-7.5-5-7.5-9.5V6z" />
        <path d="M8.5 12l2.5 2.5 4.5-5" />
      </>
    ),
  },
  {
    href: "/circle",
    label: "Circle",
    icon: (
      <>
        <circle cx="8.5" cy="9" r="3" />
        <circle cx="16.5" cy="10" r="2.4" />
        <path d="M2.5 19c.5-3.5 3-5.5 6-5.5s5.5 2 6 5.5M15.5 14c3 0 5.2 1.7 6 5" />
      </>
    ),
  },
  {
    href: "/play",
    label: "Sort It",
    icon: (
      <>
        <path d="M5 8h14l-1.3 12H6.3z" />
        <path d="M3.5 8h17M9.5 8V5h5v3" />
      </>
    ),
  },
];

const LINK_TEXT: Record<LinkState, string> = { live: "Live", connecting: "Connecting…", offline: "Offline, retrying" };

function normalize(path: string): string {
  return path.length > 1 ? path.replace(/\/+$/, "") : path;
}

export function Shell({ children }: { children: ReactNode }) {
  const path = normalize(usePathname() || "/");
  const { link, model, toasts, dismissToast, approvals } = useAtlas();
  const [settings, setSettings] = useState(false);
  const pending = approvals.items.length;

  return (
    <div className={ui.shell}>
      <header className={ui.header}>
        <span className={ui.brand}>Atlas</span>
        <div className={ui.status} title="Connection to the Atlas box, and the model that answers">
          <span className={`${ui.dot} ${link === "live" ? ui.dotLive : link === "offline" ? ui.dotOffline : ""}`} aria-hidden="true" />
          <span className={ui.statusText}>
            {LINK_TEXT[link]}
            {model ? ` · ${model}` : ""}
          </span>
        </div>
        <button type="button" className={ui.iconBtn} aria-label="Settings" onClick={() => setSettings(true)}>
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
            <path d="M4 7h10M18 7h2M4 17h2M10 17h10" />
            <circle cx="16" cy="7" r="2" />
            <circle cx="8" cy="17" r="2" />
          </svg>
        </button>
      </header>

      <div className={ui.toasts} aria-live="assertive">
        {toasts.map((t) => {
          const body = (
            <>
              <p className={ui.toastTitle}>{t.title}</p>
              <p className={ui.breakWords}>{t.message}</p>
            </>
          );
          return (
            <div key={t.id} className={`${ui.toast} ${t.tone === "alert" ? ui.toastAlert : ""}`} role="status">
              {t.href ? (
                <Link href={t.href} className={ui.toastBody} onClick={() => dismissToast(t.id)}>
                  {body}
                </Link>
              ) : (
                <div className={ui.toastBody}>{body}</div>
              )}
              <button type="button" className={ui.toastClose} aria-label="Dismiss" onClick={() => dismissToast(t.id)}>
                ×
              </button>
            </div>
          );
        })}
      </div>

      <main className={ui.main}>{children}</main>

      <nav className={ui.tabbar} aria-label="Screens">
        <div className={ui.tabs}>
          {TABS.map((t) => {
            const on = path === t.href;
            return (
              <Link key={t.href} href={t.href} className={`${ui.tab} ${on ? ui.tabOn : ""}`} aria-current={on ? "page" : undefined}>
                <svg className={ui.tabIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  {t.icon}
                </svg>
                <span>{t.label}</span>
                {t.href === "/approvals" && pending > 0 ? (
                  <span className={ui.badge} aria-label={`${pending} waiting`}>
                    {pending}
                  </span>
                ) : null}
              </Link>
            );
          })}
        </div>
      </nav>

      {settings ? <Settings onClose={() => setSettings(false)} /> : null}
    </div>
  );
}

function Settings({ onClose }: { onClose: () => void }) {
  const { link, model } = useAtlas();
  const [token, setTokenText] = useState("");
  const [saved, setSaved] = useState(false);

  useEffect(() => setTokenText(getToken()), []);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const save = (e: FormEvent) => {
    e.preventDefault();
    setToken(token.trim());
    setSaved(true);
  };

  return (
    <div className={ui.overlay} onClick={onClose}>
      <div className={ui.sheet} role="dialog" aria-modal="true" aria-label="Settings" onClick={(e) => e.stopPropagation()}>
        <div className={`${ui.row} ${ui.between}`}>
          <h2>Settings</h2>
          <button type="button" className={ui.btn} onClick={onClose}>
            Close
          </button>
        </div>
        <dl className={ui.kv}>
          <dt>Box</dt>
          <dd>{LINK_TEXT[link]}</dd>
          <dt>Model</dt>
          <dd>{model ?? "not reported yet"}</dd>
        </dl>
        <form className={ui.section} onSubmit={save}>
          <label htmlFor="atlas-token">
            <h3>Access token</h3>
          </label>
          <p className={`${ui.muted} ${ui.small}`}>
            Only needed when the box was started with a token. It is sent with questions, approvals and place sharing, and stays on this phone.
          </p>
          <input
            id="atlas-token"
            className={ui.input}
            type="password"
            autoComplete="off"
            placeholder="No token set"
            value={token}
            onChange={(e) => {
              setTokenText(e.target.value);
              setSaved(false);
            }}
          />
          <button type="submit" className={`${ui.btn} ${ui.btnPrimary}`}>
            {saved ? "Saved" : "Save token"}
          </button>
        </form>
      </div>
    </div>
  );
}
