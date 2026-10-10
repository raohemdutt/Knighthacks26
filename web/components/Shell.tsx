"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import { getToken, setToken } from "@/lib/api";
import { useAtlas, type LinkState } from "@/lib/atlas";
import ui from "./ui.module.css";

type Tab = { href: string; label: string; icon: ReactNode };

/** The five screens that fit a phone tab bar. Everything else is one tap further, under More. */
const TABS: Tab[] = [
  { href: "/", label: "Ask", icon: <path d="M4 5.5h16v10.5H10l-4.5 3.5V16H4z" /> },
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
    href: "/dayrun",
    label: "Day Run",
    icon: (
      <>
        <path d="M5 21V4M5 5h11l-2 3.5 2 3.5H5" />
      </>
    ),
  },
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
    href: "/approvals",
    label: "Approvals",
    icon: (
      <>
        <path d="M12 3l7.5 3v5.5c0 4.5-3.2 8-7.5 9.5-4.3-1.5-7.5-5-7.5-9.5V6z" />
        <path d="M8.5 12l2.5 2.5 4.5-5" />
      </>
    ),
  },
];

/** Reached through the More sheet. Their URLs are unchanged. */
const MORE: Tab[] = [
  {
    href: "/live",
    label: "Live",
    icon: (
      <>
        <rect x="3" y="6" width="18" height="13" rx="2.5" />
        <circle cx="12" cy="12.5" r="3.4" />
        <path d="M8 6l1.4-2h5.2L16 6" />
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
  {
    href: "/map",
    label: "Room map",
    icon: (
      <>
        <path d="M3.5 6.5l5.5-2 6 2 5.5-2v13l-5.5 2-6-2-5.5 2z" />
        <path d="M9 4.5v13M15 6.5v13" />
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
  const [more, setMore] = useState(false);
  const inMore = MORE.some((t) => t.href === path);
  useEffect(() => setMore(false), [path]); // the sheet closes once its link has been followed
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
          <button
            type="button"
            className={`${ui.tab} ${ui.tabButton} ${inMore || more ? ui.tabOn : ""}`}
            aria-haspopup="dialog"
            aria-expanded={more}
            onClick={() => setMore((m) => !m)}
          >
            <svg className={ui.tabIcon} viewBox="0 0 24 24" fill="currentColor" stroke="none" aria-hidden="true">
              <circle cx="5.5" cy="12" r="1.9" />
              <circle cx="12" cy="12" r="1.9" />
              <circle cx="18.5" cy="12" r="1.9" />
            </svg>
            <span>More</span>
          </button>
        </div>
      </nav>

      {more ? <MoreSheet path={path} onClose={() => setMore(false)} /> : null}
      {settings ? <Settings onClose={() => setSettings(false)} /> : null}
    </div>
  );
}

function useEscape(onClose: () => void) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
}

/** The screens that do not fit the tab bar. */
function MoreSheet({ path, onClose }: { path: string; onClose: () => void }) {
  useEscape(onClose);
  return (
    <div className={`${ui.overlay} ${ui.overlayAboveTabs}`} onClick={onClose}>
      <div className={ui.sheet} role="dialog" aria-modal="true" aria-label="More screens" onClick={(e) => e.stopPropagation()}>
        <div className={`${ui.row} ${ui.between}`}>
          <h2>More</h2>
          <button type="button" className={ui.btn} onClick={onClose}>
            Close
          </button>
        </div>
        <ul className={ui.list}>
          {MORE.map((t) => {
            const on = path === t.href;
            return (
              <li key={t.href}>
                <Link href={t.href} className={`${ui.moreLink} ${on ? ui.moreLinkOn : ""}`} aria-current={on ? "page" : undefined} onClick={onClose}>
                  <svg className={ui.tabIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                    {t.icon}
                  </svg>
                  <span>{t.label}</span>
                </Link>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}

function Settings({ onClose }: { onClose: () => void }) {
  const { link, model } = useAtlas();
  const [token, setTokenText] = useState("");
  const [saved, setSaved] = useState(false);

  useEffect(() => setTokenText(getToken()), []);
  useEscape(onClose);

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
