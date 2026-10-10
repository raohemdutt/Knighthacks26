"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useEffect, useState, type FormEvent, type ReactNode } from "react";
import { api, getToken, setToken } from "@/lib/api";
import { useAtlas, type LinkState } from "@/lib/atlas";
import { useProfileState } from "@/lib/profile";
import { useFetch } from "@/lib/useFetch";
import { Logo } from "./Logo";
import ui from "./ui.module.css";
import { VoiceSettings } from "./voice/VoiceSettings";

type Tab = { href: string; label: string; icon: ReactNode };

const TABS: Tab[] = [
  {
    href: "/now",
    label: "Now",
    icon: (
      <>
        <circle cx="12" cy="12" r="8.5" />
        <path d="M12 8v4.5l2.5 1.5" />
        <path d="M3.5 3.5l2 2M20.5 3.5l-2 2" />
      </>
    ),
  },
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

type MoreTab = Tab & { group: string };

/**
 * Reached through the More sheet, in the order a day uses them. Their URLs are unchanged. The study coach is not here:
 * it is one use of the day, reached from the coach's note on Now and from Today (/coach/ still works).
 */
const MORE: MoreTab[] = [
  {
    href: "/impact",
    group: "Your day",
    label: "Impact",
    icon: (
      <>
        <path d="M12 21c-4.5 0-7.5-3-7.5-7.2C4.500 8.500 9 5.500 19.500 4c.5 9.500-2 17-7.500 17z" />
        <path d="M5 20.500c2.500-5 6-8.500 10.500-11" />
      </>
    ),
  },
  {
    href: "/play",
    group: "Your day",
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
    group: "Home",
    label: "Room map",
    icon: (
      <>
        <path d="M3.5 6.5l5.5-2 6 2 5.5-2v13l-5.5 2-6-2-5.5 2z" />
        <path d="M9 4.5v13M15 6.5v13" />
      </>
    ),
  },
  {
    href: "/live",
    group: "Home",
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
    href: "/zones",
    group: "Home",
    label: "Zones",
    icon: (
      <>
        <rect x="3.5" y="5" width="17" height="14" rx="2" />
        <rect x="8" y="9" width="8" height="6" rx="1" strokeDasharray="2 2" />
      </>
    ),
  },
  {
    href: "/teach",
    group: "Home",
    label: "Teach",
    icon: (
      <>
        <path d="M3 9l9-4.5L21 9l-9 4.5z" />
        <path d="M7 11.2V15c0 1.4 2.2 2.8 5 2.8s5-1.4 5-2.8v-3.8M21 9v5" />
      </>
    ),
  },
  {
    href: "/lab",
    group: "Home",
    label: "Learning lab",
    icon: (
      <>
        <path d="M4 18.500l4.500-6 3.500 3 4-7 4 4.500" />
        <path d="M4 5v15h16" />
      </>
    ),
  },
  {
    href: "/circle",
    group: "Friends",
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
    href: "/privacy",
    group: "Privacy",
    label: "Privacy",
    icon: (
      <>
        <path d="M12 3.5l7 2.8v5.4c0 4.2-2.9 7.4-7 8.8-4.1-1.4-7-4.6-7-8.8V6.3z" />
        <path d="M9 12l2.2 2.2L15.5 10" />
      </>
    ),
  },
  {
    href: "/people",
    group: "Privacy",
    label: "People",
    icon: (
      <>
        <circle cx="12" cy="8.5" r="3.5" />
        <path d="M5 20c.6-4 3.4-6 7-6s6.4 2 7 6" />
      </>
    ),
  },
  {
    href: "/spot",
    group: "Inspection",
    label: "Inspection",
    icon: (
      <>
        <rect x="5" y="9" width="12" height="5" rx="1.5" />
        <path d="M7 14v5M15 14v5M17 10.5l3-2.5M9 9V6h3" />
      </>
    ),
  },
];

/** The inspection dashboard belongs to the inspection profile, or to a box that has a kit to show. */
const INSPECTION = "/spot";

const ALL = [...TABS, ...MORE];

/** How the screens are grouped, in the sidebar and in the phone's More sheet. URLs are unchanged. */
const SECTIONS: { title: string; hrefs: string[] }[] = [
  { title: "Everyday", hrefs: ["/", "/now", "/today", "/dayrun"] },
  { title: "The room", hrefs: ["/live", "/map", "/people"] },
  { title: "Plan", hrefs: ["/approvals", "/impact", "/spot"] },
  { title: "Other", hrefs: ["/timeline", "/play", "/circle", "/teach", "/lab", "/zones", "/privacy"] },
];

/** The phone tab bar: the Everyday group. Everything else is one tap further, under More. */
const BAR = ["/", "/now", "/today", "/dayrun"].map((href) => ALL.find((t) => t.href === href)!);

/** Inspection shows on the inspection profile, on a box with a kit, or when that screen is the one open. */
function useShowInspection(path: string): boolean {
  const { profile } = useProfileState();
  const spot = useFetch((s) => api.spotOverview(s));
  return profile.name === "inspection" || spot.data?.configured === true || path === INSPECTION;
}

const LINK_TEXT: Record<LinkState, string> = { live: "Live", connecting: "Connecting…", offline: "Offline, retrying" };

function normalize(path: string): string {
  return path.length > 1 ? path.replace(/\/+$/, "") : path;
}

export function Shell({ children }: { children: ReactNode }) {
  const path = normalize(usePathname() || "/");
  const { link, model, toasts, dismissToast, approvals } = useAtlas();
  const [settings, setSettings] = useState(false);
  const [more, setMore] = useState(false);
  const inMore = !BAR.some((t) => t.href === path);
  useEffect(() => setMore(false), [path]); // the sheet closes once its link has been followed
  const pending = approvals.items.length;
  const inspection = useShowInspection(path);

  if (path === "/operator") return <>{children}</>; // X15: the big-display page has its own header and no tab bar

  return (
    <div className={ui.shell}>
      <aside className={ui.sidebar} aria-label="Screens">
        <span className={ui.sideBrand}>
          <Logo size={34} />
          Atlas
        </span>
        <div className={ui.status} title="Connection to the Atlas box, and the model that answers">
          <span className={`${ui.dot} ${link === "live" ? ui.dotLive : link === "offline" ? ui.dotOffline : ""}`} aria-hidden="true" />
          <span className={ui.statusText}>
            {LINK_TEXT[link]}
            {model ? ` · ${model}` : ""}
          </span>
        </div>
        <nav className={ui.sideNav}>
          {SECTIONS.map((g) => (
            <div key={g.title} className={ui.sideGroup}>
              <span className={ui.sideGroupTitle}>{g.title}</span>
              {g.hrefs.filter((href) => href !== INSPECTION || inspection).map((href) => {
                const t = ALL.find((x) => x.href === href)!;
                const on = path === t.href;
                return (
                  <Link key={t.href} href={t.href} className={`${ui.sideLink} ${on ? ui.sideLinkOn : ""}`} aria-current={on ? "page" : undefined}>
                    <svg className={ui.sideIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      {t.icon}
                    </svg>
                    <span>{t.label}</span>
                    {t.href === "/approvals" && pending > 0 ? (
                      <span className={ui.sideBadge} aria-label={`${pending} waiting`}>
                        {pending}
                      </span>
                    ) : null}
                  </Link>
                );
              })}
            </div>
          ))}
        </nav>
        <button type="button" className={`${ui.sideLink} ${ui.sideSettings}`} onClick={() => setSettings(true)}>
          <svg className={ui.sideIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
            <path d="M4 7h10M18 7h2M4 17h2M10 17h10" />
            <circle cx="16" cy="7" r="2" />
            <circle cx="8" cy="17" r="2" />
          </svg>
          <span>Settings</span>
        </button>
      </aside>

      <header className={ui.header}>
        <span className={ui.brand}>
          <Logo size={26} />
          Atlas
        </span>
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
          {BAR.map((t) => {
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
            {pending > 0 ? (
              <span className={ui.badge} aria-label={`${pending} approvals waiting`}>
                {pending}
              </span>
            ) : null}
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

/** The screens that do not fit the tab bar, grouped the way a day uses them. */
function MoreSheet({ path, onClose }: { path: string; onClose: () => void }) {
  useEscape(onClose);
  const inspection = useShowInspection(path);
  return (
    <div className={`${ui.overlay} ${ui.overlayAboveTabs}`} onClick={onClose}>
      <div className={ui.sheet} role="dialog" aria-modal="true" aria-label="More screens" onClick={(e) => e.stopPropagation()}>
        <div className={`${ui.row} ${ui.between}`}>
          <h2>More</h2>
          <button type="button" className={ui.btn} onClick={onClose}>
            Close
          </button>
        </div>
        {SECTIONS.map((g) => (
          <div key={g.title} className={ui.moreGroup}>
            <h3 className={ui.moreGroupTitle}>{g.title}</h3>
            <ul className={ui.list}>
              {g.hrefs.filter((href) => href !== INSPECTION || inspection).map((href) => {
                const t = ALL.find((x) => x.href === href)!;
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
        ))}
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
        <VoiceSettings />
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
