"use client";

import { api, type Item } from "@/lib/api";
import { useAtlasEvents } from "@/lib/atlas";
import { useFetch } from "@/lib/useFetch";
import s from "./calendar.module.css";

/** One app that feeds the calendar, with its real logo (public/apps). `has` picks its items out of what the box has synced. */
type App = { key: string; name: string; logo: string; has: (i: Item) => boolean };

const fromFeed = (prefix: string) => (i: Item) => i.connector === "calendar" && i.external_id.startsWith(`${prefix}:`);

const APPS: App[] = [
  { key: "outlook", name: "Outlook", logo: "/apps/outlook.jpg", has: fromFeed("outlook") },
  { key: "canvas", name: "Canvas", logo: "/apps/canvas.png", has: fromFeed("canvas") },
  { key: "campus", name: "UCF Events", logo: "/apps/ucf.jpg", has: fromFeed("campus") },
  { key: "groupme", name: "GroupMe", logo: "/apps/groupme.png", has: (i) => i.connector === "groupme" },
];

/** The apps the calendar is built from, as a small row under the greeting. An app with nothing synced yet is dimmed. */
export function ConnectedApps() {
  const items = useFetch((sig) => api.items({}, sig));
  useAtlasEvents(["connector.item.synced"], () => items.reload());
  const all = items.data ?? [];
  return (
    <section className={s.apps} aria-label="Connected apps">
      <h2 className={s.appsTitle}>Connected apps:</h2>
      <ul className={s.appList}>
        {APPS.map((a) => {
          const count = all.filter(a.has).length;
          return (
            <li key={a.key} className={s.app} title={count > 0 ? `${a.name}: ${count} on your calendar` : `${a.name}: nothing synced yet`}>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img className={s.logo} src={a.logo} alt={a.name} width={26} height={26} />
            </li>
          );
        })}
      </ul>
    </section>
  );
}
