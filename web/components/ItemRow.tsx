import type { Item } from "@/lib/api";
import { dayTime, humanize, relative, safeUrl } from "@/lib/format";
import ui from "./ui.module.css";

const KIND_NAME: Record<string, string> = {
  assignment: "Assignment",
  calendar_event: "Calendar",
  message: "Message",
  post: "Post",
  page_change: "Page change",
  search_result: "Search result",
};

/** One connector item: an assignment, a calendar event, a group chat message. Links to its source when it has one. */
export function ItemRow({ item, showKind = false, countdown = false }: { item: Item; showKind?: boolean; countdown?: boolean }) {
  const url = safeUrl(item.url);
  return (
    <li className={ui.card}>
      <div className={`${ui.row} ${ui.wrap}`}>
        {showKind ? <span className={ui.tag}>{KIND_NAME[item.kind] ?? humanize(item.kind)}</span> : null}
        <span className={ui.tag}>{humanize(item.connector)}</span>
        {item.when ? (
          <span className={`${ui.muted} ${ui.small}`}>
            {dayTime(item.when)}
            {countdown ? ` · ${relative(item.when)}` : ""}
          </span>
        ) : null}
      </div>
      <p className={ui.episodeText}>
        {url ? (
          <a href={url} target="_blank" rel="noreferrer noopener">
            {item.title}
          </a>
        ) : (
          item.title
        )}
      </p>
      {item.summary && item.summary !== item.title ? <p className={`${ui.muted} ${ui.small} ${ui.breakWords}`}>{item.summary}</p> : null}
    </li>
  );
}
