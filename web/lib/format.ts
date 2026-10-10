/** Display helpers. All times are shown in the phone's own time zone. */

const TIME = new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" });
const DAY = new Intl.DateTimeFormat(undefined, { weekday: "short", month: "short", day: "numeric" });

function parse(iso: string | null | undefined): Date | null {
  if (!iso) return null;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d;
}

function sameDay(a: Date, b: Date): boolean {
  return a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();
}

/** "3:40 PM" */
export function time(iso: string | null | undefined): string {
  const d = parse(iso);
  return d ? TIME.format(d) : "";
}

/** "Today", "Tomorrow", "Yesterday" or "Sat, Oct 10" */
export function dayLabel(iso: string | null | undefined, now: Date = new Date()): string {
  const d = parse(iso);
  if (!d) return "";
  const shift = (n: number) => new Date(now.getFullYear(), now.getMonth(), now.getDate() + n);
  if (sameDay(d, now)) return "Today";
  if (sameDay(d, shift(1))) return "Tomorrow";
  if (sameDay(d, shift(-1))) return "Yesterday";
  return DAY.format(d);
}

/** "Today 3:40 PM" / "Sat, Oct 10 3:40 PM" */
export function dayTime(iso: string | null | undefined): string {
  const d = parse(iso);
  return d ? `${dayLabel(iso)} ${TIME.format(d)}` : "";
}

/** "3:40 PM to 3:52 PM", or "since 3:40 PM" for an episode that is still open. */
export function span(start: string, end: string | null): string {
  if (!end) return `since ${time(start)}`;
  const a = time(start);
  const b = time(end);
  return a === b ? a : `${a} to ${b}`;
}

/** "in 3 h", "in 2 days", "5 min ago" */
export function relative(iso: string | null | undefined, now: Date = new Date()): string {
  const d = parse(iso);
  if (!d) return "";
  const mins = Math.round((d.getTime() - now.getTime()) / 60000);
  const abs = Math.abs(mins);
  const amount = abs < 60 ? `${abs} min` : abs < 48 * 60 ? `${Math.round(abs / 60)} h` : `${Math.round(abs / 1440)} days`;
  if (abs < 1) return "now";
  return mins > 0 ? `in ${amount}` : `${amount} ago`;
}

/** 0.667 -> "67%" */
export function percent(fraction: number): string {
  return `${Math.round(fraction * 100)}%`;
}

/** 2 -> "2", 1.5 -> "1.5", 0.3333 -> "0.3" */
export function amount(n: number): string {
  return Number.isInteger(n) ? String(n) : n.toFixed(1);
}

/** "cam-desk" -> "Desk camera" */
export function cameraName(id: string): string {
  const bare = id.replace(/^cam-/, "").replace(/[-_]/g, " ");
  return bare ? `${bare.charAt(0).toUpperCase()}${bare.slice(1)} camera` : id;
}

/** "calendar_event" -> "Calendar event" */
export function humanize(s: string): string {
  const t = s.replace(/[_.-]+/g, " ").trim();
  return t.charAt(0).toUpperCase() + t.slice(1);
}

export function hostOf(url: string): string {
  try {
    return new URL(url).host;
  } catch {
    return url;
  }
}

/** Only http(s) links from the box are made clickable. */
export function safeUrl(url: string | null | undefined): string | null {
  return url && /^https?:\/\//i.test(url) ? url : null;
}
