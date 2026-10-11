import type { Egress, EgressGroup } from "@/lib/api";

/** Names and arithmetic for GET /vault/egress, shared by the Privacy screen and the operator screen (V7). */

const GROUP_LABEL: Record<string, string> = {
  lan: "Your own network",
  tailnet: "Your phones (tailnet)",
  ntp: "Clock sync",
  dns: "Name lookups",
  hub: "Learning lab",
  system: "System check",
  neon: "Circle (shared state)",
  ntfy: "Phone alerts relay",
  tailscale: "Tailscale (direct)",
  "tailscale-relays": "Tailscale relays",
};

/** Groups that stay on networks the owner controls; everything else goes out to the internet. */
const ON_YOUR_NETWORKS = new Set(["lan", "tailnet"]);

export function groupLabel(name: string): string {
  if (name.startsWith("connector:")) return `${name.slice(10).replace(/^./, (c) => c.toUpperCase())} app`;
  if (name.startsWith("extra:")) return name.slice(6);
  return GROUP_LABEL[name] ?? name;
}

export function bytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / 1024 / 1024).toFixed(1)} MB`;
  return `${(n / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

export function toInternet(g: EgressGroup): boolean {
  return !ON_YOUR_NETWORKS.has(g.name);
}

/** What left for the internet, biggest first, and its total. */
export function internetSummary(egress: Egress): { total: number; groups: EgressGroup[] } {
  const groups = egress.allowed.filter((g) => toInternet(g) && g.bytes > 0).sort((a, b) => b.bytes - a.bytes);
  return { total: groups.reduce((n, g) => n + g.bytes, 0), groups };
}

/** One line on camera data, from the box's own count of approved exports (never a fixed label). */
export function picturesLine(egress: Egress): string | null {
  const p = egress.pictures;
  if (!p) return null;
  if (p.pictures === 0) return "Camera pictures cleared to leave: none";
  const batches = p.approved_exports === 1 ? "1 approved export" : `${p.approved_exports} approved exports`;
  return `Camera pictures cleared to leave: ${p.pictures} (${bytes(p.picture_bytes)}), only in ${batches}`;
}
