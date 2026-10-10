/**
 * The one typed client for the Atlas gateway (contexts/gateway/app.py).
 *
 * Types mirror the JSON the gateway actually returns (checked by
 * scripts/check-api.mjs against a running gateway), not a wished-for shape.
 * Datetimes arrive as ISO strings.
 */

// ---------- shapes ----------

/** contracts.types.Citation: a camera keyframe, an outside link, or just a titled source. */
export interface Citation {
  episode_id: string | null;
  frame_ref: string | null; // "frames:<camera_id>:<ts_ms>"
  ts: string | null;
  url: string | null;
  title: string | null;
}

/** Every bus event carries this envelope. */
export interface EventEnvelope {
  event_id: string;
  type: string;
  version: number;
  ts: string;
}

/** POST /api/ask -> contracts.agent.AnswerGenerated */
export interface Answer extends EventEnvelope {
  type: "agent.answer.generated";
  question_id: string;
  answer: string;
  citations: Citation[];
  model_version: string;
}

export interface Health {
  ok: boolean;
  model: string;
}

/** contracts.memory.Episode */
export interface Episode {
  id: string;
  camera_id: string;
  start: string;
  end: string | null; // null while still open
  labels: string[];
  captions: string[];
  summary: string | null;
  keyframe_refs: string[];
}

/** contracts.memory.MemoryStats */
export interface MemoryStats {
  hours_recorded: number;
  episodes: number;
  captions: number;
  first_event: string | null;
}

/** contracts.memory.SearchHit */
export interface SearchHit {
  episode_id: string;
  camera_id: string;
  ts: string;
  text: string;
  score: number;
  frame_ref: string | null;
  kind: string; // "episode", or a connector item kind
  url: string | null;
  title: string | null;
}

/** contracts.memory.Item: one connector item (assignment, calendar event, message, ...). */
export interface Item {
  connector: string;
  kind: string;
  external_id: string;
  title: string;
  when: string | null;
  url: string | null;
  summary: string | null;
}

export interface HabitNudge {
  message: string;
  tone: "warn" | "celebrate";
  ts: string;
}

/** One row of GET /api/habits/today (HabitTracker._view). */
export interface Habit {
  id: string;
  name: string;
  unit: string;
  direction: "at_least" | "at_most";
  target: number;
  day: string;
  value: number;
  met: boolean;
  applies: boolean; // false when there is nothing to measure yet today
  streak: number;
  closed: boolean;
  nudge: HabitNudge | null;
}

export interface HabitsToday {
  day: string | null;
  habits: Habit[];
  latest_nudge: HabitNudge | null;
}

export interface ImpactCounts {
  items_seen: number;
  recyclable_seen: number;
  recycled: number;
  contamination: number;
  diversion_rate: number; // 0..1
}

export interface Impact {
  total: ImpactCounts;
  before_after_nudges: { before: ImpactCounts; after: ImpactCounts } | null;
}

export type ActionKind = "export_training_data" | "send_message" | "browse" | "connector_write" | "circle_share";

/** GET /api/approvals rows (contexts.workspace.domain.approvals.ActionRequest). Note: `id`, not `action_id`. */
export interface ApprovalRequest {
  id: string;
  kind: ActionKind | string;
  target: string;
  reason: string;
  params: Record<string, unknown>;
  status: "proposed" | "approved" | "denied" | "expired" | "done" | "failed";
  expires_at: string;
  result: string | null;
}

export type Decision = "approve" | "deny";

/** contracts.circle.PlaceShared */
export interface PlaceShared extends EventEnvelope {
  circle_id: string;
  member_id: string;
  place: string;
  source: "phone" | "desk";
  expires_at: string;
}

export interface ImpactLeader {
  member_id: string;
  display_name: string;
  items_seen: number;
  recycled: number;
  contamination: number;
  diversion_rate: number;
}

export interface HabitLeader {
  member_id: string;
  display_name: string;
  habit_id: string;
  streak: number;
  day: string;
}

export interface ScoreLeader {
  round_id: string;
  member_id: string;
  display_name: string;
  item: string;
  players: number;
  correct: boolean;
  seconds: number;
  score: number;
  played_at: string;
}

export interface Leaderboard {
  circle_id: string;
  impact: ImpactLeader[];
  habits: HabitLeader[];
  scores: ScoreLeader[];
}

export interface Cell {
  row: number;
  col: number;
}

/** GET /api/play/route. `path` is empty when no route exists. */
export interface PlayRoute {
  item: string;
  bin: string; // zone id of the right bin
  rows: number;
  cols: number;
  path: Cell[];
}

export interface PlayAccuracy {
  session_id: string;
  rounds: number;
  first_round: number | null; // 0..1, null until five rounds are played
  round_5_on: number | null;
}

// ---------- events on /ws ----------

export interface AlertRaised extends EventEnvelope {
  type: "agent.alert.raised";
  alert_id: string;
  rule_id: string;
  message: string;
  evidence: Citation[];
}

export interface NudgeRaised extends EventEnvelope {
  type: "habits.nudge.raised";
  nudge_id: string;
  habit_id: string;
  message: string;
  tone: "warn" | "celebrate";
  evidence: Citation[];
}

/** Anything on the bus. Narrow on `type` before reading other fields. */
export type AtlasEvent = EventEnvelope & Record<string, unknown>;

// ---------- transport ----------

const TOKEN_KEY = "atlas.token";

/**
 * Where the gateway lives. Same origin by default (the gateway serves this app).
 * NEXT_PUBLIC_ATLAS_URL points a build at a gateway elsewhere. Under `next dev`
 * HTTP stays same-origin because next.config.mjs proxies it to that URL (the
 * gateway sends no CORS headers); only the WebSocket goes there directly.
 */
const CONFIGURED = (process.env.NEXT_PUBLIC_ATLAS_URL ?? "").replace(/\/+$/, "");
const HTTP_BASE = process.env.NODE_ENV === "development" ? "" : CONFIGURED;

export function wsUrl(): string {
  const origin = CONFIGURED || window.location.origin;
  return origin.replace(/^http/, "ws") + "/ws";
}

export function getToken(): string {
  try {
    return window.localStorage.getItem(TOKEN_KEY) ?? "";
  } catch {
    return ""; // private mode or blocked storage: carry on without a token
  }
}

export function setToken(token: string): void {
  try {
    if (token) window.localStorage.setItem(TOKEN_KEY, token);
    else window.localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* nothing to do */
  }
}

export class ApiError extends Error {
  constructor(
    public status: number, // 0 = the gateway could not be reached
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
  /** 404 from an optional part of the box (Circle, Sort It, Waste Watch) means "not set up", not "broken". */
  get notConfigured(): boolean {
    return this.status === 404;
  }
  get needsToken(): boolean {
    return this.status === 401;
  }
}

type Params = Record<string, string | number | undefined | null>;

function withQuery(path: string, params?: Params): string {
  if (!params) return path;
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) if (v !== undefined && v !== null && v !== "") q.set(k, String(v));
  const s = q.toString();
  return s ? `${path}?${s}` : path;
}

async function request<T>(method: "GET" | "POST", path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const headers: Record<string, string> = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (method === "POST") {
    const token = getToken();
    if (token) headers["Authorization"] = `Bearer ${token}`;
  }
  let res: Response;
  try {
    res = await fetch(HTTP_BASE + path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body), signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError(0, "Can't reach the Atlas box. Check that you are on the same network.");
  }
  if (!res.ok) {
    let detail = res.statusText || `HTTP ${res.status}`;
    try {
      const data: unknown = await res.json();
      if (data && typeof data === "object" && "detail" in data) {
        const d = (data as { detail: unknown }).detail;
        detail = typeof d === "string" ? d : JSON.stringify(d);
      }
    } catch {
      /* not JSON: keep the status text */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

const get = <T,>(path: string, params?: Params, signal?: AbortSignal) => request<T>("GET", withQuery(path, params), undefined, signal);
const post = <T,>(path: string, body: unknown) => request<T>("POST", path, body);

/** URL of the JPEG behind a frame ref. 404s when the box has no image for it (demo fixtures have none). */
export function keyframeUrl(ref: string): string {
  return `${HTTP_BASE}/memory/keyframes/${encodeURIComponent(ref)}`;
}

/** "frames:cam-desk:1791675610000" -> "cam-desk" */
export function cameraOfFrame(ref: string): string | null {
  const parts = ref.split(":");
  return parts.length >= 3 && parts[0] === "frames" ? (parts[1] ?? null) : null;
}

export const api = {
  health: (signal?: AbortSignal) => get<Health>("/health", undefined, signal),

  ask: (text: string) => post<Answer>("/api/ask", { text }),

  episodes: (opts: { from?: string; to?: string; camera?: string; label?: string } = {}, signal?: AbortSignal) =>
    get<Episode[]>("/memory/episodes", opts, signal),
  stats: (signal?: AbortSignal) => get<MemoryStats>("/memory/stats", undefined, signal),
  search: (q: string, limit = 20, signal?: AbortSignal) => get<SearchHit[]>("/memory/search", { q, limit }, signal),
  items: (opts: { from?: string; to?: string; kind?: string } = {}, signal?: AbortSignal) =>
    get<Item[]>("/memory/items", opts, signal),

  habitsToday: (signal?: AbortSignal) => get<HabitsToday>("/api/habits/today", undefined, signal),
  impact: (signal?: AbortSignal) => get<Impact>("/api/impact", undefined, signal),

  approvals: (signal?: AbortSignal) => get<ApprovalRequest[]>("/api/approvals", undefined, signal),
  decide: (id: string, decision: Decision) => post<ApprovalRequest>(`/api/approvals/${encodeURIComponent(id)}`, { decision }),

  presence: (signal?: AbortSignal) => get<PlaceShared[]>("/api/presence", undefined, signal),
  sharePlace: (place: string, hours: number) => post<PlaceShared>("/api/presence", { place, hours }),
  leaderboard: (signal?: AbortSignal) => get<Leaderboard>("/api/circle/leaderboard", undefined, signal),

  playRoute: (item: string, signal?: AbortSignal) => get<PlayRoute>("/api/play/route", { item }, signal),
  playAccuracy: (signal?: AbortSignal) => get<PlayAccuracy>("/api/play/accuracy", undefined, signal),
};
