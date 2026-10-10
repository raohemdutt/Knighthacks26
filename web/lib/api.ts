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
  /** Only while owner recognition is on and somebody is enrolled: whose habits these are (null: the unnamed set). */
  member?: string | null;
  /** ... and the enrolled member ids that can be asked for with ?member=. */
  members?: string[];
}

/** GET /identity/settings: opt-in owner recognition (contexts/perception/api.py). */
export interface IdentitySettings {
  identity_enabled: boolean;
  available: boolean; // false when the recognition model is not on this box
  recognizer: string | null;
  reason?: string;
}

/** One row of GET /identity/members. The face template itself never leaves the box. */
export interface Member {
  member_id: string;
  name: string;
  enrolled_at: string;
}

/** GET /identity/enroll/status, and what POST /identity/enroll/start returns. */
export interface EnrollStatus {
  state: "idle" | "collecting" | "done" | "failed";
  frames_needed: number;
  frames_done?: number;
  member_id?: string;
  camera_id?: string | null;
  hint?: "no_face" | "one_person_only" | "look_at_camera" | "hold_still" | null;
  error?: string | null;
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

/** contracts.memory.Sighting: where a watch item was last seen. A zone and a text position; never coordinates. */
export interface Sighting {
  label: string;
  camera_id: string;
  zone_id: string | null;
  ts: string;
  frame_ref: string;
  position: string | null; // "left of the laptop"
}

// ---------- place packs and Day Run (contexts/play/dayrun/models.py) ----------

/** One place of a pack (contexts.play.domain.places.Place). `lat`/`lon` are absent for rooms. */
export interface Place {
  name: string;
  kind: string;
  lat?: number;
  lon?: number;
  aliases: string[];
  pack?: string; // a nested pack this place leads into (a dorm -> its apartment)
  zones?: string; // a room's zones file, as a path on the box
}

/** One row of GET /api/places. Edges are not returned. */
export interface PlacePack {
  name: string;
  kind: string; // campus | neighborhood | home
  entry: string | null;
  places: Record<string, Place>;
}

export type MoveKind = "study" | "eat" | "sleep" | "rest" | "event";
export type Activity = "idle" | "study" | "class" | "walk" | "rest" | "event" | "eat" | "sleep";

export interface FixedBlock {
  id: string;
  title: string;
  start: string;
  end: string;
  place: string | null; // "pack.place"; null = wherever you are
  place_name: string | null;
  activity: Activity;
  deadline: string | null;
  url: string | null;
}

export interface Deadline {
  id: string;
  title: string;
  due: string;
  work_left: number; // focused minutes still needed
  in_window: boolean;
  risk: number; // 0..1 at the start of the board
  url: string | null;
}

export interface FriendAt {
  member: string;
  place: string;
  place_name: string;
  start: string;
  end: string;
}

export interface OptionalEvent {
  id: string;
  title: string;
  start: string;
  end: string;
  place: string | null;
  place_name: string | null;
  url: string | null;
}

export interface MoveOption {
  kind: MoveKind;
  place: string | null;
  place_name: string | null;
  title: string;
  walk_minutes: number; // from where the slot starts
  back_minutes: number; // from there on to the next fixed block
  max_minutes: number; // what is left of the slot after both walks
  feasible: boolean;
  reason: string | null;
  friends: string[];
  event: string | null;
  route: string[];
}

export interface OpenSlot {
  id: string;
  start: string;
  end: string;
  minutes: number;
  from_place: string;
  to_place: string | null;
  next_block: string | null;
  moves: MoveOption[];
}

export interface DayRunInitial {
  energy: number;
  focus: number;
  sleep_hours: number;
  meals: number;
  sleep_debt: number;
  hungry_minutes: number;
  source: string; // "habits" when Habits supplied the numbers, "defaults" otherwise
}

export interface Board {
  start: string;
  end: string;
  tick_minutes: number;
  pack: string;
  start_place: string;
  home: string;
  initial: DayRunInitial;
  blocks: FixedBlock[];
  deadlines: Deadline[];
  slots: OpenSlot[];
  friends: FriendAt[];
  events: OptionalEvent[];
  closed: string[];
  sleep_window: [string, string] | null;
}

/** One chosen move. Optional fields may be left out; the box answers with them as null. */
export interface Move {
  slot: string;
  kind: MoveKind;
  place?: string | null;
  minutes?: number | null; // null: fill what is left of the slot
  start?: string | null;
  with_friend?: string | null;
  deadline?: string | null;
  event?: string | null;
}

export interface Tick {
  t: string;
  energy: number; // 0..100
  focus: number; // 0..100
  deadline_risk: number; // 0..1, the worst deadline
  work_left: Record<string, number>;
  activity: Activity;
  place: string;
}

/** One edge walked. `from_xy`/`to_xy` are [lat, lon], or null indoors. */
export interface RouteLeg {
  from: string;
  to: string;
  from_name: string;
  to_name: string;
  minutes: number;
  from_xy: [number, number] | null;
  to_xy: [number, number] | null;
  route: number; // index into `routes`
  depart: string;
}

export interface RouteTaken {
  index: number;
  purpose: string;
  move: number | null;
  start: string;
  end: string;
  depart: string;
  arrive: string;
  minutes: number;
  places: string[];
}

export interface Rejected {
  index: number; // position of the move in the plan that was sent
  move: Move;
  reason: string;
  minutes: number | null;
}

export interface Check {
  id: string; // "sleep", "meals", or a deadline id
  ok: boolean;
  value: number;
  target: number;
  text: string;
  status: string | null;
}

export interface Outcome {
  won: boolean;
  stars: number;
  reasons: string[];
  checks: Check[];
  summary: string;
}

export interface PlannedBlock {
  id: string; // a fixed block's id, or "m<index>" for a move
  kind: string;
  title: string;
  start: string;
  end: string;
  minutes: number;
  place: string;
  place_name: string;
  fixed: boolean;
  move: number | null;
  with_friend: string | null;
}

export interface Change {
  move: number | null;
  kind: "rerouted" | "shortened" | "friend_gone" | "late" | "route_changed" | "rejected";
  text: string;
}

export interface Totals {
  study_minutes: number;
  focused_minutes: number;
  walk_minutes: number;
  sleep_hours: number;
  meals: number;
  energy_end: number;
  energy_min: number;
  points: number; // 0..100
}

export interface SimResult {
  start: string;
  end: string;
  tick_minutes: number;
  outcome: Outcome;
  stars: number;
  timeline: Tick[];
  legs: RouteLeg[];
  routes: RouteTaken[];
  rejected: Rejected[];
  blocks: PlannedBlock[];
  changes: Change[];
  notes: string[];
  totals: Totals;
}

export interface Disruption {
  kind: "block_moved" | "friend_cancelled" | "place_closed";
  block?: string | null;
  to_place?: string | null;
  shift_minutes?: number;
  friend?: string | null;
  place?: string | null;
  text?: string;
}

export interface HardResult {
  seed: number;
  disruptions: Disruption[];
  before: SimResult;
  after: SimResult;
  changes: Change[];
}

export interface CommitResult {
  action_id: string;
  day: string;
  events: { title: string; start: string; end: string; location: string }[];
  stars: number;
  reason: string;
}

export interface Measured {
  focus_minutes: number | null;
  meals: number | null;
  sleep_hours: number | null;
  blocks: Record<string, number>;
}

export interface BlockScore {
  id: string;
  title: string;
  kind: string;
  planned_minutes: number;
  measured_minutes: number | null;
  followed: number | null; // 0..1; null when nothing can measure the block
  note: string;
}

export interface Review {
  day: string;
  score: number | null;
  blocks: BlockScore[];
  planned: Record<string, number>;
  measured: Measured;
  biggest_gap: { block: string | null; kind: string; planned: number; measured: number; text: string } | null;
  text: string;
  measured_by: string;
}

export interface ReplayResult {
  day: string;
  board: Board;
  actual: Measured;
  baseline_moves: Move[];
  baseline: SimResult;
  plan: SimResult;
  points_gained: number;
  stars_gained: number;
  text: string;
}

export interface CampaignResult {
  days: { day: string; carry_in: DayRunInitial; result: SimResult }[];
  won: boolean;
  stars: number;
  points: number;
}

export interface ForecastResult {
  score: number; // 0..100
  bias: number;
  multiplier: number;
  rows: { block: string; estimated: number; measured: number; error: number; off_by: number }[];
  unmeasured: string[];
  text: string;
}

/** What simulate, hard and commit take. `start` defaults to now on the box (or the scenario's own time). */
export interface PlanBody {
  moves: Move[];
  start?: string;
  scenario?: string;
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

export interface VaultPolicy {
  blur_faces: boolean;
  paused_cameras: string[];
  offline: boolean;
  /** Days of pictures and memory to keep; null = forever. */
  keep_days: number | null;
}

export interface EgressGroup {
  /** "lan", "ntp", "dns", "hub", "connector:calendar", ... */
  name: string;
  hosts: string[];
  packets: number;
  bytes: number;
}

export interface EgressBlock {
  dst: string;
  proto: string;
  port: string;
  count: number;
  last: string;
}

export interface Egress {
  /** false where the box runs no firewall (a laptop, or the service is off). */
  enforced: boolean;
  mode?: "online" | "offline";
  allowed: EgressGroup[];
  blocked: number;
  blocked_bytes?: number;
  recent_blocks?: EgressBlock[];
  updated?: string;
  stale?: boolean;
}

export type WipeScope = "all" | "keyframes" | "memory";

export interface WipeToken {
  token: string;
  scope: WipeScope;
  expires_at: string;
}

export interface WipeResult {
  scope: WipeScope;
  done: boolean;
  removed?: Record<string, number>;
  refused?: unknown;
}

export class ApiError extends Error {
  constructor(
    public status: number, // 0 = the gateway could not be reached
    message: string,
    /** The list a structured error carries (Day Run's commit: why the plan does not win). */
    public reasons: string[] = [],
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

async function request<T>(method: "GET" | "POST" | "PUT" | "DELETE", path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const headers: Record<string, string> = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (method !== "GET") {
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
    let reasons: string[] = [];
    try {
      const data: unknown = await res.json();
      if (data && typeof data === "object" && "detail" in data) {
        const d = (data as { detail: unknown }).detail;
        if (typeof d === "string") detail = d;
        else if (d && typeof d === "object" && typeof (d as { message?: unknown }).message === "string") {
          // {message, reasons}: Day Run's "only a winning plan can be committed"
          detail = (d as { message: string }).message;
          const r = (d as { reasons?: unknown }).reasons;
          if (Array.isArray(r)) reasons = r.filter((x): x is string => typeof x === "string");
        } else detail = JSON.stringify(d);
      }
    } catch {
      /* not JSON: keep the status text */
    }
    throw new ApiError(res.status, detail, reasons);
  }
  return (await res.json()) as T;
}

const get = <T,>(path: string, params?: Params, signal?: AbortSignal) => request<T>("GET", withQuery(path, params), undefined, signal);
const post = <T,>(path: string, body: unknown, signal?: AbortSignal) => request<T>("POST", path, body, signal);
const put = <T,>(path: string, body: unknown, signal?: AbortSignal) => request<T>("PUT", path, body, signal);

/** URL of the JPEG behind a frame ref. 404s when the box has no image for it (demo fixtures have none). */
export interface CameraStatus {
  status: "ok" | "degraded" | "down";
  fps_actual: number;
  dropped: number;
  detected: number;
  errors: number;
}

export interface TrackBox {
  label: string;
  confidence: number;
  track_id: number;
  /** Normalized 0..1, origin top left. */
  box: { x1: number; y1: number; x2: number; y2: number };
}

export interface LiveTracks {
  camera_id: string;
  ts_ms: number | null;
  tracks: TrackBox[];
}

export function keyframeUrl(ref: string): string {
  return `${HTTP_BASE}/memory/keyframes/${encodeURIComponent(ref)}`;
}

/** "frames:cam-desk:1791675610000" -> "cam-desk" */
export function cameraOfFrame(ref: string): string | null {
  const parts = ref.split(":");
  return parts.length >= 3 && parts[0] === "frames" ? (parts[1] ?? null) : null;
}

/** The newest frame from a camera (faces blurred by the box); add a changing query so the browser refetches. */
export function snapshotUrl(camera: string, stamp: number): string {
  return `${HTTP_BASE}/capture/${encodeURIComponent(camera)}/snapshot.jpg?t=${stamp}`;
}

export const api = {  health: (signal?: AbortSignal) => get<Health>("/health", undefined, signal),
  captureStatus: (signal?: AbortSignal) => get<{ cameras: Record<string, CameraStatus> }>("/capture/status", undefined, signal),
  liveTracks: (camera: string, signal?: AbortSignal) => get<LiveTracks>(`/capture/${encodeURIComponent(camera)}/tracks`, undefined, signal),

  ask: (text: string) => post<Answer>("/api/ask", { text }),

  vaultPolicy: (signal?: AbortSignal) => get<VaultPolicy>("/vault/policy", undefined, signal),
  setVaultPolicy: (body: Partial<VaultPolicy>) => put<VaultPolicy>("/vault/policy", body),
  egress: (signal?: AbortSignal) => get<Egress>("/vault/egress", undefined, signal),
  wipeRequest: (scope: WipeScope) => post<WipeToken>("/vault/wipe/request", { scope }),
  wipe: (token: string) => post<WipeResult>("/vault/wipe", { token }),

  episodes: (opts: { from?: string; to?: string; camera?: string; label?: string } = {}, signal?: AbortSignal) =>
    get<Episode[]>("/memory/episodes", opts, signal),
  stats: (signal?: AbortSignal) => get<MemoryStats>("/memory/stats", undefined, signal),
  search: (q: string, limit = 20, signal?: AbortSignal) => get<SearchHit[]>("/memory/search", { q, limit }, signal),
  items: (opts: { from?: string; to?: string; kind?: string } = {}, signal?: AbortSignal) =>
    get<Item[]>("/memory/items", opts, signal),

  habitsToday: (signal?: AbortSignal, member?: string) => get<HabitsToday>("/api/habits/today", { member }, signal),

  // People: opt-in owner recognition (P9).
  identitySettings: (signal?: AbortSignal) => get<IdentitySettings>("/identity/settings", undefined, signal),
  setIdentity: (on: boolean) => request<IdentitySettings>("PUT", "/identity/settings", { identity_enabled: on }),
  members: (signal?: AbortSignal) => get<Member[]>("/identity/members", undefined, signal),
  removeMember: (id: string) => request<{ removed: string }>("DELETE", `/identity/members/${encodeURIComponent(id)}`),
  enrollStart: (name: string) => post<EnrollStatus>("/identity/enroll/start", { name }),
  enrollStatus: (signal?: AbortSignal) => get<EnrollStatus>("/identity/enroll/status", undefined, signal),
  impact: (signal?: AbortSignal) => get<Impact>("/api/impact", undefined, signal),

  approvals: (signal?: AbortSignal) => get<ApprovalRequest[]>("/api/approvals", undefined, signal),
  decide: (id: string, decision: Decision) => post<ApprovalRequest>(`/api/approvals/${encodeURIComponent(id)}`, { decision }),

  presence: (signal?: AbortSignal) => get<PlaceShared[]>("/api/presence", undefined, signal),
  sharePlace: (place: string, hours: number) => post<PlaceShared>("/api/presence", { place, hours }),
  leaderboard: (signal?: AbortSignal) => get<Leaderboard>("/api/circle/leaderboard", undefined, signal),

  playRoute: (item: string, signal?: AbortSignal) => get<PlayRoute>("/api/play/route", { item }, signal),
  playAccuracy: (signal?: AbortSignal) => get<PlayAccuracy>("/api/play/accuracy", undefined, signal),

  places: (signal?: AbortSignal) => get<PlacePack[]>("/api/places", undefined, signal),
  lastSeen: (label: string, signal?: AbortSignal) => get<Sighting>(`/memory/objects/${encodeURIComponent(label)}/last-seen`, undefined, signal),

  // Day Run. `scenario` runs a scripted day from fixtures/play/ ("demo") instead of live sources.
  dayrunBoard: (scenario?: string, signal?: AbortSignal) => get<Board>("/api/dayrun/board", { scenario }, signal),
  dayrunSimulate: (body: PlanBody, signal?: AbortSignal) => post<SimResult>("/api/dayrun/simulate", body, signal),
  dayrunHard: (body: PlanBody & { seed?: number; count?: number; disruptions?: Disruption[] }) => post<HardResult>("/api/dayrun/hard", body),
  dayrunCommit: (body: PlanBody) => post<CommitResult>("/api/dayrun/commit", body),
  dayrunReview: (day: string, scenario?: string) => get<Review>("/api/dayrun/review", { day, scenario }),
  dayrunReplay: (body: { day: string; moves: Move[]; scenario?: string }) => post<ReplayResult>("/api/dayrun/replay", body),
  dayrunCampaign: (body: { days: Move[][]; start?: string; scenario?: string }) => post<CampaignResult>("/api/dayrun/campaign", body),
  dayrunForecast: (body: { estimates: { block: string; minutes: number }[]; measured?: Record<string, number>; day?: string; scenario?: string }) =>
    post<ForecastResult>("/api/dayrun/forecast", body),
};
