/**
 * The one typed client for the Atlas gateway (contexts/gateway/app.py).
 *
 * Types mirror the JSON the gateway actually returns (checked by
 * scripts/check-api.mjs against a running gateway), not a wished-for shape.
 * Datetimes arrive as ISO strings.
 */

import type { CampusMapData } from "./campusmap";

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
  /** Ask Atlas's tools (contexts/agent/application/ask_router.py), set on a box that routes questions to them:
   * who answered ("the box", "the network brain", "the box's rules"), the tool, its arguments and result,
   * a screen of this app to open, and the approval card the tool proposed. */
  answered_by?: string | null;
  tool?: string | null;
  tool_args?: Record<string, unknown> | null;
  tool_result?: AskToolResult | null;
  link?: string | null;
  action_id?: string | null;
}

/** What an Ask Atlas tool found: its words, whether it could do it, and the fields it returned. */
export interface AskToolResult {
  text: string;
  ok: boolean;
  events?: { title: string; when: string | null; url?: string | null; connector?: string; kind?: string }[];
  event?: { title: string; start: string; end: string; location?: string | null; attendees?: string[] };
  action_id?: string;
  dry_run?: boolean;
  [key: string]: unknown;
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

/** GET /api/digest -> contexts.agent.application.digest.DailyDigest (A6) */
export interface DailyDigest {
  day: string;
  start: string;
  end: string;
  text: string;
  episodes: number;
  cameras: Record<string, number>;
  seen_most: [string, number][];
  people_episodes: number;
  first: string | null;
  last: string | null;
  busiest_hour: string | null;
  moments: { hour: string; episodes: number; text: string; citation: Citation }[];
  citations: Citation[];
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
  coverage?: number; // sleep, focus: share of the night or day the desk camera was reporting for (0 to 1)
  reason?: string | null; // why it was not judged, when the cameras saw too little
  source?: "watch" | "camera"; // sleep only: measured by a watch that night, or estimated from the empty desk
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
  /** The retired anonymous room mode: always false now. Atlas reads nobody who has not done the face scan. */
  everyone?: boolean;
  expressions_available?: boolean; // false when the expression model is not on this box
  photos_available?: boolean; // false when the box cannot decode pictures (no OpenCV)
}

/**
 * One row of GET /identity/members. The template is never in an answer. It leaves the box only through face share
 * (off by default), and then only the box owner's own, sealed to a friend's box.
 */
export interface Member {
  member_id: string;
  name: string;
  enrolled_at: string;
  updated_at?: string | null;
  photos?: number; // how many looks the template is made of (up to 12)
  source?: "live" | "scan" | "photos" | "circle"; // circle: a friend's box shared it with this one (face share)
  expressions?: boolean; // this person switched "read my expressions" on
  circle_member_id?: string | null; // which Circle member this is, when someone said so
  /** They did the face scan (or a live look at the box's camera). False: photos only, so no expressions are read. */
  scanned?: boolean;
  /** The box matches faces to them. False for photos alone, until a scan finishes. */
  recognised?: boolean;
  shared_by?: string | null; // source "circle" only: the friend's Circle member id
  received_at?: string | null;
  expires_at?: string | null; // when this box deletes it unless the friend renews
  /** Written by the demo seed (`python -m contexts.demo circle`): made-up numbers, nobody's face. Shown with a "Demo data" tag. */
  demo?: boolean;
}

/** A box key as two friends compare it: "ab12 cd34 ef56 0a1b 2c3d". */
export interface FaceShareKey {
  known: boolean; // false until the friend's box has face sharing on and was heard from
  fingerprint: string | null;
  changed: boolean; // another key appeared under their name: sharing stops until it is compared and trusted
  new_fingerprint: string | null;
}

export interface FaceShareFriend {
  member_id: string;
  display_name: string;
  key: FaceShareKey;
  /** My own template, to this friend's box. */
  outgoing: { state: "none" | "waiting_approval" | "shared"; share_id: string | null; shared_at: string | null; expires_at: string | null; action_id: string | null };
  /** This friend's template, on this box. `usable` is false while recognition is off here or the face models differ. */
  incoming: { state: "none" | "offered" | "accepted"; received_at: string | null; expires_at: string | null; offered_at: string | null; usable: boolean | null };
  renewal_offered: boolean;
}

/** GET /api/circle/face-share (contexts/circle/face_share_api.py). 409 while it is switched off, which is the default. */
export interface FaceShareStatus {
  enabled: true;
  circle_id: string;
  member_id: string;
  expiry_days: number;
  recognition_enabled: boolean;
  recognition_available: boolean;
  explanation: string;
  me: { fingerprint: string; member: { member_id: string; name: string } | null; can_share: boolean; why_not: string | null };
  friends: FaceShareFriend[];
}

/** Why the box did not take a photo or a still; `message` is the sentence to show. */
export type PhotoReason =
  | "unreadable" | "no_face" | "two_faces" | "too_small" | "turned" | "too_dark" | "too_bright" | "blurry" | "different" | "not_them"
  | "enough_like_this";

export interface PhotoResult {
  index: number;
  accepted: boolean;
  reason: PhotoReason | null;
  message: string | null;
}

/** POST /identity/enroll/photos and POST /identity/enroll/scan/finish. */
export interface EnrollPhotosResult {
  state: "done" | "failed";
  member_id: string;
  accepted: number;
  rejected: number;
  needed: number;
  exemplars: number;
  error: "not_enough_photos" | "already_enrolled" | null;
  photos: PhotoResult[];
  /** False after photos alone: they are kept, and the person is recognised only once a face scan finishes. */
  scanned?: boolean;
  message?: string | null;
}

export type ScanPose = "front" | "left" | "right" | "up";

/** POST /identity/enroll/scan/start and .../frame: where a guided scan stands. */
export interface ScanStatus {
  session: string;
  member_id: string;
  count: number;
  needed: number;
  poses: Record<ScanPose, number>;
  want: ScanPose | null;
  prompt: string | null;
  done: boolean;
  /** Only on a frame's answer: */
  accepted?: boolean;
  reason?: PhotoReason | null;
  message?: string | null;
  pose?: ScanPose | null;
}

export type Looked = "tense" | "tired" | "upbeat" | "neutral";

/** GET /identity/expressions: stretches of apparent expression. A noisy signal, not an emotion and nothing about health. */
export interface ExpressionSeries {
  member_id: string | null; // null: the anonymous room aggregate
  hours: number;
  samples: number;
  looked: Partial<Record<Looked, number>>; // share of the counted looks
  strained_share: number | null;
  windows: {
    start: string;
    end: string;
    camera_id: string;
    dominant: string;
    looked: Looked;
    shares: Record<string, number>;
    samples: number;
    confidence: number;
    head_down: number | null;
    turned_away: number | null;
    people?: number | null;
  }[];
  note: string;
}

/** GET /identity/present: the enrolled people a camera recognises right now (ids only). */
export interface PresentNow {
  present: { member_id: string; circle_member_id: string | null; cameras: string[] }[];
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

export type ActionKind =
  | "export_training_data"
  | "send_message"
  | "browse"
  | "connector_write"
  | "circle_share"
  | "device_control";

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

// ---------- the League (contexts/play/league) ----------

export type StakeKind = "pays_for_food" | "picks_the_event" | "custom";

export interface LeagueStanding {
  member_id: string;
  display_name: string;
  rank: number;
  /** 0..100, or null when nothing of theirs was judged this week. */
  score: number | null;
  met: number;
  judged: number;
  reasons: string[];
}

export interface LeagueWho {
  member_id: string;
  display_name: string;
  score: number | null;
  reason: string;
}

export interface LeagueChallengeMember {
  member_id: string;
  display_name: string;
  met_days: number;
  judged_days: number;
  days: number;
}

export interface LeagueChallenge {
  challenge_id: string;
  title: string;
  habit_id: string;
  direction: "at_least" | "at_most";
  target: number;
  starts: string; // local date
  ends: string;
  days: number;
  started_by: string;
  status: "upcoming" | "running" | "closed";
  members: LeagueChallengeMember[];
  winners: string[];
  summary: string | null;
}

export interface LeagueTeamWaste {
  line: string; // "team: 71% diverted, target 80%"
  rate: number | null;
  target: number;
  recycled: number;
  recyclable: number;
  members: { member_id: string; display_name: string; recycled: number; recyclable: number; contamination: number; diversion_rate: number | null }[];
  contamination_of_the_week: string | null;
  /** This box's own week on the Impact screen: the latest milestone said, and the Sort It practice-set size. */
  milestone?: string | null;
  practice_size?: number | null;
}

/** GET /api/league: 200 in every state; `configured` false means no config/league.yaml on the box. */
export interface LeagueView {
  configured: boolean;
  /** true: no Circle, this box plays alone under "You". */
  local: boolean;
  circle_id: string | null;
  member_id: string | null;
  note: string | null;
  members: { member_id: string; display_name: string }[];
  week: { start: string; end: string; closes_at: string; closes_in_s: number } | null;
  targets: { id: string; label: string; weight: number; unit: string; source: "habit" | "dayrun" | "impact" }[];
  standings: LeagueStanding[];
  worst: LeagueWho | null;
  best: LeagueWho | null;
  summary: string | null;
  stake: { kind: StakeKind; text: string | null; text_for_week: string; set_for_week: string | null } | null;
  team_waste: LeagueTeamWaste | null;
  challenges: LeagueChallenge[];
  presets: { kind: string; title: string; habit_id: string; direction: "at_least" | "at_most"; target: number }[];
  last_closed: { week_start: string; worst: string | null; best: string | null; summary: string } | null;
}

export interface LeagueMe {
  member_id: string;
  display_name: string;
  week: { start: string; end: string };
  score: number | null;
  met: number;
  judged: number;
  reasons: string[];
  breakdown: { target: string; label: string; unit: string; weight: number; met: number; judged: number; share: number | null; detail: string | null }[];
}

export type RsvpAnswer = "free" | "busy" | "counter";

export interface CircleAttendee {
  member_id: string;
  /** The member id again when the Circle has no display name for them. */
  display_name: string;
  answer: RsvpAnswer;
}

export interface CircleEvent {
  id: string;
  thread_id: string | null;
  title: string;
  starts_at: string; // UTC ISO 8601
  ends_at: string;
  location: string | null;
  created_by: string | null;
  attendees: CircleAttendee[];
}

/** GET /api/circle/events: agreed events that have not ended, soonest first (contexts/circle/api.py). */
export interface CircleEvents {
  configured: boolean; // false: this box has no Circle member
  circle_id: string | null;
  /** "neon": the Circle's shared calendar. "local": only what this box itself agreed. */
  source: "neon" | "local" | "none";
  offline: boolean; // the shared calendar is set up but could not be read just now
  events: CircleEvent[];
}

export interface ShareKindStatus {
  /** habit_summary, impact_count, game_score (the outbox kinds), presence, event_plan */
  kind: string;
  on: boolean;
  queued: number | null; // null: this kind does not go through the outbox
  sent: number | null;
  last_synced_at: string | null;
}

/** POST /api/circle/proposals (contracts.circle.Proposal). Times are ISO 8601 with a UTC offset. */
export interface ProposalBody {
  title: string;
  start: string;
  end: string;
  location?: string;
  /** Member ids. The plan is agreed once all of them answer free. */
  invitees?: string[];
}

// ---------- the agents' conversations (contexts/circle/application/threads.py) ----------

export type ThreadState = "proposed" | "waiting" | "countered" | "agreed" | "declined";

export interface CircleWho {
  member_id: string;
  display_name: string;
  mine: boolean;
}

/** One line of a conversation. `line` is plain language written on the box from the structured fields only. */
export interface CircleMessage {
  id: string;
  thread_id: string;
  kind: "proposal" | "reply" | "agreed" | "note";
  answer: RsvpAnswer | null;
  at: string;
  line: string;
  from_member: string | null; // null: the agreement itself
  display_name: string | null;
  mine: boolean;
  starts_at: string | null;
  ends_at: string | null;
  location: string | null;
  /** "log": this box saw it. "store": read from the Circle. "agreement": implied by everyone having agreed. */
  source: "log" | "store" | "agreement";
}

/** What this person can do next. "propose" carries the plan a new approval card would ask about. */
export interface ThreadAction {
  kind: "approve" | "propose" | "none";
  action_id: string | null;
  proposal: { title: string; start: string; end: string; location: string | null; invitees: string[] } | null;
  text: string;
}

export interface CircleThread {
  thread_id: string;
  /** A counter-offer the proposer accepts is sent under a new id; every id of the conversation is here. */
  legs: string[];
  title: string;
  proposer: CircleWho;
  mine: boolean;
  starts_at: string;
  ends_at: string;
  location: string | null;
  state: ThreadState;
  invitees: (CircleWho & { answer: RsvpAnswer | null })[];
  waiting_on: CircleWho[];
  /** When the latest proposal was sent; null while its card waits for approval. */
  asked_at: string | null;
  last_activity: string;
  pending_action_id: string | null;
  action: ThreadAction;
  last_line: string;
  messages_count: number;
}

/** GET /api/circle/threads: 200 in every state. */
export interface CircleThreads {
  configured: boolean;
  circle_id: string | null;
  member_id: string | null;
  /** "log": this box's own log only. "log+store": plus what the Circle's store holds. */
  source: "none" | "log" | "log+store";
  offline: boolean;
  members: CircleWho[];
  threads: CircleThread[];
}

/** GET /api/circle/threads/{id} */
export interface CircleThreadDetail extends CircleThread {
  messages: CircleMessage[];
  source: "log" | "log+store";
  offline: boolean;
}

export interface CircleMember extends CircleWho {
  /** true once a box of theirs was heard from; null: not seen from this box's log (phone only so far). */
  has_box: boolean | null;
  box_seen: string | null;
  place: { place: string; source: "phone" | "desk"; shared_at: string; expires_at: string } | null;
}

/** GET /api/circle/members: 200 in every state. `connect` are the two config lines that join a Circle. */
export interface CircleMembers {
  configured: boolean;
  circle_id: string | null;
  member_id: string | null;
  connect: string[];
  members: CircleMember[];
}

/** GET /api/circle/brain: whether this member asks the group brain (G4) and where the text would go. */
export interface BrainStatus {
  configured: boolean;
  circle_id: string | null;
  member_id: string | null;
  opted_in: boolean;
  opt_in_ref: string | null;
  changed_at: string | null;
  node: string | null;
  host: string | null;
  badge: string;
}

/** POST /api/circle/brain/ask */
export interface BrainAnswerReply {
  answer: string;
  badge: string;
  ask_id: string;
  opt_in_ref: string;
  model: string;
  latency_ms: number;
}

/** The energy of the last group-brain answer, from the pod's own power samples (contexts/circle/brain_power.py). */
export interface BrainLastAnswer {
  ask_id: string;
  pending: boolean;
  asked?: string;
  answered?: string;
  seconds?: number;
  joules?: number | null;
  above_idle_j?: number;
  mean_w?: number | null;
  samples?: number;
  before_w?: number | null;
  idle_w?: number;
  low_confidence?: boolean;
  reasons?: string[];
  reason?: string;
}

/** GET /api/circle/brain/power: the teacher pod's watts, relayed by the box (the hub token never leaves it). */
export interface BrainPower {
  configured: boolean;
  node: string | null;
  host: string | null;
  where: string;
  idle_w: number;
  samples: { seq: number; ts: string; w: number }[];
  last_seq: number | null;
  offset_s: number | null;
  last_answer: BrainLastAnswer | null;
  error: string | null;
}

/** GET /api/play/campus-route */
export interface CampusRoute {
  path: string[];
  minutes: number;
}

/** GET /api/circle/sharing: what leaves this box for the Circle (contexts/circle/api.py). */
export interface CircleSharing {
  configured: boolean;
  circle_id: string | null;
  member_id: string | null;
  share_summaries: boolean; // the setting
  share_presence: boolean;
  sharing_now: boolean; // the setting is on and there is a shared database to send to
  database: "connected" | "offline" | "unknown" | "not_configured";
  database_note: string | null;
  last_reached_at: string | null;
  queued: number;
  sent: number;
  last_synced_at: string | null;
  oldest_queued_at: string | null;
  kinds: ShareKindStatus[];
  place: { place: string; source: "phone" | "desk"; expires_at: string } | null;
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

// ---------- the room twin (contexts/play/room) ----------

export type RoomWall = "north" | "east" | "south" | "west";

/** A box of furniture on the floor plan: metres, x east, y south; `rot` in degrees clockwise (0 = its front looks south). */
export interface RoomFurniture {
  id: string;
  type: string; // bed desk chair wardrobe shelf fridge nightstand table rug bin
  at: [number, number];
  size: [number, number]; // width along its front, depth
  rot: number;
  height: number;
  bin: string | null; // for a bin: its name in config/sorting.yaml
  color: string | null;
  label: string | null;
}

/** A door, a window or a poster on a wall. `at` is the centre, in metres along the wall from its west or north end. */
export interface RoomOpening {
  id?: string;
  wall: RoomWall;
  at: number;
  width: number;
  sill?: number;
  height?: number;
  art?: string;
}

/** GET /api/room: the floor plan, each camera's calibration, and the zones laid on the floor. */
export interface RoomPlan {
  name: string;
  title: string;
  size: [number, number];
  wall_height: number;
  doors: RoomOpening[];
  windows: RoomOpening[];
  posters: RoomOpening[];
  furniture: RoomFurniture[];
  /** `source`: "default" = the placeholder points the room file shipped with (positions are approximate); "set" = given for the real camera. */
  calibration: Record<string, { image: [number, number][]; floor: [number, number][]; homography: number[][]; source?: "default" | "set" }>;
  zones: Record<string, Record<string, [number, number][]>>; // camera -> zone -> floor polygon (m)
  play: { walk_speed: number; body_radius: number; start: [number, number] };
  rooms: string[];
  uncalibrated: string[];
}

/** One tracked person or thing on the floor plan. */
export interface FloorTrack {
  key: string; // "<camera>:<track id>"
  camera: string;
  track_id: number;
  label: string;
  kind: "person" | "object";
  x: number;
  y: number;
  z: number; // metres above the floor
  on: string | null; // the furniture it rests on or sits in
  held_by: string | null;
  in_bin: string | null;
  seated: boolean;
  member_id: string | null;
}

export interface RoomRouteLeg {
  player: number;
  track: string | null;
  points: { x: number; y: number }[];
  distance_m: number;
  par_s: number;
  reachable: boolean;
}

export interface SeenOnFloor {
  label: string;
  x: number;
  y: number;
  z: number;
  on: string | null;
  camera: string;
  ts: string;
}

/** GET /api/room/live. */
export interface RoomLive {
  room: string;
  ts: string | null;
  tracks: FloorTrack[];
  seen: Record<string, SeenOnFloor>;
  /** `replans`: how many times this round's route has been planned again (GET /api/play/route-search has each plan). */
  round: { round_id: string; item: string; target_bin: string; bin: string; players: number; started: string; routes: RoomRouteLeg[]; replans?: number } | null;
  result: { round_id: string; item: string; correct: boolean; seconds: number; score: number; players: number; par_s: number | null; bin: string; ts: string; right: number; rounds: number } | null;
  assist: { item: string; label: string; bin: string; bin_id: string; color: string | null; side: string; track: string; person: string; text: string } | null;
}

/** GET /api/play/room-route. `points` is empty when no way exists. */
export interface RoomRouteAnswer {
  item: string;
  bin: string;
  bin_id: string;
  zone: string;
  room: string;
  from: { x: number; y: number; track: string | null };
  reachable: boolean;
  points: [number, number][];
  distance_m: number;
  par_s: number;
}

/**
 * How a route was found (contexts/play/room/grid.py: search_json). Cells are numbers, `row * cols + col`.
 * `blocked` is one string a row: "." free, "#" furniture or a wall (grown by the body radius), "o" something in the
 * way for this plan only. `explored` is the cells the A* search expanded, in the order it expanded them (thinned
 * evenly when there are more than the box sends; `explored_total` is the real count). `path` is the cells it settled on.
 */
export interface RouteSearch {
  algorithm: string;
  cell_m: number;
  cols: number;
  rows: number;
  blocked: string[];
  explored: number[];
  explored_total: number;
  path: number[];
}

/** Something the walk bends around: a piece of furniture, a person, a thing on the floor, or the wall. */
export interface RouteAround {
  id: string;
  kind: "furniture" | "person" | "thing" | "player" | "wall";
  label: string;
  x: number;
  y: number;
}

/** A person or a thing that was in the way when the route was planned (a disc on the floor, metres). */
export interface RouteObstacle {
  kind: "person" | "thing";
  label: string;
  key: string | null;
  x: number;
  y: number;
  r: number;
}

/** The three fields GET /api/play/room-route adds with `search=1`, and every plan of a round carries. */
export interface RouteFound {
  search: RouteSearch | null; // null when no way exists
  obstacles: RouteObstacle[];
  around: RouteAround[];
}

/** One plan of a round: the first, or a re-plan with what caused it. */
export interface RoundPlan extends RouteFound {
  seq: number;
  reason: "round_started" | "replan";
  ts: string;
  rerouted: { why: "blocked" | "off_route" | "opened"; kind: string | null; label: string | null; key: string | null } | null;
  from: { x: number; y: number; track: string | null };
  bin_id: string | null;
  reachable: boolean;
  points: [number, number][];
  distance_m: number;
  par_s: number;
}

/** GET /api/play/route-search: the plans of the round that is on (`live`), else of the last one. */
export interface RoundPlans {
  room: string;
  round_id: string | null;
  live: boolean;
  item?: string;
  bin?: string;
  players?: number;
  started?: string;
  plans: RoundPlan[];
}

/** One scored round of this sitting. `estimate_s` is the route's time estimate the score is measured against. */
export interface PlayedRound {
  round: number;
  item: string;
  correct: boolean;
  seconds: number;
  estimate_s: number;
  score: number;
  players: number;
  ts: string;
}

/**
 * GET /api/play/state: where Sort It's loop is, as of the newest face count. Counts and seconds only.
 * looking: nobody is facing the camera. holding: a face has been held for `held_s` of `hold_to_start_s`.
 * round: a round is on. ending: a round is on and nobody has faced the camera for `idle_s` of `idle_to_end_s`.
 */
export interface PlayState {
  phase: "looking" | "holding" | "round" | "ending";
  faces: number; // facing the camera in the newest count
  count: number; // faces in view in the newest count
  heard: string | null; // when that count was taken; null when no face count has arrived
  held_s: number;
  hold_to_start_s: number;
  idle_s: number;
  idle_to_end_s: number;
  round: { round_id: string; item: string; players: number; started: string } | null;
  rounds: { session_id: string; played: number; first: PlayedRound | null; latest: PlayedRound | null };
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
  /** Where the pack's drawn map is served ("/api/places/map/ucf"), for packs built from real map data. */
  map?: string;
}

/** GET /api/places/route: when to leave, and the walk itself along the real footpaths. */
export interface PlaceRoute {
  from: string;
  to: string;
  from_name: string;
  to_name: string;
  mode: "arrive_by" | "depart";
  depart: string;
  arrive: string;
  minutes: number;
  walk_minutes: number; // whole minutes, rounded up, as the day plan counts walks
  meters: number | null;
  path: string[];
  polyline: [number, number][]; // [lat, lon]; empty when the packs have no path network
  sentence: string; // "Leave by 2:41 PM: 9 minutes on foot to Engineering II, 720 m"
  from_xy: [number, number] | null;
  to_xy: [number, number] | null;
  attribution: string | null;
}

/** Every vocabulary word with its plural and capitalised forms, as GET /api/profile sends them in `words`. */
export interface Words {
  workplace: string;
  Workplace: string;
  workplace_plural: string;
  Workplace_plural: string;
  activity: string;
  Activity: string;
  activity_plural: string;
  Activity_plural: string;
  event: string;
  Event: string;
  event_plural: string;
  Event_plural: string;
  deadline: string;
  Deadline: string;
  deadline_plural: string;
  Deadline_plural: string;
}

/** GET /api/profile: who this box serves (config/profiles/<name>/), and the words its screens use. */
export interface Profile {
  name: string;
  description: string;
  audience: string;
  vocabulary: { workplace: string; activity: string; event: string; deadline: string; plurals: Record<string, string> };
  words: Words;
  meters: { habits: string[]; dayrun: string[] };
  shift: { start: string; end: string; sleep_after_minutes: number; sleep_hours: number } | null;
  scenario: string | null;
  /** The Ask screen's suggested questions, in this profile's words (profile.yaml `suggestions:`). */
  suggestions: string[];
  overrides: Record<string, string>;
}

/** GET /api/dayrun/scenarios/{name}: a scripted day's own presets, never a result. */
export interface ScenarioScript {
  name: string;
  now: string;
  start_place: string | null;
  plans: Record<string, Move[]>;
  disruptions: Disruption[];
  campaign: Move[][];
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
  /** The line actually walked, [lat, lon] points along the footpaths, when the pack has a path network. */
  path?: [number, number][] | null;
  meters?: number | null;
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
  meters?: number | null; // the length walked, when the pack has a path network
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
  meals: number; // the day's meals: those eaten before the board starts plus those in its window
  meals_before?: number; // of `meals`, how many were already eaten when the board starts
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
  requested?: number | null; // seeded disruptions asked for; null when the request listed its own
  note?: string; // set when fewer came back than asked for
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
  /** Numbers the person knows better than the box: `work_left` is focused minutes by deadline id. */
  initial?: { work_left?: Record<string, number> };
}

// ---------- Today's Plan: additive to Day Run (contexts/play/dayrun: fixes.py, words.py, talk.py, history.py) ----------

/** A "Near you" event that falls in an open slot: offered in the gap, never placed (Board.suggestions). */
export interface Suggestion extends OptionalEvent {
  slot: string | null;
  walk_minutes: number | null;
  leave_by: string | null;
  note: string | null;
}

/** Board fields added for Today's Plan. Older boxes do not send them. */
export interface PlanBoard extends Board {
  suggestions?: Suggestion[];
  deadlines: (Deadline & { estimate_from?: "kind" | "default" | "yours" | null; estimate_minutes?: number | null })[];
}

/** One thing to fix, in plain words, with the edit that would mend it (the whole plan with the edit applied). */
export interface Fix {
  id: string;
  problem: string;
  label: string | null;
  moves: Move[] | null;
  changed: number | null;
  mends: boolean;
  /** No edit of its own until another problem of the list is mended: that problem's id, and the line to show in place of a button. */
  after?: string | null;
  hint?: string | null;
}

/** POST /api/dayrun/explain: what needs fixing, and one true sentence per number a person might question. */
/** Where a number comes from: `id` to key on ("canvas", "guess", "yours", "watch", "camera", "map", "set", "plan" ...), `label` the words on the chip. */
export interface NumberSource {
  id: string;
  label: string;
}

export interface Explained {
  /**
   * The source of each number that has a sentence below (contexts/play/dayrun/words.py: source_of_*), read on the
   * box from the same field the sentence is written from. `due` is null when the deadline's item names no connector.
   */
  sources?: {
    deadlines: Record<string, { due: NumberSource | null; work: NumberSource }>;
    sleep: NumberSource;
    meals: NumberSource;
    points: NumberSource;
    walks: Record<string, NumberSource>;
  };
  headline: string;
  fixes: Fix[];
  deadlines: Record<string, { title: string; due: string; work: string; status: string | null; risk: string | null }>;
  sleep: string;
  meals: string;
  points: string;
  stars: string;
  walks: Record<string, string>;
}

/** POST /api/dayrun/edit: a change said in plain words, as the new plan. `ok` false: nothing changed, see `options`. */
export interface EditResult {
  ok: boolean;
  understood: string;
  moves: Move[];
  changed: number | null;
  options: { label: string; moves: Move[]; changed: number | null }[];
  warning: string | null;
  /** Nothing in the words could be read: things that can be said about this plan (and then no `options`). */
  examples?: string[];
}

/** One day of the date strip (GET /api/dayrun/days). The last six fields are only on days still ahead. */
export interface StripDay {
  day: string;
  when: "past" | "today" | "future";
  has_data: boolean;
  label: string;
  summary: string;
  won?: boolean;
  stars?: number | null;
  score?: number | null;
  fixed?: number;
  agreed?: number;
  optional?: number;
  free_minutes?: number;
  busy_minutes?: number;
  due?: { id: string; title: string; due: string; work_left: number }[];
  pressure?: number;
}

export interface DayBlock extends PlanMove {
  mark: "done" | "missed" | null;
  moved: boolean;
}

/** GET /api/dayrun/day?day=: a past day against its review, today's plan, or a draft for a day ahead. */
export interface PlanDay extends StripDay {
  start: string | null;
  moves: Move[];
  blocks: DayBlock[];
  review: { score: number | null; stars: number; points: number; plan_stars: number; followed: string[]; slipped: string[]; change_tomorrow: string } | null;
  changes: { at: string; why: string }[];
  first: { start: string; moves: Move[] } | null;
  outcome: PlanOutcome | null;
}

// ---------- the Day loop (contracts/day.py, contexts/day) ----------

/** One block of the day as the plan simulates it: a chosen move or a fixed calendar block. */
export interface PlanMove {
  id: string;
  kind: string; // study, eat, sleep, rest, event, class
  title: string;
  start: string;
  end: string;
  minutes: number;
  place: string | null;
  place_name: string | null;
  fixed: boolean;
  deadline: string | null;
  with_friend: string | null;
  /** "optional": an optional event the person said yes to and approved. */
  origin?: string | null;
}

/** Why the plan is what it is: one input the box already had. */
export interface PlanSource {
  kind: "item" | "habit" | "member" | "place" | "review" | "config";
  id: string;
  text: string;
  origin: string | null; // "canvas", "desk camera", "watch", "circle"
}

export interface PlanOutcome {
  won: boolean;
  stars: number;
  points: number;
  summary: string;
  sleep_hours: number;
  meals: number;
  study_minutes: number;
  energy_min: number;
  deadline_risk: number;
}

export interface DayMeters {
  meals: number;
  desk_minutes: number;
  recycled: number;
  sleep_hours: number | null;
  sleep_source: string | null;
}

export interface DayPlan extends EventEnvelope {
  type: "day.plan.proposed";
  day: string;
  plan_id: string;
  start: string;
  end: string;
  moves: PlanMove[];
  dayrun_moves: Move[]; // the chosen moves in Day Run's editor shape
  outcome: PlanOutcome;
  sources: PlanSource[];
  missing: string[];
  text: string;
  action_id: string | null;
  by: "atlas" | "person";
}

export interface DayReplan extends EventEnvelope {
  type: "day.replan.proposed";
  day: string;
  plan_id: string;
  replaces: string;
  why: string;
  start: string;
  moves: PlanMove[];
  dayrun_moves: Move[];
  outcome: PlanOutcome;
  action_id: string | null;
  asked: boolean;
  for_event: string | null;
  /** The optional event ("nearby:...") a yes asked to add, or a no asked to take out. */
  for_optional?: string | null;
}

export interface NextEvent {
  id: string;
  title: string;
  start: string;
  end: string;
  place: string | null;
  place_name: string | null;
  walk_minutes: number;
  leave_by: string | null;
}

export type DayRisk = "ok" | "leave_now" | "late" | "infeasible";

export interface DayStatus extends EventEnvelope {
  type: "day.status.changed";
  day: string;
  plan_id: string | null;
  now: string;
  block_id: string | null;
  block_title: string | null;
  block_end: string | null;
  should_be: string | null;
  should_be_name: string | null;
  seen_at: string | null;
  seen_at_name: string | null;
  seen_how: string | null;
  seen_ts: string | null;
  drift_minutes: number;
  next_block: NextEvent | null;
  next_event: NextEvent | null;
  risk: DayRisk;
  deadline_risk: number;
  meters: DayMeters;
  reason: string;
}

export type CoachTrigger = "chat" | "phone" | "door" | "food" | "none";

/** contracts.day.CoachEvidence. `title` and `frame_ref` are added by the API from Memory; the event carries ids only. */
export interface CoachEvidence {
  kind: Exclude<CoachTrigger, "none">;
  id: string;
  at: string;
  label: string;
  title?: string | null;
  frame_ref?: string | null;
}

export interface CoachBreak {
  start: string;
  end: string;
  minutes: number;
  trigger: CoachTrigger;
  evidence: CoachEvidence[];
}

/** contracts.day.SessionReview: one planned block of the main activity against the desk camera. */
export interface CoachSession {
  block_id: string;
  title: string;
  start: string;
  end: string;
  place: string | null;
  place_name: string | null;
  planned_minutes: number;
  measured: boolean;
  not_measured: string | null;
  desk_minutes: number | null;
  late_start_minutes: number | null;
  ended_early_minutes: number | null;
  longest_stretch_minutes: number | null;
  longest_stretch_start: string | null;
  breaks: CoachBreak[];
  text: string;
}

export interface CoachCalibration {
  score: number;
  ratio: number;
  blocks: number;
  text: string;
}

/** contracts.day.DayCoachNoted: the coach's sentence when a block ends. */
export interface DayCoachNote extends EventEnvelope {
  type: "day.coach.noted";
  day: string;
  plan_id: string;
  block_id: string;
  text: string;
  session: CoachSession;
  suggestions: string[];
  calibration: CoachCalibration | null;
  next_block: NextEvent | null;
}

/** GET /api/day/coach?day= */
export interface DayCoach {
  day: string;
  now: string;
  plan_id: string | null;
  words: { activity: string; workplace: string };
  sessions: CoachSession[];
  text: string;
  suggestions: string[];
  calibration: CoachCalibration | null;
  block_minutes: number | null;
  desk_minutes_today: number | null;
  notes: DayCoachNote[];
  latest: DayCoachNote | null;
  limits: string[];
}

export interface DayReview extends EventEnvelope {
  type: "day.reviewed";
  day: string;
  plan_id: string;
  score: number | null;
  followed: string[];
  slipped: string[];
  text: string;
  change_tomorrow: string;
  stars: number;
  points: number;
  plan_stars: number;
  streak: number;
  /** The study coach's part; absent on reviews written before it existed. */
  sessions?: CoachSession[];
  coach_text?: string;
  suggestions?: string[];
  calibration?: CoachCalibration | null;
  block_minutes?: number | null;
}

export interface LastSeen {
  episode_id: string;
  camera_id: string;
  start: string;
  end: string | null;
  labels: string[];
  caption: string | null;
  frame_ref: string | null;
}

/** GET /api/day/now */
export interface DayNow {
  day: string;
  now: string;
  plan: DayPlan | DayReplan | null;
  committed: boolean;
  status: DayStatus | null;
  replan: DayReplan | null;
  review: DayReview | null;
  yesterday: DayReview | null;
  friends: PlaceShared[];
  last_seen: LastSeen | null;
  /** The last block the coach spoke about, for a day after it ended; evidence already cited from Memory. */
  coach?: DayCoachNote | null;
  /** The blocks before that one in the last day, newest first. */
  coach_earlier?: DayCoachNote[];
  /** The check-in to show at the top of the day, when Atlas raised one today; absent on a box without it. */
  checkin?: CheckinCard | null;
  /**
   * With no check-in on screen: "You usually start to look strained around now.", when the person has been at the
   * desk about as long as it usually takes (the look memory's own counts). For the screen only; "" most of the time.
   */
  ahead?: string;
}

export type CheckinKind = "sleep" | "resting_hr" | "expression" | "sitting" | "drink" | "history";

/** contracts.day.CheckinSignal: one thing the box noticed, in plain words. Never a raw heart number. */
export interface CheckinSignal {
  kind: CheckinKind;
  summary: string;
  detail: string;
  since: string | null;
  minutes: number | null;
  evidence_refs: string[];
}

/** What is behind each chip, looked up on the box when this phone asks. Only the kinds the check-in has. */
export interface CheckinEvidence {
  sleep?: { asleep_min: number | null; in_bed_min: number | null; stages: Record<string, number>; usual_min: number | null; nights: number };
  /** `rel` is each day against the person's own baseline (1 = baseline). The values come only with `numbers=true`. */
  resting_hr?: { series: { day: string; rel: number }[]; days: number; baseline?: number; today?: number; values?: number[] };
  expression?: { windows: { start: string; end: string; samples: number; strained: number }[] };
  sitting?: { since: string | null; minutes: number | null };
  drink?: {
    since: string | null;
    picked_up: boolean;
    frame_ref: string | null;
    last_seen: { label: string; ts: string; frame_ref: string | null; zone_id: string | null; position: string | null } | null;
  };
  history?: { days: { day: string; text: string }[]; outcome: string; by: string | null };
}

/** contracts.day.DayCheckinRaised with its words, its answer and its evidence (GET /api/day/checkin, /api/day/now). Box-only. */
export interface CheckinCard extends EventEnvelope {
  type: "day.checkin.raised";
  day: string;
  at: string;
  signals: CheckinSignal[];
  suggestions: { kind: "drink" | "break" | "move_block" | "walk"; text: string }[];
  history: { matches: number; outcome: string; by: string | null; days: { day: string; text: string }[] } | null;
  proposes_replan: boolean;
  plan_id: string | null;
  text: string;
  answered: "done" | "move" | null;
  spoken: string | null;
  display: string;
  evidence: CheckinEvidence;
  /** An earlier look this one is like, from the look memory; null when none was similar enough, or it was forgotten since. */
  callback?: CheckinCallback | null;
  /** For the card only, never said aloud or pushed: "You usually start to look strained around now. A short break tends to help." */
  ahead?: string;
}

/** contracts.day.CheckinCallback: words, a day and a label. No picture, no health value. Box-only. */
export interface CheckinCallback {
  moment_id: string;
  day: string;
  look: string;
  said: string;
  learned: string;
  lead: "drink" | "break" | "move_block" | "walk" | null;
  /** "Looked tired · like Tue" */
  chip: string;
  /** The remembered moment in one or two plain lines: what was going on, what helped. */
  lines: string[];
  /** What the counts over every remembered look say, once there are enough (stage two): `learned` is then that sentence. */
  pattern?: string;
  /** "From 9 remembered looks, 8 with a known outcome."; "" while the check-in is stage one's single look back. */
  evidence?: string;
}

/** One remembered look (GET /api/day/looks): when, what was going on, what helped. Labels and numbers only. */
export interface LookMoment {
  id: string;
  look: string;
  at: string;
  day: string;
  /** "Tuesday afternoon", "Yesterday afternoon" */
  when: string;
  /** "3 PM", to five minutes */
  time: string;
  going_on: string;
  helped: string;
}

/** GET /api/day/looks: the look memory, most recent first. Box-only: read by the owner's screen, sent nowhere. */
export interface LooksView {
  on: boolean;
  count: number;
  kept_days: number;
  moments: LookMoment[];
  note: string;
  /** What Atlas has learned from them for this person; null while nothing is remembered. */
  learned?: LooksLearned | null;
}

/** One thing Atlas learned from the remembered looks, as a sentence with its counts. `finding` false: counts only, too few to call. */
export interface LookFinding {
  kind: string;
  text: string;
  eased: number;
  n: number;
  finding: boolean;
  /** "Likely between 57% and 95% of the time, on this little." */
  band: string;
}

/**
 * What Atlas has learned about one person's looks (contexts/day/domain/look_model.py): a table of counts worked out on
 * the box from the remembered looks, when asked. Stored nowhere, so forgetting the looks forgets it.
 */
export interface LooksLearned {
  enough: boolean;
  looks: number;
  known: number;
  left_out: number;
  min_known: number;
  /** "From 13 remembered looks, 10 with a known outcome. 3 are left out: I couldn't see how they went." */
  evidence: string;
  findings: LookFinding[];
  /** "Not enough yet. I need to see how 1 more look went before I say what helps (3 of 4 so far)."; "" with enough. */
  not_enough: string;
  note: string;
  shared_note: string;
}

/** GET /api/day/checkin */
export interface CheckinView {
  now: string;
  enabled: boolean;
  latest: CheckinCard | null;
  history: { id: string; at: string; day: string; text: string; signals: string[]; answered: "done" | "move" | null }[];
  watching: { signals: string[]; agree: boolean; held: string | null };
  rule: Record<string, number | string>;
  limits: string[];
}

/** GET /api/day/plan?day= : `moves` go straight into POST /api/dayrun/simulate. */
export interface DayPlanOut {
  day: string;
  plan_id: string;
  start: string;
  committed: boolean;
  moves: Move[];
  blocks: PlanMove[];
  outcome: PlanOutcome;
  plan: DayPlan | DayReplan;
}

/** One block of the plan an optional event would run into (GET /api/day/optional `fit`). */
export interface OptionalTouched {
  id: string;
  title: string;
  kind: string;
  start: string;
  end: string;
  overlap_minutes: number;
}

/** How an optional event sits in today's plan. "nearly": only blocks Atlas chose would move, or you would leave a little early. */
export interface OptionalFit {
  fits: "yes" | "nearly" | "no";
  reason: string;
  walk_minutes: number;
  /** null when the walk is under three minutes. */
  leave_by: string | null;
  from_place: string | null;
  displaces: OptionalTouched[];
  conflicts: OptionalTouched[];
  leave_early_minutes: number;
}

export type OptionalState = "suggested" | "accepted" | "declined" | "passed";

/** GET /api/day/optional: one optional event ("Near you"), from Memory's `nearby:` calendar items. */
export interface OptionalEvent {
  /** "nearby:<slug>:<start ISO>" */
  id: string;
  slug: string;
  title: string;
  start: string;
  end: string;
  place: string | null;
  place_name: string | null;
  tags: string[];
  going: string[];
  host: string | null;
  source: string | null;
  /** accepted only once the card a yes raised was approved. */
  state: OptionalState;
  answer: "yes" | "no" | null;
  /** A card about this event (the re-plan that adds it, or takes it out) is open. */
  awaiting_approval: boolean;
  action_id: string | null;
  plan_id: string | null;
  /** When Atlas brought it up by itself, if it did. */
  suggested_at: string | null;
  /** Why Atlas is not bringing it up right now; null when it may. */
  suggest_held: string | null;
  fit: OptionalFit;
}

export interface OptionalView {
  now: string;
  plan_id: string | null;
  events: OptionalEvent[];
}

/** What a yes would do to one block Atlas had chosen. Times are local "HH:MM". */
export interface OptionalChange {
  id: string;
  title: string;
  kind: string;
  fixed: boolean;
  change: "moved" | "shortened" | "lengthened" | "dropped";
  start: string;
  end: string;
  new_start?: string;
  new_end?: string;
}

/** POST /api/day/optional/{id}. A yes never changes the plan: `action_id` is the approval card it raised. */
export interface OptionalAnswer {
  id: string;
  answer: "yes" | "no";
  state: OptionalState;
  /** A re-plan with its approval card was raised. */
  proposed: boolean;
  /** Why nothing was proposed (a yes to an event more than a day ahead). */
  reason?: string;
  plan_id?: string;
  action_id?: string | null;
  moves?: OptionalChange[];
  adds?: { id: string; title: string; kind: string; start: string; end: string }[];
  block?: PlanMove;
  fit?: OptionalFit;
}

export type DayMoment = "morning" | "now" | "replan" | "coach" | "evening" | "approval" | "checkin" | "suggest" | "start";

/** contracts.day.DayNarrated: one moment of the day in words, for the ear (`spoken`) and for the screen (`display`). */
export interface DayNarration extends EventEnvelope {
  type: "day.narrated";
  day: string;
  moment: DayMoment;
  plan_id: string | null;
  /** The source event's id, or the action id of an approval ask. */
  ref: string | null;
  spoken: string;
  display: string;
  engine: "template" | "model";
  facts_hash: string;
  /** A "now" worth saying aloud unasked: leave now, late. */
  urgent: boolean;
  /** "box" when the box's own speaker says it: the phone stays silent for this line. */
  spoken_by?: "box" | null;
  /** Said inside that narration (its event id): shown, never said or pushed again. */
  merged_into?: string | null;
}

/** GET /api/day/brief */
export interface DayBrief {
  mode: "model" | "template" | "off";
  engine: "model" | "template" | "off";
  timeout_s: number;
  gap_s: number;
  latest: Partial<Record<DayMoment, DayNarration>>;
  asks: Record<string, DayNarration>;
  model: { calls: number; used: number; unchanged?: number; rejected: number; timeouts: number; errors: number; last_problem: string | null };
}

export interface DayPattern {
  text: string;
  n: number;
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

/**
 * A piece of an answer while it is being written, on /ws only. Not an event: it is never logged and has no
 * `event_id`, `version` or `ts`. The pieces of one question, in order, spell its answer; the final
 * `agent.answer.generated` (and the reply to POST /api/ask) carries the whole text and replaces them.
 */
export interface AnswerDelta {
  type: "agent.answer.delta";
  question_id: string;
  text: string;
}

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

/** GET /runtime/models (R7): what the box has installed and what llama-server really serves. */
export interface RuntimeModels {
  current: string | null;
  previous: string | null;
  /** The folder llama-server reports it loaded; null when it did not answer. */
  serving: string | null;
  in_sync: boolean | null;
  versions: { version: string; base: string | null; task: string | null; scores: Record<string, number>; current: boolean }[];
}

export interface VaultPolicy {
  blur_faces: boolean;
  paused_cameras: string[];
  offline: boolean;
  /** Days of pictures and memory to keep; null = forever. */
  keep_days: number | null;
  /**
   * The live picture on the owner's own signed-in screen: "clear" shows every face there, "owner" only the face of
   * the person who did the face scan (everyone else blurred), "blurred" nobody's. Everyone else: always blurred.
   */
  live_faces: LiveFaces;
  /** A training export you approve: "owner_clear" lets your own scanned face stay unblurred in it. Off ("blurred") by default. */
  train_faces?: "blurred" | "owner_clear";
  /** Read-only, from the box: has the owner done the face scan, and can the box tell them right now. */
  owner?: FaceOwner;
  /** Read-only: what a signed-in screen gets right now ("owner" falls back to "blurred" until the scan is done). */
  live_faces_now?: LiveFaces;
  /** Read-only: one line saying why an owner-only choice is not in effect, or null. */
  faces_note?: string | null;
}

export type LiveFaces = "blurred" | "clear" | "owner";

export interface FaceOwner {
  scanned: boolean;
  /** Scanned, and recognition is on: "only me" can be honoured now. */
  ready: boolean;
  /** Why not, in one line ("Do the face scan first."). */
  reason: string | null;
  /** The demo seed's owner: made-up numbers, nobody's face. */
  demo: boolean;
}

/** What the owner may change with PUT /vault/policy (the read-only fields are the box's to say). */
export type VaultPolicyChange = Partial<Pick<VaultPolicy, "blur_faces" | "paused_cameras" | "offline" | "keep_days" | "live_faces" | "train_faces">>;

/** "Everyone clear" is chosen: the owner's signed-in screen shows every face. The default. */
export function liveClear(policy: Pick<VaultPolicy, "live_faces">): boolean {
  return policy.live_faces === "clear";
}

/** GET /api/voice/status: the box's own microphone and speaker (X16). "off" also means the voice service is not running. */
export interface VoiceStatus {
  state: "off" | "idle" | "listening" | "thinking" | "speaking";
  running: boolean;
  /** Why the last turn did not work, in words for the screen; "" when fine. */
  problem: string;
  heard: string;
  answer: string;
  seconds_in_state: number;
  /** True when the box is always listening for "Hey Atlas" (wake_word in config/voice.yaml). */
  wake?: boolean;
  /** True while the box listens for a reply after its answer, with no "Hey Atlas" (follow_up in config/voice.yaml). */
  follow_up?: boolean;
  mic?: { ok: boolean; detail: string };
  /** `absent`: the configured Bluetooth speaker is just not connected (off or unpaired), not broken. */
  speaker?: { ok: boolean; detail: string; absent?: boolean };
  speech_to_text?: { ok: boolean; detail: string };
  text_to_speech?: { ok: boolean; detail: string };
}

/** GET /api/zones: polygons per camera in that camera's picture, 0..1, origin top left (config/zones.yaml). */
export type ZoneSet = Record<string, Record<string, [number, number][]>>;

export interface EgressGroup {
  /** "lan", "ntp", "dns", "hub", "connector:calendar", ... */
  name: string;
  hosts: string[];
  packets: number;
  /** Every byte sent to the group (V7: counted ahead of the conntrack accept). */
  bytes: number;
  /** Connections opened to it. Absent from older firewall snapshots. */
  connections?: number;
}

/** V7: camera pictures cleared to leave the box, each batch by an approved training export. */
export interface EgressPictures {
  video_bytes: number;
  pictures: number;
  picture_bytes: number;
  approved_exports: number;
  batches: { dataset_id: string; task: string; pictures: number; bytes: number; approval: string; ts: string }[];
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
  /** Present on boxes with V7, firewall or not. */
  pictures?: EgressPictures;
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

export type LearningStage = "waiting for approval" | "exported" | "labeling" | "training" | "eval" | "packaging" | "deploying" | "done" | "failed" | "denied" | "expired";

/** One taught task, folded from the learning.* events (GET /api/learning). */
export interface LearningTask {
  task: string;
  instruction: string;
  stage: LearningStage;
  /** 0..1 within the stage. */
  progress: number;
  loss?: number | null;
  message?: string | null;
  dataset_id?: string;
  examples?: number;
  /** e.g. {current: 0.41, candidate: 0.78, teacher: 0.86} */
  scores?: Record<string, number>;
  decision?: "deploy" | "reject";
  model_version?: string;
  deployed?: boolean;
  created: string;
  updated: string;
}

export interface LearningBoard {
  tasks: LearningTask[];
  deployments: { model_version: string; previous_version: string | null; ts: string }[];
}

export interface TeachBody {
  /** A slug: "desk-objects". */
  task: string;
  instruction: string;
  camera?: string;
  label?: string;
  examples: { frame_ref: string; verdict: "up" | "down" }[];
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

/** GET /api/files -> contexts.workspace.domain.files.FileRecord (W1) */
export interface InboxFile {
  file_id: string;
  name: string;
  content_type: string;
  size: number;
  sha256: string;
  added_at: string;
  text_chars: number;
  extractor: string | null;
  excerpt: string;
}

/** POST /api/files: the file as the raw body (no multipart), its name in the query. */
export async function uploadFile(file: File): Promise<InboxFile & { duplicate: boolean }> {
  const headers: Record<string, string> = { "Content-Type": file.type || "application/octet-stream" };
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  let res: Response;
  try {
    res = await fetch(HTTP_BASE + withQuery("/api/files", { name: file.name }), { method: "POST", headers, body: file });
  } catch {
    throw new ApiError(0, "Can't reach the Atlas box. Check that you are on the same network.");
  }
  if (!res.ok) {
    let detail = res.statusText || `HTTP ${res.status}`;
    try {
      const data = (await res.json()) as { detail?: unknown };
      if (typeof data.detail === "string") detail = data.detail;
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as InboxFile & { duplicate: boolean };
}

/** Where the phone downloads an inbox file, decrypted by the box. */
export const fileUrl = (fileId: string) => `${HTTP_BASE}/api/files/${encodeURIComponent(fileId)}`;

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

// ---------- the clear live view: a ticket for this signed-in screen ----------
//
// The live picture is an <img>, which cannot send the Authorization header the rest of the app signs in with. So
// the app asks POST /vault/view-ticket (with that header) for a short-lived ticket and puts it on the picture's
// address. With a ticket, and while the Privacy screen's "Your live view" is on, the box leaves faces unblurred
// in that picture. Without one (not signed in, the box refused, the box is unreachable) the address is the old
// one and the picture is the blurred one, exactly as before. Nothing about saved pictures depends on this.

type TicketProblem = "" | "not_signed_in" | "box_has_no_token" | "unreachable";
let viewTicket: { value: string; until: number } | null = null; // `until`: this device's clock, ms
let viewTicketAsk: Promise<void> | null = null;
let viewTicketRetryAt = 0;
let viewTicketProblem: TicketProblem = "";
const TICKET_RENEW_BEFORE_MS = 120_000; // ask again two minutes before it runs out
const TICKET_RETRY_MS = 60_000; // after a refusal, not more often than this

function askViewTicket(): Promise<void> {
  if (viewTicketAsk) return viewTicketAsk;
  viewTicketAsk = post<{ ticket: string; expires_in: number }>("/vault/view-ticket", {})
    .then((t) => {
      viewTicket = { value: t.ticket, until: Date.now() + t.expires_in * 1000 };
      viewTicketProblem = "";
    })
    .catch((err) => {
      const status = err instanceof ApiError ? err.status : 0;
      viewTicketProblem = status === 401 ? "not_signed_in" : status === 403 ? "box_has_no_token" : "unreachable";
      viewTicketRetryAt = Date.now() + TICKET_RETRY_MS;
    })
    .finally(() => {
      viewTicketAsk = null;
    });
  return viewTicketAsk;
}

/** The ticket to put on a live picture's address, or "" for the blurred picture. Renews itself in the background. */
function currentViewTicket(): string {
  if (typeof window === "undefined") return "";
  const now = Date.now();
  if ((!viewTicket || viewTicket.until - now < TICKET_RENEW_BEFORE_MS) && now >= viewTicketRetryAt) void askViewTicket();
  return viewTicket && viewTicket.until > now ? viewTicket.value : "";
}

/** Resolves once this screen either holds a ticket or was told it gets none. Never rejects. */
export function viewTicketReady(): Promise<void> {
  if (typeof window === "undefined" || currentViewTicket()) return Promise.resolve();
  return viewTicketAsk ?? Promise.resolve();
}

/** Why this screen has no ticket ("" when it holds one, or has not asked yet). */
export function viewTicketState(): { held: boolean; problem: TicketProblem } {
  return { held: currentViewTicket() !== "", problem: viewTicketProblem };
}

/**
 * The newest frame from a camera; add a changing query so the browser refetches. Faces are blurred by the box
 * unless this is the owner's signed-in screen and the Privacy screen's "Your live view" is on.
 */
export function snapshotUrl(camera: string, stamp: number): string {
  const ticket = currentViewTicket();
  return `${HTTP_BASE}/capture/${encodeURIComponent(camera)}/snapshot.jpg?t=${stamp}${ticket ? `&ticket=${encodeURIComponent(ticket)}` : ""}`;
}

export const api = {  health: (signal?: AbortSignal) => get<Health>("/health", undefined, signal),
  profile: (signal?: AbortSignal) => get<Profile>("/api/profile", undefined, signal),
  captureStatus: (signal?: AbortSignal) => get<{ cameras: Record<string, CameraStatus> }>("/capture/status", undefined, signal),
  liveTracks: (camera: string, signal?: AbortSignal) => get<LiveTracks>(`/capture/${encodeURIComponent(camera)}/tracks`, undefined, signal),

  ask: (text: string) => post<Answer>("/api/ask", { text }),

  zones: (signal?: AbortSignal) => get<ZoneSet>("/api/zones", undefined, signal),
  setZones: (camera: string, zones: Record<string, [number, number][]>) => put<ZoneSet>(`/api/zones/${encodeURIComponent(camera)}`, { zones }),
  demoZones: () => post<ZoneSet>("/api/zones/demo", {}),
  vaultPolicy: (signal?: AbortSignal) => get<VaultPolicy>("/vault/policy", undefined, signal),
  runtimeModels: (signal?: AbortSignal) => get<RuntimeModels>("/runtime/models", undefined, signal),
  /** GET /runtime/stats (R5): the box's and the lab's numbers, each with its record (types in lib/stats.ts). */
  runtimeStats: (signal?: AbortSignal) => get<import("./stats").RuntimeStats>("/runtime/stats", undefined, signal),
  voiceStatus: (signal?: AbortSignal) => get<VoiceStatus>("/api/voice/status", undefined, signal),
  voiceTalk: () => post<VoiceStatus>("/api/voice/talk", {}),
  voiceStop: () => post<VoiceStatus>("/api/voice/stop", {}),
  voiceEnabled: (enabled: boolean) => post<VoiceStatus>("/api/voice/enabled", { enabled }),
  voiceListen: () => post<VoiceStatus>("/api/voice/listen", {}),
  voiceSay: (text: string) => post<VoiceStatus>("/api/voice/say", { text }),
  setVaultPolicy: (body: VaultPolicyChange) => put<VaultPolicy>("/vault/policy", body),
  egress: (signal?: AbortSignal) => get<Egress>("/vault/egress", undefined, signal),
  wipeRequest: (scope: WipeScope) => post<WipeToken>("/vault/wipe/request", { scope }),
  wipe: (token: string) => post<WipeResult>("/vault/wipe", { token }),

  learning: (signal?: AbortSignal) => get<LearningBoard>("/api/learning", undefined, signal),
  /** The Learning lab: training runs as stories (types in lib/lab.ts). The screen always passes sample=false. */
  lab: (sample: boolean, signal?: AbortSignal) => get<import("./lab").LabList>("/api/learning/lab", sample ? { sample: 1 } : undefined, signal),
  labRun: (runId: string, sample: boolean, signal?: AbortSignal) =>
    get<import("./lab").LabStory>(`/api/learning/lab/${encodeURIComponent(runId)}`, sample ? { sample: 1 } : undefined, signal),
  /** Two runs side by side (L31); without a or b, the latest solo run against the latest pooled one (types in lib/labCompare.ts). */
  labCompare: (a: string | null, b: string | null, signal?: AbortSignal) =>
    get<import("./labCompare").LabCompare>("/api/learning/lab/compare", { a, b }, signal),
  /** The lab's benchmark and energy rows, the routing what-if and the experiment timeline (types in lib/labBench.ts). */
  labBench: (signal?: AbortSignal) => get<import("./labBench").LabBench>("/api/learning/lab/bench", undefined, signal),
  labRoutingSim: (params: Partial<import("./labBench").SimParams>, signal?: AbortSignal) =>
    post<import("./labBench").LabSim>("/api/learning/lab/routing/simulate", params, signal),
  labExperiments: (signal?: AbortSignal) => get<import("./labBench").LabTimeline>("/api/learning/lab/experiments", undefined, signal),
  /** The On the box card's deploy button (L35): versions on disk, the links, the last or running swap (lib/labDeploy.ts). */
  labDeploy: (signal?: AbortSignal) => get<import("./labDeploy").DeployView>("/api/learning/lab/deploy", undefined, signal),
  labDeployStart: (version: string) => post<import("./labDeploy").DeployView>("/api/learning/lab/deploy", { version }),
  labDeployRollback: (version?: string) =>
    post<import("./labDeploy").DeployView>("/api/learning/lab/deploy/rollback", version ? { version } : {}),
  labDeployAsk: (question: string, signal?: AbortSignal) =>
    post<import("./labDeploy").DeployAnswer>("/api/learning/lab/deploy/ask", { question }, signal),
  teach: (body: TeachBody) => post<{ task: string; action_id: string }>("/api/teach", body),

  episodes: (opts: { from?: string; to?: string; camera?: string; label?: string } = {}, signal?: AbortSignal) =>
    get<Episode[]>("/memory/episodes", opts, signal),
  stats: (signal?: AbortSignal) => get<MemoryStats>("/memory/stats", undefined, signal),
  digest: (day?: string, signal?: AbortSignal) => get<DailyDigest>("/api/digest", day ? { day } : undefined, signal),
  search: (q: string, limit = 20, signal?: AbortSignal) => get<SearchHit[]>("/memory/search", { q, limit }, signal),
  files: (signal?: AbortSignal) => get<InboxFile[]>("/api/files", undefined, signal),
  items: (opts: { from?: string; to?: string; kind?: string } = {}, signal?: AbortSignal) =>
    get<Item[]>("/memory/items", opts, signal),

  habitsToday: (signal?: AbortSignal, member?: string) => get<HabitsToday>("/api/habits/today", { member }, signal),

  // People: opt-in owner recognition (P9).
  identitySettings: (signal?: AbortSignal) => get<IdentitySettings>("/identity/settings", undefined, signal),
  setIdentity: (on: boolean) => request<IdentitySettings>("PUT", "/identity/settings", { identity_enabled: on }),
  /** The anonymous room mode. Switching it on has to be confirmed by the person, on the screen that says what it means. */
  setEveryone: (on: boolean, confirm = false) => put<IdentitySettings>("/identity/settings", { everyone: on, confirm }),
  members: (signal?: AbortSignal) => get<Member[]>("/identity/members", undefined, signal),
  removeMember: (id: string) => request<{ removed: string }>("DELETE", `/identity/members/${encodeURIComponent(id)}`),
  updateMember: (id: string, change: { expressions?: boolean; circle_member_id?: string }) =>
    put<Member>(`/identity/members/${encodeURIComponent(id)}`, change),
  /** `consent`: the person being enrolled agrees. The box refuses without it. */
  enrollStart: (name: string, consent: boolean, circleMemberId?: string) =>
    post<EnrollStatus>("/identity/enroll/start", { name, consent_ack: consent, ...(circleMemberId ? { circle_member_id: circleMemberId } : {}) }),
  /** Photos as base64 (no data: prefix needed). For someone new send `name`; to add to someone send `member_id`. */
  enrollPhotos: (who: { name?: string; member_id?: string; circle_member_id?: string }, photos: string[], consent: boolean) =>
    post<EnrollPhotosResult>("/identity/enroll/photos", { ...who, consent_ack: consent, photos }),
  scanStart: (who: { name?: string; member_id?: string; circle_member_id?: string }, consent: boolean) =>
    post<ScanStatus>("/identity/enroll/scan/start", { ...who, consent_ack: consent }),
  scanFrame: (session: string, photo: string, signal?: AbortSignal) =>
    post<ScanStatus>("/identity/enroll/scan/frame", { session, photo }, signal),
  scanFinish: (session: string) => post<EnrollPhotosResult>("/identity/enroll/scan/finish", { session }),
  scanCancel: (session: string) => post<{ cancelled: boolean }>("/identity/enroll/scan/cancel", { session }),
  expressions: (memberId: string | null, hours: number, signal?: AbortSignal) =>
    get<ExpressionSeries>("/identity/expressions", { member_id: memberId ?? undefined, hours }, signal),
  presentNow: (signal?: AbortSignal) => get<PresentNow>("/identity/present", undefined, signal),
  enrollStatus: (signal?: AbortSignal) => get<EnrollStatus>("/identity/enroll/status", undefined, signal),
  impact: (signal?: AbortSignal) => get<Impact>("/api/impact", undefined, signal),
  impactStory: (member?: string, signal?: AbortSignal) => get<import("./impact").ImpactStory>("/api/impact/story", { member }, signal),
  impactPractice: (member?: string, signal?: AbortSignal) => get<import("./impact").ImpactPractice>("/api/impact/practice", { member }, signal),
  addPractice: (label: string, item_id?: string, member?: string) =>
    post<{ added: string; practice: import("./impact").ImpactPractice }>("/api/impact/practice", { label, item_id, member }),

  approvals: (signal?: AbortSignal) => get<ApprovalRequest[]>("/api/approvals", undefined, signal),
  decide: (id: string, decision: Decision) => post<ApprovalRequest>(`/api/approvals/${encodeURIComponent(id)}`, { decision }),

  presence: (signal?: AbortSignal) => get<PlaceShared[]>("/api/presence", undefined, signal),
  sharePlace: (place: string, hours: number) => post<PlaceShared>("/api/presence", { place, hours }),
  leaderboard: (signal?: AbortSignal) => get<Leaderboard>("/api/circle/leaderboard", undefined, signal),
  circleEvents: (signal?: AbortSignal) => get<CircleEvents>("/api/circle/events", undefined, signal),
  circleSharing: (signal?: AbortSignal) => get<CircleSharing>("/api/circle/sharing", undefined, signal),
  circleThreads: (signal?: AbortSignal) => get<CircleThreads>("/api/circle/threads", undefined, signal),
  circleThread: (id: string, signal?: AbortSignal) => get<CircleThreadDetail>(`/api/circle/threads/${encodeURIComponent(id)}`, undefined, signal),
  circleMembers: (signal?: AbortSignal) => get<CircleMembers>("/api/circle/members", undefined, signal),
  circleBrain: (signal?: AbortSignal) => get<BrainStatus>("/api/circle/brain", undefined, signal),
  circleBrainConsent: (on: boolean) => post<BrainStatus>("/api/circle/brain/consent", { on }),
  circleBrainAsk: (question: string, facts: string[] = []) => post<BrainAnswerReply>("/api/circle/brain/ask", { question, facts }),
  circleBrainPower: (after?: number, signal?: AbortSignal) =>
    get<BrainPower>("/api/circle/brain/power", after === undefined ? undefined : { after: String(after) }, signal),
  /** The two-pod brain live (G6): members, watts, link, serving (types in lib/sharedBrain.ts); 200 in every state. */
  circleSharedBrain: (signal?: AbortSignal) => get<import("./sharedBrain").SharedBrainPair>("/api/circle/shared-brain", undefined, signal),
  /** Face share, or null while it is switched off on this box (409), there is no Circle, or the box is older than it (404). */
  faceShare: (signal?: AbortSignal) =>
    get<FaceShareStatus>("/api/circle/face-share", undefined, signal).catch((err: unknown) => {
      if (err instanceof ApiError && (err.status === 409 || err.status === 404)) return null;
      throw err;
    }),
  /** Raises an approval card on this box; the template is sealed and sent only when it is approved. */
  faceShareAsk: (friend: string) => post<{ action_id: string }>(`/api/circle/face-share/${encodeURIComponent(friend)}/share`, undefined),
  faceShareRevoke: (friend: string) => post<{ revoked: string }>(`/api/circle/face-share/${encodeURIComponent(friend)}/revoke`, undefined),
  /** Raises the Accept card again for a friend's offer that is still waiting. */
  faceShareReview: (friend: string) => post<{ action_id: string }>(`/api/circle/face-share/${encodeURIComponent(friend)}/review`, undefined),
  faceShareForget: (friend: string) => post<{ forgotten: string }>(`/api/circle/face-share/${encodeURIComponent(friend)}/forget`, undefined),
  faceShareTrust: (friend: string, fingerprint: string) =>
    post<{ trusted: string }>(`/api/circle/face-share/${encodeURIComponent(friend)}/trust-key`, { fingerprint }),
  /** Walking route to where a friend is sharing (404 when they are not, or the place is on no map). */
  routeToFriend: (from: string, member: string, signal?: AbortSignal) =>
    get<CampusRoute>("/api/play/campus-route", { from, to: `friend:${member}` }, signal),
  /** Raises an approval card on the box; nothing is sent to the Circle until it is approved. */
  proposeTime: (body: ProposalBody) => post<{ action_id: string; thread_id: string }>("/api/circle/proposals", body),

  // The League. The two POSTs raise approval cards on the box; nothing changes until they are approved.
  league: (signal?: AbortSignal) => get<LeagueView>("/api/league", undefined, signal),
  leagueMe: (signal?: AbortSignal) => get<LeagueMe>("/api/league/me", undefined, signal),
  setStake: (kind: StakeKind, text?: string) => post<{ action_id: string; reason: string }>("/api/league/stake", { kind, text }),
  startChallenge: (body: { kind: string; days: number; invitees: string[]; target?: number }) =>
    post<{ action_id: string; challenge_id: string; reason: string }>("/api/league/challenges", body),

  playRoute: (item: string, signal?: AbortSignal) => get<PlayRoute>("/api/play/route", { item }, signal),
  playAccuracy: (signal?: AbortSignal) => get<PlayAccuracy>("/api/play/accuracy", undefined, signal),
  room: (signal?: AbortSignal) => get<RoomPlan>("/api/room", undefined, signal),
  roomLive: (signal?: AbortSignal) => get<RoomLive>("/api/room/live", undefined, signal),
  roomRoute: (item: string, fromTrack?: string | null, signal?: AbortSignal) => get<RoomRouteAnswer>("/api/play/room-route", { item, from_track: fromTrack ?? undefined }, signal),
  /** The same walk with how it was found: the grid, the cells A* looked at, what it went around. */
  roomRouteFound: (item: string, signal?: AbortSignal) => get<RoomRouteAnswer & RouteFound>("/api/play/room-route", { item, search: "1" }, signal),
  roundPlans: (signal?: AbortSignal) => get<RoundPlans>("/api/play/route-search", undefined, signal),
  playState: (signal?: AbortSignal) => get<PlayState>("/api/play/state", undefined, signal),
  setRoomCalibration: (camera: string, taps: { image: [number, number][]; floor: [number, number][] }) =>
    put<RoomPlan>(`/api/room/calibration/${encodeURIComponent(camera)}`, taps),

  places: (signal?: AbortSignal) => get<PlacePack[]>("/api/places", undefined, signal),
  /** A pack's drawn map, by the path GET /api/places gave for it. Fetch it through lib/campusmap's loadCampusMap, which keeps it. */
  placeMap: (path: string, signal?: AbortSignal) => get<CampusMapData>(path, undefined, signal),
  /** When to leave: `from` and `to` are pack ids ("ucf.eng2"), names, "friend:<member>" or "here". */
  placeRoute: (q: { from?: string; to: string; arrive_by?: string; depart?: string }, signal?: AbortSignal) => get<PlaceRoute>("/api/places/route", q, signal),
  lastSeen: (label: string, signal?: AbortSignal) => get<Sighting>(`/memory/objects/${encodeURIComponent(label)}/last-seen`, undefined, signal),
  /** The same sighting for marking it in the picture: its box when one is known (else null), and the other boxed things in that frame. */
  whereSeen: (label: string, signal?: AbortSignal) => get<import("./guide").WhereSeen>(`/memory/last-seen/${encodeURIComponent(label)}`, undefined, signal),
  /** The detector's newest tracks for a camera: normalised boxes with labels, for drawing over its snapshot. */
  captureTracks: (camera: string, signal?: AbortSignal) =>
    get<{ camera_id: string; ts_ms: number | null; tracks: { label: string; track_id: number; confidence: number; box: import("./guide").BoxN }[] }>(`/capture/${encodeURIComponent(camera)}/tracks`, undefined, signal),

  // Day Run. `scenario` runs a scripted day from fixtures/play/ ("demo") instead of live sources.
  dayrunBoard: (scenario?: string, signal?: AbortSignal) => get<Board>("/api/dayrun/board", { scenario }, signal),
  planBoard: (q: { start?: string; scenario?: string }, signal?: AbortSignal) => get<PlanBoard>("/api/dayrun/board", q, signal),
  planExplain: (body: PlanBody, signal?: AbortSignal) => post<Explained>("/api/dayrun/explain", body, signal),
  planEdit: (body: PlanBody & { text: string }) => post<EditResult>("/api/dayrun/edit", body),
  planDays: (signal?: AbortSignal) => get<{ today: string; days: StripDay[] }>("/api/dayrun/days", undefined, signal),
  planDay: (day: string, signal?: AbortSignal) => get<PlanDay>("/api/dayrun/day", { day }, signal),
  dayrunScenario: (name: string, signal?: AbortSignal) => get<ScenarioScript>(`/api/dayrun/scenarios/${encodeURIComponent(name)}`, undefined, signal),
  dayrunSimulate: (body: PlanBody, signal?: AbortSignal) => post<SimResult>("/api/dayrun/simulate", body, signal),
  dayrunHard: (body: PlanBody & { seed?: number; count?: number; disruptions?: Disruption[] }) => post<HardResult>("/api/dayrun/hard", body),
  dayrunCommit: (body: PlanBody) => post<CommitResult>("/api/dayrun/commit", body),
  dayrunReview: (day: string, scenario?: string) => get<Review>("/api/dayrun/review", { day, scenario }),
  dayrunReplay: (body: { day: string; moves: Move[]; scenario?: string }) => post<ReplayResult>("/api/dayrun/replay", body),
  dayrunCampaign: (body: { days: Move[][]; start?: string; scenario?: string }) => post<CampaignResult>("/api/dayrun/campaign", body),
  dayrunForecast: (body: { estimates: { block: string; minutes: number }[]; measured?: Record<string, number>; day?: string; scenario?: string }) =>
    post<ForecastResult>("/api/dayrun/forecast", body),

  // The Day loop (D1): the day Atlas runs unasked.
  dayNow: (signal?: AbortSignal) => get<DayNow>("/api/day/now", undefined, signal),
  dayPlan: (day?: string, signal?: AbortSignal) => get<DayPlanOut>("/api/day/plan", { day }, signal),
  /** {} remakes today's plan; {moves} replaces it with your own; {from_here: true} re-plans the rest of the day. Each raises one card. */
  dayMakePlan: (body: { moves?: Move[]; from_here?: boolean } = {}) =>
    post<{ plan_id: string; type: string; action_id: string | null; plan: DayPlan | DayReplan }>("/api/day/plan", body),
  dayCoach: (day?: string, signal?: AbortSignal) => get<DayCoach>("/api/day/coach", { day }, signal),
  /** The day in words: the latest narration of each moment and the asks for pending cards. */
  dayBrief: (signal?: AbortSignal) => get<DayBrief>("/api/day/brief", undefined, signal),
  /** Narrate a moment again from the state as it is now ("Brief me"). `fast` never waits for the model. */
  dayBriefMake: (body: { moment: Exclude<DayMoment, "approval">; fast?: boolean }) =>
    post<{ mode: string; engine: string; moment: DayMoment; fresh: boolean; narration: DayNarration }>("/api/day/brief", body),
  coachFeedback: (days?: number, signal?: AbortSignal) => get<import("./coachFeedback").CoachFeedbackView>("/api/day/coach/feedback", { days }, signal),
  coachCorrect: (body: import("./coachFeedback").CoachCorrectionBody) =>
    post<import("./coachFeedback").CoachCorrectionResult>("/api/day/coach/corrections", body),
  dayPatterns: (signal?: AbortSignal) => get<{ patterns: DayPattern[]; note?: string }>("/api/day/patterns", undefined, signal),
  /** The check-in with its evidence. `numbers` asks for the resting heart rate's values: only after an explicit tap, on this device. */
  dayCheckin: (numbers = false, signal?: AbortSignal) => get<CheckinView>("/api/day/checkin", numbers ? { numbers: "true" } : undefined, signal),
  dayCheckinDone: (id: string) => post<{ id: string; answered: "done" }>("/api/day/checkin/done", { id }),
  /** The looks Atlas remembers, most recent first (the look memory). */
  dayLooks: (memberId: string, signal?: AbortSignal) => get<LooksView>("/api/day/looks", { member: memberId }, signal),
  /** Forget one remembered look, or all of them. Each goes through the signed-in route and is kept forgotten on a restart. */
  dayForgetLook: (id: string) => request<{ forgotten: string; left: number }>("DELETE", `/api/day/looks/${encodeURIComponent(id)}`),
  dayForgetLooks: (memberId: string) => request<{ forgotten: number; left: number }>("DELETE", withQuery("/api/day/looks", { member: memberId })),
  /** "Move the block": raises a re-plan with its approval card. Nothing changes until that card is approved. */
  dayCheckinMove: (id: string) => post<{ id: string; answered: "move"; plan_id: string; action_id: string | null }>("/api/day/checkin/move", { id }),
  /** Optional events ("Near you") for today and the next days, each with its fit, walk, leave-by and state. */
  dayOptional: (signal?: AbortSignal) => get<OptionalView>("/api/day/optional", undefined, signal),
  /** Yes or no to one (`id`: its full id or its slug). A yes raises a re-plan with its approval card; nothing changes until that card is approved. */
  dayOptionalAnswer: (id: string, answer: "yes" | "no") => post<OptionalAnswer>(`/api/day/optional/${encodeURIComponent(id)}`, { answer }),

  // Inspection (SPOT): contexts/spot; shapes in components/spot/types.ts. Every GET answers 200 without a kit ({configured: false, note}).
  spotOverview: (signal?: AbortSignal) => get<Spot.Overview>("/api/spot/overview", undefined, signal),
  spotFindings: (signal?: AbortSignal) => get<{ configured: boolean; findings: Spot.Finding[] }>("/api/spot/findings", undefined, signal),
  spotTriage: (signal?: AbortSignal) => get<Spot.Triage>("/api/spot/triage", undefined, signal),
  spotLocations: (signal?: AbortSignal) => get<Spot.Locations>("/api/spot/locations", undefined, signal),
  spotAssets: (signal?: AbortSignal) => get<{ configured: boolean; assets: Spot.AssetRow[] }>("/api/spot/assets", undefined, signal),
  spotAsset: (id: string, signal?: AbortSignal) => get<Spot.AssetDetail>(`/api/spot/assets/${encodeURIComponent(id)}`, undefined, signal),
  spotMission: (id: string, signal?: AbortSignal) => get<Spot.MissionDetail>(`/api/spot/missions/${encodeURIComponent(id)}`, undefined, signal),
  spotHandoff: (signal?: AbortSignal) => get<Spot.Handoff>("/api/spot/handoff", undefined, signal),
  spotAssumptions: (signal?: AbortSignal) => get<{ assumptions: Spot.Assumption[] }>("/api/spot/assumptions", undefined, signal),
  spotValidation: (signal?: AbortSignal) => get<Spot.Validation>("/api/spot/validation", undefined, signal),
  spotModel: (signal?: AbortSignal) => get<Spot.Model>("/api/spot/model", undefined, signal),
  spotWorkOrders: (signal?: AbortSignal) => get<{ work_orders: Spot.WorkOrder[]; folder: string }>("/api/spot/work_orders", undefined, signal),
  spotReplay: (signal?: AbortSignal) => get<Spot.Replay>("/api/spot/replay/state", undefined, signal),
  spotReplayStart: () => post<Spot.Replay>("/api/spot/replay/start", {}),
  spotReplayPause: () => post<Spot.Replay>("/api/spot/replay/pause", {}),
  spotReplaySeek: (body: { index?: number; mission_id?: string; fraction?: number }) => post<Spot.Replay>("/api/spot/replay/seek", body),
  spotReplaySpeed: (speed: number) => post<Spot.Replay>("/api/spot/replay/speed", { speed }),
  spotAsk: (text: string, rephrase = false) => post<Spot.Answer>("/api/spot/ask", { text, rephrase }),
  /** The operator's first screen in plain words (no statistics): status, do-this-first, watch-out, every asset, the rounds. */
  // the look guide: "where do I look for X?" -> a place on the Inspection screen and the words to walk there (no model call)
  spotGuide: (q: string, signal?: AbortSignal) => get<import("./guide").SpotGuide>("/api/spot/guide", { q }, signal),
  spotRoute: (to: string, from?: string | null, signal?: AbortSignal) => get<import("./guideSpot").SpotRoute>("/api/spot/route", from ? { to, from } : { to }, signal),
  /** The floor traced from the kit's plan picture (rooms, doors, equipment, stairs, walkable cells), and the traced line between every pair of waypoints. */
  spotWalkable: (signal?: AbortSignal) => get<import("./guideSpot").SpotWalkable>("/api/spot/walkable", undefined, signal),
  spotWalkableLegs: (signal?: AbortSignal) => get<{ configured: boolean; available: boolean; legs: Record<string, import("./guideWalk").TracedLeg> }>("/api/spot/walkable/legs", undefined, signal),
  spotShift: (signal?: AbortSignal) => get<Spot.Shift>("/api/spot/shift", undefined, signal),
  spotRound: (id: string, signal?: AbortSignal) => get<Spot.RoundView>(`/api/spot/shift/rounds/${encodeURIComponent(id)}`, undefined, signal),
  /** The same cited answer, worded for an operator; `draft` keeps the exact wording. */
  /** Shift Run: plan a round and run it through a toy model built from the recorded rounds. Pure: nothing is written. */
  spotRunBoard: (signal?: AbortSignal) => get<Spot.RunBoard>("/api/spot/shiftrun/board", undefined, signal),
  spotRunSimulate: (body: { route: string[]; turn_back_at: number; seed?: number; hard?: boolean }) => post<Spot.RunResult>("/api/spot/shiftrun/simulate", body),
  /** The plan as a mission request: an approval card; a file on this box after Approve, nothing else. */
  spotPlan: (body: { route: string[]; turn_back_at: number }) => post<{ action_id: string; reason: string }>("/api/spot/actions/plan", body),
  spotAskPlain: (text: string) => post<Spot.Answer & { plain?: boolean }>("/api/spot/ask", { text, plain: true }),
  /** Both raise an approval card on the box; nothing is written until it is approved. */
  spotReinspect: (asset_id: string) => post<{ action_id: string; reason: string }>("/api/spot/actions/reinspect", { asset_id }),
  spotWorkOrder: (anomaly_id: string) => post<{ action_id: string; reason: string }>("/api/spot/actions/work_order", { anomaly_id }),
};

import type * as Spot from "@/components/spot/types";

/** Full URL of a gateway path the inspection screen puts in an <img> (the floor plan, the kit's sample pictures). */
export const spotUrl = (path: string): string => HTTP_BASE + path;

/**
 * POST /api/learning/lab/live-chat (contexts/learning/live_chat.py): the question streamed as Server-Sent Events, each
 * event handed to `onEvent` as it arrives. Resolves when the stream ends; throws ApiError when the route refuses.
 */
export async function liveChatStream(
  question: string,
  target: import("./liveChat").ChatTarget,
  onEvent: (e: import("./liveChat").ChatEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const { parseSse } = await import("./liveChat");
  const headers: Record<string, string> = { "Content-Type": "application/json", Accept: "text/event-stream" };
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  let res: Response;
  try {
    res = await fetch(HTTP_BASE + "/api/learning/lab/live-chat", { method: "POST", headers, body: JSON.stringify({ question, target }), signal });
  } catch (err) {
    if (err instanceof DOMException && err.name === "AbortError") throw err;
    throw new ApiError(0, "Can't reach the Atlas box. Check that you are on the same network.");
  }
  if (!res.ok || !res.body) {
    let detail = res.statusText || `HTTP ${res.status}`;
    try {
      const data: unknown = await res.json();
      if (data && typeof data === "object" && typeof (data as { detail?: unknown }).detail === "string") detail = (data as { detail: string }).detail;
    } catch {
      /* not JSON: keep the status text */
    }
    throw new ApiError(res.status, detail);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (value) buffer += decoder.decode(value, { stream: !done });
    const parsed = parseSse(done ? buffer + "\n\n" : buffer);
    buffer = parsed.rest;
    for (const e of parsed.events) onEvent(e);
    if (done) return;
  }
}
