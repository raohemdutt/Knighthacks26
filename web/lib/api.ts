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
  /** The anonymous room mode: faces that match nobody are counted together, as nobody. */
  everyone?: boolean;
  expressions_available?: boolean; // false when the expression model is not on this box
  photos_available?: boolean; // false when the box cannot decode pictures (no OpenCV)
}

/** One row of GET /identity/members. The face template itself never leaves the box. */
export interface Member {
  member_id: string;
  name: string;
  enrolled_at: string;
  updated_at?: string | null;
  photos?: number; // how many looks the template is made of (up to 12)
  source?: "live" | "scan" | "photos";
  expressions?: boolean; // this person switched "read my expressions" on
  circle_member_id?: string | null; // which Circle member this is, when someone said so
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
  calibration: Record<string, { image: [number, number][]; floor: [number, number][]; homography: number[][] }>;
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
  round: { round_id: string; item: string; target_bin: string; bin: string; players: number; started: string; routes: RoomRouteLeg[] } | null;
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

export type DayMoment = "morning" | "now" | "replan" | "coach" | "evening" | "approval";

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
}

/** GET /api/day/brief */
export interface DayBrief {
  mode: "model" | "template" | "off";
  engine: "model" | "template" | "off";
  timeout_s: number;
  gap_s: number;
  latest: Partial<Record<DayMoment, DayNarration>>;
  asks: Record<string, DayNarration>;
  model: { calls: number; used: number; rejected: number; timeouts: number; errors: number; last_problem: string | null };
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
  mic?: { ok: boolean; detail: string };
  speaker?: { ok: boolean; detail: string };
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
  profile: (signal?: AbortSignal) => get<Profile>("/api/profile", undefined, signal),
  captureStatus: (signal?: AbortSignal) => get<{ cameras: Record<string, CameraStatus> }>("/capture/status", undefined, signal),
  liveTracks: (camera: string, signal?: AbortSignal) => get<LiveTracks>(`/capture/${encodeURIComponent(camera)}/tracks`, undefined, signal),

  ask: (text: string) => post<Answer>("/api/ask", { text }),

  zones: (signal?: AbortSignal) => get<ZoneSet>("/api/zones", undefined, signal),
  setZones: (camera: string, zones: Record<string, [number, number][]>) => put<ZoneSet>(`/api/zones/${encodeURIComponent(camera)}`, { zones }),
  demoZones: () => post<ZoneSet>("/api/zones/demo", {}),
  vaultPolicy: (signal?: AbortSignal) => get<VaultPolicy>("/vault/policy", undefined, signal),
  runtimeModels: (signal?: AbortSignal) => get<RuntimeModels>("/runtime/models", undefined, signal),
  voiceStatus: (signal?: AbortSignal) => get<VoiceStatus>("/api/voice/status", undefined, signal),
  voiceTalk: () => post<VoiceStatus>("/api/voice/talk", {}),
  voiceStop: () => post<VoiceStatus>("/api/voice/stop", {}),
  voiceEnabled: (enabled: boolean) => post<VoiceStatus>("/api/voice/enabled", { enabled }),
  setVaultPolicy: (body: Partial<VaultPolicy>) => put<VaultPolicy>("/vault/policy", body),
  egress: (signal?: AbortSignal) => get<Egress>("/vault/egress", undefined, signal),
  wipeRequest: (scope: WipeScope) => post<WipeToken>("/vault/wipe/request", { scope }),
  wipe: (token: string) => post<WipeResult>("/vault/wipe", { token }),

  learning: (signal?: AbortSignal) => get<LearningBoard>("/api/learning", undefined, signal),
  teach: (body: TeachBody) => post<{ task: string; action_id: string }>("/api/teach", body),

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
  setRoomCalibration: (camera: string, taps: { image: [number, number][]; floor: [number, number][] }) =>
    put<RoomPlan>(`/api/room/calibration/${encodeURIComponent(camera)}`, taps),

  places: (signal?: AbortSignal) => get<PlacePack[]>("/api/places", undefined, signal),
  /** A pack's drawn map, by the path GET /api/places gave for it. Fetch it through lib/campusmap's loadCampusMap, which keeps it. */
  placeMap: (path: string, signal?: AbortSignal) => get<CampusMapData>(path, undefined, signal),
  /** When to leave: `from` and `to` are pack ids ("ucf.eng2"), names, "friend:<member>" or "here". */
  placeRoute: (q: { from?: string; to: string; arrive_by?: string; depart?: string }, signal?: AbortSignal) => get<PlaceRoute>("/api/places/route", q, signal),
  lastSeen: (label: string, signal?: AbortSignal) => get<Sighting>(`/memory/objects/${encodeURIComponent(label)}/last-seen`, undefined, signal),

  // Day Run. `scenario` runs a scripted day from fixtures/play/ ("demo") instead of live sources.
  dayrunBoard: (scenario?: string, signal?: AbortSignal) => get<Board>("/api/dayrun/board", { scenario }, signal),
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
  spotAsk: (text: string) => post<Spot.Answer>("/api/spot/ask", { text }),
  /** Both raise an approval card on the box; nothing is written until it is approved. */
  spotReinspect: (asset_id: string) => post<{ action_id: string; reason: string }>("/api/spot/actions/reinspect", { asset_id }),
  spotWorkOrder: (anomaly_id: string) => post<{ action_id: string; reason: string }>("/api/spot/actions/work_order", { anomaly_id }),
};

import type * as Spot from "@/components/spot/types";

/** Full URL of a gateway path the inspection screen puts in an <img> (the floor plan, the kit's sample pictures). */
export const spotUrl = (path: string): string => HTTP_BASE + path;
