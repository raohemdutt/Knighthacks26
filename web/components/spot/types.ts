/** Shapes of the inspection endpoints (contexts/spot/api.py), as the gateway returns them. */

export interface Evidence {
  missions: string[];
  actions: string[];
  readings: string[];
  anomalies: string[];
  assets: string[];
  locations: string[];
}

/** One row of a breakdown: k of n with its 95% interval. `enough` is false under the minimum n. */
export interface Bar {
  key: string;
  label: string;
  k: number;
  n: number;
  rate: number | null;
  lo: number | null;
  hi: number | null;
  enough: boolean;
  note: string | null;
  evidence: Evidence;
  extra: Record<string, unknown>;
}

export interface Finding {
  id: string;
  title: string;
  question: string;
  sentence: string;
  value: number | null;
  unit: string;
  n: number;
  interval: [number, number] | null;
  p_value: number | null;
  enough: boolean;
  caveat: string | null;
  rows: Bar[];
  evidence: Evidence;
  extra: Record<string, unknown>;
}

export interface Side {
  k?: number;
  n: number;
  value: number | null;
  lo: number | null;
  hi: number | null;
  median?: number | null;
}

/** One line of the V1 against V2 comparison (findings: version_comparison.extra.metrics). */
export interface Metric {
  key: string;
  label: string;
  kind: "rate" | "mean";
  unit?: string;
  good: "high" | "low" | "none";
  verdict: "changed" | "no clear change" | "not enough data";
  a: Side;
  b: Side;
  diff: number | null;
  diff_lo: number | null;
  diff_hi: number | null;
  p_value: number | null;
  test: string;
  evidence: { a: string[]; b: string[] };
}

export type TargetKind = "asset" | "location" | "mission" | "finding" | "validation" | "assumptions" | "metric";

export interface Target {
  kind: TargetKind;
  id: string;
}

export interface Next {
  key: string;
  title: string;
  sentence: string;
  target: Target;
  finding_id: string | null;
  score: number | null;
  evidence: Evidence;
}

export interface Kpi {
  key: string;
  label: string;
  k: number;
  n: number;
  rate: number | null;
  lo: number | null;
  hi: number | null;
  good: "high" | "low";
  note: string;
}

export interface MissionSummary {
  mission_id: string;
  mission_name: string;
  scenario: string;
  configuration_version: string;
  start_time: string;
  end_time: string;
  duration_min: number;
  start_battery_pct: number;
  end_battery_pct: number;
  mission_status: string;
  actions: number;
  failed: number;
  anomalies: number;
  worst_severity: string | null;
  hidden_failure: boolean;
  badge: string;
}

export interface ActionRow {
  action_id: string;
  mission_id: string;
  sequence: number;
  timestamp: string;
  location_id: string;
  asset_id: string;
  action_type: string;
  battery_pct: number;
  status: string;
  failure_type: string;
  duration_sec: number;
  readings: { reading_id: string; measurement_type: string; value: number; unit: string; quality: string }[];
  anomalies: { anomaly_id: string; anomaly_type: string; severity: string; description: string; confirmed: boolean }[];
}

export interface MissionDetail extends MissionSummary {
  configured: boolean;
  timeline: ActionRow[];
}

export interface Sample {
  url: string;
  label: string;
  name: string;
}

export interface Overview {
  configured: boolean;
  note?: string;
  kit_path?: string;
  now?: string | null;
  period?: { from: string; to: string };
  counts?: Record<string, number>;
  kpis?: Kpi[];
  next?: Next[];
  headlines?: { id: string; title: string; question: string; sentence: string; caveat: string | null; n: number; enough: boolean; evidence_count: number }[];
  latest_mission?: MissionSummary;
  validation?: { summary: string; agree: number; compared: number };
  issues?: { warn: number; error: number; info: number };
  min_n?: number;
  index?: { action_mission: Record<string, string>; anomaly_asset: Record<string, string> };
  samples?: Record<string, Sample | null>;
  model: { version: string | null; real: boolean; role: string };
}

export interface ScorePart {
  label: string;
  points: number;
  detail: string;
}

export interface Inspection {
  last_good_action: string | null;
  last_good_at: string | null;
  last_good_mission: string | null;
  hours_since_good: number | null;
  failed_since: string[];
  latest_attempt: string | null;
  latest_attempt_failed: boolean;
  inspections: number;
  inspections_ok: number;
}

export interface TriageItem {
  rank: number;
  kind: "asset" | "waypoint";
  id: string;
  title: string;
  score: number;
  parts: ScorePart[];
  reason: string;
  next_step: { action: "reinspect" | "dispatch" | "check_path"; text: string };
  asset_id: string | null;
  location_id: string | null;
  criticality: string | null;
  severity: string | null;
  anomaly_id: string | null;
  confirmed: boolean;
  inspection: Inspection | null;
  evidence: Evidence;
  model?: { action_id: string; score: number; flagged: boolean; why: ModelWhy[] } | null;
}

export interface ModelWhy {
  feature: string;
  why: string;
  contribution: number;
}

export interface RuleScore {
  tp: number;
  fp: number;
  fn: number;
  flagged: number;
  precision: number;
  recall: number;
  pr_auc: number | null;
}

/** GET /api/spot/model: the small anomaly scorer trained on the kit. */
export interface Model {
  configured: boolean;
  available: boolean;
  note?: string;
  name?: string;
  kind?: string;
  verdict?: string;
  threshold?: number;
  trained_on?: { missions: number; actions: number; anomalies: number };
  evaluation?: Record<"primary" | "rolling", { n: number; positives: number; model: RuleScore; status_rule: RuleScore; threshold_rule: RuleScore; test_missions?: string[] }>;
  weights?: { feature: string; weight: number; why: string }[];
  quiet_flags?: { action_id: string; mission_id: string; asset_id: string; location_id: string; action_type: string; failure_type: string; timestamp: string; score: number; has_anomaly_row: boolean; failed: boolean; why: ModelWhy[] }[];
  quiet_total?: number;
  quiet_without_anomaly_row?: number;
  use?: string;
  reproduce?: string;
}

export interface Risk extends Inspection {
  asset_id: string;
  title: string;
  criticality: string;
  location_id: string;
  at_risk: boolean;
  stale: boolean;
  text: string;
  evidence: Evidence;
}

export interface Triage {
  configured: boolean;
  items: TriageItem[];
  uninspected: Risk[];
  next: Next[];
  score?: string;
}

export interface Location {
  location_id: string;
  location_name: string;
  zone_type: string;
  zone_text: string | null;
  floor: number;
  x: number;
  y: number;
  environment: string;
  k: number;
  n: number;
  rate: number | null;
  lo: number | null;
  hi: number | null;
  enough: boolean;
  by_type: Record<string, { k: number; n: number }>;
  health: "ok" | "watch" | "bad" | "unknown";
  failed_actions: ActionRow[];
  anomalies: { anomaly_id: string; severity: string; anomaly_type: string; asset_id: string; mission_id: string; timestamp: string; confirmed: boolean }[];
  assets: string[];
  first_stop_of: number;
}

export interface Plan {
  image: string;
  image_url: string;
  width: number;
  height: number;
  panels: Record<string, { x0: number; y0: number; x1: number; y1: number }>;
  waypoints: Record<string, { floor: number; x: number; y: number }>;
  fit: { by_group: Record<string, { n: number; rms_px: number; max_px: number }>; follows_csv: boolean; note: string };
  source: string;
}

export interface Locations {
  configured: boolean;
  locations: Location[];
  plan: Plan | null;
  overall?: { k: number; n: number };
  health_rule?: string;
  zone_text_source?: string;
}

export interface AssetRow {
  asset_id: string;
  asset_name: string;
  asset_type: string;
  location_id: string;
  criticality: string;
  k: number;
  n: number;
  rate: number | null;
  lo: number | null;
  hi: number | null;
  enough: boolean;
  anomalies: number;
  worst_severity: string | null;
  triage_rank: number | null;
  triage_score: number | null;
  at_risk: boolean;
  hours_since_good: number | null;
}

export interface Series {
  asset_id: string;
  measurement_type: string;
  unit: string;
  n: number;
  n_good: number;
  points: { reading_id: string; t: string; value: number; good: boolean; action_id: string; mission_id: string }[];
  slope_per_day: number | null;
  enough: boolean;
  outside_band: string[];
  toward_edge: boolean;
  days_to_edge: number | null;
  note: string | null;
  band?: { median: number; lo: number; hi: number; n: number; unit: string; higher_is_better: boolean } | null;
}

export interface AssetDetail extends AssetRow {
  configured: boolean;
  location: { location_id: string; location_name: string; floor: number } | null;
  inspection: Inspection;
  triage: TriageItem | null;
  series: Series[];
  anomalies_list: { anomaly_id: string; severity: string; anomaly_type: string; mission_id: string; action_id: string; location_id: string; timestamp: string; description: string; confirmed: boolean }[];
  actions: ActionRow[];
  seen_at: string[];
  samples: Sample[];
}

export interface Handoff {
  configured: boolean;
  as_of?: string | null;
  lines: { text: string; evidence: Evidence; target: Target | null }[];
  text: string;
}

export interface Assumption {
  key: string;
  assumption: string;
  why: string;
}

export interface Validation {
  configured: boolean;
  summary?: string;
  agree?: number;
  compared?: number;
  rows: { group: string; name: string; ours: number | null; theirs: number | null; status: string; how: string }[];
  issues: { code: string; level: "info" | "warn" | "error"; message: string; count: number; ids: string[] }[];
}

export interface WorkOrder {
  work_order_id: string;
  kind: string;
  asset_id: string;
  anomaly_id: string | null;
  reason: string;
  path: string;
  action_id: string;
  created_at: string;
}

export interface LiveAction {
  action_id: string;
  sequence: number | null;
  action_type: string;
  location_id: string;
  asset_id: string;
  status: string; // RUNNING | SUCCESS | FAILED
  failure_type: string;
  battery_pct: number | null;
  ts: string | null;
  reading: { reading_id: string; measurement_type: string; value: number; unit: string; quality: string } | null;
  anomalies: { anomaly_id: string; anomaly_type: string; severity: string; confirmed: boolean }[];
}

export interface Alert {
  alert_id: string;
  level: "info" | "warn" | "high" | "critical";
  message: string;
  data_ts: string | null;
  seq: number;
  mission_id: string;
  action_id: string | null;
  anomaly_id: string | null;
  asset_id: string | null;
  location_id: string | null;
}

export interface Mark {
  mission_id: string;
  start: number;
  end: number;
  failures: number[];
  source: "replay" | "tables";
  status: string;
  version: string;
  failed: number;
  anomalies: number;
}

export interface LiveMission {
  mission_id: string;
  status: string;
  scenario: string;
  configuration_version: string;
  started: string | null;
  ended: string | null;
  battery_pct: number | null;
  actions: LiveAction[];
  failed: number;
  anomalies: number;
  source: string;
}

/** GET /api/spot/replay/state: the live picture, rebuilt from the event log. */
export interface Replay {
  configured: boolean;
  index: number;
  total: number;
  playing: boolean;
  speed?: number;
  speeds?: number[];
  data_ts?: string | null;
  mission?: LiveMission | null;
  robot?: { location_id: string | null; trail: string[]; floor: number | null };
  counts?: Record<string, number>;
  by_location?: Record<string, { ok: number; failed: number }>;
  by_asset?: Record<string, { ok: number; failed: number; anomalies: number; worst: string | null }>;
  alerts?: Alert[];
  marks?: Mark[];
  rebuilt_from?: number | null;
  at_end?: boolean;
}

export interface Answer {
  configured: boolean;
  intent: string;
  question: string;
  text: string;
  draft: string;
  citations: Evidence;
  target: Target | null;
  rephrased: boolean;
  model: string;
  check: string | null;
  suggested: string[];
}
