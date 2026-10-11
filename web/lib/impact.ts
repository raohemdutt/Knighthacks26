/**
 * Shapes of the Impact screen's endpoints (contexts/impact/api.py). The calls themselves are
 * `api.impactStory`, `api.impactPractice` and `api.addPractice` in lib/api.ts.
 */
import type { EventEnvelope, ImpactCounts } from "./api";

/** A low and a high estimate: nothing on the Impact screen that is not a measured count is a single number. */
export interface Range {
  low: number;
  high: number;
}

export type Confidence = "high" | "medium" | "low";

export interface ImpactEquivalentRow {
  item: string;
  count: number;
  material: string | null;
  mass_g: Range;
  each_g: Range;
  mass_source: string | null;
  mass_confidence: Confidence;
  energy_kwh: Range | null;
  co2e_kg: Range | null;
  material_source: string | null;
  material_confidence: Confidence | null;
  mass_url?: string | null;
  material_url?: string | null;
  /** The factors themselves, so the screen can show count x mass x factor. */
  energy_kwh_per_kg?: Range | null;
  co2e_kg_per_kg?: Range | null;
}

/** The week's recycled items in real units, from config/impact.yaml. Estimates, always. */
export interface ImpactEquivalents {
  estimate: true;
  disclaimer: string;
  file: string;
  items: number;
  kept_out_g: Range;
  energy_kwh: Range;
  phone_charges: Range;
  phone_charge_source: string | null;
  phone_charge_confidence: Confidence;
  phone_charge_wh?: Range;
  phone_charge_url?: string | null;
  /** What kind of energy this is (life-cycle energy of every kind, not socket electricity). */
  energy_note?: string | null;
  co2e_kg: Range;
  by_item: ImpactEquivalentRow[];
  unweighed: { item: string; count: number }[];
}

export interface ImpactDisposal {
  item_id: string;
  ts: string;
  day: string;
  label: string;
  /** The item as the rules file names it ("soda can" -> "can"). */
  item: string;
  bin: string;
  zone: string;
  correct: boolean;
  belongs_in: string;
  /** Where this item goes here, in one sentence from the active rules file. */
  rule: string;
  frame_ref: string | null;
  frame_stored: boolean;
  member_id: string | null;
  in_practice: boolean;
}

export interface ImpactDay {
  day: string;
  disposals: number;
  right: number;
  wrong: number;
  recycled: number;
  contamination: number;
  clean: boolean;
  right_share: number | null;
}

export interface ImpactSide {
  n: number;
  recycled: number;
  contamination: number;
  diversion_rate: number | null;
}

/** contracts.impact.MilestoneReached */
export interface ImpactMilestone extends EventEnvelope {
  type: "impact.milestone.reached";
  week_start: string;
  kind: "kept_out" | "clean_streak";
  threshold: number;
  value: number;
  text: string;
  member_id: string | null;
}

export interface PracticeItem {
  item: string;
  mistakes: number;
  /** Someone tapped "teach me" on it. */
  asked: boolean;
  last_wrong_at: string | null;
  went_to: string | null;
  belongs_in: string;
  /** The bin zone that scores in Sort It. */
  zone: string;
  rule: string;
  frame_ref: string | null;
  members: string[];
}

/** GET /api/impact/practice */
export interface ImpactPractice {
  week_start: string;
  member_id: string | null;
  size: number;
  items: PracticeItem[];
  rules: { file: string; name: string; place: string | null };
  how: string;
}

export interface ImpactRules {
  file: string;
  name: string;
  place: string | null;
  note: string | null;
  overridden: boolean;
  bins: string[];
  /** basis: "stated" by the place's published list, "assumed" (ours), or null when the file does not say. */
  items: { label: string; bin: string; rule: string; basis?: "stated" | "assumed" | null }[];
  default_rule: string | null;
  /** The published list the rules file was checked against. */
  source?: { title: string; publisher: string; url: string; read: string } | null;
  /** The same item under the rules around this place: values are a bin name or "not listed". */
  compare?: { item: string; [place: string]: string }[];
  compare_sources?: string[];
  swap: string;
}

export interface ImpactFlowItem {
  item_id: string;
  item: string;
  bin: string;
  correct: boolean;
  belongs_in: string;
  rule: string;
  ts: string;
}

export interface ImpactFlow {
  items: ImpactFlowItem[];
  /** Earlier disposals this week that are not in `items`. */
  more: number;
  recyclable_seen: number;
  recycled: number;
  recyclable_in_trash: number;
  /** Recyclables the camera saw that never entered a bin zone. */
  not_binned: number;
  wrong_in_recycling: number;
}

/** Mistake seen -> taught -> practised in Sort It -> the same mistakes next week. Only what the log holds. */
export interface ImpactLoop {
  seen: number;
  taught: number;
  practised: { rounds: number; right: number };
  trend: { available: boolean; previous: number | null; current: number; text: string };
  weeks: { week_start: string; disposals: number; mistakes: number; by_item: Record<string, number> }[];
}

/** GET /api/impact/story */
export interface ImpactStory {
  now: string;
  week: { start: string; end: string; today: string };
  empty: boolean;
  empty_text: string | null;
  rules: ImpactRules;
  profile: string | null;
  scope: { per_person: boolean; members: string[]; member_id: string | null; note: string };
  total: ImpactCounts;
  this_week: ImpactCounts;
  equivalents: ImpactEquivalents | null;
  contamination: {
    wrong_in_recycling: number;
    /** null: no published threshold, so none is claimed. */
    spoils_bag_at: number | null;
    at_risk: boolean;
    good_items_sharing_the_bin: number;
    text: string;
    source: string | null;
    confidence: Confidence;
    rule_of_thumb: boolean;
    /** A statement quoted word for word, who said it, and where. */
    quote?: string | null;
    attribution?: string | null;
    url?: string | null;
    rates?: { scope: string; text: string; source: string }[];
  };
  /** Measured counts in one sentence. */
  headline?: string;
  /** This week's disposals, oldest first, and where the recyclables seen ended up. */
  flow?: ImpactFlow;
  /** "The problem here": one sourced paragraph; each claim says whose figure it is. */
  problem?: { text: string; sources: { scope: string; claim: string; source: string }[] } | null;
  loop?: ImpactLoop;
  feed: ImpactDisposal[];
  goal: {
    diversion: {
      target: number;
      value: number | null;
      recycled: number;
      recyclable: number;
      met: boolean;
      progress: number | null;
      to_go: number | null;
      text: string;
    };
    contamination: { ceiling: number; value: number; met: boolean; left: number };
  };
  days: ImpactDay[];
  clean_streak: number;
  best_day: ImpactDay | null;
  projection: { enough: boolean; recycled_by_week_end: number | null; days_with_data: number; days_left: number; text: string };
  before_after: {
    available: boolean;
    enough: boolean;
    min_n: number;
    nudges_on_at: string | null;
    before: ImpactSide | null;
    after: ImpactSide | null;
    text: string;
  };
  milestones: {
    reached: ImpactMilestone[];
    latest: ImpactMilestone | null;
    next: { kind: string; threshold: number; to_go: number; text: string } | null;
  };
  practice: { size: number; items: PracticeItem[] };
}
