/**
 * Shapes of the coach's feedback endpoints (contexts/day/application/coach_feedback.py). The calls are
 * `api.coachFeedback` and `api.coachCorrect` in lib/api.ts.
 */
import type { CoachBreak, CoachSession, CoachTrigger, EventEnvelope, LearningStage } from "./api";

/** What the person can say a break was. "none": nothing in particular. */
export type CoachAnswer = "chat" | "phone" | "door" | "food" | "bathroom" | "done" | "other" | "none";

export interface CoachCorrectionView {
  right: boolean;
  answer: CoachAnswer;
  answer_text: string;
  note: string | null;
  ts: string;
}

/** A break as the feedback view gives it: `trigger` is the reason now (the person's when they answered), `guess` the coach's. */
export interface FeedbackBreak extends CoachBreak {
  index: number;
  guess: CoachTrigger;
  guess_text: string;
  frame_ref: string | null;
  correction: CoachCorrectionView | null;
}

export interface FeedbackSession extends Omit<CoachSession, "breaks"> {
  day: string;
  breaks: FeedbackBreak[];
}

export interface CoachPrecision {
  trigger: CoachTrigger;
  right: number;
  total: number;
  precision: number;
  /** "The coach is right about phone breaks 4 of 5 times." */
  text: string;
}

export interface CoachTeachProgress {
  trigger: string;
  name: string;
  answers: number;
  negatives: number;
  needed: number;
  negatives_needed: number;
  ready: boolean;
  proposed: boolean;
}

/** A teach task the coach proposed, with its stage from the learning board. */
export interface CoachLearning {
  trigger: string;
  task: string;
  name: string;
  stage: LearningStage;
  progress: number;
  learned: boolean;
  model_version: string | null;
  examples: number | null;
  proposed_at: string;
  updated: string | null;
  text: string;
}

/** GET /api/day/coach/feedback */
export interface CoachFeedbackView {
  now: string;
  words: { activity: string; workplace: string };
  options: { id: CoachAnswer; label: string }[];
  sessions: FeedbackSession[];
  suggestions: string[];
  precision: CoachPrecision[];
  answers: number;
  progress: CoachTeachProgress[];
  threshold: { min_corrections: number; min_negatives: number };
  learning: CoachLearning[];
  /** One sentence per thing the box has learned from these answers. */
  notes: string[];
}

export interface CoachCorrectionBody {
  day: string;
  block_id: string;
  break_index: number;
  right: boolean;
  answer?: CoachAnswer;
  note?: string;
}

/** contracts.day.DayCoachCorrected */
export interface CoachCorrected extends EventEnvelope {
  type: "day.coach.corrected";
  day: string;
  block_id: string;
  break_index: number;
  break_start: string;
  guess: CoachTrigger;
  right: boolean;
  answer: CoachAnswer;
  note: string | null;
  plan_id: string | null;
}

/** POST /api/day/coach/corrections */
export interface CoachCorrectionResult {
  correction: CoachCorrected;
  session: FeedbackSession;
  precision: CoachPrecision[];
  progress: CoachTeachProgress[];
  learning: CoachLearning[];
  /** trigger -> the approval card this answer raised, when it completed the threshold. */
  proposed: Record<string, string> | null;
}
