/**
 * The alone-against-with-a-friend rehearsal the team recorded on 2026-10-11, for screens that quote it where the
 * box has no recorded comparison of its own (GET /api/learning/lab/compare answers 404 there).
 *
 * Every number is copied from fixtures/learning/live-loop-2026-10-11/comparison.json (`acts.solo`, `acts.duo`,
 * `duo_over_solo`); tests/test_rehearsal_numbers.py fails if this file and that record ever disagree. Nothing
 * here is measured by this box: a screen that shows it says "recorded rehearsal" and links to the Learning lab.
 */
export const REHEARSAL = {
  source: "fixtures/learning/live-loop-2026-10-11/comparison.json",
  recorded: "2026-10-11",
  solo: { examples: 100, pods_joules: 9783.9, j_per_example: 97.8 },
  duo: { examples: 200, pods_joules: 14584.3, j_per_example: 72.9 },
  /** duo over solo: twice the examples for 1.49 times the pods' energy. */
  ratio: { examples: 2.0, pods_joules: 1.491 },
  teacher: "pod C (teacher)",
  trainer: "pod D (trainer)",
} as const;
