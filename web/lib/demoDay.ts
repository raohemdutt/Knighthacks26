/**
 * EMBEDDED COPY of fixtures/play/demo.yaml (`plans`, `disruptions`, `campaign`).
 *
 * The gateway runs the scripted Sunday when a request carries `scenario: "demo"`,
 * but no endpoint returns the scenario's own plans or disruptions, so the two
 * one-tap presets, the scripted hard mode and the scripted campaign are copied
 * here by hand. If fixtures/play/demo.yaml changes, change this file to match.
 * scripts/check-api.mjs sends these same plans to a running gateway and fails
 * when the bad plan stops failing or the good plan stops winning three stars.
 *
 * Nothing here is a result: every number on screen comes back from the box.
 */

import type { Disruption, Move } from "./api";

export const DEMO_SCENARIO = "demo";

/** No food, a lunch run that cannot make it back, cramming alone at the desk all night, a nap at dawn. */
export const DEMO_BAD_PLAN: Move[] = [
  { slot: "s1", kind: "rest" },
  { slot: "s2", kind: "eat", place: "ucf.dining_63", minutes: 40 },
  { slot: "s3", kind: "rest", minutes: 300 },
  { slot: "s3", kind: "study", place: "apartment.desk", minutes: 540 },
  { slot: "s3", kind: "sleep" },
];

/** Breakfast first, lunch near the venue, the library with Maya, dinner, a second short session, a full night. */
export const DEMO_GOOD_PLAN: Move[] = [
  { slot: "s1", kind: "eat", place: "ucf.dining_63", minutes: 30 },
  { slot: "s1", kind: "rest" },
  { slot: "s2", kind: "eat", place: "ucf.union", minutes: 25 },
  { slot: "s3", kind: "study", place: "ucf.library", minutes: 170, with_friend: "Maya" },
  { slot: "s3", kind: "eat", place: "ucf.union", minutes: 35 },
  { slot: "s3", kind: "study", place: "ucf.library", minutes: 105 },
  { slot: "s3", kind: "rest", place: "apartment.desk", minutes: 60 },
  { slot: "s3", kind: "sleep", start: "2026-10-11T22:30:00-04:00", minutes: 510 },
];

/** Hard mode in the demo: judging moves across campus and starts early; the Student Union's food closes. */
export const DEMO_DISRUPTIONS: Disruption[] = [
  { kind: "block_moved", block: "outlook:kh-judging:2026-10-11T12:00:00-04:00", to_place: "ucf.cb1", shift_minutes: -15 },
  { kind: "place_closed", place: "ucf.union" },
];

/** Campaign: Sunday as the good plan, then Monday (lecture at 10:40). */
export const DEMO_CAMPAIGN: Move[][] = [
  [
    { slot: "s1", kind: "eat", place: "ucf.dining_63", minutes: 30 },
    { slot: "s2", kind: "eat", place: "ucf.union", minutes: 25 },
    { slot: "s3", kind: "study", place: "ucf.library", minutes: 170, with_friend: "Maya" },
    { slot: "s3", kind: "eat", place: "ucf.union", minutes: 35 },
    { slot: "s3", kind: "sleep", start: "2026-10-11T22:30:00-04:00", minutes: 510 },
  ],
  [
    { slot: "s1", kind: "eat", place: "apartment.kitchen", minutes: 20 },
    { slot: "s2", kind: "eat", place: "ucf.union", minutes: 30 },
    { slot: "s2", kind: "study", place: "ucf.library", minutes: 180 },
    { slot: "s2", kind: "eat", place: "ucf.union", minutes: 30 },
    { slot: "s2", kind: "sleep", start: "2026-10-12T23:00:00-04:00", minutes: 480 },
  ],
];
