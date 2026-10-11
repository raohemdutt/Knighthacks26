"use client";

/**
 * The profile this box runs (GET /api/profile): the words the screens use for the
 * person's workplace, activity, events and deadlines. The student's words are the
 * defaults, so every screen reads exactly as before until the box answers, and on
 * a box that has no /api/profile at all.
 */

import { useEffect, useState } from "react";
import { api, type Profile, type Words } from "./api";

export type { Words };

/** The student's vocabulary, as the box would send it. */
export const STUDENT_WORDS: Words = {
  workplace: "desk",
  Workplace: "Desk",
  workplace_plural: "desks",
  Workplace_plural: "Desks",
  activity: "study",
  Activity: "Study",
  activity_plural: "studies",
  Activity_plural: "Studies",
  event: "class",
  Event: "Class",
  event_plural: "classes",
  Event_plural: "Classes",
  deadline: "assignment",
  Deadline: "Assignment",
  deadline_plural: "assignments",
  Deadline_plural: "Assignments",
};

export const STUDENT_PROFILE: Profile = {
  name: "student",
  description: "",
  audience: "",
  vocabulary: { workplace: "desk", activity: "study", event: "class", deadline: "assignment", plurals: {} },
  words: STUDENT_WORDS,
  meters: { habits: [], dayrun: ["energy", "focus", "hunger"] },
  shift: null,
  scenario: "demo",
  // The Ask screen's chips. A box that sends no list (an older build) keeps these.
  suggestions: [
    "Where did I leave my keys?",
    "What is due this week?",
    "What happened at our table overnight?",
    "What should I eat?",
    "Where should I go tonight?",
    "How's your day?",
    "Tell me a joke",
    "I'm stressed about my exam",
  ],
  overrides: {},
};

let cached: Profile | null = null;
let inflight: Promise<Profile> | null = null;

function fetchProfile(): Promise<Profile> {
  if (cached) return Promise.resolve(cached);
  inflight ??= api
    .profile()
    .then((p) => {
      cached = { ...STUDENT_PROFILE, ...p, words: { ...STUDENT_WORDS, ...p.words } };
      return cached;
    })
    .catch(() => STUDENT_PROFILE)
    .finally(() => {
      inflight = null;
    });
  return inflight;
}

/** The box's profile; the student's until the box answers (one request per page load, shared by every screen). */
export function useProfile(): Profile {
  return useProfileState().profile;
}

/** The profile and whether the box has answered yet (`ready`), for screens that must not act on the default. */
export function useProfileState(): { profile: Profile; ready: boolean } {
  const [state, setState] = useState<{ profile: Profile; ready: boolean }>({ profile: cached ?? STUDENT_PROFILE, ready: cached !== null });
  useEffect(() => {
    let live = true;
    fetchProfile().then((p) => {
      if (live) setState({ profile: p, ready: true });
    });
    return () => {
      live = false;
    };
  }, []);
  return state;
}

/** The profile's words: `w.event` ("class", "shift"), `w.Activity` ("Study", "Work"), `w.deadline_plural`, ... */
export function useWords(): Words {
  return useProfile().words;
}

/** The label for a Day Run activity or move kind in the profile's words: "class" is the event word, "study" the activity. */
export function activityLabel(words: Words, activity: string): string {
  if (activity === "study") return words.activity;
  if (activity === "class") return words.event;
  return activity;
}
