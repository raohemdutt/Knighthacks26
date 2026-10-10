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
  const [profile, setProfile] = useState<Profile>(cached ?? STUDENT_PROFILE);
  useEffect(() => {
    let live = true;
    fetchProfile().then((p) => {
      if (live) setProfile(p);
    });
    return () => {
      live = false;
    };
  }, []);
  return profile;
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
