# Profiles: the same box for a student, a shift worker, a family

Atlas is demoed as a student's assistant, but the student is one *profile*. A profile is a
folder under `config/profiles/<name>/` holding the words the screens and nudges use and the
config files that differ from the base `config/*.yaml`. Nothing in the code knows which
profile is running; it reads `contexts/profiles.load_profile(config_dir)`.

**Selection:** `ATLAS_PROFILE=<name>` in the environment, else `profile: <name>` in
`config/agent.yaml`, else `student`. An unknown name refuses to boot and lists what exists.
`GET /api/profile` tells the phone the name, description, vocabulary (`words`), meters, shift
and which files are overridden.

Shipped: `student` (the base files as they are; overrides nothing), `shift-worker`
(nights: the sleep window follows the shift, handovers, a leave-for-shift item list, the
`plant` and `flat` packs, the `night-shift` scenario), `family-home` (several people via owner
recognition, homework time, screen time at the table, bedtime, the `house` and `town` packs,
the `school-night` scenario).

```
ATLAS_PROFILE=shift-worker python -m contexts.gateway
curl -s localhost:8000/api/profile | python -m json.tool
```

## A new profile in ten lines

1. `mkdir config/profiles/lab-bench`
2. Write `profile.yaml` with `name`, `description`, `audience` and a `vocabulary` (four words, see below).
3. Pick `meters.habits`: the habit ids from `habits.yaml` this person keeps (empty = all of them).
4. Copy only the habits you change into `habits.yaml`: an entry with the same `id` is merged key by key; a new `id` is added.
5. Put the day's model in `dayrun.yaml`: `pack`, `home`, `desk`, `kinds`, `sleep`, `study_words`, `meal_words`, `work_estimates`, `win`.
6. Add `agent.yaml` for the sleep window, the `places` presence may name, the `leaving.home_place`.
7. Add a place pack under `places/` (`kind: campus` for the outside world, `kind: home` with an `entry` for the rooms; a campus place points at the home pack with `pack:`). Give the home pack its own name: a home nested under two campus places cannot be routed out of.
8. Write a scripted day under `scenarios/<name>.yaml` (same shape as `fixtures/play/demo.yaml`: `now`, `items`, `habits`, `plans.good`, `plans.bad`) and name it in `profile.yaml` as `scenario:`; the Day Run screen's demo mode plays it and its one-tap plans.
9. `ATLAS_PROFILE=lab-bench pytest -q contexts/profiles` and boot the gateway on an empty `ATLAS_DATA_DIR`.
10. Check the words on Today, Day Run and Ask, and that `plans.good` wins (`POST /api/dayrun/simulate`).

## What a profile may override

| File | How | Read by |
| --- | --- | --- |
| `profile.yaml` | required | `contexts/profiles` (vocabulary, meters, shift, scenario) |
| `habits.yaml` | layered: mappings merge by key, `habits:` merge by `id`, then `meters.habits` filters | `contexts/habits/domain/config.load_config(profile=)` |
| `dayrun.yaml` | layered (same rules); `sleep.bedtime/wake` set from `shift` when given | `contexts/play/dayrun/config.DayRunConfig.load(profile=)` |
| `agent.yaml` | layered; `sleep.bedtime/wake` set from `shift` when given | `bootstrap.build` (people, places, leaving, circle, subreddit), habits' schedule, Day Run's time zone |
| `sorting.yaml` | replaces the base file whole | Sort It and Waste Watch (`SortingRules.load`) |
| `rules.yaml` | replaces the base file whole | the alert rules (`contexts/agent/domain/rules.load_rules`) |
| `league.yaml` | replaces the base file whole, when the league context exists on the branch | `contexts/play/league` |
| `places/*.yaml` | added after `config/places/`; a pack with the same name wins | `PlaceSet.load` |
| `scenarios/*.yaml` | searched before `fixtures/play/` | `DayRun.scenario`, `GET /api/dayrun/board?scenario=` |

Everything else (`cameras.yaml`, `zones.yaml`, `connectors.yaml`, `models.yaml`) is per box,
not per audience, and stays in `config/`.

## Vocabulary keys

| Key | Student | Shift worker | Family | Used for |
| --- | --- | --- | --- | --- |
| `workplace` | desk | station | table | where the box watches: "meals at your {workplace}", "Still on the {workplace}" |
| `activity` | study | work | homework | the main activity: "Study at", "study minutes", the Planner's move kind |
| `event` | class | shift | appointment | a fixed calendar block: Ask's blurb, the Day Run "class" activity label |
| `deadline` | assignment | handover | chore | something due: Today's "Due next" wording, Day Run's board blurb |
| `plurals` | `{}` | | | irregular plurals, e.g. `{event: classes}` (regular ones are formed automatically) |

Nudge texts in `habits.yaml` may use `{workplace}`, `{event}`, `{deadline}`, `{activity}`,
their plurals (`{event_plural}`) and capitalised forms (`{Workplace}`, `{Deadline_plural}`)
next to the habit's own fields (`{value}`, `{target}`, `{title}`, `{hours}`, `{minutes}`, `{items}`).
The phone gets the same set as `words` from `GET /api/profile` (`web/lib/profile.tsx`,
`useWords()`), with the student's words as the default until the box answers.

`shift: {start, end, sleep_after_minutes, sleep_hours}` is optional; when present the sleep
window in `agent.yaml` (what Habits calls overnight) and in `dayrun.yaml` (Day Run's usual night)
both start `sleep_after_minutes` after the shift ends, so moving the rota moves the night.

## Still student-specific, in other owners' code

Behind the profile the Day Run engine still names its block kinds `study` and `class`
(`contexts/play/dayrun/models.py` `MoveKind`, `Activity`): they are identifiers, labelled with the
vocabulary on the way out. These remain hard-coded and need their owner:

- `contexts/agent/application/planner.py`: `EXAM`, `COURSE`, `STUDY_QUESTION`, `PLAN_QUESTION`
  regexes (28-36), the LLM prompt "what to study" (51), `campus="UCF"` (117, 200, 350), the
  "Study: {course}" calendar block title (218) and "Add N study block(s)" approval text (240); the
  leaving checklist (277-315) picks the next `calendar_event`/`assignment` item.
- `contexts/agent/application/qa.py`: `ITEMS_Q`, `ITEM_THING` and the per-kind regexes (184-189:
  "due", "assignments", "homework", "exams", "classes", "on campus"); the Reddit tool text "What
  students say" (81). `SleepSchedule.load` reads `config/agent.yaml` directly, so the agent's
  overnight window does not follow a profile's `agent.yaml` or `shift` (`bootstrap.py:232`).
- `contexts/connectors/`: `config/connectors.yaml` names Canvas and the UCF events feed; a
  profile cannot override it yet (per box, not per audience).
- `web/app/layout.tsx:10` metadata "Ask your desk ..."; `web/app/circle/` `SUGGESTED_PLACES`
  (Library, Dorm, Student Union) and `web/components/circle/ProposeTime.tsx:61` "Study session"
  (Circle is another owner's); `web/app/timeline/page.tsx` 192/267 ("your classes", "Assignments
  from Canvas"); `web/components/dayrun/CampusMap.tsx` and `DayPlayer.tsx` aria-labels say
  "Campus map"; `web/lib/demoDay.ts` embeds the student's plans as the fallback presets.
- `fixtures/demo/night.yaml` and `fixtures/demo/fakes.yaml` are the student's demo night; a
  profile does not get its own seeded night yet (`python -m contexts.demo seed` is student-only).
- Habits are per enrolled member with owner recognition on, but targets are one set per profile:
  a different bedtime per child needs per-member targets in `contexts/habits/domain/members.py`.
