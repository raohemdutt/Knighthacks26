# Waste Watch

Waste Watch is deliberately a small part of Atlas: one screen (`/impact/`) and one line in the friends' League. It is our entry for the OneEthos "Innovation for Good" track, and it exists because the box already has a camera on the bins.

## Brief check

The brief: "Every community faces environmental challenges. Pick an issue affecting your local community and build a solution with meaningful, measurable impact! Finding a problem worth solving is part of the challenge, so you decide what to build."

| The brief says | Where we meet it | Honest gap |
| --- | --- | --- |
| "an issue affecting your local community" | Recycling contamination at UCF. `config/sorting.yaml` is UCF Facilities' own published list, cited with the date read; the screen's "Why it matters" panel gives the sourced local picture (below). | We found no current public UCF contamination or diversion figure. The local numbers are Orange County's kerbside programme, not the campus. |
| "a solution" | A camera at a bin station sees what goes in which bin, judges it against UCF's list, shows the rule for each mistake, puts the mistake in the Sort It practice set, and tracks a weekly goal that counts in the friends' League. | The camera must be at the bins. It cannot tell a plastic bottle from a glass one, or a clean takeout tub from a dirty one, so those rules are the cautious ones. |
| "meaningful" | A wrong item is corrected at the moment it happens, with the place's own rule in one sentence, and the same mistakes become next week's practice. | One bin station's volume is tiny: the demo week is 7 disposals. Whether mistakes fall the week after is not shown yet; the log holds one week and the screen says so. |
| "measurable impact" | Measured: items seen, right or wrong bin, diversion rate (recycled / recyclables seen), wrong items in recycling. Estimated: grams, kWh, CO2e and phone charges, as ranges, from factors that each cite a document and show the unit conversion (`config/impact.yaml`). The two are labelled differently everywhere. | The factors are published US averages (EPA WARM v16 and others), not UCF's or its hauler's. Only cans and bottles have a citable mass; everything else is counted and claims no weight. |
| "finding a problem worth solving" | See "The problem here". The rules differ between the campus and the county around it, nobody is told when they get one wrong, and EPA says a wrong item can cost the whole load. | We did not interview UCF Recycling or measure a campus bin audit. The problem statement rests on published pages. |

Anything "at scale" is a projection and is not on the screen (last section).

## The problem here

UCF collects recycling single-stream: every recyclable goes in one bin, so one wrong item sits against everything else. UCF's list is narrower than the county around it: glass, plastic cups and soiled paper are trash on campus, while Orange County's kerbside carts take glass and, since May 2026, paper and clear plastic cups. People who learned one list get the other wrong. Orange County's kerbside recycling had a contamination rate of nearly 40% before its 2019 clean-up campaign, against a national average of about 17% by weight, and EPA says non-recyclable items can get a whole load thrown out.

Sources (all read 2026-10-10; the same list is in `config/impact.yaml` under `problem` and is shown on the screen):

- UCF: "UCF Recycles", UCF Facilities Operations, https://fo.ucf.edu/recycling-at-ucf/ ("UCF uses single stream recycling"; glass, plastic cups and utensils, Styrofoam and soiled paper are under "What We Don't Recycle").
- Orange County: "Think 5", Orange County Government, https://ocfl.net/WaterGarbageRecycling/Think5.aspx ; "It's To-Go Time: Orange County's Curbside Recycling Program Now Accepting Paper and Clear Plastic Cups", Orange County Government, 2026-05-06, https://newsroom.ocfl.net/media-advisories/press-releases/2026/05/its-to-go-time-orange-countys-curbside-recycling-program-now-accepting-paper-and-clear-plastic-cups/
- Orange County: "Improving the Quality of Recyclables in Orange County", The Recycling Partnership, https://recyclingpartnership.org/improving-the-quality-of-recyclables-in-orange-county/ ("had a nearly 40% contamination rate"; the campaign reduced it by 29%).
- National: "2020 State of Curbside Recycling Report", The Recycling Partnership, February 2020, https://recyclingpartnership.org/wp-content/uploads/dlm_uploads/2020/02/2020-State-of-Curbside-Recycling.pdf ("a first-of-its-kind national average inbound contamination rate estimate of 17 percent by weight").
- National: "Frequent Questions on Recycling", US EPA, page last updated 2026-01-22, https://www.epa.gov/recycle/frequent-questions-recycling ("Non-recyclable items can contaminate a whole load of recyclables, causing them to all be thrown out.").

EPA says "can", not "will", and gives no number. We found no published count of wrong items that spoils a bag, so none is claimed; the earlier "2 wrong items can spoil a bag" had no source and was removed.

## What is measured

A camera at the table counts each single-use item once (`impact.item.seen`) and records which bin zone it ends in and whether that was right (`impact.item.disposed`). That gives items seen, recyclables seen, recyclables recycled, wrong items in recycling, and the diversion rate (recycled divided by recyclables seen), plus the same numbers before and after the box started nudging about sorting, with n on each side and "not enough data yet" below five a side. These are counts from our own table. A recyclable the camera saw that never entered a bin (carried away) counts against the rate, and the story on the screen says how many there were.

## What is estimated

The Impact screen turns the week's recycled count into grams, kWh, CO2e and phone charges, always as a range, always tagged "estimate", from `config/impact.yaml`. "How we got this number" on the screen lists count × mass × factor for each item with its source.

Conversions: WARM publishes per short ton. 1 short ton = 907.185 kg and 1 million Btu = 293.071 kWh, so 1 million Btu per short ton = 0.32306 kWh/kg and 1 MTCO2E per short ton = 1.10231 kg/kg. "Recycling instead of landfill" is WARM's recycling factor minus its landfilling factor, as the WARM documentation instructs.

| Factor | Value in the file | Published figure and conversion | Source | Confidence |
| --- | --- | --- | --- | --- |
| Can, mass | 13.2–13.34 g | 0.7 lb per case of 24 = 13.2 g; "1000 aluminum cans or 13.34 kg" | "Volume-to-Weight Conversion Factors", US EPA, April 2016; WARM v16 Containers, Packaging, and Non-Durable Good Materials Chapters, US EPA, December 2023, section 2.4 | high |
| Bottle, mass | 9.25–23.9 g | average 16.9 oz water bottle 9.25 g; average soda bottle 23.9 g | "Environmental Footprint", International Bottled Water Association (Beverage Marketing Corporation data for 2014) | medium |
| Aluminium, energy | 44.5–54.4 kWh/kg | (152.76 + 0.27) million Btu/short ton × 0.32306 = 49.4, ±10 % | WARM v16 Management Practices Chapters, US EPA, December 2023, Exhibit 7-12 | medium |
| Aluminium, CO2e | 9.1–11.1 kg/kg | (9.13 + 0.02) MTCO2E/short ton × 1.10231 = 10.1, ±10 % | same document, Exhibit 1-1 | medium |
| PET, energy | 8.4–10.3 kWh/kg | (28.59 + 0.27) × 0.32306 = 9.32, ±10 % | Exhibit 7-12 | medium |
| PET, CO2e | 1.05–1.29 kg/kg | (1.04 + 0.02) × 1.10231 = 1.17, ±10 % | Exhibit 1-1 | medium |
| Paper, energy | 3.1–6.6 kWh/kg | office paper (10.08 − 0.47) → 3.10; mixed paper (20.56 − 0.17) → 6.59 | Exhibit 7-12, two grades | low |
| Paper, CO2e | 4.0–4.4 kg/kg | mixed paper (3.55 + 0.07) → 3.99; office paper (2.86 + 1.13) → 4.40 | Exhibit 1-1, two grades | low |
| Cardboard, energy | 4.3–5.3 kWh/kg | (15.10 − 0.21) × 0.32306 = 4.81, ±10 % | Exhibit 7-12 | low |
| Cardboard, CO2e | 3.3–4.0 kg/kg | (3.14 + 0.18) × 1.10231 = 3.66, ±10 % | Exhibit 1-1 | low |
| Glass, energy | 0.70–0.85 kWh/kg | (2.13 + 0.27) × 0.32306 = 0.78, ±10 % | Exhibit 7-12 | medium |
| Glass, CO2e | 0.30–0.36 kg/kg | (0.28 + 0.02) × 1.10231 = 0.33, ±10 % | Exhibit 1-1 | medium |
| One phone charge | 19.0–19.4 Wh | 28.446 Wh − (22 h × 0.411 W) = 19.4 Wh, which EPA rounds to 0.019 kWh | "Greenhouse Gas Equivalencies Calculator: Calculations and References", US EPA, page last updated 2026-08-04 | medium |

URLs are on each line of `config/impact.yaml`. Three things to know before quoting any of these:

- WARM publishes one value per material with no interval. Where the range is "±10 %", the 10 % is our allowance so the screen never shows one falsely precise number; it is not EPA's.
- WARM's energy is life-cycle energy of every kind (fuel and electricity) not spent making new material. Writing it as phone charges makes it picturable; it is not electricity from a socket.
- Paper, cardboard and glass have factors but no per-item mass, so at UCF today only cans and bottles produce an estimate. Masses for cups, sheets of paper, boxes, glasses, takeout containers, chip bags, napkins and cartons were our own guesses and were removed; the "carton" material was removed because WARM has no carton category. Those items are still counted.

## The rules, and reusing this elsewhere

`config/sorting.yaml` holds the bins, which bin each item belongs in, one sentence per item saying why, and for each item whether UCF's page states it or it is our assumption (cups made of paper, a clean takeout tub, chip bags). Checking the file against UCF's page on 2026-10-10 moved no item to a different bin; the sentences were rewritten to say what UCF says. The screen marks the assumed items and shows the same item under Orange County's and the City of Orlando's rules.

Another campus changes one file: replace `config/sorting.yaml` (or add `config/profiles/<name>/sorting.yaml` and start the box with `ATLAS_PROFILE=<name>`), and the same cameras, events and code give that place's verdicts and sentences. `config/profiles/family-home/sorting.yaml` is a complete second file, an example and not a real city's list, that sorts four items differently; `contexts/impact/tests/test_story.py` runs one event stream through both. A campus that wants its own real-unit figures also edits `config/impact.yaml`.

## The screen

`/impact/` (`GET /api/impact/story`, and `GET /api/league` for the last panel), top to bottom: a one-sentence headline of measured counts; a 15-second story in four panels (where each item went, the diversion ring against the goal, what that kept out as one estimated range, this table against the friends' tables in the League); the weekly goal; the feed of disposals with the rule and "Teach me" for a wrong one; the Sort It practice set with the loop in one row (mistakes seen, "Teach me" taps, Sort It rounds on them, and last week against this week once the log has two weeks); the rules in use. `?t=0.4` opens the story part-way, `?explain=why` or `?explain=how` opens a panel, `?motion=reduce` shows the static end state.

The friends' bet is display only: it reads each member's diversion rate from the League's `team_waste` block and the League's own stake text. Recycling is one of the League's weekly targets, so it counts toward the stake; it does not decide it alone. With no friend to compare with, the story ends on the goal line.

## At scale (a projection, not a measurement)

If the 12,280 students in UCF on-campus and affiliated housing (UCF Facts, Fall 2025, https://www.ucf.edu/about-ucf/facts/) each threw away two drink containers a day (our assumption, not a UCF figure) and their diversion rate rose from this table's 67% (six recyclables in one week, so barely a number) to the 80% goal, about 22,000 more containers a week would reach recycling (12,280 × 2 × 7 × 0.13); we do not convert that to energy because the campus mix of cans and bottles is unknown.
