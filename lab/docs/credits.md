# Credits: `python -m lab credits`

Moved here from lab/README.md (O70); the verbs list there links every page. Arguments: `python -m lab credits -h`.

```
python -m lab credits                       # accounts C,D (the SJC hub)
python -m lab credits --account A,B --window-hours 6 --until 2026-10-11T16:00:00Z --no-ledger
```

Read-only (GET requests only; it also runs while `lab/state/STOP` exists). It answers with one JSON object: per
account the figures, `alerts`, and `lines`, one sentence per account. Exit 0 when every account was read (an alert is
a result, outcome `alert`, still exit 0), 1 when one could not be (outcome `error`), 2 for bad arguments or missing
keys. `--account all` takes every letter that has both keys; an account's hub is `AUP_HUB_URL_<letter>`, else
`$LAB_HUB_URL`, else DUB.

- **Metered hub** (DUB, the other hub: quota enabled, 4 credits a minute on the R9700, the other chip). The hub deducts a session only when it
  stops, so the balance is flat while a pod runs. The tracker adds the running session (whole minutes x rate, from the
  server's `started`) and the settled ones (the usage transactions, spread over the minutes they ran) to give
  `credits_left`, `credits_per_hour` over `--window-hours` (default 3), `burn_per_hour` (the higher of that and what
  the running servers burn now), `hours_left` and `runs_out_at`. It alerts when the credits are gone or would not last
  until `--until` (default judging, Sunday 2026-10-11 12:00 ET); after that moment it never alerts.
- **Unmetered hub** (SJC, the lab: quota disabled, balance 0, unlimited). Nothing is projected and nothing alerts; the line is
  `unmetered: usage N minutes today, M sessions` from `/hub/api/stats/me`, today being the hub's UTC day.
- Times come from the hub's clock (its `Date` header), not the workstation's.
- **Ledger.** One row per account (verb `credits`, node the role the account fills (`pod-a` for C on SJC, `LAB_POD_ROLES`), `refs.account`, `hub` and `account` in conditions, outcome `ok`, `alert`,
  `unmetered` or `error`) unless `--no-ledger`. Metered rows carry `pod.balance_credits`, `pod.credits_left`,
  `pod.credits_per_hour`, `pod.credits_burn_per_hour`, `pod.credits_hours_left`; unmetered rows `pod.usage_min_today`,
  `pod.sessions_today`, `pod.usage_min_window`. A partly read answer is marked `low_confidence`. Tokens and account
  names are never printed or written.

Tests: `tests/test_lab_credits.py` replays the recordings in `fixtures/lab/credits/` and drives `lab/hub_fake.py`.
