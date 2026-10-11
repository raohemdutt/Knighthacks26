# O51: fixes deferred to the owners of lab/controller.py and lab/hub.py

The O51 review (PR #145) left findings whose fix belongs in `lab/controller.py` or `lab/hub.py`. Both files were
held by another task while the rest of O51 was fixed on branch `ctx/orchestration/o51-fix`, so they were not edited.
Each entry below gives the finding, what is still wrong on main (c39b242), the exact change and the failing test
to write first. Line numbers are main at c39b242.

The slice fixes that landed on the branch: pods-run-script-pod-scripts-report-ok-on-failure (pod scripts),
queue-queued-head-never-expires, queue-retry-reposts-create-on-503-529, queue-offline-plan-overwrites-text-columns,
modul-dup-notion-client, and the queue and referee half of modul-dup-ledger-verb-names.
xcall-slot-refusal-rows-read-as-real-by-alerts was already fixed on main (alerts.SLOT_HAPPENED, tests in
tests/test_lab_alerts.py:327-349).

## 1. pods-run-script-pod-scripts-report-ok-on-failure (controller and docs half)

The scripts now exit non-zero when their sibling file is missing, so the documented command reports `fail` instead
of a false `ok`. It still fails, because run-script uploads only the script to the Jupyter root.

- `lab/README.md:206`: document the working command:

  ```
  python -m lab.controller run-script --account B --with lab/smoke_lora.py lab/pod_b_trainer.sh smoke
  ```

  (lab/README.md is outside the O51 slice; one line, owner FYI.)
- Optional, `lab/controller.py` `run_script` (line 1111): nothing else is needed; `--with` already uploads the
  sibling next to the script and the scripts look there first.

Test (tests/test_lab_controller.py, with `fake_terminal` extended to run the uploaded files in bash in a tmp dir):
run the real `lab/pod_b_trainer.sh smoke` text with no `--with`; assert `outcome == "fail"` and
`exit_code == 2`. tests/test_lab_pod_scripts.py already proves the script side.

## 2. modul-dup-ledger-verb-names (controller half)

`lab/controller.py:151` and `:153` restate `"pod launch"`, `"pod stop"` and `"pod.balance_credits"`, which
`lab/machine/alerts.py` defines as `POD_LAUNCH`, `POD_STOP` and `BALANCE_METRIC`.

Change, `lab/controller.py`:

```python
from lab.machine import alerts, record          # line 112: add alerts
...
POD_LAUNCH, POD_STOP, POD_STATUS = alerts.POD_LAUNCH, alerts.POD_STOP, "pod status"     # line 151
BALANCE = alerts.BALANCE_METRIC                                                           # line 153
```

alerts.py imports neither controller nor hub, so there is no import cycle.

Test: `assert controller.BALANCE is alerts.BALANCE_METRIC and controller.POD_LAUNCH is alerts.POD_LAUNCH and
controller.POD_STOP is alerts.POD_STOP` (an identity check on the module constants; the slots and referee half uses
a rename test instead, because those writers now read alerts at call time).

## 3. pods-hub-address-and-accounts-pinned-to-dub-ab and modul-hard-hub-address (what is left)

Mostly fixed on main: an account's own `AUP_HUB_URL_<letter>` is read from the environment and the .env files, C and
D are accepted, and lab/README.md:57-60 documents it. What is left: `$LAB_HUB_URL` itself is read only from the
process environment (`lab/hub.py:366`, `environ.get(ENV_URL)`), so a `LAB_HUB_URL=` line in a .env is ignored.

Change, `lab/hub.py` `load_accounts` (line 218):

```python
    optional = [name for letter, _, _ in wanted for name in option_keys(letter)] + [ENV_URL]
    ...
        url, password, launch, gpu, resource = (found.get(name) for name in option_keys(letter))
        accounts[letter] = Account(letter, found[user_key], found[token_key], url or found.get(ENV_URL), password,
                                   launch, gpu, resource, roles.get(letter.upper()))
```

The order stays: base_url argument, then `AUP_HUB_URL_<letter>`, then `LAB_HUB_URL` (environment first, then
.env), then the DUB default. Also add `LAB_HUB_URL` and `AUP_HUB_URL_<letter>` to `config/.env.example` (outside the
slice).

Test (tests/test_lab_hub.py): `LAB_HUB_URL=https://sjc.example.test` only in a tmp .env, no environment value,
`Hub(load_accounts(["A"], environ={}, env_files=[env])["A"], environ={})`; assert
`base_url == "https://sjc.example.test"`.

## 4. pods-keep-alive-max-interval-no-margin

`lab/controller.py:134` allows `--interval 1800`, and `_cycle` (line 1063) does not retry a failed ping, so one
failure gives a 3600 s gap, the culler's idle threshold.

Change, `lab/controller.py:134`:

```python
KEEP_ALIVE_MAX_S = 1200             # the culler waits an hour and checks every ten minutes: one failed ping at the
                                    # longest interval still leaves two of its checks in hand
```

The `--interval` parser (line 1300) and `keep_alive` (line 1010) already read the constant.

Test (tests/test_lab_controller.py): `keep-alive --interval 1500` exits 2 with the range in the message, and
`controller.keep_alive(..., interval_s=1201)` raises ValueError; 1200 is accepted.

## 5. pods-scrub-misses-url-quoted-account-name

`Hub.request` puts the URL-quoted path in the HubUnreachable text (`lab/hub.py:445`), and `Controller.scrub`
(`lab/controller.py:425`) replaces only the raw `account.user`. A user name that `urllib.parse.quote` changes
(`a@b`, a space, a slash) reaches answers and ledger notes as `a%40b`.

Change, `lab/controller.py` `Controller.scrub`:

```python
            if len(account.user) >= MIN_NAME_CHARS:
                quoted = urllib.parse.quote(account.user, safe="")
                for name in dict.fromkeys((quoted, account.user)):     # the quoted form first: it may contain the raw
                    text = text.replace(name, f"<account {account.letter}>")
```

`urllib.parse` is already imported (line 100).

Test (tests/test_lab_controller.py): an account whose user is `amd user@1`, a transport that raises
`httpx.ConnectError` on every call; run `status` and assert neither `amd%20user%401` nor `amd user@1` is in the
JSON answer or in the ledger row's note, and `<account A>` is.

## 6. xcall-keepalive-lost-row-races-concurrent-stop

`_cycle` (`lab/controller.py:1069`) treats a server whose `pending == "stop"` as gone and `_gone` writes a
`pod keep-alive lost` row at once (line 1106), before the concurrent `stop` writes its `pod stop` row; alerts
replays rows in ts order and raises `pod_stopped` for an intended stop.

Change, `lab/controller.py` `_cycle`:

```python
            if server is not None and server.pending == "stop":
                watch.active = False
                watch.state = "stopping"
                self._row_for(problems, POD_KEEP_ALIVE, "stopping", watch.account,
                              note="the hub is stopping the pod (an intended stop)")
                return
            if server is None:
                self._gone(watch, seen, problems)
                return
```

`stopping` is in none of alerts.POD_LOST_OUTCOMES, so no alert fires, and the run's outcome falls through to
`completed` (line 1038) unless another watch reports otherwise.

Test (tests/test_lab_controller.py): one FakeHub clock, a ready pod, the hub answering the DELETE with 202 so
the server shows `pending: "stop"`; run one keep-alive cycle between the DELETE and stop's own row, then
`alerts.scan`; assert no `pod_stopped` alert and a `pod keep-alive stopping` row.
