# Working on Atlas (for AI coding agents and the people driving them)

Atlas is our Knight Hacks project: a screenless, always-on AI box (NVIDIA Jetson)
with cameras, memory, an agent, and a learning lab on two AMD R9700 GPUs. Four
people build it in parallel, and each of them works with a coding agent (Claude
Code, Codex, Cursor, and so on). This file tells every agent how to coordinate
with the team.

**The rule that matters most: Notion is the source of truth and the place the
team communicates.** The code lives in this repository. Decisions, plans,
ownership, questions and answers live in Notion. When the repo and Notion
disagree on what should be built, Notion wins. Raise the conflict on the Team
Comms page instead of quietly picking one.

---

## 1. Where things are in Notion

| Page | What it is for | Link |
| --- | --- | --- |
| **Knighthacks** (hub) | Team table: who owns what. "Start here" list. | https://app.notion.com/p/3f596ef5fec5807ebaa2dacc55626600 |
| **Atlas Design Doc** | What we are building and why: architecture, bounded contexts, contracts, model plan, demo beats. Each context has a spec subpage. | https://app.notion.com/p/3f596ef5fec5819ea5cdcaa015e69917 |
| **Task Board** | The backlog. Every task has an ID, owner, priority (P0/P1/P2), status, "Done when", and "Blocked by". | https://app.notion.com/p/3f596ef5fec581bcb319e8687c1fdd8e |
| **Demo Plan (5 min)** | The judged run of show. Everything works backwards from it. | https://app.notion.com/p/3f596ef5fec581a2bdc4f96c8da65252 |
| **Team Comms** | Questions between teammates, one page per person, answered inline. | https://app.notion.com/p/3f596ef5fec5815c8023d5058086c741 |
| **Build Log** (under Team Comms) | What has landed on `main`, how to use it, which commit. Newest first. | https://app.notion.com/p/3f596ef5fec5812fa37de5ae92086234 |
| **Overnight Status (agents)** (under Team Comms) | One section per person; each agent rewrites its own every ~30 min while unattended. People read it first in the morning. | https://app.notion.com/p/3f596ef5fec58172a2d3e0db86fc09a3 |
| **Completed tasks (archive)** (under Task Board) | Finished task pages, moved off the board to keep it fast. | https://app.notion.com/p/3f596ef5fec581d980ebc44e691ca543 |
| **Credentials** | Logins and keys. Read them; never paste them into code, commits, events, logs or chat. | linked from the Knighthacks hub |

Each teammate has a Team Comms page:

| Person | Team Comms page | Notion user id (for Assignee filters) |
| --- | --- | --- |
| Dominick (Dom) | https://app.notion.com/p/3f596ef5fec581828d8ee02ab7b912e3 (Dominick ↔ Team) | `user://13dd872b-594c-81cd-b409-0002df1ecab7` |
| Hemdutt | https://app.notion.com/p/3f596ef5fec5819f8099d245fa78799d | `user://3f5d872b-594c-8144-b8ea-0002ce9d65db` |
| Xander | https://app.notion.com/p/3f596ef5fec58177b4c6dfd0acddacce | `user://32ed872b-594c-81b8-83fa-00026e75ed3d` |
| Ilan | https://app.notion.com/p/3f596ef5fec58163bbefd0e7ea5acea7 | `user://fbcc0baf-9942-4661-9414-319688df6910` |

Every person page has the same four sections:
- **Open questions:** checkboxes, each with an `Answer:` line under it.
- **Answered:** questions that have been dealt with.
- **FYI (no reply needed):** background the owner should know: interfaces they will plug into, ownership changes, gotchas.
- **Questions for Dominick:** where that person (or their agent) asks Dom something. You can also post on **Dominick ↔ Team**; Dom's agent watches both.

### Getting Notion access
See `notion/README.md`. In short, connect the Notion MCP server (claude.ai
connector, or `claude mcp add --transport http notion https://mcp.notion.com/mcp`),
then check that the Knighthacks page can be fetched. **If your agent cannot reach
Notion, stop and tell your person.** Working without Notion means working
without the team's decisions.

---

## 2. Who am I working for?

At the start of a session, find out which teammate you are working for:
1. Check the git author (`git config user.name` / `user.email`) and the person's memory or instructions, if your tool has them.
2. If that is still unclear, ask your person once: "Which teammate are you: Dominick, Hemdutt, Xander or Ilan?"

Everything below ("your page", "your tasks") means that person's.

---

## 3. The loop: what to do, and when

Do this **at the start of every session**, **before you pick new work**, and
**after every push**. Unattended (overnight), run it as the cycle in section 8.5. During long sessions, also do it at least every 30 to 60
minutes or between tasks, because teammates keep changing the plan.

**This file changes too.** Teammates may edit `AGENTS.md` to change how agents
behave. After every `git pull`, check whether it changed
(`git diff <previous HEAD> HEAD -- AGENTS.md`, or just re-read it). If it did,
**accept the change and follow the new version from then on**. If two
instructions conflict, the newer commit wins. Ask on Dominick ↔ Team if it is
unclear.

1. **Read your Team Comms page.** Handle every unticked item under *Open questions* (see section 4).
2. **Read the FYI section** on your page for interfaces and ownership changes that affect you.
3. **Skim the Build Log** for what landed since you last looked. Use what exists; don't rebuild it.
4. **Check the Task Board** for your tasks (section 5) and pick the next one.
5. **Re-read the relevant Design Doc section** and context spec before building, so the work matches the plan and the demo.

Then build, test, push, and do the bookkeeping in section 6.

---

## 4. Answering questions addressed to your person

Questions on your person's page usually come from Dominick or Dominick's agent.
Sort each one into one of two kinds.

### A. Simple, factual questions: answer them yourself, no need to interrupt your person
If the answer can be read from the code, the config, the running system or
the environment, and it commits nobody to a new decision, find it, write it
under the question, tick the box, and move on. Examples:
- "Which port and flags is llama-server using?" Look at the launch script or the running process, then answer: "8080, `--mmproj ... --jinja`".
- "Does the detector return normalized boxes?" Read the code and answer.
- "Which file holds X?" / "What version of Y is installed?" / "Did Z land on main?"
- "Is this format compatible with your script?" Answer yourself only if you checked the script and it clearly is (or clearly is not, with the exact field that is missing).

Write answers in plain words with the evidence (file path, command output,
commit). Then mention in your session summary that you answered them, so your
person knows what was said for them.

### B. Questions that need your person's input: stop and ask them
Bring the question to your person and **wait for their answer** whenever it involves any of these:
- a **decision**: scope, priority, ownership, product behaviour, UX, what to cut, picking between designs;
- **accounts, credentials, payments, or physical hardware** (plug in a camera, flash the Jetson, log into a portal);
- **disagreeing** with a proposed contract, format, or plan;
- anything **irreversible**, or anything that commits your person to work or a deadline;
- anything you would otherwise have to **guess**.

How to ask: say clearly that this is a teammate's question from Notion and that
it needs their input. Quote it, give the options and your recommendation, and
link the page. For example:

> **Question from Dominick on Notion needs your input** (Hemdutt ↔ Dominick, Q5):
> "Do the Habits contracts match what you plan for H1–H3?" I checked `contracts/habits.py` against your H1 plan; `habits_daily` needs a `target` field the contract lacks. Options: (1) add `target: float` to `DayClosed`, or (2) keep it in config only. I recommend (1). OK to answer that way?

After they answer, write the answer under the question in Notion (say it came
from them) and tick the box.

**Never invent an answer.** If you can't verify it and it isn't your person's
decision, write `Answer: unknown — <who would know>`, or leave it and tell your person.

### Asking other people
- To ask a teammate something, add a question to **their** page under *Open questions*: a checkbox, a bold one-line question, enough context to answer without a meeting (file paths, contract names, commit), and an empty `Answer:` line. Also add it to **"Open questions at a glance"** on the Team Comms hub.
- To ask Dominick, use **Questions for Dominick** on your own page.
- Prefer Notion over chat for anything that needs context, so the answer is kept for everyone. Short "which port?" questions are fine in Notion too.

---

## 5. Picking what to do next (the backlog)

The Task Board is the backlog. To choose work:
1. Find your tasks: Assignee is your person and Status is not Done.
2. **P0 first.** Do P1 only once the P0 demo path works end to end, and P2 only if ahead.
3. Prefer tasks whose **Blocked by** tasks are all Done (lower **Wave** numbers come first).
4. Read the task's **Done when**. That text is the acceptance test.
5. **Skip anything Locked by someone else** (section 8.2). Check the lock before you take a task, every time.
6. If none of your tasks are unblocked, look for work that unblocks you, and ask on Team Comms before taking over someone else's task. If your person agrees to take over a task, change its Assignee in Notion so the board stays honest.
7. Claim the task with the lock (section 8.2): Assignee, **Locked**, **In progress**.

### Creating work when the backlog runs dry
If every task assigned to your person is Done or blocked, **it is our job to
assign more and keep going**. Don't stop, and don't wait for morning:
1. First look for **unassigned, unlocked** tasks that fit your person's area (the team table on the Knighthacks hub), and claim one.
2. Otherwise, **create the next task yourself.** Base it on the gap between the Design Doc / Demo Plan and what the Build Log says exists: a demo beat that doesn't work yet, a "Done when" clause still open on an In-progress task, a context spec item nobody has built, tests or fixtures that are missing.
3. Write it like the others: next free ID in its context (e.g. `A9`), Context, Priority (P0 only if a demo beat depends on it), a concrete **Done when**, **Blocked by**, Assignee = your person, Locked = checked, Status = In progress. Note on the task page that an agent created it and why.
4. Stay inside your person's areas and the Design Doc. Product changes, new sponsors, or anything that changes the demo go to **Dominick ↔ Team** as a proposal, not straight onto the board.

**The Design Doc is what you are building towards.** Before building a context,
read its spec page (Design Doc → section 15). Check the **Demo Plan** to see what
judges will actually see; it is the reason each P0 exists.

### Notion tool tips
- The "query data source" (SQL) tool has a **usage limit** on our plan. Prefer `search` scoped to the Task Board data source (`collection://68af27b3-01c7-487d-807e-71af7d3fc536`) and `fetch` of individual pages, and use SQL sparingly.
- Fetching the Task Board page shows the dependency graph; fetching a task page shows all its properties.

---

## 6. When you finish something

1. **Tests pass first:** `pytest -q`. Commit only on green; chain them (`pytest -q && git commit ...`).
2. **Ship through a feature branch and PR** (section 8.3), with small, focused commits whose messages say what changed and why.
3. **Build Log:** add an entry at the top. Include what landed, the commit hash, how to use it (command, endpoint, file), and anything others must do differently.
4. **Task Board:** set the task to **Done** and **uncheck Locked**, or keep it **In progress** (still locked) with a note saying which part of "Done when" is still open.
5. **Tell the people it affects:** add an FYI on their Team Comms page if it changes an interface they use.
6. **Keep Notion small:** when several tasks are Done, consolidate them. Add one "✅ Done batch N: …" row to the board, move the original task pages under **Completed tasks (archive)**, and list them in that page's table. Move pages, don't delete them. Archive pages that are fully superseded the same way.

---

## 7. Repo conventions (short version; details in the Build Log)

- **Layout:** `contracts/` (shared Pydantic events and API shapes), `bus/` (event bus + JSONL log), `runtime/` (the only way to call models), `contexts/<context>/` (one folder per bounded context: `domain/`, `application/`, `infrastructure/`, `tests/`), `config/` (YAML), `fixtures/`, `lab/` (GPU-side scripts).
- **Contexts talk only through events and small APIs.** Never read another context's tables or import its internals for side effects.
- **Contracts are shared.** Changing `contracts/` or `runtime/` needs a heads-up on Team Comms and a second person's OK. Add fields; don't rename or remove them.
- **The event log is the source of truth.** `bus.subscribe(prefix, handler, replay=True)` only for state that rebuilds from the log. Anything that sends, writes, or calls a model is live-only. Slow handlers go through `bus.in_background(...)`.
- **Use event time (`event.ts`), not the wall clock,** for durations and cooldowns, so replay matches live.
- **Every outside effect needs an approval:** messages, calendar writes, connector writes, data exports. Propose with `agent.action.proposed`; Workspace runs it after the phone approves.
- **Wire new handlers in `contexts/gateway/bootstrap.py`.** Run the app with `python -m contexts.gateway`.
- **Secrets:** `.env` only (template: `config/.env.example`). Never in code, config, events, logs, commits, or Notion pages other than Credentials.
- **Setup:** `pip install -e ".[dev]"` (add `,browser` and run `python -m patchright install chromium` for the browser connector), then `pytest -q`.

---

## 8. Working in parallel and overnight (agents talking to agents)

All four agents may run at the same time, including overnight while their
people sleep. Notion is how they coordinate; git is where their work meets.
Follow these rules so they help each other instead of colliding.

### 8.1 Message conventions (so everyone can tell who said what)
- **Sign every post.** Anything an agent writes in Notion starts with `🤖 <Person>'s agent · <Day h:mm AM/PM ET>:`, e.g. `🤖 Hemdutt's agent · Sat 2:40 AM:`. Humans don't need a tag.
- **Asking:** add `- [ ] **Q<n> (from <Person>'s agent, <time>):** <one-line question>` under *Open questions* on the **recipient's** page, with the context needed to answer it without a meeting (file paths, contract names, commit hashes), and an `Answer:` line under it. Number questions after the highest Q on that page. Questions for Dominick go on **Dominick ↔ Team**.
- **Answering:** fill in the `Answer:` line as `🤖 <Person>'s agent · <time>: …`, tick the box, and, if the answer changes what the asker builds, say so in one line.
- **Needs a human:** if an agent cannot answer without its person (section 4B), it writes `Answer: 🙋 waiting on <Person> — <what is needed>` and adds the item to **Needs a human** in its section of **Overnight Status**. It does **not** wait: it moves on to other work and checks back later.
- **Don't edit what others wrote** except their `Answer:` lines on your own page and the checkbox. Never delete another person's or agent's text.

### 8.2 The task lock (a mutex): never take work someone has already claimed
Every task on the Task Board has a **Locked** checkbox. **Locked = someone has
confirmed they own this task and are working on it.** It is a mutex: only the
lock holder (the Assignee, or their agent) works on that task.

**Check before you take.** Before starting any task, even your own:
1. **Re-fetch the task page now**, not from memory or an old search result.
2. If **Locked** is checked and the **Assignee is not your person**, the task is taken. **Do not work on it, do not edit its properties, do not "help" in its files.** Pick something else, or ask the lock holder on their Team Comms page.
3. If **Locked** is checked and the Assignee *is* your person, it's yours: continue.
4. If **Locked** is unchecked, you may claim it **only if it is assigned to your person** (or unassigned and you follow "Creating work" below).

**Claiming (acquire the lock):**
1. Set **Assignee** to your person (if it was unassigned), **Locked** = checked, **Status** = In progress, in one update.
2. Add a first line to the task page: `🤖 <Person>'s agent · <time>: locked — <plan in one line>`.
3. **Re-fetch once more.** If another agent's `locked` line appears above yours (two agents claimed at the same moment), the earlier line wins. The later agent deletes its own line, sets the Assignee back to the earlier claimant, and moves on.

**Releasing (always release what you hold):**
- **Done:** set Status = Done and **uncheck Locked**.
- **Handing off or giving up:** uncheck Locked, set Status back to Not started, and write `🤖 … released — <why, and what's already done>` on the task page.
- **Blocked for a long time:** keep the lock, set Status = Blocked, and say what it's waiting on.

**Taking over someone else's task:** you never break another person's lock.
Ask on their page. Only the lock holder (or their person) releases it or
reassigns it; then claim it normally. An **unlocked, Not started** task assigned
to someone else may be handed over by that person or their agent; they change
the Assignee and write `handed to <Person>` on the task page.

**Other people's folders:** if your work needs a one-line change in another
person's folder, keep it minimal and post an FYI on their page with the commit
hash. Anything bigger is a question for them.

### 8.3 Git: feature branches and pull requests
Nobody commits straight to `main` any more. Every piece of work goes on a
feature branch and lands through a pull request.

1. **Branch from fresh `main`:** `git checkout main && git pull --rebase && git checkout -b ctx/<context>/<feature>` (e.g. `ctx/agent/keyframe-tool`, `ctx/memory/sqlite-store`; `ctx/docs/<topic>` for docs). One task per branch.
2. **Commit small**, with messages that say what changed and why.
3. **Before pushing:** `git fetch origin && git rebase origin/main`, then `pytest -q`. Push the branch: `git push -u origin <branch>`.
4. **Open a PR** (`gh pr create --base main`) titled with the task ID, e.g. `A1: get_keyframe tool`. The body says what changed, how it was tested, and anything others must do.
5. **Merging:**
   - Your PR touches only **your own folders** (your `contexts/<context>/`, your config, your tests) and tests pass: merge it yourself (`gh pr merge --squash --delete-branch`).
   - It touches **shared files** (`contracts/`, `runtime/`, `bus/`, `contexts/gateway/bootstrap.py`, `pyproject.toml`, `AGENTS.md`): additive changes (new optional fields, new events, a new wiring block) can be merged by you, with an FYI to everyone affected. Renames, removals, or behaviour changes need a second person's (or their agent's) OK on the PR or on Team Comms first.
   - **Never** `--force` push to `main`, rewrite shared history, or use `--no-verify`.
6. **After merging:** Build Log entry (with the PR number), Task Board status, release the lock if done.

Conflicts and a red `main`:
- **Rebase conflict in your own files:** resolve it and re-run the tests. **Conflict in someone else's files:** abort (`git rebase --abort`), post on their page with the file and your branch, and work on something else until it is resolved.
- **`main` is red because of someone else's commit:** don't fix their code. Post on their page (`🤖 … main fails: <test> since <commit>`), add it under *Blocked on* in Overnight Status, and keep working on your branch.
- **Contract changes overnight** are additive only (new optional fields, new event types), always with an FYI to everyone who uses that contract. Renames and removals wait for the people.

### 8.4 What an agent must not do unattended
- No outside effects: no real messages, calendar invites, posts, purchases, emails, or data exports. Approvals exist for people; never approve on someone's behalf.
- No credentials in code, commits, events, logs or Notion (except the Credentials page, which is read-only for agents).
- No destructive commands: no deleting branches, data, or Notion pages (archive by moving instead), and no `git reset --hard` on shared work.
- No new paid services, accounts or API keys. Put those under *Needs a human*.
- Don't change the plan: the Design Doc, priorities and demo plan are the people's. Suggest changes on Dominick ↔ Team.

### 8.5 The overnight cycle (run on a loop, about every 30 minutes)
1. **Inbox:** read your person's Team Comms page and **Dominick ↔ Team**. Answer what you can (section 4A). Mark the rest `🙋 waiting on <Person>`.
2. **Replies:** check the pages where *you* asked questions. Act on new answers.
3. **Build Log + git:** pull `main`, **re-read AGENTS.md if it changed** (follow the new rules), skim what landed, and run `pytest -q` so you know `main` is green.
4. **Work:** continue the task you hold the lock on, or claim the next one: check the lock, then acquire it (sections 5 and 8.2). Work in small steps that can each be committed.
5. **Ship:** on your feature branch, rebase on `main`, run the tests, push, open the PR, merge per section 8.3, then do the Build Log entry, Task Board status (release the lock when Done), and FYIs to anyone affected (section 6).
6. **Status:** rewrite your section of **Overnight Status**: last update, working on, done tonight, blocked on, needs a human.
7. **Never idle:** if every task you hold or could claim is blocked or waiting on a human, create the next task (section 5, "Creating work when the backlog runs dry"), lock it, and build it. Only if nothing useful is left inside your person's areas, say so in Overnight Status and keep checking the inbox each cycle.

**Claude Code:** start the night with `/loop` and this prompt (no interval: the agent paces itself, about every 30 minutes):

```
/loop Follow AGENTS.md section 8.5 as <your name>'s agent: check Notion (my Team Comms page, Dominick ↔ Team, replies to my questions), pull and test main, continue the task I hold the Locked mutex on or check-then-claim my next one (never take a task Locked by someone else; if my backlog is empty, create and lock the next task from the Design Doc), push with tests green, update the Build Log, Task Board and my Overnight Status section. Never take outside actions or answer decisions that need me; mark them "🙋 waiting on me".
```

Other tools: run the same prompt on a timer (cron, a scheduler, or a manual re-run every half hour).

### 8.6 In the morning (for the people)
1. Open **Overnight Status**: what was done, and what needs you.
2. Answer the `🙋 waiting on you` questions on your Team Comms page (write the answer, tick the box). Your agent picks them up on its next cycle.
3. Skim the Build Log for what landed overnight.
