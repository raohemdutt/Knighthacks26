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
| **Completed tasks (archive)** (under Task Board) | Finished task pages, moved off the board to keep it fast. | https://app.notion.com/p/3f596ef5fec581d980ebc44e691ca543 |
| **Credentials** | Logins and keys. Read them; never paste them into code, commits, events, logs or chat. | linked from the Knighthacks hub |

Each teammate has a Team Comms page:

| Person | Team Comms page | Notion user id (for Assignee filters) |
| --- | --- | --- |
| Dominick (Dom) | asks the questions; questions *for* Dom go in "Questions for Dominick" on your own page | `user://13dd872b-594c-81cd-b409-0002df1ecab7` |
| Hemdutt | https://app.notion.com/p/3f596ef5fec5819f8099d245fa78799d | `user://3f5d872b-594c-8144-b8ea-0002ce9d65db` |
| Xander | https://app.notion.com/p/3f596ef5fec58177b4c6dfd0acddacce | `user://32ed872b-594c-81b8-83fa-00026e75ed3d` |
| Ilan | https://app.notion.com/p/3f596ef5fec58163bbefd0e7ea5acea7 | `user://fbcc0baf-9942-4661-9414-319688df6910` |

Every person page has the same four sections:
- **Open questions:** checkboxes, each with an `Answer:` line under it.
- **Answered:** questions that have been dealt with.
- **FYI (no reply needed):** background the owner should know: interfaces they will plug into, ownership changes, gotchas.
- **Questions for Dominick:** where that person (or their agent) asks Dom something.

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
**after every push**. During long sessions, also do it at least every 30 to 60
minutes or between tasks, because teammates keep changing the plan.

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
5. If none of your tasks are unblocked, look for work that unblocks you, and ask on Team Comms before taking over someone else's task. If your person agrees to take over a task, change its Assignee in Notion so the board stays honest.
6. Set the task to **In progress** when you start.

**The Design Doc is what you are building towards.** Before building a context,
read its spec page (Design Doc → section 15). Check the **Demo Plan** to see what
judges will actually see; it is the reason each P0 exists.

### Notion tool tips
- The "query data source" (SQL) tool has a **usage limit** on our plan. Prefer `search` scoped to the Task Board data source (`collection://68af27b3-01c7-487d-807e-71af7d3fc536`) and `fetch` of individual pages, and use SQL sparingly.
- Fetching the Task Board page shows the dependency graph; fetching a task page shows all its properties.

---

## 6. When you finish something

1. **Tests pass first:** `pytest -q`. Commit only on green; chain them (`pytest -q && git commit ...`).
2. **Pull, then push to `main`** (small, focused commits; message says what changed and why).
3. **Build Log:** add an entry at the top. Include what landed, the commit hash, how to use it (command, endpoint, file), and anything others must do differently.
4. **Task Board:** set the task to **Done**, or **In progress** with a note if part of "Done when" is still open (say which part).
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
