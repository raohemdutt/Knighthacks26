# Atlas (Knighthacks)

Screenless, always-on AI box with eyes. Design, contracts and tasks live in
Notion: Knighthacks > Atlas Design Doc / Task Board.

**Using an AI coding agent on this repo? Read [AGENTS.md](AGENTS.md) first.**
Notion is the source of truth and the team's communication channel, and your
agent should check your Team Comms page for questions addressed to you.

## Layout

```
contracts/   Pydantic events + API schemas (shared; changes need a second reviewer)
bus/         in-process event bus + JSONL event log (F3)
runtime/     on-device models, registry, hot-swap (shared kernel)
contexts/    capture perception memory agent learning vault workspace gateway
             each grows domain/ application/ infrastructure/ api/ fakes/ tests/
web/         phone PWA
lab/         R9700 scripts: label, train, eval, package
config/      zones.yaml, rules.yaml, models.yaml, .env.example
fixtures/    clips, sample episodes, sample datasets
jetson/      Jetson connection scripts and guide
```

Rules: contexts talk only through events in `contracts/`; no reading another
context's tables; never call llama.cpp directly, go through `runtime/`.
Branches: `ctx/<context>/<feature>`.

## Setup

```
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```
