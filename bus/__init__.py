"""In-process event bus + append-only JSONL event log (F3).

Log lives at data/events/*.jsonl and is the source of truth; context tables are
projections rebuilt by replaying it. Publish/subscribe on contracts.events.Event.
"""
