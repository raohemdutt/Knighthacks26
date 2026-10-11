"""Ask Atlas's nine demo questions, sent to a box's POST /api/ask; one JSON line each (docs/ask-atlas-diagnosis.md).

    python tooling/demo/ask_atlas_nine.py http://100.124.34.36:8000 after-box.jsonl

Each line has the question, the HTTP status, the seconds, and the answer event (answered_by, tool, tool_args, link).
ATLAS_TOKEN, when set, is sent as the bearer token."""
import json
import sys
import time

import os

import httpx

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://100.124.34.36:8000"
OUT = sys.argv[2] if len(sys.argv) > 2 else "nine.jsonl"
QUESTIONS = [
    "What is on my calendar today?",
    "Add a meeting with Sam at 3 pm tomorrow called demo prep",
    "What did I do this morning?",
    "Where is my phone?",
    "Set a reminder to call mom at 6",
    "Turn off the camera",
    "Show me the room map",
    "What is the box's model right now?",
    "How much energy did my last answer cost?",
]
with open(OUT, "a", encoding="utf-8") as f:
    for q in QUESTIONS:
        t0 = time.time()
        try:
            token = os.environ.get("ATLAS_TOKEN")
            r = httpx.post(f"{BASE}/api/ask", json={"text": q}, timeout=120,
                           headers={"Authorization": f"Bearer {token}"} if token else {})
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"raw": r.text[:500]}
            row = {"q": q, "status": r.status_code, "s": round(time.time() - t0, 1), "body": body}
        except Exception as exc:
            row = {"q": q, "status": None, "s": round(time.time() - t0, 1), "error": repr(exc)}
        f.write(json.dumps(row) + "\n")
        f.flush()
        print(json.dumps(row)[:600], flush=True)
