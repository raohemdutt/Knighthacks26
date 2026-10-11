"""Ask the box's model the held-out question of the live deploy, the way the eval asked it (L34).

    python3 tooling/demo/live_deploy_question.py                 # the water-bottle question
    python3 tooling/demo/live_deploy_question.py --all           # and the 14-hours door question

Run on the box (plain python3, no venv needed). Each prompt in live_deploy_question.json is the eval's exact text
(lab/tool_call_agreement.py prompt_for: the dataset's instruction, the agent's TOOLS, the question, the generation
prompt); its sha256 matches prompt_sha256 in the pod D held-out files. It goes to llama-server POST /completion with
n_predict 512, temperature 0.2 and top_k 1, as the eval's servers ran. The model's tool call is printed with the call
the dataset expects, so the room sees which tool the serving model picks.

The gateway's /api/ask is not used: it answers "have you seen my X" and "the last N hours" in code (qa.py prefetch)
before the model chooses a tool, so its answer reads the same on both models.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent


def post(url: str, body: dict) -> dict:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default="http://127.0.0.1:8080")
    p.add_argument("--all", action="store_true")
    p.add_argument("--questions", default=str(HERE / "live_deploy_question.json"))
    args = p.parse_args()
    props = json.loads(urllib.request.urlopen(args.url + "/props", timeout=10).read())
    print("serving:", str(props.get("model_path", "")).rsplit("/", 2)[-2])
    items = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    for item in items if args.all else items[:1]:
        began = time.time()
        body = post(args.url + "/completion", {"prompt": item["prompt"], "n_predict": 512, "temperature": 0.2,
                                               "top_k": 1, "return_tokens": True})
        tokens = body.get("tokens") or []
        text = post(args.url + "/detokenize", {"tokens": tokens})["content"] if tokens else body.get("content", "")
        print(f"\nQ: {item['question']}  ({time.time() - began:.1f} s)")
        print("model wrote:\n" + text.split("<|im_end|>")[0].strip())
        print("expected:", json.dumps(item["expected"]))


if __name__ == "__main__":
    main()
