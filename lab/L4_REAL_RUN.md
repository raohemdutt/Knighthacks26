# L4 eval gate: the real run

Everything in `lab/eval_gate.py` is proven offline against `lab/eval_gate_fake.py`
(`tests/test_lab_eval_gate.py`). This file is what is left for the hardware: the
commands in order, where each runs, the env var NAMES it needs (never values) and
what each should print. Nothing here has run yet.

## Open before the first real run (need a decision or a fix outside L4's files)

1. **The teacher is not reachable by the scorer.** `contexts.agent.eval` builds a
   plain `LlamaServerRuntime(base_url=...)`, which sends no `Authorization` header.
   The teacher (vLLM on pod-a, port 8000) is only reachable from outside through the
   hub proxy, which needs `Authorization: token <hub token>`. Options:
   (a) a small local forwarder that adds the token (a new task; `lab/proxy.py`
   `PodLink` already does the auth), giving the gate `--teacher-url http://127.0.0.1:<port>`;
   (b) a scorer flag for extra headers (a change to `contexts/agent/eval.py`,
   Dominick's file). Until one exists, step 5 cannot score the teacher.
2. **The scorer asks only the golden questions.** On the committed
   `tool-calling-fixture`, split.json holds 58 examples out, and the scorer asks only
   the 24 golden wordings among them (the 34 variants come back as `unscored`). Proven
   by `test_the_scorer_asks_only_the_golden_wordings_so_24_of_the_58_held_out_items_are_scored`.
   24 is under the 50-item floor, so the gate reports and never deploys on this set.
3. **4 groups can never reach p < 0.05.** The held-out side is 4 frame_ref groups; the
   exact two-sided sign-flip test over 4 groups has a smallest p of 2/16 = 0.125
   (`test_the_committed_fixture_scores_58_items_in_4_groups_and_cannot_deploy`). A
   deploy needs at least 6 groups, all in favour. A held-out set with more independent
   groups (a new frozen split, a new baseline) is the people's call.
4. **Greedy decoding is set on the servers, not by the client.** `runtime/llama.py`
   sends `temperature: 0.2` and `max_tokens: 512` for chat (the 512 is pinned by
   `test_the_fixed_max_output_length_is_the_one_the_runtime_sends_for_chat`). Greedy
   therefore comes from a server-side `top_k` of 1, which the request does not
   override. The flags below must be checked on the real servers (step 3), and the
   gate is then run with `--decode-attested`, which every ledger row records.
   The SPOT set (`spot-qa`) is the exception: its client (`lab/spot_score.py`) sends
   `max_tokens: 1280` (`SPOT_MAX_TOKENS`, L28), because its longest held-out answer is
   977 tokens. A server's own output limit (llama-server `--n-predict`, vLLM
   `--max-model-len` minus the prompt) must not be below 1280 for a SPOT run.

## Env var names

- `LAB_STATE_DIR` (optional; default `lab/state`): where the ledger rows land.
- `LAB_ACTOR`, `LAB_TIER`, `LAB_TASK=L4`: the writer of the ledger rows.
- Teacher reach (only once item 1 is settled): `LAB_HUB_URL` or `AUP_HUB_URL_<letter>`,
  `LAB_POD_ROLES`, and the hub token key `lab/hub.py` reads for the teacher's account.
- Jetson llama-server: `ATLAS_MODEL_DIR`, `ATLAS_LLM_PORT`, `LLAMA_SERVER` (`jetson/atlas-llm.sh`).

## Steps

1. **Check the split is frozen** (any machine with the repo):

   ```
   python lab/dataset.py fixtures/learning/tool-calling-fixture --text-only --split
   ```

   Expect `"ok": true` and a `split` entry with `held_out: 58`, `held_out_frame_refs: 4`
   and the `sha256` the gate will record.

2. **Serve current and candidate on the Jetson the way the box serves them.** The
   current model is the one `data/models/current` points at, on port 8080 (the
   service). The candidate is the L5 package, pulled with `python -m lab.transfer
   pull-package ...` into `data/models/<candidate version>/`, started as the hot-swap's
   second copy:

   ```
   ATLAS_MODEL_DIR=data/models/<candidate version> ATLAS_LLM_PORT=8090 bash jetson/atlas-llm.sh
   ```

   `atlas-llm.sh` already passes `--reasoning-budget 0` (thinking off). For greedy,
   both servers need `--top-k 1` added to the args for the gate run (current
   restarted the same way on 8080). The manifest's `gguf` must name `Q4_K_M` or the
   gate refuses.

3. **Check the decode setting on each server** (on the Jetson):

   ```
   curl -s http://127.0.0.1:8080/props | python3 -c "import json,sys; p=json.load(sys.stdin)['default_generation_settings']; print(p.get('params', p).get('top_k'))"
   curl -s http://127.0.0.1:8090/props | python3 -c "import json,sys; p=json.load(sys.stdin)['default_generation_settings']; print(p.get('params', p).get('top_k'))"
   ```

   Expect `1` from both. For the teacher (vLLM on pod-a), launch with a `top_k` of 1
   and thinking off as server defaults (vLLM's `--override-generation-config
   '{"top_k": 1}'`; for thinking, a default chat-template kwarg `enable_thinking:
   false`; check both flag names against the vLLM commit `lab/pod_a_teacher.sh` builds)
   and confirm one chat request without `chat_template_kwargs` returns no thinking text.

4. **Optional bf16 preview.** A bf16 server of the candidate (any host) may be
   passed as `--preview-url`; it is recorded under `preview.candidate_bf16` and never
   decides.

5. **Run the gate** (on the Jetson, or a laptop on its LAN with the URLs changed),
   from a clean checkout of the commit to be recorded (the gate refuses uncommitted
   changes under `contexts/`, `runtime/`, `fixtures/golden/` and `config/`):

   ```
   LAB_TASK=L4 python -m lab eval-gate --dataset fixtures/learning/tool-calling-fixture \
       --run-id <run id> --current-url http://127.0.0.1:8080 --candidate-url http://127.0.0.1:8090 \
       --teacher-url <teacher url from item 1> \
       --candidate-manifest data/models/<candidate version>/manifest.json \
       --decode-attested --events-dir data/events
   ```

   Expect one JSON line with `"ok": true`, `decision`, `reasons`, `scores`
   (`current`, `candidate`, `teacher`, dotted per-metric keys, `gate.*`), `n_items`,
   `n_groups`, `noise`, `p_value`, `unscored`, `split_sha256`, `scorer_commit`. On the
   committed fixture today: `n_items` 24, `unscored` 34, `n_groups` 4, decision `reject`
   (items 2 and 3). A refusal prints `"ok": false, "outcome": "refused"` with the reason
   and writes a `refused` ledger row.

6. **Check what it left:**

   ```
   python -m lab ledger tail 6
   ls lab/state/eval-gate/<run id>/
   ```

   Expect `eval-gate.score` rows (current twice, candidate, teacher, and preview if
   given) and one `eval-gate` row whose outcome is the decision, each with
   `conditions.decode` = `{"thinking": "off", "sampling": "greedy", "max_tokens": 512}`
   (1280 for a `spot-qa` run),
   `conditions.scorer_commit`, `conditions.split_sha256` and `decode_source` `attested`.
   The folder holds `scores.json`, `current-1.json`, `current-2.json`, `candidate.json`,
   `teacher.json` and `learning.run.evaluated.jsonl`; `data/events/<day>.jsonl` holds
   the same event for the box.

7. **Slide numbers:**

   ```
   python lab/slide_numbers.py --eval-before lab/state/eval-gate/<run id>/current-1.json \
       --eval-after lab/state/eval-gate/<run id>/candidate.json
   ```

   Expect the "Golden questions: before -> after" row with the held-out scores.

## Done when, clause by clause: what is offline and what waits here

- The real scorer subprocess, server decode settings, the Q4_K_M GGUF served by
  llama-server, and the teacher score: steps 2 to 5 (items 1 and 4).
- Every other clause is proven offline in `tests/test_lab_eval_gate.py`.
