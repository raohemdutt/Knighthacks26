# L8 real run: publish the adapter to the Hugging Face Hub

The build session had no Hub, Notion or secrets, so every test runs on the fake in `lab/hf_fake.py`. These steps
are what remains, to be run by Ilan (or the agent working for Ilan) on a machine with network access.

1. Prerequisites: the eval gate's ledger row for the run exists in the lab ledger (metrics `learning.heldout.current`,
   `.candidate`, `.teacher`, `.n_items`, `.n_groups`, `.noise`, `.p_value`; `python -m lab ledger` shows it), the trained
   adapter folder holds `adapter_config.json` (with `base_model_name_or_path` `Qwen/Qwen3.5-4B`) and
   `adapter_model.safetensors`, and the task's `TaskSpec` is saved as JSON.
2. `pip install huggingface_hub` (not a project dependency; the client imports it lazily).
3. The token is read from the environment variable `HF_TOKEN` (the write token in the workspace `.env`; export it from
   there, do not paste it anywhere):

   ```
   python -m lab publish-adapter --repo <team-owner>/<repo-name> --adapter-dir <adapter dir> --task-spec <task spec.json> --run-id <gate run id>
   ```

   Add `--private` to create the repo private. Rehearse first with `--fake` in place of a token: same steps, nothing
   uploaded.
4. A refusal prints one JSON line with the outcome: `base_license` (the base's card is no longer apache-2.0),
   `privacy` (a file in the upload folder matched; the line names the file and the kind, never the text),
   `row_incomplete`, `no_ledger_row`, `wrong_base`, `bad_adapter`, `bad_task_spec`, `missing_token`.

Deferred to this run (cannot be proven offline):
- The adapter and card actually appear on the Hub, and the card renders with the license and base model shown.
- The link goes on the Devpost, Hugging Face track.
- The real `read_card` of `Qwen/Qwen3.5-4B` returns apache-2.0 on the day of the publish.
