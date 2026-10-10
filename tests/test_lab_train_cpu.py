"""lab/train.py end to end on the CPU: the pinned training stack (torch, transformers, PEFT, TRL, datasets) trains a
tiny model of the student's own architecture on the committed fixture's rows.

The models are built here from a config (random weights, two layers: one Gated DeltaNet layer and one attention
layer, as Qwen3.5 alternates them) and saved the way `hf download --local-dir` leaves a folder, with the commit in
its metadata. The tokenizer is a small byte-level BPE trained on the fixture's text with the pre-tokenizer pattern of
the Qwen tokenizers, so a span's edges fall where they fall for the real one. The pod sampler is replaced by a
sample file written here. Nothing else is replaced, apart from what --cpu itself swaps: TRL's fused LM-head loss,
a Triton GPU kernel, by the same computation in torch.

The training stack is lab/requirements-train.txt, which the CI image does not install (`pip install -e ".[dev]"`),
so there these tests are skipped like the camera and Neon tests; install that file (and torch) to run them.
"""
from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("trl")
pytest.importorskip("datasets")
pytest.importorskip("tokenizers")

from contracts.events import parse_event  # noqa: E402
from contracts.telemetry import TrainStep  # noqa: E402
from lab import dataset, train  # noqa: E402
from lab.machine import ledger  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY = REPO_ROOT / "fixtures" / "learning" / "train-tiny"
TINY_DATASET = TINY / "tool-calling-tiny"
TINY_ROWS = TINY / "rows.jsonl"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>",
           "<tool_response>", "</tool_response>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
           "<|video_pad|>"]
QWEN_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"
              r"\s+(?!\S)|\s+")
TEXT_CONFIG = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                   num_key_value_heads=1, head_dim=16, linear_num_key_heads=2, linear_num_value_heads=2,
                   linear_key_head_dim=8, linear_value_head_dim=8, layer_types=["linear_attention", "full_attention"],
                   max_position_embeddings=4096)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "opus-test")
    monkeypatch.setenv("LAB_TIER", "opus")
    monkeypatch.delenv("LAB_TASK", raising=False)


def rows_text() -> list[str]:
    return [json.loads(line)["text"] for line in TINY_ROWS.read_text(encoding="utf-8").splitlines() if line.strip()]


def small_tokenizer(texts: list[str]):
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    core = Tokenizer(models.BPE())
    core.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(QWEN_SPLIT), behavior="isolated"),
                                                  pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    core.decoder = decoders.ByteLevel()
    core.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=420, special_tokens=SPECIAL,
                                                        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    return PreTrainedTokenizerFast(tokenizer_object=core, eos_token="<|im_end|>", pad_token="<|endoftext|>")


def downloaded(folder: Path) -> Path:
    """Mark a saved model folder the way `hf download --local-dir` does: a metadata file per file, commit first."""
    metadata = folder / ".cache" / "huggingface" / "download"
    metadata.mkdir(parents=True)
    for file in folder.iterdir():
        if file.is_file():
            (metadata / f"{file.name}.metadata").write_text(f"{COMMIT}\n{'e' * 40}\n1760000000.0\n", encoding="utf-8")
    return folder


@pytest.fixture(scope="module")
def text_model(tmp_path_factory) -> Path:
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    folder = tmp_path_factory.mktemp("models") / "tiny-qwen3.5"
    tokenizer = small_tokenizer(rows_text())
    tokenizer.save_pretrained(folder)
    torch.manual_seed(0)
    config = Qwen3_5TextConfig(vocab_size=len(tokenizer), eos_token_id=tokenizer.eos_token_id,
                               pad_token_id=tokenizer.pad_token_id, **TEXT_CONFIG)
    Qwen3_5ForCausalLM(config).save_pretrained(folder)
    return downloaded(folder)


def copy_fixture(tmp_path: Path) -> tuple[Path, Path]:
    folder = tmp_path / "datasets" / TINY_DATASET.name
    shutil.copytree(TINY_DATASET, folder)
    rows = tmp_path / "datasets" / "rows.jsonl"
    shutil.copyfile(TINY_ROWS, rows)
    return folder, rows


def sampler_file(folder: Path, watts: float) -> Path:
    """What lab/pod_power.py writes on the pod: telemetry.gpu.power samples, one a line, here every 20 ms from a
    little before now to two and a half minutes on, so even the CPU's short steps have readings inside them (a step
    shorter than the sampler's period can hold none, and its socket_w is then null)."""
    folder.mkdir(parents=True)
    start = datetime.now(timezone.utc) - timedelta(seconds=10)
    lines = []
    for n in range(8000):
        at = start + timedelta(milliseconds=20 * n)
        lines.append(json.dumps({"event_id": f"s{n}", "type": "telemetry.gpu.power", "version": 1,
                                 "ts": at.isoformat().replace("+00:00", "Z"), "node": "gpu-b", "source": "amd-smi",
                                 "period_ms": 20.0, "socket_w": watts, "gfx_mhz": None, "mem_mhz": None,
                                 "throttled": None, "power_field": "socket_power", "sampler": "lab/pod_power.py 1"}))
    path = folder / f"samples-{start:%Y-%m-%d}.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def only_row() -> dict:
    rows = [row for row in ledger.read() if row["verb"] == train.VERB]
    assert len(rows) == 1, rows
    return rows[0]


def test_a_cpu_run_trains_on_the_rows_and_records_everything_needed_to_repeat_it(tmp_path, text_model):
    folder, rows = copy_fixture(tmp_path)
    sampler_file(tmp_path / "power", 42.5)
    answer = train.main(["--cpu", "--dataset", str(folder), "--rows", str(rows), "--model", str(text_model),
                         "--out", str(tmp_path / "run"), "--run-id", "run-cpu-1", "--epochs", "2", "--batch", "2",
                         "--rank", "8", "--alpha", "16", "--lr", "1e-3", "--seed", "7", "--data-seed", "11",
                         "--workers", "0", "--power-dir", str(tmp_path / "power")])
    assert (answer["ok"], answer["outcome"]) == (True, "trained"), answer.get("error")
    refs, metrics = answer["refs"], answer["metrics"]
    split = dataset.load_split(folder, require_images=False)
    assert {key: refs[key] for key in ("run_id", "mode", "node", "dataset_id", "action_id", "split_sha256",
                                       "examples_sha256", "rows_sha256", "base_model", "base_revision", "seed",
                                       "data_seed", "model_class", "weights_dtype", "data_deleted")} == {
        "run_id": "run-cpu-1", "mode": "cpu", "node": "cpu", "dataset_id": "tool-calling-tiny",
        "action_id": "fixture:golden-questions", "split_sha256": split.sha256,
        "examples_sha256": train.file_sha256(folder / "examples.jsonl"),
        "rows_sha256": train.file_sha256(rows), "base_model": "Qwen/Qwen3.5-4B", "base_revision": COMMIT, "seed": 7,
        "data_seed": 11, "model_class": "Qwen3_5ForCausalLM", "weights_dtype": "bfloat16",
        "data_deleted": "no: a --cpu rehearsal keeps its data"}
    assert refs["loss_head"] == "torch"
    for name in ("torch", "peft", "trl", "transformers", "accelerate", "datasets"):
        assert refs[name], name
    lora = json.loads(refs["lora_config"])
    assert (lora["r"], lora["lora_alpha"], lora["lora_dropout"], lora["exclude_modules"]) == (8, 16.0, 0.0, None)
    assert "lm_head" not in lora["target_modules"]
    assert {"model.layers.0.linear_attn.in_proj_qkv", "model.layers.1.self_attn.q_proj",
            "model.layers.1.mlp.down_proj"} <= set(lora["target_modules"])
    sft = json.loads(refs["sft_config"])
    assert {key: sft[key] for key in ("bf16", "torch_compile", "seed", "data_seed", "learning_rate",
                                      "num_train_epochs", "per_device_train_batch_size", "max_length",
                                      "logging_steps", "use_cpu")} == {
        "bf16": True, "torch_compile": False, "seed": 7, "data_seed": 11, "learning_rate": 1e-3,
        "num_train_epochs": 2.0, "per_device_train_batch_size": 2, "max_length": None, "logging_steps": 1,
        "use_cpu": True}
    assert (metrics["trainer.steps"], metrics["trainer.examples"], metrics["trainer.examples_seen"]) == (6, 6, 12)
    assert metrics["trainer.samples_per_s"] == pytest.approx(12 / metrics["trainer.train_s"])
    assert refs["samples_measure"] == train.SAMPLES_MEASURE

    events = [parse_event(line) for line in (tmp_path / "run" / "telemetry.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert [type(event) for event in events] == [TrainStep] * 6
    assert [event.step for event in events] == [1, 2, 3, 4, 5, 6]
    assert {(event.run_id, event.node, event.socket_w) for event in events} == {("run-cpu-1", "cpu", 42.5)}
    assert all(math.isfinite(event.loss) and event.step_ms > 0 and event.tokens > 0 for event in events)
    assert sum(event.tokens for event in events) == metrics["trainer.tokens"]
    assert (metrics["trainer.loss_first"], metrics["trainer.loss_last"]) == (events[0].loss, events[-1].loss)
    assert (metrics["trainer.steps_with_power"], metrics["trainer.mean_socket_w"]) == (6, 42.5)
    assert (refs["power_sampler"], refs["power_source"], refs["power_field"], refs["power_period_ms"]) == \
        ("lab/pod_power.py 1", "amd-smi", "socket_power", "20")

    adapter = json.loads((tmp_path / "run" / "adapter" / "adapter_config.json").read_text(encoding="utf-8"))
    assert sorted(adapter["target_modules"]) == sorted(lora["target_modules"])
    assert (tmp_path / "run" / "adapter" / "adapter_model.safetensors").is_file()
    assert json.loads((tmp_path / "run" / "run.json").read_text(encoding="utf-8")) == answer
    assert folder.is_dir() and rows.is_file()
    row = only_row()
    assert (row["outcome"], row["refs"]["run_id"], row["metrics"]["trainer.samples_per_s"]) == \
        ("trained", "run-cpu-1", metrics["trainer.samples_per_s"])
    assert row["refs"]["lora_config"] == refs["lora_config"]


def test_the_loss_sits_on_exactly_the_span_text_with_a_qwen_style_tokenizer(text_model):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(text_model)
    for line in TINY_ROWS.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        encoded = train.span_labels(lambda text: tokenizer(text, add_special_tokens=False)["input_ids"], row["text"],
                                    [tuple(span) for span in row["spans"]],
                                    lambda ids: tokenizer.decode(list(ids), skip_special_tokens=False))
        runs, current = [], []
        for label in encoded.labels:
            if label == train.IGNORE:
                if current:
                    runs.append(current)
                current = []
            else:
                current.append(label)
        runs += [current] if current else []
        assert [tokenizer.decode(run, skip_special_tokens=False) for run in runs] == \
            [row["text"][start:end] for start, end in row["spans"]]


def test_the_cpu_loss_head_gives_the_models_own_loss_on_the_span_tokens(text_model):
    import trl.trainer.utils as trl_utils
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(text_model)
    model = AutoModelForCausalLM.from_pretrained(text_model, dtype=torch.float32)
    row = json.loads(TINY_ROWS.read_text(encoding="utf-8").splitlines()[0])
    encoded = train.span_labels(lambda text: tokenizer(text, add_special_tokens=False)["input_ids"], row["text"],
                                [tuple(span) for span in row["spans"]])
    ids = torch.tensor([encoded.ids])
    labels = torch.tensor([encoded.labels])
    kernel = trl_utils._ChunkedLogProbFunction
    with torch.no_grad():
        logits = model(input_ids=ids).logits
        expected = torch.nn.functional.cross_entropy(logits[0, :-1], labels[0, 1:], ignore_index=train.IGNORE)
        trl_utils.add_fused_lm_head(model, outputs=("log_probs", "entropy", "is_top1"))
        with train.torch_loss_head():
            fused = model(input_ids=ids, labels=labels, fused_lm_head=True)
    assert float(fused.loss) == pytest.approx(float(expected), rel=1e-5)
    assert trl_utils._ChunkedLogProbFunction is kernel


def test_a_max_length_below_the_longest_example_refuses_before_training(tmp_path, text_model):
    folder, rows = copy_fixture(tmp_path)
    answer = train.main(["--cpu", "--dataset", str(folder), "--rows", str(rows), "--model", str(text_model),
                         "--out", str(tmp_path / "run"), "--max-length", "256", "--workers", "0"])
    assert answer["outcome"] == "refused"
    assert answer["problems"][0].startswith("--max-length 256 would cut 6 of 6 examples (the longest is ")
    assert not (tmp_path / "run" / "telemetry.jsonl").exists()
    assert not (tmp_path / "run" / "adapter").exists()


def test_a_cpu_run_refuses_weights_bigger_than_a_tiny_model_before_loading_them(tmp_path, text_model):
    folder, rows = copy_fixture(tmp_path)
    big = tmp_path / "big-model"
    shutil.copytree(text_model, big)
    with (big / "more.safetensors").open("wb") as out:
        out.truncate(4 * train.CPU_PARAM_CAP + 1)
    answer = train.main(["--cpu", "--dataset", str(folder), "--rows", str(rows), "--model", str(big),
                         "--out", str(tmp_path / "run"), "--workers", "0"])
    assert (answer["outcome"], answer["problems"]) ==         ("refused", [f"--cpu takes a model of at most {train.CPU_PARAM_CAP:,} parameters; this folder's weights are "
                     "larger"])


# --- image rows --------------------------------------------------------------------------------------------------

def image_model(folder: Path) -> Path:
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    from transformers.models.qwen3_5 import Qwen3_5Config, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_vl import Qwen3VLProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor

    tokenizer = small_tokenizer(["List the objects on the desk. Keys at the left, a mug at the right."])
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessorPil(patch_size=8, temporal_patch_size=2, merge_size=2,
                                                 min_pixels=16 * 16, max_pixels=64 * 64),
        tokenizer=tokenizer, video_processor=Qwen3VLVideoProcessor(patch_size=8, temporal_patch_size=2, merge_size=2))
    processor.save_pretrained(folder)
    ids = {name: tokenizer.convert_tokens_to_ids(name) for name in SPECIAL}
    config = Qwen3_5Config(
        text_config={**TEXT_CONFIG, "vocab_size": len(tokenizer),
                     "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 3, 3],
                                         "mrope_interleaved": True, "partial_rotary_factor": 1.0}},
        vision_config=dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, patch_size=8,
                           temporal_patch_size=2, spatial_merge_size=2, out_hidden_size=32, num_position_embeddings=64),
        image_token_id=ids["<|image_pad|>"], video_token_id=ids["<|video_pad|>"],
        vision_start_token_id=ids["<|vision_start|>"], vision_end_token_id=ids["<|vision_end|>"])
    torch.manual_seed(0)
    Qwen3_5ForConditionalGeneration(config).save_pretrained(folder)
    return downloaded(folder)


def image_dataset(root: Path) -> tuple[Path, Path, dict[str, str]]:
    """An approved desk-objects export of 52 blurred 32x32 keyframes (2 to train on, 50 held out) with its frozen
    split, and the training rows for its 2 train examples: the picture in the user turn, the answer as the span."""
    from PIL import Image

    folder = root / "desk-objects-tiny"
    (folder / "images").mkdir(parents=True)
    examples, answers = [], {}
    for n in range(52):
        name = f"images/{n:04d}.jpg"
        Image.new("RGB", (32, 32), (n * 4, 100, 200 - n * 3)).save(folder / name)
        example = {"id": f"e{n:02d}", "image": name, "frame_ref": f"frames:cam-desk:f{n:02d}",
                   "ts": "2026-10-10T18:00:00Z", "context": f"caption {n}", "target": None, "verdict": None}
        if n < 2:
            answers[example["id"]] = ["Keys at the left.", "A mug at the right."][n]
            example["target"] = answers[example["id"]]
        examples.append(example)
    manifest = {"dataset_id": folder.name, "created": "2026-10-10T18:30:00Z", "examples": 52,
                "action_id": "act-approved-1",
                "spec": {"task": "desk-objects", "instruction": "List the objects on the desk.", "output_schema": {}}}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (folder / "examples.jsonl").write_text("".join(json.dumps(e) + "\n" for e in examples), encoding="utf-8")
    (folder / "split.json").write_text(json.dumps({"version": 1, "dataset_id": folder.name, "rule": "first two train",
                                                   "train": ["e00", "e01"],
                                                   "held_out": [f"e{n:02d}" for n in range(2, 52)]}), encoding="utf-8")
    rows = []
    for example in examples[:2]:
        text = ("<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>List the objects on the desk.<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n" + answers[example["id"]] + "<|im_end|>\n")
        start = text.index("</think>\n\n") + len("</think>\n\n")
        rows.append({"id": example["id"], "frame_ref": example["frame_ref"], "messages": [], "tools": [],
                     "text": text, "spans": [[start, len(text) - 1]], "images": [example["image"]]})
    path = root / "rows.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return folder, path, answers


def test_an_image_run_uses_the_image_text_class_keeps_the_vision_side_frozen_and_trains_on_the_answer(tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("torchvision")
    from transformers import AutoProcessor

    model = image_model(tmp_path / "models" / "tiny-qwen3.5-vl")
    folder, rows, answers = image_dataset(tmp_path / "datasets")
    collator = train.ImageSpanCollator(AutoProcessor.from_pretrained(model), folder)
    first = json.loads(rows.read_text(encoding="utf-8").splitlines()[0])
    item = collator.encode(first)
    tokenizer = collator.processor.tokenizer
    image_tokens = sum(token == collator.image_token for token in item["input_ids"])
    assert image_tokens == 4   # a 32x32 picture: 4x4 patches of 8, merged 2x2
    assert all(label == train.IGNORE for token, label in zip(item["input_ids"], item["labels"])
               if token == collator.image_token)
    assert tokenizer.decode([label for label in item["labels"] if label != train.IGNORE]) == \
        answers["e00"] + "<|im_end|>"
    batch = collator([first, json.loads(rows.read_text(encoding="utf-8").splitlines()[1])])
    assert batch["input_ids"].shape[0] == 2 and batch["pixel_values"].shape[0] == 32

    answer = train.main(["--cpu", "--dataset", str(folder), "--rows", str(rows), "--model", str(model),
                         "--out", str(tmp_path / "run"), "--epochs", "1", "--batch", "2", "--workers", "0"])
    assert (answer["ok"], answer["outcome"]) == (True, "trained"), answer.get("error")
    refs, metrics = answer["refs"], answer["metrics"]
    assert (refs["model_class"], refs["dataset_id"], refs["action_id"]) == \
        ("Qwen3_5ForConditionalGeneration", "desk-objects-tiny", "act-approved-1")
    lora = json.loads(refs["lora_config"])
    assert lora["exclude_modules"] == train.VISION_MODULES
    assert lora["target_modules"] and not [name for name in lora["target_modules"] if "visual" in name]
    assert metrics["trainer.longest_example_tokens"] == len(item["input_ids"])
    span_tokens = sum(len(tokenizer(answer_text + "<|im_end|>", add_special_tokens=False)["input_ids"])
                      for answer_text in answers.values())
    assert metrics["trainer.span_tokens"] == span_tokens
    events = (tmp_path / "run" / "telemetry.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(events) == metrics["trainer.steps"] == 1
