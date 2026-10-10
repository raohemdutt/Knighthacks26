"""Tests for the gfx1151 inference settings note and the Ax search space it pins.

The JSON file is what the optimization run reads, so these tests hold it to the note: every searched
parameter is on the note's shortlist, every baseline is a legal value of its parameter, every fixed
setting carries its pinned value, and the note itself passes the research-note checker.
"""

import json
import re
from pathlib import Path

from lab.machine import research_check as rc

REPO_ROOT = Path(__file__).resolve().parents[1]
KERNELS = REPO_ROOT / "lab" / "refine" / "research" / "kernels"
NOTE = KERNELS / "settings_inference_gfx1151.md"
SPACE = KERNELS / "data" / "searchspace-inference-gfx1151.json"

AX_TYPES = {"choice", "range", "fixed"}
VALUE_TYPES = {"str": str, "int": int, "float": (int, float), "bool": bool}


def space() -> dict:
    return json.loads(SPACE.read_text(encoding="utf-8"))


def shortlist_names() -> list[str]:
    """The setting names in the first column of the note's shortlist table, in table order."""
    lines = NOTE.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("## Shortlist"))
    names = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        match = re.match(r"\|\s*\d+\s*\|\s*`([^`]+)`", line)
        if match:
            names.append(match.group(1))
    return names


def test_the_note_passes_the_research_checker_with_its_sources_index():
    assert rc.check(NOTE) == []
    assert rc.check(KERNELS) == []


def test_the_search_space_names_its_chip_engine_and_note():
    data = space()
    assert data["target"]["gfx"] == "gfx1151"
    assert data["target"]["engine"].startswith("vllm 0.19.2")
    assert data["note"] == "lab/refine/research/kernels/settings_inference_gfx1151.md"


def test_every_parameter_has_an_ax_type_and_a_legal_baseline():
    for param in space()["parameters"]:
        assert param["type"] in AX_TYPES, param["name"]
        kind = VALUE_TYPES[param["value_type"]]
        baseline = space()["baseline"][param["name"]]
        assert isinstance(baseline, kind), param["name"]
        if param["type"] == "choice":
            assert len(param["values"]) >= 2, param["name"]
            assert all(isinstance(v, kind) for v in param["values"]), param["name"]
            assert baseline in param["values"], param["name"]
        elif param["type"] == "range":
            low, high = param["bounds"]
            assert low < high and low <= baseline <= high, param["name"]
        else:
            assert param["value"] == baseline, param["name"]


def test_the_searched_parameters_are_the_shortlist_and_at_most_ten():
    searched = [p["name"] for p in space()["parameters"] if p["type"] != "fixed"]
    assert len(searched) <= 10
    assert searched == shortlist_names()


def test_every_choice_value_maps_to_server_flags_or_environment():
    data = space()
    for param in data["parameters"]:
        if param["type"] == "fixed":
            continue
        mapping = data["mapping"][param["name"]]
        for value in param["values"]:
            launch = mapping[str(value).lower() if isinstance(value, bool) else str(value)]
            assert set(launch) <= {"env", "args"}, param["name"]
            assert launch.get("env") or launch.get("args") is not None, param["name"]


def test_every_shortlisted_setting_has_a_prediction_on_both_metrics():
    data = space()
    for name in shortlist_names():
        prediction = data["predictions"][name]
        assert prediction["decode_tok_s"] and prediction["j_per_token"], name
        assert prediction["confidence"] in {"low", "medium", "high"}, name


def test_the_pinned_settings_are_fixed_and_cover_the_teacher_serve_line():
    fixed = {p["name"]: p["value"] for p in space()["parameters"] if p["type"] == "fixed"}
    assert fixed["max_model_len"] == 16384
    assert fixed["gpu_memory_utilization"] == 0.90
    assert fixed["mm_encoder_attn_backend"] == "TRITON_ATTN"
    assert fixed["max_num_seqs"] == 256


def test_every_source_id_the_note_cites_is_defined_exactly_once_in_the_index():
    cited = set(re.findall(r"\[(S\d+)\]", NOTE.read_text(encoding="utf-8")))
    index = (KERNELS / "SOURCES.md").read_text(encoding="utf-8")
    defined = re.findall(r"^\| (S\d+) \|", index, flags=re.M)
    for source_id in sorted(cited):
        assert defined.count(source_id) == 1, source_id
