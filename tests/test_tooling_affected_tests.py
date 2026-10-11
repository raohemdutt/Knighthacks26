"""tooling/affected_tests.py: the test files a change maps to, on synthetic diffs over a synthetic repo tree."""

from pathlib import Path

import pytest

from tooling import affected_tests as at

REPO_ROOT = Path(__file__).resolve().parents[1]


def put(root: Path, rel: str, text: str = "") -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    """A small repo: a lab module, a machine module that imports it, tests for each, a fixture-reading test."""
    put(tmp_path, "lab/__init__.py")
    put(tmp_path, "lab/__main__.py", "VERBS = {}\n")
    put(tmp_path, "lab/widget.py", "def spin():\n    return 1\n")
    put(tmp_path, "lab/machine/__init__.py")
    put(tmp_path, "lab/machine/gear.py", "from lab import widget\n\ndef turn():\n    return widget.spin()\n")
    put(tmp_path, "lab/machine/alone.py", "def nothing():\n    return None\n")
    put(tmp_path, "contracts/__init__.py")
    put(tmp_path, "contracts/events.py", "class Event:\n    pass\n")
    put(tmp_path, "tests/test_widget.py", "from lab.widget import spin\n")
    put(tmp_path, "tests/test_lab_gear.py", "from lab.machine import gear\n")
    put(tmp_path, "tests/test_lab_other.py", "import json\n")
    put(tmp_path, "tests/test_fixture.py", 'FIX = REPO / "fixtures" / "lab" / "sample.json"\n')
    put(tmp_path, "contexts/agent/__init__.py")
    put(tmp_path, "contexts/agent/tests/__init__.py")
    put(tmp_path, "contexts/agent/tests/test_planner.py", "from contracts.events import Event\n")
    put(tmp_path, "fixtures/lab/sample.json", "{}\n")
    put(tmp_path, "docs/notes.md", "nothing\n")
    return tmp_path


def test_a_changed_module_runs_the_tests_that_import_it_directly_or_through_another_module(repo):
    selected, reasons, unmapped = at.select(["lab/widget.py"], root=repo)
    assert selected == ["tests/test_lab_gear.py", "tests/test_widget.py"]
    assert unmapped == []
    assert reasons == ["lab/widget.py: imported by 2 test file(s)"]


def test_a_changed_test_file_and_the_dispatcher_run_that_file_and_every_lab_test(repo):
    selected, reasons, unmapped = at.select(["tests/test_lab_other.py", "lab/__main__.py"], root=repo)
    assert selected == ["tests/test_lab_gear.py", "tests/test_lab_other.py"]
    assert unmapped == []
    assert reasons == ["tests/test_lab_other.py: a test file runs itself",
                       "lab/__main__.py: every tests/test_lab_* file (2)"]


def test_a_contract_change_runs_the_whole_suite(repo):
    selected, reasons, unmapped = at.select(["docs/notes.md", "contracts/events.py"], root=repo)
    assert selected is None
    assert reasons == ["contracts/events.py: shared by every test, so the whole suite runs"]
    assert at.command(None) == "python -m pytest -q"


@pytest.mark.parametrize("path", ["runtime/x.py", "bus/log.py", "pyproject.toml", "tests/conftest.py"])
def test_the_other_shared_files_run_the_whole_suite_too(repo, path):
    assert at.select([path], root=repo)[0] is None


def test_a_fixture_runs_the_tests_that_name_it_and_a_doc_is_unmapped(repo):
    selected, reasons, unmapped = at.select(["fixtures/lab/sample.json", "docs/notes.md", "lab/machine/alone.py"], root=repo)
    assert selected == ["tests/test_fixture.py"]
    assert unmapped == ["docs/notes.md", "lab/machine/alone.py"]
    assert reasons == ["fixtures/lab/sample.json: named by 1 test file(s)"]


def test_the_readme_maps_to_the_lab_tests_and_a_context_test_runs_itself(repo):
    selected, _, _ = at.select(["lab/README.md", "contexts/agent/tests/test_planner.py"], root=repo)
    assert selected == ["contexts/agent/tests/test_planner.py", "tests/test_lab_gear.py", "tests/test_lab_other.py"]


def test_the_command_lists_the_files_and_xdist_adds_the_workers():
    assert at.command(["tests/test_a.py", "tests/test_b.py"]) == "python -m pytest -q tests/test_a.py tests/test_b.py"
    assert at.command(["tests/test_a.py"], xdist=True) == "python -m pytest -q -n auto tests/test_a.py"
    assert at.command([]).startswith("#")


def test_main_prints_the_reasons_then_the_files_then_the_command(repo, monkeypatch, capsys):
    monkeypatch.setattr(at, "REPO_ROOT", repo)
    assert at.main(["--files", "lab/widget.py"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "# lab/widget.py: imported by 2 test file(s)"
    assert lines[1:3] == ["tests/test_lab_gear.py", "tests/test_widget.py"]
    assert lines[3] == "python -m pytest -q tests/test_lab_gear.py tests/test_widget.py"
    assert at.main(["--files", "docs/notes.md", "--quiet"]) == 0
    assert capsys.readouterr().out == "# no test file maps to the changed files\n"


def test_the_real_repo_maps_a_leaf_module_to_its_tests_and_a_hub_module_to_everything_downstream():
    selected, _, _ = at.select(["lab/credits.py"], root=REPO_ROOT)
    assert selected is not None
    assert "tests/test_lab_credits.py" in selected
    assert "tests/test_lab_stop_ready.py" not in selected
    selected, _, _ = at.select(["lab/machine/stop.py"], root=REPO_ROOT)
    assert {"tests/test_lab_stop_ready.py", "tests/test_lab_ledger_cli.py", "tests/test_lab_credits.py"} <= set(selected)


def test_a_path_is_found_written_as_text_or_joined_in_code():
    pattern = at.path_pattern("fixtures/lab/credits")
    assert pattern.search('FIX = REPO / "fixtures" / "lab" / "credits"')
    assert pattern.search("reads fixtures/lab/credits/ as recorded")
    assert not pattern.search("fixtures/lab/credit")
    assert not pattern.search("fixtures/telemetry/credits")


def test_module_names_cover_packages_and_the_tests_folder():
    assert at.module_names("lab/machine/stop.py") == {"lab.machine.stop"}
    assert at.module_names("contexts/agent/__init__.py") == {"contexts.agent"}
    assert at.module_names("tests/helpers.py") == {"tests.helpers", "helpers"}
    assert at.module_names("lab/README.md") == set()
