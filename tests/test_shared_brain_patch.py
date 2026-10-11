"""tooling/demo/shared_brain_patch.py: the box's gateway files get exactly the wiring this branch has (G4, G6)."""

import importlib.util
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("shared_brain_patch", REPO / "tooling" / "demo" / "shared_brain_patch.py")
patch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patch)

APP = (REPO / "contexts" / "gateway" / "app.py").read_text(encoding="utf-8")
G6_ROUTE = ("    # GET /api/circle/shared-brain: the two-pod brain live (G6): members, watts, link, serving; 200 in every state\n"
            "    app.include_router(shared_brain_router(build_brain_pair()))\n")
G6_IMPORT = "from contexts.circle.brain_pair import build_brain_pair, shared_brain_router\n"
METER_IMPORT = "from contexts.learning.answer_energy import AnswerMeter\n"
METER = ("                                                      llm_url=atlas.settings.llm_url), guard=authorized,\n"
         "                                     meter=AnswerMeter(atlas.telemetry)))  # + what each answer cost on the box (G6)\n")
BEFORE_METER = "                                                      llm_url=atlas.settings.llm_url), guard=authorized))\n"


def before_g6(text: str) -> str:
    """app.py as it was before G6: the four G6 additions taken out again."""
    for block, back in ((G6_ROUTE, ""), (G6_IMPORT, ""), (METER_IMPORT, ""), (METER, BEFORE_METER)):
        assert text.count(block) == 1, block
        text = text.replace(block, back)
    return text


@pytest.fixture
def root(tmp_path):
    for rel in ("contexts/circle/brain_power.py", "contexts/circle/brain_pair.py", "contexts/learning/answer_energy.py",
                "contexts/gateway/bootstrap.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, tmp_path / rel)
    return tmp_path


def test_a_copy_from_before_g6_gets_exactly_this_branchs_wiring(root):
    app = root / "contexts/gateway/app.py"
    app.write_text(before_g6(APP), encoding="utf-8")
    said = patch.apply(app, patch.APP, check=False)
    assert app.read_text(encoding="utf-8") == APP
    assert sum("added" in line for line in said) == 4
    assert patch.apply(app, patch.APP, check=False) and app.read_text(encoding="utf-8") == APP  # twice: no change


def test_a_copy_without_the_deploy_button_skips_the_meter_and_still_gets_the_route(root):
    app = root / "contexts/gateway/app.py"
    text = before_g6(APP).replace("from contexts.learning.lab_deploy import DeployRunner, lab_deploy_router\n", "")
    text = text.replace(BEFORE_METER, "")
    app.write_text(text, encoding="utf-8")
    said = patch.apply(app, patch.APP, check=False)
    out = app.read_text(encoding="utf-8")
    assert G6_ROUTE in out and G6_IMPORT in out and "AnswerMeter" not in out
    assert sum("skipped" in line for line in said) == 2


def test_check_changes_nothing(root):
    app = root / "contexts/gateway/app.py"
    app.write_text(before_g6(APP), encoding="utf-8")
    said = patch.apply(app, patch.APP, check=True)
    assert app.read_text(encoding="utf-8") == before_g6(APP)
    assert sum("would add" in line for line in said) == 4
