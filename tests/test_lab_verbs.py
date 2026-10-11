"""The verb registry of python -m lab is discovered from the modules' CLI_VERB declarations: what it finds
on this tree, what it refuses (a name declared twice, a declaration without its function), and that every
verb has its page under lab/docs/."""

from pathlib import Path

import pytest

from lab import __main__ as lab_cli

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS = REPO_ROOT / "lab" / "docs"

# The hand-written registry the discovery replaced (lab/__main__.py before O70), verb by verb.
REGISTRY_BEFORE_DISCOVERY = {
    "alerts": "lab.machine.alerts:main",
    "attribution-check": "lab.attribution_check:main",
    "bench": "lab.bench:main",
    "capacity": "lab.capacity:main",
    "chipsheet": "lab.kernels.chipsheet.ingest:main",
    "correctness": "lab.correctness:main",
    "credits": "lab.credits:main",
    "deploy": "lab.deploy:main",
    "desk-score": "lab.desk_score:main",
    "digest": "lab.machine.digest:main",
    "draft": "lab.machine.draft:main",
    "eval-gate": "lab.eval_gate:main",
    "experiment": "lab.machine.experiments:main",
    "gpu-power": "lab.machine.gpu_power:main",
    "handoff": "lab.machine.handoff:main",
    "interference": "lab.interference:main",
    "lab-push": "lab.lab_push:main",
    "label-agreement": "lab.label_agreement:main",
    "ledger": "lab.machine.ledger_cli:main",
    "link-speed": "lab.link_speed:main",
    "metrics": "lab.machine.metrics_report:main",
    "mirror": "lab.machine.mirror:main",
    "optimize": "lab.optimize_settings:main",
    "package": "lab.package:main",
    "placement": "lab.placement:main",
    "pool": "lab.machine.pool:main",
    "profile-decode": "lab.profile_decode:main",
    "profile-train": "lab.profile_train:main",
    "publish-adapter": "lab.publish_adapter:main",
    "quark-compare": "lab.quark_compare:main",
    "ready": "lab.machine.ready:main",
    "referee": "lab.machine.referee:main",
    "report": "lab.report:main",
    "research-check": "lab.machine.research_check:main",
    "resume": "lab.machine.stop:resume_main",
    "review": "lab.machine.review:main",
    "routing-closed-loop": "lab.routing_closed_loop:main",
    "rulings": "lab.machine.rulings:main",
    "sampler-overhead": "lab.sampler_overhead:main",
    "sit": "lab.machine.sit:main",
    "slot": "lab.machine.slots:main",
    "spot-score": "lab.spot_score:main",
    "stop": "lab.machine.stop:stop_main",
    "sweep-request": "lab.sweep_request:main",
}


def put(root: Path, rel: str, text: str) -> None:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# --- what the scan finds on this tree ---------------------------------------------------------

def test_discovery_finds_every_verb_of_the_hand_written_registry_and_nothing_it_lacked():
    """Each verb the registry held before discovery is found with the same module and function,
    and no module registers a verb the registry did not hold: a new verb lands here with its
    line (sorted) and its lab/docs page."""
    discovered = lab_cli.discover()
    assert discovered == REGISTRY_BEFORE_DISCOVERY
    assert lab_cli.VERBS == discovered


def test_the_registry_is_sorted_by_verb_name():
    assert list(lab_cli.VERBS) == sorted(lab_cli.VERBS)


def test_every_verb_has_its_page_under_lab_docs_and_a_readme_line():
    readme = (REPO_ROOT / "lab" / "README.md").read_text(encoding="utf-8")
    missing_pages = [verb for verb in lab_cli.VERBS if not (DOCS / f"{verb}.md").is_file()]
    missing_links = [verb for verb in lab_cli.VERBS if f"](docs/{verb}.md)" not in readme]
    assert missing_pages == [], f"add lab/docs/<verb>.md for {missing_pages}"
    assert missing_links == [], f"add a line linking docs/<verb>.md to lab/README.md for {missing_links}"
    stray = sorted(page.stem for page in DOCS.glob("*.md") if page.stem not in lab_cli.VERBS)
    assert stray == [], f"lab/docs pages without a verb: {stray}"


def test_the_help_listing_names_every_verb_once_in_order(capsys):
    assert lab_cli.main([]) == 0
    lines = capsys.readouterr().out.splitlines()
    listed = [line.split()[0] for line in lines[3:3 + len(lab_cli.VERBS)]]
    assert listed == list(lab_cli.VERBS)


# --- the scan on a synthetic package ---------------------------------------------------------

def test_a_duplicate_verb_name_is_refused(tmp_path):
    put(tmp_path, "one.py", 'CLI_VERB = "probe"\n\ndef main(argv):\n    return None\n')
    put(tmp_path, "machine/two.py", 'CLI_VERB = "probe"\n\ndef main(argv):\n    return None\n')
    with pytest.raises(lab_cli.RegistryError, match="'probe' is declared by both pkg.machine.two and pkg.one"):
        lab_cli.discover(tmp_path, package="pkg")


def test_the_scan_reads_single_and_multi_verb_declarations_at_any_depth_without_importing(tmp_path):
    put(tmp_path, "__init__.py", "")
    put(tmp_path, "zed.py", 'import no_such_module_anywhere\n\nCLI_VERB = "zed"   # a comment after it\n\ndef main(argv):\n    return None\n')
    put(tmp_path, "machine/switch.py",
        'CLI_VERBS = {\n    "off": "off_main",\n    "on": "on_main",\n}\n\n'
        "def off_main(argv):\n    return None\n\n\nasync def on_main(argv):\n    return None\n")
    put(tmp_path, "deep/down/here.py", 'CLI_VERB: str = "deep"\n\ndef main(argv):\n    return None\n')
    put(tmp_path, "quiet.py", 'VERB = "ledger-only"\n\ndef main(argv):\n    return None\n')
    put(tmp_path, "__main__.py", 'CLI_VERB = "never"\n\ndef main(argv):\n    return None\n')
    assert lab_cli.discover(tmp_path, package="pkg") == {
        "deep": "pkg.deep.down.here:main",
        "off": "pkg.machine.switch:off_main",
        "on": "pkg.machine.switch:on_main",
        "zed": "pkg.zed:main",
    }


@pytest.mark.parametrize("source, problem", [
    ('CLI_VERB = "probe"\n', "defines no top-level main"),
    ('CLI_VERBS = {"probe": "go"}\n\ndef main(argv):\n    return None\n', "defines no top-level go"),
    ('CLI_VERB = 7\n\ndef main(argv):\n    return None\n', "must be a string"),
    ('CLI_VERB = VERB\nVERB = "probe"\n\ndef main(argv):\n    return None\n', "must be a string literal"),
    ('CLI_VERBS = ["probe"]\n\ndef main(argv):\n    return None\n', "dict of verb name to function name"),
    ('CLI_VERB = "two words"\n\ndef main(argv):\n    return None\n', "not a verb name"),
    ('CLI_VERB = "probe"\nCLI_VERBS = {"probe": "main"}\n\ndef main(argv):\n    return None\n', "declares the verb 'probe' twice"),
])
def test_a_malformed_declaration_is_refused_with_the_module_named(tmp_path, source, problem):
    put(tmp_path, "broken.py", source)
    with pytest.raises(lab_cli.RegistryError, match=f"pkg.broken: .*{problem}"):
        lab_cli.discover(tmp_path, package="pkg")


def test_a_registered_verb_is_still_imported_only_when_called(tmp_path, monkeypatch, capsys):
    put(tmp_path, "pkg/__init__.py", "")
    put(tmp_path, "pkg/lazy.py", 'import no_such_module_anywhere\n\nCLI_VERB = "lazy"\n\ndef main(argv):\n    return None\n')
    put(tmp_path, "pkg/fine.py", 'CLI_VERB = "fine"\n\ndef main(argv):\n    """Answers fine."""\n    return {"ok": True, "outcome": "fine"}\n')
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(lab_cli, "VERBS", lab_cli.discover(tmp_path / "pkg", package="pkg"))
    assert lab_cli.main(["fine"]) == 0
    assert '"outcome": "fine"' in capsys.readouterr().out
    assert lab_cli.main(["lazy"]) == 1
    assert "cannot load pkg.lazy:main: ModuleNotFoundError" in capsys.readouterr().out
    assert "(cannot load: ModuleNotFoundError" in lab_cli.help_text()
