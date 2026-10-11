"""Which test files a change needs: python tooling/affected_tests.py [RANGE] [--xdist]

Reads `git diff --name-only RANGE` (default origin/main...HEAD; or --files to name the changed
paths yourself) and prints the test files to run, one per line, then the pytest command on the
last line. Exit 0 always; the command is the answer.

The rules, from the widest down:

1. The whole suite when anything under contracts/, runtime/ or bus/ changes, or pyproject.toml,
   or a conftest.py: these shape every test.
2. Every tests/test_lab_*.py when lab/__main__.py or lab/README.md changes: the dispatcher and
   its README are what the lab tests check together.
3. A changed test file runs itself.
4. A changed Python module runs every test file that imports it, directly or through other
   modules of this repo (the import graph is read from the sources, nothing is imported).
5. Any other changed file (a fixture, a config file, a prompt) runs the test files whose source
   names its path or a parent folder of it, written as text or joined with / in code. A file
   nothing names is listed as unmapped so the reader can decide.

Run from the repo root. The command is `python -m pytest -q` plus the files; --xdist adds
`-n auto` (pytest-xdist). When no test maps to the change the command is a comment.
"""

from __future__ import annotations

import argparse
import ast
import re
import signal
import subprocess
import sys
from collections import deque
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIRS = ("lab", "contexts", "contracts", "runtime", "bus", "tooling", "tests", "jetson")
WHOLE_SUITE_PREFIXES = ("contracts/", "runtime/", "bus/")
WHOLE_SUITE_FILES = ("pyproject.toml",)
LAB_WIDE_FILES = ("lab/__main__.py", "lab/README.md")
DEFAULT_RANGE = "origin/main...HEAD"
PYTEST = ["python", "-m", "pytest", "-q"]


def changed_files(git_range: str, root: Path = REPO_ROOT) -> list[str]:
    """The paths `git diff --name-only RANGE` prints, relative to the repo root, with slashes."""
    out = subprocess.run(["git", "diff", "--name-only", git_range], cwd=root, check=True,
                         capture_output=True, text=True).stdout
    return [line.strip() for line in out.splitlines() if line.strip()]


def is_test_file(path: str) -> bool:
    name = PurePosixPath(path).name
    return path.endswith(".py") and name.startswith("test_") and (
        path.startswith("tests/") or "/tests/" in path)


def test_files(root: Path = REPO_ROOT) -> list[str]:
    """Every test file pytest collects by default: tests/ and contexts/*/tests/, as repo-relative paths."""
    found = set()
    for base in ("tests", "contexts"):
        for path in (root / base).rglob("test_*.py"):
            rel = path.relative_to(root).as_posix()
            if is_test_file(rel):
                found.add(rel)
    return sorted(found)


def module_names(path: str) -> set[str]:
    """The dotted names a repo file can be imported by: lab/machine/stop.py is lab.machine.stop,
    tests/helpers.py is tests.helpers and helpers (pytest puts tests/ on sys.path)."""
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if not parts or not path.endswith(".py"):
        return set()
    if parts[-1] == "__init__":
        parts = parts[:-1]
    if not parts:
        return set()
    names = {".".join(parts)}
    if parts[0] == "tests" and len(parts) > 1:
        names.add(".".join(parts[1:]))
    return names


def imported_modules(path: Path, module: str) -> set[str]:
    """The dotted module names one source file imports, relative imports resolved against `module`."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return set()
    package = module.rsplit(".", 1)[0] if "." in module else ""
    if path.name == "__init__.py":
        package = module
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_parts = package.split(".") if package else []
                base_parts = base_parts[:len(base_parts) - (node.level - 1)] if node.level > 1 else base_parts
                base = ".".join(part for part in base_parts if part)
                head = f"{base}.{node.module}" if node.module and base else (node.module or base)
            else:
                head = node.module or ""
            if head:
                names.add(head)
            for alias in node.names:
                if alias.name != "*" and head:
                    names.add(f"{head}.{alias.name}")
    return names


def import_graph(root: Path = REPO_ROOT) -> tuple[dict[str, str], dict[str, set[str]]]:
    """(module name -> repo path, repo path -> the repo paths it imports) for every .py under the package dirs."""
    by_name: dict[str, str] = {}
    sources: dict[str, Path] = {}
    for base in PACKAGE_DIRS:
        folder = root / base
        if not folder.is_dir():
            continue
        for path in folder.rglob("*.py"):
            if any(part in {"__pycache__", "node_modules", ".venv"} for part in path.parts):
                continue
            rel = path.relative_to(root).as_posix()
            sources[rel] = path
            for name in module_names(rel):
                by_name.setdefault(name, rel)
    imports: dict[str, set[str]] = {}
    for rel, path in sources.items():
        primary = sorted(module_names(rel), key=len, reverse=True)[0]
        targets = set()
        for name in imported_modules(path, primary):
            hit = by_name.get(name)
            while hit is None and "." in name:
                name = name.rsplit(".", 1)[0]
                hit = by_name.get(name)
            if hit is not None and hit != rel:
                targets.add(hit)
        imports[rel] = targets
    return by_name, imports


def importers_of(changed: set[str], imports: dict[str, set[str]]) -> set[str]:
    """Every repo file that imports a changed file, directly or through other repo files."""
    reverse: dict[str, set[str]] = {}
    for source, targets in imports.items():
        for target in targets:
            reverse.setdefault(target, set()).add(source)
    seen = set(changed)
    queue = deque(changed)
    while queue:
        for importer in reverse.get(queue.popleft(), ()):
            if importer not in seen:
                seen.add(importer)
                queue.append(importer)
    return seen - set(changed)


def path_pattern(path: str) -> re.Pattern[str]:
    """A regex that finds the path written as text (fixtures/lab/credits) or joined in code
    (REPO / "fixtures" / "lab" / "credits")."""
    joiner = r"""["']?\s*/\s*["']?"""
    return re.compile(joiner.join(re.escape(part) for part in PurePosixPath(path).parts))


def mentioned_in(path: str, tests: list[str], root: Path = REPO_ROOT) -> set[str]:
    """The test files whose source names this path or a parent folder of it (two components or more:
    a bare top-level folder such as fixtures/ names nothing)."""
    parts = PurePosixPath(path)
    needles = [path, *(str(parent) for parent in parts.parents if len(parent.parts) >= 2)]
    patterns = [path_pattern(needle) for needle in needles]
    hits = set()
    for test in tests:
        text = (root / test).read_text(encoding="utf-8", errors="replace")
        if any(pattern.search(text) for pattern in patterns):
            hits.add(test)
    return hits


def select(changed: list[str], root: Path = REPO_ROOT) -> tuple[list[str] | None, list[str], list[str]]:
    """(test files to run, or None for the whole suite; the reasons; the changed files nothing maps to)."""
    tests = test_files(root)
    reasons: list[str] = []
    for path in changed:
        if path.startswith(WHOLE_SUITE_PREFIXES) or path in WHOLE_SUITE_FILES or PurePosixPath(path).name == "conftest.py":
            reasons.append(f"{path}: shared by every test, so the whole suite runs")
            return None, reasons, []
    selected: set[str] = set()
    unmapped: list[str] = []
    graph_needed = [path for path in changed if path.endswith(".py") and not is_test_file(path)]
    imports: dict[str, set[str]] = {}
    if graph_needed:
        _, imports = import_graph(root)
    for path in changed:
        if path in LAB_WIDE_FILES:
            lab_tests = [test for test in tests if test.startswith("tests/test_lab_")]
            selected.update(lab_tests)
            reasons.append(f"{path}: every tests/test_lab_* file ({len(lab_tests)})")
        if is_test_file(path):
            if path in tests:
                selected.add(path)
                reasons.append(f"{path}: a test file runs itself")
            else:
                unmapped.append(path)
            continue
        if path.endswith(".py"):
            hits = {test for test in importers_of({path}, imports) if test in tests}
            if hits:
                selected.update(hits)
                reasons.append(f"{path}: imported by {len(hits)} test file(s)")
            elif path not in LAB_WIDE_FILES:
                unmapped.append(path)
            continue
        if path in LAB_WIDE_FILES:
            continue
        hits = mentioned_in(path, tests, root)
        if hits:
            selected.update(hits)
            reasons.append(f"{path}: named by {len(hits)} test file(s)")
        else:
            unmapped.append(path)
    return sorted(selected), reasons, unmapped


def command(selected: list[str] | None, xdist: bool = False) -> str:
    """The pytest command for the selection; a comment when nothing is selected."""
    if selected is not None and not selected:
        return "# no test file maps to the changed files"
    words = [*PYTEST, *(["-n", "auto"] if xdist else []), *(selected or [])]
    return " ".join(words)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python tooling/affected_tests.py",
                                     description="List the test files a change needs and the pytest command to run them.")
    parser.add_argument("range", nargs="?", default=DEFAULT_RANGE,
                        help=f"a git diff range (default {DEFAULT_RANGE}); ignored with --files")
    parser.add_argument("--files", nargs="*", metavar="PATH", help="the changed paths, instead of asking git")
    parser.add_argument("--xdist", action="store_true", help="add -n auto to the command (needs pytest-xdist)")
    parser.add_argument("--quiet", action="store_true", help="print only the command")
    args = parser.parse_args(argv)
    changed = args.files if args.files is not None else changed_files(args.range, REPO_ROOT)
    selected, reasons, unmapped = select(changed, root=REPO_ROOT)
    if not args.quiet:
        for reason in reasons:
            print(f"# {reason}")
        for path in unmapped:
            print(f"# {path}: no test file imports or names it (unmapped)")
        for test in selected or []:
            print(test)
    print(command(selected, args.xdist))
    return 0


if __name__ == "__main__":
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)   # so `| head` ends the listing quietly
    sys.exit(main())
