"""No credential-shaped text in anything this project could publish by accident.

One scan covers three places: the files git has or would commit (tracked, plus untracked and not
ignored), every file under the lab state folder (the ledger, the notes, the review files), and the
Notion post texts kept under fixtures/secrets. It looks for two kinds of match and reports each only
as path, line and label, so a failing run never prints a value:

  shapes    the prefixes and forms of the credentials this project holds (listed below)
  literals  the value of every key config/.env.example names, and of every credential-named key in
            the .env files lab.machine.ready searches, read in memory and looked for verbatim (as
            written, JSON-escaped and percent-encoded). No .env means nothing to look for.

  hub token         the word token or bearer, then 32 hex characters (a JupyterHub API token is
                    uuid4().hex): Authorization: token ..., ?token=..., "token": "...", HUB_TOKEN=...
  hugging face      hf_ followed by 30 or more letters and digits
  notion token      ntn_ (current) or secret_ (legacy) followed by 40 or more letters and digits
  tailscale key     tskey- followed by an optional kind and 20 or more key characters
  github token      ghp_, gho_, ghu_, ghs_, ghr_ or github_pat_ followed by the token body
  private key       a PRIVATE KEY header followed by a base64 body (a header alone is documentation)
  account password  PASS, PASSWORD, PASSWD or PWD as an uppercase .env style line, or as a quoted
                    value after a password key (JSON, Python); never code that reads one

A bare 32-hex pattern is deliberately not a shape: event ids (uuid4().hex), Notion page ids in links
and sha digests all have it, so it would flag the repo itself. Values that cannot be a credential are
skipped: filler such as xxxx..., a repeated unit (0123456789abcdef twice), a run of one character, or
too few distinct characters. A fake token in a test must look like one of those, or be built at run
time. fixtures/secrets holds the proof: cases.json (true and false positives, the secret bodies
written as {placeholders} that expand to generated values so the repo never holds one) and
notion_posts.txt (post texts full of Notion links, ids, hashes and key names that must stay clean).
The scan reads its own source and those fixtures too, so neither may hold a value.

Limits: binary files and .gz archives are not read, a file is read up to 64 MB, and a hub token that
is not 32 hex or a password in free prose is found only through its .env value.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import string
import subprocess
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple
from urllib.parse import quote

import pytest

from lab.machine import ready, stop

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO_ROOT / "fixtures" / "secrets"
CASES_FILE = FIXTURE_DIR / "cases.json"
POSTS_FILE = FIXTURE_DIR / "notion_posts.txt"
ENV_TEMPLATE = REPO_ROOT / "config" / ".env.example"

MAX_BYTES = 64 * 1024 * 1024
MIN_LITERAL_CHARS = 8  # a shorter .env value is a word, not a credential: it would match prose
MIN_DISTINCT_CHARS = 9  # a random 32 hex string has fewer only 3 times in a million
BINARY_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".mp4", ".mov", ".pdf", ".gz", ".zip", ".onnx",
    ".pt", ".engine", ".gguf", ".db", ".sqlite", ".woff", ".woff2", ".ttf", ".otf", ".pyc",
}
SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "site-packages", ".next",
}
POST_SEPARATOR = "=== post ==="
CREDENTIAL_KEY = re.compile(
    r"(?:^|_)(?:TOKEN|SECRET|PASS|PASSWORD|PASSWD|PWD|API_?KEY|PRIVATE_?KEY|ACCESS_?KEY|AUTH_?KEY)(?:_|$)",
    re.IGNORECASE,
)
TEMPLATE_KEY = re.compile(r"^\s*#?\s*([A-Z][A-Z0-9_]*)=", re.MULTILINE)


class Hit(NamedTuple):
    path: str
    line: int
    label: str


class Source(NamedTuple):
    name: str
    text: str


class Shape(NamedTuple):
    label: str
    regex: re.Pattern
    hints: tuple  # lowercase; a text holding none of them cannot match, so the regex is skipped
    accept: Callable  # takes the match; False drops a match that is filler or a variable


# --- what counts as a value that cannot be a credential -------------------------------------------------

FILLER_WORDS = re.compile(
    r"(?i)example|placeholder|changeme|change-me|redacted|dummy|fake|sample|todo|your[-_ ]?(?:pass|token|secret)"
    r"|\*{4,}|x{4,}"
)
VARIABLE_NAME = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+")  # AUP_HUB_PASS_A: a key name, not a value


def is_placeholder(body: str) -> bool:
    """True for text that cannot be a credential: too few distinct characters, a repeated unit, or a long run."""
    if len(set(body)) < min(MIN_DISTINCT_CHARS, len(body) // 2):
        return True
    if re.search(r"(.)\1{9,}", body):
        return True
    size = len(body)
    return any(size % unit == 0 and body == body[:unit] * (size // unit) for unit in range(1, size // 2 + 1))


def body_of(match: re.Match) -> str:
    return next((group for group in match.groups() if group), match.group(0))


def real_body(match: re.Match) -> bool:
    return not is_placeholder(body_of(match))


def real_password(match: re.Match) -> bool:
    value = body_of(match)
    return not (is_placeholder(value) or VARIABLE_NAME.fullmatch(value) or FILLER_WORDS.search(value))


# --- the shapes -------------------------------------------------------------------------------------------

_PASS_NAME = r"(?:[A-Z][A-Z0-9]*_)*(?:PASS|PASSWORD|PASSWD|PWD)(?:_[A-Z0-9]+)*"
_PASS_VALUE = r"""([^\s"'<>${}()\[\]%,;=]{8,})"""

SHAPES = (
    Shape(
        "hub token",
        re.compile(
            r"""(?i)(?:token|bearer)(?:_[a-z0-9])?["']?\s*(?:[=:]|\s)\s*["']?"""
            r"""(?:(?:token|bearer)\s+)?([0-9a-f]{32})(?![0-9a-z])"""
        ),
        ("token", "bearer"),
        real_body,
    ),
    Shape(
        "hugging face token",
        re.compile(r"(?<![A-Za-z0-9])hf_([A-Za-z0-9]{30,})"),
        ("hf_",),
        real_body,
    ),
    Shape(
        "notion token",
        re.compile(r"(?<![A-Za-z0-9])(?:ntn|secret)_([A-Za-z0-9]{40,})"),
        ("ntn_", "secret_"),
        real_body,
    ),
    Shape(
        "tailscale key",
        re.compile(r"(?<![A-Za-z0-9])tskey-(?:[a-z]+-)?([A-Za-z0-9][A-Za-z0-9_-]{19,})"),
        ("tskey-",),
        real_body,
    ),
    Shape(
        "github token",
        re.compile(r"(?<![A-Za-z0-9])(?:gh[pousr]_([A-Za-z0-9]{36,})|github_pat_([A-Za-z0-9_]{50,}))"),
        ("ghp_", "gho_", "ghu_", "ghs_", "ghr_", "github_pat_"),
        real_body,
    ),
    Shape(
        "private key",
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----(?:\s|\\n)*"
            r"(?:[A-Za-z][A-Za-z0-9-]*:[^\n]*(?:\n|\\n)(?:\s|\\n)*)*"
            r"([A-Za-z0-9+/]{40,})"
        ),
        ("private key",),
        real_body,
    ),
    Shape(
        "account password",
        re.compile(
            r"(?m:^[ \t]*(?:export[ \t]+)?" + _PASS_NAME + r"""=["']?""" + _PASS_VALUE + r"""["']?[ \t]*(?:#.*)?\r?$)"""
            r"""|(?i:["']?(?:[a-z0-9]+[_-])*(?:password|passwd|pwd)(?:[_-][a-z0-9]+)*["']?[ \t]*[:=][ \t]*"""
            r"""["']([^\s"'<>${}()\[\]%,;]{8,})["'])"""
        ),
        ("pass", "pwd"),
        real_password,
    ),
)


def scan_text(name: str, text: str, shapes: Sequence[Shape] = SHAPES) -> list[Hit]:
    """Every credential-shaped match in text as (name, line, label); the matched text is dropped."""
    lowered = text.lower()
    hits = []
    for shape in shapes:
        if not any(hint in lowered for hint in shape.hints):
            continue
        for match in shape.regex.finditer(text):
            if shape.accept(match):
                hits.append(Hit(name, text.count("\n", 0, match.start()) + 1, shape.label))
    return hits


def shape_hits(sources: Iterable[Source]) -> list[Hit]:
    return sorted(hit for source in sources for hit in scan_text(source.name, source.text))


# --- values from the .env files ---------------------------------------------------------------------------


def template_keys(path: Path = ENV_TEMPLATE) -> set[str]:
    """Key names config/.env.example lists (commented-out ones too), minus folder settings, which are not secrets."""
    text = path.read_text(encoding="utf-8")
    return {key for key in TEMPLATE_KEY.findall(text) if not key.endswith(("_DIR", "_PATH"))}


def env_literals(files: Iterable[Path], listed: Collection[str]) -> dict[str, str]:
    """Key to value for the keys worth hiding: the listed ones and any credential-named one; first file wins."""
    literals: dict[str, str] = {}
    for path in files:
        if not path.is_file():
            continue
        for key, value in ready.read_env_file(path).items():
            if key in literals or len(value) < MIN_LITERAL_CHARS:
                continue
            if key in listed or CREDENTIAL_KEY.search(key):
                literals[key] = value
    return literals


def literal_forms(value: str) -> set[str]:
    """The value as written, as it appears inside a JSON string, and percent-encoded for a URL."""
    return {value, json.dumps(value)[1:-1], quote(value, safe="")}


def literal_needles(literals: Mapping[str, str]) -> list[tuple[str, str]]:
    return [(key, form) for key, value in literals.items() for form in sorted(literal_forms(value))]


def literal_hits(sources: Iterable[Source], needles: Sequence[tuple[str, str]]) -> list[Hit]:
    found = set()
    for source in sources:
        for key, form in needles:
            start = source.text.find(form)
            while start != -1:
                found.add(Hit(source.name, source.text.count("\n", 0, start) + 1, f"env value of {key}"))
                start = source.text.find(form, start + 1)
    return sorted(found)


# --- what gets read ---------------------------------------------------------------------------------------


def walk(root: Path) -> Iterator[Path]:
    for folder, names, files in os.walk(root):
        names[:] = [name for name in names if name not in SKIP_DIRS and not (Path(folder) == root and name == "data")]
        for name in files:
            yield Path(folder) / name


def git(root: Path, *args: str) -> bytes:
    """Run git in root without the GIT_* variables a hook sets, which would point it at the hook's repository."""
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    done = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=True, timeout=60, env=env)
    return done.stdout


def project_files(root: Path) -> list[Path]:
    """What git lists as tracked, untracked or not ignored; a folder walk when root is not a git checkout."""
    if (root / ".git").exists():
        try:
            listing = git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
            return [root / name for name in listing.decode("utf-8", "replace").split("\0") if name]
        except (OSError, subprocess.SubprocessError):
            pass
    return list(walk(root))


def display_name(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def read_source(path: Path, name: str) -> Source | None:
    """The file's text, or None for a binary file, a link, or a file that vanished since it was listed."""
    if path.suffix.lower() in BINARY_SUFFIXES or path.is_symlink() or not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            data = handle.read(min(path.stat().st_size, MAX_BYTES))
    except OSError:
        return None
    if b"\0" in data[:8192]:
        return None
    return Source(name, data.decode("utf-8", errors="replace"))


def collect_tree(root: Path) -> list[Source]:
    sources = (read_source(path, display_name(path, root)) for path in project_files(root))
    return [source for source in sources if source is not None]


def collect_state(state_dirs: Iterable[Path], root: Path) -> list[Source]:
    """Every file under the lab state folders, ignored by git or not."""
    sources: dict[str, Source] = {}
    for folder in state_dirs:
        for path in (walk(folder) if folder.is_dir() else ()):
            try:
                name = display_name(path, root)
            except ValueError:  # a state folder outside the checkout, such as $LAB_STATE_DIR
                name = "lab-state/" + path.relative_to(folder).as_posix()
            source = read_source(path, name)
            if source is not None:
                sources[name] = source
    return list(sources.values())


def state_dirs() -> list[Path]:
    found = {(REPO_ROOT / "lab" / "state").resolve(), stop.state_dir().resolve()}
    return sorted(found)


def load_posts(path: Path = POSTS_FILE) -> list[str]:
    """The post texts in the fixture: what follows each separator line, up to the next one."""
    assert path.is_file(), f"expected the recorded Notion post fixture at {path.relative_to(REPO_ROOT)}"
    chunks = path.read_text(encoding="utf-8").split(POST_SEPARATOR + "\n")[1:]
    return [chunk.strip("\n") for chunk in chunks if chunk.strip()]


def explain(hits: Sequence[Hit]) -> str:
    lines = [f"  {hit.path}:{hit.line}: {hit.label}" for hit in sorted(hits)[:50]]
    more = f"\n  ... and {len(hits) - 50} more" if len(hits) > 50 else ""
    return (
        f"{len(hits)} credential-shaped value(s) found; the values are not printed:\n"
        + "\n".join(lines)
        + more
        + "\nTake the value out of the file. If it was a real credential, tell your person:"
        + " it has to be replaced, not just deleted."
    )


# --- generated values for the fixture cases ---------------------------------------------------------------

HEX = "0123456789abcdef"
ALNUM = string.ascii_letters + string.digits
BASE64 = ALNUM + "+/"
PASSWORD_CHARS = ALNUM + "-_!"
BEGIN = "-----BEGIN "
END = "-----END "


def random_text(kind: str, salt: str, length: int, alphabet: str) -> str:
    """Deterministic, random-looking text for (kind, salt): the same fixture expands to the same values."""
    out = []
    counter = 0
    while len(out) < length:
        digest = hashlib.sha256(f"{kind}|{salt}|{counter}".encode()).digest()
        out.extend(alphabet[byte % len(alphabet)] for byte in digest)
        counter += 1
    return "".join(out[:length])


def pem(salt: str, kind: str, newline: str) -> str:
    label = f"{kind} PRIVATE KEY" if kind else "PRIVATE KEY"
    body = newline.join(random_text("pem", f"{salt}|{n}", 64, BASE64) for n in range(4))
    return f"{BEGIN}{label}-----{newline}{body}{newline}{END}{label}-----"


GENERATORS: dict[str, Callable[[str], str]] = {
    "hex32": lambda salt: random_text("hex32", salt, 32, HEX),
    "HEX32": lambda salt: random_text("hex32", salt, 32, HEX).upper(),
    "hf": lambda salt: "hf_" + random_text("hf", salt, 34, ALNUM),
    "ntn": lambda salt: "ntn_" + random_text("ntn", salt, 46, ALNUM),
    "notion_legacy": lambda salt: "secret_" + random_text("legacy", salt, 43, ALNUM),
    "tskey": lambda salt: (
        "tskey-auth-k" + random_text("ts1", salt, 6, ALNUM) + "CNTRL-" + random_text("ts2", salt, 32, ALNUM)
    ),
    "github": lambda salt: "ghp_" + random_text("gh", salt, 36, ALNUM),
    "github_pat": lambda salt: (
        "github_pat_" + random_text("pat1", salt, 22, ALNUM) + "_" + random_text("pat2", salt, 59, ALNUM)
    ),
    "pem": lambda salt: pem(salt, "RSA", "\n"),
    "pem_json": lambda salt: pem(salt, "", "\\n"),
    "openssh": lambda salt: pem(salt, "OPENSSH", "\n"),
    "password": lambda salt: random_text("password", salt, 18, PASSWORD_CHARS),
}
PLACEHOLDER = re.compile(r"\{(" + "|".join(sorted(GENERATORS, key=len, reverse=True)) + r")(?:#(\w+))?\}")


def expand(text: str, salt: str) -> str:
    """Replace each {kind} or {kind#tag} in a fixture text with a generated value that is stable per salt."""
    return PLACEHOLDER.sub(lambda m: GENERATORS[m.group(1)](f"{salt}#{m.group(2) or ''}"), text)


assert CASES_FILE.is_file(), f"expected recorded fixture {CASES_FILE.relative_to(REPO_ROOT)} for the scanner proof"
CASES = json.loads(CASES_FILE.read_text(encoding="utf-8"))


# --- the real scan ----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tree() -> list[Source]:
    return collect_tree(REPO_ROOT)


@pytest.fixture(scope="module")
def state() -> list[Source]:
    return collect_state(state_dirs(), REPO_ROOT)


def test_the_repo_tree_holds_no_secret_shape(tree):
    names = {source.name for source in tree}
    for expected in ("AGENTS.md", "tests/test_no_secrets.py", "fixtures/secrets/cases.json"):
        assert expected in names, f"the scan did not read {expected}; it read {len(names)} files"
    hits = shape_hits(tree)
    assert not hits, explain(hits)


def test_the_lab_state_logs_hold_no_secret_shape(state):
    hits = shape_hits(state)
    assert not hits, explain(hits)


def test_the_recent_notion_posts_hold_no_secret_shape():
    posts = load_posts()
    assert len(posts) >= 8, f"the post fixture holds {len(posts)} posts; keep at least 8 so it stays a corpus"
    corpus = "\n".join(posts)
    # The trap the fixture exists for: a bare 32-hex scan would flag every one of these.
    assert len(re.findall(r"\b[0-9a-f]{32}\b", corpus)) >= 8, "the posts lost their ids; a bare hex scan would pass them"
    assert "https://app.notion.com/p/" in corpus and '"event_id"' in corpus
    hits = shape_hits(Source(f"{POSTS_FILE.name} post {n}", text) for n, text in enumerate(posts, 1))
    assert not hits, explain(hits)


def test_no_value_from_the_env_files_appears_in_the_repo_the_logs_or_the_posts(tree, state):
    literals = env_literals(ready.default_env_files(), template_keys())
    hits = literal_hits([*tree, *state], literal_needles(literals))
    assert not hits, explain(hits)


# --- the scanner proves itself against the fixture --------------------------------------------------------


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_scanner_judges_each_fixture_case(case):
    hits = scan_text("case", expand(case["text"], case["id"]))
    assert sorted({hit.label for hit in hits}) == sorted(case["expect"]), f"{case['id']}: {case['why']}"
    if "line" in case:
        assert [hit.line for hit in hits] == [case["line"]], f"{case['id']}: wrong line reported"


def test_every_shape_has_a_true_case_and_a_near_miss_in_the_fixture():
    ids = [case["id"] for case in CASES]
    assert len(ids) == len(set(ids)), "two fixture cases share an id"
    shapes = {shape.label for shape in SHAPES}
    flagged = {label for case in CASES for label in case["expect"]}
    near = {case["near"] for case in CASES if not case["expect"] and "near" in case}
    assert shapes - flagged == set(), f"shapes with no true positive case: {sorted(shapes - flagged)}"
    assert shapes - near == set(), f"shapes with no near-miss case: {sorted(shapes - near)}"
    assert flagged <= shapes and near <= shapes, "a case names a shape the scanner does not have"


def test_the_generated_values_in_true_cases_are_not_filler():
    for kind, make in GENERATORS.items():
        value = make("sanity")
        assert not is_placeholder(value), f"the {kind} generator makes values the scanner would call filler"


@pytest.mark.parametrize(
    "body, filler",
    [
        ("0123456789abcdef0123456789abcdef", True),
        ("deadbeefdeadbeefdeadbeefdeadbeef", True),
        ("a" * 32, True),
        ("x" * 34, True),
        ("a1b2c3d4e5f60718293a4b5c6d7e8f90", False),
        ("Summer2026", False),
        ("12ab34cd", False),
        ("aaaaaaaa", True),
    ],
)
def test_filler_is_told_from_a_value(body, filler):
    assert is_placeholder(body) is filler


@pytest.mark.parametrize("kind", ["hex32", "hf", "ntn", "notion_legacy", "tskey", "github", "github_pat", "pem", "password"])
def test_a_secret_pasted_into_any_recorded_post_is_flagged_on_its_own_line(kind):
    expected = {
        "hex32": "hub token", "hf": "hugging face token", "ntn": "notion token", "notion_legacy": "notion token",
        "tskey": "tailscale key", "github": "github token", "github_pat": "github token",
        "pem": "private key", "password": "account password",
    }[kind]
    sentence = {
        "hex32": "Authorization: token {hex32}", "hf": "HF_TOKEN={hf}", "ntn": "NOTION_TOKEN={ntn}",
        "notion_legacy": "NOTION_TOKEN={notion_legacy}", "tskey": "tailscale up --authkey={tskey}",
        "github": "remote https://{github}@github.com/org/repo", "github_pat": "GH={github_pat}",
        "pem": "{pem}", "password": "AUP_HUB_PASS_A={password}",
    }[kind]
    for n, post in enumerate(load_posts(), 1):
        lines = post.splitlines()
        for at in (0, len(lines) // 2, len(lines)):
            planted = lines[:at] + [expand(sentence, f"post{n}")] + lines[at:]
            hits = scan_text(f"post {n}", "\n".join(planted))
            assert [(hit.line, hit.label) for hit in hits] == [(at + 1, expected)], (
                f"post {n}: a {kind} planted at line {at + 1} was not reported exactly once on that line"
            )


# --- the parts the real scan relies on --------------------------------------------------------------------


def test_the_tree_scan_reads_tracked_and_untracked_files_but_not_ignored_ones(tmp_path):
    token_line = "HUB_TOKEN=" + GENERATORS["hex32"]("tree")
    (tmp_path / ".gitignore").write_text("ignored.txt\n", encoding="utf-8")
    for name in ("tracked.txt", "untracked.txt", "ignored.txt"):
        (tmp_path / name).write_text(token_line + "\n", encoding="utf-8")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", "tracked.txt")
    hits = shape_hits(collect_tree(tmp_path))
    assert hits == [Hit("tracked.txt", 1, "hub token"), Hit("untracked.txt", 1, "hub token")]


def test_the_tree_scan_without_git_walks_the_folder_and_skips_binaries_and_environments(tmp_path):
    token_line = "HUB_TOKEN=" + GENERATORS["hex32"]("walk")
    for name in ("src/a.py", ".venv/lib/b.py", "node_modules/c.js", "data/events/d.jsonl", "img/e.png", "f.txt"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes((token_line + "\n").encode())
    (tmp_path / "g.bin").write_bytes(b"\0\0" + token_line.encode())
    hits = shape_hits(collect_tree(tmp_path))
    assert hits == [Hit("f.txt", 1, "hub token"), Hit("src/a.py", 1, "hub token")]


def test_the_state_scan_reads_every_file_under_the_state_folders_including_ignored_notes(tmp_path):
    root = tmp_path / "repo"
    folder = root / "lab" / "state"
    (folder / "notes").mkdir(parents=True)
    (folder / "ledger.jsonl").write_text('{"note": "ok"}\n', encoding="utf-8")
    (folder / "notes" / "handoff.md").write_text(
        "first\nsecond\nAUP_HUB_TOKEN_C=" + GENERATORS["hex32"]("n") + "\n", encoding="utf-8"
    )
    shared = tmp_path / "shared-state"
    (shared / "deep").mkdir(parents=True)
    (shared / "deep" / "ledger.jsonl").write_text("x\nbearer " + GENERATORS["hex32"]("e") + "\n", encoding="utf-8")
    hits = shape_hits(collect_state([folder, shared, tmp_path / "missing"], root))
    assert hits == [Hit("lab-state/deep/ledger.jsonl", 2, "hub token"), Hit("lab/state/notes/handoff.md", 3, "hub token")]


def test_the_template_lists_its_keys_commented_out_ones_too_and_not_folder_settings(tmp_path):
    template = tmp_path / ".env.example"
    template.write_text(
        "# Copy to .env\nDATA_DIR=data\nCAL_URL=\n# Optional\n# GROUP_TOKEN=\nBOT_PATH=\nPLAIN=\n", encoding="utf-8"
    )
    assert template_keys(template) == {"CAL_URL", "GROUP_TOKEN", "PLAIN"}
    real = template_keys()
    assert real >= {"OUTLOOK_ICS_URL", "NEON_DATABASE_URL"}, "the real template lost its calendar and database keys"


def write_env(path: Path, pairs: Mapping[str, str]) -> Path:
    path.write_text("".join(f"{key}={value}\n" for key, value in pairs.items()), encoding="utf-8")
    return path


def test_env_literals_take_listed_and_credential_named_keys_with_a_long_enough_value(tmp_path):
    names = ("CAL_URL", "UNLISTED_API_TOKEN", "AUP_HUB_PASS_A", "ACCOUNT_NAME", "SHORT_TOKEN", "EMPTY_TOKEN")
    values = {name: random_text("env", name, 24, ALNUM) for name in names}
    values["SHORT_TOKEN"], values["EMPTY_TOKEN"] = "abc1234", ""
    first = write_env(tmp_path / "first.env", values)
    later = {"UNLISTED_API_TOKEN": "second-file-value-is-ignored", "LATE_SECRET": "x" * 12}
    second = write_env(tmp_path / "second.env", later)
    literals = env_literals([tmp_path / "absent.env", first, second], {"CAL_URL"})
    # CAL_URL is listed; the token, password and secret keys are credential-named; ACCOUNT_NAME is neither;
    # SHORT_TOKEN is under 8 characters and EMPTY_TOKEN has no value.
    assert sorted(literals) == ["AUP_HUB_PASS_A", "CAL_URL", "LATE_SECRET", "UNLISTED_API_TOKEN"]
    assert literals["UNLISTED_API_TOKEN"] == values["UNLISTED_API_TOKEN"], "the first file defining a key must win"


def test_a_literal_env_value_is_found_with_its_key_name_in_every_written_form():
    value = random_text("lit", "a", 20, ALNUM) + '"\\@:' + random_text("lit", "b", 6, ALNUM)
    other = random_text("lit", "c", 30, ALNUM)
    needles = literal_needles({"AUP_HUB_PASS_C": value, "NEON_DATABASE_URL": other})
    sources = [
        Source("plain.txt", f"a\nAUP_HUB_PASS_C={value}\n"),
        Source("log.jsonl", json.dumps({"password": value}) + "\n"),
        Source("url.txt", f"x\ny\npostgresql://user:{quote(value, safe='')}@host/db\n"),
        Source("twice.md", f"{other} and {other}\nnothing\n{other}\n"),
        Source("clean.md", "AUP_HUB_PASS_C and NEON_DATABASE_URL are key names, not values\n"),
    ]
    assert literal_hits(sources, needles) == [
        Hit("log.jsonl", 1, "env value of AUP_HUB_PASS_C"),
        Hit("plain.txt", 2, "env value of AUP_HUB_PASS_C"),
        Hit("twice.md", 1, "env value of NEON_DATABASE_URL"),
        Hit("twice.md", 3, "env value of NEON_DATABASE_URL"),
        Hit("url.txt", 3, "env value of AUP_HUB_PASS_C"),
    ]


def test_the_report_names_the_place_and_the_key_but_never_the_value():
    secret = GENERATORS["hex32"]("report")
    password = GENERATORS["password"]("report")
    sources = [Source("lab/state/ledger.jsonl", f'{{"hdr": "Authorization: token {secret}", "pw": "{password}"}}\n')]
    hits = shape_hits(sources) + literal_hits(sources, literal_needles({"AUP_HUB_PASS_B": password}))
    message = explain(hits)
    assert "lab/state/ledger.jsonl:1: hub token" in message
    assert "lab/state/ledger.jsonl:1: env value of AUP_HUB_PASS_B" in message
    assert secret not in message and password not in message
