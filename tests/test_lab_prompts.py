"""Guards on the per-tier session prompts a sitting is launched with.

A sitting receives its tier prompt verbatim, so each file has to exist, fit in
one read, carry no credential values, stay plain ASCII, describe a role rather
than a head count, and state the duties and rails its tier is held to. The
secret scanner is proven against planted samples, built at run time so this
file holds no token-shaped literal, which makes a clean result on the real
prompts meaningful.
"""

import re
import string
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = REPO_ROOT / "lab" / "machine" / "prompts"
TIERS = ("fable", "opus", "sonnet")
WORD_LIMIT = 600

SECRET_PATTERNS = {
    "github token": r"\bgh[pousr]_[A-Za-z0-9]{20,}",
    "github fine-grained token": r"\bgithub_pat_[A-Za-z0-9_]{20,}",
    "hugging face token": r"\bhf_[A-Za-z0-9]{20,}",
    "api key": r"\bsk-[A-Za-z0-9_-]{20,}",
    "slack token": r"\bxox[abprs]-[A-Za-z0-9-]{10,}",
    "notion token": r"\b(?:ntn_|secret_)[A-Za-z0-9]{20,}",
    "aws access key id": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
    "json web token": r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}",
    "private key block": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "authorization value": r"(?i)\bauthorization\s*:\s*(?:bearer|token|basic)\s+\S{12,}",
    "bearer value": r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{20,}",
    "credential assignment": r"\b[A-Z][A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|API_?KEY|PRIVATE_?KEY)=\S{8,}",
    "credential field": r'(?i)"(?:token|secret|password|api_?key)"\s*:\s*"[^"]{8,}"',
    "token in url": r"(?i)[?&](?:token|key|secret|password)=[^\s&]{8,}",
    "long hex string": r"\b[0-9a-fA-F]{32,}\b",
    "long opaque string": r"\b[A-Za-z0-9_-]{40,}\b",
}

# Assembled from fragments so this file does not itself carry the strings it forbids.
ATTRIBUTION_MARKERS = (
    "clau" "de",
    "anthro" "pic",
    "co-authored" "-by",
    "generated " "with",
    "noreply" "@",
)

# A number (digit or word) directly before a tier name or before "sittings".
SITTING_COUNT = re.compile(
    r"(?i)\b(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
    r"(?:(?:fable|opus|sonnet)\b|(?:\w+\s+)?sittings\b)"
)

# Every verb the dispatcher is specified to define; a prompt naming any other is wrong.
KNOWN_VERBS = {
    "digest",
    "draft",
    "handoff",
    "ledger",
    "ready",
    "referee",
    "resume",
    "review",
    "rulings",
    "sit",
    "slot",
    "stop",
}

REQUIRED_EVERYWHERE = {
    "reruns the digest with its verb": r"python -m lab digest",
    "checks readiness first": r"python -m lab ready",
    "refuses other sessions' notes": r"(?i)never read\b[^.\n]*lab/state/notes",
    "runs a task through the open-ticket skill": r"/open-ticket <ID>",
    "points at AGENTS.md section 8": r"AGENTS\.md section 8",
    "points at the GPU queue section of the skill": r"section 4b",
    "reaches the GPU only through the queue verbs": r"python -m lab slot",
    "honors the stop file": r"lab/state/STOP",
    "drafts a missing verb instead of working around it": (
        r"python -m lab draft --name <[^>]+> --goal"
    ),
    "writes a handoff note with its sitting id": (
        r"python -m lab handoff --sitting <[^>]+> --note"
    ),
    "keeps the lock state honest": r"(?i)\breleased\b",
    "forbids attribution": r"(?i)attribution",
    "never merges a pull request": r"gh pr merge",
}

REQUIRED_BY_TIER = {
    "fable": {
        "names the merge duty": r"--merge\b",
        "names the referee duty": r"--referee\b",
        "runs the merge pass skill": r"/refine-merge",
        "starts the referee duty with the referee verb": (
            r"(?i)first action[^.\n]*python -m lab referee"
        ),
        "works the rulings file": r"python -m lab rulings",
        "writes the morning review": r"python -m lab review",
        "keeps rubric edits behind a ruling": r"RUBRIC\.md[^.\n]*ruling",
    },
    "opus": {
        "runs a refinement pass": r"/refine-track <track>",
        "obeys the pass rules": r"lab/refine/PASS_RULES\.md",
        "applies the old-versus-new rule": r"(?i)old versus new",
        "applies the source-forced rule": r"(?i)source-forced",
        "never edits a rubric": r"(?i)never edit[^.\n]*RUBRIC\.md",
    },
    "sonnet": {
        "handles slot verifications": r"(?i)slot verification",
        "requests a slot": r"python -m lab slot[^\n]*\brequest\b",
        "starts a slot only on its turn": r"python -m lab slot[^\n]*\bstart\b",
        "finishes a slot with its result": r"python -m lab slot[^\n]*\bdone\b",
        "complains when kept waiting": r"python -m lab slot[^\n]*\bcomplain\b",
        "prepares before asking for the card": r"(?i)prepare",
    },
}


def read_prompt(tier: str) -> str:
    path = PROMPT_DIR / f"{tier}.md"
    assert path.is_file(), (
        f"missing session prompt for the {tier} tier: expected {path.relative_to(REPO_ROOT)}"
    )
    return path.read_text(encoding="utf-8")


def find_secrets(text: str) -> list[tuple[str, int]]:
    """Return (label, line number) for every credential-shaped value in text.

    Values are never returned, so a failing assertion cannot leak one.
    """
    hits = []
    for label, pattern in SECRET_PATTERNS.items():
        for match in re.finditer(pattern, text):
            hits.append((label, text.count("\n", 0, match.start()) + 1))
    return hits


def planted(prefix: str, length: int, alphabet: str = string.ascii_letters + string.digits) -> str:
    body = (alphabet * (length // len(alphabet) + 1))[:length]
    return prefix + body


PLANTED_SECRETS = {
    "github token": planted("ghp_", 36),
    "github fine-grained token": planted("github_pat_", 40),
    "hugging face token": planted("hf_", 34),
    "api key": planted("sk-", 40),
    "slack token": planted("xoxb-", 30, string.digits),
    "notion token": planted("ntn_", 40),
    "aws access key id": planted("AKIA", 16, string.ascii_uppercase),
    "json web token": planted("eyJ", 20) + "." + planted("eyJ", 20) + "." + planted("", 20),
    "private key block": "-----BEGIN " + "RSA PRIVATE KEY-----",
    "authorization value": "Authorization: token " + planted("", 30),
    "bearer value": "send Bearer " + planted("", 30),
    "credential assignment": "NOTION_TOKEN=" + planted("", 20),
    "credential field": '{"token": "' + planted("", 20) + '"}',
    "token in url": "https://hub.example/path?token=" + planted("", 20),
    "long hex string": planted("", 40, "0123456789abcdef"),
}

CLEAN_SENTENCES = (
    "Credentials live in the workspace .env and on the Credentials page; name the key, "
    "never print a value.",
    "NOTION_TOKEN and the hub token are key names, not values.",
    "Open lab/state/review/<day>.md after python -m lab slot request for 15 minutes.",
    "Commit bf482f4 landed on main.",
)


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_file_exists_and_is_not_empty(tier):
    assert read_prompt(tier).strip(), f"{tier}.md is empty"


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_is_under_600_words(tier):
    words = len(read_prompt(tier).split())
    assert words < WORD_LIMIT, f"{tier}.md has {words} words; the limit is under {WORD_LIMIT}"


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_opens_by_naming_only_its_own_tier(tier):
    lines = read_prompt(tier).splitlines()
    assert lines and lines[0].startswith("# "), f"{tier}.md must open with a one-line title"
    title = lines[0].lower()
    assert tier in title, f"{tier}.md title does not name its tier: {lines[0]!r}"
    others = [other for other in TIERS if other != tier and other in title]
    assert not others, f"{tier}.md title names another tier: {others}"


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_names_no_secrets(tier):
    hits = find_secrets(read_prompt(tier))
    assert hits == [], f"{tier}.md holds credential-shaped values at (label, line): {hits}"


@pytest.mark.parametrize("name", sorted(PLANTED_SECRETS))
def test_scanner_flags_a_planted_secret(name):
    labels = {label for label, _ in find_secrets(f"Log in with {PLANTED_SECRETS[name]} and go.")}
    assert name in labels, f"scanner missed a planted {name}; it reported {sorted(labels)}"


@pytest.mark.parametrize("sentence", CLEAN_SENTENCES)
def test_scanner_accepts_key_names_and_locations(sentence):
    assert find_secrets(sentence) == [], f"scanner flagged clean text: {sentence!r}"


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_is_plain_ascii(tier):
    odd = sorted({ch for ch in read_prompt(tier) if ord(ch) > 127})
    assert not odd, f"{tier}.md has non-ASCII characters {odd!r}; use plain hyphens and ASCII"


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_has_no_machine_specific_paths(tier):
    found = re.findall(r"[A-Za-z]:\|/Users/|/home/|/c/", read_prompt(tier))
    assert not found, f"{tier}.md names machine-specific paths: {found}"


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_carries_no_attribution(tier):
    lowered = read_prompt(tier).lower()
    found = [marker for marker in ATTRIBUTION_MARKERS if marker in lowered]
    assert not found, f"{tier}.md carries attribution text: {found}"


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_states_no_count_of_sittings(tier):
    found = [match.group(0) for match in SITTING_COUNT.finditer(read_prompt(tier))]
    assert not found, f"{tier}.md states a count of sittings {found}; describe the role, not how many"


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_mentions_the_budget_on_one_line_with_the_decided_costs(tier):
    lines = [ln for ln in read_prompt(tier).splitlines() if re.search(r"(?i)\bbudget\b", ln)]
    assert len(lines) <= 1, f"{tier}.md mentions the budget on {len(lines)} lines; one at most"
    for line in lines:
        assert re.search(r"\b16 units\b", line), f"{tier}.md budget line lacks '16 units': {line!r}"
        assert re.search(r"(?i)sonnet[^.\d]*\b1\b", line), f"{tier}.md: Sonnet should cost 1"
        assert re.search(r"(?i)opus[^.\d]*\b2\b", line), f"{tier}.md: Opus should cost 2"
        assert re.search(r"(?i)fable[^.\d]*\b8\b", line), f"{tier}.md: Fable should cost 8"
        assert re.search(r"(?i)only sitting of your tier", line), f"{tier}.md: say why alone"


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_only_names_verbs_the_dispatcher_defines(tier):
    used = set(re.findall(r"python -m lab (\w+)", read_prompt(tier)))
    unknown = sorted(used - KNOWN_VERBS)
    assert not unknown, (
        f"{tier}.md names verbs the dispatcher does not define: {unknown}; "
        "update the prompt and KNOWN_VERBS together"
    )


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_states_the_rails_every_tier_keeps(tier):
    text = read_prompt(tier)
    missing = [what for what, rx in REQUIRED_EVERYWHERE.items() if not re.search(rx, text)]
    assert not missing, f"{tier}.md does not state: {missing}"


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_states_its_tier_duties(tier):
    text = read_prompt(tier)
    missing = [what for what, rx in REQUIRED_BY_TIER[tier].items() if not re.search(rx, text)]
    assert not missing, f"{tier}.md does not state: {missing}"


@pytest.mark.parametrize("tier", TIERS)
def test_prompt_points_at_agents_md_sections_that_exist(tier):
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    refs = sorted(set(re.findall(r"AGENTS\.md section (\d+(?:\.\d+)?)", read_prompt(tier))))
    assert refs, f"{tier}.md points at no AGENTS.md section"
    missing = []
    for ref in refs:
        level = "###" if "." in ref else "##"
        if not re.search(rf"^{level} {re.escape(ref)}[. ]", agents, re.M):
            missing.append(ref)
    assert not missing, f"{tier}.md points at AGENTS.md sections that do not exist: {missing}"
