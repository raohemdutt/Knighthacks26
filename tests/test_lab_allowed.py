"""Per-tier allow lists of a lab sitting: rule shape, tier differences, the argument
vector, and deny rules winning over allow rules."""

import re

import pytest

from lab.machine import allowed

TIERS = ("fable", "opus", "sonnet")
LISTS = {**{tier: allowed.ALLOWED[tier] for tier in TIERS}, "DENIED": allowed.DENIED}
NOTION = "mcp__plugin_design_notion__notion-"
KNOWN_TOOLS = {"Bash", "Read", "Write", "Edit", "Grep", "Glob", "WebFetch", "WebSearch", "Skill", "ToolSearch"}
MCP_TOOL = re.compile(r"mcp__[a-z0-9_]+__[a-z0-9-]+\*?")


def offenders(is_fine):
    """Every (list name, rule) pair for which is_fine(list name, rule) is false."""
    return [(name, rule) for name, rules in LISTS.items() for rule in rules if not is_fine(name, rule)]


def tool_of(rule):
    return rule.split("(", 1)[0]


# --- the data ---------------------------------------------------------------------------


def test_allowed_maps_exactly_the_three_tiers_to_lists_and_denied_is_a_list():
    assert sorted(allowed.ALLOWED) == ["fable", "opus", "sonnet"]
    assert all(isinstance(rules, list) and rules for rules in allowed.ALLOWED.values())
    assert isinstance(allowed.DENIED, list) and allowed.DENIED


def test_every_rule_is_a_tool_name_or_a_tool_with_a_specifier_ending_at_its_closing_parenthesis():
    assert offenders(lambda name, rule: re.fullmatch(r"[A-Za-z0-9_*.-]+(\(.+\))?", rule)) == []


def test_no_rule_has_a_space_outside_its_parentheses_a_comma_or_a_character_a_shell_would_eat():
    def clean(name, rule):
        return " " not in tool_of(rule) and not re.search(r'[,%^&|<>"`\r\n\t]', rule)

    assert offenders(clean) == []


def test_no_list_repeats_a_rule():
    repeats = [(name, rule) for name, rules in LISTS.items() for rule in set(rules) if rules.count(rule) > 1]
    assert repeats == []


def test_tool_names_are_built_in_tools_or_mcp_tools():
    assert offenders(lambda name, rule: tool_of(rule) in KNOWN_TOOLS or MCP_TOOL.fullmatch(tool_of(rule))) == []


def test_allow_rules_name_exact_tools_because_the_cli_skips_unanchored_allow_globs():
    assert [rule for tier in TIERS for rule in allowed.ALLOWED[tier] if "*" in tool_of(rule)] == []


def test_path_rules_exist_only_for_read_and_edit_because_the_cli_consults_no_other_tool_for_paths():
    ignored = {"Write", "Glob", "NotebookEdit", "MultiEdit"}
    assert offenders(lambda name, rule: not (tool_of(rule) in ignored and "(" in rule)) == []


def test_path_rules_never_start_with_a_single_slash_which_anchors_at_the_settings_source():
    def fine(name, rule):
        if tool_of(rule) not in ("Read", "Edit") or "(" not in rule:
            return True
        specifier = allowed.parse_rule(rule)[1]
        return not (specifier.startswith("/") and not specifier.startswith("//"))

    assert offenders(fine) == []


def test_no_bash_rule_ends_in_colon_star_which_the_cli_reads_as_a_trailing_wildcard():
    def fine(name, rule):
        return tool_of(rule) != "Bash" or not allowed.parse_rule(rule)[1].endswith(":*")

    assert offenders(fine) == []


def test_bash_allow_rules_have_no_wildcard_before_the_end_of_the_command():
    def fine(name, rule):
        if name == "DENIED" or tool_of(rule) != "Bash":
            return True
        specifier = allowed.parse_rule(rule)[1]
        return specifier.count("*") == 0 or (specifier.count("*") == 1 and specifier.endswith("*"))

    assert offenders(fine) == []


def test_mcp_rules_with_parameters_are_deny_rules_only_because_settings_files_skip_them():
    assert [rule for tier in TIERS for rule in allowed.ALLOWED[tier] if rule.startswith("mcp__") and "(" in rule] == []
    assert NOTION + "update-page(allow_deleting_content:true)" in allowed.DENIED


def test_the_model_can_evaluate_every_rule_in_every_list():
    def evaluable(name, rule):
        tool = tool_of(rule).replace("*", "x")
        allowed.rule_matches(rule, tool, "x", {"x": "x"})
        return True

    assert offenders(evaluable) == []


# --- what the tiers get -----------------------------------------------------------------


def test_web_search_is_for_opus_and_fable_only():
    assert [("WebSearch" in allowed.ALLOWED[tier]) for tier in ("fable", "opus", "sonnet")] == [True, True, False]
    assert [allowed.allows(tier, "WebSearch") for tier in ("fable", "opus", "sonnet")] == [True, True, False]


def test_every_tier_may_read_write_edit_search_and_fetch_but_not_run_subagents_or_powershell():
    for tier in TIERS:
        for tool in ("Read", "Write", "Edit", "Grep", "Glob", "WebFetch", "ToolSearch"):
            assert allowed.allows(tier, tool), (tier, tool)
        for tool in ("Agent", "PowerShell", "NotebookEdit", "AskUserQuestion"):
            assert not allowed.allows(tier, tool), (tier, tool)


def test_every_tier_gets_the_notion_tools_the_ticket_workflow_uses_by_exact_name():
    needed = [NOTION + name for name in ("search", "fetch", "query-data-sources", "update-page", "create-pages")]
    for tier in TIERS:
        assert set(needed) <= set(allowed.ALLOWED[tier]), tier
    assert NOTION + "move-pages" in allowed.ALLOWED["fable"]
    assert NOTION + "move-pages" not in allowed.ALLOWED["opus"] + allowed.ALLOWED["sonnet"]


def test_each_tier_may_run_the_skills_of_its_role():
    def skills(tier):
        return [rule for rule in allowed.ALLOWED[tier] if rule.startswith("Skill(")]

    ticket = ["Skill(open-ticket)", "Skill(open-ticket *)"]
    track = ["Skill(refine-track)", "Skill(refine-track *)"]
    merge = ["Skill(refine-merge)", "Skill(refine-merge *)"]
    assert skills("sonnet") == ticket
    assert skills("opus") == ticket + track
    assert skills("fable") == ticket + track + merge


def test_the_ticket_workflow_commands_are_allowed_for_every_tier():
    commands = [
        "git clone --quiet https://github.com/raohemdutt/Knighthacks26 work/T1",
        "git clone https://github.com/raohemdutt/Knighthacks26 work/T1",
        "git config core.hooksPath ../.githooks",
        "git fetch origin --quiet",
        "git checkout -b ctx/orchestration/feature origin/main",
        "git add lab/machine/allowed.py tests/test_lab_allowed.py",
        "git commit -F -",
        "git pull --rebase origin main",
        "git rebase origin/main",
        "git push -u origin ctx/orchestration/feature",
        "git push origin ctx/orchestration/feature",
        "python -m venv .venv",
        '.venv/Scripts/python -m pip install -e ".[dev]"',
        "python -m pytest -q",
        ".venv/Scripts/python -m pytest -q tests/test_lab_allowed.py",
        "pytest -q",
        'gh pr create --repo raohemdutt/Knighthacks26 --base main --head ctx/orchestration/feature --title "T1: x" --body-file pr.md',
        "gh pr view 38 --json state,url",
        "gh pr list --repo raohemdutt/Knighthacks26 --state all --search T1",
        "mkdir -p work",
        "git add -A && git commit -m wip && git push -u origin ctx/orchestration/feature",
    ]
    for tier in TIERS:
        assert [c for c in commands if not allowed.allows(tier, "Bash", c)] == [], tier


def test_lab_verbs_are_allowed_by_tier_in_both_interpreter_spellings():
    for python in ("python", ".venv/Scripts/python"):

        def run(tier, verb):
            return allowed.allows(tier, "Bash", f"{python} -m lab {verb}")

        for tier in TIERS:
            for verb in ("ready", "digest", "ledger add --verb x", "slot request --gpu B", "slot done", "handoff --note n",
                         "draft --name v", "experiment run e1", "metrics report", "rulings list"):
                assert run(tier, verb), (tier, verb)
        for tier in TIERS:
            assert run(tier, "rulings"), tier
            assert run(tier, "rulings --track kernels"), tier
            assert not run(tier, "rulings-apply"), tier
        for verb in ("rulings apply", "experiment add", "experiment result e1", "experiment rank"):
            assert [run(tier, verb) for tier in ("fable", "opus", "sonnet")] == [True, True, False], verb
        for verb in ("referee", "review", "mirror", "rulings mirror"):
            assert [run(tier, verb) for tier in ("fable", "opus", "sonnet")] == [True, False, False], verb


def test_the_owners_verbs_are_in_no_allow_list_and_are_denied():
    for verb in allowed.OWNER_VERBS:
        for python in allowed.PYTHONS:
            command = f"{python} -m lab {verb} --tier opus"
            assert allowed.denies("Bash", command), command
            for tier in TIERS:
                assert not any(allowed.rule_matches(rule, "Bash", command) for rule in allowed.ALLOWED[tier]), (tier, command)
                assert not allowed.allows(tier, "Bash", command)


# --- the argument vector ----------------------------------------------------------------


def options_of(argv):
    """Group an argument vector the way the CLI reads it: --name and the values up to the next --name."""
    grouped, current = {}, None
    for token in argv:
        if token.startswith("--"):
            current = token
            grouped[current] = []
        else:
            grouped[current].append(token)
    return grouped


@pytest.mark.parametrize("tier", TIERS)
def test_cli_flags_carry_the_allow_rules_the_deny_rules_and_dont_ask_mode(tier):
    flags = allowed.cli_flags(tier)
    options = options_of(flags)
    assert list(options) == ["--tools", "--allowedTools", "--disallowedTools", "--permission-mode"]
    assert options["--allowedTools"] == allowed.ALLOWED[tier]
    assert options["--disallowedTools"] == allowed.DENIED
    assert options["--permission-mode"] == ["dontAsk"]
    assert flags[-2:] == ["--permission-mode", "dontAsk"]


def test_tools_flag_names_the_built_in_tools_of_each_tier_and_no_mcp_tool():
    without_search = "Bash,Edit,Glob,Grep,Read,Skill,ToolSearch,WebFetch,Write"
    with_search = "Bash,Edit,Glob,Grep,Read,Skill,ToolSearch,WebFetch,WebSearch,Write"
    assert allowed.cli_flags("sonnet")[:2] == ["--tools", without_search]
    assert allowed.cli_flags("opus")[:2] == ["--tools", with_search]
    assert allowed.cli_flags("fable")[:2] == ["--tools", with_search]


def test_the_argument_vector_stays_well_under_the_command_line_limit_of_cmd_exe():
    # a launcher that goes through an npm .cmd shim is held to 8191 characters by cmd.exe,
    # and still has to add its own flags and the program path
    for tier in TIERS:
        assert sum(len(token) + 3 for token in allowed.cli_flags(tier)) <= 6500, tier


def test_tier_names_are_case_insensitive_and_unknown_tiers_are_refused():
    assert allowed.cli_flags("Sonnet") == allowed.cli_flags("sonnet")
    assert allowed.cli_flags(" OPUS ") == allowed.cli_flags("opus")
    for bad in ("haiku", "", "fable opus"):
        with pytest.raises(ValueError, match="unknown tier"):
            allowed.cli_flags(bad)


def test_cli_flags_returns_a_fresh_list_so_a_caller_cannot_change_the_data():
    before_allowed, before_denied = list(allowed.ALLOWED["opus"]), list(allowed.DENIED)
    flags = allowed.cli_flags("opus")
    flags.append("--dangerously-skip-permissions")
    flags[3] = "Bash"
    assert allowed.ALLOWED["opus"] == before_allowed
    assert allowed.DENIED == before_denied
    assert "--dangerously-skip-permissions" not in allowed.cli_flags("opus")


# --- deny rules win ---------------------------------------------------------------------

REFUSED = [
    # nothing is pushed to main, however the push is spelled
    "git push origin main",
    "git push origin HEAD:main",
    "git push origin HEAD:refs/heads/main",
    "git push origin +main",
    "git push origin main:main",
    "git push origin master",
    "git push -u origin ctx/orchestration/feature main",
    "git push origin main --dry-run",
    # no force, no deletes, no mirror, no bypassing the commit hook
    "git push --force origin ctx/orchestration/feature",
    "git push -f origin ctx/orchestration/feature",
    "git push origin ctx/orchestration/feature --force-with-lease",
    "git push origin --delete ctx/orchestration/feature",
    "git push origin :ctx/orchestration/feature",
    "git push --mirror origin",
    "git commit --no-verify -m wip",
    "git commit -n -m wip",
    "git push --no-verify origin ctx/orchestration/feature",
    # no merging pull requests
    "gh pr merge 12 --squash --delete-branch",
    "gh pr merge",
    "gh api -X PUT repos/raohemdutt/Knighthacks26/pulls/12/merge",
    # no recursive force delete, in any common spelling
    "rm -rf work/T1",
    "rm -fr work/T1",
    "rm -rfv work/T1",
    "rm -Rf work/T1",
    "rm -r -f work/T1",
    "rm --recursive --force work/T1",
    "rm work/T1 -rf",
    # pods are launched and stopped only through the queue verbs
    "curl -X POST https://dub.aupcloud.io/hub/api/users/amducf01/server -d '{}'",
    "curl -X DELETE https://dub.aupcloud.io/hub/api/users/amducf01/server",
    "python -c \"import httpx; httpx.delete('https://dub.aupcloud.io/hub/api/users/a/server')\"",
    "python lab/controller.py launch --account B",
    ".venv/Scripts/python lab/controller.py stop --account B",
    "python -m lab.controller launch",
    # a sitting never starts, stops or resumes the machine
    "python -m lab sit --tier opus --task T1",
    "python -m lab pool status",
    "python -m lab stop",
    ".venv/Scripts/python -m lab resume",
    # a refused part refuses the whole compound command, and so does a leading assignment
    "python -m pytest -q && git push origin main",
    "git add -A; git push origin HEAD:main",
    "LAB_TIER=fable git push origin main",
]


@pytest.mark.parametrize("tier", TIERS)
def test_dangerous_commands_are_refused_for_every_tier(tier):
    assert [command for command in REFUSED if allowed.allows(tier, "Bash", command)] == []


def test_dangerous_commands_are_matched_by_a_deny_rule_and_not_merely_missing_from_the_allow_list():
    assert [command for command in REFUSED if not allowed.denies("Bash", command)] == []


def test_a_deny_rule_beats_an_allow_rule_that_matches_the_same_command():
    for command in ("git push -u origin ctx/orchestration/feature main", "git commit --no-verify -m wip",
                    "git push origin ctx/orchestration/feature --force-with-lease"):
        for tier in TIERS:
            assert any(allowed.rule_matches(rule, "Bash", command) for rule in allowed.ALLOWED[tier]), (tier, command)
            assert allowed.denies("Bash", command)
            assert not allowed.allows(tier, "Bash", command)


def test_commands_that_only_mention_a_denied_word_are_not_denied():
    harmless = [
        "git commit -m 'explain why main stays protected'",
        "git push -u origin ctx/learning/domain-model",
        "git push origin ctx/orchestration/main-wiring",
        "python -m pytest -q tests/test_lab_allowed.py",
        "git checkout -b ctx/orchestration/feature origin/main",
    ]
    assert [command for command in harmless if allowed.denies("Bash", command)] == []


def test_the_owner_notes_and_secret_files_are_never_readable_or_writable_by_a_sitting():
    paths = [".env", "work/T1/.env", "/c/work/T1/.env", "lab/state/notes/2026-10-10.md", "work/T1/lab/state/notes/2026-10-10.md"]
    for tier in TIERS:
        for tool in ("Read", "Grep", "Glob", "Edit", "Write"):
            assert [p for p in paths if allowed.allows(tier, tool, p)] == [], (tier, tool)
    assert all(allowed.denies("Read", p) for p in paths)


def test_the_example_env_file_stays_readable_and_state_is_readable_but_changed_only_through_verbs():
    for tier in TIERS:
        assert allowed.allows(tier, "Read", "config/.env.example")
        assert allowed.allows(tier, "Read", "lab/state/alerts/2026-10-10.md")
        assert allowed.allows(tier, "Edit", "lab/refine/telemetry/RUBRIC.md")
        for tool in ("Edit", "Write"):
            for path in ("lab/state/STOP", "lab/state/ledger.jsonl", "work/T1/lab/state/review/2026-10-10.md"):
                assert not allowed.allows(tier, tool, path), (tier, tool, path)


def test_notion_page_deletion_is_refused_and_ordinary_updates_are_not():
    update = NOTION + "update-page"
    deleting = {"command": "replace_content", "allow_deleting_content": True}
    for tier in TIERS:
        assert allowed.allows(tier, update, params={"command": "insert_content"})
        assert allowed.allows(tier, update, params={"command": "replace_content", "allow_deleting_content": False})
        assert allowed.allows(tier, NOTION + "fetch", params={"id": "x"})
        assert any(allowed.rule_matches(rule, update, None, deleting) for rule in allowed.ALLOWED[tier])
        assert allowed.denies(update, params=deleting)
        assert not allowed.allows(tier, update, params=deleting)
        for tool in ("delete-pages", "trash-page"):
            assert allowed.denies(NOTION + tool)
            assert not allowed.allows(tier, NOTION + tool)


def test_no_other_mcp_server_and_no_unlisted_notion_tool_is_allowed():
    for tier in TIERS:
        assert not allowed.allows(tier, "mcp__other__send_message")
        assert not allowed.allows(tier, NOTION + "duplicate-page")
    assert [allowed.allows(tier, NOTION + "move-pages") for tier in ("fable", "opus", "sonnet")] == [True, False, False]


# --- the model of the CLI's rule matching, against the CLI's own documented examples ----


@pytest.mark.parametrize(
    "rule, command, matches",
    [
        ("Bash(npm run build)", "npm run build", True),
        ("Bash(npm run build)", "npm run build --watch", False),
        ("Bash(npm run *)", "npm run build", True),
        ("Bash(npm run *)", "npm run test --watch", True),
        ("Bash(npm run *)", "npm run", True),
        ("Bash(npm run *)", "npm install", False),
        ("Bash(git log * main)", "git log --oneline main", True),
        ("Bash(git log * main)", "git log -5 main", True),
        ("Bash(git log * main)", "git log main", False),
        ("Bash(git log * main)", "git push origin main", False),
        ("Bash(git * main)", "git merge main", True),
        ("Bash(git * main)", "git push origin main", True),
        ("Bash(git * main)", "git -c core.fsmonitor=x diff main", True),
        ("Bash(git * main)", "git log", False),
        ("Bash(* --version)", "node --version", True),
        ("Bash(* --version)", "node -v", False),
        ("Bash(ls *)", "ls -la", True),
        ("Bash(ls *)", "ls", True),
        ("Bash(ls *)", "lsof", False),
        ("Bash(ls*)", "lsof", True),
        ("Bash(* --help *)", "npm --help x", True),
        ("Bash(* --help *)", "npm --help", False),
        ("Bash(ls:*)", "ls -la", True),
        ("Bash(ls:*)", "lsof", False),
    ],
)
def test_bash_rules_match_the_way_the_cli_documents(rule, command, matches):
    assert allowed.rule_matches(rule, "Bash", command) is matches


def test_compound_commands_are_split_at_the_separators_the_cli_recognises():
    assert allowed.split_commands("a && b || c; d | e |& f & g\nh") == list("abcdefgh")
    assert allowed.split_commands("python -m pytest -q 2>&1") == ["python -m pytest -q"]
    assert allowed.split_commands("  ") == []


def test_a_command_with_a_substitution_or_an_empty_command_is_not_reported_as_allowed():
    for tier in TIERS:
        assert not allowed.allows(tier, "Bash", "python -m pytest -q $(git push origin main)")
        assert not allowed.allows(tier, "Bash", "")
        assert not allowed.allows(tier, "Bash", None)


def test_a_rule_the_model_does_not_cover_is_reported_instead_of_guessed():
    with pytest.raises(ValueError, match="not modelled"):
        allowed.rule_matches("Read(src/**)", "Read", "src/a.py")
    with pytest.raises(ValueError, match="not a permission rule"):
        allowed.parse_rule("Bash(ls) x")


# --- the module documents the verbs its rules name --------------------------------------


def test_the_dispatcher_help_is_allowed_to_every_tier_and_nothing_else_is_reachable_through_it():
    for tier in TIERS:
        for python in allowed.PYTHONS:
            for form in ("", " --help", " -h", " help"):
                assert allowed.allows(tier, "Bash", f"{python} -m lab{form}"), (tier, python, form)
            assert not allowed.allows(tier, "Bash", f"{python} -m lab unknown-verb")
            assert not allowed.allows(tier, "Bash", f"{python} -m lab --help; {python} -m lab sit")


def test_the_docstring_names_every_lab_verb_the_rules_name_and_says_which_do_not_exist_yet():
    verbs = {m for rules in LISTS.values() for rule in rules for m in re.findall(r"-m lab (\w+)", rule)} - {"help"}
    assert {"ready", "ledger", "slot", "sit", "pool", "stop", "resume", "referee"} <= verbs
    assert "do not exist yet" in allowed.__doc__
    assert [verb for verb in sorted(verbs) if not re.search(rf"\b{verb}\b", allowed.__doc__)] == []
