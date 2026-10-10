"""What a sitting of the lab machine may call, as data.

A sitting is one headless agent session started by the lab launcher. It runs under
--permission-mode dontAsk, so a tool call that no rule allows is refused without a
prompt: what a sitting can do is what this file lists, never something its prompt
says. cli_flags(tier) turns the lists into the arguments the launcher adds to its
agent command line; nothing else grants a tool. The launcher must not add another
permission mode, and must put the prompt on stdin or after cli_flags(): --allowedTools
and --disallowedTools take every argument up to the next option, so cli_flags()
ends with --permission-mode, which takes exactly one.

The data. ALLOWED maps each tier ("fable", "opus", "sonnet") to its --allowedTools
rules. DENIED is the one deny list every tier shares (--disallowedTools); deny rules
are checked first, and no allow rule can reopen what a deny rule matches. The tiers
differ in three ways: WebSearch is for opus and fable only; opus and fable also run
refinement passes (their verbs and skill); fable alone runs the merge pass and the
referee duties (their verbs and skill, and moving Notion pages).

Rule syntax, as the CLI's permissions documentation gives it (read for CLI 2.1.296):
  - A rule is Tool or Tool(specifier) and ends at the closing parenthesis.
  - Bash: * matches any text, and a trailing " *" also matches the bare command. In
    an allow rule the * goes after the subcommand, because one placed before it also
    approves options inserted there (git -c ...). A compound command (&&, ||, ;, |,
    &, newline) needs every part allowed, and a deny rule matches if any part does.
  - Path rules count only for Read and Edit (Edit also covers Write); Read, Grep and
    Glob share Read rules, and a Read deny rule also blocks Edit and Write on that
    path. "//" anchors at the filesystem root, so //**/x matches x in any directory.
  - An MCP rule with a parameter, tool(param:value), is valid only as a
    --disallowedTools value, which is how DENIED reaches the CLI; a settings file
    skips it. Do not paste DENIED into a settings file.

Left out on purpose. The owner's verbs (OWNER_VERBS): a sitting never starts, stops or
resumes the machine. The Agent tool (the pool is the parallelism, and subagents add
cost inside a sitting), PowerShell, every MCP server but Notion, and rm of any kind:
the launcher, not a sitting, deletes a finished clone. DENIED adds a second line for
the worst cases: no pod launch or stop except through the queue verbs, no push to
main or force push, no merge of a pull request, no recursive force delete, no Notion
page deletion, no bypassing the commit hook, no rebase that runs a command (--exec,
--exec=..., a shorter prefix such as --ex, or -x first or after other arguments: git
runs that command itself, so no deny rule ever sees it), nothing from the owner's notes,
the machine's state files or .env, and no allowed shell command but the interpreter
(cp, mv, touch, an output redirect) whose text names the state dir.

Verbs that do not exist yet. When this file was written lab/__main__.py registered
only ready, resume and stop. The other verbs named here do not exist yet; their
patterns are named in advance and match nothing that runs until the verb is added (a
verb with sub-verbs registers once, under its first word). VERBS in lab/__main__.py
is the authority on what exists, so nothing here changes when a verb lands.
  allowed to sittings: ready, digest, handoff, draft, ledger, metrics, slot, rulings,
    experiment, referee, review, mirror, placement
  the owner's, denied to every sitting: pool, resume, sit, stop
A pod controller script (lab/controller.py) is denied by the same logic for its
launch and stop commands; its command line is not final, so adjust _PODS when it lands.

Limits. Deny rules match the command text as written, so they are a second line and
not a boundary; the first line is that dontAsk refuses what no rule allows (apart from
what the CLI never asks about: reads inside the working directory and its built-in
read-only commands). Edit and Write are allowed at the tool level, so the launcher
chooses the directory a sitting works in. A sitting cannot read .env and has no need
to: the verbs find the keys they need (ready searches the clone and the two folders
above it), so copying .env into a clone may be refused. STATE_DIR is where the
machine keeps its state (lab/state, the default of $LAB_STATE_DIR); if the launcher
points LAB_STATE_DIR at a path that does not end in lab/state, change STATE_DIR to
match. A sitting gets LAB_ACTOR and LAB_TIER from its environment, because an inline
VAR=value prefix would not match the allow rules. The interpreter spellings are
exactly those in PYTHONS; any other is refused. The argument vector is under 6k
characters for every tier; a launcher that reaches the CLI through an npm .cmd shim is
held to 8191 by cmd.exe, so a test keeps it under 6500.

allows() and denies() model the documented matching so the lists can be tested and a
launcher can rehearse a command: they are conservative (they refuse what they cannot
parse, such as a command substitution) and they do not model wrappers like timeout,
the built-in read-only commands, working directories or protected paths. The CLI is
the authority.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Mapping, Sequence

TIERS: tuple[str, ...] = ("fable", "opus", "sonnet")

#: Passed as --permission-mode: a call no rule allows is refused, never asked.
PERMISSION_MODE = "dontAsk"

#: How a sitting writes the interpreter: the global one, or the clone's own venv.
PYTHONS: tuple[str, ...] = ("python", ".venv/Scripts/python")

#: The one repository a sitting may clone.
REPO_URL = "https://github.com/raohemdutt/Knighthacks26"

#: The name the Notion MCP server is registered under; its tools are
#: mcp__<server>__<tool>. The launcher must register the server under this name.
NOTION_SERVER = "plugin_design_notion"

#: Where the machine keeps its state: the path the deny rules for state files end in.
STATE_DIR = "lab/state"

#: Branches nothing may be pushed to.
PROTECTED_BRANCHES: tuple[str, ...] = ("main", "master")

# --- verbs of `python -m lab`, by who may call them ---------------------------------------

#: Verbs the owner alone calls: they start, stop or resume the machine itself.
OWNER_VERBS: tuple[str, ...] = ("pool", "resume", "sit", "stop")

_COMMON_VERBS: tuple[str, ...] = (
    "ready", "digest", "handoff", "draft",
    "ledger add", "ledger tail", "ledger grep",
    "metrics report", "metrics compare",
    "slot request", "slot start", "slot extend", "slot done", "slot complain", "slot status",
    "placement",
    "rulings list",
    "experiment list", "experiment run",
)

#: The dispatcher's own help, which lists the verbs and changes nothing.
_HELP: tuple[str, ...] = ("", " --help", " -h", " help")

#: Verbs whose bare form, or a form that starts with a flag, is itself a read: rulings
#: with no sub-verb lists the unanswered questions. Sub-verbs that change things (apply,
#: mirror) are separate rules below, so a flag form never reaches them.
_READ_FORMS: tuple[str, ...] = ("rulings", "rulings -*")

#: What opus and fable add: the verbs a refinement pass and experiment planning use.
_PASS_VERBS: tuple[str, ...] = ("rulings apply", "experiment add", "experiment result", "experiment rank")

#: What fable alone adds: the referee duties and the machine's account of itself.
_REFEREE_VERBS: tuple[str, ...] = ("referee", "review", "mirror", "rulings mirror")

# --- what every tier shares ---------------------------------------------------------------

_FILE_TOOLS: tuple[str, ...] = ("Read", "Write", "Edit", "Grep", "Glob")

#: The Notion tools the ticket workflow uses, by their exact names.
_NOTION_TOOLS: tuple[str, ...] = ("search", "fetch", "query-data-sources", "update-page", "create-pages")

#: Shell commands of a sitting's whole life in its clone: clone, branch, commit, push of
#: that branch, pull requests (never merged), tests, a venv. Read-only commands (ls, cat,
#: grep, git status, git diff, git log, ...) need no rule. Every * is the last character.
_SHELL: tuple[str, ...] = (
    f"git clone --quiet {REPO_URL} *",
    f"git clone {REPO_URL} *",
    "git config core.hooksPath *",
    "git fetch *",
    "git checkout *",
    "git switch *",
    "git add *",
    "git rm *",
    "git mv *",
    "git commit *",
    "git pull --rebase *",
    "git rebase *",
    "git push -u origin ctx/*",
    "git push origin ctx/*",
    "gh pr create *",
    "gh pr edit *",
    "gh pr view *",
    "gh pr list *",
    "gh pr diff *",
    "gh pr checks *",
    "gh run list *",
    "gh run view *",
    "pytest *",
    *(f"{python} -m pytest *" for python in PYTHONS),
    "python -m venv .venv",
    ".venv/Scripts/python -m pip install -e *",
    "mkdir *",
    "cp *",
    "mv *",
    "touch *",
    "date *",
)


def _allowed(*, web_search: bool, skills: Sequence[str], notion: Sequence[str], verbs: Sequence[str]) -> list[str]:
    """The --allowedTools rules of one tier."""
    rules = [*_FILE_TOOLS, "WebFetch"]
    if web_search:
        rules.append("WebSearch")
    rules.append("ToolSearch")  # loads the deferred Notion tools; it asks for no permission
    rules += [rule for name in skills for rule in (f"Skill({name})", f"Skill({name} *)")]
    rules += [f"mcp__{NOTION_SERVER}__notion-{name}" for name in notion]
    rules += [f"Bash({command})" for command in _SHELL]
    rules += [f"Bash({python} -m lab{help_form})" for help_form in _HELP for python in PYTHONS]
    rules += [f"Bash({python} -m lab {form})" for form in _READ_FORMS for python in PYTHONS]
    rules += [f"Bash({python} -m lab {verb} *)" for verb in verbs for python in PYTHONS]
    return rules


ALLOWED: dict[str, list[str]] = {
    "fable": _allowed(
        web_search=True,
        skills=("open-ticket", "refine-track", "refine-merge"),
        notion=(*_NOTION_TOOLS, "move-pages"),
        verbs=(*_COMMON_VERBS, *_PASS_VERBS, *_REFEREE_VERBS),
    ),
    "opus": _allowed(
        web_search=True,
        skills=("open-ticket", "refine-track"),
        notion=_NOTION_TOOLS,
        verbs=(*_COMMON_VERBS, *_PASS_VERBS),
    ),
    "sonnet": _allowed(
        web_search=False,
        skills=("open-ticket",),
        notion=_NOTION_TOOLS,
        verbs=_COMMON_VERBS,
    ),
}

# --- what no tier may do ------------------------------------------------------------------

#: A pod is launched and stopped through the hub API or the controller script; a sitting
#: reaches a pod only through the queue verbs. The first pattern catches any command that
#: names the hub's server endpoint, whatever client sends the request.
_PODS: list[str] = [
    "Bash(*hub/api/users/*/server*)",
    "Bash(*controller.py launch*)",
    "Bash(*controller.py stop*)",
    "Bash(*lab.controller launch*)",
    "Bash(*lab.controller stop*)",
]

#: Nothing is pushed to a protected branch, however the push is spelled. The patterns
#: need the branch as a word, after a colon, after a plus, or as refs/heads/<branch>, so a
#: branch like ctx/learning/domain-model is not caught.
_PUSH_TO_MAIN: list[str] = [
    rule
    for branch in PROTECTED_BRANCHES
    for rule in (
        f"Bash(git push * {branch}*)",
        f"Bash(git push *:{branch}*)",
        f"Bash(git push *+{branch}*)",
        f"Bash(git push *refs/heads/{branch}*)",
    )
]

#: No force push, no remote delete, no mirror of every ref, and the commit hook (which
#: strips attribution) is never bypassed. A pattern must not end in ":*": the CLI reads
#: that as its older spelling of a trailing " *", so a delete refspec is caught as :ctx/*.
_PUSH_AND_COMMIT_FLAGS: list[str] = [
    "Bash(git push *--force*)",
    "Bash(git push -f*)",
    "Bash(git push * -f*)",
    "Bash(git push *--delete*)",
    "Bash(git push -d*)",
    "Bash(git push * -d*)",
    "Bash(git push * :ctx/*)",
    "Bash(git push *--mirror*)",
    "Bash(git push *--all*)",
    "Bash(git push *--prune*)",
    "Bash(git push *--no-verify*)",
    "Bash(git commit *--no-verify*)",
    "Bash(git commit -n*)",
]

#: git rebase runs the argument of --exec (or -x) as a shell command after each commit it
#: replays. A deny rule matches the command part as written, so a push, an rm or a write to
#: the state dir inside that argument is never seen by the rules above: 'git rebase *' stays
#: allowed for ordinary rebases, and every spelling that carries a command is refused.
#: --ex* also catches --exec=<cmd> and the prefixes git accepts for --exec (--exe, --ex),
#: which no other rebase option starts with; -x is caught as the first argument (also
#: stuck to its command, -x<cmd>) and after other arguments. A short-flag group that ends
#: in x (-mx) is not caught: deny rules are a second line, not a boundary.
_REBASE_EXEC: list[str] = [
    "Bash(git rebase *--ex*)",
    "Bash(git rebase -x*)",
    "Bash(git rebase * -x*)",
]

#: A pull request is merged by a person, never by a sitting.
_MERGE: list[str] = ["Bash(gh pr merge *)", "Bash(gh api *merge*)"]

_RM_FORCE_RECURSIVE: tuple[str, ...] = (
    "-rf", "-fr", "-Rf", "-fR", "-r -f", "-f -r", "-R -f", "-f -R",
    "--recursive --force", "--force --recursive", "-r --force", "--recursive -f", "-f --recursive", "--force -r",
)

#: No recursive force delete, in the common spellings.
_RM: list[str] = [*(f"Bash(rm {flags}*)" for flags in _RM_FORCE_RECURSIVE), "Bash(rm * -rf*)", "Bash(rm * -fr*)"]

#: Notion pages are archived by moving them, never deleted: replace_content with
#: allow_deleting_content set deletes child pages, and a delete or trash tool, should
#: the server ever offer one, is refused by name.
_NOTION_DELETE: list[str] = [
    f"mcp__{NOTION_SERVER}__notion-update-page(allow_deleting_content:true)",
    f"mcp__{NOTION_SERVER}__notion-delete*",
    f"mcp__{NOTION_SERVER}__notion-trash*",
]

#: A sitting never starts, stops or resumes the machine.
_OWNER: list[str] = [f"Bash({python} -m lab {verb} *)" for verb in OWNER_VERBS for python in PYTHONS]

#: .env is never read; the owner's notes are never read or written; the machine's state
#: is read freely but changed only through the verbs (STOP, the ledger, alerts, reviews).
_PATHS: list[str] = [
    "Read(//**/.env)",
    f"Read(//**/{STATE_DIR}/notes/**)",
    f"Edit(//**/{STATE_DIR}/**)",
]

#: The state dir as a command may spell it: with forward slashes, and with the Windows
#: backslashes a quoted path keeps.
_STATE_SPELLINGS: tuple[str, ...] = (STATE_DIR, STATE_DIR.replace("/", "\\"))

#: The first words of the shell commands a sitting may run, apart from the interpreter.
#: The interpreter is left out because it is how the verbs run, and a verb may name the
#: state dir in its arguments (a handoff note that says it did not read the notes); it
#: also runs code a sitting writes, so no rule on its command text bounds what it touches.
_WRITING_COMMANDS: tuple[str, ...] = tuple(
    sorted({command.split(" ", 1)[0] for command in _SHELL} - set(PYTHONS))
)

#: The Edit rule above does not reach Bash: cp, mv, touch and mkdir take the state dir as
#: an argument, and an output redirect (date > lab/state/STOP) writes there through any
#: allowed command. So each of those commands is refused when its text names the state
#: dir. A rule cannot hold the redirect character itself (the cmd.exe shim would eat it),
#: hence one rule per command, built from _SHELL so a command added there is covered.
#: Read-only commands (ls, cat, grep, test) need no allow rule and stay free to read the
#: state dir, and the verbs stay free to change it. A commit message or pull request text
#: that names the state dir is refused too: reword it.
_STATE_WRITES: list[str] = [
    f"Bash({command} *{spelling}*)" for spelling in _STATE_SPELLINGS for command in _WRITING_COMMANDS
]

DENIED: list[str] = [
    *_PODS, *_PUSH_TO_MAIN, *_PUSH_AND_COMMIT_FLAGS, *_REBASE_EXEC, *_MERGE, *_RM, *_NOTION_DELETE,
    *_OWNER, *_PATHS, *_STATE_WRITES,
]

# --- the argument vector ------------------------------------------------------------------


def _tier(tier: str) -> str:
    key = str(tier).strip().lower()
    if key not in ALLOWED:
        raise ValueError(f"unknown tier {tier!r}; expected one of {', '.join(TIERS)}")
    return key


def builtin_tools(tier: str) -> list[str]:
    """The built-in tools that exist in a sitting of this tier (--tools): the ones with an
    allow rule. MCP tools are not built in; --tools does not touch them."""
    names = {parse_rule(rule)[0] for rule in ALLOWED[_tier(tier)]}
    return sorted(name for name in names if not name.startswith("mcp__"))


def cli_flags(tier: str) -> list[str]:
    """The permission arguments of the agent command line for a sitting of this tier:
    --tools, --allowedTools, --disallowedTools and --permission-mode dontAsk, in that
    order. A fresh list each call; the tier name is case-insensitive."""
    key = _tier(tier)
    return [
        "--tools", ",".join(builtin_tools(key)),
        "--allowedTools", *ALLOWED[key],
        "--disallowedTools", *DENIED,
        "--permission-mode", PERMISSION_MODE,
    ]


# --- a model of the CLI's documented rule matching ----------------------------------------

_RULE = re.compile(r"([A-Za-z0-9_*.\-]+)(?:\((.+)\))?", re.DOTALL)
_SEPARATORS = re.compile(r"&&|\|\||\|&|[;|&\n]")
_FD_REDIRECT = re.compile(r"\d*[<>]&\d*-?")
_ASSIGNMENT = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*=\S*\s+)+")
_SUBSTITUTION = re.compile(r"\$\(|`|<\(|>\(")


def parse_rule(rule: str) -> tuple[str, str | None]:
    """Split a rule into its tool name and its specifier (None for a bare tool name)."""
    match = _RULE.fullmatch(rule)
    if match is None:
        raise ValueError(f"not a permission rule: {rule!r}")
    return match.group(1), match.group(2)


def split_commands(command: str) -> list[str]:
    """The parts of a compound command, split at the separators the CLI recognises
    (&&, ||, ;, |, |&, &, newline). Not quote-aware: a separator inside a quoted
    argument splits too, which can only make a command look more refused."""
    return [part.strip() for part in _SEPARATORS.split(_FD_REDIRECT.sub("", command)) if part.strip()]


def _wildcard_regex(specifier: str) -> re.Pattern[str]:
    """A Bash or Skill specifier as a regular expression: * matches any text, a trailing
    " *" (or ":*") as the only wildcard also matches the bare command."""
    if specifier.endswith(":*"):
        specifier = specifier[:-2] + " *"
    if specifier.endswith(" *") and specifier.count("*") == 1:
        return re.compile(re.escape(specifier[:-2]) + r"(?: .*)?", re.DOTALL)
    return re.compile(".*".join(re.escape(part) for part in specifier.split("*")), re.DOTALL)


def _path_matches(specifier: str, path: str) -> bool:
    """Match a path against the only path shapes the lists use: //**/<name> (a file or
    directory of that name anywhere) and //**/<dir>/** (anything under that directory)."""
    if not specifier.startswith("//**/"):
        raise ValueError(f"path rule {specifier!r} is not modelled")
    tail = specifier[len("//**/"):]
    inside = tail.endswith("/**")
    if inside:
        tail = tail[: -len("/**")]
    if "*" in tail or not tail:
        raise ValueError(f"path rule {specifier!r} is not modelled")
    wanted = tail.split("/")
    segments = [segment for segment in path.replace("\\", "/").split("/") if segment]
    for start in range(len(segments) - len(wanted) + 1):
        if segments[start:start + len(wanted)] == wanted:
            end = start + len(wanted)
            if (end < len(segments)) if inside else (end == len(segments)):
                return True
    return False


def _parameter_matches(specifier: str, params: Mapping[str, object] | None) -> bool:
    """An MCP rule tool(name:value): the call carries that parameter with that value
    (booleans compare as true or false; * in the value is a wildcard)."""
    name, _, wanted = specifier.partition(":")
    name = name.strip()
    if params is None or name not in params:
        return False
    actual = params[name]
    text = ("true" if actual else "false") if isinstance(actual, bool) else str(actual)
    return fnmatch.fnmatchcase(text, wanted.strip())


def _path_rule_reaches(rule_tool: str, tool: str, deny: bool) -> bool:
    """Which tools a path rule governs: Read rules cover Read, Grep and Glob (and, as a
    deny rule, Edit and Write too); Edit rules cover Edit and Write."""
    if rule_tool == "Read":
        return tool in ("Read", "Grep", "Glob") or (deny and tool in ("Edit", "Write"))
    return tool in ("Edit", "Write")


def rule_matches(
    rule: str,
    tool: str,
    argument: str | None = None,
    params: Mapping[str, object] | None = None,
    *,
    deny: bool = False,
) -> bool:
    """Whether one rule matches one call: the tool name, a Bash command (a single command,
    see split_commands), a path, a skill name and arguments, or MCP parameters. Raises
    ValueError for a rule shape the model does not cover, instead of guessing."""
    rule_tool, specifier = parse_rule(rule)
    applies = fnmatch.fnmatchcase(tool, rule_tool)
    if not applies and specifier is not None and rule_tool in ("Read", "Edit"):
        applies = _path_rule_reaches(rule_tool, tool, deny)
    if not applies:
        return False
    if specifier is None:
        return True
    if rule_tool.startswith("mcp__"):
        return _parameter_matches(specifier, params)
    if argument is None:
        return False
    if rule_tool in ("Bash", "Skill"):
        return _wildcard_regex(specifier).fullmatch(argument) is not None
    if rule_tool in ("Read", "Edit"):
        return _path_matches(specifier, argument)
    raise ValueError(f"rule {rule!r} is not modelled")


def _calls(tool: str, argument: str | None) -> list[str | None]:
    """The calls a deny rule is tried against: the argument itself, or for Bash each part
    of the compound command, also without a leading VAR=value assignment (a deny rule
    matches past one)."""
    if tool != "Bash" or argument is None:
        return [argument]
    calls: list[str | None] = []
    for part in split_commands(argument):
        calls.append(part)
        bare = _ASSIGNMENT.sub("", part, count=1)
        if bare != part:
            calls.append(bare)
    return calls


def denies(tool: str, argument: str | None = None, params: Mapping[str, object] | None = None) -> bool:
    """Whether any DENIED rule matches the call (for Bash, any part of the command)."""
    return any(
        rule_matches(rule, tool, call, params, deny=True) for call in _calls(tool, argument) for rule in DENIED
    )


def allows(tier: str, tool: str, argument: str | None = None, params: Mapping[str, object] | None = None) -> bool:
    """Whether the rules of this tier let the call run: no DENIED rule matches and an
    allow rule does (for Bash, one for every part of the command). Calls the CLI allows
    without any rule, such as reads in the working directory, report False here."""
    rules = ALLOWED[_tier(tier)]
    if denies(tool, argument, params):
        return False
    if tool == "Bash":
        if not argument or _SUBSTITUTION.search(argument):
            return False
        parts = split_commands(argument)
        return bool(parts) and all(any(rule_matches(rule, tool, part) for rule in rules) for part in parts)
    return any(rule_matches(rule, tool, argument, params) for rule in rules)
