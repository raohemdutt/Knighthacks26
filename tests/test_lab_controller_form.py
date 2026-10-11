"""lab.controller: launching a pod through the hub's web spawn form, and noticing a REST launch the hub ignored.

Every test drives the real Controller and the real Hub client. The only doubles are the hub (lab.hub_fake.FakeHub, which
answers the web login, the spawn form and the REST launch as the SJC hub was recorded to, and starts the stock image for
a REST launch when told the hub ignores it) and, for the question which image a ready pod runs, the websocket of its
terminal, which stands for the pod's shell. Nothing here touches the network, sleeps for real, or launches a pod."""
from __future__ import annotations

import contextlib
import json
import re
import socket
import threading
from dataclasses import replace
from unittest import mock

import pytest

from lab import controller as ctl_module
from lab.controller import Controller
from lab.hub_fake import AUP_IMAGE, STOCK_IMAGE, FakeClock, FakeHub
from lab.machine import ledger

TOKEN_A = "aaaa1111bbbb2222cccc3333dddd4444"
TOKEN_C = "cccc9999dddd0000eeee1111ffff2222"
PASSWORD = "fake-correct-horse-battery"
WRONG_PASSWORD = "fake-wrong-password-77"
SJC = "https://sjc.test"
SJC_ENV = {"LAB_POD_ROLES": "C:pod-a,D:pod-b", "AUP_HUB_USER_C": "acct-c", "AUP_HUB_TOKEN_C": TOKEN_C, "AUP_HUB_URL_C": SJC}
FORM_FIELDS = {"resource_type": "gpu", "gpu_selection_gpu": "strix-halo", "runtime": "2400", "repo_url": "",
               "repo_persist": "true", "repo_branch": ""}


@pytest.fixture(autouse=True)
def lab_state(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("LAB_ACTOR", "tester")
    for name in ("LAB_TIER", "LAB_TASK"):
        monkeypatch.delenv(name, raising=False)


class Sjc:
    """The SJC hub as a fake (quota off, a REST launch it ignores) with account C, who has a password."""

    def __init__(self, *, rest_ignored=True, password=PASSWORD, **account_options):
        self.clock = FakeClock("2026-10-10T09:27:00Z")
        self.hub = FakeHub(self.clock)
        self.hub.rest_launch_ignored = rest_ignored
        account = self.hub.add_account("C", user="acct-c", token=TOKEN_C, quota_enabled=False, password=PASSWORD)
        self.account = replace(account, password=password, **account_options)

    def controller(self, base_url=SJC):
        return Controller({"C": self.account}, base_url=base_url, transport=self.hub.transport,
                          sleep=self.clock.sleep, clock=self.clock.time)

    def cli(self, *argv, environ=None):
        return ctl_module.run(list(argv), environ={**SJC_ENV, **(environ or {})}, env_files=[],
                              transport=self.hub.transport, sleep=self.clock.sleep, clock=self.clock.time)

    def paths(self):
        """The (method, path) of the calls to the hub itself that change something or log in, in order."""
        return [(call.method, call.path) for call in self.hub.calls if call.path.startswith("/hub/")
                and (call.method != "GET" or call.path.startswith(("/hub/native", "/hub/spawn")))]


def plain(conditions):
    """A row's conditions without the hub and account every pod row also records (tested on their own)."""
    return {key: value for key, value in conditions.items() if key not in ("hub", "account")}


def rows(verb):
    return ledger.read(verb=verb)


def only_row(verb):
    found = rows(verb)
    assert len(found) == 1, found
    return found[0]


def everything_written(answer, capsys=None):
    text = json.dumps(answer) + ledger.ledger_path().read_text(encoding="utf-8")
    if capsys is not None:
        captured = capsys.readouterr()
        text += captured.out + captured.err
    return text


@contextlib.contextmanager
def fake_terminal(sjc, says=None, delay_s=0.0):
    """A Jupyter terminal on localhost, standing for the pod's shell: when the controller types the command that runs
    a script, the file the script writes appears in the fake hub's file store with what the pod would have printed
    (the image it runs and whether it has the GPU device), delay_s seconds of the fake clock later, and the command is
    echoed once it is typed."""
    from websockets.sync.server import serve

    def pod_says():
        gpu = "yes" if sjc.hub.pod_has_gpu("C") else "no"
        return says if says is not None else f"image={sjc.hub.pod_image('C')}\ngpu_device={gpu}\n"

    def handler(connection):
        connection.send(json.dumps(["setup", {}]))
        for raw in connection:
            kind, payload = json.loads(raw)
            if kind == "stdin":
                match = re.search(r"> \./(\S+\.out)", payload)
                if match:
                    out, text = match.group(1), pod_says() + "### END rc=0\n"
                    if delay_s:
                        sjc.clock.call_later(delay_s, lambda: sjc.hub.files.__setitem__(("C", out), text))
                    else:
                        sjc.hub.files[("C", out)] = text
                connection.send(json.dumps(["stdout", payload]))

    with serve(handler, "127.0.0.1", 0) as server, mock.patch.object(ctl_module, "TERMINAL_SETTLE_S", 60.0):
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.socket.getsockname()[1]}"
        finally:
            server.shutdown()
            thread.join(timeout=5)


@contextlib.contextmanager
def closed_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    yield f"http://127.0.0.1:{port}"


# --- AUP_HUB_LAUNCH_<X> = form ------------------------------------------------------------------

def test_a_form_launch_logs_in_and_posts_the_form_where_the_rest_call_is_not_sent_at_all():
    sjc = Sjc(launch="form")

    answer = sjc.controller().launch("C", 2400, accelerator="strix-halo")

    assert (answer["ok"], answer["outcome"], answer["launch_path"], answer["accelerator"]) == (
        True, "running", "form", "strix-halo")
    assert answer["note"] == "quota disabled on this hub"
    assert sjc.paths() == [("GET", "/hub/native/login"), ("POST", "/hub/native/login"), ("GET", "/hub/spawn/acct-c"),
                           ("POST", "/hub/spawn/acct-c")]
    assert sjc.hub.calls_matching("POST", "/server") == []
    assert sjc.hub.calls_matching("POST", "/hub/spawn/")[0].form == FORM_FIELDS
    assert sjc.hub.pod_image("C") == AUP_IMAGE
    assert answer["start_s"] == 15.0
    assert (answer["server"]["accelerator"], answer["server"]["runtime_minutes"]) == ("strix-halo", 2400)


def test_a_form_launch_leaves_a_ledger_row_that_says_the_path_and_the_accelerator():
    sjc = Sjc(launch="form")

    sjc.controller().launch("C", 2400, accelerator="strix-halo", reason="rehearsal")

    pod = only_row("pod launch")
    assert (pod["node"], pod["outcome"], pod["note"]) == ("pod-a", "running", "rehearsal")
    assert plain(pod["conditions"]) == {"resource": "gpu", "accelerator": "strix-halo", "launch_path": "form"}
    assert pod["metrics"] == {"pod.start_s": 15.0}
    assert only_row("controller launch")["outcome"] == "running"


def test_a_form_launch_is_a_no_op_on_a_pod_that_is_already_up_and_sends_nothing_to_the_web_side():
    sjc = Sjc(launch="form")
    sjc.hub.start_server("C")

    answer = sjc.controller().launch("C", 2400)

    assert answer["outcome"] == "already_running" and "launch_path" not in answer
    assert sjc.paths() == []


def test_a_form_launch_without_a_password_is_refused_before_anything_is_sent_and_exits_2():
    sjc = Sjc(password=None)

    answer, code = sjc.cli("launch", "--account", "C", "--minutes", "30", environ={"AUP_HUB_LAUNCH_C": "form"})

    assert code == 2
    assert (answer["ok"], answer["outcome"], answer["kind"], answer["missing"]) == (
        False, "refused", "no_credentials", ["AUP_HUB_PASS_C"])
    assert sjc.hub.calls == []


def test_a_wrong_password_is_a_refused_launch_in_the_hubs_words_and_nothing_is_started(capsys):
    sjc = Sjc()

    answer, code = sjc.cli("launch", "--account", "C", "--minutes", "30",
                           environ={"AUP_HUB_LAUNCH_C": "form", "AUP_HUB_PASS_C": WRONG_PASSWORD})

    assert code == 1
    assert (answer["ok"], answer["outcome"], answer["kind"], answer["stage"]) == (False, "refused", "login_failed", "login")
    assert answer["error"] == "Invalid username or password"
    assert sjc.paths() == [("GET", "/hub/native/login"), ("POST", "/hub/native/login")]
    assert sjc.hub.running("C") is False
    assert only_row("pod launch")["outcome"] == "refused"
    assert WRONG_PASSWORD not in everything_written(answer, capsys)


def test_a_login_page_without_an_xsrf_token_is_an_error_and_the_password_is_not_posted():
    sjc = Sjc(launch="form")
    sjc.hub.omit_xsrf = True

    answer = sjc.controller().launch("C", 30)

    assert (answer["ok"], answer["outcome"], answer["kind"], answer["stage"]) == (False, "error", "xsrf_missing", "login_page")
    assert sjc.paths() == [("GET", "/hub/native/login")]
    assert sjc.hub.running("C") is False


def test_a_hub_that_drops_the_login_is_reported_unreachable_without_the_account_name_or_password(capsys):
    sjc = Sjc(launch="form")
    sjc.hub.queue_dropped("POST", "/hub/native/login")

    answer = sjc.controller().launch("C", 30)

    assert (answer["ok"], answer["outcome"], answer["kind"]) == (False, "error", "hub_unreachable")
    written = everything_written(answer, capsys)
    assert PASSWORD not in written and "acct-c" not in written


def test_an_error_page_that_echoes_the_password_is_cut_in_the_answer_and_in_the_ledger(capsys):
    sjc = Sjc(launch="form")
    sjc.hub.queue("POST", "/hub/native/login", 200, text=f'<p class="login_error">no account for {PASSWORD}</p>')

    answer = sjc.controller().launch("C", 30)

    assert (answer["outcome"], answer["kind"], answer["error"]) == ("refused", "login_failed", "no account for ***")
    assert PASSWORD not in everything_written(answer, capsys)


def test_a_spawn_the_hub_refuses_after_the_login_is_a_refused_launch():
    sjc = Sjc(launch="form")
    sjc.hub.queue("POST", "/hub/spawn/", 403, body={"status": 403, "message": "Cannot start container: no quota"})

    answer = sjc.controller().launch("C", 30)

    assert (answer["ok"], answer["outcome"], answer["kind"], answer["stage"], answer["status"]) == (
        False, "refused", "spawn_refused", "spawn", 403)
    assert answer["error"] == "Cannot start container: no quota"


def test_a_spawn_answered_with_502_that_the_hub_acted_on_is_found_by_reading_the_server_back_and_sent_once():
    sjc = Sjc(launch="form")
    sjc.hub.queue("POST", "/hub/spawn/", 502, apply=True)

    answer = sjc.controller().launch("C", 30)

    assert (answer["ok"], answer["outcome"], answer["launch_path"]) == (True, "running", "form")
    assert len(sjc.hub.calls_matching("POST", "/hub/spawn/")) == 1


def test_a_spawn_answered_with_502_that_started_nothing_is_unconfirmed_and_not_repeated():
    sjc = Sjc(launch="form")
    sjc.hub.queue("POST", "/hub/spawn/", 502)

    answer = sjc.controller().launch("C", 30)

    assert (answer["ok"], answer["outcome"], answer["kind"]) == (False, "error", "launch_unconfirmed")
    assert "not sent again" in answer["error"]
    assert len(sjc.hub.calls_matching("POST", "/hub/spawn/")) == 1
    assert sjc.hub.running("C") is False


def test_a_dry_run_of_a_form_launch_plans_the_form_without_the_password_and_sends_nothing(capsys):
    sjc = Sjc(launch="form")

    answer = sjc.controller().launch("C", 2400, accelerator="strix-halo", dry_run=True)

    assert (answer["ok"], answer["outcome"]) == (True, "dry_run")
    assert answer["plan"] == {"launch_path": "form", "login": "/hub/native/login", "method": "POST",
                              "path": "/hub/spawn/acct-c", "form": FORM_FIELDS}
    assert [call.method for call in sjc.hub.calls] == ["GET", "GET"]
    assert PASSWORD not in everything_written(answer, capsys)
    assert rows("pod launch") == []


# --- AUP_HUB_LAUNCH_<X> unset or auto: the REST call, then a look at the image -------------------------

def test_a_rest_launch_that_gave_the_stock_image_is_stopped_and_launched_again_through_the_form():
    sjc = Sjc()

    with fake_terminal(sjc) as base_url:
        answer = sjc.controller(base_url).launch("C", 600, accelerator="strix-halo")

    assert (answer["ok"], answer["outcome"], answer["launch_path"]) == (True, "running", "form")
    assert (answer["image_check"], answer["rest_pod_image"], answer["fell_back_from"]) == ("stock", STOCK_IMAGE, "rest")
    assert sjc.paths() == [("POST", "/hub/api/users/acct-c/server"), ("DELETE", "/hub/api/users/acct-c/server"),
                           ("GET", "/hub/native/login"), ("POST", "/hub/native/login"), ("GET", "/hub/spawn/acct-c"),
                           ("POST", "/hub/spawn/acct-c")]
    assert len(sjc.hub.calls_matching("POST", "/api/terminals")) == 1
    assert sjc.hub.calls_matching("POST", "/hub/api/users/acct-c/server")[0].json == {
        "runtime_minutes": 600, "resource_type": "gpu", "gpu_selection": "strix-halo"}
    assert sjc.hub.calls_matching("POST", "/hub/spawn/")[0].form["runtime"] == "600"
    assert sjc.hub.pod_image("C") == AUP_IMAGE
    assert answer["start_s"] >= 30.0
    assert answer["server"]["ready"] is True


def test_the_ledger_row_of_a_launch_that_fell_back_says_form_and_where_it_fell_back_from():
    sjc = Sjc()

    with fake_terminal(sjc) as base_url:
        sjc.controller(base_url).launch("C", 600, accelerator="strix-halo")

    pod = only_row("pod launch")
    assert pod["outcome"] == "running"
    assert plain(pod["conditions"]) == {"resource": "gpu", "accelerator": "strix-halo", "launch_path": "form",
                                 "fell_back_from": "rest"}


def test_the_start_time_of_a_rest_launch_stops_at_ready_before_the_pod_is_asked_anything():
    sjc = Sjc(rest_ignored=False)

    with fake_terminal(sjc, delay_s=20.0) as base_url:
        started = sjc.clock.time()
        answer = sjc.controller(base_url).launch("C", 600)

    assert sjc.clock.time() - started >= 35.0
    assert answer["start_s"] == 15.0
    assert only_row("pod launch")["metrics"] == {"pod.start_s": 15.0}


def test_the_pod_is_asked_by_a_small_script_that_reads_the_image_and_the_gpu_device():
    sjc = Sjc(rest_ignored=False)

    with fake_terminal(sjc) as base_url:
        sjc.controller(base_url).launch("C", 600)

    scripts = [text for (letter, name), text in sjc.hub.files.items() if name.startswith("lab_image_") and name.endswith(".sh")]
    assert len(scripts) == 1
    assert "JUPYTER_IMAGE_SPEC" in scripts[0] and "/dev/kfd" in scripts[0]


def test_a_rest_launch_that_gave_the_real_image_is_kept_and_the_form_is_never_touched():
    sjc = Sjc(rest_ignored=False)

    with fake_terminal(sjc) as base_url:
        answer = sjc.controller(base_url).launch("C", 600)

    assert (answer["ok"], answer["outcome"], answer["launch_path"]) == (True, "running", "rest")
    assert (answer["image_check"], answer["pod_image"]) == ("ok", AUP_IMAGE)
    assert "fell_back_from" not in answer
    assert sjc.hub.calls_matching(None, "/hub/native") == [] and sjc.hub.calls_matching(None, "/hub/spawn") == []
    assert sjc.hub.calls_matching("DELETE") == []
    assert plain(only_row("pod launch")["conditions"]) == {"resource": "gpu", "accelerator": "strix-halo", "launch_path": "rest"}


def test_a_pod_that_cannot_be_asked_is_left_alone_and_the_answer_says_it_is_unknown():
    sjc = Sjc()

    with closed_port() as base_url:
        answer = sjc.controller(base_url).launch("C", 600)

    assert (answer["ok"], answer["outcome"], answer["launch_path"], answer["image_check"]) == (True, "running", "rest", "unknown")
    assert "terminal" in answer["image_check_note"] and "acct-c" not in answer["image_check_note"]
    assert sjc.hub.calls_matching("DELETE") == [] and sjc.hub.calls_matching(None, "/hub/native") == []
    assert sjc.hub.running("C") is True


def test_the_image_name_settles_it_and_without_a_name_the_gpu_device_does_for_a_gpu_resource():
    for resource, says, verdict in (("gpu", "image=\ngpu_device=no\n", "stock"), ("gpu", "image=\ngpu_device=yes\n", "ok"),
                                    ("cpu", "image=\ngpu_device=no\n", "unknown"), ("gpu", "image=\n", "unknown")):
        assert ctl_module.image_verdict(**ctl_module.parse_image_answer(says), resource_type=resource) == verdict
    assert ctl_module.image_verdict("ghcr.io/amdresearch/auplc-base:latest", False, "gpu") == "ok"
    assert ctl_module.image_verdict("Quay.io/JupyterHub/Singleuser:latest", True, "gpu") == "stock"


def test_a_stock_pod_that_cannot_be_stopped_is_reported_and_no_form_is_tried():
    sjc = Sjc()
    sjc.hub.queue("DELETE", "/server", 500)

    with fake_terminal(sjc) as base_url:
        answer = sjc.controller(base_url).launch("C", 600)

    assert (answer["ok"], answer["outcome"], answer["kind"]) == (False, "error", "stock_pod_not_stopped")
    assert answer["launch_path"] == "rest" and answer["image_check"] == "stock"
    assert sjc.hub.calls_matching(None, "/hub/native") == []
    assert only_row("pod launch")["outcome"] == "error"


def test_when_the_form_fails_after_the_stock_pod_was_stopped_the_answer_says_both_and_no_pod_runs():
    sjc = Sjc(password=WRONG_PASSWORD)

    with fake_terminal(sjc) as base_url:
        answer = sjc.controller(base_url).launch("C", 600)

    assert (answer["ok"], answer["outcome"], answer["kind"], answer["launch_path"]) == (False, "refused", "login_failed", "form")
    assert answer["error"].startswith("the REST launch gave the stock image, which was stopped; the launch through the form then failed")
    assert answer["fell_back_from"] == "rest"
    assert sjc.hub.running("C") is False
    assert WRONG_PASSWORD not in json.dumps(answer) + ledger.ledger_path().read_text(encoding="utf-8")


def test_a_launch_that_does_not_wait_asks_the_pod_nothing():
    sjc = Sjc()

    answer = sjc.controller().launch("C", 600, wait=False)

    assert (answer["outcome"], answer["launch_path"]) == ("launching", "rest")
    assert "image_check" not in answer
    assert sjc.hub.calls_matching(None, "/api/terminals") == []


def test_a_launch_of_a_pod_that_is_already_up_asks_it_nothing_and_names_no_path():
    sjc = Sjc()
    sjc.hub.start_server("C", image=STOCK_IMAGE)

    answer = sjc.controller().launch("C", 600)

    assert answer["outcome"] == "already_running"
    assert "launch_path" not in answer and "image_check" not in answer
    assert sjc.hub.calls_matching(None, "/api/terminals") == [] and sjc.hub.calls_matching("POST") == []


# --- the REST path of accounts A and B is the one it was ---------------------------------------------

class Dub:
    """DUB as a fake (credits charged) with account A, who has no password, as A and B have in the .env they use."""

    def __init__(self, **account_options):
        self.clock = FakeClock("2026-10-10T09:27:00Z")
        self.hub = FakeHub(self.clock)
        account = self.hub.add_account("A", user="acct-a", token=TOKEN_A, balance=59960)
        self.account = replace(account, **account_options)

    def controller(self):
        return Controller({"A": self.account}, base_url="https://hub.test", transport=self.hub.transport,
                          sleep=self.clock.sleep, clock=self.clock.time)


def test_an_account_without_a_password_launches_by_rest_alone_exactly_as_before():
    dub = Dub()

    answer = dub.controller().launch("A", 600)

    assert [(call.method, call.path) for call in dub.hub.calls if call.method != "GET"] == [
        ("POST", "/hub/api/users/acct-a/server")]
    assert dub.hub.calls_matching("POST", "/server")[0].json == {
        "runtime_minutes": 600, "resource_type": "gpu", "gpu_selection": "strix-halo"}
    assert (answer["outcome"], answer["start_s"], answer["balance"], answer["cost_credits"]) == ("running", 15.0, 59960, 2400)
    assert answer["launch_path"] == "rest" and "image_check" not in answer
    assert dub.hub.calls_matching(None, "/api/terminals") == [] and dub.hub.calls_matching(None, "/hub/native") == []


def test_the_rest_key_keeps_an_account_that_has_a_password_on_the_rest_call_without_any_look_at_the_pod():
    dub = Dub(password="fake-password-of-account-a", launch="rest")

    answer = dub.controller().launch("A", 600)

    assert (answer["outcome"], answer["launch_path"]) == ("running", "rest")
    assert "image_check" not in answer
    assert dub.hub.calls_matching(None, "/api/terminals") == [] and dub.hub.calls_matching(None, "/hub/native") == []
    assert plain(only_row("pod launch")["conditions"]) == {"resource": "gpu", "accelerator": "strix-halo", "launch_path": "rest"}


# --- accelerator and resource: arguments, and per-account defaults ---------------------------------------

def test_the_accelerator_and_resource_come_from_the_arguments_then_the_account_keys_then_the_defaults():
    sjc = Sjc()
    env = {"AUP_HUB_LAUNCH_C": "rest"}

    sjc.cli("launch", "--account", "C", "--minutes", "30", environ=env)
    sjc.hub.drop_server("C")
    sjc.cli("launch", "--account", "C", "--minutes", "30", environ={**env, "AUP_HUB_GPU_C": "r9700",
                                                                  "AUP_HUB_RESOURCE_C": "code-gpu"})
    sjc.hub.drop_server("C")
    sjc.cli("launch", "--account", "C", "--minutes", "30", "--accelerator", "9070xt", "--resource", "gpu",
            environ={**env, "AUP_HUB_GPU_C": "r9700", "AUP_HUB_RESOURCE_C": "code-gpu"})

    assert [(call.json["resource_type"], call.json["gpu_selection"]) for call in sjc.hub.calls_matching("POST", "/server")] == [
        ("gpu", "strix-halo"), ("code-gpu", "r9700"), ("gpu", "9070xt")]


def test_the_form_names_the_accelerator_under_the_resource_the_account_launches_by_default():
    sjc = Sjc()

    answer, code = sjc.cli("launch", "--account", "C", "--minutes", "2400",
                           environ={"AUP_HUB_LAUNCH_C": "form", "AUP_HUB_PASS_C": PASSWORD, "AUP_HUB_GPU_C": "strix-halo",
                                    "AUP_HUB_RESOURCE_C": "code-gpu"})

    assert (code, answer["outcome"], answer["launch_path"], answer["resource_type"]) == (0, "running", "form", "code-gpu")
    assert sjc.hub.calls_matching("POST", "/hub/spawn/")[0].form["gpu_selection_code-gpu"] == "strix-halo"


def test_the_automatic_accelerator_is_refused_on_the_command_line_and_nothing_is_sent():
    sjc = Sjc()

    for argv in (["--accelerator", "auto"], ["--accelerator", " "], ["--resource", ""]):
        with pytest.raises(SystemExit) as caught:
            sjc.cli("launch", "--account", "C", "--minutes", "30", *argv)
        assert caught.value.code == 2

    assert sjc.hub.calls == []


def test_a_launch_mode_that_is_not_form_rest_or_auto_is_refused_and_nothing_is_sent():
    sjc = Sjc(launch="fom")

    with pytest.raises(ValueError) as caught:
        sjc.controller().launch("C", 30)

    assert "AUP_HUB_LAUNCH_C" in str(caught.value) and "form, rest, auto" in str(caught.value)
    assert sjc.hub.calls == []


def test_the_mode_is_read_without_regard_to_case_and_space():
    sjc = Sjc(launch="  FORM ")

    answer = sjc.controller().launch("C", 30)

    assert (answer["outcome"], answer["launch_path"]) == ("running", "form")


# --- the password is never printed, logged or written ---------------------------------------------------

def test_nothing_of_the_password_reaches_an_answer_a_ledger_row_or_the_output_on_any_path(capsys):
    answers = []
    sjc = Sjc(launch="form")
    answers.append(sjc.controller().launch("C", 30))
    answers.append(sjc.cli("stop", "--account", "C")[0])
    answers.append(sjc.cli("launch", "--account", "C", "--minutes", "30", "--dry-run",
                           environ={"AUP_HUB_LAUNCH_C": "form", "AUP_HUB_PASS_C": PASSWORD})[0])
    answers.append(sjc.cli("launch", "--account", "C", "--minutes", "30",
                           environ={"AUP_HUB_LAUNCH_C": "form", "AUP_HUB_PASS_C": PASSWORD})[0])
    other = Sjc()
    with fake_terminal(other) as base_url:
        answers.append(other.controller(base_url).launch("C", 30))

    written = everything_written(answers, capsys)
    assert PASSWORD not in written and TOKEN_C not in written
    assert [answer["outcome"] for answer in answers] == ["running", "ok", "dry_run", "running", "running"]


def test_the_controller_cuts_the_password_out_of_any_text_it_is_given():
    controller = Sjc().controller()

    assert controller.scrub(f"bad {PASSWORD} for acct-c with {TOKEN_C}") == "bad *** for <account C> with ***"
