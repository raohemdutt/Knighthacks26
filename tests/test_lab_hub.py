"""lab.hub: the one place the AUP hub is spoken to. Every test drives the real Hub class against an
httpx mock transport, which is the network boundary; nothing here opens a socket or reads a .env file."""
from __future__ import annotations

import datetime as dt
import json
import os
import re
from dataclasses import replace

import httpx
import pytest

from lab import hub
from lab.hub import Account, Hub, HubUnreachable, MissingCredentials, load_accounts
from lab.hub_fake import AUP_IMAGE, FakeClock, FakeHub

TOKEN = "0123456789abcdef0123456789abcdef"
OTHER_TOKEN = "fedcba9876543210fedcba9876543210"


def make_hub(handler, *, user="acct-a", sleeps=None, **kwargs):
    """A Hub for one account whose network is `handler`; sleeps collects the pauses it asked for."""
    sleeps = [] if sleeps is None else sleeps
    return Hub(Account("A", user, TOKEN), base_url="https://hub.test", transport=httpx.MockTransport(handler),
               sleep=sleeps.append, **kwargs)


def ok_json(payload, **headers):
    return httpx.Response(200, json=payload, headers=headers)


# --- what every call looks like on the wire ---------------------------------------------------

def test_every_call_carries_the_token_and_a_browser_user_agent():
    seen = []

    def handler(request):
        seen.append(request)
        return ok_json({"fine": True})

    client = make_hub(handler)
    client.get("/hub/api/users/acct-a")
    client.post("/hub/api/users/acct-a/server", json={"runtime_minutes": 5})
    client.delete("/hub/api/users/acct-a/server")
    client.put("/user/acct-a/api/contents/x.txt", json={"type": "file"})

    assert [request.method for request in seen] == ["GET", "POST", "DELETE", "PUT"]
    for request in seen:
        assert request.headers["authorization"] == f"token {TOKEN}"
        assert request.headers["user-agent"] == hub.USER_AGENT
        assert request.headers["user-agent"].startswith("Mozilla/5.0")
        assert request.url.host == "hub.test"


def test_a_json_body_goes_out_as_json_and_params_become_a_query():
    seen = []

    def handler(request):
        seen.append(request)
        return ok_json({})

    make_hub(handler).post("/hub/api/users/acct-a/server", json={"runtime_minutes": 600, "resource_type": "gpu"},
                           params={"a": "1"})
    request = seen[0]
    assert json.loads(request.content) == {"runtime_minutes": 600, "resource_type": "gpu"}
    assert request.headers["content-type"] == "application/json"
    assert request.url.query == b"a=1"


def test_a_redirect_is_handed_back_and_never_followed():
    seen = []

    def handler(request):
        seen.append(str(request.url))
        if len(seen) > 1:
            raise AssertionError("the redirect was followed")
        return httpx.Response(302, headers={"location": "https://hub.test/hub/api/oauth2/authorize?client_id=x"})

    reply = make_hub(handler).get("/user/acct-a/proxy/8000/v1/models")
    assert reply.status == 302
    assert not reply.ok
    assert len(seen) == 1


def test_the_hub_address_comes_from_the_environment_when_it_is_not_given():
    seen = []

    def handler(request):
        seen.append(request.url.host)
        return ok_json({})

    client = Hub(Account("A", "acct-a", TOKEN), transport=httpx.MockTransport(handler),
                 environ={"LAB_HUB_URL": "https://elsewhere.test/"})
    client.get("/hub/api/")
    assert seen == ["elsewhere.test"]
    assert client.base_url == "https://elsewhere.test"
    default = Hub(Account("A", "acct-a", TOKEN), transport=httpx.MockTransport(handler), environ={})
    assert default.base_url == "https://dub.aupcloud.io"


@pytest.mark.parametrize("given, own, shared, expected", [
    ("https://given.test", "https://sjc.test", "https://shared.test", "https://given.test"),
    (None, "https://sjc.test/", "https://shared.test", "https://sjc.test"),
    (None, None, "https://shared.test/", "https://shared.test"),
    (None, None, None, "https://dub.aupcloud.io"),
])
def test_the_hub_address_is_the_given_one_then_the_accounts_own_then_lab_hub_url_then_dub(given, own, shared, expected):
    seen = []

    def handler(request):
        seen.append(request.url.host)
        return ok_json({})

    environ = {"LAB_HUB_URL": shared} if shared else {}
    client = Hub(Account("C", "acct-c", TOKEN, own), base_url=given, transport=httpx.MockTransport(handler), environ=environ)
    client.get("/hub/api/")
    assert client.base_url == expected
    assert seen == [expected.split("//")[1]]


# --- replies --------------------------------------------------------------------------------

def test_a_reply_parses_json_and_reads_the_hub_clock_from_the_date_header():
    client = make_hub(lambda request: ok_json({"kind": "user"}, date="Sat, 10 Oct 2026 11:03:23 GMT"))
    reply = client.get("/hub/api/users/acct-a")
    assert reply.ok
    assert reply.body == {"kind": "user"}
    assert reply.hub_time == dt.datetime(2026, 10, 10, 11, 3, 23, tzinfo=dt.timezone.utc)
    assert reply.attempts == 1


def test_a_token_that_straddles_the_end_of_the_kept_text_leaves_no_fragment():
    text = "x" * (hub.TEXT_CHARS - 5) + TOKEN
    reply = make_hub(lambda request: httpx.Response(200, text=text)).get("/hub/api/users/acct-a")
    assert TOKEN[:5] not in reply.text
    assert reply.text.endswith("***")


def test_a_reply_without_json_or_a_date_has_no_body_and_no_hub_time():
    reply = make_hub(lambda request: httpx.Response(202, text="")).post("/hub/api/users/acct-a/server")
    assert reply.status == 202
    assert reply.body is None
    assert reply.hub_time is None


def test_detail_prefers_the_hub_error_message_then_a_stripped_error_page():
    forbidden = make_hub(lambda request: httpx.Response(403, json={"status": 403, "message": "Missing or invalid credentials."}))
    assert forbidden.get("/hub/api/users/acct-a").detail() == "Missing or invalid credentials."

    page = "<html><head><title>502</title></head><body><h1>Bad   gateway</h1>\n<p>try later</p></body></html>"
    gateway = make_hub(lambda request: httpx.Response(502, text=page), retries=0)
    assert gateway.get("/hub/api/users/acct-a").detail() == "502 Bad gateway try later"


def test_detail_of_a_redirect_does_not_repeat_the_address_it_points_at():
    reply = make_hub(lambda request: httpx.Response(
        302, headers={"location": "https://hub.test/hub/api/oauth2/authorize?client_id=jupyterhub-user-acct-a"})
    ).get("/user/acct-a/api/status")
    assert "acct-a" not in reply.detail()
    assert "redirect" in reply.detail()


# --- retries --------------------------------------------------------------------------------

def test_a_get_is_retried_after_gateway_errors_with_growing_pauses():
    answers = [502, 503, 200]
    sleeps = []

    def handler(request):
        status = answers.pop(0)
        return ok_json({"ok": 1}) if status == 200 else httpx.Response(status, text="down")

    reply = make_hub(handler, sleeps=sleeps).get("/hub/api/users/acct-a")
    assert reply.status == 200
    assert reply.attempts == 3
    assert sleeps == [1.0, 2.0]


def test_retries_stop_and_the_last_gateway_error_is_handed_back():
    calls = []
    sleeps = []

    def handler(request):
        calls.append(1)
        return httpx.Response(502, text="Bad gateway")

    reply = make_hub(handler, sleeps=sleeps, retries=3).get("/hub/api/users/acct-a")
    assert reply.status == 502
    assert reply.attempts == 4
    assert len(calls) == 4
    assert sleeps == [1.0, 2.0, 4.0]


def test_a_post_is_never_retried_because_the_first_try_may_have_landed():
    calls = []
    sleeps = []

    def handler(request):
        calls.append(request.method)
        return httpx.Response(502, text="Bad gateway")

    reply = make_hub(handler, sleeps=sleeps).post("/hub/api/users/acct-a/server", json={})
    assert reply.status == 502
    assert reply.attempts == 1
    assert calls == ["POST"]
    assert sleeps == []


def test_a_delete_is_retried_after_a_dropped_connection():
    calls = []

    def handler(request):
        calls.append(request.method)
        if len(calls) == 1:
            raise httpx.ConnectError("connection reset")
        return httpx.Response(204)

    reply = make_hub(handler).delete("/hub/api/users/acct-a/server")
    assert reply.status == 204
    assert calls == ["DELETE", "DELETE"]


def test_a_dropped_connection_that_never_recovers_raises_without_the_token():
    def handler(request):
        raise httpx.ConnectError(f"tls failure while sending Authorization: token {TOKEN}")

    with pytest.raises(HubUnreachable) as caught:
        make_hub(handler).get("/hub/api/users/acct-a")
    assert TOKEN not in str(caught.value)
    assert "***" in str(caught.value)
    assert "GET /hub/api/users/acct-a" in str(caught.value)


def test_a_timeout_is_a_transport_failure_too():
    def handler(request):
        raise httpx.ReadTimeout("slow")

    with pytest.raises(HubUnreachable):
        make_hub(handler, retries=1).get("/hub/api/users/acct-a")


# --- addresses ------------------------------------------------------------------------------

def test_user_and_pod_addresses_quote_the_account_name():
    client = Hub(Account("A", "amd user/1", TOKEN), base_url="https://hub.test", transport=httpx.MockTransport(lambda r: ok_json({})))
    assert client.user_url() == "/hub/api/users/amd%20user%2F1"
    assert client.user_url("/server") == "/hub/api/users/amd%20user%2F1/server"
    assert client.pod_url("/api/status") == "/user/amd%20user%2F1/api/status"


def test_the_terminal_address_follows_the_hub_scheme():
    secure = Hub(Account("A", "acct-a", TOKEN), base_url="https://hub.test", transport=httpx.MockTransport(lambda r: ok_json({})))
    plain = Hub(Account("A", "acct-a", TOKEN), base_url="http://127.0.0.1:8123", transport=httpx.MockTransport(lambda r: ok_json({})))
    assert secure.ws_url("/terminals/websocket/1") == "wss://hub.test/user/acct-a/terminals/websocket/1"
    assert plain.ws_url("/terminals/websocket/1") == "ws://127.0.0.1:8123/user/acct-a/terminals/websocket/1"


def test_the_websocket_headers_carry_the_token_and_the_browser_agent():
    client = make_hub(lambda request: ok_json({}))
    assert client.ws_headers() == {"Authorization": f"token {TOKEN}"}
    assert client.user_agent == hub.USER_AGENT


# --- accounts -------------------------------------------------------------------------------

ENV = {"AUP_HUB_USER_A": "acct-a", "AUP_HUB_TOKEN_A": TOKEN, "AUP_HUB_USER_B": "acct-b", "AUP_HUB_TOKEN_B": OTHER_TOKEN}


def test_accounts_are_read_from_the_process_environment():
    accounts = load_accounts(["A", "B"], environ=ENV, env_files=[])
    assert list(accounts) == ["A", "B"]
    assert (accounts["A"].letter, accounts["A"].user, accounts["A"].token, accounts["A"].node) == ("A", "acct-a", TOKEN, "pod-a")
    assert (accounts["B"].user, accounts["B"].token, accounts["B"].node) == ("acct-b", OTHER_TOKEN, "pod-b")


def test_an_account_never_shows_its_token_when_printed():
    account = load_accounts(["A"], environ=ENV, env_files=[])["A"]
    assert TOKEN not in repr(account)
    assert TOKEN not in str(account)
    assert TOKEN not in repr(make_hub(lambda request: ok_json({})))


def test_missing_keys_are_named_without_any_value_and_blank_counts_as_missing():
    environ = {"AUP_HUB_USER_A": "acct-a", "AUP_HUB_TOKEN_A": "   ", "AUP_HUB_USER_B": "acct-b"}
    with pytest.raises(MissingCredentials) as caught:
        load_accounts(["A", "B"], environ=environ, env_files=[])
    assert caught.value.missing == ["AUP_HUB_TOKEN_A", "AUP_HUB_TOKEN_B"]
    assert "acct-a" not in str(caught.value)


def test_only_the_asked_for_account_needs_its_keys():
    only_a = {key: value for key, value in ENV.items() if key.endswith("_A")}
    assert list(load_accounts(["A"], environ=only_a, env_files=[])) == ["A"]
    with pytest.raises(MissingCredentials) as caught:
        load_accounts(["B"], environ=only_a, env_files=[])
    assert caught.value.missing == ["AUP_HUB_USER_B", "AUP_HUB_TOKEN_B"]


def test_a_dotenv_file_fills_in_what_the_environment_lacks_and_the_environment_wins(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"AUP_HUB_USER_A=from-file\nAUP_HUB_TOKEN_A={OTHER_TOKEN}\nexport AUP_HUB_USER_B='acct-b'\n"
                      f"AUP_HUB_TOKEN_B={OTHER_TOKEN}\n", encoding="utf-8")
    accounts = load_accounts(["A", "B"], environ={"AUP_HUB_USER_A": "from-env"}, env_files=[dotenv])
    assert accounts["A"].user == "from-env"
    assert accounts["A"].token == OTHER_TOKEN
    assert accounts["B"].user == "acct-b"


def test_the_search_for_dotenv_files_is_the_lab_machines_own(tmp_path):
    first = tmp_path / "one.env"
    second = tmp_path / "two.env"
    first.write_text("AUP_HUB_USER_A=one\n", encoding="utf-8")
    second.write_text(f"AUP_HUB_USER_A=two\nAUP_HUB_TOKEN_A={TOKEN}\n", encoding="utf-8")
    environ = {"LAB_ENV_FILES": f"{first}{os.pathsep}{second}"}
    accounts = load_accounts(["A"], environ=environ)
    assert accounts["A"].user == "one"
    assert accounts["A"].token == TOKEN


# --- accounts beyond A and B, and the hub each lives on -------------------------------------------

SJC_ENV = {"LAB_POD_ROLES": "C:pod-a,D:pod-b", "AUP_HUB_USER_C": "acct-c", "AUP_HUB_TOKEN_C": TOKEN, "AUP_HUB_URL_C": "https://sjc.test/",
           "AUP_HUB_USER_D": "acct-d", "AUP_HUB_TOKEN_D": OTHER_TOKEN}


def test_any_letter_with_its_user_and_token_keys_is_an_account_with_its_own_node_and_hub():
    accounts = load_accounts(["C", "D"], environ=SJC_ENV, env_files=[])

    assert list(accounts) == ["C", "D"]
    assert [(a.letter, a.user, a.token, a.node, a.url) for a in accounts.values()] == [
        ("C", "acct-c", TOKEN, "pod-a", "https://sjc.test/"), ("D", "acct-d", OTHER_TOKEN, "pod-b", None)]
    assert (hub.node_of("A"), hub.node_of("B")) == ("pod-a", "pod-b")


def test_the_address_key_is_optional_a_blank_one_is_unset_and_a_dotenv_file_may_hold_it(tmp_path):
    keys = {"LAB_POD_ROLES": "C:pod-a,D:pod-b", "AUP_HUB_USER_C": "acct-c", "AUP_HUB_TOKEN_C": TOKEN}
    assert load_accounts(["C"], environ=keys, env_files=[])["C"].url is None
    assert load_accounts(["C"], environ={**keys, "AUP_HUB_URL_C": "   "}, env_files=[])["C"].url is None

    dotenv = tmp_path / ".env"
    dotenv.write_text("AUP_HUB_URL_C=https://sjc.test\nAUP_HUB_URL_D=https://elsewhere.test\n", encoding="utf-8")
    accounts = load_accounts(["C", "D"], environ={**keys, "AUP_HUB_USER_D": "acct-d", "AUP_HUB_TOKEN_D": OTHER_TOKEN,
                                                  "AUP_HUB_URL_D": "https://from-env.test"}, env_files=[dotenv])
    assert (accounts["C"].url, accounts["D"].url) == ("https://sjc.test", "https://from-env.test")


def test_the_missing_keys_of_a_letter_are_named_and_the_optional_address_is_never_one_of_them():
    with pytest.raises(MissingCredentials) as caught:
        load_accounts(["C", "D"], environ={"LAB_POD_ROLES": "C:pod-a,D:pod-b", "AUP_HUB_USER_C": "acct-c",
                                   "AUP_HUB_TOKEN_C": TOKEN}, env_files=[])
    assert caught.value.missing == ["AUP_HUB_USER_D", "AUP_HUB_TOKEN_D"]


def test_an_account_takes_its_hub_address_from_the_key_and_sends_its_calls_there():
    seen = []

    def handler(request):
        seen.append((request.url.host, request.headers["authorization"]))
        return ok_json({})

    account = load_accounts(["C"], environ=SJC_ENV, env_files=[])["C"]
    client = Hub(account, transport=httpx.MockTransport(handler), environ={"LAB_HUB_URL": "https://shared.test"})
    client.get(client.user_url())

    assert client.base_url == "https://sjc.test"
    assert seen == [("sjc.test", f"token {TOKEN}")]
    assert TOKEN not in repr(account)


PASSWORD = "fake-correct-horse-battery"


# --- the optional keys of an account: password, launch mode, accelerator, resource ---------------------

def test_the_optional_keys_are_read_like_the_others_and_none_of_them_is_ever_required():
    keys = {"LAB_POD_ROLES": "C:pod-a,D:pod-b", "AUP_HUB_USER_C": "acct-c", "AUP_HUB_TOKEN_C": TOKEN}
    bare = load_accounts(["C"], environ=keys, env_files=[])["C"]
    assert (bare.password, bare.launch, bare.gpu, bare.resource) == (None, None, None, None)

    full = {**keys, "AUP_HUB_PASS_C": PASSWORD, "AUP_HUB_LAUNCH_C": "form", "AUP_HUB_GPU_C": "strix-halo",
            "AUP_HUB_RESOURCE_C": "gpu"}
    account = load_accounts(["C"], environ=full, env_files=[])["C"]
    assert (account.password, account.launch, account.gpu, account.resource) == (PASSWORD, "form", "strix-halo", "gpu")

    blank = {**keys, "AUP_HUB_PASS_C": "  ", "AUP_HUB_LAUNCH_C": ""}
    assert load_accounts(["C"], environ=blank, env_files=[])["C"].password is None


def test_the_optional_keys_may_come_from_a_dotenv_file_and_only_for_the_account_asked_for(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"AUP_HUB_PASS_C={PASSWORD}\nAUP_HUB_PASS_D=other-password-1\nAUP_HUB_GPU_D=r9700\n",
                      encoding="utf-8")
    keys = {"LAB_POD_ROLES": "C:pod-a,D:pod-b", "AUP_HUB_USER_C": "acct-c", "AUP_HUB_TOKEN_C": TOKEN}

    account = load_accounts(["C"], environ=keys, env_files=[dotenv])["C"]

    assert (account.password, account.gpu) == (PASSWORD, None)


def test_a_password_is_never_shown_when_an_account_or_a_client_is_printed():
    account = Account("C", "acct-c", TOKEN, "https://sjc.test", PASSWORD, "form", "strix-halo", "gpu")
    client = Hub(account, transport=httpx.MockTransport(lambda request: ok_json({})))

    for shown in (repr(account), str(account), repr(client)):
        assert PASSWORD not in shown and TOKEN not in shown


def test_the_password_is_cut_out_of_a_reply_text_like_the_token():
    account = Account("C", "acct-c", TOKEN, None, PASSWORD)
    client = Hub(account, base_url="https://sjc.test", transport=httpx.MockTransport(
        lambda request: httpx.Response(400, text=f"bad {PASSWORD} and {TOKEN}")))

    reply = client.get("/anything")

    assert PASSWORD not in reply.text and TOKEN not in reply.text
    assert client.scrub(f"x {PASSWORD} y") == "x *** y"


# --- the web spawn form ----------------------------------------------------------------------


def form_world(**fake_attributes):
    """The SJC hub as a fake (quota off) with account C, who can log in, and a client for it."""
    clock = FakeClock("2026-10-10T09:27:00Z")
    fake = FakeHub(clock)
    for name, value in fake_attributes.items():
        setattr(fake, name, value)
    account = fake.add_account("C", user="acct-c", token=TOKEN, quota_enabled=False, password=PASSWORD)
    return fake, Hub(account, base_url="https://sjc.test", transport=fake.transport, sleep=clock.sleep), clock


def xsrf_of(call):
    return re.search(r"(?:^|; )_xsrf=([^;]+)", call.headers["cookie"]).group(1)


def test_the_form_logs_in_with_cookies_then_posts_the_spawn_fields_with_the_xsrf_cookie_in_the_query():
    fake, client, clock = form_world()

    result = client.spawn_form(2400, "gpu", "strix-halo")

    assert (result.ok, result.stage, result.status, result.kind, result.error) == (True, "spawn", 302, None, None)
    assert result.location == "/hub/spawn-pending/acct-c"
    assert [(call.method, call.path) for call in fake.calls] == [
        ("GET", "/hub/native/login"), ("POST", "/hub/native/login"), ("GET", "/hub/spawn/acct-c"),
        ("POST", "/hub/spawn/acct-c")]
    login_page, login, spawn_page, spawn = fake.calls
    assert "cookie" not in login_page.headers
    assert login.form == {"username": "acct-c", "password": PASSWORD, "_xsrf": xsrf_of(login)}
    assert "jupyterhub-hub-login=" in spawn_page.headers["cookie"]
    assert spawn.query == f"_xsrf={xsrf_of(spawn)}"
    assert spawn.form == {"resource_type": "gpu", "gpu_selection_gpu": "strix-halo", "runtime": "2400", "repo_url": "",
                          "repo_persist": "true", "repo_branch": ""}
    assert [call.status for call in fake.calls] == [200, 302, 200, 302]
    clock.sleep(fake.spawn_s)
    assert fake.pod_image("C") == AUP_IMAGE
    options = client.get(client.user_url()).body["servers"][""]["user_options"]
    assert options == {"gpu_selection": "strix-halo", "resource_type": "gpu", "runtime_minutes": 2400}


def test_the_form_session_never_carries_the_token_and_the_api_calls_never_carry_its_cookies():
    fake, client, _ = form_world()

    client.spawn_form(30, "gpu", "r9700")
    client.get(client.user_url())

    web, api = fake.calls[:-1], fake.calls[-1]
    assert all("authorization" not in call.headers for call in web)
    assert all(call.headers["user-agent"] == hub.USER_AGENT for call in fake.calls)
    assert api.headers["authorization"] == f"token {TOKEN}" and "cookie" not in api.headers


def test_a_wrong_password_is_a_failed_login_in_the_hubs_words_and_nothing_is_spawned():
    fake, client, _ = form_world()
    wrong = Hub(replace(client.account, password="fake-wrong-password-1"), base_url="https://sjc.test",
                transport=fake.transport)

    result = wrong.spawn_form(30, "gpu", "r9700")

    assert (result.ok, result.stage, result.kind, result.status) == (False, "login", "login_failed", 200)
    assert result.error == "Invalid username or password"
    assert [(call.method, call.path) for call in fake.calls] == [("GET", "/hub/native/login"), ("POST", "/hub/native/login")]
    assert fake.running("C") is False
    assert "fake-wrong-password-1" not in repr(result)


def test_a_login_page_without_any_xsrf_token_stops_before_the_password_is_sent():
    fake, client, _ = form_world(omit_xsrf=True)

    result = client.spawn_form(30, "gpu", "r9700")

    assert (result.ok, result.stage, result.kind) == (False, "login_page", "xsrf_missing")
    assert "_xsrf" in result.error
    assert [(call.method, call.path) for call in fake.calls] == [("GET", "/hub/native/login")]
    assert PASSWORD not in json.dumps([call.form for call in fake.calls])


def test_the_hidden_field_is_used_when_the_hub_sets_no_xsrf_cookie_but_the_spawn_needs_the_cookie():
    posted = []

    def handler(request):
        if request.url.path == "/hub/native/login" and request.method == "GET":
            filler = "<style>" + "x" * 5000 + "</style>"
            return httpx.Response(200, text=f'{filler}<input value="hidden-token" name="_xsrf" type="hidden">')
        if request.url.path == "/hub/native/login":
            posted.append(request.content.decode())
            return httpx.Response(302, headers=[("location", "/hub/"), ("set-cookie", "jupyterhub-hub-login=s; Path=/hub/")])
        return httpx.Response(200, text="<html>the spawn page</html>")

    client = Hub(Account("C", "acct-c", TOKEN, None, PASSWORD), base_url="https://sjc.test",
                 transport=httpx.MockTransport(handler))

    result = client.spawn_form(30, "gpu", "r9700")

    assert "_xsrf=hidden-token" in posted[0]
    assert (result.ok, result.stage, result.kind) == (False, "spawn_page", "xsrf_missing")


def test_an_account_without_a_password_cannot_use_the_form_and_nothing_is_sent():
    fake, client, _ = form_world()
    bare = Hub(replace(client.account, password=None), base_url="https://sjc.test", transport=fake.transport)

    with pytest.raises(MissingCredentials) as caught:
        bare.spawn_form(30, "gpu", "r9700")

    assert caught.value.missing == ["AUP_HUB_PASS_C"]
    assert fake.calls == []


def test_a_hub_that_does_not_answer_is_a_result_not_an_exception_and_leaks_nothing():
    def handler(request):
        raise httpx.ConnectError(f"cannot reach https://sjc.test with {PASSWORD}")

    client = Hub(Account("C", "acct-c", TOKEN, None, PASSWORD), base_url="https://sjc.test",
                 transport=httpx.MockTransport(handler), sleep=lambda seconds: None)

    result = client.spawn_form(30, "gpu", "r9700")

    assert (result.ok, result.stage, result.kind, result.status) == (False, "login_page", "no_answer", 0)
    assert PASSWORD not in result.error and "***" in result.error


def test_a_page_that_echoes_the_password_in_its_error_is_cut_before_it_is_returned():
    def handler(request):
        if request.method == "GET":
            return httpx.Response(200, text="", headers={"set-cookie": "_xsrf=t; Path=/hub/"})
        return httpx.Response(200, text=f'<p class="login_error">no account for {PASSWORD}</p>')

    client = Hub(Account("C", "acct-c", TOKEN, None, PASSWORD), base_url="https://sjc.test",
                 transport=httpx.MockTransport(handler))

    result = client.spawn_form(30, "gpu", "r9700")

    assert (result.kind, result.error) == ("login_failed", "no account for ***")


def test_a_get_is_tried_again_after_a_gateway_error_but_a_post_never_is():
    fake, client, _ = form_world()
    fake.queue("GET", "/hub/native/login", 502)
    first = client.spawn_form(30, "gpu", "r9700")
    assert first.ok is True
    assert [call.status for call in fake.calls_matching("GET", "/hub/native/login")] == [502, 200]

    again, other, _ = form_world()
    again.queue("POST", "/hub/native/login", 502)
    second = other.spawn_form(30, "gpu", "r9700")
    assert (second.ok, second.stage, second.kind, second.status) == (False, "login", "unexpected_status", 502)
    assert len(again.calls_matching("POST", "/hub/native/login")) == 1


def test_a_spawn_the_hub_refuses_is_reported_with_its_message_and_one_post():
    fake, client, _ = form_world()
    fake.queue("POST", "/hub/spawn/", 403, body={"status": 403, "message": "Cannot start container: Insufficient quota"})

    result = client.spawn_form(30, "gpu", "r9700")

    assert (result.ok, result.stage, result.kind, result.status) == (False, "spawn", "spawn_refused", 403)
    assert result.error == "Cannot start container: Insufficient quota"
    assert len(fake.calls_matching("POST", "/hub/spawn/")) == 1


def test_a_spawn_sent_back_to_the_login_page_is_not_a_started_server():
    def handler(request):
        if request.url.path == "/hub/native/login":
            if request.method == "GET":
                return httpx.Response(200, text="", headers={"set-cookie": "_xsrf=t; Path=/hub/"})
            return httpx.Response(302, headers=[("location", "/hub/"), ("set-cookie", "jupyterhub-hub-login=s; Path=/hub/")])
        if request.method == "GET":
            return httpx.Response(200, text="<html>the spawn page</html>")
        return httpx.Response(302, headers={"location": "/hub/login?next=%2Fhub%2Fspawn%2Facct-c"})

    client = Hub(Account("C", "acct-c", TOKEN, None, PASSWORD), base_url="https://sjc.test",
                 transport=httpx.MockTransport(handler))

    result = client.spawn_form(30, "gpu", "r9700")

    assert (result.ok, result.kind, result.status) == (False, "spawn_refused", 302)
    assert "login page" in result.error


def test_the_form_fields_name_the_accelerator_under_the_resource_it_is_for():
    assert hub.spawn_fields(2400, "gpu", "strix-halo") == {
        "resource_type": "gpu", "gpu_selection_gpu": "strix-halo", "runtime": "2400", "repo_url": "",
        "repo_persist": "true", "repo_branch": ""}
    assert hub.spawn_fields(30, "code-gpu", "r9700")["gpu_selection_code-gpu"] == "r9700"


def test_a_login_redirect_that_sets_no_session_cookie_is_not_a_login():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(200, text="", headers={"set-cookie": "_xsrf=t; Path=/hub/"})
        return httpx.Response(302, headers={"location": "/hub/native/login"})

    client = Hub(Account("C", "acct-c", TOKEN, None, PASSWORD), base_url="https://sjc.test",
                 transport=httpx.MockTransport(handler))

    result = client.spawn_form(30, "gpu", "r9700")

    assert (result.ok, result.stage, result.kind, result.status) == (False, "login", "login_failed", 302)
    assert seen == [("GET", "/hub/native/login"), ("POST", "/hub/native/login")]


# --- pod roles (LAB_POD_ROLES) ----------------------------------------------------------------

def test_the_roles_default_to_the_dub_pair_and_the_setting_is_read_from_the_environment_then_the_dotenv(tmp_path):
    assert hub.pod_roles({}, []) == {"A": "pod-a", "B": "pod-b"}
    assert hub.pod_roles({"LAB_POD_ROLES": " c:pod-a , D:POD-B "}, []) == {"C": "pod-a", "D": "pod-b"}
    dotenv = tmp_path / ".env"
    dotenv.write_text("LAB_POD_ROLES=C:pod-a,D:pod-b\n", encoding="utf-8")
    assert hub.pod_roles({}, [dotenv]) == {"C": "pod-a", "D": "pod-b"}
    assert hub.pod_roles({"LAB_POD_ROLES": "A:pod-a"}, [dotenv]) == {"A": "pod-a"}


def test_an_account_gets_its_role_as_its_node_and_one_with_no_role_is_refused_not_named_by_its_letter():
    accounts = load_accounts(["C", "D"], environ=SJC_ENV, env_files=[])
    assert (accounts["C"].node, accounts["D"].node) == ("pod-a", "pod-b")
    with pytest.raises(hub.PodRoleError, match="account C fills no pod role"):
        load_accounts(["C"], environ={k: v for k, v in SJC_ENV.items() if k != "LAB_POD_ROLES"}, env_files=[],
                      require_role=True)
    assert load_accounts(["C"], environ={k: v for k, v in SJC_ENV.items() if k != "LAB_POD_ROLES"}, env_files=[])["C"].role is None
    with pytest.raises(hub.PodRoleError):
        hub.node_of("C")
    with pytest.raises(hub.PodRoleError):
        _ = Account("D", "acct-d", TOKEN).node
    assert hub.hub_of("https://sjc.test/") == "sjc.test"
