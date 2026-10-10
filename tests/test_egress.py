import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("egress", Path(__file__).parent.parent / "jetson" / "egress.py")
egress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(egress)

CONNECTORS = """
site: {module: x, hosts: [knighthacks.org], reads: {}}
calendar: {module: y, hosts: [outlook.office365.com, events.ucf.edu]}
fake: {module: z}
"""


@pytest.fixture
def config(tmp_path):
    (tmp_path / "connectors.yaml").write_text(CONNECTORS)
    return tmp_path


def test_groups_come_from_the_connectors_the_hub_neon_and_a_remote_ntfy_only(config):
    env = {"NEON_DATABASE_URL": "postgresql://u:secret@ep-x.neon.tech/db", "NTFY_URL": "https://ntfy.example.org"}
    groups = egress.host_groups(config, env)
    assert groups == {"hub": ["dub.aupcloud.io"], "system": ["connectivity-check.ubuntu.com"], "connector:site": ["knighthacks.org"],
                      "connector:calendar": ["outlook.office365.com", "events.ucf.edu"],
                      "neon": ["ep-x.neon.tech"], "ntfy": ["ntfy.example.org"]}
    assert "secret" not in json.dumps(groups)  # only the host is kept, never the password
    lan_ntfy = egress.host_groups(config, {"NTFY_URL": "http://192.168.1.50:8090"})
    assert "ntfy" not in lan_ntfy  # a relay on the LAN needs no hole
    (config / "egress.yaml").write_text("groups: {github: [github.com]}\n")
    assert egress.host_groups(config, {})["extra:github"] == ["github.com"]


GROUPS = {"hub": ({"104.18.0.205"}, {"2606:4700::6812:cd"}), "connector:site": ({"1.2.3.4", "1.2.3.5"}, set())}
DNS = ({"192.168.1.1"}, {"2603:900b::1"})


def test_the_online_rules_open_each_address_of_each_group_and_label_it():
    v4 = egress.render(4, GROUPS, DNS, offline=False)
    assert v4.startswith("*filter\n:ATLAS_EGRESS - [0:0]\n") and v4.rstrip().endswith("COMMIT")
    assert '-A ATLAS_EGRESS -d 104.18.0.205 -m comment --comment "hub" -j ACCEPT' in v4
    assert '-A ATLAS_EGRESS -d 1.2.3.4 -m comment --comment "connector:site" -j ACCEPT' in v4
    assert "2606:4700" not in v4  # IPv6 addresses belong in the ip6tables rules
    assert '-A ATLAS_EGRESS -d 192.168.0.0/16 -m comment --comment "lan" -j ACCEPT' in v4
    assert '-A ATLAS_EGRESS -p udp --dport 123 -m comment --comment "ntp" -j ACCEPT' in v4
    assert '-d 192.168.1.1 -p udp --dport 53 -m comment --comment "dns" -j ACCEPT' in v4
    assert "-m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT" in v4
    v6 = egress.render(6, GROUPS, DNS, offline=False)
    assert '-d 2606:4700::6812:cd -m comment --comment "hub" -j ACCEPT' in v6 and "1.2.3.4" not in v6
    assert "-p ipv6-icmp" in v6 and "-p ipv6-icmp" not in v4
    assert "-d fe80::/10" in v6 and "icmp6-port-unreachable" in v6 and "icmp-port-unreachable" not in v6
    # logging and the reject come after every allow rule
    assert v4.rindex('--comment "connector:site"') < v4.index("atlas-egress-drop") < v4.index('--comment "blocked"')
    assert "--reject-with tcp-reset" in v4 and "-j REJECT --reject-with icmp-port-unreachable" in v4


def test_offline_keeps_only_the_lan_ntp_and_dns():
    v4 = egress.render(4, GROUPS, DNS, offline=True)
    assert '"hub"' not in v4 and "1.2.3.4" not in v4 and "104.18.0.205" not in v4
    assert '"lan"' in v4 and '"ntp"' in v4 and '"dns"' in v4 and '"blocked"' in v4


SAVE = """\
*filter
:ATLAS_EGRESS - [0:0]
[10:600] -A ATLAS_EGRESS -o lo -j ACCEPT
[5:1000] -A ATLAS_EGRESS -d 104.18.0.205/32 -m comment --comment "hub" -j ACCEPT
[2:300] -A ATLAS_EGRESS -d 104.18.0.206/32 -m comment --comment "hub" -j ACCEPT
[3:240] -A ATLAS_EGRESS -d 1.2.3.4/32 -m comment --comment "connector:site" -j ACCEPT
[7:420] -A ATLAS_EGRESS -p tcp -m comment --comment "blocked" -j REJECT --reject-with tcp-reset
[0:0] -A OUTPUT -j ATLAS_EGRESS
COMMIT
"""


def test_counters_are_summed_per_group_over_its_addresses():
    assert egress.parse_counters(SAVE) == {"hub": {"packets": 7, "bytes": 1300}, "connector:site": {"packets": 3, "bytes": 240},
                                           "blocked": {"packets": 7, "bytes": 420}}


KERNEL = """\
2026-10-10T07:30:01-0400 ubuntu kernel: atlas-egress-drop IN= OUT=wlP1p1s0 SRC=192.168.1.126 DST=93.184.216.34 PROTO=TCP SPT=40000 DPT=443
2026-10-10T07:30:05-0400 ubuntu kernel: atlas-egress-drop IN= OUT=wlP1p1s0 SRC=192.168.1.126 DST=93.184.216.34 PROTO=TCP SPT=40002 DPT=443
2026-10-10T07:31:00-0400 ubuntu kernel: atlas-egress-drop IN= OUT=wlP1p1s0 SRC=192.168.1.126 DST=8.8.4.4 PROTO=UDP SPT=5 DPT=53
2026-10-10T07:31:01-0400 ubuntu kernel: something else entirely
"""


def test_blocked_connections_are_grouped_newest_first():
    rows = egress.parse_drops(KERNEL)
    assert [(r["dst"], r["port"], r["count"]) for r in rows] == [("8.8.4.4", "53", 1), ("93.184.216.34", "443", 2)]


def test_snapshot_lists_allowed_groups_with_bytes_and_the_blocked_count():
    counters = egress.parse_counters(SAVE)
    snap = egress.snapshot(counters, {"hub": ["dub.aupcloud.io"]}, [], offline=False)
    assert snap["enforced"] is True and snap["mode"] == "online" and snap["blocked"] == 7
    hub = next(a for a in snap["allowed"] if a["name"] == "hub")
    assert (hub["bytes"], hub["hosts"]) == (1300, ["dub.aupcloud.io"])
    off = egress.snapshot(counters, {"hub": ["dub.aupcloud.io"]}, [], offline=True)
    assert off["mode"] == "offline" and not any(a["name"] == "hub" for a in off["allowed"])


class FakeTables:
    """Records what would be run; `save` answers like iptables-save once something has been restored."""

    def __init__(self):
        self.calls, self.loaded, self.fail_test = [], False, False

    def __call__(self, version, args, text=None):
        self.calls.append((version, args, text))
        if args[:2] == ["restore", "--test"] and self.fail_test:
            raise RuntimeError("syntax error")
        if args == ["restore", "--noflush"]:
            self.loaded = True
        if args[0] == "-C":
            raise RuntimeError("no such rule")  # so the jump gets inserted
        if args[0] == "save":
            return SAVE if self.loaded else ""
        return ""


def make(tmp_path, config, monkeypatch, offline=False):
    monkeypatch.setattr(egress, "resolve", lambda host, *a: ({"1.2.3.4"}, set()))
    monkeypatch.setattr(egress, "resolvers", lambda: ({"192.168.1.1"}, set()))
    data = tmp_path / "data"
    (data / "vault").mkdir(parents=True)
    (data / "vault" / "policy.json").write_text(json.dumps({"offline": offline}))
    tables = FakeTables()
    return egress.Egress(config, data, tmp_path / "none.env", run=tables, kernel_log=lambda: KERNEL), tables, data


def test_a_bad_ruleset_is_rejected_before_it_replaces_the_good_one(tmp_path, config, monkeypatch):
    eg, tables, _ = make(tmp_path, config, monkeypatch)
    tables.fail_test = True
    with pytest.raises(RuntimeError):
        eg.load(False)
    assert not tables.loaded and not any(a[0] == "-I" for _, a, _ in tables.calls)  # nothing was loaded


def test_loading_adds_the_jump_from_output_for_both_ip_versions(tmp_path, config, monkeypatch):
    eg, tables, _ = make(tmp_path, config, monkeypatch)
    eg.load(False)
    jumps = [(v, a) for v, a, _ in tables.calls if a[0] == "-I"]
    assert jumps == [(4, ["-I", "OUTPUT", "1", "-j", "ATLAS_EGRESS"]), (6, ["-I", "OUTPUT", "1", "-j", "ATLAS_EGRESS"])]


def test_the_policy_flag_switches_the_mode_and_totals_survive_the_reload(tmp_path, config, monkeypatch):
    eg, tables, data = make(tmp_path, config, monkeypatch)
    eg.step(resolve_due=False)
    assert eg.offline is False and tables.loaded
    n = len(tables.calls)
    eg.step(resolve_due=False)
    assert not any(a == ["restore", "--noflush"] for _, a, _ in tables.calls[n:])  # nothing changed: nothing reloaded
    (data / "vault" / "policy.json").write_text(json.dumps({"offline": True}))
    eg.step(resolve_due=False)
    assert eg.offline is True
    assert eg.base["hub"]["bytes"] == 2 * 1300  # counters of the chain that was replaced (v4 + v6 saves) are carried over
    snap = eg.write_snapshot()
    assert snap["mode"] == "offline" and json.loads((data / "vault" / "egress.json").read_text())["enforced"] is True


def test_a_resolver_blip_keeps_the_addresses_a_group_already_had(tmp_path, config, monkeypatch):
    eg, tables, _ = make(tmp_path, config, monkeypatch)
    eg.step(resolve_due=False)
    monkeypatch.setattr(egress, "resolve", lambda host, *a: (set(), set()))
    eg.step(resolve_due=True)
    assert eg.resolved["hub"] == ({"1.2.3.4"}, set())


def test_new_addresses_reload_the_chain_only_when_something_changed(tmp_path, config, monkeypatch):
    eg, tables, _ = make(tmp_path, config, monkeypatch)
    eg.step(resolve_due=False)
    loads = lambda: sum(a == ["restore", "--noflush"] for _, a, _ in tables.calls)  # noqa: E731
    before = loads()
    eg.step(resolve_due=True)  # same answers
    assert loads() == before
    monkeypatch.setattr(egress, "resolve", lambda host, *a: ({"9.9.9.9"}, set()))
    eg.step(resolve_due=True)
    assert loads() == before + 2  # v4 and v6 rules


def test_clear_removes_the_jump_the_rules_and_the_chain(tmp_path, config, monkeypatch):
    eg, tables, _ = make(tmp_path, config, monkeypatch)
    eg.clear()
    assert [a for v, a, _ in tables.calls if v == 4] == [["-D", "OUTPUT", "-j", "ATLAS_EGRESS"], ["-F", "ATLAS_EGRESS"], ["-X", "ATLAS_EGRESS"]]


def test_a_host_that_rotates_through_many_addresses_gets_all_of_them_over_time(tmp_path, config, monkeypatch):
    answers = iter([({"1.1.1.1"}, set()), ({"1.1.1.2"}, set()), ({"1.1.1.3"}, set())] * 5)
    eg, tables, _ = make(tmp_path, config, monkeypatch)
    monkeypatch.setattr(egress, "resolve", lambda host, *a: next(answers))
    first = egress.resolve_groups({"hub": ["x.example"]})
    assert first["hub"][0] == {"1.1.1.1", "1.1.1.2", "1.1.1.3"}  # three lookups, three different answers
    monkeypatch.setattr(egress, "resolve", lambda host, *a: ({"9.9.9.9"}, set()))
    second = egress.resolve_groups({"hub": ["x.example"]}, first)
    assert second["hub"][0] == {"1.1.1.1", "1.1.1.2", "1.1.1.3", "9.9.9.9"}  # kept, not replaced
    monkeypatch.setattr(egress, "resolve", lambda host, *a: (set(), set()))
    assert egress.resolve_groups({"hub": ["x.example"]}, second)["hub"][0] == second["hub"][0]  # a blip changes nothing


def test_tailscale_opens_its_servers_its_wireguard_port_and_the_tailnet_only_when_switched_on(config):
    assert "tailscale" not in egress.host_groups(config, {})
    groups = egress.host_groups(config, {"ATLAS_TAILSCALE": "1"}, relays=lambda: [])
    assert "controlplane.tailscale.com" in groups["tailscale"] and "derp1.tailscale.com" in groups["tailscale"]
    resolved = {"tailscale": ({"192.0.2.10"}, set())}
    online = egress.render(4, resolved, (set(), set()), offline=False)
    assert '-d 100.64.0.0/10 -m comment --comment "tailnet" -j ACCEPT' in online
    assert '-p udp --sport 41641 -m comment --comment "tailscale" -j ACCEPT' in online
    assert '-d 192.0.2.10 -m comment --comment "tailscale" -j ACCEPT' in online
    offline = egress.render(4, resolved, (set(), set()), offline=True)
    assert "100.64.0.0/10" in offline and "41641" not in offline and "192.0.2.10" not in offline
    assert "fd7a:115c:a1e0::/48" in egress.render(6, resolved, (set(), set()), offline=False)
    assert "100.64.0.0/10" not in egress.render(4, {}, (set(), set()), offline=False)  # off: no tailnet line


def test_every_tailscale_relay_address_comes_from_the_derp_map_uncapped(config):
    """Only some relays answer to derpN.tailscale.com; with the US ones blocked the Jetson relayed via Bengaluru."""
    class Done:
        stdout = json.dumps({"Regions": {str(r): {"Nodes": [{"IPv4": f"198.51.{r}.{n}", "IPv6": f"2001:db8:{r}::{n}"}
                                                            for n in range(1, 10)]} for r in range(1, 12)}})

    nodes = egress.derp_nodes(run=lambda *a, **k: Done())
    assert len(nodes) == 2 * 99
    groups = egress.host_groups(config, {"ATLAS_TAILSCALE": "1"}, relays=lambda: nodes)
    resolved = egress.resolve_groups({"tailscale-relays": groups["tailscale-relays"]})
    assert len(resolved["tailscale-relays"][0]) == 99 > egress.MAX_PER_GROUP  # not cut to the cap
    assert egress.derp_nodes(run=lambda *a, **k: (_ for _ in ()).throw(OSError("no tailscale"))) == []


def test_counting_rules_come_before_the_established_accept_and_count_each_address_once():
    shared = {"hub": ({"104.18.0.205"}, set()), "neon": ({"104.18.0.205", "3.3.3.3"}, set())}
    v4 = egress.render(4, shared, DNS, offline=False)
    established = v4.index("--ctstate ESTABLISHED")
    assert v4.index('--comment "acct:hub"') < established and v4.index('--comment "acct:lan"') < established
    assert '-A ATLAS_EGRESS -d 104.18.0.205 -m comment --comment "acct:hub"\n' in v4  # no target: count and fall through
    assert v4.count("-d 104.18.0.205 -m comment --comment \"acct:") == 1  # shared address: counted for hub only
    assert '-d 3.3.3.3 -m comment --comment "acct:neon"' in v4
    offline = egress.render(4, shared, DNS, offline=True)
    assert "acct:hub" not in offline and "acct:lan" in offline


def test_snapshot_reports_every_byte_from_the_counting_rule_and_connections_from_the_accept():
    counters = egress.parse_counters("""\
[900:1200000] -A ATLAS_EGRESS -d 104.18.0.205/32 -m comment --comment "acct:hub"
[3:180] -A ATLAS_EGRESS -d 104.18.0.205/32 -m comment --comment "hub" -j ACCEPT
[40:9000] -A ATLAS_EGRESS -d 100.64.0.0/10 -m comment --comment "acct:tailnet"
""")
    snap = egress.snapshot(counters, {"hub": ["dub.aupcloud.io"]}, [], offline=False)
    hub = next(a for a in snap["allowed"] if a["name"] == "hub")
    assert (hub["bytes"], hub["packets"], hub["connections"]) == (1200000, 900, 3)
    assert next(a for a in snap["allowed"] if a["name"] == "tailnet")["bytes"] == 9000
    old = egress.snapshot(egress.parse_counters(SAVE), {"hub": []}, [], offline=False)  # rules from before this change
    assert next(a for a in old["allowed"] if a["name"] == "hub")["bytes"] == 1300
    assert not any(a["name"] == "tailnet" for a in old["allowed"])
