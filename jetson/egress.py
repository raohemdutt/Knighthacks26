"""Egress firewall for the Jetson (V6), the offline switch (V5) and the per-connector allowlist and log (H4).

Only the OUTPUT side is filtered (our own chain ATLAS_EGRESS, jumped to from OUTPUT in iptables and ip6tables; the Jetson
kernel has no nftables counters or logging, so this is iptables), so ssh into the box and the phone talking to it keep
working. What may leave:

    loopback, replies to connections that came in            always
    any private / link-local / multicast address (the LAN)   always, so Wi-Fi at any venue works
    the tailnet's own addresses (100.64.0.0/10, fd7a:115c:a1e0::/48), when ATLAS_TAILSCALE=1: always, like the LAN
    NTP (udp 123), DNS to the box's own resolvers             always
    Tailscale's control and relay servers and its WireGuard
    port (udp from 41641), when ATLAS_TAILSCALE=1            online mode only
    the AUP hubs: dub.aupcloud.io plus the aupcloud.io hosts
    named by AUP_HUB_URL_<X> and LAB_HUB_URL,
    each connector's `hosts` from config/connectors.yaml,
    Neon, the ntfy relay, the Ubuntu connectivity check,
    config/egress.yaml extras                                 online mode only: their resolved IP addresses
    everything else                                           logged ("atlas-egress-drop") and rejected

Each allowed group is counted under its own comment, which is how the phone shows bytes per connector. Offline mode (the
Vault policy's `offline` flag, set from the phone or `make offline`) leaves only the always-allowed lines.

    python jetson/egress.py render        # print the rules for the current config (resolves names; changes nothing)
    sudo python jetson/egress.py apply --revert-after 90   # load them; undone in 90 s unless `touch /run/atlas-egress.ok`
    sudo python jetson/egress.py serve    # what the atlas-egress service runs: follow the policy, re-resolve, write
                                          # data/vault/egress.json for GET /vault/egress
    sudo python jetson/egress.py clear    # remove the chain

Names are resolved with the box's own resolver and refreshed every few minutes, so a host that changes address is
picked up; one that answers differently here than for the app can still be dropped (see the drop log).
"""

import argparse
import ipaddress
import json
import os
import re
import socket
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import yaml

CHAIN = "ATLAS_EGRESS"
HUB_HOSTS = ["dub.aupcloud.io"]
# The hub settings the telemetry pod reader and lab/hub.py read: one per AUP account letter, plus the all-accounts one.
HUB_URL_ACCOUNT_ENV = re.compile(r"AUP_HUB_URL_[A-Z]")
HUB_URL_ALL_ACCOUNTS_ENV = "LAB_HUB_URL"
# Only hubs under this domain may join the hub group, so a setting cannot widen the allowlist to an arbitrary host.
HUB_DOMAIN_SUFFIX = ".aupcloud.io"
# NetworkManager asks this every few minutes; blocked, it would mark the Wi-Fi as having no internet.
SYSTEM_HOSTS = ["connectivity-check.ubuntu.com"]
# Tailscale (ATLAS_TAILSCALE=1): phones on the owner's tailnet reach the box's app and ntfy from anywhere.
TAILSCALE_HOSTS = ["controlplane.tailscale.com", "login.tailscale.com", "log.tailscale.io",
                   *(f"derp{n}.tailscale.com" for n in range(1, 29))]
TAILNET = {4: ["100.64.0.0/10"], 6: ["fd7a:115c:a1e0::/48"]}
TAILSCALE_PORT = 41641  # tailscaled's WireGuard source port: direct paths to peers instead of relays
LAN = {4: ["10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16", "224.0.0.0/4", "255.255.255.255/32"],
       6: ["fe80::/10", "fc00::/7", "ff00::/8"]}
REJECT = {4: "icmp-port-unreachable", 6: "icmp6-port-unreachable"}
Resolved = tuple[set[str], set[str]]  # (ipv4, ipv6)


# --- what may leave -------------------------------------------------------------------------------------------------

def read_env(path: Path) -> dict[str, str]:
    env = dict(os.environ)
    if path.is_file():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env.setdefault(key.strip().removeprefix("export "), value.strip().strip('"').strip("'"))
    return env


def _is_private(host: str) -> bool:
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return host in ("localhost",) or host.endswith(".local")
    return ip.is_private or ip.is_loopback or ip.is_link_local


def derp_nodes(run: Callable = subprocess.run) -> list[str]:
    """Every Tailscale relay (DERP) address, from tailscaled's own map: the derpN host names cover only some of them,
    and with the nearby relays blocked the box fell back to one in Bengaluru. [] when tailscale is not running."""
    try:
        out = run(["tailscale", "debug", "derp-map"], capture_output=True, text=True, timeout=10)
        regions = json.loads(out.stdout or "{}").get("Regions") or {}
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    return sorted({ip for r in regions.values() for node in r.get("Nodes") or [] for ip in (node.get("IPv4"), node.get("IPv6"))
                   if ip and ip != "none"})


_warned: set[str] = set()


def _warn_once(message: str) -> None:
    """Report a setting that was left out, once per distinct message (the service re-reads the env every few seconds)."""
    if message not in _warned:
        _warned.add(message)
        print(f"egress: {message}", file=sys.stderr, flush=True)


def hub_hosts(env: dict[str, str]) -> list[str]:
    """HUB_HOSTS, then the aupcloud.io host of every AUP_HUB_URL_<X> and of LAB_HUB_URL, each once, in a fixed order.
    A blank value is ignored; an unparsable one, or one outside aupcloud.io, is left out with a warning."""
    hosts = list(HUB_HOSTS)
    names = sorted(k for k in env if HUB_URL_ACCOUNT_ENV.fullmatch(k)) + [HUB_URL_ALL_ACCOUNTS_ENV]
    for name in names:
        value = (env.get(name) or "").strip()
        if not value:
            continue
        try:
            host = urlparse(value).hostname
        except ValueError as e:
            _warn_once(f"{name} is not a URL ({e}): left out of the hub group")
            continue
        if not host:
            _warn_once(f"{name} has no host name: left out of the hub group")
        elif not host.endswith(HUB_DOMAIN_SUFFIX):
            _warn_once(f"{name} names {host}, which is outside {HUB_DOMAIN_SUFFIX.lstrip('.')}: left out of the hub group")
        elif host not in hosts:
            hosts.append(host)
    return hosts


def host_groups(config_dir: Path, env: dict[str, str], relays: Callable[[], list[str]] = derp_nodes) -> dict[str, list[str]]:
    """Group name -> hostnames. The group name is what the phone shows ("connector:calendar", "hub", "neon")."""
    groups: dict[str, list[str]] = {"hub": hub_hosts(env), "system": list(SYSTEM_HOSTS)}
    connectors = config_dir / "connectors.yaml"
    for name, spec in ((yaml.safe_load(connectors.read_text()) or {}) if connectors.exists() else {}).items():
        if spec and spec.get("hosts"):
            groups[f"connector:{name}"] = [str(h) for h in spec["hosts"]]
    if env.get("ATLAS_TAILSCALE") == "1":
        groups["tailscale"] = list(TAILSCALE_HOSTS)
        if nodes := relays():
            groups["tailscale-relays"] = nodes  # literal addresses, about 90 of each version: not capped
    neon = urlparse(env.get("NEON_DATABASE_URL", "")).hostname
    if neon:
        groups["neon"] = [neon]
    ntfy = urlparse(env.get("NTFY_URL", "")).hostname
    if ntfy and not _is_private(ntfy):
        groups["ntfy"] = [ntfy]
    extras = config_dir / "egress.yaml"
    for name, hosts in (((yaml.safe_load(extras.read_text()) or {}).get("groups") or {}) if extras.exists() else {}).items():
        groups.setdefault(f"extra:{name}", []).extend(str(h) for h in hosts)
    return groups


def resolve(host: str, getaddrinfo: Callable = socket.getaddrinfo) -> Resolved:
    v4: set[str] = set()
    v6: set[str] = set()
    try:
        for family, _, _, _, addr in getaddrinfo(host, None, type=socket.SOCK_STREAM):
            (v4 if family == socket.AF_INET else v6).add(addr[0])
    except OSError:
        pass
    return v4, v6


def resolvers() -> Resolved:
    """The nameservers this box uses, so DNS keeps working even when they are not on a private address."""
    v4: set[str] = set()
    v6: set[str] = set()
    for path in ("/run/systemd/resolve/resolv.conf", "/etc/resolv.conf"):
        try:
            for line in Path(path).read_text().splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[0] == "nameserver":
                    ip = ipaddress.ip_address(parts[1].split("%")[0])
                    (v4 if ip.version == 4 else v6).add(str(ip))
        except (OSError, ValueError):
            continue
    return v4, v6


MAX_PER_GROUP = 64  # addresses kept per group and IP version
LOOKUPS = 3  # rotating DNS answers (CDNs, Canonical's connectivity check) differ per lookup: ask a few times


def resolve_groups(groups: dict[str, list[str]], previous: dict[str, Resolved] | None = None) -> dict[str, Resolved]:
    """Addresses per group. Answers are added to what the group had (a host that rotates through many addresses is
    allowed on all of them, and a DNS blip changes nothing), up to MAX_PER_GROUP of each IP version."""
    out: dict[str, Resolved] = {}
    for group, hosts in groups.items():
        v4: set[str] = set(previous[group][0]) if previous and group in previous else set()
        v6: set[str] = set(previous[group][1]) if previous and group in previous else set()
        for host in hosts:
            try:
                ip = ipaddress.ip_address(host)
                (v4 if ip.version == 4 else v6).add(str(ip))
            except ValueError:
                for _ in range(LOOKUPS):
                    a, b = resolve(host)
                    v4 |= a
                    v6 |= b
        cap = None if group == "tailscale-relays" else MAX_PER_GROUP
        out[group] = (set(sorted(v4)[-cap:]) if cap else v4, set(sorted(v6)[-cap:]) if cap else v6)
    return out


# --- the rules ------------------------------------------------------------------------------------------------------

def _comment(text: str) -> str:
    return f'-m comment --comment "{text}"'


def render(version: int, groups: dict[str, Resolved], dns: Resolved, offline: bool) -> str:
    """The chain as an iptables-restore transaction for IPv4 (version 4) or IPv6 (version 6)."""
    pick = 0 if version == 4 else 1
    lines = ["*filter", f":{CHAIN} - [0:0]",
             f"-A {CHAIN} -o lo -j ACCEPT",
             f"-A {CHAIN} -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT"]
    lines += [f"-A {CHAIN} -d {net} {_comment('lan')} -j ACCEPT" for net in LAN[version]]
    if "tailscale" in groups or "tailscale-relays" in groups:  # traffic to tailnet peers goes over tailscale0 and is ours end to end, like the LAN
        lines += [f"-A {CHAIN} -d {net} {_comment('tailnet')} -j ACCEPT" for net in TAILNET[version]]
    if version == 6:  # neighbour discovery, router adverts and path-MTU messages: IPv6 does not work without them
        lines.append(f"-A {CHAIN} -p ipv6-icmp {_comment('lan')} -j ACCEPT")
    lines.append(f"-A {CHAIN} -p udp --dport 123 {_comment('ntp')} -j ACCEPT")
    for ip in sorted(dns[pick]):
        for proto in ("udp", "tcp"):
            lines.append(f"-A {CHAIN} -d {ip} -p {proto} --dport 53 {_comment('dns')} -j ACCEPT")
    if not offline:
        if "tailscale" in groups or "tailscale-relays" in groups:
            lines.append(f"-A {CHAIN} -p udp --sport {TAILSCALE_PORT} {_comment('tailscale')} -j ACCEPT")
        for group, resolved in sorted(groups.items()):
            lines += [f"-A {CHAIN} -d {ip} {_comment(group)} -j ACCEPT" for ip in sorted(resolved[pick])]
    lines += [
        f'-A {CHAIN} -m limit --limit 10/second --limit-burst 20 -j LOG --log-prefix "atlas-egress-drop " --log-level 6',
        f"-A {CHAIN} -p tcp {_comment('blocked')} -j REJECT --reject-with tcp-reset",
        f"-A {CHAIN} {_comment('blocked')} -j REJECT --reject-with {REJECT[version]}",
        "COMMIT",
        "",
    ]
    return "\n".join(lines)


# --- counters and the drop log --------------------------------------------------------------------------------------

def parse_counters(save_output: str) -> dict[str, dict[str, int]]:
    """Rule comment -> {packets, bytes} from `iptables-save -c`, summed over a group's addresses (and v4 plus v6)."""
    out: dict[str, dict[str, int]] = {}
    for line in save_output.splitlines():
        m = re.match(rf'\[(\d+):(\d+)\] -A {CHAIN} .*--comment "?([^"\s]+)"? ', line + " ")
        if m:
            c = out.setdefault(m.group(3), {"packets": 0, "bytes": 0})
            c["packets"] += int(m.group(1))
            c["bytes"] += int(m.group(2))
    return out


def parse_drops(kernel_log: str, limit: int = 20) -> list[dict]:
    """Recent blocked connections from the kernel log, newest first, one row per destination and port."""
    seen: Counter = Counter()
    last: dict[tuple, str] = {}
    for line in kernel_log.splitlines():
        if "atlas-egress-drop" not in line:
            continue
        fields = dict(re.findall(r"(\w+)=(\S*)", line))
        if "DST" not in fields:
            continue
        key = (fields["DST"], fields.get("PROTO", ""), fields.get("DPT", ""))
        seen[key] += 1
        last[key] = line.split()[0]
    rows = [{"dst": d, "proto": p, "port": port, "count": n, "last": last[(d, p, port)]}
            for (d, p, port), n in seen.items()]
    return sorted(rows, key=lambda r: r["last"], reverse=True)[:limit]


def snapshot(counters: dict[str, dict[str, int]], groups: dict[str, list[str]], drops: list[dict], offline: bool,
             now: datetime | None = None) -> dict:
    """What GET /vault/egress shows."""
    allowed = [{"name": name, "hosts": groups.get(name, []), **counters.get(name, {"packets": 0, "bytes": 0})}
               for name in ["lan", "ntp", "dns", *sorted(groups)] if not (offline and name in groups)]
    blocked = counters.get("blocked", {"packets": 0, "bytes": 0})
    return {
        "enforced": True,
        "mode": "offline" if offline else "online",
        "allowed": allowed,
        "blocked": blocked["packets"],
        "blocked_bytes": blocked["bytes"],
        "recent_blocks": drops,
        "updated": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# --- running it -----------------------------------------------------------------------------------------------------

def ipt(version: int, args: list[str], text: str | None = None) -> str:
    """Run iptables / ip6tables (version 4 / 6), or their -restore / -save when `args` starts with that word."""
    suffix = "" if version == 4 else "6"
    if args[0] in ("restore", "save"):
        cmd = [f"ip{suffix}tables-{args[0]}", *(["-w", "5"] if args[0] == "restore" else []), *args[1:]]
    else:
        cmd = [f"ip{suffix}tables", "-w", "5", *args]
    done = subprocess.run(cmd, input=text, capture_output=True, text=True)
    if done.returncode:
        raise RuntimeError(f"{' '.join(cmd)}: {done.stderr.strip()}")
    return done.stdout


class Egress:
    """Keeps the chain in step with the policy and the config; counters survive a reload."""

    def __init__(self, config_dir: Path, data_dir: Path, env_file: Path, run: Callable = ipt,
                 kernel_log: Callable[[], str] | None = None) -> None:
        self.config_dir, self.data_dir, self.env_file, self.run = config_dir, data_dir, env_file, run
        self.kernel_log = kernel_log or self._kernel_log
        self.groups: dict[str, list[str]] = {}
        self.resolved: dict[str, Resolved] = {}
        self.dns: Resolved = (set(), set())
        self.offline: bool | None = None
        self.base: dict[str, dict[str, int]] = {}  # counters carried over from earlier loads of the chain

    def _kernel_log(self) -> str:
        try:
            return subprocess.run(["journalctl", "-k", "-g", "atlas-egress-drop", "-n", "400", "--no-pager", "-o", "short-iso"],
                                  capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            return ""

    def policy_offline(self) -> bool:
        try:
            return bool(json.loads((self.data_dir / "vault" / "policy.json").read_text()).get("offline"))
        except (OSError, ValueError):
            return False

    def live_counters(self) -> dict[str, dict[str, int]]:
        total = {k: dict(v) for k, v in self.base.items()}
        for version in (4, 6):
            try:
                raw = parse_counters(self.run(version, ["save", "-c", "-t", "filter"]))
            except RuntimeError:
                continue
            for name, c in raw.items():
                t = total.setdefault(name, {"packets": 0, "bytes": 0})
                t["packets"] += c["packets"]
                t["bytes"] += c["bytes"]
        return total

    def load(self, offline: bool, groups: dict[str, list[str]] | None = None, resolved: dict[str, Resolved] | None = None,
             dns: Resolved | None = None) -> None:
        """(Re)define the chain, atomically, and make sure OUTPUT jumps to it. Totals so far are kept in `base`."""
        groups = groups if groups is not None else host_groups(self.config_dir, read_env(self.env_file))
        resolved = resolved if resolved is not None else resolve_groups(groups, self.resolved)
        dns = dns if dns is not None else resolvers()
        texts = {v: render(v, resolved, dns, offline) for v in (4, 6)}
        for v, text in texts.items():
            self.run(v, ["restore", "--test", "--noflush"], text)  # syntax check first: a bad ruleset must never replace a good one
        self.base = self.live_counters()
        for v, text in texts.items():
            self.run(v, ["restore", "--noflush"], text)
            try:
                self.run(v, ["-C", "OUTPUT", "-j", CHAIN])
            except RuntimeError:
                self.run(v, ["-I", "OUTPUT", "1", "-j", CHAIN])
        self.groups, self.resolved, self.dns, self.offline = groups, resolved, dns, offline

    def step(self, resolve_due: bool) -> None:
        want = self.policy_offline()
        groups = host_groups(self.config_dir, read_env(self.env_file))
        if self.offline is None or want != self.offline or groups != self.groups:
            self.load(want, groups)
        elif resolve_due:
            resolved, dns = resolve_groups(self.groups, self.resolved), resolvers()
            if resolved != self.resolved or dns != self.dns:  # a reload zeroes the kernel counters, so only when something moved
                self.load(want, self.groups, resolved, dns)

    def clear(self) -> None:
        for v in (4, 6):
            for args in (["-D", "OUTPUT", "-j", CHAIN], ["-F", CHAIN], ["-X", CHAIN]):
                try:
                    self.run(v, args)
                except RuntimeError:
                    pass

    def unresolved(self) -> list[str]:
        return [g for g, (v4, v6) in self.resolved.items() if not (v4 or v6)]

    def write_snapshot(self) -> dict:
        snap = snapshot(self.live_counters(), self.groups, parse_drops(self.kernel_log()), bool(self.offline))
        out = self.data_dir / "vault" / "egress.json"
        if not out.parent.exists():  # created by root here: hand it to the box's user, who writes policy.json in it
            out.parent.mkdir(parents=True)
            if hasattr(os, "chown") and os.geteuid() == 0:
                owner = self.data_dir.stat()
                os.chown(out.parent, owner.st_uid, owner.st_gid)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(snap))
        os.replace(tmp, out)
        return snap


def serve(egress: Egress, interval: float = 5, resolve_every: float = 300, snapshot_every: float = 10) -> None:
    last_resolve = last_snap = 0.0
    while True:
        now = time.monotonic()
        try:
            # a group with no addresses (the network was not up yet) is retried every 30 s, not every 5 minutes
            due = now - last_resolve >= (30 if egress.unresolved() else resolve_every)
            egress.step(resolve_due=due)
            if due:
                last_resolve = now
            if now - last_snap >= snapshot_every:
                egress.write_snapshot()
                last_snap = now
        except Exception as e:  # keep running: the rules already loaded keep enforcing
            print(f"egress: {e}", file=sys.stderr, flush=True)
        time.sleep(interval)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["render", "apply", "serve", "clear", "status"])
    p.add_argument("--config-dir", type=Path, default=Path("config"))
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--env-file", type=Path, default=Path(".env"))
    p.add_argument("--offline", action="store_true", help="render/apply the offline rules")
    p.add_argument("--revert-after", type=int, default=0, help="apply: remove the chain again after N s unless /run/atlas-egress.ok exists")
    args = p.parse_args(argv)
    eg = Egress(args.config_dir, args.data_dir, args.env_file)

    if args.command == "render":
        groups = host_groups(args.config_dir, read_env(args.env_file))
        resolved, dns = resolve_groups(groups), resolvers()
        print(render(4, resolved, dns, args.offline))
        print(render(6, resolved, dns, args.offline))
    elif args.command == "apply":
        confirm = Path("/run/atlas-egress.ok")
        confirm.unlink(missing_ok=True)
        eg.load(args.offline)
        print(f"loaded ({'offline' if args.offline else 'online'}); groups: {', '.join(sorted(eg.groups))}", flush=True)
        if args.revert_after:
            print(f"will be removed in {args.revert_after} s unless you run: touch {confirm}", flush=True)
            deadline = time.monotonic() + args.revert_after
            while time.monotonic() < deadline and not confirm.exists():
                time.sleep(1)
            if not confirm.exists():
                eg.clear()
                print("not confirmed: removed")
                return 1
            print("confirmed: staying")
    elif args.command == "clear":
        eg.clear()
        print("removed")
    elif args.command == "status":
        out = args.data_dir / "vault" / "egress.json"
        print(out.read_text() if out.exists() else "no snapshot yet (is atlas-egress running?)")
    else:
        serve(eg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
