import httpx
import pytest

from contexts.demo import kit
from contexts.demo.__main__ import main


def box(*, app=True, manifest=True, ntfy=True, topic=True, failing=()):
    """A fake box: gateway on :8000, ntfy on :8090. Records every request so tests can prove nothing is sent."""
    seen: list[httpx.Request] = []

    def handle(r: httpx.Request) -> httpx.Response:
        seen.append(r)
        path = r.url.path
        if r.url.port == 8090:
            if not ntfy:
                raise httpx.ConnectError("refused", request=r)
            if path == "/v1/health":
                return httpx.Response(200, json={"healthy": True})
            if path == "/atlas/json":
                return httpx.Response(200 if topic else 403, text="")
            return httpx.Response(404)
        if path == "/":
            return httpx.Response(200 if app else 404, text="<!DOCTYPE html><html><body>Atlas</body></html>")
        if path == "/manifest.webmanifest":
            return httpx.Response(200, json={"name": "Atlas"}) if manifest else httpx.Response(404)
        if path == "/api/preflight":
            return httpx.Response(200, json={"checks": [{"name": n, "ok": n not in failing, "detail": "", "fix": ""}
                                                        for n in ("phone link", "ntfy", "cameras")]})
        return httpx.Response(404)

    return httpx.Client(transport=httpx.MockTransport(handle)), seen


def by_name(checks):
    return {c.name: c for c in checks}


def test_the_checklist_has_every_item_of_the_done_when():
    text = kit.checklist()
    for word in ("keychain", "package", "door zone", "mounts", "Power strip", "USB cables", "hotspot",
                 "2 charged demo phones", "QR to the Devpost"):
        assert word.lower() in text.lower(), word
    assert sum(it.required for it in kit.KIT) == 9  # D6's nine items; the rest are Demo Plan extras


def test_a_ready_box_passes_and_only_reads(tmp_path):
    pytest.importorskip("qrcode")
    client, seen = box()
    checks = kit.software_checks("https://box:8000", None, "atlas", "https://devpost.com/software/atlas", tmp_path, client)
    assert all(c.ok for c in checks), kit.report(checks)
    assert {r.method for r in seen} == {"GET"}  # nothing published to ntfy, nothing changed on the box
    assert any(r.url.port == 8090 for r in seen)  # ntfy found beside the gateway
    card = (tmp_path / "devpost-qr.html").read_text(encoding="utf-8")
    assert "https://devpost.com/software/atlas" in card and "<svg" in card
    assert (tmp_path / "devpost-qr.svg").read_text(encoding="utf-8").lstrip().startswith("<?xml")


def test_each_missing_piece_fails_with_a_fix(tmp_path):
    client, _ = box(app=False, manifest=False, topic=False, failing=("cameras",))
    checks = by_name(kit.software_checks("http://box:8000", None, "atlas", None, tmp_path, client))
    for name in ("phone app served", "app installable (manifest)", "https for install and mic", "alert topic readable",
                 "box preflight (D9)", "Devpost link"):
        assert not checks[name].ok and checks[name].fix, name
    assert "cameras" in checks["box preflight (D9)"].detail
    assert checks["ntfy answers"].ok
    assert "Devpost QR printed" not in checks  # no link, no card
    assert not (tmp_path / "devpost-qr.html").exists()


def test_ntfy_down_and_a_wrong_link(tmp_path):
    client, _ = box(ntfy=False)
    checks = by_name(kit.software_checks("http://box:8000", "http://box:8090", "atlas", "https://example.com/x", tmp_path, client))
    assert not checks["ntfy answers"].ok and "systemctl" in checks["ntfy answers"].fix
    assert not checks["Devpost link"].ok


def test_the_cli_lists_the_kit(capsys):
    assert main(["kit", "--list"]) == 0
    out = capsys.readouterr().out
    assert "[ ] Bright keychain" in out and "Software" not in out
