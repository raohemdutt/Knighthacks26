"""The Jetson firewall's hub group follows the AUP hub settings the telemetry pod reader reads (T3)."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("egress", Path(__file__).parent.parent / "jetson" / "egress.py")
egress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(egress)


@pytest.fixture
def config(tmp_path):
    return tmp_path


def test_hub_group_admits_env_named_aupcloud_hosts_once_after_the_default(config):
    env = {
        "AUP_HUB_URL_C": "https://sjc.aupcloud.io",
        "AUP_HUB_URL_D": "https://sjc.aupcloud.io/",
        "LAB_HUB_URL": "https://dub.aupcloud.io",
        "AUP_HUB_URL_E": "https://evil.example.org",
    }
    groups = egress.host_groups(config, env)
    assert groups["hub"] == ["dub.aupcloud.io", "sjc.aupcloud.io"]


def test_env_cannot_widen_the_allowlist_to_a_host_outside_aupcloud(config):
    env = {
        "AUP_HUB_URL_C": "https://sjc.aupcloud.io",
        "AUP_HUB_URL_E": "https://evil.example.org",
        "LAB_HUB_URL": "https://aupcloud.io.evil.example.org",
    }
    groups = egress.host_groups(config, env)
    every_host = [h for hosts in groups.values() for h in hosts]
    assert "sjc.aupcloud.io" in groups["hub"]
    assert "evil.example.org" not in every_host
    assert "aupcloud.io.evil.example.org" not in every_host


def test_hub_group_is_the_default_hub_alone_with_an_empty_env(config):
    assert egress.host_groups(config, {})["hub"] == ["dub.aupcloud.io"]
