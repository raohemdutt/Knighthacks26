from contexts.gateway.bootstrap import circle_from_env

BASE = {"member_id": None, "circle_id": None, "share_presence": False, "share_summaries": False,
        "names": {"dom": "Dominick", "hemdutt": "Hemdutt"}}


def test_env_joins_a_circle_without_touching_the_rest():
    out = circle_from_env(BASE, {"ATLAS_CIRCLE_MEMBER": "hemdutt", "ATLAS_CIRCLE_ID": "kh-demo-1", "ATLAS_CIRCLE_SHARE": "1"})
    assert (out["member_id"], out["circle_id"], out["share_presence"], out["share_summaries"]) == ("hemdutt", "kh-demo-1", True, True)
    assert out["names"] == BASE["names"]
    assert BASE["member_id"] is None  # the config itself is not changed


def test_no_env_means_the_config_as_it_is():
    assert circle_from_env(BASE, {}) == BASE
    assert circle_from_env({}, {}) == {}
