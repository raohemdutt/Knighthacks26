from contexts.connectors.servers.site import check, summarize_change, visible_text

V1 = "<html><head><title>x</title><script>var a=1</script></head><body><h1>Schedule</h1><p>Opening 7 PM</p></body></html>"
V2 = V1.replace("Opening 7 PM", "Opening 8 PM")
PAGES = [{"title": "Knight Hacks schedule", "url": "https://example.test/schedule"}]


def test_visible_text_skips_scripts_and_head():
    assert visible_text(V1) == ["Schedule", "Opening 7 PM"]


def test_first_fetch_is_baseline_then_changes_are_items(tmp_path):
    pages = {"html": V1}
    fetch = lambda url: pages["html"]  # noqa: E731
    assert check(PAGES, tmp_path, fetch) == []
    assert check(PAGES, tmp_path, fetch) == []
    pages["html"] = V2
    [item] = check(PAGES, tmp_path, fetch)
    assert item["title"] == "Knight Hacks schedule changed"
    assert item["summary"].startswith("Opening 8 PM")
    assert item["url"] == "https://example.test/schedule"


def test_summary_counts_extra_changes():
    assert summarize_change(["a"], ["a"]) is None
    assert summarize_change(["a", "b"], ["c", "d"]) == "c (+3 more changes)"
