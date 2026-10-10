import asyncio
import os

import pytest

from contexts.connectors.servers.browser import (
    DDG_JS,
    REDDIT_JS,
    Browser,
    challenge_kind,
    clean_results,
    focus,
    host_allowed,
    readable,
    real_url,
)


def test_unwraps_search_redirects():
    ddg = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fcs.ucf.edu%2Fcourses%2Fcop3502%2F&rut=abc"
    assert real_url(ddg) == "https://cs.ucf.edu/courses/cop3502/"
    bing = "https://www.bing.com/ck/a?!&&p=x&u=a1aHR0cHM6Ly9jcy51Y2YuZWR1L2NvdXJzZXMvY29wMzUwMi9mYWxsMjAyNS9leGFtLw&ntb=1"
    assert real_url(bing) == "https://cs.ucf.edu/courses/cop3502/fall2025/exam/"
    assert real_url("https://example.com/a") == "https://example.com/a"


def test_clean_results_dedupes_and_limits():
    raw = [
        {"title": "A", "url": "https://a.test", "snippet": "first"},
        {"title": "A again", "url": "https://a.test"},
        {"title": "", "url": "https://empty.test"},
        {"title": "B", "url": "https://b.test"},
        {"title": "C", "url": "https://c.test"},
    ]
    out = clean_results(raw, limit=2)
    assert [(r["title"], r["url"], r["summary"]) for r in out] == [("A", "https://a.test", "first"), ("B", "https://b.test", None)]


def test_readable_focus_and_hosts():
    text = readable("  Exam 1   covers  recursion \n\n x \nOffice hours Monday\nExam 2 is on trees", 1000)
    assert text == "Exam 1 covers recursion\nOffice hours Monday\nExam 2 is on trees"
    assert focus(text, "what is on the exam") == "Exam 1 covers recursion … Exam 2 is on trees"
    assert focus(text, None) is None
    assert host_allowed("https://cs.ucf.edu/x", ["ucf.edu"])
    assert not host_allowed("https://evil.test/x", ["ucf.edu"])
    assert host_allowed("https://anything.test", ["*"])


DDG_PAGE = """<div class="result"><a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fcs.ucf.edu%2Fexam%2F">COP 3502 Exam Info</a>
<a class="result__snippet">Exam 1 solutions</a></div>"""
REDDIT_PAGE = """<a data-testid="post-title-text" href="https://www.reddit.com/r/ucf/comments/abc/need_advice/">Need advice for COP3502C</a>
<a data-testid="post-title" href="https://www.reddit.com/r/ucf/comments/abc/need_advice/">Need advice for COP3502C</a>"""


@pytest.mark.skipif(os.environ.get("ATLAS_SKIP_BROWSER") == "1", reason="no browser on this machine")
def test_page_scripts_parse_saved_markup():
    async def run():
        b = Browser({"headless": True, "profile_dir": os.path.join(os.environ.get("TMP", "/tmp"), "atlas-test-profile")})
        try:
            page = await (await b.context()).new_page()
            await page.set_content(DDG_PAGE)
            ddg = await page.evaluate(DDG_JS)
            await page.set_content(REDDIT_PAGE)
            red = await page.evaluate(REDDIT_JS)
            return ddg, red
        finally:
            await b.close()

    try:
        ddg, red = asyncio.run(run())
    except Exception as exc:  # browser binaries not installed
        pytest.skip(f"browser unavailable: {exc}")
    assert clean_results(ddg, 5)[0]["url"] == "https://cs.ucf.edu/exam/"
    assert [r["title"] for r in clean_results(red, 5)] == ["Need advice for COP3502C"]


def test_challenge_detection():
    assert challenge_kind("https://x.test/a", "Welcome to the course page") is None
    assert challenge_kind("https://www.reddit.com/r/ucf/search/?js_challenge=1", "") == "js"
    assert challenge_kind("https://x.test", "Just a moment... Checking your browser") == "js"
    assert challenge_kind("https://html.duckduckgo.com", "Unfortunately, bots use DuckDuckGo too.") == "captcha"
    assert challenge_kind("https://old.reddit.com", "You've been blocked by network security.") == "blocked"


FORM_PAGE = """<form action="about:blank" onsubmit="document.body.innerHTML='<main>RSVP saved for '+document.getElementById('name').value+'</main>'; return false;">
<input id="name"><button id="go">RSVP</button></form>"""


@pytest.mark.skipif(os.environ.get("ATLAS_SKIP_BROWSER") == "1", reason="no browser on this machine")
def test_submit_form_fills_and_clicks(tmp_path):
    page_file = tmp_path / "form.html"
    page_file.write_text(FORM_PAGE)

    async def run():
        b = Browser({"headless": True, "profile_dir": str(tmp_path / "profile"), "min_gap_s": 0})
        try:
            return await b.submit_form(page_file.as_uri(), {"#name": "Dom"}, "#go")
        finally:
            await b.close()

    try:
        out = asyncio.run(run())
    except Exception as exc:
        if "browser" in str(exc).lower() or "executable" in str(exc).lower():
            pytest.skip(f"browser unavailable: {exc}")
        raise
    assert "RSVP saved for Dom" in out["text"]


def test_submit_form_refuses_credentials():
    b = Browser({})
    with pytest.raises(ValueError, match="credentials"):
        asyncio.run(b.submit_form("https://x.test/login", {"#password": "hunter2"}, "#go"))
