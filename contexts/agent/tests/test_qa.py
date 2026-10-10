import asyncio
from datetime import UTC, datetime, timedelta

from contexts.agent.application.memory_port import FakeMemory
from contexts.agent.application.qa import NO_EVIDENCE, QA
from contracts.experience import QuestionAsked
from contracts.memory import Episode, Sighting
from contracts.types import ChatResult, ToolCall
from runtime import FakeRuntime

NOW = datetime(2026, 10, 11, 9, 0, tzinfo=UTC)


def ep(id, hours_ago, captions, camera="cam-desk", labels=("person",)):
    start = NOW - timedelta(hours=hours_ago)
    return Episode(
        id=id,
        camera_id=camera,
        start=start,
        end=start + timedelta(minutes=5),
        labels=list(labels),
        captions=captions,
        keyframe_refs=[f"frames:{camera}:{id}"],
    )


MEMORY = FakeMemory(
    episodes=[
        ep("e1", 8, ["A person packs a laptop and leaves the table."]),
        ep("e2", 3, ["The table is empty, lights dim."]),
        ep("e3", 0.5, ["A courier drops a box at the door."], camera="cam-door", labels=("person", "package")),
        ep("e4", 30, ["Old episode from two days ago."]),
    ],
    sightings=[
        Sighting(label="keys", camera_id="cam-desk", zone_id="desk", ts=NOW - timedelta(hours=2), frame_ref="frames:cam-desk:k1", position="next to the mug"),
    ],
)


def qa(*replies):
    rt = FakeRuntime(chat_replies=[ChatResult(content=r, model_version="v1") if isinstance(r, str) else r for r in replies])
    return QA(rt, MEMORY, clock=lambda: NOW), rt


def evidence_prompt(rt):
    return rt.calls[-1][1][-1].content


def test_where_are_my_keys_prefetches_last_seen():
    q, rt = qa("Your keys are on the desk next to the mug.")
    a = asyncio.run(q.ask("Where did I leave my keys?"))
    assert a.citations[0].frame_ref == "frames:cam-desk:k1"
    assert "next to the mug" in evidence_prompt(rt)
    assert a.model_version == "v1"


def test_overnight_uses_hour_window_and_skips_old_episodes():
    q, rt = qa("Someone left, then it was quiet.")
    a = asyncio.run(q.ask("What happened at our table overnight?"))
    prompt = evidence_prompt(rt)
    assert "packs a laptop" in prompt and "two days ago" not in prompt
    assert {c.episode_id for c in a.citations} == {"e1", "e2", "e3"}


def test_last_hour_door():
    q, rt = qa("Yes, a courier dropped a box.")
    a = asyncio.run(q.ask("Did anyone come to the door in the last hour?"))
    assert [c.episode_id for c in a.citations] == ["e3"]


def test_no_evidence_does_not_call_the_model():
    q, rt = qa()
    a = asyncio.run(q.ask("Where is my wallet?"))
    assert a.text == NO_EVIDENCE
    assert not any(name == "chat" for name, _ in rt.calls)


def test_open_question_runs_the_tool_loop():
    call = ChatResult(content="", tool_calls=[ToolCall(id="c1", name="search_memory", arguments={"q": "courier box"})], model_version="v1")
    q, rt = qa(call, "A courier dropped a box at the door.")
    a = asyncio.run(q.ask("Anything delivered?"))
    assert a.text == "A courier dropped a box at the door."
    assert a.citations[0].episode_id == "e3"
    tool_msg = rt.calls[-1][1][-1]
    assert tool_msg.role == "tool" and "courier" in tool_msg.content


def test_tool_loop_without_evidence_says_so():
    q, _ = qa("Probably nothing happened.")
    assert asyncio.run(q.ask("Anything weird?")).text == NO_EVIDENCE


def test_on_question_emits_answer_event():
    q, _ = qa("On the desk.")
    out = asyncio.run(q.on_question(QuestionAsked(question_id="q1", text="where are my keys")))
    assert (out.question_id, out.answer, out.model_version) == ("q1", "On the desk.", "v1")


def test_overnight_follows_the_sleep_schedule():
    from datetime import time
    from zoneinfo import ZoneInfo

    from contexts.agent.application.qa import SleepSchedule

    ny = ZoneInfo("America/New_York")
    default = SleepSchedule.load("config/agent.yaml")
    after_waking = datetime(2026, 10, 11, 9, 30, tzinfo=ny)
    assert default.last_night(after_waking) == (datetime(2026, 10, 10, 20, tzinfo=ny), datetime(2026, 10, 11, 8, tzinfo=ny))
    still_asleep = datetime(2026, 10, 11, 3, 0, tzinfo=ny)
    assert default.last_night(still_asleep) == (datetime(2026, 10, 10, 20, tzinfo=ny), still_asleep)
    night_owl = SleepSchedule(bedtime=time(1), wake=time(9), tz=ny)
    assert night_owl.last_night(after_waking) == (datetime(2026, 10, 11, 1, tzinfo=ny), datetime(2026, 10, 11, 9, tzinfo=ny))


def test_web_tools_cite_links():
    from contexts.agent.application.web_port import FakeWeb

    web = FakeWeb(
        results=[{"title": "COP 3502 Exam Info", "url": "https://cs.ucf.edu/exam/", "summary": "Exam 1 covers recursion"}],
        posts=[{"title": "Exam 1 tips", "url": "https://reddit.com/r/ucf/comments/x/"}],
        pages={"https://blocked.test/": {"url": "https://blocked.test/", "blocked": "captcha"}},
    )
    calls = ChatResult(
        content="",
        tool_calls=[
            ToolCall(id="1", name="web_search", arguments={"query": "COP 3502 exam"}),
            ToolCall(id="2", name="reddit_search", arguments={"query": "COP 3502 exam"}),
            ToolCall(id="3", name="read_page", arguments={"url": "https://blocked.test/"}),
        ],
        model_version="v1",
    )
    rt = FakeRuntime(chat_replies=[calls, ChatResult(content="Exam 1 covers recursion.", model_version="v1")])
    q = QA(rt, MEMORY, clock=lambda: NOW, web=web)
    a = asyncio.run(q.ask("What's on the COP 3502 exam?"))
    assert a.text == "Exam 1 covers recursion."
    assert [c.url for c in a.citations] == ["https://cs.ucf.edu/exam/", "https://reddit.com/r/ucf/comments/x/", "https://blocked.test/"]
    assert ("reddit", "ucf", "COP 3502 exam") in web.calls
    tool_results = [m.content for m in rt.calls[-1][1] if m.role == "tool"]
    assert "bot check" in tool_results[2]
    assert any(t["function"]["name"] == "web_search" for t in q.tools)


def test_without_web_the_model_only_sees_memory_tools():
    q, _ = qa()
    assert {t["function"]["name"] for t in q.tools} == {"search_memory", "last_seen", "episodes_between", "item_history",
                                                         "upcoming"}


def test_get_keyframe_tool_captions_a_stored_frame():
    call = ChatResult(content="", tool_calls=[ToolCall(id="k", name="get_keyframe", arguments={"frame_ref": "frames:cam-desk:e02", "question": "Are there keys?"})], model_version="v1")
    rt = FakeRuntime(chat_replies=[call, ChatResult(content="Yes, keys by the mug.", model_version="v1")], caption_text="Keys next to a mug.")
    q = QA(rt, MEMORY, clock=lambda: NOW, load_keyframe=lambda ref: b"jpeg:" + ref.encode())
    a = asyncio.run(q.ask("Is anything on the desk?"))
    assert a.text == "Yes, keys by the mug." and a.citations[0].frame_ref == "frames:cam-desk:e02"
    assert ("caption", "Are there keys?") in rt.calls


def test_due_this_week_answers_from_connector_items_with_links():
    from contracts.memory import Item

    sunday_10am = datetime(2026, 10, 11, 14, 0, tzinfo=UTC)  # Sun 10:00 AM in New York
    items = [
        Item(connector="canvas", kind="assignment", external_id="a", title="HW 4", when=sunday_10am + timedelta(days=2), url="https://canvas/a"),
        Item(connector="canvas", kind="assignment", external_id="b", title="Essay", when=sunday_10am + timedelta(days=9), url="https://canvas/b"),
        Item(connector="calendar", kind="calendar_event", external_id="c", title="Judging", when=sunday_10am + timedelta(hours=1)),
    ]
    rt = FakeRuntime(chat_replies=[ChatResult(content="HW 4 is due Tuesday.", model_version="v1")])
    q = QA(rt, FakeMemory(items=items), clock=lambda: sunday_10am)
    a = asyncio.run(q.ask("What is due this week?"))
    assert [c.url for c in a.citations] == ["https://canvas/a"]  # the essay is next week; judging isn't "due"
    cal = asyncio.run(QA(FakeRuntime(), FakeMemory(items=items), clock=lambda: sunday_10am).ask("What's on my calendar today?"))
    assert [c.title for c in cal.citations] == ["Judging"]
    assert asyncio.run(QA(FakeRuntime(), FakeMemory(), clock=lambda: sunday_10am).ask("Anything due tomorrow?")).text == NO_EVIDENCE


def test_search_hits_for_connector_items_cite_their_link():
    from contracts.memory import SearchHit

    class ItemSearch(FakeMemory):
        def search(self, q, limit=5):
            return [SearchHit(episode_id="item:canvas:a", camera_id="", ts=NOW, text="HW 4 covers recursion", score=1.0,
                              kind="assignment", url="https://canvas/a", title="HW 4")]

    call = ChatResult(content="", tool_calls=[ToolCall(id="1", name="search_memory", arguments={"q": "recursion"})], model_version="v1")
    rt = FakeRuntime(chat_replies=[call, ChatResult(content="HW 4.", model_version="v1")])
    a = asyncio.run(QA(rt, ItemSearch(), clock=lambda: NOW).ask("Anything about recursion?"))
    assert a.citations[0].url == "https://canvas/a" and a.citations[0].episode_id is None


def test_where_is_ignores_filler_words_around_the_item():
    """ "Where is the package now?" asked about "package now" and found nothing."""
    from contexts.agent.application.qa import item_name

    for question in ("Where are my keys now?", "Where's my keys right now?", "where are the keys at the moment",
                     "Where is a keys currently?!", "Where did I leave my keys today?", "where are the keys again?",
                     "Where are my keys?"):
        q, rt = qa("On the desk.")
        a = asyncio.run(q.ask(question))
        assert a.text == "On the desk." and a.citations[0].frame_ref == "frames:cam-desk:k1", question
        assert "keys last seen" in evidence_prompt(rt)
    assert item_name("the wallet right now") == "wallet" and item_name("my package now again") == "package"
    assert item_name("snow globe") == "snow globe"  # only whole filler words are dropped
    q, _ = qa()
    assert asyncio.run(q.ask("Where is the wallet now?")).text == NO_EVIDENCE  # never seen: still says so


def test_last_hour_cites_each_episode_and_keeps_only_the_named_camera():
    """Round 2 (T3): the hour cited one desk frame and ignored that the question named the door."""
    mem = FakeMemory(episodes=[
        ep("d1", 0.8, ["A person types at the desk."]),
        ep("door1", 0.6, ["A courier rings the bell."], camera="cam-door"),
        ep("door2", 0.2, ["A neighbour knocks and leaves."], camera="cam-door"),
    ])
    rt = FakeRuntime(chat_replies=[ChatResult(content="Two people came to the door.", model_version="v1")])
    a = asyncio.run(QA(rt, mem, clock=lambda: NOW).ask("Did anyone come to the door in the last hour?"))
    assert [(c.episode_id, c.frame_ref) for c in a.citations] == [("door1", "frames:cam-door:door1"), ("door2", "frames:cam-door:door2")]
    assert "types at the desk" not in evidence_prompt(rt)
    rt = FakeRuntime(chat_replies=[ChatResult(content="Typing, then visitors.", model_version="v1")])
    a = asyncio.run(QA(rt, mem, clock=lambda: NOW).ask("What happened in the last hour?"))
    assert [c.episode_id for c in a.citations] == ["d1", "door1", "door2"]


def test_yesterday_and_what_happened_to_are_answered_from_memory():
    """Round 2 (T5): both went to the tool loop and the fake model said "I didn't see that"."""
    from contexts.agent.application.qa import time_window

    start, end = time_window("how many energy drinks did i have yesterday?", NOW)
    assert (start, end) == (datetime(2026, 10, 10, 4, tzinfo=UTC), datetime(2026, 10, 11, 4, tzinfo=UTC))  # New York midnights
    mem = FakeMemory(episodes=[ep("drink", 20, ["A person opens an energy drink can."]), ep("today", 1, ["Coffee."])])
    rt = FakeRuntime(chat_replies=[ChatResult(content="One energy drink.", model_version="v1")])
    a = asyncio.run(QA(rt, mem, clock=lambda: NOW).ask("How many energy drinks did I have yesterday?"))
    assert a.text == "One energy drink." and "energy drink can" in evidence_prompt(rt) and "Coffee" not in evidence_prompt(rt)

    mem = FakeMemory(episodes=[ep("p1", 2, ["A courier leaves a package at the door."], camera="cam-door"),
                               ep("p2", 1, ["A person carries the package to the desk."])])
    rt = FakeRuntime(chat_replies=[ChatResult(content="It was delivered, then moved to the desk.", model_version="v1")])
    a = asyncio.run(QA(rt, mem, clock=lambda: NOW).ask("What happened to the package?"))
    assert a.text.startswith("It was delivered") and [c.episode_id for c in a.citations] == ["p2", "p1"]  # newest first


def test_due_lists_are_answered_from_the_data_and_hedges_are_dropped():
    """On the Jetson the 4B model answered: The most recent item is due at 11:59 PM. I didn't see that."""
    from contexts.agent.application.qa import tidy
    from contracts.memory import Item

    hw = Item(connector="canvas", kind="assignment", external_id="hw4", title="HW 4: linked lists",
              when=NOW + timedelta(days=2), url="https://canvas/hw4")
    exam = Item(connector="canvas", kind="assignment", external_id="ex1", title="Exam 1",
                when=NOW + timedelta(days=4), url="https://canvas/ex1")
    rt = FakeRuntime()
    a = asyncio.run(QA(rt, FakeMemory(items=[exam, hw]), clock=lambda: NOW).ask("Any assignments due this week?"))
    assert a.text.startswith("Due this week: HW 4: linked lists, ") and "; Exam 1, " in a.text
    assert [c.url for c in a.citations] == ["https://canvas/hw4", "https://canvas/ex1"] and not rt.calls  # no model call
    assert tidy("Your keys were last seen at 7:41 PM. I didn't see that.") == "Your keys were last seen at 7:41 PM."
    assert tidy(NO_EVIDENCE) == NO_EVIDENCE


def visit_ep(id, minutes_ago, length, caption, labels=("person",), camera="cam-desk"):
    start = NOW - timedelta(minutes=minutes_ago)
    return Episode(id=id, camera_id=camera, start=start, end=start + timedelta(minutes=length), labels=list(labels),
                   captions=[caption], keyframe_refs=[f"frames:{camera}:{id}"])


def test_a_visit_is_the_latest_stretch_of_people_and_everything_seen_during_it():
    memory = FakeMemory(episodes=[
        visit_ep("old", 60, 5, "Two people eat lunch at the table."),  # ended 55 min ago: an earlier visit
        visit_ep("v1", 20, 4, "Two people stop at the table and look at the box."),
        visit_ep("obj", 17, 1, "A phone is left on the desk.", labels=("cell phone",)),
        visit_ep("v2", 14, 6, "One of them asks Atlas a question and points at the phone screen."),  # 2 min after v1 ended
    ])
    a = QA(FakeRuntime(), memory, clock=lambda: NOW)
    ev = a.prefetch("What happened while the judges were here?")
    assert [c.episode_id for c in ev.citations] == ["v1", "obj", "v2"]
    assert ev.direct.startswith("The latest visit started") and "phone is left on the desk" in ev.direct
    assert "Two people eat lunch" not in ev.direct  # the earlier visit


def test_no_person_in_view_recently_means_no_visit():
    memory = FakeMemory(episodes=[visit_ep("obj", 10, 1, "A cup on the desk.", labels=("cup",))])
    a = QA(FakeRuntime(), memory, clock=lambda: NOW)
    assert asyncio.run(a.ask("Recap their visit")).text == NO_EVIDENCE


def test_while_we_were_asleep_is_still_the_night_not_a_visit():
    a = QA(FakeRuntime(), MEMORY, clock=lambda: NOW)
    assert a.prefetch("What went on while we were asleep?").direct is None
    assert a.prefetch("Recap their visit").direct.startswith("The latest visit")


# --- A13: fall back to the tool loop when the fixed path is unsure; more tools ---

def web_qa(*replies, web=None, memory=MEMORY):
    from contexts.agent.application.web_port import FakeWeb

    rt = FakeRuntime(chat_replies=[ChatResult(content=r, model_version="v1") if isinstance(r, str) else r for r in replies])
    web = web or FakeWeb(results=[{"title": "Knights 31, Gators 24", "url": "https://espn.test/ucf-uf", "summary": "Final"}],
                         pages={"https://espn.test/ucf-uf": {"url": "https://espn.test/ucf-uf", "title": "UCF vs UF",
                                                             "summary": "UCF beat UF 31-24 on Sep 5."}})
    return QA(rt, memory, clock=lambda: NOW, web=web), rt, web


def calls(*named):
    return ChatResult(content="", model_version="v1",
                      tool_calls=[ToolCall(id=f"c{n}", name=name, arguments=args) for n, (name, args) in enumerate(named)])


def test_a_score_question_searches_the_web_then_reads_the_page():
    q, rt, web = web_qa(calls(("web_search", {"query": "UCF vs UF most recent game score"})),
                        calls(("read_page", {"url": "https://espn.test/ucf-uf", "goal": "final score"})),
                        "UCF beat UF 31-24 on Sep 5.")
    a = asyncio.run(q.ask("What was the score in the most recent game against UF?"))
    assert a.text == "UCF beat UF 31-24 on Sep 5."
    assert {c.url for c in a.citations} == {"https://espn.test/ucf-uf"}
    assert ("browse", "https://espn.test/ucf-uf", "final score") in web.calls
    assert "sports scores" in rt.calls[-1][1][0].content  # the web hint is in the tool loop's system prompt


def test_web_questions_get_four_tool_calls():
    q, rt, _ = web_qa(*[calls(("web_search", {"query": f"try {n}"})) for n in range(5)], "done")
    asyncio.run(q.ask("Who won the game?"))
    assert sum(1 for name, _ in rt.calls if name == "chat") == 5  # 4 tool rounds, then one answer from the evidence


def test_where_is_my_wallet_still_says_so_at_once_with_web_tools_on():
    q, rt, web = web_qa("should not be asked")
    for text in ("Where is my wallet?", "Have you seen my wallet?", "What happened to my wallet?", "Did my wallet arrive?"):
        assert asyncio.run(q.ask(text)).text == NO_EVIDENCE
    assert not any(name == "chat" for name, _ in rt.calls)
    assert web.calls == []


def test_an_empty_fixed_path_falls_back_to_the_tool_loop_with_web_tools():
    q, rt, web = web_qa(calls(("web_search", {"query": "UCF last win"})), "UCF last won on Sep 5.", memory=FakeMemory())
    a = asyncio.run(q.ask("When was the last time UCF won a football game?"))
    assert a.text == "UCF last won on Sep 5."
    assert web.calls[0] == ("search", "UCF last win")


def test_without_web_an_empty_fixed_path_still_answers_at_once():
    rt = FakeRuntime(chat_replies=[ChatResult(content="should not be asked", model_version="v1")])
    q = QA(rt, FakeMemory(), clock=lambda: NOW)
    assert asyncio.run(q.ask("When was the last time UCF won a football game?")).text == NO_EVIDENCE
    assert not any(name == "chat" for name, _ in rt.calls)


def test_an_unsure_fixed_answer_is_replaced_by_the_tool_loop_and_only_the_final_one_streams():
    q, rt, web = web_qa(NO_EVIDENCE, calls(("web_search", {"query": "table"})), "Found it on the web.")
    pieces = []
    a = asyncio.run(q.ask("When did someone last leave the table?", stream=pieces.append))
    assert a.text == "Found it on the web."
    assert pieces == ["Found it on the web."]


def test_a_sure_fixed_answer_with_web_tools_stays_one_model_call():
    q, rt, web = web_qa("At 1:00 AM someone packed a laptop and left.")
    pieces = []
    a = asyncio.run(q.ask("When did someone last leave the table?", stream=pieces.append))
    assert a.text.startswith("At 1:00 AM")
    assert sum(1 for name, _ in rt.calls if name == "chat") == 1
    assert pieces == [a.text] and web.calls == []


def test_item_history_tool_gathers_the_last_sighting_and_the_moments_it_appeared():
    q, rt = qa(calls(("item_history", {"label": "package", "hours": 6})), "A courier left a box at the door.")
    asyncio.run(q.ask("Anything about deliveries?"))
    result = [m.content for m in rt.calls[-1][1] if m.role == "tool"][0]
    assert "courier drops a box" in result
    assert "Old episode" not in result


def test_upcoming_tool_lists_connector_items_matching_the_query():
    from contracts.memory import Item

    items = [Item(connector="canvas", kind="assignment", external_id="a1", title="COP 4533 Milestone 1",
                  when=NOW + timedelta(days=2), url="https://canvas.test/a1"),
             Item(connector="calendar", kind="event", external_id="e1", title="Dinner with Maya",
                  when=NOW + timedelta(days=1), url="https://cal.test/e1")]
    q, rt = qa(calls(("upcoming", {"query": "milestone", "days": 7})), "Milestone 1 is due Tuesday.")
    q.memory = FakeMemory(items=items)
    a = asyncio.run(q.ask("Anything I should work on?"))
    result = [m.content for m in rt.calls[-1][1] if m.role == "tool"][0]
    assert "COP 4533 Milestone 1" in result and "Dinner" not in result
    assert a.citations[0].url == "https://canvas.test/a1"


def test_prepare_for_runs_the_planner_as_a_tool():
    from contracts.agent import PlanCreated
    from contracts.types import Citation

    class Planner:
        def wants(self, question):
            return False

        async def answer(self, question):
            return PlanCreated(plan_id="p1", subject=question, text="Study recursion tonight, sleep by midnight.",
                               citations=[Citation(url="https://cs.ucf.edu/exam/")], model_version="v1")

    q, rt = qa(calls(("prepare_for", {"event": "COP 3502 exam"})), "Study recursion tonight.")
    q.planner = Planner()
    a = asyncio.run(q.ask("I have a test soon, what now?"))
    assert a.citations[0].url == "https://cs.ucf.edu/exam/"
    assert "Study recursion tonight, sleep by midnight." in [m.content for m in rt.calls[-1][1] if m.role == "tool"][0]


def test_a_model_that_refuses_without_trying_gets_a_web_search_of_the_question():
    q, rt, web = web_qa("I don't know the score. I can't browse the web or check sports news from here.",
                        "UCF beat UF 31-24 on Sep 5.")
    pieces = []
    a = asyncio.run(q.ask("What was the score in the most recent game against UF?", stream=pieces.append))
    assert web.calls[0] == ("search", "What was the score in the most recent game against UF?")
    assert a.text == "UCF beat UF 31-24 on Sep 5." and "".join(pieces) == a.text
    assert a.citations[0].url == "https://espn.test/ucf-uf"


def test_a_refusal_on_the_fixed_path_counts_as_unsure():
    q, rt, web = web_qa("As an AI I don't have access to that.", calls(("web_search", {"query": "table"})), "Found it.")
    assert asyncio.run(q.ask("When did someone last leave the table?")).text == "Found it."


def test_no_second_search_when_the_model_already_searched():
    from contexts.agent.application.web_port import FakeWeb

    q, rt, web = web_qa(calls(("web_search", {"query": "x"})), "I don't know.", web=FakeWeb())
    assert asyncio.run(q.ask("Who won the game?")).text == NO_EVIDENCE  # nothing found anywhere
    assert [c[0] for c in web.calls] == ["search"]


def test_unsure_spots_refusals_but_not_real_answers():
    from contexts.agent.application.qa import unsure

    assert unsure(NO_EVIDENCE) and unsure("I can't browse the web.") and unsure("I'm just an AI running on this box")
    assert not unsure("Your keys are on the desk.") and not unsure("UF won 24-13, I know it was close.")
