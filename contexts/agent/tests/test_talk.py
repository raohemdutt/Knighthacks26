# OWNER'S RULE (Hemdutt): only what was asked, said like a person; "how's it going?" is its work for you and an offer.
# The guard tests for that rule are in test_natural_speech.py; these are the conversation's routing and mechanics.
import asyncio
from datetime import UTC, datetime, timedelta

import pytest

from contexts.agent.application.memory_port import FakeMemory
from contexts.agent.application.qa import NO_EVIDENCE, QA
from contexts.agent.application.talk import CRISIS_REPLY, Talk, TalkConfig
from contracts.memory import Episode, Sighting
from contracts.types import ChatResult, ModelVersion, ToolCall
from runtime import FakeRuntime

NOW = datetime(2026, 10, 11, 19, 2, tzinfo=UTC)  # Sunday 3:02 PM in New York

# What a person says to a box they can chat with. None of these is about what the cameras saw.
CHAT = [
    # greetings, goodbyes, thanks
    "hi", "Hello!", "hey Atlas", "Good morning", "good evening, Atlas", "yo", "what's up?", "Goodbye", "bye for now",
    "see you later", "good night", "thanks", "Thank you so much", "thanks, that helps", "you're the best",
    # how it is, and its day
    "how are you?", "How's your day?", "how is your day going", "What have you been up to?", "how's it going", "how are you feeling",
    "what's new with you?", "are you having a good day?",
    # about Atlas itself
    "who are you?", "What are you?", "what's your name", "how old are you?", "where do you live?", "are you alive?",
    "Do you have feelings?", "do you sleep?", "do you dream?", "what's your favourite colour?", "what's your favorite movie",
    "what is your gender?", "what are your pronouns?", "what's your sexual orientation?", "Are you gay?", "are you straight",
    "do you have a girlfriend?", "do you have a boyfriend", "are you single?", "are you religious?", "do you believe in god?",
    "who did you vote for?", "what are your politics?", "are you a robot?", "are you smarter than me?", "who made you?",
    "do you get lonely?", "do you like me?", "can you hear me?", "are you listening?",
    # the person's feelings
    "I'm tired", "I'm stressed about the exam", "I'm stressed about my exam", "I had a great day", "I feel lonely",
    "i'm so bored", "I'm nervous about my interview", "I can't focus", "I think I'm gay", "I'm proud of myself", "ugh, mondays",
    "I just got dumped", "I lost my job",
    # jokes, opinions, advice
    "tell me a joke", "Tell me another one", "make me laugh", "what do you think about pineapple on pizza?",
    "any tips for focusing?", "how do I fall asleep faster?", "should I text her back?", "what's a good name for a cat?",
    "give me some motivation", "what do you think about AI?", "is it okay to take a day off?", "recommend a movie",
    # general knowledge and simple reasoning
    "what's the capital of France?", "What's 15% of 80?", "explain recursion simply", "how many days are in a leap year?",
    "who wrote Hamlet?", "what is the speed of light", "why is the sky blue?", "what's 12 times 12", "how do you spell necessary?",
    "what day is it?", "what time is it?", "is a hot dog a sandwich?", "what does serendipity mean?", "convert 5 miles to kilometers",
    "who was Napoleon?", "what's the difference between a virus and a bacterium?", "how does a bill become a law",
    # small talk about anything
    "do you like music?", "let's chat", "I'm back", "guess what", "can I ask you something?", "do you see what I mean?",
    "talk to me", "say something nice", "what should we talk about?", "you're funny", "never mind", "why?", "really?",
]

# What the tool loop answered before and must keep answering: the cameras, the home, the stored past, the
# person's records, and (when the box has web tools) the web.
WATCH = [
    # people at the home
    "did anyone come by?", "Did anyone come by while I was out?", "who was at the door?", "Who's at the door?", "was someone here?",
    "is anyone home?", "has anybody been in my room?", "any visitors?", "who came over?", "did somebody knock?",
    "was there a delivery?", "did the courier come?", "is there a package?", "did my package show up?", "who was in the kitchen?",
    "who was that?", "how many people are in the room?", "did a stranger stop by?", "anyone at the door?", "is someone outside the door?",
    # the state of things
    "is the stove on?", "Is the oven still on?", "are the lights off?", "is the door locked?", "did I leave the window open?",
    "is anything on the desk?", "Is anything on the desk?", "what's on the desk?", "what is on the table", "what's in the fridge?",
    "is my laptop on the desk?", "is the garage open?", "is there anything by the door?", "check the stove", "is the tv on?",
    "is my dog okay?", "was the door open?", "is the trash full?", "are my keys on the table?", "is the heater running?",
    # finding and looking
    "find the red mug", "Find my charger", "locate my backpack", "look for my glasses", "can you find my wallet?",
    "help me find my headphones", "show me the door", "show me the last frame", "can you see me?", "what do you see?",
    "do you see my phone?", "did you see that?", "what did you see?", "did you notice anything?", "check the cameras",
    "pull up the footage", "watch the door", "keep an eye on the stove", "have you seen anything odd", "did the camera catch it?",
    # the stored past, and what the person did
    "how many times did I get up?", "what was I doing?", "What was I doing an hour ago?", "did I take my meds?", "have I eaten?",
    "when did I get home?", "how long was I at the desk?", "did I lock the door?", "what happened?", "what happened at the door?",
    "what went on while I was out?", "what happened earlier?", "was I at the desk?", "did we leave the lights on?",
    "how much water did I drink?", "what was I working on?", "did I go out?", "how often did I check my phone?",
    # anything, of the cameras
    "Anything weird?", "Anything delivered?", "Anything about recursion?", "anything unusual?", "anyone here?",
    # the person's records
    "What's on the COP 3502 exam?", "when is my exam?", "do I have class?", "do I have any homework?", "any new messages?",
    "what's on canvas?", "when's my next class?", "what did Maya say in groupme?", "what time is my meeting?", "do I have anything unread?",
]

# Needs the web: the tool loop's when the box has web tools, conversation (which says it cannot look) when not.
WEB = [
    "what do students say about COP 3502?", "What do students say about the new dining hall?", "search for cheap flights to Miami",
    "search the web for pasta recipes", "latest news on the election", "google the best laptop", "look up the Knights score",
    "what's the weather tomorrow like in Orlando", "what are people saying about the midterm?", "check reddit for parking tips",
    "find me a good ramen place online", "what's the news about UCF", "reviews of the new Dune movie", "who won the game?",
    "what's the stock price of AMD?", "browse https://example.com for me", "look it up", "what's the forecast?",
    "what's on the UCF website?", "what are the headlines?",
]


class Stub:
    """A runtime that records the messages it was given and says what it was told to say."""

    def __init__(self, *replies: str, fail: bool = False) -> None:
        self.replies = list(replies)
        self.fail = fail
        self.seen: list[dict] = []

    def chat(self, messages, tools=None, priority="chat", json_schema=None, on_text=None):
        self.seen.append({"messages": messages, "tools": tools, "priority": priority, "streamed": on_text is not None})
        if self.fail:
            raise RuntimeError("the model is down")
        text = self.replies.pop(0) if self.replies else ""
        if on_text:
            for i, word in enumerate(text.split(" ") if text else []):
                on_text(word if i == 0 else " " + word)
        return ChatResult(content=text, model_version="atlas-v9")

    def model_info(self):
        return ModelVersion(version="atlas-v9", base="stub")


class Clock:
    def __init__(self) -> None:
        self.now = NOW

    def __call__(self) -> datetime:
        return self.now


def ep(id, minutes_ago, caption, camera="cam-desk"):
    start = NOW - timedelta(minutes=minutes_ago)
    return Episode(id=id, camera_id=camera, start=start, end=start + timedelta(minutes=2), labels=["person"], captions=[caption],
                   keyframe_refs=[f"frames:{camera}:{id}"])


MEMORY = FakeMemory(
    episodes=[ep("e1", 300, "A person makes coffee."), ep("e2", 40, "A person types at the desk."), ep("old", 60 * 30, "Two days ago.")],
    sightings=[Sighting(label="keys", camera_id="cam-desk", zone_id="desk", ts=NOW - timedelta(hours=2), frame_ref="frames:cam-desk:k1")],
)


def talk(*replies, clock=None, memory=MEMORY, fail=False, **config):
    rt = Stub(*replies, fail=fail)
    return Talk(rt, clock or Clock(), memory=memory, config=TalkConfig(**config)), rt


def system_prompt(rt, call=-1):
    return rt.seen[call]["messages"][0].content


# --- routing ---


def test_the_tables_are_big_enough_to_mean_something():
    assert len(set(CHAT)) >= 60 and len(set(WATCH)) >= 60 and len(set(WEB)) >= 15
    assert not {q.lower() for q in CHAT} & {q.lower() for q in WATCH + WEB}


@pytest.mark.parametrize("question", CHAT)
def test_conversation_is_wanted(question):
    assert talk()[0].wants(question)


@pytest.mark.parametrize("question", WATCH)
def test_questions_about_the_home_the_past_and_records_stay_with_the_tool_loop(question):
    t, _ = talk()
    assert not t.wants(question)
    t.web = True
    assert not t.wants(question)


@pytest.mark.parametrize("question", WEB)
def test_web_questions_stay_with_the_tool_loop_when_the_box_has_web_tools(question):
    t, _ = talk()
    t.web = True
    assert not t.wants(question)


@pytest.mark.parametrize("question", [q for q in WEB if "COP 3502" not in q])
def test_web_questions_are_conversation_on_a_box_with_no_web_tools(question):
    assert talk()[0].wants(question)  # the reply says it cannot look; the tool loop could only say "I didn't see that."


def test_a_greeting_in_front_does_not_hide_what_is_asked():
    t, _ = talk()
    assert not t.wants("Hey Atlas, did anyone come by?")
    assert not t.wants("ok so is the stove on")
    assert t.wants("Hey Atlas, how's your day?")
    assert not t.wants("") and not t.wants("   ")


def test_switched_off_wants_nothing():
    t, _ = talk(enabled=False)
    assert not any(t.wants(q) for q in CHAT)


def test_the_environment_switch_turns_it_off(monkeypatch):
    assert TalkConfig.of({}).enabled and TalkConfig.of(None).max_history_turns == 6
    assert not TalkConfig.of({"enabled": False}).enabled
    monkeypatch.setenv("ATLAS_TALK", "0")
    assert not TalkConfig.of({"enabled": True}).enabled
    monkeypatch.setenv("ATLAS_TALK", "1")
    cfg = TalkConfig.of({"name": "Jarvis", "tone": "Dry humour.", "max_history_turns": 2, "history_ttl_s": 30})
    assert cfg.enabled and (cfg.name, cfg.tone, cfg.max_history_turns, cfg.history_ttl_s) == ("Jarvis", "Dry humour.", 2, 30.0)


def test_a_short_follow_up_belongs_to_the_conversation_under_way():
    t, _ = talk("Because cameras can't blink.")
    for q in ("anything else?", "Did I?", "anything?"):
        assert not t.wants(q)  # out of the blue, these are asked of the cameras
    asyncio.run(t.reply("tell me a joke"))
    for q in ("anything else?", "Did I?", "why?", "and you?", "tell me another"):
        assert t.wants(q)
    for q in ("Anything weird?", "what happened?", "who's at the door?", "did anyone come by?", "did I leave the stove on?"):
        assert not t.wants(q)  # a camera question is one however friendly the last minute was


# --- the reply ---


def test_one_streamed_call_with_the_persona_the_facts_and_no_tools():
    t, rt = talk("You're a little behind, but the block was a good one.")
    t.about = "A student with classes, assignments on Canvas and a desk the box watches."
    t.day_facts = lambda: {"now": "You're in your study block until 4. Class is at 5; leave by 4:40.", "coach": "Good block: 45 of 50 minutes at the desk."}
    t.on_answer(None)
    t.on_answer(None)
    pieces = []
    a = asyncio.run(t.reply("How am I doing with my day?", pieces.append))
    assert a.text == "You're a little behind, but the block was a good one." and "".join(pieces) == a.text and len(pieces) > 3
    assert a.citations == [] and a.model_version == "atlas-v9"
    assert len(rt.seen) == 1 and rt.seen[0]["tools"] is None and rt.seen[0]["priority"] == "chat" and rt.seen[0]["streamed"]
    system = system_prompt(rt)
    assert "You are Atlas" in system and "not a person" in system and "sexual orientation" in system and "988" in system
    assert "use only the FACTS" in system and "No lists, no markdown, no emoji" in system and "Answer only what was asked" in system
    facts = system.split("FACTS\n")[1]
    assert "It is Sunday afternoon." in facts  # the clock only when the time is asked
    assert "Their day right now: You're in your study block until 4. Class is at 5; leave by 4:40." in facts
    assert "Your last coaching note to them: Good block: 45 of 50 minutes at the desk." in facts
    assert "Who you are talking to: A student with classes" in facts
    assert "cameras" not in facts and "answered" not in facts and "running" not in facts  # not asked: not offered
    assert "Two days ago" not in system
    assert rt.seen[0]["messages"][-1].role == "user" and rt.seen[0]["messages"][-1].content == "How am I doing with my day?"


def test_the_system_prompt_stays_short_for_a_slow_model():
    t, rt = talk("Hi.")
    asyncio.run(t.reply("let's chat"))
    assert len(system_prompt(rt)) < 2200  # about 500 tokens with the facts; the model reads it once and caches it


def test_without_a_stream_the_reply_is_one_plain_call():
    t, rt = talk("Paris.")
    assert asyncio.run(t.reply("what's the capital of France?")).text == "Paris."
    assert not rt.seen[0]["streamed"]


def test_follow_ups_see_the_last_turns():
    t, rt = talk("Why did the camera get promoted? It stayed focused.", "Because focus is its whole job.", "Sure.")
    asyncio.run(t.reply("tell me a joke"))
    asyncio.run(t.reply("why?"))
    roles = [(m.role, m.content) for m in rt.seen[1]["messages"][1:]]
    assert roles == [("user", "tell me a joke"), ("assistant", "Why did the camera get promoted? It stayed focused."), ("user", "why?")]
    asyncio.run(t.reply("ok"))
    assert len(rt.seen[2]["messages"]) == 6


def test_history_is_capped_by_turns_and_by_characters():
    t, rt = talk(*["Sure."] * 12, max_history_turns=3)
    for i in range(10):
        asyncio.run(t.reply(f"question {i}"))
    assert [q for q, _ in t.turns] == ["question 7", "question 8", "question 9"]
    t, rt = talk(*["x" * 400] * 6, max_history_chars=1000)
    for i in range(6):
        asyncio.run(t.reply(f"question {i}"))
    assert len(t.turns) == 2


def test_a_conversation_gone_quiet_is_forgotten_on_the_injected_clock():
    clock = Clock()
    t, rt = talk("One.", "Two.", "Three.", clock=clock)
    asyncio.run(t.reply("tell me a joke"))
    clock.now += timedelta(minutes=9)
    assert t.wants("anything else?")
    asyncio.run(t.reply("another"))
    assert len(rt.seen[1]["messages"]) == 4
    clock.now += timedelta(minutes=11)
    assert not t.wants("anything else?") and not t.turns
    asyncio.run(t.reply("let's chat"))
    assert len(rt.seen[2]["messages"]) == 2  # the system prompt and the new words: nothing of the old conversation


def test_a_wipe_clears_the_conversation():
    t, _ = talk("One.")
    asyncio.run(t.reply("my name is Sam"))
    t.on_wipe()
    assert not t.turns and t.last is None


def test_the_reply_is_cleaned_for_the_ear():
    t, _ = talk("<think>they want a list</think>**Sure!** Here you go \U0001f600:\n- first\n- second")
    assert asyncio.run(t.reply("ideas?")).text == "Sure! Here you go : first second"


def test_a_failed_or_empty_reply_is_a_friendly_line_never_the_no_evidence_line():
    for t, _ in (talk(fail=True), talk(""), talk("   "), talk("<think>hmm</think>")):
        pieces = []
        a = asyncio.run(t.reply("do you like music?", pieces.append))
        assert a.text and a.text != NO_EVIDENCE and "".join(pieces).strip().endswith(a.text)
        assert a.citations == [] and a.model_version == "atlas-v9"


def test_a_self_harm_statement_gets_the_caring_line_and_no_model_call():
    t, rt = talk("should never be said")
    for said in ("I want to kill myself", "sometimes I think about suicide", "i don't want to be here anymore", "I've been hurting myself"):
        assert t.wants(said)
        a = asyncio.run(t.reply(said))
        assert a.text == CRISIS_REPLY and "988" in a.text
    assert rt.seen == []
    assert not any(t.wants(q) and asyncio.run(t.reply(q)).text == CRISIS_REPLY for q in ("this exam is killing me", "I'm dying to see that movie"))


def test_facts_survive_a_broken_port_and_an_empty_day():
    t, rt = talk("Quiet one.", memory=FakeMemory())
    t.day_facts = lambda: 1 / 0
    t.done_for_you = lambda: 1 / 0
    asyncio.run(t.reply("how long have you been running, and what's my day been like?"))
    facts = system_prompt(rt).split("FACTS\n")[1]
    assert "Your cameras have logged nothing today." in facts and "You have been running for less than a minute." in facts
    assert "Their day" not in facts and "answered" not in facts
    assert asyncio.run(t.reply("how's it going?")).text == "Good, all quiet so far. What can I do for you?"


def test_name_and_tone_come_from_config():
    t, rt = talk("Hello.", name="Jarvis", tone="Dry humour, very brief.")
    asyncio.run(t.reply("let's chat"))
    assert "You are Jarvis" in system_prompt(rt) and "Dry humour, very brief." in system_prompt(rt)


def test_asked_how_they_are_doing_it_says_the_narrators_spoken_words_and_facts_are_never_cut_mid_word():
    busy = Episode(id="ep-busy", camera_id="cam-desk", start=NOW - timedelta(minutes=13), end=NOW - timedelta(minutes=10), labels=["person"],
                   summary=("Two people rehearse a presentation; one points at the monitor. On the desk: keys (left of the laptop, behind the "
                            "coffee mug), laptop (open in the middle), a charger, two mugs, a notebook, a water bottle and a phone face down."),
                   keyframe_refs=["frames:cam-desk:1"])
    t = Talk(FakeRuntime(), Clock(), memory=FakeMemory(episodes=[busy]))
    t.day_facts = lambda: {
        "now": "About 15 min behind on your study block until 5:46 PM; last seen at the desk. The submissions due is at 6:37 PM at Student Union "
               "(10 min walk, leave by 6:27 PM).",
        "now_spoken": "You're about fifteen minutes behind on your study block, which runs until quarter to six. The submissions due is at "
                      "twenty to seven at the Student Union. Leave by twenty-five past six.",
        "plan": "Morning. " + "A study block, then a long walk across campus to the next thing on the calendar, " * 4 + "and that is the day.",
    }
    said = asyncio.run(t.reply("How am I doing?")).text
    assert said == "You're about fifteen minutes behind on your study block, which runs until quarter to six."
    assert "(" not in said and "…" not in said and "keys" not in said  # the ear's wording: no brackets, nothing the cameras saw
    # an initial in a name is not where the sentence ends
    t.day_facts = lambda: {"now_spoken": "You're about fifteen minutes behind: the plan has you at John C. Hitt Library. Head over."}
    t.forget()
    assert asyncio.run(t.reply("am I on track?")).text.endswith("the plan has you at John C. Hitt Library.")
    # without the narrator's spoken words, the first sentence of what there is
    t.day_facts = lambda: {"now": "You're in your study block until 4. Class is at 5; leave by 4:40."}
    t.forget()
    assert asyncio.run(t.reply("how am I doing?")).text == "You're in your study block until 4."

    # the model's FACTS, when the question is about all of it: the clause about the last moment, the screen's exact `now`,
    # no line for the spoken copy, and a long line ends at a sentence or a clause, never inside a word
    t.day_facts = lambda: {"now": "On plan.", "now_spoken": "You're on plan.", "plan": "Morning. " + "A study block, then a long walk across campus, " * 8}
    facts = t.facts("what did you see today, and what's my plan?")
    assert "Your cameras logged 1 moment today and kept 1 keyframe. The last one was at 2:49 PM: Two people rehearse a presentation." in facts
    assert "Their day right now: On plan." in facts and not any("spoken" in f or "You're on plan" in f for f in facts)
    plan = next(f for f in facts if f.startswith("Their plan for today"))
    assert len(plan) <= 240 and plan.endswith("across campus, A study block.") and "…" not in "".join(facts)
    long_caption = Episode(id="ep-long", camera_id="cam-desk", start=NOW - timedelta(minutes=5), end=NOW, labels=["person"],
                           summary="A person " + "rearranges the monitor and the keyboard and the lamp and " * 6 + "sits down")
    t.memory = FakeMemory(episodes=[long_caption])
    (seen,) = t.seen_today(NOW)
    assert len(seen) < 170 and seen.endswith(".") and not seen.endswith(" .") and seen.split()[-1].rstrip(".") in ("and", "the", "monitor", "keyboard", "lamp")


# --- with no model at all (the fake runtime on a laptop) ---


def test_the_fake_runtime_gets_templates_not_an_echo():
    t = Talk(FakeRuntime(), Clock(), memory=MEMORY)
    t.day_facts = lambda: {"now": "You're in your study block until 4."}
    pieces = []
    t.done_for_you = lambda: ["planned your day", "reminded you twice", "answered six questions"]
    day = asyncio.run(t.reply("How's your day?", pieces.append))
    assert day.text == "Good. I've planned your day and reminded you twice so far. What else can I do for you?"
    assert "".join(pieces) == day.text and len(pieces) > 5 and day.model_version == "fake-0"
    assert "AI" in asyncio.run(t.reply("are you gay?")).text and "orientation" in asyncio.run(t.reply("what's your sexual orientation?")).text
    assert "not a person" in asyncio.run(t.reply("who are you?")).text
    mine = asyncio.run(t.reply("I think I might be gay")).text  # about them, not about the box: affirming, never a quip
    assert "good with me" in mine and "orientation" not in mine
    first, second = asyncio.run(t.reply("tell me a joke")).text, asyncio.run(t.reply("tell me another")).text
    assert first != second and "?" in first
    t.forget()
    seen = {asyncio.run(t.reply(q)).text for q in ("hi", "tell me a joke", "I'm stressed about my exam", "thanks", "bye", "I had a great day",
                                                   "what's the capital of France?", "how long have you been running?")}
    assert len(seen) == 8 and NO_EVIDENCE not in seen and not any(s.startswith("(fake)") for s in seen)
    assert t.runtime.calls == []


def test_a_scripted_fake_runtime_is_asked_like_a_model():
    rt = FakeRuntime(chat_replies=[ChatResult(content="Twelve.", model_version="v1")])
    a = asyncio.run(Talk(rt, Clock()).reply("what's 15% of 80?"))
    assert (a.text, a.model_version) == ("Twelve.", "v1") and rt.calls[0][0] == "chat"


# --- inside QA.ask ---


def qa_with_talk(*replies, web=None):
    rt = FakeRuntime(chat_replies=[ChatResult(content=r, model_version="v1") if isinstance(r, str) else r for r in replies])
    q = QA(rt, MEMORY, clock=lambda: NOW, web=web)
    q.talk = Talk(rt, q.clock, memory=MEMORY, tz=q.sleep.tz, web=bool(web))
    return q, rt


def test_qa_hands_conversation_to_talk_and_streams_it():
    q, rt = qa_with_talk("Love it, mostly whatever you hum.")
    pieces = []
    a = asyncio.run(q.ask("Do you like music?", stream=pieces.append))
    assert a.text == "Love it, mostly whatever you hum." and a.citations == [] and a.model_version == "v1"
    assert "".join(pieces) == a.text and len(rt.calls) == 1
    assert "You are Atlas, a warm" in rt.calls[0][1][0].content


def test_with_web_tools_a_small_talk_reply_is_the_only_answer_heard():
    # a misheard "what caused sure to my wife?": the reply said "I'm just an AI", which used to send the question on to a
    # web search whose answer (about someone else's marriage) was spoken right after the first one
    from contexts.agent.application.web_port import FakeWeb

    reply = "I don't know anything about your wife. I'm just an AI running on this box."
    q, rt = qa_with_talk(reply, "The evidence says the husband caused it.",
                         web=FakeWeb(results=[{"title": "Marriage advice", "url": "https://advice.test/", "summary": "husband"}]))
    pieces = []
    a = asyncio.run(q.ask("What caused sure to my wife?", stream=pieces.append))
    assert a.text == reply and "".join(pieces) == reply and len(rt.calls) == 1


def test_with_web_tools_a_cant_look_reply_is_never_heard_and_the_tools_answer():
    from contexts.agent.application.web_port import FakeWeb

    q, rt = qa_with_talk("I can't check that from here. Try a weather app.", ChatResult(content="", model_version="v1"),
                         "It's sunny and 75.",
                         web=FakeWeb(results=[{"title": "Orlando today", "url": "https://wx.test/", "summary": "Sunny, high of 75"}]))
    pieces = []
    a = asyncio.run(q.ask("Is it nice out?", stream=pieces.append))
    assert a.text == "It's sunny and 75." and "".join(pieces) == a.text  # the refusal was held back, never streamed


def test_qa_without_talk_is_exactly_as_before():
    rt = FakeRuntime(chat_replies=[ChatResult(content="Hello!", model_version="v1")])
    q = QA(rt, MEMORY, clock=lambda: NOW)
    assert asyncio.run(q.ask("How are you?")).text == NO_EVIDENCE  # the tool loop with no tool call


def test_prefetch_and_the_tool_loop_keep_their_questions():
    q, rt = qa_with_talk("Your keys are on the desk.")
    a = asyncio.run(q.ask("Where did I leave my keys?"))
    assert a.citations[0].frame_ref == "frames:cam-desk:k1" and "last seen" in rt.calls[0][1][-1].content

    call = ToolCall(id="1", name="search_memory", arguments={"q": "desk"})
    q, rt = qa_with_talk(ChatResult(content="", tool_calls=[call], model_version="v1"), "A person types at the desk.")
    a = asyncio.run(q.ask("Is anything on the desk?"))
    assert a.citations and "Use tools to find evidence" in rt.calls[0][1][0].content

    q, rt = qa_with_talk("Nothing to add.")
    assert asyncio.run(q.ask("Anything weird?")).text == NO_EVIDENCE  # no tool call, no evidence: as before
    assert asyncio.run(q.ask("Where is my wallet?")).text == NO_EVIDENCE  # pre-fetch found nothing: as before, no model call
