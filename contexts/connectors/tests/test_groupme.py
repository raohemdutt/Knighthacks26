from datetime import UTC, datetime
from pathlib import Path

import pytest

from contexts.connectors.servers.groupme import find_group, post_message, read_messages, seed, split_speaker, to_items

T0 = datetime(2026, 10, 10, 14, 0, tzinfo=UTC)
ENV = {"GROUPME_BOT_CLASS": "bot-class", "GROUPME_BOT_FRIENDS": "bot-friends"}
CLASS = {"key": "class", "name": "COP4533 group chat", "bot_env": "GROUPME_BOT_CLASS", "group_id": "g1"}
FRIENDS = {"key": "friends", "name": "Friends", "bot_env": "GROUPME_BOT_FRIENDS", "group_id": "g2"}


def options(tmp_path) -> dict:
    return {"state_dir": str(tmp_path), "groups": [CLASS, FRIENDS]}


def test_find_group_defaults_to_the_first():
    opts = {"groups": [CLASS, FRIENDS]}
    assert find_group(opts, None) == CLASS
    assert find_group(opts, "friends") == FRIENDS
    with pytest.raises(RuntimeError):
        find_group(opts, "work")


def test_each_group_posts_through_its_own_bot(tmp_path):
    sent = []
    send = lambda bot, text: sent.append((bot, text))  # noqa: E731
    post_message(options(tmp_path), "study at 2?", "class", "Maya", send, T0, ENV)
    post_message(options(tmp_path), "running late", "friends", None, send, T0, ENV)
    assert sent == [("bot-class", "Maya: study at 2?"), ("bot-friends", "running late")]
    with pytest.raises(RuntimeError):
        post_message(options(tmp_path), "hi", "class", None, send, T0, env={})


def test_posted_messages_are_read_back_without_a_token(tmp_path):
    post_message(options(tmp_path), "study at 2?", "class", "Maya", lambda bot, text: None, T0, ENV)
    [item] = read_messages(options(tmp_path), fetch=None, env={})
    assert item["kind"] == "message"
    assert item["external_id"].startswith("class:")
    assert item["title"] == "Maya: study at 2?"
    assert item["summary"] == "Maya in COP4533 group chat: study at 2?"
    assert item["when"] == T0.isoformat()


def test_live_groups_are_read_when_a_token_is_set(tmp_path):
    live = {
        "g1": [
            {"id": "2", "name": "Atlas", "sender_type": "bot", "text": "Jordan: weeks 5 and 6", "created_at": 1760104900},
            {"id": "1", "name": "Hemdutt", "sender_type": "user", "text": "which lectures?", "created_at": 1760104800},
            {"id": "3", "name": "GroupMe", "text": "Atlas was added", "created_at": 1760104700, "system": True},
            {"id": "4", "name": "Priya", "text": None, "created_at": 1760104950},  # image only
        ],
        "g2": [{"id": "9", "name": "Sam", "sender_type": "user", "text": "food?", "created_at": 1760105000}],
    }
    items = read_messages(options(tmp_path), lambda group, token, limit: live[group], {"GROUPME_ACCESS_TOKEN": "t"})
    assert [i["external_id"] for i in items] == ["class:1", "class:2", "friends:9"]
    assert items[1]["title"] == "Jordan: weeks 5 and 6"  # the bot's prefix is read back as the speaker


def test_split_speaker_only_trusts_our_bots_prefix():
    assert split_speaker("Atlas", "Maya: hi", bot=True) == ("Maya", "hi")
    assert split_speaker("Sam", "note: bring chargers", bot=False) == ("Sam", "note: bring chargers")
    assert split_speaker("Atlas", "Package left at the door", bot=True) == ("Atlas", "Package left at the door")


def test_to_items_caps_the_title():
    [item] = to_items([{"id": "1", "name": "Maya", "text": "x" * 200, "created_at": 1760104800}], CLASS)
    assert len(item["title"]) == 80


@pytest.mark.parametrize("script, bot", [("class_chat.yaml", "bot-class"), ("friends_chat.yaml", "bot-friends")])
def test_seed_posts_the_whole_script_to_its_group(tmp_path, script, bot):
    sent = []
    count = seed(
        options(tmp_path),
        Path("fixtures/groupme") / script,
        lambda b, text: sent.append((b, text)),
        sleep=lambda s: None,
        env=ENV,
    )
    assert count == len(sent) >= 8
    assert {b for b, _ in sent} == {bot}
    assert all(": " in text for _, text in sent)
