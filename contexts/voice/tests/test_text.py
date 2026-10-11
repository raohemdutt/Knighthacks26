from contexts.voice.text import (MAX_SPOKEN_CHARS, SentenceStream, clean_transcript, closing_reply, find_wake, is_echo, speech_for_answer,
                                 without_wake)


def test_times_links_and_frame_refs_are_made_sayable():
    said = speech_for_answer("**Keys** on the desk at 15:40. See https://x.test/a and frames:cam-desk:1760000000000.", 2)
    assert said == "Keys on the desk at 3:40 PM. Based on 2 pieces of evidence."


def test_one_citation_is_singular_and_none_is_silent():
    assert speech_for_answer("Yes.", 1).endswith("Based on 1 piece of evidence.")
    assert speech_for_answer("Yes.", 0) == "Yes."


def test_long_answers_are_cut_at_a_sentence():
    said = speech_for_answer(" ".join(f"Sentence number {i} is here." for i in range(60)))
    assert len(said) <= MAX_SPOKEN_CHARS and said.endswith(".") and "number 59" not in said


def test_empty_answer_has_a_fallback():
    assert speech_for_answer("") == "I don't have an answer to read out for that."


def test_the_offline_demo_model_reads_its_evidence_lines_only():
    said = speech_for_answer("(fake) Evidence:\n- Keys on the desk\n- Package at the door\n\nQuestion: where?")
    assert "Keys on the desk" in said and "Package at the door" in said and "Question" not in said and "Evidence" not in said


def test_whisper_noise_tags_are_dropped():
    assert clean_transcript(" [BLANK_AUDIO] ") == ""
    assert clean_transcript("(music) Where are my keys? [noise]") == "Where are my keys?"


def test_words_whisper_invents_for_a_noise_are_not_a_question():
    for heard in (" Thank you.", "you", "Thanks for watching!", " Bye. ", " ♪♪♪", "..."):
        assert clean_transcript(heard) == ""
    assert clean_transcript("Thank you, where are my keys?") == "Thank you, where are my keys?"


def test_a_phrase_whisper_wrote_twice_is_kept_once():
    assert clean_transcript("nothing much are you. Nothing much are you.") == "nothing much are you."
    assert clean_transcript("How are you doing today? How are you doing today?") == "How are you doing today?"
    assert clean_transcript("Where are my keys?") == "Where are my keys?"
    assert clean_transcript("very very good") == "very very good"


def test_hey_atlas_is_the_wake_phrase_however_whisper_spells_it():
    for heard in ("Hey Atlas.", "hey, atlas", "Hey Atlus!", "Hey at last", "Okay Atlas", "  Hay Atlas  ", "Hey Alice", "Atlas.",
                  "Okay, Atlas Okay Atlas Okay Atlas Okay Atlas"):
        assert find_wake(heard) == "", heard


def test_a_question_in_the_same_breath_is_passed_on():
    assert find_wake("Hey Atlas, where are my keys?") == "where are my keys?"
    assert find_wake("Hey Atlas. you") == ""  # a stray word is not a question


def test_other_speech_is_not_a_wake():
    for heard in ("where are my keys", "the atlas is on the shelf", "hey there", "Alice is in the kitchen", ""):
        assert find_wake(heard) is None, heard


def test_an_answer_is_cut_into_sentences_as_it_arrives():
    s = SentenceStream()
    got = []
    for piece in ["You left them", " on the desk at 8 PM.", " They were next to", " the laptop.", " Anything else?"]:
        got += s.feed(piece)
    assert got == ["You left them on the desk at 8 PM.", "They were next to the laptop."]  # the last is not finished yet
    assert s.flush() == "Anything else?"


def test_a_very_short_sentence_is_joined_to_the_next_one():
    s = SentenceStream()
    assert s.feed("Sure. The keys are on the desk. ") == ["Sure. The keys are on the desk."]
    assert s.feed("1. ") == [] and s.flush() == "1."


def test_a_time_with_a_dot_is_not_a_sentence_end():
    s = SentenceStream()
    assert s.feed("It was at 3.40 in the afternoon, I think. ") == ["It was at 3.40 in the afternoon, I think."]


def test_hey_atlas_said_again_is_not_part_of_the_question():
    assert without_wake("hey Atlas hey Atlas what time is it") == "what time is it"
    assert without_wake("Hey Atlas.") == ""
    assert find_wake("Hey Atlas, hey Atlas, where are my keys") == "where are my keys"


SPOKEN = ["You left them on the desk at 3:40 PM.", "They were next to the laptop."]


def test_the_speakers_own_voice_is_an_echo():
    assert is_echo("You left them on the desk at 3:40 PM. They were next to the laptop.", SPOKEN)  # all of it
    assert is_echo("you left them on the desk", SPOKEN)  # the start of it
    assert is_echo("They were next to the laptop", SPOKEN)  # one sentence of it
    assert is_echo("you left them on a desk at 3:40 PM", SPOKEN)  # most of the words, misheard a little
    assert is_echo("Okay.", ["Okay."])


def test_a_reply_that_shares_a_word_with_the_answer_is_not_an_echo():
    assert not is_echo("which desk was it", SPOKEN)
    assert not is_echo("the desk?", SPOKEN)
    assert not is_echo("why were they next to my bag", SPOKEN)
    assert not is_echo("why?", SPOKEN) and not is_echo("", SPOKEN) and not is_echo("why?", [])


def test_a_goodbye_said_alone_ends_the_conversation():
    for said in ("Thanks.", "thank you", "Thank you so much!", "No thanks.", "Okay, thanks Atlas.", "Thanks, bye."):
        assert closing_reply(said) == "Any time.", said
    for said in ("That's all.", "that\u2019s it", "Never mind.", "Stop.", "Goodbye.", "Bye!", "No.", "Nope", "I'm good.",
                 "No, that's all for now."):
        assert closing_reply(said) == "Okay.", said


def test_a_question_that_only_contains_a_goodbye_word_is_not_one():
    for said in ("thanks, and what about tomorrow?", "no, I meant the other one", "stop the timer", "that's it on the desk?",
                 "why?", "what about bye week", ""):
        assert closing_reply(said) is None, said
