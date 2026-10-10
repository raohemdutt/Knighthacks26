from contexts.voice.text import clean_transcript, speech_for_answer


def test_times_links_and_frame_refs_are_made_sayable():
    said = speech_for_answer("**Keys** on the desk at 15:40. See https://x.test/a and frames:cam-desk:1760000000000.", 2)
    assert said == "Keys on the desk at 3:40 PM. Based on 2 pieces of evidence."


def test_one_citation_is_singular_and_none_is_silent():
    assert speech_for_answer("Yes.", 1).endswith("Based on 1 piece of evidence.")
    assert speech_for_answer("Yes.", 0) == "Yes."


def test_long_answers_are_cut_at_a_sentence():
    said = speech_for_answer(" ".join(f"Sentence number {i} is here." for i in range(60)))
    assert len(said) <= 420 and said.endswith(".")


def test_empty_answer_has_a_fallback():
    assert speech_for_answer("") == "I don't have an answer to read out for that."


def test_the_offline_demo_model_reads_its_evidence_lines_only():
    said = speech_for_answer("(fake) Evidence:\n- Keys on the desk\n- Package at the door\n\nQuestion: where?")
    assert "Keys on the desk" in said and "Package at the door" in said and "Question" not in said and "Evidence" not in said


def test_whisper_noise_tags_are_dropped():
    assert clean_transcript(" [BLANK_AUDIO] ") == ""
    assert clean_transcript("(music) Where are my keys? [noise]") == "Where are my keys?"
