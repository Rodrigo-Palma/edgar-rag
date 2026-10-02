import json

import numpy as np
import pytest

from edgar_rag.domain import Chunk, EmbedderSpec, Generation, NotRecorded, ScoredChunk
from edgar_rag.eval.replay import (
    BRIER,
    BRIER_PATH,
    EMBED,
    GENERATE,
    Tape,
    TapedBrierScores,
    TapedEmbedder,
    TapedGenerator,
    TapeMiss,
    brier_request,
    tape_key,
)
from tests.fakes import EXAMPLE, FakeEmbedder, FakeGenerator

SPEC = EmbedderSpec(model="nomic-embed-text", lowercase=True)
OPTIONS = {"temperature": 0, "seed": 0, "num_ctx": 8192, "think": False}


def test_the_key_is_the_canonical_json_so_key_order_does_not_matter():
    assert tape_key({"a": 1, "b": "x"}) == tape_key({"b": "x", "a": 1})
    assert tape_key({"a": 1}) != tape_key({"a": 2})


def test_a_recorded_vector_replays_byte_for_byte(tmp_path):
    vector = [0.1234567, -2.5e-8, 3.0]
    recording = TapedEmbedder(Tape.open(tmp_path), SPEC, live=FakeEmbedder({"Revenue?": vector}))
    recorded = recording.embed(("Revenue?",))
    recording.tape.save({})

    replayed = TapedEmbedder(Tape.open(tmp_path), SPEC).embed(("Revenue?",))

    assert replayed.dtype == np.float32
    assert replayed.tobytes() == recorded.tobytes()


def test_the_embedding_key_is_the_text_the_model_receives_after_lowercasing(tmp_path):
    tape = Tape.open(tmp_path)
    TapedEmbedder(tape, SPEC, live=FakeEmbedder({"Apple revenue": [1.0, 0.0]})).embed(
        ("Apple revenue",)
    )

    # Lower-cased, the model sees the same bytes: a hit, no live model needed.
    assert TapedEmbedder(tape, SPEC).embed(("APPLE REVENUE",)).shape == (1, 2)
    # Sent as written, it is another call.
    with pytest.raises(TapeMiss):
        TapedEmbedder(tape, EmbedderSpec(SPEC.model, lowercase=False)).embed(("Apple revenue",))


def test_a_miss_in_replay_says_to_re_record_locally(tmp_path):
    with pytest.raises(TapeMiss, match="re-record locally"):
        TapedEmbedder(Tape.open(tmp_path), SPEC).embed(("never recorded",))
    with pytest.raises(TapeMiss, match="re-record locally"):
        TapedGenerator(Tape.open(tmp_path), "qwen3:8b", OPTIONS).generate("a prompt")


def test_a_miss_is_the_domain_s_not_recorded_so_the_service_can_map_it(tmp_path):
    with pytest.raises(NotRecorded):
        TapedEmbedder(Tape.open(tmp_path), SPEC).embed(("never recorded",))


def test_a_generation_replays_with_its_tokens_and_seconds(tmp_path):
    live = FakeGenerator("Revenue was $10 million [1].")
    recording = TapedGenerator(Tape.open(tmp_path), "qwen3:8b", OPTIONS, live=live)
    first = recording.generate("the prompt")
    recording.tape.save({})

    replayed = TapedGenerator(Tape.open(tmp_path), "qwen3:8b", OPTIONS).generate("the prompt")

    assert replayed == Generation(
        text=first.text,
        prompt_tokens=first.prompt_tokens,
        completion_tokens=first.completion_tokens,
        seconds=0.0,
    )


def test_a_tape_hit_does_not_call_the_live_model_again(tmp_path):
    live = FakeGenerator("text [1].")
    taped = TapedGenerator(Tape.open(tmp_path), "qwen3:8b", OPTIONS, live=live)

    taped.generate("same prompt")
    taped.generate("same prompt")

    assert live.prompts == ["same prompt"]


def test_another_model_or_seed_is_another_call(tmp_path):
    tape = Tape.open(tmp_path)
    TapedGenerator(tape, "qwen3:8b", OPTIONS, live=FakeGenerator("x [1].")).generate("p")

    with pytest.raises(TapeMiss):
        TapedGenerator(tape, "qwen3:32b", OPTIONS).generate("p")
    with pytest.raises(TapeMiss):
        TapedGenerator(tape, "qwen3:8b", {**OPTIONS, "seed": 1}).generate("p")


def test_rows_are_appended_as_recorded_and_saved_sorted_by_key(tmp_path):
    tape = Tape.open(tmp_path)
    live = FakeGenerator("x [1].")
    taped = TapedGenerator(tape, "m", OPTIONS, live=live)
    for prompt in ("c", "a", "b"):
        taped.generate(prompt)

    # Before saving, a run that stopped still left every row on disk.
    assert len((tmp_path / f"{GENERATE}.jsonl").read_text().splitlines()) == 3

    tape.save({"generation": {"name": "m", "digest": None}})
    keys = [json.loads(line)["key"] for line in (tmp_path / f"{GENERATE}.jsonl").open()]
    assert keys == sorted(keys)
    assert Tape.open(tmp_path).meta == {"generation": {"name": "m", "digest": None}}


def test_a_corrupt_tape_row_is_refused_with_its_line(tmp_path):
    (tmp_path / f"{EMBED}.jsonl").write_text('{"no key": 1}\n')

    with pytest.raises(ValueError, match="embed.jsonl:1"):
        Tape.open(tmp_path)


def test_a_git_lfs_pointer_in_place_of_a_tape_file_says_to_pull_it(tmp_path):
    pointer = "version https://git-lfs.github.com/spec/v1\noid sha256:ab\nsize 4194304\n"
    (tmp_path / f"{EMBED}.jsonl").write_text(pointer)

    with pytest.raises(ValueError, match=r"embed.jsonl is a Git LFS pointer.*git lfs pull"):
        Tape.open(tmp_path)


QUESTION = "what does the company design?"


def _passages(*scores: float) -> tuple[ScoredChunk, ...]:
    """Passages whose text is ``passage <n>``, retrieved with the given scores."""
    return tuple(
        ScoredChunk(
            chunk=Chunk(chunk_id=f"c{n}", item=f"Item {n}", title="t", text=f"passage {n}"),
            score=score,
        )
        for n, score in enumerate(scores, start=1)
    )


def _brier_tape(root, *replies: tuple[int, object]) -> Tape:
    """A tape holding brier's reply to passage n of ``_passages`` as replies[n-1]."""
    tape = Tape.open(root)
    for position, (status, body) in enumerate(replies, start=1):
        call = {"path": BRIER_PATH, "body": brier_request(QUESTION, f"passage {position}")}
        key, _ = tape.lookup(BRIER, call)
        tape.record(BRIER, key, call, {"status": status, "body": body})
    tape.save({})
    return Tape.open(root)


def _yes(confidence: float) -> tuple[int, object]:
    return 200, {"answers": [{"probabilities": [1 - confidence, confidence]}]}


def test_brier_replays_the_most_confident_passage_and_admits_whatever_it_scored(tmp_path):
    tape = _brier_tape(tmp_path, _yes(0.3), _yes(0.95), _yes(0.0))

    decision = TapedBrierScores(tape).admits(QUESTION, _passages(0.8, 0.7, 0.6), EXAMPLE)

    assert decision.admitted is True
    assert decision.scores == {"brier": 0.95}
    assert decision.degraded is False


def test_a_recorded_confidence_of_zero_is_an_answer_not_a_missing_one(tmp_path):
    tape = _brier_tape(tmp_path, _yes(0.0))

    decision = TapedBrierScores(tape).admits(QUESTION, _passages(0.8), EXAMPLE)

    assert decision.scores == {"brier": 0.0}
    assert decision.degraded is False


@pytest.mark.parametrize(
    "failed",
    [(503, {"detail": "down"}), (200, {"answers": [{"probabilities": [0.5, 1.5]}]})],
    ids=["error status", "malformed probability"],
)
def test_a_recorded_failure_leaves_its_passage_unjudged_and_the_decision_degraded(tmp_path, failed):
    tape = _brier_tape(tmp_path, failed, _yes(0.4))

    decision = TapedBrierScores(tape).admits(QUESTION, _passages(0.8, 0.7), EXAMPLE)

    assert decision.scores == {"brier": 0.4}
    assert decision.degraded is True


def test_a_tape_with_no_valid_brier_reply_for_a_question_stops_the_replay(tmp_path):
    tape = _brier_tape(tmp_path, (503, {}))

    with pytest.raises(TapeMiss, match="no passage has a valid brier reply"):
        TapedBrierScores(tape).admits(QUESTION, _passages(0.8), EXAMPLE)


def test_a_passage_brier_never_scored_is_a_miss_never_a_call(tmp_path):
    tape = _brier_tape(tmp_path, _yes(0.9))

    with pytest.raises(TapeMiss, match="re-record locally"):
        TapedBrierScores(tape).admits("another question", _passages(0.8), EXAMPLE)


def test_brier_replays_nothing_retrieved_as_a_rejection_scored_zero(tmp_path):
    decision = TapedBrierScores(Tape.open(tmp_path)).admits(QUESTION, (), EXAMPLE)

    assert decision.admitted is False
    assert decision.scores == {"brier": 0.0}
