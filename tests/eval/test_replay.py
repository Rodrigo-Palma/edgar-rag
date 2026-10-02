import json

import httpx
import numpy as np
import pytest

from edgar_rag.domain import EmbedderSpec, Generation, NotRecorded
from edgar_rag.eval.replay import (
    BRIER,
    EMBED,
    GENERATE,
    Tape,
    TapedEmbedder,
    TapedGenerator,
    TapedTransport,
    TapeMiss,
    tape_key,
)
from tests.fakes import FakeEmbedder, FakeGenerator

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


def _brier(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"answers": [{"probabilities": [0.2, 0.8]}]})


def test_brier_replays_from_the_body_whatever_the_host(tmp_path):
    tape = Tape.open(tmp_path)
    body = {"state": "passage", "questions": [{"name": "relevance"}]}
    with httpx.Client(transport=TapedTransport(tape, live=httpx.MockTransport(_brier))) as client:
        recorded = client.post("http://127.0.0.1:8100/decide", json=body).json()
    tape.save({})

    replay = TapedTransport(Tape.open(tmp_path))
    with httpx.Client(transport=replay) as client:
        replayed = client.post("http://elsewhere:9/decide", json=body)
        assert replayed.json() == recorded
        with pytest.raises(TapeMiss, match="re-record locally"):
            client.post("http://elsewhere:9/decide", json={**body, "state": "other"})
    assert Tape.open(tmp_path).has(BRIER)
