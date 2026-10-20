from __future__ import annotations

import json

import pytest

from tools.convert.resources import load_resources, token_domain


def test_final_resources_override_defaults_without_hash_pinning(tmp_path):
    (tmp_path / "tokenizer.json").write_text("invalid default")
    selected = tmp_path / "custom-tokenizer.json"
    selected.write_text(
        json.dumps({"model": {"type": "BPE", "vocab": {"a": 0, "b": 1}}})
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {"added_tokens_decoder": {"2": {"content": "<end>", "special": True}}}
        )
    )
    (tmp_path / "generation_config.json").write_text(
        '{"eos_token_id":2,"temperature":1.0}'
    )
    template = tmp_path / "custom.jinja"
    template.write_text("custom {{ messages }}")
    references, payloads, count, special = load_resources(
        tmp_path,
        vocab_size=4,
        overrides={"tokenizer.json": selected, "chat_template.jinja": template},
    )
    assert count == 3 and special == (2,)
    assert set(references) == {"text"}
    assert payloads[references["text"]["chat_template.jinja"]] == template.read_bytes()
    assert (
        payloads[references["text"]["generation_config.json"]]
        == (tmp_path / "generation_config.json").read_bytes()
    )


def test_special_tokens_merge_both_resources_with_consistent_flags():
    tokenizer = {
        "model": {"vocab": {"a": 0}},
        "added_tokens": [{"id": 1, "content": "<start>", "special": True}],
    }
    config = {"added_tokens_decoder": {"2": {"content": "<end>", "special": True}}}
    assert token_domain(tokenizer, config, 4) == (3, (1, 2))
    config["added_tokens_decoder"]["1"] = {"content": "<start>", "special": False}
    with pytest.raises(ValueError, match="special flag"):
        token_domain(tokenizer, config, 4)


def _bundled_checkpoint(path):
    image = {
        "patch_size": 16,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "size": {"shortest_edge": 65536, "longest_edge": 16777216},
        "image_mean": [0.5, 0.5, 0.5],
    }
    video = {
        **image,
        "fps": 2,
        "size": {"shortest_edge": 4096, "longest_edge": 25165824},
    }
    for name, value in {
        "tokenizer.json": {
            "model": {"vocab": {"a": 0, "b": 1}},
            "added_tokens": [{"id": 2, "content": "<end>", "special": True}],
        },
        "tokenizer_config.json": {"eos_token": "<end>"},
        "config.json": {
            "bos_token_id": None,
            "eos_token_id": 2,
            "pad_token_id": 0,
            "text_config": {"eos_token_id": 0},
        },
        "processor_config.json": {"image_processor": image, "video_processor": video},
    }.items():
        (path / name).write_text(json.dumps(value))
    (path / "chat_template.jinja").write_text("{{ messages }}")
    return image, video


def test_bundled_processors_and_outer_generation_defaults(tmp_path):
    image, video = _bundled_checkpoint(tmp_path)
    refs, payloads, count, special = load_resources(
        tmp_path,
        vocab_size=3,
        vision_config={"patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2},
    )
    assert count == 3 and special == (2,)
    assert json.loads(payloads[refs["vision"]["preprocessor_config.json"]]) == image
    assert json.loads(payloads[refs["vision"]["video_preprocessor_config.json"]]) == video
    assert json.loads(payloads[refs["text"]["generation_config.json"]]) == {
        "bos_token_id": None,
        "eos_token_id": 2,
        "pad_token_id": 0,
    }
    assert not (tmp_path / "generation_config.json").exists()
    assert not (tmp_path / "preprocessor_config.json").exists()


def test_explicit_processor_and_stop_resources_take_precedence(tmp_path):
    image, _ = _bundled_checkpoint(tmp_path)
    selected = tmp_path / "selected.json"
    selected.write_text(
        json.dumps({**image, "size": {"shortest_edge": 2048, "longest_edge": 4096}})
    )
    stop = b'{"eos_token_id":[1,2],"temperature":0.0}\n'
    (tmp_path / "generation_config.json").write_bytes(stop)
    refs, payloads, _, _ = load_resources(
        tmp_path,
        vocab_size=3,
        vision_config={"patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2},
        overrides={"preprocessor_config.json": selected},
    )
    assert payloads[refs["vision"]["preprocessor_config.json"]] == selected.read_bytes()
    assert payloads[refs["text"]["generation_config.json"]] == stop
    with pytest.raises(FileNotFoundError):
        load_resources(
            tmp_path,
            vocab_size=3,
            vision_config={"patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2},
            overrides={"preprocessor_config.json": tmp_path / "missing.json"},
        )


def test_derived_generation_defaults_require_an_explicit_model_eos(tmp_path):
    _bundled_checkpoint(tmp_path)
    (tmp_path / "config.json").write_text('{"text_config":{"eos_token_id":0}}')
    with pytest.raises(ValueError, match="missing eos_token_id"):
        load_resources(tmp_path, vocab_size=3)


def test_bundled_processor_geometry_is_validated(tmp_path):
    _bundled_checkpoint(tmp_path)
    with pytest.raises(ValueError, match="patch_size differs"):
        load_resources(
            tmp_path,
            vocab_size=3,
            vision_config={"patch_size": 32, "temporal_patch_size": 2, "spatial_merge_size": 2},
        )
