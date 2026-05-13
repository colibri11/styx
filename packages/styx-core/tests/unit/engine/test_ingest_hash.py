"""Unit-тесты для ingest_hash (волна 23).

Pure функции ``canonicalize`` + ``compute_content_hash`` +
``is_content_ref_empty``. Ключевое требование — детерминизм:
канонизация должна давать одинаковый hash для семантически
одинаковых payload'ов независимо от порядка ключей.
"""

from __future__ import annotations

import hashlib

import pytest

from styx.engine.ingest_hash import (
    canonicalize,
    compute_content_hash,
    is_content_ref_empty,
)


# ── canonicalize: примитивы ──────────────────────────────────────────


def test_canonicalize_none() -> None:
    assert canonicalize(None) == "null"


def test_canonicalize_bool() -> None:
    assert canonicalize(True) == "true"
    assert canonicalize(False) == "false"


def test_canonicalize_int() -> None:
    assert canonicalize(42) == "42"
    assert canonicalize(-7) == "-7"
    assert canonicalize(0) == "0"


def test_canonicalize_float() -> None:
    assert canonicalize(1.5) == "1.5"


def test_canonicalize_string() -> None:
    assert canonicalize("hello") == '"hello"'


def test_canonicalize_string_with_unicode() -> None:
    # ensure_ascii=False — кириллица сохраняется как есть.
    assert canonicalize("привет") == '"привет"'


def test_canonicalize_string_with_quotes() -> None:
    assert canonicalize('say "hi"') == '"say \\"hi\\""'


# ── canonicalize: контейнеры ─────────────────────────────────────────


def test_canonicalize_empty_list() -> None:
    assert canonicalize([]) == "[]"


def test_canonicalize_list_preserves_order() -> None:
    # Список — не сортируется (порядок значим).
    assert canonicalize([3, 1, 2]) == "[3,1,2]"


def test_canonicalize_empty_dict() -> None:
    assert canonicalize({}) == "{}"


def test_canonicalize_dict_sorts_keys() -> None:
    # Семантически одинаковые объекты с разным порядком ключей →
    # одинаковая каноническая строка.
    a = canonicalize({"b": 1, "a": 2})
    b = canonicalize({"a": 2, "b": 1})
    assert a == b == '{"a":2,"b":1}'


def test_canonicalize_filters_none_values_from_dict() -> None:
    # None-значения отфильтровываются — иначе {} и {"x": None} дали бы
    # разные hash'и, что ломает идемпотентность.
    assert canonicalize({"x": None}) == "{}"
    assert canonicalize({"a": 1, "b": None, "c": 2}) == '{"a":1,"c":2}'


def test_canonicalize_nested_dict_in_list() -> None:
    out = canonicalize([{"b": 1, "a": 2}, {"d": 3, "c": 4}])
    assert out == '[{"a":2,"b":1},{"c":4,"d":3}]'


def test_canonicalize_deeply_nested() -> None:
    payload = {"outer": {"inner": [{"k": "v"}]}}
    assert canonicalize(payload) == '{"outer":{"inner":[{"k":"v"}]}}'


def test_canonicalize_unknown_type_returns_null() -> None:
    # Memorybox parity: неожиданный input не кидает, а даёт "null"
    # чтобы hash оставался детерминированным.
    class Custom:
        pass

    assert canonicalize(Custom()) == "null"


# ── compute_content_hash ─────────────────────────────────────────────


def test_compute_hash_returns_hex_sha256() -> None:
    h = compute_content_hash(
        pipeline_id="audiobox",
        pipeline_version="v1.0",
        content_ref={"file_path": "/recordings/a.wav"},
    )
    # sha256 hex — 64 символа [0-9a-f].
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_compute_hash_deterministic() -> None:
    args = {
        "pipeline_id": "audiobox",
        "pipeline_version": "v1.0",
        "content_ref": {"file_path": "/a.wav", "size": 1024},
    }
    h1 = compute_content_hash(**args)
    h2 = compute_content_hash(**args)
    assert h1 == h2


def test_compute_hash_independent_of_dict_key_order() -> None:
    # Семантически одинаковый payload с разным порядком ключей
    # content_ref → одинаковый hash. Это фундамент идемпотентности.
    h1 = compute_content_hash(
        pipeline_id="p",
        pipeline_version="v1",
        content_ref={"file_path": "/a", "url": "http://b"},
    )
    h2 = compute_content_hash(
        pipeline_id="p",
        pipeline_version="v1",
        content_ref={"url": "http://b", "file_path": "/a"},
    )
    assert h1 == h2


def test_compute_hash_changes_with_pipeline_version() -> None:
    args = {"pipeline_id": "p", "content_ref": {"file_path": "/a"}}
    h1 = compute_content_hash(pipeline_version="v1", **args)
    h2 = compute_content_hash(pipeline_version="v2", **args)
    assert h1 != h2


def test_compute_hash_changes_with_content_ref() -> None:
    args = {"pipeline_id": "p", "pipeline_version": "v1"}
    h1 = compute_content_hash(content_ref={"file_path": "/a"}, **args)
    h2 = compute_content_hash(content_ref={"file_path": "/b"}, **args)
    assert h1 != h2


def test_compute_hash_changes_with_pipeline_id() -> None:
    args = {"pipeline_version": "v1", "content_ref": {"file_path": "/a"}}
    h1 = compute_content_hash(pipeline_id="audiobox", **args)
    h2 = compute_content_hash(pipeline_id="textbox", **args)
    assert h1 != h2


def test_compute_hash_matches_manual_sha256() -> None:
    # Защита от изменения формы canonicalisation: явно строим
    # ожидаемую каноническую строку и hash'им вручную.
    payload = {
        "pipeline_id": "p",
        "pipeline_version": "v1",
        "content_ref": {"file_path": "/a"},
    }
    canonical = canonicalize(payload)
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    actual = compute_content_hash(
        pipeline_id="p",
        pipeline_version="v1",
        content_ref={"file_path": "/a"},
    )
    assert actual == expected


# ── is_content_ref_empty ─────────────────────────────────────────────


def test_is_content_ref_empty_none() -> None:
    assert is_content_ref_empty(None) is True


def test_is_content_ref_empty_dict() -> None:
    assert is_content_ref_empty({}) is True


def test_is_content_ref_all_none_values() -> None:
    # zod-style {} с явными undefined полями — семантически пусто.
    assert is_content_ref_empty({"file_path": None, "url": None}) is True


def test_is_content_ref_with_value() -> None:
    assert is_content_ref_empty({"file_path": "/a"}) is False
    assert is_content_ref_empty({"x": 0}) is False
    # Пустая строка — это значение, не отсутствие.
    assert is_content_ref_empty({"file_path": ""}) is False


# ── property: hash + canonicalize composition ───────────────────────


@pytest.mark.parametrize(
    "content_ref",
    [
        {"file_path": "/a"},
        {"url": "http://example.com", "size": 1024},
        {"inline_text": "hello"},
        {"a": [1, 2, 3], "b": {"nested": True}},
        {"deep": {"deeper": {"deepest": [1, "two", None, False]}}},
    ],
)
def test_compute_hash_stable_across_calls(content_ref: dict) -> None:
    h1 = compute_content_hash(
        pipeline_id="audio",
        pipeline_version="v2.1",
        content_ref=content_ref,
    )
    h2 = compute_content_hash(
        pipeline_id="audio",
        pipeline_version="v2.1",
        content_ref=content_ref,
    )
    assert h1 == h2
