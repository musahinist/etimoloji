"""Hakasça sözlük tanıkları: ayrıştırma, süzgeçler, aday arama (veri gerekmez)."""

from __future__ import annotations

import json

import pytest

from engine.fetchers import khakas_dict
from engine.fetchers.khakas_dict import KhakasDictFetcher, _clean_gloss, _explanatory_entry


@pytest.fixture
def fake_data(tmp_path, monkeypatch):
    russian = [
        {"word": "кӱн", "semgloss": "солнце", "part": "NOMEN", "etym": "", "rest": "солнце"},
        {"word": "тас", "semgloss": "камень", "part": "NOMEN", "etym": "", "rest": "камень"},
        {"word": "килерге", "semgloss": "приходить", "part": "VERBUM", "etym": "", "rest": "приходить"},
        {"word": "тастирға", "semgloss": "бросать", "part": "VERBUM", "etym": "", "rest": "бросать"},
        {"word": "лента", "semgloss": "лента", "part": "NOMEN", "etym": "rus", "rest": "лента"},
        {"word": "Абакан", "semgloss": "Абакан", "part": "NOMEN", "etym": "", "rest": ""},
    ]
    explanatory = [
        {"headword_fix": "САС", "field_fix": "**САС** *адал.* Пас тӱгі. -- Волосы. *Узун сас.*"},
        {"headword_fix": "КӰН", "field_fix": "**КӰН** *адал.* Хызыл. -- День."},
    ]
    for name, rows in (("khakas_russian.jsonl", russian), ("khakas_explanatory.jsonl", explanatory)):
        (tmp_path / name).write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
        )
    monkeypatch.setattr(khakas_dict, "DATA_DIR", tmp_path)
    predicted = {"gel": {"khk": "kil"}}  # öğrenilmiş ses denkliklerinin yerine
    monkeypatch.setattr(khakas_dict, "_predicted_forms", lambda word: predicted.get(word, {}))
    khakas_dict._index.cache_clear()
    yield
    khakas_dict._index.cache_clear()


def _words(query: str) -> dict[str, dict]:
    return {e["word"]: e for e in KhakasDictFetcher().fetch(query)["turkic_languages"]}


def test_clean_gloss_strips_numbering_and_abbreviations():
    assert _clean_gloss("1) зоол. леопард // леопардовый; шкура") == "леопард"


def test_explanatory_entry_takes_russian_translation():
    head, gloss, is_verb = _explanatory_entry(
        {"headword_fix": "ІРЛЕНЕРГЕ \\[ірлен-\\]",
         "field_fix": "**ІРЛЕНЕРГЕ** *иділ.* Хырызарға. -- Браниться, ругать. *Örnek.*"}
    )
    assert (head, gloss, is_verb) == ("ірленерге", "браниться, ругать", True)


def test_candidates_are_local_and_meaning_checked(fake_data):
    found = _words("gün")
    assert "кӱн" in found
    entry = found["кӱн"]
    assert entry["lang_code"] == "khk" and entry["meaning_check"] is True
    assert KhakasDictFetcher.is_local and not KhakasDictFetcher.is_seed_source


def test_russian_loans_and_proper_names_are_dropped(fake_data):
    assert "лента" not in _words("lenta")
    assert not any(w.lower() == "абакан" for w in _words("abakan"))


def test_verbs_match_only_verbs_via_stem(fake_data):
    assert "килерге" in _words("gelmek")
    assert "тастирға" not in _words("taş")  # ad sorgusu fiil maddesine gitmez
    assert "тас" in _words("taş")


def test_explanatory_dictionary_fills_missing_headwords(fake_data):
    assert _words("saç")["сас"]["meaning"] == "волосы"


def test_no_data_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(khakas_dict, "DATA_DIR", tmp_path)
    khakas_dict._index.cache_clear()
    try:
        assert KhakasDictFetcher().fetch("gün")["turkic_languages"] == []
    finally:
        khakas_dict._index.cache_clear()
