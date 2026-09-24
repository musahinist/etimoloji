"""Zemberek kök varyantları: öznitelik ayrıştırma, çıkarım ve iki yön.

Kural mantığı sentetik maddelerle sınanır (sözlük gerekmez); gerçek
sözlükle koşan denetim ``data/zemberek/master-dictionary.dict`` yoksa atlanır.
"""

from __future__ import annotations

import pytest

from engine.nlp import root_variants as rv
from engine.nlp.root_variants import _item, _parse_line, modified_root


def _modified(line: str, **kw) -> str:
    item = _item(*_parse_line(line))
    assert item is not None
    return modified_root(item, **kw)


def test_parse_line_reads_pos_and_attributes():
    assert _parse_line("hak [P:Noun; A:Doubling, InverseHarmony ; Index:1]") == (
        "hak", "Noun", {"Doubling", "InverseHarmony"})
    assert _parse_line("kitap") == ("kitap", "", set())
    assert _parse_line("# yorum") is None


@pytest.mark.parametrize("line, expected", [
    ("kitap", "kitab"),                        # çok heceli p/ç/t/k: varsayılan yumuşama
    ("at", "at"),                              # tek heceli: yumuşamaz
    ("tat [A:Voicing]", "tad"),                # tek heceli ama sözlükte işaretli
    ("ahenk", "aheng"),                        # nk -> ng
    ("psikolog", "psikoloğ"),                  # og -> oğ
    ("saat [A:InverseHarmony, NoVoicing]", "saat"),
    ("hak [A:Doubling]", "hakk"),
    ("ret [A:Voicing, Doubling]", "redd"),
    ("burun [A:LastVowelDrop]", "burn"),
    ("kayıp [A:Voicing, LastVowelDrop]", "kayb"),
    ("ayırmak [A:LastVowelDrop]", "ayr"),      # fiil: mastarsız gövde
    ("ağlamak", "ağl"),                        # fiil ünlüyle bitiyor: -Iyor düşmesi
])
def test_modified_root_follows_zemberek(line, expected):
    assert _modified(line) == expected


def test_progressive_drop_can_be_left_out():
    assert _modified("ağlamak", progressive=False) == "ağla"


def test_general_rules_without_lexicon(monkeypatch):
    monkeypatch.setattr(rv, "_lexicon", lambda: ({}, {}))
    assert rv.surface_variants("kitap") == ["kitap", "kitab"]
    assert rv.surface_variants("kavun") == ["kavun"]          # ünlü düşmesi tahmin edilmez
    assert rv.surface_variants("at") == ["at"]
    assert rv.root_candidates("kitab") == ["kitap"]
    assert rv.root_candidates("aheng") == ["ahenk"]
    assert rv.root_candidates("burn") == []


@pytest.mark.skipif(not (rv.LEXICON_DIR / "master-dictionary.dict").is_file(),
                    reason="Zemberek sözlüğü indirilmemiş")
def test_real_lexicon_both_directions():
    assert rv.surface_variants("burun") == ["burun", "burn"]
    assert rv.surface_variants("ağız") == ["ağız", "ağz"]
    assert rv.surface_variants("hak") == ["hak", "hakk"]
    assert "kitab" in rv.surface_variants("kitap")
    assert rv.root_candidates("burn") == ["burun"]
    assert rv.root_candidates("ağz") == ["ağız"]
    assert rv.root_candidates("hakk") == ["hak"]
    assert rv.root_candidates("kitab") == ["kitap"]
    # Şimdiki zaman düşmesi ters yönde kullanılmaz (`ad` -> `ada-` olmaz).
    assert rv.root_candidates("ad") == []
