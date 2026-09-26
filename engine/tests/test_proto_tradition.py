"""Gelenekten bağımsız eşdeğerlik anahtarı (Wiktionary ↔ Starling/EDAL)."""

from __future__ import annotations

import pytest

from engine.utils.proto_notation import normalize_proto, same_root_across_traditions, tradition_key


@pytest.mark.parametrize(
    ("wiktionary", "starling"),
    [
        ("*tāt-", "*dāt-"),        # söz başı Oğuz *d- (EDAL) ~ *t- (Wiktionary)
        ("*tāš", "*tāĺ"),          # trk-cmn-pro *š ~ EDAL *ĺ
        ("*tālᶴ", "*tāĺ"),         # trk-pro *lᶴ ~ EDAL *ĺ
        ("*tuz", "*tūŕ"),          # trk-cmn-pro *z ~ EDAL *ŕ, uzunluk
        ("*benʸi", "*bẹńi"),       # Wiktionary kılavuzundaki örnek
        ("*āl", "*ăl"),            # makron ~ breve
        ("*kel-", "*käl-"),        # *e ~ *ä
        ("*yol", "*jol"),
        ("*adak", "*aδak"),
        ("*kapuk", "*Kāpuk"),
    ],
)
def test_same_root_different_notation(wiktionary: str, starling: str) -> None:
    assert same_root_across_traditions(wiktionary, starling)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("*bït", "*büt"),          # ünlü farkı gösterim değil, iddia farkıdır
        ("*ar", "*at"),
        ("*kara", "*kar"),         # türemiş/kısaltılmış biçim
        ("*bāš", "*pāš"),          # *b-/*p- iki gelenekte de *b-: tabloda yok
        ("*ada", "*ata"),          # d/t yalnız SÖZ BAŞINDA eşitlenir
    ],
)
def test_different_roots_stay_apart(a: str, b: str) -> None:
    assert not same_root_across_traditions(a, b)


def test_breve_only_on_vowels() -> None:
    assert tradition_key("*ağ") == "ağ"          # ğ'nin breve'si korunur
    assert tradition_key("*ă") == "a"


def test_superset_of_exact_normalization() -> None:
    for a, b in [("*jeŋi", "*yeŋi"), ("*kemük", "*kemük-"), ("*ḳara", "*qara")]:
        assert normalize_proto(a) == normalize_proto(b)
        assert same_root_across_traditions(a, b)


def test_empty() -> None:
    assert not same_root_across_traditions("*", "")
