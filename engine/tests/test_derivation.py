"""Türetme çözümleyicisi: şablon dili ve sözlükte doğrulanmış çözümler."""

import pytest

from engine.nlp import derivation as dv
from engine.nlp import root_variants as rv


@pytest.mark.parametrize(
    ("template", "prev", "before_vowel", "expected"),
    [
        ("lI~k", "güzel", False, "lik"),
        ("lI~k", "kul", True, "luğ"),
        (">cI", "kitap", False, "çı"),
        (">cI", "süt", False, "çü"),
        (">cI", "yol", False, "cu"),
        ("+Im", "kazı", False, "m"),
        ("+Im", "sür", False, "üm"),
        (">gIn", "küs", False, "kün"),
        (">gAn", "çalış", False, "kan"),
        ("+A~k", "dur", False, "ak"),
    ],
)
def test_surface(template, prev, before_vowel, expected):
    assert dv.surface(template, prev, before_vowel=before_vowel) == expected


needs_lexicon = pytest.mark.skipif(
    not (rv.LEXICON_DIR / "master-dictionary.dict").is_file(), reason="Zemberek sözlüğü yok"
)


@needs_lexicon
@pytest.mark.parametrize(
    ("word", "formula"),
    [
        ("güzellik", "güzel + +lIk"),
        ("kitapçı", "kitap + +CI"),
        ("salgın", "sal + -gIn"),
        ("çalışkan", "çalış + -gAn"),
        ("kolaylaşmak", "kolay + +lAş"),
    ],
)
def test_analyze_finds(word, formula):
    assert formula in [a.formula for a in dv.analyze(word)]


@needs_lexicon
def test_pos_gate_and_order():
    # -Im fiil kökü ister: `adım` -> ad (ad) + -Im üretilmez.
    assert not any(a.root == "ad" for a in dv.analyze("adım"))
    # Sığ çözüm önce: `sevgili` -> sevgi + +lI, sonra sev + -gI + +lI.
    assert [a.formula for a in dv.analyze("sevgili")][:2] == ["sevgi + +lI", "sev + -gI + +lI"]
