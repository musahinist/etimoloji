"""
9n — şans düzeyindeki verici etiketinin dürüst gösterimi.

Tanı (``data/cache/work/donor9n/PREREG.md``): Türkçe etiketlerin önemli kısmında gösterilen verici
biçimi doğru etimon değil şans eşleşmesi (sultan ~ صلى, kudret ~ قطر). Testler korur:

1. Kesinlik kuralı: mesafe ≤ 0,25 + şans yüzdeliği 0 + ortak içerik sözcüğü, ya da ünsüz
   iskeleti eşi (≤ 0,35); anlam yalnız işlev sözcüğüyle eşleştiyse hiçbir zaman kesin değil.
2. Kesin olmayan etikette biçim gösterilmez; a1 aile, a2 önsel+ipucu tek dil, a3 önsel-aile.
3. ``off`` = eski davranış (her etiket kesin, biçim gösterilir).
4. Önsel sonsalı olasılık dağılımıdır; model yoksa yalnız doğal önsel (Arapça en sık).
"""

from __future__ import annotations

import math
from unittest import mock

import pytest

from engine.nlp import donor_prior
from engine.nlp import donor_proximity as dp


def _att(**kw):
    base = {"lang_code": "ar", "word": "صلى", "comparison": "salla", "gloss": "to pray", "distance": 0.36,
            "null_distance": 0.4, "chance_percentile": 0.25, "sense_overlap": 1, "skeleton_match": False}
    base.update(kw)
    return dp.DonorAttribution(**base)


def test_certain_rule():
    assert dp.attribution_certain(_att(distance=0.1, chance_percentile=0.0))
    assert not dp.attribution_certain(_att(distance=0.1, chance_percentile=0.0, sense_overlap=0))
    assert not dp.attribution_certain(_att(distance=0.1, chance_percentile=0.1))
    assert not dp.attribution_certain(_att(distance=0.3, chance_percentile=0.0))
    assert not dp.attribution_certain(_att(distance=0.1, chance_percentile=None))
    # iskelet eşi: şans yüzdeliğinden bağımsız, 0,35'e kadar
    assert dp.attribution_certain(_att(distance=0.34, skeleton_match=True))
    assert not dp.attribution_certain(_att(distance=0.36, skeleton_match=True))
    assert not dp.attribution_certain(None)


def test_off_keeps_form():
    h = dp.honest_label(_att(), "sultan", "off")
    assert h.certain and h.show_form and h.lang_code == "ar" and h.code == "ar"
    assert h.describe(_att()) == _att().describe()


def test_a1_family_without_form():
    h = dp.honest_label(_att(), "sultan", "a1")
    assert not h.show_form and h.family == ("ar", "fa") and h.code == "ar|fa"
    text = h.describe(_att())
    assert "Arapça ya da Farsça" in text and "صلى" not in text and "yakın biçim bulunamadı" in text
    h = dp.honest_label(_att(lang_code="it"), "kordon", "a1")
    assert h.family == ("fr", "it") and h.text().startswith("Batı dili")


def test_a2_prior_single_language_without_form():
    with mock.patch.object(donor_prior, "posterior", return_value={"fr": 0.7, "ar": 0.2, "fa": 0.1}):
        h = dp.honest_label(_att(), "sultan", "a2")
    assert not h.show_form and h.lang_code == "fr" and h.basis == "önsel"
    text = h.describe(_att())
    assert "muhtemelen Fransızca" in text and "0,70" in text and "صلى" not in text


def test_a3_family_by_posterior_mass():
    with mock.patch.object(donor_prior, "posterior", return_value={"fr": 0.45, "ar": 0.30, "fa": 0.25}):
        h = dp.honest_label(_att(), "sultan", "a3")
    assert h.family == ("ar", "fa") and math.isclose(h.probability, 0.55)


def test_certain_label_keeps_form_in_every_mode():
    a = _att(distance=0.05, chance_percentile=0.0)
    for mode in ("a1", "a2", "a3"):
        h = dp.honest_label(a, "sultan", mode)
        assert h.certain and h.lang_code == "ar" and h.describe(a) == a.describe()


def test_unknown_mode():
    with pytest.raises(ValueError):
        dp.honest_label(_att(), "sultan", "zz")


def test_posterior_is_distribution_and_prior_without_model():
    with mock.patch.object(donor_prior, "load_model", return_value=None):
        post = donor_prior.posterior("sultan", ["ar", "fa", "fr"])
    assert math.isclose(sum(post.values()), 1.0)
    assert max(post, key=post.get) == "ar"
    total = sum(donor_prior.NATURAL_DONOR_COUNTS[k] for k in ("ar", "fa", "fr"))
    assert math.isclose(post["fr"], donor_prior.NATURAL_DONOR_COUNTS["fr"] / total)


def test_cue_model_prefers_french_suffix():
    model = donor_prior.CueModel({
        "ar": {"": {"a": 5, "k": 3, "$": 2}, "#": {"m": 2}, "##": {"m": 2}},
        "fr": {"": {"o": 5, "n": 3, "$": 2, "s": 3, "y": 3}, "#": {"s": 2}, "##": {"s": 2},
               "yo": {"n": 5}, "on": {"$": 5}},
    })
    with mock.patch.object(donor_prior, "load_model", return_value=model):
        assert donor_prior.best("syon", ["ar", "fr"], 1.0)[0] == "fr"


def test_detector_hides_chance_form_only_for_turkish_donor_set():
    from engine.nlp import borrowing_detector as bd

    match = dp.DonorMatch("ar", "صلى", "salla", "to pray", 0.30, True, None)
    label = _att()
    with mock.patch.object(bd, "nearest_donor", lambda *a, **k: match), \
            mock.patch.object(bd, "attribute_donor", lambda *a, **k: label), \
            mock.patch.object(dp, "DONOR_HONEST", "a1"):
        tr = bd.BorrowingDetector._donor_signal("sultan", "sultan", list(bd.TURKISH_DONORS))
        other = bd.BorrowingDetector._donor_signal("sultan", "sultan", ["ru", "mn"])
    assert "صلى" not in tr.explanation and "Arapça ya da Farsça" in tr.explanation
    assert tr.evidence["attributed_lang"] == "ar|fa" and tr.evidence["attribution"]["donor_certain"] is False
    assert "صلى" in other.explanation and other.evidence["attributed_lang"] == "ar"
    with mock.patch.object(bd, "nearest_donor", lambda *a, **k: match), \
            mock.patch.object(bd, "attribute_donor", lambda *a, **k: label), \
            mock.patch.object(dp, "DONOR_HONEST", "off"):
        off = bd.BorrowingDetector._donor_signal("sultan", "sultan", list(bd.TURKISH_DONORS))
    assert "صلى" in off.explanation and off.evidence["attributed_lang"] == "ar"
    assert tr.strength == off.strength  # güç değişmez
