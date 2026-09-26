"""
9o — biçim-öncelikli ikinci arama (``donor_proximity.DONOR_FORM_FIRST``).

Tanı (``data/cache/work/donor9o/PREREG.md``): kesin verici etiketi verilemeyen maddelerde doğru
etimon çoğu zaman sözlükte var ama anlam havuzuna girmiyor (sultan "A monarchic title …" ~ سلطان
"sultan"). Testler korur:

1. Aday yalnız a2'nin DİLİNDE aranır; biçim eşiği (c1 0,15, c3 0,20) ve geniş anlam sözcükleriyle
   en az bir ortak içerik sözcüğü şarttır; dilbilgisi göndermesi anlamları sayılmaz.
2. Önsel Arapça, eşleşen Farsça ise aynı Arap yazısı iskeletli Arapça madde gösterilir (``via="fa"``).
3. ``honest_label``: bulunan biçim kesin etiket olur, dili a2'ninki; anlam havuzu boşsa
   (``attribution`` None) yalnız biçim-öncelikli arama etiket verir; kapalıyken davranış 9n ile aynı.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from engine.db import donor_skeleton, sense_bridge
from engine.nlp import donor_prior
from engine.nlp import donor_proximity as dp

ROWS = [
    {"id": 1, "lang_code": "ar", "word": "سلطان", "comparison": "sultan", "gloss": "sultan"},
    {"id": 2, "lang_code": "fr", "word": "sultane", "comparison": "sultane", "gloss": "sultana, wife of a sultan"},
    {"id": 3, "lang_code": "ar", "word": "صلى", "comparison": "salla", "gloss": "to pray"},
    {"id": 4, "lang_code": "fa", "word": "معصوم", "comparison": "masum", "gloss": "innocent"},
    {"id": 5, "lang_code": "ar", "word": "مَعْصُوم", "comparison": "masum", "gloss": "infallible, sinless"},
    {"id": 6, "lang_code": "ar", "word": "معصومة", "comparison": "masuma", "gloss": "innocent (feminine)"},
    {"id": 7, "lang_code": "ar", "word": "مسوم", "comparison": "musim", "gloss": "genitive plural of x"},
]
DIST = {("sultan", "sultan"): 0.0, ("sultan", "sultane"): 0.08, ("sultan", "salla"): 0.40,
        ("masum", "masum"): 0.0, ("masum", "masuma"): 0.18, ("masum", "musim"): 0.05}


@pytest.fixture()
def env():
    bridge = {"sultan": "sultan"}
    with mock.patch.object(donor_skeleton, "by_skeleton", lambda *a, **k: [dict(r) for r in ROWS]), \
            mock.patch.object(dp, "_index", lambda: SimpleNamespace(exists=True, path="x")), \
            mock.patch.object(dp, "_pairwise", lambda: object()), \
            mock.patch.object(dp, "_null_distance", lambda *a: 0.5), \
            mock.patch.object(dp, "label_distance", lambda a, b: DIST.get((a, b), 0.9)), \
            mock.patch.object(dp, "SENSE_BRIDGE", False), \
            mock.patch.object(sense_bridge, "english_sense", lambda c, *a: bridge.get(c, "")), \
            mock.patch.object(sense_bridge, "ottoman_sense", lambda c: ""):
        dp._word_bridge.cache_clear()
        yield bridge
    dp._word_bridge.cache_clear()


def test_finds_etymon_in_prior_language(env):
    a = dp.form_first_attribution("sultan", "A monarchic title for Sunni Muslim monarchs.", "ar", "c1")
    assert a is not None and a.word == "سلطان" and a.lang_code == "ar" and a.distance == 0.0 and a.via == ""
    fr = dp.form_first_attribution("sultan", "A monarchic title for Sunni Muslim monarchs.", "fr", "c1")
    assert fr is not None and fr.word == "sultane"
    assert dp.form_first_attribution("sultan", "A monarchic title.", "fa", "c1") is None


def test_needs_shared_content_word(env):
    env.clear()  # başlık köprüsü yok: "monarchic title" ile "sultan" ortak sözcük taşımaz
    assert dp.form_first_attribution("sultan", "A monarchic title for Sunni Muslim monarchs.", "ar", "c1") is None


def test_off_and_threshold(env):
    assert dp.form_first_attribution("sultan", "sultan", "ar", "off") is None
    # masuma (0,18) c1 eşiğinin (0,15) dışında, c3'ün (0,20) içinde; dilbilgisi göndermesi (musim) sayılmaz
    assert dp.form_first_attribution("masum", "innocent", "ar", "c1").word == "مَعْصُوم"


def test_persian_twin_to_arabic(env):
    a = dp.form_first_attribution("masum", "innocent person", "ar", "c1")
    # Farsça معصوم "innocent" anlamla eşleşir; Arapça aynı iskeletli madde gösterilir
    assert a.lang_code == "ar" and a.word == "مَعْصُوم" and a.via == "fa"


def test_threshold_c3(env):
    rows = [r for r in ROWS if r["id"] == 6]
    with mock.patch.object(donor_skeleton, "by_skeleton", lambda *a, **k: [dict(r) for r in rows]):
        assert dp.form_first_attribution("masum", "innocent", "ar", "c1") is None
        assert dp.form_first_attribution("masum", "innocent", "ar", "c3").word == "معصومة"


def test_turkish_sense_word_bridge_only_in_c2(env):
    env["masum"] = ""
    env["suçsuz"] = "innocent"
    rows = [r for r in ROWS if r["id"] == 4]
    with mock.patch.object(donor_skeleton, "by_skeleton", lambda *a, **k: [dict(r) for r in rows]):
        assert dp.form_first_attribution("masum", "suçsuz kimse", "fa", "c1") is None
        assert dp.form_first_attribution("masum", "suçsuz kimse", "fa", "c2").word == "معصوم"


def _att(**kw):
    base = {"lang_code": "fr", "word": "question", "comparison": "kuestion", "gloss": "question",
            "distance": 0.38, "null_distance": 0.4, "chance_percentile": 0.3, "sense_overlap": 0}
    base.update(kw)
    return dp.DonorAttribution(**base)


def test_honest_label_uses_form_first(env):
    sense = "A monarchic title for Sunni Muslim monarchs."
    with mock.patch.object(donor_prior, "posterior", return_value={"ar": 0.8, "fr": 0.15, "fa": 0.05}):
        off = dp.honest_label(_att(), "sultan", "a2", sense=sense, form_first="off")
        on = dp.honest_label(_att(), "sultan", "a2", sense=sense, form_first="c1")
        none_on = dp.honest_label(None, "sultan", "a2", sense=sense, form_first="c1")
        none_off = dp.honest_label(None, "sultan", "a2", sense=sense, form_first="off")
        no_sense = dp.honest_label(_att(), "sultan", "a2", form_first="c1")
    assert not off.certain and off.form is None
    assert on.certain and on.lang_code == "ar" and on.form.word == "سلطان" and on.basis == "biçim-öncelikli"
    assert "سلطان" in on.describe(_att()) and "question" not in on.describe(_att())
    assert none_on is not None and none_on.form.word == "سلطان"
    assert none_off is None
    assert not no_sense.certain  # anlam verilmezse (eski çağrı) 9n davranışı


def test_certain_attribution_untouched(env):
    a = _att(lang_code="ar", word="سلطان", comparison="sultan", distance=0.0, chance_percentile=0.0, sense_overlap=1)
    h = dp.honest_label(a, "sultan", "a2", sense="sultan", form_first="c3")
    assert h.certain and h.form is None and h.describe(a) == a.describe()


def test_detector_shows_form_first_even_without_sense_pool(env):
    from engine.nlp import borrowing_detector as bd

    match = dp.DonorMatch("ar", "صلى", "salla", "to pray", 0.30, True, None)
    sense = "A monarchic title for Sunni Muslim monarchs."
    with mock.patch.object(bd, "nearest_donor", lambda *a, **k: match), \
            mock.patch.object(bd, "attribute_donor", lambda *a, **k: None), \
            mock.patch.object(donor_prior, "posterior", return_value={"ar": 0.8, "fr": 0.15, "fa": 0.05}), \
            mock.patch.object(dp, "DONOR_HONEST", "a2"):
        with mock.patch.object(dp, "DONOR_FORM_FIRST", "c1"):
            on = bd.BorrowingDetector._donor_signal("sultan", sense, list(bd.TURKISH_DONORS))
            other = bd.BorrowingDetector._donor_signal("sultan", sense, ["ru", "mn"])
        off = bd.BorrowingDetector._donor_signal("sultan", sense, list(bd.TURKISH_DONORS))
    assert "سلطان" in on.explanation and "biçim-öncelikli" in on.explanation
    assert on.evidence["attributed_lang"] == "ar" and on.evidence["attribution"]["donor_certain"] is True
    assert "attribution" not in off.evidence and "attribution" not in other.evidence
    assert on.strength == off.strength  # güç değişmez
