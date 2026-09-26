"""Clauson 1972 EDT: TurkicWorld HTML ayrıştırıcısı ve yerel tanık yardımcıları.

Ayrıştırıcı sentetik (kısaltılmış, uydurma gövdeli) paragraflarla sınanır;
indirilmiş veri gerekmez. Gerçek veriyle koşan denetim
``data/clauson/clauson_edt.jsonl`` yoksa atlanır.
"""

from __future__ import annotations

import pytest

from engine.db import clauson
from engine.db.clauson import attestations, parse_html, split_headword
from engine.fetchers import clauson_edt as ce

BLUE = '<font color="#0000FF"><i>({})</i></font>'

HTML = "<html><body>" + "".join(f"<p>{p}</p>" for p in [
    "<b>Dis. ŠNS</b>",
    "D <b>taŋsuk</b> Den. N./A. fr. 2 <b>taŋ</b>; ‘wonderful, rare’. Xak. xı taŋsuk Kaš. III 382: KB 100.",
    f"D <b>savur-</b> {BLUE.format('scatter')} Caus. f. of <b>sav-</b> {BLUE.format('turn away')}; "
    "‘to scatter’. Xak. xı Kaš. II 1.",
    "<b>taŋna-</b>; SW Osm. taŋla- (devam paragrafı).",
    "S <b>čaput</b> See <b>čapğut</b>.",
    f"?D <b>čapğut</b> {BLUE.format('rag, patch')} perhaps Dev. N. fr. <b>čap-</b>. Xak. xı Kaš. I 1.",
    "<b>kočrja:r</b> ‘ram’; Türkü vııı I E 1: Uyğ. vııı ff. Civ. kočğar.",
    "S <b>kočğar</b> See <b>kočga:r</b>.",
    f"<b>tı:n (d-)</b> {BLUE.format('breath')} ‘breath’. Uyğ. vııı ff. Bud. tın.",
    "D <b>2 ellig</b> P.N./A. fr. 1 <b>e:l</b>; ‘having a realm’.",
    "D <b>tapıš</b> N.Ac. (with a connotation of mutuality) fr. 1 <b>tap-</b>. Xak. xı",
    "<b>yarlıkančsız</b> Priv. N./A. fr. a Dev. N. fr. yarlıkan-. Uyğ. vııı ff.",
    "F <b>čawga:n</b> See <b>čögen</b>.",
]) + "</body></html>"


@pytest.fixture(scope="module")
def records():
    out = parse_html(HTML, "test")
    for i, r in enumerate(out):
        r["id"] = i
    return {r["headword"]: r for r in out}


def test_entries_sections_and_continuations(records):
    assert "Dis. ŠNS" not in records and "taŋna-" not in records
    assert set(records) >= {"taŋsuk", "savur-", "čaput", "čapğut", "kočŋa:r", "tı:n", "ellig", "tapıš"}


def test_base_gloss_and_blue_gloss(records):
    t = records["taŋsuk"]
    assert (t["prefix"], t["base"], t["base_homonym"]) == ("D", "taŋ", 2)
    assert t["gloss"] == "wonderful, rare" and t["gloss_tw"] == ""  # sonraki mavi parantez tabanın
    s = records["savur-"]
    assert s["base"] == "sav-" and s["gloss_tw"] == "scatter"
    assert records["tapıš"]["base"] == "tap-"  # "of mutuality" taban değil
    assert records["ellig"]["homonym"] == 2 and records["ellig"]["base_homonym"] == 1


def test_prefixless_derivation_and_references(records):
    assert records["yarlıkančsız"]["base"] == "yarlıkan-"
    assert records["čaput"]["see"] == "čapğut"
    assert records["čawga:n"]["see"] == "čögen"  # F önekli ama salt gönderme


def test_ocr_eng_and_initial_note(records):
    assert "kočŋa:r" in records  # "rj" -> ŋ
    assert records["tı:n"]["initial_note"] == "d-"


def test_split_headword():
    h = split_headword("1 bele:- (be:le:-)")
    assert (h["homonym"], h["headword"], h["variants"]) == (1, "bele:-", ["be:le:-"])
    assert split_headword("*2 üp")["hypothetical"] is True
    assert split_headword("mengü:/meŋgü:")["variants"] == ["meŋgü:"]


def test_attestations_and_works():
    a = attestations("Türkü vııı I E 35: Uyğ. vııı ff. Man.: Xak. xı Kaš. I 40: KB 1424")
    works = [x["work"] for x in a if "work" in x]
    assert works == ["orhun", "dlt", "kb"]
    assert {"dialect": "Uyğ", "century": 8, "ff": True} in a
    # "Türkü vııı ff." Orhun değildir (Yenisey / Maniheist metinler)
    assert not any(x.get("work") == "orhun" for x in attestations("Türkü vııı ff. Man. M I 5"))


def test_comparison_and_see_resolution(records, monkeypatch):
    assert ce.clauson_comparison("kočŋa:r") == "koçnar"
    assert ce.clauson_comparison("uč-") == "uç"
    monkeypatch.setattr(clauson, "load_records", lambda path=None: list(records.values()))
    ce._records.cache_clear()
    try:
        assert ce.resolve(records["čaput"])["headword"] == "čapğut"
        assert ce.resolve(records["kočğar"])["headword"] == "kočŋa:r"  # kaba sınıfta eş
        assert ce.base_form(records["taŋsuk"], "head") == "taŋsuk"
        assert ce.base_form(records["taŋsuk"], "base") == "taŋ"
        dated = ce.earliest_attestation(records["kočŋa:r"])
        assert (dated["year"], dated["precision"]) == (732, "point")
        period = ce.earliest_attestation(records["tı:n"])
        assert period["precision"] == "period" and period["year"] == 1350
    finally:
        ce._records.cache_clear()


@pytest.mark.skipif(not (clauson.CLAUSON_DIR / clauson.JSONL_NAME).is_file(), reason="make clauson yok")
def test_real_data_counts():
    recs = clauson.load_records()
    assert 8500 < len(recs) < 9600
    heads = {r["headword"] for r in recs}
    assert {"taŋsuk", "čapğut", "kočŋa:r", "savur-"} <= heads
