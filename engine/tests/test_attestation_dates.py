"""Kaynak → yıl haritasının TEK tanımı (``engine.utils.attestation_dates``)."""
from engine.db import starling
from engine.nlp import historical_attestation_verifier as hav
from engine.utils import attestation_dates as ad


def test_single_definition_is_shared():
    """Starling etiketleri ve serbest metin eşleyicisi aynı tablodan okur."""
    assert starling.SOURCE_DATES is ad.STARLING_SOURCE_DATES
    assert [y for _, y, _ in hav.DATED_SOURCES] == [w.year for w in ad.POINT_WORKS]


def test_same_work_same_year_everywhere():
    dlt = ad.work("dlt").year
    assert starling.SOURCE_DATES["MK"] == dlt
    for text in ("Divan-i Lugat-it Türk (1070)", "Divanü Lugati't-Türk (1074)",
                 "Dîvânu Lugâti't-Türk", "Kaşgarlı Mahmud"):
        assert ad.canonical_year(text) == dlt, text
    ah = ad.work("ah").year
    assert starling.SOURCE_DATES["AH"] == ah
    assert ad.canonical_year("Atebet-ül Hakayık (1300 yılından önce)") == ah
    assert starling.SOURCE_DATES["Orkh."] == ad.canonical_year("Orhun Yazıtları") == 732


def test_unknown_work_keeps_source_year():
    assert ad.canonical_year("Kitab-ı Dede Korkut (1500)", 1500) == 1500
    assert ad.canonical_year("", None) is None


def test_old_uyghur_is_a_period_not_a_point_year():
    """Wilkens tanığı dönem aralığıdır (üst sınır 1350); serbest metinde nokta
    yıl vermez. Tarihsiz "Eski Uygurca" dil adı hiç eşleşmez."""
    from engine.fetchers import wilkens_old_uyghur as wk

    oui = ad.work("oui")
    assert oui.precision == "period" and oui.range == (wk.PERIOD_START, wk.attestation_year()) == (800, 1350)
    assert ad.canonical_year("Eski Uygurca") is None
    assert ad.canonical_year("Eski Uygurca (9.-14. yy), Wilkens 2021, s. 12", None) is None
    assert all(w.precision == "point" for w in ad.POINT_WORKS)
