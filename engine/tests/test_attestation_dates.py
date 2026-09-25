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


def test_old_turkic_inscription_citations_map_to_orhun():
    """Vikisözlük otk tanık atıfları (İngilizce yazıt adları) Orhun 732'dir;
    tarihsiz ya da başka yazıt (Ongin, Yenisey) nokta yıl vermez."""
    for ref in ("8th century CE, Kültegin Inscription, S5", "Kül Tégin Inscription E8",
                "Bilge Khagan Inscription, N11", "20th of September 735, Bilge Ḳaġan Inscription N6",
                "c. 716 CE, Bilgä Toɲuquq, Toɲuquq Inscription"):
        assert ad.canonical_year(ref) == 732, ref
    assert ad.canonical_year("c. 728 CE, Ongin Inscription line 11") is None
    assert ad.canonical_year("8-10th century CE, Begre e-11") is None
    assert ad.canonical_year("Yollïġ Tigin") is None


def test_runic_entry_without_dated_citation_is_a_period():
    """Runik madde yazıt atfı yoksa dönem tanığıdır (üst sınır 1000); daha
    geç nokta tarih (Codex Cumanicus 1303) ilk tanıklık sayılmaz."""
    verifier = hav.HistoricalAttestationVerifier()
    entries = [
        {"lang_code": "otk", "lang_name": "Eski Türkçe", "word": "𐰖𐰆𐰞", "meaning": "road"},
        {"lang_code": "qwm", "lang_name": "Kıpçakça (Codex Cumanicus)", "word": "iol", "meaning": "road"},
    ]
    out = verifier.verify_attestation("yol", entries)
    assert out["first_attestation_precision"] == "period"
    assert out["first_attestation_year"] == 1000
    assert out["first_attestation_range"] == [700, 1000]
    # Atıf tarihli yazıtı adlandırıyorsa nokta yıl: Orhun 732.
    entries[0]["attestation_ref"] = "8th century CE, Kültegin Inscription, S5"
    out = verifier.verify_attestation("yol", entries)
    assert (out["first_attestation_year"], out["first_attestation_precision"]) == (732, "point")
    # Runik tanık yoksa Kumanca tanık 1303 kalır.
    out = verifier.verify_attestation("yol", entries[1:])
    assert (out["first_attestation_year"], out["first_attestation_precision"]) == (1303, "point")
    # Ses varyantıyla bulunan başka kelime (`iz` -> 𐰃𐰾 iş) yıl vermez.
    other = {"lang_code": "otk", "lang_name": "Eski Türkçe", "word": "𐰃𐰾", "comparison": "iş",
             "attestation_ref": "8th century CE, Kültegin Inscription, S5"}
    assert verifier.verify_attestation("iz", [other])["verified"] is False
