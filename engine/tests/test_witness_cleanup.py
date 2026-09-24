"""Tanık listesi temizliği (search_engine): akrabalık beyanı, tekrarlı
tanık, anlamı sorgunun kendisi olan tanık."""
import re

from engine.search_engine import (
    _ATTRIBUTED_FORM,
    _drop_tautological_meanings,
    _first_sentence_loan,
    _merge_duplicate_witnesses,
)


def _attributed(word: str, text: str) -> bool:
    return bool(re.compile(_ATTRIBUTED_FORM.format(q=re.escape(word)), re.IGNORECASE).search(text))


def test_query_inside_quoted_gloss_is_not_an_attributed_form():
    """master: 'Proto-Turkic *ūŕ ("master, craftsman")' Proto-Türkçe `master` demek değil."""
    assert not _attributed("master", "From Proto-Turkic *ūŕ (“master, craftsman”) a derivation")
    assert not _attributed("master", 'Cognate with Turkish hoca ("master, teacher").')
    assert _attributed("göz", "Inherited from Proto-Turkic *köŕ. Cognate with Turkish göz.")
    assert _attributed("köz", "Old Turkic 𐰚𐰇𐰕 (köz)")


def test_semicolon_inside_gloss_does_not_end_the_first_sentence():
    """Başkurtça хужа: 'From *xoja ("owner; host"), from Persian خواجه'."""
    assert _first_sentence_loan("From *xoja (“owner; host”), from Persian خواجه (xâje, “dignitary”).")
    assert not _first_sentence_loan("Inherited from Proto-Turkic *köŕ (“eye”).")


def test_duplicate_witnesses_merge_and_keep_sources():
    entries = [
        {"lang_code": "tt", "lang_name": "Tatarca", "word": "пычак", "comparison": "pıçak",
         "meaning": "", "source": "NorthEuraLex", "origin": "local"},
        {"lang_code": "tt", "lang_name": "Tatarca", "word": "pıçak", "comparison": "pıçak",
         "meaning": "bıçak", "source": "Çağdaş Türk Dilleri", "origin": "seed"},
        # Ayrı bölge etiketli ağız kaydı ayrı tanıktır.
        {"lang_code": "tr", "lang_name": "Türk Ağızları (Sinop)", "word": "pıçak", "comparison": "pıçak",
         "meaning": "", "source": "Derleme", "origin": "live"},
    ]
    out = _merge_duplicate_witnesses(entries)
    assert len(out) == 2
    assert out[0]["word"] == "пычак" and out[0]["meaning"] == "bıçak"
    assert out[0]["also_sources"] == [{"source": "Çağdaş Türk Dilleri", "origin": "seed"}]


def test_merged_sources_still_count_for_triangulation():
    from engine.nlp.hypothesis_validation_protocol import CrossCognateTriangulator

    split = [{"lang_code": "tt", "word": "pıçak", "source": "A", "origin": "local"},
             {"lang_code": "tt", "word": "пычак", "source": "B", "origin": "local"}]
    merged = _merge_duplicate_witnesses([dict(e, lang_name="Tatarca", comparison="pıçak") for e in split])
    tri = CrossCognateTriangulator()
    assert tri.verify("bıçak", split)["source_count"] == tri.verify("bıçak", merged)["source_count"] == 2


def test_meaning_equal_to_query_is_dropped():
    entries = [{"lang_code": "tr", "lang_name": "Türk Ağızları (Diyarbakır)", "word": "Kemal", "meaning": "Kemal",
                "dialect": True},
               {"lang_code": "tt", "lang_name": "Tatarca", "word": "pıçaq", "meaning": "bıçak."},
               {"lang_code": "az", "lang_name": "Azerice", "word": "bıçaq", "meaning": "kesici alet"}]
    _drop_tautological_meanings("kemal", entries[:1])
    _drop_tautological_meanings("bıçak", entries[1:])
    assert entries[0]["meaning"] == "" and entries[0]["meaning_same_as_query"]
    assert entries[1]["meaning"] == "" and entries[1]["meaning_same_as_query"]
    assert entries[2]["meaning"] == "kesici alet" and "meaning_same_as_query" not in entries[2]
    # Sorgunun kendi dil çizgisinin kaydı başlık kökü seçiminde kullanılır; dokunulmaz.
    own = [{"lang_code": "ota", "lang_name": "Osmanlı Türkçesi", "word": "kan", "meaning": "kan"}]
    _drop_tautological_meanings("kan", own)
    assert own[0]["meaning"] == "kan"
