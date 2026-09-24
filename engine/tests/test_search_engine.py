"""
SearchEngine Entegrasyon Testleri

Eski test canlı ağa çıkıyor, ~80 HTTP isteği atıyor ve 30-60 saniye sürüyordu.
``SearchEngine`` artık fetcher enjeksiyonuna izin verdiği için (``fetchers=``)
tüm entegrasyon ağsız ve milisaniyeler içinde koşar.
"""
from __future__ import annotations

import os
import tempfile
import unittest

from engine import config
from engine.db.database import DatabaseManager
from engine.search_engine import SearchEngine, default_fetchers, translate_meaning
from engine.tests.data_guards import needs_torch
from engine.tests.fakes import EmptyFetcher, FailingFetcher, FakeFetcher

GOZ_FORMS = [("tr", "göz"), ("az", "göz"), ("kk", "көз"), ("tt", "күз"),
             ("cv", "куҫ"), ("otk", "köz"), ("ky", "көз"), ("ba", "күҙ")]


class TestSearchEngineIntegration(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db = DatabaseManager(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def _engine(self, fetchers):
        return SearchEngine(db_manager=self.db, fetchers=fetchers)

    def _goz_fetcher(self, **kw):
        return FakeFetcher(
            name="Sahte Sözlük", entries=GOZ_FORMS,
            meaning="göz, görme organı", only_for="göz", **kw
        )

    # --- Temel akış ------------------------------------------------------

    def test_search_produces_correct_reconstruction(self):
        res = self._engine([self._goz_fetcher()]).search("göz", save_to_db=False)
        self.assertEqual(res["root"]["proto_turkic"], "*köŕ")
        real = [e for e in res["turkic_languages"] if e["lang_code"] != "donor"]
        self.assertEqual(len(real), 8)

    def test_result_is_deterministic(self):
        """Aynı girdi -> BİREBİR aynı çıktı.

        Regresyon: hash() tohumlaması ve set sıralaması yüzünden skorlar ve
        aranan varyantlar her çalıştırmada değişiyordu.
        """
        import json

        engine = self._engine([self._goz_fetcher()])
        a = json.dumps(self._strip_timings(engine.search("göz", save_to_db=False)), sort_keys=True)
        b = json.dumps(self._strip_timings(engine.search("göz", save_to_db=False)), sort_keys=True)
        self.assertEqual(a, b)

    @staticmethod
    def _strip_timings(finding):
        f = dict(finding)
        f.pop("diagnostics", None)
        return f

    def test_failing_fetcher_does_not_break_search(self):
        """Bir kaynak çökerse arama DEVAM etmeli, hata GÖRÜNÜR olmalı."""
        engine = self._engine([self._goz_fetcher(), FailingFetcher(), EmptyFetcher()])
        res = engine.search("göz", save_to_db=False)
        real = [e for e in res["turkic_languages"] if e["lang_code"] != "donor"]
        self.assertEqual(len(real), 8)
        diag = res["diagnostics"]["sources"]
        self.assertEqual(diag["Patlayan Kaynak"]["status"], "error")
        self.assertTrue(diag["Patlayan Kaynak"]["errors"])
        self.assertEqual(diag["Boş Kaynak"]["status"], "empty")

    def test_diagnostics_report_real_timings(self):
        """Teşhis GERÇEK aşama sürelerini taşımalıdır (panelin sahte
        setTimeout simülasyonunun yerini alan veri)."""
        res = self._engine([self._goz_fetcher()]).search("göz", save_to_db=False)
        timings = res["diagnostics"]["stage_timings_ms"]
        for stage in ("morphology", "fetch", "nlp", "total"):
            self.assertIn(stage, timings)
            self.assertIsInstance(timings[stage], int)

    def test_variant_count_is_capped(self):
        res = self._engine([self._goz_fetcher()]).search("göz", save_to_db=False)
        self.assertLessEqual(len(res["diagnostics"]["variants_used"]), config.MAX_VARIANTS)

    # --- Uydurma üretmeme ------------------------------------------------

    def test_unknown_word_gets_no_fabricated_root(self):
        """Kanıt yoksa `*<kelime>` biçiminde ata kök UYDURULMAMALIDIR."""
        res = self._engine([EmptyFetcher()]).search("zzzqx", save_to_db=False)
        self.assertNotEqual(res["root"]["proto_turkic"], "*zzzqx")
        hypo = res["nlp_analysis"]["proven_hypothesis"]
        if hypo and hypo.get("validation_report"):
            self.assertNotIn("VALIDATED (kanıta dayalı)", hypo["validation_report"]["badge"])

    def test_unknown_word_badge_is_not_validated(self):
        res = self._engine([EmptyFetcher()]).search("qqqwwweee", save_to_db=False)
        hypo = res["nlp_analysis"]["proven_hypothesis"] or {}
        report = hypo.get("validation_report") or {}
        self.assertNotEqual(report.get("status_code"), "VALIDATED")

    # --- Önbellek --------------------------------------------------------

    def test_cache_round_trip(self):
        engine = self._engine([self._goz_fetcher()])
        first = engine.search("göz", save_to_db=True)
        self.assertFalse(first["from_cache"])
        second = engine.search("göz", save_to_db=True)
        self.assertTrue(second["from_cache"])
        self.assertEqual(second["root"]["proto_turkic"], first["root"]["proto_turkic"])

    def test_cache_can_be_disabled(self):
        engine = self._engine([self._goz_fetcher()])
        engine.search("göz", save_to_db=True)
        original = config.CACHE_ENABLED
        try:
            config.CACHE_ENABLED = False
            self.assertFalse(engine.search("göz", save_to_db=False)["from_cache"])
        finally:
            config.CACHE_ENABLED = original

    # --- Zenginleştirme --------------------------------------------------

    def test_entries_get_transliteration_and_references(self):
        """README'nin vaat ettiği transkripsiyon motoru boru hattına BAĞLI olmalı."""
        res = self._engine([self._goz_fetcher()]).search("göz", save_to_db=False)
        cyrillic = [e for e in res["turkic_languages"] if e.get("script") == "Cyrillic"]
        self.assertTrue(cyrillic)
        self.assertTrue(any("latin_transliteration" in e for e in cyrillic))

    def test_faz6_modules_are_wired(self):
        res = self._engine([self._goz_fetcher()]).search("göz", save_to_db=False)
        nlp = res["nlp_analysis"]
        for key in ("loanword_detection", "cognate_clusters", "historical_morphology",
                    "induced_sound_laws", "proven_hypothesis"):
            self.assertIn(key, nlp)

    def test_query_length_is_capped(self):
        res = self._engine([EmptyFetcher()]).search("a" * 500, save_to_db=False)
        self.assertLessEqual(len(res["query_word"]), config.MAX_QUERY_LENGTH)

    def test_empty_query(self):
        res = self._engine([EmptyFetcher()]).search("", save_to_db=False)
        self.assertEqual(res["query_word"], "")

    # --- Kalıcılık -------------------------------------------------------

    def test_saved_finding_round_trips(self):
        engine = self._engine([self._goz_fetcher()])
        engine.search("göz", save_to_db=True)
        stored = self.db.get_finding("göz")
        self.assertIsNotNone(stored)
        self.assertEqual(stored["root"]["proto_turkic"], "*köŕ")


class _SlowFetcher(FakeFetcher):
    """Geç biten kaynak: bitiş sırasının anlam seçimini belirlemediğini sınar."""

    def fetch(self, word):
        import time

        time.sleep(0.05)
        return super().fetch(word)


class _DialectFetcher(FakeFetcher):
    """TDK Derleme gibi `tr` kodlu ama ağız işaretli kayıt döndürür."""

    def fetch(self, word):
        result = super().fetch(word)
        for entry in result["turkic_languages"]:
            entry["lang_name"] = "Türk Ağızları (Isparta)"
            entry["dialect"] = True
        return result


class TestMeaningsBySource(unittest.TestCase):
    """terlik regresyonu: başlıkta "bk. derlik", TDK anlamı kayıp."""

    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db = DatabaseManager(self.db_path)
        self._cache = config.CACHE_ENABLED
        config.CACHE_ENABLED = False

    def tearDown(self):
        config.CACHE_ENABLED = self._cache
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def _search(self, fetchers):
        return SearchEngine(db_manager=self.db, fetchers=fetchers).search("terlik", save_to_db=False)

    def _fetchers(self):
        return [
            _DialectFetcher(name="TDK Derleme", entries=[("tr", "terlik")],
                            meaning="takke, başlık", only_for="terlik"),
            FakeFetcher(name="TDK Tarama", entries=[("otk", "terlik")],
                        meaning="bk. derlik", only_for="terlik"),
            _SlowFetcher(name="TDK (Türk Dil Kurumu)", entries=[("tr", "terlik")],
                         meaning="ev içinde giyilen ayak giysisi", only_for="terlik"),
        ]

    def test_primary_meaning_is_tdk_even_when_it_finishes_last(self):
        res = self._search(self._fetchers())
        self.assertEqual(res["root"]["meaning"], "ev içinde giyilen ayak giysisi")

    def test_every_source_meaning_is_reported_tdk_first(self):
        groups = self._search(self._fetchers())["root"]["meanings"]
        self.assertEqual(groups[0], {"source": "TDK (Türk Dil Kurumu)",
                                     "meanings": ["ev içinde giyilen ayak giysisi"]})
        self.assertIn({"source": "TDK Derleme", "meanings": ["takke, başlık"]}, groups)

    def test_cross_reference_is_not_a_meaning(self):
        res = self._search(self._fetchers())
        self.assertNotIn("TDK Tarama", [g["source"] for g in res["root"]["meanings"]])
        self.assertNotEqual(res["root"]["meaning"], "bk. derlik")

    def test_dialect_entry_does_not_replace_standard_entry(self):
        res = self._search(self._fetchers())
        tr = sorted(e["meaning"] for e in res["turkic_languages"]
                    if e["lang_code"] == "tr" and e["word"] == "terlik")
        self.assertEqual(tr, ["ev içinde giyilen ayak giysisi", "takke, başlık"])


class _FormationFetcher(FakeFetcher):
    """Yapıyı veren tarihî sözlük maddesi (indeksin `formation` sütunu)."""

    def fetch(self, word):
        result = super().fetch(word)
        for entry in result["turkic_languages"]:
            entry["formation"] = "biti- + -g"
        return result


class TestRootFromFormation(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db = DatabaseManager(self.db_path)
        self._cache = config.CACHE_ENABLED
        config.CACHE_ENABLED = False

    def tearDown(self):
        config.CACHE_ENABLED = self._cache
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_formation_gives_root_when_engine_finds_none(self):
        fetcher = _FormationFetcher(name="Tarihî", entries=[("otk", "bitig")],
                                    meaning="inscription", only_for="bitig")
        res = SearchEngine(db_manager=self.db, fetchers=[fetcher]).search("bitig", save_to_db=False)
        self.assertEqual(res["morphology"], "biti- + -g (sözlük maddesine göre)")
        self.assertEqual(res["root"]["proto_turkic"], "biti-")
        self.assertIn("sözlük maddesinin yapısı", res["root"]["provenance"])
        self.assertIn("TÜRETİLMEDİ", res["nlp_analysis"]["reconstruction"]["reconstruction_notes"])

    def test_formation_does_not_override_reconstructed_root(self):
        fetcher = _FormationFetcher(name="Sahte Sözlük", entries=GOZ_FORMS,
                                    meaning="göz, görme organı", only_for="göz")
        res = SearchEngine(db_manager=self.db, fetchers=[fetcher]).search("göz", save_to_db=False)
        self.assertEqual(res["root"]["proto_turkic"], "*köŕ")


class TestEtymologyMentions(unittest.TestCase):
    """Ters bağlantı: `betik`in etimolojisi "from Old Turkic bitig" der."""

    def _mentions(self, rows):
        from unittest import mock

        index = mock.Mock(exists=True)
        index.search.return_value = rows
        with mock.patch("engine.db.lexicon_index.LexiconIndex", return_value=index):
            return SearchEngine._etymology_mentions("bitig")

    def test_reverse_link_found_and_self_excluded(self):
        res = self._mentions([
            {"lang_code": "tr", "word": "betik", "comparison": "betik", "gloss": "book",
             "etymology": "Learned borrowing from Old Turkic bitig."},
            {"lang_code": "otk", "word": "bitig", "comparison": "bitig", "gloss": "inscription",
             "etymology": "biti- + -g"},
        ])
        self.assertEqual([i["word"] for i in res["items"]], ["betik"])

    def test_capitalised_title_and_suffix_are_not_citations(self):
        res = self._mentions([
            {"lang_code": "kk", "word": "және", "comparison": "jene", "gloss": "and",
             "etymology": "Old Turkic jana (in Irk Bitig)"},
        ])
        self.assertEqual(res["items"], [])


class TestAssertedCognates(unittest.TestCase):
    """Kaynağın akraba dediği kayıtlar tanık olur; alıntı, eşsesli ve türev olmaz."""

    @staticmethod
    def _item(lang, word, gloss, text, comparison=None):
        return {"lang_code": lang, "lang_name": lang, "word": word, "gloss": gloss,
                "comparison": comparison or word, "etymology_full": text}

    def _run(self, word, items):
        from engine.search_engine import _asserted_cognates

        return [c["word"] for c in _asserted_cognates(word, {"items": items})]

    def test_kinship_claims_are_accepted(self):
        items = [
            self._item("cv", "пӗтӳ", "amulet", "From Proto-Turkic *bitig, ultimately from *biti-.", "petu"),
            self._item("ba", "бетеү", "документ", "Родственно др.-тюрк. 𐰋𐰃𐱅𐰏 (bitig, «письмо»).", "beteü"),
        ]
        self.assertEqual(self._run("bitig", items), ["пӗтӳ", "бетеү"])

    def test_borrowings_are_rejected_anywhere_in_note(self):
        items = [
            self._item("tr", "betik", "book", "Learned borrowing from Old Turkic bitig, from Proto-Turkic *bitig."),
            self._item("az", "kitab", "book", "Borrowed from Arabic كِتَاب. Compare Turkish kitap."),
            self._item("ba", "ҡәләм", "pencil", "From Arabic قَلَم. Cognate with Turkish kalem.", "kelem"),
        ]
        self.assertEqual(self._run("bitig", items[:1]), [])
        self.assertEqual(self._run("kitap", items[1:2]), [])
        self.assertEqual(self._run("kalem", items[2:]), [])

    def test_query_must_be_named_as_turkic_form(self):
        """`el` "halk" eşseslisi: "Uyghur ئەل (el)" Türkçe `el` demek değildir."""
        item = self._item("ug", "ئەل", "people", "From Proto-Turkic *ēl. Cognate with Uyghur ئەل (el).", "el")
        self.assertEqual(self._run("el", [item]), [])

    def test_compounds_and_form_of_glosses_are_rejected(self):
        items = [
            self._item("gag", "göz yaşı", "tear", "Cognate with Turkish göz."),
            self._item("tr", "gözyaşı", "tear", "Compare Turkish göz."),
            self._item("az", "eyləmək", "dated form of eləmək", "Compare Turkish el."),
        ]
        self.assertEqual(self._run("göz", items[:2]), [])
        self.assertEqual(self._run("el", items[2:]), [])


class TestHomonymFilter(unittest.TestCase):
    def test_no_meanings_means_no_filtering(self):
        from engine.search_engine import _homonym_filter

        cands = [{"word": "x", "meaning": "to sow"}]
        self.assertEqual(_homonym_filter(cands, []), (cands, []))

    @needs_torch
    def test_low_similarity_is_dropped(self):
        from unittest import mock

        import torch

        from engine.search_engine import _homonym_filter

        model = mock.Mock()
        model.encode.side_effect = lambda texts, **_: torch.tensor(
            [[1.0, 0.0] if t in ("bread", "ekmek anlamı") else [0.0, 1.0] for t in texts]
        )
        with (
            mock.patch("engine.nlp.diachronic_semantic_engine.has_semantic_model", return_value=True),
            mock.patch("engine.nlp.diachronic_semantic_engine.get_sentence_transformer", return_value=model),
        ):
            kept, dropped = _homonym_filter(
                [{"word": "ötmek", "meaning": "bread"}, {"word": "ekmää", "meaning": "to sow"}],
                ["ekmek anlamı"],
            )
        self.assertEqual([k["word"] for k in kept], ["ötmek"])
        self.assertEqual([d["word"] for d in dropped], ["ekmää"])


class TestTranslateMeaning(unittest.TestCase):
    def test_whole_word_matching(self):
        """Regresyon: substring eşleşmesi 'Sunday' -> 'güneş, gün' yapıyordu."""
        self.assertEqual(translate_meaning("sun"), "güneş, gün")
        self.assertEqual(translate_meaning("Sunday"), "Sunday")
        self.assertEqual(translate_meaning("consumed"), "consumed")
        self.assertEqual(translate_meaning("sunset glow"), "sunset glow")

    def test_phrase_matching(self):
        self.assertEqual(translate_meaning("the sea"), "deniz, büyük göl")

    def test_word_inside_longer_gloss_is_not_translated(self):
        """Regresyon: Osmanlıca `ekmek`in anlamı "water" yüzünden "su, sıvı" oluyordu."""
        gloss = "bread, a foodstuff prepared from a dough of flour and water"
        self.assertEqual(translate_meaning(gloss), gloss)
        self.assertEqual(translate_meaning("water; river"), "su, sıvı")

    def test_wiki_markup_stripped(self):
        self.assertNotIn("{{", translate_meaning("{{lb|tr}} eye"))

    def test_online_placeholder_passthrough(self):
        self.assertTrue(translate_meaning("Online Kazakça").startswith("Online"))

    def test_empty(self):
        self.assertEqual(translate_meaning(""), "")


class TestDefaultPortfolio(unittest.TestCase):
    def test_portfolio_instantiates(self):
        fetchers = default_fetchers()
        self.assertGreaterEqual(len(fetchers), 15)
        for f in fetchers:
            self.assertTrue(f.source_name)


if __name__ == "__main__":
    unittest.main()
