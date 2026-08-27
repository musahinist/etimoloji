"""
Alıntı geçiş zinciri ve uyarlama kuralı testleri.

Kullanıcının sorusu buydu: kelime Arapçadan/Farsçadan geçtiyse **nasıl**
geçtiğini ve **geçerken nasıl değiştiğini** çıkarabiliyor muyuz?
"""

from __future__ import annotations

import unittest

from engine.config import LEXICON_DIR
from engine.nlp.borrowing_chain import (
    ABJAD_LANGUAGES,
    AdaptationRule,
    AdaptationRuleLearner,
    ChainExtractor,
    ChainLink,
    language_name,
)

HAS_LEXICON = any(LEXICON_DIR.glob("tr.jsonl*"))


def record(word, lang, templates):
    return {
        "word": word,
        "lang_code": lang,
        "etymology_templates": [
            {"name": name, "args": {"1": lang, "2": donor, "3": form}}
            for name, donor, form in templates
        ],
    }


class TestChainExtraction(unittest.TestCase):
    def setUp(self):
        self.extractor = ChainExtractor()

    def test_single_link(self):
        chain = self.extractor.extract(record("kitap", "tr", [("inh", "ota", "كتاب")]))
        self.assertEqual(chain.depth, 1)
        self.assertEqual(chain.ultimate_origin, "ota")

    def test_multi_step_chain_is_ordered_nearest_first(self):
        """``sabun``: Türkçe <- Osmanlıca <- Arapça."""
        chain = self.extractor.extract(
            record("sabun", "tr", [("inh", "ota", "صابون"), ("der", "ar", "صَابُون")])
        )
        self.assertEqual(chain.depth, 2)
        self.assertEqual(chain.path(), ["Türkçe", "Osmanlı Türkçesi", "Arapça"])
        self.assertEqual(chain.ultimate_origin, "ar")

    def test_duplicate_inh_plus_is_collapsed(self):
        """Wiktionary aynı halkayı ``inh`` ve ``inh+`` ile iki kez verebilir."""
        chain = self.extractor.extract(
            record("duvar", "tr", [("inh", "ota", "دیوار"), ("inh+", "ota", "دیوار")])
        )
        self.assertEqual(chain.depth, 1)

    def test_helper_templates_are_ignored(self):
        chain = self.extractor.extract(
            record("x", "tr", [("yesno", "i", "I"), ("glossary", None, None), ("inh", "ota", "y")])
        )
        self.assertEqual(chain.depth, 1)

    def test_pure_inheritance_is_not_a_borrowing(self):
        """Türkçe <- Osmanlıca <- Ana Türkçe bir MİRAS zinciridir."""
        chain = self.extractor.extract(
            record("on", "tr", [("inh", "ota", "اون"), ("inh", "trk-pro", "*ōn")])
        )
        self.assertFalse(chain.is_borrowed)

    def test_chain_with_a_borrowed_link_is_a_borrowing(self):
        chain = self.extractor.extract(
            record("sabun", "tr", [("inh", "ota", "صابون"), ("bor", "ar", "صَابُون")])
        )
        self.assertTrue(chain.is_borrowed)

    def test_no_templates_yields_nothing(self):
        self.assertIsNone(self.extractor.extract(record("kar", "tr", [])))

    def test_missing_word_yields_nothing(self):
        self.assertIsNone(self.extractor.extract({"lang_code": "tr"}))

    def test_describe_is_readable(self):
        chain = self.extractor.extract(
            record("çay", "tr", [("inh", "ota", "چای"), ("der", "fa", "چای")])
        )
        self.assertIn("←", chain.describe())
        self.assertIn("Farsça", chain.describe())


class TestAdaptationRules(unittest.TestCase):
    def test_learns_a_repeated_change(self):
        learner = AdaptationRuleLearner()
        for source, target in [("lampe", "lamba"), ("rampe", "ramba"), ("pompe", "pomba")]:
            learner.observe_link(ChainLink("fr", source, "tr", target, "alıntı"))
        changes = [r for r in learner.rules() if r.is_change]
        self.assertTrue(changes)

    def test_single_observation_is_not_a_rule(self):
        learner = AdaptationRuleLearner()
        learner.observe_link(ChainLink("fr", "lampe", "tr", "lamba", "alıntı"))
        self.assertEqual([r for r in learner.rules() if r.support >= 2], [])

    def test_too_short_forms_are_skipped(self):
        learner = AdaptationRuleLearner()
        self.assertFalse(learner.observe_link(ChainLink("fr", "a", "tr", "b", "alıntı")))

    def test_regularity_is_one_for_perfectly_regular_pairs(self):
        learner = AdaptationRuleLearner()
        for word in ("kitap", "kitap", "kitap"):
            learner.observe_link(ChainLink("ar", word, "tr", word, "alıntı"))
        self.assertEqual(learner.regularity("ar", "tr"), 1.0)

    def test_regularity_is_zero_for_unknown_pair(self):
        self.assertEqual(AdaptationRuleLearner().regularity("xx", "yy"), 0.0)

    def test_script_artifact_is_flagged(self):
        """Abjad'dan gelen ünlü ekleme kuralı ses değişimi DEĞİLDİR.

        ``صابون`` çeviriyazıda ünlüsüz görünür; Türkçe ``sabun`` ile
        hizalandığında sanki ünlü türetilmiş gibi çıkar. Oysa ünlü zaten
        söyleniyordu, yalnız yazılmıyordu.
        """
        artifact = AdaptationRule("ar", "ota", "medial", "-", "a", 100)
        self.assertTrue(artifact.is_script_artifact)

    def test_genuine_consonant_change_is_not_an_artifact(self):
        genuine = AdaptationRule("fr", "tr", "medial", "c", "k", 397)
        self.assertFalse(genuine.is_script_artifact)

    def test_latin_source_gap_is_not_an_artifact(self):
        """Fransızca ünlüleri yazar; oradaki ünlü düşmesi gerçektir."""
        genuine = AdaptationRule("fr", "tr", "final", "e", "-", 1202)
        self.assertFalse(genuine.is_script_artifact)

    def test_abjad_list_covers_the_main_donors(self):
        for code in ("ar", "fa", "ota"):
            self.assertIn(code, ABJAD_LANGUAGES, code)


class TestLanguageNames(unittest.TestCase):
    def test_known_codes_are_translated(self):
        self.assertEqual(language_name("ar"), "Arapça")
        self.assertEqual(language_name("ota"), "Osmanlı Türkçesi")

    def test_unknown_code_falls_back_to_itself(self):
        self.assertEqual(language_name("xyz"), "xyz")


@unittest.skipUnless(HAS_LEXICON, "sözlük dökümü indirilmemiş")
class TestRealChains(unittest.TestCase):
    def test_multi_step_chains_exist_in_real_data(self):
        from engine.nlp.borrowing_chain import iter_chains

        deep = [c for c in iter_chains("tr") if c.depth >= 2]
        self.assertGreater(len(deep), 100)

    def test_ottoman_is_the_dominant_intermediary(self):
        """Türkçedeki Arapça/Farsça alıntıların çoğu Osmanlıca üzerinden gelir.

        Bu yüzden "doğrudan Arapça alıntı" saymak yanıltıcıdır.
        """
        from engine.nlp.borrowing_chain import iter_chains

        via_ottoman = sum(
            1 for c in iter_chains("tr") if any(link.from_lang == "ota" for link in c.links)
        )
        self.assertGreater(via_ottoman, 1000)


if __name__ == "__main__":
    unittest.main()
