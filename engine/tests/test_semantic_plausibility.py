"""
Semantik makullük testleri.

Etimolojik iddia iki ayak üzerinde durur: ses ve anlam. Bu modül anlam
ayağını sezgiden veriye taşır; testler o bağın kopmamasını korur.
"""

from __future__ import annotations

import unittest

from engine.nlp.semantic_plausibility import (
    MANUAL_BRIDGE,
    SemanticPlausibility,
    concreteness,
    load_colexifications,
    to_concepticon,
)

HAS_COLEX = bool(load_colexifications())


class TestConcreteness(unittest.TestCase):
    def test_concrete_and_abstract(self):
        self.assertEqual(concreteness("el"), "somut")
        self.assertEqual(concreteness("korku"), "soyut")

    def test_unknown_is_not_forced(self):
        """Bilinmeyen kavram somut veya soyut diye ZORLANMAZ."""
        self.assertEqual(concreteness("zzzqx"), "bilinmiyor")

    def test_empty(self):
        self.assertEqual(concreteness(""), "bilinmiyor")


class TestBridge(unittest.TestCase):
    def test_known_turkish_terms_map_to_concepticon(self):
        self.assertEqual(to_concepticon("ağaç"), "tree")
        self.assertEqual(to_concepticon("su"), "water")

    def test_unknown_term_passes_through(self):
        """Bulunamayan kelime uydurulmaz, olduğu gibi geçer."""
        self.assertEqual(to_concepticon("zzzqx"), "zzzqx")

    def test_manual_bridge_covers_basic_vocabulary(self):
        for term in ("ağaç", "ateş", "baş", "taş", "yol"):
            self.assertIn(term, MANUAL_BRIDGE, term)


class TestPlausibilityWithoutData(unittest.TestCase):
    def setUp(self):
        self.engine = SemanticPlausibility(colexifications={}, relations={})

    def test_identical_meaning_is_certain(self):
        verdict = self.engine.assess("göz", "göz")
        self.assertEqual(verdict.score, 1.0)

    def test_missing_data_says_so_instead_of_guessing(self):
        verdict = self.engine.assess("zzzqx", "qqqzz")
        self.assertEqual(verdict.verdict, "kanıt yok")
        self.assertFalse(verdict.is_plausible)

    def test_concrete_to_abstract_is_favoured(self):
        """Xu ve ark. 2023: dillerin %90'ında beklenen yön budur."""
        forward = self.engine.assess("el", "yardım")
        backward = self.engine.assess("yardım", "el")
        self.assertGreater(forward.score, backward.score)

    def test_reverse_direction_is_flagged_not_forbidden(self):
        verdict = self.engine.assess("yardım", "el")
        self.assertTrue(any("tersi" in a for a in verdict.against))

    def test_empty_input(self):
        self.assertEqual(self.engine.assess("", "x").score, 0.0)


@unittest.skipUnless(HAS_COLEX, "CLICS eş-adlandırma verisi kurulmamış")
class TestPlausibilityWithData(unittest.TestCase):
    def setUp(self):
        self.engine = SemanticPlausibility()

    def test_attested_colexification_is_plausible(self):
        """``ağaç ~ odun`` 59 dil ailesinde aynı kelimedir."""
        verdict = self.engine.assess("ağaç", "odun")
        self.assertTrue(verdict.is_plausible)
        self.assertGreater(verdict.colexification_count, 10)

    def test_unattested_pair_is_not_plausible(self):
        verdict = self.engine.assess("ağaç", "korku")
        self.assertFalse(verdict.is_plausible)

    def test_evidence_names_the_source(self):
        verdict = self.engine.assess("ağaç", "odun")
        self.assertTrue(any("CLICS" in e for e in verdict.evidence))

    def test_symmetric(self):
        """Eş-adlandırma yönsüzdür; kanıt iki yönde de bulunmalı."""
        forward = self.engine.assess("ağaç", "odun").colexification_count
        backward = self.engine.assess("odun", "ağaç").colexification_count
        self.assertEqual(forward, backward)

    def test_serialisable(self):
        data = self.engine.assess("ağaç", "odun").as_dict()
        self.assertIn("explanation", data)
        self.assertIn("colexification_count", data)


if __name__ == "__main__":
    unittest.main()
