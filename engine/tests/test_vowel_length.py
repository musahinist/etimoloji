"""
Ünlü uzunluğu çıkarımı ve rekonstrüksiyonu testleri.

Proto-Türkçe'de uzunluk ayırt edicidir (``*ōt`` "ateş" ≠ ``*ot`` "ot") ama
Ortak Türkçe kollarının çoğu onu kaybetmiştir. Bu testler hem çıkarımın
doğruluğunu hem de **ölçülen sınırların** korunmasını denetler.
"""

from __future__ import annotations

import unittest

from engine.config import LEXICON_DIR
from engine.db.lexicon_index import extract_long_vowels
from engine.nlp.vowel_length import (
    APPLY_LENGTH_TO_PROTO,
    LENGTH_LOSING,
    LENGTH_PRESERVING,
    LengthEvidence,
    apply_length,
    gather_evidence,
)

HAS_INDEX = (LEXICON_DIR / "index.db").exists()


class TestLongVowelExtraction(unittest.TestCase):
    """``ː`` aramak yetmez: ünsüzden sonra gelirse İKİZLEŞMEDİR."""

    def test_long_vowel_is_found(self):
        self.assertEqual(extract_long_vowels("/aː.biˈde/"), "aː")
        self.assertEqual(extract_long_vowels("[biːt]"), "iː")

    def test_geminate_consonant_is_not_vowel_length(self):
        """``борщ [buɔɐ̯rɕː]`` bir uzunluk tanığı DEĞİLDİR."""
        self.assertEqual(extract_long_vowels("[bu͡ɔɐ̯rɕː]"), "")
        self.assertEqual(extract_long_vowels("/ɕː/"), "")
        self.assertEqual(extract_long_vowels("[pɘˈrːe]"), "")

    def test_multiple_long_vowels_keep_order(self):
        self.assertEqual(extract_long_vowels("/aːɯː/"), "aːɯː")

    def test_no_length_marks(self):
        self.assertEqual(extract_long_vowels("/köz/"), "")
        self.assertEqual(extract_long_vowels(""), "")

    def test_ascii_colon_also_counts(self):
        self.assertEqual(extract_long_vowels("[o:t]"), "oː")


class TestLanguageRoles(unittest.TestCase):
    def test_khalaj_is_the_strongest_witness(self):
        """Halaçça hem uzunluğu hem söz başı *h-'yi korur (Doerfer)."""
        self.assertEqual(max(LENGTH_PRESERVING.values()), LENGTH_PRESERVING["klj"])

    def test_turkmen_is_the_oghuz_length_witness(self):
        self.assertIn("tk", LENGTH_PRESERVING)
        self.assertNotIn("tk", LENGTH_LOSING)

    def test_turkish_is_not_a_length_witness(self):
        """Türkçedeki uzun ünlüler çoğunlukla Arapça/Farsça alıntıdandır."""
        self.assertIn("tr", LENGTH_LOSING)
        self.assertNotIn("tr", LENGTH_PRESERVING)

    def test_roles_do_not_overlap(self):
        self.assertEqual(set(LENGTH_PRESERVING) & LENGTH_LOSING, set())


class TestApplyLength(unittest.TestCase):
    def test_no_evidence_leaves_form_untouched(self):
        self.assertEqual(apply_length("*ot", LengthEvidence()), "*ot")

    def test_unconstrained_evidence_is_not_applied(self):
        """⚠️ Anlam kısıtı olmadan kesinlik 0,303 — uygulamak ZARAR verir.

        Ölçüldü: kısıtsız uygulandığında NED 0,384 -> 0,398, tam doğruluk
        0,230 -> 0,205. Sebep eşadlılık: Türkmence ``ot`` hem "ateş"
        (``*ōt``) hem "ot/bitki" (``*ot``) demektir.
        """
        evidence = LengthEvidence(positions={0: 1.0}, sense_constrained=False)
        self.assertEqual(apply_length("*ot", evidence), "*ot")

    def test_sense_constrained_evidence_is_applied(self):
        """Anlam kısıtıyla kesinlik 0,583'e çıkıyor — uygulanabilir."""
        evidence = LengthEvidence(positions={0: 1.0}, sense_constrained=True)
        self.assertEqual(apply_length("*ot", evidence), "*ōt")

    def test_force_overrides_for_experiments(self):
        evidence = LengthEvidence(positions={0: 1.0}, sense_constrained=False)
        self.assertEqual(apply_length("*ot", evidence, force=True), "*ōt")

    def test_only_the_marked_vowel_becomes_long(self):
        evidence = LengthEvidence(positions={1: 1.0}, sense_constrained=True)
        self.assertEqual(apply_length("*kapuk", evidence), "*kapūk")

    def test_weak_evidence_does_not_reach_the_threshold(self):
        """Tuvaca tek başına (0,75) eşiği geçemez."""
        evidence = LengthEvidence(positions={0: 0.75}, sense_constrained=True)
        self.assertEqual(apply_length("*ot", evidence), "*ot")

    def test_star_prefix_is_preserved(self):
        evidence = LengthEvidence(positions={0: 1.0}, sense_constrained=True)
        self.assertTrue(apply_length("*ot", evidence).startswith("*"))
        self.assertFalse(apply_length("ot", evidence).startswith("*"))

    def test_switch_is_documented_as_enabled(self):
        self.assertTrue(APPLY_LENGTH_TO_PROTO)


@unittest.skipUnless(HAS_INDEX, "sözlük indeksi kurulmamış")
class TestEvidenceGathering(unittest.TestCase):
    def test_length_preserving_witness_yields_evidence(self):
        evidence = gather_evidence({"tk": "ot"}, sense="fire")
        self.assertTrue(evidence.any_evidence)
        self.assertTrue(evidence.sense_constrained)

    def test_length_losing_witness_is_ignored(self):
        """Türkçe biçimde uzun ünlü görmek ata biçim hakkında kanıt değildir."""
        evidence = gather_evidence({"tr": "abide"})
        self.assertFalse(evidence.any_evidence)

    def test_no_sense_means_unconstrained(self):
        evidence = gather_evidence({"tk": "ot"})
        self.assertFalse(evidence.sense_constrained)

    def test_describe_is_readable(self):
        evidence = gather_evidence({"tk": "ot"}, sense="fire")
        self.assertIn("uzunluk", evidence.describe())

    def test_empty_witnesses(self):
        self.assertFalse(gather_evidence({}).any_evidence)


if __name__ == "__main__":
    unittest.main()
