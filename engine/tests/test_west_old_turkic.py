"""
Batı Eski Türkçe (Oğur) tanık bağlayıcısı testleri.

Bu modülün tek işi **yanlış kümeye tanık takmamaktır**. Kavram düzeyinde
eşleme tek başına yetmez: altın standartta GRASS'ı 4, JUMP'ı 4, WIND ve
BURN'ü 3'er küme paylaşıyor. Testler süzgeçlerin bu durumda susmasını korur.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from engine.db import west_old_turkic as wot
from engine.nlp.proto_phonology import LIVE_OGHUR_CODES, OGHUR_CODES

FORMS = """ID,Language_ID,Parameter_ID,Form,Segments
WOT-knee,WOT,0_knee,tīr,t iː r
WOT-wind,WOT,1_wind,śēl,ɕ eː l
WOT-grass,WOT,2_grass,otan,o t a n
WOT-grass2,WOT,2_grass,otam,o t a m
H-knee,H,0_knee,térd,t eː r d
"""

PARAMS = """ID,Name,Concepticon_ID,Concepticon_Gloss
0_knee,knee,1371,KNEE
1_wind,wind,1315,WIND
2_grass,grass,1030,GRASS
"""


def _dataset(directory: Path) -> Path:
    (directory / "forms.csv").write_text(FORMS, encoding="utf-8")
    (directory / "parameters.csv").write_text(PARAMS, encoding="utf-8")
    return directory


class TestLoading(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        wot.reset_cache()
        self.dir = _dataset(Path(self._tmp.name))
        self._forms = wot.load_forms(self.dir)

    def tearDown(self):
        self._tmp.cleanup()
        wot.reset_cache()

    def test_only_wot_forms_are_loaded(self):
        """Macarca satırlar Oğur tanığı DEĞİLDİR."""
        every = [f.form for items in self._forms.values() for f in items]
        self.assertIn("tīr", every)
        self.assertNotIn("térd", every)

    def test_indexed_by_concepticon_gloss(self):
        self.assertEqual(len(self._forms["GRASS"]), 2)

    def test_missing_dataset_disables_the_module(self):
        with TemporaryDirectory() as empty:
            wot.reset_cache()
            self.assertEqual(wot.load_forms(Path(empty)), {})


class TestOghurCorrespondences(unittest.TestCase):
    def test_rhotacism_reflex_is_offered(self):
        """Oğur ``-r`` ~ Ortak Türkçe ``-z``; çevrilmezse ``tir``/``diz``
        düzenli denkliği "fark" sayılır ve doğru tanık elenir."""
        self.assertIn("tiz", wot.to_common_turkic_reflex("tir"))

    def test_initial_sibilant_reflex_is_offered(self):
        self.assertIn("yel", wot.to_common_turkic_reflex("sel"))

    def test_original_form_is_always_a_candidate(self):
        self.assertIn("kum", wot.to_common_turkic_reflex("kum"))

    def test_empty_input(self):
        self.assertEqual(wot.to_common_turkic_reflex(""), set())


class TestLinking(unittest.TestCase):
    """⚠️ Sahte veri kümesiyle koşar.

    ``load_forms`` doğrudan yamalanır: ``lru_cache`` anahtarı argümana bağlı
    olduğu için argümansız çağrı gerçek ``data/cldf/`` dizinini okurdu ve
    testler CI'da veri varlığına bağlı hâle gelirdi.
    """

    def setUp(self):
        self._tmp = TemporaryDirectory()
        wot.reset_cache()
        forms = wot.load_forms(_dataset(Path(self._tmp.name)))
        wot.reset_cache()
        self._patch = mock.patch.object(wot, "load_forms", lambda *a, **k: forms)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()
        wot.reset_cache()

    def test_rhotacism_witness_is_linked(self):
        linked = wot.link_witness("KNEE", {"Turkish": "diz", "Turkmen": "dyz"})
        self.assertIsNotNone(linked)
        self.assertEqual(linked.form, "tīr")

    def test_unrelated_root_is_refused(self):
        """Kavram eşleşse bile kök tutmuyorsa tanık takılmaz."""
        self.assertIsNone(wot.link_witness("KNEE", {"Turkish": "kulak"}))

    def test_ambiguous_concept_is_refused(self):
        """``otan`` ve ``otam`` tanığa aynı uzaklıkta: hangisinin bu kümeye
        ait olduğu **veriden çıkmıyor** demektir. Yanlış takmaktansa
        susulur — yanlış kümeye takılmış tanık, hiç tanık olmamasından
        kötüdür."""
        self.assertIsNone(wot.link_witness("GRASS", {"Turkish": "otar"}))

    def test_derivational_variant_is_refused_on_length(self):
        """``sek`` ile ``*sekir`` aynı köktendir ama aynı BİÇİM değildir;
        taşımadığı hece hakkında oy kullanamaz."""
        self.assertIsNone(
            wot.link_witness("WIND", {"Turkish": "yelpaze", "Kazakh": "jelpaze"})
        )

    def test_no_witnesses_no_link(self):
        self.assertIsNone(wot.link_witness("KNEE", {}))

    def test_unknown_concept(self):
        self.assertIsNone(wot.link_witness("VOLCANO", {"Turkish": "diz"}))


class TestOghurLevels(unittest.TestCase):
    """``wot`` Oğurdur ama ATTESTE DEĞİLDİR — ayrımı kod düzeyinde korur."""

    def test_wot_counts_as_oghur(self):
        self.assertIn("wot", OGHUR_CODES)

    def test_wot_is_not_a_live_witness(self):
        """Geri kurulmuş bir biçimden ``*PT`` iddia etmek, zincirleme
        belirsizliği tek bir iddianın arkasına saklamaktır."""
        self.assertNotIn("wot", LIVE_OGHUR_CODES)
        self.assertIn("cv", LIVE_OGHUR_CODES)

    def test_flag_is_off_because_it_measured_worse(self):
        """Ölçüldü: tam 0,2350 -> 0,2325. Kazanç yok, işaret negatif."""
        self.assertFalse(wot.USE_WEST_OLD_TURKIC)


class TestDistance(unittest.TestCase):
    def test_identical_is_zero(self):
        self.assertEqual(wot._normalised_distance("kum", "kum"), 0.0)

    def test_empty_is_one(self):
        self.assertEqual(wot._normalised_distance("", "kum"), 1.0)

    def test_is_normalised(self):
        self.assertAlmostEqual(wot._normalised_distance("kum", "kun"), 1 / 3)


if __name__ == "__main__":
    unittest.main()
