"""
Verici dil yakınlığı testleri — alıntı tespitinin en güçlü sinyali.

Ölçüldü (WOLD/Sakha, n=769): bu sinyal motorun F skorunu **0,3850 -> 0,5664**
çıkardı ve tek başına F 0,5613 alıyor. Testler iki şeyi korur:

1. **Mesafe SCA'dır**, düz Levenshtein değil. Sakha Rusça ``stol``u
   ``ostuol`` yapar; düz uzaklık 0,50 verip eşiğin üstünde kalır.
2. **Verici indeksi Türki indeksten ayrıdır**. Karışsalardı Rusça ``море``
   Türki bir akraba adayı olarak dönerdi.
"""

from __future__ import annotations

import gzip
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from engine.db.donor_index import DonorIndex, is_from_turkic, iter_donor_entries
from engine.nlp import donor_proximity as dp

ARMENIAN = [
    {
        "word": "գյոզ",
        "forms": [{"form": "gyoz", "tags": ["romanization"]}],
        "senses": [{"glosses": ["eye"]}],
        "etymology_text": "From Ottoman Turkish گوز (göz, “eye”).",
    },
    {
        "word": "կուռ",
        "forms": [{"form": "kuŕ", "tags": ["romanization"]}],
        "senses": [{"glosses": ["arm"]}],
        "etymology_text": "Inherited from Old Armenian կուռն (kuṙn).",
    },
]

RUSSIAN = [
    {"word": "стол", "senses": [{"glosses": ["table"]}]},
    {"word": "море", "senses": [{"glosses": ["sea"]}]},
    {"word": "мыло", "senses": [{"glosses": ["soap"]}]},
    {"word": "о", "senses": [{"glosses": ["about"]}]},  # tek harf — elenir
    {"word": "", "senses": []},                     # boş — elenir
]


def _dump(directory: Path, code: str, records: list[dict]) -> Path:
    path = directory / f"{code}.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


class TestDonorEntryParsing(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_cyrillic_is_normalised_for_comparison(self):
        """Rusça Kirildir, Sakha ölçütü Latin çevriyazıdır; ortak biçim şart."""
        entries = list(iter_donor_entries(_dump(self.dir, "ru", RUSSIAN), "ru"))
        forms = {e.word: e.comparison for e in entries}
        self.assertEqual(forms["стол"], "stol")
        self.assertEqual(forms["море"], "more")

    def test_too_short_entries_are_dropped(self):
        """Tek harflik madde her sorguya yakın çıkar; yalnız gürültü üretir.

        ⚠️ Ölçüt **çevriyazı sonrası** uzunluktur: Kiril ``я`` tek harftir
        ama Latin karşılığı ``ya``dır ve elenmez."""
        entries = list(iter_donor_entries(_dump(self.dir, "ru", RUSSIAN), "ru"))
        self.assertNotIn("о", {e.word for e in entries})

    def test_glosses_are_kept(self):
        entries = {
            e.word: e.gloss
            for e in iter_donor_entries(_dump(self.dir, "ru", RUSSIAN), "ru")
        }
        self.assertEqual(entries["море"], "sea")

    def test_malformed_lines_are_skipped(self):
        """900 MB'lik bir dökümde tek bozuk satır indeksi engellememeli."""
        path = self.dir / "x.jsonl"
        path.write_text('{bozuk\n{"word": "стол", "senses": []}\n', encoding="utf-8")
        self.assertEqual(len(list(iter_donor_entries(path, "ru"))), 1)


class TestDonorIndexQueries(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.index = DonorIndex(self.dir / "donors.db")
        self.index.build(sources={"ru": _dump(self.dir, "ru", RUSSIAN)})

    def tearDown(self):
        self._tmp.cleanup()

    def test_sense_restricted_lookup(self):
        rows = self.index.by_sense("table")
        self.assertEqual([r["word"] for r in rows], ["стол"])

    def test_sense_lookup_is_empty_for_unknown_concept(self):
        self.assertEqual(self.index.by_sense("VOLCANO"), [])

    def test_short_sense_tokens_are_ignored(self):
        """``I``, ``we`` gibi tek-iki harfli kavramlar FTS'te gürültüdür."""
        self.assertEqual(self.index.by_sense("I"), [])

    def test_unconstrained_candidates_use_a_length_window(self):
        """``n`` düzenleme uzaklığındaki biçmin uzunluk farkı da en çok
        ``n``dir; pencere dışı zaten elenirdi."""
        rows = self.index.candidates("stol", max_length_gap=0)
        self.assertTrue(all(len(r["comparison"]) == 4 for r in rows))

    def test_missing_index_is_safe(self):
        empty = DonorIndex(self.dir / "yok.db")
        self.assertFalse(empty.exists)
        self.assertEqual(empty.by_sense("table"), [])
        self.assertEqual(empty.candidates("stol"), [])
        self.assertFalse(empty.stats()["exists"])

    def test_build_without_sources_raises(self):
        with self.assertRaises(FileNotFoundError):
            DonorIndex(self.dir / "y.db").build(sources={})

    def test_stats_reports_totals(self):
        self.assertEqual(self.index.stats()["total_entries"], 3)


class TestReverseBorrowingFilter(unittest.TestCase):
    """⚠️ Verici sözlüğü **Türkiden alınmış** kelimeleri de içerir.

    Ölçüldü: Türkçe ``göz`` Ermenice ``գյոզ``a SCA 0,040 uzaklıktaydı ve
    "alıntı kanıtı" sayılıyordu — oysa o madde tam TERSİNİN kanıtıdır.
    Süzgeç WOLD/Sakha'da motorun F'sini 0,5664'ten 0,5839'a çıkardı.
    """

    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.index = DonorIndex(self.dir / "donors.db")
        self.index.build(sources={"hy": _dump(self.dir, "hy", ARMENIAN)})

    def tearDown(self):
        self._tmp.cleanup()

    def test_turkic_sourced_entry_is_excluded_from_evidence(self):
        self.assertEqual([r["word"] for r in self.index.by_sense("eye")], [])

    def test_genuinely_inherited_donor_entry_is_kept(self):
        self.assertEqual([r["word"] for r in self.index.by_sense("arm")], ["կուռ"])

    def test_excluded_count_is_reported_not_hidden(self):
        self.assertEqual(self.index.stats()["from_turkic_excluded"], 1)

    def test_comparison_uses_kaikki_romanisation(self):
        """``to_comparison_form`` Ermeni alfabesini tümden siler
        (``գիրք`` -> ``""``); kaikki'nin kendi çevriyazısı kullanılır."""
        rows = self.index.by_sense("arm")
        self.assertEqual(rows[0]["comparison"], "kuŕ")


class TestTurkicSourceDetection(unittest.TestCase):
    def test_borrowed_from_turkic_is_detected(self):
        self.assertTrue(is_from_turkic("Borrowed from Turkic. Compare Azerbaijani dəmir."))

    def test_ottoman_turkish_is_detected(self):
        self.assertTrue(is_from_turkic("From Ottoman Turkish گوز (göz)."))

    def test_mere_comparison_is_not_a_borrowing_claim(self):
        """"Compare Turkish …" bir alıntı beyanı DEĞİLDİR; ayırmazsak Türki
        bir adı anan her madde elenir ve gerçek vericiler de kaybolur."""
        self.assertFalse(is_from_turkic("From Proto-Iranian. Compare Turkish demir."))

    def test_unrelated_etymology(self):
        self.assertFalse(is_from_turkic("Inherited from Old Armenian կուռն."))

    def test_empty(self):
        self.assertFalse(is_from_turkic(""))


class TestScaDistance(unittest.TestCase):
    """⚠️ Bu testler LingPy varsa anlamlıdır; yoksa atlanır."""

    def setUp(self):
        dp.reset_cache()
        if dp._pairwise() is None:
            self.skipTest("LingPy kurulu değil")

    def test_adapted_borrowing_is_close(self):
        """Sakha Rusça ``stol``u ``ostuol`` yapar: öntüreme ünlü + ikizünlü.

        Düz Levenshtein 3/6 = 0,50 verir ve eşiğin üstünde kalır; SCA ses
        sınıflarıyla çalıştığı için aynı çift 0,22 civarındadır.
        """
        self.assertLess(dp.sca_distance("ostuol", "stol"), dp.DONOR_DISTANCE_THRESHOLD)

    def test_unrelated_forms_are_far(self):
        self.assertGreater(dp.sca_distance("bagana", "çuçka"), 0.5)

    def test_empty_input_is_no_evidence(self):
        """Kanıt yokluğu 1,0'dır — yanlış bir kanıt değil."""
        self.assertEqual(dp.sca_distance("", "stol"), 1.0)


class TestProximityStrength(unittest.TestCase):
    def _match(self, distance: float) -> dp.DonorMatch:
        return dp.DonorMatch("ru", "стол", "stol", "table", distance, True)

    def test_below_threshold_is_full_strength(self):
        self.assertEqual(dp.proximity_strength(self._match(0.10)), 1.0)

    def test_above_ceiling_is_zero(self):
        self.assertEqual(dp.proximity_strength(self._match(0.90)), 0.0)

    def test_transition_is_gradual(self):
        """0,349 ile 0,351'in kararı ters çevirmesi için bir sebep yok."""
        middle = dp.proximity_strength(self._match(0.475))
        self.assertGreater(middle, 0.0)
        self.assertLess(middle, 1.0)

    def test_no_match_is_zero(self):
        self.assertEqual(dp.proximity_strength(None), 0.0)


class TestSignalIsDisabledWithoutData(unittest.TestCase):
    def test_missing_donor_index_returns_none(self):
        """Verici indeksi yoksa sinyal susar — uydurma bir mesafe vermez."""
        with TemporaryDirectory() as tmp:
            dp.reset_cache()
            with mock.patch.object(dp, "_index", lambda: DonorIndex(Path(tmp) / "yok.db")):
                self.assertIsNone(dp.nearest_donor("ostuol", "table"))
        dp.reset_cache()

    def test_missing_lingpy_disables_the_signal(self):
        """⚠️ Düz Levenshtein'a DÜŞÜLMEZ: yayınlanmış F1 0,806 SCA ile
        ölçülmüştür, başka mesafeyle o sayı iddia edilemez."""
        dp.reset_cache()
        with mock.patch.object(dp, "_pairwise", lambda: None):
            self.assertIsNone(dp.nearest_donor("ostuol", "table"))
            self.assertEqual(dp.sca_distance("ostuol", "stol"), 1.0)
        dp.reset_cache()


if __name__ == "__main__":
    unittest.main()
