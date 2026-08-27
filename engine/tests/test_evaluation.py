"""
Değerlendirme koşum takımı testleri.

Bu testler "kod çalışıyor mu"dan çok **ölçüm geçerli mi** sorusunu korur:
sızıntı açılırsa, kabul kuralları gevşerse veya altın biçim ayrıştırması
bozulursa buradan kırmızı yanar.

Ağ veya indirilmiş veri gerektiren testler, veri yoksa atlanır — CI'da
veri indirilmeden de koşabilmelidir.
"""

from __future__ import annotations

import unittest

from engine.config import CLDF_DIR
from engine.evaluation.baselines import BASELINES, copy_anchor, copy_random_daughter
from engine.evaluation.gold import SPLIT_RATIOS, assign_split
from engine.evaluation.metrics import (
    bcubed_fscore,
    best_match,
    edit_distance,
    feature_error_rate,
    is_acceptable,
    normalize_proto,
    normalized_edit_distance,
    parse_gold_form,
    score_reconstructions,
)

HAS_CLDF = (CLDF_DIR / "savelyevturkic" / "forms.csv").exists()


class TestNormalisation(unittest.TestCase):
    def test_star_and_notes_removed(self):
        self.assertEqual(normalize_proto("*Kāpuk"), "kāpuk")
        self.assertEqual(normalize_proto("*Kāpuk", strip_length=True), "kapuk")

    def test_alternatives_cut_at_first(self):
        self.assertEqual(normalize_proto("*Kūrɨk,gak"), "kūrɨk")


class TestAcceptability(unittest.TestCase):
    """Kabul kuralları gevşerse doğruluk yapay olarak şişer."""

    def test_archiphoneme_tolerated(self):
        self.assertTrue(is_acceptable("*kapuk", "*Kāpuk"))
        self.assertTrue(is_acceptable("*gapuk", "*Kāpuk"))

    def test_unrelated_sound_not_tolerated(self):
        self.assertFalse(is_acceptable("*mapuk", "*Kāpuk"))

    def test_notation_only_difference_tolerated(self):
        self.assertTrue(is_acceptable("*teŋiŕ", "*teñir"))

    def test_disputed_initial_tolerated(self):
        """*g-/*k- tartışması çözülmemiştir; motor taraf tutmak zorunda değil."""
        self.assertTrue(is_acceptable("*kel", "*gẹl"))

    def test_missed_rotacism_is_not_acceptable(self):
        """Regresyon koruması: kör 'ED <= 1' kuralı bunları kabul ediyordu.

        Rotasizmi/lambdaizmi kaçırmak motorun düzeltmesi gereken ASIL
        hatadır; 'yakındı' diye kabul edilirse ölçüm anlamını yitirir.
        """
        self.assertFalse(is_acceptable("*tuz", "*tūŕ"))
        self.assertFalse(is_acceptable("*köz", "*köŕ"))
        self.assertFalse(is_acceptable("*bas", "*baĺ"))


class TestGoldFormParsing(unittest.TestCase):
    def test_ascii_colon_is_vowel_length(self):
        self.assertEqual(parse_gold_form("*o:t"), ["*ōt"])

    def test_equivalent_alternatives_kept(self):
        self.assertEqual(parse_gold_form("*jaŋï / *jeŋi"), ["*jaŋï", "*jeŋi"])

    def test_bibliography_removed(self):
        self.assertEqual(parse_gold_form("*ubak (ЭСТЯ 1, 561)"), ["*ubak"])

    def test_free_text_comment_removed(self):
        self.assertEqual(parse_gold_form("*üčük (?); According to VEWT, < Mong."), ["*üčük"])

    def test_suffix_fragment_is_not_an_alternative(self):
        """``*Kūrɨk,gak`` ikinci parçası ``*gak`` DEĞİL, ``*Kūrgak``tır.

        Parçayı ayrı aday saymak motora bedava bir doğru cevap verir.
        """
        self.assertEqual(parse_gold_form("*Kūrɨk,gak"), ["*Kūrɨk"])

    def test_explicit_starred_alternative_kept(self):
        self.assertEqual(parse_gold_form("*ti:ŕ, *tü:ŕ"), ["*tīŕ", "*tǖŕ"])


class TestBestMatch(unittest.TestCase):
    def test_any_equivalent_counts_as_correct(self):
        gold, exact, _ = best_match("*jeŋi", ("*jaŋï", "*jeŋi"))
        self.assertTrue(exact)
        self.assertEqual(gold, "*jeŋi")

    def test_no_candidates(self):
        self.assertEqual(best_match("*x", ()), ("", False, False))


class TestMetrics(unittest.TestCase):
    def test_edit_distance(self):
        self.assertEqual(edit_distance("kat", "kat"), 0)
        self.assertEqual(edit_distance("kat", "kut"), 1)
        self.assertEqual(edit_distance("", "abc"), 3)

    def test_normalized_edit_distance_bounded(self):
        self.assertEqual(normalized_edit_distance("abc", "abc"), 0.0)
        self.assertEqual(normalized_edit_distance("abc", "xyz"), 1.0)

    def test_feature_error_rate_separates_near_from_far(self):
        """ED ikisini de 1 sayar; FER yakın hatayı alakasız hatadan ayırır."""
        near = feature_error_rate("*göŕ", "*köŕ")
        far = feature_error_rate("*mat", "*köŕ")
        self.assertLess(near, far)

    def test_bcubed_perfect_and_worst(self):
        perfect = bcubed_fscore({"a": 1, "b": 1}, {"a": 9, "b": 9})
        self.assertEqual(perfect["fscore"], 1.0)
        split = bcubed_fscore({"a": 1, "b": 2}, {"a": 9, "b": 9})
        self.assertLess(split["fscore"], 1.0)

    def test_abstentions_count_against_coverage_not_accuracy(self):
        """Çekimserlik hata değildir ama bedava da değildir."""
        score = score_reconstructions([("*a", "*a")], abstentions=1)
        self.assertEqual(score.n, 2)
        self.assertEqual(score.exact, 1)
        self.assertEqual(score.accuracy, 0.5)
        self.assertEqual(score.coverage, 0.5)


class TestSplitAssignment(unittest.TestCase):
    def test_deterministic(self):
        """Aynı kavram her koşuda aynı bölüme düşmeli — tohum yok, özet var."""
        self.assertEqual(assign_split("1_fingernailn"), assign_split("1_fingernailn"))

    def test_all_splits_reachable(self):
        seen = {assign_split(f"kavram-{i}") for i in range(400)}
        self.assertEqual(seen, set(SPLIT_RATIOS))

    def test_ratios_roughly_respected(self):
        counts = {name: 0 for name in SPLIT_RATIOS}
        for i in range(4000):
            counts[assign_split(f"c{i}")] += 1
        for name, ratio in SPLIT_RATIOS.items():
            self.assertAlmostEqual(counts[name] / 4000, ratio, delta=0.03)


class TestBaselines(unittest.TestCase):
    def test_copy_anchor_returns_query_unchanged(self):
        out = copy_anchor("göz", [])
        self.assertEqual(out["reconstructed_root"], "*göz")

    def test_random_daughter_is_seeded(self):
        entries = [{"lang_code": "tr", "word": "a"}, {"lang_code": "kk", "word": "b"}]
        first = copy_random_daughter("x", entries)["reconstructed_root"]
        second = copy_random_daughter("x", entries)["reconstructed_root"]
        self.assertEqual(first, second)

    def test_all_baselines_return_required_keys(self):
        entries = [{"lang_code": "tr", "word": "göz"}, {"lang_code": "cv", "word": "kus"}]
        for name, fn in BASELINES.items():
            with self.subTest(baseline=name):
                out = fn("göz", entries)
                self.assertIn("reconstructed_root", out, name)
                self.assertIn("is_reconstructible", out, name)


@unittest.skipUnless(HAS_CLDF, "CLDF verisi indirilmemiş (make data)")
class TestGoldStandardIntegrity(unittest.TestCase):
    """İndirilmiş gerçek veri üzerinde sızıntı ve bütünlük denetimi."""

    @classmethod
    def setUpClass(cls):
        from engine.evaluation.gold import GoldStandard

        cls.gold = GoldStandard.build("savelyevturkic")

    def test_no_concept_leaks_across_splits(self):
        """Bir kavram iki bölüme birden düşerse ölçüm geçersizdir."""
        self.assertEqual(self.gold.concept_leakage(), [])

    def test_test_split_requires_explicit_consent(self):
        with self.assertRaises(PermissionError):
            self.gold.split("test")
        self.assertTrue(self.gold.split("test", i_am_writing_the_final_report=True))

    def test_every_item_has_a_reconstruction(self):
        for item in self.gold.items:
            self.assertTrue(item.gold_candidates, item.set_id)
            self.assertTrue(item.gold_form.startswith("*"), item.gold_form)

    def test_proto_level_labelled(self):
        """Çuvaşça tanığı yoksa iddia *PT değil *PCT olmalıdır."""
        for item in self.gold.items:
            self.assertIn(item.proto_level, ("PT", "PCT"))
        pt_items = [i for i in self.gold.items if i.proto_level == "PT"]
        self.assertLess(
            len(pt_items) / len(self.gold.items),
            0.5,
            "Türki verisinde kümelerin çoğunda Oğur tanığı YOKTUR; "
            "PT oranı %50'yi aşıyorsa etiketleme bozulmuştur.",
        )


@unittest.skipUnless(HAS_CLDF, "CLDF verisi indirilmemiş (make data)")
class TestHarnessLeakage(unittest.TestCase):
    def test_witnesses_never_contain_the_gold_form(self):
        """Motora verilen girdide altın cevap bulunmamalıdır."""
        from engine.db.cldf_wordlist import CldfWordlist
        from engine.db.language_mapping import build_mapping
        from engine.evaluation.gold import GoldStandard
        from engine.evaluation.harness import _witnesses_for

        gold = GoldStandard.build("savelyevturkic")
        mapping = build_mapping(CldfWordlist.load("savelyevturkic"))
        for item in gold.items[:50]:
            witnesses = _witnesses_for(item, mapping)
            words = {w["word"] for w in witnesses}
            for candidate in item.gold_candidates:
                self.assertNotIn(candidate, words, item.set_id)
            for word in words:
                self.assertFalse(word.startswith("*"), f"{item.set_id}: {word}")


if __name__ == "__main__":
    unittest.main()
