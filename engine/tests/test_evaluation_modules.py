"""
Değerlendirme modüllerinin testleri: koşum takımı, akraba/tahmin/alıntı
ölçümleri ve rapor üretimi.

Bu modüller sayıyı ÜRETEN koddur; bozulurlarsa yanlış sayı raporlanır ve
kimse fark etmez. Testler özellikle **ölçümü geçersiz kılan** durumları
hedefler: sızıntı, döngüsellik, eşleşmeyen diziler, eşiğin yanlış veride
seçilmesi.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from engine.config import CLDF_DIR
from engine.evaluation import cognate_eval, prediction_eval
from engine.evaluation.borrowing_eval import (
    PRF,
    BorrowingCase,
    always_borrowed,
    always_inherited,
    load_wold_cases,
    score_system,
    tune_threshold,
)
from engine.evaluation.harness import (
    HarnessResult,
    _anchor_for,
    _witnesses_for,
    classify_error,
    comparative_reconstructor,
    run,
)
from engine.evaluation.metrics import ReconstructionScore

HAS_CLDF = (CLDF_DIR / "savelyevturkic" / "forms.csv").exists()
HAS_WOLD = (CLDF_DIR / "wold" / "forms.csv").exists()


class TestErrorClassification(unittest.TestCase):
    """Hata modu dökümü, toplam skordan daha çok yol gösterir."""

    def _item(self, has_length=True):
        from engine.evaluation.gold import GoldItem

        return GoldItem(
            set_id="x",
            gold_form="*köŕ",
            gold_candidates=("*köŕ",),
            concept="c",
            concepticon_gloss="EYE",
            proto_level="PT",
            witnesses={},
            has_length_witness=has_length,
            split="dev",
        )

    def test_correct(self):
        self.assertEqual(classify_error("*köŕ", "*köŕ", self._item()), "dogru")

    def test_empty_is_abstention(self):
        self.assertEqual(classify_error("", "*köŕ", self._item()), "cekimser")

    def test_vowel_length_only(self):
        self.assertEqual(classify_error("*tur", "*tūr", self._item()), "unlu_uzunlugu")

    def test_missed_rotacism(self):
        self.assertEqual(
            classify_error("*koz", "*koŕ", self._item()), "rotasizm_lambdaizm_kacirildi"
        )

    def test_unstripped_suffix(self):
        self.assertEqual(classify_error("*icmek", "*ic", self._item()), "ek_soyulmamis")

    def test_too_short(self):
        self.assertEqual(classify_error("*su", "*subak", self._item()), "capa_kisa")

    def test_wrong_initial(self):
        self.assertEqual(classify_error("*mor", "*kor", self._item()), "soz_basi_yanlis")


class TestAnchorSelection(unittest.TestCase):
    def test_prefers_turkish(self):
        witnesses = [
            {"lang_code": "kk", "word": "köz"},
            {"lang_code": "tr", "word": "göz"},
        ]
        self.assertEqual(_anchor_for(witnesses), ("göz", "tr"))

    def test_falls_back_to_any_language(self):
        form, lang = _anchor_for([{"lang_code": "cv", "word": "kus"}])
        self.assertEqual((form, lang), ("kus", "cv"))

    def test_skips_forms_that_normalise_to_nothing(self):
        """Normalize edilince boş kalan biçim çapa olamaz.

        Regresyon: `?` gibi belirsizlik işaretleri çapa seçilince 24 madde
        haksız yere çekimser sayılıyordu.
        """
        witnesses = [
            {"lang_code": "tr", "word": "???"},
            {"lang_code": "kk", "word": "köz"},
        ]
        self.assertEqual(_anchor_for(witnesses), ("köz", "kk"))

    def test_empty(self):
        self.assertEqual(_anchor_for([]), ("", ""))


class TestHarnessRun(unittest.TestCase):
    def _items(self, n=3, witnesses=None):
        from engine.evaluation.gold import GoldItem

        return [
            GoldItem(
                set_id=f"s{i}",
                gold_form="*köŕ",
                gold_candidates=("*köŕ",),
                concept=f"c{i}",
                concepticon_gloss="EYE",
                proto_level="PT",
                witnesses=witnesses
                or {"Turkish": "göz", "Kazakh": "köz", "Chuvash": "kuś"},
                has_length_witness=False,
                split="dev",
            )
            for i in range(n)
        ]

    def _mapping(self):
        return {"Turkish": "tr", "Kazakh": "kk", "Chuvash": "cv"}

    def test_every_item_leaves_a_record(self):
        """Eşleşmiş anlamlılık testi ancak diziler aynı maddeleri kapsarsa geçerli."""
        result = run(comparative_reconstructor(), self._items(4), mapping=self._mapping())
        self.assertEqual(len(result.item_correct), 4)
        self.assertEqual(len(result.item_ids), 4)

    def test_anchor_language_is_excluded_by_language_not_by_form(self):
        """Kelimeye göre çıkarmak, aynı biçmi paylaşan bütün dilleri siler."""
        items = self._items(
            1, witnesses={"Turkish": "tırnak", "Kazakh": "tırnak", "Chuvash": "çĕrne"}
        )
        result = run(comparative_reconstructor(), items, mapping=self._mapping())
        self.assertEqual(len(result.item_correct), 1)

    def test_including_the_anchor_is_a_separate_condition(self):
        with_anchor = run(
            comparative_reconstructor(),
            self._items(3),
            mapping=self._mapping(),
            exclude_anchor_language=False,
        )
        self.assertEqual(len(with_anchor.item_correct), 3)

    def test_unmapped_languages_are_dropped_not_crashed(self):
        items = self._items(1, witnesses={"Klingon": "xyz", "Kazakh": "köz"})
        result = run(comparative_reconstructor(), items, mapping=self._mapping())
        self.assertEqual(len(result.item_correct), 1)

    def test_witnesses_never_leak_the_gold_form(self):
        from engine.evaluation.gold import GoldItem

        item = GoldItem(
            set_id="s",
            gold_form="*köŕ",
            gold_candidates=("*köŕ",),
            concept="c",
            concepticon_gloss="EYE",
            proto_level="PT",
            witnesses={"Kazakh": "köz"},
            has_length_witness=False,
            split="dev",
        )
        forms = {w["word"] for w in _witnesses_for(item, self._mapping())}
        self.assertNotIn("*köŕ", forms)

    def test_result_is_serialisable(self):
        result = run(comparative_reconstructor(), self._items(2), mapping=self._mapping())
        data = result.as_dict()
        for key in ("accuracy", "acceptable", "ED", "NED", "FER", "coverage"):
            self.assertIn(key, data)

    def test_empty_items(self):
        result = HarnessResult(split="dev", system="x", score=ReconstructionScore())
        self.assertEqual(result.as_dict()["n"], 0)


class TestCognateEvalClusterers(unittest.TestCase):
    def _task(self):
        return cognate_eval.ConceptTask(
            concept="c",
            forms={"a": "göz", "b": "köz", "c": "kitap"},
            gold={"a": "1", "b": "1", "c": "2"},
        )

    def test_all_together_makes_one_cluster(self):
        self.assertEqual(len(set(cognate_eval.cluster_all_together(self._task()).values())), 1)

    def test_all_apart_makes_n_clusters(self):
        clusters = cognate_eval.cluster_all_apart(self._task())
        self.assertEqual(len(set(clusters.values())), 3)

    def test_edit_distance_separates_unrelated_forms(self):
        clusters = cognate_eval.cluster_edit_distance(self._task(), threshold=0.4)
        self.assertNotEqual(clusters["a"], clusters["c"])

    def test_edit_distance_joins_similar_forms(self):
        clusters = cognate_eval.cluster_edit_distance(self._task(), threshold=0.5)
        self.assertEqual(clusters["a"], clusters["b"])

    def test_evaluate_system_reports_bcubed(self):
        row = cognate_eval.evaluate_system(
            "x", cognate_eval.cluster_all_apart, [self._task()]
        )
        self.assertIn("bcubed_f", row)
        self.assertEqual(row["n_concepts"], 1)

    def test_skipped_clusterers_do_not_crash(self):
        row = cognate_eval.evaluate_system("x", lambda t: {}, [self._task()])
        self.assertEqual(row["skipped"], 1)

    def test_tuning_returns_a_threshold(self):
        threshold, score = cognate_eval.tune_edit_distance_threshold([self._task()])
        self.assertGreater(threshold, 0.0)
        self.assertGreaterEqual(score, 0.0)


class TestPredictionEvalSystems(unittest.TestCase):
    def _case(self):
        return prediction_eval.PredictionCase(
            concept="c",
            source_lang="tr",
            source_form="göz",
            target_lang="kk",
            target_form="köz",
        )

    def test_identity_copies_the_source(self):
        self.assertEqual(prediction_eval.predict_identity(self._case()), "göz")

    def test_handwritten_is_an_oracle(self):
        """Elle yazılmış kurallar GERÇEK CEVABA en yakın adayı seçiyor.

        Bu haksız bir avantajdır ve raporda öyle işaretlenir; öğrenilmiş
        sistemin bunu yakalaması anlamlıdır.
        """
        self.assertEqual(prediction_eval.predict_handwritten(self._case()), "köz")

    def test_scoring_counts_exact_and_near(self):
        cases = [self._case()]
        exact = prediction_eval.score(lambda c: "köz", cases)
        near = prediction_eval.score(lambda c: "küz", cases)
        self.assertEqual(exact.accuracy, 1.0)
        self.assertEqual(near.accuracy, 0.0)
        self.assertEqual(near.near_miss_rate, 1.0)

    def test_empty_prediction_is_handled(self):
        result = prediction_eval.score(lambda c: "", [self._case()])
        self.assertEqual(result.accuracy, 0.0)

    def test_score_dict_shape(self):
        data = prediction_eval.score(lambda c: "köz", [self._case()]).as_dict()
        for key in ("n", "accuracy", "within_one_edit", "ED", "NED"):
            self.assertIn(key, data)


class TestBorrowingEvalMechanics(unittest.TestCase):
    def _cases(self):
        return [
            BorrowingCase(word="kitap", lang_code="tr", is_borrowed=True),
            BorrowingCase(word="göz", lang_code="tr", is_borrowed=False),
            BorrowingCase(word="duvar", lang_code="tr", is_borrowed=True),
            BorrowingCase(word="deniz", lang_code="tr", is_borrowed=False),
        ]

    def test_prf_arithmetic(self):
        result = score_system(always_borrowed, self._cases())
        self.assertEqual(result.tp, 2)
        self.assertEqual(result.fp, 2)
        self.assertEqual(result.recall, 1.0)
        self.assertEqual(result.precision, 0.5)

    def test_always_inherited_has_zero_recall(self):
        result = score_system(always_inherited, self._cases())
        self.assertEqual(result.recall, 0.0)
        self.assertEqual(result.fscore, 0.0)

    def test_accuracy_and_fscore_disagree_for_trivial_systems(self):
        """Bu yüzden ikisi birden raporlanır."""
        borrowed = score_system(always_borrowed, self._cases())
        inherited = score_system(always_inherited, self._cases())
        self.assertGreater(borrowed.fscore, inherited.fscore)
        self.assertEqual(borrowed.accuracy, inherited.accuracy)

    def test_empty_prf(self):
        empty = PRF()
        self.assertEqual(empty.fscore, 0.0)
        self.assertEqual(empty.accuracy, 0.0)

    def test_threshold_tuning_picks_something_sensible(self):
        threshold, fscore = tune_threshold(self._cases(), use_chain=True)
        self.assertGreaterEqual(threshold, 0.0)
        self.assertLessEqual(threshold, 1.0)
        self.assertGreaterEqual(fscore, 0.0)


@unittest.skipUnless(HAS_WOLD, "WOLD indirilmemiş")
class TestWoldGold(unittest.TestCase):
    """Etiket okuma testleri — tanık iliştirme KAPALI.

    ``with_witnesses=True`` her madde için ileri tahmin + sözlük araması
    yapar (1.500 madde × 12 dil). Etiketlerin doğru okunduğunu sınamak için
    o iş gereksiz; testi dakikalarca uzatır.
    """

    def test_wold_yields_labelled_cases(self):
        cases = load_wold_cases(with_witnesses=False)
        self.assertGreater(len(cases), 100)

    def test_uncertain_middle_grade_is_excluded(self):
        """Kararsız kademe ölçüme sokulmaz — hem sistemi hem ölçütü cezalandırır."""
        cases = load_wold_cases(with_witnesses=False)
        self.assertTrue(all(isinstance(c.is_borrowed, bool) for c in cases))

    def test_both_classes_are_present(self):
        cases = load_wold_cases(with_witnesses=False)
        self.assertTrue(any(c.is_borrowed for c in cases))
        self.assertTrue(any(not c.is_borrowed for c in cases))


@unittest.skipUnless(HAS_CLDF, "CLDF verisi indirilmemiş")
class TestSplitIsolation(unittest.TestCase):
    """Eşik ve tablolar TRAIN'de öğrenilir, sonuç DEV'de raporlanır."""

    def test_cognate_tasks_respect_the_split(self):
        train = {t.concept for t in cognate_eval.build_tasks(split="train")}
        dev = {t.concept for t in cognate_eval.build_tasks(split="dev")}
        self.assertTrue(train)
        self.assertTrue(dev)
        self.assertEqual(train & dev, set())

    def test_prediction_cases_respect_the_split(self):
        train = {c.concept for c in prediction_eval.build_cases(split="train")}
        dev = {c.concept for c in prediction_eval.build_cases(split="dev")}
        self.assertEqual(train & dev, set())


class TestReportWriting(unittest.TestCase):
    def test_markdown_report_includes_the_honest_condition(self):
        from engine.evaluation.report import write_markdown

        payload = {
            "dataset": "d",
            "dataset_ref": "v1",
            "dataset_commit": "abcdef123456",
            "measured_at": "2026-01-01T00:00:00+00:00",
            "split": "all",
            "gold_summary": {
                "total": 10,
                "splits": {"train": 6, "dev": 2, "test": 2},
                "concept_leakage": 0,
                "proto_levels": {"PT": 3, "PCT": 7},
            },
            "unmapped_languages": [],
            "negative_controls": [
                {"battery": "b", "n": 2, "reconstructed": 0,
                 "false_positive_rate": 0.0, "strong_claim_rate": 0.0}
            ],
            "conditions": {
                "tum_veri_capa_haric": {
                    "note": "dürüst",
                    "n_items": 10,
                    "systems": {
                        "comparative": {"accuracy": 0.2, "acceptable": 0.3, "ED": 1.0,
                                        "NED": 0.2, "FER": 0.1, "coverage": 1.0},
                        "copy_anchor": {"accuracy": 0.1, "acceptable": 0.2, "ED": 2.0,
                                        "NED": 0.4, "FER": 0.3, "coverage": 1.0},
                    },
                    "significance": [{
                        "system": "comparative", "vs": "copy_anchor", "difference": 0.1,
                        "ci95": [-0.1, 0.3], "permutation_p": 0.4, "mcnemar_p": 0.4,
                        "significant_after_fdr": False,
                    }],
                },
                "15_tanik_capa_dahil": {
                    "note": "kolay", "n_items": 5,
                    "systems": {
                        "comparative": {"accuracy": 0.4, "acceptable": 0.5, "ED": 1.0,
                                        "NED": 0.2, "FER": 0.1, "coverage": 1.0},
                        "copy_anchor": {"accuracy": 0.2, "acceptable": 0.3, "ED": 2.0,
                                        "NED": 0.4, "FER": 0.3, "coverage": 1.0},
                    },
                    "significance": [],
                },
            },
        }
        with TemporaryDirectory() as tmp:
            path = write_markdown(payload, Path(tmp) / "R.md")
            text = path.read_text(encoding="utf-8")
        self.assertIn("Negatif kontroller", text)
        self.assertIn("anlamlı DEĞİL", text)
        self.assertIn("Kavram sızıntısı", text)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAS_WOLD and HAS_CLDF, "veri eksik")
class TestWitnessAttachment(unittest.TestCase):
    """⚠️ Tanıkların iliştirilmesi ölçümün geçerliliği için ZORUNLUDUR.

    Bir dönem ``witnesses`` alanı hiç doldurulmuyordu; dört alıntı
    sinyalinden ikisi tanık gerektirdiği için **yapısal olarak devre
    dışıydı** ve ablasyon "yalnız fonotaktik" ile birebir aynı çıkıyordu.
    O sonuç "sinyaller katkı sağlamıyor" diye raporlanmıştı; doğrusu
    "sinyaller hiç çalıştırılmadı"ydı.
    """

    def test_witnesses_are_attached_by_default(self):
        from engine.evaluation.borrowing_eval import load_wiktionary_cases

        cases = load_wiktionary_cases(limit=20)
        if not cases:
            self.skipTest("sözlük indeksi yok")
        self.assertTrue(any(c.witnesses for c in cases))

    def test_witness_dependent_signals_actually_fire(self):
        """Tanıklı maddede ses kanunu / yayılım sinyalleri ateşlenebilmeli."""
        from engine.evaluation.borrowing_eval import find_witnesses
        from engine.nlp.borrowing_detector import BorrowingDetector

        witnesses = find_witnesses("kitap")
        if len(witnesses) < 3:
            self.skipTest("yeterli tanık bulunamadı (sözlük indeksi eksik olabilir)")
        entries = [{"lang_code": c, "word": w} for c, w in witnesses]
        verdict = BorrowingDetector().detect("kitap", entries)
        names = {s.name for s in verdict.signals}
        self.assertIn("ses_kanunu_ihlali", names)
        self.assertIn("değişimsiz_yayılım", names)
        # Tanıksız çağrıda bu iki sinyal "yeterli tanık yok" der ve ASLA
        # ateşlenemez; tanıklıda en azından değerlendirilebilir olmalı.
        evaluated = [s for s in verdict.signals if "yeterli tanık yok" not in s.explanation]
        self.assertGreaterEqual(len(evaluated), 3)
