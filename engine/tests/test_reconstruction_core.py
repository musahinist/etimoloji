"""
Rekonstrüksiyon çekirdeği testleri — hizalama ve ata ses seçimi.

Buradaki her test **ölçülmüş bir hataya** karşılık gelir. Regresyon olursa
hangi bozulmanın geri geldiği test adından okunur.
"""

from __future__ import annotations

import unittest

from engine.nlp.comparative_reconstruction import ComparativeReconstructor
from engine.nlp.multi_alignment import ORTHO_TO_IPA, align_forms
from engine.nlp.proto_phonology import (
    ARCHAISM_WEIGHTS,
    VOWEL_ARCHAISM_WEIGHTS,
    pick_proto_sound,
    weight_for,
)


def entries(pairs):
    return [{"lang_code": code, "word": word} for code, word in pairs]


class TestMultipleAlignment(unittest.TestCase):
    def test_two_forms_produce_columns(self):
        columns = align_forms({"tr": "göz", "cv": "kus"})
        self.assertEqual(len(columns), 3)

    def test_single_form_yields_nothing(self):
        self.assertEqual(align_forms({"tr": "göz"}), [])

    def test_width_comes_from_longest_not_from_anchor(self):
        """Regresyon: ata biçim sorgu kelimesinin uzunluğuna KİLİTLİYDİ.

        `tr` yalnız iki harfli olsa da `sub`/`suv` üçüncü bir sütun üretmeli;
        aksi hâlde ``*sub`` yerine ``*su`` çıkar.
        """
        columns = align_forms({"tr": "su", "otk": "sub", "tk": "suv"})
        self.assertGreaterEqual(len(columns), 3)

    def test_initial_consonant_gets_its_own_column(self):
        """Regresyon: LingPy ``yol``u ``['yo','l']`` diye bölüyordu.

        Bölünme yanlış olunca söz başı ünsüz sütunu hiç oluşmuyor ve
        ``*jol`` yerine ``*yol`` üretiliyordu.
        """
        columns = align_forms({"tr": "yol", "kk": "jol", "cv": "sul"})
        first = columns[0].present
        self.assertIn("tr", first)
        self.assertEqual(first["tr"], "y")

    def test_columns_report_original_orthography_not_ipa(self):
        """Hizalama IPA üzerinden yapılır ama sütunlar özgün harf taşımalı."""
        columns = align_forms({"tr": "göz", "kk": "köz"})
        sounds = {s for column in columns for s in column.distinct}
        self.assertIn("ö", sounds)
        self.assertNotIn("ø", sounds)

    def test_ipa_map_is_one_char_to_one_char(self):
        """Konumların geri eşlenebilmesi buna bağlıdır."""
        for source, target in ORTHO_TO_IPA.items():
            with self.subTest(source=source):
                self.assertEqual(len(source), 1)
                self.assertEqual(len(target), 1)


class TestArchaismWeights(unittest.TestCase):
    def test_chuvash_heavy_on_consonants_light_on_vowels(self):
        """Oğur kolu ünsüzde tanısal, ünlüde güvenilmezdir.

        Regresyon: tek ağırlık tablosu ``*köŕ`` yerine ``*kuŕ`` üretiyordu —
        Çuvaşça'nın ``u``su dört dilin ``ö``südüne karşı kazanıyordu.
        """
        self.assertGreater(weight_for("cv", "r"), weight_for("tr", "r"))
        self.assertLess(weight_for("cv", "ö"), weight_for("tr", "ö"))

    def test_length_preserving_languages_lead_on_vowels(self):
        for lang in ("klj", "tk", "sah"):
            with self.subTest(lang=lang):
                self.assertGreater(weight_for(lang, "a"), weight_for("tr", "a"))

    def test_oghuz_is_least_trusted_on_initials(self):
        """Oğuz kolu söz başı ötümlüleşme YENİLİĞİNİ yapmıştır."""
        self.assertLess(ARCHAISM_WEIGHTS["tr"], ARCHAISM_WEIGHTS["otk"])
        self.assertLess(ARCHAISM_WEIGHTS["az"], ARCHAISM_WEIGHTS["klj"])

    def test_every_vowel_weight_has_a_consonant_counterpart(self):
        for lang in VOWEL_ARCHAISM_WEIGHTS:
            with self.subTest(lang=lang):
                self.assertIn(lang, ARCHAISM_WEIGHTS, lang)


class TestProtoSoundSelection(unittest.TestCase):
    def _column(self, sounds):
        columns = align_forms({lang: form for lang, form in sounds.items()})
        return columns[0] if columns else None

    def test_rotacism_needs_oghur_witness(self):
        from engine.nlp.multi_alignment import AlignedColumn

        with_oghur = AlignedColumn(0, {"cv": "r", "tr": "z", "kk": "z"}, 1)
        self.assertEqual(pick_proto_sound(with_oghur, "final").sound, "ŕ")

        without_oghur = AlignedColumn(0, {"tt": "r", "tr": "z", "kk": "z"}, 1)
        self.assertNotEqual(pick_proto_sound(without_oghur, "final").sound, "ŕ")

    def test_rotacism_does_not_fire_word_initially(self):
        """Regresyon: ``*jan`` yerine ``*ŕan`` üretiliyordu (11 kelime).

        Rotasizm ve lambdaizm söz başında YOKTUR.
        """
        from engine.nlp.multi_alignment import AlignedColumn

        column = AlignedColumn(0, {"cv": "s", "tr": "j", "kk": "z"}, 1)
        self.assertNotEqual(pick_proto_sound(column, "initial").sound, "ŕ")

    def test_dental_rule_requires_diagnostic_sound(self):
        """Regresyon: ``{d,t}`` sütunu ``*d`` veriyordu.

        ``*d̮`` denkliğinin tanısal sesleri ``y`` ve ``z``dir; yoksa ortada
        yalnız ötümlülük değişimi vardır ve ata ses ``*t``tir
        (``*jumurtka`` yerine ``*yumurdka`` çıkıyordu).
        """
        from engine.nlp.multi_alignment import AlignedColumn

        plain = AlignedColumn(0, {"tr": "d", "kk": "t"}, 1)
        self.assertEqual(pick_proto_sound(plain, "medial").sound, "t")

        diagnostic = AlignedColumn(0, {"tr": "y", "kk": "z", "otk": "d"}, 1)
        self.assertEqual(pick_proto_sound(diagnostic, "medial").sound, "d")

    def test_partial_overlap_does_not_fire_a_rule(self):
        """Kural sütundaki BÜTÜN sesleri açıklamalıdır.

        Regresyon: kısmi örtüşme ``*arka -> *arca`` gibi bozulmalar üretiyordu.
        """
        from engine.nlp.multi_alignment import AlignedColumn

        column = AlignedColumn(0, {"tr": "k", "kk": "k", "cv": "ç"}, 1)
        decision = pick_proto_sound(column, "medial")
        self.assertNotEqual(decision.method, "denklik")

    def test_unanimous_column_is_kept_as_is(self):
        from engine.nlp.multi_alignment import AlignedColumn

        column = AlignedColumn(0, {"tr": "l", "kk": "l", "cv": "l"}, 1)
        decision = pick_proto_sound(column, "final")
        self.assertEqual(decision.sound, "l")
        self.assertEqual(decision.agreement, 1.0)


class TestReconstructionEndToEnd(unittest.TestCase):
    """Bilinen ata biçimler — Türkolojide tartışmasız olanlar."""

    def setUp(self):
        self.engine = ComparativeReconstructor()

    def test_known_proto_forms(self):
        cases = {
            "*köŕ": [("tr", "göz"), ("kk", "көз"), ("cv", "куҫ"), ("tt", "күз")],
            "*teŋiŕ": [("tr", "deniz"), ("kk", "теңіз"), ("cv", "тинӗс"), ("tk", "deňiz")],
            "*jol": [("tr", "yol"), ("kk", "жол"), ("cv", "ҫул"), ("tt", "юл")],
        }
        for expected, pairs in cases.items():
            with self.subTest(expected=expected):
                out = self.engine.reconstruct(pairs[0][1], entries(pairs))
                self.assertEqual(out["reconstructed_root"], expected)

    def test_final_consonant_not_truncated_by_query_length(self):
        """``*sub`` regresyonu: sorgu ``su`` iki harfli ama ata biçim üç."""
        out = self.engine.reconstruct(
            "su", entries([("otk", "sub"), ("cv", "şıv"), ("tk", "suv")])
        )
        self.assertEqual(out["reconstructed_root"], "*sub")

    def test_proto_level_labelled_pt_when_oghur_present(self):
        out = self.engine.reconstruct(
            "göz", entries([("tr", "göz"), ("cv", "куҫ"), ("kk", "көз")])
        )
        self.assertEqual(out["proto_level"], "PT")

    def test_proto_level_falls_back_to_pct_without_oghur(self):
        """Çuvaşça yoksa iddia Proto-Türkçe DEĞİL, Ana Ortak Türkçe'dir."""
        out = self.engine.reconstruct(
            "göz", entries([("tr", "göz"), ("kk", "көз"), ("tt", "күз")])
        )
        self.assertEqual(out["proto_level"], "PCT")
        self.assertIn("Ana Ortak Türkçe", out["proto_level_note"])

    def test_confidence_is_lower_without_oghur_witness(self):
        with_oghur = self.engine.reconstruct(
            "göz", entries([("tr", "göz"), ("kk", "көз"), ("cv", "куҫ")])
        )
        without = self.engine.reconstruct(
            "göz", entries([("tr", "göz"), ("kk", "көз"), ("tt", "күз")])
        )
        self.assertGreater(with_oghur["confidence"], without["confidence"])

    def test_column_agreement_dominates_confidence(self):
        """Ölçüm: sütun uyumu tek gerçek sinyal (AUC 0,730).

        Uyumsuz bir küme, tanık sayısı fazla olsa bile düşük güven almalı.
        """
        agreeing = self.engine.reconstruct(
            "kan", entries([("kk", "kan"), ("tt", "kan"), ("ky", "kan")])
        )
        disagreeing = self.engine.reconstruct(
            "kan", entries([("kk", "xun"), ("tt", "qam"), ("ky", "sil"), ("uz", "pot")])
        )
        self.assertGreater(agreeing["confidence"], disagreeing["confidence"])

    def test_single_datum_is_labelled_as_not_comparative(self):
        """Tek veri noktasında karşılaştırmalı yöntem UYGULANAMAZ.

        Motor sorgu biçmini aday olarak döndürür ama `method` alanı bunun
        türetilmediğini söyler ve güven sıfırdır. Sessizce "rekonstrüksiyon"
        diye sunmak, yapılmamış bir işi yapılmış göstermek olurdu.
        """
        out = self.engine.reconstruct("göz", entries([("tr", "göz")]))
        self.assertEqual(out.get("method"), "anchor_fallback")
        self.assertFalse(out["evidence_available"])
        self.assertEqual(out.get("confidence"), 0.0)

    def test_query_word_counts_as_a_witness(self):
        """Regresyon: iki dilli kümelerde motor gereksiz çekimser kalıyordu.

        Sorgu kelimesi tanıktan farklıysa o da bir veri noktasıdır
        (400 maddenin 70'i bu yüzden cevapsız kalıyordu).
        """
        out = self.engine.reconstruct("göz", entries([("cv", "kus")]))
        self.assertTrue(out["is_reconstructible"])

    def test_deterministic(self):
        pairs = entries([("tr", "göz"), ("kk", "көз"), ("cv", "куҫ")])
        first = self.engine.reconstruct("göz", pairs)
        second = self.engine.reconstruct("göz", pairs)
        self.assertEqual(first["reconstructed_root"], second["reconstructed_root"])
        self.assertEqual(first["confidence"], second["confidence"])


if __name__ == "__main__":
    unittest.main()


class TestCenterStarFallback(unittest.TestCase):
    """LingPy yoksa hizalama durmaz, merkez-yıldız yedeğine düşer."""

    def test_fallback_produces_columns(self):
        from unittest import mock

        from engine.nlp import multi_alignment

        with mock.patch.object(multi_alignment, "_lingpy_multiple", lambda: None):
            columns = multi_alignment.align_forms({"tr": "göz", "kk": "köz", "cv": "kus"})
        self.assertGreaterEqual(len(columns), 3)

    def test_fallback_handles_different_lengths(self):
        from unittest import mock

        from engine.nlp import multi_alignment

        with mock.patch.object(multi_alignment, "_lingpy_multiple", lambda: None):
            columns = multi_alignment.align_forms({"tr": "su", "otk": "sub", "tk": "suv"})
        self.assertGreaterEqual(len(columns), 3)

    def test_lingpy_failure_falls_back_instead_of_crashing(self):
        from unittest import mock

        from engine.nlp import multi_alignment

        class Exploding:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("patlak")

        with mock.patch.object(multi_alignment, "_lingpy_multiple", lambda: Exploding):
            columns = multi_alignment.align_forms({"tr": "göz", "kk": "köz"})
        self.assertGreaterEqual(len(columns), 2)

    def test_needleman_wunsch_aligns_and_gaps(self):
        from engine.nlp.multi_alignment import _needleman_wunsch

        a, b = _needleman_wunsch("sub", "su")
        self.assertEqual(len(a), len(b))
        self.assertIn("-", b)

    def test_empty_forms_are_dropped(self):
        from engine.nlp.multi_alignment import align_forms

        self.assertEqual(align_forms({"tr": "", "kk": ""}), [])


class TestAlignedColumnProperties(unittest.TestCase):
    def test_gap_ratio_and_distinct(self):
        from engine.nlp.multi_alignment import GAP, AlignedColumn

        column = AlignedColumn(0, {"a": "k", "b": "k", "c": GAP}, 1)
        self.assertAlmostEqual(column.gap_ratio, 1 / 3)
        self.assertEqual(column.distinct, frozenset({"k"}))
        self.assertEqual(set(column.present), {"a", "b"})

    def test_empty_column(self):
        from engine.nlp.multi_alignment import AlignedColumn

        column = AlignedColumn(0, {}, 0)
        self.assertEqual(column.gap_ratio, 1.0)
        self.assertEqual(column.distinct, frozenset())


class TestPlausibilityGuard(unittest.TestCase):
    """Ata biçmin kendisi Türkçe olabilir mi?"""

    def test_well_formed_roots_pass(self):
        from engine.nlp.proto_phonology import proto_plausibility

        for form in ("*köŕ", "*teŋiŕ", "*sub", "*kāpuk", "*tūŕ"):
            with self.subTest(form=form):
                score, violations = proto_plausibility(form)
                self.assertEqual(score, 1.0, f"{form}: {violations}")

    def test_vowelless_root_is_rejected(self):
        from engine.nlp.proto_phonology import proto_plausibility

        score, violations = proto_plausibility("*zzzky")
        self.assertEqual(score, 0.0)
        self.assertTrue(any("ünlü" in v for v in violations))

    def test_prohibited_initial_is_penalised(self):
        from engine.nlp.proto_phonology import proto_plausibility

        score, violations = proto_plausibility("*firak")
        self.assertLess(score, 1.0)
        self.assertTrue(any("söz başı" in v for v in violations))

    def test_length_marks_do_not_count_as_missing_vowels(self):
        """``ā`` bir ünlüdür; normalize edilmezse *kāpuk 'ünlüsüz' sayılıyordu."""
        from engine.nlp.proto_phonology import proto_plausibility

        self.assertEqual(proto_plausibility("*kāpuk")[0], 1.0)

    def test_empty_form(self):
        from engine.nlp.proto_phonology import proto_plausibility

        self.assertEqual(proto_plausibility("")[0], 0.0)
