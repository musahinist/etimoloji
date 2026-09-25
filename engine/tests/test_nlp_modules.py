"""
NLP Katmanı Birim Testleri

Daha önce test edilmemiş 10 NLP modülünü ve Faz 6'da eklenen yeni modülleri
kapsar.
"""
from __future__ import annotations

import unittest

from engine.nlp.cognate_alignment import CognateAlignmentEngine
from engine.nlp.cognate_clustering import CognateClusterEngine
from engine.nlp.comparative_reconstruction import ComparativeReconstructor
from engine.nlp.derivation_network import DerivationNetworkBuilder
from engine.nlp.diachronic_semantic_engine import DiachronicSemanticEngine
from engine.nlp.donor_lexicon import DonorLexicon, ipa_distance, levenshtein
from engine.nlp.donor_search import DonorSearchEngine
from engine.nlp.historical_attestation_verifier import HistoricalAttestationVerifier
from engine.nlp.historical_morphology import HistoricalMorphologyAnalyzer
from engine.nlp.iterative_hypothesis_engine import IterativeHypothesisEngine
from engine.nlp.loanword_classifier import LoanwordClassifier
from engine.nlp.loanword_detector import LoanwordDetector
from engine.nlp.neologism_detector import NeologismDetector
from engine.nlp.reconstruction import ProtoTurkicReconstructor
from engine.nlp.sound_law_induction import SoundLawInductionEngine
from engine.nlp.unsupervised_morpheme_segmenter import UnsupervisedMorphemeSegmenter
from engine.tests.data_guards import needs_index


def entries(pairs, origin="live"):
    return [
        {"lang_code": c, "word": w, "source": f"S{i}", "origin": origin,
         "lang_name": c, "meaning": "test", "script": "Latin"}
        for i, (c, w) in enumerate(pairs)
    ]


GOZ = entries([("tr", "göz"), ("az", "göz"), ("kk", "көз"), ("tt", "күз"),
               ("cv", "куҫ"), ("otk", "köz"), ("ky", "көз"), ("ba", "күҙ")])


class TestAttestedLoanBlocksReconstruction(unittest.TestCase):
    """Sözlüğün DOĞRUDAN tanıklığı miras rekonstrüksiyonunu engellemeli.

    Alıntı tuzaklarında `zincir_kanıtı` ateşleniyor ama toplam skor tam
    0.50'de kalıp muhafazakâr eşiğin (0.55) kılpayı altında kalıyordu;
    yalnız `kitap` (0.59) engelleniyordu. Ölçüldü: batarya
    `alinti_tuzagi` oranı 0.800 -> 0.000.
    """

    TRAPS = ("kitap", "duvar", "çorap", "pencere", "sabun")
    #: Engellenmemesi gerekenler. Üçü ölçülmüş risk vakası:
    #: `gül` (miras kanıtı `gülmek` mastarında), `yaz`/`öküz` (indekste
    #: ters yönlü "alıntı" kaydı var, çekirdek Ortak Türkçe).
    PROTECTED = ("gül", "yaz", "öküz", "göz", "bardak", "deniz", "su", "çay", "kat")

    @needs_index
    def test_traps_are_blocked(self):
        from engine.nlp.borrowing_detector import BorrowingDetector

        d = BorrowingDetector()
        for w in self.TRAPS:
            with self.subTest(word=w):
                self.assertTrue(d.detect(w).blocks_inherited_reconstruction, w)

    def test_native_words_are_not_blocked(self):
        from engine.nlp.borrowing_detector import BorrowingDetector

        d = BorrowingDetector()
        for w in self.PROTECTED:
            with self.subTest(word=w):
                self.assertFalse(d.detect(w).blocks_inherited_reconstruction, w)

    @needs_index
    def test_blocked_trap_yields_no_root(self):
        """Engellenen kelimeye ata biçim ÜRETİLMEMELİ."""
        from engine.nlp.comparative_reconstruction import ComparativeReconstructor

        entries = [
            {"lang_code": "az", "word": "divar"},
            {"lang_code": "tk", "word": "diwar"},
            {"lang_code": "uz", "word": "devor"},
        ]
        res = ComparativeReconstructor().reconstruct("duvar", entries)
        self.assertFalse(res.get("is_reconstructible"))
        self.assertEqual(res.get("reconstructed_root"), "")

    @needs_index
    def test_infinitive_evidence_rescues_homonym(self):
        """`gül` indekste yalnız alıntı görünür; miras kanıtı `gülmek`tedir."""
        from engine.nlp.borrowing_detector import _index_attests_loan

        self.assertFalse(_index_attests_loan("gül"))
        self.assertTrue(_index_attests_loan("duvar"))

    @needs_index
    def test_native_material_rows_do_not_block(self):
        """9c: Türkçe malzemeli türetme ve Türk dili verici engellemez."""
        from engine.nlp.borrowing_detector import _index_attests_loan

        for word in ("örgüt", "kurmay", "bakşı"):
            with self.subTest(word=word):
                self.assertFalse(_index_attests_loan(word), word)

    def test_native_material_loan_row(self):
        from engine.nlp.borrowing_detector import _native_material_loan_row

        self.assertTrue(_native_material_loan_row({"donor_lang": "oui", "etymology": ""}))
        self.assertTrue(_native_material_loan_row(
            {"donor_lang": "fr", "etymology": "Calque of French sous-marin"}))
        self.assertFalse(_native_material_loan_row(
            {"donor_lang": "ar", "etymology": "Borrowed from Arabic"}))

    def test_cldf_forms_do_not_collide_with_turkish_headwords(self):
        """Tam eşleşme şartı — normalleştirme çarpışması regresyonu.

        `index.lookup` karşılaştırma biçimi üzerinden arar ve CLDF
        işaretlerini normalleştirir (`š`->`ş`, `ï`->`ı`, uzunluk `:` düşer).
        Ölçüm hattı proto/tanık biçimleri beslediği için alakasız Türkçe
        maddelere çarpıyordu:

            keš     -> 'keş'   (fa "drug addict")
            kïrba:  -> 'kırba' (ar "waterskin")

        Bu iki çarpışma altın dev kümesinde 2 maddeyi çekimser bırakıp
        NED'i 0.302 -> 0.3261 bozuyordu.
        """
        from engine.nlp.borrowing_detector import _index_attests_loan

        for form in ("keš", "kïrba:"):
            with self.subTest(form=form):
                self.assertFalse(_index_attests_loan(form), form)


class TestWitnessAttestationDiagnostic(unittest.TestCase):
    """Tanık tanıklığı RAPORLANIR, soymayı engellemez.

    Sütun uyumu tanıkların birbiriyle uyuşmasını ölçer, gerçek olup
    olmadıklarını değil: uydurma bir kökün uydurulmuş tanıkları da
    kusursuz uyumludur.

    ⚠️ Sert kapı (en az bir tanık indekste olsun) ölçülerek REDDEDİLDİ —
    sahtelerin 7/8'ini elerken GERÇEK altın maddelerin %12'sini de elerdi
    (73/83 maddede tanık indekste). Gerekçe `comparative_reconstruction`
    içindeki notta.
    """

    @needs_index
    def test_fabricated_witnesses_report_zero(self):
        from engine.nlp.comparative_reconstruction import ComparativeReconstructor

        entries = [
            {"lang_code": "kk", "word": "kalgır"},
            {"lang_code": "tt", "word": "qalğır"},
            {"lang_code": "ky", "word": "kalgır"},
        ]
        res = ComparativeReconstructor().reconstruct("kalgır", entries)
        self.assertEqual(res.get("attested_witness_count"), 0)
        self.assertIn("bulunamadı", res.get("attestation_note", ""))
        # Engellemez: kök yine üretilir, yalnız dayanaksızlığı görünür olur.
        self.assertTrue(res.get("reconstructed_root"))

    @needs_index
    def test_real_witnesses_report_positive(self):
        from engine.nlp.comparative_reconstruction import ComparativeReconstructor

        entries = [
            {"lang_code": "az", "word": "göz"},
            {"lang_code": "tt", "word": "küz"},
            {"lang_code": "ky", "word": "köz"},
        ]
        res = ComparativeReconstructor().reconstruct("göz", entries)
        count = res.get("attested_witness_count")
        self.assertIsNotNone(count)
        self.assertGreater(count, 0)

    def test_missing_index_is_none_not_zero(self):
        """"Sıfır doğrulandı" ile "ölçemedim" karıştırılmamalı."""
        from unittest import mock

        from engine.nlp.comparative_reconstruction import ComparativeReconstructor

        with mock.patch(
            "engine.db.lexicon_index.LexiconIndex.exists",
            new_callable=mock.PropertyMock,
            return_value=False,
        ):
            self.assertIsNone(
                ComparativeReconstructor._attested_witness_count(["göz", "köz"])
            )


class TestComparativeReconstruction(unittest.TestCase):
    def test_produces_correct_proto_forms(self):
        """Karşılaştırmalı yöntem bilinen ata biçimleri üretmelidir."""
        r = ComparativeReconstructor()
        cases = {
            "göz": ("*köŕ", [("tr", "göz"), ("kk", "көз"), ("cv", "куҫ"), ("tt", "күз")]),
            "deniz": ("*teŋiŕ", [("tr", "deniz"), ("kk", "теңіз"), ("cv", "тинӗс"), ("tk", "deňiz")]),
            "yol": ("*jol", [("tr", "yol"), ("kk", "жол"), ("cv", "ҫул"), ("tt", "юл")]),
        }
        for word, (expected, pairs) in cases.items():
            with self.subTest(word=word):
                out = r.reconstruct(word, entries(pairs))
                self.assertEqual(out["reconstructed_root"], expected)

    def test_single_witness_is_not_presented_as_evidence(self):
        """Tek tanıkla KARŞILAŞTIRMALI ata biçim üretilmemelidir.

        Motor bir aday döndürebilir (ölçümde cevapsızlık mümkün olan en kötü
        NED'i alır ve çekimserliği ödüllendirmek yanlıştır) ama bunu
        **karşılaştırmalı yöntemin sonucu gibi sunamaz**: `method` alanı
        `anchor_fallback` demeli ve `evidence_available` false olmalı.
        """
        r = ComparativeReconstructor()
        out = r.reconstruct("göz", entries([("tr", "göz")]))
        self.assertFalse(out["evidence_available"])
        self.assertEqual(out.get("method"), "anchor_fallback")
        self.assertEqual(out.get("confidence"), 0.0)

    def test_confidence_grows_with_evidence(self):
        r = ComparativeReconstructor()
        few = r.reconstruct("göz", entries([("tr", "göz"), ("kk", "көз")]))
        many = r.reconstruct("göz", GOZ)
        self.assertGreater(many["confidence"], few["confidence"])

    def test_branch_diversity_is_reported(self):
        out = ComparativeReconstructor().reconstruct("göz", GOZ)
        self.assertGreaterEqual(out["branch_count"], 3)
        self.assertIn("oghur", out["branches"])

    def test_empty_word(self):
        out = ComparativeReconstructor().reconstruct("", GOZ)
        self.assertFalse(out["evidence_available"])

    def test_facade_rejects_non_turkic_initial(self):
        out = ProtoTurkicReconstructor().reconstruct_proto_form("fistan", GOZ)
        self.assertFalse(out["is_reconstructible"])


class TestCognateAlignment(unittest.TestCase):
    def test_spreading_uses_real_language_count(self):
        """Yayılım, koda gömülü sabit yerine gerçek dil sayısına bölünmelidir.

        Sayı sabitlenmez: dil haritası büyüdükçe (Faz 3'te ünlü uzunluğu
        tanıkları eklendi) bu testin de büyümesi gerekir, yoksa harita
        genişletildiğinde yanlış yere kırmızı yanar.
        """
        from engine.fetchers.base import (
            RECONSTRUCTED_CORPORA,
            SMALL_HISTORICAL_CORPORA,
            TURKIC_LANGUAGES_MAP,
        )

        res = CognateAlignmentEngine().evaluate_cognate_distribution("göz", GOZ)
        # ⚠️ Payda ATTESTE dillerin sayısıdır. Geri kurulmuş külliyatlar
        # (``wot``) tanık olarak sayılır ama hiçbir getirici onlardan kayıt
        # üretemez; paydaya katılsalardı her yayılım oranı sessizce düşerdi.
        # Küçük tarihî külliyatlar (Eski Uygurca, Eski Anadolu Türkçesi) da
        # aynı gerekçeyle paydaya girmez.
        self.assertEqual(
            res["total_language_count"],
            len(set(TURKIC_LANGUAGES_MAP) - RECONSTRUCTED_CORPORA - SMALL_HISTORICAL_CORPORA),
        )
        self.assertEqual(res["present_dialects_count"], 8)

    def test_pseudo_codes_excluded(self):
        rich = GOZ + [{"lang_code": "donor", "word": "X"}, {"lang_code": "ai", "word": "Y"}]
        res = CognateAlignmentEngine().evaluate_cognate_distribution("göz", rich)
        self.assertEqual(res["present_dialects_count"], 8)

    def test_no_fabricated_baseline_score(self):
        """alignment_score sabit 85.0 tabanından GELMEMELİDİR."""
        res = CognateAlignmentEngine().evaluate_cognate_distribution("göz", GOZ)
        self.assertIsNotNone(res["alignment_score"])
        self.assertLessEqual(res["alignment_score"], 1.0)

    def test_empty_entries_have_no_evidence(self):
        res = CognateAlignmentEngine().evaluate_cognate_distribution("zzz", [])
        self.assertFalse(res["evidence_available"])
        self.assertIsNone(res["alignment_score"])


class TestCognateClustering(unittest.TestCase):
    def test_separates_distinct_roots(self):
        """*köŕ ailesi ile Sibirya *karak kökü AYRI kümelere düşmelidir.

        ⚠️ Eşik 8 değil 7: düzenleme benzerliğine geçince Çuvaşça ``куҫ``
        (``kus``) bu örnekte kümeden düşüyor (``küz`` ile benzerlik 0,33).
        Tek vaka; savelyevturkic'te ölçülen Çuvaşça duyarlılığı ise
        ARTIYOR: train 0,511 -> 0,663, dev 0,594 -> 0,692.
        """
        mixed = GOZ + entries([("sah", "харах"), ("tyv", "карак")])
        res = CognateClusterEngine().cluster(mixed)
        self.assertTrue(res["evidence_available"])
        self.assertGreaterEqual(res["cluster_count"], 2)
        self.assertGreaterEqual(res["largest_cluster_size"], 7)

    def test_core_cognate_set_flag(self):
        res = CognateClusterEngine().cluster(GOZ)
        self.assertTrue(res["clusters"][0]["is_core_cognate_set"])

    def test_needs_two_forms(self):
        res = CognateClusterEngine().cluster(entries([("tr", "göz")]))
        self.assertFalse(res["evidence_available"])


class TestLoanwordClassifier(unittest.TestCase):
    def test_known_loanwords_are_not_native(self):
        """Regresyon: kitap/kalem/müdür/televizyon %100 'Öz Türkçe' çıkıyordu."""
        c = LoanwordClassifier()
        for w in ("kitap", "kalem", "müdür", "televizyon", "fistan", "coğrafya"):
            with self.subTest(word=w):
                res = c.classify(w)
                self.assertLess(res["probabilities"]["p_native_turkic"], 0.55, w)

    def test_native_words_stay_native(self):
        c = LoanwordClassifier()
        for w in ("göz", "deniz", "su", "ayak", "okul", "yastık"):
            with self.subTest(word=w):
                self.assertGreaterEqual(c.classify(w)["probabilities"]["p_native_turkic"], 0.55, w)

    @needs_index
    def test_attestation_overrides_phonotactics(self):
        """Tamamen Türkçeleşmiş alıntıları fonotaktik göremez, sözlük söyler.

        `makas`, `ahır`, `tasa`, `maruz` fonotaktik olarak kusursuz Türkçedir;
        ihlal aramak onları bulamaz ve hepsi "Asli Öz Türkçe" çıkıyordu.
        Ölçüldü (387 gerçek alıntı, özel adlar elendi): sınıflandırıcı
        %32,0'sine "öz Türkçe" diyordu, aile isabeti %24,8 idi. Sözlük
        tanıklığı fonotaktik tahmini ezince %98'e çıktı.
        """
        c = LoanwordClassifier()
        for w in ("makas", "ahır", "tasa", "maruz", "elektrik"):
            with self.subTest(word=w):
                res = c.classify(w)
                self.assertLess(res["probabilities"]["p_native_turkic"], 0.55, w)
                self.assertNotIn("Öz Türkçe", res["classification"], w)

    def test_reversed_loan_direction_stays_native(self):
        """Sözlükte YÖNÜ TERS kaydedilmiş maddeler alıntı sayılmamalı.

        Macarca ve Sırp-Hırvatça bu sözcükleri Türkçeden almıştır; Wiktionary
        ters yazmış. Önce iki ilkeli ayırt edici denendi ve ölçülerek
        çürütüldü (Türki yayılım; verici dilin Balkan olması) — ayrıntı
        `KNOWN_REVERSED_LOAN_DIRECTION` tanımında.
        """
        c = LoanwordClassifier()
        for w in ("yaz", "öküz", "diz", "gece", "balta", "yastık", "okul"):
            with self.subTest(word=w):
                self.assertGreaterEqual(
                    c.classify(w)["probabilities"]["p_native_turkic"], 0.55, w
                )

    @needs_index
    def test_genuine_balkan_loans_still_detected(self):
        """Ters-yön listesi gerçek Balkan alıntılarını kırmamalı."""
        c = LoanwordClassifier()
        for w in ("haydut", "soba", "tabur", "varoş", "çete", "voyvoda"):
            with self.subTest(word=w):
                self.assertLess(
                    c.classify(w)["probabilities"]["p_native_turkic"], 0.55, w
                )

    def test_probabilities_sum_to_one(self):
        for w in ("kitap", "göz", "tren"):
            p = LoanwordClassifier().classify(w)["probabilities"]
            self.assertAlmostEqual(sum(p.values()), 1.0, places=2)

    def test_compound_words_exempt_from_vowel_harmony(self):
        """Bileşikler ünlü uyumunu doğal olarak bozar; alıntı sayılmamalıdır."""
        res = LoanwordClassifier().classify("bilgisayar")
        self.assertGreaterEqual(res["probabilities"]["p_native_turkic"], 0.55)

    def test_cross_dialect_spread_strengthens_native(self):
        c = LoanwordClassifier()
        wide = c.classify("kitap", spreading_ratio=0.8)["probabilities"]["p_native_turkic"]
        narrow = c.classify("kitap", spreading_ratio=0.05)["probabilities"]["p_native_turkic"]
        self.assertGreater(wide, narrow)

    def test_empty_word(self):
        self.assertFalse(LoanwordClassifier().classify("")["evidence_available"])


class TestLoanwordDetector(unittest.TestCase):
    def test_four_layer_pipeline(self):
        res = LoanwordDetector().detect("göz", GOZ)
        self.assertEqual(res["verdict"], "native")
        for layer in ("layer1_phonotactics", "layer2_cross_dialect",
                      "layer3_probabilities", "layer4_donor_neighbours"):
            self.assertIn(layer, res["layers"])

    def test_donor_lexicon_match_overrides_rules(self):
        res = LoanwordDetector().detect("kitap")
        self.assertEqual(res["verdict"], "loanword")
        self.assertIn("Arap", res["classification"])

    def test_method_is_declared_rule_based(self):
        """Sezgisel yöntem çıktıda AÇIKÇA bildirilmelidir."""
        res = LoanwordDetector().detect("göz", GOZ)
        self.assertEqual(res["method"], "rule_based")
        self.assertIn("ML", res["method_note"])

    def test_empty_word(self):
        self.assertFalse(LoanwordDetector().detect("")["evidence_available"])


class TestBorrowingLineageFilter(unittest.TestCase):
    """Türkçenin ATA katmanı verici dil sayılmamalıdır.

    Sözlük `donor_lang` alanını ata biçimi kaydetmek için de kullanıyor:
    `bardak` satırı ('bardak', 'alıntı', 'trk-eog', 'برت') ve trk-eog =
    Eski Oğuzca, yani Türkçenin atası. Süzülmediğinde öz Türkçe kelime
    1.0 güçle "alıntı" ilan ediliyor ve DOĞRU miras hipotezi bu gerekçeyle
    reddediliyordu (ölçüldü: `bardak` -> "ALINTI — trk-eog" seçiliyordu).
    """

    def test_ancestor_code_is_not_loan_evidence(self):
        from engine.nlp.borrowing_detector import BorrowingDetector

        d = BorrowingDetector()
        for w in ("bardak", "bilge", "betik", "kamu", "sav"):
            verdict = d.detect(w)
            fired = {s.name for s in verdict.signals if s.fired}
            self.assertNotIn("zincir_kanıtı", fired, w)
            self.assertFalse(verdict.donor_language, w)

    @needs_index
    def test_real_loans_still_detected(self):
        """Süzgeç gerçek alıntıyı öldürmemeli.

        `ota` (Osmanlıca) bilerek süzülmüyor: hem doğrudan atadır hem de
        Arapça/Farsça alıntıların geçiş katmanıdır. `reis` ve `ziyaret`
        gerçek Arapça alıntılardır ve Osmanlıca üzerinden gelmişlerdir.
        """
        from engine.nlp.borrowing_detector import BorrowingDetector

        d = BorrowingDetector()
        for w, donor in (("kitap", "ar"), ("reis", "ota"), ("ziyaret", "ota")):
            verdict = d.detect(w)
            fired = {s.name for s in verdict.signals if s.fired}
            self.assertIn("zincir_kanıtı", fired, w)
            self.assertEqual(verdict.donor_language, donor, w)


class TestDonorLexicon(unittest.TestCase):
    def test_levenshtein(self):
        self.assertEqual(levenshtein("abc", "abc"), 0)
        self.assertEqual(levenshtein("", "abc"), 3)
        self.assertEqual(levenshtein("abc", ""), 3)
        self.assertEqual(levenshtein("kitten", "sitting"), 3)

    def test_ipa_distance_within_threshold(self):
        self.assertLessEqual(ipa_distance("kitap", "kitāb"), 2)
        self.assertGreater(ipa_distance("deniz", "kitap"), 2)

    def test_nearest_neighbours_finds_seed_donors(self):
        d = DonorLexicon()
        self.assertTrue(d.nearest_neighbours("kitap"))
        self.assertFalse(d.nearest_neighbours("zzzqxwv"))

    def test_empty_word(self):
        self.assertEqual(DonorLexicon().nearest_neighbours(""), [])


class TestDonorSearch(unittest.TestCase):
    def test_no_match_returns_none_not_placeholder(self):
        """Regresyon: eşleşme yokken donor_language'a '10 Komşu Dilde Canlı
        Taranıyor…' gibi SAHTE bir durum metni yazılıyordu."""
        res = DonorSearchEngine().search_donor_neighbors("zzzqx")
        self.assertFalse(res["found_match"])
        self.assertIsNone(res["donor_language"])

    def test_known_loanword_matches(self):
        res = DonorSearchEngine().search_donor_neighbors("kitap")
        self.assertTrue(res["found_match"])
        self.assertTrue(res["donor_language"])

    def test_empty(self):
        self.assertFalse(DonorSearchEngine().search_donor_neighbors("")["found_match"])


class TestNeologismDetector(unittest.TestCase):
    def test_compound_neologism(self):
        res = NeologismDetector().detect("bilgisayar")
        self.assertTrue(res["is_neologism"])
        self.assertEqual(res["components"], ["bilgi", "sayar"])

    def test_arabic_loans_not_flagged_as_neologisms(self):
        """Regresyon: -im/-ım eki 'hüküm', 'zulüm' gibi Arapça alıntıları
        'Cumhuriyet neolojizmi' sayıyordu."""
        d = NeologismDetector()
        for w in ("hüküm", "zulüm"):
            self.assertIsNone(d.detect(w), w)
        for w in ("resim", "takım"):
            res = d.detect(w)
            self.assertFalse(res["is_neologism"], w)

    def test_strong_suffixes(self):
        d = NeologismDetector()
        for w in ("toplumsal", "kurultay"):
            self.assertTrue(d.detect(w)["is_neologism"], w)

    def test_short_and_empty(self):
        self.assertIsNone(NeologismDetector().detect(""))
        self.assertIsNone(NeologismDetector().detect("ab"))


class TestHistoricalMorphology(unittest.TestCase):
    def test_builds_derivation_tree(self):
        t = HistoricalMorphologyAnalyzer().build_tree("güzellik")
        self.assertEqual(t["root"], "güzel")
        self.assertEqual(t["depth"], 1)

    def test_old_turkic_suffixes(self):
        """Plan §2.5'teki tarihsel ekler (+gU, -Ik, -gA, -bA) tanınmalıdır."""
        a = HistoricalMorphologyAnalyzer()
        self.assertEqual(a.build_tree("bitig")["root"], "biti")
        self.assertEqual(a.build_tree("susuz")["root"], "su")
        # ⚠️ Bu satır eskiden `depth >= 2` istiyordu ve analizörün ÇÖP bir ara
        # biçim üretmesini şart koşuyordu: zincir `toplumsal -> toplum (+sAl)
        # -> topl (-Im)` idi ve `topl` diye bir kelime yok (indekste bulunmaz,
        # Türkçe kök değildir). Soymaya tanıklık kapısı eklenince analiz
        # `toplum` ("society") üzerinde duruyor — doğru davranış budur.
        # Ölçüldü: kapı `menengiç -> mene`, `avsunlu -> avs`, `köremez -> köre`,
        # `garametli -> gara` gibi sahte kökleri de eliyor.
        toplumsal = a.build_tree("toplumsal")
        self.assertEqual(toplumsal["root"], "toplum")
        self.assertEqual(toplumsal["depth"], 1)

    @needs_index
    def test_derivation_formula_opens_strip(self):
        """Kaynağın kendi türetme formülü soymayı açmalı; formül yoksa açmamalı.

        `pervasız` uzun süre soyulamıyordu: `perva` indekste sözlükbirim olarak
        tanıklanmıyor. Çözüm için denenen TDK yedeği `bardak -> barda`yı
        bozduğu için geri alınmıştı (bkz. 1815aea). Wiktionary'nin KAYITLI
        türetme formülü ikisini ayırıyor:

            pervasız -> "equivalent to perva + -sız"   => formül açar
            bardak   -> formül yok                     => tanıklık kapısı reddeder

        Gloss örtüşmesi ve CLICS bu çifti ayıramamıştı; formül ayırıyor.
        """
        a = HistoricalMorphologyAnalyzer()
        self.assertEqual(a.build_tree("pervasız")["root"], "perva")
        self.assertEqual(a.build_tree("bardak")["depth"], 0)

    def test_formula_rule_is_additive_only(self):
        """Formül BAŞKA kök söylüyorsa soyma reddedilmemeli.

        Formül nihai tabanı yazar ("adalet + -siz + -lik"), soyucu tek katman
        soyar (`adaletsiz`). "Eşleşmiyorsa reddet" dalı 600 sözlükbirimde
        ölçüldü: 3 doğru kabul getirirken 5 DOĞRU soymayı kapatıyordu.
        Kural bu yüzden yalnız ekleyicidir.
        """
        a = HistoricalMorphologyAnalyzer()
        self.assertEqual(a.build_tree("adaletsizlik")["root"], "adaletsiz")
        self.assertEqual(a.build_tree("eczacı")["root"], "ecza")

    @needs_index
    def test_deverbal_gate_blocks_nonverb_stems(self):
        """Fiilden türeten ek FİİL kök ister.

        ⚠️ Bu kapı bir kez konup GERİ ALINMIŞTI (a670fd4): `bitig -> biti`yi
        kırıyordu, çünkü `biti`nin Türkçe mastarı yok ve otk kayıtları o
        sırada runik çeviriyazıyla `bıtı` diye indeksleniyordu. Sözlük
        indeksi kaynaktaki romanizasyonu kullanmaya başlayınca `𐰋𐰃𐱅𐰃`
        artık `biti` olarak bulunuyor ve kapı yeniden mümkün oldu.
        """
        a = HistoricalMorphologyAnalyzer()
        # Kapının hedefi: fiil olmayan köke soyma engellenir.
        self.assertEqual(a.build_tree("tabut")["depth"], 0)
        self.assertEqual(a.build_tree("kavut")["depth"], 0)
        # Kapıyı bir kez öldüren vaka — tarihî fiil kaydı sayesinde korunur.
        self.assertEqual(a.build_tree("bitig")["root"], "biti")
        # Mastarı olan gerçek fiil kökleri etkilenmez.
        for word, root in (("umut", "um"), ("yoğurt", "yoğur"), ("kanıt", "kan")):
            with self.subTest(word=word):
                self.assertEqual(a.build_tree(word)["root"], root)

    def test_inflection_lookalike_suffixes_are_exempt(self):
        """`-Im`/`-In` tabloda "fiilden ad" ama Türkçede İYELİK/İLGİ ekidir.

        Kapı bunlara da uygulanınca 600 kelimelik örneklemde 27 yeni ret
        çıkıyordu ve 20'si tam bu iki ekten geliyordu — hepsi çekim.
        Muafiyetle yeni ret 27 -> 7'ye indi.
        """
        a = HistoricalMorphologyAnalyzer()
        for word, root in (
            ("düşmanım", "düşman"),
            ("kontum", "kont"),
            ("altıgenin", "altıgen"),
        ):
            with self.subTest(word=word):
                self.assertEqual(a.build_tree(word)["root"], root)

    def test_bare_root_has_zero_depth(self):
        self.assertEqual(HistoricalMorphologyAnalyzer().build_tree("at")["depth"], 0)

    def test_empty(self):
        self.assertFalse(HistoricalMorphologyAnalyzer().build_tree("")["evidence_available"])


class TestDerivationNetwork(unittest.TestCase):
    def test_generates_harmonic_candidates(self):
        """Ek seçimi ünlü uyumuna uymalıdır."""
        cands = {c["word"] for c in DerivationNetworkBuilder().candidate_derivations("göz")}
        self.assertIn("gözlük", cands)
        self.assertIn("gözsüz", cands)
        self.assertNotIn("gözluk", cands)

    def test_unverified_mode_needs_no_network(self):
        res = DerivationNetworkBuilder().build("göz", verify=False)
        self.assertTrue(res["evidence_available"])
        self.assertFalse(res["verified"])

    def test_short_root_rejected(self):
        self.assertFalse(DerivationNetworkBuilder().build("a", verify=False)["evidence_available"])


class TestSoundLawInduction(unittest.TestCase):
    def test_single_pair_has_no_confidence(self):
        """Tek çiftten güven skoru ÇIKARILAMAZ (eskiden sabit 0.95 veriliyordu)."""
        res = SoundLawInductionEngine().induce_sound_law("köz", "göz")
        self.assertIsNone(res["rule_confidence"])
        self.assertFalse(res["evidence_available"])

    def test_multi_pair_induction(self):
        pairs = [("köz", "göz"), ("kök", "gök"), ("kel", "gel"), ("kör", "gör"), ("küç", "güç")]
        res = SoundLawInductionEngine().induce_from_pairs(pairs)
        self.assertTrue(res["evidence_available"])
        rule = res["induced_rules"][0]
        self.assertEqual(rule["rule_id"], "INITIAL_K_G")
        self.assertEqual(rule["observed_in_pairs"], 5)
        self.assertGreater(rule["confidence"], 0.8)

    def test_needs_two_pairs(self):
        self.assertFalse(SoundLawInductionEngine().induce_from_pairs([("a", "b")])["evidence_available"])

    def test_identical_words_induce_nothing(self):
        res = SoundLawInductionEngine().induce_sound_law("göz", "göz")
        self.assertFalse(res["has_induced_rule"])


class TestSemanticEngine(unittest.TestCase):
    def test_deterministic_across_calls(self):
        e = DiachronicSemanticEngine()
        v1, _ = e.vectorizer.vectorise("deniz, büyük su")
        v2, _ = e.vectorizer.vectorise("deniz, büyük su")
        self.assertEqual(v1, v2)

    def test_missing_data_yields_no_evidence(self):
        res = DiachronicSemanticEngine().evaluate_diachronic_trajectory("", "")
        self.assertFalse(res["evidence_available"])
        self.assertIsNone(res["is_plausible"])

    def test_no_fabricated_acceleration_field(self):
        """Sahte 'ivme' alanı (mesafe × 1.10) kaldırılmalıydı."""
        res = DiachronicSemanticEngine().evaluate_diachronic_trajectory("göz", "göz")
        self.assertNotIn("semantic_acceleration", res)


class TestAttestationVerifier(unittest.TestCase):
    def test_no_fabricated_attestation(self):
        """Regresyon: kanıt yokken '13.-19. yy Osmanlı/Çağatay' cümlesi
        uyduruluyor, sonra A-HVP bunu 1850 yılı olarak okuyordu."""
        res = HistoricalAttestationVerifier().verify_attestation("zzzqx")
        self.assertFalse(res["verified"])
        self.assertIsNone(res["first_attestation_record"])
        self.assertIsNone(res["first_attestation_year"])

    def test_detects_known_sources(self):
        v = HistoricalAttestationVerifier()
        res = v.verify_attestation("deniz", [{"lang_name": "Divanü Lugati't-Türk (1074)"}])
        self.assertTrue(res["verified"])
        self.assertEqual(res["first_attestation_year"], 1074)

    def test_prefers_earliest(self):
        v = HistoricalAttestationVerifier()
        res = v.verify_attestation("su", [
            {"lang_name": "Kamus-ı Türkî"},
            {"lang_name": "Orhun Yazıtları"},
        ])
        self.assertEqual(res["first_attestation_year"], 732)

    def test_fetcher_attestation_is_used(self):
        v = HistoricalAttestationVerifier()
        res = v.verify_attestation("deniz", [], [{"first_attestation": {"year": 1245, "source": "X yazması"}}])
        self.assertEqual(res["first_attestation_year"], 1245)

    def test_known_work_gets_single_year(self):
        """Aynı eserin tek yılı var: EtimolojiTürkçe DLT'ye 1070, Starling MK
        1072, yerel kayıtlar 1074 diyordu (``utils.attestation_dates``)."""
        v = HistoricalAttestationVerifier()
        from_fetcher = v.verify_attestation(
            "deniz", [], [{"first_attestation": {"year": 1070, "source": "Divan-i Lugat-it Türk (1070)"}}])
        from_corpus = v.verify_attestation("deniz", [{"lang_name": "Divanü Lugati't-Türk (1074)"}])
        self.assertEqual(from_fetcher["first_attestation_year"], from_corpus["first_attestation_year"])


class TestMorphemeSegmenter(unittest.TestCase):
    def test_segments_suffixes(self):
        res = UnsupervisedMorphemeSegmenter().segment_morphemes("yağmurluk")
        self.assertTrue(res["is_segmented"])

    def test_short_word_not_over_segmented(self):
        res = UnsupervisedMorphemeSegmenter().segment_morphemes("at")
        self.assertGreaterEqual(len(res["stem"]), 2)


class TestComparableOriginForm(unittest.TestCase):
    """`origin_form` bir biçim değil, bileşik bir açıklamadır.

    Hizalamaya ham hâliyle verilince `kalem` 1. aşamada 0.244 alıyor ve
    rozet 🔴 çıkıyordu; oysa 2.-4. aşamaların üçü de geçiyordu.
    Ölçüm (10 tohum kaydı): 4 iyileşme, 0 gerileme.
    """

    def test_parenthetical_transliteration_wins(self):
        from engine.nlp.hypothesis_validation_protocol import _comparable_origin_form

        self.assertEqual(
            _comparable_origin_form("قلم (qalam) / Grekçe κάλαμος (kálamos)"), "qalam"
        )

    def test_parenthesis_before_alternative(self):
        """Sıra ters olursa `herkil` kırılır: `/` ile bölmek Ermeni parçayı alır."""
        from engine.nlp.hypothesis_validation_protocol import _comparable_origin_form

        self.assertEqual(
            _comparable_origin_form("յարգել / յարգիլ (harkil / hargel)"), "harkil"
        )

    def test_macrons_are_kept(self):
        """Unicode ADIYLA test şart: regex sınıfı `ā`/`ū`yu eleyip kanıtı yok ediyordu."""
        from engine.nlp.hypothesis_validation_protocol import _comparable_origin_form

        self.assertEqual(_comparable_origin_form("كتاب (kitāb)"), "kitāb")
        self.assertEqual(_comparable_origin_form("روزگار (rūzgār)"), "rūzgār")

    def test_plain_latin_untouched(self):
        from engine.nlp.hypothesis_validation_protocol import _comparable_origin_form

        self.assertEqual(_comparable_origin_form("télévision"), "télévision")

    def test_unextractable_falls_back_to_raw(self):
        """Kanıt asla kaybedilmez."""
        from engine.nlp.hypothesis_validation_protocol import _comparable_origin_form

        self.assertEqual(_comparable_origin_form("كتاب"), "كتاب")
        self.assertEqual(_comparable_origin_form(""), "")


class TestBestWitnessSemanticDrift(unittest.TestCase):
    """3. aşama, tanıklanmış anlamların EN YAKININA karşı ölçülmeli.

    Tek bir önceden seçilmiş tanığa bakmak `bardak`ı yanlış reddettiriyordu:
    biçim sıralaması tam eşleşen `bardak` tanığını seçiyor, glossu dar
    anlamlı "testicik"; oysa `bart` tanığı "su içilen kap" diyor.
    Ölçüldü: bardak 0.7863 -> 0.2202, göz 0.2843 -> 0.0311,
    el 0.4066 -> 0.1921; deniz/su/ayak kanıt kazandı; gerileme yok.
    """

    MODERN = "Su vb. şeyleri içmek için kullanılan, camdan yapılan kap"

    def _distance(self, res):
        return (res.get("trajectory_details") or {}).get("total_shift_distance")

    def test_candidates_never_worsen_the_distance(self):
        """Değişmez kural: aday havuzu eklemek mesafeyi ARTIRAMAZ.

        Bu iddia gömme modelinden bağımsızdır — min bir üst kümede alınıyor.
        """
        from engine.nlp.hypothesis_validation_protocol import SemanticDriftEvaluator

        ev = SemanticDriftEvaluator()
        tek = self._distance(ev.verify("testicik", self.MODERN))
        cok = self._distance(ev.verify("testicik", self.MODERN, ["su içilen kap"]))
        self.assertIsNotNone(tek)
        self.assertIsNotNone(cok)
        self.assertLessEqual(cok, tek)

    def test_no_candidates_keeps_old_behaviour(self):
        from engine.nlp.hypothesis_validation_protocol import SemanticDriftEvaluator

        ev = SemanticDriftEvaluator()
        a = self._distance(ev.verify("su içilen kap", self.MODERN))
        b = self._distance(ev.verify("su içilen kap", self.MODERN, None))
        self.assertEqual(a, b)

    def test_empty_candidates_are_skipped(self):
        from engine.nlp.hypothesis_validation_protocol import SemanticDriftEvaluator

        ev = SemanticDriftEvaluator()
        res = ev.verify("su içilen kap", self.MODERN, ["", "   ", None])  # type: ignore[list-item]
        self.assertIsNotNone(self._distance(res))


class TestHistoricalGlossReachesHypothesis(unittest.TestCase):
    """Miras dalı, tarihî anlama MODERN anlamın kopyasını yazmamalı.

    Kusur: `_select_hypothesis` karşılaştırmalı yöntem dalında
    `historical_meaning` alanına `modern_meaning`'i yazıyordu. A-HVP
    3. aşaması çiftini oradan aldığı için anlamı kendisiyle karşılaştırıyor,
    mesafe tanımı gereği 0 çıkıyor ve aşama BEDAVA ✅ veriyordu.

    Ölçüldü (`--json`, iki düğüm ayrı ayrı):
        göz    stage3: 'göz, görme organı' ~ 'göz, görme organı'  özdeş
        bardak stage3: TDK tanımı          ~ TDK tanımı           özdeş
    Aynı koşuda `search_engine` yolu sağlıklıydı (göz 0.4981,
    bardak 0.2202) — yani veri MEVCUTTU, bu dala taşınmıyordu.
    """

    RECONSTRUCTION = {
        "evidence_available": True,
        "reconstructed_root": "*köŕ",
        "reconstruction_notes": "3 dil tanığı",
        "witness_count": 3,
    }

    def test_historical_gloss_is_used_not_modern_copy(self):
        from engine.nlp.iterative_hypothesis_engine import IterativeHypothesisEngine

        hypo = IterativeHypothesisEngine._select_hypothesis(
            "göz",
            {"meaning": "göz, görme organı"},
            None,
            None,
            self.RECONSTRUCTION,
            "göz, görmek",
        )
        self.assertEqual(hypo["historical_meaning"], "göz, görmek")
        self.assertNotEqual(hypo["historical_meaning"], hypo["modern_meaning"])

    def test_missing_witness_leaves_gloss_empty(self):
        """Tanık yoksa UYDURULMAZ; boş kalır ve aşama 'ölçülemedi' der."""
        from engine.nlp.iterative_hypothesis_engine import IterativeHypothesisEngine

        hypo = IterativeHypothesisEngine._select_hypothesis(
            "göz", {"meaning": "göz, görme organı"}, None, None, self.RECONSTRUCTION
        )
        self.assertEqual(hypo["historical_meaning"], "")

    def test_junk_glosses_are_skipped(self):
        """Kitap tarama artığı ve runik HARF adı tarihî anlam sayılmamalı.

        `archive_org.py:42` bir tam-metin arama isabetini `lang_code="otk"`
        diye işaretleyip anlam alanına KİTAP BAŞLIĞI yazıyor; indeksteki
        otk kayıtlarının 73/470'i (%15,5) ise alfabe maddesi.
        Ölçülen mesafeler (süzgeçten önce -> sonra):
            su  0.9523 -> 0.5345   'Kitap: 3 Bogatyr…'  -> 'su, akarsu'
            el  0.7098 -> 0.1921   'Kitap: The Land…'   -> 'el'
            baş 0.7996 -> 0.6501   runik harf adı       -> 'kafa, … reis'
        """
        from engine.nlp.iterative_hypothesis_engine import _historical_gloss

        self.assertEqual(
            _historical_gloss([
                {"lang_code": "otk", "meaning": "Kitap: 3 Bogatyr bikers (Yazar: X)"},
                {"lang_code": "otk", "meaning": "su, akarsu"},
            ]),
            "su, akarsu",
        )
        self.assertEqual(
            _historical_gloss([
                {"lang_code": "otk",
                 "meaning": "A letter of the Old Turkic runic script, representing /bɑʃ/"},
                {"lang_code": "otk", "meaning": "kafa, lider"},
            ]),
            "kafa, lider",
        )

    def test_citation_prefix_is_stripped_not_rejected(self):
        """Atıf öneki kırpılmalı; gloss önekten sonra gerçekten duruyor."""
        from engine.nlp.iterative_hypothesis_engine import _historical_gloss

        self.assertEqual(
            _historical_gloss([
                {"lang_code": "otk",
                 "meaning": "Divanü Lugati't-Türk (1074): göz, görme organı"}
            ]),
            "göz, görme organı",
        )

    def test_non_latin_parenthetical_and_quotes_are_stripped(self):
        """Parantez içi yabancı yazı ve tırnaklar anlam değildir.

        Ölçüldü: `deniz` tarihî glossu "teŋiz (تِںِزْ) 'deniz, ulu göl'."
        idi; Arap harfli biçim kodlanınca mesafe şişiyordu.
        A-HVP 3. aşama mesafesi 0.6725 -> 0.3241.

        ⚠️ LATİN parantezler KORUNUR: "(mecazi)" gibi açıklamalar anlamın
        parçasıdır.
        """
        from engine.nlp.iterative_hypothesis_engine import _historical_gloss

        temiz = _historical_gloss(
            [{"lang_code": "otk", "meaning": "teŋiz (تِںِزْ) 'deniz, ulu göl'."}]
        )
        self.assertNotIn("تِںِزْ", temiz)
        self.assertIn("deniz", temiz)

        korunan = _historical_gloss(
            [{"lang_code": "otk", "meaning": "kafa, (mecazi) lider, reis"}]
        )
        self.assertEqual(korunan, "kafa, (mecazi) lider, reis")

    def test_closest_form_witness_wins(self):
        """Tanık seçimi BİÇİME bakmalı; ilk kayıt başka kelime olabilir.

        Ölçüldü: `bilge` için 1. tanık `belgü` "işaret, alamet" — başka bir
        kelime (belgü -> belge). Doğru tanık (`bilge` "Âlim, hakim, bilgin.")
        hemen arkasındaydı ve A-HVP mesafesi 0.8147 çıkıyordu; en yakın biçim
        seçilince 0.2891.

        ⚠️ Bu bir EŞİK DEĞİL SIRALAMADIR. Biçim mesafesine eşik konamaz:
        `bilge~belge` oranı 0.20 ve bu meşru ses denkliklerinden DAHA İYİ
        (`göz~köz` 0.33, `deniz~teŋiz` 0.40, `el~elig` 0.50). Eşik yanlışı
        eleyemeden doğruları keserdi.
        """
        from engine.nlp.iterative_hypothesis_engine import _historical_gloss

        entries = [
            {"lang_code": "otk", "word": "belgü", "meaning": "işaret, alamet"},
            {"lang_code": "otk", "word": "bilge", "meaning": "Âlim, hakim, bilgin."},
            {"lang_code": "otk", "word": "belge", "meaning": "işaret, alamet"},
        ]
        self.assertEqual(_historical_gloss(entries, "bilge"), "Âlim, hakim, bilgin")

    def test_without_word_first_witness_is_kept(self):
        """`word` verilmezse eski davranış (ilk tanık) korunmalı."""
        from engine.nlp.iterative_hypothesis_engine import _historical_gloss

        entries = [
            {"lang_code": "otk", "word": "belgü", "meaning": "işaret, alamet"},
            {"lang_code": "otk", "word": "bilge", "meaning": "Âlim, hakim, bilgin."},
        ]
        self.assertEqual(_historical_gloss(entries), "işaret, alamet")

    def test_gloss_extractor_picks_first_real_witness(self):
        from engine.nlp.iterative_hypothesis_engine import _historical_gloss

        entries = [
            {"lang_code": "tr", "meaning": "modern tanım"},
            {"lang_code": "otk", "meaning": ""},
            {"lang_code": "otk", "meaning": "su içilen kap"},
            {"lang_code": "ota", "meaning": "başka"},
        ]
        self.assertEqual(_historical_gloss(entries), "su içilen kap")
        self.assertEqual(_historical_gloss([]), "")
        self.assertEqual(_historical_gloss(None), "")


class TestHypothesisEngine(unittest.TestCase):
    def test_no_hypothesis_without_evidence(self):
        res = IterativeHypothesisEngine().prove_etymological_hypothesis(
            "qqqwww", {"root": {"proto_turkic": "", "meaning": ""}}, [], []
        )
        self.assertFalse(res["hypothesis_available"])
        self.assertIsNone(res["proven_hypothesis"])

    def test_comparative_hypothesis_from_cognates(self):
        res = IterativeHypothesisEngine().prove_etymological_hypothesis(
            "göz", {"root": {"proto_turkic": "", "meaning": "göz"}}, GOZ, []
        )
        self.assertTrue(res["hypothesis_available"])
        self.assertEqual(res["proven_hypothesis"]["origin_form"], "*köŕ")

    def test_donor_match_takes_priority(self):
        res = IterativeHypothesisEngine().prove_etymological_hypothesis(
            "kitap", {"root": {"proto_turkic": "", "meaning": "kitap"}}, [], []
        )
        self.assertTrue(res["hypothesis_available"])
        self.assertEqual(res["proven_hypothesis"]["evidence_kind"], "donor_lexicon")


if __name__ == "__main__":
    unittest.main()
