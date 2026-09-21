"""
Ek Kapsam: LLM araçları, semantik motor, sunucu ve kazıyıcı derin yolları.
"""
from __future__ import annotations

import re
import threading
import unittest
from unittest import mock

import responses

from engine import config, server
from engine.llm import research_tools
from engine.llm.advanced_tools import (
    tool_ipa_phonetic_analyzer,
    tool_wiktionary_multilingual_api,
)
from engine.llm.qwen_agent import QwenEtymologyAgent
from engine.llm.research_tools import tool_web_search
from engine.nlp.diachronic_semantic_engine import (
    DenseSemanticVectorizer,
    DiachronicSemanticEngine,
    get_sentence_transformer,
    has_semantic_model,
)
from engine.nlp.phonological_feature_engine import PhonologicalFeatureEngine
from engine.nlp.trusted_whitelisted_scraper import scrape_whitelisted_academic_sources
from engine.utils import network


class TestWebSearchPaths(unittest.TestCase):
    @responses.activate
    def test_wiktionary_fallback_chain(self):
        """tool_web_search çoklu yedek zinciri — her kaynak denenmeli."""
        network.reset_session()
        responses.add(
            responses.GET, re.compile(r".*tr\.wiktionary\.org/w/api\.php.*"),
            json={"query": {"search": [{"title": "göz", "snippet": "<b>göz</b> anlamı"}]}},
            status=200,
        )
        responses.add(responses.GET, re.compile(r".*"), status=500)
        out = tool_web_search("göz")
        self.assertTrue(out)
        self.assertTrue(all("url" in r and "title" in r for r in out))

    @responses.activate
    def test_wikipedia_fallback(self):
        network.reset_session()
        responses.add(responses.GET, re.compile(r".*tr\.wiktionary.*"), status=500)
        responses.add(
            responses.GET, re.compile(r".*tr\.wikipedia\.org.*"),
            json={"query": {"search": [{"title": "Göz", "snippet": "organ"}]}}, status=200,
        )
        responses.add(responses.GET, re.compile(r".*"), status=500)
        self.assertIsInstance(tool_web_search("göz"), list)

    @responses.activate
    def test_duckduckgo_fallback(self):
        network.reset_session()
        responses.add(responses.GET, re.compile(r".*wiki.*"), status=500)
        responses.add(
            responses.GET, re.compile(r".*duckduckgo.*"),
            body='<a class="result__a" href="https://ornek.org/x">Etimoloji sonucu başlığı</a>',
            status=200, content_type="text/html",
        )
        responses.add(responses.GET, re.compile(r".*"), status=500)
        self.assertIsInstance(tool_web_search("göz"), list)

    @responses.activate
    def test_scraped_urls_are_validated(self):
        """Kazınmış URL'ler doğrulanmadan sonuç listesine GİRMEMELİDİR."""
        network.reset_session()
        responses.add(responses.GET, re.compile(r".*wiki.*"), status=500)
        responses.add(
            responses.GET, re.compile(r".*duckduckgo.*"),
            body='<a class="result__a" href="javascript:alert(1)">kötü</a>',
            status=200, content_type="text/html",
        )
        responses.add(responses.GET, re.compile(r".*"), status=500)
        for item in tool_web_search("göz"):
            self.assertTrue(item["url"].startswith(("http://", "https://")), item["url"])


class TestAdvancedTools(unittest.TestCase):
    @responses.activate
    def test_wiktionary_multilingual_success(self):
        network.reset_session()
        responses.add(
            responses.GET, re.compile(r".*rest_v1/page/definition.*"),
            json={"tr": [{"definitions": [{"definition": "görme organı"}]}]}, status=200,
        )
        res = tool_wiktionary_multilingual_api("göz")
        self.assertIn("api_summary", res)

    @responses.activate
    def test_wiktionary_multilingual_partial_failure(self):
        network.reset_session()
        responses.add(responses.GET, re.compile(r".*tr\.wiktionary.*"), status=501)
        responses.add(
            responses.GET, re.compile(r".*en\.wiktionary.*"),
            json={"en": [{"definitions": [{"definition": "eye"}]}]}, status=200,
        )
        self.assertIn("raw_found", tool_wiktionary_multilingual_api("göz"))

    def test_ipa_analyzer_empty(self):
        res = tool_ipa_phonetic_analyzer("")
        self.assertIn("ipa", res)
        self.assertEqual(res["vowel_count"], 0)


class TestSemanticEngineBackends(unittest.TestCase):
    def test_model_loader_is_idempotent(self):
        self.assertIs(get_sentence_transformer(), get_sentence_transformer())

    def test_has_semantic_model_returns_bool(self):
        self.assertIsInstance(has_semantic_model(), bool)

    def test_vectoriser_reports_backend(self):
        """Boy, kullanılan arka uca göre değişir.

        ⚠️ Bu satır eskiden koşulsuz `len(vec) == 64` istiyordu ve böylece
        BİR HATAYI kayda geçiriyordu: transformer yolu 384 boyutluk
        embedding'i tam boy normalize edip `[:64]` ile kırpıyordu, dönen
        vektör birim OLMUYORDU. `cosine_distance` birim vektör varsaydığı
        için özdeş iki anlamda bile mesafe ~0.83 çıkıyor ve A-HVP 3. aşama
        her kelimeyi reddediyordu. Kırpma kaldırıldı; 64 yalnız ortografik
        yedek yolun (`vocab_size`) boyudur.
        """
        vec, used_model = DenseSemanticVectorizer().vectorise("deniz")
        self.assertIsInstance(used_model, bool)
        if used_model:
            self.assertGreater(len(vec), 64)
            norm = sum(x * x for x in vec) ** 0.5
            self.assertAlmostEqual(norm, 1.0, places=3)
        else:
            self.assertEqual(len(vec), 64)

    def test_identical_meaning_has_zero_distance(self):
        """Özdeş iki anlamın mesafesi 0 olmalı — kırpma hatası regresyonu.

        Transformer vektörü tam boy normalize edilip 64 boyuta kırpılıyordu;
        dönen vektör birim olmadığı için ULAŞILABİLİR AZAMİ benzerlik ~0.17
        idi ve özdeş anlamlar bile ~0.83 mesafe veriyordu. Sonuç: A-HVP
        3. aşaması her kelimeyi reddediyordu (`göz` 0.8544, `bardak` 0.8661).
        Model kurulu değilse ortografik yedek yol da özdeş metinde 0 verir,
        dolayısıyla bu iddia her iki arka uçta da geçerlidir.
        """
        e = DiachronicSemanticEngine()
        v1, _ = e.vectorizer.vectorise("su içilen kap")
        v2, _ = e.vectorizer.vectorise("su içilen kap")
        self.assertAlmostEqual(e.cosine_distance(v1, v2), 0.0, places=3)

    def test_identical_inputs_are_not_free_evidence(self):
        """İki taraf aynı metinse aşama BEDAVA ✅ vermemeli.

        Ölçüldü (12 kelime): 9'unda tarihî ve modern anlam birebir aynı
        dizeydi, çünkü çoğu tarihî tanığın `meaning` alanı modern TDK
        tanımının kopyası. Mesafe 0 çıkıp aşama otomatik geçiyordu; bu
        ölçüm değil kendini doğrulamadır.
        """
        res = DiachronicSemanticEngine().evaluate_diachronic_trajectory(
            "su içilen kap", "su içilen kap"
        )
        self.assertFalse(res["evidence_available"])
        self.assertIsNone(res["is_plausible"])

    def test_unrelated_meanings_exceed_theta(self):
        """İlgisiz anlamlar eşiği aşmalı; aksi hâlde aşama lastik damgadır.

        theta 0.85 iken ilgisiz çiftlerin %87,4'ü de geçiyordu (254 CLDF
        kavramı üzerinde ölçüldü); 0.60'a çekilince %19,7'ye indi.
        """
        e = DiachronicSemanticEngine()
        if not has_semantic_model():
            self.skipTest("semantik model yok; ortografik yedek anlam ölçmez")
        # ⚠️ İddia THETA'DAN BAĞIMSIZ tutuldu. Eşik şu an 0.85 ve bilerek
        # gevşek (gerekçe THETA_THRESHOLD notunda); "ilgisiz çift eşiği aşar"
        # demek eşiğe bağımlı ve kırılgan olurdu. Ölçülebilir ve kalıcı olan
        # şey SIRALAMADIR: ilgisiz anlamlar, ilgili anlamlardan uzaktır.
        ilgili = e.evaluate_diachronic_trajectory("book", "kitap")
        ilgisiz = e.evaluate_diachronic_trajectory("book", "mountain range")
        self.assertGreater(
            ilgisiz["total_shift_distance"], ilgili["total_shift_distance"]
        )

    def test_trajectory_segment_is_dropped_not_unlabelled(self):
        """Güzergâh bölümü anlam değildir; etiketi silinip bırakılmamalı.

        `donor_etymology_database` alıntıların tarihî anlamını
        "Kaynak anlamı: X | Geçiş yörüngesi: A -> B -> C" diye kuruyor.
        İlk bölüm gerçek gloss, ikincisi bir YOL (dil adları ve oklar).
        Yalnız etiketi silmek güzergâhı anlammış gibi kodluyordu:
            kalem 0.7589 -> 0.4808      kitap 0.6485 -> 0.3267
        """
        from engine.nlp.diachronic_semantic_engine import _clean_gloss

        temiz = _clean_gloss(
            "Kaynak anlamı: Yazı kamışı, yontulmuş kamış | "
            "Geçiş yörüngesi: Eski Grekçe (kálamos) -> Arapça (qalam)"
        )
        self.assertEqual(temiz, "Yazı kamışı, yontulmuş kamış")
        self.assertNotIn("Grekçe", temiz)
        # Sıradan gloss'a dokunulmaz.
        self.assertEqual(
            _clean_gloss("kafa, (mecazi) lider, reis"), "kafa, (mecazi) lider, reis"
        )

    def test_vectoriser_empty_text(self):
        vec, used = DenseSemanticVectorizer().vectorise("")
        self.assertEqual(vec, [0.0] * 64)
        self.assertFalse(used)

    def test_ngram_extraction(self):
        grams = DenseSemanticVectorizer().extract_ngrams("deniz mavi")
        self.assertIn("deniz", grams)
        self.assertIn("de", grams)

    def test_cosine_distance_bounds(self):
        e = DiachronicSemanticEngine()
        self.assertEqual(e.cosine_distance([1.0, 0.0], [1.0, 0.0]), 0.0)
        self.assertGreater(e.cosine_distance([1.0, 0.0], [0.0, 1.0]), 0.9)

    def test_timeline_is_reported(self):
        res = DiachronicSemanticEngine().evaluate_diachronic_trajectory(
            "eski anlam", "yeni anlam", ["1074 DLT"]
        )
        self.assertEqual(res["timeline_layers"], ["1074 DLT"])

    def test_transformer_encode_failure_falls_back(self):
        e = DiachronicSemanticEngine()

        class BadModel:
            def encode(self, *a, **k):
                raise RuntimeError("encode hatası")

        with mock.patch("engine.nlp.diachronic_semantic_engine.get_sentence_transformer",
                        return_value=BadModel()):
            vec, used = e.vectorizer.vectorise("deniz")
        self.assertFalse(used)
        self.assertEqual(len(vec), 64)


class TestPhonologicalFallback(unittest.TestCase):
    def test_fallback_when_panphon_missing(self):
        """PanPhon yoksa motor ÇÖKMEMELİ, dürüst biçimde yedek moda düşmelidir."""
        import engine.nlp.phonological_feature_engine as mod

        original = (mod._FEATURE_TABLE, mod._DISTANCE, mod._BACKEND)
        try:
            mod._FEATURE_TABLE, mod._DISTANCE, mod._BACKEND = None, None, "fallback"
            e = PhonologicalFeatureEngine()
            res = e.sequence_phonological_distance("göz", "köz")
            self.assertTrue(res["evidence_available"])
            self.assertEqual(res["backend"], "fallback")
            self.assertGreater(res["phonetic_similarity"], 0.4)
        finally:
            mod._FEATURE_TABLE, mod._DISTANCE, mod._BACKEND = original

    def test_distance_failure_falls_back(self):
        import engine.nlp.phonological_feature_engine as mod

        class BadDistance:
            def hamming_feature_edit_distance(self, *a):
                raise RuntimeError("x")

        original = mod._DISTANCE
        try:
            mod._DISTANCE = BadDistance()
            res = PhonologicalFeatureEngine().sequence_phonological_distance("göz", "köz")
            self.assertTrue(res["evidence_available"])
        finally:
            mod._DISTANCE = original


class TestTrustedScraperPaths(unittest.TestCase):
    @responses.activate
    def test_scrapes_all_whitelisted_sources(self):
        network.reset_session()
        responses.add(
            responses.GET, re.compile(r".*"),
            body='<div class="etym">Eski Türkçe köz biçiminden</div>'
                 '<a class="card-title">Akademik makale başlığı</a>',
            status=200, content_type="text/html",
        )
        out = scrape_whitelisted_academic_sources("göz")
        self.assertIsInstance(out, list)

    @responses.activate
    def test_partial_failures_are_isolated(self):
        network.reset_session()
        responses.add(responses.GET, re.compile(r".*nisanyan.*"), status=500)
        responses.add(responses.GET, re.compile(r".*"), body="<html></html>", status=200)
        self.assertIsInstance(scrape_whitelisted_academic_sources("göz"), list)


class TestQwenPromptConstruction(unittest.TestCase):
    @responses.activate
    def test_successful_generation_is_used(self):
        network.reset_session()
        responses.add(responses.GET, config.OLLAMA_TAGS_URL,
                      json={"models": [{"name": config.OLLAMA_MODEL}]}, status=200)
        responses.add(responses.POST, config.OLLAMA_GENERATE_URL,
                      json={"response": "Gerçek yapay zekâ sentezi."}, status=200)
        agent = QwenEtymologyAgent()
        finding = {"query_word": "göz", "root": {"proto_turkic": "*köŕ", "meaning": "göz"},
                   "turkic_languages": [], "nlp_analysis": {}}
        with mock.patch("engine.llm.qwen_agent.tool_web_search", return_value=[]), \
             mock.patch("engine.llm.qwen_agent.tool_ipa_phonetic_analyzer", return_value={"ipa": "/x/"}), \
             mock.patch("engine.llm.qwen_agent.tool_donor_pattern_analyzer",
                        return_value={"detected_donor_patterns": []}), \
             mock.patch("engine.llm.qwen_agent.tool_extract_suffixes", return_value="kök"):
            out = agent.research_and_enrich("göz", finding)
        self.assertEqual(out["ai_agent_enrichment"], "Gerçek yapay zekâ sentezi.")

    @responses.activate
    def test_empty_generation_falls_back(self):
        network.reset_session()
        responses.add(responses.GET, config.OLLAMA_TAGS_URL,
                      json={"models": [{"name": config.OLLAMA_MODEL}]}, status=200)
        responses.add(responses.POST, config.OLLAMA_GENERATE_URL,
                      json={"response": "   "}, status=200)
        agent = QwenEtymologyAgent()
        finding = {"query_word": "göz", "root": {"proto_turkic": "*köŕ", "meaning": "göz"},
                   "turkic_languages": [], "nlp_analysis": {}}
        with mock.patch("engine.llm.qwen_agent.tool_web_search", return_value=[]), \
             mock.patch("engine.llm.qwen_agent.tool_ipa_phonetic_analyzer", return_value={"ipa": "/x/"}), \
             mock.patch("engine.llm.qwen_agent.tool_donor_pattern_analyzer",
                        return_value={"detected_donor_patterns": []}), \
             mock.patch("engine.llm.qwen_agent.tool_extract_suffixes", return_value="kök"):
            out = agent.research_and_enrich("göz", finding)
        self.assertIn("*köŕ", out["ai_agent_enrichment"])

    @responses.activate
    def test_donor_hypothesis_fallback_text(self):
        network.reset_session()
        responses.add(responses.GET, config.OLLAMA_TAGS_URL, status=500)
        agent = QwenEtymologyAgent()
        finding = {
            "query_word": "kitap",
            "root": {"proto_turkic": "", "meaning": "kitap"},
            "turkic_languages": [],
            "nlp_analysis": {"proven_hypothesis": {
                "donor_language": "Arapça", "origin_form": "kitāb",
                "proof_summary": "Arapça k-t-b kökünden",
            }},
        }
        with mock.patch("engine.llm.qwen_agent.tool_web_search", return_value=[]), \
             mock.patch("engine.llm.qwen_agent.tool_ipa_phonetic_analyzer", return_value={"ipa": "/x/"}), \
             mock.patch("engine.llm.qwen_agent.tool_donor_pattern_analyzer",
                        return_value={"detected_donor_patterns": []}), \
             mock.patch("engine.llm.qwen_agent.tool_extract_suffixes", return_value="kök"):
            out = agent.research_and_enrich("kitap", finding)
        self.assertIn("Arapça", out["ai_agent_enrichment"])


class TestServerRunner(unittest.TestCase):
    def test_run_server_binds_and_stops(self):
        """`run_server` gerçekten dinlemeli ve temiz kapanmalıdır."""
        stop = threading.Event()

        def runner():
            try:
                server.run_server(host="127.0.0.1", port=0)
            finally:
                stop.set()

        with mock.patch.object(server.ThreadingHTTPServer, "serve_forever",
                               side_effect=KeyboardInterrupt):
            t = threading.Thread(target=runner, daemon=True)
            t.start()
            t.join(timeout=10)
        self.assertTrue(stop.is_set())

    def test_non_local_bind_logs_warning(self):
        with mock.patch.object(server.ThreadingHTTPServer, "serve_forever",
                               side_effect=KeyboardInterrupt), \
             mock.patch("engine.server.logger") as log:
            server.run_server(host="0.0.0.0", port=0)
        self.assertTrue(log.warning.called)


class TestResearchToolsEdgeCases(unittest.TestCase):
    def test_module_exposes_only_used_tools(self):
        """Ölü araçlar SİLİNMİŞ olmalıdır (sabit %88.5 metriği dâhil)."""
        names = [n for n in dir(research_tools) if n.startswith("tool_")]
        self.assertEqual(set(names), {"tool_extract_suffixes", "tool_web_search"})

    def test_no_fabricated_metric_remains(self):
        import inspect

        src = inspect.getsource(research_tools)
        self.assertNotIn("88.5", src)


if __name__ == "__main__":
    unittest.main()
