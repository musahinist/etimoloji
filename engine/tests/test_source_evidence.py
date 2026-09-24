"""Kaynakların verdiği bilginin (Starling, alıntı zinciri, akraba
listesi) doğru okunduğunu sabitleyen testler."""

import unittest

from engine.db.starling import StarlingEtymology, _turkish_forms, decode
from engine.nlp.borrowing_chain import source_loan_step
from engine.search_engine import _cited_cognates, _english_query_gloss, _query_source_proto


class TestStarlingDecode(unittest.TestCase):
    def test_special_letters(self):
        # ō = 0xD2, č = 0xB6 (boncuk: *bōnčok)
        self.assertEqual(decode(b"*b\xd2n\xb6ok"), "*bōnčok")

    def test_combining_mark_binds_to_previous_letter(self):
        # 0xB1 birleşik vurgu: *kul± -> *kuĺ (ayrı harf DEĞİL)
        self.assertEqual(decode(b"*kul\xb1"), "*kuĺ")

    def test_cp866_russian(self):
        # "кол" CP866: 0xAA 0xAE 0xAB
        self.assertEqual(decode(b"\xaa\xae\xab"), "кол")

    def test_italic_tags_are_stripped(self):
        self.assertEqual(decode(b"\\Ibit-\\i 2"), "bit- 2")

    def test_turkish_forms_map_transcription(self):
        self.assertEqual(_turkish_forms("jol 1"), {"yol"})
        self.assertEqual(_turkish_forms("uč- 'летать'"), {"uç-"})
        self.assertIn("kuş", _turkish_forms("kuš 1, (dial.) kuš-"))

    def test_earliest_dated_source(self):
        etym = StarlingEtymology(1, "*bōnčok", "beads", {"ATU": "mončuq (Orkh., OUygh.)"})
        self.assertEqual(etym.earliest_dated_source(), (732, "Orkh."))


INDEX_SOURCE = "Tarihî Katman (yerel sözlük indeksi: Eski Türkçe, Osmanlıca, Çağatayca)"


class TestSourceEvidence(unittest.TestCase):
    def test_source_loan_step_requires_loan_relation(self):
        derived = {"lang_code": "donor", "word": "facies", "relation": "Ses evrimi (evolution)"}
        loan = {"lang_code": "donor", "word": "faccia", "relation": "Alıntı (loan)"}
        self.assertIsNone(source_loan_step([derived]))
        self.assertEqual(source_loan_step([derived, loan])["word"], "faccia")

    def _own(self, origin, cognates=None, **extra):
        return {"lang_code": "ota", "lang_name": "Osmanlı Türkçesi", "word": "بونجق",
                "comparison": "boncuk", "lexicon_origin": origin, "source": INDEX_SOURCE,
                "source_cognates": cognates or [], **extra}

    def test_cited_cognates_only_from_inherited_record(self):
        cognates = [{"lang": "kk", "form": "моншақ"}, {"lang": "mn", "form": "x"}]
        inherited = _cited_cognates("boncuk", [self._own("miras", cognates)])
        self.assertEqual([c["lang_code"] for c in inherited], ["kk"])
        # Alıntının "akrabaları" paralel alıntıdır; tanık olmaz.
        self.assertEqual(_cited_cognates("boncuk", [self._own("alıntı", cognates)]), [])

    def test_query_source_proto(self):
        entry = self._own("miras", donor_lang="trk-pro", donor_form="*bōnčuk", meaning="bead")
        self.assertEqual(_query_source_proto("boncuk", [entry])[0], "*bōnčuk")

    def test_english_gloss_skips_redirects(self):
        entries = [
            {"lang_code": "ota", "word": "x", "comparison": "köpük", "meaning": "alternative spelling of كوپوك",
             "source": INDEX_SOURCE},
            {"lang_code": "ota", "word": "y", "comparison": "köpük", "meaning": "foam, froth", "source": INDEX_SOURCE},
        ]
        self.assertEqual(_english_query_gloss("köpük", entries), "foam, froth")


if __name__ == "__main__":
    unittest.main()


class TestWiktionaryProtoLink(unittest.TestCase):
    """Proto-Türkçe kök yalnız Türkçe bölümündeki MİRAS bağlantısından alınır."""

    def test_loanword_takes_no_proto_root(self):
        from engine.fetchers.wiktionary import _turkish_proto_link

        page = "==Kumyk==\n{{inh|kum|trk-pro|*bar}}\n==Turkish==\n{{bor+|tr|fa|پارچه}}\n"
        self.assertIsNone(_turkish_proto_link(page))

    def test_inherited_root_and_meaning(self):
        from engine.fetchers.wiktionary import _turkish_proto_link

        page = "==Turkish==\nFrom {{inh|tr|trk-pro|*sub|t=water}}\n==Uzbek==\n{{inh|uz|trk-pro|*x}}"
        self.assertEqual(_turkish_proto_link(page), ("sub", "water"))

    def test_named_desc_args_are_not_forms(self):
        from engine.fetchers.wiktionary import _desc_parts

        self.assertEqual(_desc_parts("бар|tr=bar"), ("бар", "bar"))
        self.assertEqual(_desc_parts("-|der=1"), ("", ""))


class TestTextDonor(unittest.TestCase):
    def test_cognate_list_is_not_a_donor(self):
        from engine.db.lexicon_index import donor_from_text

        text = "Inherited from Ottoman Turkish گرك.\nCognates\nNorthern Kurdish gerek\nUzbek kerak"
        self.assertEqual(donor_from_text(text), ("", ""))
        self.assertEqual(donor_from_text("From Persian پارچه (pârče). Cognate with Kurdish x.")[0], "fa")


class TestNewLocalSources(unittest.TestCase):
    def test_northeuralex_has_no_mongolic_khk(self):
        """NorthEuraLex `khk` = Halha Moğolcası; motorda `khk` = Hakasça."""
        from engine.fetchers.northeuralex import LANGUAGES

        self.assertNotIn("khk", LANGUAGES)
        self.assertNotIn("khk", LANGUAGES.values())

    def test_same_concept_other_root_is_dropped(self):
        from unittest import mock

        from engine.fetchers import northeuralex

        data = ({"pencere": {"window"}}, {"window": [("tr", "pencere"), ("kk", "терезе"), ("az", "pəncərə")]},
                {"window": "WINDOW"})
        with mock.patch.object(northeuralex, "_load", return_value=data):
            words = [e["word"] for e in northeuralex.NorthEuraLexFetcher().fetch("pencere")["turkic_languages"]]
        self.assertEqual(words, ["pəncərə"])


class TestWithinOne(unittest.TestCase):
    def test_matches_full_edit_distance(self):
        import random

        from engine.db.lexicon_index import _within_one
        from engine.evaluation.metrics import edit_distance

        rng = random.Random(0)
        for _ in range(2000):
            a = "".join(rng.choice("abç") for _ in range(rng.randint(0, 5)))
            b = "".join(rng.choice("abç") for _ in range(rng.randint(0, 5)))
            self.assertEqual(_within_one(a, b), min(edit_distance(a, b), 2), (a, b))


class TestCircuitBreaker(unittest.TestCase):
    def tearDown(self):
        from engine.utils.network import reset_circuits

        reset_circuits()

    def test_opens_after_consecutive_failures(self):
        from unittest import mock

        import requests

        from engine.utils import network

        network.reset_circuits()
        session = mock.Mock()
        session.get.side_effect = requests.Timeout("yavaş")
        with mock.patch.object(network, "get_session", return_value=session), \
             mock.patch.object(network.time, "sleep"):
            for _ in range(network.CIRCUIT_FAILURES):
                self.assertIsNone(network.fetch("https://sozluk.gov.tr/gts?ara=x", max_retries=0))
            calls = session.get.call_count
            self.assertIsNone(network.fetch("https://sozluk.gov.tr/gts?ara=y", max_retries=0))
            self.assertEqual(session.get.call_count, calls, "devre açıkken istek atılmamalı")
            # Başka sunucu etkilenmez.
            network.fetch("https://example.org/", max_retries=0)
            self.assertEqual(session.get.call_count, calls + 1)


class TestProtoTurkicLocal(unittest.TestCase):
    def test_wiktionary_notation_key(self):
        from engine.fetchers.proto_turkic_local import _key

        self.assertEqual(_key("*köŕ"), _key("körᶻ"))
        self.assertEqual(_key("*bōnčuk"), _key("bončuk"))

    def test_borrowed_and_reshaped_branches_are_skipped(self):
        from engine.fetchers.proto_turkic_local import _is_borrowed

        self.assertTrue(_is_borrowed({"raw_tags": ["borrowed", "uncertain"]}))
        self.assertTrue(_is_borrowed({"raw_tags": ["reshaped by analogy or addition of morphemes"]}))
        self.assertFalse(_is_borrowed({"raw_tags": ["inherited"]}))


class TestPersistentHttpCache(unittest.TestCase):
    def test_dictionary_response_is_served_from_disk(self):
        import tempfile
        from pathlib import Path
        from unittest import mock

        from engine.utils import network

        with tempfile.TemporaryDirectory() as tmp:
            session = mock.Mock()
            body = '[{"madde": "su"}]'  # /gts JSON döner; geçersiz gövde önbelleğe girmez
            session.get.return_value = mock.Mock(status_code=200, text=body, content=b"g", encoding="utf-8",
                                                 headers={}, apparent_encoding="utf-8")
            with mock.patch.object(network, "HTTP_CACHE_PATH", Path(tmp) / "http.db"), \
                 mock.patch.object(network, "_persistent_enabled", True), \
                 mock.patch.object(network, "_decode_body", return_value=body), \
                 mock.patch.object(network, "get_session", return_value=session):
                self.assertEqual(network.fetch("https://sozluk.gov.tr/gts?ara=su"), body)
                self.assertEqual(network.fetch("https://sozluk.gov.tr/gts?ara=su"), body)
                self.assertEqual(session.get.call_count, 1, "ikinci istek diskten gelmeli")
                # Listede olmayan sunucu önbelleğe alınmaz.
                network.fetch("https://example.org/x")
                network.fetch("https://example.org/x")
                self.assertEqual(session.get.call_count, 3)


class TestCircuitPersistence(unittest.TestCase):
    def test_open_circuit_survives_process_restart(self):
        import tempfile
        from pathlib import Path
        from unittest import mock

        from engine.utils import network

        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(network, "CIRCUIT_STATE_PATH", Path(tmp) / "circuits.json"), \
             mock.patch.object(network, "_persistent_enabled", True):
            network.reset_circuits()
            for _ in range(network.CIRCUIT_FAILURES):
                network._record("sozluk.gov.tr", failed=True)
            self.assertTrue((Path(tmp) / "circuits.json").exists())
            # Yeni süreç: bellek boş, durum diskten okunur.
            network._circuits.clear()
            network._circuits_loaded = False
            self.assertTrue(network._circuit_open("sozluk.gov.tr"))
            network.reset_circuits()


class TestApertium(unittest.TestCase):
    def _fetch(self, word, table):
        from unittest import mock

        from engine.fetchers import apertium

        with mock.patch.object(apertium, "_table", return_value=table), \
             mock.patch.object(apertium, "_predicted_forms", return_value={}):
            return [(e["lang_code"], e["word"]) for e in apertium.ApertiumFetcher().fetch(word)["turkic_languages"]]

    def test_translation_that_is_not_cognate_is_dropped(self):
        table = {("pencere", False): [("ky", "терезе"), ("tk", "penjire"), ("uz", "Deniz")]}
        self.assertEqual(self._fetch("pencere", table), [("tk", "penjire")])

    def test_infinitive_looks_only_at_verbs(self):
        table = {("kırk", True): [("az", "qırx")], ("kırk", False): [("ky", "кырк")]}
        self.assertEqual(self._fetch("kırkmak", table), [("az", "qırx")])
