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
