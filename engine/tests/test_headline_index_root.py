"""Başlık kökünün indeks yedeği: sorgunun kendi Türkçe miras kaydı."""
import unittest
from unittest import mock

from engine.search_engine import _index_source_proto, _infinitive_of, _query_source_proto


def _row(word, origin, donor_lang="", donor_form="", pos="noun", lang="tr", gloss="x"):
    return {"word": word, "comparison": word, "lang_code": lang, "gloss": gloss, "pos": pos,
            "origin": origin, "donor_lang": donor_lang, "donor_form": donor_form}


class _FakeIndex:
    exists = True
    rows: dict[str, list[dict]] = {}

    def lookup(self, query, languages=None, limit=10):
        return [r for r in self.rows.get(query, []) if not languages or r["lang_code"] in languages]


def _with_rows(rows):
    fake = type("Fake", (_FakeIndex,), {"rows": rows})
    return mock.patch("engine.db.lexicon_index.LexiconIndex", fake)


class IndexSourceProtoTest(unittest.TestCase):
    def test_turkish_record_gives_root(self):
        # Fetcher'lar `tr` kaydını getirmiyordu: `anız` "kök belirlenemedi" diyordu.
        with _with_rows({"anız": [_row("anız", "miras", "trk-pro", "*aŋïŕ")]}):
            self.assertEqual(_index_source_proto("anız"), ("*aŋïŕ", "Türkiye Türkçesi"))

    def test_bare_verb_stem_uses_infinitive_verb_record(self):
        rows = {"ayırmak": [_row("ayırmak", "miras", "trk-pro", "*adïr-", pos="verb")]}
        with _with_rows(rows):
            self.assertEqual(_index_source_proto("ayır")[0], "*adïr-")

    def test_infinitive_noun_record_is_not_used(self):
        # `ulu` sorgusu `ulumak`ın ad kaydına düşmesin; yalnız fiil kaydı.
        rows = {"ulumak": [_row("ulumak", "miras", "trk-pro", "*ūlï-", pos="noun")]}
        with _with_rows(rows):
            self.assertEqual(_index_source_proto("ulu"), ("", ""))

    def test_loan_and_non_prototurkic_records_ignored(self):
        rows = {"tay": [_row("tay", "alıntı", "ar", "طَيِّء"), _row("tay", "miras", "ota", "طاى")]}
        with _with_rows(rows):
            self.assertEqual(_index_source_proto("tay"), ("", ""))

    def test_empty_template_form_rejected(self):
        # İndeksteki `*-` şablonu `kavuk` başlığına "*-" basıyordu.
        entry = {"source": "yerel sözlük indeksi", "word": "kavuk", "lang_code": "ota", "meaning": "cap",
                 "lexicon_origin": "miras", "donor_lang": "trk-pro", "donor_form": "*-"}
        self.assertEqual(_query_source_proto("kavuk", [entry]), ("", ""))

    def test_infinitive_harmony(self):
        self.assertEqual(_infinitive_of("ayır"), "ayırmak")
        self.assertEqual(_infinitive_of("sil"), "silmek")
        self.assertEqual(_infinitive_of("üt"), "ütmek")


class IndexTurkishGlossTest(unittest.TestCase):
    def test_english_detection(self):
        from engine.search_engine import looks_english

        self.assertTrue(looks_english("army, a highly organized military force"))
        self.assertTrue(looks_english("clipping of akümülatör"))
        self.assertTrue(looks_english("cosine (abbreviated form: cos)"))
        self.assertFalse(looks_english("Orduda görev yapan erden generale kadar herkes."))
        self.assertFalse(looks_english("Uğursuz"))

    def test_first_turkish_gloss_of_own_record(self):
        from engine.search_engine import _index_turkish_gloss

        rows = {"kar": [
            _row("kâr", None, gloss="Alışveriş işlerinin sağladığı kazanç"),  # başka kelime
            _row("kar", "miras", "trk-pro", "*kār", gloss="snow"),
            _row("Kar", None, pos="name", gloss="Bir soyadı."),
            _row("kar", None, gloss="(Artvin ağzı) bir tür ölçek"),
            _row("kar", None, gloss="Buz kristallerinden oluşan yağış"),
        ]}
        with _with_rows(rows):
            self.assertEqual(_index_turkish_gloss("kar"), "Buz kristallerinden oluşan yağış")

    def test_no_turkish_gloss(self):
        from engine.search_engine import _index_turkish_gloss

        with _with_rows({"neft": [_row("neft", "alıntı", "fa", "نفت", gloss="naphtha")]}):
            self.assertEqual(_index_turkish_gloss("neft"), "")


if __name__ == "__main__":
    unittest.main()
