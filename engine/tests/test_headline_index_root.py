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


if __name__ == "__main__":
    unittest.main()
