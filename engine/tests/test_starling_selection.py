"""Starling: ad/fiil okuması, eşsesli kök seçimi ve fetcher sözleşmesi."""

import struct
import unittest
from unittest import mock

from engine.db import starling as db
from engine.db.starling import StarlingEtymology, lookup_turkish
from engine.fetchers import starling as fetcher_module
from engine.fetchers.starling import StarlingFetcher, select_by_meaning

CREAM_VERB = StarlingEtymology(657, "*KAj-", "1 to turn back 2 to show respect")
SLIDE = StarlingEtymology(1227, "*Kāj-", "1 to slide 2 to swim 3 skis")
BREAD = StarlingEtymology(1649, "*et-mek", "bread", {"ATU": "etmek (MK)"})
MAKE = StarlingEtymology(1198, "*ēt-", "to organize, to make")
SONG = StarlingEtymology(900, "*ɨr", "song")
RISE = StarlingEtymology(901, "*Kạl(ɨ)-", "1 to rise 2 jump up")
HAND = StarlingEtymology(38, "*el, -ig", "hand", {"ATU": "älig (Orkh.)"})
PEOPLE = StarlingEtymology(1190, "*ēl", "1 peace 2 people, country", {"ATU": "el (Orkh.)"})

TABLE = {
    "kay-": (CREAM_VERB, SLIDE),
    "etmek": (BREAD,),
    "et-": (MAKE,),
    "ır": (SONG,),
    "kalk": (RISE,),
    "el": (HAND, PEOPLE),
}


def _patched_table():
    return mock.patch.object(db, "turkish_lookup", return_value=TABLE)


class TestLookupTurkish(unittest.TestCase):
    def test_mastar_returns_noun_and_verb_readings(self):
        with _patched_table():
            self.assertEqual(lookup_turkish("etmek"), (BREAD, MAKE))

    def test_verb_only(self):
        with _patched_table():
            self.assertEqual(lookup_turkish("kaymak"), (CREAM_VERB, SLIDE))

    def test_same_spelled_noun_is_not_a_verb_root(self):
        # `ırmak` "river" -> *ɨr "song" sahte eşleşmesi.
        with _patched_table():
            self.assertEqual(lookup_turkish("ırmak"), ())

    def test_unhyphenated_verb_root_with_verbal_meaning(self):
        with _patched_table():
            self.assertEqual(lookup_turkish("kalkmak"), (RISE,))


class TestSelectByMeaning(unittest.TestCase):
    def _select(self, word, matches, glosses, scores):
        with mock.patch.object(fetcher_module, "_query_glosses", return_value=glosses), \
                mock.patch.object(fetcher_module, "_meaning_scores", return_value=scores):
            return select_by_meaning(word, matches)

    def test_single_candidate_kept(self):
        self.assertIs(select_by_meaning("x", (BREAD,))[0], BREAD)

    def test_meaning_picks_the_right_homonym(self):
        chosen, scores = self._select("kaymak", (CREAM_VERB, SLIDE), ["to slide"], [0.57, 0.85])
        self.assertIs(chosen, SLIDE)
        self.assertEqual(scores, {657: 0.57, 1227: 0.85})

    def test_both_meanings_present_is_ambiguous(self):
        self.assertIsNone(self._select("yüz", (HAND, PEOPLE), ["face", "hundred"], [1.0, 1.0])[0])

    def test_no_meaning_is_ambiguous(self):
        self.assertIsNone(self._select("el", (HAND, PEOPLE), [], None)[0])

    def test_no_model_is_ambiguous(self):
        self.assertIsNone(self._select("el", (HAND, PEOPLE), ["hand"], None)[0])

    def test_nothing_above_floor_is_ambiguous(self):
        self.assertIsNone(self._select("kur", (HAND, PEOPLE), ["exchange rate"], [0.07, 0.08])[0])


class TestStarlingFetcherDatabase(unittest.TestCase):
    def _fetch(self, word, chosen):
        fetcher = StarlingFetcher(use_database=True)
        with _patched_table(), mock.patch.object(
            fetcher_module, "select_by_meaning", return_value=(chosen, {})
        ):
            return fetcher.fetch(word)

    def test_ambiguous_root_is_not_presented_as_certain(self):
        result = self._fetch("el", None)
        root = result["root"]
        self.assertEqual(root["proto_turkic"], "")
        self.assertNotIn("starling_proto", root)
        self.assertNotIn("first_attestation", result)
        self.assertIn("*el, -ig", root["reconstruction_notes"])
        self.assertIn("*ēl", root["reconstruction_notes"])
        self.assertEqual([c["number"] for c in root["starling_candidates"]], [38, 1190])

    def test_selected_root_names_the_other_candidates(self):
        result = self._fetch("el", PEOPLE)
        self.assertEqual(result["root"]["proto_turkic"], "*ēl")
        self.assertIn("öbür aday", result["root"]["reconstruction_notes"])
        self.assertEqual(result["first_attestation"]["year"], 732)


class TestStarlingFetcherContract(unittest.TestCase):
    def test_truncated_database_does_not_break_construction(self):
        with mock.patch.object(db, "load_turcet", side_effect=struct.error("unpack requires a buffer")):
            fetcher = StarlingFetcher()
        self.assertFalse(fetcher.has_database)
        self.assertTrue(fetcher.is_seed_source)

    def test_fetch_never_raises(self):
        fetcher = StarlingFetcher(use_database=True)
        with mock.patch.object(db, "turkish_lookup", side_effect=struct.error("boom")):
            self.assertEqual(fetcher.fetch("göz"), fetcher.empty_result())


if __name__ == "__main__":
    unittest.main()


class TestReflexWitnesses(unittest.TestCase):
    """Starling tanıkları: anlam numarası, ad/fiil ve türev süzgeci."""

    EYE = StarlingEtymology(
        1, "*göŕ / *gör-", "1 eye 2 to see",
        {"TRK": "göz 1, gör- 2", "DOLG": "kör- 2; köhün- 'to be seen'",
         "TOF": "kösküt- 'to show', kör- 2", "SJG": "köz 1, gör- 1",
         "CHG": "göz", "UIG": "qi(r)q", "KHAL": "*munǯuq", "KRMX": "Guš (< Az.)"},
    )

    def _forms(self, word):
        return {w["lang_code"]: w["word"] for w in db.reflex_witnesses(self.EYE, word)}

    def test_sense_and_kind(self):
        forms = self._forms("göz")
        self.assertEqual(forms["ybe"], "köz")
        self.assertNotIn("dlg", forms)  # yalnız 2. anlam (görmek) ve türev
        self.assertNotIn("kim", forms)
        self.assertEqual(self._forms("görmek")["dlg"], "kör-")
        self.assertNotIn("chg", self._forms("görmek"))  # ad biçimi fiile tanık olmaz

    def test_optional_sound_reconstruction_and_loan(self):
        forms = self._forms("göz")
        self.assertEqual(forms["ug"], "qirq")
        self.assertNotIn("klj", forms)  # yeniden kurulmuş biçim
        self.assertNotIn("crh", forms)  # alıntı işaretli

    def test_derivative_turkish_form_gives_nothing(self):
        etym = StarlingEtymology(2, "*dāt-", "1 to taste 2 taste",
                                 {"TRK": "tat- 1, tadɨm, tatɨk 2", "KRG": "tatɨq 2"})
        self.assertEqual(db.reflex_witnesses(etym, "tadım"), [])
        self.assertEqual(db.reflex_witnesses(etym, "tatık")[0]["word"], "tatɨq")
