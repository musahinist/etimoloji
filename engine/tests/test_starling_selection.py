"""Starling: `-mak/-mek` kelimenin ad ve fiil okuması."""

import unittest
from unittest import mock

from engine.db import starling as db
from engine.db.starling import StarlingEtymology, lookup_turkish

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


if __name__ == "__main__":
    unittest.main()
