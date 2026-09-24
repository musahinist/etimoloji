"""Yerel kaynaklar: fetch() sözleşmesi — bozuk veri istisna atmaz, boş sonuç döner."""

import unittest
from unittest import mock

from engine.fetchers import apertium, northeuralex, proto_turkic_local
from engine.fetchers.apertium import ApertiumFetcher
from engine.fetchers.northeuralex import NorthEuraLexFetcher
from engine.fetchers.proto_turkic_local import LocalProtoTurkicFetcher


class TestFetchNeverRaises(unittest.TestCase):
    def test_northeuralex_broken_csv(self):
        fetcher = NorthEuraLexFetcher()
        with mock.patch.object(northeuralex, "_load", side_effect=KeyError("ID")):
            self.assertEqual(fetcher.fetch("göz"), fetcher.empty_result())

    def test_apertium_broken_table(self):
        fetcher = ApertiumFetcher()
        with mock.patch.object(apertium, "_table", side_effect=ValueError("bozuk .dix")):
            self.assertEqual(fetcher.fetch("göz"), fetcher.empty_result())

    def test_proto_local_broken_cache(self):
        fetcher = LocalProtoTurkicFetcher()
        with mock.patch.object(proto_turkic_local, "_cache_ready", side_effect=OSError("kilitli")):
            self.assertEqual(fetcher.fetch("göz"), fetcher.empty_result())

    def test_proto_local_broken_index(self):
        fetcher = LocalProtoTurkicFetcher()
        with mock.patch("engine.db.lexicon_index.LexiconIndex.lookup", side_effect=RuntimeError("disk")), \
                mock.patch("engine.db.lexicon_index.LexiconIndex.exists", new_callable=mock.PropertyMock,
                           return_value=True), \
                mock.patch.object(proto_turkic_local, "_cache_ready", return_value=True):
            self.assertEqual(fetcher.fetch("göz"), fetcher.empty_result())


if __name__ == "__main__":
    unittest.main()
