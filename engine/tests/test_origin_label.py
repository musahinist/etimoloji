"""Kayıt köken etiketi: live / local / seed (sağlık ucu ve web paneli dahil)."""

import unittest

from engine.fetchers.apertium import ApertiumFetcher
from engine.fetchers.base import BaseFetcher
from engine.fetchers.historical_index import HistoricalIndexFetcher, ModernIndexFetcher
from engine.fetchers.northeuralex import NorthEuraLexFetcher
from engine.fetchers.proto_turkic_local import LocalProtoTurkicFetcher
from engine.fetchers.starling import StarlingFetcher


class _Probe(BaseFetcher):
    @property
    def source_name(self):
        return "Probe"

    def fetch(self, word):
        return self.empty_result()


class TestOriginLabel(unittest.TestCase):
    def _origin(self, *, seed=False, local=False):
        probe = _Probe()
        probe.is_seed_source, probe.is_local = seed, local
        return probe.make_entry("kk", "көз")["origin"]

    def test_three_labels(self):
        self.assertEqual(self._origin(), "live")
        self.assertEqual(self._origin(seed=True), "seed")
        self.assertEqual(self._origin(local=True), "local")

    def test_downloaded_sources_are_local_not_seed(self):
        for cls in (NorthEuraLexFetcher, ApertiumFetcher, LocalProtoTurkicFetcher,
                    HistoricalIndexFetcher, ModernIndexFetcher):
            with self.subTest(cls=cls.__name__):
                fetcher = cls()
                self.assertEqual(fetcher.origin_label, "local")
                self.assertFalse(fetcher.is_seed_source)

    def test_starling_database_is_local_not_live(self):
        self.assertEqual(StarlingFetcher(use_database=True).origin_label, "local")
        self.assertEqual(StarlingFetcher(use_database=False).origin_label, "seed")


class TestHealthCounts(unittest.TestCase):
    def test_every_source_in_exactly_one_class(self):
        import json

        from engine.tests.fakes import FakeFetcher
        from engine.tests.test_server_and_cli import ServerHarness

        local = FakeFetcher(name="Yerel")
        local.is_local = True
        fetchers = [FakeFetcher(name="Canlı"), FakeFetcher(name="Tohum", seed=True), local]
        with ServerHarness(fetchers=fetchers) as h:
            _, _, body = h.get("/api/health")
        data = json.loads(body)
        self.assertEqual(
            (data["live_sources"], data["local_sources"], data["seed_sources"], data["fetcher_count"]),
            (1, 1, 1, 3),
        )


class TestWebPanelLabels(unittest.TestCase):
    def test_panel_knows_local_origin(self):
        from engine.config import PROJECT_ROOT

        html = (PROJECT_ROOT / "web" / "index.html").read_text(encoding="utf-8")
        self.assertIn("local:", html)
        self.assertIn("yerel veri", html)
        self.assertNotIn('item.origin === "seed" ? "tohum veri" : "canlı kaynak"', html)


if __name__ == "__main__":
    unittest.main()
