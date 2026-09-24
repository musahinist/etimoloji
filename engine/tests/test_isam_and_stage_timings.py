"""İSAM canlı yolunun bayrağa bağlanması ve aşama sürelerinin eksiksizliği."""
from __future__ import annotations

import os
import re
import tempfile
import unittest

import responses

from engine import config
from engine.db.database import DatabaseManager
from engine.fetchers.isam_ansiklopedi import IsamAnsiklopediFetcher
from engine.search_engine import SearchEngine
from engine.tests.fakes import FakeFetcher
from engine.utils import network


class TestIsamLiveFlag(unittest.TestCase):
    def setUp(self):
        network.reset_session()

    def test_default_is_off(self):
        self.assertFalse(config.LIVE_ISAM)

    @responses.activate
    def test_seed_only_makes_no_request(self):
        fetcher = IsamAnsiklopediFetcher(live=False)
        res = fetcher.fetch("tanrı")
        self.assertEqual(len(responses.calls), 0)
        self.assertTrue(fetcher.is_seed_source)
        self.assertFalse(fetcher.exact_query_only)
        self.assertIn("tohum", fetcher.source_name)
        self.assertNotIn("canlı", fetcher.source_name)
        self.assertEqual(res["turkic_languages"][0]["origin"], "seed")

    @responses.activate
    def test_live_label_and_headword_only(self):
        responses.add(responses.GET, re.compile(r".*islamansiklopedisi\.org\.tr.*"),
                      body="<p>Kitap kelimesi Arapça ktb kökünden gelir; etimoloji bakımından yazmak.</p>",
                      status=200, content_type="text/html")
        fetcher = IsamAnsiklopediFetcher(live=True)
        self.assertFalse(fetcher.is_seed_source)
        self.assertTrue(fetcher.exact_query_only)
        self.assertIn("canlı", fetcher.source_name)
        res = fetcher.fetch("kitap")
        self.assertEqual(len(responses.calls), 1)
        self.assertEqual(res["turkic_languages"][0]["origin"], "live")

    @responses.activate
    def test_paragraph_with_only_kok_is_not_etymology(self):
        """Regresyon: içinde yalnız "kök" geçen paragraf etimoloji sayılıyordu."""
        responses.add(responses.GET, re.compile(r".*"),
                      body="<p>Osmanlı döneminde kök salmış bir kurum olarak kitap ticareti gelişti.</p>",
                      status=200, content_type="text/html")
        res = IsamAnsiklopediFetcher(live=True).fetch("kitap")
        self.assertEqual(res["turkic_languages"], [])

    def test_persistent_cache_host(self):
        self.assertIn("islamansiklopedisi.org.tr", network.PERSISTENT_CACHE_HOSTS)


class TestStageTimingsCoverTotal(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.db = DatabaseManager(self.db_path)

    def tearDown(self):
        os.remove(self.db_path)

    def test_stages_sum_to_total(self):
        fetcher = FakeFetcher(name="Sahte", entries=[("tr", "göz"), ("az", "göz")],
                              meaning="göz", only_for="göz")
        res = SearchEngine(db_manager=self.db, fetchers=[fetcher]).search("göz", save_to_db=False)
        timings = res["diagnostics"]["stage_timings_ms"]
        covered = sum(v for k, v in timings.items() if k != "total")
        # Her aşama ms'ye kesilir: aşama başına en çok 1 ms kayıp.
        self.assertLessEqual(covered, timings["total"])
        self.assertGreaterEqual(covered, timings["total"] - len(timings))
        for stage in ("morphology", "fetch", "witness_filter", "nlp", "graph"):
            self.assertIn(stage, timings)


if __name__ == "__main__":
    unittest.main()
