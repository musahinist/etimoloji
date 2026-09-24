"""Proto-Türkçe yerel döküm önbelleği: dökümden doğrudan okumayla birebir aynı sonuç."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from engine.fetchers import proto_turkic_local as ptl


def _clear() -> None:
    ptl._pages.cache_clear()
    ptl._cache_ready.cache_clear()
    ptl._entry.cache_clear()


def _dump_based(proto: str) -> tuple[list[tuple[str, str, str]], str]:
    page = ptl._pages().get(ptl._key(proto))
    return ptl._page_descendants(page), ptl._page_gloss(page)


class _TempCache(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmp.name) / "trk-pro.sqlite3"
        patcher = mock.patch.object(ptl, "CACHE", self.cache)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(_clear)
        _clear()


@unittest.skipUnless(ptl.DUMP.exists(), "trk-pro dökümü yok")
class TestCacheMatchesDump(_TempCache):
    def test_every_key_identical(self):
        keys = list(ptl._pages())
        expected = {key: _dump_based(key) for key in keys}
        _clear()
        self.assertGreater(len(keys), 50)
        for key in keys:
            self.assertEqual((ptl.descendants(key), ptl.gloss(key)), expected[key], key)
        self.assertEqual(ptl.descendants("*yokböyleköken"), [])
        self.assertEqual(ptl.gloss("*yokböyleköken"), "")


class TestCacheInvalidation(_TempCache):
    def _write_dump(self, path: Path, word: str, gloss: str) -> None:
        record = {
            "word": word,
            "senses": [{"glosses": [gloss]}],
            "descendants": [
                {"lang_code": "az", "word": "göz", "roman": "", "raw_tags": ["inherited"]},
                {"lang_code": "grc", "word": "x", "raw_tags": ["borrowed"]},
                {"lang_code": "cv", "word": "куҫлӗх", "raw_tags": ["reshaped by analogy"]},
                {"lang_code": "kk", "word": "көз", "roman": "köz", "descendants": [{"lang_code": "ky", "word": "*köz"}]},
            ],
        }
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def test_rebuilds_when_dump_changes(self):
        dump = Path(self._tmp.name) / "trk-pro.jsonl.gz"
        self._write_dump(dump, "göz", "eye")
        with mock.patch.object(ptl, "DUMP", dump):
            self.assertEqual(ptl.descendants("*göz"), [("az", "göz", ""), ("kk", "көз", "köz")])
            self.assertEqual(ptl.gloss("*göz"), "eye")
            self.assertTrue(self.cache.exists())

            self._write_dump(dump, "göz", "eye; sight")
            stat = dump.stat()
            os.utime(dump, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
            _clear()
            self.assertEqual(ptl.gloss("*göz"), "eye; sight")

    def test_missing_dump_is_empty(self):
        with mock.patch.object(ptl, "DUMP", Path(self._tmp.name) / "yok.jsonl.gz"):
            self.assertEqual(ptl.descendants("*göz"), [])
            self.assertEqual(ptl.gloss("*göz"), "")
            self.assertFalse(self.cache.exists())


if __name__ == "__main__":
    unittest.main()
