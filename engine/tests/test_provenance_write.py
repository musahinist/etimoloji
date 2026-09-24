"""Künye yalnız veri değişince yazılır: taze klonda `make bootstrap` aynı veriyi
indirip commit edilmiş künyeyi yalnız indirme zamanı yüzünden değiştirmesin."""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from engine.utils.provenance import deterministic_gzip, write_if_changed

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_{name}_prov", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestWriteIfChanged(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "_provenance.json"

    def _write(self, data):
        return write_if_changed(self.path, data, json.dumps(data, ensure_ascii=False, indent=2))

    def test_same_data_keeps_committed_file(self):
        old = {"retrieved_at": "2026-01-01T00:00:00+00:00", "files": {"a": {"sha256": "x"}}}
        self._write(old)
        before = self.path.read_bytes()
        kept = self._write({**old, "retrieved_at": "2026-09-24T12:00:00+00:00"})
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(kept["retrieved_at"], "2026-01-01T00:00:00+00:00")

    def test_changed_data_rewrites(self):
        self._write({"retrieved_at": "t1", "files": {"a": {"sha256": "x"}}})
        self._write({"retrieved_at": "t2", "files": {"a": {"sha256": "y"}}})
        self.assertEqual(json.loads(self.path.read_text())["files"]["a"]["sha256"], "y")

    def test_unreadable_existing_file_is_replaced(self):
        self.path.write_text("{bozuk", encoding="utf-8")
        self._write({"retrieved_at": "t", "files": {}})
        self.assertEqual(json.loads(self.path.read_text())["files"], {})


class TestDeterministicGzip(unittest.TestCase):
    def test_same_content_same_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            digests = []
            for name in ("a.gz", "b.gz"):
                path = Path(tmp) / name
                with deterministic_gzip(path) as handle:
                    handle.write(b'{"word": "su"}\n')
                digests.append(hashlib.sha256(path.read_bytes()).hexdigest())
                time.sleep(1.05)  # gzip başlığındaki zaman damgası saniye çözünürlüklü
            self.assertEqual(digests[0], digests[1])
            self.assertEqual(gzip.open(Path(tmp) / "a.gz").read(), b'{"word": "su"}\n')


class TestProtoPatternsSave(unittest.TestCase):
    def test_retraining_same_table_keeps_file(self):
        from engine.nlp.proto_patterns import ProtoPatternTable, save

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proto_patterns.json"
            table = ProtoPatternTable(trained_at="2026-01-01T00:00:00+00:00")
            table.observe("tr", "g", "k")
            save(table, path)
            before = path.read_bytes()
            table.trained_at = "2026-09-24T00:00:00+00:00"
            save(table, path)
            self.assertEqual(path.read_bytes(), before)
            table.n_sets += 1
            save(table, path)
            self.assertNotEqual(path.read_bytes(), before)


class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.status_code = 200


class TestApertiumBootstrap(unittest.TestCase):
    def test_redownload_of_same_files_keeps_provenance(self):
        module = _load("download_apertium")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            session = mock.MagicMock()
            session.get.side_effect = lambda url, timeout: _FakeResponse(b"<dictionary/>")
            with mock.patch.object(module, "TARGET", target), \
                    mock.patch.object(module.requests, "Session", return_value=session):
                module.main([])
                before = (target / "_provenance.json").read_bytes()
                for dix in target.glob("*.dix"):  # taze klon: künye var, veri yok
                    dix.unlink()
                with mock.patch.object(module, "datetime") as clock:
                    clock.now.return_value.isoformat.return_value = "2099-01-01T00:00:00+00:00"
                    module.main([])
                self.assertEqual((target / "_provenance.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
