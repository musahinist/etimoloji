"""Soğuk başlangıç: tembel yüklemelerin iş parçacığı güvenliği ve ön ısıtma."""
from __future__ import annotations

import threading
import time
import unittest
from unittest import mock

from engine.nlp import diachronic_semantic_engine as sem
from engine.nlp import phonological_feature_engine as phon


class TestSentenceTransformerLoading(unittest.TestCase):
    def setUp(self):
        self._saved = (sem._ST_MODEL, sem._ST_TRIED)
        sem._ST_MODEL, sem._ST_TRIED = None, False

    def tearDown(self):
        sem._ST_MODEL, sem._ST_TRIED = self._saved

    def _slow_loader(self, calls: list[int]):
        sentinel = object()

        def load():
            calls.append(1)
            time.sleep(0.05)
            sem._ST_MODEL = sentinel
            return sentinel
        return load, sentinel

    def test_concurrent_callers_get_the_loaded_model(self):
        """Regresyon: yükleme sürerken ikinci çağıran `None` alıyordu."""
        calls: list[int] = []
        load, sentinel = self._slow_loader(calls)
        results: list[object] = []
        with mock.patch.object(sem, "_load_sentence_transformer", load):
            threads = [threading.Thread(target=lambda: results.append(sem.get_sentence_transformer()))
                       for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        self.assertEqual(len(calls), 1)
        self.assertEqual(results, [sentinel] * 4)

    def test_prewarm_then_get_returns_same_model(self):
        calls: list[int] = []
        load, sentinel = self._slow_loader(calls)
        with mock.patch.object(sem, "_load_sentence_transformer", load):
            sem.prewarm_sentence_transformer()
            self.assertIs(sem.get_sentence_transformer(), sentinel)
            sem.prewarm_sentence_transformer()  # yüklendiyse iş parçacığı açmaz
        self.assertEqual(len(calls), 1)


class TestPhonologyBackendLoading(unittest.TestCase):
    def test_backend_published_after_epitran(self):
        """`_BACKEND` dolu görünüyorsa Epitran denemesi de bitmiş olmalı."""
        saved = (phon._FEATURE_TABLE, phon._DISTANCE, phon._EPITRAN, phon._BACKEND)
        phon._BACKEND = None
        seen: list[tuple[object, object]] = []
        try:
            threads = [threading.Thread(target=lambda: seen.append((phon._load_backend(), phon._EPITRAN)))
                       for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(len({backend for backend, _ in seen}), 1)
            self.assertEqual(len({id(epi) for _, epi in seen}), 1)
        finally:
            phon._FEATURE_TABLE, phon._DISTANCE, phon._EPITRAN, phon._BACKEND = saved


if __name__ == "__main__":
    unittest.main()
