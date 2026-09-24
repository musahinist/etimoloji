"""ApertiumFetcher: ad/fiil okuması, boş anlam alanı, Çuvaşça fiil kanıt toplamı."""

from __future__ import annotations

import unittest
from unittest import mock

from engine.fetchers import apertium

_TABLE = {
    ("parmak", False): [("ky", "бармак")],
    ("kay", True): [("az", "qay")],
    ("kaymak", False): [("tt", "каймак")],
    ("master", False): [("crh", "master")],
    # Çuvaşça fiiller: yazılış 0,33 + tahmin 1,0 -> geçer; yazılış 0 + tahmin 0,67 -> elenir.
    ("sev", True): [("cv", "сав")],
    ("yaz", True): [("cv", "ҫыр")],
}
_PREDICTED = {
    "sevmek": {"cv": "savma"},
    "yazmak": {"cv": "surma"},
}


class TestApertiumFetcher(unittest.TestCase):
    def setUp(self) -> None:
        patches = [
            mock.patch.object(apertium, "_table", lambda: _TABLE),
            mock.patch.object(apertium, "_predicted_forms", lambda word: _PREDICTED.get(word, {})),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.fetcher = apertium.ApertiumFetcher()

    def _forms(self, word: str) -> list[tuple[str, str]]:
        return [(e["lang_code"], e["word"]) for e in self.fetcher.fetch(word)["turkic_languages"]]

    def test_mak_noun_reading_is_searched(self) -> None:
        self.assertEqual(self._forms("parmak"), [("ky", "бармак")])

    def test_both_readings_pass_similarity_on_own_lemma(self) -> None:
        self.assertEqual(self._forms("kaymak"), [("az", "qay"), ("tt", "каймак")])

    def test_meaning_is_empty_and_lemma_kept_separately(self) -> None:
        entry = self.fetcher.fetch("master")["turkic_languages"][0]
        self.assertEqual(entry["meaning"], "")
        self.assertEqual(entry["translation_of"], "master")

    def test_chuvash_verb_needs_both_kinds_of_evidence(self) -> None:
        self.assertEqual(self._forms("sevmek"), [("cv", "сав")])
        self.assertEqual(self._forms("yazmak"), [])


if __name__ == "__main__":
    unittest.main()
