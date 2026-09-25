"""Wiktionary/kaikki dil kodları: Hakasça `kjh` -> `khk`, Salarca `slr` -> `slq`;
Wiktionary'nin `khk`sı (Halha Moğolcası) Hakasça SAYILMAZ."""

from __future__ import annotations

import gzip
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from engine.db.lexicon_index import TURKIC_FAMILY_CODES, _cognates_from_templates, _origin_from_templates
from engine.fetchers import proto_turkic_local as ptl
from engine.db.lexicon_index import discover_lexicons, iter_entries
from engine.fetchers.base import TURKIC_LANGUAGES_MAP, lang_code_from_wiktionary, lang_code_from_wiktionary_header

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


class TestMapping(unittest.TestCase):
    def test_iso_codes_map_to_engine_codes(self):
        self.assertEqual(lang_code_from_wiktionary("kjh"), "khk")
        self.assertEqual(lang_code_from_wiktionary("slr"), "slq")
        self.assertEqual(lang_code_from_wiktionary("kk"), "kk")

    def test_wiktionary_khk_is_khalkha_not_khakas(self):
        code = lang_code_from_wiktionary("khk")
        self.assertNotEqual(code, "khk")
        self.assertNotIn(code, TURKIC_LANGUAGES_MAP)


def _node(code: str, word: str) -> dict:
    return {"lang_code": code, "word": word, "raw_tags": ["inherited"]}


class TestProtoTurkicDescendants(unittest.TestCase):
    def test_khakas_and_salar_kept_khalkha_dropped(self):
        page = {"descendants": [_node("kjh", "хараң"), _node("slr", "göz"), _node("khk", "хар")]}
        self.assertEqual(
            [(code, word) for code, word, _ in ptl._page_descendants(page)],
            [("khk", "хараң"), ("slq", "göz")],
        )

    def test_cache_signature_covers_the_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            dump = Path(tmp) / "trk-pro.jsonl.gz"
            dump.write_bytes(b"")
            with mock.patch.object(ptl, "DUMP", dump):
                before = ptl._dump_signature()
                with mock.patch.dict("engine.fetchers.base.WIKTIONARY_CODE_ALIASES", {"xyz": "kk"}):
                    self.assertNotEqual(ptl._dump_signature(), before)


class TestLexiconIndexCodes(unittest.TestCase):
    def test_cognate_codes_mapped(self):
        record = {"etymology_templates": [
            {"name": "cog", "args": {"1": "kjh", "2": "көс"}},
            {"name": "cog", "args": {"1": "slr", "2": "göz"}},
            {"name": "cog", "args": {"1": "khk", "2": "нүд"}},
        ]}
        langs = [c["lang"] for c in json.loads(_cognates_from_templates(record))]
        self.assertEqual(langs, ["khk", "slq", "mn"])

    def test_salar_internal_derivation_is_not_a_loan(self):
        # slq `öxsirik`: "Derived from Salar öxsirğüsi" — alıntı çıkıyordu.
        record = {"etymology_templates": [{"name": "der", "args": {"1": "slr", "2": "slr", "3": "öxsirğüsi"}}]}
        self.assertEqual(_origin_from_templates(record)[0], "miras")

    def test_family_uses_wiktionary_codes(self):
        self.assertIn("kjh", TURKIC_FAMILY_CODES)
        self.assertIn("slr", TURKIC_FAMILY_CODES)
        self.assertNotIn("khk", TURKIC_FAMILY_CODES)  # Wiktionary'de Halha Moğolcası


def _load_download_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / "download_lexicons.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestNorthernAltai(unittest.TestCase):
    """Kuzey Altayca (``atv``) ayrı dildir; Güney Altaycaya (``alt``) katılmaz."""

    def test_code_is_separate_and_unmapped(self):
        self.assertIn("atv", TURKIC_LANGUAGES_MAP)
        self.assertEqual(lang_code_from_wiktionary("atv"), "atv")
        self.assertEqual(lang_code_from_wiktionary("alt"), "alt")
        self.assertEqual(lang_code_from_wiktionary_header("==Northern Altai=="), "atv")
        self.assertEqual(lang_code_from_wiktionary_header("Southern Altai"), "alt")

    def test_download_script_lists_it(self):
        module = _load_download_script("_download_lexicons_atv")
        self.assertEqual(module.LEXICONS["Northern_Altai"], "atv")
        self.assertEqual(
            module.kaikki_url("Northern_Altai"),
            "https://kaikki.org/dictionary/Northern%20Altai/kaikki.org-dictionary-NorthernAltai.jsonl",
        )

    def test_dump_is_discovered_and_parsed(self):
        from engine.db import lexicon_index

        rec = {"word": "кӧл", "lang_code": "atv", "pos": "noun",
               "etymology_text": "From Proto-Turkic *kȫl.",
               "etymology_templates": [{"name": "inh", "args": {"1": "atv", "2": "trk-pro", "3": "*kȫl"}}],
               "senses": [{"glosses": ["lake"]}]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "atv.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            with mock.patch.object(lexicon_index, "LEXICON_DIR", Path(tmp)):
                self.assertEqual(discover_lexicons(), {"atv": path})
            entry = next(iter_entries(path, "atv"))
        self.assertEqual((entry.lang_code, entry.comparison, entry.gloss), ("atv", "köl", "lake"))
        self.assertEqual((entry.origin, entry.donor_lang), ("miras", "trk-pro"))


class _Response:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self) -> None:
        pass

    def iter_content(self, chunk_size: int):
        yield self._payload


class TestTrEditionSplit(unittest.TestCase):
    def test_split_maps_codes(self):
        module = _load_download_script("_download_lexicons_codes")
        lines = [{"lang_code": "kjh", "word": "хараң"}, {"lang_code": "slr", "word": "göz"},
                 {"lang_code": "khk", "word": "хар"}, {"lang_code": "tr", "word": "göz"}]
        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb") as gz:
            gz.write("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines).encode())
        session = mock.Mock()
        session.get.return_value = _Response(buffer.getvalue())
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(module, "LEXICON_DIR", Path(tmp)):
            counts = module.download_tr_edition(session=session)
            self.assertEqual(counts, {"khk": 1, "slq": 1, "tr": 1})
            directory = Path(tmp) / module.TR_SUBDIR
            self.assertFalse((directory / "mn.jsonl.gz").exists())
            with gzip.open(directory / "khk.jsonl.gz", "rt", encoding="utf-8") as handle:
                self.assertEqual(json.loads(handle.readline())["word"], "хараң")


if __name__ == "__main__":
    unittest.main()
