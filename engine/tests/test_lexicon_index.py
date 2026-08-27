"""
Yerel sözlük indeksi testleri.

İndeks iki soruya cevap verir: "bu biçim şu dilde var mı?" (öngörü testi) ve
"bu kelime hangi dilden, nasıl gelmiş?" (alıntı zinciri). Bu testler her
ikisinin de doğru çalıştığını ve **arama indeksi ile karar katmanının
karışmadığını** korur.
"""

from __future__ import annotations

import gzip
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from engine.db.lexicon_index import (
    SCHEMA,
    TURKIC_FAMILY_CODES,
    LexiconEntry,
    LexiconIndex,
    donor_from_text,
    iter_entries,
    parse_tree_template,
)


def record(word, templates=None, etymology="", gloss="anlam", lang="tr"):
    return {
        "word": word,
        "lang_code": lang,
        "pos": "noun",
        "etymology_text": etymology,
        "etymology_templates": templates or [],
        "senses": [{"glosses": [gloss]}],
        "sounds": [{"ipa": f"/{word}/"}],
    }


def write_dump(directory: Path, name: str, records: list[dict]) -> Path:
    path = directory / f"{name}.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for item in records:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return path


class TestTreeTemplateParsing(unittest.TestCase):
    """``etymon``/``ety`` şablonları bambaşka bir yapı kullanır."""

    def test_simple_tree_template(self):
        parsed = parse_tree_template(
            {"name": "etymon", "args": {"1": "tr", "2": ":inh", "3": "ota:كتاب"}}
        )
        self.assertEqual(parsed, [("inh", "ota", "كتاب")])

    def test_nested_chain_is_extracted(self):
        parsed = parse_tree_template(
            {"args": {"2": ":inh", "3": "ota:صلا\n<ety:der<ar:صَلاَة>>"}}
        )
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[1][1], "ar")

    def test_trailing_id_markup_is_stripped(self):
        parsed = parse_tree_template(
            {"args": {"2": ":bor", "3": "fr:robot<id:Q11012>"}}
        )
        self.assertEqual(parsed, [("bor", "fr", "robot")])

    def test_empty_args(self):
        self.assertEqual(parse_tree_template({"args": {}}), [])


class TestDonorFromText(unittest.TestCase):
    def test_finds_arabic(self):
        code, _ = donor_from_text("Etymology tree\nArabic كِتَاب (kitāb)bor.")
        self.assertEqual(code, "ar")

    def test_prefers_the_longer_name(self):
        """"Classical Persian" "Persian"dan önce denenmeli."""
        code, _ = donor_from_text("From Classical Persian دِیوَار")
        self.assertEqual(code, "fa-cls")

    def test_returns_nothing_when_absent(self):
        self.assertEqual(donor_from_text("Inherited from Old Turkic"), ("", ""))

    def test_empty(self):
        self.assertEqual(donor_from_text(""), ("", ""))


class TestOriginDetection(unittest.TestCase):
    """Zincirin İLK halkasına bakmak, alıntıları miras sayar."""

    def _origin(self, rec):
        return next(iter_entries(self._dump(rec), "tr")).origin

    def _dump(self, rec):
        self._tmp = TemporaryDirectory()
        return write_dump(Path(self._tmp.name), "tr", [rec])

    def tearDown(self):
        if hasattr(self, "_tmp"):
            self._tmp.cleanup()

    def test_direct_borrowing(self):
        self.assertEqual(
            self._origin(record("robot", [{"name": "bor", "args": {"2": "fr", "3": "robot"}}])),
            "alıntı",
        )

    def test_pure_inheritance(self):
        self.assertEqual(
            self._origin(record("on", [
                {"name": "inh", "args": {"2": "ota", "3": "اون"}},
                {"name": "inh", "args": {"2": "trk-pro", "3": "*ōn"}},
            ])),
            "miras",
        )

    def test_inheritance_that_leaves_the_family_is_a_borrowing(self):
        """``sabun``: ilk halka Osmanlıcadan MİRAS, ikinci halka Arapçaya çıkıyor.

        İlk halkaya bakan uygulama bunu "miras" sayıyordu; Türkçedeki Arapça
        alıntı 48 görünüyordu, gerçek sayı 3.958.
        """
        self.assertEqual(
            self._origin(record("sabun", [
                {"name": "inh", "args": {"2": "ota", "3": "صابون"}},
                {"name": "der", "args": {"2": "ar", "3": "صَابُون"}},
            ])),
            "alıntı",
        )

    def test_etymology_text_is_a_fallback(self):
        """``kitap``ın Arapça halkası yalnız serbest metinde."""
        self.assertEqual(
            self._origin(record(
                "kitap",
                [{"name": "etymon", "args": {"2": ":inh", "3": "ota:كتاب"}}],
                etymology="Etymology tree\nArabic كِتَاب (kitāb)bor.\nOttoman Turkish كتاب",
            )),
            "alıntı",
        )

    def test_no_templates_no_origin(self):
        self.assertIsNone(self._origin(record("kar")))

    def test_turkic_family_covers_the_intermediaries(self):
        for code in ("ota", "otk", "trk-pro", "cv", "sah"):
            self.assertIn(code, TURKIC_FAMILY_CODES, code)


class TestIndexOperations(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        write_dump(self.dir, "tr", [
            record("göz", gloss="eye"),
            record("kitap", [{"name": "bor", "args": {"2": "ar", "3": "كتاب"}}], gloss="book"),
            record("kar", gloss="snow"),
        ])
        write_dump(self.dir, "kk", [record("köz", gloss="eye", lang="kk")])
        self.index = LexiconIndex(self.dir / "index.db")
        self.index.build(sources={"tr": self.dir / "tr.jsonl.gz", "kk": self.dir / "kk.jsonl.gz"})

    def tearDown(self):
        self._tmp.cleanup()

    def test_exact_lookup(self):
        rows = self.index.lookup("göz")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["gloss"], "eye")

    def test_lookup_can_be_language_scoped(self):
        self.assertEqual(len(self.index.lookup("köz", languages=["tr"])), 0)
        self.assertEqual(len(self.index.lookup("köz", languages=["kk"])), 1)

    def test_fuzzy_lookup_finds_near_misses(self):
        """İleri tahminin %75,5'i bir harf içinde — bulanık arama şart."""
        rows = self.index.fuzzy_lookup("kör", max_distance=1)
        self.assertTrue(any(r["word"] == "göz" or r["edit_distance"] <= 1 for r in rows))

    def test_fuzzy_lookup_reports_distance(self):
        rows = self.index.fuzzy_lookup("göz", max_distance=1)
        self.assertEqual(rows[0]["edit_distance"], 0)

    def test_full_text_search(self):
        self.assertTrue(self.index.search("eye"))

    def test_borrowings_are_queryable(self):
        rows = self.index.borrowings("tr")
        self.assertEqual([r["word"] for r in rows], ["kitap"])

    def test_donor_counts(self):
        self.assertEqual(self.index.donor_counts("tr"), [("ar", 1)])

    def test_stats_reports_totals(self):
        stats = self.index.stats()
        self.assertTrue(stats["exists"])
        self.assertEqual(stats["total_entries"], 4)
        self.assertEqual(stats["borrowed"], 1)

    def test_empty_lookup_is_safe(self):
        self.assertEqual(self.index.lookup(""), [])
        self.assertEqual(self.index.fuzzy_lookup(""), [])

    def test_missing_index_reports_absence(self):
        self.assertFalse(LexiconIndex(self.dir / "yok.db").stats()["exists"])

    def test_build_without_sources_raises(self):
        with self.assertRaises(FileNotFoundError):
            LexiconIndex(self.dir / "x.db").build(sources={})


class TestEntryParsing(unittest.TestCase):
    def test_gloss_and_ipa_are_extracted(self):
        with TemporaryDirectory() as tmp:
            path = write_dump(Path(tmp), "tr", [record("göz", gloss="eye")])
            entry = next(iter_entries(path, "tr"))
            self.assertEqual(entry.gloss, "eye")
            self.assertEqual(entry.ipa, "/göz/")

    def test_arabic_script_is_transliterated(self):
        """Arap kefi tanınmazsa Uygurca kayıtların %97'si düşüyordu."""
        with TemporaryDirectory() as tmp:
            path = write_dump(Path(tmp), "ug", [record("كۆز", lang="ug")])
            entry = next(iter_entries(path, "ug"))
            self.assertEqual(entry.comparison, "köz")

    def test_malformed_lines_are_skipped(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.jsonl"
            path.write_text('{"broken\n{"word": "göz", "lang_code": "tr"}\n', encoding="utf-8")
            self.assertEqual(len(list(iter_entries(path, "tr"))), 1)

    def test_row_shape_matches_schema(self):
        """Satır uzunluğu INSERT'teki sütun sayısıyla birebir olmalı.

        Şemaya sütun eklenip ``as_row`` güncellenmezse veya tersi olursa,
        indeks kurulumu sessizce yanlış sütuna yazar.
        """
        entry = LexiconEntry(lang_code="tr", word="a", comparison="a")
        columns = [
            line.split()[0]
            for line in SCHEMA.split("CREATE TABLE IF NOT EXISTS entries (")[1]
            .split(");")[0]
            .strip()
            .splitlines()
            if line.strip() and not line.strip().startswith("--")
        ]
        # ``id`` otomatik atanır, satıra girmez.
        self.assertEqual(len(entry.as_row()), len(columns) - 1)


if __name__ == "__main__":
    unittest.main()
