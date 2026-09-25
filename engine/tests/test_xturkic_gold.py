"""
Türk dilleri arası alıntı altını (plan X1) — etiket kuralı, bölme, mühür
ve kör indeks testleri.

Etiket tanımı plandadır ve bağlayıcıdır; bu testler tanımın sessizce
kaymasını engeller (ör. ilk halka Türk vericiliyse "alıntı" dememek).
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from engine.config import PROJECT_ROOT
from engine.db.lexicon_index import SCHEMA
from engine.evaluation.xturkic_gold import (
    SPLIT_RATIOS,
    XTURKIC_DIR,
    assign_split,
    donor_macro,
    etymon_key,
    label_record,
    load_split,
)


def rec(*templates, text="", word="сөз"):
    return {
        "word": word,
        "etymology_templates": [{"name": n, "args": {"1": "kk", "2": d, "3": f}} for n, d, f in templates],
        "etymology_text": text,
    }


class LabelRuleTests(unittest.TestCase):
    def test_direct_foreign_borrowing_is_loan(self):
        lab = label_record(rec(("bor", "ru", "маши́на"), ("der", "fr", "machine")), "kk")
        self.assertEqual(lab.label, "alıntı")
        self.assertEqual(lab.donor_macro, "ru")

    def test_classical_persian_maps_to_fa(self):
        self.assertEqual(donor_macro("fa-cls"), "fa")
        self.assertEqual(donor_macro("xng"), "mn")
        self.assertEqual(donor_macro("hy"), "diğer")

    def test_inherited_all_turkic(self):
        lab = label_record(rec(("inh", "trk-pro", "*taĺ")), "kk")
        self.assertEqual(lab.label, "miras")
        self.assertTrue(lab.etymon.startswith("trk:"))

    def test_turkic_internal_borrowing_excluded(self):
        lab = label_record(rec(("bor", "tt", "сүз")), "kk")
        self.assertIsNone(lab.label)
        self.assertEqual(lab.reason, "türk_içi_alıntı")

    def test_loan_with_turkic_inheritance_is_conflict(self):
        lab = label_record(rec(("bor", "xng", "x"), ("inh", "trk-pro", "*y")), "kk")
        self.assertIsNone(lab.label)
        self.assertEqual(lab.reason, "çelişen_şablon")

    def test_mediated_loan_excluded(self):
        # uz ← Çağatayca (miras) ← Arapça: ilk halka miras, plan tanımına uymaz.
        lab = label_record(rec(("inh", "chg", "x"), ("bor", "ar", "y")), "uz")
        self.assertIsNone(lab.label)

    def test_der_only_excluded(self):
        lab = label_record(rec(("der", "ar", "x")), "kk")
        self.assertEqual(lab.reason, "yalnız_der")

    def test_hedged_text_excluded(self):
        lab = label_record(rec(("bor", "fa", "x"), text="Possibly from Persian x."), "kk")
        self.assertEqual(lab.reason, "çekince_metni")

    def test_calque_excluded(self):
        lab = label_record(rec(("cal", "ru", "x")), "kk")
        self.assertEqual(lab.reason, "öyküntü")

    def test_tuvan_only_mongolic_layer(self):
        self.assertEqual(label_record(rec(("bor", "ru", "x")), "tyv").reason, "tyv_moğolca_dışı")
        self.assertEqual(label_record(rec(("bor", "cmg", "x")), "tyv").label, "alıntı")


class SplitTests(unittest.TestCase):
    def test_split_deterministic_and_known(self):
        self.assertEqual(assign_split("fa:ارزان"), assign_split("fa:ارزان"))
        self.assertIn(assign_split("ru:x"), SPLIT_RATIOS)
        self.assertAlmostEqual(sum(SPLIT_RATIOS.values()), 1.0)

    def test_etymon_key_ignores_vocalisation(self):
        # Aynı Arapça etimon, harekeli ve harekesiz: aynı grup (diller arası sızıntı olmasın).
        self.assertEqual(etymon_key("ar", "كِتَاب"), etymon_key("ar", "كتاب"))

    def test_test_split_cannot_be_loaded(self):
        with self.assertRaises(PermissionError):
            load_split("test")

    @unittest.skipUnless((XTURKIC_DIR / "SEAL.json").exists(), "altın kurulmamış")
    def test_seal_matches_files(self):
        from engine.evaluation.xturkic_gold import verify_seal

        self.assertTrue(all(verify_seal().values()))


class BlindIndexTests(unittest.TestCase):
    def test_env_var_selects_index(self):
        out = subprocess.run(
            [sys.executable, "-c", "from engine.db.lexicon_index import INDEX_PATH; print(INDEX_PATH)"],
            cwd=PROJECT_ROOT, capture_output=True, text=True,
            env={**os.environ, "ETY_LEXICON_INDEX": "/tmp/kör.db"}, check=True,
        )
        self.assertEqual(out.stdout.strip(), "/tmp/kör.db")

    def test_blind_copy_nulls_origin_and_keeps_source(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from build_blind_index import build

        with TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "index.db", Path(tmp) / "blind.db"
            with sqlite3.connect(src) as c:
                c.executescript(SCHEMA)
                c.execute(
                    "INSERT INTO entries (lang_code, word, comparison, gloss, etymology, origin, donor_lang, donor_form)"
                    " VALUES ('kk', 'кітап', 'kitap', 'book', 'From Arabic', 'alıntı', 'ar', 'كتاب')"
                )
                c.execute("INSERT INTO entries_fts(entries_fts) VALUES('rebuild')")
            report = build(src, out)
            self.assertEqual(report["origin_rows_before"], 1)
            with sqlite3.connect(out) as c:
                row = c.execute("SELECT word, gloss, origin, donor_lang, etymology FROM entries").fetchone()
                info = dict(c.execute("SELECT key, value FROM build_info").fetchall())
            self.assertEqual(row, ("кітап", "book", None, None, None))
            self.assertEqual(info["blind"], "1")
            with sqlite3.connect(src) as c:
                self.assertEqual(c.execute("SELECT origin FROM entries").fetchone()[0], "alıntı")
            json.dumps(report)


class R3Tests(unittest.TestCase):
    """X5: R3 bölümü — güçlendirilmiş melez süzgeci ve mühür/sızıntı."""

    def test_hybrid_filter_foreign_stem_and_text(self):
        from engine.evaluation.xturkic_gold import hybrid_reason

        # ug نۇقسانسىز: miras (Çağatay) ama yüzey çözümlemesi alıntı kök + -siz
        record = {
            "word": "نۇقسانسىز",
            "etymology_templates": [
                {"name": "inh", "args": {"1": "ug", "2": "chg", "3": "نقصانسیز"}},
                {"name": "surf", "args": {"1": "ug", "2": "نۇقسان<t:defect>", "3": "ـسىز<t:-less>"}},
            ],
            "etymology_text": "Inherited from Chagatai نقصانسیز. By surface analysis, نۇقسان + ـسىز.",
        }
        self.assertEqual(hybrid_reason(record, "miras", {"نۇقسان"}), "melez_kök")
        self.assertEqual(hybrid_reason(record, "miras", set()), "")
        chagatai = rec(("inh", "chg", "حرارت"), text="Inherited from Chagatai حرارت, borrowed from Classical Persian.")
        self.assertEqual(hybrid_reason(chagatai, "miras", set()), "miras_metninde_yabancı_dil")
        cognate = rec(("inh", "trk-pro", "*kan"), text="From Proto-Turkic *kan. Cognate with Russian x, Mongolian y.")
        self.assertEqual(hybrid_reason(cognate, "miras", set()), "")
        hedged = rec(("bor", "ltc", "擺子"), text="Borrowed from Middle Chinese 擺子.\nOr from *bezgäk.")
        self.assertEqual(hybrid_reason(hedged, "alıntı", set()), "çekince_or")

    def test_r3_seal_and_no_etymon_leak(self):
        from engine.evaluation.xturkic_gold import R3_SEAL, file_sha256

        seal_path = XTURKIC_DIR / R3_SEAL
        if not seal_path.exists():
            self.skipTest("R3 kurulmamış")
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        self.assertEqual(file_sha256(XTURKIC_DIR / seal["file"]), seal["checksum"])
        x1 = json.loads((XTURKIC_DIR / "SEAL.json").read_text(encoding="utf-8"))["checksums"]
        self.assertEqual(seal["x1_seal_checksums"], x1)
        # Etimon sızıntısı: R3 etimonları tune/R1/R2'de yok (test okunmaz; kurulumda denetlendi).
        r3 = {json.loads(line)["etymon"] for line in (XTURKIC_DIR / seal["file"]).read_text(encoding="utf-8").splitlines()}
        for split in ("tune", "r1", "r2"):
            self.assertFalse(r3 & {i["etymon"] for i in load_split(split)}, split)


if __name__ == "__main__":
    unittest.main()
