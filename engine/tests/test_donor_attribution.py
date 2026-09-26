"""
Verici dil etiketi testleri — "alıntı mı?"dan ayrı "kimden?" adımı.

Ölçüldü (``make eval-donor``, WOLD Saha, n=440): tek havuzdaki en yakın
maddenin dili Moğolca alıntıların 79/166'sını Rusça etiketliyordu. Dil başına
en yakın madde + dilin null'ı + Starling ``monget`` doğruluğu 0,652'den
0,714'e çıkardı. Testler üç şeyi korur:

1. Etiket sinyal GÜCÜNE girmez (``monget`` güce katılınca WOLD F düşüyordu).
2. Moğolca adaylar Starling ``monget``ten gelir, kaikki'den değil.
3. Seçim ham mesafeyle değil, dilin kendi null'ına göre yapılır.
"""

from __future__ import annotations

import unittest
from unittest import mock

from engine.db import starling
from engine.nlp import borrowing_detector as bd
from engine.nlp import donor_proximity as dp


class TestMongolicFormParsing(unittest.TestCase):
    def test_references_and_glosses_are_dropped(self):
        self.assertEqual(
            starling._mongolic_forms("dölü (L 272), döl 'flame' [dial.]"), ["dölü", "döl"]
        )

    def test_starling_j_is_aligned_with_sakha_y(self):
        self.assertEqual(starling._mongolic_forms("jeke"), ["yeke"])

    def test_empty_field(self):
        self.assertEqual(starling._mongolic_forms(""), [])

    def test_missing_tables_give_nothing(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        starling.load_monget.cache_clear()
        with TemporaryDirectory() as tmp:
            self.assertEqual(starling.load_monget(Path(tmp)), ())
        starling.load_monget.cache_clear()


class TestAttributionControls(unittest.TestCase):
    def test_every_length_has_enough_controls(self):
        """⚠️ Şans denetiminin kontrolleri 6 harften uzun sorguda yetersizdi;
        etiket null'ı her uzunlukta kurulabilmeli."""
        for length in range(2, 13):
            with self.subTest(length=length):
                controls = dp._attribution_controls(length)
                self.assertGreaterEqual(len(controls), 8)
                self.assertTrue(all(len(c) == length for c in controls))


class _FakeIndex:
    exists = True

    def __init__(self, rows):
        self.rows = rows

    def by_sense(self, sense, languages=None, limit=200, per_language=False):
        return [r for r in self.rows if languages is None or r["lang_code"] in languages]


def _row(lang, word, comparison, gloss="x"):
    return {"lang_code": lang, "word": word, "comparison": comparison, "gloss": gloss}


class TestAttributeDonor(unittest.TestCase):
    def setUp(self):
        dp.reset_cache()
        if dp._pairwise() is None:
            self.skipTest("LingPy kurulu değil")

    def tearDown(self):
        dp.reset_cache()

    def _run(self, rows, monget, nulls, comparison="moğoy", languages=("ru", "mn", "evn")):
        entries = tuple((c, w, "snake", frozenset({"snake"})) for c, w in monget)
        with mock.patch.object(dp, "_index", lambda: _FakeIndex(rows)), \
                mock.patch.object(dp, "_monget_entries", lambda: entries), \
                mock.patch.object(dp, "_null_distance", lambda length, pool: nulls.get(pool[0][:2], 0.0)):
            return dp.attribute_donor(comparison, "snake", languages=list(languages))

    def test_monget_replaces_kaikki_mongolian(self):
        rows = [_row("ru", "змея", "zmeya"), _row("mn", "хорхой", "horhoy")]
        result = self._run(rows, [("moğay", "moɣai")], {})
        self.assertEqual(result.lang_code, "mn")
        self.assertEqual(result.word, "moɣai")
        self.assertEqual(result.source, "starling-monget")

    def test_selection_uses_the_language_null(self):
        """Ham mesafede Rusça daha yakın (0,10 < 0,20), ama rastgele Türkçe
        kelimeler de Rusça havuza 0,15'e düşüyor, Moğolca havuza ise 0,50'ye:
        Rusça eşleşme şanstan pek ayrılmıyor, etiket Moğolca olmalı."""
        rows = [_row("ru", "могай", "mogay")]
        entries = (("moğay", "moɣai", "snake", frozenset({"snake"})),)
        with mock.patch.object(dp, "_index", lambda: _FakeIndex(rows)), \
                mock.patch.object(dp, "_monget_entries", lambda: entries), \
                mock.patch.object(dp, "best_sca",
                                  lambda q, c: (0.1, c[0]) if c[0] == "mogay" else (0.2, c[0])), \
                mock.patch.object(dp, "_null_distance",
                                  lambda length, pool: 0.15 if pool == ("mogay",) else 0.5):
            chosen = dp.attribute_donor("moğoy", "snake", languages=["ru", "mn"])
        self.assertEqual(chosen.lang_code, "mn")
        self.assertEqual([alt[0] for alt in chosen.alternatives], ["ru"])

    def test_far_label_is_marked_uncertain(self):
        rows = [_row("ru", "удав", "udav")]
        result = self._run(rows, [], {}, comparison="moğoy", languages=("ru",))
        self.assertTrue(result.uncertain)
        self.assertIn("verici belirsiz", result.describe())
        self.assertTrue(result.as_dict()["donor_uncertain"])

    def test_no_candidates_is_none(self):
        self.assertIsNone(self._run([], [], {}))


class TestArabicViaPersian(unittest.TestCase):
    """9d: Farsça kazanan aday Farsçadaki Arapça alıntıysa etiket Arapça.

    Ölçüldü (ön kayıt ``data/cache/work/donor9d/PREREG.md``): Türkçe DEV
    etiket doğruluğu 0,484 -> 0,641 (D1, McNemar 10/0, Holm p=0,004).
    """

    def setUp(self):
        dp.reset_cache()
        if dp._pairwise() is None:
            self.skipTest("LingPy kurulu değil")

    def tearDown(self):
        dp.reset_cache()

    def _run(self, rows, comparison, rule="d1", loans=frozenset()):
        with mock.patch.object(dp, "_index", lambda: _FakeIndex(rows)), \
                mock.patch.object(dp, "_monget_entries", lambda: ()), \
                mock.patch.object(dp, "ARABIC_VIA_RULE", rule), \
                mock.patch.object(dp, "persian_arabic_loans", lambda: loans), \
                mock.patch.object(dp, "_null_distance", lambda length, pool: 0.5):
            return dp.attribute_donor(comparison, "invitation", languages=["ar", "fa", "fr"])

    def test_same_script_skeleton_relabels_to_arabic(self):
        rows = [_row("ar", "دعوة", "dava"), _row("fa", "دعوت", "davat")]
        off = self._run(rows, "davet", rule="off")
        self.assertEqual(off.lang_code, "fa")
        on = self._run(rows, "davet")
        self.assertEqual((on.lang_code, on.via, on.word), ("ar", "fa", "دعوة"))
        self.assertIn("Farsça aracılığıyla", on.describe())
        self.assertEqual(on.as_dict()["donor_via"], "fa")
        self.assertNotIn("ar", [alt[0] for alt in on.alternatives])

    def test_etymology_mark_relabels_without_skeleton_match(self):
        rows = [_row("ar", "كلمة", "kalima"), _row("fa", "دعوت", "davat", gloss="invitation")]
        on = self._run(rows, "davet", loans=frozenset({("دعوت", "invitation")}))
        self.assertEqual((on.lang_code, on.via), ("ar", "fa"))

    def test_native_persian_stays_persian(self):
        rows = [_row("ar", "رخيص", "rahis"), _row("fa", "ارزان", "arzan")]
        self.assertEqual(self._run(rows, "arzan").lang_code, "fa")

    def test_d1_leaves_non_persian_winner(self):
        rows = [_row("ar", "دعوة", "dava"), _row("fr", "davet", "davet")]
        chosen = self._run(rows, "davet")
        self.assertEqual((chosen.lang_code, chosen.via), ("fr", ""))

    def test_skeletons(self):
        self.assertEqual(dp.script_skeleton("عَظَمَة"), dp.script_skeleton("عظمت"))
        self.assertIn(dp.consonant_skeleton("gadab"), dp._query_skeletons("gazap"))
        self.assertIn(dp.consonant_skeleton("dava"), dp._query_skeletons("davet"))


class TestFrenchVia(unittest.TestCase):
    """9e F2: Farsça/Ermenice/Yunanca kazanan biçim aslında o dilin Fransızca
    alıntısıysa (pozitron, metro) etiket Fransızca, ``via`` = kazanan dil."""

    def setUp(self):
        dp.reset_cache()
        if dp._pairwise() is None:
            self.skipTest("LingPy kurulu değil")

    def tearDown(self):
        dp.reset_cache()

    def _run(self, rows, comparison, rule="f2", loans=frozenset()):
        with mock.patch.object(dp, "_index", lambda: _FakeIndex(rows)), \
                mock.patch.object(dp, "_monget_entries", lambda: ()), \
                mock.patch.object(dp, "FRENCH_RULE", rule), \
                mock.patch.object(dp, "dump_loans", lambda lang, phrase: loans), \
                mock.patch.object(dp, "_null_distance",
                                  lambda length, pool: 0.5 if "pozitron" in pool else 0.0):
            return dp.attribute_donor(comparison, "positron", languages=["ar", "fa", "hy", "el", "fr", "it"])

    def test_near_french_form_relabels(self):
        rows = [_row("fa", "پوزیترون", "pozitron"), _row("fr", "positron", "positron")]
        self.assertEqual(self._run(rows, "pozitron", rule="f1").lang_code, "fa")
        on = self._run(rows, "pozitron")
        self.assertEqual((on.lang_code, on.via, on.word), ("fr", "fa", "positron"))
        self.assertIn("aracılığıyla", on.describe())

    def test_etymology_mark_relabels(self):
        rows = [_row("hy", "մետրո", "pozitron", gloss="positron"), _row("fr", "électron", "elektron")]
        on = self._run(rows, "pozitron", loans=frozenset({("մետրո", "positron")}))
        self.assertEqual((on.lang_code, on.via), ("fr", "hy"))

    def test_unrelated_winner_stays(self):
        rows = [_row("fa", "پوزیترون", "pozitron"), _row("fr", "gare", "gar")]
        self.assertEqual(self._run(rows, "pozitron").lang_code, "fa")

    def test_italian_winner_is_not_touched(self):
        rows = [_row("it", "positrone", "pozitron"), _row("fr", "gare", "gar")]
        on = self._run(rows, "pozitron")
        self.assertEqual((on.lang_code, on.via), ("it", ""))


class TestWesternFrenchPrior(unittest.TestCase):
    """9f: Batı alıntısında Fransızca önceliği — H1 sonek koşutluğu, H2 yakın
    beraberlik (İtalyanca kazanan, Fransızca ≤ ε daha uzak)."""

    def setUp(self):
        dp.reset_cache()
        if dp._pairwise() is None:
            self.skipTest("LingPy kurulu değil")

    def tearDown(self):
        dp.reset_cache()

    def _run(self, rows, comparison, rule):
        with mock.patch.object(dp, "_index", lambda: _FakeIndex(rows)), \
                mock.patch.object(dp, "_monget_entries", lambda: ()), \
                mock.patch.object(dp, "WESTERN_RULE", rule), \
                mock.patch.object(dp, "dump_loans", lambda lang, phrase: frozenset()), \
                mock.patch.object(dp, "_null_distance",
                                  lambda length, pool: 0.1 if "posa" in pool else 0.0):
            return dp.attribute_donor(comparison, "x", languages=["ar", "fa", "hy", "el", "fr", "it"])

    def test_suffix_pair_relabels(self):
        rows = [_row("it", "organizzazione", "organizzazione"), _row("fr", "organisation", "organisation")]
        self.assertEqual(self._run(rows, "organizasyon", "off").lang_code, "it")
        on = self._run(rows, "organizasyon", "h1")
        self.assertEqual((on.lang_code, on.word, on.via), ("fr", "organisation", ""))

    def test_suffix_needs_french_counterpart(self):
        rows = [_row("it", "organizzazione", "organizzazione"), _row("fr", "organiser", "organiser")]
        self.assertEqual(self._run(rows, "organizasyon", "h1").lang_code, "it")

    def test_near_tie_prefers_french(self):
        rows = [_row("it", "posa", "posa"), _row("fr", "pose", "pose")]
        self.assertEqual(self._run(rows, "poz", "off").lang_code, "it")
        self.assertEqual(self._run(rows, "poz", "h2").lang_code, "fr")

    def test_clear_italian_stays(self):
        rows = [_row("it", "senato", "senato"), _row("fr", "gare", "gar")]
        self.assertEqual(self._run(rows, "senato", "h12").lang_code, "it")


class TestSignalStrengthIsUntouched(unittest.TestCase):
    """⚠️ Etiket güce girerse WOLD "alıntı mı?" F'si değişir."""

    def test_attribution_changes_label_not_strength(self):
        match = dp.DonorMatch("ru", "змея", "zmeya", "snake", 0.2, True, 0.0)
        label = dp.DonorAttribution("mn", "moɣai", "moğay", "snake", 0.05, 0.6, "starling-monget")
        with mock.patch.object(bd, "nearest_donor", lambda *a, **k: match):
            with mock.patch.object(bd, "attribute_donor", lambda *a, **k: None):
                plain = bd.BorrowingDetector._donor_signal("moɣoy", "snake", ["ru", "mn"])
            with mock.patch.object(bd, "attribute_donor", lambda *a, **k: label):
                labelled = bd.BorrowingDetector._donor_signal("moɣoy", "snake", ["ru", "mn"])
        self.assertEqual(plain.strength, labelled.strength)
        self.assertEqual(plain.fired, labelled.fired)
        self.assertEqual(labelled.evidence["donor_lang"], "ru")  # en yakın madde aynı kalır
        self.assertEqual(labelled.evidence["attributed_lang"], "mn")
        self.assertIn("verici etiketi", labelled.explanation)

    def test_no_label_when_signal_does_not_fire(self):
        far = dp.DonorMatch("ru", "змея", "zmeya", "snake", 0.9, True, None)
        called = mock.Mock()
        with mock.patch.object(bd, "nearest_donor", lambda *a, **k: far), \
                mock.patch.object(bd, "attribute_donor", called):
            signal = bd.BorrowingDetector._donor_signal("moɣoy", "snake", ["ru"])
        self.assertFalse(signal.fired)
        called.assert_not_called()


if __name__ == "__main__":
    unittest.main()


class TestLabelOnlyPools9g(unittest.TestCase):
    """9g: eski dil havuzları ve çekim süzgeci yalnız ETİKET adımında, varsayılan kapalı."""

    def _index(self, tmp, rows):
        import gzip
        import json
        from pathlib import Path

        from engine.db.donor_index import DonorIndex

        sources = {}
        for lang, items in rows.items():
            path = Path(tmp) / f"{lang}.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                for word, gloss in items:
                    handle.write(json.dumps({"word": word, "senses": [{"glosses": [gloss]}]}) + "\n")
            sources[lang] = path
        index = DonorIndex(Path(tmp) / "d.db")
        index.build(sources=sources)
        return index

    def test_defaults_off(self):
        self.assertFalse(dp.OLD_DONOR_LABELS)
        self.assertFalse(dp.LABEL_FORM_FILTER)

    def test_form_of_rows_skipped_only_for_listed_languages(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            index = self._index(tmp, {
                "el": [("kalos", "plural of kala"), ("kala", "good")],
                "fr": [("bonne", "feminine singular of bon"), ("bon", "good")],
            })
            words = lambda rows: sorted(r["word"] for r in rows)  # noqa: E731
            self.assertEqual(words(index.by_sense("plural singular good", clean=False)), ["bon", "bonne", "kala", "kalos"])
            skipped = index.by_sense("plural singular good", clean=False, skip_form_of=frozenset({"el"}))
            self.assertEqual(words(skipped), ["bon", "bonne", "kala"])

    def test_old_pool_maps_to_family_code(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            main = self._index(tmp + "", {"fa": [("zzzzz", "lamp")]})
            from pathlib import Path

            sub = Path(tmp) / "label"
            sub.mkdir()
            label = self._index(sub, {"grc": [("lampas", "lamp, torch")]})
            with mock.patch.object(dp, "_index", return_value=main), \
                    mock.patch.object(dp, "_label_index", return_value=label), \
                    mock.patch.object(dp, "OLD_DONOR_LABELS", True):
                result = dp.attribute_donor("lamba", "lamp", languages=["el", "fa"])
            self.assertIsNotNone(result)
            self.assertEqual(result.lang_code, "el")
            self.assertEqual(result.source, "kaikki-grc")
            self.assertIn("Eski Yunanca", result.describe())


class TestItalianLabels9j(unittest.TestCase):
    """9j: İtalyanca imla (I1), Venedikçe/Cenevizce havuzu (I2), eski dil SCA sınırı (G1')."""

    _index = TestLabelOnlyPools9g._index

    def test_italian_phonetic(self):
        cases = {"scialuppa": "şalupa", "ceppo": "çepo", "giranta": "ciranta", "chiglia": "kilya",
                 "organizzazione": "organizazyon", "bocciarda": "boçarda", "bagno": "banyo", "timon": "timon"}
        for italian, expected in cases.items():
            self.assertEqual(dp.italian_phonetic(italian), expected)

    def test_venetan_rows_join_italian_group(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            main = self._index(tmp, {"fa": [("zzzzzz", "rudder")], "it": [("qqqqq", "rudder")]})
            sub = Path(tmp) / "label"
            sub.mkdir()
            label = self._index(sub, {"vec": [("timon", "rudder")]})
            with mock.patch.object(dp, "_index", return_value=main), \
                    mock.patch.object(dp, "_label_index", return_value=label), \
                    mock.patch.object(dp, "VENETAN_LABELS", True):
                result = dp.attribute_donor("timon", "rudder", languages=["fa", "it"])
            self.assertEqual(result.lang_code, "it")
            self.assertEqual(result.source, "kaikki-vec")
            self.assertIn("Venedikçe", result.describe())

    def test_old_pool_capped_by_distance(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            main = self._index(tmp, {"fa": [("lamba", "lamp")]})
            sub = Path(tmp) / "label"
            sub.mkdir()
            label = self._index(sub, {"grc": [("xyzqwv", "lamp")]})
            with mock.patch.object(dp, "_index", return_value=main), \
                    mock.patch.object(dp, "_label_index", return_value=label), \
                    mock.patch.object(dp, "OLD_DONOR_LABELS", True), \
                    mock.patch.object(dp, "OLD_DONOR_MAX", 0.35):
                result = dp.attribute_donor("lamba", "lamp", languages=["el", "fa"])
            self.assertEqual(result.lang_code, "fa")


class TestSenseBridge9l(unittest.TestCase):
    """9l: Türkçe anlam köprüsü (S1), ayrı dil havuz sorgusu (S2), tam eşleşmede ham mesafe (S3)."""

    _index = TestLabelOnlyPools9g._index

    def test_defaults(self):
        self.assertTrue(dp.SENSE_BRIDGE)
        self.assertEqual(dp.EXTRA_POOL_LANGS, ())
        self.assertIsNone(dp.EXACT_MATCH_EPS)

    def test_is_english_sense(self):
        from engine.db import sense_bridge as sb

        with mock.patch.object(sb, "_known_english", side_effect=lambda t: sum(w in {"lamp"} for w in t)):
            self.assertTrue(sb.is_english_sense("a kind of boat used for fishing"))
            self.assertTrue(sb.is_english_sense("lamp"))
            self.assertFalse(sb.is_english_sense("Bir tür büyük balıkçı teknesi"))
            self.assertFalse(sb.is_english_sense("kasaphane"))
            self.assertFalse(sb.is_english_sense(""))

    def test_bridge_only_for_non_english_sense(self):
        from engine.db import sense_bridge as sb

        with mock.patch.object(dp, "SENSE_BRIDGE", True), \
                mock.patch.object(sb, "english_sense", return_value="band"), \
                mock.patch.object(sb, "ottoman_sense", return_value=""), \
                mock.patch.object(sb, "is_english_sense", side_effect=lambda s: s == "music band"):
            self.assertEqual(dp.bridged_sense("bando", "Bir müzik topluluğu"), "band")
            self.assertEqual(dp.bridged_sense("bando", "music band"), "music band")
            self.assertEqual(dp.bridged_sense("bando", ""), "")
            self.assertEqual(dp.bridged_sense("bando", "дом"), "дом")
        with mock.patch.object(dp, "SENSE_BRIDGE", False):
            self.assertEqual(dp.bridged_sense("bando", "Bir müzik topluluğu"), "Bir müzik topluluğu")

    def test_extra_pool_query(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            index = self._index(tmp, {"it": [("balo", "ball")], "ar": [("xyz", "ball")]})
            calls = []
            real = index.by_sense

            def spy(sense, **kw):
                calls.append(kw.get("languages"))
                return real(sense, **kw)

            with mock.patch.object(dp, "_index", return_value=index), mock.patch.object(index, "by_sense", spy), \
                    mock.patch.object(dp, "EXTRA_POOL_LANGS", ("it",)):
                result = dp.attribute_donor("balo", "ball", languages=["ar", "it", "fr"])
            self.assertIn(["it"], calls)
            self.assertEqual(result.lang_code, "it")

    def test_exact_match_beats_null(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            index = self._index(tmp, {"it": [("arma", "coat of arms")], "ar": [("arima", "coat of arms")]})
            with mock.patch.object(dp, "_index", return_value=index), \
                    mock.patch.object(dp, "_null_distance", side_effect=lambda n, pool: 0.9 if "arima" in pool else 0.0), \
                    mock.patch.object(dp, "EXACT_MATCH_EPS", 0.0):
                result = dp.attribute_donor("arma", "coat of arms", languages=["ar", "it"])
            self.assertEqual(result.lang_code, "it")


class TestFrenchArabicGuard9m(unittest.TestCase):
    """9m: R1 (H1'den belirsiz sonek çıkarma), R2 (Arapça iskelet koruması)."""

    def setUp(self):
        dp.reset_cache()
        if dp._pairwise() is None:
            self.skipTest("LingPy kurulu değil")

    def tearDown(self):
        dp.reset_cache()

    def test_defaults_off(self):
        self.assertEqual(dp.WESTERN_SUFFIX_DROP, ())
        self.assertFalse(dp.FRENCH_ARABIC_GUARD)

    def test_suffix_drop(self):
        self.assertEqual(dp._french_suffix_pair("dinamik"), ("ikue",))
        with mock.patch.object(dp, "WESTERN_SUFFIX_DROP", ("ik",)):
            self.assertEqual(dp._french_suffix_pair("dinamik"), ())
            self.assertEqual(dp._french_suffix_pair("aktör"), ("eur",))

    def _run(self, rows, comparison, guard, western=frozenset()):
        with mock.patch.object(dp, "_index", lambda: _FakeIndex(rows)), \
                mock.patch.object(dp, "_monget_entries", lambda: ()), \
                mock.patch.object(dp, "FRENCH_RULE", "f2"), \
                mock.patch.object(dp, "WESTERN_RULE", "h1"), \
                mock.patch.object(dp, "FRENCH_ARABIC_GUARD", guard), \
                mock.patch.object(dp, "FRENCH_ARABIC_GUARD_SKIP_WESTERN", True), \
                mock.patch.object(dp, "dump_loans",
                                  lambda lang, phrase: western if lang == "ar" else frozenset()), \
                mock.patch.object(dp, "_null_distance", lambda length, pool: 0.0):
            return dp.attribute_donor(comparison, "x", languages=["ar", "fa", "fr", "it"])

    def test_skeleton_match_blocks_suffix_rule(self):
        rows = [_row("it", "tattico", "tatiko"), _row("fr", "tactique", "taktikue"),
                _row("ar", "تطبيق", "tatbik")]
        self.assertEqual(self._run(rows, "tatbik", False).lang_code, "fr")
        self.assertEqual(self._run(rows, "tatbik", True).lang_code, "ar")

    def test_western_arabic_candidate_does_not_guard(self):
        rows = [_row("it", "tattico", "tatiko"), _row("fr", "tactique", "taktikue"),
                _row("ar", "تطبيق", "tatbik", gloss="x")]
        western = frozenset({("تطبيق", "x")})
        self.assertEqual(self._run(rows, "tatbik", True, western).lang_code, "fr")

    def test_g2_pool_winner_yields_to_arabic_skeleton(self):
        class _SharedWithoutFrench(_FakeIndex):
            def by_sense(self, sense, languages=None, limit=200, per_language=False):
                if languages == ["fr"]:
                    return [r for r in self.rows if r["lang_code"] == "fr"]
                return [r for r in self.rows if r["lang_code"] != "fr"]

        rows = [_row("ar", "معذرة", "madira"), _row("fr", "mazure", "mazure")]
        for guard, expected in ((False, "fr"), (True, "ar")):
            with mock.patch.object(dp, "_index", lambda: _SharedWithoutFrench(rows)), \
                    mock.patch.object(dp, "_monget_entries", lambda: ()), \
                    mock.patch.object(dp, "FRENCH_RULE", "g2"), \
                    mock.patch.object(dp, "FRENCH_ARABIC_GUARD", guard), \
                    mock.patch.object(dp, "dump_loans", lambda lang, phrase: frozenset()), \
                    mock.patch.object(dp, "_null_distance", lambda length, pool: 0.0):
                self.assertEqual(dp.attribute_donor("mazur", "x", languages=["ar", "fr"]).lang_code, expected)
