"""Anlam süzgeci (X3) — e5 yüklemeden sınanabilen kısımlar."""

import os
import unittest
from unittest import mock

from engine.nlp import sense_match


class TestSpec(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(sense_match.parse_spec("s1:0.86").tau, 0.86)
        self.assertEqual(sense_match.parse_spec("s2:5").k, 5)
        self.assertTrue(sense_match.parse_spec("s3:strict").strict)
        self.assertFalse(sense_match.parse_spec("s3:fallback").strict)
        with self.assertRaises(ValueError):
            sense_match.parse_spec("s9:1")

    def test_flag_default_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ETY_DONOR_SENSE_FILTER", None)
            self.assertIsNone(sense_match.filter_from_env())
        with mock.patch.dict(os.environ, {"ETY_DONOR_SENSE_FILTER": "0"}):
            self.assertIsNone(sense_match.filter_from_env())

    def test_flag_on_follows_accepted_spec(self):
        with mock.patch.dict(os.environ, {"ETY_DONOR_SENSE_FILTER": "1"}):
            sense_match._from_spec.cache_clear()
            expected = sense_match.ACCEPTED_SPEC
            got = sense_match.filter_from_env()
            if expected is None:
                self.assertIsNone(got)
            else:
                self.assertEqual(got.spec, sense_match.parse_spec(expected).spec)
        with mock.patch.dict(os.environ, {"ETY_DONOR_SENSE_FILTER": "s3:strict"}):
            self.assertEqual(sense_match.filter_from_env().spec, "s3:strict")


class TestConcepts(unittest.TestCase):
    def test_parts_and_concepts(self):
        if not sense_match.concept_table():
            self.skipTest("yerel CLDF yok")
        self.assertTrue(sense_match.concepts_of("army") & sense_match.concepts_of("soldier; army, force"))
        self.assertFalse(sense_match.concepts_of("to make beautiful"))
        # çekim açıklaması: "X of Y:" sonrası alınır
        self.assertTrue(sense_match.concepts_of("beak") & sense_match.concepts_of("diminutive of нос (nos): (little) nose; bill, beak"))

    def test_s3_filter(self):
        if not sense_match.concept_table():
            self.skipTest("yerel CLDF yok")
        rows = [{"gloss": "soldier; army, force"}, {"gloss": "to make beautiful"}]
        strict = sense_match.parse_spec("s3:strict")
        self.assertEqual(strict.filter("army", rows), rows[:1])
        self.assertEqual(strict.filter("to make soft", rows), [])
        self.assertEqual(sense_match.parse_spec("s3:fallback").filter("to make soft", rows), rows)


if __name__ == "__main__":
    unittest.main()
