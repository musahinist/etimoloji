"""
ASJP-PMI mesafesi ve kavram hizalı Moğol/Tunguz verici listeleri.

İkisi de ölçüldü ve üretimde KAPALI (bkz. ``donor_proximity``
``CONCEPT_DONOR_LABELS`` / ``LABEL_DISTANCE`` / ``STRENGTH_DISTANCE``).
Testler hizalamanın doğruluğunu, alfabe dönüşümünü ve bayrakların varsayılan
olarak üretimi değiştirmediğini korur.
"""

from __future__ import annotations

import random
import unittest
from functools import lru_cache

from engine.db import concept_donors as cd
from engine.nlp import donor_proximity as dp
from engine.nlp import pmi_distance as pmi


def _reference(a: str, b: str) -> float:
    """Bütün hizalamaları gezen afin boşluklu global hizalama (Biopython globalds)."""
    table, gap_open, gap_extend = pmi._matrix()

    @lru_cache(maxsize=None)
    def best(i: int, j: int, last: str) -> float:
        if i == len(a) and j == len(b):
            return 0.0
        out = float("-inf")
        if i < len(a) and j < len(b):
            out = max(out, table[(a[i], b[j])] + best(i + 1, j + 1, "M"))
        if i < len(a):
            out = max(out, (gap_extend if last == "X" else gap_open) + best(i + 1, j, "X"))
        if j < len(b):
            out = max(out, (gap_extend if last == "Y" else gap_open) + best(i, j + 1, "Y"))
        return out

    return best(0, 0, "S")


@unittest.skipUnless(pmi.available(), "data/asjp PMI matrisi yok")
class TestPmiDistance(unittest.TestCase):
    def test_gotoh_matches_exhaustive_alignment(self):
        rng = random.Random(1)
        symbols = "aeiou3ptkbdgmnlrsSCjxyNh"
        for _ in range(200):
            a = "".join(rng.choice(symbols) for _ in range(rng.randint(1, 7)))
            b = "".join(rng.choice(symbols) for _ in range(rng.randint(1, 7)))
            self.assertAlmostEqual(pmi.pmi_score(a, b), _reference(a, b), places=9)

    def test_identity_is_zero_and_unrelated_is_far(self):
        self.assertEqual(pmi.pmi_distance("stol", "stol"), 0.0)
        self.assertLess(pmi.pmi_distance("mıla", "mılo"), 0.3)
        self.assertGreaterEqual(pmi.pmi_distance("bagana", "çuçka"), 1.0)

    def test_comparison_alphabet_is_mapped_to_asjp(self):
        self.assertEqual(pmi.to_asjp("çaşığ"), "CaS3x")
        self.assertEqual(pmi.to_asjp("cöŋ"), "joN")


class TestSegmentComparison(unittest.TestCase):
    def test_ipa_glides_and_affricates(self):
        # IPA j = Türk y, IPA y = ü; d͡ʒ = c, t͡ʃ = ç
        self.assertEqual(cd.segment_comparison("j a d͡ʒ y t͡ʃ"), "yacüç")

    def test_slash_notation_takes_sound_value(self):
        self.assertEqual(cd.segment_comparison("č/tʃ è/e m è/e"), "çeme")

    def test_boundaries_and_length_are_dropped(self):
        self.assertEqual(cd.segment_comparison("aː + l _ ɮ"), "all")


class TestPools(unittest.TestCase):
    def test_khalkha_goes_to_mongolic_pool_not_khakas(self):
        # NorthEuraLex khk = Halha Moğolcası; motorda khk = Hakasça.
        self.assertEqual(cd.NORTHEURALEX_POOLS["khk"], "mn")
        self.assertEqual(set(cd.NORTHEURALEX_POOLS.values()), {"mn", "evn"})

    def test_flags_are_off_in_production(self):
        self.assertFalse(dp.CONCEPT_DONOR_LABELS)
        self.assertEqual(dp.LABEL_DISTANCE, "sca")
        self.assertEqual(dp.STRENGTH_DISTANCE, "sca")
        self.assertEqual(dp.ATTRIBUTION_SCORE, "median")


if __name__ == "__main__":
    unittest.main()
