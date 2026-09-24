"""Başlık / verici tanıma / kronoloji ölçümlerinin saf yardımcıları.

Motor koşusu gerektirmez: arama hattı ``make eval-headline`` ve
``make eval-chronology`` ile koşar, burada yalnız puanlama mantığı sınanır.
"""

from __future__ import annotations

import pytest

from engine.evaluation import chronology_eval, donor_id_eval, headline_eval


def test_wilson_bounds():
    low, high = headline_eval.wilson(5, 10)
    assert 0.0 < low < 0.5 < high < 1.0
    assert headline_eval.wilson(0, 0) == (0.0, 0.0)
    assert headline_eval.wilson(10, 10)[1] == 1.0


def test_score_item_exact_and_ned():
    hit = headline_eval.score_item("*kemük", ["*kemük"])
    assert hit["exact"] and hit["ned"] == 0.0
    # Yıldızsız başlık (sorgu kelimesinin kendisi) de puanlanır, ceza yer.
    miss = headline_eval.score_item("kemik", ["*kemük"])
    assert not miss["exact"] and miss["ned"] > 0
    # Eşdeğer adaylardan herhangi biri yeter.
    assert headline_eval.score_item("*jeŋi", ["*jaŋï", "*jeŋi"])["exact"]


def test_from_starling_uses_provenance():
    assert headline_eval.from_starling("tanıklı — Starling (Dybo & Starostin 2005) (Proto-Türkçe *sɨb)")
    assert not headline_eval.from_starling("tanıklı — Osmanlı Türkçesi sözlük kaydı")


def test_evaluate_subset_filters_and_baseline():
    items = [("su", ["*sɨb"]), ("kemik", ["*kemük"])]
    runs = {
        "su": {"headline": "*sɨb", "provenance": "tanıklı — Starling"},
        "kemik": {"headline": "*kemük", "provenance": "tanıklı — sözlük"},
    }
    everything = headline_eval.evaluate_subset(items, runs)
    assert everything["engine"]["n"] == 2 and everything["engine"]["exact"] == 1.0
    assert everything["baseline_identity"]["exact"] == 0.0
    no_starling = headline_eval.evaluate_subset(
        items, runs, keep=lambda r: not headline_eval.from_starling(r["provenance"]))
    assert no_starling["engine"]["n"] == 1


def test_summarize_finding_reads_stage2_year():
    finding = {
        "root": {"proto_turkic": "*sɨb", "provenance": "x"},
        "nlp_analysis": {
            "proven_hypothesis": {"validation_report": {
                "stage_breakdown": {"stage2_time_lock": {"attestation_year": 1072}}}},
            "unattested_word_reconstruction": {"attestation": {"first_attestation_record": "MK"}},
        },
    }
    out = headline_eval.summarize_finding(finding)
    assert out["headline"] == "*sɨb" and out["attestation_year"] == 1072
    assert headline_eval.summarize_finding({})["attestation_year"] is None


def test_is_turkish_word_rejects_non_turkish_letters():
    assert not headline_eval.is_turkish_word("arqun")
    assert not headline_eval.is_turkish_word("")


def test_donor_class_mappings():
    assert donor_id_eval.wold_class("Kalmyk") == "Moğolca"
    assert donor_id_eval.wold_class("Sanskrit") == "diğer"
    assert donor_id_eval.engine_class("xgn-pro") == "Moğolca"
    assert donor_id_eval.engine_class("zle-mru") == "Rusça"
    assert donor_id_eval.engine_class("evn") == "Tunguzca"
    assert donor_id_eval.engine_class("") == donor_id_eval.NO_PREDICTION
    assert donor_id_eval.engine_class("ar") == "diğer"


def test_donor_score_counts_abstention_as_wrong():
    gold = ["Rusça", "Rusça", "Moğolca", "Moğolca"]
    pred = ["Rusça", donor_id_eval.NO_PREDICTION, "Moğolca", "Rusça"]
    s = donor_id_eval.score(gold, pred)
    assert s["accuracy"] == 0.5
    assert s["coverage"] == 0.75
    assert s["accuracy_when_predicted"] == round(2 / 3, 4)
    assert s["confusion"]["Moğolca"]["Rusça"] == 1


def test_donor_cases_have_known_donor():
    cases = donor_id_eval.load_cases()
    if not cases:
        pytest.skip("WOLD indirilmemiş")
    assert all(c.gold and c.languoid != "Unidentified" for c in cases)


def test_chronology_agreement():
    same = chronology_eval.agreement(1074, 1072)
    assert same["same_source"] and same["within_century"] and same["not_later"]
    later = chronology_eval.agreement(1303, 1072)
    assert not later["not_later"] and not later["within_century"]
    none = chronology_eval.agreement(None, 1072)
    assert not none["has_year"] and none["diff"] is None


def test_chronology_summary_separates_coverage():
    rows = [chronology_eval.agreement(1074, 1072), chronology_eval.agreement(None, 732)]
    s = chronology_eval.summarize(rows)
    assert s["coverage"] == 0.5
    assert s["within_century|covered"] == 1.0
    assert s["within_century|all"] == 0.5
