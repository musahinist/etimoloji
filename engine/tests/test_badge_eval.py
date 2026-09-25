"""A-HVP rozet kalibrasyonu ölçümü (``make eval-badge``)."""
from engine.evaluation import badge_eval


def test_auc_orders_and_ties():
    assert badge_eval.auc([3, 3, 0], [True, True, False]) == 1.0
    assert badge_eval.auc([2, 2], [True, False]) == 0.5
    assert badge_eval.auc([1], [True]) is None


def test_summary_per_badge_and_green_vs_yellow_flag():
    rows = [
        {"badge": "VALIDATED", "score": 0.8, "verdict": "CONSISTENT", "verdict_score": 0.9, "correct": True},
        {"badge": "VALIDATED", "score": 0.8, "verdict": "SUSPECT", "verdict_score": 0.2, "correct": False},
        {"badge": "NEEDS_REVIEW", "score": 0.5, "verdict": "CONSISTENT", "verdict_score": 0.8, "correct": True},
        {"badge": None, "score": None, "verdict": "NOT_EVALUATED", "verdict_score": None, "correct": False},
    ]
    s = badge_eval.summarize(rows)
    stage = s["stage_verdict"]
    assert (stage["by_badge"]["VALIDATED"]["n"], stage["by_badge"]["VALIDATED"]["accuracy"]) == (2, 0.5)
    assert stage["by_badge"]["hipotez_yok"]["n"] == 1
    assert s["n_with_hypothesis"] == 3
    # 🟢 %50 < 🟡 %100: eski aşama kararı güveni sıralamıyor ve bu açıkça işaretleniyor.
    assert stage["green_above_yellow"] is False
    v = s["verdict_badge"]
    # Değerlendirilmeyen kelime AUC'ye girmez, ayrı sınıfta sayılır.
    assert v["by_badge"]["NOT_EVALUATED"]["n"] == 1
    assert v["auc_score"] == 1.0 and v["consistent_above_suspect"] is True


def test_fisher_p():
    assert badge_eval.fisher_p(34, 34, 1, 2) == 0.0556
    assert badge_eval.fisher_p(5, 10, 5, 10) == 1.0
    assert badge_eval.fisher_p(1, 1, 0, 0) is None


def test_hypothesis_direction():
    assert badge_eval.hypothesis_is_loan({"hypothesis_donor": "Proto-Türkçe"}) is False
    assert badge_eval.hypothesis_is_loan({"hypothesis_donor": "Türkçe (modern türetme)"}) is False
    assert badge_eval.hypothesis_is_loan({"hypothesis_donor": "Arapça"}) is True
    assert badge_eval.hypothesis_is_loan({"hypothesis_donor": ""}) is True
    assert badge_eval.hypothesis_is_loan({}) is None


def test_turkish_items_never_include_test_split():
    from engine.evaluation.gold import assign_split

    items = badge_eval.turkish_items(n=10_000)
    assert items, "Türkçe altın küme yok"
    assert all(assign_split(f"tr-gold:{w}") != "test" for w, _ in items)
