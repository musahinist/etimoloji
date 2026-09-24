"""A-HVP rozet kalibrasyonu ölçümü (``make eval-badge``)."""
from engine.evaluation import badge_eval


def test_auc_orders_and_ties():
    assert badge_eval.auc([3, 3, 0], [True, True, False]) == 1.0
    assert badge_eval.auc([2, 2], [True, False]) == 0.5
    assert badge_eval.auc([1], [True]) is None


def test_summary_per_badge_and_green_vs_yellow_flag():
    rows = [
        {"badge": "VALIDATED", "score": 0.8, "correct": False},
        {"badge": "VALIDATED", "score": 0.8, "correct": True},
        {"badge": "NEEDS_REVIEW", "score": 0.5, "correct": True},
        {"badge": None, "score": None, "correct": False},
    ]
    s = badge_eval.summarize(rows)
    assert (s["by_badge"]["VALIDATED"]["n"], s["by_badge"]["VALIDATED"]["accuracy"]) == (2, 0.5)
    assert s["by_badge"]["hipotez_yok"]["n"] == 1
    assert s["n_with_hypothesis"] == 3
    # 🟢 %50 < 🟡 %100: rozet güveni sıralamıyor ve bu açıkça işaretleniyor.
    assert s["green_above_yellow"] is False


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
