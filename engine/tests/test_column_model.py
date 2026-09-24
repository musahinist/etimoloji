"""Sütun modeli: saf Python çıkarım, etiketleme, Starling uyumlaştırma, kanca."""

from __future__ import annotations

import json

import pytest

from engine.nlp import column_model as cm
from engine.nlp.multi_alignment import AlignedColumn


def _columns(forms: dict[str, str]) -> list[AlignedColumn]:
    width = max(len(f) for f in forms.values())
    return [
        AlignedColumn(index=i, sounds={lang: (f[i] if i < len(f) else "-") for lang, f in forms.items()}, width=width)
        for i in range(width)
    ]


@pytest.fixture(autouse=True)
def _no_global_model():
    cm.set_model(None)
    yield
    cm.reset_model_cache()


def test_label_columns_marks_suffix_columns_as_null():
    cols = _columns({"kk": "jaksı", "ky": "jakşı", "tt": "yahşı"})
    labels = cm.label_columns(["*jak"], cols)
    assert labels[:3] == ["j", "a", "k"]
    assert labels[3:] == [cm.NULL, cm.NULL]


def test_label_columns_picks_cheapest_equivalent_gold():
    cols = _columns({"kk": "at", "ky": "at"})
    assert cm.label_columns(["*öt", "*at"], cols) == ["a", "t"]


def test_starling_harmonisation():
    assert cm.starling_first_form("elgün (Osm. XIV c.), elgün 'realm'") == "elgün"
    assert cm.starling_first_form("bür- (MK, KB), büz- (IM)") == "bür"
    assert cm.starling_protos("*büŕ- / *bür-") == ["*büŕ", "*bür"]
    assert cm.starling_protos("*Küte(re)") == ["*Küte"]
    assert cm.starling_form("qɨrɣuj") == "kırğuy"


def test_starling_leakage_filter():
    sets = [
        cm.StarlingSet(1, ("*kül",), {"kül"}, []),
        cm.StarlingSet(2, ("*ōt",), {"ot"}, []),
        cm.StarlingSet(3, ("*taš",), {"taş"}, []),
    ]
    kept = cm.starling_allowed(sets, excluded_turkish={"kül"}, excluded_protos={"ot"})
    assert [s.number for s in kept] == [3]


def test_fit_logistic_separates_simple_data():
    rows = [({"x": 1.0}, 1)] * 20 + [({"x": -1.0}, 0)] * 20 + [({"x": 1.0}, 0)] * 2
    weights, intercept = cm.fit_logistic(rows)
    model = cm.ColumnModel(weights, intercept)
    assert model.prob({"x": 1.0}) > 0.8
    assert model.prob({"x": -1.0}) < 0.2


def test_model_roundtrip(tmp_path):
    model = cm.ColumnModel({"share": 2.0, "null": -1.5}, -0.25, {"C": 1.0})
    path = cm.save(model, tmp_path / "m.json")
    loaded = cm.load(path)
    assert loaded is not None
    assert loaded.weights == model.weights and loaded.intercept == model.intercept
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["_schema"] == cm.SCHEMA


def test_load_missing_or_wrong_schema_returns_none(tmp_path):
    assert cm.load(tmp_path / "yok.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"_schema": "baska", "weights": {}, "intercept": 0}), encoding="utf-8")
    assert cm.load(bad) is None


def test_decide_follows_weights_and_never_empties_the_root():
    cols = _columns({"kk": "at", "ky": "at", "tt": "at"})
    # Yalnız paya bakan model: sütundaki sesi seçer.
    keep = cm.ColumnModel({"share": 5.0, "null": -5.0}, 0.0)
    assert [d.sound for d in keep.decide(cols, None)] == ["a", "t"]
    # Her şeyi atmak isteyen model bile en az bir ses bırakır.
    drop = cm.ColumnModel({"null": 10.0}, 0.0)
    sounds = [d.sound for d in drop.decide(cols, None)]
    assert sum(1 for s in sounds if s) == 1


def test_hook_returns_none_without_model():
    cm.set_model(None)
    assert cm.decide(_columns({"kk": "at", "ky": "at"})) is None


def test_column_features_are_reported():
    cols = _columns({"kk": "at", "ky": "at", "tt": "et"})
    decisions = cm.ColumnModel({"share": 5.0, "null": -5.0}, 0.0).decide(cols, None)
    feats = cm.column_features(cols, decisions)
    assert set(feats) >= {
        "column_margin_min", "column_margin_mean", "nbest_margin",
        "table_vote_confidence_min", "table_vote_confidence_mean", "agreement_min", "method_shares",
    }
    assert feats["method_shares"] == {"sutun_modeli": 1.0}
    assert feats["column_margin_min"] <= feats["column_margin_mean"]
