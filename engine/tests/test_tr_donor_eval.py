"""Türkçe verici dil ölçümü (``tr_donor_eval``) — sentetik, ağsız."""

import json

from engine.evaluation import tr_donor_eval as t
from engine.evaluation.gold import assign_split


def test_gold_name_and_classes():
    assert t.gold_name("Arapça ʿasker") == "Arapça"
    assert t.gold_name("Rumca") == "Yunanca"
    assert t.name_class("Orta Farsça") == "Farsça"
    assert t.name_class("Eski Yunanca") == "Yunanca"
    assert t.name_class("Venedikçe") == "İtalyanca"
    assert t.name_class("İngilizce") == "diğer"
    assert t.name_class("") == t.NO_PREDICTION


def test_engine_class_codes_and_names():
    assert t.engine_class("ar") == "Arapça"
    assert t.engine_class("grc") == "Yunanca"
    assert t.engine_class("fa-cls") == "Farsça"
    assert t.engine_class("ota") == "diğer"  # ata katmanı verici sınıfı değil
    assert t.engine_class("Fransızca") == "Fransızca"
    assert t.engine_class("") == t.NO_PREDICTION


def test_search_donor_only_from_borrowed_verdict():
    assert t.search_donor({"kind": "borrowed", "claim": "ALINTI — Arapça"}) == "Arapça"
    assert t.search_donor({"kind": "borrowed", "claim": "ALINTI"}) == ""
    assert t.search_donor({"kind": "inherited", "claim": "MİRAS — x"}) == ""


def _words(n):
    return [f"kelime{i}" for i in range(n)]


def test_load_cases_agreement_and_test_split_sealed(tmp_path):
    words = _words(60)
    items = []
    for i, w in enumerate(words):
        if i % 3 == 0:
            items.append({"word": w, "label": "alıntı", "tdk_source": "Arapça x", "nisanyan_source": "Arapça"})
        elif i % 3 == 1:  # kaynaklar uyuşmuyor
            items.append({"word": w, "label": "alıntı", "tdk_source": "Fransızca y", "nisanyan_source": "İngilizce"})
        else:
            items.append({"word": w, "label": "miras", "tdk_source": "", "nisanyan_source": "Eski Türkçe"})
    items.append({"word": "rumkelime", "label": "alıntı", "tdk_source": "Rumca", "nisanyan_source": "Yunanca"})
    path = tmp_path / "gold.json"
    path.write_text(json.dumps({"items": items}, ensure_ascii=False), encoding="utf-8")
    cases = t.load_cases(path)
    got = {c.word for c in cases}
    expected = {w for i, w in enumerate(words) if i % 3 == 0 and assign_split(f"tr-gold:{w}") != "test"}
    if assign_split("tr-gold:rumkelime") != "test":
        expected.add("rumkelime")
    assert got == expected
    assert all(c.split in ("train", "dev") for c in cases)
    assert {c.gold for c in cases} <= {"Arapça", "Yunanca"}


def test_evaluate_scores_confusion_and_top_error():
    cases = [t.TrDonorCase(f"w{i}", g, g, "", "", "train")
             for i, g in enumerate(["Arapça"] * 6 + ["Farsça"] * 4)]
    full = {}
    for i, c in enumerate(cases):
        # Farsça -> Arapça karışması iki kez; bir madde tahminsiz.
        label = {"Arapça": "ar", "Farsça": "fa"}[c.gold]
        if i in (6, 7):
            label = "ar"
        full[c.word] = {"word": c.word, "label": label, "search": "" if i == 0 else c.gold,
                        "chain": "", "proximity": ""}
    blind = {c.word: {"word": c.word, "label": "", "proximity": "fa"} for c in cases}
    result = t.evaluate(cases, {"full": full, "blind": blind})
    systems = result["systems"]
    assert systems["(a) etiket"]["accuracy"] == 0.8
    assert systems["(a) etiket"]["top_errors"][0] == {"gold": "Farsça", "pred": "Arapça", "n": 2}
    assert systems["(b) arama"]["accuracy"] == 0.9
    assert systems["(b) arama"]["coverage"] == 0.9
    assert systems["(c) kör"]["accuracy"] == 0.4
    assert systems["çoğunluk"]["accuracy"] == 0.6
    assert systems["(a) etiket"]["confusion"]["Farsça"] == {"Arapça": 2, "Farsça": 2}
    assert set(result["vs_majority_mcnemar"]) == {"(a) etiket", "(a) etiket (gösterilen)", "(b) arama",
                                                  "(b) zincir", "(b) dedektör", "(c) kör"}
    # önbellekte gösterilen etiket yoksa ham etikete düşer
    assert systems["(a) etiket (gösterilen)"]["accuracy"] == 0.8


def test_capture_refuses_wrong_environment(monkeypatch):
    import pytest

    monkeypatch.setenv("ETY_DONOR_CLEAN", "1")
    monkeypatch.setenv("ETY_DONOR_RAMP_CHANCE", "1")
    monkeypatch.delenv("ETY_LEXICON_INDEX", raising=False)
    with pytest.raises(SystemExit):
        t._check_env("full", "off")
    with pytest.raises(SystemExit):
        t._check_env("blind", "on")
    t._check_env("full", "on")


def test_natural_accuracy_reweights_classes():
    gold = ["Arapça"] * 8 + ["Fransızca"] * 2
    all_ar = ["Arapça"] * 10
    # ham doğruluk 0,8; doğal ağırlıkla yalnız Arapça payı
    w_ar = t.NATURAL_COUNTS["Arapça"] / (t.NATURAL_COUNTS["Arapça"] + t.NATURAL_COUNTS["Fransızca"])
    assert abs(t.natural_accuracy(gold, all_ar) - round(w_ar, 4)) < 1e-4
    assert t.natural_accuracy(gold, gold) == 1.0


def test_family_label_partial_credit():
    assert t.engine_class("ar|fa") == "Arapça|Farsça"
    assert t.credit("Farsça", "Arapça|Farsça") == 0.5
    assert t.credit("Fransızca", "Arapça|Farsça") == 0.0
    assert t.credit("Arapça", "Arapça") == 1.0
    gold = ["Arapça", "Fransızca"]
    w = t.NATURAL_COUNTS
    got = t.natural_accuracy(gold, ["Arapça|Farsça", "Fransızca"])
    assert abs(got - round((0.5 * w["Arapça"] + w["Fransızca"]) / (w["Arapça"] + w["Fransızca"]), 4)) < 1e-4


def test_form_metrics_precision_and_coverage(tmp_path):
    import sqlite3

    db = tmp_path / "index.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE entries (lang_code, word, donor_lang, donor_form, etymology)")
    con.execute("INSERT INTO entries VALUES ('tr', 'sultan', 'ar', 'سُلْطَان', '')")
    con.commit()
    cases = [t.TrDonorCase("sultan", "Arapça", "Arapça", "Arapça sulṭān", "Arapça", "train"),
             t.TrDonorCase("kudret", "Arapça", "Arapça", "Arapça ḳudret", "Arapça", "train"),
             t.TrDonorCase("aktör", "Fransızca", "Fransızca", "Fransızca acteur", "Fransızca", "train")]
    rows = {
        "sultan": {"label": "ar", "label_word": "سلطان", "label_comparison": "sultan", "label_certain": True},
        "kudret": {"label": "ar", "label_word": "قطر", "label_comparison": "katar", "label_certain": False},
        "aktör": {"label": "fr", "label_word": "acteur", "label_comparison": "acteur", "label_certain": True},
    }
    out = t.form_metrics(cases, rows, db)
    raw, shown = out["(a) etiket"], out["(a) etiket (gösterilen)"]
    assert (raw["shown"], raw["shown_correct"], raw["coverage"]) == (3, 2, 1.0)
    assert (shown["shown"], shown["shown_correct"], shown["form_precision"]) == (2, 2, 1.0)
    assert abs(shown["coverage"] - 0.6667) < 1e-4
