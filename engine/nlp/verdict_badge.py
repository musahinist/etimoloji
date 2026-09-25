"""
Kullanıcıya gösterilen hüküm rozeti — A-HVP aşama kararının yerine.

Neden: A-HVP'nin dört aşamalı kararı (🟢/🟡/⚪/🔴, ``_decide``) doğruyu
yanlıştan AYIRMIYORDU. ``make eval-badge`` (adb167d): Türkçe altın AUC 0,494,
savelyev dev 0,458; 🔴 kutusu 🟢'dan kötü değildi. Aşama skorları, kapsam,
üçgenleme, ``column_features`` tek başına ayar kümesinde AUC 0,47–0,55.

Ölçüm (ön-kayıt: ``data/cache/work/badge/PREREG.md``). Doğruluk: A-HVP
hipotezinin yönü (miras/alıntı) Türkçe altınla aynı. AYAR = Türkçe altın
train (183 hipotezli kelime, 11 yanlış), RAPOR = dev (bir kez).

* Tek girdi AUC (ayar): sıralayıcının seçtiği hipotez skoru 0,779 ·
  ``loanword_detection`` güveni 0,772 · sıralayıcı marjı 0,734 · sınıflayıcı
  yönünün hipotezle uyumu 0,676 · A-HVP final skoru 0,636 · kronoloji
  kanıtı var 0,611 · eski rozet sırası 0,513 · ``column_features`` ≤ 0,53 ·
  ``conflicts`` ayarda hiç görülmedi.
* Kural (ilk üç bağımsız girdinin ayar kümesi ampirik CDF'leri ortalaması)
  5-kat CV AUC 0,915; 8 girdili lojistik 0,913 → basit kural seçildi.
* RAPOR (dev, bir kez): AUC 0,984 — ama 62 hipotezli kelimede YALNIZ 1
  yanlış var; alt %10 eşiği altı 2 kelime (doğruluk 0,5), üstü 60 kelime
  (1,0). Wilson aralıkları örtüşüyor, Fisher p ≈ 0,06: ön-kayıttaki
  "🟢 🔴'dan anlamlı yüksek" şartı **GEÇMEDİ**. savelyev dev (30): AUC 0,68
  ama eşikte ayrım yok (0,667 / 0,667).
* Ayarda 🟢/🟡 ayrımı ölçülemedi (üst yarı 0,989, orta 0,986; dev'de ikisi de
  1,0). Bu yüzden rozet **iki sınıf + değerlendirilmedi**:

  ``CONSISTENT`` 🟢 TUTARLI  — ayarda doğruluk 0,988 (n=164)
  ``SUSPECT``    🔴 ŞÜPHELİ  — ayarda 0,526 (n=19); ölçülmüş tek ayrım budur
  ``NOT_EVALUATED`` ∅        — A-HVP hipotezi yok (Türkçe altının ~%58'i);
                               renk verilmez, "kanıt yetersiz" DEMEZ.

⚠️ Döngüsellik: sıralayıcı ağırlıkları Türkçe altının tamamında eğitildi ve
yön doğruluğu kısmen Wiktionary↔TDK/Nişanyan uyumudur; %94 taban doğruluk
bu yüzden iyimser. Rozet mutlak doğruluk değil, göreli şüphe bildirir.
"""
from __future__ import annotations

import json
from bisect import bisect_right
from functools import lru_cache
from typing import Any

from engine.config import PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "data" / "models" / "verdict_badge.json"
_INHERITED_DONORS = ("Proto-Türkçe", "Türkçe (modern türetme)", "Ana Türkçe", "Öz Türkçe", "Eski Türkçe")

LABELS = {
    "CONSISTENT": "🟢 TUTARLI (hüküm ile bağımsız sinyaller uyumlu)",
    "SUSPECT": "🔴 ŞÜPHELİ (hüküm zayıf ya da sinyallerle çelişiyor)",
    "NOT_EVALUATED": "∅ DEĞERLENDİRİLMEDİ (sınanacak hipotez yok)",
}


@lru_cache(maxsize=1)
def _model() -> dict[str, Any]:
    return json.loads(MODEL_PATH.read_text(encoding="utf-8"))


def _hypothesis_is_loan(hypothesis: dict[str, Any]) -> bool:
    # ``badge_eval.hypothesis_is_loan`` ile aynı: boş verici (fonotaktik aday)
    # alıntı yönüdür; burada yalnız miras adları sayılır.
    donor = hypothesis.get("donor_language")
    return donor is not None and donor not in _INHERITED_DONORS


def inputs(nlp_analysis: dict[str, Any]) -> dict[str, float] | None:
    """Üç girdi; hipotez yoksa ``None``."""
    hypothesis = nlp_analysis.get("proven_hypothesis") or {}
    if not hypothesis.get("validation_report"):
        return None
    ranked = nlp_analysis.get("ranked_hypotheses") or {}
    selected = ranked.get("selected") or {}
    detection = nlp_analysis.get("loanword_detection") or {}
    key = (nlp_analysis.get("loanword_classification") or {}).get("classification_key")
    agree = float(bool(key) and (key != "native") == _hypothesis_is_loan(hypothesis))
    return {
        "rank_score": round(float(selected.get("score") or 0.0), 4),
        "detect_conf": round(float(detection.get("confidence") or 0.0), 4),
        "agree_classifier": agree,
    }


def score(values: dict[str, float]) -> float:
    """Ayar kümesi ampirik CDF'lerinin ortalaması (0–1)."""
    ref = _model()["reference"]
    return round(sum(bisect_right(ref[k], values[k]) / len(ref[k]) for k in ref) / len(ref), 4)


def assess(nlp_analysis: dict[str, Any]) -> dict[str, Any]:
    values = inputs(nlp_analysis)
    if values is None:
        return {"code": "NOT_EVALUATED", "label": LABELS["NOT_EVALUATED"], "score": None, "inputs": {}}
    s = score(values)
    code = "SUSPECT" if s < _model()["thresholds"]["suspect_below"] else "CONSISTENT"
    return {"code": code, "label": LABELS[code], "score": s, "inputs": values}


def apply(nlp_analysis: dict[str, Any]) -> dict[str, Any]:
    """Rozeti ``nlp_analysis["verdict_badge"]``'a yazar; A-HVP raporunda
    gösterilen ``badge`` metni bu rozettir, eski aşama kararı
    ``stage_badge`` alanında kalır (``status_code`` değişmez)."""
    result = assess(nlp_analysis)
    nlp_analysis["verdict_badge"] = result
    report = (nlp_analysis.get("proven_hypothesis") or {}).get("validation_report")
    if report:
        report.setdefault("stage_badge", report.get("badge"))
        report["badge"] = result["label"]
        report["verdict_badge"] = result["code"]
    return result
