"""Hüküm rozeti: ölçülmüş iki sınıf + değerlendirilmedi (``engine.nlp.verdict_badge``)."""
from engine.nlp import verdict_badge


def _nlp(rank_score, detect_conf, key, donor="Proto-Türkçe"):
    return {
        "proven_hypothesis": {"donor_language": donor,
                              "validation_report": {"status_code": "VALIDATED", "badge": "🟢 DOĞRULANDI"}},
        "ranked_hypotheses": {"selected": {"kind": "inherited", "score": rank_score}},
        "loanword_detection": {"confidence": detect_conf},
        "loanword_classification": {"classification_key": key},
    }


def test_no_hypothesis_is_not_evaluated_and_colourless():
    nlp = {"ranked_hypotheses": {"selected": {"score": 0.9}}}
    result = verdict_badge.apply(nlp)
    assert result["code"] == "NOT_EVALUATED" and result["score"] is None
    assert not any(c in result["label"] for c in "🟢🟡⚪🔴")
    assert nlp["verdict_badge"]["code"] == "NOT_EVALUATED"


def test_strong_agreeing_evidence_is_consistent_and_replaces_shown_badge():
    nlp = _nlp(0.9, 1.0, "native")
    result = verdict_badge.apply(nlp)
    assert result["code"] == "CONSISTENT"
    report = nlp["proven_hypothesis"]["validation_report"]
    assert report["badge"] == result["label"]
    # Eski aşama kararı saklanır; iç kararlar için status_code değişmez.
    assert report["stage_badge"] == "🟢 DOĞRULANDI" and report["status_code"] == "VALIDATED"
    verdict_badge.apply(nlp)  # ikinci kez uygulamak aşama rozetini ezmez
    assert report["stage_badge"] == "🟢 DOĞRULANDI"


def test_weak_disagreeing_evidence_is_suspect():
    # Miras hipotezi, sınıflayıcı alıntı diyor, sıralayıcı skoru ve dedektör güveni düşük.
    assert verdict_badge.assess(_nlp(0.15, 0.5, "western"))["code"] == "SUSPECT"


def test_classifier_agreement_follows_hypothesis_direction():
    assert verdict_badge.inputs(_nlp(0.3, 0.7, "western", donor="Arapça"))["agree_classifier"] == 1.0
    assert verdict_badge.inputs(_nlp(0.3, 0.7, "native", donor="Arapça"))["agree_classifier"] == 0.0
    assert verdict_badge.inputs(_nlp(0.3, 0.7, None))["agree_classifier"] == 0.0


def test_score_monotone_in_inputs():
    low = verdict_badge.score({"rank_score": 0.2, "detect_conf": 0.6, "agree_classifier": 0.0})
    high = verdict_badge.score({"rank_score": 0.5, "detect_conf": 0.9, "agree_classifier": 1.0})
    assert 0.0 <= low < high <= 1.0
