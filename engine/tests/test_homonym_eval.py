"""Eşsesli eval ayrıştırıcısı ve ölçütleri — sentetik kaikki kayıtlarıyla."""

from engine.evaluation.homonym_eval import (
    aggregate,
    assign_gloss,
    assign_root,
    homonyms_from_records,
    proto_roots,
    score_word,
)


def _rec(word, number, pos, gloss, templates=(), text="", form_of=False):
    sense = {"glosses": [gloss]}
    if form_of:
        sense["form_of"] = [{"word": word}]
    return {
        "word": word, "pos": pos, "etymology_number": number, "senses": [sense],
        "etymology_templates": list(templates), "etymology_text": text,
    }


def _inh(lang, form):
    return {"name": "inh", "args": {"1": "tr", "2": lang, "3": form}}


def _bor(lang, form):
    return {"name": "bor", "args": {"1": "tr", "2": lang, "3": form}}


RECORDS = [
    _rec("zor", "1", "noun", "face", [_inh("trk-pro", "*zör")]),
    _rec("zor", "2", "num", "hundred", [_inh("trk-pro", "*zöz")]),
    _rec("zor", "2", "noun", "hundredth part", [_inh("trk-pro", "*zöz")]),
    # biçim göndermesi ve lema göstergesi sayılmaz
    _rec("zor", "3", "verb", "imperative of zormak", form_of=True),
    _rec("zor", "4", "noun", "x", text="See the etymology of the corresponding lemma form."),
    # tek etimolojili başlık eşsesli değil
    _rec("tek", None, "noun", "single", [_inh("trk-pro", "*tek")]),
    # alıntı + miras: kökle değil vericiyle ayrılır
    _rec("çağ", "1", "noun", "tea plant", [_bor("fa", "چای")]),
    _rec("çağ", "2", "noun", "small river, brook", [_inh("trk-pro", "*čāy")]),
    # ortak kök: ayırt edilemez
    _rec("yağ", "1", "noun", "oil", [_inh("trk-pro", "*yāg")]),
    _rec("yağ", "2", "noun", "fat", [_inh("trk-pro", "*yāg")]),
]


def test_homonyms_from_records_groups_numbered_lemmas():
    homs = {h["word"]: h for h in homonyms_from_records(RECORDS)}
    assert set(homs) == {"zor", "çağ", "yağ"}
    zor = homs["zor"]
    assert [e["number"] for e in zor["etymologies"]] == [1, 2]
    assert zor["etymologies"][1]["glosses"] == ["hundred", "hundredth part"]
    assert zor["etymologies"][1]["pos"] == ["num", "noun"]
    assert zor["etymologies"][0]["proto_roots"] == ["*zör"]
    assert zor["distinguishable"]
    assert homs["çağ"]["distinguishable"]
    assert homs["çağ"]["etymologies"][0]["origin"] == "alıntı"
    assert not homs["yağ"]["distinguishable"]


def test_proto_roots_text_fallback_takes_first_only():
    rec = {"etymology_text": "From Proto-Turkic *kol (“arm”). Compare Proto-Turkic *kōl.", "etymology_templates": []}
    assert proto_roots(rec) == ["*kol"]


def test_assign_gloss_and_root():
    etyms = {h["word"]: h for h in homonyms_from_records(RECORDS)}["zor"]["etymologies"]
    assert assign_gloss("one hundred", etyms) == 2
    assert assign_gloss("FACE", etyms) == 1
    assert assign_gloss("yüz", etyms) is None  # Türkçe anlam atanamaz
    assert assign_root("*zör", etyms) == 1
    assert assign_root("*qqq", etyms) == "hiçbiri"


def _summary(**kw):
    base = {"headline": "", "meaning_en": "", "verdict_kind": "inherited", "stars": [],
            "borrowed_claims": [], "homonym_cognates": [], "homonym_text": [], "witnesses": []}
    base.update(kw)
    return base


def test_score_word_mixed_headline_and_witnesses():
    item = {h["word"]: h for h in homonyms_from_records(RECORDS)}["zor"]
    # 'face' kökü + 'hundred' anlamı ve tanıkları: yüz türü karışma
    row = score_word(item, _summary(
        headline="*zör", meaning_en="hundred", stars=["*zör"],
        witnesses=[{"meaning": "hundred"}, {"meaning": "hundred"}, {"meaning": "face"}, {"meaning": "?"}],
    ))
    assert row["b_headline_etym"] == 1 and row["b_meaning_etym"] == 2
    assert row["b_tutarsiz_baslik"]
    assert row["c_assigned"] == 3 and row["c_foreign"] == 2
    assert row["c_karisik"] and row["c_baslik_tanik_catismasi"]
    assert not row["a_fark_ediyor"]


def test_score_word_awareness_signals():
    item = {h["word"]: h for h in homonyms_from_records(RECORDS)}["çağ"]
    row = score_word(item, _summary(
        headline="*čāy", stars=["*čāy"], borrowed_claims=["Farsça çay"],
        homonym_cognates=[{"word": "x", "meaning": "tea"}],
    ))
    assert row["a_cok_koken"] and row["a_uyari_tanik"] and row["a_fark_ediyor"]
    assert row["surfaced_etymologies"] == [1, 2]
    summary = aggregate([row])
    assert summary["a"]["a_fark_ediyor"]["k"] == 1
    assert summary["b"]["baslik_etimolojisi"] == {"E2": 1}


def test_shared_root_does_not_count_as_two_etymologies():
    item = {h["word"]: h for h in homonyms_from_records(RECORDS)}["yağ"]
    row = score_word(item, _summary(headline="*yāg", stars=["*yāg"]))
    assert row["b_headline_etym"] == "belirsiz"
    assert not row["a_cok_koken"]
