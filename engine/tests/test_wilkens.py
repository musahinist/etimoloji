"""Wilkens Eski Uygurca sözlüğü: PDF satır ayrıştırıcısı ve yerel tanık fetcher'ı.

Ayrıştırıcı sentetik satırlarla sınanır (PDF gerekmez); gerçek veriyle
koşan denetim ``data/wilkens/wilkens_oui.jsonl`` yoksa atlanır.
"""

from __future__ import annotations

import json

import pytest

from engine.db import wilkens
from engine.db.wilkens import (
    COLUMN_LEFT,
    FIRST_PAGE,
    Line,
    Token,
    parse_lines,
    split_gloss,
)
from engine.fetchers import wilkens_old_uyghur as wou
from engine.fetchers.wilkens_old_uyghur import WilkensOldUyghurFetcher, coarse, gloss_items

LEFT = COLUMN_LEFT[0]


def _line(*tokens: tuple[str, str], x: float = LEFT, small: set[int] | None = None) -> Line:
    small = small or set()
    return Line(0, x, 0.0, [Token(style, i in small, text) for i, (style, text) in enumerate(tokens)],
                FIRST_PAGE)


# --- ayrıştırıcı ----------------------------------------------------------------

def test_main_entries_subentries_and_continuations():
    lines = [
        _line(("B", "1"), ("B", "ačıt-"), ("r", " bekümmern, quälen || "), small={0}),
        _line(("r", "üzmek, acı vermek")),
        _line(("I", "ačıt"), ("r", "- "), ("I", "agrıt"), ("r", "- quälen"), ("r", "2"),
              ("r", " || acıtmak"), x=LEFT + 12, small={4}),
        _line(("B", "2"), ("B", "ačıt- "), ("r", "umsorgt werden || ilgilenilmek"), small={0}),
    ]
    first, second = parse_lines(lines)
    assert (first["headword"], first["homonym"]) == ("ačıt-", 1)
    assert first["de"] == ["bekümmern, quälen"] and first["tr"] == ["üzmek, acı vermek"]
    assert first["subentries"] == [{"form": "ačıt- agrıt-", "de": ["quälen"], "tr": ["acıtmak"]}]
    assert (second["headword"], second["homonym"], second["tr"]) == ("ačıt-", 2, ["ilgilenilmek"])


def test_italic_continuation_with_other_initial_is_not_a_subentry():
    lines = [
        _line(("B", "abidarmakoš"), ("r", " Werktitel || bir metnin başlığı")),
        _line(("I", "abidarmakoš šasdr"), ("r", " << Skt. "), ("I", "abhidharmakośa-"), x=LEFT + 12),
        _line(("I", "śāstra"), ("r", " Werktitel || bir metnin başlığı"), x=LEFT + 12),
    ]
    (entry,) = parse_lines(lines)
    assert [s["form"] for s in entry["subentries"]] == ["abidarmakoš šasdr"]


def test_line_end_hyphenation_is_joined_in_glosses():
    lines = [_line(("B", "adro-"), ("r", " vorankommen || ilerle-")), _line(("r", "mek"))]
    (entry,) = parse_lines(lines)
    assert entry["tr"] == ["ilerlemek"]


def test_donor_chain_and_flags():
    lines = [
        _line(("B", "ačite"), ("r", " < TochB "), ("I", "ajite"), ("r", " < Skt. "), ("I", "ajita"),
              ("r", " n. pr. (ein Gandharva) || bir Gandharva’nın adı")),
        _line(("B", "abidarim"), ("r", " < TochA "), ("I", "abhidharm"), ("r", " / < TochB "),
              ("I", "abhidhārm"), ("r", " < Skt. "), ("I", "abhidharma"), ("r", " Scholastik || skolastik")),
        _line(("B", "abag "), ("r", "(br) → "), ("I", "apıg")),
        _line(("B", "abam"), ("r", " † → "), ("I", "apam")),
        _line(("B", "arbuda "), ("r", "<"), ("B", " "), ("r", " Skt. "), ("I", "arbuda "),
              ("r", "ein Bezirk || bir alan")),
    ]
    loan, chain3, redirect, error, split_arrow = parse_lines(lines)
    assert [(s["lang"], s["form"], s["direct"]) for s in loan["donor_chain"]] == [
        ("TochB", "ajite", True), ("Skt.", "ajita", True)]
    assert loan["proper_name"] and loan["tr"] == ["bir Gandharva’nın adı"]
    assert [s["lang_name"] for s in chain3["donor_chain"]] == ["Toharca", "Toharca", "Sanskritçe"]
    assert chain3["de"] == ["Scholastik"]
    assert redirect["see"] == "apıg" and redirect["marks"] == ["brahmi"] and not redirect.get("error")
    assert error["error"] and error["see"] == "apam"
    assert split_arrow["donor_chain"][0]["form"] == "arbuda" and split_arrow["de"] == ["ein Bezirk"]


def test_split_gloss_ignores_semicolons_inside_parentheses():
    german, turkish = split_gloss(
        "Fleisch (einer der sieben Dhātus; auch Äquivalent von Skt. māṃsa) || et (yedi Dhātu’dan "
        "biri; Skt. māṃsa’nın eş değeri); Haut || deri"
    )
    assert german == ["Fleisch (einer der sieben Dhātus; auch Äquivalent von Skt. māṃsa)", "Haut"]
    assert turkish == ["et (yedi Dhātu’dan biri; Skt. māṃsa’nın eş değeri)", "deri"]


def test_small_digits_in_glosses_are_dropped():
    lines = [_line(("B", "adrıl-"), ("r", " sich trennen"), ("r", "2"), ("r", " || ayrılmak"), small={2})]
    (entry,) = parse_lines(lines)
    assert entry["de"] == ["sich trennen"]


# --- fetcher --------------------------------------------------------------------

RECORDS = [
    {"headword": "köz", "homonym": 1, "variants": [], "page": 401, "proper_name": False,
     "de": ["Auge"], "tr": ["göz(ler) (Skt. cakṣus’un da eş değeri)", "bakış"]},
    {"headword": "közlüg", "homonym": None, "variants": [], "page": 402, "proper_name": False,
     "de": ["mit Augen"], "tr": ["gözlü, göz …"]},
    {"headword": "kök", "homonym": 2, "variants": [], "page": 401, "proper_name": False,
     "de": ["Film"], "tr": ["ince zar, göz bebeklerinin perdelenmesi"]},
    {"headword": "taš", "homonym": 1, "variants": [], "page": 690, "proper_name": False,
     "de": ["Stein"], "tr": ["taş"]},
    {"headword": "taš-", "homonym": None, "variants": [], "page": 690, "proper_name": False,
     "de": ["überlaufen"], "tr": ["taşmak"]},
    {"headword": "bil-", "homonym": 1, "variants": [], "page": 170, "proper_name": False,
     "de": ["wissen"], "tr": ["bilmek, anlamak"]},
    {"headword": "ešik", "homonym": None, "variants": [], "page": 120, "proper_name": False,
     "de": ["Tür"], "tr": ["kapı, eşik"]},
    {"headword": "kapıg", "homonym": None, "variants": [], "page": 330, "proper_name": False,
     "de": ["Tor"], "tr": ["büyük kapı", "kapı"],
     "donor_chain": [{"lang": "Sogd.", "lang_name": "Soğdca", "direct": True, "form": "x"}]},
    {"headword": "Kara", "homonym": None, "variants": [], "page": 340, "proper_name": True,
     "de": ["n. pr."], "tr": ["kara"]},
    {"headword": "kara", "homonym": 9, "variants": [], "page": 340, "error": True, "see": "kar",
     "proper_name": False, "de": ["† → kar"], "tr": ["kara"]},
]


@pytest.fixture
def fake_records(monkeypatch):
    monkeypatch.setattr(wilkens, "load_records", lambda path=None: json.loads(json.dumps(RECORDS)))
    monkeypatch.setattr(wou, "_predicted_old_turkic", lambda stem: "")
    wou._index.cache_clear()
    yield
    wou._index.cache_clear()


def _heads(word: str) -> list[str]:
    return [e["word"] for e in WilkensOldUyghurFetcher().fetch(word)["turkic_languages"]]


def test_gloss_items_drop_parentheses_and_derivative_markers():
    assert gloss_items("göz(ler) (Skt. cakṣus’un da eş değeri), bakış") == ["göz", "bakış"]
    assert gloss_items("gözlü, göz …") == ["gözlü"]


def test_coarse_sound_classes():
    assert coarse("köz") == coarse("göz")
    assert coarse("ačıg") == coarse("acı")
    assert coarse("taŋ") == coarse("tan")


def test_turkish_gloss_and_form_must_both_match(fake_records):
    assert _heads("göz") == ["köz"]  # *kök* "göz bebeği…" ve türev *közlüg* değil
    assert _heads("kapı") == ["kapıg"]  # *ešik* "kapı" çeviri eşdeğeri, akraba değil
    assert _heads("bil") == ["bil-"] and _heads("bilmek") == ["bil-"]


def test_bare_query_prefers_noun_over_verb(fake_records):
    assert _heads("taş") == ["taš"]
    assert _heads("taşmak") == ["taš-"]


def test_proper_names_and_errors_are_not_witnesses(fake_records):
    assert _heads("kara") == []


def test_attestation_is_dated_by_period_label(fake_records):
    result = WilkensOldUyghurFetcher().fetch("göz")
    attestation = result["first_attestation"]
    assert attestation["year"] == 1350  # "9.-14. yy": dönemin sonu, en geç tanık
    assert "Eski Uygurca (9.-14. yy)" in attestation["source"] and "s. 401" in attestation["source"]
    entry = result["turkic_languages"][0]
    assert entry["lang_code"] == "oui" and not entry.get("meaning_check")
    assert WilkensOldUyghurFetcher.is_local and WilkensOldUyghurFetcher.exact_query_only


def test_donor_chain_is_carried(fake_records):
    (entry,) = WilkensOldUyghurFetcher().fetch("kapı")["turkic_languages"]
    assert entry["etymology"] == "Wilkens: < Sogd. x"
    assert entry["donor_chain"][0]["lang_name"] == "Soğdca"


def test_no_data_returns_empty(monkeypatch):
    monkeypatch.setattr(wilkens, "load_records", lambda path=None: [])
    wou._index.cache_clear()
    try:
        result = WilkensOldUyghurFetcher().fetch("göz")
        assert result["turkic_languages"] == [] and "first_attestation" not in result
    finally:
        wou._index.cache_clear()


# --- indirici --------------------------------------------------------------------

def test_downloader_skips_only_when_files_match_provenance(tmp_path):
    import hashlib
    import importlib.util

    from engine.config import PROJECT_ROOT

    spec = importlib.util.spec_from_file_location(
        "download_wilkens", PROJECT_ROOT / "scripts" / "download_wilkens.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    is_current = module.is_current

    (tmp_path / "a.pdf").write_bytes(b"%PDF-1")
    files = {"a.pdf": {"sha256": hashlib.sha256(b"%PDF-1").hexdigest()}}
    (tmp_path / "_provenance.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    assert is_current(tmp_path)
    (tmp_path / "a.pdf").write_bytes(b"%PDF-2")
    assert not is_current(tmp_path)


# --- gerçek veri (varsa) -----------------------------------------------------------

@pytest.mark.skipif(not (wilkens.WILKENS_DIR / wilkens.JSONL_NAME).exists(),
                    reason="Wilkens verisi yok (make wilkens)")
def test_parsed_dictionary_sanity():
    records = wilkens.load_records()
    assert len(records) > 29000
    by_head = {(r["headword"], r["homonym"]): r for r in records}
    assert by_head[("köz", 1)]["tr"][0].startswith("göz")
    assert by_head[("til", 1)]["de"][0].startswith("Zunge")
    assert [s["lang"] for s in by_head[("ačite", 1)]["donor_chain"]] == ["TochB", "Skt."]


# --- dönem düzeyinde tarih (aralık, üst sınır) --------------------------------------

PERIOD_ATT = {"first_attestation": {"year": 1350, "precision": "period", "range": [800, 1350],
                                    "label": "Eski Uygurca dönemi (9.–14. yy) içinde tanıklı; kesin yer yok",
                                    "source": "Eski Uygurca (9.-14. yy), Wilkens 2021"}}


def _verify(*results, entries=None):
    from engine.nlp.historical_attestation_verifier import HistoricalAttestationVerifier

    return HistoricalAttestationVerifier().verify_attestation("göz", entries or [], list(results))


def test_period_only_attestation_is_a_range_not_a_point():
    out = _verify(PERIOD_ATT)
    assert out["first_attestation_precision"] == "period"
    assert out["first_attestation_range"] == [800, 1350] and out["first_attestation_year"] == 1350
    assert "kesin yer yok" in out["first_attestation_record"]


def test_point_date_always_wins_over_period():
    starling = {"first_attestation": {"year": 732, "source": "Orkh. (Starling #1)"}}
    out = _verify(PERIOD_ATT, starling)
    assert (out["first_attestation_year"], out["first_attestation_precision"]) == (732, "point")


def test_point_date_later_than_period_bound_is_not_first_attestation():
    late = {"first_attestation": {"year": 1876, "source": "Lehce-i Osmânî"}}
    assert _verify(PERIOD_ATT, late)["first_attestation_precision"] == "period"


def test_period_witness_entry_does_not_yield_a_corpus_year():
    entry = {"lang_name": "Eski Uygurca", "word": "köz", "meaning": "göz",
             "source": "Wilkens 2021, Handwörterbuch des Altuigurischen (yerel, Eski Uygurca)",
             "attestation_precision": "period"}
    assert _verify(PERIOD_ATT, entries=[entry])["first_attestation_precision"] == "period"


def test_time_lock_reports_range_as_upper_bound():
    from engine.nlp.hypothesis_validation_protocol import ChronologicalTimeLock

    stage2 = ChronologicalTimeLock().verify("Sanskritçe", _verify(PERIOD_ATT))
    assert stage2["attestation_precision"] == "period" and stage2["attestation_range"] == [800, 1350]
    assert "en geç 1350" in stage2["reason"]


def test_chronology_counts_point_and_period_coverage_separately():
    from engine.evaluation.chronology_eval import agreement, precision_split

    rows = [agreement(732, 732), agreement(1350, 1072), agreement(None, 1072)]
    runs = [{"attestation_precision": "point"}, {"attestation_precision": "period"}, {}]
    split = precision_split(rows, runs)
    assert split["coverage_point"] == round(1 / 3, 4) and split["coverage_period"] == round(1 / 3, 4)
    assert split["within_century|point"] == 1.0
