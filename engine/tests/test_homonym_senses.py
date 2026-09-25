"""Eşsesli tek anlam çapası (`_reconcile_homonym_senses`).

Anlam modeli yerine sözcük örtüşmesi (Jaccard) kullanılır; kayıtlar
indeks yerine sahte kaynaktan gelir. Sınanan: çapa seçimi, verici kaydının
ana anlama göre seçimi, tanık ayırma ve geri alma.
"""

from __future__ import annotations

import re

import pytest

import engine.search_engine as S


def _jaccard(texts, refs):
    def toks(t):
        return set(re.findall(r"[a-z]+", t.lower()))
    return [[len(toks(a) & toks(b)) / max(1, len(toks(a) | toks(b))) for b in refs] for a in texts]


def _rec(lang, origin, donor_lang, form, meaning):
    return {"lang_code": lang, "word": "x", "lexicon_origin": origin, "donor_lang": donor_lang,
            "donor_form": form, "meaning": meaning}


@pytest.fixture
def fake(monkeypatch):
    def install(records):
        monkeypatch.setattr(S, "_own_sense_records", lambda word, entries: records)
        monkeypatch.setattr(S, "_similarity_matrix", _jaccard)
    return install


def test_tek_etimoloji_dokunmaz(fake):
    fake([_rec("tr", "miras", "trk-pro", "*teg-", "till until")])
    entries = [{"lang_code": "kk", "word": "a", "meaning": "squirrel"}]
    out = S._reconcile_homonym_senses("değin", entries, "", "*teg-", "inherited")
    assert out["kept"] == entries and not out["dropped"] and not out["headline"]


def test_baslik_kokunun_etimolojisi_cipa_olur(fake):
    fake([_rec("tr", "miras", "trk-pro", "*teg-", "till until"),
          _rec("tr", "miras", "ota", "*tegiŋ", "squirrel")])
    entries = [{"lang_code": "kk", "word": "a", "meaning": "until to"},
               {"lang_code": "kk", "word": "b", "meaning": "squirrel"},
               {"lang_code": "ky", "word": "c", "meaning": ""}]
    out = S._reconcile_homonym_senses("değin", entries, "", "*teg-", "inherited")
    assert [e["word"] for e in out["dropped"]] == ["b"]
    assert [e["word"] for e in out["kept"]] == ["a", "c"]  # anlamsız tanık kalır


def test_alinti_vericisi_ana_anlama_gore(fake):
    fake([_rec("tr", "alıntı", "en", "cola", "cola plant"),
          _rec("tr", "alıntı", "it", "colla", "laundry starch clothes")])
    out = S._reconcile_homonym_senses("kola", [], "starch for clothes", "İngilizce cola", "borrowed")
    assert out["headline"] == "İtalyanca colla"
    assert "cola plant" in out["other"]


def test_alinti_vericisi_zaten_dogruysa_degismez(fake):
    fake([_rec("tr", "alıntı", "en", "cola", "cola plant"),
          _rec("tr", "alıntı", "it", "colla", "laundry starch clothes")])
    out = S._reconcile_homonym_senses("kola", [], "cola plant", "İngilizce cola", "borrowed")
    assert out["headline"] == ""


def test_erken_ayrilan_akraba_geri_alinir(fake):
    fake([_rec("tr", "miras", "trk-pro", "*ïrak", "far distant remote"),
          _rec("tr", "alıntı", "ar", "عِرَاق", "makam tone music")])
    held = [{"lang_code": "alt", "word": "ыраак", "meaning": "far"}]
    out = S._reconcile_homonym_senses("ırak", [], "makam tone music", "*ırak", "inherited", held_out=held)
    assert out["readmitted"] == held
