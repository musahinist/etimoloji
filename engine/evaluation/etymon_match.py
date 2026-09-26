"""
Gösterilen verici biçimi doğru etimon mu? — "biçim kesinliği" ölçütünün eşleştiricisi (9n).

Referans: altının etimon çevriyazısı (TDK ``lisan`` etimonu, Nişanyan kaynağı) + TAM sözlük
indeksindeki Türkçe maddenin Wiktionary şablon argümanı (``donor_form``, ör. عَسْكَر) ve etimoloji
metnindeki "dil adı + biçim (çevriyazı)" çiftleri. Yalnız PUANLAMADA kullanılır; etiketleyici görmez.

Eşleşme ("aynı sözcük"): Arap yazısında harekesiz iskelet eşitliği (``ال`` farkı serbest), Latin/Yunan
yazısında aksansız eşitlik, çevriyazıda karşılaştırma biçimi eşitliği ya da ≥ 4 harfte normalleştirilmiş
düzenleme uzaklığı ≤ 0,20. Yalnız ünsüz iskeleti tutan türev/şans (asker ~ عسكري) eşleşme SAYILMAZ.
Tanı ve el denetimi: ``data/cache/work/donor9n/PREREG.md``.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from collections.abc import Iterable

LANG_NAMES = {
    "Arabic": "ar", "Arapça": "ar", "Persian": "fa", "Farsça": "fa", "French": "fr", "Fransızca": "fr",
    "Italian": "it", "İtalyanca": "it", "Venetian": "it", "Venedikçe": "it", "Genoese": "it",
    "Greek": "el", "Yunanca": "el", "Rumca": "el", "Armenian": "hy", "Ermenice": "hy",
}
_REF = re.compile(r"(Ancient Greek|Byzantine Greek|Medieval Greek|Modern Greek|Classical Persian|"
                  + "|".join(sorted(LANG_NAMES, key=len, reverse=True))
                  + r")\s+([^\s(),;:.]+)(?:\s+\(([^,);]+))?")
_ARABIC = re.compile(r"[؀-ۿ]")
_NON_LATIN = re.compile(r"[^\x00-ɏḀ-ỿ]")


def _lang_of(name: str) -> str:
    for key, code in LANG_NAMES.items():
        if key in name:
            return code
    return ""


def _plain(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text or "") if not unicodedata.combining(c)).casefold()


def reference_forms(word: str, connection: sqlite3.Connection) -> set[tuple[str, str]]:
    """TAM indeksteki Türkçe maddenin (dil, biçim) referansları."""
    out: set[tuple[str, str]] = set()
    rows = connection.execute(
        "SELECT donor_lang, donor_form, etymology FROM entries WHERE lang_code='tr' AND word=?", (word,))
    for lang, form, ety in rows:
        if form:
            out.add((lang or "", re.sub(r"<[^<>]*>", "", form).strip()))
        for m in _REF.finditer(ety or ""):
            code = _lang_of(m.group(1))
            out.add((code, m.group(2)))
            if m.group(3):
                out.add((code, m.group(3).strip()))
    return {(lang, f) for lang, f in out if f and f not in ("from", "word", "term")}


def source_translits(*sources: str) -> list[str]:
    """Altın kaynak alanlarından ("Arapça ʿasker", "Arapça ʿabd + ʿāciz") çevriyazılar."""
    out: list[str] = []
    for src in sources:
        parts = (src or "").split(" ", 1)
        if len(parts) == 2:
            out += [p.strip() for p in re.split(r"\s*\+\s*|,", parts[1]) if p.strip()]
    return out


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def is_etymon(word: str, comparison: str, refs: Iterable[tuple[str, str]], translits: Iterable[str]) -> bool:
    """Gösterilen verici biçimi (``word``, karşılaştırma biçimi ``comparison``) referansla aynı sözcük mü?"""
    from engine.nlp.donor_proximity import script_skeleton
    from engine.utils.orthography import to_comparison_form

    refs = list(refs)
    arabic = bool(_ARABIC.search(word or ""))
    skeleton = script_skeleton(word) if arabic else ""
    for _, form in refs:
        if arabic and _ARABIC.search(form):
            ref = script_skeleton(form)
            if ref and (ref == skeleton or ref.removeprefix("ال") == skeleton or skeleton.removeprefix("ال") == ref):
                return True
        elif not arabic and _plain(form) == _plain(word):
            return True
    latin = [to_comparison_form(f) for _, f in refs if not _NON_LATIN.search(f)]
    latin += [to_comparison_form(t) for t in translits]
    for x in (x for x in latin if x):
        if x == comparison or (len(x) >= 4 and _lev(x, comparison) / max(len(x), len(comparison)) <= 0.2):
            return True
    return False
