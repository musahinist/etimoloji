"""
Türki Diller Derin Akraba Kelime ve Türev Ağı (Cognate Network Builder)
Araması yapılan kelimelerin aynı kökten türeyen akraba sözcük kümesini
yalnızca gerçek sözlük kayıtları ve doğrulanmış diyalekt denkliği üzerinden tespit eder.
"""
from typing import Any

from engine.utils.orthography import to_comparison_form

#: Türkiye Türkçesinin KENDİ soy çizgisi: bu dillerde sorguyla aynı karşılaştırma
#: biçimini taşıyan kayıt kelimenin kendi (eski) yazımıdır, akrabası değil
#: (laakal ~ Osmanlıca لااقل, asker ~ عسكر).
_OWN_LINEAGE = frozenset({"tr", "ota", "trk-oat"})

#: Başlıklar hükme göre: alıntıda liste "aynı kökten türeyen akraba" DEĞİLDİR,
#: aynı vericiden başka Türk dillerine ayrı ayrı geçmiş PARALEL ALINTILARDIR.
COGNATE_HEADINGS: dict[str, tuple[str, str]] = {
    "borrowed": ("PARALEL ALINTILAR", "Aynı vericiden başka Türk dillerine geçmiş paralel alıntılar"),
    "default": ("KÖK AKRABA SÖZCÜK AĞI", "Aynı kökten türeyen akraba kelimeler"),
}


def cognate_heading(finding: dict[str, Any]) -> tuple[str, str]:
    """Sıralayıcının hükmüne göre ``(bölüm başlığı, satır etiketi)``."""
    nlp = finding.get("nlp_analysis") or {}
    kind = ((nlp.get("ranked_hypotheses") or {}).get("selected") or {}).get("kind", "")
    return COGNATE_HEADINGS["borrowed" if kind == "borrowed" else "default"]


def _is_own_spelling(entry: dict[str, Any], query_comparison: str) -> bool:
    if entry.get("lang_code") not in _OWN_LINEAGE:
        return False
    forms = {
        str(entry.get("comparison") or ""),
        to_comparison_form(str(entry.get("latin_transliteration") or "")),
        to_comparison_form(str(entry.get("word") or "")),
    }
    return query_comparison in forms


def get_related_cognates(word: str, entries: list[dict[str, Any]] | None = None) -> list[str]:
    """Herhangi bir kelime için yalnızca GERÇEK sözlük kayıtları ve doğrulanmış Türki dil denklerini toplar."""
    w = (word or "").strip().lower()
    if not w:
        return []

    cognate_set = []
    seen = set()
    query_comparison = to_comparison_form(w)

    # 1. Sözlük kayıtlarından (20+ fetcher çıktısı) gerçek kelimeleri topla
    if entries:
        for entry in entries:
            # Kaynak dil kayıtları (Latince facies, İtalyanca faccia) alıntı
            # zinciridir; "aynı kökten türeyen Türki akraba" değildir.
            if entry.get("lang_code") in ("donor", "ai"):
                continue
            # Kelimenin kendi yazımı (Osmanlıca عسكر) akrabası değildir.
            if _is_own_spelling(entry, query_comparison):
                continue
            ew = (entry.get("word") or "").strip()
            ew_clean = ew.lower().lstrip("*")
            if ew_clean and ew_clean != w and ew_clean not in seen:
                seen.add(ew_clean)
                cognate_set.append(ew)

    # ⚠️ BURADA BİR YEDEK VARDI VE KANIT UYDURUYORDU.
    #
    # "3'ten az akraba toplandıysa" `generate_turkic_cognate_candidates()`
    # çağrılıp ÜRETİLMİŞ ses varyantları listeye ekleniyordu — yanında
    # "Sadece gerçek kök haritasından gelen kelimeler" yorumuyla, ki doğru
    # değildi. Bulunmamış biçimler kullanıcıya "aynı kökten türeyen akraba
    # kelimeler" diye sunuluyordu.
    #
    # Ölçüldü (iki kelimede de %100 uydurma):
    #   pervasız -> pervasır, pervasıs, pirvasız, первасыз…  12/12 varyant
    #   herkil   -> herkel, hirkil, härkil, хиркил…          hepsi varyant
    # Bu biçimler o aramada sorgu adayı olarak üretilip HİÇBİR sözlükte
    # bulunamamıştı; yani liste, aramanın başarısızlığını başarı gibi
    # gösteriyordu.
    #
    # Aynı hastalık `hypothesis_validation_protocol` içinde bir kez
    # düzeltilmişti (bkz. oradaki not: "KENDİ ÜRETTİĞİ varyantları
    # sayıyordu, skor daima 0.95 çıkıyordu"); bu ikinci kopyaydı.
    #
    # Akraba listesi artık YALNIZ bulunan tanıklardan gelir. Boş dönmesi
    # dürüst sonuçtur: akraba bulunamadı demektir.
    return cognate_set[:12]

