"""
Türki Diller Morfolojik Analizör ve Akıllı Kök Ayrıştırıcı (Turkic Stemmer & Morphological Analyzer)
Karmaşık türemiş ve çekim ekli kelimeleri köklerine ayrıştırır. Alıntı kelimelerde yapay ek kesimini engeller.
"""

# Öz Türkçede söz başında KESİNLİKLE bulunmayan (veya aşırı nadir olan) ünsüzler
NON_TURKIC_INITIAL_CONSONANTS = ['f', 'h', 'p', 'v', 'j', 'z']

# Türkçe ve Türki diller yapım ve çekim ekleri
TURKIC_SUFFIXES = [
    "cılık", "cilik", "çılık", "çilik", "culuk", "cülük",
    "daş", "deş", "taş", "teş",
    "lik", "lık", "luk", "lük", "ci", "cı", "cu", "cü", "çi", "çı",
    "li", "lı", "lu", "lü",
    "siz", "sız", "suz", "süz",
    "gi", "gı", "gu", "gü", "ki", "kı", "ku", "kü",
    "sel", "sal", "gil",
    "ler", "lar", "dan", "den", "tan", "ten",
    "daki", "deki", "taki", "teki",
    "mak", "mek", "ma", "me", "ış", "iş", "uş", "üş"
]

#: Wiktionary çekim gloss'larının dilbilgisel belirteçleri. Bu belirteci
#: TAŞIYAN satır bir sözlükbirim tanımı değil, başka bir lemmanın çekimidir:
#: "third-person singular indicative aorist of canlanmak", "verbal noun of
#: yapılmak", "locative singular of bar", "inflection of yağ:".
#: ⚠️ Yalnız " of " aramak yetmez — "palm of hand" gibi GERÇEK tanımlar da
#: onu içerir; bu yüzden dilbilgisi terimi aranır.
INFLECTION_GLOSS_MARKERS: tuple[str, ...] = (
    "inflection of", "plural of", "singular of", "verbal noun of",
    "-person", "aorist of", "imperative of", "participle of",
    "dative of", "accusative of", "genitive of", "ablative of",
    "locative of", "nominative of", "optative of", "necessitative of",
    "past of", "present of", "future of", "negative of",
    "causative of", "passive of", "reflexive of", "reciprocal of",
)


def is_lexeme(word: str) -> bool:
    """Girdi bir sözlükbirim mi? Ek, sayı, öbek ve özel ad değilse evet.

    Etimoloji üretilecek girdinin sözlükbirim olması gerekir. Kapı yokken
    ağız havuzunun ilk beş kaydı ``-acağım``, ``-akalmak``, ``-amaç``,
    ``-anak``, ``-arak`` (hepsi **ek**) geliyordu ve motor bunların ikisine
    "güçlü aday" (0,40) dedi. Havuzda ayrıca sıra sayıları (``1'inci``),
    özel adlar (``Aydoğan``) ve öbekler (``av köpeği``) var.

    ⚠️ Büyük harf denetimi, kelime küçük harfe ÇEVRİLMEDEN uygulanmalıdır.
    """
    w = (word or "").strip()
    if len(w) < 2:
        return False
    if w.startswith("-") or w.endswith("-"):
        return False  # yapım/çekim eki
    if " " in w or "\t" in w:
        return False  # çok sözcüklü öbek
    if w[:1].isupper():
        return False  # özel ad
    if any(ch.isdigit() for ch in w):
        return False  # sayı / sıra sayısı
    if "'" in w or "’" in w:
        return False  # kesme işaretli çekim
    return True


def is_inflection_gloss(gloss: str) -> bool:
    """Bu gloss bir çekim tanımı mı (kelime başka bir lemmanın biçimi mi)?

    Sözlük indeksinin %69'u çekim satırı taşıdığı için, indekste bir biçimin
    "bulunması" tek başına o biçimin bir kök olduğunu göstermez.
    """
    g = (gloss or "").lower()
    return any(marker in g for marker in INFLECTION_GLOSS_MARKERS)


def analyze_morphology(word: str) -> tuple[str, list[str]]:
    w = word.strip().lower()

    # 1. Alıntı Kelime Koruması: f-, h-, p-, v-, j-, z- ile başlayan kelimelerde mekanik ek kesimi yapma!
    if w and w[0] in NON_TURKIC_INITIAL_CONSONANTS:
        return w, []

    detected_suffixes = []
    stem = w

    # 2. Mekanik ek kesimi (sadece geçerli Öz Türkçe kök adayları için)
    for suf in TURKIC_SUFFIXES:
        if stem.endswith(suf) and len(stem) - len(suf) >= 3:
            detected_suffixes.append("-" + suf)
            stem = stem[:-len(suf)]

    return stem, detected_suffixes
