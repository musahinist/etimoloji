"""
Türetme çözümleyicisi — türemiş Türkçe kelimeden sözlük köküne (ZA5 + Z3).

``historical_morphology`` ekleri düz listeyle, sondan açgözlü söker; ek
SIRASI, ünlü uyumu ve ünsüz uyumu denetlenmez. Bu modül Zemberek-NLP'nin
ek şablon dilini (``SurfaceTransition``/``SuffixTemplateTokenizer``) YALNIZ
yapım ekleri alt kümesi için yeniden yazar; Zemberek koduna bağımlılık yok:

======  ==============================================================
``I``   dörtlü uyumlu ek ünlüsü (ı/i/u/ü)
``A``   ikili uyumlu ek ünlüsü (a/e)
``+I``  ``+A``: önceki harf ünlüyse düşen tampon ünlü (``kazı+k``)
``>c``  önceki harf ötümsüzse sertleşen ünsüz (c→ç, d→t, g→k)
``~k``  ardından ünlü gelirse yumuşayan ünsüz (k→ğ, ç→c)
======  ==============================================================

Çözümleme üretimin tersidir: yüzeyin her ön ekini aday kök sayar, kök
Zemberek sözlüğünde (``root_variants``; ünsüz yumuşaması/ünlü düşmesi
değişmiş gövdesiyle) varsa, POS durum makinesi üzerinde ek zincirini
üretip kalan yüzeyle karşılaştırır. POS uyumu şarttır: fiilden ad yapan ek
(``-Im``, ``-Ik``, ``-gIn``…) yalnız FİİL köküne, addan ad/fiil yapan ek
(``+lIk``, ``+lA``…) yalnız AD/SIFAT köküne. Sıra denetimi durum
makinesinden gelir (``-lA`` fiil yapar, ardından ``-Iş`` gelebilir;
``+CI`` ad yapar, ardından ``-Im`` gelemez). Kökü sözlükte doğrulanmayan
bölme ÜRETİLMEZ (eşsesli/yanlış bölme riski).

⚠️ Döngüsellik: Zemberek sözlüğü TDK tabanlıdır; yalnız madde başı ve
sözcük türü okunur, köken bilgisi yok.

⚠️ OLUMSUZ SONUÇ — başlık köküne BAĞLANMADI (2026-09-26; ön kayıt
``data/cache/work/deriv/PREREG.md``; `make eval-headline` düzeni, önbellekli
koşudan benzetim — değişim yalnız "kök belirlenemedi" maddelerine dokunur):

* Hata payı: Starling kapalı 140 başlık hatasının ~38'i türemiş biçim
  (kök+yapım eki). ~19'unda başlık kelimenin KENDİ kaydından gelen türemiş
  biçimdir (``adım`` *ātïm ↔ *āt; kural gereği dokunulmaz), ~19'unda kök
  hiç belirlenemedi (``alık``, ``bitiş``, ``salgın``, ``tatık``…).
* Yedek: kök yoksa çözümün kökü için Starling (portföydeyse) sonra yerel
  indeks Proto-Türkçe kaydı. Seçim yarısında (118) hiçbir yapılandırma
  (sığ/derin × çıplak kök evet/hayır) fark yaratmadı (+0 −0).
  Rapor yarısı (122), Starling kapalı: tam 0,434 -> 0,426 (+0 −1, McNemar
  p=1,0; kayıp ``yelme``: Starling *jElme'yi kendi kökü sayar, çözüm yel+mA).
  Starling açık ve savelyev dev (32): değişmez. Kabul ölçütü karşılanmadı.
* Neden: kök doğru bulunsa da (``alık`` al, ``salgın`` sal, ``tatık``
  tat, ``bitiş`` bit) indeksin Proto-Türkçe biçimi Starling yazımıyla tam
  eşleşmiyor (*āl↔*ăl, *sāl↔*sal, *tāt-↔*dāt, *bït↔*büt): hata kök
  bulmada değil, başvuru yazımı/ünlü niceliği farkında.
* Çözümleyici kesinliği (Starling TRK, Zemberek'te bulunan maddeler): 241
  çözümlemenin 111'inde sığ çözümün kökü Starling köküyle uyumlu, 70'inde
  Starling kelimeyi basit sayıyor (yanlış bölme: ``burun``→bur+In,
  ``bulut``→bul+It, ``gece``→ge+CA, ``yer``→ye+Ar), 60'ında başka kök.
  Starling'in kök-ek diye böldüğü 76 maddenin 23'ü çözümlendi, 14'ünde kök
  doğru. Kısa (1 heceli) kök + tek ünlülü ek bölmeleri en riskli sınıftır.

⚠️ TEKRAR, gelenek-denk ölçütle — yine BAĞLANMADI (2026-09-26; ön kayıt
``data/cache/work/noroot/PREREG.md``, ``engine_halves`` A seçim / B rapor):

* "Kök yok" sınıflaması (Starling kapalı, 45 madde + 5 alıntı hükmü): en
  büyük sınıf (23) yerel veride HİÇ etimolojik kayıt olmayan kelimeler
  (``uçarı``, ``koçkar``, ``tansık``, ``çaput``, ``tın``…); tek bilgi
  kaynağı Starling, Starling açıkken 23'ü de doğru. Probe: 37 "yok"
  maddesinin 29'unda indekste hiç Türk dili tanığı yok; tanık olup kök
  yasağına (``withheld_reconstruction``) takılan 0. Sonra: 8 başlık
  ``*<kelime>`` (yazım/eşsesli), 5 miras kaydı var ama trk-pro bağı yok
  (``apaçık``, ``köstek``, ``çapa``…), 5 çözümleyici kökü yanlış (``boyan``
  boya, ``sinle`` sin), yalnız 4 bu yedeğin çözebildiği sınıf (``alık``,
  ``salgın``, ``karım``, ``tatık``).
* Seçim (A): derin çözüm (gelenek 69 -> 71). Rapor (B, n=120): tam
  48 -> 47 (−1 ``yelme``), gelenek 64 -> 65, ikisi de McNemar p=1,0.
  savelyev dev gelenek −1 (``atla`` -> *at); 50 sahte kelimenin 1'ine kök
  yazılır (``akçık`` -> *āk). Kabul ölçütü (tam VE gelenek anlamlı artış)
  karşılanmadı. Kalan açığı kapatmak veri gerektirir (Starling dışı kök
  kaynağı), çözümleyici değil.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from engine.nlp import root_variants
from engine.utils.orthography import to_comparison_form

_VOWELS = frozenset("aeıioöuü")
_BACK = frozenset("aıou")
_ROUND = frozenset("oöuü")
_VOICELESS = frozenset("çfhkpsşt")
_DEVOICE_FIRST = {"c": "ç", "d": "t", "g": "k"}
_VOICE_LAST = {"k": "ğ", "ç": "c"}

#: Sözcük türü sınıfları: Zemberek POS -> durum.
_NOMINAL = {"Noun": "N", "Adj": "N"}
_VERB = {"Verb": "V"}

#: (etiket, şablon, girdi durumu, çıktı durumu). Kaynak: ZEMBEREK_ALGO §1
#: yapım eki listesi + Zemberek ``TurkishMorphotactics`` türetim geçişleri.
SUFFIXES: tuple[tuple[str, str, str, str], ...] = (
    # addan ad / sıfat
    ("+lIk", "lI~k", "N", "N"),
    ("+CI", ">cI", "N", "N"),
    ("+lI", "lI", "N", "N"),
    ("+sIz", "sIz", "N", "N"),
    ("+CA", ">cA", "N", "N"),
    ("+CIk", ">cI~k", "N", "N"),
    ("+DAş", ">dAş", "N", "N"),
    ("+sAl", "sAl", "N", "N"),
    # addan fiil
    ("+lA", "lA", "N", "V"),
    ("+lAş", "lAş", "N", "V"),
    ("+lAn", "lAn", "N", "V"),
    # fiilden ad / sıfat
    ("-Im", "+Im", "V", "N"),
    ("-Ik", "+I~k", "V", "N"),
    ("-I", "+I", "V", "N"),
    ("-Iş", "+Iş", "V", "N"),
    ("-gI", ">gI", "V", "N"),
    ("-gIn", ">gIn", "V", "N"),
    ("-gAn", ">gAn", "V", "N"),
    ("-gAç", ">gA~ç", "V", "N"),
    ("-gIç", ">gI~ç", "V", "N"),
    ("-mAn", "mAn", "V", "N"),
    ("-Ar", "+Ar", "V", "N"),
    ("-It", "+It", "V", "N"),
    ("-In", "+In", "V", "N"),
    ("-mA", "mA", "V", "N"),
    ("-Ak", "+A~k", "V", "N"),
    ("-Inç", "+Inç", "V", "N"),
)

#: Azami ek sayısı (kelime başına).
MAX_SUFFIXES = 3
#: Kök en az bu kadar harf.
MIN_ROOT = 2


def _tokens(template: str) -> list[str]:
    out, i = [], 0
    while i < len(template):
        if template[i] in "+>~" and i + 1 < len(template):
            out.append(template[i : i + 2])
            i += 2
        else:
            out.append(template[i])
            i += 1
    return out


def _last_vowel(text: str) -> str:
    for ch in reversed(text):
        if ch in _VOWELS:
            return ch
    return ""


def _harmonize(kind: str, prev: str) -> str:
    v = _last_vowel(prev) or "e"
    back, rnd = v in _BACK, v in _ROUND
    if kind == "A":
        return "a" if back else "e"
    return ("u" if rnd else "ı") if back else ("ü" if rnd else "i")


def surface(template: str, prev: str, *, before_vowel: bool = False) -> str:
    """Şablonun ``prev`` gövdesinden sonraki yüzeyi (Zemberek ``generateSurface``).

    ``before_vowel``: ardından ünlüyle başlayan ek geliyor (``~k`` -> ``ğ``).
    """
    out = ""
    toks = _tokens(template)
    for idx, tok in enumerate(toks):
        cur = prev + out
        if tok in ("+I", "+A"):
            if cur and cur[-1] in _VOWELS:
                continue
            out += _harmonize(tok[1], cur)
        elif tok in ("I", "A"):
            out += _harmonize(tok, cur)
        elif tok[0] == ">":
            ch = tok[1]
            out += _DEVOICE_FIRST.get(ch, ch) if cur and cur[-1] in _VOICELESS else ch
        elif tok[0] == "~":
            ch = tok[1]
            last = idx == len(toks) - 1
            out += _VOICE_LAST.get(ch, ch) if (last and before_vowel) else ch
        else:
            out += tok
    return out


@dataclass(frozen=True)
class Analysis:
    """Bir çözüm: sözlük kökü, kökün türü, ek zinciri, bölünmüş yüzey."""

    root: str
    root_pos: str
    suffixes: tuple[str, ...]
    segments: tuple[str, ...]

    @property
    def formula(self) -> str:
        return " + ".join((self.root, *self.suffixes))


def _root_surfaces(prefix: str) -> list[tuple[str, str, bool]]:
    """``prefix`` yüzeyine denk düşen (kök, durum, yalnız-ünlü-önünde) üçlüleri.

    Değişmiş gövde (``kitab``, ``burn``) yalnız ünlüyle başlayan ek önünde
    geçerlidir.
    """
    out: list[tuple[str, str, bool]] = []
    for item in root_variants.lookup(prefix):
        state = _NOMINAL.get(item.pos) or _VERB.get(item.pos)
        if state:
            out.append((item.root, state, False))
    for root in root_variants._lexicon()[1].get(prefix, ()):
        for item in root_variants.lookup(root):
            state = _NOMINAL.get(item.pos) or _VERB.get(item.pos)
            if state and root_variants.modified_root(item, progressive=False) == prefix:
                out.append((item.root, state, True))
    return list(dict.fromkeys(out))


def _chains(word: str, stem: str, state: str, need_vowel: bool, depth: int):
    """``stem``den ``word``e giden ek zincirleri (etiketler, yüzeyler)."""
    rest = word[len(stem):]
    if not rest:
        if not need_vowel:
            yield (), ()
        return
    if depth >= MAX_SUFFIXES:
        return
    for label, template, src, dst in SUFFIXES:
        if src != state:
            continue
        for before_vowel in (False, True):
            surf = surface(template, stem, before_vowel=before_vowel)
            if before_vowel and surf == surface(template, stem):
                continue  # ~ yok; aynı yüzey iki kez denenmesin
            if not surf or not rest.startswith(surf):
                continue
            if need_vowel and surf[0] not in _VOWELS:
                continue
            for labels, segs in _chains(word, stem + surf, dst, before_vowel, depth + 1):
                yield (label, *labels), (surf, *segs)


@lru_cache(maxsize=4096)
def analyze(word: str) -> tuple[Analysis, ...]:
    """Yüzey -> sözlükte doğrulanmış (kök, ek zinciri) çözümleri.

    Sıra: en uzun kök önce (sığ çözüm), eşitlikte daha az ek. En az bir ek
    soyulmuş çözümler döner; kelimenin kendisi sözlükte olsa da (``alık``
    Zemberek maddesi) altındaki çözüm listelenir. Sözlük yoksa boş.
    """
    w = to_comparison_form(word)
    if not w or not root_variants.lexicon_available():
        return ()
    # Mastar (`kolaylaşmak`) çekim ekidir; gövde çözümlenir.
    if len(w) > 5 and w.endswith(("mak", "mek")):
        w = w[:-3]
    found: list[Analysis] = []
    for i in range(len(w) - 1, MIN_ROOT - 1, -1):
        prefix = w[:i]
        for root, state, need_vowel in _root_surfaces(prefix):
            for labels, segs in _chains(w, prefix, state, need_vowel, 0):
                if labels:
                    pos = "Verb" if state == "V" else "Noun"
                    found.append(Analysis(root, pos, labels, (prefix, *segs)))
    found = list(dict.fromkeys(found))
    found.sort(key=lambda a: (-len(a.segments[0]), len(a.suffixes)))
    return tuple(found)
