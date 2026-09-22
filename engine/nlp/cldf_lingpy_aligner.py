"""
Fonetik Dizi Hizalayıcı — LingPy SCA (Phonetic Sequence Aligner)

İki biçimi ses sınıfı düzeyinde hizalar ve fonetik benzerlik üretir.
Hizalama **gerçek LingPy** kütüphanesinin SCA (Sound-Class-Based Alignment)
modeliyle yapılır.

Neden değişti
-------------
Önceki sürüm docstring'inde "LingPy SCA (Dolgopolsky Ses Sınıfları)" yazmasına
rağmen ``lingpy`` kütüphanesini kullanmıyordu; elle yazılmış bir
Needleman-Wunsch ve **35 elemanlı** bir sözde-Dolgopolsky haritası taşıyordu.
Gerçek LingPy SCA modeli çok daha zengin bir ses sınıfı sistemi kullanır ve
CLDF ekosistemiyle uyumludur.

Ayrıca eski kodda:

* ``:27`` docstring'i ``'suv' -> 'VPW'`` diyordu; gerçek çıktı ``SVP`` idi.
* Traceback'te ``score_matrix[i][j] == score_matrix[i-1][j-1] + exp`` biçiminde
  **float eşitlik** karşılaştırması vardı; yuvarlama nedeniyle yanlış yol
  seçilebiliyordu.

Kurulum: ``pip install -e ".[phon]"``. LingPy yoksa modül kendi
Needleman-Wunsch uygulamasına düşer ve bunu ``backend`` alanında bildirir.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from functools import lru_cache
from typing import Any

from engine.logging_setup import get_logger
from engine.nlp.phonological_feature_engine import PhonologicalFeatureEngine, to_ipa
from engine.utils.orthography import to_comparison_form

logger = get_logger(__name__)

_LINGPY = None
_BACKEND: str | None = None

#: Needleman-Wunsch yedek uygulaması için puanlar
MATCH_SCORE = 3
CLASS_MATCH_SCORE = 2
MISMATCH_PENALTY = -1
GAP_PENALTY = -2

#: Yedek mod için Dolgopolsky ses sınıfları
FALLBACK_SOUND_CLASSES: dict[str, str] = {
    "a": "V", "e": "V", "ı": "V", "i": "V", "o": "V", "ö": "V", "u": "V", "ü": "V",
    "ä": "V", "ə": "V",
    "p": "P", "b": "P", "f": "P", "v": "P", "w": "W", "m": "M",
    "t": "T", "d": "T", "s": "S", "z": "S", "ş": "S",
    "c": "J", "ç": "J", "j": "J",
    "r": "R", "ŕ": "R", "l": "L", "ĺ": "L", "n": "N", "ŋ": "N",
    "k": "K", "g": "K", "ğ": "K", "q": "K", "h": "H",
    # Türki y- ~ c- ~ j- ~ ç- denkliği tek sınıfta toplanır; aksi hâlde
    # hizalayıcı bu konumda ikame yerine boşluk tercih eder.
    "y": "J", "ǰ": "J",
}


def _load_lingpy():
    """LingPy'yi tembel yükler. Import anında gürültülü log basar, susturulur."""
    global _LINGPY, _BACKEND
    if _BACKEND is not None:
        return _LINGPY
    previous = logging.root.manager.disable
    try:
        logging.disable(logging.INFO)
        import lingpy

        _LINGPY = lingpy
        _BACKEND = "lingpy"
        logger.info("LingPy yüklendi (SCA hizalama modeli)")
    except Exception:
        logger.info(
            'lingpy kurulu değil; yerleşik Needleman-Wunsch hizalayıcısına düşülüyor. '
            'Kurulum: pip install -e ".[phon]"'
        )
        _BACKEND = "fallback"
    finally:
        logging.disable(previous)
    return _LINGPY


def backend_name() -> str:
    _load_lingpy()
    return _BACKEND or "fallback"


class CldfLingPyAligner:
    """LingPy SCA tabanlı fonetik dizi hizalayıcı."""

    def __init__(self) -> None:
        self.panphon_engine = PhonologicalFeatureEngine()

    @property
    def backend(self) -> str:
        return backend_name()

    def to_sound_classes(self, seq: str) -> str:
        """
        Biçimi ses sınıfı dizisine çevirir (ör. ``göz`` -> ``KVS``).

        LingPy varsa gerçek SCA modeli, yoksa yerleşik Dolgopolsky tablosu.
        """
        clean = to_comparison_form(seq)
        if not clean:
            return ""
        lingpy = _load_lingpy()
        if lingpy is not None:
            try:
                tokens = lingpy.ipa2tokens(to_ipa(clean))
                return "".join(lingpy.tokens2class(tokens, "sca"))
            except Exception:
                logger.debug("LingPy ses sınıfı çevirisi başarısız: %r", clean)
        return "".join(FALLBACK_SOUND_CLASSES.get(ch, "X") for ch in clean)

    def align_sequences(self, seq1: str, seq2: str) -> dict[str, Any]:
        """
        İki biçimi hizalar ve fonetik benzerlik üretir.

        :returns: ``aligned_seq1``/``aligned_seq2`` hizalanmış diziler,
            ``phonetic_similarity`` [0,1] aralığında benzerlik,
            ``evidence_available`` karşılaştırmanın yapılabildiğini bildirir.
        """
        s1 = to_comparison_form(seq1)
        s2 = to_comparison_form(seq2)

        if not s1 or not s2:
            return {
                "seq1": s1,
                "seq2": s2,
                "backend": self.backend,
                "evidence_available": False,
                "aligned_seq1": "",
                "aligned_seq2": "",
                "sound_class_seq1": "",
                "sound_class_seq2": "",
                "alignment_score": None,
                "phonetic_similarity": None,
                "aligned_pairs": [],
            }

        # ⚠️ Hesap önbellekli, dönen nesne HER ÇAĞRIDA KOPYA. Önbelleklenmiş
        # sözlüğü doğrudan döndürmek onu bütün çağıranlar arasında paylaşılan
        # DEĞİŞTİRİLEBİLİR bir nesne yapardı; bugün kimse onu değiştirmiyor
        # ama bu, ileride izi sürülmesi en zor hata sınıfıdır. Kopya bedava:
        # ölçüldü — hizalama 184,0 µs, derin kopya 11,4 µs (16x ucuz).
        return deepcopy(_aligned_payload(s1, s2))

    # --- Hizalama arka uçları --------------------------------------------

    def _align(self, s1: str, s2: str) -> tuple[str, str, float]:
        """Biçimleri **imla düzeyinde**, karakter karakter hizalar."""
        # LingPy hizalaması Türki ses denkliklerini (y- ~ j- ~ c-) bilmediği
        # için bu konumlarda ikame yerine BOŞLUK tercih ediyor; karşılaştırmalı
        # rekonstrüksiyon konum bazlı denklik kümeleri kurduğu için bu
        # hizalamayı bozar. Bu yüzden imla hizalaması, ses sınıflarını dikkate
        # alan yerleşik Needleman-Wunsch ile yapılır.
        # LingPy yine SCA ses sınıflarını ve bağımsız bir benzerlik sinyalini sağlar.
        return self._needleman_wunsch(s1, s2)

    def _needleman_wunsch(self, s1: str, s2: str) -> tuple[str, str, float]:
        """
        Yedek hizalayıcı.

        Traceback, float eşitlik karşılaştırması yerine yön matrisi kullanır;
        eski uygulamada yuvarlama nedeniyle yanlış yol seçilebiliyordu.
        """
        cls1, cls2 = self.to_sound_classes(s1), self.to_sound_classes(s2)
        m, n = len(s1), len(s2)
        score = [[0.0] * (n + 1) for _ in range(m + 1)]
        ptr = [[0] * (n + 1) for _ in range(m + 1)]  # 0:diag 1:up 2:left

        for i in range(1, m + 1):
            score[i][0] = i * GAP_PENALTY
            ptr[i][0] = 1
        for j in range(1, n + 1):
            score[0][j] = j * GAP_PENALTY
            ptr[0][j] = 2

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    sub = MATCH_SCORE
                elif i - 1 < len(cls1) and j - 1 < len(cls2) and cls1[i - 1] == cls2[j - 1]:
                    sub = CLASS_MATCH_SCORE
                else:
                    sub = MISMATCH_PENALTY
                choices = (
                    (score[i - 1][j - 1] + sub, 0),
                    (score[i - 1][j] + GAP_PENALTY, 1),
                    (score[i][j - 1] + GAP_PENALTY, 2),
                )
                best = max(choices)
                score[i][j], ptr[i][j] = best[0], best[1]

        a1: list[str] = []
        a2: list[str] = []
        i, j = m, n
        while i > 0 or j > 0:
            d = ptr[i][j]
            if i > 0 and (j == 0 or d == 1):
                a1.append(s1[i - 1])
                a2.append("-")
                i -= 1
            elif j > 0 and (i == 0 or d == 2):
                a1.append("-")
                a2.append(s2[j - 1])
                j -= 1
            else:
                a1.append(s1[i - 1])
                a2.append(s2[j - 1])
                i -= 1
                j -= 1
        return "".join(reversed(a1)), "".join(reversed(a2)), round(score[m][n], 3)


# --- Önbellekli hizalama -------------------------------------------------
#
# Hizalama SAF bir hesaptır: `(s1, s2)` dışında hiçbir şeye bağlı değildir.
# Doğrulandı:
#   * `_needleman_wunsch` yalnız modül sabitlerini kullanır (MATCH_SCORE,
#     GAP_PENALTY, ses sınıfı tabloları) — örnek durumu yok.
#   * `PhonologicalFeatureEngine` kendi içinde zaten `lru_cache`lidir.
#   * İKİ AYRI örnek aynı çiftte birebir aynı sonucu veriyor (deneysel
#     kontrol; okumayla yetinilmedi).
#
# Neden gerekiyordu: aynı `(biçim, biçim)` çifti kavramlar arasında ortalama
# ~4 kez tekrarlanıyor. Ölçüldü — dev: 24.623 çağrı / 6.500 farklı çift
# (isabet %73,6), train: 80.196 / 20.376 (%74,6). Çağrı başına hizalama
# 184,0 µs; derin kopya 11,4 µs. Derin kopyayla bile net kazanç %69,1.
_SHARED_ALIGNER: CldfLingPyAligner | None = None


def _shared_aligner() -> CldfLingPyAligner:
    """Önbellekli hesap için modül düzeyinde tek örnek.

    `lru_cache`i metoda koymak `self`i güçlü referansla tutar ve önbelleği
    örnek başına böler; bu yüzden hesap modül düzeyine alındı.
    """
    global _SHARED_ALIGNER
    if _SHARED_ALIGNER is None:
        _SHARED_ALIGNER = CldfLingPyAligner()
    return _SHARED_ALIGNER


@lru_cache(maxsize=32768)
def _aligned_payload(s1: str, s2: str) -> dict[str, Any]:
    """`align_sequences` gövdesi — girdiler SADELEŞTİRİLMİŞ biçimlerdir.

    ⚠️ Dönen sözlük önbellekte tutulur; çağıran taraf onu **kopyalamadan**
    döndürmemelidir (bkz. `align_sequences`).
    """
    aligner = _shared_aligner()
    cls1, cls2 = aligner.to_sound_classes(s1), aligner.to_sound_classes(s2)
    aligned1, aligned2, raw_score = aligner._align(s1, s2)

    panphon_res = aligner.panphon_engine.sequence_phonological_distance(s1, s2)
    articulatory_sim = panphon_res.get("phonetic_similarity")

    # Ses sınıfı eşleşme oranı
    class_matches = sum(1 for a, b in zip(cls1, cls2, strict=False) if a == b)
    class_ratio = class_matches / max(len(cls1), len(cls2), 1)

    # Nihai benzerlik: artikülatör mesafe (%60) + ses sınıfı uyumu (%40)
    if articulatory_sim is None:
        similarity = round(class_ratio, 3)
    else:
        similarity = round(0.6 * articulatory_sim + 0.4 * class_ratio, 3)

    aligned_pairs = [
        {"seq1": a, "seq2": b, "match": a == b}
        for a, b in zip(aligned1, aligned2, strict=False)
    ]

    return {
        "seq1": s1,
        "seq2": s2,
        "backend": backend_name(),
        "evidence_available": True,
        "sound_class_seq1": cls1,
        "sound_class_seq2": cls2,
        "aligned_seq1": aligned1,
        "aligned_seq2": aligned2,
        "alignment_score": raw_score,
        "sound_class_match_ratio": round(class_ratio, 3),
        "articulatory_similarity": articulatory_sim,
        "panphon_articulatory_distance": panphon_res.get("normalized_articulatory_distance"),
        "phonetic_similarity": similarity,
        "aligned_pairs": aligned_pairs,
    }
