"""
Çoklu dizi hizalama (multiple sequence alignment) — rekonstrüksiyonun temeli.

**Neden ayrı bir modül?** Önceki rekonstrüktör her tanığı tek tek *çapa
kelimeye* hizalıyordu ve sütunları ``[[] for _ in anchor]`` ile kuruyordu.
Bunun iki sonucu vardı:

1. **Ata biçim çapa kelimenin uzunluğuna kilitliydi.** Çapadan uzun tanıklar
   budanıyordu: ``*sub`` yerine ``*su`` üretiliyordu.
2. **Çapa ayrıcalıklıydı.** Modern Türkiye Türkçesi biçimi, Çuvaşça veya
   Halaçça tanıktan daha belirleyici oluyordu — oysa karşılaştırmalı yöntemde
   arkaik tanık daha ağır basar.

Doğrusu: **bütün tanıklar birbirine** hizalanır, ata biçmin uzunluğu
hizalamanın genişliğinden çıkar.

LingPy'nin SCA tabanlı ilerlemeli hizalaması (``Multiple.prog_align``)
kullanılır; kurulu değilse merkez-yıldız (center-star) yedeği devreye girer.
"""

from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from engine.logging_setup import get_logger

logger = get_logger(__name__)

GAP = "-"

#: **İmla -> IPA eşlemesi, yalnız hizalama için.**
#:
#: LingPy'nin SCA ses sınıfları IPA varsayar. Karşılaştırma biçimimiz ise
#: Türkçe imlasına yakındır ve bu iki gelenek bazı harflerde ÇAKIŞIR:
#:
#: * ``y`` IPA'da ön yuvarlak ÜNLÜdür [y]; LingPy ``yol``un ``y``sini ünlü
#:   sayıp ünsüz sütununa hiç koymuyordu. Sonuç: ``*jol`` yerine ``*yol``.
#: * ``c`` IPA'da damaksıl patlayıcıdır, Türkçe imlada [dʒ].
#: * ``ı``, ``ö``, ``ü``, ``ş``, ``ç``, ``ğ`` IPA'da başka harflerle yazılır.
#:
#: Eşleme **birebir** (tek karakter -> tek karakter) tutulur; böylece hizalama
#: konumları birebir geri eşlenebilir ve sütunlarda ÖZGÜN harfler görünür.
ORTHO_TO_IPA = {
    "y": "j",
    "j": "ʒ",
    "c": "ʤ",
    "ç": "ʧ",
    "ş": "ʃ",
    "ı": "ɯ",
    "ö": "ø",
    "ü": "y",
    "ğ": "ɣ",
    "ŕ": "r",
    "ĺ": "l",
}


def _to_ipa(form: str) -> str:
    """Hizalama için IPA'ya çevirir. Uzunluk birebirdir."""
    return "".join(ORTHO_TO_IPA.get(ch, ch) for ch in form)


@dataclass(frozen=True)
class AlignedColumn:
    """Hizalamanın tek bir sütunu: hangi dilde hangi ses var."""

    index: int
    sounds: dict[str, str]
    width: int

    @property
    def present(self) -> dict[str, str]:
        """Boşluk olmayan sesler."""
        return {lang: s for lang, s in self.sounds.items() if s and s != GAP}

    @property
    def gap_ratio(self) -> float:
        if not self.sounds:
            return 1.0
        gaps = sum(1 for s in self.sounds.values() if not s or s == GAP)
        return gaps / len(self.sounds)

    @property
    def distinct(self) -> frozenset[str]:
        return frozenset(self.present.values())


@lru_cache(maxsize=1)
def _lingpy_multiple() -> Any | None:
    try:
        from lingpy import Multiple
    except ImportError:
        logger.info("LingPy yok; merkez-yıldız hizalama yedeği kullanılacak")
        return None
    return Multiple


def _needleman_wunsch(a: str, b: str, *, gap_penalty: int = -1) -> tuple[str, str]:
    """İki dizi için klasik global hizalama — yedek yol."""
    rows, cols = len(a) + 1, len(b) + 1
    score = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        score[i][0] = i * gap_penalty
    for j in range(1, cols):
        score[0][j] = j * gap_penalty
    for i in range(1, rows):
        for j in range(1, cols):
            match = score[i - 1][j - 1] + (2 if a[i - 1] == b[j - 1] else -1)
            score[i][j] = max(match, score[i - 1][j] + gap_penalty, score[i][j - 1] + gap_penalty)

    out_a: list[str] = []
    out_b: list[str] = []
    i, j = len(a), len(b)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and score[i][j] == score[i - 1][j - 1] + (2 if a[i - 1] == b[j - 1] else -1):
            out_a.append(a[i - 1])
            out_b.append(b[j - 1])
            i, j = i - 1, j - 1
        elif i > 0 and score[i][j] == score[i - 1][j] + gap_penalty:
            out_a.append(a[i - 1])
            out_b.append(GAP)
            i -= 1
        else:
            out_a.append(GAP)
            out_b.append(b[j - 1])
            j -= 1
    return "".join(reversed(out_a)), "".join(reversed(out_b))


def _center_star(forms: dict[str, str]) -> dict[str, list[str]]:
    """Merkez-yıldız çoklu hizalama — LingPy yoksa kullanılır.

    En uzun biçim merkez seçilir; her biçim merkeze hizalanır ve merkezdeki
    boşluklar bütün satırlara yayılır.
    """
    if not forms:
        return {}
    center_lang = max(forms, key=lambda k: len(forms[k]))
    center = forms[center_lang]

    aligned: dict[str, str] = {}
    center_versions: dict[str, str] = {}
    for lang, form in forms.items():
        if lang == center_lang:
            aligned[lang] = center
            center_versions[lang] = center
            continue
        c_aligned, f_aligned = _needleman_wunsch(center, form)
        center_versions[lang] = c_aligned
        aligned[lang] = f_aligned

    # Merkezin tüm sürümlerindeki boşlukları birleştir
    merged_center: list[str] = []
    cursors = dict.fromkeys(forms, 0)
    positions: dict[str, list[str]] = {lang: [] for lang in forms}
    max_len = max(len(v) for v in center_versions.values())

    for _ in range(max_len * 2):
        if all(cursors[lang] >= len(center_versions[lang]) for lang in forms):
            break
        # Bu adımda merkezde boşluk isteyen var mı?
        wants_gap = [
            lang
            for lang in forms
            if cursors[lang] < len(center_versions[lang]) and center_versions[lang][cursors[lang]] == GAP
        ]
        if wants_gap:
            for lang in forms:
                if lang in wants_gap:
                    positions[lang].append(aligned[lang][cursors[lang]])
                    cursors[lang] += 1
                else:
                    positions[lang].append(GAP)
            merged_center.append(GAP)
            continue
        for lang in forms:
            if cursors[lang] < len(center_versions[lang]):
                positions[lang].append(aligned[lang][cursors[lang]])
                cursors[lang] += 1
            else:
                positions[lang].append(GAP)
        merged_center.append(center[len(merged_center)] if len(merged_center) < len(center) else GAP)

    return positions


#: Boşluk-yönelimli hizalama budaması açık mı? (Faz D4)
#:
#: Blum & List (2023, ``lingrex.trimming``) budamanın 10 ailenin 10'unda
#: düzenli denklik oranını artırdığını ölçüyor (+0,03…+0,07).
#: ⚠️ Rekonstrüksiyon **doğruluğuna** etkisi yayınlanmamıştır; bizim
#: ölçümümüz aşağıdaki sabitte.
TRIM_ALIGNMENTS = True

#: Budama eşiği: bu orandan çok boşluk içeren sütunlar budama adayıdır.
TRIM_THRESHOLD = 0.5


def _trim(rows: dict[str, list[str]]) -> dict[str, list[str]]:
    """Boşluk-yönelimli hizalama budaması (Blum & List 2023).

    Tek bir dilin kendi eklemesi olan sütunlar hizalamayı genişletir ve
    ata biçme yanlış konum ekler. ``lingrex`` bunları CV iskeletini bozmadan
    budar.

    ⚠️ ``lingrex`` yoksa veya budama satırları eşitsiz bırakırsa **hiçbir
    şey yapılmaz**: yarı budanmış bir hizalama, budanmamıştan kötüdür.
    """
    if not TRIM_ALIGNMENTS or len(rows) < 2:
        return rows
    try:
        from lingrex.trimming import Sites
    except ImportError:
        return rows
    langs = sorted(rows)
    try:
        sites = Sites([rows[lang] for lang in langs], gap=GAP)
        trimmed = sites.trimmed(threshold=TRIM_THRESHOLD, strategy="gap-oriented")
        matrix = trimmed.to_alignment()
    except Exception:
        logger.debug("Hizalama budaması başarısız; budanmamış hizalama kullanılıyor",
                     exc_info=True)
        return rows
    if len(matrix) != len(langs) or not matrix[0]:
        return rows
    widths = {len(row) for row in matrix}
    if len(widths) != 1:
        return rows
    return {lang: list(matrix[i]) for i, lang in enumerate(langs)}


def align_forms(forms: dict[str, str], *, trim: bool = True) -> list[AlignedColumn]:
    """``dil_kodu -> biçim`` eşlemesini sütunlara ayırır.

    :param trim: boşluk-yönelimli budama uygulansın mı? (Faz D4)

        ⚠️ **Ölçüm hizalamalarında ``False`` OLMALIDIR.** Budama, bir
        tarafta boşluk olan sütunları atar; ``metrics.reconstruction_bcubed``
        tahmin ile altını hizalarken bunu yaparsa **uyuşmazlıkları
        siler** ve skor yapay olarak şişer. Ölçüldü: budama metrik yoluna
        sızdığında bütün sistemlerin B-Cubed F'si birden yükseldi
        (``majority_character`` 0,571 -> 0,696) — tahminleri hiç
        değişmemiş olmasına rağmen.

    :returns: hizalamanın sütunları; uzunluk **en uzun tanığa göre** belirlenir,
        çapa kelimeye göre değil.
    """
    usable = {lang: form for lang, form in forms.items() if form}
    if len(usable) < 2:
        return []

    rows: dict[str, list[str]] = {}
    multiple = _lingpy_multiple()
    if multiple is not None:
        try:
            langs = sorted(usable)
            # ⚠️ LingPy'ye DİZGİ verilirse kendi IPA bölütleyicisini çalıştırır
            # ve ``yol`` kelimesini ``['yo', 'l']`` diye böler. Bölütlemeyi biz
            # yapıp harf listesi veriyoruz — ve harfleri IPA'ya çeviriyoruz,
            # yoksa SCA ses sınıfları imla harflerini yanlış sınıflandırır.
            with contextlib.redirect_stdout(io.StringIO()):
                aligner = multiple([list(_to_ipa(usable[lang])) for lang in langs])
                aligner.prog_align()
            matrix = aligner.alm_matrix
            if len(matrix) == len(langs):
                # IPA hizalamasındaki boşluk konumları ÖZGÜN harflere geri
                # eşlenir: çıktıda IPA değil, karşılaştırma biçmi görünmelidir.
                rows = {}
                for i, lang in enumerate(langs):
                    original = usable[lang]
                    cursor = 0
                    row: list[str] = []
                    for cell in matrix[i]:
                        if cell == GAP:
                            row.append(GAP)
                        else:
                            row.append(original[cursor] if cursor < len(original) else cell)
                            cursor += 1
                    rows[lang] = row
        except Exception:
            logger.warning("LingPy çoklu hizalama başarısız; yedeğe geçiliyor", exc_info=True)
            rows = {}

    if not rows:
        rows = _center_star(usable)
    if not rows:
        return []

    if trim:
        rows = _trim(rows)

    width = max(len(r) for r in rows.values())
    columns: list[AlignedColumn] = []
    for index in range(width):
        sounds = {lang: (row[index] if index < len(row) else GAP) for lang, row in rows.items()}
        columns.append(AlignedColumn(index=index, sounds=sounds, width=width))
    return columns
