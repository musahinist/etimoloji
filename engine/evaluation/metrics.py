"""
Değerlendirme metrikleri.

**Neden dört metrik?** List (2019, *Computational Linguistics* 45(1)) tek
başına edit distance ile yapılan değerlendirmeleri açıkça yetersiz buluyor:
kısa kelimelerde ED küçük görünür, uzun kelimelerde büyük; normalize edilmezse
veri kümeleri arası karşılaştırma yapılamaz. Bu yüzden burada dördü birden
raporlanır ve hiçbiri tek başına kullanılmaz:

===============  ==========================================================
``ED``           Levenshtein uzaklığı — ham hata büyüklüğü
``NED``          uzunluğa bölünmüş ED — veri kümeleri arası karşılaştırılabilir
``accuracy``     tam eşleşme oranı — en katı ölçüt
``B-Cubed F``    akraba kümeleme kalitesi (Bagga & Baldwin; Hauer & Kondrak)
===============  ==========================================================

Ek olarak :func:`feature_error_rate` fonolojik özellik düzeyinde hata verir:
``*köŕ`` yerine ``*göŕ`` demek ile ``*köŕ`` yerine ``*mat`` demek arasındaki
farkı ED göremez, FER görür.

Karşılaştırma noktaları (aynı metriklerle ölçülmüş yayınlar):

* LexStat-Infomap akraba tespiti **B-Cubed F ≈ 0,89** (List, Greenhill & Gray
  2017, *PLOS ONE*)
* Transformer rekonstrüksiyon **%53 Roman / %39,5 Sinitik** tam doğruluk
  (Kim ve ark. 2023, ACL) — ama **8.799 eğitim örneğiyle**
"""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

#: Rekonstrüksiyon karşılaştırmasında yok sayılan işaretler. Ata biçim
#: gösteriminde ``*`` yalnızca "bu bir rekonstrüksiyondur" demektir, sesin
#: parçası değildir; parantez ise belirsizlik/isteğe bağlılık işaretidir.
_STRIP_CHARS = "*()[]{}?"

#: Cevaplanmayan bir madde ED ortalamasına ne kadar katkı yapsın?
#: Altın biçimlerin ortalama uzunluğu ~5; boş cevabın ED'si o uzunluktur.
WORST_CASE_EDIT_DISTANCE = 5

#: Arşifonem gösterimi: büyük harf "bu konumda iki ses arasında karar
#: verilmemiştir" demektir (``*Kāpuk`` = *k-* mi *g-* mi belli değil).
#: "Kabul edilebilir" eşleşmede bu belirsizlik tolere edilir.
ARCHIPHONEME_EQUIVALENTS: dict[str, frozenset[str]] = {
    "K": frozenset("kgq"),
    "T": frozenset("td"),
    "P": frozenset("pb"),
    "S": frozenset("sz"),
    "Č": frozenset({"č", "ǰ", "c"}),
    "A": frozenset("aä"),
    "E": frozenset({"e", "ẹ", "ä"}),
    "U": frozenset("uü"),
    "I": frozenset({"ı", "i", "ɨ"}),
    "O": frozenset("oö"),
}


#: **Salt yazım geleneği farkları.** Aynı sesi farklı yazan okulların
#: uzlaştırılması. Bunlar EXACT ölçümde de eşitlenir: ``*yol`` ile ``*jol``
#: aynı rekonstrüksiyondur, farklı yazımdır. Bunları hata saymak dilbilimi
#: değil, çeviriyazı geleneğini ölçmek olurdu.
#:
#: Ölçüldü: "söz başı yanlış" sayılan 93 hatanın 31'i yalnız buydu
#: (``y``/``j`` 18, ``c``/``č`` 5, ``ı``/``ï`` 3, ``k``/``ḳ`` 2 …).
TRANSCRIPTION_VARIANTS: dict[str, str] = {
    "ï": "ı", "ɨ": "ı", "ɯ": "ı",
    "y": "j", "ǰ": "j", "ɟ": "j",
    "č": "ç", "c": "ç", "ʧ": "ç",
    "š": "ş", "ʃ": "ş",
    "ñ": "ŋ", "ń": "ŋ", "ṅ": "ŋ",
    "ḳ": "k", "q": "k", "ḵ": "k",
    "ġ": "g", "ǧ": "ğ", "ɣ": "ğ",
    "ẹ": "e", "ė": "e", "ạ": "a", "ǝ": "e", "ə": "e",
    "ẓ": "z", "ṣ": "s", "ṭ": "t", "ḏ": "d",
    "ʼ": "", "ʔ": "", "ʲ": "", "ˊ": "", "'": "", "ʻ": "",
}


def _fold_transcription(text: str) -> str:
    return "".join(TRANSCRIPTION_VARIANTS.get(ch, ch) for ch in text)


def normalize_proto(form: str, *, strip_length: bool = False) -> str:
    """Ata biçmi karşılaştırılabilir hâle getirir.

    ``*Kāpuk`` -> ``kāpuk``. ``strip_length=True`` ise uzunluk da atılır
    (``kapuk``) — "uzunluk dışında doğru mu?" sorusunu ayrıca ölçmek için.

    Yazım geleneği farkları (:data:`TRANSCRIPTION_VARIANTS`) burada eşitlenir.
    ⚠️ ``ŕ`` ve ``ĺ`` **eşitlenmez**: Türkolojide bunlar ``r``/``l``den ayrı
    sesbirimlerdir; birleştirmek ``*ar`` ile ``*aŕ``ı aynı sayardı.
    """
    text = unicodedata.normalize("NFC", form.strip())
    for ch in _STRIP_CHARS:
        text = text.replace(ch, "")
    text = text.split(",")[0].split("/")[0].strip()  # "*Kūrɨk,gak" -> "*Kūrɨk"
    text = _fold_transcription(text.casefold())
    if strip_length:
        decomposed = unicodedata.normalize("NFD", text)
        text = unicodedata.normalize("NFC", "".join(c for c in decomposed if c != "̄")).replace("ː", "")
    return text


def edit_distance(a: str, b: str) -> int:
    """Levenshtein uzaklığı. Sürüm bağımlılığı olmasın diye elde yazılmıştır."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,  # silme
                    current[j - 1] + 1,  # ekleme
                    previous[j - 1] + (ca != cb),  # değiştirme
                )
            )
        previous = current
    return previous[-1]


def normalized_edit_distance(a: str, b: str) -> float:
    """ED'yi daha uzun dizginin uzunluğuna böler → [0, 1].

    Normalizasyon olmadan uzun kelimeler kısa kelimelerden "daha kötü"
    görünür ve veri kümeleri arası karşılaştırma anlamsızlaşır (List 2019).
    """
    longest = max(len(a), len(b))
    return edit_distance(a, b) / longest if longest else 0.0


#: **Salt gösterim farkları.** Aynı sesi farklı yazan geleneklerin
#: uzlaştırılması; hiçbiri fonolojik iddia taşımaz.
#: ``*teŋiŕ`` ile ``*teñiŕ`` aynı rekonstrüksiyondur.
NOTATIONAL_EQUIVALENTS: dict[str, str] = {
    "ẹ": "e",
    "ė": "e",
    "ä": "e",
    "ɨ": "ı",
    "ï": "ı",
    "ı": "ı",
    "ǰ": "j",
    "y": "j",
    "š": "ş",
    "č": "ç",
    "ń": "ŋ",
    "ñ": "ŋ",
    "ĺ": "l",  # Lir-Şaz lambdaizminin ata sesi; *ĺ ve *l₂ aynı şeydir
    "ŕ": "r",  # rotasizmin ata sesi; *ŕ ve *r₂ aynı şeydir
    "ʼ": "",
    "ʔ": "",
    "ˊ": "",
    "'": "",
}

#: **Gerçek bilimsel tartışmalar.** Uzmanların anlaşamadığı konularda motorun
#: taraf tutmaması beklenir; iki cevap da savunulabilirdir.
#: Konum duyarlıdır: söz başı ``*g-`` ~ ``*k-`` tartışması (Doerfer'e karşı
#: Tekin) yalnız söz başında geçerlidir, söz içinde değil.
DISPUTED_INITIAL: dict[str, str] = {"g": "k", "b": "b", "d": "t"}


def _fold(form: str) -> str:
    """Gösterim farklarını ve uzunluğu silerek "aynı iddia mı?" karşılaştırması."""
    text = normalize_proto(form, strip_length=True)
    folded = "".join(NOTATIONAL_EQUIVALENTS.get(ch, ch) for ch in text)
    if folded:
        head = DISPUTED_INITIAL.get(folded[0], folded[0])
        folded = head + folded[1:]
    return folded


def is_acceptable(predicted: str, gold: str) -> bool:
    """ "Kabul edilebilir" eşleşme — tam değil ama savunulabilir.

    Üç tolerans tanınır. Hepsi **adı konmuş** bir sebebe dayanır; "yakındı,
    say gitsin" toleransı **yoktur**:

    1. **Ünlü uzunluğu** — uzunluk tanığı (Halaçça, Dolganca…) bulunmayan bir
       kümede uzunluğu türetememek yöntem hatası değil, veri eksikliğidir.
    2. **Gösterim farkı** — ``*teŋiŕ`` ~ ``*teñir``; aynı iddianın farklı
       yazımı (:data:`NOTATIONAL_EQUIVALENTS`).
    3. **Uzmanların tartıştığı konular** — arşifonem belirsizliği
       (``*Kāpuk``ta *K-* hem *k-* hem *g-* olabilir) ve söz başı ``*g-``/``*k-``
       tartışması (:data:`DISPUTED_INITIAL`).

    ⚠️ Kör bir "edit distance ≤ 1" kuralı **bilerek kullanılmaz**: o kural
    ``*tuz`` ~ ``*tūŕ`` gibi **rotasizmin kaçırıldığı** durumu da kabul
    edilebilir sayardı; oysa bu, motorun düzeltmesi gereken asıl hatadır.
    """
    if normalize_proto(predicted) == normalize_proto(gold):
        return True
    if _fold(predicted) == _fold(gold):
        return True
    return _matches_archiphonemes(_fold(predicted), gold)


def _matches_archiphonemes(predicted_folded: str, gold_raw: str) -> bool:
    """Tahmin, altın biçimdeki arşifonem belirsizliğinin içine düşüyor mu?

    Büyük harf bilgisi :func:`normalize_proto` tarafından silindiği için
    altın biçim burada HAM hâliyle okunur; yalnız uzunluk ve gösterim
    farkları temizlenir.
    """
    gold_chars = unicodedata.normalize("NFC", gold_raw.strip())
    for ch in _STRIP_CHARS:
        gold_chars = gold_chars.replace(ch, "")
    gold_chars = gold_chars.split(",")[0].split("/")[0].strip()
    gold_chars = unicodedata.normalize(
        "NFC", "".join(c for c in unicodedata.normalize("NFD", gold_chars) if c != "\u0304")
    ).replace("ː", "")
    gold_chars = "".join(
        ch if ch in ARCHIPHONEME_EQUIVALENTS else NOTATIONAL_EQUIVALENTS.get(ch, ch) for ch in gold_chars
    )
    if len(gold_chars) != len(predicted_folded):
        return False
    for pc, gc in zip(predicted_folded, gold_chars, strict=True):
        if pc == gc.casefold():
            continue
        if pc in ARCHIPHONEME_EQUIVALENTS.get(gc, frozenset()):
            continue
        return False
    return True


@dataclass
class ReconstructionScore:
    """Bir rekonstrüksiyon koşusunun metrik dörtlüsü ve hata dökümü."""

    n: int = 0
    exact: int = 0
    acceptable: int = 0
    total_ed: int = 0
    total_ned: float = 0.0
    abstained: int = 0
    per_item: list[dict[str, object]] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.exact / self.n if self.n else 0.0

    @property
    def acceptable_rate(self) -> float:
        return self.acceptable / self.n if self.n else 0.0

    @property
    def mean_ed(self) -> float:
        return self.total_ed / self.n if self.n else 0.0

    @property
    def mean_ned(self) -> float:
        return self.total_ned / self.n if self.n else 0.0

    @property
    def coverage(self) -> float:
        """Motor kaç maddede cevap verdi? Çekimserlik bir özelliktir, hata değil."""
        attempted = self.n - self.abstained
        return attempted / self.n if self.n else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "n": self.n,
            "accuracy": round(self.accuracy, 4),
            "acceptable": round(self.acceptable_rate, 4),
            "ED": round(self.mean_ed, 4),
            "NED": round(self.mean_ned, 4),
            "coverage": round(self.coverage, 4),
            "abstained": self.abstained,
        }


def score_reconstructions(
    pairs: Iterable[tuple[str, str]],
    *,
    abstentions: int = 0,
) -> ReconstructionScore:
    """``(tahmin, altın)`` çiftlerini puanlar.

    :param pairs: tahmin edilen ve altın ata biçimler
    :param abstentions: motorun cevap vermediği madde sayısı — bunlar ``n``e
        dahildir ama doğru sayılmaz; ``coverage`` ile ayrıca raporlanır.
    """
    score = ReconstructionScore(abstained=abstentions)
    for predicted, gold in pairs:
        p, g = normalize_proto(predicted), normalize_proto(gold)
        exact = p == g
        acceptable = exact or is_acceptable(predicted, gold)
        ed = edit_distance(p, g)
        score.n += 1
        score.exact += exact
        score.acceptable += acceptable
        score.total_ed += ed
        score.total_ned += normalized_edit_distance(p, g)
        score.per_item.append(
            {"predicted": predicted, "gold": gold, "exact": exact, "acceptable": acceptable, "ed": ed}
        )
    # ⚠️ Çekimserlik BEDAVA DEĞİLDİR. Eskiden yalnız ``n`` artırılıyordu;
    # ``total_ned`` ve ``total_ed`` değişmediği için çekimser madde ortalamaya
    # **NED=0 (mümkün olan en iyi)** olarak giriyordu. Bu, cevap vermemeyi
    # kusursuz cevap vermekle bir tutuyor ve çekimser kalmayı ödüllendiriyordu.
    #
    # Doğrusu: cevaplanmayan madde mümkün olan **en kötü** değeri alır.
    # Anlamlılık testi de aynı muhasebeyi kullanır; ikisi ayrışırsa tablo ile
    # test birbiriyle çelişir.
    score.n += abstentions
    score.total_ned += float(abstentions)
    score.total_ed += abstentions * WORST_CASE_EDIT_DISTANCE
    return score


def bcubed_fscore(
    predicted: dict[str, str] | Sequence[tuple[str, str]],
    gold: dict[str, str] | Sequence[tuple[str, str]],
) -> dict[str, float]:
    """B-Cubed precision/recall/F — akraba kümelemenin standart ölçüsü.

    Her **öğe** için, kendi kümesindeki öğelerin ne kadarının altın kümesinde
    de birlikte olduğuna bakılır; sonuç öğeler üzerinden ortalanır. Küme
    sayısı farklı olsa bile anlamlı kalması bu yüzdendir.

    Referans: Bagga & Baldwin 1998; tarihsel dilbilimde kullanımı için
    Hauer & Kondrak 2011 ve List, Greenhill & Gray 2017 (*PLOS ONE*), orada
    LexStat-Infomap **F ≈ 0,89** alır — bu modülün karşılaştırma hedefi.

    :param predicted: ``öğe_kimliği -> küme_kimliği``
    :param gold: ``öğe_kimliği -> küme_kimliği``
    """
    pred = dict(predicted)
    truth = dict(gold)
    shared = sorted(set(pred) & set(truth))
    if not shared:
        return {"precision": 0.0, "recall": 0.0, "fscore": 0.0, "n": 0}

    pred_members: dict[str, set[str]] = defaultdict(set)
    gold_members: dict[str, set[str]] = defaultdict(set)
    for item in shared:
        pred_members[pred[item]].add(item)
        gold_members[truth[item]].add(item)

    precision = recall = 0.0
    for item in shared:
        p_cluster = pred_members[pred[item]]
        g_cluster = gold_members[truth[item]]
        overlap = len(p_cluster & g_cluster)
        precision += overlap / len(p_cluster)
        recall += overlap / len(g_cluster)

    precision /= len(shared)
    recall /= len(shared)
    fscore = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "fscore": round(fscore, 4),
        "n": len(shared),
    }


#: Kaba fonolojik özellik sınıfları. PanPhon kuruluysa oradaki 24 özellikli
#: vektörler tercih edilir; bu tablo bağımlılık olmadan da çalışmak içindir.
_FEATURE_CLASSES: dict[str, frozenset[str]] = {
    "labial": frozenset("pbmfvw"),
    "coronal": frozenset({"t", "d", "n", "s", "z", "r", "l", "š", "ž", "č", "ǰ"}),
    "dorsal": frozenset({"k", "g", "q", "ɣ", "x", "ŋ"}),
    "voiced": frozenset({"b", "d", "g", "z", "ž", "ǰ", "v", "ɣ", "m", "n", "ŋ", "r", "l", "j", "y"}),
    "vowel": frozenset("aeıioöuüäɨėẹ"),
    "front": frozenset("eiöüä"),
    "high": frozenset("iıuü"),
    "nasal": frozenset("mnŋ"),
}


def _features(ch: str) -> frozenset[str]:
    return frozenset(name for name, members in _FEATURE_CLASSES.items() if ch in members)


def feature_error_rate(predicted: str, gold: str) -> float:
    """Fonolojik özellik düzeyinde hata oranı → [0, 1].

    ED ``*göŕ`` ile ``*mat``ı aynı uzaklıkta görebilir; FER görmez. Sesbirim
    yerine özellik saydığı için "yakın hata" ile "alakasız hata" ayrılır.
    """
    p, g = normalize_proto(predicted), normalize_proto(gold)
    if not p and not g:
        return 0.0
    length = max(len(p), len(g))
    if not length:
        return 0.0
    wrong = 0.0
    for i in range(length):
        pc = p[i] if i < len(p) else ""
        gc = g[i] if i < len(g) else ""
        if pc == gc:
            continue
        pf, gf = _features(pc), _features(gc)
        union = pf | gf
        wrong += 1.0 if not union else len(pf ^ gf) / len(union)
    return wrong / length


# --- Altın biçim ayrıştırma ------------------------------------------------
#
# ``savelyevturkic``ın ``Root`` alanı serbest metindir; şu gerçek örnekler
# 399 maddelik altın standartta bulunuyor::
#
#     *o:t                    ASCII iki nokta = ünlü uzunluğu (96 madde)
#     *jaŋï / *jeŋi           eşdeğer iki rekonstrüksiyon (30 madde)
#     *ubak (ЭСТЯ 1, 561)     kaynak künyesi (52 madde parantez içeriyor)
#     *kI:n: KN               sondaki ": KN" arşifonem açıklaması
#     *üčük (?); According…   serbest yorum
#
# Bunlar temizlenmeden ölçüm anlamsızdır: motor ``*ot`` dediğinde altın
# ``*o:t (ЭСТЯ …)`` ile karşılaştırılırsa daima yanlış sayılır.

_COMMENT_SPLIT = ";"
_PAREN_PATTERN = re.compile(r"\([^)]*\)")
_TRAILING_GLOSS = re.compile(r":\s*[A-ZŠČŊĹŔ]{2,}\s*$")
_COLON_LENGTH = re.compile(r"([aeıioöuüäɨėẹAEIOUÄ]):")
#: ``*sV: sẹ`` — arşifonemli biçim, iki nokta, sonra somut biçim.
_CONCRETE_AFTER_COLON = re.compile(r"^\s*(\*?[^\s:]+)\s*:\s+([^\s:]+)\s*$")


def parse_gold_form(raw: str) -> list[str]:
    """Ham ``Root`` alanını karşılaştırılabilir ata biçim adaylarına çevirir.

    Birden çok aday dönebilir: ``*jaŋï / *jeŋi`` iki eşdeğer rekonstrüksiyondur
    ve motorun **herhangi birini** bulması doğrudur. Boş liste dönerse madde
    altın standarda alınmamalıdır.
    """
    text = raw.split(_COMMENT_SPLIT)[0]
    text = _PAREN_PATTERN.sub("", text)
    text = _TRAILING_GLOSS.sub("", text)
    # ``*sV: sẹ`` kalıbı: soldaki arşifonemli biçim ve sağdaki somut biçim
    # AYNI rekonstrüksiyonun iki gösterimidir. İkisi de aday sayılır; yoksa
    # ``: `` iki nokta uzunluk kuralına takılıp ``*sV̄ sẹ`` gibi çöp üretir.
    concrete = _CONCRETE_AFTER_COLON.match(text)
    if concrete:
        text = f"{concrete.group(1)} / {concrete.group(2)}"
    text = _COLON_LENGTH.sub(lambda m: m.group(1) + "̄", text)
    text = unicodedata.normalize("NFC", text).strip()

    candidates: list[str] = []
    for chunk in text.replace(" / ", "/").split("/"):
        parts = [p.strip() for p in chunk.split(",") if p.strip()]
        if not parts:
            continue
        head_length = len(parts[0].lstrip("*"))
        for index, part in enumerate(parts):
            # Virgülden sonraki parça iki türlü olabilir:
            #   ``*ti:ŕ, *tü:ŕ``   -> ``*`` taşıyor, gerçek bir alternatif
            #   ``*Kūrɨk,gak``     -> ``*`` yok ve kısa, bu bir EK PARÇASIDIR
            #                         (``*Kūrgak`` demek isteniyor, ``*gak`` değil)
            # İkincisini aday saymak ölçümü haksız yere kolaylaştırır.
            if index > 0 and not part.startswith("*"):
                if len(part.lstrip("*")) < 0.7 * head_length:
                    continue
            form = part if part.startswith("*") else "*" + part
            if len(form.lstrip("*")) >= 2 and form not in candidates:
                candidates.append(form)
    return candidates


def best_match(predicted: str, gold_candidates: Sequence[str]) -> tuple[str, bool, bool]:
    """En iyi eşleşen altın adayı seçer.

    :returns: ``(seçilen_altın, tam_mı, kabul_edilebilir_mi)``. Motorun
        **herhangi bir** eşdeğer rekonstrüksiyonu bulması başarıdır; en katı
        adaya göre puanlamak haksız olurdu.
    """
    if not gold_candidates:
        return "", False, False
    best = gold_candidates[0]
    best_ed = None
    exact = acceptable = False
    for candidate in gold_candidates:
        if normalize_proto(predicted) == normalize_proto(candidate):
            return candidate, True, True
        if is_acceptable(predicted, candidate):
            best, acceptable = candidate, True
            best_ed = 0
            continue
        ed = edit_distance(normalize_proto(predicted), normalize_proto(candidate))
        if best_ed is None or ed < best_ed:
            best, best_ed = candidate, ed
    return best, exact, acceptable


# --- Rekonstrüksiyon için B-Cubed F ----------------------------------------
#
# ⚠️ Bu, akraba kümeleme B-Cubed'inden FARKLI bir ölçüdür ve farklı bir soru
# sorar. List (2019, *Beyond Edit Distances: Comparing linguistic
# reconstruction systems*, Theoretical Linguistics 45) tam olarak şunu
# gösteriyor: edit distance **sistematik** hataları her örnekte yeniden
# cezalandırır. Bir sistem ``*č``yi tutarlı biçimde ``*c`` yazıyorsa bu tek
# bir yanlış karardır ve geri döndürülebilir; ED ise onu yüz kelimede yüz kez
# cezalandırır.
#
# B-Cubed bunu düzeltir: hataların ne kadar TUTARLI olduğunu ölçer. Yüksek
# B-Cubed, "yanlış ama sistemli" demektir — yani yeniden eşlenebilir.
#
# SIGTYP 2022'nin resmi metrikleri ED, NED, **B-Cubed F** ve BLEU'dur; tam
# doğruluk hiç yer almaz.


def reconstruction_bcubed(pairs: Sequence[tuple[str, str]]) -> dict[str, float]:
    """Veri kümesi düzeyinde B-Cubed F — hataların tutarlılığını ölçer.

    Her ``(tahmin, altın)`` çifti hizalanır; hizalamanın her sütunu bir
    ``(tahmin_sesi, altın_sesi)`` karşılığı üretir. Sonra:

    * **kesinlik**: aynı tahmin sesini taşıyan konumların ne kadarı aynı
      altın sesine karşılık geliyor
    * **duyarlılık**: aynı altın sesini taşıyan konumların ne kadarı aynı
      tahmin sesinden geliyor

    Rastgele hata yapan bir sistem düşük, tutarlı biçimde yanlış eşleyen bir
    sistem yüksek alır — ve ikincisi düzeltilebilir bir sistemdir.

    :param pairs: ``(tahmin, altın)`` çiftleri (yıldızlı ham biçimler)
    """
    from engine.nlp.multi_alignment import GAP, align_forms

    predicted_groups: dict[str, list[int]] = defaultdict(list)
    gold_groups: dict[str, list[int]] = defaultdict(list)
    position = 0

    for raw_prediction, raw_gold in pairs:
        prediction = normalize_proto(raw_prediction)
        gold = normalize_proto(raw_gold)
        if not prediction or not gold:
            continue
        # ⚠️ ``trim=False`` ŞART: budama bir tarafta boşluk olan sütunları
        # atar, yani tam da ölçmek istediğimiz uyuşmazlıkları siler.
        columns = align_forms({"pred": prediction, "gold": gold}, trim=False)
        for column in columns:
            sounds = column.sounds
            left = sounds.get("pred", GAP) or GAP
            right = sounds.get("gold", GAP) or GAP
            if left == GAP and right == GAP:
                continue
            predicted_groups[left].append(position)
            gold_groups[right].append(position)
            position += 1

    if not position:
        return {"precision": 0.0, "recall": 0.0, "fscore": 0.0, "n": 0}

    membership_predicted = {
        index: sound for sound, indices in predicted_groups.items() for index in indices
    }
    membership_gold = {
        index: sound for sound, indices in gold_groups.items() for index in indices
    }

    precision = recall = 0.0
    for index in range(position):
        own_predicted = set(predicted_groups[membership_predicted[index]])
        own_gold = set(gold_groups[membership_gold[index]])
        overlap = len(own_predicted & own_gold)
        precision += overlap / len(own_predicted)
        recall += overlap / len(own_gold)

    precision /= position
    recall /= position
    fscore = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "fscore": round(fscore, 4),
        "n": position,
    }
