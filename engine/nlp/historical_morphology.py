"""
Tarihsel Ek Ağacı Çözücüsü (Historical Suffix Tree Solver)

Tarihsel morfotaktik çözümleme hedefi:

    "Kelimeyi tarihsel yapım eklerine (+gU, -ik, -gə, -ba) bölerek kelimenin
     ham kökünü ayrıştırma."

Mevcut ``utils/morphology.py`` ve ``nlp/unsupervised_morpheme_segmenter.py``
yalnızca **modern Türkiye Türkçesi** eklerini tanıyordu; plandaki Eski Türkçe
yapım ekleri hiç yoktu ve çıktı düz bir liste, "ağaç" değildi.

Bu modül tarihsel yapım eklerini tanır ve katmanlı bir türetme AĞACI kurar:

    güzellik  ->  güzel  +lIk
              ->  gö(r)-  +z   (tarihsel katman)
"""
from __future__ import annotations

from typing import Any

from engine.logging_setup import get_logger
from engine.utils.orthography import to_comparison_form

logger = get_logger(__name__)

#: Eski/Orta Türkçe yapım ekleri.
#: Her giriş: (yüzey biçimleri, etiket, işlev, tarihsel katman)
HISTORICAL_SUFFIXES: list[tuple[tuple[str, ...], str, str, str]] = [
    # --- Fiilden ad ---
    (("gu", "gü", "ğu", "ğü", "qu"), "+gU", "fiilden ad/araç adı (Eski Türkçe +gU)", "old_turkic"),
    (("g", "ğ", "k", "q"), "+G", "fiilden ad (Eski Türkçe +G)", "old_turkic"),
    (("ik", "ık", "uk", "ük"), "-Ik", "fiilden sıfat/ad (Eski Türkçe -Ik)", "old_turkic"),
    (("ge", "ga", "gə", "qa"), "-gA", "fiilden ad (Eski Türkçe -gA)", "old_turkic"),
    (("gi", "gı", "gu", "gü", "ki", "kı"), "-gI", "fiilden ad (Eski Türkçe -gI: bilgi, sevgi)", "old_turkic"),
    (("nç", "inç", "ınç", "unç", "ünç"), "-nÇ", "fiilden duygu adı (sevinç, korkunç)", "old_turkic"),
    (("gıç", "giç", "guç", "güç"), "-gIç", "fiilden araç adı (dalgıç, bilgiç)", "old_turkic"),
    (("ba", "be", "pa", "pe"), "-bA", "fiilden ad (Eski Türkçe -bA)", "old_turkic"),
    (("gan", "gen", "qan", "ken"), "-gAn", "sıfat-fiil (Eski Türkçe -gAn)", "old_turkic"),
    (("miş", "mış", "muş", "müş"), "-mIş", "geçmiş sıfat-fiil", "old_turkic"),
    # Fiilden eylem adı: taşı-ma, bulaş-ma, kavur-ma. Tabloda yoktu; `taşıma`
    # bütün hâliyle aranıyor, akraba tanığı bulunamıyor ve motor çekimser
    # kalıyordu. ⚠️ `elma`, `yama`, `dolma` gibi yanlış adaylar
    # `search_engine`'deki asgari kelime uzunluğu (6) ve yeni tanıklık
    # denetimi tarafından eleniyor.
    (("ma", "me"), "-mA", "fiilden eylem adı (Eski Türkçe -mA)", "old_turkic"),
    (("m", "im", "ım", "um", "üm"), "-Im", "fiilden eylem adı", "old_turkic"),
    (("t", "it", "ıt", "ut", "üt"), "-It", "fiilden ad", "old_turkic"),
    (("n", "in", "ın", "un", "ün"), "-In", "fiilden ad / dönüşlü", "old_turkic"),
    (("z",), "-z", "eski ad yapım eki / ikilik (köz, teŋiz)", "old_turkic"),
    # --- Addan ad ---
    (("lik", "lık", "luk", "lük", "lig", "lıg", "lug", "lüg"), "+lIK",
     "addan soyut ad (Eski Türkçe +lIg)", "old_turkic"),
    (("çi", "çı", "çu", "çü", "ci", "cı", "cu", "cü"), "+çI", "meslek/fail adı", "old_turkic"),
    (("daş", "deş", "taş", "teş"), "+dAş", "ortaklık adı", "old_turkic"),
    (("lı", "li", "lu", "lü", "lıg", "lig"), "+lI", "addan sıfat (Eski Türkçe +lIg)", "old_turkic"),
    (("sız", "siz", "suz", "süz"), "+sIz", "yokluk sıfatı", "old_turkic"),
    (("cak", "cek", "çak", "çek"), "+çAK", "küçültme", "middle_turkic"),
    # --- Modern katman ---
    (("sal", "sel"), "+sAl", "Cumhuriyet dönemi sıfat eki", "modern"),
    (("tay", "tey"), "+tAy", "Cumhuriyet dönemi kurum adı", "modern"),
    (("men", "man"), "+mAn", "Cumhuriyet dönemi meslek eki", "modern"),
]

#: Bir kökün altına düşmemesi gereken asgari uzunluk.
MIN_STEM_LENGTH = 2
#: Azami türetme derinliği (sonsuz döngü koruması).
#: ⚠️ Çözümleme artık İLK TANIKLI KÖKTE durduğu için bu tavana pratikte
#: ulaşılmaz; sabit, zincirleme soymaya dönülürse diye korunuyor.
MAX_DEPTH = 4


#: Tanıklık denetimi süreç ömrü boyunca önbelleklenir; aynı kök defalarca
#: sorulur (aynı ek zinciri farklı kelimelerde tekrar eder).
_ATTESTATION_CACHE: dict[str, bool] = {}


def _stem_is_attested(stem: str) -> bool:
    """Soyulan kök Türkçede gerçek bir sözlükbirim mi?

    İndeks yoksa ya da sorgu düşerse **True** döner: kapı, veri eksikliğinde
    çözümlemeyi büsbütün durdurmamalı (deponun başka yerlerindeki "veri yoksa
    sessizce devre dışı kal" davranışıyla aynı).
    """
    if stem in _ATTESTATION_CACHE:
        return _ATTESTATION_CACHE[stem]

    verdict = True
    try:
        from engine.db.lexicon_index import LexiconIndex

        index = LexiconIndex()
        if index.exists:
            verdict = index.is_attested_stem(stem)
            # ⚠️ TDK YEDEĞİ DENENDİ VE GERİ ALINDI (2026-09-21).
            #
            # İndeks yaygın alıntı köklerini kaçırıyor: `perva` TDK'da madde
            # başı ("lisan: Farsça pervā") ama indekste yok, bu yüzden
            # `pervasız` soyulamıyor. İndeks "hayır" dediğinde TDK'ya sormak
            # denendi — ama NET KAZANÇ VERMEDİ:
            #
            #   kazanılan : pervasız -> perva   (doğru)
            #   kaybedilen: bardak   -> barda   (YANLIŞ; `barda` da TDK'da
            #               madde başı ama `bardak` ile ilgisiz —
            #               Nişanyan: bardak < Eski Türkçe `bart`)
            #
            # Yani "sözlükte var mı" bu iki durumu ayırt edemiyor; ikisi de
            # var. Yüzey uzunluğu da ayırmıyor: `bardak`(+G, 'k') yanlış ama
            # `bitig`(+G, 'g') DOĞRU ve teste bağlı.
            #
            # Ayrım anlamsal: kök ile kelimenin anlamca ilişkili olması
            # gerekir. O denetim şu an kapalı (sentence-transformers kurulu
            # değil, A-HVP 3. aşama hiçbir kelimede ölçülemiyor). Semantik
            # aşama açıldığında bu yedek yeniden değerlendirilmeli.
    except Exception:
        logger.debug("Kök tanıklık denetimi yapılamadı: %s", stem, exc_info=True)

    _ATTESTATION_CACHE[stem] = verdict
    return verdict


class HistoricalMorphologyAnalyzer:
    """Kelimeyi tarihsel yapım eklerine göre katmanlı bir ağaca ayırır."""

    def build_tree(self, word: str) -> dict[str, Any]:
        """
        Kelimenin tarihsel türetme ağacını kurar.

        :returns: ``root`` ham kök, ``layers`` her katmanda soyulan ek,
            ``depth`` türetme derinliği.
        """
        w = to_comparison_form(word)
        if not w:
            return {
                "word": word,
                "evidence_available": False,
                "root": "",
                "layers": [],
                "depth": 0,
                "reason": "Kelime çözümlenebilir bir biçime indirgenemedi.",
            }

        layers: list[dict[str, Any]] = []
        stem = w

        # İki kural birlikte çalışır.
        #
        # 1) TANIKLIK KAPISI — soyulan kök Türkçede sözlükbirim değilse
        #    soyma yapılmaz. Zincirleme soyma kelime OLMAYAN kökler
        #    üretiyor, o kısa parçalar başka dillerde tesadüfen eşleşiyordu:
        #      menengiç -> mene [az]   avsunlu   -> avs  [ota]
        #      köremez  -> köre [kdr]  garametli -> gara [tk]
        #
        # 2) EN SIĞ TANIKLI KÖKTE DUR — kapı tek başına yetmedi, çünkü
        #    zincir gerçek kökü geçip 2-3 harflik parçalara iniyor ve o
        #    parçalar tesadüfen Türkçe kelime olduğu için kapıyı da geçiyordu:
        #      kanatlı -> kana   altlık -> al   çıtlık -> çı
        #      damcı   -> da     değin  -> de   küncü  -> kü
        #
        # ⚠️ Ölçüldü (400 ağız maddesi): kapıdan geçen iki+ katmanlı 12
        # analizin HEPSİ bu türdendi; meşru çok katmanlı türetme ÇIKMADI.
        # Dolayısıyla ilk tanıklı kökte durmanın ölçülen maliyeti sıfır,
        # kazancı 12 hatalı analiz. Testlerin dayandığı dört kelime
        # (güzellik, bitig, susuz, toplumsal) aynen korunur.
        match = self._strip_one(stem)
        if match is not None:
            new_stem, label, function, layer = match
            if _stem_is_attested(new_stem):
                layers.append({
                    "surface": stem[len(new_stem):],
                    "suffix": label,
                    "function": function,
                    "historical_layer": layer,
                    "stem_before": stem,
                    "stem_after": new_stem,
                })
                stem = new_stem

        return {
            "word": word,
            "normalised": w,
            "evidence_available": bool(layers),
            "root": stem,
            "layers": layers,
            "depth": len(layers),
            "derivation_path": " + ".join([stem, *[lay["suffix"] for lay in reversed(layers)]])
            if layers
            else stem,
            "oldest_layer": layers[-1]["historical_layer"] if layers else None,
        }

    @staticmethod
    def _strip_one(stem: str) -> tuple[str, str, str, str] | None:
        """En uzun eşleşen eki soyar. Kök çok kısalırsa soymaz."""
        best: tuple[int, str, str, str, str] | None = None
        for surfaces, label, function, layer in HISTORICAL_SUFFIXES:
            for surface in surfaces:
                if not stem.endswith(surface):
                    continue
                remaining = stem[: -len(surface)]
                if len(remaining) < MIN_STEM_LENGTH:
                    continue
                cand = (len(surface), remaining, label, function, layer)
                if best is None or cand[0] > best[0]:
                    best = cand
        if best is None:
            return None
        return best[1], best[2], best[3], best[4]
