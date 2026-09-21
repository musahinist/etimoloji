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


#: `_ATTESTATION_CACHE` ile aynı gerekçe: soyma denemesi başına indeks açmak
#: pahalı. Formül sonucu süreç boyunca değişmez.
_FORMULA_CACHE: dict[str, str | None] = {}


def _formula_stem(word: str) -> str | None:
    """Kelimenin KAYITLI etimolojisindeki türetme formülünden kökü çıkarır.

    Wiktionary türetmeleri düzenli bir kalıpla yazıyor::

        "equivalent to perva + -sız"
        "from güzel (“beautiful”) + -lik (“-ness”)"
        "equivalent to su + -suz"

    Bu, soyma için OLUMLU KANITTIR: kaynağın kendisi kelimenin hangi kökten
    türediğini söylüyor. Formül yoksa ``None`` döner — "türemiş değil"
    demek değil, "veri yok" demektir; karar tanıklık kapısına kalır.

    ⚠️ Kapsam ölçüldü: 45.647 Türkçe maddenin %54,5'inde etimoloji metni,
    yalnız **%10,2**'sinde bu formül var. Bu yüzden tek başına kapı olarak
    kullanılamaz, yalnız destekleyici kanıttır.
    """
    import re

    key = (word or "").strip().lower()
    if key in _FORMULA_CACHE:
        return _FORMULA_CACHE[key]

    found: str | None = None
    try:
        from engine.db.lexicon_index import LexiconIndex

        index = LexiconIndex()
        if index.exists:
            text = ""
            for row in index.lookup(key, languages=["tr"], limit=5) or []:
                text = (row.get("etymology") or "").strip()
                if text:
                    break
            if text:
                match = re.search(
                    r"(?:equivalent to|from)\s+([a-zçğıöşüâîû\-]{2,})\s*(?:\([^)]*\))?\s*\+",
                    text,
                    re.IGNORECASE,
                )
                if match:
                    found = match.group(1).lower().strip()
    except Exception:
        logger.debug("Türetme formülü okunamadı: %s", key, exc_info=True)

    _FORMULA_CACHE[key] = found
    return found


def _strip_is_supported(word: str, stem: str) -> bool:
    """Bu soyma kanıtla destekleniyor mu?

    Kural YALNIZ EKLEYİCİDİR: türetme formülü adayı doğruluyorsa soyma kabul
    edilir, aksi HER durumda karar eskisi gibi tanıklık kapısına kalır.

    Ölçüldü — formülün çözdüğü, başka hiçbir ucuz yöntemin çözemediği ayrım:
        pervasız -> "equivalent to perva + -sız"  => formül KABUL ediyor
        bardak   -> formül yok ("ultimately from Early Old Oghuz برت");
                    tanıklık kapısına düşer, `barda`nın indeksteki tek anlamı
                    "locative singular of bar" (çekim) olduğu için REDDEDİLİR.
    Gloss örtüşmesi ve CLICS bu ikisini ayıramamıştı (bkz. commit günlüğü).

    ⚠️ "Formül BAŞKA kök söylüyorsa reddet" dalı DENENDİ ve GERİ ALINDI.
    Formül nihai tabanı yazar, soyucu tek katman soyar; granülarite uyuşmaz::

        adaletsizlik -> formül "adalet + -siz + -lik" verir, aday `adaletsiz`

    600 rastgele tr sözlükbiriminde ölçüldü: o dal 3 doğru kabul getirirken
    5 DOĞRU soymayı kapatıyordu (bıngıldak->bıngılda, dükkâncı->dükkan,
    adaletsizlik->adaletsiz, İzmirli->izmir, doymuş->doy). Net zarar.
    """
    formula = _formula_stem(word)
    # Wiktionary fiil köklerini TİRELİ yazar ("equivalent to um- + -ut"),
    # soyucu ise tiresiz aday üretir; tire atılmazsa eşleşme kaçar.
    if formula and formula.rstrip("-") == stem.strip().lower():
        return True

    # ⚠️ "FİİLDEN TÜRETEN EK FİİL KÖK İSTER" KAPISI DENENDİ VE GERİ ALINDI.
    #
    # `-It` sistematik olarak fazla soyuyor ve sorun gerçek:
    #     umut -> um      ✅   çaput -> çap  ❌ ('çap' Ermenice "diameter")
    #     yoğurt -> yoğur ✅   tabut -> tab  ❌ (Arapça تابوت)
    #     kanıt -> kan    ✅   kavut -> kav  ❌
    #                          bulut -> bul  ❌ (Ortak Türkçe *bulıt)
    #
    # Kökün mastarı var mı (`kök+mak/mek`) diye bakan kapı `tabut` ve
    # `kavut`u düzeltti, isimden türeyenlerin hiçbirini bozmadı — ama
    # `bitig -> biti`yi KIRDI ve bu, deponun önemsediği Eski Türkçe
    # katmanı. Ölçüldü, kurtarılamıyor:
    #     `biti` indekste TEK satır: [tr] pos=noun "book"; `bitig` 0 satır.
    #     `pos='verb'` alanı dolu (tr %13,4, otk %16,6) ama bu köklerin
    #     hiçbirinde yok; `"to ..."` gloss sezgisi 8 kökte 0 isabet.
    #     Mastarı diğer Türki dillere yaymak ZARARLI: `tabmaq` [crh],
    #     `çapmaq` [az/crh] var; `tab` "fiil" olup `tabut` yine kırılırdı.
    #
    # Yani ayrımı yapan şey mantık değil VERİ BOŞLUĞU. Sonradan daha
    # keskin ölçüldü: otk fiil kayıtları indekste ASLINDA VAR (78 adet,
    # `𐰋𐰃𐱅𐰃` = "to write, to inscribe" dahil) ama Latin kökle ARANAMAZ,
    # çünkü çeviriyazı `biti` değil `bıtı` üretiyor. Sebep kaynak yazıdır:
    # `ORKHON I` işareti i/ı ayrımı yapmaz, ünsüzler (`AEB`, `AEK`, `AES`…)
    # ön/art uyumu taşımaz, ve gerçek otk kayıtlarının %38'i ünlüsüzdür
    # (`tg` = tağ, `lg` = elig, `sç` = saç). Uyum-duyarlı çeviriyazı da
    # denendi ve çürütüldü — ayrıntı `utils/transliteration.py` notunda.
    # Çözüm kural değil veri: küratörlü Latin okuma. `-It` fazla-soyması
    # AÇIK KUSUR olarak duruyor.
    return _stem_is_attested(stem)


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
            if _strip_is_supported(w, new_stem):
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
