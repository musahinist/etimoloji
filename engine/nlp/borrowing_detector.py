"""
Açıklamalı alıntı tespiti — "neden alıntı?" sorusuna cevap veren katman.

Mevcut sistemler (seabor, PyBor) **ikili sınıflandırıcıdır**: "alıntı" veya
"miras" der, gerekçe vermez. Buradaki fark, her kararın adı konmuş
kanıtlara dayanması ve **miras olsaydı beklenen biçmin** hesaplanmasıdır::

    kitap  →  ALINTI
      · Türki dillerde biçim neredeyse aynı (ç=0,95): miras kelimeler
        düzenli ses farkları gösterir, bu göstermiyor
      · söz başı /k/ + söz içi /t/ Oğuz ötümlüleşmesinden geçmemiş;
        miras olsaydı beklenen Türkçe biçim: *gidap
      · zincir: Türkçe kitap ← Osmanlıca كتاب ← Arapça كِتَاب

Ölçülmüş gerekçe: negatif kontrol bataryasında alıntı tuzakları (kitap,
duvar, çorap, sabun, pencere, çay) **%100 yanlış-pozitif** veriyordu — motor
hepsini rekonstrükte edilebilir sayıyordu.

Dört bağımsız sinyal kullanılır; hiçbiri tek başına karar vermez:

============================  =============================================
``zincir_kanıtı``             sözlükte tanıklanmış verici dil ve yol
``fonotaktik_ihlal``          Proto-Türkçe'de bulunmayan ses veya dizim
``ses_kanunu_ihlali``         beklenen refleks tutmuyor (özgün katkı)
``değişimsiz_yayılım``        bütün dillerde neredeyse aynı biçim
``verici_yakınlığı``          verici dil sözlüğünde aynı kavramın karşılığı
                              fonetik olarak neredeyse aynı (sabor)
``fonotaktik_model``          eğitilmiş dizilim modeli kelimeyi alıntı
                              sınıfında daha olası buluyor (PyBor)
============================  =============================================
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from engine.logging_setup import get_logger
from engine.nlp.donor_proximity import attribute_donor, nearest_donor, proximity_strength
from engine.nlp.proto_phonology import PROHIBITED_INITIALS
from engine.utils.orthography import to_comparison_form
from engine.utils.phonotactics import VOWELS, has_vowel_harmony

logger = get_logger(__name__)

#: Ağırlıklar. Zincir kanıtı en güçlüsüdür çünkü doğrudan tanıklamadır;
#: ötekiler dolaylı göstergedir.
#:
#: ``verici_yakınlığı`` alanın **ölçülmüş en güçlü tek sinyalidir**
#: (Miller & List 2023, ``sabor``: F1 0,806, kesinlik 0,931). Zincir
#: kanıtından sonra en yüksek ağırlığı alır; ondan düşük olmasının sebebi
#: zincirin DOĞRUDAN tanıklama, bunun ise çıkarım olmasıdır.
#:
#: ⚠️ ``ses_kanunu_ihlali`` ve ``değişimsiz_yayılım`` ağırlığı **sıfırdır**.
#: Sinyaller hesaplanmaya ve kullanıcıya gösterilmeye devam eder (gerekçe
#: değeri taşırlar) ama **karara katılmazlar**. Ölçüldü (WOLD/Sakha, n=769,
#: her ablasyonda eşik yeniden ayarlanmış)::
#:
#:     çıkarılan sinyal                              F      doğruluk
#:     yok                                        0,6235     0,7581
#:     ses_kanunu_ihlali                          0,6318     0,7529
#:     değişimsiz_yayılım                         0,6304     0,7529
#:     ses_kanunu_ihlali + değişimsiz_yayılım     0,6417     0,7763
#:
#: ⚠️ **Yukarıdaki tablo RAPOR yarısındandır ve karar gerekçesi DEĞİLDİR.**
#: Rapor yarısına bakıp ağırlık seçmek, ölçümün içine ayar sızdırmaktır.
#: Karar AYAR yarısının kendi içinde bölünerek verildi (iç-ayar / iç-doğrulama,
#: n=385; rapor yarısı hiç görülmedi)::
#:
#:     çıkarılan sinyal                            F      doğruluk
#:     yok                                      0,6016     0,7455
#:     ses_kanunu_ihlali                        0,6016     0,7455
#:     değişimsiz_yayılım                       0,6016     0,7455
#:     ses_kanunu_ihlali + değişimsiz_yayılım   0,6016     0,7455
#:
#: Yani ayar verisinde iki sinyal **hiçbir kararı değiştirmiyor** — ne iyi
#: ne kötü. Gerekçe budur: hiçbir karara katkısı olmayan bir sinyal
#: toplamın %20'sini taşımamalıdır. Rapor yarısındaki +0,023'lük iyileşme
#: ayar verisinde ÖNGÖRÜLMEMİŞTİR; bağımsız olarak doğrulanmadı ve üst
#: sınır sayılmalıdır.
#:
#: Eğitilmiş birleştirici de bağımsız olarak aynı yöne varıyor: bu ikisine
#: +0,118 ve -0,022 katsayı veriyor, verici yakınlığına +1,534.
#:
#: ⚠️ Kavramsal açıklama **sınandı ve doğrulanmadı**. Denklikleri alıntıların
#: da içinde olduğu veriden öğrendiğimiz için sinyalin zayıf olduğunu
#: düşünüyorduk; yalnız uzmanın ata biçim verdiği kümelerden ikinci bir
#: tablo öğrenildi (bkz. ``INHERITED_CORRESPONDENCE_PATH``) ve sinyalin
#: katkısı -0,0101'den yalnız -0,0083'e geldi. Yani teşhis yanlıştı;
#: yöntemin kendisi bu görevde zayıf.
#: ⚠️ Bu tablo **yedek yoldur**. Asıl karar eğitilmiş birleştiricidedir
#: (``borrowing_combiner``); bu ağırlıklar yalnız model dosyası yokken
#: kullanılır ve o durum ``verdict.is_trained == False`` ile ilan edilir.
#:
#: ⚠️ ``fonotaktik_model`` ağırlığı **sıfırdır** ve bu bilinçlidir. Sinyal
#: eğitilmiş birleştiriciye girer (orada +0,878 katsayı alır) ama elle
#: ağırlıklandırılmış toplama girmez. Ölçüldü (WOLD/Sakha, n=769)::
#:
#:     el ağırlıklı toplam, fonotaktik_model DIŞARIDA       F 0,6461
#:     el ağırlıklı toplam, öğrenilen katsayılar normalize  F 0,6137
#:     eğitilmiş birleştirici (sigmoid + sabit terim)       F 0,6513
#:
#: Öğrenilen katsayıları normalize edip doğrusal toplama koymak lojistik
#: modelin davranışını **yeniden üretmiyor**: sabit terim (-1,993) ve
#: sigmoid, kararın parçasıdır. Doğrusal yedek yol bu sinyali taşıyamıyor.
SIGNAL_WEIGHTS: dict[str, float] = {
    "zincir_kanıtı": 0.50,
    "verici_yakınlığı": 0.32,
    "fonotaktik_ihlal": 0.18,
    "fonotaktik_model": 0.0,
    "ses_kanunu_ihlali": 0.0,
    "değişimsiz_yayılım": 0.0,
}

#: Proto-Türkçe'de söz başında bulunmayan ama bu dilde DÜZENLİ miras
#: refleksi olan sesler. ``fonotaktik_ihlal`` bunlar için ateşlenmez.
#:
#: * ``sah``: karşılaştırma biçimindeki söz başı ``h`` iki düzenli miras
#:   refleksidir: ``χ`` < *k- art ünlü önünde (``χaːr`` "kar" < *kār,
#:   ``χaːn`` "kan" < *kān) ve ``һ`` < *s-. Ölçüldü (WOLD Saha, AYAR yarısı
#:   n=770): söz başı h- 49 miras / 20 alıntı — alıntı oranı 0,29, taban
#:   0,28; ayırt edici değil ve sinyalin Saha'daki ateşlemelerinin çoğu buydu.
#: * ``ba``: Başkurtçada da *s- > h- düzenlidir (``һыу`` "su"). Ölçülmedi —
#:   bu dilde etiketli alıntı kümesi yok; kural ses tarihinden.
REGULAR_INITIALS: dict[str, str] = {"sah": "h", "ba": "h"}

#: Birleşik söz bileşenlerini ayıran işaretler (WOLD ``kün_ortoto``).
_COMPOUND_SEPARATOR = re.compile(r"[_\-\s]+")

#: Bu eşiğin üstünde kelime **alıntı olarak raporlanır**.
BORROWING_THRESHOLD = 0.45

#: Bu eşiğin üstünde miras rekonstrüksiyonu **hiç yapılmaz**.
#:
#: ⚠️ İki ayrı eşik gerekiyor çünkü zincir sinyali tek başına EŞADLILARA
#: takılabiliyor. Ölçüldü: tek eşikle altın standarttaki 400 maddenin 26'sı
#: yanlışlıkla engellendi (`bil`, `ben`, `bär`, `dal` — hepsi miras, ama
#: sözlükte aynı yazılışta bir alıntı madde de var). Rekonstrüksiyon
#: doğruluğu %22,3'ten %20,0'a düşüyordu.
#:
#: Engelleme yalnız zincir kanıtı BAŞKA bir sinyalle desteklendiğinde
#: yapılır. Arada kalan kelimeler "alıntı" diye raporlanır ama motor yine de
#: miras hipotezini kurar ve kullanıcı iki okumayı da görür.
BLOCK_THRESHOLD = 0.55


#: Fiil kökünün sözlükteki mastar biçimi (dil -> ekler). Sözlük fiili
#: MASTARLA tutar: ``duy`` satırı yalnız ad (Fr. *douille*) ve emir kipi
#: çekimidir, miras kayıt ``duymak`` (*tuy-) satırındadır.
INFINITIVE_SUFFIXES: dict[str, tuple[str, ...]] = {"tr": ("mak", "mek")}


def _lexical_origin_rows(
    index: Any, word: str, lang: str, *, exact: bool
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """``(alıntı kayıtları, miras kayıtları, bütün satırlar)`` — eşadlılık oranı için.

    ``_chain_signal`` ve ``_index_attests_loan`` AYNI kuralı kullanır; eskiden
    ikisi ayrı kopyaydı ve mastar kanıtı yalnız birinde vardı.

    Ölçülmüş hatalar (başlık × Starling, G6): ``duy``, ``sek``, ``ser``,
    ``tak``, ``gül``, ``büz``, ``dik``, ``kar`` miras fiil/ad köküne 1,0 güçle
    ``zincir_kanıtı`` ateşleniyordu. Dört ayrı sebep:

    * **Mastar.** Fiilin miras kaydı mastardadır (``duymak`` *tuy-); çıplak
      biçimde yalnız eşsesli alıntı ad görünür. Mastar kaydı miras sayılır.
    * **Ek satırları.** ``-kâr`` (Fa. ـکَار), ``-dik`` (*-tuk) karşılaştırma
      biçiminde ``kar``/``dik`` ile çakışır; ek, kelime değildir.
    * **Farklı yazılış.** ``kâr`` (Pehl. "kazanç") ≠ ``kar`` (*kār "snow").
      Karşılaştırma biçimi düzeltme işaretini siliyor. Birebir aynı yazılışta
      kayıt varsa yalnız onlar sayılır; yoksa (Saha IPA sorgusu ~ Kiril
      madde) eski karşılaştırma eşleşmesine dönülür.
    * **Tekrarlı alıntı satırı.** ``sek`` sıfat + zarf, ``dik`` sıfat + emir
      kipi aynı vericiyi (Fr. *sec*, Ç. 直) iki kez sayıyordu; 2 alıntı / 1
      miras = 0,67 eşadlılık eşiğini (0,6) geçiyordu. Alıntılar
      ``(verici, biçim)`` başına bir kez sayılır.

    :param exact: ``True`` ise yalnız birebir aynı yazılış (engelleme kararı
        için; bkz. ``_index_attests_loan``).
    """
    from engine.nlp.borrowing_chain import TURKIC_LINEAGE_CODES

    key = (word or "").strip()
    folded = key.casefold()

    def usable(rows: list[dict[str, Any]], headword: str) -> list[dict[str, Any]]:
        target = headword.casefold()
        rows = [r for r in rows if not _is_affix_row(r)]
        if headword == headword.lower():
            # Küçük harfli cins ad sorgusu için özel ad kaydı kanıt değildir.
            rows = [
                r
                for r in rows
                if r.get("pos") != "name" and not str(r.get("word") or "")[:1].isupper()
            ]
        same = [r for r in rows if str(r.get("word") or "").strip().casefold() == target]
        return same if (same or exact) else rows

    rows = usable(index.lookup(key, languages=[lang], limit=10) or [], key)

    def is_loan(r: dict[str, Any]) -> bool:
        donor = str(r.get("donor_lang") or "")
        return r.get("origin") == "alıntı" and bool(donor) and donor not in TURKIC_LINEAGE_CODES

    def is_inherited(r: dict[str, Any]) -> bool:
        return r.get("origin") == "miras" or (
            r.get("origin") == "alıntı"
            and str(r.get("donor_lang") or "") in TURKIC_LINEAGE_CODES
        )

    borrowed: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for r in rows:
        if not is_loan(r):
            continue
        donor_key = (str(r.get("donor_lang") or ""), str(r.get("donor_form") or ""))
        if donor_key in seen:
            continue
        seen.add(donor_key)
        borrowed.append(r)
    inherited = [r for r in rows if is_inherited(r)]
    if borrowed and folded:
        for suffix in INFINITIVE_SUFFIXES.get(lang, ()):
            infinitive = key + suffix
            found = usable(index.lookup(infinitive, languages=[lang], limit=5) or [], infinitive)
            match = next((r for r in found if is_inherited(r)), None)
            if match is not None:
                inherited.append(match)
                break
    return borrowed, inherited, rows


def _is_affix_row(row: dict[str, Any]) -> bool:
    """``-kâr``, ``-dik`` gibi ek satırı mı?"""
    return row.get("pos") in ("suffix", "prefix", "infix", "affix", "interfix") or str(
        row.get("word") or ""
    ).strip().startswith("-")


def _index_attests_loan(word: str) -> bool:
    """Sözlük bu kelimeyi DOĞRUDAN alıntı olarak tanıklıyor mu?

    Toplam skor muhafazakâr eşiğin (0.55) altında kalsa bile, indeksin
    doğrudan tanıklığı miras rekonstrüksiyonunu engellemeye yeter — bu,
    mümkün olan en güçlü kanıttır.

    Ölçüldü (negatif kontrol, alıntı tuzakları): dördünde de `zincir_kanıtı`
    ateşleniyor ama skor tam 0.50'de kalıp eşiğin kılpayı altında kalıyordu::

        kitap   0.59  engellenirdi        duvar   0.50  engellenmiyordu (fa-cls)
        çorap   0.50  engellenmiyordu     pencere 0.50  engellenmiyordu (fa-cls)
        sabun   0.50  engellenmiyordu (ar)

    Eşiği düşürmek yerine tanıklığa bakılıyor; eşik muhafazakâr kalıyor.

    Üç koruma — üçü de ÖLÇÜLMÜŞ vakalardan:
      * Soy kodları (`trk-pro`, `otk`…) verici değildir.
      * Ters yönlü kayıtlar (`KNOWN_REVERSED_LOAN_DIRECTION`): `öküz`
        indekste `hu` vericili "alıntı" görünür ama çekirdek Ortak
        Türkçedir; süzülmezse miras rekonstrüksiyonu engellenirdi.
      * Eşadlılık — MASTAR KANIDI DAHİL. `gül` indekste yalnız alıntı
        (fa-cls "rose") görünür, miras kanıtı `gülmek`tedir. Mastar
        sayılmazsa altın kümedeki `gül=*kül` (LAUGH) engellenirdi.

    Etki alanı ölçüldü: 5 tuzağın 5'i engellenir; `göz`, `bardak`, `deniz`,
    `su`, `yaş`, `kat`, `çay`, `gül`, `yaz`, `öküz` engellenmez; altın dev
    kümesinde engellenen madde sayısı 0.
    """
    key = (word or "").strip().lower()
    if not key:
        return False
    try:
        from engine.db.lexicon_index import LexiconIndex
        from engine.nlp.loanword_classifier import KNOWN_REVERSED_LOAN_DIRECTION

        if key in KNOWN_REVERSED_LOAN_DIRECTION:
            return False

        index = LexiconIndex()
        if not index.exists:
            return False

        # ⚠️ TAM EŞLEŞME ŞART. `index.lookup` karşılaştırma biçmi üzerinden
        # arıyor ve CLDF işaretlerini normalleştiriyor (`š`->`ş`, `ï`->`ı`,
        # uzunluk `:` düşüyor). Ölçüm hattı proto/tanık biçimleri besliyor
        # ve bu yüzden alakasız Türkçe maddelere çarpıyordu::
        #
        #     keš     -> 'keş'   (fa "drug addict")
        #     kïrba:  -> 'kırba' (ar "waterskin")
        #
        # Bu iki çarpışma altın dev kümesinde 2 maddeyi çekimser bırakıp
        # NED'i 0.302'den 0.3261'e bozuyordu. Tuzak kelimelerin hepsi zaten
        # tam eşleşiyor (duvar, kitap, çorap, sabun, pencere), yani şart
        # onları etkilemiyor.
        # Mastar kanıtı, ek satırları ve tekrarlı alıntı satırları:
        # bkz. `_lexical_origin_rows`.
        loans, inherited, _ = _lexical_origin_rows(index, key, "tr", exact=True)
        if not loans:
            return False

        if inherited and len(loans) / (len(loans) + len(inherited)) < 0.6:
            return False
        return True
    except Exception:
        logger.debug("Alıntı tanıklığı okunamadı: %s", key, exc_info=True)
        return False

#: Bu benzerlik oranının üstündeki yayılım şüphelidir. Miras kelimeler
#: bin yılda düzenli ses farkları biriktirir; birikmemişse yayılım yenidir.
UNIFORMITY_SUSPICION = 0.85


def _language_name(code: str) -> str:
    from engine.nlp.borrowing_chain import language_name

    return language_name(code)


def own_sense(word: str, lang: str) -> str:
    """Kelimenin sözlük indeksindeki KENDİ kaydının anlamı; yoksa boş dizgi.

    ⚠️ Verici yakınlığı sinyali anlam kısıtlıdır: anlam yoksa hiç aday
    aranmaz ve sinyal ATEŞLENMEZ. Ölçüm hattı anlamı kendisi veriyordu
    (Türkçe altın küme: indeksin ``gloss`` alanı; WOLD: kavram adı) ama
    üretimdeki çağıranlar (``HypothesisRanker.rank``,
    ``ComparativeReconstructor._borrowing_verdict``) ``sense`` geçmiyordu.
    Sonuç: WOLD'da ölçülmüş en güçlü sinyal (birleştirici katsayısı +1,55)
    arama yolunda HİÇ yoktu. Ölçüldü (denetim örneklemi, tohum 21, 20 alıntı
    + 20 miras, yalnız yerel kaynaklar): önce 0/40 kelimede ateşleme; sonra
    alıntıların 15/20'sinde, mirasların 5/20'sinde (duyar, eğilmek, aslan,
    yılmaz, gaga — sıralayıcı bunları ALINTI'ya çevirdi; altın uyumu
    28/40 -> 26/40). Ölçüm hattıyla aynı sinyal artık üretimde; yanlış
    pozitiflerin ikisi (eğilmek, yılmaz) şans denetimi YAPILAMAYAN uzun
    kelimeler (``donor_proximity._controls`` 8'den az kontrol).

    Kural ölçüm hattıyla **aynıdır** (``borrowing_eval._turkish_glosses``):
    aynı dil, birebir aynı yazılış, boş olmayan ilk ``gloss``. Farklı bir
    kural (ör. TDK'nın Türkçe tanımı) verici sözlüklerinin İngilizce
    anlamlarıyla örtüşmez ve ölçülmemiş bir sinyal üretirdi.
    """
    key = (word or "").strip()
    if not key or not lang:
        return ""
    try:
        from engine.db.lexicon_index import LexiconIndex

        index = LexiconIndex()
        if not index.exists:
            return ""
        with index.connect() as connection:
            row = connection.execute(
                "SELECT gloss FROM entries WHERE lang_code = ? AND word = ? "
                "AND gloss IS NOT NULL AND gloss != '' LIMIT 1",
                (lang, key),
            ).fetchone()
        return str(row["gloss"]) if row else ""
    except Exception:
        logger.debug("Kelimenin anlamı okunamadı: %s", key, exc_info=True)
        return ""


def default_donors(lang: str) -> list[str] | None:
    """Dilin verici sözlükleri — ölçüm hattıyla aynı küme.

    ⚠️ ``None`` bütün verici sözlükleri demektir; havuz büyüdükçe şans
    benzerliği artar (bkz. ``donor_proximity.CHANCE_CONTROL_COUNT``).
    Üretim ``None`` ile, ölçüm ``donors_for(lang)`` ile çalışsaydı ölçülen
    sinyal üretimdekiyle aynı sinyal olmazdı.
    """
    from engine.evaluation.borrowing_eval import donors_for

    return donors_for(lang)


@dataclass
class Signal:
    """Tek bir kanıt kalemi."""

    name: str
    fired: bool
    strength: float
    explanation: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "signal": self.name,
            "fired": self.fired,
            "strength": round(self.strength, 3),
            "explanation": self.explanation,
            "evidence": self.evidence,
        }


@dataclass
class BorrowingVerdict:
    """Bir kelime için alıntı kararı ve tam gerekçesi."""

    word: str
    score: float
    signals: list[Signal] = field(default_factory=list)
    expected_if_inherited: str = ""
    chain: list[str] = field(default_factory=list)
    donor_language: str = ""
    #: **Eğitilmiş** birleştiricinin olasılığı. ``None`` ise model yok ve
    #: el ağırlıkları kullanılıyor — bu durum ilan edilir, gizlenmez.
    trained_probability: float | None = None
    #: Eğitilmiş modelin künyesi (hangi veride, kaç örnekle, hangi hedefle).
    combiner_note: str = ""
    #: Sorgunun dili — ``expected_if_inherited`` bu dilin beklenen biçimidir.
    lang: str = "tr"

    @property
    def is_trained(self) -> bool:
        return self.trained_probability is not None

    @property
    def is_borrowed(self) -> bool:
        """Karar. Eğitilmiş model varsa **o** karar verir.

        ⚠️ El ağırlıklı toplam ölçüldü ve en güçlü sinyalin kararını
        bozuyordu: madde başına doğrulukta beş sinyalli motor (0,7035)
        yalnız verici yakınlığının (0,7334) ANLAMLI biçimde altındaydı
        (fark -0,030, %95 GA [-0,049, -0,010], p=0,004). Eğitilmiş
        birleştirici F'yi 0,5839'dan 0,5982'ye çıkarıyor.
        """
        if self.trained_probability is not None:
            return self.trained_probability >= self._trained_threshold
        return self.score >= BORROWING_THRESHOLD

    #: Eğitilmiş modelin karar eşiği; ``detect`` doldurur.
    _trained_threshold: float = 0.5

    @property
    def blocks_inherited_reconstruction(self) -> bool:
        """Kanıt, miras rekonstrüksiyonunu hiç denememeyi haklı çıkarıyor mu?

        ⚠️ Engelleme kararı **el skoruna** bağlı kalır. Eğitilmiş model F
        için ayarlıdır, yani bilinçli olarak duyarlılık yönüne kayar; o
        eşikle rekonstrüksiyonu engellemek miras kelimeleri susturur.
        Engelleme daha muhafazakâr bir karardır ve muhafazakâr eşikte kalır.
        """
        # Skor eşiği muhafazakâr kalır; ama sözlüğün DOĞRUDAN tanıklığı
        # tek başına yeter (bkz. `_index_attests_loan`). Eşiği düşürmek
        # miras kelimeleri susturacaktı, tanıklığa bakmak susturmuyor.
        return self.score >= BLOCK_THRESHOLD or _index_attests_loan(self.word)

    @property
    def verdict(self) -> str:
        """Sözel karar. ``is_borrowed`` ile **aynı** kaynaktan gelmeli.

        ⚠️ Eskiden ikisi ayrı hesaplanıyordu; eğitilmiş model devreye
        girince ``is_borrowed=True`` ile ``verdict="miras adayı"`` aynı anda
        çıkabilirdi.
        """
        if self.trained_probability is not None:
            if self.trained_probability >= self._trained_threshold:
                return "alıntı"
            if self.trained_probability >= self._trained_threshold / 2:
                return "belirsiz"
            return "miras adayı"
        if self.score >= BORROWING_THRESHOLD:
            return "alıntı"
        if self.score >= BORROWING_THRESHOLD / 2:
            return "belirsiz"
        return "miras adayı"

    def explain(self) -> str:
        """İnsan-okunur gerekçe — çıktının asıl değeri budur."""
        lines = [f"{self.word} → {self.verdict.upper()} (skor {self.score:.2f})"]
        for signal in self.signals:
            if signal.fired:
                lines.append(f"  · {signal.explanation}")
        if self.expected_if_inherited:
            lines.append(
                f"  · miras olsaydı beklenen {_language_name(self.lang)} biçim: "
                f"{self.expected_if_inherited}"
            )
        if self.chain:
            lines.append(f"  · zincir: {' ← '.join(self.chain)}")
        if not any(s.fired for s in self.signals):
            lines.append("  · alıntı göstergesi bulunamadı")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "verdict": self.verdict,
            "is_borrowed": self.is_borrowed,
            "score": round(self.score, 3),
            "donor_language": self.donor_language,
            "chain": self.chain,
            "expected_if_inherited": self.expected_if_inherited,
            "signals": [s.as_dict() for s in self.signals],
            "trained": self.is_trained,
            "trained_probability": (
                round(self.trained_probability, 4)
                if self.trained_probability is not None
                else None
            ),
            "combiner_note": self.combiner_note,
            "explanation": self.explain(),
        }


class BorrowingDetector:
    """Dört sinyalli, gerekçeli alıntı tespiti."""

    #: Eğitilmiş birleştirici bir kez yüklenir; yoksa ``None`` kalır.
    _COMBINER: Any = None
    _COMBINER_LOADED = False

    @property
    def combiner(self) -> Any:
        """Eğitilmiş birleştirici — yoksa ``None``.

        ⚠️ Model dosyası yoksa **el ağırlıklarına dönülür ama bu ilan
        edilir** (``verdict.is_trained == False``). Kalibre edilmemiş bir
        skoru kalibreymiş gibi sunmak, hiç kalibre etmemekten kötüdür.
        """
        cls = type(self)
        if not cls._COMBINER_LOADED:
            from engine.nlp.borrowing_combiner import load

            cls._COMBINER = load()
            cls._COMBINER_LOADED = True
        return cls._COMBINER

    @classmethod
    def reset_combiner_cache(cls) -> None:
        cls._COMBINER = None
        cls._COMBINER_LOADED = False

    def __init__(self, index: Any = None, predictor: Any = None):
        self._index = index
        self._predictor = predictor
        self._inherited_predictor = predictor

    @property
    def index(self) -> Any:
        if self._index is None:
            from engine.db.lexicon_index import LexiconIndex

            self._index = LexiconIndex()
        return self._index

    @property
    def predictor(self) -> Any:
        if self._predictor is None:
            from engine.nlp.cognate_prediction import CognatePredictor

            self._predictor = CognatePredictor()
        return self._predictor

    @property
    def inherited_predictor(self) -> Any:
        """Yalnız MİRAS kümelerden öğrenilmiş tablolarla tahmin eder.

        ⚠️ Miras tablosu yoksa ana tabloya dönülür — ama o tablo alıntıları
        da içerir ve sinyal ölçülmüş biçimde zayıflar. Dönüş sessiz değildir:
        günlüğe yazılır.
        """
        if self._inherited_predictor is None:
            from engine.nlp.cognate_prediction import (
                CognatePredictor,
                load_inherited_tables,
            )

            tables = load_inherited_tables()
            if not tables:
                logger.info(
                    "Miras denklik tablosu yok; ana tabloya dönülüyor "
                    "(alıntılar da içinde — ses kanunu sinyali zayıflar)"
                )
                self._inherited_predictor = self.predictor
            else:
                self._inherited_predictor = CognatePredictor(tables)
        return self._inherited_predictor

    # -- sinyaller ----------------------------------------------------------

    def _chain_signal(self, word: str, lang: str) -> tuple[Signal, list[str], str]:
        """Sözlükte tanıklanmış alıntı kaydı var mı?

        En güçlü sinyal, çünkü dolaylı gösterge değil doğrudan tanıklamadır.
        Zincir çok halkalı olabilir: Türkçe ← Osmanlıca ← Arapça.
        """
        from engine.nlp.borrowing_chain import language_name

        if not getattr(self.index, "exists", False):
            return (
                Signal("zincir_kanıtı", False, 0.0, "sözlük indeksi yok", {"no_data": True}),
                [],
                "",
            )
        # ⚠️ ÖZEL AD ELEMESİ. Arama `comparison` alanı üzerinden yapılır ve o
        # alan büyük/küçük harf ayırmaz; böylece özel adlar cins adın
        # sorgusuna düşer. Ölçüldü: `aya` için indekste doğru kayıt VAR
        # (miras, Eski Türkçe *hāya "elin iç tarafı") ama büyük harfli `Aya`
        # (alıntı, Yunanca Αγία "aziz") kayıtları da dönüyordu. Aşağıdaki
        # eşadlılık koruması bunu yakalayamadı: 2 alıntı / 1 miras = 0,667,
        # eşik 0,6 — koruma ateşlenmedi ve özel ad, cins adı 1.0 güçle alıntı
        # ilan edip MİRAS hipotezini reddettirdi.
        # Küçük harfli bir cins ad sorgusu için özel ad kaydı kanıt DEĞİLDİR.
        #
        # ⚠️ ATA KATMANI VERİCİ DEĞİLDİR. Sözlük `donor_lang` alanını ata
        # biçimi kaydetmek için de kullanıyor; `bardak` satırı
        # ('bardak', 'alıntı', 'trk-eog', 'برت') diyor ve trk-eog =
        # Eski Oğuzca, yani Türkçenin ATASI. Süzülmezse öz Türkçe kelime
        # 1.0 güçle "alıntı" ilan edilir ve DOĞRU miras hipotezi bu
        # gerekçeyle reddedilir — `bardak` çıktısında ölçülen tam buydu.
        #
        # Aynı koruma `search_engine.py`'de vardı (bkz. oradaki not:
        # "Proto-Türkçe bir verici dil DEĞİLDİR... ölçüldü: göz -> yanlış
        # damga") ama sıralayıcıyı besleyen bu yola konmamıştı.
        #
        # Ölçüldü (indeks, tr): `origin='alıntı'` + soy kodu = 39 kayıt,
        # hepsi öz Türkçe (bilge, betik <- *bitig, kamu, sav, başkan,
        # anlamak, karınca, küsmek, evren, keçe, tin, bardak...).
        # `ota` ve kardeş Türki diller bilerek süzülmüyor — gerekçe
        # `TURKIC_LINEAGE_CODES` tanımında.
        #
        # Mastar kanıtı, ek satırları, farklı yazılış ve tekrarlı alıntı
        # satırları (G6: duy, sek, ser, tak, gül, büz, dik, kar): bkz.
        # `_lexical_origin_rows`.
        borrowed, inherited, rows = _lexical_origin_rows(self.index, word, lang, exact=False)

        # ⚠️ EŞADLILIK. Aynı yazılışta hem miras hem alıntı madde olabilir:
        # Türkçe `su` (miras, "water") ile Fransızca kökenli bir `su` maddesi
        # sözlükte yan yanadır. Yalnız "alıntı kaydı var mı" diye bakmak,
        # miras kelimeyi alıntı ilan eder. Kanıt gücü, alıntı kayıtlarının
        # ORANIYLA ölçülür; miras kayıt varsa sinyal zayıflar.
        if borrowed and inherited:
            share = len(borrowed) / (len(borrowed) + len(inherited))
            if share < 0.6:
                return (
                    Signal(
                        "zincir_kanıtı",
                        False,
                        share,
                        (
                            f"eşadlılık: sözlükte {len(inherited)} miras, "
                            f"{len(borrowed)} alıntı kaydı var — kanıt belirsiz"
                        ),
                        {"borrowed_entries": len(borrowed), "inherited_entries": len(inherited)},
                    ),
                    [],
                    "",
                )
        if not borrowed:
            return (
                Signal(
                    "zincir_kanıtı",
                    False,
                    0.0,
                    # Yalnız `lang` sözlüğüne bakıldı; "sözlükte" demek, başka
                    # katmanda alıntı kaydı olan kelimede (bitig: otk kaydı
                    # Orta Çince 筆 der) düpedüz yanlış bir cümleydi.
                    f"{language_name(lang)} sözlüğünde alıntı kaydı yok",
                    {"checked_entries": len(rows)},
                ),
                [],
                "",
            )
        row = borrowed[0]
        donor = str(row.get("donor_lang") or "")
        chain = [
            f"{language_name(lang)} {row['word']}",
            f"{language_name(donor)} {row.get('donor_form') or '?'}",
        ]
        return (
            Signal(
                "zincir_kanıtı",
                True,
                1.0,
                f"sözlükte alıntı olarak tanıklanmış: verici dil {language_name(donor)}",
                {"donor_lang": donor, "donor_form": row.get("donor_form", "")},
            ),
            chain,
            donor,
        )

    @staticmethod
    def _phonotactic_signal(word: str, lang: str = "tr") -> Signal:
        """Proto-Türkçe'de bulunmayan ses veya dizim var mı?

        ⚠️ Kurallar dile özgüdür (bkz. :data:`REGULAR_INITIALS`): Saha'da
        söz başı ``h-`` (*k- > χ-, *s- > һ-) düzenli miras refleksidir ve
        alıntı göstergesi değildir.

        ⚠️ Ünlü uyumu **bileşenlere ayrı ayrı** bakılır. WOLD Saha maddelerinin
        bir kısmı birleşik sözdür (``kün_ortoto``, ``uon_biːr``); karşılaştırma
        biçimi ayracı sildiği için iki uyumlu kelime tek "uyumsuz" kelime
        sayılıyordu.
        """
        form = to_comparison_form(word)
        violations: list[str] = []
        if not form:
            return Signal("fonotaktik_ihlal", False, 0.0, "biçim çözümlenemedi", {"no_data": True})

        if form[0] in PROHIBITED_INITIALS and form[0] not in REGULAR_INITIALS.get(lang, ""):
            violations.append(f"Proto-Türkçe'de söz başı *{form[0]}- bulunmaz")
        if len(form) >= 2 and form[0] not in VOWELS and form[1] not in VOWELS:
            violations.append("söz başı ünsüz kümesi — Türkçede bulunmaz")
        parts = [to_comparison_form(p) for p in _COMPOUND_SEPARATOR.split(word or "")]
        if any(
            len([ch for ch in part if ch in VOWELS]) >= 2 and not has_vowel_harmony(part)
            for part in parts
            if part
        ):
            violations.append("ünlü uyumu ihlali")

        strength = min(1.0, len(violations) / 2)
        return Signal(
            "fonotaktik_ihlal",
            bool(violations),
            strength,
            "; ".join(violations) if violations else "fonotaktik olarak Türkçeye uygun",
            {"violations": violations},
        )

    def _sound_law_signal(
        self, word: str, witnesses: dict[str, str], lang: str = "tr"
    ) -> tuple[Signal, str]:
        """**Özgün katkı:** miras olsaydı beklenen biçim tutuyor mu?

        Miras bir kelime, akraba dillerdeki biçimleriyle **düzenli** ses
        denklikleri gösterir. Öğrenilmiş denklik tablolarıyla her tanıktan
        ``lang`` dilindeki biçim tahmin edilir; tahminler gerçek biçme
        uymuyorsa kelime ses kanunlarının işlediği dönemde dilde yoktu demektir.

        ⚠️ **Hedef, sorgunun KENDİ dilidir.** Eskiden hedef sabit ``"tr"``di:
        Saha kelimesi için akrabalardan beklenen TÜRKÇE refleks Saha biçimiyle
        kıyaslanıyordu (``küöl`` ~ beklenen ``göl``) ve sinyal Saha'da
        yapısal olarak anlamsızdı — WOLD birleştiricisi ona −0,27 katsayı
        veriyordu (ayar yarısı analizinde −0,38). ``(kaynak, lang)`` denklik
        tablosu olmayan tanık tahmin üretmez (güven 0); dilin hiç tablosu
        yoksa sinyal "tahmin üretmedi" ile kapanır.

        ⚠️ **Beklenti MİRAS tablosundan okunur** (bkz.
        ``cognate_prediction.INHERITED_CORRESPONDENCE_PATH``). Ana tablo
        bütün akraba kümelerinden öğrenilir ve alıntıları da içerir; Arapça
        alıntılar bütün Oğuz dillerinde aynı biçimde uyarlandığı için kendi
        düzenli denkliklerini yaratıp bu testi geçiyorlardı. Ölçüldü:
        sinyalin katkısı o hâlde **-0,0101** (zararlı) idi.
        """
        actual = to_comparison_form(word)
        if not actual or len(witnesses) < 2:
            return (
                Signal("ses_kanunu_ihlali", False, 0.0, "yeterli tanık yok", {"no_data": True}),
                "",
            )

        expectations: list[str] = []
        for source_lang, source_form in sorted(witnesses.items()):
            if source_lang == lang:
                continue
            prediction = self.inherited_predictor.predict(source_form, source_lang, lang)
            if prediction.form and prediction.confidence > 0:
                expectations.append(prediction.form)
        if not expectations:
            return (
                Signal("ses_kanunu_ihlali", False, 0.0, "denklik tablosu tahmin üretmedi", {"no_data": True}),
                "",
            )

        expected, votes = Counter(expectations).most_common(1)[0]
        agreement = sum(1 for e in expectations if e == actual) / len(expectations)

        # ⚠️ Sinyal ancak beklenti KENDİ İÇİNDE tutarlıysa anlamlıdır.
        # Tek bir gürültülü tahmin ("Çuvaşça şıv -> Türkçe şa") miras bir
        # kelimeyi alıntı ilan edebilir. Üç koşul birden aranır:
        #   1. en az üç tanıktan tahmin üretilmiş olmalı
        #   2. tahminlerin çoğunluğu AYNI biçimde birleşmeli
        #   3. o ortak beklenti gerçek biçimden farklı olmalı
        consensus = votes / len(expectations)
        fired = (
            len(expectations) >= 3
            and consensus >= 0.5
            and agreement < 0.34
            and expected != actual
        )
        return (
            Signal(
                "ses_kanunu_ihlali",
                fired,
                1.0 - agreement,
                (
                    f"akraba biçimlerden beklenen {_language_name(lang)} refleks {expected!r}, "
                    f"gerçek biçim {actual!r} — ses kanunları işlememiş"
                    if fired
                    else f"beklenen refleks tutuyor (uyum {agreement:.2f})"
                ),
                {
                    "expected": expected,
                    "actual": actual,
                    "agreement": round(agreement, 3),
                    "target_lang": lang,
                },
            ),
            expected if fired else "",
        )

    @staticmethod
    def _uniformity_signal(witnesses: dict[str, str]) -> Signal:
        """Bütün dillerde neredeyse aynı biçim — yeni yayılım göstergesi.

        Miras kelimeler bin yılda düzenli ses farkları biriktirir
        (``göz ~ көз ~ küz ~ куҫ``). Farklar birikmemişse yayılım yenidir
        ve büyük olasılıkla ortak bir verici dilden gelmiştir
        (``kitap ~ kitap ~ kitap ~ kitob``).
        """
        forms = [to_comparison_form(f) for f in witnesses.values() if f]
        forms = [f for f in forms if f]
        if len(forms) < 3:
            return Signal("değişimsiz_yayılım", False, 0.0, "yeterli tanık yok", {"no_data": True})

        from engine.evaluation.metrics import normalized_edit_distance

        pairs = [
            1.0 - normalized_edit_distance(a, b)
            for i, a in enumerate(forms)
            for b in forms[i + 1 :]
        ]
        similarity = sum(pairs) / len(pairs) if pairs else 0.0
        fired = similarity >= UNIFORMITY_SUSPICION
        return Signal(
            "değişimsiz_yayılım",
            fired,
            max(0.0, (similarity - UNIFORMITY_SUSPICION) / (1 - UNIFORMITY_SUSPICION))
            if fired
            else 0.0,
            (
                f"Türki dillerde biçim neredeyse aynı (benzerlik {similarity:.2f}): "
                f"miras kelimeler düzenli ses farkları gösterir, bu göstermiyor"
                if fired
                else f"diller arası düzenli farklar var (benzerlik {similarity:.2f})"
            ),
            {"similarity": round(similarity, 3), "n_forms": len(forms)},
        )

    # -- karar --------------------------------------------------------------

    @staticmethod
    def _phonotactic_model_signal(word: str, lang: str) -> Signal:
        """Eğitilmiş dizilim modeli (PyBor, Miller ve ark. 2020).

        Elle yazılmış fonotaktik kurallar (ünlü uyumu, yasak söz başı ses)
        WOLD/Sakha'da tek başına **F 0,215** alıyor. Aynı veride eğitilmiş
        iki modelli sınıflandırıcı **F 0,568** alıyor — yayınlanmış PyBor
        ortalamasının (0,59-0,61) hemen altında, bağımsız bir yeniden
        üretim.

        ⚠️ Model **dile özgüdür**; başka dilin modeline dönülmez.
        Fonotaktik dilden dile değişir ve zaten ölçtüğü şey odur.
        """
        from engine.nlp.phonotactic_lm import load

        classifier = load(lang)
        if classifier is None:
            return Signal(
                "fonotaktik_model",
                False,
                0.0,
                f"{lang} için eğitilmiş dizilim modeli yok",
                {"no_data": True},
            )
        strength = classifier.strength(word)
        if strength <= 0.0:
            return Signal(
                "fonotaktik_model",
                False,
                0.0,
                "dizilim modeli miras sınıfını daha olası buluyor",
            )
        return Signal(
            "fonotaktik_model",
            True,
            strength,
            (
                f"eğitilmiş dizilim modeli alıntı sınıfını daha olası buluyor "
                f"(log oran {classifier.score(word):+.3f}, eşik "
                f"{classifier.threshold:+.3f})"
            ),
            {"log_ratio": round(classifier.score(word), 4), "language": lang},
        )

    @staticmethod
    def _donor_signal(word: str, sense: str, donors: list[str] | None) -> Signal:
        """Verici dil sözlüğüne fonetik yakınlık (sabor, Miller & List 2023).

        ⚠️ Mesafe **SCA**'dır, düz Levenshtein değil: Sakha Rusça ``stol``u
        ``ostuol`` yapar; düz uzaklık 0,50 verip eşiğin üstünde kalır, SCA
        0,216 verir.

        ⚠️ Arama **anlam kısıtlıdır**. Kısıtsız arama 440.910 maddelik Rusça
        sözlüğe yayılır ve şans benzerliğine açılır.
        """
        comparison = to_comparison_form(word)
        match = nearest_donor(comparison, sense, languages=donors)
        strength = proximity_strength(match)
        if match is None or strength <= 0.0:
            return Signal(
                "verici_yakınlığı",
                False,
                0.0,
                "verici sözlüğünde yakın karşılık yok",
            )
        # ⚠️ "Kimden?" ayrı bir adımdır (bkz. ``donor_proximity.attribute_donor``):
        # tek havuzdaki en yakın maddenin dili Moğolca alıntıların 79/166'sını
        # Rusça etiketliyordu. Etiket GÜCE girmez; güç yukarıda hesaplandı.
        evidence = match.as_dict()
        explanation = (
            f"verici sözlüğünde aynı kavramın karşılığı fonetik olarak yakın: "
            f"{match.describe()}"
        )
        attribution = attribute_donor(comparison, sense, languages=donors)
        if attribution is not None:
            evidence["attributed_lang"] = attribution.lang_code
            evidence["attribution"] = attribution.as_dict()
            explanation += f"; verici etiketi: {attribution.describe()}"
        return Signal("verici_yakınlığı", True, strength, explanation, evidence)

    def detect(
        self,
        word: str,
        entries: list[dict[str, Any]] | None = None,
        *,
        lang: str = "tr",
        sense: str = "",
        donors: list[str] | None = None,
    ) -> BorrowingVerdict:
        """Bir kelimenin alıntı olup olmadığına gerekçeli karar verir.

        :param sense: kelimenin anlamı. Verici yakınlığı sinyali bunsuz
            çalışmaz (anlam kısıtı yayınlanmış kurulumun parçasıdır). Boşsa
            sözlük indeksindeki kendi kaydından okunur (:func:`own_sense`).
        :param donors: bakılacak verici dil kodları. ``None`` ise dilin
            ölçüm hattındaki kümesi (:func:`default_donors`).
        """
        if not sense:
            sense = own_sense(word, lang)
        if donors is None:
            donors = default_donors(lang)
        witnesses = {
            e["lang_code"]: e.get("word", "")
            for e in (entries or [])
            if e.get("lang_code") and e.get("word")
        }
        witnesses.setdefault(lang, word)

        chain_signal, chain, donor = self._chain_signal(word, lang)
        phonotactic = self._phonotactic_signal(word, lang)
        sound_law, expected = self._sound_law_signal(word, witnesses, lang)
        uniformity = self._uniformity_signal(witnesses)
        donor_proximity = self._donor_signal(word, sense, donors)
        phonotactic_model = self._phonotactic_model_signal(word, lang)

        signals = [
            chain_signal,
            phonotactic,
            sound_law,
            uniformity,
            donor_proximity,
            phonotactic_model,
        ]
        score = sum(
            SIGNAL_WEIGHTS[signal.name] * signal.strength for signal in signals if signal.fired
        )
        verdict = BorrowingVerdict(
            word=word,
            score=round(score, 3),
            signals=signals,
            expected_if_inherited=expected,
            chain=chain,
            donor_language=donor,
            lang=lang,
        )

        combiner = self.combiner
        if combiner is not None and combiner.is_trained:
            strengths = {s.name: (s.strength if s.fired else 0.0) for s in signals}
            verdict.trained_probability = combiner.probability(strengths)
            verdict._trained_threshold = combiner.threshold
            note = (
                f"eğitilmiş birleştirici ({combiner.trained_on}, n={combiner.n}, "
                f"hedef={combiner.objective}): {combiner.explain()}"
            )
            # ⚠️ Model Sakha'da eğitildi. Başka bir dile uygulamak ALAN DIŞI
            # kullanımdır: sinyal dağılımı dilden dile değişir ve ölçülmüş
            # F 0,598 o dilde geçerli değildir.
            if lang not in combiner.trained_on:
                note += f" ⚠️ ALAN DIŞI: model {combiner.trained_on} verisinde eğitildi, sorgu dili {lang}"
            verdict.combiner_note = note
        else:
            verdict.combiner_note = (
                "⚠️ EĞİTİLMEMİŞ — el ağırlıkları kullanılıyor. Ölçüldü: el "
                "ağırlıklı toplam en güçlü sinyalin kararını bozuyor."
            )
        return verdict


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Açıklamalı alıntı tespiti")
    ap.add_argument("words", nargs="*", default=[])
    ap.add_argument("--controls", action="store_true", help="negatif kontrol bataryasını koş")
    args = ap.parse_args()

    detector = BorrowingDetector()

    if args.controls:
        from engine.evaluation.negative_controls import ALL_BATTERIES

        for name, items in ALL_BATTERIES.items():
            print(f"\n--- {name}")
            for item in items:
                entries = [{"lang_code": c, "word": w} for c, w in item.witnesses]
                verdict = detector.detect(item.query, entries)
                if name == "alinti_tuzagi":
                    expected = verdict.is_borrowed
                elif name == "eşadlı":
                    # Eşadlıda doğru cevap KESİN KARAR DEĞİL, belirsizliktir.
                    expected = not verdict.blocks_inherited_reconstruction
                else:
                    expected = not verdict.is_borrowed
                mark = "OK " if expected else "!! "
                print(f"  {mark}{verdict.word:10} {verdict.verdict:12} {verdict.score:.2f}")
        return 0

    for word in args.words or ["kitap", "göz", "çorap", "deniz", "sabun", "yol"]:
        print(detector.detect(word).explain())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
