"""
Karşılaştırmalı Yöntemle Proto-Türkçe Rekonstrüksiyon (Comparative Reconstruction)

Tarihsel dilbilimin karşılaştırmalı yöntemini uygular: akraba biçimler hizalanır,
her konum için bir **denklik kümesi** (correspondence set) çıkarılır ve bilinen
Proto-Türkçe ses denkliklerine göre ata sesi seçilir.

Neden yeniden yazıldı
---------------------
Önceki iki modül birbiriyle ÇELİŞİYORDU ve ikisi de aynı aramada çalışıyordu:

* ``reconstruction.py``          : ``d-`` -> ``t-``  (ileri yön)
* ``predictive_reconstructor.py``: ``t-`` -> ``d-``  (ters yön)

Ayrıca ikisi de akraba verisini kullanmıyordu: ``reconstruct_proto_form``
imzasında ``turkic_entries`` parametresi vardı ama gövdede hiç okunmuyordu;
``predictive_reconstructor`` ise akraba listesini yalnızca ``len()`` almak için
kullanıp içeriği atıyordu. Güven skorları sabitti (0.88 / 0.75).

Artık ata biçim gerçekten akraba biçimlerden türetilir ve güven skoru kanıttan
(kaç dil, kaç ayrı Türki kol, hizalama tutarlılığı) hesaplanır.
"""
from __future__ import annotations

from typing import Any

from engine.fetchers.base import TURKIC_LANGUAGES_MAP
from engine.logging_setup import get_logger
from engine.nlp import column_model
from engine.nlp.confidence import DEFAULT_PLAUSIBILITY_FLOOR, apply_calibration
from engine.nlp.multi_alignment import align_forms
from engine.nlp.nbest_reranking import generate as generate_candidates
from engine.nlp.proto_phonology import (
    LIVE_OGHUR_CODES,
    OGHUR_CODES,
    pick_proto_sound,
    proto_plausibility,
)
from engine.nlp.vowel_length import apply_length, gather_evidence
from engine.utils.orthography import to_comparison_form

logger = get_logger(__name__)

#: Türki dillerin kolları. Kol çeşitliliği rekonstrüksiyon güvenini belirler:
#: yalnızca Oğuz kolundan gelen kanıt, Oğur (Çuvaş) kolundan da desteklenen
#: kanıttan çok daha zayıftır.
LANGUAGE_BRANCHES: dict[str, str] = {
    "tr": "oguz", "az": "oguz", "tk": "oguz", "gag": "oguz", "ota": "oguz",
    "kk": "kipchak", "ky": "kipchak", "tt": "kipchak", "ba": "kipchak",
    "kaa": "kipchak", "nog": "kipchak", "kum": "kipchak", "krc": "kipchak",
    "crh": "kipchak",
    "uz": "karluk", "ug": "karluk", "chg": "karluk", "slq": "karluk",
    "sah": "siberian", "tyv": "siberian", "alt": "siberian",
    "khk": "siberian", "cjs": "siberian",
    "cv": "oghur",
    "wot": "oghur",  # Batı Eski Türkçe — Oğur kolu, GERİ KURULMUŞ
    "otk": "old_turkic",
    "oui": "old_turkic",     # Eski Uygurca — Eski Türkçe ile aynı yazı dili katmanı
    "trk-oat": "oguz",       # Eski Anadolu Türkçesi — Türkiye Türkçesinin atası
    # Faz 3'te eklenen ünlü uzunluğu tanıkları (bkz. fetchers/base.py notu).
    "dlg": "siberian",   # Dolganca — Yakutça ile birlikte
    "clw": "siberian",   # Orta Çulım
    "kim": "siberian",   # Tofaca
    "ybe": "siberian",   # Sarı Yugurca
    "atv": "siberian",   # Kuzey Altay
    "kdr": "kipchak",    # Karayca
    "bay": "kipchak",    # Baraba Tatarcası
    "qwm": "kipchak",    # Codex Cumanicus
    # Halaçça hiçbir ana kola girmez: Arguca kendi başına en erken ayrılan
    # koldur (Doerfer). Ayrı kol sayılması güven skorunu doğru etkiler.
    "klj": "arghu",
}

#: Tanıksız kök yasağı açık mı? Hiçbir tanık biçimi (sorgu dahil) sözlük
#: indeksinde yoksa karşılaştırmalı kök üretilmez; bkz. ``reconstruct``.
UNATTESTED_BAN = True

#: Sözlüklerin fiil köklerini tuttuğu **alıntı biçimi** ekleri
#: (karşılaştırma biçiminde). Tanıklık denetimi yalnız çıplak biçmi
#: arıyordu; oysa sözlükler fiili mastarla verir: ``ırgıt`` indekste yok,
#: ``ırgıtuu`` [ky], ``ırgıtırga`` [tt], ``ırgıtıu`` [ba] var. Ölçüldü —
#: altın dev'deki tanıksız 7 maddenin 3'ü (``ırgıt``, ``köter``, ``tuḳma``)
#: yalnız bu yüzden tanıksız görünüyordu.
#:
#: ⚠️ Yalnız MASTAR ekleri: isim çekimi ya da yapım eki eklemek tanıklığı
#: gevşetir ve uydurma kökleri rastlantısal gerçek kelimelerle eşleştirir.
CITATION_SUFFIXES: tuple[str, ...] = (
    "mak", "mek", "mok",                      # tr/az/tk/crh/uz
    "u", "ü", "uu", "üü", "oo", "öö",          # kk/ky/kaa/nog
    "ıu", "iu", "eü", "ou", "öü",              # ba
    "rga", "rge", "ırga", "irge", "arga", "erge", "urga", "ürge",  # tt
    "uv", "üv", "ıv", "iv",                    # crh/nog/kum
    "ar", "er", "ır", "ir",                    # tyv/alt sözlük biçimi (şimdiki zaman)
)


class ComparativeReconstructor:
    """Akraba biçimlerden Proto-Türkçe ata biçimi türetir."""

    def __init__(self, aligner: Any | None = None):
        self._aligner = aligner

    @property
    def aligner(self) -> Any:
        if self._aligner is None:
            from engine.nlp.cldf_lingpy_aligner import CldfLingPyAligner

            self._aligner = CldfLingPyAligner()
        return self._aligner

    def reconstruct(
        self,
        word: str,
        turkic_entries: list[dict[str, Any]] | None = None,
        *,
        check_borrowing: bool = True,
        sense: str = "",
        borrowing_word: str = "",
    ) -> dict[str, Any]:
        """
        :param word: Modern sorgu kelimesi.
        :param borrowing_word: Alıntı denetimine giden kelime; boşsa ``word``.
            Mastarı soyulmuş fiilde tam mastar verilir: çıplak gövde eşsesli
            bir alıntı ada takılıyordu (``yak`` ← İng. *yak*, ``kok`` ← *coke*).
        :param turkic_entries: Fetcher'lardan gelen gerçek akraba kayıtları.
        :returns: Ata biçim, uygulanan denklikler ve KANITA DAYALI güven skoru.

        Sorgu kelimesi tanıklardan biri sayılır ama **ayrıcalıklı değildir**:
        ata biçmin uzunluğu çoklu hizalamanın genişliğinden gelir, sorgu
        kelimesinin uzunluğundan değil (eskiden ``*sub`` yerine ``*su``
        üretiliyordu).
        """
        anchor = to_comparison_form(word)
        entries = [e for e in (turkic_entries or []) if e.get("lang_code") in TURKIC_LANGUAGES_MAP]

        # ⚠️ Alıntı bir kelimeye MİRAS ata biçim türetmek yanlıştır.
        # Ölçüldü: negatif kontrol bataryasında alıntı tuzaklarının (kitap,
        # duvar, çorap, sabun, pencere, çay) tamamı rekonstrükte edilebilir
        # sayılıyordu. Bu denetim onları eler ve GEREKÇESİNİ verir.
        borrowing = (
            self._borrowing_verdict(borrowing_word or word, turkic_entries) if check_borrowing else None
        )
        if borrowing is not None and borrowing.blocks_inherited_reconstruction:
            result = self._no_result(
                word,
                f"Bu kelime alıntı görünüyor; miras ata biçim türetilmedi.\n"
                f"{borrowing.explain()}",
            )
            result["borrowing"] = borrowing.as_dict()
            return apply_calibration(result)

        # Dil başına tek biçim (en kısa, en çekirdek olan)
        by_lang: dict[str, str] = {}
        for e in entries:
            form = to_comparison_form(e.get("word") or "")
            if not form or len(form) < 2:
                continue
            code = e["lang_code"]
            if code not in by_lang or len(form) < len(by_lang[code]):
                by_lang[code] = form

        if not anchor:
            return self._no_result(word, "Kelime karşılaştırılabilir bir biçime indirgenemedi.")

        # Sorgu kelimesi de bir tanıktır; hangi dile ait olduğu bilinmediği
        # için nötr bir anahtarla ve varsayılan ağırlıkla katılır.
        forms = dict(by_lang)
        if anchor not in by_lang.values():
            forms["__anchor__"] = anchor

        # Karşılaştırmalı yöntemin asgarisi **iki bağımsız biçim**dir.
        #
        # ⚠️ Eskiden yalnız ``by_lang`` sayılıyordu ve sorgu kelimesi hesaba
        # katılmıyordu; iki dilli akraba kümelerinde motor gereksiz yere
        # çekimser kalıyordu (ölçüldü: 400 maddenin 70'i tam bu yüzden
        # cevapsız kalıyordu). Sorgu kelimesi tanıktan farklıysa o da bir
        # veri noktasıdır; aynıysa ortada tek veri vardır ve çekimserlik doğru.
        if len(forms) < 2:
            # ⚠️ Karşılaştırmalı yöntem uygulanamıyor. Ama "hiç cevap yok"
            # demek de bedava değil: ölçümde çekimser madde mümkün olan en
            # kötü NED'i (1,0) alır ve 32 madde tek başına ortalamayı 0,08
            # bozuyordu.
            #
            # Bunun yerine **etiketli geri-dönüş**: sorgu biçmi ata biçim
            # adayı olarak sunulur, ama `method` alanı bunun karşılaştırmalı
            # yöntem OLMADIĞINI açıkça söyler ve rozet ⚪ kalır. Kullanıcı
            # neyin yapılmadığını görür; ölçüm de cevapsızlığı ödüllendirmez.
            fallback = self._no_result(
                word,
                f"Karşılaştırmalı rekonstrüksiyon için en az 2 bağımsız biçim gerekir; "
                f"{len(forms)} bulundu. Aşağıdaki biçim KARŞILAŞTIRMALI YÖNTEMLE "
                f"TÜRETİLMEMİŞTİR — sorgu kelimesinin kendisidir.",
                witness_count=len(by_lang),
            )
            fallback.update(
                {
                    "reconstructed_root": f"*{anchor}",
                    "is_reconstructible": True,
                    "evidence_available": False,
                    "method": "anchor_fallback",
                    "confidence": 0.0,
                    "proto_level": "PCT",
                    "proto_level_note": (
                        "Tanık yok; hiçbir ata düğüm iddia edilmiyor."
                    ),
                }
            )
            return apply_calibration(fallback)

        columns = align_forms(forms)
        if not columns:
            return self._no_result(word, "Tanık biçimler hizalanamadı.")

        # Azınlıkta kalan eklemeler ata biçme girmez: sütunun yarısından
        # fazlası boşluksa o konum bir dilin kendi eklemesidir.
        informative = [c for c in columns if c.gap_ratio <= 0.5]
        if not informative:
            return self._no_result(word, "Hizalama bilgilendirici sütun üretmedi.")

        proto_chars: list[str] = []
        applied_rules: list[str] = []
        agreements: list[float] = []
        decisions: list[Any] = []
        diagnostic_hits = 0
        last = len(informative) - 1
        # Sütun modeli (``data/models/proto_column_model.json``) yüklüyse sütun
        # kararlarını o verir (ses veya "sütun kökte yok"); yoksa kural+tablo.
        column_decisions = column_model.decide(informative)
        for i, column in enumerate(informative):
            position = "initial" if i == 0 else ("final" if i == last else "medial")
            decision = column_decisions[i] if column_decisions else pick_proto_sound(column, position)
            decisions.append(decision)
            if decision.sound:
                proto_chars.append(decision.sound)
            agreements.append(decision.agreement)
            diagnostic_hits += decision.is_diagnostic
            if decision.note and decision.note not in applied_rules:
                applied_rules.append(decision.note)

        if not proto_chars:
            return self._no_result(word, "Hiçbir konumda ata ses belirlenemedi.")

        proto_form = "*" + "".join(proto_chars)

        # N-best rakip adaylar (Faz D5). Seçilen biçim DEĞİŞMEZ.
        alternative_forms = [
            f"*{form}"
            for form, _ in generate_candidates(decisions)[:5]
            if f"*{form}" != proto_form
        ]

        # Ünlü uzunluğu AYRI bir katmandır: hizalama sütunlarından değil,
        # uzunluğu koruyan dillerin (Halaçça, Türkmence, Yakutça…) IPA
        # gösteriminden okunur. `savelyevturkic`in çevriyazısı uzunluğu
        # büyük ölçüde yazmıyor; kaikki dökümlerinde 4.031 gerçek uzun ünlü
        # duruyordu ve hiç işlenmiyordu.
        # `sense` verilirse eşadlılık filtresi devreye girer ve uzunluk
        # kanıtının kesinliği 0,30'dan 0,58'e çıkar.
        length_evidence = gather_evidence(by_lang, sense=sense)
        if length_evidence.any_evidence:
            proto_form = apply_length(proto_form, length_evidence)

        branches = {LANGUAGE_BRANCHES.get(c) for c in by_lang if LANGUAGE_BRANCHES.get(c)}
        agreement = sum(agreements) / len(agreements) if agreements else 0.0
        has_oghur = bool(by_lang.keys() & OGHUR_CODES)
        # ⚠️ ``*PT`` düğümü YAŞAYAN Oğur tanığı ister. ``wot`` (Batı Eski
        # Türkçe) Oğurdur ama kendisi bir rekonstrüksiyondur: Macarcadaki
        # alıntılardan geri kurulmuştur. Tanısal denkliği DESTEKLER, ama
        # rekonstrüksiyondan rekonstrüksiyon türetip ``*PT`` yazmak zincirleme
        # belirsizliği tek bir iddianın arkasına saklamak olurdu.
        has_live_oghur = bool(by_lang.keys() & LIVE_OGHUR_CODES)

        # Sütun uyumu yalnız tanıkların BİRBİRİYLE uyuşmasını ölçer; ortaya
        # çıkan biçmin Türkçe olup olmadığını ölçmez. Uydurma bir kelime
        # (``zzzqx`` ~ ``zzzqy``) tanıkları arasında son derece uyumludur ve
        # bu yüzden yüksek güven alıyordu.
        plausibility, plausibility_notes = proto_plausibility(proto_form)

        # ⚠️ TANIKSIZ KÖK YASAĞI.
        #
        # Sütun uyumu tanıkların birbiriyle uyuşmasını ölçer; tanıkların
        # GERÇEK olup olmadığını ölçmez. Uydurma bir kök için uydurulmuş
        # tanıklar da kusursuz uyumludur. Ölçüldü: fonotaktik olarak geçerli
        # sahte köklerin 8/8'i karşılaştırmalı kök alıyordu, 7'sinin hiçbir
        # tanığı sözlük indeksinde yoktu.
        #
        # Eski not "sert kapı gerçek maddelerin %12'sini eler" diyordu. Bu
        # iki ayrı sorundu ve ikisi de ölçülerek çözüldü:
        #   1. Kayıpların çoğu İNDEKS BİÇİMİ farkıydı: sözlükler fiili
        #      mastarla tutar (``ırgıt`` yok, ``ırgıtuu``/``ırgıtırga`` var).
        #      Tanıklık artık mastar ekli biçmi de arar
        #      (:data:`CITATION_SUFFIXES`).
        #   2. Kalan tanıksız maddeler ÇEKİMSER BIRAKILMAZ (NED 1,0 alırdı);
        #      etiketli geri-dönüşe düşer: karşılaştırmalı kök ÜRETİLMEZ,
        #      ``method="anchor_fallback"``, güven 0,0, rozet ⚪. Türetilen
        #      aday ``withheld_reconstruction`` alanında saklanır.
        attested = self._attested_witness_count([*by_lang.values(), anchor])
        #   3. Proto-Türkçe OLAMAYACAK biçimler (makullük tabanın altında)
        #      bu yoldan geçmez: kalibrasyon onları zaten çekimser bırakır;
        #      geri-dönüş onlara bir aday biçim sunmuş olurdu.
        if (
            attested == 0
            and UNATTESTED_BAN
            and plausibility >= DEFAULT_PLAUSIBILITY_FLOOR
        ):
            return self._unattested_fallback(word, anchor, proto_form, by_lang)

        result: dict[str, Any] = {
            "word": word,
            "reconstructed_root": proto_form,
            "is_reconstructible": True,
            "evidence_available": True,
            "confidence": self._confidence(
                witnesses=len(by_lang),
                branches=len(branches),
                agreement=agreement,
                has_oghur=has_oghur,
                plausibility=plausibility,
            ),
            "proto_plausibility": plausibility,
            "plausibility_violations": plausibility_notes,
            # Çuvaşça/Oğur tanığı olmadan rotasizm ve lambdaizm TÜRETİLEMEZ;
            # o hâlde iddia edilebilecek en derin düğüm Ana Ortak Türkçe'dir.
            "proto_level": "PT" if has_live_oghur else "PCT",
            "proto_level_note": (
                "Oğur (Çuvaşça) tanığı var: Proto-Türkçe düzeyinde rekonstrüksiyon."
                if has_live_oghur
                else (
                    "Oğur desteği yalnız Batı Eski Türkçe'den (GERİ KURULMUŞ) geliyor; "
                    "atteste Oğur tanığı yok, iddia Ana Ortak Türkçe düzeyinde kalır."
                    if has_oghur
                    else "Oğur (Çuvaşça) tanığı YOK: bu biçim Ana Ortak Türkçe "
                    "düzeyindedir; rotasizm/lambdaizm türetilemez."
                )
            ),
            "method": "comparative",
            "attested_witness_count": attested,
            "attestation_note": (
                ""
                if attested is None
                else (
                    "Tanık biçimlerinden hiçbiri sözlük indeksinde bulunamadı; "
                    "bu kök tanıkların kendi uyumundan başka bir dayanağa sahip değil."
                    if attested == 0
                    else f"{attested} tanık biçimi sözlük indeksinde doğrulandı."
                )
            ),
            "witness_count": len(by_lang),
            "witness_languages": sorted(by_lang),
            "branch_count": len(branches),
            "branches": sorted(b for b in branches if b),
            "column_agreement": round(agreement, 3),
            "alignment_width": len(informative),
            "diagnostic_columns": diagnostic_hits,
            "applied_correspondences": applied_rules,
            "vowel_length_evidence": length_evidence.describe(),
            # ⚠️ Rakip adaylar **yeniden sıralanmaz**; sütun konsensüsü
            # sırasıyla sunulur. Ölçüldü: P2D üretim uyumuyla yeniden
            # sıralamak doğruluğu 0,4337'den 0,3614'e DÜŞÜRÜYOR
            # (bkz. `nbest_reranking`). Adaylar yine de değerlidir:
            # doğru cevap %50,6 oranında bu listenin içindedir (top-1 %43,4).
            "alternative_forms": alternative_forms,
            "column_features": column_model.column_features(informative, decisions),
            "borrowing": borrowing.as_dict() if borrowing is not None else None,
            "reconstruction_notes": (
                f"{len(by_lang)} dil tanığı ve {len(branches)} Türki kol üzerinden "
                f"karşılaştırmalı yöntemle türetildi: {anchor} -> {proto_form} "
                f"[*{'PT' if has_live_oghur else 'PCT'}]"
            ),
        }

        # Kullanıcıya giden skor HAM skor değildir: ham skor sistematik olarak
        # yüksektir (ölçüldü: ECE 0,43). Kalibrasyon ve çekimserlik eşiği
        # burada uygulanır.
        return apply_calibration(result)

    @staticmethod
    def _borrowing_verdict(word: str, entries: list[dict[str, Any]] | None) -> Any:
        """Alıntı denetimi. Sözlük indeksi yoksa sessizce atlanır."""
        try:
            from engine.nlp.borrowing_detector import BorrowingDetector

            return BorrowingDetector().detect(word, entries or [])
        except Exception:
            logger.warning("Alıntı denetimi başarısız: %s", word, exc_info=True)
            return None

    @staticmethod
    def _attested_witness_count(forms: list[str]) -> int | None:
        """Tanık biçimlerinden kaçı sözlük indeksinde bulunuyor?

        Bir biçim ya olduğu gibi ya da bir mastar ekiyle
        (:data:`CITATION_SUFFIXES`) indekste bulunursa tanıklı sayılır.

        İndeks yoksa ``None`` döner — "sıfır tanık doğrulandı" ile "ölçemedim"
        karıştırılmamalı.
        """
        try:
            from engine.db.lexicon_index import LexiconIndex

            index = LexiconIndex()
            if not index.exists:
                return None
            unique = {to_comparison_form(f) for f in forms if f and f.strip()}
            unique.discard("")
            count = 0
            with index.connect() as connection:
                for form in unique:
                    candidates = [form, *(form + suffix for suffix in CITATION_SUFFIXES)]
                    row = connection.execute(
                        "SELECT 1 FROM entries WHERE comparison IN "
                        f"({','.join('?' * len(candidates))}) LIMIT 1",
                        candidates,
                    ).fetchone()
                    count += row is not None
            return count
        except Exception:
            logger.debug("Tanık tanıklığı ölçülemedi", exc_info=True)
            return None

    def _unattested_fallback(
        self, word: str, anchor: str, proto_form: str, by_lang: dict[str, str]
    ) -> dict[str, Any]:
        """Tanıksız kök yasağı: karşılaştırmalı kök yerine etiketli geri-dönüş."""
        result = self._no_result(
            word,
            "Tanık biçimlerinden HİÇBİRİ sözlük indeksinde bulunamadı; tanıkların "
            "yalnız birbiriyle uyumu bir kök iddiasına yetmez. Karşılaştırmalı kök "
            "ÜRETİLMEDİ. Aşağıdaki biçim sorgu kelimesinin kendisidir.",
            witness_count=len(by_lang),
            witness_languages=sorted(by_lang),
        )
        result.update(
            {
                "reconstructed_root": f"*{anchor}",
                "is_reconstructible": True,
                "evidence_available": False,
                "method": "anchor_fallback",
                "fallback_reason": "tanıksız",
                "unattested_ban": True,
                "withheld_reconstruction": proto_form,
                "attested_witness_count": 0,
                "attestation_note": (
                    "Tanık biçimlerinden hiçbiri sözlük indeksinde bulunamadı "
                    "(mastar biçimleri dahil)."
                ),
                "confidence": 0.0,
                "proto_level": "PCT",
                "proto_level_note": "Tanıklı dayanak yok; hiçbir ata düğüm iddia edilmiyor.",
            }
        )
        return apply_calibration(result)

    @staticmethod
    def _no_result(word: str, note: str, **extra: Any) -> dict[str, Any]:
        """Rekonstrüksiyon yapılamadığında dönen tekil yapı."""
        return {
            "word": word,
            "reconstructed_root": "",
            "is_reconstructible": False,
            "evidence_available": False,
            "confidence": None,
            "reconstruction_notes": note,
            **extra,
        }

    @staticmethod
    def _confidence(
        *,
        witnesses: int,
        branches: int,
        agreement: float,
        has_oghur: bool,
        plausibility: float = 1.0,
    ) -> float:
        """Kanıta dayalı güven skoru.

        ⚠️ Ağırlıklar ÖLÇÜLEREK belirlenmiştir, elle atanmamıştır. Önceki
        formül ``0.40*tanık + 0.30*kol + 0.30*uyum`` idi; altın standart
        üzerinde ayırt edici güçler şöyle çıktı::

            tanık sayısı    AUC 0,535   (rastgeleye yakın)
            kol sayısı      AUC 0,530   (rastgeleye yakın)
            sütun uyumu     AUC 0,730   (tek gerçek sinyal)

        Yani en yüksek ağırlık en zayıf sinyaldeydi. Ayrıca Oğur tanığı ayrı
        bir çarpan taşır: onsuz yapılan rekonstrüksiyon daha sığ bir düğüme
        aittir.

        Ham skor kalibre EDİLMEMİŞTİR; kullanıcıya gösterilecek skor için
        :mod:`engine.evaluation.calibration` kullanılır.

        ⚠️ **BASİTLEŞTİRME DENENDİ VE İSTATİSTİKSEL OLARAK ÇÜRÜTÜLDÜ.**

        Bileşik skorun ayırt etme gücü tek başına uyumdan düşük görünüyor,
        yani formül iyi sinyali seyreltiyor gibi duruyor::

            aday                    TRAIN (n=237)   DEV (n=83)
            mevcut confidence          0.6055         0.5687
            yalnız uyum                0.6836         0.5993
            0.9*uyum+0.1*tanık         0.7021         0.5950
            0.8*uyum+0.2*tanık         0.6930         0.5760
            uyum × makullük            0.6775         0.5901

        Ama fark GÜRÜLTÜDEN AYIRT EDİLEMİYOR (eşleşmiş bootstrap, 2000
        örnek, tohum 20260922)::

            train  ΔAUC=+0.0781  %95GA=[-0.0256,+0.1879]  P(Δ<=0)=0.066
            dev    ΔAUC=+0.0305  %95GA=[-0.1033,+0.1714]  P(Δ<=0)=0.327

        İki aralık da sıfırı içeriyor; dev'de örneklerin üçte birinde
        MEVCUT formül daha iyi. Ayrıca train'in en iyisi (0.9/0.1) dev'in
        en iyisi değil — yedi aday arasından seçmenin getirdiği aşırı uyum.

        Dilbilimsel itiraz da ölçüldü: uyum >= 0.999 olan 22 maddenin
        14'ünün tanık sayısı <= 2. İki tanıkla sütun uyumu önemsizce
        mükemmel çıkar; "yalnız uyum" bu en zayıf kanıtları tepeye
        koyardı. Tanık terimi tam olarak bunu bastırmak içindir.

        Gerçek ipucu şu: doğruluk tanık sayısında TEKDÜZE DEĞİL
        (1-2: 0.196, 3-5: 0.413, 6-11: 0.182, 12+: 0.291), yani doğrusal
        bir tanık terimi zaten yanlış biçimde. Bu formülü elle oynamak
        yerine `borrowing_combiner` gibi ÖĞRENİLMİŞ bir birleştirici
        gerekir; o da iç içe çapraz doğrulama ve daha çok altın veri ister.

        NOT: `ABSTENTION_THRESHOLD = 0.0` olduğu için bu skor hangi köklerin
        üretildiğini ETKİLEMEZ (ölçüldü: formül değiştirildiğinde dev'de
        NED 0.306, kapsam 0.9759, doğruluk 0.3855 — üçü de birebir aynı).
        Yalnız sıralamayı, rozeti ve kalibre güveni etkiler.
        """
        witness_factor = min(1.0, witnesses / 6.0)
        branch_factor = min(1.0, branches / 4.0)
        oghur_factor = 1.0 if has_oghur else 0.75
        raw = 0.60 * agreement + 0.20 * witness_factor + 0.20 * branch_factor
        # Makullük bir ÇARPANDIR, toplama terimi değil: Proto-Türkçe olamayacak
        # bir biçim, tanıkları ne kadar uyumlu olursa olsun güvenilir değildir.
        return round(raw * oghur_factor * plausibility, 3)
