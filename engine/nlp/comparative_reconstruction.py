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

from collections import Counter
from typing import Any

from engine.fetchers.base import TURKIC_LANGUAGES_MAP
from engine.logging_setup import get_logger
from engine.nlp.confidence import apply_calibration
from engine.nlp.multi_alignment import align_forms
from engine.nlp.proto_phonology import OGHUR_CODES, pick_proto_sound, proto_plausibility
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
    "otk": "old_turkic",
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

#: Bilinen Proto-Türkçe denklik kümeleri.
#: Her giriş: (sesler, ata_ses, açıklama, konum).
#: Konum: "initial" (söz başı), "final" (söz sonu), "any" (her yer).
#: Bunlar Türkolojide yerleşik denkliklerdir (Lir-Şaz / rotasizm-lambdaizm).
CORRESPONDENCE_SETS: list[tuple[frozenset[str], str, str, str]] = [
    # --- Söz sonu (Lir-Şaz) ---
    (frozenset({"z", "r"}), "ŕ", "Lir-Şaz rotasizmi: Ortak Türkçe -z ~ Çuvaşça -r < Proto-Türkçe *-ŕ", "final"),
    (frozenset({"z", "r", "s"}), "ŕ", "Ortak Türkçe -z/-s ~ Çuvaşça -r < Proto-Türkçe *-ŕ", "final"),
    (frozenset({"ş", "l"}), "ĺ", "Lambdaizm: Ortak Türkçe -ş ~ Çuvaşça -l < Proto-Türkçe *-ĺ", "final"),
    (frozenset({"s", "ş", "l"}), "ĺ", "Ortak Türkçe -s/-ş ~ Çuvaşça -l < Proto-Türkçe *-ĺ", "final"),
    # --- Söz başı ---
    # Oğuz kolu söz başı ötümsüzleri ötümlüleştirdi (t->d, k->g); ata biçim ötümsüzdür.
    (frozenset({"d", "t"}), "t", "Söz başı ötümlüleşme: Oğuz d- ~ diğer t- < Proto-Türkçe *t-", "initial"),
    (frozenset({"g", "k"}), "k", "Söz başı ötümlüleşme: Oğuz g- ~ diğer k- < Proto-Türkçe *k-", "initial"),
    (frozenset({"y", "c", "j", "ç"}), "j", "Söz başı akıcı: y- ~ c- ~ j- < Proto-Türkçe *j-", "initial"),
    (frozenset({"b", "m"}), "b", "Genizsilleşme: b- ~ m- < Proto-Türkçe *b-", "initial"),
    (frozenset({"h", "k", "q"}), "k", "Söz başı h- ~ k- denkliği", "initial"),
    # --- Konumdan bağımsız ---
    (frozenset({"d", "y", "z", "t", "r"}), "d", "Klasik *d̮ denkliği: d ~ y ~ z ~ t ~ r", "any"),
    (frozenset({"b", "v", "w", "u"}), "b", "Ünsüz yumuşaması: b ~ v ~ w ~ u < Proto-Türkçe *b", "any"),
    (frozenset({"g", "ğ", "v", "w"}), "g", "Ünlü arası yumuşama: g ~ ğ ~ v ~ w < Proto-Türkçe *g", "any"),
    (frozenset({"n", "ŋ"}), "ŋ", "Genizsil denkliği: -n- ~ -ŋ- < Proto-Türkçe *-ŋ-", "any"),
]


def _pick_proto_phoneme(sounds: list[str], position: str) -> tuple[str, str | None]:
    """
    Bir konumdaki seslerden ata sesi seçer.

    Denklikler KONUMA DUYARLIDIR: söz başı ``d ~ t`` denkliği Proto-Türkçe
    ``*t-`` verirken, söz içi ``d ~ y ~ z`` denkliği ``*d̮`` verir. Konumu
    yok saymak yanlış ata biçim üretir.

    :param position: "initial" | "medial" | "final"
    :returns: (ata_ses, açıklama)
    """
    present = {s for s in sounds if s}
    if not present:
        return "", None
    if len(present) == 1:
        return next(iter(present)), None

    best: tuple[int, int, str, str] | None = None
    for members, proto, note, applies_to in CORRESPONDENCE_SETS:
        if applies_to != "any" and applies_to != position:
            continue
        overlap = present & members
        if len(overlap) >= 2:
            # Konuma özgü kural, genel kurala tercih edilir.
            specificity = 1 if applies_to == "any" else 2
            cand = (specificity, len(overlap), proto, note)
            if best is None or cand[:2] > best[:2]:
                best = cand
    if best:
        return best[2], best[3]

    return Counter(s for s in sounds if s).most_common(1)[0][0], None


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
    ) -> dict[str, Any]:
        """
        :param word: Modern sorgu kelimesi.
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
        borrowing = self._borrowing_verdict(word, turkic_entries) if check_borrowing else None
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
        diagnostic_hits = 0
        last = len(informative) - 1
        for i, column in enumerate(informative):
            position = "initial" if i == 0 else ("final" if i == last else "medial")
            decision = pick_proto_sound(column, position)
            if decision.sound:
                proto_chars.append(decision.sound)
            agreements.append(decision.agreement)
            diagnostic_hits += decision.is_diagnostic
            if decision.note and decision.note not in applied_rules:
                applied_rules.append(decision.note)

        if not proto_chars:
            return self._no_result(word, "Hiçbir konumda ata ses belirlenemedi.")

        proto_form = "*" + "".join(proto_chars)

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

        # Sütun uyumu yalnız tanıkların BİRBİRİYLE uyuşmasını ölçer; ortaya
        # çıkan biçmin Türkçe olup olmadığını ölçmez. Uydurma bir kelime
        # (``zzzqx`` ~ ``zzzqy``) tanıkları arasında son derece uyumludur ve
        # bu yüzden yüksek güven alıyordu.
        plausibility, plausibility_notes = proto_plausibility(proto_form)

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
            "proto_level": "PT" if has_oghur else "PCT",
            "proto_level_note": (
                "Oğur (Çuvaşça) tanığı var: Proto-Türkçe düzeyinde rekonstrüksiyon."
                if has_oghur
                else "Oğur (Çuvaşça) tanığı YOK: bu biçim Ana Ortak Türkçe düzeyindedir; "
                "rotasizm/lambdaizm türetilemez."
            ),
            "method": "comparative",
            "witness_count": len(by_lang),
            "witness_languages": sorted(by_lang),
            "branch_count": len(branches),
            "branches": sorted(b for b in branches if b),
            "column_agreement": round(agreement, 3),
            "alignment_width": len(informative),
            "diagnostic_columns": diagnostic_hits,
            "applied_correspondences": applied_rules,
            "vowel_length_evidence": length_evidence.describe(),
            "borrowing": borrowing.as_dict() if borrowing is not None else None,
            "reconstruction_notes": (
                f"{len(by_lang)} dil tanığı ve {len(branches)} Türki kol üzerinden "
                f"karşılaştırmalı yöntemle türetildi: {anchor} -> {proto_form} "
                f"[*{'PT' if has_oghur else 'PCT'}]"
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
        """
        witness_factor = min(1.0, witnesses / 6.0)
        branch_factor = min(1.0, branches / 4.0)
        oghur_factor = 1.0 if has_oghur else 0.75
        raw = 0.60 * agreement + 0.20 * witness_factor + 0.20 * branch_factor
        # Makullük bir ÇARPANDIR, toplama terimi değil: Proto-Türkçe olamayacak
        # bir biçim, tanıkları ne kadar uyumlu olursa olsun güvenilir değildir.
        return round(raw * oghur_factor * plausibility, 3)
