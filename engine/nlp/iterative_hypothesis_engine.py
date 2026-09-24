"""
Etimolojik Hipotez Kurucu (Etymological Hypothesis Engine)

Kelime için en iyi desteklenen etimoloji hipotezini kurar ve A-HVP
protokolünden geçirir.

Düzeltilen sorun
----------------
Önceki sürümde dört hipotez dalının dördü de **sabit güven skoru** taşıyordu
(0.99 / 0.98 / 0.85 / 0.90) ve son ``else`` dalı her kelimeye mutlaka bir
"Asli Öz Türkçe Köken Hipotezi" kuruyordu — akraba tanığı olmasa bile.
Bu yüzden ``bilgisayar`` için ``*bilgisayar``, ``zzzqx`` için ``*zzzqx``
gibi var olmayan ata biçimler üretiliyordu.

Sabit skorlar zaten A-HVP çıktısıyla eziliyordu (yani ölü koddu) ama kod
okuyanı ve JSON tüketicisini yanıltıyordu. Artık hipotez yalnızca kanıt
varsa kurulur ve tek güven kaynağı A-HVP'dir.
"""
from __future__ import annotations

from typing import Any

from engine.logging_setup import get_logger
from engine.nlp.comparative_reconstruction import ComparativeReconstructor
from engine.nlp.donor_etymology_database import DeepDonorEtymologyDatabase
from engine.nlp.historical_attestation_verifier import HistoricalAttestationVerifier
from engine.nlp.hypothesis_validation_protocol import HypothesisValidationProtocol
from engine.nlp.neologism_detector import NeologismDetector
from engine.utils.phonotactics import initial_consonant_violation
from engine.utils.reference_resolver import is_cross_reference

logger = get_logger(__name__)


#: Tarihî tanık katmanları. Aynı üçlü `search_engine.py:413` ve
#: `fetchers/historical_index.py:49`'da da kullanılıyor.
HISTORICAL_WITNESS_LANGUAGES: tuple[str, ...] = ("otk", "ota", "chg")


#: Gloss'un SONUNA yapışan dil adları. Anlamın parçası değil, kaynak
#: notunun artığıdır (ölçüldü: `kalem` -> "Yazı kamışı, yontulmuş kamış
#: Eski Grekçe"). Yalnız dizenin sonunda ve boşlukla ayrılmışsa atılır;
#: "Eski Grekçe kamışı" gibi anlam içi kullanımlar korunur.
_TRAILING_LANGUAGE_NAMES: tuple[str, ...] = (
    "Eski Grekçe", "Eski Yunanca", "Orta Farsça", "Klasik Farsça",
    "Osmanlı Türkçesi", "Eski Türkçe", "Ana Türkçe", "Arapça", "Farsça",
    "Yunanca", "Latince", "Ermenice", "Moğolca", "Soğdca", "Çince",
)


def _form_distance(query: str, witness_form: str) -> float:
    """Sorgu ile tanık biçimi arasındaki normalize düzenleme mesafesi (0=aynı).

    Tanık `word` alanı çok biçimli olabiliyor ("göz / kör-",
    "bilgi, (bilgü)"); en yakın alt biçim esas alınır.
    """
    import re

    from engine.nlp.donor_lexicon import levenshtein
    from engine.utils.orthography import to_comparison_form

    q = to_comparison_form(query or "")
    if not q:
        return 1.0
    best = 1.0
    # Parantez de ayırıcıdır: Tarama "derlik (terlik)" yazar, ikinci biçim
    # sorgunun kendisidir ve parantez yüzünden hiç karşılaştırılmıyordu.
    for piece in re.split(r"[/,;()]", witness_form or ""):
        f = to_comparison_form(piece.strip().strip("()*-"))
        if not f:
            continue
        best = min(best, levenshtein(q, f) / max(len(q), len(f), 1))
    return best


def _historical_gloss_candidates(entries: list[dict[str, Any]] | None) -> list[str]:
    """Temizlenmiş TÜM tarihî gloss adayları (sıralamasız).

    3. aşama tek bir tanığa değil, tanıklanmış anlamların EN YAKININA karşı
    ölçülür: "tanıklanmış hiçbir tarihsel anlam modern anlama yakın değil
    mi?" sorusu filolojik olarak meşrudur ve bir kelimenin birden çok
    tarihsel anlamı olabilir.
    """
    return _historical_gloss(entries, return_all=True)  # type: ignore[return-value]


def _historical_gloss(
    entries: list[dict[str, Any]] | None, word: str = "", *, return_all: bool = False
) -> str | list[str]:
    """Tarihî tanıkların ilk GERÇEK anlamını döndürür; yoksa boş dize.

    ⚠️ Bu yardımcı bir kusuru kapatmak için eklendi: miras dalı (aşağıdaki
    3. seçenek) `historical_meaning` alanına MODERN anlamın kopyasını
    yazıyordu. A-HVP 3. aşaması da çiftini buradan aldığı için anlamı
    kendisiyle karşılaştırıyor, mesafe tanımı gereği 0 çıkıyor ve aşama
    BEDAVA ✅ veriyordu.

    Ölçüldü (`--json` çıktısında iki düğüm ayrı ayrı okundu):
        göz    stage3: 'göz, görme organı' ~ 'göz, görme organı'  -> özdeş
        bardak stage3: TDK tanımı          ~ TDK tanımı           -> özdeş
    Oysa aynı koşuda `search_engine` yolu sağlıklı çift üretiyordu:
        göz    'DLT (1074): göz…' ~ 'göz, görme organı'   0.4981
        bardak 'su içilen kap'    ~ TDK tanımı            0.2202
    Yani veri MEVCUTTU, yalnızca bu dala taşınmıyordu.

    Tanık yoksa boş döner; motor o zaman dürüstçe "ölçülemedi" der,
    uydurma kanıt üretmez.
    """
    import re

    #: Atıf öneki: "Divanü Lugati't-Türk (1074): göz…" -> gloss iki nokta
    #: sonrasıdır. Önek atılmazsa kaynak adı ve yıl anlamla birlikte
    #: kodlanıyor ve mesafeyi şişiriyor (ölçüldü: `göz` 0.4981, `deniz` 0.8806).
    citation = re.compile(r"^[^:]{3,60}\(\d{3,4}\)\s*:\s*")

    candidates: list[tuple[float, str]] = []
    for entry in entries or []:
        if entry.get("lang_code") not in HISTORICAL_WITNESS_LANGUAGES:
            continue
        meaning = (entry.get("meaning") or "").strip()
        if not meaning:
            continue

        # ⚠️ ÇÖP GLOSS ELEMESİ — üçü de ölçülmüş gerçek vakalar.
        #
        # 1. Kitap tarama artığı. `archive_org.py:42` bir tam-metin arama
        #    isabetini `lang_code="otk"` diye işaretleyip anlam alanına
        #    KİTAP BAŞLIĞI yazıyor; `local_pdf_books.py:128,144` aynısını
        #    yapıyor. Kelimenin taranmış bir kitapta geçmesi onu Eski Türkçe
        #    tanığı yapmaz. Ölçüldü:
        #      su -> "Kitap: 3 Bogatyr bikers For CNC…"  mesafe 0.9523
        #      el -> "Kitap: The Land Created from Light…" mesafe 0.7098
        # 2. Runik HARF adı. Alfabe maddesidir, sözlükbirim değil; indekste
        #    otk'nin 73/470'i (%15,5) ve ota'da 33 kayıt böyle. Ölçüldü:
        #      baş -> "A letter of the Old Turkic runic script…" mesafe 0.7996
        if meaning.startswith("Kitap:") or "letter of the" in meaning.lower():
            continue
        # 5. Gönderme: "bk. derlik" anlam değil, başka maddeye yönlendirme.
        #    Ölçüldü: `terlik` için biçimce en yakın tanık (mesafe 0.0) bu
        #    göndermeydi ve `derlik (terlik)` "Üstten giyilen ince elbise"
        #    tanımını yeniyordu; başlığa "bk. derlik" basılıyordu.
        if is_cross_reference(meaning):
            continue

        # 3. Atıf öneki yalnız KIRPILIR, kayıt atılmaz: gloss önekten sonra
        #    gerçekten duruyor.
        cleaned = citation.sub("", meaning).strip()

        # 4. İkinci kademe gürültü (ölçüldü):
        #      deniz -> "teŋiz (تِںِزْ) 'deniz, ulu göl'."  parantezde Arap
        #               harfli biçim + tırnak
        #      kalem -> "Yazı kamışı, yontulmuş kamış Eski Grekçe"  sona
        #               yapışmış DİL ADI
        #    Latin harfi içermeyen parantezli ekler ve sondaki dil adı
        #    anlam değildir; kodlanırsa mesafeyi şişirirler.
        cleaned = re.sub(
            r"\(\s*[^)a-zçğıöşüâîûA-ZÇĞİÖŞÜ]{1,40}\s*\)", " ", cleaned
        )
        cleaned = cleaned.strip().strip("\"'“”‘’ .,;")
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
        # ⚠️ Bu kuralı `kalem` için dize ORTASINDAKİ dil adını da kesecek
        # şekilde genelletmeyi denedim; birim testte çalıştı ama gerçek
        # vakayı hiç etkilemedi (0.7589 -> 0.7589). Sebebi katman hatasıydı:
        # `kalem` bir ALINTI, `_select_hypothesis`'in donor_lexicon dalından
        # geçiyor ve tarihî anlamı `donor_etymology_database`'den geliyor —
        # bu fonksiyona hiç uğramıyor. Doğru yer tek boğaz noktası olan
        # `diachronic_semantic_engine._clean_gloss`; genelleme geri alındı.
        for name in _TRAILING_LANGUAGE_NAMES:
            if cleaned.endswith(" " + name):
                cleaned = cleaned[: -len(name)].strip().strip(",;")
                break

        if cleaned:
            candidates.append((_form_distance(word, str(entry.get("word") or "")), cleaned))

    if return_all:
        seen: set[str] = set()
        out: list[str] = []
        for _, text in candidates:
            if text not in seen:
                seen.add(text)
                out.append(text)
        return out

    if not candidates:
        return ""

    # ⚠️ EŞİK DEĞİL SIRALAMA. Seçici eskiden İLK tanığı alıyordu ve biçime
    # hiç bakmıyordu; `bilge` için 1. tanık `belgü` "işaret, alamet" —
    # BAŞKA BİR KELİME (belgü -> belge). Doğru tanık (`bilge` "Âlim, hakim,
    # bilgin.") hemen arkasındaydı. Mesafe 0.8147 çıkıyordu.
    #
    # Biçim mesafesine EŞİK konamaz, ölçüldü: `bilge~belge` oranı 0.20 ve
    # bu, meşru ses denkliklerinden DAHA İYİ — `göz~köz` 0.33,
    # `deniz~teŋiz` 0.40, `el~elig` 0.50. Eşik yanlışı eleyemeden
    # doğruları keserdi.
    #
    # En yakın biçimi seçmek ise hiçbir şeyi elemez, yalnız sırayı düzeltir:
    # `bilge`(0.00) `belgü`(0.40) ve `belge`(0.20)'yi yener; `deniz`de tek
    # aday `teŋiz` zaten seçilir. `word` verilmezse eski davranış (ilk
    # tanık) korunur.
    if not (word or "").strip():
        return candidates[0][1]
    return min(candidates, key=lambda c: c[0])[1]


def _attested_proto_root(
    word: str,
    root: dict[str, Any],
    entries: list[dict[str, Any]],
    fetcher_results: list[dict[str, Any]] | None = None,
) -> str:
    """Kaynağın açıkça verdiği Proto-Türkçe kök; yoksa ``""``.

    Başlıkla AYNI kaynaklar ve öncelik (``search_engine``): önce sorgunun kendi
    miras kaydının verdiği biçim (``_query_source_proto``), sonra sorgunun
    KENDİ Starling kökü.

    ⚠️ Yalnız VERİLEN girdiler okunur: Starling kökü ``fetcher_results``te
    yoksa veritabanından çekilmez (çağıran kaynağı getirmediyse kök tanıklı
    değildir). Starling kök varyantlarıyla da sorgulanır (`kulluk` -> `kul`);
    sonucun sorgunun kendisine ait olduğu doğrudan aramayla doğrulanır.
    ⚠️ İlk fetcher'ın kökü (``root.proto_turkic``) kullanılmaz:
    EtimolojiTürkçe'nin ETü adımı çoğu kez türetme tabanıdır (`açıkgöz` ->
    *açuk) ve başlık onu göstermez.
    """
    from engine.search_engine import _query_source_proto

    form, _lang = _query_source_proto(word, entries, primary=str(root.get("meaning") or ""))
    if form:
        return form
    fetched = {
        str((r.get("root") or {}).get("proto_turkic") or "")
        for r in fetcher_results or []
        if (r.get("root") or {}).get("starling_proto")
    } - {""}
    if not fetched:
        return ""
    try:
        from engine.fetchers.starling import StarlingFetcher

        own = StarlingFetcher().fetch(word).get("root") or {}
    except Exception:
        return ""
    own_root = str(own.get("proto_turkic") or "")
    return own_root if own.get("starling_proto") and own_root in fetched else ""


class IterativeHypothesisEngine:
    def __init__(self) -> None:
        self.donor_db = DeepDonorEtymologyDatabase()
        self.neologism_detector = NeologismDetector()
        self.attestation_verifier = HistoricalAttestationVerifier()
        self.validator_protocol = HypothesisValidationProtocol()
        self.reconstructor = ComparativeReconstructor()

    def prove_etymological_hypothesis(
        self,
        word: str,
        initial_finding: dict[str, Any],
        turkic_entries: list[dict[str, Any]] | None = None,
        fetcher_results: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        w = (word or "").strip().lower()
        entries = turkic_entries if turkic_entries is not None else initial_finding.get("turkic_languages", [])
        root = initial_finding.get("root", {}) or {}

        neologism = self.neologism_detector.detect(w)
        donor_match = self.donor_db.lookup(w)
        if not donor_match:
            # Tohum veritabanı 10 kelimelik; kaynağın açık "Alıntı" adımı
            # varsa verici o. Yoksa A-HVP "kaynak dil belirlenemedi" diyordu
            # (`faça`: EtimolojiTürkçe İtalyanca faccia veriyordu).
            from engine.nlp.borrowing_chain import source_loan_step

            step = source_loan_step(entries)
            if step:
                donor_match = {
                    "donor_language": step.get("lang_name") or "",
                    "origin_form": step.get("word") or "",
                    "etymology": f"Kaynağın alıntı zinciri ({step.get('source') or 'kaynak'}): "
                                 f"{step.get('lang_name')} {step.get('word')}",
                    "historical_meaning": step.get("meaning") or "",
                    "hypothesis_type": "Kaynakta alıntı olarak kayıtlı",
                }
        attestation = self.attestation_verifier.verify_attestation(w, entries, fetcher_results)
        reconstruction = self.reconstructor.reconstruct(w, entries)

        hypothesis = self._select_hypothesis(
            w,
            root,
            neologism,
            donor_match,
            reconstruction,
            str(_historical_gloss(entries, w)),
            _historical_gloss_candidates(entries),
            attested_root=_attested_proto_root(w, root, entries, fetcher_results),
        )

        if hypothesis is None:
            return {
                "word": w,
                "hypothesis_available": False,
                "proven_hypothesis": None,
                "attestation": attestation,
                "validation_report": None,
                "reason": (
                    "Ne donör sözlük eşleşmesi, ne modern türetme kalıbı, ne de yeterli "
                    "akraba tanığı bulundu. Hipotez ÜRETİLMEDİ (uydurma yapılmaz)."
                ),
            }

        val_report = self.validator_protocol.validate_hypothesis(w, hypothesis, attestation, entries)
        hypothesis["validation_report"] = val_report
        hypothesis["confidence_score"] = val_report["final_confidence_score"]

        return {
            "word": w,
            "hypothesis_available": True,
            "proven_hypothesis": hypothesis,
            "attestation": attestation,
            "validation_report": val_report,
        }

    @staticmethod
    def _select_hypothesis(
        w: str,
        root: dict[str, Any],
        neologism: dict[str, Any] | None,
        donor_match: dict[str, Any] | None,
        reconstruction: dict[str, Any],
        historical_gloss: str = "",
        historical_candidates: list[str] | None = None,
        attested_root: str = "",
    ) -> dict[str, Any] | None:
        """Kanıt gücüne göre en iyi hipotezi seçer. Kanıt yoksa ``None``.

        :param attested_root: kaynağın AÇIKÇA verdiği Proto-Türkçe biçim
            (başlık kökü). Varsa A-HVP motorun rekonstrüksiyonunu değil onu
            sınar.
        """
        modern_meaning = root.get("meaning", "") or ""

        # 1. Donör sözlük eşleşmesi — en güçlü doğrudan kanıt
        if donor_match:
            return {
                "hypothesis_type": donor_match.get("hypothesis_type") or "Doğrulanmış alıntı kökeni",
                "donor_language": donor_match["donor_language"],
                "origin_form": donor_match["origin_form"],
                "proof_summary": donor_match.get("etymology", ""),
                "historical_meaning": donor_match.get("historical_meaning", ""),
                "modern_meaning": modern_meaning,
                "evidence_kind": "donor_lexicon",
            }

        # 2. Modern türetme (bileşik veya güçlü özleştirme eki)
        if neologism and neologism.get("is_neologism"):
            return {
                "hypothesis_type": "Modern Türkçe türetme (Cumhuriyet dönemi)",
                "donor_language": "Türkçe (modern türetme)",
                "origin_form": "+".join(neologism.get("components", [])) or w,
                "proof_summary": neologism.get("etymology_details", ""),
                "historical_meaning": "",
                "modern_meaning": modern_meaning,
                "evidence_kind": "morphological_derivation",
                "is_modern_derivation": True,
            }

        # 3. Karşılaştırmalı yöntemle ata biçim (en az 2 bağımsız dil tanığı)
        if reconstruction.get("evidence_available") and reconstruction.get("reconstructed_root"):
            # ⚠️ Tanıklı kök varken A-HVP motorun KENDİ rekonstrüksiyonunu
            # sınıyordu: başlık `uçmak` *uč- (kaynak) derken A-HVP *uça'yı
            # sınayıp onayı/reddi ona veriyordu (105 kelimelik denetim: 14
            # kelimede başlık kökü ≠ A-HVP ata biçimi). Kaynak kökü verdiyse
            # sınanan odur; motorun biçimi karşılaştırma için ayrı alanda kalır.
            engine_root = str(reconstruction["reconstructed_root"])
            tested_root = attested_root or engine_root
            return {
                "hypothesis_type": (
                    "Asli Proto-Türkçe kök (tanıklı kök, karşılaştırmalı yöntemle sınandı)"
                    if attested_root else "Asli Proto-Türkçe kök (karşılaştırmalı yöntem)"
                ),
                "donor_language": "Proto-Türkçe",
                "origin_form": tested_root,
                "engine_reconstruction": engine_root,
                "origin_form_attested": bool(attested_root),
                "proof_summary": reconstruction.get("reconstruction_notes", ""),
                # ⚠️ Burada `modern_meaning` yazıyordu; iki alan aynı olunca
                # A-HVP 3. aşaması anlamı kendisiyle karşılaştırıyordu.
                # Gerçek tarihî gloss artık tanıklardan geliyor (bkz.
                # `_historical_gloss`); tanık yoksa boş kalır ve aşama
                # dürüstçe "ölçülemedi" der.
                "historical_meaning": historical_gloss,
                "historical_meaning_candidates": historical_candidates,
                "modern_meaning": modern_meaning,
                "evidence_kind": "comparative_method",
                "witness_count": reconstruction.get("witness_count", 0),
                "branches": reconstruction.get("branches", []),
                "applied_correspondences": reconstruction.get("applied_correspondences", []),
            }

        # 4. Fonotaktik ihlal — alıntı adayı, kaynak dil belirsiz
        violated, reason = initial_consonant_violation(w)
        if violated:
            return {
                "hypothesis_type": "Alıntı adayı (kaynak dil belirlenemedi)",
                "donor_language": "",
                "origin_form": w,
                "proof_summary": f"{reason}. Kaynak dil için donör sözlük kanıtı gerekiyor.",
                "historical_meaning": "",
                "modern_meaning": modern_meaning,
                "evidence_kind": "phonotactic_only",
                "is_loan_candidate": True,
            }

        return None
