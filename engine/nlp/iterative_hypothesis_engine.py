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

logger = get_logger(__name__)


#: Tarihî tanık katmanları. Aynı üçlü `search_engine.py:413` ve
#: `fetchers/historical_index.py:49`'da da kullanılıyor.
HISTORICAL_WITNESS_LANGUAGES: tuple[str, ...] = ("otk", "ota", "chg")


def _historical_gloss(entries: list[dict[str, Any]] | None) -> str:
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

        # 3. Atıf öneki yalnız KIRPILIR, kayıt atılmaz: gloss önekten sonra
        #    gerçekten duruyor.
        cleaned = citation.sub("", meaning).strip()
        if cleaned:
            return cleaned
    return ""


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
        attestation = self.attestation_verifier.verify_attestation(w, entries, fetcher_results)
        reconstruction = self.reconstructor.reconstruct(w, entries)

        hypothesis = self._select_hypothesis(
            w, root, neologism, donor_match, reconstruction, _historical_gloss(entries)
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
    ) -> dict[str, Any] | None:
        """Kanıt gücüne göre en iyi hipotezi seçer. Kanıt yoksa ``None``."""
        modern_meaning = root.get("meaning", "") or ""

        # 1. Donör sözlük eşleşmesi — en güçlü doğrudan kanıt
        if donor_match:
            return {
                "hypothesis_type": "Doğrulanmış alıntı kökeni",
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
            return {
                "hypothesis_type": "Asli Proto-Türkçe kök (karşılaştırmalı yöntem)",
                "donor_language": "Proto-Türkçe",
                "origin_form": reconstruction["reconstructed_root"],
                "proof_summary": reconstruction.get("reconstruction_notes", ""),
                # ⚠️ Burada `modern_meaning` yazıyordu; iki alan aynı olunca
                # A-HVP 3. aşaması anlamı kendisiyle karşılaştırıyordu.
                # Gerçek tarihî gloss artık tanıklardan geliyor (bkz.
                # `_historical_gloss`); tanık yoksa boş kalır ve aşama
                # dürüstçe "ölçülemedi" der.
                "historical_meaning": historical_gloss,
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
