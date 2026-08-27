"""
Denklik düzenliliği teşhisi — CoPaR (List 2019, *Computational Linguistics*).

⚠️ **Plan bunu elle yazılmış denkliklerin YERİNE koymayı öngörüyordu.**
Ölçüm o planı değiştirdi: elle yazılmış denklikleri öğrenilmiş sayımların
önüne koymak dilbilimsel olarak yerleşik iki kararı bozuyordu::

    {tr: y, kk: z, otk: d}  ->  *j       (doğrusu *d̮)
    *teŋiŕ                  ->  *teniŕ   (ŋ sütunu kayboldu)

Elle yazılmış denklikler dar ve küratörlüdür (bkz.
``proto_phonology.CORRESPONDENCES``); öğrenilmiş katman onların **arkasında**
duruyor ve orada katkı sağlıyor (bkz. ``nlp/proto_patterns.py``).

Bu modül CoPaR'ı **teşhis** olarak kullanır, karar katmanı olarak değil:
verimizin ne kadarı düzenli denklik örüntüleriyle açıklanabiliyor?

## Neden önemli

Düzenlilik oranı, rekonstrüksiyon doğruluğunun **üst sınırını** belirler.
Sütunların yarısı hiçbir örüntüye oturmuyorsa, kural tabanlı bir sistemin o
sütunlarda doğru ata sesi bulması ancak şansa kalır. Blum & List (2023)
hizalama budamasının bu oranı 10 ailenin 10'unda artırdığını ölçüyor —
bizde budamanın rekonstrüksiyon kazancı da (tam 0,361 -> 0,386) bununla
tutarlı.

⚠️ ``lingrex`` yoksa modül **hiçbir sayı üretmez**.
"""

from __future__ import annotations

import contextlib
import io
import json
from dataclasses import dataclass
from typing import Any

from engine.logging_setup import get_logger

logger = get_logger(__name__)

#: CoPaR'ın bir örüntüyü "düzenli" sayması için gereken asgari sütun sayısı.
#:
#: ⚠️ Sayı doğrudan sonucu belirler: 2 dersek neredeyse her sütun bir
#: örüntüye oturur ve oran anlamını yitirir.
MIN_REFS = 3


@dataclass(frozen=True)
class RegularityReport:
    n_cognate_sets: int
    n_sites: int
    n_patterns: int
    n_singleton_patterns: int
    regular_ratio: float
    purity: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_cognate_sets": self.n_cognate_sets,
            "n_sites": self.n_sites,
            "n_patterns": self.n_patterns,
            "n_singleton_patterns": self.n_singleton_patterns,
            "regular_ratio": round(self.regular_ratio, 4),
            "purity": round(self.purity, 4) if self.purity is not None else None,
            "min_refs": MIN_REFS,
        }


def _cv(form: str) -> str:
    """Boşlukla ayrılmış CV iskeleti — CoPaR'ın ``structure`` sütunu.

    ⚠️ İskelet **karşılaştırma biçmi** üzerinden kurulur, IPA üzerinden
    değil; bölütlemeyle birebir hizalanmalı, yoksa CoPaR konumları kaydırır.
    """
    from engine.utils.phonotactics import VOWELS

    return " ".join("V" if character in VOWELS else "C" for character in form)


def _wordlist_rows(dataset: str, split: str | None) -> list[list[Any]]:
    """CoPaR'ın beklediği LingPy tablosu."""
    from engine.db.cldf_wordlist import CldfWordlist
    from engine.db.language_mapping import build_mapping
    from engine.evaluation.gold import assign_split
    from engine.utils.orthography import to_comparison_form

    wordlist = CldfWordlist.load(dataset)
    mapping = build_mapping(wordlist)
    # ⚠️ LingPy ``ipa`` sütununu zorunlu tutuyor (``Alignments`` onu
    # ``transcription`` diye arıyor); yoksa KeyError ile çöküyor.
    # ⚠️ CoPaR ``structure`` sütununu ZORUNLU tutuyor: örüntüler prozodik
    # konuma (söz başı / iç / son) göre ayrılıyor. Sütun yoksa ValueError.
    rows: list[list[Any]] = [
        ["doculect", "concept", "ipa", "tokens", "structure", "cogid"]
    ]
    for index, cognate_set in enumerate(wordlist.cognate_sets(min_languages=2), start=1):
        concept = cognate_set.concept or cognate_set.id
        if split and assign_split(concept) != split:
            continue
        for language, form in cognate_set.forms_by_language().items():
            code = mapping.get(language)
            comparison = to_comparison_form(form)
            if not code or not comparison:
                continue
            rows.append(
                [code, concept, comparison, list(comparison), _cv(comparison), index]
            )
    return rows


def measure(dataset: str = "savelyevturkic", split: str | None = "train") -> RegularityReport | None:
    """Verinin ne kadarının düzenli denklik örüntüleriyle açıklandığını ölçer."""
    try:
        from lingpy import Wordlist
        from lingrex.copar import CoPaR
    except ImportError:
        logger.info("lingrex/lingpy yok — düzenlilik teşhisi atlanıyor")
        return None

    rows = _wordlist_rows(dataset, split)
    if len(rows) < 20:
        return None
    # ⚠️ Tek dilli akraba kümeleri CoPaR'ı çökertiyor; hizalanacak bir şey
    # yok. Ölçüme de bilgi katmazlar.
    from collections import Counter

    per_set = Counter(row[5] for row in rows[1:])
    rows = [rows[0]] + [row for row in rows[1:] if per_set[row[5]] >= 2]

    data = {index: row for index, row in enumerate(rows[1:], start=1)}
    wordlist = Wordlist({0: rows[0], **data})
    try:
        # ⚠️ CoPaR ilerleme çubuklarını stdout'a basıyor ve ölçüm çıktısını
        # okunmaz kılıyor; bastırılıyor.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            copar = CoPaR(wordlist, ref="cogid", fuzzy=False, minrefs=MIN_REFS)
            copar.add_alignments()
            copar.get_sites()
            copar.cluster_sites()
            copar.sites_to_pattern()
    except Exception:
        logger.warning("CoPaR koşulamadı", exc_info=True)
        return None

    patterns = getattr(copar, "clusters", {}) or {}
    sites = getattr(copar, "sites", {}) or {}
    singleton = sum(1 for members in patterns.values() if len(members) < 2)
    total_sites = len(sites)
    covered = sum(len(members) for members in patterns.values() if len(members) >= 2)
    try:
        purities = copar.purity(ref="cogid")
        purity = sum(purities.values()) / max(len(purities), 1)
    except Exception:
        logger.debug("örüntü saflığı hesaplanamadı", exc_info=True)
        purity = None

    return RegularityReport(
        n_cognate_sets=len({row[5] for row in rows[1:]}),
        n_sites=total_sites,
        n_patterns=len(patterns),
        n_singleton_patterns=singleton,
        regular_ratio=covered / total_sites if total_sites else 0.0,
        purity=purity,
    )


def main() -> int:
    from engine.evaluation.report import EVAL_DIR

    report = measure()
    if report is None:
        print(
            "Düzenlilik teşhisi koşulamadı (lingrex yok veya veri yetersiz).\n"
            "⚠️ Sayı üretilmiyor — uydurma bir oran, hiç oran vermemekten kötüdür."
        )
        return 1
    print("=== DENKLİK DÜZENLİLİĞİ (CoPaR, TRAIN bölümü) ===")
    print(
        f"{report.n_cognate_sets} akraba kümesi · {report.n_sites} hizalama sütunu"
    )
    print(
        f"{report.n_patterns} örüntü ({report.n_singleton_patterns} tekil) · "
        f"**düzenli sütun oranı {report.regular_ratio:.4f}**"
    )
    if report.purity is not None:
        print(f"örüntü saflığı {report.purity:.4f}")
    print(
        "\n⚠️ Bu oran rekonstrüksiyon doğruluğunun ÜST SINIRINI belirler:\n"
        "   hiçbir örüntüye oturmayan sütunda kural tabanlı bir sistem\n"
        "   ancak şansa kalır."
    )
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out = EVAL_DIR / "regularity.json"
    out.write_text(
        json.dumps(
            {"_schema": "turkic-etymology-regularity/v1", **report.as_dict()},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nJSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
