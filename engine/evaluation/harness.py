"""
Değerlendirme koşum takımı — motoru altın standarda karşı koşar.

**Sızıntı önleme burada uygulanır.** Motora yalnız *tanık biçimler* verilir;
altın ata biçim, kavram adı ve küme kimliği **verilmez**. Motorun gördüğü şey
gerçek kullanımdaki girdinin aynısıdır: bir avuç akraba kelime.

Ayrıca ``baselines`` ile birlikte koşulur: Cui ve ark. (2024) rastgele taban
çizginin %3,68, modelin %54 aldığını gösteriyor — bu farkı **göstermek
zorunludur**, yoksa "%29 aldık" cümlesi hiçbir şey ifade etmez.

Kullanım::

    python -m engine.evaluation.harness --split dev
    python -m engine.evaluation.harness --split dev --json rapor.json
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine.db.cldf_wordlist import CldfWordlist
from engine.db.language_mapping import build_mapping
from engine.evaluation.gold import GoldItem, GoldStandard
from engine.evaluation.metrics import (
    ReconstructionScore,
    best_match,
    feature_error_rate,
    normalize_proto,
    score_reconstructions,
)
from engine.logging_setup import get_logger

logger = get_logger(__name__)

#: Motora verilen bir tanık kaydı. Alanlar bilerek azdır: fetcher'lardan
#: gelen gerçek kaydın taşıdığı kadarını taşır, fazlasını değil.
Witness = dict[str, str]

#: ``(word, entries) -> sonuç sözlüğü`` imzalı rekonstrüktör.
Reconstructor = Callable[[str, list[Witness]], dict[str, Any]]


@dataclass
class HarnessResult:
    """Bir koşunun tüm çıktısı — metrikler, hata modları, per-item kayıt."""

    split: str
    system: str
    score: ReconstructionScore
    error_modes: Counter = field(default_factory=Counter)
    by_proto_level: dict[str, dict[str, float]] = field(default_factory=dict)
    #: Kalibrasyon için: yalnız motorun CEVAP VERDİĞİ maddeler.
    confidences: list[float] = field(default_factory=list)
    correctness: list[bool] = field(default_factory=list)
    #: Anlamlılık testi için: **her** madde, çekimserlik ``False`` sayılarak.
    #: Eşleşmiş test ancak diziler aynı maddeleri kapsarsa geçerlidir.
    item_correct: list[bool] = field(default_factory=list)
    item_ids: list[str] = field(default_factory=list)
    mean_fer: float = 0.0
    mean_witnesses: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "system": self.system,
            **self.score.as_dict(),
            "FER": round(self.mean_fer, 4),
            "mean_witnesses": round(self.mean_witnesses, 2),
            "error_modes": dict(self.error_modes.most_common()),
            "by_proto_level": self.by_proto_level,
        }


def classify_error(predicted: str, gold: str, item: GoldItem) -> str:
    """Hata modunu adlandırır — toplam skordan çok bu döküm yol gösterir."""
    p, g = normalize_proto(predicted), normalize_proto(gold)
    if p == g:
        return "dogru"
    if not p:
        return "cekimser"

    p_nolen = normalize_proto(predicted, strip_length=True)
    g_nolen = normalize_proto(gold, strip_length=True)
    if p_nolen == g_nolen:
        return "unlu_uzunlugu"
    if not item.has_length_witness and len(g) != len(g_nolen):
        return "unlu_uzunlugu_taniksiz"
    if len(p) > len(g) + 1:
        return "ek_soyulmamis"
    if len(p) < len(g) - 1:
        return "capa_kisa"
    if g and p and g[-1] in "ŕĺ" and p[-1] in "zş":
        return "rotasizm_lambdaizm_kacirildi"
    if p[:1] != g[:1]:
        return "soz_basi_yanlis"
    return "ses_hatasi"


def _witnesses_for(item: GoldItem, mapping: dict[str, str]) -> list[Witness]:
    """Altın maddeyi motorun göreceği biçime çevirir — **cevap sızdırmadan**."""
    out: list[Witness] = []
    for cldf_lang, form in sorted(item.witnesses.items()):
        code = mapping.get(cldf_lang)
        if not code or not form:
            continue
        out.append({"lang_code": code, "word": form, "source": "gold-harness"})
    return out


def _anchor_for(witnesses: list[Witness]) -> tuple[str, str]:
    """Sorgu (çapa) biçmini ve **dilini** seçer.

    Gerçek kullanımda kullanıcı Türkçe bir kelime yazar. Altın maddede Türkçe
    tanık varsa o kullanılır; yoksa en çok konuşulan modern dilden biri.
    Çapa **tanıkların içinden** gelir — altın cevaptan değil.

    :returns: ``(biçim, dil_kodu)``. Dil kodu döndürülmesi şart: çapayı
        girdiden çıkarırken **dile** göre çıkarmak gerekir. Kelimeye göre
        çıkarmak, aynı biçmi paylaşan bütün dilleri birden siler — Türki
        dillerde ``tïrnaḳ`` gibi biçimler on dilde birden aynıdır, ve bu
        tanık sayısını sıfıra düşürüp motoru haksız yere çekimser bırakır.
    """
    from engine.utils.orthography import to_comparison_form

    # Normalize edilince boş kalan biçim çapa olamaz: rekonstrüksiyon hiç
    # başlamaz ve madde haksız yere çekimser sayılır.
    by_code = {
        w["lang_code"]: w["word"]
        for w in witnesses
        if w.get("word") and to_comparison_form(w["word"])
    }
    for preferred in ("tr", "az", "tk", "gag", "kk", "ky", "tt", "uz", "ug"):
        if preferred in by_code:
            return by_code[preferred], preferred
    for code, form in by_code.items():
        return form, code
    return "", ""


def run(
    reconstructor: Reconstructor,
    items: list[GoldItem],
    *,
    mapping: dict[str, str],
    split: str = "dev",
    system: str = "comparative",
    exclude_anchor_language: bool = True,
) -> HarnessResult:
    """Motoru altın maddeler üzerinde koşar ve puanlar.

    :param exclude_anchor_language: çapa olarak kullanılan dilin tanığı
        girdiden çıkarılır mı? Çıkarılmazsa motor kendi sorusunu cevap olarak
        geri görür — küçük ama gerçek bir sızıntıdır.
    """
    pairs: list[tuple[str, str]] = []
    result = HarnessResult(split=split, system=system, score=ReconstructionScore())
    abstentions = 0
    fers: list[float] = []
    witness_counts: list[int] = []
    per_level: dict[str, list[bool]] = {}

    for item in items:
        # Her madde sonuç dizisine MUTLAKA bir kayıt bırakır; aksi hâlde
        # sistemler arası eşleşmiş test kurulamaz.
        result.item_ids.append(item.set_id)
        result.item_correct.append(False)
        witnesses = _witnesses_for(item, mapping)
        anchor, anchor_lang = _anchor_for(witnesses)
        if not anchor:
            abstentions += 1
            result.error_modes["esleyen_dil_yok"] += 1
            continue
        if exclude_anchor_language:
            witnesses = [w for w in witnesses if w["lang_code"] != anchor_lang]
        witness_counts.append(len(witnesses))

        try:
            output = reconstructor(anchor, witnesses)
        except Exception:
            logger.warning("Rekonstrüksiyon çöktü: %s", anchor, exc_info=True)
            abstentions += 1
            result.error_modes["cokme"] += 1
            continue

        predicted = str(output.get("reconstructed_root") or "")
        if not predicted or not output.get("is_reconstructible"):
            abstentions += 1
            result.error_modes["cekimser"] += 1
            continue

        # Eşdeğer rekonstrüksiyonlardan hangisi tutuyorsa ona göre puanlanır:
        # altın ``*jaŋï / *jeŋi`` derken motorun ikisinden birini bulması
        # başarıdır, en katı adaya göre puanlamak haksız olurdu.
        matched_gold, is_exact, _ = best_match(predicted, item.gold_candidates)
        pairs.append((predicted, matched_gold))
        result.error_modes[classify_error(predicted, matched_gold, item)] += 1
        fers.append(feature_error_rate(predicted, matched_gold))

        result.item_correct[-1] = is_exact
        confidence = output.get("confidence")
        if isinstance(confidence, (int, float)):
            result.confidences.append(float(confidence))
            result.correctness.append(is_exact)
        per_level.setdefault(item.proto_level, []).append(is_exact)

    result.score = score_reconstructions(pairs, abstentions=abstentions)
    result.mean_fer = sum(fers) / len(fers) if fers else 0.0
    result.mean_witnesses = sum(witness_counts) / len(witness_counts) if witness_counts else 0.0
    result.by_proto_level = {
        level: {"n": len(flags), "accuracy": round(sum(flags) / len(flags), 4)}
        for level, flags in sorted(per_level.items())
    }
    return result


def comparative_reconstructor() -> Reconstructor:
    """Motorun mevcut karşılaştırmalı rekonstrüktörü."""
    from engine.nlp.comparative_reconstruction import ComparativeReconstructor

    engine = ComparativeReconstructor()
    return lambda word, entries: engine.reconstruct(word, entries)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Altın standarda karşı rekonstrüksiyon ölçümü")
    ap.add_argument("--split", default="dev", choices=("train", "dev", "test"))
    ap.add_argument("--dataset", default="savelyevturkic")
    ap.add_argument("--json", type=Path, help="sonucu JSON olarak yaz")
    ap.add_argument(
        "--final-report",
        action="store_true",
        help="test bölümünü açmak için gerekli bilinçli onay",
    )
    args = ap.parse_args()

    gold = GoldStandard.build(args.dataset)
    items = gold.split(args.split, i_am_writing_the_final_report=args.final_report)
    mapping = build_mapping(CldfWordlist.load(args.dataset))

    result = run(
        comparative_reconstructor(),
        items,
        mapping=mapping,
        split=args.split,
        system="comparative",
    )

    payload = result.as_dict()
    print(f"\n=== {args.dataset} / {args.split} · n={payload['n']} ===")
    for key in ("accuracy", "acceptable", "ED", "NED", "FER", "coverage", "mean_witnesses"):
        print(f"  {key:16} {payload[key]}")
    print("\n  ata düğüme göre:")
    for level, stats in result.by_proto_level.items():
        print(f"    {level:5} n={stats['n']:<4} doğruluk={stats['accuracy']}")
    print("\n  hata modları:")
    for mode, count in result.error_modes.most_common():
        print(f"    {mode:32} {count}")

    if args.json:
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON yazıldı: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
