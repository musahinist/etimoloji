"""
Taban çizgisi raporu — ``make eval-baseline``.

Bu modülün varlık sebebi: plandaki **ön ölçüm** sayıları (%29 tam / %45 kabul
edilebilir) oturum içi geçici betiklerle üretilmiş ve commit edilmemişti,
yani kimse yeniden üretemiyordu. Bu, planın kendi tekrarlanabilirlik ilkesiyle
çelişiyordu.

Burada üretilen ``data/eval/baseline.json`` ve ``BASELINE.md`` her koşuda
sıfırdan hesaplanır; veri kümesi sürümü, checksum'ı ve ölçüm koşulları da
yazılır. Sayı değişirse **neden değiştiği** dosyadan görülür.

Rapor dört koşulu yan yana verir; aradaki fark ölçümün ne kadar kolaylaştığını
gösterir::

    çapa dili DIŞLANIR / DAHİL        anchor sızıntısının etkisi
    tüm veri / 15+ tanık              kolay altküme seçmenin etkisi
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.config import PROJECT_ROOT
from engine.db.cldf_wordlist import CldfWordlist
from engine.db.language_mapping import build_mapping, unmapped_languages
from engine.evaluation.baselines import BASELINES
from engine.evaluation.gold import GoldStandard
from engine.evaluation.harness import comparative_reconstructor, run
from engine.evaluation.negative_controls import ALL_BATTERIES, run_battery
from engine.evaluation.significance import compare_systems
from engine.logging_setup import get_logger

logger = get_logger(__name__)

EVAL_DIR = PROJECT_ROOT / "data" / "eval"


@dataclass(frozen=True)
class Condition:
    """Bir ölçüm koşulu — hangi maddeler, hangi sızıntı ayarıyla."""

    name: str
    min_witnesses: int
    exclude_anchor: bool
    note: str


CONDITIONS: tuple[Condition, ...] = (
    Condition(
        "tum_veri_capa_haric",
        0,
        True,
        "DÜRÜST KOŞUL — tüm altın standart, çapa dilinin tanığı girdiden çıkarılmış",
    ),
    Condition(
        "tum_veri_capa_dahil",
        0,
        False,
        "çapa dili girdide bırakılmış — motor kendi sorusunu tanık olarak görüyor",
    ),
    Condition(
        "15_tanik_capa_haric",
        15,
        True,
        "yalnız 15+ tanıklı kolay altküme, çapa çıkarılmış",
    ),
    Condition(
        "15_tanik_capa_dahil",
        15,
        False,
        "kolay altküme + çapa sızıntısı — ÖN ÖLÇÜMÜN koşuluna en yakın hâli",
    ),
)


def measure(dataset: str = "savelyevturkic", split: str | None = None) -> dict[str, Any]:
    """Motoru ve tüm taban çizgilerini her koşulda ölçer."""
    gold = GoldStandard.build(dataset)
    wordlist = CldfWordlist.load(dataset)
    mapping = build_mapping(wordlist)
    unmapped = unmapped_languages(wordlist)

    pool = gold.items if split is None else gold.split(split)
    systems: dict[str, Any] = {"comparative": comparative_reconstructor(), **BASELINES}

    results: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        items = [i for i in pool if i.witness_count >= condition.min_witnesses]
        per_system: dict[str, Any] = {}
        for name, system in systems.items():
            outcome = run(
                system,
                items,
                mapping=mapping,
                split=split or "all",
                system=name,
                exclude_anchor_language=condition.exclude_anchor,
            )
            per_system[name] = outcome.as_dict()
            per_system[name]["_correct_flags"] = outcome.item_correct
            if name == "comparative":
                per_system[name]["by_proto_level"] = outcome.by_proto_level

        # ⚠️ Anlamlılık testi yalnız AYNI maddeler üzerinde yapılabilir.
        # Sistemler farklı maddelerde çekimser kaldığında bayrak dizileri
        # farklı uzunlukta olur ve eşleşmiş test geçersizdir.
        flags = {
            name: stats.pop("_correct_flags")
            for name, stats in per_system.items()
        }
        reference = max(
            (n for n in flags if n != "comparative"),
            key=lambda n: per_system[n]["accuracy"],
            default=None,
        )
        comparisons: list[dict[str, Any]] = []
        if reference:
            comparisons = compare_systems(
                {"comparative": flags["comparative"], reference: flags[reference]},
                reference=reference,
            )
        results[condition.name] = {
            "note": condition.note,
            "n_items": len(items),
            "systems": per_system,
            "significance": comparisons,
            "significance_note": (
                "Eşleşmiş permütasyon ve McNemar testi; bootstrap %95 güven "
                "aralığı sıfırı içeriyorsa fark anlamlı değildir."
            ),
        }

    # Negatif kontroller ANA SONUCUN YANINDA raporlanır: yüksek doğruluk,
    # yüksek yanlış-pozitif oranıyla birlikte anlamsızdır.
    engine_fn = systems["comparative"]
    controls = [
        run_battery(engine_fn, battery_items, name).as_dict()
        for name, battery_items in ALL_BATTERIES.items()
    ]

    return {
        "_schema": "turkic-etymology-baseline/v1",
        "negative_controls": controls,
        "measured_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset": dataset,
        "dataset_ref": wordlist.provenance.get("ref", ""),
        "dataset_commit": wordlist.provenance.get("commit", ""),
        "split": split or "all",
        "gold_summary": gold.summary(),
        "unmapped_languages": unmapped,
        "conditions": results,
    }


def write_markdown(payload: dict[str, Any], path: Path) -> Path:
    """İnsan-okunur taban çizgisi tablosu."""
    gold = payload["gold_summary"]
    lines = [
        "# Taban çizgisi ölçümü",
        "",
        "Bu dosya `make eval-baseline` tarafından **otomatik üretilir** — elle",
        "düzenlemeyin. Her sayı, adı geçen veri kümesi sürümünden sıfırdan",
        "hesaplanır.",
        "",
        f"- **Veri kümesi:** `{payload['dataset']}` `{payload['dataset_ref']}` "
        f"(commit `{payload['dataset_commit'][:12]}`)",
        f"- **Ölçüm:** {payload['measured_at']}",
        f"- **Bölüm:** `{payload['split']}`",
        f"- **Altın standart:** {gold['total']} madde · "
        f"train {gold['splits']['train']} / dev {gold['splits']['dev']} / test {gold['splits']['test']}",
        f"- **Kavram sızıntısı:** {gold['concept_leakage']} (0 olmalı)",
        f"- **Ata düğüm:** PT {gold['proto_levels'].get('PT', 0)} · "
        f"PCT {gold['proto_levels'].get('PCT', 0)} — Çuvaşça tanığı olmayan kümede "
        f"iddia edilebilecek en derin düğüm Ana Ortak Türkçe'dir",
        "",
    ]
    if payload["unmapped_languages"]:
        lines += [
            f"> ⚠️ {len(payload['unmapped_languages'])} dil motor koduna eşlenemedi ve "
            f"değerlendirmeye girmedi: {', '.join(payload['unmapped_languages'])}",
            "",
        ]

    for name, block in payload["conditions"].items():
        lines += [
            f"## `{name}` — n={block['n_items']}",
            "",
            f"> {block['note']}",
            "",
            "| Sistem | tam | kabul edilebilir | ED | NED | FER | kapsam |",
            "|---|---|---|---|---|---|---|",
        ]
        for system, stats in block["systems"].items():
            marker = "**" if system == "comparative" else ""
            lines.append(
                f"| {marker}{system}{marker} | {stats['accuracy']:.3f} | {stats['acceptable']:.3f} "
                f"| {stats['ED']:.2f} | {stats['NED']:.3f} | {stats['FER']:.3f} | {stats['coverage']:.3f} |"
            )
        lines.append("")
        for comparison in block.get("significance", []):
            low, high = comparison["ci95"]
            verdict = (
                "**anlamlı**"
                if comparison["significant_after_fdr"]
                else "anlamlı DEĞİL — güven aralığı sıfırı içeriyor"
            )
            lines += [
                f"> `comparative` vs `{comparison['vs']}`: fark "
                f"**{comparison['difference']:+.4f}**, %95 GA [{low:+.4f}, {high:+.4f}], "
                f"permütasyon p={comparison['permutation_p']:.3f}, "
                f"McNemar p={comparison['mcnemar_p']:.3f} → {verdict}.",
                "",
            ]

    honest = payload["conditions"]["tum_veri_capa_haric"]["systems"]
    leaky = payload["conditions"]["15_tanik_capa_dahil"]["systems"]
    lines += [
        "## Yorum",
        "",
        f"Dürüst koşulda motor **%{honest['comparative']['accuracy'] * 100:.1f} tam** "
        f"(%{honest['comparative']['acceptable'] * 100:.1f} kabul edilebilir) alıyor.",
        "",
        f"Aynı motor, kolay altküme seçilip çapa sızıntısı bırakıldığında "
        f"**%{leaky['comparative']['accuracy'] * 100:.1f} tam** "
        f"(%{leaky['comparative']['acceptable'] * 100:.1f}) gösteriyor. Aradaki fark yöntemden "
        "değil, ölçüm kurgusundan geliyor — bu yüzden raporlanan sayı daima "
        "dürüst koşulun sayısıdır.",
        "",
        "Motorun aşması gereken çıta, en iyi trivial taban çizgisidir: "
        + ", ".join(f"`{s}` %{honest[s]['accuracy'] * 100:.1f}" for s in honest if s != "comparative")
        + ".",
        "",
        "### Negatif kontroller",
        "",
        "Yüksek doğruluk, yüksek yanlış-pozitif oranıyla birlikte anlamsızdır.",
        "**Güçlü iddia oranı** en kritik sütundur: motorun uydurma veya alıntı",
        "bir kelimeye 🟢/🟡 rozet verme oranı sıfır olmalıdır.",
        "",
        "| Batarya | n | rekonstrükte | yanlış-pozitif | güçlü iddia |",
        "|---|---|---|---|---|",
    ]
    for control in payload.get("negative_controls", []):
        lines.append(
            f"| `{control['battery']}` | {control['n']} | {control['reconstructed']} "
            f"| {control['false_positive_rate']:.3f} | **{control['strong_claim_rate']:.3f}** |"
        )
    lines += [
        "",
        "### İstatistiksel durum",
        "",
        "Her koşulun altındaki satır, motor ile en iyi trivial taban çizgisi",
        "arasındaki farkın eşleşmiş permütasyon ve McNemar testiyle sınanmış",
        "sonucunu verir. **Bootstrap güven aralığı sıfırı içeriyorsa fark",
        "anlamlı değildir** ve öyle raporlanır — bu, motorun kötü olduğunu",
        "değil, farkın henüz kanıtlanmadığını söyler.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Taban çizgisi ölçümü ve raporu")
    ap.add_argument("--dataset", default="savelyevturkic")
    ap.add_argument("--split", default=None, help="varsayılan: tüm veri")
    ap.add_argument("--out", type=Path, default=EVAL_DIR)
    args = ap.parse_args()

    payload = measure(args.dataset, args.split)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "baseline.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = write_markdown(payload, args.out / "BASELINE.md")

    for name, block in payload["conditions"].items():
        stats = block["systems"]["comparative"]
        best_baseline = max(
            (s for s in block["systems"] if s != "comparative"),
            key=lambda s: block["systems"][s]["accuracy"],
        )
        print(
            f"{name:24} n={block['n_items']:<4} motor={stats['accuracy']:.3f} "
            f"kabul={stats['acceptable']:.3f}  "
            f"en iyi taban çizgi={best_baseline} {block['systems'][best_baseline]['accuracy']:.3f}"
        )
    print(f"\nRapor: {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
