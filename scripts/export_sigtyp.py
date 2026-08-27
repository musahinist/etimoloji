#!/usr/bin/env python3
"""
SIGTYP 2022 biçiminde veri kümesi dışa aktarımı — Faz E4.

Altın standardımızı alanın **yayınlanmış paylaşılan görev biçimine** çevirir
ki başka sistemler bu veride ölçülebilsin ve bizim sayılarımız bağımsız
olarak yeniden üretilebilsin.

SIGTYP 2022 (*Shared Task on the Prediction of Cognate Reflexes*) biçimi::

    <dizin>/
        cognates.tsv          COGID + dil sütunları, hücreler boşlukla
                              ayrılmış bölütler; ata dil bir sütundur
        training-0.10.tsv     kız dillerin %10'u GİZLİ, gerisi görünür
        test-0.10.tsv         gizlenen hücreler "?" ile işaretli
        solutions-0.10.tsv    gizlenen hücrelerin gerçek değerleri
        …0.20 …0.30 …0.40 …0.50

⚠️ **Bu bir katkı önerisi değil, veri yayınıdır.** ST2022'ye dış veri PR
ile eklenmemiş; organizatörler veri kümelerini Lexibank'tan derlemiş.
Gerçekçi rota: Lexibank uyumlu CLDF + Zenodo DOI yayınlamak ve ST2022
bölmelerini ondan türetmek.

⚠️ **Bölmeler deterministiktir.** Rastgelelik ``SPLIT_SEED`` ile tohumlanır;
aynı komut aynı bölmeleri üretir, yoksa "aynı veride ölçtük" iddiası
kurulamaz.

Kullanım::

    python scripts/export_sigtyp.py
    python scripts/export_sigtyp.py --out data/sigtyp --dataset savelyevturkic
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.config import PROJECT_ROOT  # noqa: E402

#: Gizlenen hücre işareti — ST2022 sözleşmesi.
MASK = "?"

#: Ata dil sütununun adı.
PROTO_COLUMN = "Proto-Turkic"

#: Üretilecek gizleme oranları.
PROPORTIONS = (0.10, 0.20, 0.30, 0.40, 0.50)

#: ⚠️ Bölmeler deterministik olmalı; aynı komut aynı bölmeleri üretmeli.
SPLIT_SEED = 20260827

#: Bir kümenin dışa aktarılması için gereken asgari kız dil tanığı.
MIN_WITNESSES = 2


def _segments(form: str) -> str:
    """Biçmi boşlukla ayrılmış bölütlere çevirir.

    ⚠️ ST2022 hücreleri **bölütlenmiş** ister. Bizde bölütleme harf
    düzeyindedir; IPA çok-karakterli bölütleri (``t͡ʃ``) korunmaz. Bu bir
    kısıttır ve künyeye yazılır.
    """
    return " ".join(form)


def collect(dataset: str) -> tuple[list[str], list[dict[str, str]]]:
    """Altın standarttan ST2022 satırlarını toplar."""
    from engine.db.cldf_wordlist import CldfWordlist
    from engine.db.language_mapping import build_mapping
    from engine.evaluation.gold import GoldStandard
    from engine.utils.orthography import to_comparison_form

    gold = GoldStandard.build(dataset)
    mapping = build_mapping(CldfWordlist.load(dataset))

    rows: list[dict[str, str]] = []
    languages: set[str] = set()
    for item in gold.items:
        proto = to_comparison_form(item.gold_form)
        forms = {
            mapping[lang]: to_comparison_form(form)
            for lang, form in item.witnesses.items()
            if lang in mapping and to_comparison_form(form)
        }
        if not proto or len(forms) < MIN_WITNESSES:
            continue
        languages |= set(forms)
        row = {"COGID": item.set_id, PROTO_COLUMN: _segments(proto)}
        row.update({lang: _segments(form) for lang, form in forms.items()})
        rows.append(row)
    return sorted(languages), rows


def _write(path: Path, columns: list[str], rows: list[dict[str, str]]) -> int:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", restval="")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def _mask(
    rows: list[dict[str, str]], languages: list[str], proportion: float, seed: int
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Kız dil hücrelerinin ``proportion`` kadarını gizler.

    ⚠️ Gizlenen hücreler **kız dillerden** seçilir; ata sütunu her zaman
    görünürdür. ST2022 görevi "kız dillerden ata biçmi tahmin et" değil,
    "eksik refleksi tahmin et"tir; iki görevi karıştırmak biçimin anlamını
    bozar.

    ⚠️ Her satırda en az bir kız hücre **görünür kalır**; hepsi gizlenirse
    tahmin edilecek bağlam kalmaz.
    """
    rng = random.Random(seed)
    training: list[dict[str, str]] = []
    test: list[dict[str, str]] = []
    solutions: list[dict[str, str]] = []

    for row in rows:
        present = [lang for lang in languages if row.get(lang)]
        if len(present) < 2:
            training.append(dict(row))
            continue
        count = max(1, min(len(present) - 1, round(len(present) * proportion)))
        hidden = set(rng.sample(present, count))

        train_row = {k: v for k, v in row.items() if k not in hidden}
        test_row = dict(row)
        solution_row = {"COGID": row["COGID"]}
        for lang in hidden:
            solution_row[lang] = row[lang]
            test_row[lang] = MASK
        training.append(train_row)
        test.append(test_row)
        solutions.append(solution_row)
    return training, test, solutions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(dataset: str, out: Path) -> dict[str, object]:
    languages, rows = collect(dataset)
    if not rows:
        raise RuntimeError("dışa aktarılacak küme yok")

    out.mkdir(parents=True, exist_ok=True)
    columns = ["COGID", PROTO_COLUMN, *languages]
    files: dict[str, dict[str, object]] = {}

    path = out / "cognates.tsv"
    files["cognates.tsv"] = {"rows": _write(path, columns, rows), "sha256": _sha256(path)}

    for proportion in PROPORTIONS:
        tag = f"{proportion:.2f}"
        training, test, solutions = _mask(rows, languages, proportion, SPLIT_SEED)
        for name, data, cols in (
            (f"training-{tag}.tsv", training, columns),
            (f"test-{tag}.tsv", test, columns),
            (f"solutions-{tag}.tsv", solutions, columns),
        ):
            target = out / name
            files[name] = {
                "rows": _write(target, cols, data),
                "sha256": _sha256(target),
            }

    provenance = {
        "_schema": "turkic-etymology-sigtyp-export/v1",
        "format": "SIGTYP 2022 Shared Task on Cognate Reflex Prediction",
        "source_dataset": dataset,
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "split_seed": SPLIT_SEED,
        "n_cognate_sets": len(rows),
        "n_languages": len(languages),
        "languages": languages,
        "proto_column": PROTO_COLUMN,
        "caveats": [
            "Bölütleme HARF düzeyindedir; IPA çok-karakterli bölütleri "
            "(t͡ʃ gibi) korunmaz.",
            "Gizlenen hücreler yalnız kız dillerden seçilir; ata sütunu her "
            "zaman görünürdür.",
            "Her satırda en az bir kız hücre görünür kalır.",
            "ST2022'ye dış veri PR ile eklenmemiştir; bu bir VERİ YAYINIDIR, "
            "katkı önerisi değil.",
        ],
        "files": files,
    }
    (out / "_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return provenance


def main() -> int:
    ap = argparse.ArgumentParser(description="SIGTYP 2022 biçiminde dışa aktarım")
    ap.add_argument("--dataset", default="savelyevturkic")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "sigtyp"))
    args = ap.parse_args()

    provenance = export(args.dataset, Path(args.out))
    print(
        f"{provenance['n_cognate_sets']} akraba kümesi · "
        f"{provenance['n_languages']} dil · {args.out}"
    )
    for name, info in provenance["files"].items():
        print(f"  {name:22} {info['rows']:>5} satır  {info['sha256'][:12]}")
    print("\nUyarılar:")
    for caveat in provenance["caveats"]:
        print(f"  ⚠️ {caveat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
