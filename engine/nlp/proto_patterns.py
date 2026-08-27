"""
Denklik örüntüsünden ata ses öğrenimi — denetimli katman (Faz D2).

⚠️ **Bu modül "sıfır eğitim verisi" iddiasını bitirir.** Kullanıcı kararıyla
400 uzman etiketli küme artık eğitim verisidir. Gerekçe ölçülmüştür:

===========================  ==========================
sistem sınıfı                Rom-phon tam doğruluk
===========================  ==========================
CorPaR (kural/örüntü)        %22,2
SVM+PosStrIni (kural/örüntü) %24,7
**bizim kural katmanımız**   **%23,5**
RNN (denetimli)              %52,3
Transformer (denetimli)      %53,8
===========================  ==========================

Kural tabanlı paradigma tavanı bizim uygulamamızda değil, paradigmadadır.

## Yöntem

Hizalanmış her sütun bir **denklik örüntüsüdür**: hangi dilde hangi ses
görünüyor. Örüntü, altın kümelerden o sütuna karşılık gelen ata sesle
eşleştirilir ve sayılır. Test sırasında sütunun örüntüsü tabloda aranır.

⚠️ **Örüntü tam eşleşmesi işe yaramaz.** Ölçüldü: 400 kümenin dil kümeleri
neredeyse hiç birebir örtüşmüyor, tam örüntü eşleşmesi test kümesinde
neredeyse sıfır isabet veriyor. Bu yüzden örüntü **dil-ses çiftlerine**
ayrıştırılır ve her çift ayrı bir oy verir::

    sütun {cv: r, tr: z, kk: z}  ->  ("cv","r")  ("tr","z")  ("kk","z")

Her çift için ``(dil, ses) -> ata ses`` dağılımı öğrenilir. Bu, elle yazılmış
``ARCHAISM_WEIGHTS`` oylamasının **veriden öğrenilmiş** karşılığıdır: bir
dilin bir sesinin ata sese ne kadar tanıklık ettiği, o dile atfedilen genel
"arkaiklik" yerine gerçek sayımdan gelir.

⚠️ **Bölme kavram bazlıdır** (``gold.assign_split``). Aynı kavramın başka
bir kümesi eğitimde görünürse sızıntı olur.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from engine.config import PROJECT_ROOT
from engine.logging_setup import get_logger

logger = get_logger(__name__)

MODEL_PATH = PROJECT_ROOT / "data" / "models" / "proto_patterns.json"

#: Bir ``(dil, ses)`` çiftinin oy kullanabilmesi için gereken asgari gözlem.
#:
#: ⚠️ Tek gözleme dayanan bir çift, gürültüyü kural sanmaktır. Eşik
#: ``MIN_SUPPORT``tan düşük çiftler tabloya girmez.
MIN_SUPPORT = 3

#: Öğrenilmiş oyun elle yazılmış kurala baskın gelmesi için gereken güven.
#:
#: ⚠️ Ölçüldü (önceki tur): öğrenilmiş ters-refleks tablosu sütun düzeyinde
#: +5,7 puan veriyor ama **kelime düzeyinde kaybettiriyor** (0,324 -> 0,257).
#: Yani tabloyu koşulsuz üstün tutmak zarar veriyor; yalnız yüksek güvenli
#: kararlarda devreye girmeli.
MIN_CONFIDENCE = 0.60


@dataclass
class ProtoPatternTable:
    """``(dil, ses) -> ata ses`` sayımları."""

    counts: dict[tuple[str, str], Counter] = field(
        default_factory=lambda: defaultdict(Counter)
    )
    trained_on: str = ""
    n_sets: int = 0
    n_columns: int = 0
    trained_at: str = ""

    @property
    def is_trained(self) -> bool:
        return bool(self.counts)

    def observe(self, language: str, sound: str, proto: str) -> None:
        self.counts[(language, sound)][proto] += 1

    def vote(self, column_sounds: dict[str, str]) -> tuple[str, float, int]:
        """Sütundaki dil-ses çiftlerinden ata ses oyu.

        :returns: ``(ata_ses, güven, destek)``. Hiçbir çift yeterli desteğe
            sahip değilse ``("", 0.0, 0)``.
        """
        tally: Counter = Counter()
        support = 0
        for language, sound in column_sounds.items():
            observed = self.counts.get((language, sound))
            if not observed:
                continue
            total = sum(observed.values())
            if total < MIN_SUPPORT:
                continue
            support += 1
            # ⚠️ Oy **olasılıkla** ağırlıklandırılır, ham sayımla değil.
            # Ham sayım büyük dilleri (çok tanıklı olanları) kayırırdı.
            for proto, count in observed.items():
                tally[proto] += count / total
        if not tally:
            return ("", 0.0, 0)
        winner, score = tally.most_common(1)[0]
        return (winner, score / sum(tally.values()), support)

    def as_dict(self) -> dict[str, Any]:
        return {
            "_schema": "turkic-etymology-proto-patterns/v1",
            "trained_on": self.trained_on,
            "trained_at": self.trained_at,
            "n_sets": self.n_sets,
            "n_columns": self.n_columns,
            "min_support": MIN_SUPPORT,
            "counts": {
                f"{language}|{sound}": dict(protos.most_common())
                for (language, sound), protos in sorted(self.counts.items())
                if sum(protos.values()) >= MIN_SUPPORT
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProtoPatternTable:
        table = cls(
            trained_on=str(data.get("trained_on", "")),
            n_sets=int(data.get("n_sets", 0)),
            n_columns=int(data.get("n_columns", 0)),
            trained_at=str(data.get("trained_at", "")),
        )
        for key, protos in (data.get("counts") or {}).items():
            language, sound = key.split("|", 1)
            table.counts[(language, sound)] = Counter(
                {k: int(v) for k, v in protos.items()}
            )
        return table


def save(table: ProtoPatternTable, path: Path | None = None) -> Path:
    target = path or MODEL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(table.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return target


def load(path: Path | None = None) -> ProtoPatternTable | None:
    source = path or MODEL_PATH
    if not source.exists():
        return None
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("ata ses örüntü tablosu okunamadı: %s", source)
        return None
    table = ProtoPatternTable.from_dict(data)
    return table if table.is_trained else None


def learn(split: str = "train", *, dataset: str = "savelyevturkic") -> ProtoPatternTable:
    """Altın kümelerden ``(dil, ses) -> ata ses`` tablosunu öğrenir.

    ⚠️ Yalnız ``split`` kavramları kullanılır; bölme **kavram bazlıdır**.
    Aynı kavramın başka bir kümesi eğitimde görünürse sızıntı olur.
    """
    from engine.evaluation.gold import GoldStandard
    from engine.nlp.multi_alignment import align_forms
    from engine.utils.orthography import to_comparison_form

    gold = GoldStandard.build(dataset)
    from engine.db.cldf_wordlist import CldfWordlist
    from engine.db.language_mapping import build_mapping

    mapping = build_mapping(CldfWordlist.load(dataset))

    table = ProtoPatternTable(trained_on=f"{dataset}/{split}")
    for item in gold.split(split):
        proto = to_comparison_form(item.gold_form)
        forms = {
            mapping[lang]: to_comparison_form(form)
            for lang, form in item.witnesses.items()
            if lang in mapping and to_comparison_form(form)
        }
        if len(forms) < 2 or not proto:
            continue
        columns = align_forms(forms)
        informative = [c for c in columns if c.gap_ratio <= 0.5]
        # ⚠️ Sütun sayısı ata biçim uzunluğuyla tutmuyorsa hizalama ile ata
        # biçim arasında birebir eşleme kurulamaz. Ölçüldü: 8.135 uzman
        # hizalamasının genişliği altın ``Root`` uzunluğuyla yalnız **%33**
        # tutuyor. Tutmayan kümeler ATLANIR — yanlış hizalanmış bir sütundan
        # öğrenmek, hiç öğrenmemekten kötüdür.
        if len(informative) != len(proto):
            continue
        table.n_sets += 1
        for column, proto_sound in zip(informative, proto, strict=True):
            table.n_columns += 1
            for language, sound in column.present.items():
                if sound:
                    table.observe(language, sound, proto_sound)

    table.trained_at = datetime.now(UTC).isoformat(timespec="seconds")
    logger.info(
        "Ata ses örüntü tablosu: %d küme, %d sütun, %d (dil, ses) çifti",
        table.n_sets,
        table.n_columns,
        len(table.counts),
    )
    return table


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Ata ses örüntü tablosunu öğren")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()
    table = learn(args.split)
    if not table.is_trained:
        print("Örüntü öğrenilemedi.")
        return 1
    path = save(table)
    usable = sum(1 for c in table.counts.values() if sum(c.values()) >= MIN_SUPPORT)
    print(
        f"{table.n_sets} küme · {table.n_columns} sütun · "
        f"{len(table.counts)} çift ({usable} destekli) · {path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
