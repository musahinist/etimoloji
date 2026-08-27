"""
İleri yönde akraba tahmini — zincirin kopuk ilk halkası.

Etimolojisi yapılmamış bir kelimeyi çözmek için **akrabasını bulmak** gerekir.
Akrabanın nasıl göründüğü bilinmediğinden ses kanunları ileri yönde uygulanıp
tahmin edilmeli, sonra o tahmin aranmalıdır::

    akraba keşfi  ->  tanık sayısı  ->  rekonstrüksiyon

Bu modül ilk halkayı kurar: "Türkçe ``kar`` Kazakçada ne olur?" sorusuna
cevap verir.

**Kurallar elle yazılmaz, veriden öğrenilir.** Önceki uygulama
(``sound_shifts.SOUND_SHIFT_RULES``) elle yazılmış bir listeydi ve ölçülen
hataların kaynağıydı: ``ş ~ s`` denkliği hiç yoktu, bu yüzden ``baş ~ бас``
ve ``taş ~ тас`` bulunamıyordu.

Burada denklikler ``savelyevturkic``in **train** kavramlarından, uzman
akraba kümeleri hizalanarak sayılır. Her dil çifti için konum duyarlı bir
tablo çıkar::

    (tr, kk) konum=final :  ş -> s  (14 kez)  |  ş -> ş (2 kez)

⚠️ Tablolar yalnız TRAIN kavramlarından öğrenilir; dev/test kavramlarını
görmek ölçümü geçersiz kılardı.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path

from engine.config import PROJECT_ROOT
from engine.logging_setup import get_logger
from engine.nlp.multi_alignment import GAP, align_forms
from engine.utils.orthography import to_comparison_form

logger = get_logger(__name__)

CORRESPONDENCE_PATH = PROJECT_ROOT / "data" / "correspondences" / "learned.json"

#: Bir denkliğin tabloya girmesi için gereken asgari gözlem sayısı.
#: Tek gözlemli denklik kural değil, gürültüdür.
MIN_SUPPORT = 2

#: Konum etiketleri — ses kanunları konuma duyarlıdır.
POSITIONS = ("initial", "medial", "final")


def position_of(index: int, length: int) -> str:
    if index == 0:
        return "initial"
    if index >= length - 1:
        return "final"
    return "medial"


@dataclass
class CorrespondenceTable:
    """Bir dil çifti için öğrenilmiş, konum duyarlı denklik tablosu."""

    source: str
    target: str
    #: ``(konum, kaynak_ses) -> {hedef_ses: sayım}``
    counts: dict[tuple[str, str], Counter] = field(default_factory=lambda: defaultdict(Counter))

    def observe(self, position: str, source_sound: str, target_sound: str) -> None:
        self.counts[(position, source_sound)][target_sound] += 1

    def predict(self, position: str, sound: str) -> tuple[str, float]:
        """En olası hedef sesi ve olasılığını döndürür.

        Konuma özgü kayıt yoksa konumdan bağımsız toplama düşülür; o da
        yoksa ses **olduğu gibi** bırakılır (en muhafazakâr tahmin).
        """
        specific = self.counts.get((position, sound))
        if specific and sum(specific.values()) >= MIN_SUPPORT:
            best, count = specific.most_common(1)[0]
            return best, count / sum(specific.values())

        pooled: Counter = Counter()
        for pos in POSITIONS:
            pooled.update(self.counts.get((pos, sound), Counter()))
        if pooled and sum(pooled.values()) >= MIN_SUPPORT:
            best, count = pooled.most_common(1)[0]
            return best, count / sum(pooled.values())
        return sound, 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "target": self.target,
            "rules": {
                f"{position}|{sound}": dict(targets.most_common())
                for (position, sound), targets in sorted(self.counts.items())
                if sum(targets.values()) >= MIN_SUPPORT
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> CorrespondenceTable:
        table = cls(source=data["source"], target=data["target"])
        for key, targets in data.get("rules", {}).items():
            position, sound = key.split("|", 1)
            table.counts[(position, sound)] = Counter(targets)
        return table


@dataclass
class PredictedForm:
    """Bir dil için tahmin edilen biçim ve gerekçesi."""

    language: str
    form: str
    confidence: float
    steps: list[dict[str, object]] = field(default_factory=list)
    support: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "language": self.language,
            "predicted_form": self.form,
            "confidence": round(self.confidence, 3),
            "correspondence_support": self.support,
            "steps": self.steps,
        }


class CognatePredictor:
    """Öğrenilmiş denkliklerle ileri yönde akraba biçim tahmini."""

    def __init__(self, tables: dict[tuple[str, str], CorrespondenceTable] | None = None):
        self.tables = tables if tables is not None else load_tables()

    def predict(self, word: str, source: str, target: str) -> PredictedForm:
        """``source`` dilindeki ``word``ün ``target`` dilindeki beklenen biçmi."""
        form = to_comparison_form(word)
        if not form:
            return PredictedForm(target, "", 0.0)

        table = self.tables.get((source, target))
        if table is None:
            # Denklik tablosu yoksa uydurma yapılmaz: kaynak biçim olduğu gibi
            # döner ve güven sıfırdır. "Bilmiyorum" demek, yanlış tahminden iyidir.
            return PredictedForm(target, form, 0.0, [{"note": "denklik tablosu yok"}])

        chars: list[str] = []
        probabilities: list[float] = []
        steps: list[dict[str, object]] = []
        for index, sound in enumerate(form):
            position = position_of(index, len(form))
            predicted, probability = table.predict(position, sound)
            if predicted != GAP:
                chars.append(predicted)
            probabilities.append(probability)
            if predicted != sound:
                steps.append(
                    {
                        "position": position,
                        "from": sound,
                        "to": predicted,
                        "probability": round(probability, 3),
                    }
                )

        confidence = sum(probabilities) / len(probabilities) if probabilities else 0.0
        support = sum(sum(c.values()) for c in table.counts.values())
        return PredictedForm(target, "".join(chars), confidence, steps, support)

    def predict_all(self, word: str, source: str = "tr") -> list[PredictedForm]:
        """Bilinen bütün hedef diller için tahmin — akraba araması bununla yapılır."""
        targets = sorted({target for (src, target) in self.tables if src == source})
        predictions = [self.predict(word, source, target) for target in targets]
        return sorted(predictions, key=lambda p: (-p.confidence, p.language))


# --- Öğrenme ---------------------------------------------------------------


def learn_tables(
    dataset: str = "savelyevturkic",
    *,
    split: str = "train",
) -> dict[tuple[str, str], CorrespondenceTable]:
    """Uzman akraba kümelerinden denklik tablolarını çıkarır.

    ⚠️ Yalnız ``split`` kavramları kullanılır. dev/test kavramlarını görmek
    ileri tahmin ölçümünü geçersiz kılardı.
    """
    from engine.db.cldf_wordlist import CldfWordlist
    from engine.db.language_mapping import build_mapping
    from engine.evaluation.gold import assign_split

    wordlist = CldfWordlist.load(dataset)
    mapping = build_mapping(wordlist)
    tables: dict[tuple[str, str], CorrespondenceTable] = {}
    used_sets = 0

    for cognate_set in wordlist.cognate_sets(min_languages=2):
        if assign_split(cognate_set.concept or cognate_set.id) != split:
            continue
        forms = {
            mapping[lang]: to_comparison_form(form)
            for lang, form in cognate_set.forms_by_language().items()
            if lang in mapping and to_comparison_form(form)
        }
        if len(forms) < 2:
            continue
        columns = align_forms(forms)
        if not columns:
            continue
        used_sets += 1

        width = len(columns)
        for index, column in enumerate(columns):
            position = position_of(index, width)
            present = column.sounds
            for source, source_sound in present.items():
                if not source_sound or source_sound == GAP:
                    continue
                for target, target_sound in present.items():
                    if source == target or not target_sound:
                        continue
                    key = (source, target)
                    if key not in tables:
                        tables[key] = CorrespondenceTable(source=source, target=target)
                    tables[key].observe(position, source_sound, target_sound)

    logger.info(
        "Denklik tabloları öğrenildi: %d dil çifti, %d akraba kümesinden (%s)",
        len(tables),
        used_sets,
        split,
    )
    return tables


def save_tables(
    tables: dict[tuple[str, str], CorrespondenceTable],
    *,
    trained_on: str,
    path: Path | None = None,
) -> Path:
    out = path or CORRESPONDENCE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "_schema": "turkic-etymology-correspondences/v1",
                "trained_on": trained_on,
                "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "min_support": MIN_SUPPORT,
                "n_pairs": len(tables),
                "note": (
                    "Yalnız TRAIN kavramlarından öğrenilmiştir. Kurallar elle "
                    "yazılmamış, uzman akraba kümeleri hizalanarak sayılmıştır."
                ),
                "tables": [table.as_dict() for table in tables.values()],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Denklik tabloları yazıldı: %s (%d çift)", out, len(tables))
    return out


@lru_cache(maxsize=1)
def load_tables(path: Path | None = None) -> dict[tuple[str, str], CorrespondenceTable]:
    """Öğrenilmiş tabloları diskten okur. Yoksa boş sözlük."""
    source = path or CORRESPONDENCE_PATH
    if not source.exists():
        logger.info("Denklik tablosu yok (%s); ileri tahmin devre dışı", source)
        return {}
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Denklik tablosu okunamadı: %s", source, exc_info=True)
        return {}
    tables: dict[tuple[str, str], CorrespondenceTable] = {}
    for entry in data.get("tables", []):
        table = CorrespondenceTable.from_dict(entry)
        tables[(table.source, table.target)] = table
    return tables


def reset_table_cache() -> None:
    load_tables.cache_clear()


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Denklik tablolarını öğren ve yaz")
    ap.add_argument("--dataset", default="savelyevturkic")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    tables = learn_tables(args.dataset, split=args.split)
    if not tables:
        print("Denklik öğrenilemedi.")
        return 1
    path = save_tables(tables, trained_on=f"{args.dataset}/{args.split}")

    rules = sum(len(t.as_dict()["rules"]) for t in tables.values())
    print(f"{len(tables)} dil çifti · {rules} kural · {path}")

    predictor = CognatePredictor(tables)
    print("\nörnek tahminler (tr ->):")
    for word in ("kar", "baş", "taş", "göz", "yol", "tuz"):
        row = []
        for target in ("kk", "tt", "cv", "sah"):
            prediction = predictor.predict(word, "tr", target)
            row.append(f"{target}:{prediction.form or '—'}")
        print(f"  {word:6} {'  '.join(row)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
