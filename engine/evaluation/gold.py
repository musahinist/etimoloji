"""
Altın standart kurulumu ve **sızıntı önleyici** veri ayrımı.

Planın en kritik tek maddesi burada uygulanır. Hakem hükmü şuydu:

    "Sızıntı tek tek fazlarda değil, mimaride: aynı kaynaklar hem sinyal, hem
    eğitim, hem sınav. Bu düzeltilmeden plandaki diğer her şey — en parlak
    katkı iddiası dahil — 'kendi verisinde kendini sınamış sistem'e indirgenir."

Üç önlem alınır:

1. **kaikki/Wiktionary altın standarda GİRMEZ.** Wiktionary'nin Proto-Türkçe
   rekonstrüksiyonları büyük ölçüde EDAL/Starostin soyundandır ve amatör
   elden geçmiştir; motor da zaten Wiktionary okur. Aynı kaynaktan hem soru
   hem cevap alınamaz (Häuser & Stamatakis 2025).

2. **Ayrım kavram (Concepticon) bazındadır**, küme veya dil bazında değil.
   Küme bazlı ayrım aynı kavramın farklı dillerdeki biçimlerini iki tarafa
   düşürür — klasik sızıntı.

3. **Test bölümü dondurulur.** ``data/gold/test.frozen.json`` bir kez yazılır,
   checksum'ı kaydedilir ve geliştirme boyunca açılmaz. Yalnız nihai rapor
   için koşulur. Yeniden üretim deterministiktir: bölme, kavram kimliğinin
   SHA-256'sına göre yapılır — rastgele tohum yoktur, dolayısıyla makineler
   arasında da aynıdır.

Kullanım::

    from engine.evaluation.gold import GoldStandard

    gold = GoldStandard.build()          # savelyevturkic'ten kurar
    train = gold.split("train")
    dev = gold.split("dev")
    # test yalnız nihai raporda:
    test = gold.split("test", i_am_writing_the_final_report=True)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from engine.config import GOLD_DIR
from engine.db.cldf_wordlist import CldfWordlist
from engine.evaluation.metrics import parse_gold_form
from engine.logging_setup import get_logger

logger = get_logger(__name__)

#: Bölme oranları. Kavram kimliğinin özetine göre deterministik atanır.
SPLIT_RATIOS: dict[str, float] = {"train": 0.60, "dev": 0.20, "test": 0.20}

#: Bölme tuzu. Değiştirilirse **tüm ayrım değişir** ve dondurulmuş test seti
#: geçersiz olur; bu yüzden asla değiştirilmez.
SPLIT_SALT = "turkic-etymology-gold-v1"


def assign_split(concept_id: str) -> str:
    """Bir kavramı deterministik olarak train/dev/test'e atar.

    Rastgele tohum kullanılmaz: aynı kavram her makinede, her koşuda aynı
    bölüme düşer. Bu, "test setini yanlışlıkla eğitimde kullandık" hatasını
    yeniden üretilebilir biçimde imkânsızlaştırır.
    """
    digest = hashlib.sha256(f"{SPLIT_SALT}:{concept_id}".encode()).digest()
    position = int.from_bytes(digest[:8], "big") / float(1 << 64)
    cumulative = 0.0
    for name, ratio in SPLIT_RATIOS.items():
        cumulative += ratio
        if position < cumulative:
            return name
    return "test"


@dataclass(frozen=True)
class GoldItem:
    """Altın standarttaki tek bir rekonstrüksiyon sorusu."""

    set_id: str
    gold_form: str
    #: ``Root`` alanı serbest metindir; ``*jaŋï / *jeŋi`` gibi **eşdeğer**
    #: rekonstrüksiyonlar içerebilir. Motorun herhangi birini bulması doğrudur.
    gold_candidates: tuple[str, ...]
    concept: str
    concepticon_gloss: str
    proto_level: str
    witnesses: dict[str, str]
    has_length_witness: bool
    split: str

    @property
    def witness_count(self) -> int:
        return len(self.witnesses)


class GoldStandard:
    """Uzman kararlarından kurulmuş, bölünmüş ve mühürlenmiş altın standart."""

    def __init__(self, items: list[GoldItem], *, source: str, source_ref: str = ""):
        self.items = items
        self.source = source
        self.source_ref = source_ref

    # -- kurulum ------------------------------------------------------------

    @classmethod
    def build(
        cls,
        dataset: str = "savelyevturkic",
        *,
        min_languages: int = 2,
        notation: str = "transliteration",
    ) -> GoldStandard:
        """CLDF veri kümesinden altın standardı kurar.

        Yalnız ``*`` ile başlayan ``Root`` değerleri alınır: ``savelyevturkic``
        v2.1'de 905 kümenin **519'u** rekonstrüksiyon taşır, kalanların
        ``Root``u ``DIRNAK`` gibi bir Türkçe kavram etiketidir ve ata biçim
        değildir.
        """
        wordlist = CldfWordlist.load(dataset)
        items: list[GoldItem] = []
        skipped = 0
        for cognate_set in wordlist.cognate_sets(min_languages=min_languages, reconstructed_only=True):
            candidates = parse_gold_form(cognate_set.root)
            if not candidates:
                # Ayrıştırılamayan ``Root`` (yalnız künye veya yorum içeren)
                # altın standarda alınmaz — sessizce yanlış puanlanmasındansa
                # hiç sorulmaması doğrudur.
                skipped += 1
                continue
            items.append(
                GoldItem(
                    set_id=cognate_set.id,
                    gold_form=candidates[0],
                    gold_candidates=tuple(candidates),
                    concept=cognate_set.concept,
                    concepticon_gloss=cognate_set.concepticon_gloss,
                    proto_level=cognate_set.proto_level,
                    witnesses=cognate_set.forms_by_language(notation=notation),
                    has_length_witness=cognate_set.has_length_witness,
                    split=assign_split(cognate_set.concept or cognate_set.id),
                )
            )
        ref = str(wordlist.provenance.get("ref", ""))
        logger.info(
            "Altın standart kuruldu: %s (%s) — %d madde (%d ayrıştırılamayan Root atlandı)",
            dataset,
            ref,
            len(items),
            skipped,
        )
        return cls(items, source=dataset, source_ref=ref)

    # -- bölümler -----------------------------------------------------------

    def split(
        self,
        name: str,
        *,
        i_am_writing_the_final_report: bool = False,
    ) -> list[GoldItem]:
        """Bir bölümü döndürür.

        ``test`` bölümü **kasıtlı olarak zor erişilir**: açıkça
        ``i_am_writing_the_final_report=True`` demeden alınamaz. Amaç teknik
        bir kilit değil — geliştirme sırasında "bir de test setinde deneyeyim"
        refleksini bilinçli bir karara dönüştürmektir.
        """
        if name == "test" and not i_am_writing_the_final_report:
            raise PermissionError(
                "Test bölümü dondurulmuştur. Geliştirme sırasında 'dev' kullanın. "
                "Nihai rapor yazıyorsanız i_am_writing_the_final_report=True geçin — "
                "ve bunu yaptığınız koşuyu raporda belirtin."
            )
        return [item for item in self.items if item.split == name]

    def counts(self) -> dict[str, int]:
        counts = dict.fromkeys(SPLIT_RATIOS, 0)
        for item in self.items:
            counts[item.split] = counts.get(item.split, 0) + 1
        return counts

    def concept_leakage(self) -> list[str]:
        """Birden çok bölüme düşen kavram var mı? Boş liste dönmeli."""
        seen: dict[str, set[str]] = {}
        for item in self.items:
            seen.setdefault(item.concept, set()).add(item.split)
        return sorted(c for c, splits in seen.items() if len(splits) > 1)

    # -- mühürleme ----------------------------------------------------------

    def freeze(self, directory: Path | None = None) -> dict[str, str]:
        """Bölümleri diske yazar ve test setini checksum'la mühürler."""
        out_dir = directory or GOLD_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        checksums: dict[str, str] = {}

        for name in SPLIT_RATIOS:
            items = [i for i in self.items if i.split == name]
            filename = "test.frozen.json" if name == "test" else f"{name}.json"
            payload = json.dumps(
                {
                    "_schema": "turkic-etymology-gold/v1",
                    "split": name,
                    "source": self.source,
                    "source_ref": self.source_ref,
                    "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "count": len(items),
                    "items": [asdict(i) for i in items],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            path = out_dir / filename
            path.write_text(payload, encoding="utf-8")
            checksums[name] = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        seal = {
            "_schema": "turkic-etymology-gold-seal/v1",
            "source": self.source,
            "source_ref": self.source_ref,
            "salt": SPLIT_SALT,
            "ratios": SPLIT_RATIOS,
            "counts": self.counts(),
            "checksums": checksums,
            "sealed_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "note": (
                "test.frozen.json geliştirme boyunca açılmaz. Checksum değişmişse "
                "ya veri kümesi sürümü ya bölme tuzu değişmiştir; her iki durumda "
                "da önceki ölçümler karşılaştırılabilir değildir."
            ),
        }
        (out_dir / "SEAL.json").write_text(json.dumps(seal, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Altın standart mühürlendi: %s", out_dir)
        return checksums

    def summary(self) -> dict[str, object]:
        by_level: dict[str, int] = {}
        for item in self.items:
            by_level[item.proto_level] = by_level.get(item.proto_level, 0) + 1
        witness_counts = [i.witness_count for i in self.items]
        return {
            "source": self.source,
            "source_ref": self.source_ref,
            "total": len(self.items),
            "splits": self.counts(),
            "concept_leakage": len(self.concept_leakage()),
            "proto_levels": by_level,
            "pt_ratio": round(by_level.get("PT", 0) / len(self.items), 3) if self.items else 0.0,
            "with_length_witness": sum(1 for i in self.items if i.has_length_witness),
            "mean_witnesses": round(sum(witness_counts) / len(witness_counts), 2) if witness_counts else 0.0,
        }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Altın standardı kur, böl ve mühürle")
    ap.add_argument("--dataset", default="savelyevturkic")
    ap.add_argument("--freeze", action="store_true", help="diske yaz ve mühürle")
    args = ap.parse_args()

    gold = GoldStandard.build(args.dataset)
    for key, value in gold.summary().items():
        print(f"{key:22} {value}")

    leakage = gold.concept_leakage()
    if leakage:
        print(f"\n!! SIZINTI: {len(leakage)} kavram birden çok bölümde: {leakage[:5]}")
        return 1
    print("\nKavram sızıntısı yok.")

    if args.freeze:
        checksums = gold.freeze()
        print(f"\nMühürlendi -> {GOLD_DIR}")
        for name, digest in checksums.items():
            print(f"  {name:6} {digest[:16]}…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
