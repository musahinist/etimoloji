"""
Semantik makullük — "bu anlam kayması olur mu?" sorusunu veriye bağlar.

Etimolojik bir iddia iki ayak üzerinde durur: **ses** ve **anlam**. Ses
tarafı ölçülüyor (denklikler, refleksler); anlam tarafı bugüne kadar
sezgiseldi. Bu modül onu veriye dayandırır.

İki kaynak:

**CLICS³ — eş-adlandırma (colexification) ağı.**
    Dünya dillerinde aynı kelimeyle ifade edilen kavram çiftleri. ``ağaç``
    ile ``odun``un yüzlerce dilde tek kelime olması, aralarındaki kaymanın
    **doğal** olduğunu gösterir. ``ağaç`` ile ``korku`` arasında böyle bir
    kanıt yoktur.

**Concepticon — kavram ilişkileri.**
    Üst/alt kavram (``köpek`` ⊂ ``hayvan``), parça-bütün, karşıtlık.

Ek olarak bir **yön kuralı** uygulanır:

    Somuttan soyuta doğru kayma, dillerin **%90'ında** ters yönden daha sık
    görülür (Xu ve ark. 2023). ``el`` → ``yardım`` beklenen yöndür;
    ``yardım`` → ``el`` ekstra kanıt ister.

⚠️ **Diakronik word embedding kullanılmaz.** Dubossarsky ve ark. bunların
büyük ölçüde model artefaktı olduğunu gösterdi; Türki diller için
tarihlendirilmiş derlem de yoktur. Sezgiyi sayıya çevirmek, sezgiden kötüdür.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from engine.config import PROJECT_ROOT
from engine.logging_setup import get_logger

logger = get_logger(__name__)

SEMANTIC_DIR = PROJECT_ROOT / "data" / "semantic"
COLEXIFICATION_PATH = SEMANTIC_DIR / "colexifications.json"
CONCEPT_RELATIONS_PATH = SEMANTIC_DIR / "conceptrelations.tsv"

#: Somutluk göstergeleri. Tam bir somutluk ölçeği yok; Türkçe için elle
#: derlenmiş bu kaba sınıflandırma, yön kuralının uygulanabilmesi içindir.
#:
#: ⚠️ Kaba olduğu için yön kuralı **tek başına karar vermez**, yalnız
#: eş-adlandırma kanıtının yanında bir ağırlık taşır.
CONCRETE_MARKERS = frozenset(
    {
        "el", "ayak", "baş", "göz", "kulak", "burun", "ağız", "diş", "dil",
        "su", "ateş", "taş", "ağaç", "yaprak", "kök", "toprak", "gök", "güneş",
        "ay", "yıldız", "dağ", "deniz", "yol", "ev", "kapı", "at", "köpek",
        "kuş", "balık", "et", "kan", "kemik", "deri", "tüy", "boynuz",
    }
)

ABSTRACT_MARKERS = frozenset(
    {
        "sevgi", "korku", "umut", "akıl", "düşünce", "bilgi", "inanç", "hak",
        "adalet", "zaman", "yardım", "güç", "kuvvet", "onur", "utanç", "sabır",
        "özgürlük", "gerçek", "yalan", "iyilik", "kötülük", "anlam", "amaç",
    }
)


@dataclass
class SemanticVerdict:
    """Bir anlam kaymasının makullük değerlendirmesi."""

    source: str
    target: str
    score: float
    evidence: list[str] = field(default_factory=list)
    against: list[str] = field(default_factory=list)
    colexification_count: int = 0
    direction: str = ""

    @property
    def is_plausible(self) -> bool:
        return self.score >= 0.4

    @property
    def verdict(self) -> str:
        if self.score >= 0.6:
            return "makul"
        if self.score >= 0.4:
            return "olabilir"
        if self.score > 0.0:
            return "zayıf"
        return "kanıt yok"

    def explain(self) -> str:
        lines = [f"{self.source} → {self.target}: {self.verdict.upper()} ({self.score:.2f})"]
        lines += [f"  + {e}" for e in self.evidence]
        lines += [f"  − {a}" for a in self.against]
        if not self.evidence and not self.against:
            lines.append("  · bu kavram çifti için veri bulunamadı")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "score": round(self.score, 3),
            "verdict": self.verdict,
            "is_plausible": self.is_plausible,
            "colexification_count": self.colexification_count,
            "direction": self.direction,
            "evidence": self.evidence,
            "against": self.against,
            "explanation": self.explain(),
        }


@lru_cache(maxsize=1)
def load_colexifications() -> dict[str, dict[str, int]]:
    """CLICS³ eş-adlandırma sayıları: ``kavram -> {kavram: kaç dilde}``."""
    if not COLEXIFICATION_PATH.exists():
        logger.info(
            "Eş-adlandırma verisi yok (%s). "
            "Kurmak için: python -m engine.nlp.semantic_plausibility --build",
            COLEXIFICATION_PATH,
        )
        return {}
    try:
        data = json.loads(COLEXIFICATION_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Eş-adlandırma verisi okunamadı", exc_info=True)
        return {}
    return {k: dict(v) for k, v in data.get("colexifications", {}).items()}


@lru_cache(maxsize=1)
def load_concept_relations() -> dict[tuple[str, str], str]:
    """Concepticon kavram ilişkileri: ``(a, b) -> ilişki türü``."""
    if not CONCEPT_RELATIONS_PATH.exists():
        return {}
    relations: dict[tuple[str, str], str] = {}
    with CONCEPT_RELATIONS_PATH.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            source = (row.get("SOURCE_GLOSS") or "").strip().lower()
            target = (row.get("TARGET_GLOSS") or "").strip().lower()
            relation = (row.get("RELATION") or "").strip()
            if source and target and relation:
                relations[(source, target)] = relation
    return relations


def reset_caches() -> None:
    load_colexifications.cache_clear()
    load_concept_relations.cache_clear()
    load_turkish_bridge.cache_clear()


#: Türkçe kelime -> Concepticon etiketi köprüsü.
#:
#: ⚠️ Concepticon'da **Türkçe kavram listesi yoktur** (13 listenin hiçbiri
#: Türkçe kaynaklı değil). Köprü bu yüzden kendi verimizden türetilir:
#: ``savelyevturkic``ta her kavramın hem Türkçe biçmi hem Concepticon
#: etiketi vardır, ikisi eşleştirilir.
#:
#: Kapsam Swadesh düzeyiyle sınırlıdır (~250 kavram). Bu listede olmayan
#: bir Türkçe kelime için eş-adlandırma kanıtı **aranamaz** ve modül bunu
#: "veri yok" diye söyler — uydurmaz.
TURKISH_CONCEPT_PATH = SEMANTIC_DIR / "turkish_concepts.json"


#: Veriden türetilen köprüyü tamamlayan **elle eklenmiş** kavramlar.
#:
#: ``savelyevturkic``ın 254 kavramı Swadesh listesidir ve bazı çok temel
#: Türkçe kelimeler orada Türkçe biçimle temsil edilmez (kavram var ama
#: Türkçe sütunu boş veya farklı bir kelime). Bu ek onları kapatır.
MANUAL_BRIDGE: dict[str, str] = {
    "ağaç": "tree",
    "ateş": "fire",
    "baş": "head",
    "yol": "road",
    "yıl": "year",
    "söz": "word",
    "iş": "work",
    "taş": "stone",
    "odun": "wood",
    "gün": "day",
    "yer": "earth (soil)",
    "gök": "sky",
    "deniz": "sea",
    "dağ": "mountain",
    "kuş": "bird",
    "balık": "fish",
    "et": "meat",
    "kemik": "bone",
    "deri": "skin",
    "diş": "tooth",
    "kulak": "ear",
    "burun": "nose",
    "ağız": "mouth",
    "boyun": "neck",
    "kalp": "heart",
    "yürek": "heart",
    "ev": "house",
    "kapı": "door",
    "yaprak": "leaf",
    "kök": "root",
    "tuz": "salt",
    "yıldız": "star",
    "ay": "moon",
    "güneş": "sun",
    "gece": "night",
    "ad": "name",
    "yaş": "age",
    "can": "soul",
    "dil (konuşma)": "language",
}


@lru_cache(maxsize=1)
def load_turkish_bridge() -> dict[str, str]:
    """``Türkçe kelime -> CONCEPTICON ETİKETİ``."""
    if not TURKISH_CONCEPT_PATH.exists():
        return dict(MANUAL_BRIDGE)
    try:
        data = json.loads(TURKISH_CONCEPT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("Türkçe kavram köprüsü okunamadı", exc_info=True)
        return {}
    bridge = {k.lower(): v.lower() for k, v in data.get("bridge", {}).items()}
    # Elle eklenenler veriden geleni EZMEZ; yalnız boşluk doldurur.
    for term, gloss in MANUAL_BRIDGE.items():
        bridge.setdefault(term, gloss)
    return bridge


def to_concepticon(term: str) -> str:
    """Türkçe kelimeyi Concepticon etiketine çevirir; bulamazsa olduğu gibi.

    ⚠️ **Eşadlılık sınırı.** Köprü yazılışa göre çalışır, anlama göre değil.
    Türkçe ``yüz`` üç ayrı kelimedir (yüz = surat / yüzmek / 100) ve köprü
    yalnız birini seçebilir. Bu yüzden semantik skor **tek başına karar
    vermez**, yalnız bir kanıt kalemidir.
    """
    text = (term or "").strip().lower()
    return load_turkish_bridge().get(text, text)


def build_turkish_bridge(dataset: str = "savelyevturkic") -> dict[str, str]:
    """CLDF verisinden Türkçe -> Concepticon köprüsünü kurar."""
    from engine.db.cldf_wordlist import CldfWordlist

    wordlist = CldfWordlist.load(dataset)
    bridge: dict[str, str] = {}
    for cognate_set in wordlist.cognate_sets():
        gloss = (cognate_set.concepticon_gloss or "").strip().lower()
        if not gloss:
            continue
        for entry in cognate_set.entries:
            if entry.language != "Turkish":
                continue
            form = (entry.transliteration or entry.form).strip().lower()
            if form and form not in bridge:
                bridge[form] = gloss
    return bridge


def concreteness(concept: str) -> str:
    """``somut`` | ``soyut`` | ``bilinmiyor``."""
    text = (concept or "").strip().lower()
    if not text:
        return "bilinmiyor"
    head = text.split()[0].split(",")[0]
    if head in CONCRETE_MARKERS:
        return "somut"
    if head in ABSTRACT_MARKERS:
        return "soyut"
    return "bilinmiyor"


class SemanticPlausibility:
    """Anlam kaymalarını eş-adlandırma verisine ve yön kuralına göre puanlar."""

    def __init__(
        self,
        colexifications: dict[str, dict[str, int]] | None = None,
        relations: dict[tuple[str, str], str] | None = None,
    ):
        self._colex = colexifications
        self._relations = relations

    @property
    def colexifications(self) -> dict[str, dict[str, int]]:
        if self._colex is None:
            self._colex = load_colexifications()
        return self._colex

    @property
    def relations(self) -> dict[tuple[str, str], str]:
        if self._relations is None:
            self._relations = load_concept_relations()
        return self._relations

    def assess(self, source: str, target: str) -> SemanticVerdict:
        """Bir anlam kaymasının makullüğü."""
        # Türkçe girdiler Concepticon etiketine çevrilir; CLICS verisi
        # o etiketler üzerinden indekslenmiştir.
        #
        # ⚠️ Somutluk ÇEVİRİDEN ÖNCEKİ Türkçe kelimeye bakar: somutluk
        # tablosu Türkçe derlenmiştir ve ``el -> "arm or hand"`` çevirisi
        # yapıldıktan sonra eşleşme kaybolur.
        raw_source = (source or "").strip().lower()
        raw_target = (target or "").strip().lower()
        a = to_concepticon(source)
        b = to_concepticon(target)
        verdict = SemanticVerdict(source=a, target=b, score=0.0)
        if not a or not b:
            return verdict
        if a == b:
            verdict.score = 1.0
            verdict.evidence.append("anlam değişmemiş")
            return verdict

        score = 0.0

        # 1 — Eş-adlandırma: en güçlü kanıt, çünkü tanıklanmış dil verisidir.
        count = self.colexifications.get(a, {}).get(b, 0) or self.colexifications.get(b, {}).get(a, 0)
        verdict.colexification_count = count
        if count > 0:
            strength = min(1.0, count / 20.0)
            score += 0.60 * strength
            verdict.evidence.append(
                f"{count} dilde aynı kelimeyle ifade ediliyor (CLICS eş-adlandırma)"
            )
        else:
            verdict.against.append("hiçbir dilde eş-adlandırma tanıklanmamış")

        # 2 — Concepticon kavram ilişkisi.
        relation = self.relations.get((a, b)) or self.relations.get((b, a))
        if relation:
            score += 0.25
            verdict.evidence.append(f"Concepticon ilişkisi: {relation}")

        # 3 — Yön kuralı (Xu ve ark. 2023).
        source_kind = concreteness(raw_source) or concreteness(a)
        target_kind = concreteness(raw_target) or concreteness(b)
        if source_kind == "bilinmiyor":
            source_kind = concreteness(a)
        if target_kind == "bilinmiyor":
            target_kind = concreteness(b)
        if source_kind == "somut" and target_kind == "soyut":
            score += 0.15
            verdict.direction = "somut→soyut"
            verdict.evidence.append(
                "somuttan soyuta kayma — dillerin %90'ında beklenen yön (Xu ve ark. 2023)"
            )
        elif source_kind == "soyut" and target_kind == "somut":
            score -= 0.15
            verdict.direction = "soyut→somut"
            verdict.against.append(
                "soyuttan somuta kayma — beklenenin tersi, ekstra kanıt gerektirir"
            )
        else:
            verdict.direction = f"{source_kind}→{target_kind}"

        verdict.score = max(0.0, min(1.0, score))
        return verdict


# --- Veri kurulumu ----------------------------------------------------------


def build_from_clics(
    network_path: Path, *, min_families: int = 3
) -> dict[str, dict[str, int]]:
    """CLICS³ GML ağından eş-adlandırma sayılarını çıkarır.

    ⚠️ Dosya 70 MB ve satırların çoğu devasa ``words``/``wofam`` alanlarıdır
    (tek satır megabaytlarca olabilir). Bu yüzden tüm metni belleğe alıp
    regex çalıştırmak yerine **satır satır** okunur ve yalnız ihtiyaç
    duyulan alanlar tutulur.

    :param min_families: bir çiftin sayılması için gereken asgari **dil
        ailesi** sayısı. Dil sayısı değil aile sayısı kullanılır: aynı
        ailenin on dilinde görülen bir eş-adlandırma tek bir olay olabilir
        (ortak miras), farklı ailelerde görülmesi ise bağımsız kanıttır.
    """
    colex: dict[str, dict[str, int]] = {}
    node_labels: dict[str, str] = {}

    in_node = in_edge = False
    node_id = gloss = ""
    source = target = ""
    weight = 0

    with network_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped == "node [":
                in_node, node_id, gloss = True, "", ""
                continue
            if stripped == "edge [":
                in_edge, source, target, weight = True, "", "", 0
                continue
            if stripped == "]":
                if in_node and node_id and gloss:
                    node_labels[node_id] = gloss
                elif in_edge and source and target and weight >= min_families:
                    a, b = node_labels.get(source), node_labels.get(target)
                    if a and b and a != b:
                        colex.setdefault(a, {})[b] = weight
                        colex.setdefault(b, {})[a] = weight
                in_node = in_edge = False
                continue

            if in_node:
                if stripped.startswith("id "):
                    node_id = stripped[3:].strip().strip('"')
                elif stripped.startswith('Gloss "'):
                    gloss = stripped[7:].rstrip('"').strip().lower()
            elif in_edge:
                if stripped.startswith("source "):
                    source = stripped[7:].strip().strip('"')
                elif stripped.startswith("target "):
                    target = stripped[7:].strip().strip('"')
                elif stripped.startswith("FamilyWeight "):
                    try:
                        weight = int(float(stripped[13:].strip()))
                    except ValueError:
                        weight = 0
    return colex


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Semantik makullük")
    ap.add_argument("--build", type=Path, help="CLICS³ GML dosyasından veri kur")
    ap.add_argument("--build-bridge", action="store_true", help="Türkçe kavram köprüsünü kur")
    ap.add_argument("--pair", nargs=2, metavar=("KAYNAK", "HEDEF"))
    args = ap.parse_args()

    if args.build:
        colex = build_from_clics(args.build)
        SEMANTIC_DIR.mkdir(parents=True, exist_ok=True)
        COLEXIFICATION_PATH.write_text(
            json.dumps(
                {
                    "_schema": "turkic-etymology-colexification/v1",
                    "source": "CLICS³",
                    "n_concepts": len(colex),
                    "colexifications": colex,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        reset_caches()
        print(f"{len(colex)} kavram için eş-adlandırma yazıldı: {COLEXIFICATION_PATH}")
        return 0

    if args.build_bridge:
        bridge = build_turkish_bridge()
        SEMANTIC_DIR.mkdir(parents=True, exist_ok=True)
        TURKISH_CONCEPT_PATH.write_text(
            json.dumps(
                {
                    "_schema": "turkic-etymology-concept-bridge/v1",
                    "source": "savelyevturkic (Türkçe biçim ↔ Concepticon etiketi)",
                    "n": len(bridge),
                    "note": (
                        "Concepticon'da Türkçe kavram listesi yoktur; köprü kendi "
                        "verimizden türetilmiştir. Kapsam Swadesh düzeyiyle sınırlı."
                    ),
                    "bridge": bridge,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        reset_caches()
        print(f"{len(bridge)} Türkçe kavram köprüsü yazıldı: {TURKISH_CONCEPT_PATH}")
        return 0

    engine = SemanticPlausibility()
    if not engine.colexifications:
        print(
            "⚠️ Eş-adlandırma verisi yok; yalnız yön kuralı ve Concepticon\n"
            "   ilişkileri kullanılacak. Kurmak için CLICS³ ağını indirip\n"
            "   --build ile geçin."
        )

    pairs = [tuple(args.pair)] if args.pair else [
        ("ağaç", "odun"),
        ("el", "yardım"),
        ("göz", "kaynak"),
        ("dil", "konuşma"),
        ("ağaç", "korku"),
        ("yardım", "el"),
    ]
    for source, target in pairs:
        print(engine.assess(source, target).explain())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
