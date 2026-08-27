"""
Alıntı geçiş zinciri — kelime hangi dillerden geçerek geldi?

Kullanıcının sorusu buydu: *"kelime Arapçadan, Yunancadan, Farsçadan geçtiyse
nasıl geçtiğini ve geçerken nasıl değiştiğini de çıkarabiliyor muyuz?"*

Zincir tek adımlı değildir. Ölçüldü — Türkçe ``sabun`` şu yoldan gelmiştir::

    Türkçe sabun  <-miras-  Osmanlıca صابون  <-alıntı-  Arapça صَابُون

Bu yüzden "Türkçedeki Arapça alıntı" saymak yanıltıcıdır: kaikki'nin
doğrudan ``bor`` etiketiyle Arapçaya bağladığı Türkçe kelime yalnız 48'dir,
çünkü **büyük çoğunluk Osmanlıca üzerinden gelir** ve ilk halka ``inh``
(miras) olarak etiketlenir.

İki katman:

:class:`ChainExtractor`
    Wiktionary'nin **sıralı** ``etymology_templates`` dizisinden zinciri
    okur. Bu bir *çıkarım* değil, okumadır — kaynak zaten söylüyor.
:class:`AdaptationRuleLearner`
    Her halkadaki biçim çiftinden (``صَابُون`` -> ``صابون`` -> ``sabun``)
    **uyarlama kurallarını indükler**: hangi ses hangi konumda neye dönüşüyor.

⚠️ Ayrım önemlidir ve raporda korunur: zinciri *okumak* yapılmış bir iştir
(EtymDB, Etymological Wordnet); zinciri *çıkarsamak* ve **her halkaya ayrı
kural indüklemek** yapılmamıştır. Bu modül okumayı yapar ve kural
indüksiyonunun eğitim verisini üretir.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from engine.config import PROJECT_ROOT
from engine.logging_setup import get_logger
from engine.nlp.cognate_prediction import MIN_SUPPORT, position_of
from engine.nlp.multi_alignment import GAP, align_forms
from engine.utils.orthography import to_comparison_form
from engine.utils.transliteration import transliterate_to_latin

logger = get_logger(__name__)

CHAINS_PATH = PROJECT_ROOT / "data" / "borrowing" / "chains.json"
RULES_PATH = PROJECT_ROOT / "data" / "borrowing" / "adaptation_rules.json"

#: Zincir halkası kuran şablonlar. Sıra önemlidir: ``etymology_templates``
#: en yakın kaynaktan en uzağa doğru sıralıdır.
CHAIN_TEMPLATES: dict[str, str] = {
    "inh": "miras",
    "inh+": "miras",
    "bor": "alıntı",
    "bor+": "alıntı",
    "ubor": "uyarlanmamış alıntı",
    "lbor": "öğrenilmiş alıntı",
    "slbor": "yarı öğrenilmiş alıntı",
    "obor": "orfografik alıntı",
    "der": "türev",
    "calque": "öyküntü",
    "psm": "fono-semantik eşleme",
}

#: Zincire girmeyen yardımcı şablonlar (biçimlendirme, sözlükçe bağlantısı…).
IGNORED_TEMPLATES = frozenset(
    {"yesno", "glossary", "doublet", "cog", "m", "l", "w", "q", "qualifier", "sense"}
)

#: Dil kodu -> Türkçe ad. Zincirin okunabilir yazılması için.
DONOR_LANGUAGE_NAMES: dict[str, str] = {
    "tr": "Türkçe",
    "ota": "Osmanlı Türkçesi",
    "ar": "Arapça",
    "fa": "Farsça",
    "fa-cls": "Klasik Farsça",
    "pal": "Orta Farsça",
    "grc": "Eski Yunanca",
    "el": "Yunanca",
    "la": "Latince",
    "it": "İtalyanca",
    "fr": "Fransızca",
    "en": "İngilizce",
    "de": "Almanca",
    "ru": "Rusça",
    "hy": "Ermenice",
    "ka": "Gürcüce",
    "he": "İbranice",
    "arc": "Aramice",
    "sa": "Sanskritçe",
    "zhx": "Çince (lehçe)",
    "zh": "Çince",
    "cmn": "Mandarin Çincesi",
    "mn": "Moğolca",
    "xng": "Orta Moğolca",
    "sog": "Soğdca",
    "otk": "Eski Türkçe",
    "trk-pro": "Ana Türkçe",
    "bg": "Bulgarca",
    "sr": "Sırpça",
    "ro": "Rumence",
    "hu": "Macarca",
    "ady": "Adigece",
    "ku": "Kürtçe",
    "az": "Azerbaycan Türkçesi",
    "es": "İspanyolca",
    "pt": "Portekizce",
    "nl": "Felemenkçe",
}


#: **Ünsüz yazıları (abjad).** Kısa ünlüleri yazmazlar; harekeler isteğe
#: bağlıdır ve kaynakların çoğunda yoktur.
#:
#: ⚠️ Bu, uyarlama kuralı indüksiyonunda ciddi bir yanlılık kaynağıdır:
#: ``صابون`` çeviriyazıda ``sabvn`` olur ve Türkçe ``sabun`` ile
#: hizalandığında sanki bir ünlü TÜRETİLMİŞ gibi görünür. Oysa ünlü zaten
#: oradaydı, yalnız YAZILMAMIŞTI. Bu tür kurallar ses değişimi değil,
#: **yazı sistemi artefaktıdır** ve ayrı işaretlenir.
ABJAD_LANGUAGES = frozenset({"ar", "fa", "fa-cls", "pal", "ota", "chg", "he", "arc", "ug"})


def language_name(code: str) -> str:
    return DONOR_LANGUAGE_NAMES.get(code, code)


@dataclass(frozen=True)
class ChainLink:
    """Zincirin tek bir halkası: bir dilden bir dile geçiş."""

    from_lang: str
    from_form: str
    to_lang: str
    to_form: str
    relation: str

    @property
    def from_latin(self) -> str:
        return to_comparison_form(transliterate_to_latin(self.from_form))

    @property
    def to_latin(self) -> str:
        return to_comparison_form(transliterate_to_latin(self.to_form))

    def describe(self) -> str:
        return (
            f"{language_name(self.to_lang)} {self.to_form} "
            f"←{self.relation}— {language_name(self.from_lang)} {self.from_form}"
        )


@dataclass
class BorrowingChain:
    """Bir kelimenin tam geçiş yolu."""

    word: str
    lang_code: str
    links: list[ChainLink] = field(default_factory=list)
    etymology_text: str = ""

    @property
    def depth(self) -> int:
        return len(self.links)

    @property
    def ultimate_origin(self) -> str:
        """Zincirin en uzak ucundaki dil — kelimenin nihai kaynağı."""
        return self.links[-1].from_lang if self.links else self.lang_code

    @property
    def is_borrowed(self) -> bool:
        """Zincirde en az bir **alıntı** halkası var mı?

        Yalnız ``miras`` halkalarından oluşan zincir alıntı değildir:
        Türkçe ← Osmanlıca ← Eski Türkçe bir miras zinciridir.
        """
        return any("alıntı" in link.relation or link.relation in ("türev", "öyküntü") for link in self.links)

    def path(self) -> list[str]:
        """``[Türkçe, Osmanlı Türkçesi, Arapça]`` biçiminde dil yolu."""
        if not self.links:
            return [language_name(self.lang_code)]
        return [language_name(self.lang_code)] + [
            language_name(link.from_lang) for link in self.links
        ]

    def describe(self) -> str:
        forms = [self.word] + [link.from_form for link in self.links]
        parts = [
            f"{language_name(lang)} {form}"
            for lang, form in zip(
                [self.lang_code] + [link.from_lang for link in self.links], forms, strict=True
            )
        ]
        return " ← ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "word": self.word,
            "lang_code": self.lang_code,
            "depth": self.depth,
            "is_borrowed": self.is_borrowed,
            "ultimate_origin": self.ultimate_origin,
            "path": self.path(),
            "chain": self.describe(),
            "links": [
                {
                    "from_lang": link.from_lang,
                    "from_form": link.from_form,
                    "to_lang": link.to_lang,
                    "to_form": link.to_form,
                    "relation": link.relation,
                }
                for link in self.links
            ],
        }


class ChainExtractor:
    """``etymology_templates`` dizisinden geçiş zincirini okur."""

    def extract(self, record: dict[str, Any]) -> BorrowingChain | None:
        word = str(record.get("word", "")).strip()
        lang_code = str(record.get("lang_code", "")).strip()
        if not word or not lang_code:
            return None

        chain = BorrowingChain(
            word=word,
            lang_code=lang_code,
            etymology_text=str(record.get("etymology_text", "")),
        )

        current_lang = lang_code
        current_form = word
        for template in record.get("etymology_templates", []) or []:
            name = str(template.get("name", "")).lower()
            if name in IGNORED_TEMPLATES or name not in CHAIN_TEMPLATES:
                continue
            args = template.get("args", {}) or {}
            donor_lang = str(args.get("2", "") or "").strip()
            donor_form = str(args.get("3", "") or "").strip()
            if not donor_lang or not donor_form:
                continue
            # ``inh`` ve ``inh+`` aynı halkayı iki kez verebilir.
            if chain.links and chain.links[-1].from_lang == donor_lang:
                continue
            chain.links.append(
                ChainLink(
                    from_lang=donor_lang,
                    from_form=donor_form,
                    to_lang=current_lang,
                    to_form=current_form,
                    relation=CHAIN_TEMPLATES[name],
                )
            )
            current_lang, current_form = donor_lang, donor_form

        return chain if chain.links else None


def iter_chains(lang_code: str = "tr") -> Iterator[BorrowingChain]:
    """İndirilmiş dökümden bütün zincirleri okur."""
    import gzip

    from engine.config import LEXICON_DIR

    path = next(iter(LEXICON_DIR.glob(f"{lang_code}.jsonl*")), None)
    if path is None:
        raise FileNotFoundError(
            f"{lang_code} dökümü yok. Önce indirin: "
            f"python scripts/download_lexicons.py"
        )
    extractor = ChainExtractor()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:  # type: ignore[operator]
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            chain = extractor.extract(record)
            if chain is not None:
                yield chain


# --- Uyarlama kuralı indüksiyonu -------------------------------------------


@dataclass
class AdaptationRule:
    """Bir halkada gözlenen ses uyarlaması."""

    from_lang: str
    to_lang: str
    position: str
    source_sound: str
    target_sound: str
    support: int

    @property
    def is_change(self) -> bool:
        return self.source_sound != self.target_sound

    @property
    def is_script_artifact(self) -> bool:
        """Bu kural ses değişimi mi, yazı sistemi artefaktı mı?

        Bir ünsüz yazısından (abjad) gelen ünlü ekleme/silme kuralları
        gerçek bir uyarlama değildir: ünlü zaten söyleniyordu, yalnız
        yazılmıyordu. Bunları "ses kanunu" diye raporlamak yanlış olur.
        """
        involves_gap = GAP in (self.source_sound, self.target_sound)
        touches_abjad = self.from_lang in ABJAD_LANGUAGES or self.to_lang in ABJAD_LANGUAGES
        vowel = (self.source_sound + self.target_sound).strip(GAP)
        return involves_gap and touches_abjad and vowel in set("aeıioöuüâîû")

    def describe(self) -> str:
        arrow = f"{self.source_sound or '∅'} → {self.target_sound or '∅'}"
        return (
            f"{language_name(self.from_lang)} → {language_name(self.to_lang)}, "
            f"{self.position}: {arrow} ({self.support} örnek)"
        )


class AdaptationRuleLearner:
    """Zincir halkalarındaki biçim çiftlerinden uyarlama kurallarını indükler.

    ⚠️ arXiv'de ``"loanword adaptation"`` sorgusu 0 makale döndürüyor ama bu
    bir arama artefaktıdır: **Tsvetkov, Ammar & Dyer (NAACL 2015)** tanıklanmış
    verici–alıntı çiftlerinden OT kısıt sıralaması öğreniyor. Buradaki katkı
    "ilk kez kural indüksiyonu" değil, **çok halkalı zincirde her halkaya
    ayrı kural** indüklemektir.
    """

    def __init__(self) -> None:
        #: ``(kaynak_dil, hedef_dil, konum, kaynak_ses) -> {hedef_ses: sayım}``
        self.counts: dict[tuple[str, str, str, str], Counter] = defaultdict(Counter)
        self.pair_counts: Counter = Counter()

    def observe_link(self, link: ChainLink) -> bool:
        """Bir halkadan kural gözlemi çıkarır. Hizalanamazsa ``False``."""
        source = link.from_latin
        target = link.to_latin
        if len(source) < 2 or len(target) < 2:
            return False

        columns = align_forms({"src": source, "tgt": target})
        if not columns:
            return False

        width = len(columns)
        for index, column in enumerate(columns):
            source_sound = column.sounds.get("src", GAP)
            target_sound = column.sounds.get("tgt", GAP)
            if source_sound == GAP and target_sound == GAP:
                continue
            position = position_of(index, width)
            key = (link.from_lang, link.to_lang, position, source_sound)
            self.counts[key][target_sound] += 1
        self.pair_counts[(link.from_lang, link.to_lang)] += 1
        return True

    def rules(self, *, min_support: int = MIN_SUPPORT) -> list[AdaptationRule]:
        """Yeterince desteklenen kuralları döndürür."""
        out: list[AdaptationRule] = []
        for (from_lang, to_lang, position, source_sound), targets in self.counts.items():
            for target_sound, support in targets.items():
                if support < min_support:
                    continue
                out.append(
                    AdaptationRule(
                        from_lang=from_lang,
                        to_lang=to_lang,
                        position=position,
                        source_sound=source_sound,
                        target_sound=target_sound,
                        support=support,
                    )
                )
        return sorted(out, key=lambda r: (-r.support, r.from_lang, r.to_lang, r.position))

    def regularity(self, from_lang: str, to_lang: str) -> float:
        """Bir dil çiftinin uyarlamasının **düzenlilik oranı**.

        Her (konum, kaynak ses) için en sık hedef sesin payı. Yüksek oran,
        uyarlamanın kurallı olduğunu gösterir; düşük oran ya veri gürültüsü
        ya da uyarlamanın gerçekten düzensiz olduğu anlamına gelir
        (Mao & Hulden 2016, Japoncada "alıntı fonolojisinin iç tutarsızlığı").
        """
        relevant = [
            targets
            for (source, target, _, _), targets in self.counts.items()
            if source == from_lang and target == to_lang
        ]
        if not relevant:
            return 0.0
        total = sum(sum(t.values()) for t in relevant)
        dominant = sum(max(t.values()) for t in relevant)
        return dominant / total if total else 0.0

    def as_dict(self, *, min_support: int = MIN_SUPPORT) -> dict[str, Any]:
        pairs = sorted(self.pair_counts.items(), key=lambda kv: -kv[1])
        return {
            "_schema": "turkic-etymology-adaptation-rules/v1",
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "min_support": min_support,
            "note": (
                "Kurallar tanıklanmış alıntı çiftlerinden İNDÜKLENMİŞTİR, elle "
                "yazılmamıştır. Karşılaştırma noktası: Tsvetkov, Ammar & Dyer "
                "(NAACL 2015), Arapça→Svahili OT kısıt sıralaması öğrenimi."
            ),
            "language_pairs": [
                {
                    "from": from_lang,
                    "to": to_lang,
                    "examples": count,
                    "regularity": round(self.regularity(from_lang, to_lang), 4),
                }
                for (from_lang, to_lang), count in pairs
            ],
            "rules": [
                {
                    "from": rule.from_lang,
                    "to": rule.to_lang,
                    "position": rule.position,
                    "source": rule.source_sound,
                    "target": rule.target_sound,
                    "support": rule.support,
                    "script_artifact": rule.is_script_artifact,
                }
                for rule in self.rules(min_support=min_support)
            ],
        }


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Alıntı geçiş zinciri ve uyarlama kuralları")
    ap.add_argument("--lang", default="tr")
    ap.add_argument("--min-depth", type=int, default=1)
    ap.add_argument("--show", type=int, default=12, help="kaç örnek zincir göster")
    ap.add_argument("--save", action="store_true", help="zincirleri ve kuralları yaz")
    args = ap.parse_args()

    chains = [c for c in iter_chains(args.lang) if c.depth >= args.min_depth]
    borrowed = [c for c in chains if c.is_borrowed]
    deep = [c for c in chains if c.depth >= 2]

    print(f"\n=== {args.lang} geçiş zincirleri ===")
    print(f"zincir taşıyan kayıt : {len(chains):,}")
    print(f"alıntı zinciri       : {len(borrowed):,}")
    print(f"2+ halkalı zincir    : {len(deep):,}")

    depths = Counter(c.depth for c in chains)
    print("\nhalka sayısına göre:")
    for depth in sorted(depths):
        print(f"  {depth} halka: {depths[depth]:,}")

    origins = Counter(c.ultimate_origin for c in borrowed)
    print("\nnihai kaynak dile göre (en sık 12):")
    for code, count in origins.most_common(12):
        print(f"  {language_name(code):22} {count:>6,}")

    print("\nörnek çok halkalı zincirler:")
    for chain in deep[: args.show]:
        print(f"  {chain.describe()}")

    learner = AdaptationRuleLearner()
    aligned = 0
    for chain in chains:
        for link in chain.links:
            aligned += learner.observe_link(link)
    rules = learner.rules()
    print(f"\nuyarlama kuralı indüksiyonu: {aligned:,} halka hizalandı, {len(rules):,} kural")
    genuine = [r for r in rules if r.is_change and not r.is_script_artifact]
    artifacts = [r for r in rules if r.is_change and r.is_script_artifact]
    print(f"  bunlardan {len(artifacts):,} tanesi YAZI ARTEFAKTI (abjad ünlü yazmıyor)")
    print("\nen çok desteklenen 12 GERÇEK ses uyarlaması:")
    for rule in genuine[:12]:
        print(f"  {rule.describe()}")

    print("\ndil çiftlerinin düzenlilik oranı (en çok örnekli 8):")
    for pair, count in learner.pair_counts.most_common(8):
        regularity = learner.regularity(*pair)
        print(
            f"  {language_name(pair[0]):20} → {language_name(pair[1]):20} "
            f"n={count:>5,} düzenlilik={regularity:.3f}"
        )

    if args.save:
        CHAINS_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHAINS_PATH.write_text(
            json.dumps(
                {
                    "_schema": "turkic-etymology-chains/v1",
                    "language": args.lang,
                    "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "n_chains": len(chains),
                    "n_borrowed": len(borrowed),
                    "n_multi_step": len(deep),
                    "chains": [c.as_dict() for c in chains],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        RULES_PATH.write_text(
            json.dumps(learner.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\nyazıldı: {CHAINS_PATH}\n         {RULES_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
