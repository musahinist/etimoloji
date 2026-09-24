import re
from typing import Any

from engine.fetchers.base import TURKIC_LANGUAGES_MAP, BaseFetcher
from engine.logging_setup import get_logger
from engine.utils.seed import load_seed_entries, seed_source_label

logger = get_logger(__name__)

# Dahili Yıldız/Starling Proto-Türkçe Etimoloji Sözlük Dizini (Core Turkic Lexicon Index)
#: Tohum (seed) veri. Kod içinde değil, data/seed/lexicon/starling.json dosyasında tutulur.
#: ⚠️ Yalnız gerçek Starling veritabanı (`make starling`) indirilmemişse kullanılır.
SEED_PATH = "lexicon/starling.json"
STARLING_OFFLINE_LEXICON = load_seed_entries(SEED_PATH)

#: Kök alanındaki ilk yıldızlı biçim: "*göŕ ( = *gör-s) / *gör-" -> "*göŕ".
_FIRST_PROTO = re.compile(r"\*[^\s,/();]+")


#: Adayın sorgunun bir anlamını "taşıdığı" en düşük benzerlik. Arama
#: motorunun eşsesli süzgeciyle aynı ölçüt (``HOMONYM_SIMILARITY_FLOOR``,
#: paraphrase-multilingual-MiniLM); orada ölçüldü: eşsesliler 0,13-0,27,
#: gerçek anlamlar 0,385+.
MEANING_FLOOR = 0.30

#: En iyi aday, ikinciyi bu kadar geçmiyorsa seçim yapılmaz.
MEANING_MARGIN = 0.15


def _database_available() -> bool:
    """Veritabanı okunabiliyor mu? Yarım inmiş ``.dbf`` ``struct.error``
    verir; bu, ``SearchEngine()`` kurulumunu düşürmemeli."""
    try:
        from engine.db.starling import load_turcet

        return bool(load_turcet())
    except Exception:
        logger.warning("Starling veritabanı okunamadı; tohum veriye dönülüyor", exc_info=True)
        return False


def _query_glosses(word: str) -> list[str]:
    """Sorgunun yerel sözlük indeksindeki Türkçe/Osmanlıca anlamları
    (çekim ve yönlendirme satırları hariç), indeks sırasıyla."""
    from engine.db.lexicon_index import LexiconIndex
    from engine.utils.morphology import is_inflection_gloss

    index = LexiconIndex()
    if not index.exists:
        return []
    glosses: list[str] = []
    for row in index.lookup(word, languages=["tr", "ota"], limit=40):
        gloss = str(row.get("gloss") or "").strip()
        if (
            str(row.get("word") or "").strip().lower() == word and gloss
            and not is_inflection_gloss(gloss) and not re.search(r"\b(?:form|spelling) of\b", gloss, re.I)
            and gloss not in glosses
        ):
            glosses.append(gloss)
    return glosses[:12]


def _meaning_scores(candidates: list[str], glosses: list[str]) -> list[float] | None:
    """Her aday anlamın sorgu anlamlarından en yakınına kosinüs benzerliği;
    model yoksa ``None`` (n-gram yedeği anlam ayırmaz)."""
    from engine.nlp.diachronic_semantic_engine import get_sentence_transformer, has_semantic_model

    if not has_semantic_model():
        return None
    from sentence_transformers.util import cos_sim

    model = get_sentence_transformer()
    # Starling çok anlamı numaralar: "1 to slide 2 to swim 3 skis".
    texts = [re.sub(r"(?:^|\s)\d+\s", "; ", c).strip("; ") or c for c in candidates]
    sims = cos_sim(model.encode(texts, show_progress_bar=False), model.encode(glosses, show_progress_bar=False))
    return [float(v) for v in sims.max(dim=1).values.tolist()]


def select_by_meaning(word: str, matches: tuple[Any, ...]) -> tuple[Any | None, dict[int, float]]:
    """Eşsesli Starling adaylarından sorgunun anlamına uyanı seçer.

    ``(seçilen ya da None, {kök no: benzerlik})``. Tek aday olduğu gibi
    döner. Birden çok adayda seçim ancak TEK aday sorgunun bir anlamını
    taşıyorsa ya da en iyisi ikinciyi açıkça geçiyorsa yapılır. `el`
    ("hand" ve "country") ya da `yüz` ("face" ve "hundred") gibi sorgunun
    kendisi eşsesliyse ya da anlam/model yoksa ``None``: ilk aday kesin kök
    gibi sunulmaz. Eskiden hep ilk aday alınıyordu (`kaymak` -> *KAj- "to
    turn back").
    """
    if len(matches) == 1:
        return matches[0], {}
    glosses = _query_glosses(word)
    if not glosses:
        return None, {}
    try:
        values = _meaning_scores([m.meaning for m in matches], glosses)
    except Exception:
        logger.warning("Starling anlam seçimi çalışmadı", exc_info=True)
        values = None
    if values is None:
        return None, {}
    scores = {m.number: round(v, 3) for m, v in zip(matches, values, strict=True)}
    ranked = sorted(zip(values, range(len(matches)), strict=True), reverse=True)
    (best, best_i), (second, _) = ranked[0], ranked[1]
    if best >= MEANING_FLOOR and (second < MEANING_FLOOR or best - second >= MEANING_MARGIN):
        return matches[best_i], scores
    return None, scores


class StarlingFetcher(BaseFetcher):
    def __init__(self, *, use_database: bool | None = None) -> None:
        """:param use_database: ``None`` = veritabanı indirilmişse onu kullan;
        ``False`` = tohum veri (testler makinedeki veriden bağımsız kalsın)."""
        #: Gerçek veritabanı varsa tohum veriye hiç bakılmaz.
        self.has_database = _database_available() if use_database is None else use_database
        #: Tohum veri yerel ve elle yazılmıştır, canlı bir servis DEĞİLDİR.
        self.is_seed_source = not self.has_database
        #: İndirilmiş veritabanı yereldir, canlı değildir.
        self.is_local = self.has_database

    @property
    def source_name(self) -> str:
        if self.has_database:
            return "Starling Türk Etimoloji Veritabanı (Dybo & Starostin 2005, yerel)"
        return seed_source_label("Starling Etymological Database", SEED_PATH)

    def fetch(self, word: str) -> dict[str, Any]:
        word_clean = (word or "").strip().lower()
        try:
            if self.has_database:
                return self._fetch_database(word_clean)
            return self._fetch_seed(word_clean)
        except Exception:
            # Sözleşme: fetch() istisna atmaz (yarım inmiş .dbf -> struct.error).
            logger.warning("%s: kaynak işlenemedi", self.source_name, exc_info=True)
            return self.empty_result()

    def _fetch_database(self, word: str) -> dict[str, Any]:
        """Kök, anlam ve tarihli ilk tanıklama; TANIK ÜRETMEZ.

        Starling tanıkları kendi transkripsiyonundadır (`qɨrq`, `mončuq`);
        sözlük indeksindeki Kiril/Latin biçimlerle yan yana tanık sayılırsa
        aynı dil iki kez görünür. Tanık olarak kullanımı ayrıca ölçülmeli.
        """
        from engine.db.starling import lookup_turkish

        result = self.empty_result()
        matches = lookup_turkish(word)
        if not matches:
            return result
        etym, scores = select_by_meaning(word, matches)
        if etym is None:
            # Eşsesli kökler anlamla ayrılamadı: hiçbiri başlık kökü ya da
            # ilk tanıklama olarak sunulmaz, adaylar yalnız notta durur.
            result["root"]["reconstruction_notes"] = (
                f"Starling'de {len(matches)} aday kök; sorgunun anlamı tek köke "
                "bağlanamadı (eşsesli): "
                + "; ".join(f"#{m.number} {m.proto} “{m.meaning}”" for m in matches)
            )
            result["root"]["starling_candidates"] = [
                {"number": m.number, "proto": m.proto, "meaning": m.meaning,
                 **({"meaning_similarity": scores[m.number]} if m.number in scores else {})}
                for m in matches
            ]
            return result
        proto = _FIRST_PROTO.search(etym.proto)
        result["root"]["proto_turkic"] = proto.group(0) if proto else etym.proto
        result["root"]["starling_proto"] = etym.proto
        result["root"]["meaning"] = etym.meaning
        old_turkic = etym.reflexes.get("ATU", "")
        result["root"]["reconstruction_notes"] = (
            f"Starling #{etym.number}: {etym.proto} “{etym.meaning}”"
            + (f"; Eski Türkçe: {old_turkic}" if old_turkic else "")
            + (f"; kaynakça: {etym.reference}" if etym.reference else "")
            + (
                "; anlamca seçildi, öbür aday(lar): "
                + ", ".join(f"#{m.number} {m.proto} “{m.meaning}”" for m in matches if m is not etym)
                if len(matches) > 1 else ""
            )
        )
        dated = etym.earliest_dated_source()
        if dated:
            year, tag = dated
            result["first_attestation"] = {
                "form": old_turkic.split("(")[0].strip() or etym.reflexes.get("KRH", ""),
                "meaning": etym.meaning,
                "source": f"{tag} (Starling #{etym.number})",
                "year": year,
            }
        return result

    def _fetch_seed(self, word_clean: str) -> dict[str, Any]:
        result = {
            "root": {
                "proto_turkic": "",
                "meaning": "",
                "reconstruction_notes": ""
            },
            "turkic_languages": []
        }

        if word_clean in STARLING_OFFLINE_LEXICON:
            entry = STARLING_OFFLINE_LEXICON[word_clean]
            result["root"]["proto_turkic"] = entry["proto_turkic"]
            result["root"]["meaning"] = entry["meaning"]
            result["root"]["reconstruction_notes"] = f"Starling / Tower of Babel Proto-Turkic reconstruction {entry['proto_turkic']}"

            for lang_code, cognate in entry["cognates"].items():
                if lang_code in TURKIC_LANGUAGES_MAP:
                    display_word = cognate["word"]
                    result["turkic_languages"].append({
                        "lang_code": lang_code,
                        "lang_name": TURKIC_LANGUAGES_MAP[lang_code],
                        "word": display_word,
                        "meaning": cognate["meaning"],
                        "script": "Cyrillic" if re.search(r'[\u0400-\u04FF]', display_word) else ("Arabic" if re.search(r'[\u0600-\u06FF]', display_word) else "Latin")
                    })

        return result
