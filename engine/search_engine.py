import concurrent.futures
import re
import time
from functools import lru_cache
from typing import Any

from engine import config
from engine.db.database import DatabaseManager
from engine.db.graph_database import GraphDatabaseManager
from engine.fetchers.academic_turkology import AcademicTurkologyFetcher
from engine.fetchers.apertium import ApertiumFetcher
from engine.fetchers.archive_org import ArchiveOrgFetcher
from engine.fetchers.base import TURKIC_LANGUAGES_MAP, BaseFetcher
from engine.fetchers.etimoloji_turkce import EtimolojiTurkceFetcher
from engine.fetchers.historical_index import HistoricalIndexFetcher, ModernIndexFetcher
from engine.fetchers.historical_modern import HistoricalModernLexiconFetcher
from engine.fetchers.isam_ansiklopedi import IsamAnsiklopediFetcher
from engine.fetchers.loanword_donor_etymology import LoanwordDonorEtymologyFetcher
from engine.fetchers.local_pdf_books import LocalPdfBooksFetcher
from engine.fetchers.multilang_wiktionary import MultiLangWiktionaryFetcher
from engine.fetchers.northeuralex import NorthEuraLexFetcher
from engine.fetchers.osmanlica_lugat import OsmanlicaLugatFetcher
from engine.fetchers.proto_turkic_local import LocalProtoTurkicFetcher
from engine.fetchers.starling import StarlingFetcher
from engine.fetchers.tdk_historical import TdkDerlemeFetcher, TdkTaramaFetcher
from engine.fetchers.tdk_nisanyan import NisanyanFetcher, TdkFetcher
from engine.fetchers.tietze_altaica import TietzeAltaicaFetcher
from engine.fetchers.turkic_national_dictionaries import TurkicNationalDictionariesFetcher
from engine.fetchers.wiktextract_local import WiktextractFetcher
from engine.fetchers.wiktionary import WiktionaryFetcher
from engine.llm.qwen_agent import QwenEtymologyAgent
from engine.logging_setup import get_logger
from engine.nlp.cldf_lingpy_aligner import CldfLingPyAligner
from engine.nlp.cognate_alignment import CognateAlignmentEngine
from engine.nlp.cognate_clustering import CognateClusterEngine
from engine.nlp.derivation_network import DerivationNetworkBuilder
from engine.nlp.diachronic_semantic_engine import DiachronicSemanticEngine
from engine.nlp.donor_search import DonorSearchEngine
from engine.nlp.historical_morphology import HistoricalMorphologyAnalyzer, infinitive_stem, strip_infinitive
from engine.nlp.iterative_hypothesis_engine import (
    IterativeHypothesisEngine,
    _historical_gloss,
)
from engine.nlp.iterative_hypothesis_prover import IterativeHypothesisProver
from engine.nlp.loanword_classifier import LoanwordClassifier
from engine.nlp.loanword_detector import LoanwordDetector
from engine.nlp.reconstruction import ProtoTurkicReconstructor
from engine.nlp.sound_law_induction import SoundLawInductionEngine
from engine.utils.cognates import get_related_cognates
from engine.utils.geo_tagger import tag_geographical_region
from engine.utils.morphology import analyze_morphology, is_inflection_gloss
from engine.utils.network import Diagnostics, RequestRecord, capture_requests, unanswered_status
from engine.utils.orthography import to_comparison_form
from engine.utils.phonetic_rules import analyze_phonetic_shifts
from engine.utils.reference_resolver import extract_cross_references, is_cross_reference
from engine.utils.seed import load_seed_entries
from engine.utils.transliteration import transliterate_to_latin
from engine.utils.variant_expander import generate_dynamic_phonetic_variants


@lru_cache(maxsize=1)
def _meaning_translations() -> dict[str, str]:
    """İngilizce -> Türkçe temel anlam eşlemesi (tohum veri)."""
    return {k.lower(): v for k, v in load_seed_entries("meaning_translations.json").items()}


def translate_meaning(meaning: str) -> str:
    """
    İngilizce sözlük tanımını Türkçeleştirir.

    Eşleşme TAM KELİME sınırındadır. Eski sürüm substring araması yapıyordu:
    ``"sun"`` girdisi ``"Sunday"``, ``"consume"``, ``"sunset"`` gibi
    kelimelerde eşleşip anlamı "güneş, gün" olarak DEĞİŞTİRİYORDU.
    """
    m = (meaning or "").strip()
    if not m or m.startswith("Online"):
        return m

    m_clean = re.sub(r"\{\{.*?\}\}", "", m).strip()
    m_clean = re.sub(r"\[\[(.*?)\]\]", r"\1", m_clean).strip()

    # ⚠️ YALNIZ anlamın TAMAMI ya da İLK öbeği sözlük maddesiyse çevrilir.
    # Eskiden metnin HERHANGİ bir yerindeki ilk eşleşme bütün anlamın yerine
    # geçiyordu. Ölçüldü: Osmanlıca `ekmek` "bread, a foodstuff prepared from
    # a dough of flour and water" -> "su, sıvı"; bu yanlış anlam hem sözlük
    # listesine hem A-HVP'nin tarihî anlam girdisine giriyordu.
    translations = _meaning_translations()
    lowered = m_clean.lower().strip()
    first = re.split(r"[,;]", lowered, maxsplit=1)[0].strip()
    for candidate in (lowered, first):
        candidate = re.sub(r"^(?:the|a|an|to)\s+", "", candidate).strip(" .")
        if candidate in translations:
            return translations[candidate]

    return m_clean or meaning

logger = get_logger(__name__)


#: Ana (başlık) anlamının öncelikli kaynağı: ölçünlü Türkçe sözlük.
_PRIMARY_MEANING_SOURCE = "TDK (Türk Dil Kurumu)"
#: TDK yokken başlık anlamının Türkçe yedeği (bkz. `_index_turkish_gloss`).
_INDEX_TR_MEANING_SOURCE = "Türkçe Vikisözlük (yerel sözlük indeksi)"


def _add_meaning(bucket: list[str], meaning: str) -> None:
    """Anlamı kaynak listesine ekler; boş, yer tutucu ve göndermeleri atlar.

    "bk. derlik" bir anlam değil, sözlüğün başka maddeye yönlendirmesidir;
    anlam diye basılırsa okuru yanıltır (ölçüldü: `terlik` başlığı).
    """
    m = (meaning or "").strip()
    if (
        not m
        or m.startswith("Online")
        or m.endswith("madde mevcut")  # çok dilli Wiktionary yer tutucusu
        or is_cross_reference(m)
        or m in bucket
    ):
        return
    bucket.append(m)


def _names_word(form: str, word: str) -> bool:
    """Kayıt biçimi sorgu kelimesini adlandırıyor mu.

    Tarama çift biçim yazar: "derlik (terlik)" — ikinci biçim sorgunun kendisidir.
    """
    return any(p.strip().strip("*-").lower() == word for p in re.split(r"[/,;()]", form))


#: Sorgunun KENDİ dil çizgisi: Türkiye Türkçesi ve ataları (Osmanlıca, Eski
#: Anadolu Türkçesi, Eski Türkçe). Başlık anlamı ve kelime düzeyi köken yalnız
#: bu dillerin kaydından okunur. ⚠️ Eskiden yalnız biçim eşleşmesine
#: bakılıyordu ve kardeş dilin EŞYAZIMLI kelimesi sorgunun kaydı sayılıyordu
#: (105 kelimelik denetimde ölçüldü: 31 kelimede başlık anlamı başka Türk
#: dilinden — `bale` "беда" Nogayca, `set` "sofa, couch", `tüp` "bottom",
#: `özen` "river"; 24 kelimede köken katmanı — `nice` "alıntı — Rusça как
#: (Gagavuzca)").
OWN_LINE_CODES = frozenset({"tr", "ota", "otk", "trk-oat"})


def _is_own_record(entry: dict[str, Any], word: str) -> bool:
    """Kayıt sorgu kelimesinin KENDİ (Türkiye Türkçesi çizgisindeki) kaydı mı.

    Ağız kaydı (`dialect`) sayılmaz: Derleme'nin `tüp` "Alt, dip" kaydı
    yöresel bir eşseslidir, ölçünlü dilin anlamı değil.
    """
    if entry.get("lang_code") not in OWN_LINE_CODES or entry.get("dialect"):
        return False
    return _names_word(entry.get("word") or "", word) or (
        bool(entry.get("comparison")) and entry["comparison"] == to_comparison_form(word)
    )


def _snippet(text: str, term: str, width: int = 160) -> str:
    """Metnin terimi içeren kısmı."""
    text = re.sub(r"\s+", " ", text)
    at = text.lower().find(term.lower())
    if at < 0 or len(text) <= width:
        return text[:width]
    start = max(0, at - width // 2)
    return ("…" if start else "") + text[start:start + width] + ("…" if start + width < len(text) else "")


def _root_note(form: str, lang_code: str) -> str:
    """Kökün aynı dildeki sözlük maddesinden anlamı ve etimoloji notu."""
    try:
        from engine.db.lexicon_index import LexiconIndex

        index = LexiconIndex()
        if not index.exists or not lang_code:
            return ""
        rows = index.lookup(form.strip("-"), languages=[lang_code], limit=3)
    except Exception:
        logger.warning("Kök notu okunamadı: %s", form, exc_info=True)
        return ""
    for row in rows:
        gloss, etymology = str(row.get("gloss") or ""), str(row.get("etymology") or "")
        if gloss or etymology:
            return f"“{gloss}” — {etymology}" if gloss else etymology
    return ""


def _origin_layers(
    entries: list[dict[str, Any]], word: str, formation_entry: dict[str, Any] | None
) -> list[str]:
    """Kelime düzeyi ile kök düzeyi kökeni AYRI satırlarda.

    `bitig` Türkçe içinde yapılmıştır (biti- + -g), yalnız kökünün Orta
    Çince 筆'den geldiği düşünülür. Tek bir "alıntı / öz Türkçe" etiketi
    bunu anlatamaz: rapor hem "Asli Öz Türkçe" hem Çince kök notu basıyor
    ve okur hangisinin neyi söylediğini göremiyordu.

    ⚠️ Yalnız rapordur; alıntı sınıflayıcılarının kararını değiştirmez.
    """
    from engine.nlp.borrowing_chain import TURKIC_LINEAGE_CODES, language_name

    layers: list[str] = []
    if formation_entry:
        layers.append(
            f"Kelime: Türki içi yapım — {formation_entry['formation']} "
            f"({formation_entry.get('lang_name') or formation_entry.get('lang_code')} sözlük maddesi)"
        )
    for entry in entries:
        # Kardeş dilin eşyazımlı kaydı sorgunun kökenini söylemez
        # (`nice` ≠ Gagavuzca *nice* < Rusça как).
        if not _is_own_record(entry, word):
            continue
        donor = str(entry.get("donor_lang") or "")
        if entry.get("lexicon_origin") != "alıntı" or not donor or donor in TURKIC_LINEAGE_CODES:
            continue
        form = re.sub(r"<[^<>]*>", "", str(entry.get("donor_form") or "")).strip()
        where = entry.get("lang_name") or entry.get("lang_code")
        if formation_entry:
            layers.append(f"Kökün uzak kaynağı: {language_name(donor)} {form} ({where} sözlük kaydına göre)")
        else:
            layers.append(f"Sözlük kaydı: alıntı — {language_name(donor)} {form} ({where})")
        break
    return layers


ASSERTED_COGNATE_SOURCE = "Sözlük indeksi — etimoloji notundaki akrabalık beyanı"

#: Akrabalık ipucu: not kaydın MİRAS/akraba olduğunu söylüyor.
_KINSHIP_CUE = re.compile(
    r"\binherited\b|\bfrom proto-turkic\b|\bcognate|\bcompare\b|родствен|восход|пратюрк",
    re.IGNORECASE,
)

#: Kaydın herhangi bir yerinde geçerse kayıt alıntı/ikizlemedir, tanık değil.
#: Cümle düzeyinde bakmak yetmedi — ölçüldü: Azerice `kitab` "Borrowed from
#: Arabic كِتَاب. Compare Turkish kitap." ikinci cümle yüzünden geçiyordu.
#: Aynı kalıpla `betik` ("Learned borrowing from Old Turkic bitig") ve
#: Sahaca `бичик` ("Borrowed from Mongolian бичиг, from Proto-Turkic *bitig")
#: elenir.
_NOT_KINSHIP = re.compile(r"borrow|doublet|calque|заимств", re.IGNORECASE)

#: Sorgu kelimesi Türkçe / Eski Türkçe / Proto-Türkçe biçim olarak ANILMALI.
#: ⚠️ Yalnız "kelime geçiyor" yetmez — ölçüldü: `el` araması "Uyghur ئەل (el)"
#: üzerinden "el = halk, ülke" maddelerini getiriyordu; Türkçe `el` "el
#: organı" ile eşsesli. `{q}` sorgu kelimesiyle doldurulur.
#: Uzunluk işaretleri bilerek normalleştirilmez: *ēl "ülke" `el`e eşlenirse
#: aynı eşseslilik geri gelir.
_ATTRIBUTED_FORM = (
    r"(?:turkish|old turkic|proto-turkic|др\.-тюрк\.?|турецк\w*|тур\.)"
    # Ara boşluk cümle sınırını (nokta) geçmez: "Proto-Turkic *ēl. Cognate
    # with Uyghur ئەل (el)" Uygurca biçimi Proto-Türkçeye bağlamamalı.
    r"[^,;.]{{0,40}}?(?<![\w-])\*?{q}(?![\w-])"
)


#: Semantik benzerlik alt sınırı: altındaki "akraba", sorgunun EŞSESLİSİNİN
#: akrabasıdır. Ölçüldü (paraphrase-multilingual-MiniLM, tanığın gloss'u ile
#: sorgunun sözlük anlamları arasındaki en yüksek kosinüs):
#:     eşsesli : ekmek~"to sow" 0.22/0.16, el~"fifty" 0.26, su~"healthy" 0.13,
#:               su~"to milk" 0.16, el~"eyləmək" 0.27
#:     gerçek  : su~"вода" 0.385 (en düşük), bitig~"amulet" 0.539,
#:               baş~"голова" 0.485, deniz~"sea" 0.447, ekmek~"bread" 0.588
#: Pay dar (0.27 ile 0.385) ve örnek 22 çift; eşik gözden geçirilmeli.
#: Türevler (göz~"mirror" 0.47, göz~"to see" 0.41) geçer — eşsesli değil,
#: anlamca bağlı kelimelerdir.
HOMONYM_SIMILARITY_FLOOR = 0.30

#: Yerel çağdaş dil adayları (yazılışla bulunur) için ANLAM alt sınırı.
#: Eşsesli süzgecinin 0,30'u burada yetmiyor. 50 kelimede elle sayıldı
#: (rastgele 25'er kayıt): 0,50 altında ~%24 sahte akraba (`tozmak` ~ туз
#: "tuz", `gerek` ~ кӗрӗк "kürk", `kırkmak` ~ кырк "kırk"), üstünde ~%4
#: (`çığlık` ~ çığ). Bedel: 225 adaydan 73'ü kalır; `deniz`in 0,46-0,49'daki
#: doğru Karayca/Kırım Tatarca biçimleri de elenir.
#: Bağımsız doğrulama (100 YENİ kelime, rastgele 50 tanık elle): 48 doğru,
#: 1 sahte (Rusça yönlendirme anlamı, artık süzülüyor), 1 sınırda.
LOCAL_WITNESS_FLOOR = 0.50

#: Yerel çağdaş dil adayı, en iyi eşleşen adayın bu kadar altındaysa elenir
#: (başka anlamın, yani eşseslinin kaydıdır).
LOCAL_WITNESS_MARGIN = 0.35


def _first_sentence_loan(text: str) -> bool:
    """İlk cümle aile dışı bir dilden "From X" diyor mu (ҡәләм: "From Arabic قَلَم")."""
    from engine.db.lexicon_index import ETYMOLOGY_TEXT_DONORS

    first = re.split(r"(?<=[.;])\s+", text.strip(), maxsplit=1)[0]
    names = [*ETYMOLOGY_TEXT_DONORS, "Middle Chinese", "Old Chinese"]
    return any(re.search(rf"\bfrom {re.escape(n)}\b", first, re.IGNORECASE) for n in names)


def _homonym_filter(
    candidates: list[dict[str, Any]], query_meanings: list[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Anlamca sorgunun HİÇBİR anlamına yakın olmayan adayları ayırır.

    Model yoksa ya da karşılaştırılacak anlam yoksa süzgeç uygulanmaz
    (n-gram yedek vektörleri anlam ayırmaz; yanlış eleme yapmasın).
    """
    glossed = [c for c in candidates if c.get("meaning")]
    if not glossed or not query_meanings:
        return candidates, []
    try:
        from engine.nlp.diachronic_semantic_engine import get_sentence_transformer, has_semantic_model

        if not has_semantic_model():
            return candidates, []
        from sentence_transformers.util import cos_sim

        model = get_sentence_transformer()
        q = model.encode(query_meanings[:12], show_progress_bar=False)
        g = model.encode([c["meaning"] for c in glossed], show_progress_bar=False)
        sims = cos_sim(g, q).max(dim=1).values.tolist()
    except Exception:
        logger.warning("Eşsesli süzgeci çalışmadı; adaylar süzülmeden alındı", exc_info=True)
        return candidates, []
    kept, dropped = [c for c in candidates if not c.get("meaning")], []
    for cand, sim in zip(glossed, sims, strict=True):
        cand["meaning_similarity"] = round(float(sim), 3)
        (kept if sim >= HOMONYM_SIMILARITY_FLOOR else dropped).append(cand)
    return kept, dropped


def _asserted_cognates(word: str, mentions: dict[str, Any]) -> list[dict[str, Any]]:
    """Etimoloji notu sorgu kelimesiyle akrabalık İDDİA EDEN Türki kayıtlar.

    `bitig` tek tanıkla kalıyor ve rapor "dar/lokal yayılım (ağız terimi
    veya son dönem alıntı)" diyordu; oysa sözlüğün kendisi Çuvaşça *пӗтӳ*
    ("From Proto-Turkic *bitig"), Başkurtça *бетеү* ("Родственно др.-тюрк.
    bitig") ve Türkçe *biti* ("Inherited from Proto-Turkic *bitig") için
    akrabalığı AÇIKÇA söylüyor.

    ⚠️ Bunlara ANLAM SÜZGECİ UYGULANMAZ, bilerek: *пӗтӳ* "amulet" anlamca
    "inscription"dan uzaktır ama gerçek akrabadır (Başkurtça kayıt da
    «письмо, надпись; амулет» der). Kaynağın açık iddiası anlam
    benzerliğinden güçlü kanıttır. Anlam süzgeci, ses varyantıyla BULUNAN
    (kimsenin akraba demediği) tanıklar için gereklidir.

    Birleşik ve türemiş kelimeler (*göz yaşı*, *gözyaşı*, *baş burmaq*)
    akraba değil, sorgu kelimesinin türevidir; elenir.
    """
    from engine.fetchers.base import detect_script

    own = to_comparison_form(word)
    attributed = re.compile(_ATTRIBUTED_FORM.format(q=re.escape(word.lower())), re.IGNORECASE)
    out: list[dict[str, Any]] = []
    for item in mentions.get("items", []):
        if item["lang_code"] not in TURKIC_LANGUAGES_MAP:
            continue
        form = str(item.get("comparison") or "")
        if " " in str(item["word"]).strip() or (form != own and own and own in form):
            continue
        # Anlamı başka biçime gönderme olan kayıt ("dated form of eləmək",
        # "plural of …") anlam taşımaz; akraba adayı değildir.
        gloss = str(item.get("gloss") or "")
        if is_inflection_gloss(gloss) or re.search(r"\bform of\b", gloss, re.IGNORECASE):
            continue
        text = str(item.get("etymology_full") or item.get("etymology") or "")
        if _NOT_KINSHIP.search(text) or not _KINSHIP_CUE.search(text) or _first_sentence_loan(text):
            continue
        if not attributed.search(text):
            continue
        out.append({
            "lang_code": item["lang_code"],
            "lang_name": item["lang_name"],
            "word": item["word"],
            "meaning": item.get("gloss") or "",
            "script": detect_script(item["word"]),
            "origin": "seed",
            "source": ASSERTED_COGNATE_SOURCE,
            "comparison": form,
            "etymology": text,
            "asserted_cognate": True,
        })
    return out


CITED_COGNATE_SOURCE = "Sözlük indeksi — kaynak kaydının akraba listesi"


def _own_lexicon_entries(word: str, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sorgu kelimesinin KENDİ sözlük indeksi kayıtları (köken sınıfı taşıyanlar)."""
    own = to_comparison_form(word)
    return [
        e for e in entries
        if e.get("lexicon_origin") is not None
        and (_names_word(e.get("word") or "", word) or e.get("comparison") == own)
    ]


def _cited_cognates(word: str, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sorgunun kendi MİRAS kaydının akraba listesindeki Türki biçimler.

    `boncuk`un Osmanlıca kaydı "Cognate with Azerbaijani muncuq, Kazakh
    моншақ, Kyrgyz мончок, Turkmen monjuk, Uyghur مونچاق and Uzbek munchoq"
    diyor; bu liste yalnız not olarak basılıyor, tanık sayılmıyordu ve rapor
    ortak Türkçe bir kelime için "%12 dar/lokal yayılım" veriyordu. Ölçüldü:
    akraba listesi olan 8 kelimenin 7'sinde listedeki diller tanıkta yoktu.

    ⚠️ Yalnız MİRAS kayıtları: alıntı bir kelimenin "akrabaları" paralel
    alıntılardır (Azerice kitab ~ Türkçe kitap) ve tanık sayılırsa yayılım
    sinyali alıntıyı yerli gösterir. Anlam süzgeci uygulanmaz; kaynağın
    açık akrabalık iddiası anlam benzerliğinden güçlüdür (bkz.
    `_asserted_cognates`).
    """
    from engine.fetchers.base import detect_script

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for entry in _own_lexicon_entries(word, entries):
        if entry.get("lexicon_origin") != "miras":
            continue
        for cognate in entry.get("source_cognates") or []:
            lang = str(cognate.get("lang") or "")
            form = str(cognate.get("form") or "").strip()
            if lang not in TURKIC_LANGUAGES_MAP or not form or " " in form:
                continue
            # Arap yazılı Uygurca biçim (مونچاق) karşılaştırma biçimine
            # çevrilemiyor ve boş kalıyordu; ölçüldü: Uygurca 4 kelimede
            # listeden düşüyordu. Kaba harf çevirisi, biçimi kaybetmekten iyidir.
            comparison = to_comparison_form(str(cognate.get("reading") or "") or form) or (
                to_comparison_form(transliterate_to_latin(form))
            )
            if not comparison or (lang, comparison) in seen:
                continue
            seen.add((lang, comparison))
            out.append({
                "lang_code": lang,
                "lang_name": TURKIC_LANGUAGES_MAP[lang],
                "word": form,
                "meaning": str(cognate.get("gloss") or ""),
                "script": detect_script(form),
                "origin": "seed",
                "source": CITED_COGNATE_SOURCE,
                "comparison": comparison,
                "etymology": f"{entry.get('lang_name') or entry.get('lang_code')} "
                             f"{entry.get('word')} kaydının akraba listesi",
                "asserted_cognate": True,
            })
    return out


_REDIRECT_GLOSS = re.compile(r"\b(?:form|spelling) of\b", re.IGNORECASE)


#: Hüküm ALINTI iken çeviri/indeks tanıklarının rolü. Sıralayıcının ve
#: akraba listesinin (`engine/utils/cognates.py`) kullandığı etiketle aynı.
PARALLEL_LOAN_LABEL = "paralel alıntı"


def _mark_parallel_loans(
    entries: list[dict[str, Any]], translation_sources: set[str]
) -> list[dict[str, Any]]:
    """Alıntı kelimede çeviri/indeks tanıklarını paralel alıntı diye işaretler.

    Apertium çevirisi, NorthEuraLex kavramı ve çağdaş dil indeksindeki
    eşyazımlı madde yalnız BİÇİMCE benzer karşılıktır. Kelime alıntıysa bu
    benzerlik ortak atadan değil, aynı vericiden ayrı ayrı alınmaktan gelir:
    `bant` ~ Tatarca бинт, `bale` ~ Kırgızca/Tatarca балет (105 kelimelik
    denetimde 36 kelime). Miras akraba gibi gösterilmez, yayılım ve miras
    kanıtı sayılmaz. Kaynağın açık akrabalık beyanı (`asserted_cognate`) ve
    sorgunun kendi dil çizgisinin kaydı işaretlenmez.

    İşaretlenmeyen (kanıt sayılacak) kayıtları döndürür.
    """
    evidence: list[dict[str, Any]] = []
    for entry in entries:
        if (
            entry.get("source") in translation_sources
            and entry.get("lang_code") in TURKIC_LANGUAGES_MAP
            and entry.get("lang_code") not in OWN_LINE_CODES
            and not entry.get("asserted_cognate")
        ):
            entry["parallel_loan"] = True
            entry["witness_role"] = PARALLEL_LOAN_LABEL
        else:
            evidence.append(entry)
    return evidence


def _rank_own_by_meaning(word: str, entries: list[dict[str, Any]], primary: str) -> list[tuple[float, dict[str, Any]]]:
    """Sorgunun KENDİ kayıtları, anlamlarının sorgunun ana anlamına benzerliğiyle.

    Eşsesli kelimede ilk kayıt yanlış anlamın kaydı olabilir: `el` için
    Osmanlıca "people" (*ēl) kaydı "hand" kaydından önce geliyordu ve başlık
    *ēl, eşsesli süzgeci "halk" anlamlı akrabaları geçiriyordu (ölçüldü).
    Mutlak eşik yok: 'bead' ~ TDK'nın uzun Türkçe tanımı zaten 0,22; kayıtlar
    birbirine göre sıralanır. Model ya da ana anlam yoksa sıra korunur (1,0).
    """
    own = to_comparison_form(word)
    # Yalnız sözlük indeksi kayıtları: TDK kaydı ana anlamın KENDİSİDİR
    # (benzerliği 1,0 çıkar), yerel çağdaş dil adayları ise başka dildir.
    candidates = [
        e for e in entries
        if "yerel sözlük indeksi" in str(e.get("source") or "") and not e.get("meaning_check")
        and (_names_word(e.get("word") or "", word) or e.get("comparison") == own)
        and e.get("meaning") and not is_inflection_gloss(str(e["meaning"]))
        and not _REDIRECT_GLOSS.search(str(e["meaning"]))
    ]
    if not primary or len(candidates) < 2:
        return [(1.0, e) for e in candidates]
    try:
        from engine.nlp.diachronic_semantic_engine import get_sentence_transformer, has_semantic_model

        if not has_semantic_model():
            return [(1.0, e) for e in candidates]
        from sentence_transformers.util import cos_sim

        model = get_sentence_transformer()
        sims = cos_sim(
            model.encode([str(e["meaning"]) for e in candidates], show_progress_bar=False),
            model.encode([primary], show_progress_bar=False),
        )[:, 0].tolist()
    except Exception:
        logger.warning("Kendi kayıtları anlamca sıralanamadı", exc_info=True)
        return [(1.0, e) for e in candidates]
    return sorted(zip(sims, candidates, strict=True), key=lambda pair: -pair[0])


#: En iyi anlamla "aynı anlam grubu" sayılan benzerlik farkı.
_SAME_SENSE_MARGIN = 0.10


def _query_source_proto(word: str, entries: list[dict[str, Any]], primary: str = "") -> tuple[str, str]:
    """Sorgunun kendi miras kaydının verdiği Proto-Türkçe biçim ve kaydın dili.

    Ölçüldü: kaynakta Proto-Türkçe biçim bulunan 12 kelimenin yaklaşık
    5'inde başlık motorun kendi, farklı rekonstrüksiyonunu gösteriyordu
    (`uçmak` *uça ↔ kaynak *uč-, `kırkmak` *kırko ↔ *kïrk, `boncuk`
    *bonjuk ↔ *bōnčuk).
    """
    ranked = _rank_own_by_meaning(word, entries, primary)
    if not ranked:
        return "", ""
    best = ranked[0][0]
    for similarity, entry in ranked:
        if best - similarity > _SAME_SENSE_MARGIN:
            break  # buradan sonrası başka bir anlamın (eşseslinin) kaydı
        form = re.sub(r"<[^<>]*>", "", str(entry.get("donor_form") or "")).strip()
        # Harfsiz biçim (`*-`, indeksteki boş şablon) kök değildir: `kavuk`
        # başlığı "*-" basıyordu.
        if (entry.get("lexicon_origin") == "miras" and entry.get("donor_lang") == "trk-pro"
                and re.search(r"\w", form.strip("*-"))):
            return (form if form.startswith("*") else f"*{form}",
                    str(entry.get("lang_name") or entry.get("lang_code")))
    return "", ""


_ENGLISH_GLOSS = re.compile(
    r"^[A-Za-z0-9 ,;:()'\"\-.!?/]+$|\b(?:of|the|an?|and|or|to|clipping|abbreviat\w*)\b"
)
#: Ağız kaydı (`(Artvin, Erzincan ağzı) …`) ölçünlü dilin anlamı değildir.
_DIALECT_GLOSS = re.compile(r"^\([^)]*ağz[ıi][^)]*\)")


def looks_english(text: str) -> bool:
    """Anlam İngilizce mi (kaba ama açık ölçüt: ASCII ya da İngilizce işlev sözcüğü)."""
    return bool(_ENGLISH_GLOSS.search(text.strip()))


def _index_turkish_gloss(word: str) -> str:
    """Sorgunun indeksteki Türkçe kaydının (Türkçe Vikisözlük) ilk TÜRKÇE anlamı.

    TDK cevap vermeyince başlık anlamı ya boş (kelimenin kendisi) ya da
    İngilizce Wiktionary anlamı oluyordu; indeksteki Türkçe sürüm kaydı
    hiç okunmuyordu (105 kelimelik denetim, yalnız yerel kaynak: 76 boş,
    27 İngilizce, 2 Türkçe). Yalnız sorgunun kendisi (`kâr` ≠ `kar`),
    özel ad, ağız kaydı ve yönlendirme olmayan anlam.
    """
    try:
        from engine.db.lexicon_index import LexiconIndex

        index = LexiconIndex()
        if not index.exists:
            return ""
        for row in index.lookup(word, languages=["tr"], limit=30):
            gloss = str(row.get("gloss") or "").strip()
            if (row.get("word") != word or row.get("pos") == "name" or not gloss
                    or looks_english(gloss) or _DIALECT_GLOSS.match(gloss)
                    or is_cross_reference(gloss) or is_inflection_gloss(gloss)):
                continue
            return gloss
    except Exception:
        logger.debug("İndeks Türkçe anlamı okunamadı: %s", word, exc_info=True)
    return ""


def _infinitive_of(stem: str) -> str:
    """Çıplak fiil gövdesinin sözlük madde başı (`ayır` -> `ayırmak`)."""
    vowels = [c for c in stem if c in "aıoueiöü"]
    return stem + ("mak" if vowels and vowels[-1] in "aıou" else "mek")


def _index_source_proto(word: str, primary: str = "") -> tuple[str, str]:
    """Sorgunun sözlük indeksindeki KENDİ miras kaydının Proto-Türkçe biçimi.

    `_query_source_proto` yalnız fetcher'ların getirdiği kayıtlara bakar;
    oysa indeks fetcher'ları Türkiye Türkçesi (`tr`) kaydını HİÇ getirmez
    (tarihî katman ota/otk/chg, çağdaş katman tr hariç). `anız` (*aŋïŕ),
    `bön`, `sığ` gibi kelimelerin Türkçe kaydı Proto-Türkçe biçimi açıkça
    verdiği hâlde başlık "kök belirlenemedi" diyordu. Çıplak fiil gövdesi
    (`ayır`, `sil`, `uyan`) indekste mastarlı madde başıyla (`ayırmak`)
    durur; gövdenin kendi kaydı yoksa yalnız mastarın FİİL kaydına bakılır.

    Ölçüldü (başlık ölçümü, Starling kapalı, 240 kelime): tam 0,312 -> 0,412;
    Starling açık üretim ayarı değişmez (bu yedek Starling'den SONRA gelir).
    """
    try:
        from engine.db.lexicon_index import LexiconIndex

        index = LexiconIndex()
        if not index.exists:
            return "", ""
        stem = to_comparison_form(word)
        for query, verbs_only in ((stem, False), (_infinitive_of(stem), True)):
            entries = [
                {"source": "yerel sözlük indeksi", "word": row.get("word") or "",
                 "comparison": row.get("comparison") or "", "lang_code": row.get("lang_code"),
                 "lang_name": TURKIC_LANGUAGES_MAP.get(str(row.get("lang_code") or ""), ""),
                 "meaning": row.get("gloss") or "", "lexicon_origin": row.get("origin"),
                 "donor_lang": row.get("donor_lang"), "donor_form": row.get("donor_form")}
                for row in index.lookup(query, languages=["tr", "ota"], limit=20)
                if not verbs_only or row.get("pos") == "verb"
            ]
            form, lang = _query_source_proto(query, entries, primary)
            if form:
                return form, lang
    except Exception:
        logger.debug("İndeks kaynak kökü okunamadı: %s", word, exc_info=True)
    return "", ""


def _english_query_gloss(word: str, entries: list[dict[str, Any]], primary: str = "") -> str:
    """Sorgunun kendi Türkçe/Osmanlıca sözlük kaydının İLK İngilizce anlamı.

    Eşsesli süzgeci tanığın İngilizce anlamını ("bead") TDK'nın uzun
    Türkçe tanımıyla ("Cam, taş, sedef… süs tanesi") karşılaştırıyor ve
    gerçek akrabaları eliyordu (ölçüldü: `boncuk`~Uygurca "bead" 0,219;
    `bağlamak`~Kırgızca "tie" 0,269; `gerek`~Hakasça 0,249). Yalnız İLK
    kaydın ilk anlamı alınır: bütün anlamlar eklenirse sorgunun kendi
    eşseslisi (`ekmek` "to sow") süzgeci yeniden gevşetir.
    """
    for _similarity, entry in _rank_own_by_meaning(word, entries, primary):
        if entry.get("lang_code") in ("tr", "ota"):
            return re.split(r"[;(]", str(entry["meaning"]))[0].strip()
    return ""

#: Kaynak zincirindeki verici dil adı -> sınıflandırıcının aile anahtarı.
_SOURCE_DONOR_FAMILY = {
    "Arapça": "arabic_persian", "Farsça": "arabic_persian",
    "Yunanca": "greek_latin", "Eski Yunanca": "greek_latin",
    "Latince": "greek_latin", "Ermenice": "greek_latin",
    "Fransızca": "western", "İngilizce": "western", "İtalyanca": "western",
    "Rusça": "western", "Almanca": "western",
}


def _apply_source_loan_family(loan_eval: dict[str, Any], entries: list[dict[str, Any]]) -> None:
    """Kaynak açık bir alıntı adımı veriyorsa sınıflandırmayı ona göre düzeltir.

    `LoanwordClassifier` yalnız kelimenin ses yapısına bakar; `faça` için
    "verici dil ailesi belirlenemedi" (Doğu %32,3 = Batı %32,3) diyordu,
    aynı çıktıda kaynak "Alıntı: İtalyanca faccia" derken. Olasılık dağılımı
    DEĞİŞTİRİLMEZ; o hâlâ yalnız ses yapısının söylediğidir.
    """
    from engine.nlp.borrowing_chain import source_loan_step
    from engine.nlp.loanword_classifier import CLASSIFICATION_LABELS

    step = source_loan_step(entries)
    family = _SOURCE_DONOR_FAMILY.get(str((step or {}).get("lang_name") or ""))
    if not step or not family or loan_eval.get("classification_key") == family:
        return
    loan_eval["phonotactic_classification"] = loan_eval.get("classification")
    loan_eval["classification"] = CLASSIFICATION_LABELS[family]
    loan_eval["classification_key"] = family
    loan_eval["source_override"] = (
        f"kaynağın alıntı zinciri: {step.get('lang_name')} {step.get('word')} "
        f"({step.get('source') or 'kaynak'}); ses yapısına göre sınıf: "
        f"{loan_eval['phonotactic_classification']} — aşağıdaki dağılım yalnız ses yapısıdır"
    )


def _source_proto_forms(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Kaynakların AÇIKÇA verdiği Proto-Türkçe biçimler, kaç kayıtta geçtiğiyle.

    Motorun rekonstrüksiyonu ile kaynağın iddiası ayrışabilir (ölçüldü:
    `bitig` için motor *biti kuruyor — söz sonu -g'nin Kıpçak/Çuvaş
    kollarındaki düşüşünü modellemiyor — ama üç kayıt "Proto-Turkic
    *bitig" diyor). İkisi yan yana gösterilir; okur tahmini iddiayla
    karıştırmasın.
    """
    from collections import Counter

    counts: Counter[str] = Counter()
    for entry in entries:
        if not (entry.get("asserted_cognate") or entry.get("formation")):
            continue
        for form in set(re.findall(r"Proto-Turkic \*([^\s,.;()“”\"]+)", str(entry.get("etymology") or ""))):
            counts[form.rstrip("-")] += 1
    return [{"form": f"*{form}", "count": n} for form, n in counts.most_common(3)]


def default_fetchers() -> list[BaseFetcher]:
    """Üretimde kullanılan varsayılan veri toplayıcı portföyü."""
    return [
        AcademicTurkologyFetcher(),
        HistoricalIndexFetcher(),
        ModernIndexFetcher(),
        NorthEuraLexFetcher(),
        ApertiumFetcher(),
        LocalProtoTurkicFetcher(),
        HistoricalModernLexiconFetcher(),
        # Yerel tohum veri; canlı site yalnız ETY_LIVE_ISAM=1 ile (madde başı).
        IsamAnsiklopediFetcher(),
        *([ArchiveOrgFetcher()] if config.LIVE_ARCHIVE_ORG else []),
        OsmanlicaLugatFetcher(),
        TurkicNationalDictionariesFetcher(),
        LoanwordDonorEtymologyFetcher(),
        LocalPdfBooksFetcher(),
        TietzeAltaicaFetcher(),
        EtimolojiTurkceFetcher(),
        StarlingFetcher(),
        NisanyanFetcher(),
        TdkFetcher(),
        TdkTaramaFetcher(),
        TdkDerlemeFetcher(),
        # Canlı İngilizce Wiktionary: yerel karşılıkları ModernIndexFetcher
        # (aynı yazılışlı maddeler, anlam doğrulamalı) ve LocalProtoTurkicFetcher
        # (kök torunları). Ağ ancak açıkça istenirse (ETY_LIVE_WIKTIONARY=1).
        *([WiktionaryFetcher(), WiktextractFetcher()] if config.LIVE_WIKTIONARY else []),
        *([MultiLangWiktionaryFetcher()] if config.LIVE_WIKTIONARY_EDITIONS else []),
    ]


class SearchEngine:
    def __init__(
        self,
        db_manager: DatabaseManager | None = None,
        fetchers: list[BaseFetcher] | None = None,
    ):
        """
        :param db_manager: Kalıcılık katmanı; verilmezse varsayılan SQLite yöneticisi.
        :param fetchers: Veri toplayıcı listesi. Testlerde sahte (fake) toplayıcı
            enjekte etmek için kullanılır; verilmezse üretim portföyü kurulur.
        """
        self.db = db_manager or DatabaseManager()
        self.graph_db = GraphDatabaseManager()
        self.qwen_agent = QwenEtymologyAgent()

        # NLP & İleri Hesaplamalı Modüller
        self.loanword_classifier = LoanwordClassifier()
        self.cognate_alignment_engine = CognateAlignmentEngine()
        self.reconstructor = ProtoTurkicReconstructor()
        self.donor_search_engine = DonorSearchEngine()
        self.hypothesis_engine = IterativeHypothesisEngine()
        self.hypothesis_prover = IterativeHypothesisProver()
        self.lingpy_aligner = CldfLingPyAligner()
        self.semantic_engine = DiachronicSemanticEngine()
        self.sound_law_induction = SoundLawInductionEngine()
        # Alıntı keşfi, akraba kümeleme ve tarihsel morfoloji katmanları
        self.loanword_detector = LoanwordDetector(classifier=self.loanword_classifier)
        self.cognate_cluster_engine = CognateClusterEngine()
        self.historical_morphology = HistoricalMorphologyAnalyzer()
        self.derivation_builder = DerivationNetworkBuilder()

        self.fetchers: list[BaseFetcher] = fetchers if fetchers is not None else default_fetchers()
        # Yerel sözlük indeksi bir KAYNAKTIR: ters bağlantı araması ve kök notu
        # da ona sorar, bu yüzden yalnız portföyde onun fetcher'ı varsa
        # çalışırlar (sahte fetcher'lı testler gerçek indekse sızmasın).
        self.uses_lexicon_index = any(isinstance(f, HistoricalIndexFetcher) for f in self.fetchers)


    @staticmethod
    def _etymology_mentions(word: str, limit: int = 0) -> dict[str, Any]:
        """Etimoloji metninde sorgu kelimesi geçen sözlük kayıtları (ters bağlantı).

        `bitig` araması `bitig` biçimini arar; oysa Türkçe `betik`in maddesi
        "Learned borrowing from Old Turkic bitig" der. Bu bağ ancak etimoloji
        METNİNDE aranarak bulunur. İndeks bu sütunu FTS5'e açıyordu ama
        arama hattı hiç sormuyordu.

        ⚠️ Bunlar TANIK DEĞİLDİR ve `turkic_languages`a girmez: `betik` bir
        dil devrimi türetmesidir, akraba gibi sayılırsa rekonstrüksiyonu
        bozar. Yalnız raporlanır.
        """
        try:
            from engine.db.lexicon_index import LexiconIndex

            index = LexiconIndex()
            if not index.exists:
                return {"total": 0, "items": []}
            term = word.replace('"', "")
            rows = index.search(f'etymology : "{term}"', limit=200)
        except Exception:
            logger.warning("Etimoloji metni araması başarısız: %s", word, exc_info=True)
            return {"total": 0, "items": []}

        own = to_comparison_form(word)
        # FTS büyük/küçük harf ayırmaz. Büyük harfle geçiş özel addır, biçim
        # atfı değil (ölçüldü: `bitig` araması "Irk Bitig" kitap adını anan
        # 7 alakasız maddeyi getiriyordu: jana "again", köznök "window"…).
        # Başında tire olan geçiş EKTİR, kelime değil: "pamuk + -su" (ölçüldü:
        # `su` araması *pamuksu*, *otsu*, *odunsu* getiriyordu).
        cited = re.compile(rf"(?<![\w-]){re.escape(term.lower())}(?![\w])")
        items: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for row in rows:
            key = (str(row.get("lang_code")), str(row.get("word")))
            if row.get("comparison") == own or key in seen:
                continue
            if not cited.search(str(row.get("etymology") or "")):
                continue
            seen.add(key)
            items.append({
                "lang_code": key[0],
                "lang_name": TURKIC_LANGUAGES_MAP.get(key[0], key[0]),
                "word": key[1],
                "comparison": row.get("comparison") or "",
                "gloss": row.get("gloss") or "",
                "etymology": _snippet(str(row.get("etymology") or ""), term),
                "etymology_full": str(row.get("etymology") or ""),
            })
        # Tümü döner: akrabalık beyanları bütün listede aranır. Ekrana
        # basılan kısım CLI'da kırpılır.
        return {"total": len(items), "items": items[:limit] if limit else items}

    def _consult_descendants(
        self, word: str, mentions: dict[str, Any], limit: int = 2
    ) -> list[tuple[str, str]]:
        """Ters bağlantılı Türkçe maddeleri (bitig -> betik) canlı kaynaklara sorar.

        Nişanyan'da `bitig` maddesi yok ama `betik` maddesi "Eski Türkçe
        bitig “yazı”" der; sorgu yalnız `bitig` diye yapıldığı için bu bilgi
        hiç okunmuyordu. Her maddenin canlı sonucu `mentions` kaydına eklenir;
        madde açıkça SORGU KELİMESİNDEN söz ediyorsa verdiği anlam döner.
        """
        by_type = {type(f): f for f in self.fetchers}
        tdk, nisanyan = by_type.get(TdkFetcher), by_type.get(NisanyanFetcher)
        if not (tdk or nisanyan):
            return []
        own = to_comparison_form(word)
        found: list[tuple[str, str]] = []
        items = [m for m in mentions.get("items", []) if m.get("lang_code") == "tr"][:limit]
        # Canlı istekler paralel atılır (soğuk süreçte sıralı 4 istek ~6,5 s
        # sürüyordu); sonuçlar yine madde sırasıyla işlenir.
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, 2 * len(items))) as pool:
            pending = [
                (item,
                 pool.submit(tdk.fetch, item["word"]) if tdk else None,
                 pool.submit(nisanyan.fetch, item["word"]) if nisanyan else None)
                for item in items
            ]
        for item, tdk_future, nisanyan_future in pending:
            live: dict[str, str] = {}
            if tdk_future is not None:
                t = tdk_future.result()
                if t["root"].get("meaning"):
                    live[tdk.source_name] = t["root"]["meaning"]
            if nisanyan_future is not None:
                n = nisanyan_future.result()
                note = n["root"].get("reconstruction_notes") or ""
                if note:
                    live[nisanyan.source_name] = note
                named = to_comparison_form((n["root"].get("proto_turkic") or "").strip("*"))
                if named == own and n["root"].get("meaning"):
                    found.append((f"{nisanyan.source_name} ({item['word']} maddesi)", n["root"]["meaning"]))
            if live:
                item["live"] = live
        return found

    def _rank_hypotheses(self, word: str, entries: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Rakip köken hipotezlerini sıralar; başarısız olursa hattı durdurmaz."""
        try:
            from engine.nlp.hypothesis_ranking import HypothesisRanker

            return HypothesisRanker().rank(word, entries).as_dict()
        except Exception:
            logger.warning("Hipotez sıralaması başarısız: %s", word, exc_info=True)
            return None

    def search(
        self, query: str, save_to_db: bool = True, use_qwen_agent: bool = False, use_cache: bool = True
    ) -> dict[str, Any]:
        word_clean = query.strip().lower()[: config.MAX_QUERY_LENGTH]
        search_started = time.perf_counter()
        diagnostics = Diagnostics()
        # Kaynak bazlı teşhis: hangi fetcher ne kadar sürdü, ne döndürdü, neden düştü.
        source_diagnostics: dict[str, dict[str, Any]] = {}
        stage_timings: dict[str, int] = {}
        # Aşama süreleri ardışık "tur"larla ölçülür: her `_lap(ad)` bir önceki
        # işaretten bu yana geçen süreyi o aşamaya yazar. Böylece aşamalar
        # boşluksuz bitişir ve toplamları `total`a eşit olur (eskiden soğuk
        # aramanın ~8 s'si hiçbir aşamaya yazılmıyordu).
        _lap_mark = [search_started]

        def _lap(name: str) -> None:
            now = time.perf_counter()
            stage_timings[name] = stage_timings.get(name, 0) + int((now - _lap_mark[0]) * 1000)
            _lap_mark[0] = now

        stem, suffixes = analyze_morphology(word_clean)

        # Varyant patlamasını sınırla: her varyant 21 fetcher'a ayrı istek demek.
        all_variants = list(dict.fromkeys([word_clean, stem, *generate_dynamic_phonetic_variants(word_clean)]))
        search_variants = all_variants[: config.MAX_VARIANTS]
        if len(all_variants) > len(search_variants):
            logger.info(
                "Varyant sayısı %d -> %d olarak sınırlandı (MAX_VARIANTS)",
                len(all_variants), len(search_variants),
            )
        _lap("morphology")

        if config.CACHE_ENABLED and use_cache and not use_qwen_agent:
            cached = self.db.get_finding(word_clean, max_age_seconds=config.CACHE_TTL_SECONDS)
            if cached:
                cached["from_cache"] = True
                logger.info("Önbellekten döndürüldü: %r", word_clean)
                return cached
        _lap("cache_lookup")
        if self.uses_lexicon_index:
            # Anlam süzgeçlerinin modeli ağ beklenirken yüklensin (bkz.
            # `prewarm_sentence_transformer`); sonucu değiştirmez.
            from engine.nlp.diachronic_semantic_engine import prewarm_sentence_transformer

            prewarm_sentence_transformer()

        proto_root = ""
        # Kökün NEREDEN geldiği: sözlükten alıntılanan bilgi ile motorun kendi
        # türettiği hipotez aynı alana yazılıyor ve aynı güven hattından
        # geçiyordu; kullanıcı hangisinin tanık hangisinin tahmin olduğunu
        # göremiyordu. Her atama noktası bu damgayı da koyar.
        proto_root_provenance = ""
        root_meaning = ""
        # Kaynak başına anlamlar. Eskiden tek bir anlam seçiliyordu ve seçim
        # fetcher'ların BİTİŞ SIRASINA bağlıydı (`as_completed`): aynı
        # kelime koşudan koşuya farklı anlamla çıkıyordu, `terlik` TDK'nın
        # "ayak giysisi" tanımı yerine Tarama'nın "bk. derlik" göndermesini
        # basıyordu. Artık her sözlüğün anlamı ayrı ayrı raporlanır.
        meanings_by_source: dict[str, list[str]] = {}
        # Kaynağın kendi ANA anlam alanı (TDK: ilk iki anlamın birleşimi).
        # Ana anlam buradan seçilir, liste ise bütün anlamları gösterir.
        # İkisi ayrı tutulmazsa gösterim değişikliği A-HVP 3. aşamasının
        # girdisini sessizce değiştiriyordu (ölçüldü: 14 kelimenin 14'ünde).
        primary_by_source: dict[str, str] = {}
        sources = []
        turkic_entries_map = {}
        raw_fetcher_results: list[dict[str, Any]] = []

        def fetch_worker(fetcher: BaseFetcher):
            results = []
            started = time.perf_counter()
            errors: list[str] = []
            # Kaynağın dış istekleri: boş sonucun "veri yok" mu, "cevap
            # alınamadı" mı olduğunu ayırmak için (bkz. `unanswered_status`).
            requests_log: list[RequestRecord] = []
            variants = search_variants[:1] if getattr(fetcher, "exact_query_only", False) else search_variants
            for var in variants:
                try:
                    with capture_requests() as book:
                        res = fetcher.fetch(var)
                    requests_log.extend(book.records)
                except Exception as exc:  # fetcher sözleşmesi istisna atmamalı; atarsa görünür olsun
                    logger.warning(
                        "Fetcher istisna attı: %s (varyant=%r)", fetcher.source_name, var, exc_info=True
                    )
                    errors.append(f"{type(exc).__name__}: {exc}")
                    continue
                # ⚠️ Kök `res["root"]["proto_turkic"]` içindedir; eskiden en üst
                # düzeyde aranıyordu ve tanık üretmeyen her sonuç atılıyordu:
                # alıntı kelimede Nişanyan'ın kökü, Starling'in kökü ve yalnız
                # tanıklama tarihi döndüren sonuçlar hiç işlenmiyordu.
                root = (res or {}).get("root") or {}
                if res and (res.get("turkic_languages") or root.get("proto_turkic")
                            or res.get("first_attestation")):
                    # Hangi varyantla bulunduğu anlam listesi için gerekli:
                    # kök varyantının ("terlik" -> "ter") anlamı sorgunun anlamı değildir.
                    results.append((var, res))
            elapsed = int((time.perf_counter() - started) * 1000)
            if not results and not errors:
                failure = unanswered_status(requests_log)
                if failure:
                    return fetcher, results, elapsed, failure[1], failure[0]
            status = "ok" if results else ("error" if errors else "empty")
            return fetcher, results, elapsed, errors, status

        fetcher_order = {f.source_name: i for i, f in enumerate(self.fetchers)}
        hypothesis_historical_meaning = ""
        # Starling'in Proto-Türkçe biçimi (kelimenin kendi sözlük kaydı biçim
        # vermiyorsa başlıkta gösterilir; bkz. aşağıdaki kaynak kökü adımı).
        starling_root = ""

        with concurrent.futures.ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
            # Sonuçlar PORTFÖY SIRASIYLA işlenir, bitiş sırasıyla değil.
            # `as_completed` ile ilk biten kaynak başlık kökünü (`proto_root`)
            # ve aynı (dil, kelime) anahtarlı tanığı kapıyordu; aynı kelime
            # koşudan koşuya farklı kökle çıkabiliyordu. İşleme hafiftir,
            # istekler yine paralel koşar.
            future_to_fetcher = {executor.submit(fetch_worker, f): f for f in self.fetchers}
            for future, fetcher in future_to_fetcher.items():
                try:
                    _fetcher_obj, results, elapsed_ms, errors, status = future.result()
                    source_diagnostics[fetcher.source_name] = {
                        # ok | empty (cevap geldi, veri yok) | error | circuit_open
                        "status": status,
                        "duration_ms": elapsed_ms,
                        "result_count": len(results),
                        "errors": errors or None,
                    }
                    if not results:
                        logger.debug("Kaynak veri döndürmedi: %s (%d ms)", fetcher.source_name, elapsed_ms)
                    for variant, res in results:
                        raw_fetcher_results.append(res)
                        root_info = res.get("root", {})
                        # Kök varyantının ("kulluk" -> "kul") Starling kökü
                        # sorgunun kökü değildir.
                        if root_info.get("starling_proto") and not starling_root and variant == word_clean:
                            starling_root = str(root_info.get("proto_turkic") or "")
                        if root_info.get("proto_turkic") and not proto_root:
                            proto_root = root_info.get("proto_turkic")
                            proto_root_provenance = f"tanıklı — {fetcher.source_name}"
                        source_meanings = meanings_by_source.setdefault(fetcher.source_name, [])
                        if variant == word_clean:
                            primary = translate_meaning(root_info.get("meaning") or "")
                            if primary and not is_cross_reference(primary):
                                primary_by_source.setdefault(fetcher.source_name, primary)
                            # Kaynak anlamları tek tek verdiyse (TDK) onlar, yoksa tek alan.
                            for m in root_info.get("meanings") or [root_info.get("meaning") or ""]:
                                _add_meaning(source_meanings, translate_meaning(m))

                        for entry in res.get("turkic_languages", []):
                            entry["meaning"] = translate_meaning(entry.get("meaning", ""))
                            entry["phonetic_shift"] = analyze_phonetic_shifts(
                                word_clean, entry.get("word", ""), entry.get("lang_name", "")
                            )
                            # Kiril/Arap yazımlı biçimlerin Latin okunuşu
                            # (README'nin vaat ettiği transkripsiyon motoru;
                            #  daha önce import edilip hiç çağrılmıyordu)
                            if entry.get("script") in ("Cyrillic", "Arabic", "Runic"):
                                entry["latin_transliteration"] = transliterate_to_latin(entry.get("word", ""))
                            # Ağız kayıtlarındaki coğrafi etiket (TDK Derleme: "(Sinop)")
                            geo = tag_geographical_region(entry.get("lang_name", "") + " " + (entry.get("meaning") or ""))
                            # Yalnızca GERÇEK bir bölge tespit edildiyse ekle;
                            # "Genel Türki Coğrafya" gibi jenerik yedek etiket bilgi taşımaz.
                            if geo and geo.get("geo_coordinates"):
                                entry["geo"] = geo
                            # Sözlük tanımlarındaki "-> herkil" çapraz göndermeleri
                            refs = extract_cross_references(entry.get("meaning") or "")
                            if refs:
                                entry["cross_references"] = refs
                            # Yalnız sorgunun KENDİ dilinin kaydı sorgunun anlamıdır;
                            # kardeş dilin eşyazımlı kelimesi ve ağız kaydı değil
                            # (bkz. `OWN_LINE_CODES`).
                            if _is_own_record(entry, word_clean):
                                _add_meaning(source_meanings, entry.get("meaning") or "")
                            # Ağız kaydı ile ölçünlü dil kaydı aynı (dil, kelime)
                            # çiftini taşıyabiliyor; ayrı tutulmazsa hangisinin
                            # kalacağını bitiş sırası belirliyordu.
                            key = (entry["lang_code"], entry["word"], bool(entry.get("dialect")))
                            if key not in turkic_entries_map:
                                turkic_entries_map[key] = entry
                            elif turkic_entries_map[key].get("meaning") in ["", f"Online {TURKIC_LANGUAGES_MAP.get(entry['lang_code'], '')} Sözlük kaydı"]:
                                if entry.get("meaning") and not entry.get("meaning").startswith("Online"):
                                    turkic_entries_map[key] = entry

                        if res.get("turkic_languages") or root_info.get("proto_turkic"):
                            sources.append(fetcher.source_name)
                except Exception as exc:
                    logger.warning("Fetcher sonucu işlenemedi: %s", fetcher.source_name, exc_info=True)
                    source_diagnostics[fetcher.source_name] = {
                        "status": "error",
                        "duration_ms": 0,
                        "result_count": 0,
                        "errors": [f"{type(exc).__name__}: {exc}"],
                    }
        _lap("fetch")

        # Ana anlam SABİT bir öncelikle seçilir: ölçünlü TDK sözlüğü önce,
        # sonra fetcher portföyünün sırası. Bitiş sırası artık belirleyici değil.
        meanings_by_source = {k: v for k, v in meanings_by_source.items() if v}
        primary_source = ""
        for source_name in sorted(
            meanings_by_source,
            key=lambda n: (n != _PRIMARY_MEANING_SOURCE, fetcher_order.get(n, len(fetcher_order))),
        ):
            candidate = next(
                (m for m in [primary_by_source.get(source_name, ""), *meanings_by_source[source_name]]
                 if m and m != word_clean),
                "",
            )
            if candidate:
                root_meaning = candidate
                primary_source = source_name
                break

        # Kaynağın AÇIKÇA akraba dediği kayıtlar tanık olur (bkz.
        # `_asserted_cognates`). Ters bağlantı araması burada bir kez yapılır.
        etymology_mentions = (
            self._etymology_mentions(word_clean) if self.uses_lexicon_index else {"total": 0, "items": []}
        )
        # Eşsesli akrabaları ayırmak için sorgunun ANA anlamı (başlıktaki
        # sözlükbirim). ⚠️ Bütün anlamlar kullanılamaz: sorgunun kendisi
        # eşsesli olabilir — Tarama/Derleme `ekmek` için "tohum atmak"
        # anlamını da veriyor ve Gagavuzca *ekmää* "to sow" bu yüzden
        # süzgeçten geçiyordu (ölçüldü).
        # Ana kaynağın İLK anlamı: TDK'nın birleşik alanı ikinci anlamları da
        # taşıyor ("İnsanı geçindirecek iş; kazanç") ve süzgeci gevşetiyordu
        # (ölçüldü: *eyləmək* "yapmak" `el` için 0.443, *ekmää* 0.398).
        query_meanings = meanings_by_source.get(primary_source, [])[:1] or [
            m for group in meanings_by_source.values() for m in group
        ]
        english_gloss = _english_query_gloss(
            word_clean, list(turkic_entries_map.values()),
            primary=(meanings_by_source.get(primary_source) or [""])[0],
        )
        if english_gloss and english_gloss not in query_meanings:
            query_meanings = [*query_meanings, english_gloss]

        # Yerel çağdaş dil kayıtları yazılışla bulundu; akraba oldukları ancak
        # ANLAMLA doğrulanır (bkz. `ModernIndexFetcher`). Doğrulanamayan
        # (anlamsız kayıt, model ya da sorgu anlamı yok) tanık OLMAZ.
        unverified = [e for e in turkic_entries_map.values() if e.get("meaning_check")]
        if unverified:
            glossed = [e for e in unverified if e.get("meaning")]
            kept, _ = _homonym_filter(glossed, query_meanings) if glossed else ([], [])
            scored = [e for e in kept if "meaning_similarity" in e]
            # İki kesim: mutlak alt sınır (0,50) ve en iyi eşleşmeye göre göreli
            # kesim (`el`: "hand" adayları 1,0, "people" eşseslileri 0,43).
            best = max((e["meaning_similarity"] for e in scored), default=0.0)
            floor = max(LOCAL_WITNESS_FLOOR, best - LOCAL_WITNESS_MARGIN)
            verified = {id(e) for e in scored if e["meaning_similarity"] >= floor}
            turkic_entries_map = {
                k: v for k, v in turkic_entries_map.items()
                if not v.get("meaning_check") or id(v) in verified
            }
            etymology_mentions["local_witnesses"] = {
                "found": len(unverified), "verified": len(verified),
            }
        asserted, homonyms = _homonym_filter(
            _asserted_cognates(word_clean, etymology_mentions), query_meanings
        )
        etymology_mentions["homonym_cognates"] = [
            {"lang_name": h["lang_name"], "word": h["word"], "meaning": h["meaning"],
             "similarity": h.get("meaning_similarity")}
            for h in homonyms
        ]
        for entry in asserted:
            key = (entry["lang_code"], entry["word"], False)
            if key not in turkic_entries_map:
                entry["phonetic_shift"] = analyze_phonetic_shifts(
                    word_clean, entry["word"], entry["lang_name"]
                )
                if entry.get("script") in ("Cyrillic", "Arabic", "Runic"):
                    entry["latin_transliteration"] = transliterate_to_latin(entry["word"])
                turkic_entries_map[key] = entry
                if ASSERTED_COGNATE_SOURCE not in sources:
                    sources.append(ASSERTED_COGNATE_SOURCE)

        present = {(e["lang_code"], e.get("comparison") or to_comparison_form(e.get("word") or ""))
                   for e in turkic_entries_map.values()}
        for entry in _cited_cognates(word_clean, list(turkic_entries_map.values())):
            if (entry["lang_code"], entry["comparison"]) in present:
                continue
            present.add((entry["lang_code"], entry["comparison"]))
            entry["phonetic_shift"] = analyze_phonetic_shifts(word_clean, entry["word"], entry["lang_name"])
            if entry.get("script") in ("Cyrillic", "Arabic", "Runic"):
                entry["latin_transliteration"] = transliterate_to_latin(entry["word"])
            turkic_entries_map[(entry["lang_code"], entry["word"], False)] = entry
            if CITED_COGNATE_SOURCE not in sources:
                sources.append(CITED_COGNATE_SOURCE)

        sorted_entries = sorted(
            list(turkic_entries_map.values()),
            key=lambda x: (0 if x["lang_code"] == "otk" else (0.3 if x["lang_code"] == "ai" else (0.5 if x["lang_code"] == "donor" else 1)), x["lang_name"])
        )

        # 4. KÖKEN NLP VE OTONOM İNATÇI HİPOTEZ REKONSTRÜKSİYONU
        # Eğer kök anlamı henüz atanmadıysa sorted_entries içindeki gerçek sözlük tanımından çek
        if not root_meaning or root_meaning == word_clean:
            # Yedek de yalnız sorgunun kendi kaydından: ilk sıradaki kayıt
            # başka bir Türk dilinin (ya da vericinin) kaydı olabilir.
            for entry in sorted_entries:
                if not _is_own_record(entry, word_clean):
                    continue
                m = entry.get("meaning", "").strip()
                if m and not m.startswith("Online") and m != word_clean and not is_cross_reference(m):
                    root_meaning = m
                    break
        _lap("witness_filter")

        # Katman 2 (çapraz lehçe yayılımı) önce hesaplanır; Katman 1'e girdi olur.
        cognate_eval = self.cognate_alignment_engine.evaluate_cognate_distribution(word_clean, sorted_entries)
        loan_eval = self.loanword_classifier.classify(
            word_clean, spreading_ratio=cognate_eval.get("spreading_ratio")
        )
        _apply_source_loan_family(loan_eval, sorted_entries)
        # 4 katmanlı alıntı keşif hattı (master plan Katman 1-4)
        loanword_detection = self.loanword_detector.detect(word_clean, sorted_entries)
        # Çoklu dizi hizalama ile akraba kümeleri (plan §2.1'in asıl hedefi)
        cognate_clusters = self.cognate_cluster_engine.cluster(sorted_entries)
        # Tarihsel yapım eki ağacı (plan §2.5)
        historical_morphology = self.historical_morphology.build_tree(word_clean)

        # Morfolojik kök rekonstrüksiyona GEÇİRİLİR. Eskiden `build_tree`
        # sonucu yalnız çıktı sözlüğüne konuyor, rekonstrüktöre ham kelime
        # gidiyordu; sözlük madde başı `içmek` için `*içmek` üretiliyordu.
        #
        # ⚠️ ÖLÇÜM NOTU. Altın standart üzerinde ek soymanın katkısı **net
        # sıfır** çıktı (289 kez ateşlendi: 6 iyileşme, 6 bozulma). Sebep
        # ölçüm verisinin doğası: CLDF biçimleri zaten ÇIPLAK KÖK'tür, sözlük
        # madde başı değil. Yani ölçüm bu katmanı sınayamıyor.
        #
        # 2026-09-21 doğrulaması: `-mA` eki eklenip soyma tanıklığa
        # bağlandıktan sonra `harness --split dev` yeniden koşuldu —
        # NED 0,302 · tam 0,3855 · BCFS 0,5951, yani taban çizgisiyle BİREBİR
        # aynı. Sebebi ölçüldü: dev kümesindeki **83 maddenin 0'ı** soyma için
        # gereken asgari kelime uzunluğunu (6) bile geçmiyor, dolayısıyla
        # katman bu veride HİÇ ateşlenmiyor. "İyileşme yok" değil,
        # "ölçüm sınayamıyor" — bu yüzden katman geri alınmadı.
        #
        # Gerçek kullanımda girdi TDK madde başıdır (`içmek`, `üzerinde`) ve
        # orada soymak şart. Bu yüzden katman kalıyor ama muhafazakâr bir
        # korumayla: yalnız çok heceli kelimelerde ve makul uzunlukta bir kök
        # bırakıyorsa. Koruma olmadan `yan -> ya`, `karın -> kar` gibi aşırı
        # soymalar oluyordu.
        MIN_WORD_FOR_STRIPPING = 6
        MIN_ROOT_AFTER_STRIPPING = 4
        morphological_root = str(historical_morphology.get("root") or "")
        def _root_is_attested(root: str) -> bool:
            """Soyulan kök gerçekten var mı? Önce tanıklar, sonra sözlük indeksi.

            Uzunluk koruması tek başına yetmiyordu: `bardak` (7 harf) -> `barda`
            (5 harf) her iki eşiği de geçiyor ama `barda` diye bir kök yok ve
            rekonstrüksiyon yanlış girdiyle çalışıyordu.
            """
            target = root.strip().lower().rstrip("-")
            if not target:
                return False
            if any(
                (entry.get("word") or "").strip().lower().rstrip("-") == target
                for entry in sorted_entries
            ):
                return True
            try:
                from engine.db.lexicon_index import LexiconIndex

                # Kural tek yerde: `LexiconIndex.is_attested_stem`. Burada
                # ikinci bir kopya tutmak ikisinin zamanla ayrışması demekti.
                index = LexiconIndex()
                if not index.exists:
                    return True  # indeks yoksa eski davranış (yalnız uzunluk)
                return index.is_attested_stem(target)
            except Exception:
                logger.debug("Kök tanıklık denetimi yapılamadı: %s", target, exc_info=True)
                return True

        strip_ok = (
            morphological_root
            and morphological_root != word_clean
            and len(word_clean) >= MIN_WORD_FOR_STRIPPING
            and len(morphological_root) >= MIN_ROOT_AFTER_STRIPPING
            and _root_is_attested(morphological_root)
        )
        reconstruction_input = morphological_root if strip_ok else word_clean
        # Fiil mastarı: sorgudan -mAk, tanıklardan dilin kendi mastar eki
        # (`gülmək`, `күлүү`, `kulmoq`) soyulur; alıntı denetimi tam mastarla
        # yapılır. Ölçüldü (406 miras fiil, 2026-09-24 ağacı): NED 0,427 -> 0,311,
        # tam 0,212 -> 0,357 (ΔNED %95 GA −0,133…−0,100).
        # Bkz. `historical_morphology.infinitive_stem`.
        verb_stem = infinitive_stem(word_clean, root_meaning)
        reconstruction_witnesses = sorted_entries
        if verb_stem:
            reconstruction_input = verb_stem
            reconstruction_witnesses = [
                {**e, "word": strip_infinitive(e.get("lang_code") or "", e.get("word") or "")}
                for e in sorted_entries
            ]
        reconstruction_eval = self.reconstructor.reconstruct_proto_form(
            reconstruction_input, reconstruction_witnesses,
            borrowing_word=word_clean if verb_stem else "",
        )
        if reconstruction_input != word_clean:
            reconstruction_eval["stripped_from"] = word_clean
            reconstruction_eval["stripped_suffixes"] = ["-mAk"] if verb_stem else [
                layer.get("suffix") for layer in historical_morphology.get("layers", [])
            ]

        # Rakip hipotezler ve red gerekçeleri (Faz 9). Reddedilen köken
        # önerileri çıktıda KALIR; gerekçesiyle birlikte.
        ranked_hypotheses = self._rank_hypotheses(word_clean, sorted_entries)
        # Hüküm alıntıysa çeviri/indeks tanıkları paralel alıntıdır: bundan
        # sonraki miras/yayılım kanıtına (A-HVP üçgenlemesi, ses kanunu
        # indüksiyonu, akraba listesi, yayılım raporu) girmez.
        evidence_entries = sorted_entries
        if ((ranked_hypotheses or {}).get("selected") or {}).get("kind") == "borrowed":
            evidence_entries = _mark_parallel_loans(sorted_entries, {
                f.source_name for f in self.fetchers
                if isinstance(f, (ApertiumFetcher, NorthEuraLexFetcher, ModernIndexFetcher))
            })
            parallel_count = len(sorted_entries) - len(evidence_entries)
            if parallel_count:
                cognate_eval = self.cognate_alignment_engine.evaluate_cognate_distribution(
                    word_clean, evidence_entries
                )
                cognate_eval["parallel_loans_excluded"] = parallel_count
        donor_eval = self.donor_search_engine.search_donor_neighbors(word_clean)

        finding_temp = {"root": {"proto_turkic": proto_root, "meaning": root_meaning}}
        # Hipotez motorlarına GERÇEK akraba kayıtları ve ham fetcher çıktıları verilir;
        # eskiden bu veriler geçilmiyor, motorlar kendi ürettikleri varyantları
        # "kanıt" sayıyordu.
        proven_hypothesis_eval = self.hypothesis_engine.prove_etymological_hypothesis(
            word_clean, finding_temp, evidence_entries, raw_fetcher_results
        )
        unattested_prover_eval = self.hypothesis_prover.prove_unattested_word(word_clean, evidence_entries)

        # Sözlüklerden kök bulunamadıysa karşılaştırmalı rekonstrüksiyona başvur.
        # ÖNEMLİ: kanıt yoksa `*<kelime>` biçiminde bir kök UYDURULMAZ.
        if not proto_root or proto_root == word_clean:
            hypo_pr = unattested_prover_eval.get("proven_hypothesis")
            if hypo_pr and hypo_pr.get("origin_form"):
                proto_root = hypo_pr["origin_form"]
                proto_root_provenance = "türetilmiş — tanıksız kelime kanıtlayıcısı"
                if not proven_hypothesis_eval.get("hypothesis_available"):
                    proven_hypothesis_eval["proven_hypothesis"] = hypo_pr
                    proven_hypothesis_eval["hypothesis_available"] = True
            elif reconstruction_eval.get("evidence_available"):
                proto_root = reconstruction_eval.get("reconstructed_root", "")
                proto_root_provenance = "türetilmiş — karşılaştırmalı yöntem"

        lingpy_eval = self.lingpy_aligner.align_sequences(proto_root or word_clean, word_clean)
        # Tarihsel anlam, EN ESKİ TANIĞIN anlamıdır. Eskiden modern anlam iki
        # kez geçiliyordu — yani bir anlam kendisiyle karşılaştırılıyor ve
        # sonuç tanımı gereği "kayma yok" çıkıyordu; `diachronic_semantic_drift`
        # alanı anlamsız veri taşıyordu. `sorted_entries` tarihî katmanı başa
        # sıraladığı için ilk tarihî tanık buradan alınır. Tanık yoksa boş
        # geçilir: motor zaten `evidence_available: False` döndürür
        # (diachronic_semantic_engine.py:152-164), uydurma skor üretmez.
        # ⚠️ Burada seçim KALİTE SÜZGECİSİZDİ ve "ilk boş olmayan anlam"ı
        # alıyordu; kitap tarama artıkları ile runik harf adları tarihî
        # anlam diye geçiyordu (su -> "Kitap: 3 Bogatyr bikers…" 0.9523,
        # baş -> "A letter of the Old Turkic runic script…" 0.7996).
        # Süzgeç `_historical_gloss` içinde tek yerde toplandı; A-HVP dalı
        # da aynı yardımcıyı kullanıyor, iki yol ayrışmasın.
        historical_meaning = _historical_gloss(sorted_entries, word_clean)
        semantic_eval = self.semantic_engine.evaluate_diachronic_trajectory(
            historical_meaning, root_meaning or ""
        )

        # Gerçek ses kanunu indüksiyonu: TÜM akraba çiftleri üzerinden.
        # Eskiden tek çiftten sabit 0.95 güven skoru üretiliyordu.
        induction_pairs = [
            (proto_root or word_clean, e["word"])
            for e in evidence_entries
            if e.get("word") and e.get("lang_code") in TURKIC_LANGUAGES_MAP
        ]
        sound_law_induced = (
            self.sound_law_induction.induce_from_pairs(induction_pairs)
            if len(induction_pairs) >= 2
            else self.sound_law_induction.induce_sound_law(proto_root or word_clean, word_clean)
        )


        if donor_eval and donor_eval.get("found_match"):
            donor_lang = donor_eval.get("donor_language")
            origin_form = donor_eval.get("origin_form")
            donor_meaning = donor_eval.get("donor_meaning")
            proto_root = f"[{donor_lang}] {origin_form}"
            # ⚠️ Tohum dosyası TANIKLIK DEĞİLDİR. `data/seed/donor/
            # donor_etymology.json` elle yazılmış 10 kayıtlık bir çekirdek
            # (herkil, herkel, harkil, efendi, rüzgar, kitap, kalem, dünya,
            # sümen, televizyon). Ölçüldü: `herkil` bu dosyanın ilk maddesi ve
            # rapor onu "tanıklı — donör dil etimoloji veritabanı" diye,
            # sözlük tanıklığıymış gibi gösteriyordu.
            _donor_source = str(donor_eval.get("evidence_source") or "donör veritabanı")
            _donor_is_seed = "tohum" in _donor_source.lower()
            proto_root_provenance = (
                f"{'tohum verisi' if _donor_is_seed else 'tanıklı'} — {_donor_source}"
            )
            sources.append(f"Donör Dil Etimoloji Veritabanı ({donor_lang})")
            sorted_entries.insert(0, {
                "lang_code": "donor",
                "lang_name": f"Kaynak Dil Etimolojisi ({donor_lang})",
                "word": origin_form,
                "meaning": donor_meaning,
                "script": "Original"
            })

        _hypo = proven_hypothesis_eval.get("proven_hypothesis") or {}
        _report = _hypo.get("validation_report") or {}
        # Eskiden sabit 0.95 eşiği vardı ve pratikte hiç tetiklenmiyordu.
        # Artık A-HVP rozetine bakılır.
        if _report.get("status_code") in ("VALIDATED", "NEEDS_REVIEW") and _hypo.get("donor_language"):
            hypo = _hypo
            proto_root = hypo.get("origin_form") or proto_root
            # ⚠️ Bu dal MİRAS kelimelerde de ateşleniyor: `göz` için
            # `donor_language` alanı "Proto-Türkçe" geliyor ve Proto-Türkçe
            # bir verici dil DEĞİLDİR. Ayrım yapılmazsa miras kök "alıntı
            # kökeni" diye damgalanır (ölçüldü: göz -> yanlış damga).
            _donor_lang = str(hypo.get("donor_language") or "")
            # ⚠️ Rozet "inceleme gerekli" (NEEDS_REVIEW) iken köken satırı
            # "doğrulanmış" diyordu; 60 kelimelik taramada 6 kelimede çelişki.
            _verdict = "doğrulanmış" if _report.get("status_code") == "VALIDATED" else "inceleme gerektiren"
            _inherited = _donor_lang in ("", "Proto-Türkçe", "Ana Türkçe", "Öz Türkçe", "Eski Türkçe")
            if _inherited:
                proto_root_provenance = (
                    f"türetilmiş — A-HVP {_verdict} miras kökü ({_donor_lang or 'Türki'})"
                )
            else:
                proto_root_provenance = (
                    f"tanıklı — A-HVP {_verdict} alıntı kökeni ({_donor_lang})"
                )
            # ⚠️ Eskiden burada `root_meaning` EZİLİYORDU: başlıktaki anlam
            # modern sözlük tanımı yerine hipotezin tarihî anlamı oluyordu
            # (`terlik` -> "bk. derlik"). Tarihî anlam ayrı alanda taşınır.
            hypothesis_historical_meaning = hypo.get("historical_meaning") or ""
            # ⚠️ Miras kökte A-HVP'nin ata biçimi motorun KENDİ
            # rekonstrüksiyonudur: ne bir kaynak ne de bir "kaynak dil".
            # Eskiden burada da portföyde olmayan "Derin Komşu Diller
            # Etimoloji Veritabanı (Proto-Türkçe)" kaynağı yazılıyor ve
            # *uça gibi tahminler "Köken zinciri — kaynak diller" altında
            # tanık gibi basılıyordu (105 kelimelik denetim: 51 miras
            # kelimenin 32'sinde). Ata biçimi başlıkta, kaynağı damgada kalır.
            if not _inherited:
                sources.append(f"Derin Komşu Diller Etimoloji Veritabanı ({hypo.get('donor_language')})")
            if not _inherited and not any(e.get("lang_code") == "donor" for e in sorted_entries):
                sorted_entries.insert(0, {
                    "lang_code": "donor",
                    "lang_name": f"Kaynak Dil Etimolojisi ({hypo.get('donor_language')})",
                    "word": hypo.get("origin_form"),
                    "meaning": hypo.get("proof_summary"),
                    "script": "Original"
                })

        timeline = []
        for entry in sorted_entries:
            lname = entry.get("lang_name", "")
            if "Divanü Lugati't-Türk" in lname or "1074" in lname or "Orhun" in lname or "Eski Türkçe" in lname or "İSAM" in lname:
                timeline.append(f"M.Ö. III. YY - 11. YY (Hun / Orhun / DLT / İSAM): {entry.get('word')} - {(entry.get('meaning') or '')[:60]}")
            elif "1303" in lname or "Codex Cumanicus" in lname:
                timeline.append(f"14. YY (Kıpçakça / Codex Cumanicus): {entry.get('word')}")
            elif "1901" in lname or "Kamus-ı Türkî" in lname or "13.-19." in lname or "Osmanlıca Lügat" in lname:
                timeline.append(f"19. YY (Osmanlıca / Lehçe-i Osmanî / Kamus-ı Türkî): {entry.get('word')}")

        _lap("nlp")

        morphology_info = f"Kök: {stem} + Ekler: {', '.join(suffixes)}" if suffixes else "Yalın Kök"
        # Sorgu kelimesinin yapısını açıkça veren sözlük maddesi ("biti- + -g").
        # Yalnız sorgunun kendi dil çizgisinin maddesi: Azerice `qaçmaq + -ır`
        # Türkçe `kaçırmak`ın yapısı diye basılıyordu (denetim: 9 kelime).
        formation_entry = next(
            (e for e in sorted_entries if e.get("formation") and _is_own_record(e, word_clean)),
            None,
        )
        if not suffixes and formation_entry:
            # Kural tabanlı çözümleyici Eski Türkçe eklerini tanımıyor (`bitig`
            # -> "Yalın Kök"); sözlük maddesi yapıyı açıkça veriyorsa o yazılır.
            morphology_info = f"{formation_entry['formation']} (sözlük maddesine göre)"
        related_cognates = get_related_cognates(word_clean, evidence_entries)

        _lap("report")
        # 5. Neo4j Uyumlu Graf Veritabanı Düğüm Şeması Oluşturma
        graph_export = self.graph_db.build_etymology_graph(
            word=word_clean,
            root_form=proto_root or word_clean,
            hypothesis=proven_hypothesis_eval.get("proven_hypothesis", {}),
            attestations=timeline,
            cognates=related_cognates
        )
        _lap("graph")

        # ⚠️ BAŞLIK İLE HÜKÜM AYRI KAYNAKLARDAN BESLENİYORDU.
        #
        # `proto_root` yukarıda BEŞ ayrı yerde atanıyor (fetcher, tanıksız
        # kanıtlayıcı, karşılaştırmalı rekonstrüksiyon, donör veritabanı,
        # A-HVP). Sıralayıcı ise bağımsız çalışıp bir hipotez SEÇİYOR. İkisi
        # çeliştiğinde kullanıcının İLK GÖRDÜĞÜ satır yanlış olanı
        # gösterebiliyordu. Ölçüldü:
        #
        #   pinti  -> başlık "*pinti" (yıldızlı Proto-Türkçe kök) ama
        #             sıralayıcı "ALINTI — Ermenice" seçiyor ve MİRAS'ı
        #             reddediyor; aynı çıktının NLP bölümü de "Proto-Türkçe
        #             rekonstrüksiyon uygulanmaz" diyor. Üç bölüm birbirini
        #             yalanlıyordu. (Doğrusu: Ermenice փնթի "pis, murdar".)
        #   herkil -> başlık Ermenice kökeni kesin gibi veriyor, sıralayıcı
        #             "Kökeni belirlenemedi" diyor.
        #
        # Hüküm sıralayıcınındır; başlık onu İZLER.
        _selected = (ranked_hypotheses or {}).get("selected") or {}
        _sel_kind = _selected.get("kind")
        _sel_detail = _selected.get("detail") or {}
        _sel_claim = _selected.get("claim") or ""
        _ranker_root = ""
        if _sel_kind == "borrowed":
            # Alıntıda yıldızlı ata biçim göstermek yanlıştır: zincirin son
            # halkası verici dil + özgün biçimdir (ör. "Ermenice փնթի").
            _chain = [link for link in (_sel_detail.get("chain") or []) if link]
            _ranker_root = _chain[-1] if _chain else ""
        elif _sel_kind == "inherited":
            _ranker_root = str(_sel_detail.get("reconstructed_root") or "")

        if _ranker_root and _ranker_root != proto_root:
            proto_root = _ranker_root
            proto_root_provenance = f"sıralayıcı hükmü — {_sel_claim}"
        elif _sel_kind not in ("borrowed", "inherited") and proto_root:
            # Sıralayıcı bir köken seçemediyse başlık kesinlik iddia edemez.
            proto_root_provenance = (
                f"{proto_root_provenance or 'kaynak belirsiz'} "
                f"⚠️ sıralayıcı: {_sel_claim or 'kökeni belirlenemedi'}"
            )

        # Kaynağın kendi miras kaydı Proto-Türkçe biçimi AÇIKÇA veriyorsa
        # başlık onu gösterir; motorun rekonstrüksiyonu NLP bölümünde kalır.
        # Sıralayıcı alıntı dediyse dokunulmaz. Yalnız rapordur: skorlar
        # yukarıda hesaplandı.
        source_root, source_root_lang = _query_source_proto(
            word_clean, sorted_entries, primary=(meanings_by_source.get(primary_source) or [""])[0]
        )
        source_label = f"{source_root_lang} sözlük kaydı"
        if not source_root and starling_root:
            source_root, source_label = starling_root, "Starling (Dybo & Starostin 2005)"
        if not source_root:
            # Starling'den SONRA: önce yerleştirilince Starling'li ayarda
            # savelyev dev tam 0,656 -> 0,594 düşüyordu.
            source_root, index_lang = _index_source_proto(
                word_clean, primary=(meanings_by_source.get(primary_source) or [""])[0]
            )
            source_label = f"{index_lang} sözlük kaydı (yerel indeks)"
        if source_root and _sel_kind != "borrowed" and source_root != proto_root:
            engine_root = proto_root
            proto_root = source_root
            proto_root_provenance = (
                f"tanıklı — {source_label} (Proto-Türkçe {source_root})"
                + (f"; motorun rekonstrüksiyonu: {engine_root}" if engine_root else "")
            )

        # Motor hiçbir yöntemle kök bulamadıysa ama sözlük maddesi yapıyı
        # veriyorsa, kök o yapının ilk parçasıdır (`bitig` -> `biti-`).
        # ⚠️ Skorlar hesaplandıktan SONRA atanır: yalnız rapor içindir,
        # A-HVP ve sıralayıcının girdisini değiştirmez.
        root_note = ""
        if not proto_root and formation_entry:
            base = formation_entry["formation"].split(" + ")[0].strip()
            if base and not base.startswith("-"):
                proto_root = base
                proto_root_provenance = (
                    f"tanıklı — sözlük maddesinin yapısı ({formation_entry['formation']}; "
                    f"{formation_entry.get('lang_name') or formation_entry.get('lang_code')})"
                )
                root_note = (
                    _root_note(base, str(formation_entry.get("lang_code") or ""))
                    if self.uses_lexicon_index else ""
                )
                # Rekonstrüksiyon notu "aşağıdaki biçim sorgu kelimesinin
                # kendisidir" diyordu; başlık artık başka bir kök gösterdiği
                # için iki satır birbirini yalanlıyordu.
                reconstruction_eval["reconstruction_notes"] = (
                    "Karşılaştırmalı yöntem uygulanamadı (yeterli bağımsız tanık yok). "
                    f"Başlıktaki kök ({base}) karşılaştırmalı yöntemle TÜRETİLMEDİ; "
                    f"sözlük maddesinin verdiği yapıdan alındı: {formation_entry['formation']}."
                )

        origin_layers = _origin_layers(sorted_entries, word_clean, formation_entry)
        source_proto = _source_proto_forms(sorted_entries)

        _lap("headline")
        # (Sorgu TDK'da yoksa) ters bağlantılı maddelerin canlı sonuçları.
        # ⚠️ Anlam seçiminden ve skorlardan SONRA: yalnız rapor.
        if _PRIMARY_MEANING_SOURCE not in meanings_by_source:
            for source_name, meaning in self._consult_descendants(word_clean, etymology_mentions):
                bucket = meanings_by_source.setdefault(source_name, [])
                _add_meaning(bucket, meaning)
        _lap("descendants")

        # Başlık anlamı TDK'dan gelmediyse ve boş/İngilizceyse indeksteki
        # Türkçe kayıt gösterilir. ⚠️ Yalnız GÖSTERİM: `root_meaning` A-HVP
        # 3. aşamasının ve eşsesli süzgecinin girdisidir, değiştirilmez.
        display_meaning = root_meaning or word_clean
        if primary_source != _PRIMARY_MEANING_SOURCE and (
            display_meaning == word_clean or looks_english(display_meaning)
        ):
            turkish_gloss = _index_turkish_gloss(word_clean)
            if turkish_gloss:
                display_meaning = turkish_gloss
                _add_meaning(meanings_by_source.setdefault(_INDEX_TR_MEANING_SOURCE, []), turkish_gloss)

        finding = {
            "query_word": word_clean,
            "morphology": morphology_info,
            "turkic_languages": sorted_entries,
            "etymology_mentions": etymology_mentions,
            "root": {
                "proto_turkic": proto_root or word_clean,
                "meaning": display_meaning,
                # Tek anlam seçmek bilgi kaybıydı; her sözlüğün anlamı kendi
                # adıyla, portföy sırasıyla (TDK önce).
                "meanings": [
                    {"source": name, "meanings": meanings_by_source[name]}
                    for name in sorted(
                        meanings_by_source,
                        key=lambda n: (n != _PRIMARY_MEANING_SOURCE, fetcher_order.get(n, len(fetcher_order))),
                    )
                ],
                "historical_meaning": hypothesis_historical_meaning or historical_meaning or "",
                # Kök hiç atanmadıysa sorgu kelimesi yazılıyor; bu bir bulgu
                # değildir, o yüzden damgası da "yok".
                "provenance": proto_root_provenance or "yok — kök belirlenemedi",
                "root_note": root_note,
                "origin_layers": origin_layers,
                "source_proto_forms": source_proto,
                "reconstruction_notes": reconstruction_eval.get("reconstruction_notes", "")
            },
            "nlp_analysis": {
                "loanword_classification": loan_eval,
                "cognate_distribution": cognate_eval,
                "reconstruction": reconstruction_eval,
                "donor_matching": donor_eval,
                "proven_hypothesis": proven_hypothesis_eval.get("proven_hypothesis"),
                "unattested_word_reconstruction": unattested_prover_eval,
                "lingpy_alignment": lingpy_eval,
                "diachronic_semantic_drift": semantic_eval,
                "induced_sound_laws": sound_law_induced,
                "loanword_detection": loanword_detection,
                "cognate_clusters": cognate_clusters,
                "historical_morphology": historical_morphology,
                "ranked_hypotheses": ranked_hypotheses,
            },
            "graph_database": graph_export,
            "timeline": list(dict.fromkeys(timeline)),
            "related_cognates": related_cognates,
            "sources": sorted(set(sources)),
            "from_cache": False,
        }

        _lap("assemble")
        if use_qwen_agent:
            finding = self.qwen_agent.research_and_enrich(word_clean, finding)
            _lap("ai_enrichment")

        # Gerçek telemetri: web panelindeki sahte setTimeout simülasyonunun yerini alır.
        stage_timings["total"] = int((time.perf_counter() - search_started) * 1000)
        finding["diagnostics"] = {
            "stage_timings_ms": stage_timings,
            "sources": source_diagnostics,
            "http": diagnostics.summary(),
            "variants_used": search_variants,
            "live_source_count": sum(1 for d in source_diagnostics.values() if d["status"] == "ok"),
        }
        logger.info(
            "Arama tamamlandı: %r — %d ms, %d/%d kaynak veri döndürdü",
            word_clean, stage_timings["total"],
            finding["diagnostics"]["live_source_count"], len(self.fetchers),
        )

        if save_to_db:
            self.db.save_finding(finding)

        return finding
