"""
Merkezî Yapılandırma (Central Configuration)

Projedeki tüm ağ adresleri, zaman aşımları, model adları, eşik değerleri ve
dosya yolları burada tek kaynaktan yönetilir. Her değer `ETY_` önekli bir ortam
değişkeniyle ezilebilir.

Bu modül, daha önce 32 ayrı `timeout=`, 32 gömülü URL ve 11 farklı User-Agent
olarak dağılmış olan sabitleri tek yerde toplar.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Yardımcılar -----------------------------------------------------------

def _env_str(name: str, default: str) -> str:
    return os.environ.get(f"ETY_{name}", default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(f"ETY_{name}", default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(f"ETY_{name}", default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(f"ETY_{name}")
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "evet"}


# --- Yollar ----------------------------------------------------------------

#: **Motor sürümü — önbellek geçersizleştirme anahtarı.**
#:
#: ⚠️ Motorun analiz mantığı her değiştiğinde ARTTIRILMALIDIR. Aksi hâlde
#: önbellek, eski ve düzeltilmiş hatalarla üretilmiş sonuçları geri verir:
#: kullanıcı `göz` için hâlâ `*kuŕ` görür, `kitap` hâlâ miras sayılır.
#: Sürüm değişince eski kayıtlar otomatik olarak ıskalama sayılır.
ENGINE_VERSION = "4.3.4"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path(_env_str("DB_PATH", str(PROJECT_ROOT / "etymology.db")))
SCHEMA_PATH = PROJECT_ROOT / "engine" / "db" / "schema.sql"
SEED_DIR = Path(_env_str("SEED_DIR", str(PROJECT_ROOT / "data" / "seed")))
BOOKS_DIR = Path(_env_str("BOOKS_DIR", str(PROJECT_ROOT / "data" / "books")))
CLDF_DIR = Path(_env_str("CLDF_DIR", str(PROJECT_ROOT / "data" / "cldf")))
GOLD_DIR = Path(_env_str("GOLD_DIR", str(PROJECT_ROOT / "data" / "gold")))
LEXICON_DIR = Path(_env_str("LEXICON_DIR", str(PROJECT_ROOT / "data" / "lexicons")))
PREDICTIONS_DIR = Path(_env_str("PREDICTIONS_DIR", str(PROJECT_ROOT / "data" / "predictions")))

# --- Ağ --------------------------------------------------------------------

# Üç kademeli zaman aşımı. Daha önce 2–10 sn arası 32 farklı değer kullanılıyordu.
HTTP_TIMEOUT_SHORT = _env_float("HTTP_TIMEOUT_SHORT", 3.0)   # hızlı JSON API'ler
HTTP_TIMEOUT_MEDIUM = _env_float("HTTP_TIMEOUT_MEDIUM", 6.0)  # HTML kazıma
HTTP_TIMEOUT_LONG = _env_float("HTTP_TIMEOUT_LONG", 12.0)     # büyük sayfalar / wikitext

HTTP_MAX_RETRIES = _env_int("HTTP_MAX_RETRIES", 2)
HTTP_BACKOFF_BASE = _env_float("HTTP_BACKOFF_BASE", 0.3)
# Devre kesici: bir sunucu art arda bu kadar ağ hatası / 5xx verirse
# CIRCUIT_COOLDOWN saniye boyunca ona istek atılmaz. Ölçüldü (2026-09-24):
# TDK erişilemezken her istek ~19 sn bekliyor, arama başına 4-6 istek;
# 60 kelimelik tarama 10 dakikada yalnız 4 kelime ilerledi.
CIRCUIT_FAILURES = _env_int("CIRCUIT_FAILURES", 3)
CIRCUIT_COOLDOWN = _env_float("CIRCUIT_COOLDOWN", 300.0)
# Kalıcı yanıt önbelleği: toplu dökümü olmayan canlı sözlüklerin (TDK,
# Nişanyan, EtimolojiTürkçe) başarılı cevapları diske yazılır; bir kez sorulan
# kelime bir daha ağa çıkmaz ve ölçümler tekrarlanabilir olur.
HTTP_CACHE = _env_bool("HTTP_CACHE", True)
HTTP_CACHE_TTL_DAYS = _env_float("HTTP_CACHE_TTL_DAYS", 90.0)
HTTP_CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "http.db"
USER_AGENT = _env_str(
    "USER_AGENT",
    "TurkicEtymologyEngine/3.0 (academic research; +https://github.com/)",
)

# --- Arama motoru ----------------------------------------------------------

MAX_WORKERS = _env_int("MAX_WORKERS", 10)
# Varyant patlamasını sınırlar: 21 fetcher × N varyant = N×21 dış istek.
MAX_VARIANTS = _env_int("MAX_VARIANTS", 4)
CACHE_ENABLED = _env_bool("CACHE_ENABLED", True)
# Türki dillerin KENDİ Wiktionary'lerine canlı sorgu (14 site × aday biçim,
# arama başına ~28 istek). Varsayılan KAPALI. Ölçüldü (50 kelime): yalnız
# 2 kelimede kayıt getirdi, sık sık HTTP 503 / zaman aşımı; ayrıca sayfanın
# hangi dil bölümünden geldiğine bakmıyordu. Aynı dillerin çoğu yerel
# Türkçe/Rusça Wiktionary dökümlerinde doğru dil etiketiyle var.
LIVE_WIKTIONARY_EDITIONS = _env_bool("LIVE_WIKTIONARY_EDITIONS", False)
# Canlı İngilizce Wiktionary (sayfa + REST). Varsayılan KAPALI: aynı veri yerel
# kaikki dökümlerinde (indeks + Proto-Türkçe torunları). Wikimedia 503/hız
# sınırı ve sayfa ayrıştırma hataları (`parça` -> *bar) canlı yolun bedeliydi.
LIVE_WIKTIONARY = _env_bool("LIVE_WIKTIONARY", False)
# Canlı TDV İslâm Ansiklopedisi madde sayfası. Varsayılan KAPALI: yerel tohum
# veri her zaman okunur; canlı yol ölçümde sıcak aramanın 6,1/6,3 s'sini
# yiyordu (4 varyant × istek) ve eşleştirmesi gürültülüydü.
LIVE_ISAM = _env_bool("LIVE_ISAM", False)
# Arama yolunda verici yakınlığı (anlamı indeksten doldurarak). Kapalı: 150
# kelimede sıralayıcı uyumu 110 vs açıkken 100 (c0e8dd7). Sıralayıcı eğitilmiş
# birleştiriciye geçince de (Türkçe altın dev, data/cache/work/ranker) kapalı
# 129/135, açık 119/135 (McNemar p=0,031). Ölçüm hattını etkilemez.
SEARCH_DONOR_PROXIMITY = _env_bool("SEARCH_DONOR_PROXIMITY", False)
# Türkçeye özgü iki ses sinyali (alıntı dedektörü): Zemberek InverseHarmony/
# ImplicitPlural işareti (``ters_uyum``) ve söz sonu iki ünsüz
# (``söz_sonu_ünsüz_kümesi``). Yalnız lang == "tr"de ateşlenir. Bkz.
# borrowing_detector._inverse_harmony_signal. KAPALI: Türkçe altın rapor
# yarısında F farkı -0,0038 [-0,0132, +0,0055] (Z1, data/cache/work/z1).
BORROWING_TR_SOUND_SIGNALS = _env_bool("BORROWING_TR_SOUND_SIGNALS", False)
# Archive.org tam metin araması. Varsayılan KAPALI: ölçümlerde tanık üretmedi.
LIVE_ARCHIVE_ORG = _env_bool("LIVE_ARCHIVE_ORG", False)
CACHE_TTL_SECONDS = _env_int("CACHE_TTL_SECONDS", 7 * 24 * 3600)
MAX_QUERY_LENGTH = _env_int("MAX_QUERY_LENGTH", 64)

# --- LLM (Ollama) ----------------------------------------------------------

OLLAMA_BASE_URL = _env_str("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_GENERATE_URL = f"{OLLAMA_BASE_URL}/api/generate"
OLLAMA_TAGS_URL = f"{OLLAMA_BASE_URL}/api/tags"
OLLAMA_MODEL = _env_str("OLLAMA_MODEL", "qwen2.5:14b")
OLLAMA_TIMEOUT = _env_float("OLLAMA_TIMEOUT", 180.0)  # qwen2.5:14b yerel donanımda yavaş olabilir
OLLAMA_NUM_CTX = _env_int("OLLAMA_NUM_CTX", 1024)
OLLAMA_NUM_PREDICT = _env_int("OLLAMA_NUM_PREDICT", 250)
OLLAMA_TEMPERATURE = _env_float("OLLAMA_TEMPERATURE", 0.15)
# Kazınmış içeriğin isteme girebileceği azami karakter (prompt injection yüzeyini daraltır).
MAX_UNTRUSTED_CHARS = _env_int("MAX_UNTRUSTED_CHARS", 1500)

# --- REST API sunucusu -----------------------------------------------------

# Varsayılan olarak yalnızca yerel arayüz. Daha önce '' (0.0.0.0) idi.
API_HOST = _env_str("API_HOST", "127.0.0.1")
API_PORT = _env_int("API_PORT", 8000)
# CORS: virgülle ayrılmış origin listesi. '*' bilinçli bir tercih olmalıdır.
CORS_ALLOW_ORIGINS = tuple(
    o.strip() for o in _env_str("CORS_ALLOW_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",") if o.strip()
)
# Hata gövdesinde iç istisna metni döndürülsün mü (yalnızca geliştirme).
API_DEBUG_ERRORS = _env_bool("API_DEBUG_ERRORS", False)

# --- A-HVP hakem protokolü -------------------------------------------------

# Dört aşamanın ağırlıkları. Bir aşama kanıt üretemezse ağırlığı toplamdan
# düşülür ve skor katkıda bulunan aşamalara normalize edilir.
A_HVP_WEIGHTS = {
    "phonetic": _env_float("AHVP_W_PHONETIC", 0.35),
    "chronology": _env_float("AHVP_W_CHRONOLOGY", 0.30),
    "semantic": _env_float("AHVP_W_SEMANTIC", 0.15),
    "triangulation": _env_float("AHVP_W_TRIANGULATION", 0.20),
}

BADGE_THRESHOLDS = {
    "validated": _env_float("AHVP_T_VALIDATED", 0.75),
    "needs_review": _env_float("AHVP_T_NEEDS_REVIEW", 0.50),
}

# Kanıtlanan aşama ağırlığı bu oranın altındaysa rozet en fazla
# INSUFFICIENT_EVIDENCE olabilir. "Kanıt yoksa puan da yok" ilkesi.
MIN_EVIDENCE_COVERAGE = _env_float("AHVP_MIN_EVIDENCE_COVERAGE", 0.50)

# --- Anlam benzerliği -------------------------------------------------------

# Bir biçimin (tanık / Starling adayı) sorgunun bir anlamını "taşıdığı" en
# düşük MiniLM benzerliği. Arama motorunun eşsesli süzgeci
# (``search_engine.HOMONYM_SIMILARITY_FLOOR``) ile Starling aday seçimi
# (``fetchers.starling.MEANING_FLOOR``) aynı ölçüte bakar; ölçüm ve gerekçe
# ``search_engine.py``'dedir (eşsesliler 0,13-0,27, gerçek anlamlar 0,385+).
MEANING_SIMILARITY_FLOOR = 0.30

# --- Kazıma güvenliği ------------------------------------------------------

TRUSTED_DOMAINS = tuple(
    d.strip()
    for d in _env_str(
        "TRUSTED_DOMAINS",
        "nisanyansozluk.com,lugatim.com,dergipark.org.tr,wiktionary.org,"
        "wikipedia.org,sozluk.gov.tr,tdk.gov.tr,islamansiklopedisi.org.tr,"
        "archive.org,etimolojiturkce.com,starling.rinet.ru",
    ).split(",")
    if d.strip()
)

# --- Loglama ---------------------------------------------------------------

LOG_LEVEL = _env_str("LOG_LEVEL", "WARNING")
