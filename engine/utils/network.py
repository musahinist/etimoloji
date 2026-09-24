"""
Ortak HTTP İstemcisi (Shared HTTP Client)

Projedeki TÜM dış ağ erişiminin tek kapısı. Daha önce 21 fetcher, 3 kazıyıcı ve
2 LLM araç modülü kendi `urllib.request` çağrısını, kendi User-Agent'ını ve
kendi zaman aşımını yazıyordu (32 farklı `timeout=`, 11 farklı UA); retry/backoff
mantığı bu dosyada yazılmış ama hiçbir yerden çağrılmıyordu.

Sağladıkları:
  * `requests.Session` üzerinde bağlantı havuzu
  * Üstel geri çekilmeli (exponential backoff) yeniden deneme
  * Merkezî User-Agent ve üç kademeli zaman aşımı
  * SSRF koruması: özel/loopback adres reddi, opsiyonel alan adı beyaz listesi
  * Her isteğin süresini ve sonucunu kaydeden teşhis (diagnostics) kancası
"""
from __future__ import annotations

import ipaddress
import json
import socket
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import requests

from engine.config import (
    CIRCUIT_COOLDOWN,
    CIRCUIT_FAILURES,
    HTTP_BACKOFF_BASE,
    HTTP_CACHE,
    HTTP_CACHE_PATH,
    HTTP_CACHE_TTL_DAYS,
    HTTP_MAX_RETRIES,
    HTTP_TIMEOUT_MEDIUM,
    TRUSTED_DOMAINS,
    USER_AGENT,
)
from engine.logging_setup import get_logger

logger = get_logger(__name__)


class UnsafeURLError(ValueError):
    """URL güvenlik denetiminden geçemedi (SSRF koruması)."""


@dataclass
class RequestRecord:
    """Tek bir dış isteğin teşhis kaydı."""
    url: str
    status: str          # "ok" | "http_error" | "network_error" | "blocked"
    duration_ms: int
    http_status: int | None = None
    error: str | None = None


@dataclass
class Diagnostics:
    """Bir arama boyunca yapılan tüm isteklerin toplandığı kayıt defteri."""
    records: list[RequestRecord] = field(default_factory=list)

    def add(self, record: RequestRecord) -> None:
        self.records.append(record)

    @property
    def total_requests(self) -> int:
        return len(self.records)

    @property
    def total_ms(self) -> int:
        return sum(r.duration_ms for r in self.records)

    def summary(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        for r in self.records:
            by_status[r.status] = by_status.get(r.status, 0) + 1
        return {
            "total_requests": self.total_requests,
            "total_ms": self.total_ms,
            "by_status": by_status,
        }


def _decode_body(resp: requests.Response) -> str:
    """
    Gövdeyi doğru karakter kodlamasıyla çözer.

    Bazı kaynaklar (ör. etimolojiturkce.com) ``Content-Type: text/html``
    başlığını charset bildirmeden gönderir; ``requests`` bu durumda RFC gereği
    ISO-8859-1 varsayar ve Türkçe karakterler mojibake olur ("eş" -> "eÅ").
    Sunucu charset bildirmediyse içerikten sezilen kodlama kullanılır.
    """
    declared = (resp.headers.get("Content-Type") or "").lower()
    if "charset=" not in declared:
        resp.encoding = resp.apparent_encoding or "utf-8"
    return resp.text


_session: requests.Session | None = None


def get_session() -> requests.Session:
    """Süreç ömrü boyunca paylaşılan HTTP oturumu."""
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept-Language": "tr,en;q=0.8",
        })
    return _session


def reset_session() -> None:
    """Testlerin oturumu sıfırlaması için."""
    global _session
    if _session is not None:
        _session.close()
    _session = None
    reset_circuits()


# --- Kalıcı yanıt önbelleği ---------------------------------------------------
#: Toplu dökümü olmayan canlı sözlükler. Başka hiçbir sunucu önbelleğe alınmaz.
PERSISTENT_CACHE_HOSTS = frozenset({
    "sozluk.gov.tr", "www.nisanyansozluk.com", "nisanyansozluk.com",
    "www.etimolojiturkce.com", "etimolojiturkce.com",
})
#: Testler bunu kapatır (bkz. ``engine/tests/conftest.py``).
_persistent_enabled = HTTP_CACHE
_cache_lock = threading.Lock()


def _cache_key(url: str, params: dict[str, Any] | None) -> str:
    if not params:
        return url
    return url + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))


def _cache_connect() -> sqlite3.Connection:
    HTTP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(HTTP_CACHE_PATH, timeout=10)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, body TEXT NOT NULL, fetched_at REAL NOT NULL)"
    )
    return connection


def _cache_get(key: str) -> str | None:
    try:
        with _cache_lock, _cache_connect() as connection:
            row = connection.execute("SELECT body, fetched_at FROM responses WHERE key = ?", (key,)).fetchone()
    except sqlite3.Error:
        logger.debug("Yanıt önbelleği okunamadı", exc_info=True)
        return None
    if not row or time.time() - row[1] > HTTP_CACHE_TTL_DAYS * 86400:
        return None
    return row[0]


def _cache_put(key: str, body: str) -> None:
    try:
        with _cache_lock, _cache_connect() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO responses (key, body, fetched_at) VALUES (?, ?, ?)",
                (key, body, time.time()),
            )
    except sqlite3.Error:
        logger.debug("Yanıt önbelleğine yazılamadı", exc_info=True)


# --- Devre kesici ----------------------------------------------------------
# Sunucu başına ardışık hata sayısı ve devrenin açık kalacağı an.
_circuits: dict[str, tuple[int, float]] = {}
_circuit_lock = threading.Lock()


#: Açık devreler diske de yazılır: her yeni süreç erişilemeyen sunucuyu
#: yeniden denemesin (TDK çökükken her arama ilk ~40 sn'yi zaman aşımında
#: geçiriyordu). Kalıcı önbellekle aynı anahtara bağlıdır; testler kapatır.
CIRCUIT_STATE_PATH = HTTP_CACHE_PATH.parent / "circuits.json"
_circuits_loaded = False


def reset_circuits() -> None:
    global _circuits_loaded
    with _circuit_lock:
        _circuits.clear()
        _circuits_loaded = True  # diskteki durum da yok sayılır


def _load_circuits() -> None:
    global _circuits_loaded
    if _circuits_loaded:
        return
    _circuits_loaded = True
    if not _persistent_enabled or not CIRCUIT_STATE_PATH.exists():
        return
    try:
        for host, open_until in json.loads(CIRCUIT_STATE_PATH.read_text(encoding="utf-8")).items():
            if float(open_until) > time.time():
                _circuits[host] = (0, float(open_until))
    except (OSError, ValueError, TypeError):
        logger.debug("Devre durumu okunamadı", exc_info=True)


def _save_circuits() -> None:
    if not _persistent_enabled:
        return
    try:
        CIRCUIT_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        state = {h: until for h, (_, until) in _circuits.items() if until > time.time()}
        CIRCUIT_STATE_PATH.write_text(json.dumps(state), encoding="utf-8")
    except OSError:
        logger.debug("Devre durumu yazılamadı", exc_info=True)


def _circuit_open(host: str) -> bool:
    with _circuit_lock:
        _load_circuits()
        _, open_until = _circuits.get(host, (0, 0.0))
        return open_until > time.time()


def _record(host: str, *, failed: bool) -> None:
    with _circuit_lock:
        if not failed:
            if _circuits.pop(host, (0, 0.0))[1]:
                _save_circuits()
            return
        failures = _circuits.get(host, (0, 0.0))[0] + 1
        open_until = 0.0
        if failures >= CIRCUIT_FAILURES:
            open_until = time.time() + CIRCUIT_COOLDOWN
            logger.warning(
                "Devre açıldı: %s art arda %d kez başarısız; %.0f sn istek atılmayacak",
                host, failures, CIRCUIT_COOLDOWN,
            )
            failures = 0
        _circuits[host] = (failures, open_until)
        if open_until:
            _save_circuits()


# --- Güvenlik --------------------------------------------------------------

def _is_private_host(hostname: str) -> bool:
    """Hostname özel/loopback/link-local bir adrese mi çözümleniyor?"""
    try:
        infos = socket.getaddrinfo(hostname, None)
    except (socket.gaierror, UnicodeError):
        return True  # çözümlenemiyorsa güvenli tarafta kal
    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return True
    return False


def is_url_allowed(url: str, *, trusted_only: bool = False, allow_private: bool = False) -> tuple[bool, str]:
    """URL'nin çekilmesi güvenli mi? (izin, gerekçe) döndürür.

    ``allow_private`` yalnızca bilinçli yerel servisler için kullanılır
    (ör. Ollama ``localhost:11434``).
    """
    parsed = urlparse(url or "")
    if parsed.scheme not in ("http", "https"):
        return False, f"desteklenmeyen şema: {parsed.scheme!r}"
    if not parsed.hostname:
        return False, "host yok"
    if not allow_private:
        try:
            socket.getaddrinfo(parsed.hostname, None)
        except (socket.gaierror, UnicodeError):
            return False, f"host çözümlenemedi: {parsed.hostname}"
        if _is_private_host(parsed.hostname):
            return False, f"özel/loopback adres reddedildi: {parsed.hostname}"
    if trusted_only:
        host = parsed.hostname.lower()
        if not any(host == d or host.endswith("." + d) for d in TRUSTED_DOMAINS):
            return False, f"beyaz listede değil: {host}"
    return True, "ok"


# --- İstek ----------------------------------------------------------------

def _retry_after_seconds(resp: requests.Response, *, cap: float = 5.0) -> float | None:
    """``Retry-After`` başlığını saniyeye çevirir; yoksa/saçmaysa ``None``.

    Üst sınır şart: sunucu dakikalarca bekleme isteyebilir, ama tek bir
    kelime araması onlarca kaynağa paralel gidiyor — orada beklemek bütün
    aramayı kilitler. Bu durumda beklemek yerine o isteği bırakmak doğrudur.
    """
    raw = (resp.headers.get("Retry-After") or "").strip()
    if not raw:
        return None
    try:
        seconds = float(raw)
    except ValueError:
        return None  # HTTP-date biçimi: bu hat için beklemeye değmez
    if seconds <= 0 or seconds > cap:
        return None
    return seconds


def fetch(
    url: str,
    *,
    timeout: float = HTTP_TIMEOUT_MEDIUM,
    max_retries: int = HTTP_MAX_RETRIES,
    trusted_only: bool = False,
    allow_private: bool = False,
    diagnostics: Diagnostics | None = None,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
) -> str | None:
    """
    URL'yi çeker ve gövdeyi metin olarak döndürür; başarısızlıkta ``None``.

    Hata asla sessizce yutulmaz — her başarısızlık loglanır ve varsa
    ``diagnostics`` defterine yazılır.
    """
    allowed, reason = is_url_allowed(url, trusted_only=trusted_only, allow_private=allow_private)
    if not allowed:
        logger.warning("URL engellendi (%s): %s", reason, url)
        if diagnostics is not None:
            diagnostics.add(RequestRecord(url=url, status="blocked", duration_ms=0, error=reason))
        return None

    host = urlparse(url).hostname or ""
    cache_key = _cache_key(url, params) if _persistent_enabled and host in PERSISTENT_CACHE_HOSTS else ""
    if cache_key:
        cached = _cache_get(cache_key)
        if cached is not None:
            if diagnostics is not None:
                diagnostics.add(RequestRecord(url=url, status="ok", duration_ms=0, http_status=200, error="kalıcı önbellek"))
            return cached
    if _circuit_open(host):
        logger.debug("Devre açık, istek atlanıyor: %s", url)
        if diagnostics is not None:
            diagnostics.add(RequestRecord(url=url, status="circuit_open", duration_ms=0, error="devre açık"))
        return None

    session = get_session()
    started = time.perf_counter()
    last_error: str | None = None
    http_status: int | None = None

    for attempt in range(max_retries + 1):
        try:
            resp = session.get(url, timeout=timeout, headers=headers, params=params)
            http_status = resp.status_code
            if resp.status_code == 200:
                _record(host, failed=False)
                elapsed = int((time.perf_counter() - started) * 1000)
                if diagnostics is not None:
                    diagnostics.add(RequestRecord(url=url, status="ok", duration_ms=elapsed, http_status=200))
                body = _decode_body(resp)
                if cache_key:
                    _cache_put(cache_key, body)
                return body
            last_error = f"HTTP {resp.status_code}"
            # ⚠️ 429 "yavaşla" demektir; körlemesine yeniden denemek yükü
            # ARTIRIR. Eskiden 429 geçici hata sayılıp 0,3s ve 0,6s arayla iki
            # kez daha deneniyordu, yani her reddedilen istek ÜÇE katlanıyordu.
            # Ölçüldü (`herkil`): 4 varyant × 14 Wiktionary sürümü ≈ 60 istek,
            # yeniden denemelerle ~180'e çıkıp Wikimedia tarafından toptan
            # reddedildi ve terminal uyarıya boğuldu.
            # Sunucu makul bir `Retry-After` verirse bir kez beklenir;
            # vermezse bu istek için pes edilir.
            if resp.status_code == 429:
                wait = _retry_after_seconds(resp)
                if wait is None or attempt >= max_retries:
                    break
                time.sleep(wait)
                continue
            if resp.status_code not in (500, 502, 503, 504):
                break
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        if attempt < max_retries:
            time.sleep(HTTP_BACKOFF_BASE * (2 ** attempt))

    elapsed = int((time.perf_counter() - started) * 1000)
    status = "http_error" if http_status else "network_error"
    # Yalnız sunucunun ULAŞILAMAZ olduğunu gösteren hatalar devreyi besler;
    # 404 "kelime yok", 429 ise hız sınırıdır.
    if http_status is None or http_status >= 500:
        _record(host, failed=True)
    elif http_status == 404:
        _record(host, failed=False)
    # 404 ve 429 bu hatta BEKLENEN durumlardır: aranan kelime o sözlükte
    # gerçekten olmayabilir (404), ya da tek bir arama onlarca kaynağa
    # paralel gittiği için hız sınırına çarpılabilir (429). Bunları WARNING
    # basmak terminali dolduruyor ve GERÇEK arızayı görünmez kılıyor —
    # ölçüldü: tek bir `herkil` aramasında 180'den fazla uyarı satırı.
    _expected = http_status in (404, 429)
    (logger.debug if _expected else logger.warning)(
        "İstek başarısız (%s) %s — %s", last_error, url, f"{elapsed}ms"
    )
    if diagnostics is not None:
        diagnostics.add(
            RequestRecord(url=url, status=status, duration_ms=elapsed, http_status=http_status, error=last_error)
        )
    return None


def fetch_json(url: str, **kwargs: Any) -> Any | None:
    """`fetch` ile çeker ve JSON olarak ayrıştırır; başarısızlıkta ``None``."""
    import json

    body = fetch(url, **kwargs)
    if body is None:
        return None
    try:
        return json.loads(body)
    except (ValueError, TypeError) as exc:
        logger.warning("JSON ayrıştırılamadı %s: %s", url, exc)
        return None


# Geriye dönük uyumluluk: eski ad.
fetch_url_safe = fetch


def post_json(
    url: str,
    payload: Any,
    *,
    timeout: float = HTTP_TIMEOUT_MEDIUM,
    allow_private: bool = False,
    diagnostics: Diagnostics | None = None,
) -> Any | None:
    """JSON gövdeli POST atar ve JSON yanıtı döndürür; başarısızlıkta ``None``."""
    allowed, reason = is_url_allowed(url, allow_private=allow_private)
    if not allowed:
        logger.warning("POST URL engellendi (%s): %s", reason, url)
        if diagnostics is not None:
            diagnostics.add(RequestRecord(url=url, status="blocked", duration_ms=0, error=reason))
        return None

    started = time.perf_counter()
    try:
        resp = get_session().post(url, json=payload, timeout=timeout)
        elapsed = int((time.perf_counter() - started) * 1000)
        if resp.status_code == 200:
            if diagnostics is not None:
                diagnostics.add(RequestRecord(url=url, status="ok", duration_ms=elapsed, http_status=200))
            return resp.json()
        logger.warning("POST başarısız HTTP %s: %s", resp.status_code, url)
        if diagnostics is not None:
            diagnostics.add(
                RequestRecord(url=url, status="http_error", duration_ms=elapsed, http_status=resp.status_code)
            )
    except (requests.RequestException, ValueError) as exc:
        elapsed = int((time.perf_counter() - started) * 1000)
        logger.warning("POST hatası %s: %s", url, exc)
        if diagnostics is not None:
            diagnostics.add(
                RequestRecord(url=url, status="network_error", duration_ms=elapsed, error=str(exc))
            )
    return None
