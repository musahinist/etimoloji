import json
import re
import urllib.parse
import urllib.request
from typing import Any

from engine import config
from engine.fetchers.base import TURKIC_LANGUAGES_MAP, BaseFetcher
from engine.logging_setup import get_logger
from engine.utils.network import fetch as http_get

logger = get_logger(__name__)


class TdkFetcher(BaseFetcher):
    @property
    def source_name(self) -> str:
        return "TDK (Türk Dil Kurumu)"

    def fetch(self, word: str) -> dict[str, Any]:
        word_clean = word.strip().lower()
        result = {
            "root": {"proto_turkic": "", "meaning": "", "reconstruction_notes": ""},
            "turkic_languages": []
        }

        url = f"https://sozluk.gov.tr/gts?ara={urllib.parse.quote(word_clean)}"
        try:
            _body = http_get(url, timeout=config.HTTP_TIMEOUT_MEDIUM)
            if _body is not None:
                data = json.loads(_body)
                if isinstance(data, list) and len(data) > 0 and "anlamlarListe" in data[0]:
                    meanings = [item["anlam"] for item in data[0]["anlamlarListe"] if "anlam" in item]
                    meaning_str = "; ".join(meanings[:2])

                    lisan = data[0].get("lisan", "")

                    result["turkic_languages"].append({
                        "lang_code": "tr",
                        "lang_name": TURKIC_LANGUAGES_MAP["tr"],
                        "word": word_clean,
                        "meaning": meaning_str,
                        "script": "Latin"
                    })
                    result["root"]["meaning"] = meaning_str
                    # Raporda maddenin TÜM anlamları ayrı ayrı gösterilir.
                    result["root"]["meanings"] = meanings
                    if lisan:
                        result["root"]["reconstruction_notes"] = f"TDK Köken Bilgisi: {lisan}"
        except Exception:
            logger.warning("%s: kaynak işlenemedi", self.source_name if hasattr(self, "source_name") else __name__, exc_info=True)
        return result


# --- Nişanyan düzyazı ayrıştırıcısı ---------------------------------------
#
# Nişanyan köken CÜMLESİ şu kalıptadır:
#
#     <Kaynak dil> [kök/vezin tarifi] <biçim> [yabancı yazı] “<anlam>”
#     sözcüğünden <FİİL>
#
# ve HÜKMÜ veren şey fiildir: ``alıntıdır`` -> alıntı, ``evrilmiştir`` ->
# miras, ``türetilmiştir`` -> türetme.
#
# Eski ayrıştırıcının iki kırılma noktası vardı (ölçüldü: 12 rastgele
# kelimenin 9'unda sıfır çıktı, 1'inde uydurma köken):
#
# 1. Karakter sınıfı ``[a-zçğıöşüA-ZÇĞİÖŞÜ]`` transkripsiyon harflerini
#    kapsamıyordu; ``teŋiz``, ``kȫz``, ``tapuġ`` tek harf yüzünden kaçıyordu.
# 2. ``<Dil> <kelime> “<anlam>”`` bitişikliği varsayılıyordu; Nişanyan araya
#    "√ˁẓm kökünden gelen faˁlala(t) vezninde ... olan" koyduğu için Arapça
#    maddelerin tamamı düşüyordu. Kalıp eşleşmeyince metindeki HERHANGİ bir
#    yabancı dil adı köken sanılıyordu: ``deniz`` için Nişanyan'ın "Anlam
#    bağı için karş. Latince aequor" BENZETMESİ köken diye kaydediliyordu.

#: Benzetme/gönderme işaretleri. Bunlardan SONRASI köken iddiası değildir.
_ANALOGY_RE = re.compile(r"(?i)\b(?:karş|krş|bkz)\.")

#: Kaynak dil adları — uzun olan önce gelmeli ki "Eski Farsça", "Farsça"
#: tarafından yutulmasın.
_LANGUAGES: tuple[str, ...] = (
    "Eski Anadolu Türkçesi", "Türkiye Türkçesi", "Proto-Türkçe", "Eski Türkçe",
    "Orta Türkçe", "Aramice-Süryanice", "Eski Farsça", "Eski Yunanca",
    "Ermenice", "Fransızca", "İtalyanca", "İngilizce", "Süryanice", "Almanca",
    "Latince", "Yunanca", "Grekçe", "Arapça", "Farsça", "Rumca", "Rusça",
    "Moğolca", "Soğdca", "Akatça", "İbranice", "Çince", "Macarca",
)

#: Türki ata katmanları: bunlar VERİCİ dil değil, mirasın kendisidir.
_TURKIC_ANCESTORS = {
    "Eski Türkçe": "otk",
    "Orta Türkçe": "otk",
    "Eski Anadolu Türkçesi": "otk",
    "Türkiye Türkçesi": "tr",
    "Proto-Türkçe": None,
}

#: Fiil -> hüküm. Uzun biçimler önce denenmeli ("alıntı olabilir" vs "alıntıdır").
_VERDICT_BY_VERB: dict[str, str] = {
    "alıntı olabilir": "alıntı",
    "alıntıdır": "alıntı",
    "evrilmiş olabilir": "miras",
    "evrilmiştir": "miras",
    "türetilmiş olabilir": "türetme",
    "türetilmiştir": "türetme",
}

#: Biçim adayı olamayacak tarif sözcükleri.
_FILLER_TOKENS = frozenset({
    "aynı", "anlama", "gelen", "kökünden", "kökünün", "vezninde", "olan",
    "ve", "veya", "ile", "eş", "tanıksız", "fiil", "fiili", "fiilinden",
    "masdarı", "sıfat", "sıfatı", "edilgen", "murabba", "dörtlü", "yalnız",
    "bir", "adı", "özel", "sözcüğü", "sözcük", "biçiminden", "çoğulu",
})

#: En az bir Latin harfi taşıyan belirteç (Arapça/İbranice yazımı eler).
_HAS_LATIN_RE = re.compile(r"[A-Za-zÀ-ɏḀ-ỿ]")

_CLAIM_RE = re.compile(
    r"(?P<lang>" + "|".join(_LANGUAGES) + r")"
    r"(?P<middle>[^“”.]{0,160}?)"
    r"(?:“(?P<meaning>[^”]{1,200})”\s*)?"
    r"(?:sözcüğünden|sözünden|fiilinden|adından|biçiminden)\s+"
    # Türetme cümlelerinde kaynak biçim ile fiil arasına EK TARİFİ girer:
    #   "... bulġa- “karıştırmak” fiilinden  Türkiye Türkçesi +Iş- ekiyle
    #    türetilmiştir"
    # Bu araya girme olmadan `bulaşmak`, `çığlık`, `taslamak` gibi türemiş
    # kelimelerin tamamı eşleşmeden düşüyordu.
    r"(?:[^.“”]{0,90}?ekiyle\s+)?"
    r"(?P<verb>" + "|".join(_VERDICT_BY_VERB) + r")"
)


def _extract_form(middle: str) -> str | None:
    """Dil adı ile anlam arasındaki tarif metninden BİÇİMİ ayıklar.

    Nişanyan araya kök ve vezin tarifi koyar; aranan biçim bu tarifin
    sonundaki son Latin harfli belirteçtir::

        "√ftl kökünden gelen faˁīl vezninde sıfat olan fatīl veya fatīla(t) فتيل"
        -> "fatīla(t)"
    """
    candidates: list[str] = []
    for raw in middle.split():
        token = raw.strip(",;:()[]").strip()
        if not token or token.startswith("√"):
            continue
        if token.lower() in _FILLER_TOKENS:
            continue
        if not _HAS_LATIN_RE.search(token):  # Arapça/İbranice yazım
            continue
        candidates.append(raw.strip(",;:").strip())
    return candidates[-1] if candidates else None


def parse_etymology_claim(text: str) -> dict[str, Any] | None:
    """Nişanyan metninden İLK köken iddiasını çıkarır.

    ``karş.`` / ``krş.`` / ``bkz.`` işaretlerinden sonrası benzetmedir ve
    ayrıştırmaya hiç girmez. Köken cümlesi bulunamazsa ``None`` döner —
    uydurmaktansa boş dönmek yeğdir.
    """
    if not text:
        return None
    primary = _ANALOGY_RE.split(text, maxsplit=1)[0]
    match = _CLAIM_RE.search(primary)
    if not match:
        return None
    form = _extract_form(match.group("middle") or "")
    if not form:
        return None
    meaning = (match.group("meaning") or "").strip() or None
    return {
        "language": match.group("lang"),
        "form": form,
        "meaning": meaning,
        "verb": match.group("verb"),
        "verdict": _VERDICT_BY_VERB[match.group("verb")],
    }


class NisanyanFetcher(BaseFetcher):
    @property
    def source_name(self) -> str:
        return "Nişanyan Etimoloji Sözlüğü"

    @staticmethod
    def _parse_nisanyan_text(text_full: str, result: dict[str, Any]) -> None:
        """Çıkarılan köken iddiasını fetcher sözleşmesine yazar."""
        claim = parse_etymology_claim(text_full)
        if claim is None:
            result["root"]["reconstruction_notes"] = f"Nişanyan Etimoloji: {text_full[:300]}..."
            return

        lang, meaning = claim["language"], claim["meaning"]
        # Nişanyan biçimlerde görünmez sözcük birleştirici (U+2060) kullanıyor;
        # temizlenmezse `*\u2060bonçuk` başka hiçbir biçimle eşleşmez.
        form = re.sub(r"[\u200b-\u200d\u2060\ufeff]", "", claim["form"])
        # Yıldızlı biçim Nişanyan'ın yeniden kurduğu, TANIKLANMAMIŞ biçimdir;
        # tanıklı Eski Türkçe kaydı gibi tanık listesine girmemeli.
        reconstructed = form.startswith("*")
        form = form.lstrip("*")
        if meaning:
            result["root"]["meaning"] = meaning

        if lang in _TURKIC_ANCESTORS:
            # Miras: verici dil YOKTUR, biçim ata katmandır.
            result["root"]["proto_turkic"] = f"*{form}"
            lang_code = _TURKIC_ANCESTORS[lang]
            if lang_code and not reconstructed:
                result["turkic_languages"].append({
                    "lang_code": lang_code,
                    "lang_name": TURKIC_LANGUAGES_MAP.get(lang_code, lang),
                    "word": form,
                    "meaning": meaning or "",
                    "script": "Latin",
                })
            result["root"]["reconstruction_notes"] = (
                f"Nişanyan: {lang} {form}"
                f"{f' “{meaning}”' if meaning else ''} — hüküm: {claim['verdict']}"
                f" ({claim['verb']})"
            )
        else:
            result["root"]["proto_turkic"] = f"[{lang}] {form}"
            result["root"]["reconstruction_notes"] = (
                f"Nişanyan Alıntı Kaynağı: {lang} '{form}'"
                f"{f' ({meaning})' if meaning else ''}"
            )

    def fetch(self, word: str) -> dict[str, Any]:
        word_clean = word.strip().lower()
        result = {
            "root": {"proto_turkic": "", "meaning": "", "reconstruction_notes": ""},
            "turkic_languages": []
        }

        url = f"https://www.nisanyansozluk.com/kelime/{urllib.parse.quote(word_clean)}"
        try:
            _body = http_get(url, timeout=config.HTTP_TIMEOUT_LONG)
            if _body is not None:
                tokens = re.findall(r'text:\"([^\"]+)\"', _body)
                if not tokens:
                    return result
                self._parse_nisanyan_text("".join(tokens), result)

        except Exception:
            logger.warning("%s: kaynak işlenemedi", self.source_name if hasattr(self, "source_name") else __name__, exc_info=True)
        return result
