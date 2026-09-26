"""
Eren (1999) ve Gülensoy (2007) — Türkiye Türkçesi anahtarlı etimoloji sözlükleri.

Kaynaklar (archive.org, tesseract OCR ``_djvu.txt``; ``make erengul`` =
``scripts/download_erengul.py`` ile ``data/erengul/`` altına iner):

* Hasan Eren, *Türk Dilinin Etimolojik Sözlüğü*, Ankara 1999 (Bizim Büro).
  EDAL'dan (2003) ÖNCE yayımlandı: Starling'den bağımsız görüş.
* Tuncer Gülensoy, *Türkiye Türkçesindeki Türkçe Sözcüklerin Köken Bilgisi
  Sözlüğü*, 2 cilt, Ankara 2007 (TDK). Yalnız Türkçe kökenli sözcükler.

⚠️ Metinler telifli: ``_djvu.txt`` ve JSONL repoya ALINMAZ (``.gitignore``);
yalnız künye ve SHA-256'lar ``data/erengul/_provenance.json`` ile commit edilir.

Çıkarılan alanlar (madde başına bir kayıt):

* ``lemma`` — Türkiye Türkçesi madde başı (fiillerde Eren ``-mAk`` mastarı,
  Gülensoy ``kök-`` biçimi); ``key`` — eşleşme anahtarı (fiilde mastar/tire
  atılmış gövde: ``savurmak`` / ``savur-`` -> ``savur``). Eşleşme TAMdır:
  benzerlik yok.
* ``origin`` — köken hükmü: ``turkish`` / ``loan`` (``donor`` dil kısaltması)
  / ``unknown`` ("Kökenini bilmiyoruz") / ``""`` (hüküm çıkarılamadı).
* ``root`` — önerilen kök/taban, kaynağın yazdığı gibi (``çap-``, ``*kapa``);
  ``suffixes`` — türetme ekleri (``-gut``); ``ot_form`` — Eski/Orta Türkçe
  biçim (``çapğut``); ``see`` — ``Bk. X`` göndermesi; ``refs`` — Räsänen /
  Clauson / ЭСТЯ sayfa atıfları (yalnız Eren).
* ``redirect`` — ``X bk. Y`` biçimli gönderme maddesi.

Kök çıkarma kuralları (yalnız kaynağın kendi sesi; reddedilen görüş alınmaz):

* Eren: ``X kökünden`` / ``X kökünün`` (Eren'in naklettiği ``Clauson'a göre``
  dahil; ``yanlış`` geçen cümle dışarıda), yoksa ``< X- + -ek`` satırı.
* Gülensoy: türetme satırı ``< a-b-c`` (OCR'da ``<`` işareti ``€``, ``x``,
  ``cx``, ``m``, ``«`` okunmuş) -> ilk parça kök, kalanı ek; ``< OT. kepi- +
  -r`` biçiminde ``OT.`` atılır.

⚠️ Bu kaynaklar TANIK DEĞİLDİR (Clauson dersi, 148fe1d): yeniden kurucuya
girmez; yalnız başlık / köken notu kaynağıdır ("Eren: *çap- + -gut").
Gösterim Türkiye Türkolojisi çeviriyazısıdır (ğ, ŋ, ı), Starling/EDAL
notasyonu değildir.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from engine.config import PROJECT_ROOT

ERENGUL_DIR = PROJECT_ROOT / "data" / "erengul"
JSONL_NAME = "erengul.jsonl"

#: archive.org ögeleri ve ``_djvu.txt`` dosya adları (yerel ad -> (öge, uzak ad)).
SOURCES: dict[str, tuple[str, str]] = {
    "eren_djvu.txt": (
        "turk-dilinin-etimolojik-sozlugu-hasan-eren",
        "Türk Dilinin Etimolojik Sözlügü Hasan Eren_djvu.txt",
    ),
    "gulensoy1_djvu.txt": (
        "turkiye-turkcesindeki-turkce-sozcuklerin-koken-bilgisi-sozlugu-1.-cilt",
        "Türkiye Türkçesindeki Türkçe Sözcüklerin Köken Bilgisi Sözlüğü 1. cilt_djvu.txt",
    ),
    "gulensoy2_djvu.txt": (
        "turkiye-turkcesindeki-turkce-sozcuklerin-koken-bilgisi-sozlugu-1.-cilt",
        "Türkiye Türkçesindeki Türkçe Sözcüklerin Köken Bilgisi Sözlüğü 2. cilt_djvu.txt",
    ),
}

_L = "a-zçğıöşüâîûéŋāēīōūȫǖä"
_LEMMA = rf"[{_L}][{_L}']*(?:[ -][{_L}]+)?"
_QUOTE = "‘'“\"*’”"

# --- ortak -------------------------------------------------------------------


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def lemma_key(lemma: str) -> str:
    """Eşleşme anahtarı: küçük harf, fiil mastarı / gövde tiresi atılmış."""
    key = _nfc(lemma).strip().casefold().rstrip("-|").strip()
    return key


def verb_key(lemma: str) -> str:
    """``savurmak`` -> ``savur`` (yalnız mastar biçimli madde başında)."""
    key = lemma_key(lemma)
    if len(key) > 4 and key.endswith(("mak", "mek")) and " " not in key:
        return key[:-3]
    return ""


def _join(lines: list[str]) -> str:
    """Satırları birleştirir; satır sonu tiresiyle bölünmüş sözü yapıştırır.

    ⚠️ Kök gösterimi de tireyle biter (``çap-``); satır sonundaki ``çap-``
    kök mü bölünmüş söz mü ayırt edilemez. Kural: tireden sonraki satır küçük
    harfle başlıyor VE tireden önce en az 2 harf varsa yapıştır — ``çap-\\nkökünden``
    yanlışlıkla ``çapkökünden`` olur; bu yüzden ``kökün``/``ek`` ile başlayan
    devam satırında yapıştırma yapılmaz."""
    out = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if out.endswith("-") and re.match(rf"[{_L}]", line) and not re.match(r"(kökün|ek|\+)", line):
            out = out[:-1] + line
        else:
            out = (out + " " + line) if out else line
    return re.sub(r"\s+", " ", out).strip()


def fix_ot_ocr(form: str) -> str:
    """Gülensoy OCR'ında Eski Türkçe söz başı ``t`` italikte ``f`` okunmuş
    (``fang``, ``fın-``); Eski Türkçede yerli söz başı ``f-`` yoktur."""
    return "t" + form[1:] if form.startswith("f") else form


def clean_root(form: str) -> str:
    """``*kapa`` / ``çap-`` / ``ob-/op-`` -> ``kapa`` / ``çap`` / ``ob``."""
    form = _nfc(form).strip().split("/")[0].split("~")[0].strip()
    form = form.strip("*()[]{}.,;:?!'\"‘’“”").rstrip("-").strip()
    return form


# --- Eren ----------------------------------------------------------------------

_EREN_HEAD = re.compile(
    rf"^(?P<lemma>{_LEMMA})"
    rf"(?P<mid>(?:,\s*{_LEMMA}|\s+(?:\d\.|yer\.|argo|esk\.|hlk\.|den\.|<\s*{_LEMMA}|\([^)]{{0,30}}\)))*)"
    rf"\s*(?P<tail>[‘'“\"*]\s*[{_L}(\d]|bk\.)"
)
#: Eren'in alıntı işareti: ``< Far …``, ``< Ar …``, ``< Erm …`` (büyük harfli dil kısaltması).
_EREN_LOAN = re.compile(r"(?:^|\s)<\s*(?P<donor>[A-ZÇİÖŞÜ][a-zçğıöşü]{0,6})\b\.?(?:\s*\([^)]*\))?\s")
_TURKIC_ABBR = {"Tkm", "Az", "Krg", "Kzk", "Özb", "Tat", "TatK", "Bşk", "Nog", "Tuv", "Yak", "Çuv", "Alt",
                "Uyg", "Kmk", "Blk", "KKlp", "Hak", "Şor", "Tel", "Kır", "Gag", "OT", "Kıp", "Tü", "Çağ"}
_EREN_ROOT = re.compile(
    rf"(?<![\w'’])(?P<root>\*?[{_L}]+-?(?:\s*/\s*[{_L}]+-?)?)\s*"
    rf"(?:\((?:<\s*)?[^)]{{0,40}}\)\s*)?(?:[‘'“][^’'”]{{0,60}}[’'”]\s*)?"
    rf"kökün(?:den|ün|e)"
)
_EREN_SUFFIX = re.compile(r"(?P<suf>-\(?[" + _L + r"(), ]{1,12}?\)?)\s*(?:eki|ekiyle|ekleriyle|ekinden)")
_EREN_DERIV = re.compile(
    rf"(?:^|\s)<\s*:?\s*(?P<root>\*?[{_L}]+-?)\s*(?:[‘'“][^’'”]{{0,60}}[’'”]\s*)?"
    rf"(?:\+\s*(?P<suf>-[{_L}()]+)|(?=[.;,(]|\s*Bk\.|\s*$))"
)
#: ``çay + -lak eki`` / ``kunda- + -k`` (düzyazıda, ``<`` işaretsiz).
_EREN_PLUS = re.compile(rf"(?<![\w'’])(?P<root>\*?[{_L}]+-?)\s*\+\s*(?P<suf>-[{_L}()]+)")
_EREN_OT = re.compile(rf"~\s*OT\s+(?P<form>\*?[{_L}:]+)")
_EREN_MT = re.compile(rf"(?:Orta|Eski) Türkçede\s+(?:d[ae]\s+)?(?P<form>\*?[{_L}:]+)(?:\s*[‘'“]|\s+(?:olarak|biçimi))")
_EREN_SEE = re.compile(rf"\bBk\.\s*(?P<see>{_LEMMA})")
_EREN_REF = re.compile(
    r"(?P<who>R[äâaö]s[äâa]nen|Clauson|[ÈE]STJa|Doerfer|Tietze)\s*:\s*(?P<where>[A-Za-z]{1,6}\s*[\d\s:,\-]*\d\s*[ab]?\b)"
)


def _sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.!?])\s+(?=[A-ZÇİÖŞÜ~<])", text)


_ROOT_STOP = {"türkçe", "bu", "aynı", "bir", "ayrı", "başka", "şu", "fiil", "ad", "isim", "söz", "sözcük",
              "biçim", "türev", "eski", "yeni", "ortak", "o", "kendi", "onun", "bunun"}
_EREN_LOAN_PROSE = re.compile(
    r"\b(?!Türkçe)(?:Arap|Fars|Rum|İtalyan|Erme?ni|Moğol|Fransız|Slav|Bulgar|Sırp|Rus|Alman|İngiliz|Latin|Kürt|"
    r"Gürcü|Macar|Süryani|İbrani|Çin|Soğd)\w{0,3}(?:ça|ce|ca)(?:\s*\([^)]*\))?(?:\s+veya\s+\w+)?(?:dan|den|tan|ten)"
    r"\s+(?:alın|geldi|geçti)"
)


def _header_end(text: str, lemma_end: int) -> int:
    """Madde başı + anlam bölümünün bittiği yer (kök araması gövdede yapılır;
    anlamda geçen "kökünden" — ``engir`` 'üzüm çubuklarının kökünden çıkan' —
    kök bildirimi değildir)."""
    first = len(_sentences(text)[0])
    cut = re.search(r"\s[~<=]", text[lemma_end + 3:])
    if cut:
        first = min(first, lemma_end + 3 + cut.start())
    if first < 250:
        return first
    quote = re.search(r"[’”']", text[lemma_end:])
    return lemma_end + (quote.end() if quote else 0)


def parse_eren_entry(lines: list[str]) -> dict[str, Any] | None:
    text = _join(lines)
    match = _EREN_HEAD.match(text)
    if not match:
        return None
    lemma = match.group("lemma").strip()
    rest = text[match.end("lemma"):]
    rec: dict[str, Any] = {"source": "eren", "lemma": lemma, "key": lemma_key(lemma), "verb_key": verb_key(lemma),
                           "origin": "", "donor": "", "root": "", "suffixes": [], "ot_form": "", "see": "",
                           "redirect": "", "refs": []}
    redirect = re.match(rf"\s*(?:yer\.\s*|argo\s*)?bk\.\s*(?P<to>{_LEMMA})\s*\.?\s*$", rest)
    if redirect:
        rec["redirect"] = redirect.group("to")
        return rec
    body = text[_header_end(text, match.end("lemma")):]
    loan = _EREN_LOAN.search(" " + body)
    if loan and loan.group("donor") not in _TURKIC_ABBR:
        rec["origin"], rec["donor"] = "loan", loan.group("donor")
    if rec["origin"] != "loan":
        for sentence in _sentences(body):
            if rec["root"]:
                break
            if re.search(r"yanlış|yanılmış|katılm|kabul edilemez|doğru değil|birleştirilemez", sentence):
                continue
            for root in _EREN_ROOT.finditer(sentence):
                candidate = root.group("root").strip()
                bare = candidate.strip("*-").casefold()
                if bare in _ROOT_STOP or len(bare) < 2 or re.search(r"(?:n[ıiuü]n|l[ae]r[ıi]n)$", bare):
                    continue
                rec["root"] = candidate
                suffix = _EREN_SUFFIX.search(sentence[root.end():])
                if suffix:
                    rec["suffixes"] = [suffix.group("suf").strip().rstrip(",")]
                break
        if not rec["root"]:
            for sentence in _sentences(body):
                if re.search(r"yanlış|yanılmış|katılm|kabul edilemez|doğru değil|birleştirilemez", sentence):
                    continue
                plus = _EREN_PLUS.search(sentence)
                if plus and plus.group("root").strip("*-").casefold() not in _ROOT_STOP:
                    rec["root"], rec["suffixes"] = plus.group("root"), [plus.group("suf")]
                    break
        if not rec["root"]:
            deriv = _EREN_DERIV.search(" " + body)
            if deriv and deriv.group("root").strip("*-").casefold() not in _ROOT_STOP:
                rec["root"] = deriv.group("root")
                rec["suffixes"] = [deriv.group("suf")] if deriv.group("suf") else []
    ot = _EREN_OT.search(body) or _EREN_MT.search(body)
    if ot:
        rec["ot_form"] = ot.group("form")
    see = _EREN_SEE.search(body)
    if see:
        rec["see"] = see.group("see")
    rec["refs"] = [f"{m.group('who')}: {m.group('where').strip()}" for m in _EREN_REF.finditer(body)][:8]
    if not rec["origin"]:
        turkic_cognates = re.search(r"(?:^|\s)~\s*[A-ZÇÖŞÜ]", body)
        if re.search(r"Kökenini bilmiyoruz|Kökeni bilinmiyor", body) or (
                re.search(r"Kökünü bilmiyoruz", body) and not turkic_cognates):
            rec["origin"] = "unknown"
        elif rec["root"] or rec["ot_form"] or re.search(
                r"(?:^|\s)~\s*[A-ZÇÖŞÜ]|Yalnız Türkçede|[Dd]iyalektlerde|Eski Kıpçakçada|Eski Türkçe", body):
            rec["origin"] = "turkish"
        elif _EREN_LOAN_PROSE.search(body):
            rec["origin"] = "loan"
    if rec["origin"] == "turkish" and not rec["root"] and _EREN_LOAN_PROSE.search(body[:300]) \
            and not re.search(r"(?:^|\s)~\s*[A-ZÇÖŞÜ]", body):
        rec["origin"] = "loan"
    return rec


def _eren_blocks(lines: list[str]) -> Iterator[list[str]]:
    current: list[str] = []
    prev_blank = True
    prev_text = ""
    for raw in lines:
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            prev_blank = True
            continue
        if re.fullmatch(r"\d{1,4}|[A-ZÇĞİÖŞÜ]{1,2}|[ivxlc]{1,5}", stripped):
            continue  # sayfa numarası / harf başlığı
        starts = ((prev_blank or re.search(r"[.’”'!?]$", prev_text)) and not prev_text.endswith("-")
                  and _EREN_HEAD.match(stripped)
                  and not re.match(r"(?:ve|ile|veya|da|de|olarak|biçimi|gibi|bu|bir)\b", stripped))
        if starts and current:
            yield current
            current = []
        current.append(stripped)
        prev_blank = False
        prev_text = stripped
    if current:
        yield current


def parse_eren(path: Path) -> list[dict[str, Any]]:
    lines = _nfc(path.read_text(encoding="utf-8")).splitlines()
    # Ön söz / kısaltmalar atlanır: ilk "a ..." maddesinden başla (``aba``).
    start = next((i for i, ln in enumerate(lines) if re.match(r"^aba\s+1\.", ln.strip())), 0)
    out = []
    for block in _eren_blocks(lines[start:]):
        rec = parse_eren_entry(block)
        if rec:
            out.append(rec)
    return out


# --- Gülensoy -------------------------------------------------------------------

_GUL_HEAD = re.compile(
    rf"^(?P<lemma>{_LEMMA}-?\|?)"
    rf"(?P<mid>(?:\s*(?:\((?:hik|bik|hlk|Hik)\.?\)|\[[^\]]{{0,40}}\]|\d\.|,\s*bk\.))*)"
    rf"\s*(?P<tail>[“\"‘'*]\s*(?:\d\.\s*)?[A-ZÇĞİÖŞÜ(]|Bk\.|bk\.)"
)
#: Türetme satırı: ``<``'in OCR okumaları (``€``, ``x``, ``cx``, ``m``, ``«``, ``<?``).
_GUL_DERIV = re.compile(
    rf"^(?:<|€|«|(?:cx|x|m|c)(?=\s))\s*\??\s*(?P<ot>(?:(?:OT|ET|or)[.,]?\s*(?:\([^)]*\)\s*)?)*)[\"“]?(?P<form>\*?[{_L}][{_L}()\-:]*-?)"
)
_GUL_OT = re.compile(rf"^(?:—|-|=|~)?\s*(?:OT|or|ET)[.,]\s*(?P<form>\*?[{_L}:]+-?)")
_GUL_LOAN_TAG = re.compile(r"\((?:Kürt|Uyg|Kırg|Kzk|Bşk|TatK|Özb|Az|Trkm|Kar|Kum|Nog|Alt|Tuv|Yak|Çuv)\.\)")


def _gul_split(form: str) -> tuple[str, list[str]]:
    """``uç-ar-ı`` -> (``uç-``, [``-ar``, ``-ı``]); ``*kapa`` -> (``*kapa``, [])."""
    form = form.strip()
    if "-" not in form.strip("-"):
        return form, []
    parts = [p for p in form.split("-")]
    root = parts[0] + "-" if parts[0] else ""
    return root, ["-" + p for p in parts[1:] if p]


def _gul_trim_plus(form: str, lemma: str) -> str:
    """OCR'da ``+`` işareti ``t``, ``tH``, ``H``, ``d`` ya da ``(`` okunmuş
    (``sağtla-y-ıcı`` = ``sağ+la-y-ıcı``, ``acıtHla-n-`` = ``acı+la-n-``). Biçim
    madde başının öneki değilse ve ortak önekten hemen sonra bu artıklardan biri
    geliyorsa kök ortak önektir."""
    bare = form.lstrip("*")
    lem = lemma_key(lemma)
    common = 0
    while common < min(len(bare), len(lem)) and bare[common] == lem[common]:
        common += 1
    if common >= 2 and common < len(bare) and not lem.startswith(bare.rstrip("-")) \
            and re.match(r"(?:tH|H|t|d|\(|Y)", bare[common:]):
        return form[: len(form) - len(bare) + common]
    return form


#: Söz başı ses denklikleri (kök ile madde başının ilk sesi uyuşmalı): Oğuz
#: ötümlüleşmesi t/d, k/g, b/p/v; y- düşmesi/eklenmesi ünlüye karşı.
_ONSET = [set("td"), set("kg"), set("bpvm"), set("çc"), set("sş")]


def _onset_ok(root: str, lemma: str) -> bool:
    a, b = root.lstrip("*")[:1], lemma_key(lemma)[:1]
    if not a or not b:
        return False
    return a == b or any(a in grp and b in grp for grp in _ONSET) or (a == "y" and b in "aeıioöuü") \
        or (b == "y" and a in "aeıioöuü")


def _gul_root(form: str, lemma: str, is_ot: bool, line: str) -> tuple[str, list[str]]:
    """Türetme satırından (kök, ekler). Temkinli: biçim OCR artığı taşıyorsa
    (büyük harf, rakam, ``$``, ``#``) ya da kökün ilk sesi madde başıyla
    uyuşmuyorsa kök BOŞ döner — yanlış kök göstermektense kök yok."""
    form = fix_ot_ocr(form) if is_ot else form
    form = form.split("+")[0] if "+" in form else form
    first = form.split("-")[0]
    trimmed = _gul_trim_plus(first, lemma)
    if trimmed != first:
        root, suffixes = trimmed, []
    else:
        root, suffixes = _gul_split(form)
    root = root.split("(")[0] if not root.startswith("(") else root
    bare = root.strip("*-:")
    if len(bare) < 2 or not re.fullmatch(rf"[{_L}:]+", bare) or not _onset_ok(root, lemma):
        return "", []
    if not suffixes:
        plus = re.search(rf"\+\s*-?(?P<suf>[{_L}]+)", line)
        suffixes = ["-" + plus.group("suf")] if plus else []
    return root, suffixes


def parse_gulensoy_entry(lines: list[str]) -> dict[str, Any] | None:
    head = lines[0]
    match = _GUL_HEAD.match(head)
    if not match:
        return None
    lemma = re.sub(r"-[fl]$", "-", match.group("lemma").strip())  # OCR: "dil-|" -> "dil-f"
    rec: dict[str, Any] = {"source": "gulensoy", "lemma": lemma.rstrip("|").strip(), "key": lemma_key(lemma),
                           "verb_key": "", "origin": "turkish", "donor": "", "root": "", "suffixes": [],
                           "ot_form": "", "see": "", "redirect": "", "refs": []}
    if _GUL_LOAN_TAG.search(head[: match.end()]):
        return None  # Kürtçe vb. madde başı (Türkçeden alıntı), Türkiye Türkçesi değil
    redirect = re.match(rf"^{re.escape(lemma)}\s*,?\s*[Bb]k\.\s*(?P<to>{_LEMMA}-?)", head)
    if redirect:
        rec["redirect"] = redirect.group("to")
        return rec
    for raw in lines[1:]:
        line = raw.strip()
        if re.match(r"(?:TT\.?\s*:|An\.\s*ağl|Tü\.|Eren\s*\(|Nişanyan|Kış\.|Krş\.)", line):
            break  # çözümleme bölümü bitti (türevler, ağız biçimleri, başka yazarlar)
        if not rec["ot_form"]:
            ot = _GUL_OT.match(line)
            if ot:
                rec["ot_form"] = fix_ot_ocr(ot.group("form"))
                continue
        if not rec["root"]:
            deriv = _GUL_DERIV.match(line)
            if deriv and len(deriv.group("form").strip("*-()")) >= 2:
                if deriv.group("ot") and not rec["ot_form"] and "-" not in deriv.group("form").strip("-") \
                        and "+" not in line:
                    # "< OT. evin" — türetme değil, Eski Türkçe biçim
                    rec["ot_form"] = fix_ot_ocr(deriv.group("form"))
                    continue
                root, suffixes = _gul_root(deriv.group("form"), lemma, bool(deriv.group("ot")), line)
                if root:
                    rec["root"], rec["suffixes"] = root, suffixes
                break  # yalnız İLK türetme satırı; OCR bozuksa kök boş kalır (tahmin yok)
        see = re.match(rf"^[Bb]k\.\s*(?P<see>{_LEMMA})", line)
        if see and not rec["see"]:
            rec["see"] = see.group("see")
    return rec


def _gul_blocks(lines: list[str]) -> Iterator[list[str]]:
    current: list[str] = []
    prev_text = ""
    for raw in lines:
        stripped = raw.strip()
        if not stripped or re.fullmatch(r"\d{1,4}|Prof\. Dr\. Tuncer G[ÜU]LENSOY.*|.*KÖKEN BİLGİSİ SÖZLÜĞÜ.*",
                                        stripped):
            continue
        starts = (not prev_text.endswith("-") and _GUL_HEAD.match(stripped)
                  and not re.match(r"(?:ve|ile|veya|da|de|olarak|biçimi|gibi|bu|bir|krş)\b", stripped))
        if starts and current:
            yield current
            current = []
        if starts or current:
            current.append(stripped)
        prev_text = stripped
    if current:
        yield current


def parse_gulensoy(paths: Iterable[Path]) -> list[dict[str, Any]]:
    out = []
    for path in paths:
        lines = _nfc(path.read_text(encoding="utf-8")).splitlines()
        start = next((i for i, ln in enumerate(lines) if re.match(r"^a(?:ba)?\s+[“\"]", ln.strip())), 0)
        for block in _gul_blocks(lines[start:]):
            rec = parse_gulensoy_entry(block)
            if rec:
                out.append(rec)
    return out


# --- yazma / okuma ------------------------------------------------------------------


def parse_dir(directory: Path = ERENGUL_DIR) -> list[dict[str, Any]]:
    records = parse_eren(directory / "eren_djvu.txt")
    records += parse_gulensoy([directory / "gulensoy1_djvu.txt", directory / "gulensoy2_djvu.txt"])
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


_CACHE: dict[str, list[dict[str, Any]]] | None = None


def load_index(path: Path | None = None) -> dict[str, list[dict[str, Any]]]:
    """``key`` (ve fiilde ``verb_key``) -> kayıtlar. Dosya yoksa boş sözlük."""
    global _CACHE
    if _CACHE is not None and path is None:
        return _CACHE
    path = path or ERENGUL_DIR / JSONL_NAME
    index: dict[str, list[dict[str, Any]]] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            for key in {rec.get("key"), rec.get("verb_key")} - {"", None}:
                index.setdefault(key, []).append(rec)
    if path == ERENGUL_DIR / JSONL_NAME:
        _CACHE = index
    return index
