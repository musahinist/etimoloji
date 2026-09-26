"""
Clauson 1972, *An Etymological Dictionary of Pre-Thirteenth-Century Turkish*
(EDT) — TurkicWorld Unicode HTML ayrıştırıcısı.

Kaynak: Sir Gerard Clauson, *EDT*, Oxford: Clarendon Press 1972 (OUP
telifi). N. Kisamov'un TurkicWorld sitesindeki "substantially corrected and
annotated" Unicode HTML transkripsiyonu (10 parça, ≈7,5 MB) ``make clauson``
(``scripts/download_clauson.py``) ile ``data/clauson/`` altına indirilir ve bu
modülle JSONL'e ayrıştırılır. HTML de JSONL de repoya ALINMAZ (telifli metin);
künye ve SHA-256'lar ``_provenance.json`` ile commit edilir.

Madde yapısı (HTML ``<p>`` başına bir madde):

* ``[önek] <b>[eşsesli no] madde başı</b> <font blue><i>(anlam)</i></font> gövde``
* Önek (Clauson'un işaretleri): ``D`` türemiş, ``S`` ikincil biçim / gönderme
  (``S čaput See čapğut``), ``F`` yabancı (alıntı), ``VU``/``PU`` seslendirme /
  söyleniş belirsiz, ``C`` bileşik, ``E`` hatalı okuma ya da hayalet söz; ``?``
  ve ayraç (``?D``, ``(D)``) Clauson'un kuşkusudur ve korunur.
* Mavi italik parantez (``<font color="#0000FF">``) Kisamov'un EKLEDİĞİ kısa
  İngilizce (bazen Rusça) anlamdır — Clauson'un değil. Ayrı alanda
  (``gloss_tw``) tutulur; Clauson'un kendi anlamı gövdedeki ilk tırnaklı
  ifadedir (``gloss``). İkisi de İngilizcedir.
* Türemişlerde taban: ``Dev. N. fr. čap-``, ``Den. N./A. fr. 2 taŋ``,
  ``Aor. Particip. of uč-``, ``Caus. f. of sav-`` -> ``base``.
* Tanıklık: gövdede lehçe + yüzyıl etiketleri (``Türkü vııı``, ``Uyğ. vııı
  ff.``, ``Xak. xı``, ``KB``, ``Kaš.``, ``Čağ. xv ff.`` …); OCR'ın Roma
  rakamında i yerine ı yazması düzeltilir.
* ``<b>Dis. V. SBĞ-</b>`` gibi paragraflar ünsüz iskeletine göre bölüm
  başlığıdır, madde değildir. Madde başı ile başlamayan ya da madde başının
  hemen ardından noktalama gelen paragraflar önceki maddenin devamıdır
  (HTML'de satır kırılması paragraf açmış).

⚠️ Clauson Proto-Türkçe kök VERMEZ: madde başı Eski Türkçe (8.-13. yy)
biçimdir; ``base`` da Eski Türkçe tabandır. Motor bunu "Eski Türkçe tabanı
(Clauson)" diye gösterir, Proto-Türkçe kök yerine koymaz.

⚠️ Döngüsellik: Starling turcet REFERENCE alanında EDT 1.504 kez atıflıdır
(EDAL'ın üst kaynağı). Clauson ile Starling uyumu bağımsız doğrulama değildir.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from engine.config import PROJECT_ROOT

CLAUSON_DIR = PROJECT_ROOT / "data" / "clauson"
JSONL_NAME = "clauson_edt.jsonl"
#: TurkicWorld parça adları (basılı sayfa aralığı, PDF sayfa aralığı).
PARTS: tuple[str, ...] = (
    "1-100(47-146)", "101-201(147-247)", "202-300(248-346)", "301-400(347-446)",
    "401-500(447-546)", "501-600(547-646)", "601-700(647-746)", "701-800(747-846)",
    "801-900(847-946)", "901-988(947-1035)",
)
BASE_URL = "http://s155239215.onlinehome.us/turkic/40_Language/ClausonEDT/Clauson_EDTp{}.htm"


def part_file(part: str) -> str:
    return f"Clauson_EDTp{part}.htm"


_BOLD_OPEN, _BOLD_CLOSE = "\x01", "\x02"
_BLUE = re.compile(r'<font[^>]*color="?#0000FF"?[^>]*>(.*?)</font>', re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_PARA = re.compile(r"<p[^>]*>(.*?)</p>", re.S | re.I)
#: Bölüm başlığı: "Mon. V. A-", "Dis. SBA", "Tris. V. BRN-", "Preliminary note."
_SECTION = re.compile(r"^(?:Mon|Mono|Dis|Tris|Tetras|Pent|Poly)\.?\s|^Preliminary note", re.I)
_PREFIX = r"(?P<pre>(?:[(?]*\s*(?:VUPF|VU|PU|D|S|F|E|C)\s*[?)]*\s*){0,3})(?P<num0>\d\s+)?"
_ENTRY_BOLD = re.compile(_PREFIX + _BOLD_OPEN + r"(?P<head>[^" + _BOLD_CLOSE + r"]{1,80})" + _BOLD_CLOSE + r"(?P<rest>.*)", re.S)
_ENTRY_PLAIN = re.compile(r"(?P<pre>(?:[(?]*\s*(?:VUPF|VU|PU|D|S|F|E|C)\s*[?)]*\s*){1,3})\s+(?P<head>[^\s" + _BOLD_OPEN + r"]{1,40})(?P<rest>\s.*)", re.S)
_HEAD_OK = re.compile(r"^[a-zA-Zçčğıöüšŋñäéıīāūōēİ:'()\-/ ,?.]+$")
_NUM = re.compile(r"^(\d)\s+")
_SEE = re.compile(r"^\W{0,3}See\s+" + _BOLD_OPEN + r"?\s*(?:(\d)\s+)?([^\s" + _BOLD_CLOSE + r",;.]+)")
#: Türemiş maddede taban: "fr." ya da "... f. of" / "Particip. of" ardından ilk biçim.
_BASE_FR = re.compile(
    r"\bfr\.?\s+(?:an?\s+[^,;]{0,40}?\s+fr\.?\s*)?(?:(\d)\s+)?" + _BOLD_OPEN + r"?\s*(?:(\d)\s+)?\W?"
    r"(\*?[a-zçčğıöüšŋñäé:'\-]+)"
)
_BASE_OF = re.compile(
    r"\b(?:f\.|Particip\.|Co-op\.|Ger\.|Dim\.)\s+of\s+(?:(\d)\s+)?" + _BOLD_OPEN + r"?\s*(?:(\d)\s+)?"
    r"(\*?[a-zçčğıöüšŋñäé:'\-]+)"
)
_DERIVATION_START = re.compile(
    r"^(?:Hap\. leg\.[;,]?\s*)?(?:[A-Z][\w./-]*\s+){1,5}(?:\(?[A-Za-z. ]{0,30}\)?\s*)?(?:fr\.|of)\s"
)
_QUOTE = re.compile(r"‘(?P<g>[^’‘]{2,160}?)[’'](?![a-zA-Z])")
_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8, "ix": 9,
          "x": 10, "xi": 11, "xii": 12, "xiii": 13, "xiv": 14, "xv": 15, "xvi": 16, "xvii": 17,
          "xviii": 18, "xix": 19, "xx": 20}
#: Clauson'un lehçe etiketleri (Giriş, s. xxvii vd.).
DIALECTS: tuple[str, ...] = ("Türkü", "Uyğ", "Xak", "Oğuz", "Arğu", "Čiğil", "Yağma", "Kip", "Xwar",
                             "Čağ", "Osm", "Kom", "Tkm", "Az")
_ATTEST = re.compile(
    r"\b(" + "|".join(re.escape(d) for d in DIALECTS) + r")\.?\s*([xvıi]{1,5})\b(\s*ff\.)?"
)


def _roman(text: str) -> int | None:
    return _ROMAN.get(text.replace("ı", "i").lower())


def _clean(fragment: str) -> str:
    text = fragment.replace("<b>", _BOLD_OPEN).replace("</b>", _BOLD_CLOSE)
    text = re.sub(r"<b\s[^>]*>", _BOLD_OPEN, text)
    text = _TAG.sub("", text)
    text = html.unescape(text).replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip()


def _plain(text: str) -> str:
    return text.replace(_BOLD_OPEN, "").replace(_BOLD_CLOSE, "")


def split_headword(raw: str) -> dict[str, Any]:
    """``"1 bele:- (be:le:-)"`` -> no 1, ``bele:-``, varyant ``be:le:-``;
    ``"tilge: (d-)"`` -> ``d-`` sonraki dillerde söz başı ötümlüleşme notu."""
    raw = raw.strip().rstrip(".,;").strip()
    out: dict[str, Any] = {"homonym": None, "variants": [], "initial_note": None, "hypothetical": False}
    if raw.startswith("*"):
        out["hypothetical"] = True
        raw = raw[1:].lstrip()
    m = _NUM.match(raw)
    if m:
        out["homonym"] = int(m.group(1))
        raw = raw[m.end():]
    for paren in re.findall(r"\(([^)]*)\)", raw):
        paren = paren.strip()
        if re.fullmatch(r"[dgbčy]-", paren):
            out["initial_note"] = paren
        elif paren and not paren.startswith(("or ", "sic")):
            out["variants"].append(paren)
    raw = re.sub(r"\([^)]*\)", " ", raw).strip()
    parts = [p.strip() for p in re.split(r"[/,]", raw) if p.strip()]
    head = parts[0] if parts else raw
    out["variants"] = [*parts[1:], *out["variants"]]
    if head.startswith("*"):
        out["hypothetical"] = True
        head = head[1:]
    # OCR: ŋ kimi yerde "rj" okunmuş (kočrja:r, arju:la:-); Türkçe sözde rj yok.
    out["headword"] = head.strip().replace("rj", "ŋ")
    return out


def attestations(body: str) -> list[dict[str, Any]]:
    """Gövdedeki lehçe/yüzyıl etiketleri ve tarihli eserler (sıralı, tekil)."""
    text = _plain(body)
    found: list[dict[str, Any]] = []
    seen: set[tuple[str, int | None, bool]] = set()
    for m in _ATTEST.finditer(text):
        century = _roman(m.group(2))
        if century is None:
            continue
        key = (m.group(1), century, bool(m.group(3)))
        if key not in seen:
            seen.add(key)
            found.append({"dialect": m.group(1), "century": century, "ff": bool(m.group(3))})
    works = []
    if re.search(r"\bKaš\.", text):
        works.append("dlt")
    if re.search(r"\bKB\b", text):
        works.append("kb")
    if any(a["dialect"] == "Türkü" and a["century"] == 8 and not a["ff"] for a in found):
        works.insert(0, "orhun")
    return [*found, *({"work": w} for w in works)]


def _base(rest: str) -> tuple[int | None, str] | None:
    head = rest[:260]
    found = [m for m in (_BASE_FR.search(head), _BASE_OF.search(head)) if m]
    if not found:
        return None
    m = min(found, key=lambda m: m.start())
    number = m.group(1) or m.group(2)
    form = m.group(3).strip(".,;:'")
    if len(form.replace(":", "").replace("-", "")) < 2 or form in {"the", "a", "an", "this", "it"}:
        return None
    return (int(number) if number else None), form


_MARK = "\x03"


def _own_gloss(marked: str, blue: list[str]) -> str:
    """Madde başının HEMEN ardındaki mavi anlam (Kisamov); gövdenin ilerisindeki
    mavi parantez tabanın ya da gönderilen maddenin anlamıdır, alınmaz."""
    close = marked.find(_BOLD_CLOSE)
    tail = marked[close + 1:] if close >= 0 else marked
    for m in re.finditer(_MARK + r"(\d+)" + _MARK + r"|(\S)", tail):
        if m.group(2) is not None:
            if m.group(2) in (_BOLD_OPEN, _BOLD_CLOSE):
                continue
            return ""
        gloss = _plain(blue[int(m.group(1))]).strip()
        if gloss.strip("() ") and not re.fullmatch(r"[\d\s\\]+", gloss):
            return re.sub(r"^\(|\)$", "", gloss).strip()
    return ""


def parse_paragraphs(paragraphs: Iterable[tuple[str, str]], keep_raw: bool = False) -> Iterator[dict[str, Any]]:
    """``(parça, <p> iç HTML'i)`` akışından madde kayıtları."""
    current: dict[str, Any] | None = None
    for part, fragment in paragraphs:
        blue = [_clean(b) for b in _BLUE.findall(fragment)]
        pieces = _BLUE.split(fragment)
        # split: [metin, mavi, metin, mavi, ...] (tek yakalama grubu)
        marked = _clean("".join(
            piece if k % 2 == 0 else f" {_MARK}{k // 2}{_MARK} " for k, piece in enumerate(pieces)))
        text = re.sub(r"\s+", " ", re.sub(_MARK + r"\d+" + _MARK, " ", marked)).strip()
        plain = _plain(text).strip()
        if not plain:
            continue
        if _SECTION.match(plain):
            if current:
                yield current
            current = None
            continue
        m = _ENTRY_BOLD.match(text) or _ENTRY_PLAIN.match(text)
        head_ok = bool(m) and bool(_HEAD_OK.match(re.sub(r"^\*?\d?\s*\*?", "", m.group("head").strip()))) and any(
            c.isalpha() for c in m.group("head"))
        rest = m.group("rest") if m else ""
        continuation = (not m or not head_ok
                        or (not m.group("pre").strip() and re.match(r"^\s*[;,.:)\]0-9]", rest)))
        if continuation:
            if current is not None:
                current["body"] += " " + text
            continue
        if current:
            yield current
        record = split_headword((m.groupdict().get("num0") or "") + m.group("head"))
        pre = re.sub(r"\s+", "", m.group("pre"))
        gloss_tw = _own_gloss(marked, blue)
        current = {
            "part": part,
            "prefix": pre,
            **record,
            "gloss_tw": gloss_tw,
            "body": rest.strip(),
        }
        if keep_raw:
            current["_raw"] = _plain(text)[:400]
    if current:
        yield current


def finalize(record: dict[str, Any]) -> dict[str, Any]:
    """Gövdeden gönderme, taban, anlam ve tanıklığı çıkarır; gövdeyi atar
    (telifli metin JSONL'de yalnız kısa alıntı olarak kalır)."""
    body = record.pop("body", "")
    pre = record.get("prefix", "")
    see = _SEE.match(body)
    # "S X See Y" ya da öneki başka olup gövdesi yalnız gönderme olan madde
    # ("F čawga:n See čögen").
    is_ref = bool(see) and ("S" in pre or len(_plain(body)) < 60)
    record["see"] = see.group(2).strip(".,;") if is_ref else None
    record["see_homonym"] = int(see.group(1)) if is_ref and see.group(1) else None
    # Önek OCR'da düşmüş olabilir ("yarlıkančsız Priv. N./A. fr. …"): öneksiz
    # maddede taban yalnız gövde bir türetme formülüyle BAŞLIYORSA alınır.
    derived = "D" in pre or (not pre and bool(_DERIVATION_START.match(_plain(body))))
    base = _base(body) if derived else None
    record["base"] = base[1] if base else None
    record["base_homonym"] = base[0] if base else None
    quote = _QUOTE.search(_plain(body))
    record["gloss"] = quote.group("g").strip() if quote else ""
    record["attestations"] = attestations(body)
    record["derivation_note"] = _plain(body[: body.find(";")] if 0 < body.find(";") < 90 else "")[:90]
    return record


def parse_html(text: str, part: str) -> list[dict[str, Any]]:
    body_start = text.find("<body")
    paragraphs = ((part, p) for p in _PARA.findall(text[body_start:]))
    return [finalize(r) for r in parse_paragraphs(paragraphs)]


def parse_dir(directory: Path = CLAUSON_DIR) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for part in PARTS:
        path = directory / part_file(part)
        records.extend(parse_html(path.read_text(encoding="utf-8-sig"), part))
    for i, record in enumerate(records):
        record["id"] = i
    return records


def write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_records(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or CLAUSON_DIR / JSONL_NAME
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
