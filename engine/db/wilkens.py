"""
Wilkens 2021, *Handwörterbuch des Altuigurischen* — PDF ayrıştırıcı.

Kaynak: Jens Wilkens, *Handwörterbuch des Altuigurischen. Altuigurisch –
Deutsch – Türkisch*, Göttingen: Universitätsverlag Göttingen 2021,
DOI 10.17875/gup2021-1590, CC BY-SA 4.0. ``make wilkens`` açık erişim
PDF'ini ``data/wilkens/`` altına indirir ve bu modülle JSONL'e ayrıştırır;
ikisi de repoya alınmaz, künye (SHA-256) commit edilir.

Sözlüğün yapısı (Vorbemerkungen, s. II-VI):

* İki sütun. Ana madde başı KALIN (dolgu+kontur ile basılmış; pdfminer'da
  ``graphicstate.scolor`` sıfırdan farklı), sütunun sol kenarından başlar;
  eşsesli numarası kalın üst simge (``¹ačıt-``). Devam satırları da sol
  kenardan başlar, yani madde sınırını YALNIZ kalın yazı belirler.
* Alt madde (ikileme, kalıp söz) İTALİK, ~12 pt içeriden.
* Anlam: ``Almanca || Türkçe``; birden çok anlam ``; `` ile sıralanır
  (``DE1 || TR1; DE2 || TR2``). Anlamdaki üst simge rakam (``quälen²``)
  ikileme sayısıdır, atılır.
* Köken: madde başının hemen ardından ``< Dil biçim`` (doğrudan) ya da
  ``<< Dil biçim`` (dolaylı) zinciri; dil kısaltması düz, biçim italik
  (``< TochB ajite < Skt. ajita``).
* ``†`` hatalı okuma (madde değil), ``→`` gönderme; ``(r)`` runik, ``(br)``
  Brāhmī, ``(m)`` Maniheist, ``(c)`` Hristiyan, ``(tib)`` Tibet yazısı;
  ``n. pr.`` özel ad, ``n. loc.`` yer adı.

⚠️ Sözlük **tanık yeri vermez** ("Stellenangaben können … nicht gegeben
werden", s. II): maddeye özgü metin ya da yıl yoktur. Eski Uygurca metinler
9.-14. yüzyıla yayılır; bir maddenin tanığı bu dönemin HERHANGİ bir
yerinde olabilir.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from engine.config import PROJECT_ROOT

WILKENS_DIR = PROJECT_ROOT / "data" / "wilkens"
PDF_NAME = "Wilkens_handwoerterbuch.pdf"
JSONL_NAME = "wilkens_oui.jsonl"
#: Sözlüğün ilk sayfası (0 tabanlı PDF sayfa sırası; basılı s. 1).
FIRST_PAGE = 15

#: Anlamdaki ve madde başındaki üst simgeler bu boyuttan küçüktür (9,2 pt'ye karşı 5,4).
SMALL_SIZE = 7.0
#: Aynı satır sayılan taban çizgisi farkı (pt). Kaydırılmış italik biçimler
#: (``abhayakῑrti`` 400'de, satır 399'da) satıra katılmalı.
LINE_TOLERANCE = 2.0
#: Alt madde girintisi sütun kenarından ~12 pt; bundan büyük kayma alt maddedir.
INDENT = 6.0
#: Sütunların sol kenarı (pt). Ölçüldü (s. 1-105, 2.818 kalın satır başı):
#: sol sütun hep 109,4, sağ sütun hep 312,2 — tek/çift sayfada aynı.
COLUMN_LEFT = {0: 109.4, 1: 312.2}

#: Köken zincirindeki dil kısaltması -> motorun Türkçe dil adı. Kısaltmalar
#: sözlüğün "Abkürzungen" listesinden (s. VII-VIII).
DONOR_LANGUAGES: dict[str, str] = {
    "Skt.": "Sanskritçe", "BHS": "Sanskritçe",
    "Toch": "Toharca", "TochA": "Toharca", "TochB": "Toharca", "PrototochA": "Toharca",
    "Sogd.": "Soğdca", "MSogd.": "Soğdca", "CSogd.": "Soğdca",
    "Chin.": "Çince", "chin.": "Çince", "Spätmittelchin.": "Çince",
    "Mo.": "Moğolca", "MMo.": "Moğolca",
    "MP": "Orta Farsça", "Parth.": "Partça", "Mitteliran.": "Orta İranca",
    "Neupers.": "Farsça", "Pers.": "Farsça",
    "Baktr.": "Baktriyaca", "Khotansak.": "Hotan Sakacası", "Altkhotansak.": "Hotan Sakacası",
    "Spätkhotansak.": "Hotan Sakacası", "Tumšuksak.": "Tumşuk Sakacası",
    "Tib.": "Tibetçe", "tib.": "Tibetçe", "Tang.": "Tangutça",
    "Syr.": "Süryanice", "Gr.": "Grekçe", "Hebr.": "İbranice", "Arab.": "Arapça",
    "Lat.": "Latince", "Mittelind.": "Orta Hint dili",
}

_MARKS = {"(r)": "runic", "(br)": "brahmi", "(m)": "manichaean", "(c)": "christian",
          "(tib)": "tibetan", "(p)": "phags_pa", "(H)": "hybrid"}
_DONOR_RE = re.compile(r"^\s*\(?\s*(<<?)\s*(?:zu\s+)?([A-Za-zÄÖÜäöüšŠ]+\.?)\s*")
_MARK_PREFIX = re.compile(r"^\s*(?:\((?:r|br|m|c|tib|p|H)\)\s*)+")


# --- PDF -> satır -------------------------------------------------------------

@dataclass
class Token:
    """Aynı biçemli ardışık karakterler: ``B`` kalın, ``I`` italik, ``r`` düz."""

    style: str
    small: bool
    text: str


@dataclass
class Line:
    column: int
    x: float          # ilk boş olmayan karakterin x'i
    y: float          # taban çizgisi
    tokens: list[Token] = field(default_factory=list)
    page: int = 0

    @property
    def text(self) -> str:
        return "".join(t.text for t in self.tokens)


def _chars(obj: Any, out: list) -> None:
    from pdfminer.layout import LTChar

    if isinstance(obj, LTChar):
        out.append(obj)
        return
    if hasattr(obj, "__iter__"):
        for child in obj:
            _chars(child, out)


def _style(char: Any) -> str:
    scolor = getattr(char.graphicstate, "scolor", None)
    if scolor not in (None, 0, 0.0) and scolor != (0,) and scolor != [0]:
        return "B"
    return "I" if "Italic" in char.fontname else "r"


def page_lines(page: Any, page_no: int = 0) -> list[Line]:
    """Bir PDF sayfasının karakterlerini sütun + taban çizgisiyle satırlara böler."""
    raw: list = []
    _chars(page, raw)
    # Kalın harflerin bir kısmı (satırın ilk harfi) iki kez basılır: bir düz,
    # bir kalın. Aynı yerdeki aynı harf bir kez sayılır, kalın olan tutulur.
    seen: dict[tuple[str, int, int], Any] = {}
    for c in raw:
        key = (c.get_text(), round(c.x0 * 2), round(c.y0 * 2))
        if key not in seen or _style(c) == "B":
            seen[key] = c
    chars = list(seen.values())
    half = page.width / 2
    big = [c for c in chars if c.size >= SMALL_SIZE]
    small = [c for c in chars if c.size < SMALL_SIZE]
    rows: list[tuple[int, float, list]] = []
    for c in sorted(big, key=lambda c: (c.x0 > half, -c.y0)):
        col = int(c.x0 > half)
        for row in rows:
            if row[0] == col and abs(row[1] - c.y0) <= LINE_TOLERANCE:
                row[2].append(c)
                break
        else:
            rows.append((col, c.y0, [c]))
    # Üst simge, taban çizgisi kendisinin 0-7 pt altındaki satıra aittir.
    for c in small:
        col = int(c.x0 > half)
        best = None
        for row in rows:
            if row[0] == col and -1.0 <= c.y0 - row[1] <= 7.0:
                if best is None or c.y0 - row[1] < c.y0 - best[1]:
                    best = row
        if best is not None:
            best[2].append(c)
    lines: list[Line] = []
    for col, y, members in rows:
        members.sort(key=lambda c: c.x0)
        tokens: list[Token] = []
        for c in members:
            style, is_small = _style(c), c.size < SMALL_SIZE
            if tokens and tokens[-1].style == style and tokens[-1].small == is_small:
                tokens[-1].text += c.get_text()
            else:
                tokens.append(Token(style, is_small, c.get_text()))
        first = next((c for c in members if c.get_text().strip()), None)
        if first is None:
            continue
        lines.append(Line(col, first.x0, y, tokens, page_no))
    lines.sort(key=lambda ln: (ln.column, -ln.y))
    return lines


def iter_pdf_lines(path: Path, pages: Iterable[int] | None = None) -> Iterator[Line]:
    from pdfminer.high_level import extract_pages

    numbers = list(pages) if pages is not None else None
    for index, page in enumerate(extract_pages(str(path), page_numbers=numbers, laparams=None)):
        page_no = numbers[index] if numbers is not None else index
        if page_no < FIRST_PAGE:
            continue
        lines = page_lines(page, page_no)
        # Sayfa numarası (alt kenar) ve boş başlık satırları.
        lines = [ln for ln in lines if not re.fullmatch(r"\s*\d*\s*", ln.text)]
        yield from lines


# --- satır -> madde -----------------------------------------------------------

def _is_main_start(line: Line, left: float) -> bool:
    body = [t for t in line.tokens if t.text.strip()]
    return bool(body) and body[0].style == "B" and line.x <= left + INDENT


def _is_sub_start(line: Line, left: float) -> bool:
    body = [t for t in line.tokens if t.text.strip() and not t.small]
    return bool(body) and body[0].style == "I" and line.x > left + INDENT


def _initial(text: str) -> str:
    m = re.search(r"[^\W\d_]", text)
    return m.group(0).lower() if m else ""


def _same_initial(line: Line, entry: dict[str, Any]) -> bool:
    """Alt madde hep ana maddenin ilk öğesiyle başlar (s. I); italik başlayan
    devam satırı (sarkan köken biçimi ``śāstra``) alt madde sayılmaz."""
    first = next(t for t in line.tokens if t.text.strip() and not t.small)
    head = next((t for t in entry["lines"][0] if t.text.strip() and not t.small), None)
    return head is None or _initial(first.text) == _initial(head.text)


def group_entries(lines: Iterable[Line]) -> list[dict[str, Any]]:
    """Satırları ana madde (+ alt maddeler) token listelerine böler.

    Sütunun sol kenarı sabittir (``COLUMN_LEFT``); sayfadaki en küçük x
    kullanılamaz, çünkü yalnız alt maddelerden oluşan sütunda (s. 583,
    ``sansarlıg …``) o x alt madde girintisidir.
    """
    lines = list(lines)
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    target: list[list[Token]] | None = None
    for ln in lines:
        left = COLUMN_LEFT[ln.column]
        if _is_main_start(ln, left):
            current = {"page": ln.page + 1 - FIRST_PAGE, "lines": [ln.tokens], "subs": []}
            entries.append(current)
            target = current["lines"]
        elif current is not None and _is_sub_start(ln, left) and _same_initial(ln, current):
            current["subs"].append([ln.tokens])
            target = current["subs"][-1]
        elif target is not None:
            target.append(ln.tokens)
    return entries


def _join(lines: list[list[Token]]) -> list[Token]:
    """Satırları tek token dizisine birleştirir; satır sonu tirelemesini (düz
    metinde ``ilerle-`` + ``mek``) kaldırır, küçük düz rakamları (ikileme
    sayısı) atar."""
    out: list[Token] = []
    for index, tokens in enumerate(lines):
        tokens = [Token(t.style, t.small, t.text) for t in tokens
                  if not (t.small and t.style != "B" and t.text.strip().isdigit())]
        if index and out and tokens:
            prev = out[-1]
            first = next((t for t in tokens if t.text.strip()), None)
            stripped = prev.text.rstrip()
            if (prev.style == "r" and stripped.endswith("-") and first is not None
                    and first.text.lstrip()[:1].islower() and first.style == "r"
                    and not stripped.endswith(" -")):
                prev.text = stripped[:-1]
                first.text = first.text.lstrip()
            else:
                prev.text = prev.text.rstrip() + " "
        for t in tokens:
            if not t.text.strip() and not t.small:
                # boşluk biçem taşımaz ("<" + kalın " " + " Skt.")
                t = Token(out[-1].style if out else "r", False, t.text)
            if out and out[-1].style == t.style and out[-1].small == t.small:
                out[-1].text += t.text
            else:
                out.append(t)
    return out


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" ,;")


def _last_top_level(text: str, char: str) -> int:
    """Parantez dışındaki son ``char`` konumu (yoksa -1): "(… Dhātus; auch
    Äquivalent …)" içindeki noktalı virgül anlam ayracı değildir."""
    depth, found = 0, -1
    for i, ch in enumerate(text):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth = max(0, depth - 1)
        elif ch == char and depth == 0:
            found = i
    return found


def split_gloss(text: str) -> tuple[list[str], list[str]]:
    """``DE1 || TR1; DE2 || TR2`` -> (Almanca anlamlar, Türkçe anlamlar)."""
    parts = [p.strip() for p in text.split("||")]
    if len(parts) < 2:
        return ([_clean(text)] if _clean(text) else []), []
    german, turkish = [parts[0]], []
    for middle in parts[1:-1]:
        cut = _last_top_level(middle, ";")
        if cut >= 0:
            turkish.append(middle[:cut])
            german.append(middle[cut + 1:])
        else:  # ayraç yok: bütünü Türkçe say (Almanca anlam kaybolur)
            turkish.append(middle)
    turkish.append(parts[-1])
    return [g for g in map(_clean, german) if g], [t for t in map(_clean, turkish) if t]


def parse_donor_chain(tokens: list[Token]) -> tuple[list[dict[str, str]], list[Token]]:
    """Madde başından sonraki ``< Dil biçim`` zincirini ayırır.

    Dil kısaltması düz, biçim italik yazılır. Zincir, italik biçimden sonra
    ``<``, ``~`` ya da ``/`` ile sürmeyen ilk düz metinde biter.
    """
    chain: list[dict[str, str]] = []
    rest = list(tokens)
    while rest:
        head = rest[0]
        if head.style == "I" and chain and not chain[-1]["form"]:
            chain[-1]["form"] = _clean(head.text)
            rest.pop(0)
            continue
        if head.style == "I" and chain:  # "~ ikinci biçim"
            rest.pop(0)
            continue
        if head.style not in ("r", "B"):
            break
        text = head.text
        m = _DONOR_RE.match(text)
        if m:
            abbr = m.group(2)
            chain.append({"lang": abbr, "lang_name": DONOR_LANGUAGES.get(abbr, ""),
                          "direct": m.group(1) == "<", "form": ""})
            remainder = text[m.end():]
            # "<< Skt. *" -> yıldız biçime aittir
            remainder = remainder.lstrip("*")
            if remainder.strip():
                rest[0] = Token(head.style, head.small, remainder)
                if chain[-1]["form"] == "" and rest[0].style == "r":
                    # biçim italik değil (nadir) — zinciri burada bırak
                    break
            else:
                rest.pop(0)
            continue
        if chain and re.match(r"\s*[~/,)]", text):
            # "/ < TochB", "(< TochA/B biçim ~ TochB biçim < Skt. …)", "~ biçim"
            text = re.sub(r"^\s*[~/,)]\s*(?:[AB]\b\s*)?", "", text)
            alt = re.match(r"([A-Za-zÄÖÜäöüšŠ]+\.?)\s*", text)
            if alt and alt.group(1) in DONOR_LANGUAGES and not text.lstrip().startswith("<"):
                text = text[alt.end():]
            if text.strip():
                rest[0] = Token(head.style, head.small, text)
            else:
                rest.pop(0)
            continue
        break
    return chain, rest


def parse_entry(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Gruplanmış bir ana maddeyi kayda çevirir; madde başı yoksa ``None``."""
    tokens = _join(raw["lines"])
    homonym = None
    head_parts: list[str] = []
    rest = list(tokens)
    while rest and (rest[0].style == "B" or not rest[0].text.strip()):
        t = rest.pop(0)
        if t.small and t.text.strip().isdigit() and not head_parts:
            homonym = int(t.text.strip()[-1])
        elif not t.small:
            head_parts.append(t.text)
    headword_text = _clean("".join(head_parts))
    if not headword_text:
        return None
    variants = [_clean(v) for v in headword_text.split("~") if _clean(v)]
    headword = variants[0]
    marks_text = "".join(t.text for t in rest)[:14]
    if rest and _MARK_PREFIX.match(rest[0].text):
        rest[0] = Token(rest[0].style, rest[0].small, _MARK_PREFIX.sub("", rest[0].text))
    chain, rest = parse_donor_chain(rest)
    body = _clean("".join(t.text for t in rest))
    record: dict[str, Any] = {
        "headword": headword,
        "homonym": homonym,
        "variants": variants[1:],
        "page": raw["page"],
    }
    if chain:
        record["donor_chain"] = chain
    marks = [name for mark, name in _MARKS.items() if mark in marks_text or mark in headword_text]
    if marks:
        record["marks"] = marks
    if body.startswith("†") or " † " in f" {body[:4]} ":
        record["error"] = True
    if "→" in body[:4] or (body.startswith("(") and "→" in body[:12] and "||" not in body):
        target = body.split("→", 1)[1].strip()
        record["see"] = _clean(target.split("||")[0])
    body = re.sub(r"^(?:\((?:r|br|m|c|tib|p|H)\)\s*)+", "", body)
    record["proper_name"] = bool(re.match(r"n\. (?:pr|loc)\.", body)) or headword[:1].isupper()
    german, turkish = split_gloss(body) if "||" in body else ([_clean(body)] if body else [], [])
    record["de"] = german
    record["tr"] = turkish
    subs = []
    for sub_lines in raw["subs"]:
        lead: list[str] = []
        srest = list(_join(sub_lines))
        while srest:
            t = srest[0]
            if t.style == "I" or not t.text.strip():
                lead.append(srest.pop(0).text)
            elif lead and t.text.startswith("-"):
                # fiil tiresi düz basılmış: "ačıt" + "- " + "agrıt" + "- anlam"
                lead.append("- " if t.text[1:2].isspace() else "-")
                srest[0] = Token(t.style, t.small, t.text[1:])
                if not srest[0].text.strip():
                    srest.pop(0)
            else:
                break
        form = _clean("".join(lead))
        sbody = _clean("".join(t.text for t in srest))
        sde, str_ = split_gloss(sbody) if "||" in sbody else ([sbody] if sbody else [], [])
        if form:
            subs.append({"form": form, "de": sde, "tr": str_})
    if subs:
        record["subentries"] = subs
    return record


def parse_lines(lines: Iterable[Line]) -> list[dict[str, Any]]:
    out = []
    for raw in group_entries(lines):
        record = parse_entry(raw)
        if record is not None:
            out.append(record)
    return out


def parse_pdf(path: Path, pages: Iterable[int] | None = None) -> list[dict[str, Any]]:
    return parse_lines(iter_pdf_lines(path, pages))


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_records(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or WILKENS_DIR / JSONL_NAME
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
