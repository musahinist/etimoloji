"""
Tarihli eserler → yıl haritası (TEK TANIM).

Eskiden aynı eserin yılı üç yerde, üç ayrı değerle yazılıydı:

* ``db/starling.SOURCE_DATES``: MK 1072, Orkh. 732, AH 1300
* ``nlp/historical_attestation_verifier.DATED_SOURCES``: DLT 1074, Orhun 735,
  Atebetü'l-Hakayık 1150
* EtimolojiTürkçe (Nişanyan türevi) metni: "Divan-i Lugat-it Türk (1070)"

Aynı kelime hangi kaynaktan tanık aldığına göre 1070, 1072 ya da 1074;
Atebet'te 1150 ya da 1300 alıyordu. Artık bütün okuyucular yılı buradan alır.

Yıl seçme ilkesi: **eserin o tarihte VAR OLDUĞU kesin olan en erken yıl**
(tamamlanma yılı; tarihi tartışmalıysa üst sınır). Gerekçe: bu yıl A-HVP
2. aşamasında (``ChronologicalTimeLock``) "verici dil teması tanıklamadan
SONRA olamaz" kilidinin girdisidir. Erken bir yıl tartışmalı bir tarihte
yanlış ANAKRONİZM üretir; geç bir yıl yalnız "en geç o yılda tanıklı" der,
bu da doğrudur. Aynı ilke Wilkens Eski Uygurca tanıklarında da uygulanıyor
(dönem etiketi "9.-14. yy" → dönemin sonu).

Değerler ve kaynakları
----------------------
* **Orhun yazıtları — 732.** Köl Tigin yazıtı 732'de dikildi (Çince yüzü
  tarihli), Bilge Kağan 735. Tonyukuk ~720'ler ama kesin tarihli değil.
  Starling'in ``Orkh.`` etiketi hepsini kapsar; kesin tarihli EN ERKEN yazıt
  Köl Tigin'dir. (Tekin, *A Grammar of Orkhon Turkic*, 1968, s. 3-8.)
* **Kutadgu Bilig — 1069.** Yusuf Has Hâcib eseri H. 462'de (1069-70)
  bitirdiğini yazar. (Arat, *Kutadgu Bilig I. Metin*, 1947, giriş.)
* **Dîvânu Lugâti't-Türk — 1074.** Kâşgarlı Mahmud yazmaya H. 464'te (1072)
  başladı, H. 466'da (1074) bitirdi. 1072 başlangıçtır; "1070" (EtimolojiTürkçe/
  Nişanyan) hiçbir tarihle örtüşmüyor. Tamamlanma yılı alınır.
  (Dankoff & Kelly, *Compendium of the Turkic Dialects*, 1982, I, s. 1-7;
  Atalay, *Divanü Lûgat-it-Türk Tercümesi*, 1939, giriş.)
* **Atebetü'l-Hakayık — 1300.** Edib Ahmed Yüknekî'nin tarihi tartışmalı:
  12. yy'dan 13. yy sonuna kadar öneriler var; en eski nüsha 1444 (Semerkant).
  Nişanyan "1300 yılından önce" diyor, Starling ``AH`` 1300. Eski 1150 değeri
  aralığın ALT ucuydu ve ilkeye aykırıydı. (Arat, *Atebetü'l-Hakayık*, 1951,
  giriş; Nişanyan Sözlük kaynakçası.)
* **İbn Mühennâ lügati — 1245** (yaklaşık; 13. yy ortası; Starling ``IM``).
  Tarih tartışmalı (13. yy ortası - 14. yy başı); Starling'in değeri korundu
  çünkü başvuru ölçümü (``chronology_eval``) bu etiketi kullanıyor ve
  değiştirecek kaynaklı bir tek değer yok.
* **Codex Cumanicus — 1303.** Venedik nüshasının ilk yaprağındaki tarih
  (11 Temmuz 1303). (Grønbech, *Komanisches Wörterbuch*, 1942, giriş.)
* **Abuşka — 1500** (16. yy Çağatayca-Osmanlıca sözlük; yaklaşık, Starling).
* **Senglâh (Sanglax) — 1760.** Mehdî Han, *Senglâh*, 1759-60 (Clauson 1960
  tıpkıbasım girişi).
* **Tarama Sözlüğü — 1300.** 13.-19. yy Osmanlıca metinlerin taraması;
  derlemenin en erken metinleri 13. yy sonu, yani üst sınır değil ALT uç.
  Bu bir eser değil derleme olduğu için yalnız "en geç 13. yy sonu" kabulüyle
  tutuluyor (eski değer korunuyor).
* **Lehce-i Osmânî — 1876** (Ahmed Vefik Paşa, 1. baskı 1876).
* **Kâmûs-ı Türkî — 1901** (Şemseddin Sâmi, 1899-1901; ikinci cilt 1901).
* **Eski Uygurca dönem — [800, 1350]** (Wilkens 2021 tanıkları; dönem
  "9.-14. yy", sözlük tanık yeri vermiyor). NOKTA YIL DEĞİL: 1350 yalnız üst
  sınırdır, nokta tarih her zaman kazanır. Bkz.
  ``fetchers/wilkens_old_uyghur.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class DatedWork:
    key: str
    label: str
    year: int
    #: Serbest metinde (kaynak notu, dil adı, EtimolojiTürkçe "[ … ]") eser adı.
    pattern: re.Pattern[str]
    #: Starling turcet tanıklarındaki kaynak etiketleri ("mončuq (Orkh., OUygh.)").
    starling_tags: tuple[str, ...] = ()
    #: ``point``: eserin tarihi (nokta yıl). ``period``: dönem etiketi; ``year``
    #: yalnız ÜST SINIRDIR, ``range`` aralığı taşır (Wilkens Eski Uygurca:
    #: sözlük tanık yeri vermiyor). Dönem tanığını doğrulayıcı nokta yıl
    #: saymaz (bkz. ``HistoricalAttestationVerifier``).
    precision: str = "point"
    range: tuple[int, int] | None = None


WORKS: tuple[DatedWork, ...] = (
    DatedWork("orhun", "Orhun Yazıtları", 732,
              re.compile(r"orhun|orkhon|köktürk|kül\s*tigin|köl\s*tigin|bilge\s*kağan", re.I), ("Orkh.",)),
    DatedWork("kb", "Kutadgu Bilig (Yusuf Has Hacib)", 1069,
              re.compile(r"kutadgu\s*bilig", re.I), ("KB",)),
    DatedWork("dlt", "Divânu Lugâti't-Türk (Kâşgarlı Mahmud)", 1074,
              re.compile(r"d[iî]v[aâ]n.?[uıi]?[\s-]*lu[gğ][aâ]t|k[aâ]şgarl|\bdlt\b", re.I), ("MK",)),
    DatedWork("ah", "Atebetü'l-Hakayık (Edib Ahmed)", 1300,
              re.compile(r"atebet.?[üu]l.?hakay[ıi]k|atabet", re.I), ("AH",)),
    DatedWork("im", "İbn Mühennâ Lügati", 1245,
              re.compile(r"[iİı]bn[\s-]*m[üu]henn", re.I), ("IM",)),
    DatedWork("cc", "Codex Cumanicus", 1303,
              re.compile(r"codex\s*cumanicus", re.I), ()),
    # Yalnız Wilkens tanığı: dil adı "Eski Uygurca" tek başına bir ESER
    # değildir (indeksin oui kayıtları tarihsiz Vikisözlük maddeleri).
    DatedWork("oui", "Eski Uygurca (9.-14. yy), Wilkens 2021", 1350,
              re.compile(r"wilkens|handwörterbuch\s*des\s*altuigur", re.I), (),
              precision="period", range=(800, 1350)),
    DatedWork("abush", "Abuşka", 1500, re.compile(r"abu[şs]ka", re.I), ("Abush.",)),
    DatedWork("sangl", "Senglâh (Mehdî Han)", 1760, re.compile(r"sengl[aâ]h|sanglax", re.I), ("Sangl.",)),
    DatedWork("tarama", "TDK Tarama Sözlüğü (13.-19. yy)", 1300, re.compile(r"tarama\s*sözlü", re.I), ()),
    DatedWork("lehce", "Lehce-i Osmânî", 1876, re.compile(r"lehçe-?i\s*osman|lehce-?i\s*osman", re.I), ()),
    DatedWork("kamus", "Kâmûs-ı Türkî (Şemseddin Sâmi)", 1901,
              re.compile(r"kamus-?ı\s*türk[iî]|şemse?t?d?din\s*sami", re.I), ()),
)

#: Nokta tarihli eserler: serbest metin taramasında (``DATED_SOURCES``) yalnız
#: bunlar yıl verir; dönem etiketi yıl değil üst sınırdır.
POINT_WORKS: tuple[DatedWork, ...] = tuple(w for w in WORKS if w.precision == "point")

#: Starling kaynak etiketi → yıl (``StarlingEtymology.earliest_dated_source``).
STARLING_SOURCE_DATES: dict[str, int] = {tag: w.year for w in WORKS for tag in w.starling_tags}

_BY_KEY = {w.key: w for w in WORKS}


def work(key: str) -> DatedWork:
    return _BY_KEY[key]


def works_in(text: str) -> list[DatedWork]:
    """Serbest metinde adı geçen tarihli eserler (harita sırasıyla)."""
    if not text:
        return []
    return [w for w in WORKS if w.pattern.search(text)]


def canonical_year(text: str, fallback: int | None = None) -> int | None:
    """Metinde adı geçen eserin haritadaki yılı; eser tanınmazsa ``fallback``.

    Kaynağın kendi yazdığı yıl ("Divan-i Lugat-it Türk (1070)") eser
    tanınıyorsa KULLANILMAZ: aynı eserin tek yılı vardır. Birden çok eser
    geçiyorsa en erkeni.
    """
    found = [w for w in works_in(text) if w.precision == "point"]
    return min(w.year for w in found) if found else fallback
