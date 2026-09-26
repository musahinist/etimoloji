# 9l — Verici etiketinde ANLAM ARAMASI: Türkçe anlam köprüsü / ayrı dil havuzu / tam eşleşme (ön kayıt)

Tarih 2026-09-26. Bu dosya YENİ rapor bölümü açılmadan commit edilir. Üretim tabanı:
`ARABIC_VIA_RULE = "d1"`, `FRENCH_RULE = "g2"`, `WESTERN_RULE = "h1"`; 9j/9g bayrakları kapalı.
Yalnız ETİKET adımı (`donor_proximity.attribute_donor`); alıntı GÜCÜ yolu (`nearest_donor`,
`proximity_strength`), `donors.db`, `index.db`, ENGINE_VERSION değişmez.
Gerekçe: 9j SONUÇ — TDK rapor Türkçe anlamlı 493 maddede doğruluk 0,075 (337'si etiketsiz),
İngilizce anlamlı 217'de 0,52; İngilizce anlamlıda doğru İtalyanca etimon paylaşılan sıralamasız
`LIMIT 200`'e girmiyor (9f 50/120) ya da 0,0 mesafeli İtalyanca biçim `mesafe − null`da yeniliyor (41/120).

## Adaylar (3; kod bu commit'te, varsayılan KAPALI)
- **S1** `SENSE_BRIDGE = True` — anlam metni İngilizce değilse (`sense_bridge.is_english_sense`:
  İng./Tr. durak sözcükleri, Türkçe harf, yoksa İngilizce başlık sözlüğü çoğunluğu; kör indekste
  en katmanı %99,1 İngilizce, tr katmanı %0,44 yanlış İngilizce) ve Latin yazılıysa, verici anlam
  araması kelimenin **Vikisözlük çeviri karşılıklarıyla** yapılır (en çok 4, sık olan önce):
  trwiktionary Türkçe maddelerin çeviri bölümü (tr->en), trwiktionary İngilizce maddelerin tek
  sözcüklük Türkçe tanımı (ters), enwiktionary İngilizce maddelerin çeviri tabloları (ters); yedek:
  aynı karşılaştırma biçimli Osmanlıca (`ota`, en-Wiktionary) maddenin İngilizce anlamı. Anahtar
  kelimenin karşılaştırma biçimi. **Kullanılmayan**: TDK tanımı (Türkçe anlam metni köprülenmez),
  etimoloji alanları, kategoriler ("… kökenli"), İngilizce dışı çeviriler (İtalyanca/Fransızca
  çeviri çoğu zaman etimonun kendisi). Tablo `data/lexicons/sense_bridge/tr_en.db` (git-ignored;
  29.346 Türkçe biçim + 884.644 İngilizce başlık; künye `tr_en.provenance.json`, `engine/db/sense_bridge.py`).
  Kör/tam indeksten bağımsız; ağ gerektirmez.
- **S2** `EXTRA_POOL_LANGS = ("it",)` — G2'nin Fransızca için yaptığı ayrı 200'lük anlam sorgusu
  İtalyanca için de (paylaşılan havuz aynen, yalnız eklenir).
- **S3** `EXACT_MATCH_EPS = 0.05` — en yakın biçimi SCA ≤ 0,05 olan diller varsa seçim yalnız
  onların arasında ham mesafeyle (tam eşleşme null düzeltmesine yenilmez); D1/G2/H1 sonrası kuralları aynen.

## Ayar (yalnız 9j TDK ayar bölümü, n=439; `res_tdk_ayar.json`, `ayar.log`, `ayar2.log`)
| koşul | TDK ayar | McN (aday/off) | en n=157 | tr n=282 | etiketsiz | TR train+dev n=293 | xturkic | Saha |
|---|---|---|---|---|---|---|---|---|
| off | **0,2096** (it 13/116) | — | 72 | 20 | 183 | 0,6416 | 0,7459 | 0,7136 |
| **S1** | **0,2779** (it 23) | **30 / 0** | 72 | 50 | 125 | 0,6416 (0/0) | 0,7459 | 0,7136 |
| S2 `it` | 0,2005 (it 15) | 3 / 7 | 67 | 21 | 183 | **0,6041** (1/12) ✗ | 0,7459 | 0,7136 |
| S2b `it,el,hy` (seçilmedi) | 0,2005 | 6 / 10 | 67 | 21 | 183 | 0,5700 ✗ | 0,7459 | 0,7136 |
| S3 ε=0 (seçilmedi) | 0,2118 | 1 / 0 | 73 | 20 | 183 | 0,6416 | 0,7659 | 0,7136 |
| **S3** ε=0,05 | 0,2141 | 2 / 0 | 74 | 20 | 183 | 0,6451 (1/0) | 0,7677 | 0,7136 |
| S1+S3 (birleşik) | 0,2847 (it 25) | 34 / 1 | 74 | 51 | 125 | 0,6451 (1/0) | 0,7677 | 0,7136 |

Seçimler (ayarda, rapordan önce): S2 varyantı `it` (s2b ile aynı doğruluk, daha az uyuşmazlık ve
TR korumasında daha az düşüş); S3 ε=0,05 (ayarda 2/0 ≥ 1/0). S1'in tek sürümü var (en çok 4 karşılık).
⚠️ Önceden görülen: **S2 Türkçe TDK+Nişanyan korumasını ayarda DÜŞÜRDÜ** (0,6041 < 0,6416;
Fransızca->İtalyanca 11->19, Arapça->İtalyanca 5->16) — rapor sonucu ne olursa olsun kabul edilemez;
yine rapor edilir (yön). S1 Türkçe train+dev'de hiç uyuşmazlık üretmiyor (o maddelerin anlamı İngilizce).
S3 xturkic ayarı 0,746 -> 0,768 yükseltiyor.

## Yeni rapor altını (`build_gold.py` -> `gold.json`, tuz `donor9l-tdk-v1`, n = 441)
TDK GTS v12 `lisan` (künye 9j PREREG; ham metin repoda yok), 9j TDK madde kuralıyla aynı (tek sözcük,
özel ad değil, tek dil, kayıtlar tutarlı, kör indekste anlamı var, biçim ≥ 3 harf); **dışarıda**:
TDK+Nişanyan (items+disagreements), 9e, 9f, 9g, 9j-TDK (ayar+rapor), 9j-TETTL (ayar+rapor) altınlarının
kelimeleri ve (verici, etimon) grupları; grup başına tek madde. Tek bölüm `rapor`.

| it | el | hy | fr | ar | fa | toplam | İngilizce anlamlı |
|---|---|---|---|---|---|---|---|
| **19 (tümü)** | 12 (tümü) | 0 (yok) | 150 | 130 | 130 | **441** | 212 (it 4) |

Mevcut: ar 4.363, fr 4.111, fa 579, it 19, el 12. Düşenler: önceki altında kelime 3.118, etimon 56,
anlam yok 602, aynı grup 111. ⚠️ İtalyanca 19'la sınırlı (TDK'nın `lisan: İtalyanca` maddelerinin
neredeyse tümü 9j TDK/TETTL'de kullanıldı); birincil ölçüt tüm sınıflarda.
**Mühür**: `gold.json` sha256 `68f26b6d05b9fc19128f982cf761b7bde7355650eb968e7deacc4d0648f57cdb`.

## Sızıntı (K2, `k2.json`, n = 441; doğruluk hesaplanmadı) — GEÇTİ
Ağ kapalı (`socket.connect` engelli), kör indeks; koşu sonunda yüklü getirici **yok**; kör indekste
köken sütunları 0 dolu; anlam kör = tam 441/441; etiket (off, s1, s2, s3b, s13) kör vs tam 441/441
aynı. Havuz dilleri yalnız kaikki. S1 köprüsü 67 maddede devreye giriyor; köprü anlamında dil adı
yok (6 madde yalnız "from" içeriyor: "resin from extinct … trees", "air from the outside" …).

## Rapor (bir kez, bu commit'ten sonra)
`NINEL_CONDS=off,s1,s2,s3b,s13 python data/cache/work/donor9l/harness.py gold 9l rapor`.
Birincil: TÜM sınıflarda etiket doğruluğu; S1, S2, S3 her biri `off`a karşı iki yönlü kesin McNemar,
Holm (3 aday), α = 0,05. Bilgi: sınıf başına, anlam dili katmanı, etiketsiz sayısı, S1+S3 birleşik;
sonra 9e/9f/9g/9j raporları (önceden açılmış, bilgi).

Kabul (aday için hepsi):
1. Rapor artış yönünde ve Holm p < 0,05;
2. Türkçe tr_donor train+dev ≥ 0,6416 — ölçüldü (S1 0,6416; S2 0,6041 ✗; S3 0,6451);
3. Saha eval-donor motor ≥ 0,7136 — ölçüldü (üçü aynı);
4. xturkic ayar verici ≥ 0,7459 — ölçüldü (S1/S2 aynı, S3 0,7677);
5. eval-borrowing F'ler aynı — yapı gereği (yalnız etiket); kabul edilirse `make eval-borrowing` ile doğrulanır.

Karar: kabul edilen her aday varsayılan AÇIK. S1 ve S3 ikisi de kabul edilirse birleşik açılır
(birleşiğin 2–4 korumaları ayarda ölçüldü: 0,6451 / 0,7136 / 0,7677; raporu bilgi); yalnız biri
kabul edilirse yalnız o. Hiçbiri -> üretim değişmez (bayraklar kapalı, kod ve tablo bayrak arkasında).
