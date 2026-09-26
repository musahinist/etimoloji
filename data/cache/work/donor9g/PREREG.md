# 9g — Rumca/Yunanca ve Ermenice verici etiketi (ön kayıt)

Tarih 2026-09-26. Bu dosya YENİ rapor bölümü açılmadan commit edilir. Üretim
tabanı: `ARABIC_VIA_RULE = "d1"`, `FRENCH_RULE = "g2"`, `WESTERN_RULE = "h1"`
(79290de). Yalnız ETİKET adımı (`donor_proximity.attribute_donor`); alıntı GÜCÜ
yolu ve `donors.db` değişmez.

## Yeni değerlendirme verisi (`build_gold.py` -> `gold.json`, tuz `donor9g-v1`)
Kaynak: kaikki en dökümleri **tr, ota, az, crh**. Madde: ilk verici şablonu
(9e `donor_of` birebir: bor/bor+/lbor/der/…, önünde türetme şablonu yok) Yunanca
ailesi (el, grc, gkm, pnt) ya da Ermenice ailesi (hy, xcl, axm); aynı (dil,
kelime) kayıtları vericide anlaşıyor; özel ad/ek/deyim değil; karşılaştırma
biçimi ≥ 3 harf; TDK+Nişanyan altınında (items + disagreements) YOK; 9e ve 9f
altınlarındaki kelimeler ve (aile, etimon) grupları YOK (142 düştü); kör
indekste (xtr/index_blind.db) o dilde anlamı var; aynı (aile, etimon, biçim)
birden çok dilde varsa bir kez (tr > ota > az > crh; 75 düştü). Bölme (aile,
etimon) grubuna göre, ayar payı 0,35.

| bölüm | Yunanca | Ermenice | toplam |
|---|---|---|---|
| ayar | 80 (tavan; 102 mevcut) | 36 (tümü) | 116 |
| **rapor** | **150** (tavan; 182 mevcut) | **78** (tümü) | **228** |

Dil dağılımı (rapor): el tr 87 / ota 45 / az 11 / crh 7; hy tr 57 / ota 10 / az 11.
Verici kodu (tümü): el 129, grc 69, gkm 24, pnt 8; hy 105, axm 8, xcl 1.
**Ağız (`dialectal`) etiketli:** rapor hy 41/78, el 3/150; ayar hy 16/36, el 3/80.
Keşif sayımı (süzgeç öncesi, ilk verici şablonu): tr el-ailesi 492 / hy-ailesi 131
(tr'de `dialectal` etiketli: hy 61, el 12); ota 357 / 51; az 41 / 37; crh 28 / 1.
Hedef "rapor el ≥ 60, hy ≥ 60" tuttu (hy 78, gerçek sayı; mevcudun tümü).

Kalite (`audit30.tsv`, el 15 + hy 15, hash sırası, **LLM ön-etiketi — Claude;
insan onayı yok**): 27 doğru, 3 şüpheli (Ege: yer adı niteliğinde; daşd: Ermenice
tašt'ın kendisi Farsçadan, Farsça doğrudan olası; hamol: etimon Latin harfli,
kaynak belirsiz), 0 açıkça yanlış.

Sızıntı (K2, `k2.json`, n=344, ayar+rapor; doğruluk hesaplanmadı): kör indekste
köken sütunları 0 dolu; anlam kör = tam 344/344; etiket (off ve g3) kör vs tam
344/344 aynı. Anlamda dil/kaynak adı: 15 (7'si gerçekten Yunan/Ermeni anıyor:
drahmi "Greek currency", palikarya "Greek man", dığa "an Armenian boy", ota
"Barekendan… Armenian Carnival", "Greek orphan", "late Byzantine title", pilaki
"Greek etymon"; kalanı "from"/"borrowed" dışı rastlantı) — anlamın kendisi,
etimoloji sütunu değil; motor anlam metnindeki dil adını kullanmıyor. GEÇTİ.

Sınıf eşlemesi: tahmin el/grc/gkm -> Yunanca, hy/xcl -> Ermenice, diğerleri
`tr_donor_eval.engine_class`. Verici kümesi `TURKISH_DONORS` (ar fa el hy fr it)
her dil için. Sorgu biçimi kör indeksin `comparison` sütunu, anlam kör indeksin
ilk anlamı.

## Havuzlar (yalnız etiket; `data/lexicons/donors_label/`, git-ignored)
`scripts/download_lexicons.py --label-donors` (künye commit edilir) ->
`python -m engine.db.donor_index --build-label` -> `donors_label.db`:
**grc 67.809, xcl 8.472** madde (Eski Yunanca ham 382,5 MB / 68.196 kayıt;
Eski Ermenice 45,1 MB / 8.541). Orta Yunanca (gkm) kaikki'de YOK (dizin
listesi: Ancient Greek, Greek, Mariupol Greek, Mycenaean Greek; Old/Middle
Armenian). Çekim/biçim göndermesi (`is_form_of`): grc 35.618, xcl 2.307; güç
havuzunda el 49.062 / 84.722, hy 1.989 / 21.748.

## Adaylar (3; kod bu commit'te sabit, varsayılan KAPALI)
- **G1** `OLD_DONOR_LABELS = True` (mod `separate`): grc ve xcl AYRI grup,
  kendi null'ı; seçilirse etiket aile kodu (el/hy), kaynak `kaikki-grc/xcl`
  ("Eski Yunanca biçimi").
- **G2** `LABEL_FORM_FILTER = True`: el, hy (ve grc, xcl) maddelerinden anlamı
  YALNIZ çekim/biçim göndermesi olanlar `by_sense`te atlanır, yerlerine sıradaki
  maddeler girer (X4 (b) kuralı; `CLEAN` bayrağı ve güç yolu değişmez).
- **G3** = G1 + G2.

### Ayar ölçümü (9g ayar n=116; tabana karşı) ve korumalar (rapordan ÖNCE ölçüldü)
| koşul | 9g ayar | McNemar (aday/taban) | TR train+dev n=293 | xturkic ayar | Saha motor | 9f rapor (bilgi) |
|---|---|---|---|---|---|---|
| off (üretim) | **0,2414** (Yun 19/80, Erm 9/36) | — | 0,6416 | 0,7459 | 0,7136 | 0,4375 |
| G1 | 0,3276 (26/80, 12/36) | 12/2, p=0,013 | **0,6143** (4/12) | 0,7459 | 0,7136 | 0,4292 |
| G2 | 0,2328 (18/80, 9/36) | 0/1 | 0,6416 (0/0) | 0,7459 | 0,7136 | 0,4333 |
| G3 | 0,3103 (24/80, 12/36) | 11/3, p=0,057 | **0,6075** (3/13, p=0,021) | 0,7459 | 0,7136 | 0,4250 |

Üretimin ayar karışıklık matrisi (off, `gold_ayar.json`): Yunanca -> Fransızca 23,
Yunanca 19, Arapça 16, İtalyanca 9, Ermenice 8, Farsça 5; Ermenice -> Arapça 12,
Ermenice 9, Fransızca 8, Farsça 3, Yunanca 3, İtalyanca 1. Hatalı etiketlerin
çoğu anlam kısıtlı havuzdaki rastlantı adayı (enerji -> fr énergie gibi Batı
aracılığı gerçek; ama yakamoz -> ar tammuz, pancar -> ar yanayir rastlantı).

⚠️ **Önceden görülen sonuç:** G1 ve G3 Türkçe koruması (≥ 0,642) ayarda
DÜŞTÜ (Arapça -> Yunanca 6 -> 17/19: Eski Yunanca havuzu anlam kısıtlı ikinci
bir "bilet" ve Arapça alıntılara rastlantı eşi veriyor). Tanı (bilgi; aday
değil): `merge` modu (eski dil ailenin grubuna, null birleşik) ayar 0,2845 /
TR 0,6212 — o da düşüyor. G2 ayarda etkisiz: çekim göndermesi anlamları
sorgu anlamıyla nadiren örtüşüyor (FTS eşleşmesi "plural/form" sözcüklerine
dayanmıyor). Rapor yine bir kez açılır: üretimin el/hy doğruluğunun bağımsız
tahmini ve adayların yönü için.

## Rapor (bir kez, bu commit'ten sonra)
`python data/cache/work/donor9g/harness.py gold rapor` (n=228). Birincil:
el+hy etiket doğruluğu; G1, G2, G3 her biri `off`a karşı iki yönlü kesin
McNemar, Holm (3 aday), α = 0,05.

Kabul (aday için hepsi):
1. Rapor artış yönünde ve Holm p < 0,05;
2. Türkçe tr_donor train+dev ≥ 0,6416 — ölçüldü (G1 0,614, G2 0,642, G3 0,608);
3. Saha eval-donor motor ≥ 0,7136 — ölçüldü (değişmez);
4. xturkic ayar verici tanıma ≥ 0,7459 — ölçüldü (değişmez);
5. eval-borrowing F'ler aynı — yapı gereği (etiket yalnız sinyal ateşlenince
   hesaplanır); kabul edilirse doğrulanır (çıktı scratch'e).

Bilgi (karar ölçütü değil): 9f rapor bölümü (yukarıda); ağız etiketli alt küme.

Karar: kabul koşullarını geçenlerden rapor doğruluğu en yüksek olan (eşitlikte
basit olan: G2 < G1 < G3). Hiçbiri -> üretim değişmez (`OLD_DONOR_LABELS =
LABEL_FORM_FILTER = False`); havuz ve kod bayrak arkasında kalır.

Mühür: `gold.json` sha256 `61cf43e54270cfd414f4792410bd67f77c299e9d411bfbd2dc93f28f7eddbc0e`.
