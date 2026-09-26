# 9n — şans düzeyindeki verici etiketinde dürüst gösterim (ön kayıt)

Tarih 2026-09-26. Bu dosya YENİ rapor bölümü açılmadan commit edilir. Üretim tabanı (4.3.1):
`ARABIC_VIA_RULE = "d1"`, `FRENCH_RULE = "g2"`, `WESTERN_RULE = "h1"`, `SENSE_BRIDGE = True`,
`DONOR_HONEST = "off"`. Yalnız ETİKETİN GÖSTERİMİ ve Türkçe dil etiketi değişir; `attribute_donor`un
seçtiği dil ve biçim, alıntı GÜCÜ (`nearest_donor`, `proximity_strength`), `donors.db`, `index.db` değişmez.

Soru: 9m tanısı Türkçe verici etiketinin önemli kısmının doğru etimona değil şans eşleşmesine
dayandığını gösterdi (sultan ~ صلى, kudret ~ قطر "Katar", imza ~ عيسى). Kullanıcıya yanlış gerekçeli
bir verici BİÇİMİ gösteriliyor, dil tesadüfen doğru olsa bile. Şans düzeyindeki eşleşme etiketin
iç özellikleriyle ayırt edilebilir mi; ayrılınca biçim gizlenip dil daha dürüst verilebilir mi?

## 1. Tanı (yalnız ayar: Türkçe TDK+Nişanyan train+dev, 9j TDK ayar, 9j TETTL ayar; `diag.py` -> `diag_rows.json`, `diag_summary.py`)

Kör indeks, ağ kapalı, üretim etiketi. **Etimon referansı** (yalnız puanlama): altının etimon çevriyazısı
(TDK/TETTL `etymon`; Türkçe altında `tdk_source`/`nisanyan_source`'un dil adından sonrası) + TAM
indeksteki Türkçe maddenin Wiktionary şablon argümanı (`donor_form`, ör. عَسْكَر) ve etimoloji metnindeki
dil adı + biçim çiftleri. **Doğru etimon** = gösterilen biçim referansla aynı sözcük (Arap yazısında
harekesiz iskelet eşitliği, `ال` farkı serbest; Latin/Yunan yazısında aksansız eşitlik; çevriyazıda
karşılaştırma biçimi eşitliği ya da ≥ 4 harfte normalleştirilmiş düzenleme uzaklığı ≤ 0,20) VE dil sınıfı
altınla aynı. Yalnız ünsüz iskeleti tutan (asker ~ عسكري "askerî" gibi türev/şans) doğru sayılmaz.
El denetimi: 35 "şans" + 25 "etimon" örneğinde sınıflama doğru; kalan hatalar türev/kök akrabası
(meşveret ~ مشاورة) ve kısa iskelet çakışması (site ~ CEDEAO, "iskelet" sınıfı; doğru sayılmadı).

| altın | n | doğru etimon | şans (dil doğru, biçim yanlış) | yanlış dil | etiket yok | dil doğru, etimon bilinmiyor |
|---|---|---|---|---|---|---|
| Türkçe train+dev | 293 | **138 (0,471)** | **50 (0,171)** | 104 (0,355) | 1 | 0 |
| 9j TDK ayar | 439 | 75 (0,171) | 44 (0,100) | 192 (0,437) | 125 | 3 |
| 9j TETTL ayar | 286 | 57 (0,199) | 26 (0,091) | 128 (0,448) | 63 | 12 |
| toplam | 1.018 | 270 | 120 | 424 | 189 | 15 |

Etiketli + referanslı 739 maddede gösterilen biçimin doğru etimon olma oranı (**biçim kesinliği**)
0,365 (Türkçe train+dev 0,473; TDK ayar 0,274; TETTL ayar 0,330). Dil doğru olan etiketlerin
120/390'ında (%31) biçim şans eşleşmesi. Şans eşleşmelerinin 92'si paylaşılan havuzdan doğrudan,
16'sı D1 (Farsça üzerinden), 9'u G2'nin ayrı Fransızca havuzundan; 34'ünde anlam yalnız işlev ya da
genel sözcükle eşleşmiş (ortak içerik sözcüğü 0); yalnız 19'unda doğru etimon havuzda vardı (seçilmedi).

**Ayıran özellikler** (739 madde, biçim referansla eşleşiyor mu — dil koşulsuz — AUC; `analyze.py`): mesafe 0,846 · mesafe−null marjı 0,837 ·
ünsüz iskeleti eşi 0,754 · şans yüzdeliği (12 Türkçe kontrol biçiminin aynı havuza en yakın
mesafelerinden bu eşleşmeninkine eşit/küçük olanların payı) 0,753 · ortak içerik sözcüğü 0,669 ·
ikinci dile fark 0,591 · havuz büyüklüğü 0,589. Mesafe dilimlerinde doğru etimon oranı:
≤ 0,10 **0,80** · 0,10–0,15 0,62 · 0,15–0,20 0,54 · 0,20–0,25 0,42 · 0,25–0,30 0,23 · 0,30–0,35 0,18 ·
0,35–0,45 0,09 · > 0,45 0,03. Ortak içerik sözcüğü olmayan 118 maddenin 5'i doğru.

**Kural (seçildi, ayarda; `donor_proximity.attribution_certain`)**: etiket KESİN ⇔ anlamda ortak içerik
sözcüğü ≥ 1 VE [(mesafe ≤ 0,25 VE şans yüzdeliği = 0) YA DA (ünsüz iskeleti eşi VE mesafe ≤ 0,35)].
Taramada (T ∈ {0,15 … 0,30} × içerik × şans × iskelet) kesinlik 0,64–0,81; bu kural (dil koşulsuz biçim eşi) kesinlik 0,777
(394 kesin, 306 eşleşen; gizlenen eşleşen 35) ile kapsam/kesinlik dengesinde seçildi.

| ayar | kesin n | doğru etimon / şans / yanlış dil | belirsiz n | doğru etimon / şans / yanlış dil |
|---|---|---|---|---|
| Türkçe train+dev | 170 | 125 / 12 / 33 | 122 | 13 / 38 / 71 |
| 9j TDK ayar | 133 | 66 / 7 / 60 | 141 | 9 / 37 / 95 |
| 9j TETTL ayar | 91 | 50 / 9 / 32 | 82 | 7 / 17 / 58 |

Kesin dilimde kalan yanlışların çoğu **şans değil koşut alıntı**: aynı uluslararası sözcüğün başka
havuzdaki biçimi (benzin ~ بنزين, taksi ~ تكسي, dedektif ~ դետեկտիվ). Belirsiz dilimde etiketin dil
doğruluğu 132/425 (%31); hep "Arapça" demek bile doğal ağırlıkla daha iyi (Türkçe train+dev 0,602 -> 0,641).

## 2. Adaylar (3; kod bu commit'te, varsayılan KAPALI: `DONOR_HONEST = "off"`)

Hepsinde kesin olmayan etikette verici BİÇİMİ gösterilmez (CLI/web açıklaması "verici belirsiz — yakın
biçim bulunamadı (en yakın aday şans düzeyinde)"; yakınlık sinyalinin kendi açıklamasında da biçim
yok). Kesin etiket 4.3.1 ile aynı. Yalnız Türkçe verici kümesinde (`TURKISH_DONORS`; `tr`, `ota`):
önsel Türkçenin; Saha/xturkic `attribute_donor`un dil kodunu okur, yapı gereği etkilenmez.

- **A1** `a1` — dil yalnız aile düzeyinde: ar/fa -> "Arapça ya da Farsça", fr/it -> "Batı dili
  (Fransızca ya da İtalyanca)", el/hy -> "Yunanca ya da Ermenice" (en yakın adayın dilinin ailesi).
- **A2** `a2` — dil = argmax önsel · P(biçim | dil)^τ, aday diller ar/fa/fr. Önsel: TDK GTS `lisan`
  doğal dağılımı (`natural.json` `tum`: ar 6.638, fr 5.676, fa 1.432). Biçim ipucu: dil başına karakter
  3-gram modeli (`engine/nlp/donor_prior.py`, `data/models/donor_cue_tr.json`), Türkçe Vikisözlük
  `alıntı` maddelerinin `donor_lang`ından; **bütün verici altınlarının kelimeleri eğitimden çıkarıldı**
  (TDK+Nişanyan tümü dahil test, 9e, 9f, 9g, 9j, 9l, 9m ve bu 9n altını; 9n altınının 600 kelimesinin
  600'ü dışlamada). Eğitim: ar 2.238, fr 1.567, fa 412 (it 6, el 21, hy 14 — önceki altınlarda tükendi;
  bu yüzden aday diller ar/fa/fr). τ ayarda {0; 0,25; 0,5; 1} içinden: 1 (tam Bayes).
- **A3** `a3` — A2'nin sonsalıyla AİLE (ar+fa kütlesi vs fr), aile düzeyinde gösterilir.

A1/A2/A3 aynı kesinlik kuralını kullandığı için **(i) üçünde özdeş**; ayrıştıkları yer dil etiketidir.

## 3. Ölçütler

- **(i) biçim kesinliği (BİRİNCİL)** = gösterilen biçimlerden doğru etimon olanların payı (etimon
  referansı olan maddeler). Test: üretimde biçim gösterilen maddelerde (aday tarafından korunan /
  gizlenen) × (doğru etimon / değil) 2×2, kesin Fisher, iki yönlü; Holm (3 aday).
- **(ii) doğal ağırlıklı dil doğruluğu** `acc_nat` (9m tanımı, `natural.json` `tum`). **Aile etiketi**
  altın sınıf ailedeyse **1/2 puan** (iki üyeli aile = rastgele seçimin beklenen puanı), değilse 0.
  Test: madde ağırlıklı işaret çevirme (9m `signflip`, ≤ 20 uyuşmazlıkta kesin, yoksa 200.000 örnek,
  tohum 9), iki yönlü, Holm. Ayrıca ham doğruluk ve "aile doğruluğu" (ailede ya da tam isabet) bilgi.
- **(iii) kapsama** = kesin (biçimli) etiket verilen madde payı; raporlanır (kayıp beklenir).

## 4. Ayar sonuçları (`ayar.log`, `res_tr.json`, `res_tdk_ayar.json`, `res_tettl_ayar.json`)

| altın | koşul | acc_nat | ham | aile | biçim kesinliği | gösterilen (yanlış) | kapsama | (ii) işaret D (daha iyi/kötü), p |
|---|---|---|---|---|---|---|---|---|
| Türkçe train+dev | prod | 0,6020 | 0,6416 | 0,6416 | 0,473 | 292 (154) | 0,997 | — |
| | A1 | 0,5521 | 0,5836 | 0,6997 | 0,735 | 170 (45) | 0,580 | −0,050 (17/51), 0,0001 |
| | **A2** | **0,7257** | 0,7816 | 0,7816 | 0,735 | 170 (45) | 0,580 | **+0,124 (51/10), 5e-6** |
| | A3 | 0,5989 | 0,6416 | 0,8157 | 0,735 | 170 (45) | 0,580 | −0,003 (58/51), 0,86 |
| 9j TDK ayar | prod | 0,4519 | 0,2779 | — | 0,274 | 274 (199) | 0,715 | — |
| | A1 | 0,3981 | 0,2631 | 0,3576 | 0,496 | 133 (67) | 0,312 | −0,054, 0,003 |
| | **A2** | **0,5729** | 0,2916 | — | 0,496 | 133 (67) | 0,312 | **+0,121 (29/23), 0,0001** |
| | A3 | 0,4453 | 0,2654 | 0,3622 | 0,496 | 133 (67) | 0,312 | −0,007, 0,78 |
| 9j TETTL ayar | prod | 0,5073 | 0,3322 | — | 0,330 | 173 (116) | 0,780 | — |
| | A1 | 0,4525 | 0,3059 | 0,3951 | 0,550 | 91 (41) | 0,339 | −0,055, 0,001 |
| | **A2** | **0,6183** | 0,3706 | — | 0,550 | 91 (41) | 0,339 | **+0,111 (24/13), 6e-5** |
| | A3 | 0,5069 | 0,3287 | 0,4406 | 0,550 | 91 (41) | 0,339 | −0,000, 0,99 |

(i) Fisher (korunan/gizlenen doğru etimon): Türkçe 125/45 vs 13/109, TDK 66/67 vs 9/132, TETTL 50/41
vs 7/75 — üçünde p < 1e-15. τ taraması (acc_nat, Türkçe / TDK / TETTL): τ=0 (yalnız önsel) 0,641 /
0,504 / 0,557 · 0,25: 0,712 / 0,565 / 0,605 · 0,5: 0,717 / 0,571 / 0,612 · **1: 0,726 / 0,573 / 0,618**.

Korumalar (ölçüldü, `guard.log`): Saha `eval-donor` motor **0,7136** her `DONOR_HONEST` değerinde
(yakınlık 0,6659; verici kümesi ru/mn/evn, dürüst etiket uygulanmaz); xturkic ayar verici **0,7459**
(ayar maddelerinde Türkçe verici kümesi 0 madde; `attribute_donor` dili değişmez).

⚠️ Önceden görülen: **A1 Türkçe train+dev korumasını düşürdü** (0,552 < 0,592) ve üç ayarda da (ii)'yi
anlamlı düşürdü — rapor ne derse desin kabul edilemez; yine raporlanır. A3 ayarda (ii)'de eşit.

## 5. Yeni rapor altını (`build_gold.py` -> `gold.json`, tuz `donor9n-tdk-v1`, n = 600)

9m kuralı (TDK GTS v12 `lisan`, tek sözcük, tek dil, kör indekste anlam, grup başına tek madde);
**dışarıda**: TDK+Nişanyan (items + disagreements), 9e, 9f, 9g, 9j-TDK (ayar+rapor), 9j-TETTL
(ayar+rapor), 9l, **9m** altınlarının kelimeleri ve (verici, etimon) grupları (önceki altınlarla kelime
kesişimi 0, denetlendi). Doğal oranlar (`tum`, kalan sınıflar): ar 290 / fr 248 / fa 62 (İngilizce
anlamlı 165 / 93 / 38). Mevcut ar 3.940, fr 3.713, fa 387. Etimonu boş 1 madde.
**Mühür**: `gold.json` sha256 `7a8e3a7bd0a78b8a8662c8975dca1fc2d215d55ef7aae761559075eed0515482`.

## Sızıntı (K2, `k2.json`, n = 600; doğruluk hesaplanmadı) — GEÇTİ
Ağ kapalı, kör indeks; getirici yüklenmedi; kör indekste köken sütunları 0 dolu; anlam kör = tam
600/600; etiket ve dört koşulun dürüst etiketi kör vs tam anlamla 600/600 aynı; biçim ipucu modeli
9n altınını dışlama listesinde taşıyor ve 600 kelimenin 600'ü dışlanmış.

## Rapor (bir kez, bu commit'ten sonra)
`python data/cache/work/donor9n/harness.py gold 9n` (`run_rapor.sh`).

Kabul (aday için hepsi):
1. (i) rapor biçim kesinliği prod'dan yüksek VE Fisher Holm p < 0,05.
2. (ii) rapor `acc_nat` (aile = 1/2) Δ ≥ **−0,01** VE anlamlı düşüş yok (işaret çevirme Holm p < 0,05 ve
   D < 0 ise red).
3. Türkçe train+dev `acc_nat` ≥ prod − 0,01 = 0,592 — ölçüldü (A1 0,552 ✗, A2 0,726, A3 0,599).
4. Saha `eval-donor` motor ≥ 0,7136, xturkic ayar ≥ 0,7459 — ölçüldü (aynı). "Belirsiz" bu iki ölçümde
   yoktur (Türkçe verici kümesi dışı); `tr_donor_eval`de aile etiketi 1/2 puan.
5. `eval-borrowing` F'ler aynı — yapı gereği (güç yolu değişmez); kabul edilirse `make eval-borrowing`.

Karar: kabul edilenlerden rapor `acc_nat`ı en yüksek olan `DONOR_HONEST` varsayılanı olur; hiçbiri ->
üretim değişmez (bayrak kapalı, kod bayrak arkasında). Kabul varsa `tr_donor_eval`e kalıcı
"biçim kesinliği" ve "kapsama" ölçütleri, CLI/web gösterimi, README verici satırı, ENGINE_VERSION 4.3.2.
