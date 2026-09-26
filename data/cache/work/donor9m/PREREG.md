# 9m — Fransızca kurallarının (G2, H1) Arapça bedeli; DOĞAL dağılım ağırlıklı doğruluk (ön kayıt)

Tarih 2026-09-26. Bu dosya YENİ rapor bölümü açılmadan commit edilir. Üretim tabanı (4.3.1):
`ARABIC_VIA_RULE = "d1"`, `FRENCH_RULE = "g2"`, `WESTERN_RULE = "h1"`, `SENSE_BRIDGE = True`.
Yalnız ETİKET adımı (`donor_proximity.attribute_donor`); alıntı GÜCÜ (`nearest_donor`,
`proximity_strength`), `donors.db`, `index.db` değişmez.

Soru: G2 (dd630e3) ve H1 (79290de) Türkçe TDK+Nişanyan train+dev etiket doğruluğunu 0,563 -> 0,642
yükseltti ama Arapça duyarlılığı 0,804 -> 0,757 düştü (119 -> 112/148). Kurallar Fransızca ağırlıklı
altınlarda kabul edilmişti; Türkçenin doğal alıntı dağılımında net etki olumsuz olabilir mi?

## 1. Tanı (görülmüş veri: Türkçe train+dev, 9j TDK/TETTL ayar, 9l rapor; `diag_ar.py` -> `diag_ar.json`)

Koşullar: `base` G2 ve H1 kapalı · `g1` yalnız G2'nin ayrı 200'lük Fransızca havuzu · `g2` G2 tam
(havuz + Fransızca aracılı) · `h1` yalnız H1 · `prod` G2+H1. Kör indeks, `harness.py`.

**Kaybedilen Arapça maddelerin TAMAMI G2'nin ayrı Fransızca havuzundan** (Fransızca kazanan doğrudan
havuzdan gelir; Fransızca aracılı `_french_via` 0, H1 soneki 0, D1 ile sıra 0 madde). Hiçbir altında
G2/H1 bir Arapça madde KAZANDIRMIYOR.

| altın | Arapça n | kayıp | kelimeler (taban Arapça etiketi -> üretim Fransızca etiketi) |
|---|---|---|---|
| Türkçe train+dev | 148 | 7 | badire (bahira -> parterre), eşhas (şuhhad -> exos), mazur (mahara -> essayer), menfaat (infarada -> bienfait), sultan (salla -> question), tahvilat (ahirat -> excellent), velinimet (alana -> vouvoiement) |
| 9j TDK ayar | 60 | 4 | askerî (alkahira -> square), imza (isa -> impose), itaat (dat -> étant), kudret (katar -> Madrid) |
| 9j TETTL ayar | 60 | 3 | fetret (fakid -> être), gurbet (karida -> gourmet), tevatür (tavvaka -> épater) |
| 9l rapor (bilgi) | 130 | 3 | bilhassa (basbas -> filasse), defaat (abad -> avant), vükela (vuzara -> voyageurs) |

⚠️ 17 kaybın 17'sinde taban Arapça etiketin gösterdiği etimon YANLIŞ (sultan ~ صلى "dua etti",
kudret ~ قطر "Katar", imza ~ عيسى "İsa", tahvilat ~ عاهرات): doğru etimon anlam havuzunda yok; taban
Arapça'yı yalnız paylaşılan `LIMIT 200` havuzunu Arapça doldurduğu için (sınıf önseli) "doğru"
buluyordu. G2'nin ayrı havuzu bu şans eşleşmesine bir Fransızca şans eşleşmesi ekliyor. Yani kayıp
bir biçim ya da sonek hatası değil, havuz önselinin kaybı. H1'in soneki hiçbir Arapça maddeyi bozmuyor
(Türkçe train+dev'de H1'in dokunduğu Arapça yok).

## 2. Doğal dağılım (`natural_dist.py` -> `natural.json`; TDK GTS v12, ham döküm repoda yok)

| sayım | Arapça | Fransızca | Farsça | İtalyanca | Yunanca (Rumca) | Ermenice | diğer |
|---|---|---|---|---|---|---|---|
| **tum** (`lisan` dolu her başlık, ilk dil; BİRİNCİL) | **0,4176** (6.638) | **0,3571** (5.676) | 0,0901 | 0,0386 | 0,0307 | 0,0018 | 0,0641 |
| altin (tek sözcük, tek dil, özel ad değil; duyarlılık) | 0,3891 (5.418) | 0,3919 (5.457) | 0,0745 | 0,0424 | 0,0332 | 0,0019 | 0,0671 |

⚠️ Önsel beklentinin tersine TDK başlık sayımında Arapça Fransızcadan **çok daha büyük değil**:
%41,8 / %35,7 (tüm başlıklar; fark çoğunlukla "Arapça + Farsça/Türkçe" bileşikleri), tek sözcüklük
yalın alıntılarda eşit (%38,9 / %39,2). (Sayım TÜR sayımıdır; metin sıklığı değil.)

**Doğal ağırlıklı doğruluk** `acc_nat` = Σ_c w_c · duyarlılık_c / Σ_c w_c, w = `tum` payları, yalnız
altında bulunan sınıflar (Ermenice sınıfı olmayan altında Ermenice payı "diğer"e). `natural.py`.

Mevcut kuralların etkisi, TÜM mevcut altınlar (hepsi görülmüş, bilgi; ham / **acc_nat tum** / acc_nat altin):

| altın | n | base | g1 | g2 | h1 | prod | Arapça duy. base -> prod | Fransızca duy. base -> prod |
|---|---|---|---|---|---|---|---|---|
| Türkçe train+dev | 293 | 0,563 / **0,505** / 0,491 | 0,563 / 0,513 | 0,628 / 0,587 | 0,584 / 0,529 | 0,642 / **0,602** / 0,601 | 0,804 -> 0,757 | 0,359 -> 0,685 |
| 9j TDK ayar | 439 | 0,246 / **0,375** / 0,365 | 0,248 / 0,369 | 0,267 / 0,428 | 0,264 / 0,413 | 0,278 / **0,452** / 0,455 | 0,533 -> 0,467 | 0,250 -> 0,538 |
| 9j TETTL ayar | 286 | 0,308 / **0,451** / 0,444 | 0,308 / 0,448 | 0,329 / 0,501 | 0,315 / 0,463 | 0,332 / **0,507** / 0,511 | 0,617 -> 0,567 | 0,383 -> 0,600 |
| 9j TDK rapor | 710 | 0,249 / **0,391** / 0,378 | 0,246 / 0,368 | 0,280 / 0,434 | 0,273 / 0,434 | 0,303 / **0,475** / 0,477 | 0,610 -> 0,520 | 0,213 -> 0,540 |
| 9j TETTL rapor | 573 | 0,241 / **0,376** / 0,367 | 0,258 / 0,396 | 0,290 / 0,454 | 0,257 / 0,399 | 0,300 / **0,469** / 0,473 | 0,570 -> 0,520 | 0,253 -> 0,560 |
| 9l rapor | 441 | 0,360 / **0,399** / 0,389 | 0,365 / 0,401 | 0,406 / 0,447 | 0,386 / 0,427 | 0,417 / **0,459** / 0,457 | 0,569 -> 0,546 | 0,267 -> 0,467 |
| 9e ayar | 290 | 0,469 / **0,545** / 0,530 | 0,486 / 0,559 | 0,552 / 0,634 | 0,517 / 0,599 | 0,586 / **0,673** / 0,675 | 0,788 -> 0,738 | 0,340 -> 0,740 |
| 9e rapor | 560 | 0,463 / **0,545** / 0,530 | 0,484 / 0,562 | 0,548 / 0,632 | 0,512 / 0,599 | 0,588 / **0,674** / 0,675 | 0,819 -> 0,775 | 0,330 -> 0,720 |
| 9f (fr+it) | 240 | 0,263 / **0,333** | 0,313 / 0,423 | 0,400 / 0,581 | 0,304 / 0,408 | 0,442 / **0,656** | — | 0,350 -> 0,708 |
| 9g ayar (el+hy) | 116 | 0,293 / **0,309** | 0,276 / 0,285 | 0,241 / 0,238 | 0,293 / 0,309 | 0,241 / **0,238** | — | — |
| 9g rapor (el+hy) | 228 | 0,320 / **0,298** | 0,307 / 0,279 | 0,281 / 0,241 | 0,320 / 0,298 | 0,281 / **0,241** | — | — |

Sonuç (bilgi): doğal ağırlıkla G2+H1'in net etkisi Arapça/Fransızca içeren her altında **açıkça
olumlu** (+0,06 ile +0,10; Arapça −0,02..−0,09, Fransızca +0,20..+0,40). Tek olumsuz: yalnız
Yunanca/Ermenice altını 9g (0,298 -> 0,241; G2 Yunancayı Fransızcaya çekiyor). G2'nin yalnız havuz
kısmı (g1) TDK raporunda doğal ağırlıkla olumsuz (0,391 -> 0,368); kazancın çoğu Fransızca aracılı ve H1'den.

## 3. Adaylar (3; kod bu commit'te, varsayılan KAPALI)

- **R1** `WESTERN_SUFFIX_DROP = ("ik",)` — H1'den belirsiz sonek çıkarılır. Seçim ölçütü RAPORDAN ve
  TDK'dan bağımsız: tam indeksin Türkçe Vikisözlük maddelerinde (`donor_lang` dolu) H1 sonekiyle biten
  maddelerin Fransızca dışı vericiye (İngilizce hariç) payı ≥ %10 olan sonek. `-ik`: fr 188 / ota 76,
  ar 43, trk 37, hy 7 (%45 Fransızca dışı) -> çıkar; `-ör` fr 98 / fa 1 (%1), `-ist` fr 41 / fa 1,
  diğerleri %0-2 -> kalır. (Tanı: H1 hiçbir Arapça maddeyi bozmuyor; R1'in Arapçaya etkisi beklenmez.)
- **R2** `FRENCH_ARABIC_GUARD = True` (+ `FRENCH_ARABIC_GUARD_MIN = 3`, `FRENCH_ARABIC_GUARD_SKIP_WESTERN
  = True`, `FRENCH_ARABIC_GUARD_MAX = None`) — sorgunun ünsüz iskeleti (D2 mantığı, `_query_skeletons`,
  6370422) Arapça havuzdaki bir adayınkiyle aynıysa, iskelet ≥ 3 ünsüzse ve o Arapça aday Arapça dökümde
  kendisi Batı alıntısı değilse (`from French/Italian/English`; `kobalt` ~ كوبالت) Fransızca kuralları
  devreye girmez: G2'nin ayrı havuzundan gelen Fransızca kazanan -> iskelet eşi Arapça aday; Fransızca
  aracılı ve H1 atlanır.
- **R3** R1 + R2.

R2 "güçlü eşleşme" ayarı (yalnız ayar: Türkçe train+dev, 9j TDK/TETTL ayar; `r2var.log`; acc_nat tum):

| varyant | TR train+dev | TDK ayar | TETTL ayar |
|---|---|---|---|
| prod | 0,6020 (ham 0,6416) | 0,4519 | 0,5073 |
| a: iskelet ≥ 3 | 0,5999 (0,6416) | 0,4572 | 0,5084 |
| b: a + Arapça aday SCA ≤ 0,35 | 0,6009 (0,6416) | 0,4423 | 0,5073 |
| c: b + Batı alıntısı Arapça sayılmaz | 0,6048 (0,6451) | 0,4471 | 0,5073 |
| **d: a + Batı alıntısı Arapça sayılmaz** (seçildi) | 0,6037 (0,6451) | **0,4620** | 0,5084 |

## 4. Ayar sonuçları (seçilen adaylar; `ayar.log`, `res_tr.json`, `res_tdk_ayar.json`, `res_tettl_ayar.json`)

| koşul | TR train+dev ham | TR acc_nat | TR Arapça | TR Fransızca | TDK ayar acc_nat | TETTL ayar acc_nat | xturkic | Saha |
|---|---|---|---|---|---|---|---|---|
| prod | 0,6416 | 0,6020 | 112/148 (0,757) | 63/92 | 0,4519 | 0,5073 | 0,7459 | 0,7136 |
| R1 | **0,6348 ✗** | 0,5942 | 112 | 61 (diyakronik, oftalmolojik -> it) | 0,4471 | 0,5073 | 0,7459 | 0,7136 |
| R2 | 0,6451 | 0,6037 | 114 (0,770) | 62 (aktör -> ar أحضر kaybı; kobalt korunur) | 0,4620 | 0,5084 | 0,7459 | 0,7136 |
| R3 | **0,6382 ✗** | 0,5960 | 114 | 60 | 0,4572 | 0,5084 | 0,7459 | 0,7136 |

R2'nin ayardaki 8 uyuşmazlığı: kazanç mazur (معذرة, doğru etimon), eşhas (شهد), askerî (ظهر), kudret
(قطر), gurbet (هرب) — mazur dışındakiler sınıf doğru ama etimon YANLIŞ (şans iskelet eşi); kayıp aktör
(acteur -> أحضر), kulis (coulisse -> قلد), kadran (-> حزيران).
Ağırlıklı işaret testi (aday/prod, ayar): R2 TR 2/1 p=1,0; TDK ayar 2/1 p=0,5; TETTL 1/1 p=1,0.
⚠️ Önceden görülen: **R1 ve R3 Türkçe train+dev korumasını DÜŞÜRDÜ** (0,6348 / 0,6382 < 0,6416) —
rapor sonucu ne olursa olsun kabul edilemez; yine raporlanır. **Hiçbir aday Türkçe train+dev Arapça
duyarlılığını 0,80'e döndürmüyor** (en çok R2 0,770): kabulün "eşitlik + Arapça ≥ 0,80" yolu ayarda
zaten kapalı; kabul yalnız birincil ölçütte anlamlı artışla mümkün. Tanıya göre (kayıplar şans
eşleşmesi) büyük etki beklenmiyor.

## 5. Yeni rapor altını (`build_gold.py` -> `gold.json`, tuz `donor9m-tdk-v1`, n = 600)

TDK GTS v12 `lisan`, 9j/9l madde kuralı; **dışarıda**: TDK+Nişanyan (items+disagreements), 9e, 9f, 9g,
9j-TDK (ayar+rapor), 9j-TETTL (ayar+rapor), 9l altınlarının kelimeleri ve (verici, etimon) grupları;
grup başına tek madde. **Doğal oranlarda**: kota `tum` payları kalan sınıflar üzerinde yeniden
normalleştirilmiş (İtalyanca/Yunanca/Ermenice önceki altınlarda tükendi: mevcut 0).

| ar | fr | fa | toplam | İngilizce anlamlı |
|---|---|---|---|---|
| 290 (0,483) | 248 (0,413) | 62 (0,104) | **600** | 295 |

Mevcut: ar 4.232, fr 3.961, fa 449. Düşenler: önceki altında kelime 3.560, etimon 65, anlam yok 602,
aynı grup 102, biçim 20. Tek bölüm `rapor`.
**Mühür**: `gold.json` sha256 `8a4811d250fc45e2c10a41c7f12b30a4e2629200cac13369fad35c20449378a6`.

## Sızıntı (K2, `k2.json`, n = 600; doğruluk hesaplanmadı) — GEÇTİ
Ağ kapalı (`socket.connect` engelli), kör indeks; koşu sonunda yüklü getirici **yok**; kör indekste
köken sütunları 0 dolu; anlam kör = tam 600/600; etiket (prod, r1, r2, r3) kör vs tam 600/600 aynı.
Havuz dilleri yalnız kaikki (+ Starling monget). Anlamda "from" geçen 3 madde (hasır "mat woven from
reeds", sahtiyan, melas) — dil adı yok.

## Rapor (bir kez, bu commit'ten sonra)
`NINEM_CONDS=base,g1,g2,h1,prod,r1,r2,r3 python data/cache/work/donor9m/harness.py gold 9m`.

**Birincil**: `acc_nat` (tum), her aday `prod`a karşı, eşleştirilmiş ağırlıklı işaret çevirme testi
(`natural.signflip`; ≤ 20 uyuşmazlıkta kesin, yoksa 200.000 örnek, tohum 9), iki yönlü, Holm (3 aday),
α = 0,05. Bilgi: ham doğruluk + McNemar, `acc_nat` altin, sınıf başına, base/g1/g2/h1.

Kabul (aday için hepsi):
1. Birincil: (a) rapor `acc_nat` artış yönünde ve Holm p < 0,05; YA DA (b) eşitlik — Δacc_nat ≥ −0,005
   ve anlamlı düşüş yok — VE Türkçe train+dev Arapça duyarlılığı ≥ 0,80 VE rapor Arapça duyarlılığı
   prod'dan düşük değil. ((b) ayarda hiçbir aday için sağlanmıyor: R2 0,770.)
2. Rapor Fransızca duyarlılığı prod'dan en çok **0,02** düşük (≤ 4/248 madde).
3. Türkçe tr_donor train+dev ham doğruluk ≥ 0,6416 — ölçüldü (R1 0,6348 ✗, R2 0,6451, R3 0,6382 ✗);
   doğal ağırlıklı sürümü de raporlanır (prod 0,6020).
4. Saha `eval-donor` motor ≥ 0,7136 — ölçüldü (üçü aynı); xturkic ayar verici ≥ 0,7459 — ölçüldü (aynı).
5. `eval-borrowing` F'ler aynı — yapı gereği (yalnız etiket); kabul edilirse `make eval-borrowing` ile doğrulanır.

Karar: kabul edilen aday varsayılan AÇIK (R3 ancak R1 ve R2 ikisi de kabulse; R1/R3 3. koşulu ayarda
sağlamadığı için pratikte yalnız R2 açılabilir). Hiçbiri -> üretim değişmez (bayraklar kapalı, kod
bayrak arkasında). Her durumda `tr_donor_eval`e `accuracy_natural` kalıcı ölçüt olarak eklenir.
