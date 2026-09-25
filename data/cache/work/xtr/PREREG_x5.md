# Ön-kayıt — X5: birleşik tek aday (X4 A2 + X2 mean + X3 S3 fallback), R3'te bir kez

Yazıldı: 2026-09-25, R3 bölümü HİÇ açılmadan (R3'te hiçbir motor sinyali
yakalanmadı, hiçbir karar görülmedi). Önceki ön kayıtlar: `PREREG_pmi.md`
(X2, R1), `PREREG_sense.md` (X3, R1), `PREREG_x4.md` (X4, R2). Üçü de
"belirsiz (güç yetersiz)" bitti; yönleri tutarlı: X2 mean +0,021
[−0,003, +0,045], X3 S3 fallback +0,025 [−0,002, +0,053], X4 A2 +0,014
[−0,003, +0,031]. R1 ve R2 görüldü; yeni, daha büyük, dokunulmamış bölüm R3.

## 1. R3 bölümü (`data/gold/xturkic/r3.jsonl`, `SEAL_r3.json`, `stats_r3.json`)

Kod: `engine/evaluation/xturkic_gold.py` `build-r3` (X1 altını ve mührü
DEĞİŞMEZ; `verify` dört bölümde de doğru). Etiket kuralı X1'inkiyle aynı
(`label_record` + `item_exclusion`), üstüne yalnız R3'te:

- **Güçlendirilmiş melez türetme süzgeci** (X1 kullanıcı denetimi izi,
  ug نۇقسانسىز): miras + yapım şablonu (af/suf/surf/com…) bileşeni aynı
  dökümde yabancı kökenli (`melez_kök`); miras + metnin akraba/karşılaştırma
  cümleleri dışında Türk dışı dil adı (Çağatay aracılı "inherited"
  alıntılar: uz harorat, ug مۇسۇلمان; `miras_metninde_yabancı_dil`); miras +
  `root` şablonu Türk dışı (`miras_yabancı_kök`); ve iki sınıfta da
  "Or from …" / "or an early borrowing" çekincesi (`çekince_or`, Q4'te alt
  безгек). Bilinen örnekler (ug نۇقسانسىز, ug مۇسۇلمان, uz harorat)
  süzgeçte düşüyor. Süzgecin R3 adaylarından düşürdüğü: metinde yabancı dil 50, melez kök 34, çekince_or 6, yabancı kök 2 (13 dil, bütün adaylar).
- **Kaynaklar:** (a) X1'de kullanılmayan diller, kaikki en: az, crh, gag,
  kum, nog, alt. kaa: kaikki en dökümü yok (404); krc: yalnız Rusça sürüm
  (plan: ru sürümü altın değil); sah: WOLD. (b) X1 dillerinde 250 sınırı
  yüzünden seçilmemiş maddeler.
- **Etimon sızıntısı yok:** X1 seçimi bellekte yeniden hesaplandı (dosyalar
  okunmadı; yeniden hesaplanan dört bölümün sha256'sı `SEAL.json` ile
  birebir aynı) ve X1'in herhangi bir bölümünde (test DAHİL; testten yalnız
  etimon anahtarı) bulunan her etimon grubu dışlandı. Dışlanan: 5.308 aday
  (çoğu ortak Proto-Türkçe biçimli miras).
- **Dengeleme:** dil × sınıf ≤ 350, Rusça ≤ %50, Moğolcanın hepsi; hücrede
  < 20 madde kalırsa hücre alınmaz (Q4 denetlenemez: nog/alıntı 2).
- **Q4 (yeni diller, LLM ön-etiketi — Claude, X5 ajanı):** R3 havuzundan
  dil × sınıf başına 20 madde, motor çıktısı görülmeden, kanıt kartıyla
  (`x5/audit_cards_r3.txt`, `x5/audit_verdicts_r3.csv`, `x5/Q4_r3.json`).
  10 hücre 20/20 (Wilson alt 0,839) kabul; **alt/alıntı 18/20** (alt тура
  "ev" büyük olasılıkla Türkçe *tura; безгек çekinceli) — havuzda 27 madde
  olduğundan 25/27 bile Wilson alt 0,766 < 0,80 → hücre MÜHÜRDEN ÖNCE
  çıkarıldı. Kullanıcı ≥%10'u bağımsız denetlemeli (plan). X1 dillerinin
  artığı X1 Q4'ünden geçmiş dillerdir.

**Boyut (mühürlü): n = 3.706**, 2.679 etimon grubu; alıntı 2.097 / miras
1.609; yeni dil 1.792, X1 artığı 1.914. Verici: ar 899, ru 684, fa 385,
diğer 128, mn 1.

| dil | alıntı | miras | | dil | alıntı | miras |
|---|---|---|---|---|---|---|
| az | 350 | 350 | | kk | 350 | 209 |
| crh | 350 | 97 | | ug | 350 | 89 |
| gag | 37 | 131 | | uz | 350 | 0 |
| kum | 60 | 61 | | tk | 208 | 0 |
| nog | 0 | 55 | | ba | 42 | 171 |
| alt | 0 | 301 | | ky | 0 | 117 |
| | | | | tt | 0 | 28 |

Tek sınıflı diller (uz, tk yalnız alıntı; ky, tt, nog, alt yalnız miras)
etimon sızıntısı dışlamasının sonucudur; F birleşik (havuzlanmış) ölçülür.
**Önceden yazılır:** yeni dillerde (ve ayarda olmayan dillerde) fonotaktik
dizilim modeli yoktur (`fonotaktik_model` = 0); bu alan kayması taban ve
adayda AYNIDIR, eşleşmiş farkı yanlılaştırmaz ama mutlak F'leri düşürebilir.

## 2. Güç (MDE) — `x5/MDE_r3.json`

Ayar bölümünde (n=1.379, kat dışı) aday − taban `engine_trained` F farkının
kümelenmiş eşleşmiş bootstrap SE'si, `sqrt(n_ayar / n)` ile ölçeklendi
(α=0,05 iki yönlü, güç 0,80). Birincil aday x5c: ayar SE 0,0114 →
**MDE ≤ 0,02 için n ≥ 3.487**; bunu sağlayan en küçük sınır (50'lik adım)
350 → n=3.706, **MDE ≈ 0,019**. Bileşenler: A2 0,010, mean 0,014, S3 0,013.
(X1'in vekil çiftleri 0,038–0,053 veriyordu; o vekiller aday farkı değil,
iki farklı sistem farkıydı — gerçek aday çiftleri daha az uyumsuz.)

## 3. Tek aday C = `x5c` (Holm YOK)

`xborrowing_eval.apply_variant("x5c")`: `ETY_DONOR_CLEAN=1`,
`ETY_DONOR_RAMP_CHANCE=1`, `ETY_DONOR_SENSE_FILTER=s3:fallback`,
`STRENGTH_DISTANCE="mean"`, eşik 0,60 / tavan 0,85. Hepsi önceki ön
kayıtlardaki SABİT değerler; yeniden ayar YOK.

**Birleştirme kuralı (tek yeni öğe, ölçümden ÖNCE tanımlandı):** X4 A2'nin
rampa null'ı SCA ölçeğindeydi (o zaman güç mesafesi de SCA idi). mean'de
rampa 0,60–0,85 ortalama-mesafe ölçeğindedir; SCA null'ıyla karşılaştırmak
ölçek karışıklığı olur (neredeyse bütün rampayı öldürür — mekanizmayı yapay
olarak "geçirirdi"). Null bu yüzden güç mesafesinin ölçeğinde:
`donor_proximity._ramp_null` (aynı kontrol kelimeleri, aynı havuz, medyan;
`sca`'da X4 A2 ile birebir). Parametre yok. Bu kural ayar yakalamasından
önce yazıldı; ayardan hiçbir şey seçilmedi.

Taban: `sca` (X1 üretimi: sca 0,35/0,60, üç bayrak kapalı; artık açıkça
kurulur). İkincil (bilgi, karar DEĞİL): her bileşen tek başına — `x4a2`,
`mean`, `sense_s3fallback`.

## 4. Ayar doğrulaması (kat dışı, n=1.379; betimsel, seçim YOK)

| çeşit | F | doğr. | dp kes. | dp duy. | verici tanıma | rampa miras | rampa alıntı | ΔF [GA95] |
|---|---|---|---|---|---|---|---|---|
| taban sca | 0,7611 | 0,7382 | 0,7978 | 0,7320 | 0,5499 | 0,5571 | 0,1694 | |
| **x5c** | **0,8155** | 0,8165 | 0,8793 | 0,7614 | 0,5774 | **0,6400** | 0,1591 | +0,054 [+0,033, +0,077] |
| x4a2 | 0,7672 | 0,7527 | 0,8139 | 0,7408 | 0,5568 | 0,4214 | 0,1252 | +0,006 [−0,006, +0,018] |
| mean | 0,7773 | 0,7549 | 0,8339 | 0,7541 | 0,5550 | 0,7329 | 0,1856 | +0,016 [+0,000, +0,032] |
| s3 fallback | 0,7753 | 0,7600 | 0,8404 | 0,7290 | 0,5678 | 0,5843 | 0,1826 | +0,014 [−0,001, +0,029] |

Mirasta verici yakınlığı (ayar): sca yakın 0,181 / rampa 0,557 / herhangi
0,739; x5c 0,101 / 0,640 / 0,741; mean 0,146 / 0,733 / 0,879.

**Önceden yazılır — mekanizma ölçütünün TUTMAMASI bekleniyor:** mean'in
geniş bandı (0,60–0,85) mirasta rampayı büyütüyor (0,557 → 0,733); A2
denetimi bunu 0,640'a indiriyor ama sca tabanının altına indirmiyor
(ayarda McNemar 390 → 448, p ≈ 1). Rampa, ölçekler farklı olduğu için
tabanla birebir aynı nesne değildir; yine de ölçüt görevde verildiği gibi
(taban sca'ya göre) uygulanır ve DEĞİŞTİRİLMEZ. Birincil F ölçütü tutup
mekanizma tutmazsa sonuç "KABUL EDİLMEDİ (mekanizma ölçütü)" yazılır.
Ayrıca betimsel raporlanır: A2'nin birleşik içindeki etkisi (x5c'nin rampası
mean'e göre), mirasta herhangi bir ateşlenme, mirasta yakın (güç 1) oranı.

## 5. R3 protokolü (BİR KEZ)

```
ETY_LEXICON_INDEX=$PWD/data/cache/work/xtr/index_blind.db python -m engine.evaluation.xborrowing_eval capture --split r3 --prereg data/cache/work/xtr/PREREG_x5.md
… capture --split r3 --variant x5c --tag x5c --prereg …
… capture --split r3 --variant x4a2 --tag x4a2 --prereg …
… capture --split r3 --variant mean --tag mean --prereg …
… capture --split r3 --variant sense:s3:fallback --tag sense_s3fallback --prereg …
python -m engine.evaluation.xborrowing_eval compare --split r3 --prereg data/cache/work/xtr/PREREG_x5.md --tags x5c x4a2 mean sense_s3fallback
```
Her çeşidin fonotaktik dizilim modeli + birleştiricisi KENDİ ayar
önbelleğinin tamamında BELLEKTE eğitilir (K7) ve R3'te ölçülür. Kilit
`.opened_r3_x5`. Kör indeks + zincir kapalı (ana ölçüm). `compare` birden
çok etikette Holm'lu p de yazar; **x5c kararı Holm'suz** (tek aday),
bileşenlerin p'si yalnız bilgidir.

## 6. Kabul ölçütü (x5c; hepsi birden)

1. **Birincil:** `engine_trained` F(x5c) − F(sca), etimona göre kümelenmiş
   eşleşmiş bootstrap (2.000, tohum 20260925): %95 GA alt ucu > 0.
2. **Mekanizma:** mirasta rampa ateşlenmesi azalır (0 < güç < 1; kesin
   McNemar, tek yönlü p < 0,05; taban sca).
3. **İkincil (gerekli):** verici tanıma (`donor_identification`,
   engine_trained) x5c ≥ sca − 0,02.
4. **Koruma** (görülmüş veri; yalnız gerekli şart): `x5/run_beval_x5.py x5c`
   (make eval-borrowing eşdeğeri, izlenen dosyalara yazmaz): WOLD
   `engine_trained` F ≥ 0,6482 VE Türkçe `engine_trained` F ≥ 0,8773;
   arama yolu: `x5/run_replay_x5.py x5c` (`ranker/replay.py new dev`
   eşdeğeri) bayrak kapalıyken uyum ≥ 129/135. `ETY_SEARCH_DONOR_PROXIMITY`
   açıkken uyum ayrı karar: ≥129/135 değilse arama yolu bayrağı kapalı
   kalır (bugün de kapalı; üretim 119/135).

**Kabul →** üretim varsayılanları açılır: `donor_index.CLEAN_DEFAULT = True`,
`donor_proximity.RAMP_CHANCE_DEFAULT = True`, `sense_match.ACCEPTED_SPEC =
"s3:fallback"` ve `ETY_DONOR_SENSE_FILTER` varsayılan açık,
`STRENGTH_DISTANCE = "mean"`, eşik 0,60 / tavan 0,85 (bayraklarla
kapatılabilir). Koruma koşuları üretim ayarıyla yinelenir, commit.
**Red →** varsayılanlar kapalı kalır; sonuç JSON + kod yorumu.

Ek raporlanır (karar değil): alt kümeler yeni dil / X1 artığı (eşleşmiş ΔF),
dil ve verici kırılımları, dp kesinliği, kolay/zor duyarlılık, miras
özgüllüğü, bileşenlerin ΔF'si. Verici tanıma ölçümünde Wiktionary verici
etiketi hatası beklenir (X1 notu: kk зақым; Q4'te crh kotlet Fransızca
etiketli, büyük olasılıkla Rusça aracılı).
