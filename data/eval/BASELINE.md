# Taban çizgisi ölçümü

Bu dosya `make eval-baseline` tarafından **otomatik üretilir** — elle
düzenlemeyin. Her sayı, adı geçen veri kümesi sürümünden sıfırdan
hesaplanır.

- **Veri kümesi:** `savelyevturkic` `v2.1` (commit `4a540590580f`)
- **Ölçüm:** 2026-08-27T09:20:55+00:00
- **Bölüm:** `all`
- **Altın standart:** 400 madde · train 237 / dev 83 / test 80
- **Kavram sızıntısı:** 0 (0 olmalı)
- **Ata düğüm:** PT 115 · PCT 285 — Çuvaşça tanığı olmayan kümede iddia edilebilecek en derin düğüm Ana Ortak Türkçe'dir

## `tum_veri_capa_haric` — n=400

> DÜRÜST KOŞUL — tüm altın standart, çapa dilinin tanığı girdiden çıkarılmış

**Birincil metrikler NED ve B-Cubed F'tir** (SIGTYP 2022 resmi
metrikleri; List 2019). Tam doğruluk ikincildir ve alanda tek başına
raporlanmaz — n=400'de en oynak ölçüdür.

| Sistem | **NED**↓ | **BCFS**↑ | ED↓ | FER↓ | tam | kapsam |
|---|---|---|---|---|---|---|
| **comparative** | **0.381** | **0.516** | 1.93 | 0.358 | 0.235 | 0.985 |
| copy_anchor | **0.417** | **0.466** | 2.13 | 0.392 | 0.203 | 1.000 |
| copy_random_daughter | **0.453** | **0.427** | 2.29 | 0.414 | 0.168 | 1.000 |
| copy_longest | **0.552** | **0.362** | 3.04 | 0.541 | 0.100 | 1.000 |
| majority_character | **0.382** | **0.500** | 1.94 | 0.363 | 0.237 | 1.000 |

> **BİRİNCİL (NED)** `comparative` vs `majority_character`: fark **-0.0008** (düşük olan iyi), %95 GA [-0.0170, +0.0164] → anlamlı DEĞİL — güven aralığı sıfırı içeriyor.

> ikincil (tam doğruluk) `comparative` vs `majority_character`: fark **-0.0025**, %95 GA [-0.0300, +0.0250], permütasyon p=1.000, McNemar p=1.000 → anlamlı DEĞİL — güven aralığı sıfırı içeriyor.

## `tum_veri_capa_dahil` — n=400

> çapa dili girdide bırakılmış — motor kendi sorusunu tanık olarak görüyor

**Birincil metrikler NED ve B-Cubed F'tir** (SIGTYP 2022 resmi
metrikleri; List 2019). Tam doğruluk ikincildir ve alanda tek başına
raporlanmaz — n=400'de en oynak ölçüdür.

| Sistem | **NED**↓ | **BCFS**↑ | ED↓ | FER↓ | tam | kapsam |
|---|---|---|---|---|---|---|
| **comparative** | **0.383** | **0.519** | 1.94 | 0.352 | 0.242 | 0.973 |
| copy_anchor | **0.417** | **0.466** | 2.13 | 0.392 | 0.203 | 1.000 |
| copy_random_daughter | **0.457** | **0.428** | 2.30 | 0.422 | 0.158 | 1.000 |
| copy_longest | **0.560** | **0.360** | 3.11 | 0.550 | 0.080 | 1.000 |
| majority_character | **0.395** | **0.487** | 2.02 | 0.368 | 0.223 | 1.000 |

> **BİRİNCİL (NED)** `comparative` vs `majority_character`: fark **-0.0114** (düşük olan iyi), %95 GA [-0.0324, +0.0099] → anlamlı DEĞİL — güven aralığı sıfırı içeriyor.

> ikincil (tam doğruluk) `comparative` vs `majority_character`: fark **+0.0200**, %95 GA [-0.0125, +0.0525], permütasyon p=0.272, McNemar p=0.280 → anlamlı DEĞİL — güven aralığı sıfırı içeriyor.

## `15_tanik_capa_haric` — n=158

> yalnız 15+ tanıklı kolay altküme, çapa çıkarılmış

**Birincil metrikler NED ve B-Cubed F'tir** (SIGTYP 2022 resmi
metrikleri; List 2019). Tam doğruluk ikincildir ve alanda tek başına
raporlanmaz — n=400'de en oynak ölçüdür.

| Sistem | **NED**↓ | **BCFS**↑ | ED↓ | FER↓ | tam | kapsam |
|---|---|---|---|---|---|---|
| **comparative** | **0.347** | **0.606** | 1.52 | 0.336 | 0.279 | 0.981 |
| copy_anchor | **0.365** | **0.568** | 1.62 | 0.351 | 0.260 | 1.000 |
| copy_random_daughter | **0.446** | **0.470** | 1.97 | 0.399 | 0.165 | 1.000 |
| copy_longest | **0.632** | **0.355** | 3.35 | 0.650 | 0.038 | 1.000 |
| majority_character | **0.340** | **0.603** | 1.49 | 0.330 | 0.285 | 1.000 |

> **BİRİNCİL (NED)** `comparative` vs `majority_character`: fark **+0.0075** (düşük olan iyi), %95 GA [-0.0188, +0.0369] → anlamlı DEĞİL — güven aralığı sıfırı içeriyor.

> ikincil (tam doğruluk) `comparative` vs `majority_character`: fark **-0.0063**, %95 GA [-0.0506, +0.0380], permütasyon p=1.000, McNemar p=1.000 → anlamlı DEĞİL — güven aralığı sıfırı içeriyor.

## `15_tanik_capa_dahil` — n=158

> kolay altküme + çapa sızıntısı — ÖN ÖLÇÜMÜN koşuluna en yakın hâli

**Birincil metrikler NED ve B-Cubed F'tir** (SIGTYP 2022 resmi
metrikleri; List 2019). Tam doğruluk ikincildir ve alanda tek başına
raporlanmaz — n=400'de en oynak ölçüdür.

| Sistem | **NED**↓ | **BCFS**↑ | ED↓ | FER↓ | tam | kapsam |
|---|---|---|---|---|---|---|
| **comparative** | **0.344** | **0.608** | 1.51 | 0.332 | 0.285 | 0.981 |
| copy_anchor | **0.365** | **0.568** | 1.62 | 0.351 | 0.260 | 1.000 |
| copy_random_daughter | **0.460** | **0.465** | 1.99 | 0.418 | 0.158 | 1.000 |
| copy_longest | **0.628** | **0.355** | 3.34 | 0.652 | 0.038 | 1.000 |
| majority_character | **0.341** | **0.604** | 1.49 | 0.330 | 0.285 | 1.000 |

> **BİRİNCİL (NED)** `comparative` vs `majority_character`: fark **+0.0034** (düşük olan iyi), %95 GA [-0.0226, +0.0331] → anlamlı DEĞİL — güven aralığı sıfırı içeriyor.

> ikincil (tam doğruluk) `comparative` vs `majority_character`: fark **+0.0000**, %95 GA [-0.0443, +0.0443], permütasyon p=1.000, McNemar p=1.000 → anlamlı DEĞİL — güven aralığı sıfırı içeriyor.

## Yorum

Dürüst koşulda motor **%23.5 tam** (%32.0 kabul edilebilir) alıyor.

Aynı motor, kolay altküme seçilip çapa sızıntısı bırakıldığında **%28.5 tam** (%42.4) gösteriyor. Aradaki fark yöntemden değil, ölçüm kurgusundan geliyor — bu yüzden raporlanan sayı daima dürüst koşulun sayısıdır.

Motorun aşması gereken çıta, en iyi trivial taban çizgisidir: `copy_anchor` %20.2, `copy_random_daughter` %16.8, `copy_longest` %10.0, `majority_character` %23.8.

### Negatif kontroller

Yüksek doğruluk, yüksek yanlış-pozitif oranıyla birlikte anlamsızdır.
**Güçlü iddia oranı** en kritik sütundur: motorun uydurma veya alıntı
bir kelimeye 🟢/🟡 rozet verme oranı sıfır olmalıdır.

| Batarya | n | rekonstrükte | yanlış-pozitif | güçlü iddia |
|---|---|---|---|---|
| `fonotaktik_gecerli_sahte` | 8 | 8 | 1.000 | **0.000** |
| `bariz_sahte` | 4 | 0 | 0.000 | **0.000** |
| `sahte_akraba` | 4 | 4 | 1.000 | **0.000** |
| `alinti_tuzagi` | 5 | 3 | 0.600 | **0.000** |
| `eşadlı` | 3 | 3 | 1.000 | **0.000** |

### İstatistiksel durum

Her koşulun altındaki satır, motor ile en iyi trivial taban çizgisi
arasındaki farkın eşleşmiş permütasyon ve McNemar testiyle sınanmış
sonucunu verir. **Bootstrap güven aralığı sıfırı içeriyorsa fark
anlamlı değildir** ve öyle raporlanır — bu, motorun kötü olduğunu
değil, farkın henüz kanıtlanmadığını söyler.
