# Taban çizgisi ölçümü

Bu dosya `make eval-baseline` tarafından **otomatik üretilir** — elle
düzenlemeyin. Her sayı, adı geçen veri kümesi sürümünden sıfırdan
hesaplanır.

- **Veri kümesi:** `savelyevturkic` `v2.1` (commit `4a540590580f`)
- **Ölçüm:** 2026-08-27T12:34:27+00:00
- **Bölüm:** `dev`
- **Altın standart:** 400 madde · train 237 / dev 83 / test 80
- **Kavram sızıntısı:** 0 (0 olmalı)
- **Ata düğüm:** PT 115 · PCT 285 — Çuvaşça tanığı olmayan kümede iddia edilebilecek en derin düğüm Ana Ortak Türkçe'dir

## `tum_veri_capa_haric` — n=83

> DÜRÜST KOŞUL — tüm altın standart, çapa dilinin tanığı girdiden çıkarılmış

**Birincil metrikler NED ve B-Cubed F'tir** (SIGTYP 2022 resmi
metrikleri; List 2019). Tam doğruluk ikincildir ve alanda tek başına
raporlanmaz — n=400'de en oynak ölçüdür.

| Sistem | **NED**↓ | **BCFS**↑ | ED↓ | FER↓ | tam | kapsam |
|---|---|---|---|---|---|---|
| **comparative** | **0.306** | **0.583** | 1.48 | 0.286 | 0.361 | 1.000 |
| copy_anchor | **0.392** | **0.529** | 1.88 | 0.353 | 0.217 | 1.000 |
| copy_random_daughter | **0.403** | **0.520** | 1.86 | 0.353 | 0.229 | 1.000 |
| copy_longest | **0.502** | **0.429** | 2.58 | 0.514 | 0.145 | 1.000 |
| majority_character | **0.341** | **0.571** | 1.60 | 0.299 | 0.325 | 1.000 |

> **BİRİNCİL (NED)** `comparative` vs `majority_character`: fark **-0.0345** (düşük olan iyi), %95 GA [-0.0707, +0.0016] → anlamlı DEĞİL — güven aralığı sıfırı içeriyor.

> ikincil (tam doğruluk) `comparative` vs `majority_character`: fark **+0.0361**, %95 GA [-0.0361, +0.1084], permütasyon p=0.554, McNemar p=0.549 → anlamlı DEĞİL — güven aralığı sıfırı içeriyor.

## `tum_veri_capa_dahil` — n=83

> çapa dili girdide bırakılmış — motor kendi sorusunu tanık olarak görüyor

**Birincil metrikler NED ve B-Cubed F'tir** (SIGTYP 2022 resmi
metrikleri; List 2019). Tam doğruluk ikincildir ve alanda tek başına
raporlanmaz — n=400'de en oynak ölçüdür.

| Sistem | **NED**↓ | **BCFS**↑ | ED↓ | FER↓ | tam | kapsam |
|---|---|---|---|---|---|---|
| **comparative** | **0.294** | **0.594** | 1.42 | 0.276 | 0.386 | 1.000 |
| copy_anchor | **0.392** | **0.529** | 1.88 | 0.353 | 0.217 | 1.000 |
| copy_random_daughter | **0.438** | **0.471** | 2.05 | 0.384 | 0.193 | 1.000 |
| copy_longest | **0.542** | **0.412** | 2.78 | 0.555 | 0.072 | 1.000 |
| majority_character | **0.370** | **0.539** | 1.76 | 0.326 | 0.277 | 1.000 |

> **BİRİNCİL (NED)** `comparative` vs `majority_character`: fark **-0.0762** (düşük olan iyi), %95 GA [-0.1195, -0.0375] → **motor daha iyi**.

> ikincil (tam doğruluk) `comparative` vs `majority_character`: fark **+0.1084**, %95 GA [+0.0241, +0.1928], permütasyon p=0.021, McNemar p=0.022 → **anlamlı**.

## `15_tanik_capa_haric` — n=34

> yalnız 15+ tanıklı kolay altküme, çapa çıkarılmış

**Birincil metrikler NED ve B-Cubed F'tir** (SIGTYP 2022 resmi
metrikleri; List 2019). Tam doğruluk ikincildir ve alanda tek başına
raporlanmaz — n=400'de en oynak ölçüdür.

| Sistem | **NED**↓ | **BCFS**↑ | ED↓ | FER↓ | tam | kapsam |
|---|---|---|---|---|---|---|
| **comparative** | **0.258** | **0.669** | 1.18 | 0.235 | 0.412 | 1.000 |
| copy_anchor | **0.362** | **0.630** | 1.65 | 0.336 | 0.235 | 1.000 |
| copy_random_daughter | **0.401** | **0.608** | 1.71 | 0.366 | 0.206 | 1.000 |
| copy_longest | **0.596** | **0.443** | 3.03 | 0.653 | 0.029 | 1.000 |
| majority_character | **0.318** | **0.669** | 1.41 | 0.290 | 0.353 | 1.000 |

> **BİRİNCİL (NED)** `comparative` vs `majority_character`: fark **-0.0608** (düşük olan iyi), %95 GA [-0.1044, -0.0240] → **motor daha iyi**.

> ikincil (tam doğruluk) `comparative` vs `majority_character`: fark **+0.0588**, %95 GA [+0.0000, +0.1471], permütasyon p=0.498, McNemar p=0.500 → anlamlı DEĞİL — güven aralığı sıfırı içeriyor.

## `15_tanik_capa_dahil` — n=34

> kolay altküme + çapa sızıntısı — ÖN ÖLÇÜMÜN koşuluna en yakın hâli

**Birincil metrikler NED ve B-Cubed F'tir** (SIGTYP 2022 resmi
metrikleri; List 2019). Tam doğruluk ikincildir ve alanda tek başına
raporlanmaz — n=400'de en oynak ölçüdür.

| Sistem | **NED**↓ | **BCFS**↑ | ED↓ | FER↓ | tam | kapsam |
|---|---|---|---|---|---|---|
| **comparative** | **0.253** | **0.691** | 1.15 | 0.235 | 0.412 | 1.000 |
| copy_anchor | **0.362** | **0.630** | 1.65 | 0.336 | 0.235 | 1.000 |
| copy_random_daughter | **0.435** | **0.539** | 1.88 | 0.398 | 0.176 | 1.000 |
| copy_longest | **0.596** | **0.441** | 3.03 | 0.653 | 0.029 | 1.000 |
| majority_character | **0.326** | **0.660** | 1.44 | 0.293 | 0.353 | 1.000 |

> **BİRİNCİL (NED)** `comparative` vs `majority_character`: fark **-0.0730** (düşük olan iyi), %95 GA [-0.1235, -0.0304] → **motor daha iyi**.

> ikincil (tam doğruluk) `comparative` vs `majority_character`: fark **+0.0588**, %95 GA [+0.0000, +0.1471], permütasyon p=0.498, McNemar p=0.500 → anlamlı DEĞİL — güven aralığı sıfırı içeriyor.

## Yorum

Dürüst koşulda motor **%36.1 tam** (%44.6 kabul edilebilir) alıyor.

Aynı motor, kolay altküme seçilip çapa sızıntısı bırakıldığında **%41.2 tam** (%58.8) gösteriyor. Aradaki fark yöntemden değil, ölçüm kurgusundan geliyor — bu yüzden raporlanan sayı daima dürüst koşulun sayısıdır.

Motorun aşması gereken çıta, en iyi trivial taban çizgisidir: `copy_anchor` %21.7, `copy_random_daughter` %22.9, `copy_longest` %14.5, `majority_character` %32.5.

### Negatif kontroller

Yüksek doğruluk, yüksek yanlış-pozitif oranıyla birlikte anlamsızdır.
**Güçlü iddia oranı** en kritik sütundur: motorun uydurma veya alıntı
bir kelimeye 🟢/🟡 rozet verme oranı sıfır olmalıdır.

| Batarya | n | rekonstrükte | yanlış-pozitif | güçlü iddia |
|---|---|---|---|---|
| `fonotaktik_gecerli_sahte` | 8 | 8 | 1.000 | **0.000** |
| `bariz_sahte` | 4 | 0 | 0.000 | **0.000** |
| `sahte_akraba` | 4 | 4 | 1.000 | **0.000** |
| `alinti_tuzagi` | 5 | 4 | 0.800 | **0.000** |
| `eşadlı` | 3 | 3 | 1.000 | **0.000** |

### İstatistiksel durum

Her koşulun altındaki satır, motor ile en iyi trivial taban çizgisi
arasındaki farkın eşleşmiş permütasyon ve McNemar testiyle sınanmış
sonucunu verir. **Bootstrap güven aralığı sıfırı içeriyorsa fark
anlamlı değildir** ve öyle raporlanır — bu, motorun kötü olduğunu
değil, farkın henüz kanıtlanmadığını söyler.
