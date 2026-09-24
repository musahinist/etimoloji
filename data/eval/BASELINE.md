# Taban çizgisi ölçümü

Bu dosya `make eval-baseline` tarafından **otomatik üretilir** — elle
düzenlemeyin. Her sayı, adı geçen veri kümesi sürümünden sıfırdan
hesaplanır.

- **Veri kümesi:** `savelyevturkic` `v2.1` (commit `4a540590580f`)
- **Ölçüm:** 2026-09-24T09:15:12+00:00
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
| **comparative** | **0.309** | **0.597** | 1.51 | 0.264 | 0.398 | 0.976 |
| copy_anchor | **0.390** | **0.529** | 1.87 | 0.351 | 0.229 | 1.000 |
| copy_random_daughter | **0.401** | **0.520** | 1.84 | 0.351 | 0.241 | 1.000 |
| copy_longest | **0.500** | **0.429** | 2.57 | 0.512 | 0.157 | 1.000 |
| majority_character | **0.339** | **0.571** | 1.59 | 0.297 | 0.337 | 1.000 |

> **BİRİNCİL (NED)** `comparative` vs `majority_character`: fark **-0.0301** (düşük olan iyi), %95 GA [-0.0679, +0.0069] → anlamlı DEĞİL — güven aralığı sıfırı içeriyor.

> ikincil (tam doğruluk) `comparative` vs `majority_character`: fark **+0.0602**, %95 GA [-0.0120, +0.1325], permütasyon p=0.228, McNemar p=0.227 → anlamlı DEĞİL — güven aralığı sıfırı içeriyor.

## `tum_veri_capa_dahil` — n=83

> çapa dili girdide bırakılmış — motor kendi sorusunu tanık olarak görüyor

**Birincil metrikler NED ve B-Cubed F'tir** (SIGTYP 2022 resmi
metrikleri; List 2019). Tam doğruluk ikincildir ve alanda tek başına
raporlanmaz — n=400'de en oynak ölçüdür.

| Sistem | **NED**↓ | **BCFS**↑ | ED↓ | FER↓ | tam | kapsam |
|---|---|---|---|---|---|---|
| **comparative** | **0.299** | **0.603** | 1.47 | 0.265 | 0.410 | 0.976 |
| copy_anchor | **0.390** | **0.529** | 1.87 | 0.351 | 0.229 | 1.000 |
| copy_random_daughter | **0.436** | **0.471** | 2.04 | 0.382 | 0.205 | 1.000 |
| copy_longest | **0.540** | **0.412** | 2.77 | 0.553 | 0.084 | 1.000 |
| majority_character | **0.368** | **0.539** | 1.75 | 0.324 | 0.289 | 1.000 |

> **BİRİNCİL (NED)** `comparative` vs `majority_character`: fark **-0.0690** (düşük olan iyi), %95 GA [-0.1122, -0.0282] → **motor daha iyi**.

> ikincil (tam doğruluk) `comparative` vs `majority_character`: fark **+0.1205**, %95 GA [+0.0361, +0.2048], permütasyon p=0.012, McNemar p=0.013 → **anlamlı**.

## `15_tanik_capa_haric` — n=34

> yalnız 15+ tanıklı kolay altküme, çapa çıkarılmış

**Birincil metrikler NED ve B-Cubed F'tir** (SIGTYP 2022 resmi
metrikleri; List 2019). Tam doğruluk ikincildir ve alanda tek başına
raporlanmaz — n=400'de en oynak ölçüdür.

| Sistem | **NED**↓ | **BCFS**↑ | ED↓ | FER↓ | tam | kapsam |
|---|---|---|---|---|---|---|
| **comparative** | **0.268** | **0.675** | 1.21 | 0.240 | 0.412 | 1.000 |
| copy_anchor | **0.362** | **0.630** | 1.65 | 0.336 | 0.235 | 1.000 |
| copy_random_daughter | **0.401** | **0.608** | 1.71 | 0.366 | 0.206 | 1.000 |
| copy_longest | **0.596** | **0.443** | 3.03 | 0.653 | 0.029 | 1.000 |
| majority_character | **0.318** | **0.669** | 1.41 | 0.290 | 0.353 | 1.000 |

> **BİRİNCİL (NED)** `comparative` vs `majority_character`: fark **-0.0501** (düşük olan iyi), %95 GA [-0.0931, -0.0130] → **motor daha iyi**.

> ikincil (tam doğruluk) `comparative` vs `majority_character`: fark **+0.0588**, %95 GA [+0.0000, +0.1471], permütasyon p=0.498, McNemar p=0.500 → anlamlı DEĞİL — güven aralığı sıfırı içeriyor.

## `15_tanik_capa_dahil` — n=34

> kolay altküme + çapa sızıntısı — ÖN ÖLÇÜMÜN koşuluna en yakın hâli

**Birincil metrikler NED ve B-Cubed F'tir** (SIGTYP 2022 resmi
metrikleri; List 2019). Tam doğruluk ikincildir ve alanda tek başına
raporlanmaz — n=400'de en oynak ölçüdür.

| Sistem | **NED**↓ | **BCFS**↑ | ED↓ | FER↓ | tam | kapsam |
|---|---|---|---|---|---|---|
| **comparative** | **0.266** | **0.679** | 1.21 | 0.253 | 0.412 | 1.000 |
| copy_anchor | **0.362** | **0.630** | 1.65 | 0.336 | 0.235 | 1.000 |
| copy_random_daughter | **0.435** | **0.539** | 1.88 | 0.398 | 0.176 | 1.000 |
| copy_longest | **0.596** | **0.441** | 3.03 | 0.653 | 0.029 | 1.000 |
| majority_character | **0.326** | **0.660** | 1.44 | 0.293 | 0.353 | 1.000 |

> **BİRİNCİL (NED)** `comparative` vs `majority_character`: fark **-0.0596** (düşük olan iyi), %95 GA [-0.1081, -0.0181] → **motor daha iyi**.

> ikincil (tam doğruluk) `comparative` vs `majority_character`: fark **+0.0588**, %95 GA [+0.0000, +0.1471], permütasyon p=0.498, McNemar p=0.500 → anlamlı DEĞİL — güven aralığı sıfırı içeriyor.

## Yorum

Dürüst koşulda motor **%39.8 tam** (%48.2 kabul edilebilir) alıyor.

Aynı motor, kolay altküme seçilip çapa sızıntısı bırakıldığında **%41.2 tam** (%58.8) gösteriyor. Aradaki fark yöntemden değil, ölçüm kurgusundan geliyor — bu yüzden raporlanan sayı daima dürüst koşulun sayısıdır.

Motorun aşması gereken çıta, en iyi trivial taban çizgisidir: `copy_anchor` %22.9, `copy_random_daughter` %24.1, `copy_longest` %15.7, `majority_character` %33.7.

### Negatif kontroller

Yüksek doğruluk, yüksek yanlış-pozitif oranıyla birlikte anlamsızdır.
**Güçlü iddia oranı** en kritik sütundur: motorun uydurma veya alıntı
bir kelimeye 🟢/🟡 rozet verme oranı sıfır olmalıdır.

| Batarya | n | rekonstrükte | yanlış-pozitif | güçlü iddia |
|---|---|---|---|---|
| `fonotaktik_gecerli_sahte` | 58 | 1 | 0.017 | **0.000** |
| `bariz_sahte` | 4 | 0 | 0.000 | **0.000** |
| `sahte_akraba` | 4 | 0 | 0.000 | **0.000** |
| `alinti_tuzagi` | 5 | 0 | 0.000 | **0.000** |
| `eşadlı` | 3 | 3 | 1.000 | **0.000** |

### İstatistiksel durum

Her koşulun altındaki satır, motor ile en iyi trivial taban çizgisi
arasındaki farkın eşleşmiş permütasyon ve McNemar testiyle sınanmış
sonucunu verir. **Bootstrap güven aralığı sıfırı içeriyorsa fark
anlamlı değildir** ve öyle raporlanır — bu, motorun kötü olduğunu
değil, farkın henüz kanıtlanmadığını söyler.
