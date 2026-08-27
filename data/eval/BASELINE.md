# Taban çizgisi ölçümü

Bu dosya `make eval-baseline` tarafından **otomatik üretilir** — elle
düzenlemeyin. Her sayı, adı geçen veri kümesi sürümünden sıfırdan
hesaplanır.

- **Veri kümesi:** `savelyevturkic` `v2.1` (commit `4a540590580f`)
- **Ölçüm:** 2026-08-27T01:21:21+00:00
- **Bölüm:** `all`
- **Altın standart:** 400 madde · train 237 / dev 83 / test 80
- **Kavram sızıntısı:** 0 (0 olmalı)
- **Ata düğüm:** PT 115 · PCT 285 — Çuvaşça tanığı olmayan kümede iddia edilebilecek en derin düğüm Ana Ortak Türkçe'dir

## `tum_veri_capa_haric` — n=400

> DÜRÜST KOŞUL — tüm altın standart, çapa dilinin tanığı girdiden çıkarılmış

| Sistem | tam | kabul edilebilir | ED | NED | FER | kapsam |
|---|---|---|---|---|---|---|
| **comparative** | 0.215 | 0.292 | 1.79 | 0.346 | 0.361 | 0.920 |
| copy_anchor | 0.203 | 0.268 | 2.13 | 0.417 | 0.392 | 1.000 |
| copy_random_daughter | 0.168 | 0.242 | 2.29 | 0.453 | 0.414 | 1.000 |
| copy_longest | 0.100 | 0.147 | 3.04 | 0.552 | 0.541 | 1.000 |
| majority_character | 0.237 | 0.330 | 1.94 | 0.382 | 0.363 | 1.000 |

> `comparative` vs `majority_character`: fark **-0.0225**, %95 GA [-0.0500, +0.0050], permütasyon p=0.172, McNemar p=0.175 → anlamlı DEĞİL — güven aralığı sıfırı içeriyor.

## `tum_veri_capa_dahil` — n=400

> çapa dili girdide bırakılmış — motor kendi sorusunu tanık olarak görüyor

| Sistem | tam | kabul edilebilir | ED | NED | FER | kapsam |
|---|---|---|---|---|---|---|
| **comparative** | 0.230 | 0.315 | 1.86 | 0.362 | 0.356 | 0.970 |
| copy_anchor | 0.203 | 0.268 | 2.13 | 0.417 | 0.392 | 1.000 |
| copy_random_daughter | 0.158 | 0.233 | 2.30 | 0.457 | 0.422 | 1.000 |
| copy_longest | 0.080 | 0.133 | 3.11 | 0.560 | 0.550 | 1.000 |
| majority_character | 0.223 | 0.307 | 2.02 | 0.395 | 0.368 | 1.000 |

> `comparative` vs `majority_character`: fark **+0.0075**, %95 GA [-0.0225, +0.0375], permütasyon p=0.742, McNemar p=0.743 → anlamlı DEĞİL — güven aralığı sıfırı içeriyor.

## `15_tanik_capa_haric` — n=158

> yalnız 15+ tanıklı kolay altküme, çapa çıkarılmış

| Sistem | tam | kabul edilebilir | ED | NED | FER | kapsam |
|---|---|---|---|---|---|---|
| **comparative** | 0.279 | 0.418 | 1.43 | 0.329 | 0.334 | 0.981 |
| copy_anchor | 0.260 | 0.335 | 1.62 | 0.365 | 0.351 | 1.000 |
| copy_random_daughter | 0.165 | 0.272 | 1.97 | 0.446 | 0.399 | 1.000 |
| copy_longest | 0.038 | 0.089 | 3.35 | 0.632 | 0.650 | 1.000 |
| majority_character | 0.285 | 0.418 | 1.49 | 0.340 | 0.330 | 1.000 |

> `comparative` vs `majority_character`: fark **-0.0063**, %95 GA [-0.0506, +0.0380], permütasyon p=1.000, McNemar p=1.000 → anlamlı DEĞİL — güven aralığı sıfırı içeriyor.

## `15_tanik_capa_dahil` — n=158

> kolay altküme + çapa sızıntısı — ÖN ÖLÇÜMÜN koşuluna en yakın hâli

| Sistem | tam | kabul edilebilir | ED | NED | FER | kapsam |
|---|---|---|---|---|---|---|
| **comparative** | 0.279 | 0.418 | 1.42 | 0.327 | 0.331 | 0.981 |
| copy_anchor | 0.260 | 0.335 | 1.62 | 0.365 | 0.351 | 1.000 |
| copy_random_daughter | 0.158 | 0.266 | 1.99 | 0.460 | 0.418 | 1.000 |
| copy_longest | 0.038 | 0.089 | 3.34 | 0.628 | 0.652 | 1.000 |
| majority_character | 0.285 | 0.418 | 1.49 | 0.341 | 0.330 | 1.000 |

> `comparative` vs `majority_character`: fark **-0.0063**, %95 GA [-0.0443, +0.0316], permütasyon p=1.000, McNemar p=1.000 → anlamlı DEĞİL — güven aralığı sıfırı içeriyor.

## Yorum

Dürüst koşulda motor **%21.5 tam** (%29.2 kabul edilebilir) alıyor.

Aynı motor, kolay altküme seçilip çapa sızıntısı bırakıldığında **%27.9 tam** (%41.8) gösteriyor. Aradaki fark yöntemden değil, ölçüm kurgusundan geliyor — bu yüzden raporlanan sayı daima dürüst koşulun sayısıdır.

Motorun aşması gereken çıta, en iyi trivial taban çizgisidir: `copy_anchor` %20.2, `copy_random_daughter` %16.8, `copy_longest` %10.0, `majority_character` %23.8.

### Negatif kontroller

Yüksek doğruluk, yüksek yanlış-pozitif oranıyla birlikte anlamsızdır.
**Güçlü iddia oranı** en kritik sütundur: motorun uydurma veya alıntı
bir kelimeye 🟢/🟡 rozet verme oranı sıfır olmalıdır.

| Batarya | n | rekonstrükte | yanlış-pozitif | güçlü iddia |
|---|---|---|---|---|
| `fonotaktik_gecerli_sahte` | 8 | 8 | 1.000 | **0.000** |
| `bariz_sahte` | 4 | 0 | 0.000 | **0.000** |
| `sahte_akraba` | 4 | 0 | 0.000 | **0.000** |
| `alinti_tuzagi` | 5 | 3 | 0.600 | **0.000** |
| `eşadlı` | 3 | 3 | 1.000 | **0.000** |

### İstatistiksel durum

Her koşulun altındaki satır, motor ile en iyi trivial taban çizgisi
arasındaki farkın eşleşmiş permütasyon ve McNemar testiyle sınanmış
sonucunu verir. **Bootstrap güven aralığı sıfırı içeriyorsa fark
anlamlı değildir** ve öyle raporlanır — bu, motorun kötü olduğunu
değil, farkın henüz kanıtlanmadığını söyler.
