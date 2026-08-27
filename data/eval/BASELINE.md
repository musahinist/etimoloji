# Taban çizgisi ölçümü

Bu dosya `make eval-baseline` tarafından **otomatik üretilir** — elle
düzenlemeyin. Her sayı, adı geçen veri kümesi sürümünden sıfırdan
hesaplanır.

- **Veri kümesi:** `savelyevturkic` `v2.1` (commit `4a540590580f`)
- **Ölçüm:** 2026-08-27T00:04:25+00:00
- **Bölüm:** `all`
- **Altın standart:** 400 madde · train 237 / dev 83 / test 80
- **Kavram sızıntısı:** 0 (0 olmalı)
- **Ata düğüm:** PT 115 · PCT 285 — Çuvaşça tanığı olmayan kümede iddia edilebilecek en derin düğüm Ana Ortak Türkçe'dir

## `tum_veri_capa_haric` — n=400

> DÜRÜST KOŞUL — tüm altın standart, çapa dilinin tanığı girdiden çıkarılmış

| Sistem | tam | kabul edilebilir | ED | NED | FER | kapsam |
|---|---|---|---|---|---|---|
| **comparative** | 0.092 | 0.193 | 1.71 | 0.340 | 0.438 | 0.733 |
| copy_anchor | 0.120 | 0.210 | 2.52 | 0.490 | 0.436 | 1.000 |
| copy_random_daughter | 0.060 | 0.122 | 3.02 | 0.567 | 0.510 | 1.000 |
| copy_longest | 0.040 | 0.068 | 4.05 | 0.624 | 0.608 | 1.000 |
| majority_character | 0.077 | 0.168 | 2.77 | 0.520 | 0.462 | 1.000 |

## `tum_veri_capa_dahil` — n=400

> çapa dili girdide bırakılmış — motor kendi sorusunu tanık olarak görüyor

| Sistem | tam | kabul edilebilir | ED | NED | FER | kapsam |
|---|---|---|---|---|---|---|
| **comparative** | 0.128 | 0.255 | 2.34 | 0.461 | 0.431 | 0.998 |
| copy_anchor | 0.120 | 0.210 | 2.52 | 0.490 | 0.436 | 1.000 |
| copy_random_daughter | 0.077 | 0.158 | 2.81 | 0.536 | 0.479 | 1.000 |
| copy_longest | 0.033 | 0.068 | 4.12 | 0.631 | 0.613 | 1.000 |
| majority_character | 0.113 | 0.225 | 2.59 | 0.497 | 0.437 | 1.000 |

## `15_tanik_capa_haric` — n=158

> yalnız 15+ tanıklı kolay altküme, çapa çıkarılmış

| Sistem | tam | kabul edilebilir | ED | NED | FER | kapsam |
|---|---|---|---|---|---|---|
| **comparative** | 0.158 | 0.329 | 1.85 | 0.429 | 0.402 | 0.994 |
| copy_anchor | 0.177 | 0.304 | 1.93 | 0.432 | 0.376 | 1.000 |
| copy_random_daughter | 0.051 | 0.127 | 2.84 | 0.590 | 0.528 | 1.000 |
| copy_longest | 0.019 | 0.032 | 4.75 | 0.697 | 0.712 | 1.000 |
| majority_character | 0.101 | 0.240 | 2.38 | 0.499 | 0.425 | 1.000 |

## `15_tanik_capa_dahil` — n=158

> kolay altküme + çapa sızıntısı — ÖN ÖLÇÜMÜN koşuluna en yakın hâli

| Sistem | tam | kabul edilebilir | ED | NED | FER | kapsam |
|---|---|---|---|---|---|---|
| **comparative** | 0.183 | 0.373 | 1.80 | 0.409 | 0.392 | 1.000 |
| copy_anchor | 0.177 | 0.304 | 1.93 | 0.432 | 0.376 | 1.000 |
| copy_random_daughter | 0.114 | 0.209 | 2.50 | 0.532 | 0.473 | 1.000 |
| copy_longest | 0.019 | 0.038 | 4.67 | 0.684 | 0.705 | 1.000 |
| majority_character | 0.171 | 0.348 | 2.16 | 0.451 | 0.388 | 1.000 |

## Yorum

Dürüst koşulda motor **%9.2 tam** (%19.2 kabul edilebilir) alıyor.

Aynı motor, kolay altküme seçilip çapa sızıntısı bırakıldığında **%18.4 tam** (%37.3) gösteriyor. Aradaki fark yöntemden değil, ölçüm kurgusundan geliyor — bu yüzden raporlanan sayı daima dürüst koşulun sayısıdır.

Motorun aşması gereken çıta, en iyi trivial taban çizgisidir: `copy_anchor` %12.0, `copy_random_daughter` %6.0, `copy_longest` %4.0, `majority_character` %7.8.
