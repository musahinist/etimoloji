# 9c — Alıntı engeli yanlış pozitifleri: ön kayıt

Tarih: 2026-09-26. Test bölümleri OKUNMAZ (savelyev test, Türkçe altın test).

## Tanı (yaygınlık, `prevalence.py base`, TRAIN+DEV)

Engel = `BorrowingVerdict.blocks_inherited_reconstruction`
(el skoru ≥ 0,55 VEYA `_index_attests_loan`).

| küme | engellenen | skor yolu | indeks yolu |
|---|---|---|---|
| savelyev miras (n=320) | 3 (%0,9) | 1 | 3 |
| Türkçe altın miras (n=239) | 9 (%3,8) | 2 | 9 |
| Türkçe altın alıntı (n=334) | 308 (%92,2) | 199 | 308 |

Engellenen miras maddelerinin HEPSİ indeks tanıklığı yolundan geçiyor;
skor yolu (BLOCK_THRESHOLD) tek başına hiçbirini engellemiyor. Sinyal: her
birinde `zincir_kanıtı` (3'ünde + `fonotaktik_ihlal`). Maddeler ve indeks
satırının köken metni:

- Türkçe malzemeli türetme, verici yalnız MODEL: `aday` (savelyev çapası;
  "Coined by TDK", fr *nominé*), `örgüt` ("By surface analysis ör- + -güt",
  fr), `kurmay` ("Coined by TDK from kur", fr), `kumul` ("From kum + -ul …
  phono-semantic", fr), `denizaltı` ("Calque of French sous-marin").
- Verici bir Türk dili: `bakşı`, `diremek` (donor `oui`, Eski Uygurca —
  `TURKIC_LINEAGE_CODES`'ta yok).
- Wiktionary gerçekten alıntı diyor (altınla kaynak çatışması): `hem` (fa),
  `pin` (hy), `hedik` (hy), `yas` (ar), `amca` (ar).

Eğitilmiş birleştirici olasılığına bağlama (sıralayıcı 396affe gibi) elendi:
indeks yolunu p ≥ 0,39 ile koşullamak Türkçe miras engelini 9 → 4'e indirir
ama Türkçe alıntı engelini 308 → 276'ya düşürür (−32, %10) — ölçüt K2'yi
açıkça bozar. Tanıklı miras kökü kuralı: engellenen 12 mirasın yalnız 1'inde
(`diremek`) yakalanmış tanıklı kök var; ComparativeReconstructor'da kök
bilgisi de yok.

## Tek aday (C1)

`_index_attests_loan` yalnız engelleme kararında şu indeks satırlarını ALINTI
TANIKLIĞI saymaz (zincir sinyali, el skoru, birleştirici DEĞİŞMEZ):

1. Vericisi bir Türk dili olan satır (`TURKIC_LINEAGE_CODES` ∪
   `TURKIC_LANGUAGES_MAP` ∪ {oui, xqa, chg, ota, otk}).
2. Köken metni Türkçe malzemeli türetmeyi söyleyen satır: `Calque of`,
   `Coined by`, `By surface analysis`, `Semantic loan`, `phono-semantic`
   (büyük/küçük harf duyarsız; metnin başında ya da içinde).

Parametre yok; kalıplar Wiktionary köken şablonlarının sabit ifadeleri.

## Ölçütler (hepsi tutmalı; aksi hâlde aday reddedilir, kod değişmez)

- K1: miras maddelerde (savelyev + Türkçe altın, TRAIN+DEV) engel sayısı
  AZALIR (12'den kesin küçük).
- K2: Türkçe altın alıntılarda engel en çok 3 madde (%1) düşer.
- K3: `make eval-cv` (savelyev, 5 kat) `rules` ve `column_model` NED
  artmaz, tam doğruluk düşmez (data/eval/crossval.json'a göre).
- K4: `make eval-controls` — alıntı tuzakları engellenmeye devam eder,
  sahte kök bataryası (kök yasağı a49bee6) bozulmaz: başarısız madde sayısı
  data/eval/negative_controls.json'dakinden artmaz.
- Koruma: pytest (`-n 2`) yeşil; `_index_attests_loan` testleri (gül, duvar,
  tuzaklar) aynı.

## Sonuç (C1, bir kez; 2026-09-26) — KABUL

- K1 ✓ miras engeli 12 → 6: savelyev 3 → 2 (`aday` açıldı; `hem`, `pin`
  kaldı), Türkçe miras 9 → 4 (açılan: örgüt, kurmay, kumul, bakşı, diremek;
  kalan: amca, yas, hedik, denizaltı — sonuncusu skor yolundan 0,59).
- K2 ✓ Türkçe alıntı engeli 308 → 307 (−1: `ehliyet`).
- K3 ✓ eval-cv (A2 kapalı): rules tam 0,2531 → 0,2531, NED 0,3677 → 0,3669;
  column_model tam 0,2969 → 0,2969, NED 0,3282 → 0,3274; learned_table NED
  0,3424 → 0,3417. (Ölçüt dışı: BCFS rules 0,5354 → 0,5335, column_model
  0,5566 → 0,5548.)
- K4 ✓ negative_controls batteries birebir aynı.

Kalan `hem`/`pin`/`hedik`/`yas`/`amca`: Wiktionary satırı açıkça alıntı
diyor (altın TDK/Nişanyan ile kaynak çatışması); bu adayın kapsamı dışı.
Çıktılar: prevalence_{base,c1}.json, data/cache/work/a2/out/c1/.
