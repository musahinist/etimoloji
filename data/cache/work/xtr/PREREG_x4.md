# Ön-kayıt — X4: verici havuzu temizliği + rampa şans denetimi

Yazıldı: 2026-09-25, R2 bölümü HİÇ açılmadan (R1, X2/X3'te görüldü; bu
ön-kayıt R1'i kullanmaz). Dayanak: X3 tanısı (`PREREG_sense.md` §1). Ayar
bölümünde yalnız DOĞRULAMA yapıldı; hiçbir parametre seçilmedi.

## Adaylar (2; Holm, m=2)

Kod: `engine/db/donor_index.py` (`ETY_DONOR_CLEAN`), `engine/nlp/donor_proximity.py`
(`ETY_DONOR_RAMP_CHANCE`); ölçüm çeşitleri `xborrowing_eval.apply_variant("x4a1"|"x4a2")`.

- **A1 = temizlik paketi (a)+(b)+(c), birlikte:**
  (a) `by_sense` eşleşme anahtarı yalnız İÇERİK sözcükleri: 2 harften uzun,
  sabit `FUNCTION_WORDS` listesi (İngilizce işlev sözcükleri + tanıdaki
  genel sözcükler: make, one, thing, person, something …) dışı, ilk 6.
  Sorgu yalnız işlev sözcüklerinden oluşuyorsa ("all", "one", "how much")
  eski sözcükler kullanılır. Aynı anahtar monget ve kavram havuzlarında.
  (b) Anlamı YALNIZ dilbilgisi göndermesi olan verici maddeleri havuzdan
  çıkar (`is_form_of`: en az bir parça "<dilbilgisi sözcükleri> of X", öbür
  parçalar da yalnız dilbilgisi sözcükleri; ":" sonrası gerçek anlam
  taşıyanlar — "diminutive of нос: (little) nose" — kalır). Lemmaya
  yönlendirme YOK (lemma maddesi havuzda zaten ayrıca var).
  (c) Adaylar FTS `bm25` sırasıyla (`ORDER BY rank`), süzgeç öncesi en çok
  4.000 satır (`CLEAN_FETCH`, hız sınırı — sıralı olduğu için en ilgili
  4.000), süzgeçten sonra ilk 200 (sınır aynı, artık sıralı).
- **A2 = A1 + rampa şans denetimi (d):** eşik ile tavan arasındaki (0,35–0,60)
  en yakın madde yalnız mesafesi o maddenin DİLİNİN null'ından küçükse güç
  alır. Null = 1facc40'taki verici etiketi null'ı (`_null_distance`: aynı
  uzunlukta 12 Türkçe kontrol kelimesinin o dilin süzülmüş havuzuna medyan
  SCA uzaklığı). Parametre yok.

Değişmeyen: SCA, eşik/tavan, eşik altı şans denetimi (%10 yüzdelik),
verici etiketi kuralı, birleştirici özellikleri.

## Ayar doğrulaması (kat dışı, n=1.379; betimsel, seçim YOK)

| | F | doğr. | dp kes. | dp duy. | verici tanıma | rampa miras | rampa alıntı |
|---|---|---|---|---|---|---|---|
| taban | 0,7611 | 0,7382 | 0,7978 | 0,7320 | 0,5499 | 0,5571 | 0,1694 |
| A1 | 0,7607 | 0,7404 | 0,8139 | 0,7408 | 0,5576 | 0,5829 | 0,1620 |
| A2 | 0,7672 | 0,7527 | 0,8139 | 0,7408 | 0,5568 | 0,4214 | 0,1252 |

ΔF: A1 −0,000 [−0,010, +0,009]; A2 +0,006 [−0,006, +0,018]. Temiz havuz
medyanı 53 (önce 57); 200'e dayanan sorgu 251 (önce 290, artık sıralı).
Mirasta rampa: A1'de AZALMIYOR (0,557 -> 0,583; havuz küçülünce en yakın
madde rampaya kayıyor), A2'de azalıyor (-> 0,421, McNemar p<0,001).
**Önceden yazılır:** A1'in mekanizma ölçütünü geçmesi beklenmiyor; birincil
etkiler MDE'nin (0,038–0,053) çok altında → güç YETERSİZ, ret "belirsiz".

## R2 protokolü (BİR KEZ)

```
ETY_LEXICON_INDEX=$PWD/data/cache/work/xtr/index_blind.db python -m engine.evaluation.xborrowing_eval capture --split r2 --prereg data/cache/work/xtr/PREREG_x4.md
… capture --split r2 --variant x4a1 --tag x4a1 --prereg …
… capture --split r2 --variant x4a2 --tag x4a2 --prereg …
python -m engine.evaluation.xborrowing_eval compare --split r2 --prereg data/cache/work/xtr/PREREG_x4.md --tags x4a1 x4a2
```
Her çeşit kendi ayar önbelleğinin tamamında bellekte eğitilir. Kilit
`.opened_r2_x4`. (Taban ayar önbelleği `signals_tune.jsonl`; A1/A2
`signals_tune_x4a1/x4a2.jsonl`.)

## Kabul ölçütü (aday başına; hepsi birden)

1. **Birincil:** `engine_trained` F(aday) − F(taban), kümelenmiş eşleşmiş
   bootstrap (2.000, tohum 20260925); tek yönlü p Holm (m=2) < 0,025
   (≡ Holm'lu %95 GA alt ucu > 0).
2. **Mekanizma:** mirasta rampa ateşlenmesi azalır (kesin McNemar, tek
   yönlü p < 0,05).
3. **Koruma** (yalnız 1–2'yi geçen aday için): `x3/run_beval_x4.py <aday>`
   (make eval-borrowing eşdeğeri, izlenen dosyalara yazmaz) WOLD
   `engine_trained` F ≥ 0,6482 ve Türkçe `engine_trained` F ≥ 0,8773.

İkisi de geçerse A2 (A1'i kapsar) alınır. Kabul -> veri düzeltmesi olduğu
için VARSAYILAN AÇIK: `donor_index.CLEAN_DEFAULT = True` (+ A2 ise
`donor_proximity.RAMP_CHANCE_DEFAULT = True`); bayraklarla kapatılabilir.
donors.db yeniden kurulmaz (süzgeç sorguda). Hiçbiri geçmezse varsayılanlar
kapalı kalır, kod ölçüm için durur.

**Arama yolu bayrağı** `ETY_SEARCH_DONOR_PROXIMITY` ayrı karar, yalnız
koruma: kabul edilen ayarla `data/cache/work/ranker/replay.py new dev`
bayrak açıkken uyum ≥ 129/135; tutmazsa bayrak kapalı kalır (üretimde
bugün de kapalı).

Ek raporlanır: dp kesinliği, verici tanıma, dil/verici kırılımları.
