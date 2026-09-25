# Ön-kayıt — X2: SCA+PMI ortalaması alıntı gücünde, Türk dilleri arası altında

Yazıldı: 2026-09-25, R1 bölümü HİÇ açılmadan, `mean` çeşidinin ayar
yakalaması da yapılmadan önce. Plan: `data/cache/work/research/PLAN_XTURKIC.md`
İŞ 3. Önceki ön-kayıt (WOLD/Türkçe, olumsuz): `data/cache/work/donor2/PREREG.md`.

## Soru
`donor_proximity` alıntı GÜCÜ adımında SCA yerine `(SCA + min(1, PMI)) / 2`
(ASJP-PMI, Jäger 2018) kullanmak, bağımsız Türk dilleri arası altında
(kaikki en şablonları; kk ky tt ba uz ug tk + tyv) birleştiricinin F'sini
artırır mı?

## Tek aday — yeniden ayar YOK
- Aday `mean`: `STRENGTH_DISTANCE = "mean"`, `DONOR_DISTANCE_THRESHOLD = 0.60`,
  `DONOR_DISTANCE_CEILING = 0.85` — donor2/PREREG.md'de WOLD ayar yarısında
  seçilmiş değerler, AYNEN. Şans denetimi de aynı mesafeyle (üretim kodu).
- Taban `sca`: üretim (`sca`, 0,35 / 0,60).
- Verici ETİKETİ adımı iki çeşitte de üretimde (`LABEL_DISTANCE = "sca"`).
- Tek aday: çoklu karşılaştırma düzeltmesi yok.

## Protokol
- Ana ölçüm: KÖR indeks (`data/cache/work/xtr/index_blind.db`) + zincir kapalı.
- Çeşit süreç içinde uygulanır: `xborrowing_eval.apply_variant("mean")`.
- Yakalamalar (`data/cache/work/heavy.sh` ile, tek ağır süreç):
  ```
  ETY_LEXICON_INDEX=$PWD/data/cache/work/xtr/index_blind.db python -m engine.evaluation.xborrowing_eval capture --split tune --variant mean --tag mean
  ETY_LEXICON_INDEX=... capture --split r1 --prereg data/cache/work/xtr/PREREG_pmi.md
  ETY_LEXICON_INDEX=... capture --split r1 --variant mean --tag mean --prereg data/cache/work/xtr/PREREG_pmi.md
  ```
  (taban ayar önbelleği `signals_tune.jsonl` X1'de alındı, commit fae680c.)
- Ayar yakalaması yalnız betimseldir: hiçbir parametre ondan seçilmez.
- R1 BİR KEZ:
  ```
  python -m engine.evaluation.xborrowing_eval compare --split r1 --prereg data/cache/work/xtr/PREREG_pmi.md --tags mean
  ```
  Her çeşidin fonotaktik dizilim modeli + birleştiricisi KENDİ ayar
  önbelleğinin tamamında BELLEKTE eğitilir (LM A yarısı, birleştirici B
  yarısı; K7: `save` kapalı) ve R1'de ölçülür. Kilit
  `data/cache/work/xtr/.opened_r1_pmi`.

## Kabul ölçütü (R1, üçü birden)
1. **Birincil:** `engine_trained` F(mean) − F(sca), etimona göre kümelenmiş
   eşleşmiş bootstrap (2.000 örnek, tohum 20260925): %95 GA alt ucu > 0.
2. `engine_trained` doğruluk farkı (mean − sca) ≥ 0 (nokta tahmini).
3. Verici tanıma (`donor_identification`, engine_trained): doğruluk
   mean ≥ sca − 0,02.

Geçerse: `STRENGTH_DISTANCE = "mean"` (+ eşikler) üretime ayrı commit'le;
R2'de birleşik doğrulama bir kez (X3 kabul edilirse onunla birlikte). Geçmezse
C1 (PMI gücü) kapanır; üretim değişmez.

## Güç — ÖNCEDEN yazılır: GÜÇ YETERSİZ
`MDE.json` (ayar SE'si n=684'e ölçeklenmiş, α=0,05 iki yönlü, güç 0,80):
en yakın vekil çift için MDE F = **0,038**, diğerleri 0,053–0,055. Beklenen
etki ~0,014 F (WOLD'da mean−sca engine_trained F +0,014: 0,6582 -> 0,6721).
MDE beklenen etkinin ~2,7 katı; bu etkiyi yakalama gücü ~%20'nin altında.
Bu yüzden: ölçüt 1 tutmazsa sonuç **"belirsiz (güç yetersiz)"** diye
raporlanır, "etki yok" diye DEĞİL. C1 yine de kapanır (üretim değişmez), ama
kayda "etkisizliği gösterilmedi" diye geçer. Nokta tahmini ve GA raporlanır.

## Ek raporlanır (karar ölçütü değil)
`donor_proximity_only` kesinlik/F, rampa ateşlenme oranı (miras/alıntı),
dil ve verici kırılımları, kolay/zor duyarlılık, miras özgüllüğü.
