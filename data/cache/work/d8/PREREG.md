# Ön-kayıt — D8: birleştiricide zor negatifler (M9) ve PRED mesafesi (M8), R4'te bir kez

Yazıldı: 2026-09-26, R4 bölümü HİÇ açılmadan (R4'te hiçbir motor sinyali
yakalanmadı). Plan: `research/PLAN.md` D8; yöntemler `research/METHODS.md`
M8 (Mi ve ark. 2018, PRED) ve M9 (Nath ve ark. 2022, zor negatifler).
X1 bölümleri (tune/R1/R2) ve R3 GÖRÜLDÜ; test mühürlü ve açılmaz.

## 1. Tanı (ayar, kat dışı, n=1.379; betimsel)

Taban (`sca`, X1 üretimi; yeni yakalama `signals_tune_d8.jsonl`): F 0,7611,
kesinlik 0,691, duyarlılık 0,847; **miras özgüllüğü 0,633** (miras 700'ün
257'si alıntı denmiş). Bu 257 yanlış pozitifin 145'inde verici yakınlığı
rampada (0 < güç < 1), 107'sinde yakın (güç 1), 64'ünde fonotaktik ihlal;
mirasta rampa ateşlenmesi %55,7.

## 2. R4 bölümü (`data/gold/xturkic/r4.jsonl`, `SEAL_r4.json`, `stats_r4.json`)

Kod: `engine/evaluation/xturkic_gold.py` `build-r4` (X1 ve R3 dosyaları ve
mühürleri DEĞİŞMEZ). Havuz = R3 kurulumunda seçilmeyen maddeler (R3
havuzunun artığı) + X1/R3'te hiç kullanılmamış dört çağdaş Türk dili
(kaikki en): **atv** (Kuzey Altayca), **cv** (Çuvaşça), **klj** (Halaçça),
**slq** (Salarca). Etiket kuralı ve süzgeçler R3'ünkiyle aynı
(`extract_candidates_r3`: güçlendirilmiş melez türetme süzgeci + "Or from"
çekincesi). **Etimon sızıntısı yok:** X1'in (test DAHİL; bellekte yeniden
hesaplanır, sha256 mühürle aynı) ve R3'ün etimon grupları dışlanır.
Tarihî diller (ota, chg, otk, oui) alınmaz; tyv (X1 Q4'te düştü) alınmaz.
Dengeleme `balance_r3`: dil × sınıf ≤ 350, Rusça ≤ %50, Moğolcanın hepsi,
hücre ≥ 20.

**Q4 (yeni diller, LLM ön-etiketi — Claude, D8 ajanı):** R4 havuzundan dil ×
sınıf başına 20 madde, motor çıktısı görülmeden, kanıt kartıyla
(`d8/audit_cards_r4.txt`, `d8/audit_verdicts_r4.csv`, `d8/Q4_r4.json`;
ölçüt sınıf doğruluğu, kesin etimon değil). 6 hücre 20/20 kabul;
**slq/alıntı 17/20** (3 belirsiz: siru, dumçuq, yogo) ve **cv/alıntı 18/20**
(Kumanca = Türk dili verici; zayıf Moğolca kaynak) → Wilson alt < 0,80,
hücreler MÜHÜRDEN ÖNCE çıkarıldı (cv/alıntı zaten Rusça payı sınırıyla
< 20'ye düşüyordu). R3'ün dışladığı alt/alıntı burada da dışlı. Kullanıcı
≥%10'u bağımsız denetlemeli.

**Boyut (mühürlü): n = 1.588**, 1.454 etimon grubu; **alıntı 1.350 / miras
238**; yeni dil 443, R3 artığı 1.145. Verici: ar 468, ru 386, fa 384,
diğer 74, mn 38.

| dil | alıntı | miras | | dil | alıntı | miras |
|---|---|---|---|---|---|---|
| az | 350 | 0 | | atv | 45 | 69 |
| uz | 243 | 0 | | klj | 160 | 34 |
| ug | 242 | 0 | | cv | 0 | 51 |
| kk | 198 | 0 | | slq | 0 | 84 |
| crh | 112 | 0 | | | | |

**Önceden yazılır — bileşim:** etimon sızıntısı dışlaması mirası tüketiyor
(ortak Proto-Türkçe etimonlar X1/R3'te); R4'teki mirasın hepsi dört yeni
dilden. Miras payı %15 (ayarda %51). F bu bileşimde duyarlılığa çok,
mirastaki yanlış pozitife az duyarlıdır; bu, özgüllüğü artırıp duyarlılığı
düşüren bir adayın F ölçütünü tutturmasını ZORLAŞTIRIR. Ölçüt görevde
verildiği gibi uygulanır ve DEĞİŞTİRİLMEZ; doğal oranla ağırlıklı doğruluk
yalnız bilgi olarak raporlanır. Yeni dillerde fonotaktik dizilim modeli
yoktur (alan kayması taban ve adaylarda aynı).

## 3. Güç (MDE) — `d8/MDE_r4.json`

Ayarda (kat dışı) aday − taban F farkının SE'si, R4 bileşimine göre sınıf
tabakalı bootstrap (alıntı 1.350 / miras 238). Sınır 300 (n=1.538) ve 350
(n=1.588) karşılaştırıldı; 350 seçildi (R3 kuralı, n ≥ 1.500). **MDE:
C1 0,0134, C2 0,0140** (α=0,05 iki yönlü, güç 0,80); Holm ilk adımında
(α/2) ≈ 0,015.

**Önceden yazılır — beklenti:** R4 bileşiminde ayardan izdüşürülen fark
C1 **−0,036** (özgüllük artar, duyarlılık düşer), C2 **+0,012** (duyarlılık
artar, özgüllük DÜŞER: ayarda 0,633 → 0,594). İki adayın da birleşik
ölçütü (F VE özgüllük) tutturması BEKLENMİYOR: M9 ile M8 aynı eksende
zıt yönde çalışıyor.

## 4. Adaylar (en fazla 2; Holm)

Taban: `sca` (X1 üretimi; `apply_variant("sca")`: 0,35/0,60, temizlik/rampa
şans/anlam süzgeci kapalı), zor negatif kapalı (`HARD_NEG_OFF`, bayrak
`ETY_COMBINER_HARD_NEG` ölçümü etkilemez).

**C1 — M9 zor negatifler, etiket `d8@hn2:ramp:thr`.** Birleştirici (lojistik
regresyon) eğitiminde miras AMA verici yakınlığı rampada (0 < güç < 1)
olan maddelerin kayıp ağırlığı 2; karar eşiği de aynı ağırlıklı eğitim
verisinde (F hedefli) seçilir. Sinyaller tabanla AYNI önbellekten
(`signals_*_d8.jsonl`); yalnız eğitim değişir.
`borrowing_combiner.fit(hard_negative=(2.0, "ramp", True))`.
*Ayar seçimi (yalnız tune, kat dışı; `d8/tune_select.json`):* ızgara
ağırlık {2, 3, 5} × kural {ramp, ramp_phon (rampa ∨ fonotaktik ihlal)} ×
eşik {ağırlıksız, ağırlıklı}. Ağırlıksız eşikle hiçbir yapılandırma
özgüllüğü değiştirmedi (257 → 256–271 YP: eşik ağırlığı geri alıyor);
ağırlıklı eşikle özgüllük 0,75–0,85. Kural: mirasta YP'yi azaltanlar
(McNemar tek yönlü p < 0,05) arasında en yüksek ayar F'si → **hn2:ramp:thr**
(F 0,7616 vs 0,7611; özgüllük 0,633 → 0,753; ΔF +0,0005 [−0,016, +0,018]).

**C2 — M8 PRED, etiket `d8pred`.** Verici yakınlığı gücünün mesafesi
`pred_distance` (`donor_proximity.STRENGTH_DISTANCE = "pred"`): alıcı
kelimenin sonundan en çok 4 harf cezasız kırpılır (gövde ≥ 3 harf ve ≥
verici uzunluğu), en küçük SCA mesafesi alınır; eşikler SCA'nınki
(0,35/0,60); şans denetimi kontrol kelimeleri de aynı mesafeyi görür.
Parametre ayarlanmadı (4 = tipik tek-çift ek uzunluğu, önceden sabit).
*Kök üzerinden mesafe (`derivation.py` 71a6ae4, `root_variants.py`)
UYGULANMADI:* ikisi Zemberek Türkçe sözlüğüne bağlı; R4 dillerinde
(az, kk, uz, ug, crh, atv, klj…) kök çıkaramaz. Dile bağımsız sondan kırpma
PRED'in (Mi 2018) kendisidir. Ayar: F 0,7622 (ΔF +0,001 [−0,016, +0,020]),
özgüllük 0,594.

## 5. R4 protokolü (BİR KEZ)

```
export ETY_LEXICON_INDEX=$PWD/data/cache/work/xtr/index_blind.db
P=data/cache/work/d8/PREREG.md
python -m engine.evaluation.xborrowing_eval capture --split r4 --variant sca --tag d8 --prereg $P
python -m engine.evaluation.xborrowing_eval capture --split r4 --variant pred --tag d8pred --prereg $P
python -m engine.evaluation.xborrowing_eval compare --split r4 --prereg $P --base-tag d8 --tags "d8@hn2:ramp:thr" d8pred
```
Her aday fonotaktik model + birleştiricisini KENDİ ayar önbelleğinin
tamamında BELLEKTE eğitir (K7), R4'te ölçülür. Kör indeks + zincir kapalı.
Kilit `data/cache/work/xtr/.opened_r4_PREREG`.

## 6. Kabul ölçütü (her aday için; hepsi birden)

1. **Birincil:** `engine_trained` F(aday) − F(taban), etimona göre kümelenmiş
   eşleşmiş bootstrap (2.000, tohum 20260925), tek yönlü p **Holm**
   düzeltmeli (iki aday) < 0,025 (≡ %95 GA alt ucu > 0).
2. **Özgüllük:** R4'te `engine_trained` miras özgüllüğü artar — mirasta YP
   azalır, kesin McNemar tek yönlü p < 0,05 (`fp_inherited_mcnemar`).
3. **Koruma** (görülmüş veri; gerekli şart): `d8/run_beval_d8.py <aday>`
   (make eval-borrowing eşdeğeri, izlenen dosyalara yazmaz): WOLD
   `engine_trained` F ≥ 0,6482 VE Türkçe `engine_trained` F ≥ 0,8773;
   arama yolu `d8/run_replay_d8.py <aday>` (`ranker/replay.py new dev`
   eşdeğeri, adayın birleştiricisiyle) bayrak kapalıyken uyum ≥ 129/135.

İkisi de geçerse Holm p'si küçük olan seçilir. **Kabul →** C1:
`borrowing_combiner.HARD_NEGATIVE_WEIGHT = 2.0`, `HARD_NEGATIVE_RULE =
"ramp"`, `HARD_NEGATIVE_THRESHOLD = True`; C2: `donor_proximity.STRENGTH_DISTANCE
= "pred"`; `make eval-borrowing` ile üretim birleştiricisi
(`data/models/borrowing_combiner.json`) yeniden eğitilir, commit.
**Red →** varsayılanlar değişmez, model dosyası değişmez; olumsuz sonuç
`borrowing_combiner.py` (C1) ve `donor_proximity.py` (C2) docstring'lerine.

Ek raporlanır (karar değil): alt kümeler yeni dil / R3 artığı, dil ve
verici kırılımları, verici tanıma, doğal oranla ağırlıklı doğruluk,
mirasta rampa oranı (C2).
