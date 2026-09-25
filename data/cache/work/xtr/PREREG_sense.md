# Ön-kayıt — X3: anlam bilgili verici eşleştirmesi

Plan: `data/cache/work/research/PLAN_XTURKIC.md` İŞ 2. R1 bölümü HİÇ
açılmadan yazıldı. Bölüm 1 (tanı) adaylar tanımlanmadan ÖNCE yazıldı; yalnız
AYAR bölümü (n=1.379, kör indeks + zincir kapalı, taban önbellek
`signals_tune.jsonl`) kullanıldı. Betik: `data/cache/work/xtr/x3/diag_sense.py`
(çıktılar `x3/diag.json`, `x3/fp50.tsv`).

## 1. Tanı (ayar)

**`DonorIndex.by_sense`**: anlamın 2 harften uzun ilk 6 sözcüğü FTS `OR` ile
aranır, sıralama yok, `LIMIT 200`. Havuz medyanı 57 madde; 290/1.379 sorgu
200 sınırına dayanıyor (ilk 200 rowid sırası — anlamca en iyisi değil);
18 sorguda havuz boş. Durak sözcük süzgeci yok (`the`, `for`, `one`, `make`
eşleşmeyi sağlayan sözcükler arasında).

**Ateşlenme** (verici yakınlığı sinyali):

| | n | yakın (güç 1) | rampa (0 < güç < 1) |
|---|---|---|---|
| miras | 700 | 0,181 | **0,557** |
| alıntı | 679 | 0,732 | 0,169 |

Rampa maddelerinin alıntı payı 0,228 (taban 0,492): rampa ağırlıkla MİRAS
kelimelerde ateşleniyor.

**Eşleşmeyi sağlayan sözcük** (sorgu sözcükleri ∩ en yakın verici maddesinin
anlamı): ateşlenen 517 mirasın %97,5'inde TEK sözcük (alıntıda %86,1).
Verici anlamı çekim/türetim açıklaması ("genitive plural of …",
"diminutive of …") mirasta %30,4, alıntıda %18,6. Mirasta en sık sözcükler:
the 8, wing 5, forty 5, one 4, for 4, door 4, green 4, thirty 4, sand 4,
write 4, make 4, snake 4, nine 4 … — çoğu temel söz varlığı kavramı.

**50 yanlış pozitif elle** (miras + sinyal ateşlendi + engine_trained alıntı
dedi; havuz 252, tohum 20260925; motor kararı görülerek ama altın etiketi
dışında bilgi kullanılmadan sınıflandı):

| sınıf | n | örnek |
|---|---|---|
| anlam uyumsuz (yan anlam, genel sözcük, özel ad) | **20** | жумшоо "to make soft" ~ ar jammala "to make beautiful"; авыл "village" ~ ar Tirbil (köy adı); сүз "word" ~ ar "to coin (a word)"; оюнчук "toy" ~ ru волчок "little wolf; spinning top (toy)"; näçe "how much" ~ ar "forgetting much" |
| anlam uyumlu, biçim tesadüfü | **28** | qayiq "boat" ~ ru коч; ilon/елан "snake" ~ ar afʕā; tütün "smoke" ~ ar duxān; бир "one" ~ ar fard; gözel ~ fa ġazāl "gazelle; beautiful" |
| gerçek ortak alıntı / etiket hatası | **2** | черүү ~ mn čerig (Türk–Moğol temas); uz harorat ~ fa/ar ḥarārat (altında miras — etiket hatası) |

Sonuç: anlam süzgeci yanlış pozitiflerin en çok ~%40'ını hedefleyebilir;
çoğunluk (%56) anlamı doğru eşleşmiş ama rampada şans benzerliği. Beklenen
etki bu yüzden sınırlı; süzgecin şans denetimini küçük havuzla yeniden
kurması (null havuzu da süzülür) ikinci mekanizmadır.

## 2. Adaylar (≤3; tanıdan SONRA tanımlandı)

Kod: `engine/nlp/sense_match.py`; `donor_proximity.nearest_donor` ve
`attribute_donor`'a `sense_filter=None` (None = `ETY_DONOR_SENSE_FILTER`
bayrağı, varsayılan KAPALI). Süzgeç `by_sense`in (değişmemiş, `LIMIT 200`)
adaylarına uygulanır; en yakın madde VE şans denetiminin kontrol havuzu
süzülmüş havuzdan kurulur (null yeniden kurulur); verici etiketi adımı da
süzülmüş havuzu (kaikki + monget) kullanır. Havuz genişletilmez (saf süzgeç
etkisi).

- **S1** `s1:τ` — e5-small (`intfloat/multilingual-e5-small`, `query: `
  önekiyle, normalize) kosinüsü(sorgu anlamı, verici anlamı[:200]) ≥ τ.
- **S2** `s2:k` — çift yönlü: ileri yön `by_sense` sözcük eşleşmesi; geri
  yönde verici anlamının e5 gömmesine en yakın kavramlar arasında (başvuru
  havuzu: yerel CLDF `parameters.csv` Concepticon_Gloss'ları, 1.887 kavram;
  sorgu anlamı havuza eklenir, aynısı çıkarılır) sorgu anlamı ilk k içinde.
- **S3** `s3:strict|fallback` — tam kavram eşitliği: anlam `; , /` ile
  parçalanır, parantez/"X of Y:" öneki atılır, parçalar yerel CLDF
  (NorthEuraLex, WOLD, savelyev, robbeets, hruschka, ronatas, starostin)
  ad/Concepticon_Gloss tablosuyla Concepticon kimliklerine eşlenir; kesişim
  boş değilse tutulur. `fallback`: sorgu anlamı hiçbir kavrama eşlenmezse
  süzgeç uygulanmaz.

## 3. Ayar seçimi (yalnız ayar, kat dışı kararlar)

Kural: her aile için ayar
`engine_trained` F'si en büyük parametre. Izgara: τ ∈ {0,80, 0,82, …, 0,92},
k ∈ {3, 5, 10}, S3 ∈ {strict, fallback}. Önbellekler
`signals_tune_sense_*.jsonl`, betik `x3/grid_capture.py`, tablo
`x3/tune_table.py`.

| çeşit | F | doğr. | dp kes. | dp duy. | verici tanıma | rampa miras | rampa alıntı | ΔF [GA95] |
|---|---|---|---|---|---|---|---|---|
| taban sca | 0,7611 | 0,7382 | 0,7978 | 0,7320 | 0,5499 | 0,5571 | 0,1694 | |
| s1:0,80 | 0,7484 | 0,7157 | 0,7984 | 0,7231 | 0,5438 | 0,5614 | 0,1841 | −0,013 [−0,024, −0,003] |
| s1:0,82 | 0,7569 | 0,7302 | 0,7990 | 0,7202 | 0,5368 | 0,5557 | 0,1915 | −0,004 |
| s1:0,84 | 0,7477 | 0,7201 | 0,7987 | 0,7069 | 0,5501 | 0,5743 | 0,1929 | −0,013 |
| s1:0,86 | 0,7523 | 0,7259 | 0,8180 | 0,6951 | 0,5685 | 0,6114 | 0,1944 | −0,009 |
| **s1:0,88** | **0,7652** | 0,7418 | 0,8596 | 0,6672 | 0,5616 | 0,6271 | 0,2106 | +0,004 [−0,016, +0,024] |
| s1:0,90 | 0,7456 | 0,7194 | 0,8700 | 0,5817 | 0,5528 | 0,5271 | 0,2239 | −0,016 |
| s1:0,92 | 0,7363 | 0,7179 | 0,8781 | 0,4669 | 0,5439 | 0,3743 | 0,2091 | −0,025 |
| s2:3 | 0,7624 | 0,7469 | 0,8290 | 0,7069 | 0,5677 | 0,6143 | 0,1944 | +0,001 |
| **s2:5** | **0,7638** | 0,7368 | 0,8111 | 0,7084 | 0,5568 | 0,5929 | 0,1944 | +0,003 [−0,013, +0,018] |
| s2:10 | 0,7549 | 0,7273 | 0,8127 | 0,7158 | 0,5522 | 0,5929 | 0,1959 | −0,006 |
| s3:strict | 0,6989 | 0,6251 | 0,7252 | 0,2371 | 0,5503 | 0,4514 | 0,0692 | −0,062 |
| **s3:fallback** | **0,7753** | 0,7600 | 0,8404 | 0,7290 | 0,5678 | 0,5843 | 0,1826 | +0,014 [−0,001, +0,029] |

**Seçilen: S1 = `s1:0.88`, S2 = `s2:5`, S3 = `s3:fallback`.**

Ayar gözlemi (önceden yazılır): süzgeçler verici-yakınlığı-yalnız
KESİNLİĞİNİ artırıyor (S1 +0,062, S3 +0,043), ama mirasta RAMPA
ateşlenmesini hiçbirinde AZALTMIYOR (0,557 -> 0,59–0,63): küçük havuzda en
yakın madde uzaklaşıp eşiğin altından rampaya kayıyor. Mekanizma ölçütünün
rampa kısmının R1'de tutması beklenmiyor. Seçilen parametreler ayar F'sini
en büyütenlerdir (kazananın laneti: ayar farkları iyimser).

## 4. R1 protokolü (BİR KEZ)

Taban R1 önbelleği `signals_r1.jsonl` (PREREG_pmi ile alındı; açılmadı).
```
ETY_LEXICON_INDEX=$PWD/data/cache/work/xtr/index_blind.db python -m engine.evaluation.xborrowing_eval capture --split r1 --variant sense:s1:0.88 --tag sense_s10.88 --prereg data/cache/work/xtr/PREREG_sense.md
… --variant sense:s2:5 --tag sense_s25 …
… --variant sense:s3:fallback --tag sense_s3fallback …
python -m engine.evaluation.xborrowing_eval compare --split r1 --prereg data/cache/work/xtr/PREREG_sense.md --tags sense_s10.88 sense_s25 sense_s3fallback
```
Sıra: önce X2 (PREREG_pmi) R1 karşılaştırması, sonra bu.

## 5. Kabul ölçütü (aday başına; hepsi birden)

1. **Birincil:** `engine_trained` F(aday) − F(sca), etimona göre kümelenmiş
   eşleşmiş bootstrap (2.000, tohum 20260925), tek yönlü p (örneklerin ≤0
   payı) Holm ile (m=3) düzeltilmiş < 0,025 (≡ Holm'lu %95 GA alt ucu > 0).
2. **Mekanizma** (ikisi de, tek yönlü p < 0,05): `donor_proximity_only`
   kesinliği artar (eşleşmiş kümelenmiş bootstrap) VE mirasta rampa
   ateşlenmesi azalır (kesin McNemar).
3. **Koruma:** süzgeç açıkken `make eval-borrowing` eşdeğeri
   (`x3/run_beval_sense.py`, izlenen dosyalara yazmaz): WOLD
   `engine_trained` F ≥ 0,6482 ve Türkçe altın `engine_trained` F ≥ 0,8773
   (üretim 0,6582 / 0,8873 eksi 0,01). Yalnız 1 ve 2'yi geçen aday için koşulur.

Birden çok aday geçerse ayar F'si en yüksek olan üretime alınır.
Kabul -> `sense_match.ACCEPTED_SPEC` = o tanım, `ETY_DONOR_SENSE_FILTER`
varsayılanı açık; R2'de birleşik doğrulama bir kez (X2 de kabulse ikisi
birlikte). Hiçbiri geçmezse bayrak KAPALI kalır, kod ölçüm için durur.
Güç: MDE (n=684) F 0,038–0,053; ayar etkileri +0,003–+0,014 → güç
YETERSİZ; ret "belirsiz" diye raporlanır.

**Arama yolu bayrağı** `ETY_SEARCH_DONOR_PROXIMITY` ayrı karar: yalnız
koruma — süzgeç kabul edilirse Türkçe dev'de bayrak açıkken uyum ≥129/135
(`data/cache/work/ranker/replay.py`) aranır; kabul yoksa ölçülmez.
