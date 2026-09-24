# Türki Diller Etimoloji Araştırma Motoru

Yerel çalışan, kaynak-şeffaf bir etimoloji araştırma motoru. Bir Türkçe kelimenin
Türki dillerdeki karşılıklarını toplar, **karşılaştırmalı yöntemle** Proto-Türkçe
ata biçimini türetir, alıntı olup olmadığını sınıflandırır ve ürettiği her
hipotezi dört aşamalı bir hakem protokolünden geçirir.

Temel ilke: **kanıt yoksa puan da yok.** Motor bir sonucu ancak ölçebildiği
kanıt kadar destekler; ölçemediği aşamayı skora katmaz ve eksikliği açıkça
raporlar.

```
$ python -m engine.cli search göz

  Ana Kök / Rekonstrüksiyon : *köŕ  [*PT]
  Yöntem                    : karşılaştırmalı yöntem, 8 dil tanığı / 5 Türki kol
  Uygulanan denklikler      : g- ~ k- (Proto-Türkçe *k-)
                              Ortak Türkçe -z ~ Çuvaşça -r (Lir-Şaz rotasizmi)
  Kalibre güven             : 0,26  🟠 ZAYIF KANIT
  Rakip hipotezler          : 1. MİRAS 0,23 ✓  2. MODERN TÜRETME 0,05 ✗
                              3. ALINTI 0,00 ✗ (sözlükte alıntı kaydı yok)
```

---

## ⚠️ Ölçülmüş doğruluk

Bu bölüm reklam değil, **ölçüm**dür. Bütün sayılar `make eval-baseline` ile
yeniden üretilebilir ve `data/eval/BASELINE.md` içinde sürüm damgasıyla
saklanır.

### Nerede duruyoruz — özet

| Ölçüm | Değer | Taban çizgi | Hüküm |
|---|---|---|---|
| Alıntı F — **Türkçe** (TDK+Nişanyan, n=349) | **0,889** | 0,749 hepsi-alıntı | ✅ **anlamlı** (+0,272 doğruluk, p=0,0001) |
| Alıntı F — WOLD/Sakha (n=769) | **0,657** | 0,464 hepsi-alıntı | ✅ fonotaktiğe karşı anlamlı (+0,121 doğruluk, p=0,0001) · ⚠️ yalnız verici yakınlığına karşı anlamlı **değil** (−0,007, p=0,608) |
| Rekonstrüksiyon NED (dev, n=83, çapa hariç) | **0,304** | 0,339 `majority_character` | ⚠️ anlamlı **değil** (GA [−0,075, +0,007]) |
| Rekonstrüksiyon tam (dev, çapa hariç) | **0,398** | 0,337 | ⚠️ anlamlı **değil** (p=0,188) |
| Rekonstrüksiyon NED (5 katlı ÇD, train+dev n=320) | **0,343** | 0,371 `majority_character` | ✅ **anlamlı** (fark −0,028, GA [−0,047, −0,009]) · tam 0,272 vs 0,253 anlamlı değil · BCFS 0,550 vs 0,519 ✅ anlamlı (+0,031, GA [+0,012, +0,047]) → **H2 destekleniyor** · `make eval-cv` |
| Akraba tespiti B-Cubed F (dev) | 0,931 | 0,934 düzenleme uzaklığı | ⚠️ taban çizgisine **eşit** (kümeleyici artık aynı ölçüyü kullanıyor) |
| Uzman uyuşmazlık bandı | **0,914** | — | otomatik sistemin gerçekçi tavanı |
| Denklik düzenliliği (CoPaR, TRAIN) | **0,713** | — | kural tabanlı doğruluğun üst sınırı |

Doğrulanmamış ya da kazanç vermeyen şeyler de aynı ayrıntıda raporlanıyor:
bağlam kodlaması (D3), N-best yeniden sıralama (D5), Batı Eski Türkçe
entegrasyonu (B2), ağaç-uyumsuz dağılım (C3), ses kanunu sinyalinin miras
tablosuyla kurtarılamaması (C4).

### Rekonstrüksiyon (dev bölümü, n=83, **çapa hariç**)

**Birincil metrikler NED ve B-Cubed F'tir.** Bu, keyfî bir tercih değil alan
standardı: SIGTYP 2022'nin resmi metrikleri ED, NED, B-Cubed F ve BLEU'dur ve
**tam doğruluk hiç yer almaz**; Bouchard-Côté ve ark. 2013 yalnız normalize
Levenshtein raporlar; Meloni ve ark. 2021'in ana metriği ortalama ED'dir.
List 2019 (*Beyond Edit Distances*) sistematik hataların ED tarafından her
örnekte yeniden cezalandırıldığını gösterip B-Cubed F önerir.

⚠️ **Ölçüm artık `dev` bölümündedir, tüm veride değil.** Motor denetimli bir
katman taşıyor (`proto_patterns`, TRAIN kavramlarından öğrenilmiş); tüm
veride ölçmek eğitim maddelerini ölçüme sokar ve raporlanan sayı motorun
performansı değil **ezberi** olur. `engine.evaluation.report` bu durumu
kendi tespit edip uyarı basıyor ve künyeye yazıyor.

| Sistem | **NED**↓ | **BCFS**↑ | ED↓ | tam |
|---|---|---|---|---|
| **motor** | **0,304** | **0,600** | **1,47** | **0,398** |
| `majority_character` (trivial) | 0,339 | 0,571 | 1,59 | 0,337 |
| `copy_anchor` (hiçbir şey yapma) | 0,390 | 0,529 | 1,87 | 0,229 |
| `copy_random_daughter` | 0,401 | 0,520 | 1,84 | 0,241 |
| `copy_longest` | 0,500 | 0,429 | 2,57 | 0,157 |

Birincil metrikte fark **−0,0347** (motor lehine), %95 GA [−0,0753,
+0,0065] → **anlamlı DEĞİL**; aralık sıfırı kılpayı içeriyor. n=83'te daha
fazlası gösterilemiyor. `test` bölümü **dondurulmuş** durumda ve Faz D
bitene kadar açılmayacak.

Tur başında (denetimli katman ve budama yokken) aynı bölümde motor NED
0,334 · tam 0,337 alıyordu.

#### Denetimli katman: örüntüden ata ses (Faz D2)

⚠️ **"Sıfır eğitim verisi" iddiası bırakıldı.** Kullanıcı kararıyla 400
uzman kümesi artık eğitim verisidir. Gerekçe ölçülmüştür — kural tabanlı
paradigmanın tavanı bizim uygulamamızda değil, paradigmadadır:

| Sistem sınıfı | Rom-phon tam doğruluk |
|---|---|
| rastgele kız dil | %0,06 |
| CorPaR (kural/örüntü) | %22,2 |
| SVM+PosStrIni (kural/örüntü) | %24,7 |
| RNN (denetimli) | %52,3 |
| Transformer (denetimli) | %53,8 |

Yöntem: hizalanmış her sütun bir denklik örüntüsüdür. ⚠️ **Örüntü tam
eşleşmesi işe yaramıyor** — 400 kümenin dil kümeleri neredeyse hiç birebir
örtüşmüyor. Bu yüzden örüntü **dil-ses çiftlerine** ayrıştırılıyor ve her
çift ayrı oy veriyor. Oy **olasılıkla** ağırlıklandırılıyor, ham sayımla
değil: ham sayım çok tanıklı dilleri kayırırdı.

Bu, elle atanmış `ARCHAISM_WEIGHTS` oylamasının veriden öğrenilmiş
karşılığıdır.

**Katkı (dev, n=83, dürüst koşul):**

| | tam | NED | BCFS | ED |
|---|---|---|---|---|
| öğrenilmiş tablo yok | 0,337 | 0,334 | 0,562 | 1,60 |
| tablo var | **0,361** | **0,306** | **0,583** | **1,48** |

⚠️ **Sıra ölçümle belirlendi ve ilk seçim yanlıştı.** Öğrenilmiş oy önce
elle yazılmış denkliklerin ÖNÜNE kondu; dilbilimsel olarak yerleşik iki
kararı bozdu:

    {tr: y, kk: z, otk: d}  ->  *j       (doğrusu *d̮)
    *teŋiŕ                  ->  *teniŕ   (ŋ sütunu kayboldu)

Elle yazılmış denklikler dar ve küratörlüdür; 135 kümeden öğrenilmiş bir
sayım onları geçemez. Öğrenilmiş oyun doğru yeri **arkaiklik ağırlıklı oyun
önü**: o yol en çok kullanılan (426 sütun) ama en zayıf (0,606) karar
yoludur ve elle atanmış katsayılara dayanır.

Karar sırası: tanısal (Lir-Şaz) → elle yazılmış denklik → **öğrenilmiş
örüntü** → arkaiklik ağırlıklı oy.

⚠️ Öğrenilmiş oy yalnız **güven eşiğinin (0,5) üstünde** devreye giriyor.
Koşulsuz üstün tutmak ölçüldü ve zarar veriyordu (sütun düzeyinde +5,7
puan, kelime düzeyinde 0,324 → 0,257). Sütun düzeyi ölçümü (dev, n=206):

| eşik | kural | öğrenilmiş | melez | öğrenilmiş oyun payı |
|---|---|---|---|---|
| 0,0 | 0,762 | 0,782 | 0,796 | %98 |
| **0,5** | 0,762 | 0,782 | **0,801** | %85 |
| 0,6 | 0,762 | 0,782 | 0,786 | %72 |
| 0,7 | 0,762 | 0,782 | 0,767 | %60 |

⚠️ Tablo yalnız 135 kümeden öğrenilebildi: hizalama genişliği ata biçim
uzunluğuyla tutmayan kümeler atlanıyor. Yanlış hizalanmış bir sütundan
öğrenmek, hiç öğrenmemekten kötüdür.

#### Hizalama budaması (Faz D4)

Blum & List 2023 (`lingrex.trimming`) boşluk-yönelimli budamanın 10 ailenin
10'unda düzenli denklik oranını artırdığını ölçüyor (+0,03…+0,07). Tek bir
dilin kendi eklemesi olan sütunlar hizalamayı genişletir ve ata biçme yanlış
konum ekler.

⚠️ Rekonstrüksiyon **doğruluğuna** etkisi yayınlanmamıştı; ölçtük:

| | tam | NED | BCFS | ED |
|---|---|---|---|---|
| budamasız | 0,361 | 0,306 | 0,583 | 1,48 |
| **budamalı** | **0,386** | **0,302** | **0,595** | **1,45** |

Yan kazanç: örüntü tablosu 135 yerine **142** kümeden öğrenilebiliyor
(hizalama genişliği ata biçim uzunluğuyla daha sık tutuyor).

⚠️ Çapa dahil koşulda tersi oluyor: 0,386 → 0,373. Birincil koşul çapa
hariç olandır.

> **⚠️ Ölçüm bozulması — bulundu ve düzeltildi.** Budama ilk uygulandığında
> `metrics.reconstruction_bcubed`in kendi hizalamasına da sızdı. O metrik
> tahmin ile altın biçmi hizalar; budama bir tarafta boşluk olan sütunları
> attığı için **tam da ölçmek istediği uyuşmazlıkları siliyordu**. Sonuç:
> bütün sistemlerin B-Cubed F'si birden yükseldi (`majority_character`
> 0,571 → **0,696**) — tahminleri hiç değişmemiş olmasına rağmen. Şimdi
> `align_forms(..., trim=False)` ölçüm ve fark sayma yollarında zorunlu.

#### Denklik düzenliliği: doğruluğun üst sınırı (Faz D1, CoPaR)

⚠️ **Plan CoPaR'ı elle yazılmış denkliklerin YERİNE koymayı öngörüyordu;
ölçüm o planı değiştirdi.** Elle yazılmış denklikleri öğrenilmiş sayımların
önüne koymak dilbilimsel olarak yerleşik iki kararı bozuyordu (yukarıda).
Elle yazılmış tablo dar ve küratörlüdür; öğrenilmiş katman onun **arkasında**
duruyor ve orada katkı sağlıyor.

CoPaR bu yüzden **teşhis** olarak koşuluyor, karar katmanı olarak değil:

| | değer |
|---|---|
| akraba kümesi (TRAIN) | 395 |
| hizalama sütunu | 1.565 |
| denklik örüntüsü | 762 (450'si **tekil**) |
| **düzenli sütun oranı** | **0,713** |

Tekil örüntü = o sütunun denkliği başka hiçbir yerde görülmüyor. Sayısı
düzensizliğin doğrudan ölçüsüdür.

⚠️ **Bu oran rekonstrüksiyon doğruluğunun üst sınırını belirler:** hiçbir
örüntüye oturmayan %29'luk sütunda kural tabanlı bir sistem ancak şansa
kalır. Blum & List'in budamayla düzenliliği artırma bulgusu bizde de
tutarlı: budama rekonstrüksiyon doğruluğunu 0,361 → 0,386 çıkardı.

#### ⚠️ Denenmiş ve KAZANÇ VERMEYEN dört şey

**Bağlam kodlaması (Faz D3).** List ve ark. 2022 Pos/Str/Ini kodlamasının
CorPaR'ın ED'sini %11 düşürdüğünü ölçüyor. Tablo anahtarı
`(konum, dil, ses)` yapıldı, konuma özgü destek yetmezse konumsuz toplama
geri çekilecek şekilde. Sonuç: NED 0,3063 → 0,3067, tam doğruluk aynı, çapa
dahil koşulda 0,386 → 0,373 (hafif **kötü**). Sebebi veri azlığı: 515 sütun
üçe bölününce konuma özgü sayımların çoğu destek eşiğini geçemiyor. Basit
olan tutuldu.

**N-best yeniden sıralama (Faz D5).** Lu, Wang & Mortensen 2024 (P2D) dört
veri setinde +0,9…+3,1 puan ölçüyor. Bizde:

| sıralama ölçütü | tam doğruluk |
|---|---|
| konsensüs (= top-1, mevcut) | **0,434** |
| yalnız P2D üretim uyumu | 0,361 |
| konsensüs + 0,2·P2D | 0,422 |
| konsensüs + 0,5·P2D | 0,434 |
| **N-best oracle (tavan)** | **0,506** |

Hiçbir karışım konsensüsü geçmiyor. Doğru cevap adayların içinde — oracle
top-1'in **7 puan** üstünde — ama sıralayıcı onu öne çıkaramıyor. Kök neden
büyük olasılıkla üreteç zayıflığı: `pt → X` denklikleri yalnız ~237
rekonstrüksiyonlu eğitim kümesinden öğreniliyor; P2D'nin yayınlanmış
kazançları sinir ağı üreteçlerle elde edilmiş.

⚠️ İlk sürüm üretim uyumunu **jenerik** Ortak Türkçe refleksiyle
hesaplıyordu ve daha da kötüydü (0,313). Gerçek P2D için ata dil, denklik
tablolarına **sözde dil** olarak katıldı (`pt`) ve her tanık dilin kendi
biçmi üretiliyor — bu 0,313 → 0,361 yaptı ama yetmedi.

**Karar:** adaylar üretiliyor ve çıktıda **rakip hipotez** olarak
gösteriliyor (`alternative_forms`), ama seçilen biçim değişmiyor.

**Hata profili ayrıştırıldı — mekanik hedef çıkmadı.** "Motor tabanı neden
anlamlı geçemiyor" sorusu hata kovalarına inilerek arandı. dev'de cevaplanan
81 maddenin dağılımı: doğru 33 · `ses_hatasi` 19 · `soz_basi_yanlis` 12 ·
`ek_soyulmamis` 11 · `unlu_uzunlugu` 3 · `capa_kisa` 2 · rotasizm 1.
Üç büyük kovanın üçü de hedef vermedi:

- **`ek_soyulmamis` (11).** Yaklaşık 6-7'si motorun hatası DEĞİL, altın
  gelenek farkı: ANT kümesinde 17 tanığın **hepsi** `-sḳa` taşıyor
  (`ḳumursḳa`, `χomursɣa`…) ama altın çıplak `*Kumïr` istiyor. Tanıkların
  tamamı eki taşıyorsa karşılaştırmalı yöntem ekli biçmi kurmakta haklıdır.
  Kalan vakalarda (`al`/`*alın`, `ör`/`*örle`) daha kısa tanık gerçekten
  vardı; iki kural **train'de** ölçüldü ve ikisi de net zararlı:

  | kural | aday | iyi | kötü | eşit | net |
  |---|---|---|---|---|---|
  | en kısa tanığı al | 53 | 14 | 35 | 4 | **−21** |
  | ≥2 dilde tanıklı çıplak biçmi al | 14 | 4 | 7 | 3 | **−3** |

- **`soz_basi_yanlis` (12).** 4'ü zaten `is_acceptable` sayılıyor
  (`DISPUTED_INITIAL`: *t-/d-*, *k-/g-*). dev'de "kabul ama tam değil" olan
  **tüm** maddeler yalnız 7 tanedir (4 söz başı + 3 ünlü uzunluğu), yani
  tam doğrulukla kabul arasındaki fark bütünüyle bu iki bilinen sınıftır.

- **`ses_hatasi` (train'de 77).** Yalnız **23'ü** aynı uzunlukta tek
  karakterlik fark; kalan %70 çok karakterli. O 23 de 14 ayrı ikame tipine
  dağılıyor, en sığı `v → b` ile 3 vaka. Sistematik bir denklik boşluğu yok.

⚠️ **Sonuç:** tabana olan açık düzeltilebilir bir kusurdan değil, paradigma
tavanından geliyor — yukarıdaki Rom-phon tablosu da aynı şeyi söylüyor
(kural/örüntü %22-25, denetimli %52-54). Buradan kazanç, kural katmanına
yama atarak değil **yöntem değiştirerek** gelir.

**Sütun düzeyine inildi — tavan orada da yok.** Kelime doğruluğunu sütun
kararları belirlediği için ölçüm sütun düzeyine taşındı (`learn()` ile aynı
hizalama sözleşmesi; dev'in **%70'i** hizalanabiliyor, 58 madde / 206 sütun
— kayıtlı `n=206` ile birebir tuttu):

| karar yolu | sütun | doğruluk | yük |
|---|---|---|---|
| `tek_ses` | 86 | 0,907 | %41,7 |
| `ogrenilmis_oruntu` | 78 | 0,833 | %37,9 |
| `arkaik_agirlik` | 21 | **0,571** | %10,2 |
| `denklik` | 19 | 0,789 | %9,2 |
| `tanisal` | 2 | 1,000 | %1,0 |

Sütun top-1 **0,835**, aynı 58 maddede kelime **0,586**. Bu altküme tüm
dev'den (0,398) kolaydır; buradaki kazanç raporlanan ölçüte birebir geçmez.

⚠️ Öğrenilmiş tablo, "en çok kullanılan ama en zayıf" diye bilinen
`arkaik_agirlik` yolunun yükünü **devralmış**: 426 sütundan 21'e inmiş.
Onu tümüyle düzeltmek bile en çok 9 sütun kazandırır.

Üç müdahale denendi, üçü de kazanç vermedi:

- **`VOWEL_ARCHAISM_WEIGHTS` hak etmiyor.** Elle bakımı yapılan 14 dillik
  ayrı ünlü tablosu ablasyonla kaldırıldı: train +1 sütun (0,8386 /
  0,8341), dev **tam sıfır** (65/82 birebir aynı). Docstring'indeki "tek
  tablo kullanmak ölçülen bir hata kaynağıydı" iddiası artık ölçümle
  desteklenmiyor — öğrenilmiş tablo o sütunları devraldı.
- **Ünlü uyumu kısıtı ölü doğdu.** Ünlüler sütunların %40'ı ama hataların
  %50'si (ünlü 0,793 / ünsüz 0,863) ve ünlü oracle açığı 13,4 puan; kısıt
  cazip görünüyordu. Ama motorun **kendi çıktısı çok ünlülü kelimelerin
  %99,0'ında zaten uyumlu**, altın biçimler ise yalnız %90,3. Kısıt yapı
  gereği sağlanıyor (tanıklar uyuma uyduğu için oylar da uyuyor), sonda 144
  maddenin **1'inde** ateşleniyor. Dahası motor altından DAHA uyumlu: sert
  kural olarak dayatmak, altının uyumsuz %9,7'sinden (`*kȫpek`, `*tīĺla`,
  `*iagïr`) uzaklaştırır.
- **Sütun adaylarını yeniden sıralamanın tavanı yok denecek kadar dar.**
  Yanlış sütunlarda altın cevap **train'de %58, dev'de %53 oranında aday
  listesinde HİÇ YOK** — yani hataların yarıdan fazlası sıralama değil
  **üretim** başarısızlığı ve hiçbir sıralayıcı onlara dokunamaz. Listede
  olanlarda da 1. adayla altın arasındaki puan farkı medyan 0,53 (train) /
  0,38 (dev): kıl payı değil, skorlayıcı emin biçimde yanılıyor. Mükemmel
  bir sıralayıcının dev tavanı 34 sütunun 16'sı (%7,8 oracle açığıyla
  tutarlı) ve bu, kelimede +2-4 maddedir — n=83'te gösterilemez.

⚠️ **Çekimserlik bedava değildir.** Cevaplanmayan madde ortalamaya mümkün
olan en kötü NED'i (1,0) katar. Bir dönem yalnızca cevaplanan maddeler
ortalanıyordu; o muhasebe cevap vermemeyi kusursuz cevap vermekle bir
tutuyor ve çekimser kalmayı ödüllendiriyordu.

### Akraba tespiti (B-Cubed F, dev kavramları)

| Sistem | F | kesinlik | duyarlılık |
|---|---|---|---|
| ayarlı düzenleme uzaklığı | **0,934** | 0,944 | 0,932 |
| SCA benzeri (LingPy) | 0,854 | 0,816 | 0,959 |
| LexStat benzeri | 0,824 | 0,749 | 0,991 |
| **motorun kümeleyicisi** | 0,931 | 0,944 | 0,927 |
| motorun eski kümeleyicisi (LingPy hizalayıcı ≥ 0,62) | 0,833 | 0,954 | 0,765 |
| hepsi tek küme (trivial) | 0,743 | 0,642 | 1,000 |

Referans: LexStat-Infomap **F ≈ 0,89** (List, Greenhill & Gray 2017).
⚠️ Motorun kümeleyicisi eskiden LingPy hizalayıcısının fonetik
benzerliğini kullanıyordu ve düz düzenleme uzaklığının gerisindeydi.
Hizalayıcı hiçbir eşikte (en iyisi 0,40: F 0,857) ve hiçbir karışımda
(en iyisi 0,3·hizalayıcı + 0,7·düzenleme: F 0,923) düzenleme uzaklığını
geçemedi. Kümeleyici bu yüzden düzenleme benzerliğine geçti
(`COGNATE_THRESHOLD = 0,50`, train'de F'yi maksimize eden eşik). Bedeli
train kesinliğinin 0,934'ten 0,883'e inmesi; karşılığında dev
duyarlılığı 0,765'ten 0,927'ye çıktı. Motor artık taban çizgisinin
altında değil, ama onu geçmiyor da: **ikisi aynı algoritma.** Kalan
0,003'lük fark, motorun 2 harften kısa ve yinelenen biçimleri elemesinden.
Çuvaşça (tek Ogur tanığı) duyarlılığı da arttı: dev 0,594 → 0,692.

#### Uzman uyuşmazlık bandı (Faz E1) — tavan 1,00 değildir

⚠️ **Bu ölçüm olmadan yukarıdaki hiçbir sayı yorumlanamaz.** "Motor F 0,82
aldı" cümlesi, uzmanların birbiriyle ne kadar uyuştuğu bilinmeden
anlamsızdır. List, Walworth ve ark. (2018, *JLE*) bu boşluğu açıkça ilan
ediyor — akraba kümesi kararlarında uzmanlar arası uyumu sistematik ölçen
bir çalışma **yok**.

İki bağımsız uzman derlemesi karşılaştırıldı:

| | değer |
|---|---|
| ortak öğe | 1.833 · 26 dil |
| küme sayısı | 234 (Savelyev) vs 173 (Hruschka) |
| B-Cubed kesinlik | 0,975 |
| B-Cubed duyarlılık | 0,861 |
| **B-Cubed F** | **0,914** |
| **Ayarlanmış Rand İndeksi** | **0,912** |

⚠️ **Kavram köprüsü kurulamıyor**: `hruschkaturkic`te Concepticon glossu
**hiç yok** — parametreler `Etymon 2` gibi künye etiketleri. Köprü bu
yüzden **öğe düzeyinde** kuruldu: bir `(dil, karşılaştırma biçmi)` çifti
iki derlemede de geçiyorsa ortak öğedir.

Örnek uyuşmazlıklar (Savelyev ayırıyor, Hruschka birleştiriyor):

    alt ak    ~ az ağ    ~ khk ah   ~ slq ah      "ak"
    alt at    ~ gag at   ~ uz ot                  "at"
    alt ağaş  ~ az ağaç                           "ağaç"

Kesinlik 0,975 / duyarlılık 0,861 örüntüsü şunu söylüyor: Savelyev'in
kümeleri **daha ince**; her Savelyev kümesi neredeyse tümüyle bir Hruschka
kümesinin içinde ama Hruschka kümeleri birden çok Savelyev kümesine
dağılıyor.

⚠️ **Bu sayı yukarıdaki tabloyla DOĞRUDAN KARŞILAŞTIRILAMAZ.** Tablodaki
skorlar `savelyevturkic`in dev kavramlarında tahmin-vs-altın; band ise iki
ayrı derlemenin kesişiminde bölümleme-vs-bölümleme. Kavram listeleri ve
küme inceliği farklı, dolayısıyla band **aşağı yanlıdır**. Bandın söylediği
tek şey: tavan 1,00 değil.

⚠️ Bu ölçümün kendi kısıtları: ortak öğe tanımı biçim eşleşmesine dayanır
(çevriyazı farkları öğeyi düşürür, örneklem uyuşmanın kolay tarafına
kayabilir); iki derleme aynı kavram listesini kullanmıyor, kesişim rastgele
değil; küme büyüklüğü dağılımları farklı ve B-Cubed buna duyarlı.

### İleri akraba tahmini (n=903 çift, tr → 31 dil)

| Sistem | tam | %95 GA | ≤1 harf |
|---|---|---|---|
| **öğrenilmiş denklikler** | **%47,6** | [%44,3 %50,8] | **%75,5** |
| elle yazılmış kurallar (oracle) | %47,7 | | %75,1 |
| kimlik (kopyala) | %34,6 | [%31,5 %37,7] | %65,6 |

Kimlik taban çizgisine karşı fark **+%13,1**, permütasyon p = 0,0001 →
**anlamlı**. Sözlük araması için asıl önemli sayı "≤1 harf" oranıdır.

### Güven kalibrasyonu (n=372)

| Skor | ECE | %95 GA | Brier | AUC |
|---|---|---|---|---|
| ham | 0,406 | [0,363 0,446] | 0,342 | 0,648 |
| kalibre (izotonik) | 0,057 | [0,037 0,104] | 0,177 | 0,620 |
| **kalibre (Platt)** | **0,037** | [0,016 0,079] | 0,178 | 0,621 |

Ham skor sistematik olarak **+0,41 aşırı güvenli**ydi (ortalama %64,6 güven,
gerçek doğruluk %23,9). Kullanıcıya gösterilen skor artık kalibre skordur.

⚠️ **Kalibrasyon AUC'yi düşürüyor (0,648 → 0,621) ve bu beklenen bir
bedeldir, tutarsızlık değil.** Platt monoton bir dönüşümdür; tek bir
kalibratör uygulansa AUC'yi *değiştirmemesi* gerekirdi. Düşüşün sebebi
`cross_validated_calibration`: her madde **kendisinin katılmadığı**
katmanlarda eğitilen kalibratörle dönüştürülür (aynı veride hem kalibre
edip hem ölçmek ECE'yi yapay olarak sıfıra yaklaştırırdı). Beş ayrı
kalibratör beş ayrı altkümeye uygulanınca küresel eşleme monoton olmaktan
çıkar ve sıralama bilgisinin bir kısmı silinir. Takas bilinçlidir:
ECE 0,406 → 0,037 karşılığında AUC'den ~0,03 verilmiştir.

### Alıntı tespiti

**Birincil ölçüt — WOLD** (uzman derlemesi, Wiktionary'den bağımsız), Sakha, n=769:

| Sistem | F | kesinlik | duyarlılık | doğruluk |
|---|---|---|---|---|
| **motor (eğitilmiş birleştirici)** | **0,651** | 0,635 | 0,668 | 0,784 |
| yalnız verici yakınlığı | 0,624 | 0,697 | 0,565 | 0,795 |
| motor (doğrusal yedek yol) | 0,623 | 0,547 | 0,724 | 0,736 |
| yalnız dizilim modeli (PyBor) | 0,558 | 0,556 | 0,560 | 0,732 |
| hepsi alıntı (trivial) | 0,464 | 0,302 | 1,000 | 0,302 |
| yalnız fonotaktik kural | 0,215 | 0,372 | 0,151 | 0,667 |

**F 0,385 → 0,651.** Plan hedefi (≥0,60) aşıldı; PyBor'un WOLD 41 dil
ortalamasının (0,59–0,61) üstünde.

⚠️ **Ablasyon hükmü WOLD'da anlamlı DEĞİL**: gönderdiğimiz sistem
(eğitilmiş birleştirici) vs yalnız fonotaktik fark +0,030, %95 GA
[−0,018, +0,077], p=0,251. Doğrusal yedek yol için aynı fark +0,069
(p=0,004) ama üretimde kullanılan sistem o değil; hüküm **gönderdiğimiz
sisteme** göre verilmelidir.

Anlamlı üstünlük **ikincil ölçütte** (aşağıda, TDK + Nişanyan) elde
edildi.

⚠️ Motor, tek başına verici yakınlığından **anlamlı biçimde iyi değil**:
madde başına doğrulukta fark −0,010, %95 GA [−0,030, +0,009], p=0,354.
Yani "istatistiksel olarak berabere" — üstünlük iddia edilmiyor. (Bu turun
başında aynı fark −0,030, p=0,004 ile motorun **aleyhineydi**.)

#### Kazancı sağlayan üç şey

**1. Verici dil sözlüğüne SCA yakınlığı** (`sabor`, Miller & List 2023).
Sinyal ablasyonu (n=769, her sinyal tek tek çıkarılır, eşik her seferinde
yeniden ayarlanır):

| çıkarılan sinyal | kalan F | katkısı |
|---|---|---|
| `verici_yakınlığı` | 0,384 | **+0,239** |
| `zincir_kanıtı` | 0,611 | +0,013 |
| `fonotaktik_ihlal` | 0,635 | −0,012 |
| `fonotaktik_model` | 0,626 | −0,002 (ağırlığı sıfır) |
| `ses_kanunu_ihlali` | 0,626 | −0,002 (ağırlığı sıfır) |
| `değişimsiz_yayılım` | 0,626 | −0,002 (ağırlığı sıfır) |

(Tablo **doğrusal yedek yol** içindir, F=0,623. Sıfır ağırlıklı sinyallerin
−0,002'si eşik seçiminin gürültüsüdür.)

**`ses_kanunu_ihlali` ve `değişimsiz_yayılım` artık karara katılmıyor.**
Hesaplanmaya ve kullanıcıya gösterilmeye devam ediyorlar (gerekçe değeri
taşırlar) ama ağırlıkları sıfır.

⚠️ **Karar rapor yarısına bakılarak verilmedi** — o, ölçümün içine ayar
sızdırmak olurdu. Ayar yarısı kendi içinde ikiye bölündü (iç-ayar /
iç-doğrulama, n=385; rapor yarısı hiç görülmedi) ve orada iki sinyal
**hiçbir kararı değiştirmiyordu**: F 0,6016 ve doğruluk 0,7455, dört
kombinasyonda da birebir aynı. Gerekçe budur — hiçbir karara katkısı
olmayan bir sinyal toplamın %20'sini taşımamalıdır.

⚠️ Rapor yarısındaki +0,023'lük iyileşme ayar verisinde **öngörülmemişti**;
bağımsız doğrulanmadı, üst sınır sayılmalıdır.

⚠️ **Kavramsal teşhis sınandı ve DOĞRULANMADI.** Sinyalin zayıflığını
"denklikleri alıntıların da içinde olduğu veriden öğrendik"e bağlıyorduk.
Yalnız uzmanın ata biçim verdiği kümelerden ikinci bir denklik tablosu
öğrenildi (395 eğitim kümesinin 237'si) ve sinyalin katkısı −0,0101'den
yalnız −0,0083'e geldi. Teşhis yanlıştı: yöntemin kendisi bu görevde zayıf.
İkinci tablo yine de kullanımda — kavramsal olarak doğru ve ölçüm nötr.

**2. Yön süzgeci.** Verici sözlüğü Türkiden **alınmış** kelimeleri de
içeriyor ve sinyali ters yönden tetikliyordu:

    Türkçe göz   ~ Ermenice գյոզ (gyoz)  SCA 0,040   "From Ottoman Turkish"
    Türkçe demir ~ Farsça   تمر  (tamor) SCA 0,075   "Borrowed from Turkic"

Verici maddesinin kendi etimolojisi Türki kaynağa işaret ediyorsa madde
kanıttan çıkarılıyor (1.674.418'in 2.900'ü). "Compare Turkish …" alıntı
beyanı sayılmıyor; sayılsaydı Türki bir adı anan her madde elenirdi.

**3. Şans benzerliği denetimi** (Kessler 2001). Ham mesafe eşiği **verici
havuzunun büyüklüğüne gizlice bağlıdır**: Sakha ölçütünde havuz 3 dil /
448.000 madde, Türkçede 6 dil / 1.600.000. Aynı 0,35 eşiği ikisinde aynı
şeyi ölçmez.

    Türkçe baş   ~ Fransızca pou     SCA 0,092  ama kontrollerin %17'si de bu kadar yakın
    Türkçe balık ~ Fransızca béluga  SCA 0,146  kontrollerin %17'si
    Türkçe kitap ~ Arapça    كتاب    SCA 0,000  kontrollerin %0'ı  -> BULGU

Aynı havuza karşı 12 kontrol kelimesi ölçülüyor; gözlenen mesafe kontrol
dağılımının %10'undan düşük değilse sinyal ateşlenmiyor. Havuz büyüdükçe
null da kayar ve eşik kendini ayarlar.

#### Eğitilmiş dizilim modeli (PyBor)

Elle yazılmış fonotaktik kurallar (ünlü uyumu, yasak söz başı ses)
WOLD/Sakha'da tek başına **F 0,215** alıyor. Aynı veride eğitilmiş iki
modelli sınıflandırıcı — biri miras kelimelerden, biri alıntılardan, karar
log olasılık farkıyla — **F 0,558** alıyor. Yayınlanmış PyBor ortalaması
0,59–0,61; bu bağımsız bir yeniden üretimdir.

Uygulama: karakter 3-gram Markov, Witten-Bell yumuşatmalı. LSTM sürümü
yayında biraz daha iyi (0,61 vs 0,59) ama bağımlılık gerektiriyor.

⚠️ **Model dile özgüdür**; başka dilin modeline dönülmez. Fonotaktik dilden
dile değişir ve zaten ölçtüğü şey odur.

> **⚠️ Yığın sızıntısı — ölçülmüş ve düzeltilmiş.** Model tüm ayar
> yarısında eğitilip eşik de aynı yarıda ayarlanınca, eşiğe modelin
> gerçekte sahip olmadığı bir ayırt etme gücü varsaydırıldı:
>
>     ayar yarısı (model burada eğitildi)  sınıf ayrımı 1,9197
>     rapor yarısı (hiç görülmedi)         sınıf ayrımı 0,6278  → 3,1 kat şişkin
>
> Sonuç: motorun F'si 0,646'dan **0,587'ye düştü** ve ablasyon "verici
> yakınlığı zararlı" gibi saçma bir sonuç verdi. Düzeltme: ayar yarısı ikiye
> bölünüyor — model ilk parçada eğitiliyor, eşik ikinci parçada ayarlanıyor.
> Bunun bedeli de var: eşik yarı veriyle seçildiği için doğrusal yedek yol
> 0,646'dan 0,623'e indi.

⚠️ Dizilim modelinin **doğrusal yedek yolda ağırlığı sıfır**. Öğrenilen
katsayıları normalize edip doğrusal toplama koymak lojistik modelin
davranışını yeniden üretmiyor — sabit terim (−1,993) ve sigmoid kararın
parçasıdır:

    doğrusal toplam, dizilim modeli dışarıda        F 0,646
    doğrusal toplam, öğrenilen katsayılar normalize F 0,614
    eğitilmiş birleştirici (sigmoid + sabit terim)  F 0,651

#### Eğitilmiş birleştirici

El ile konmuş ağırlıklı toplam (düzeltme öncesi), en güçlü sinyalin kararını
**bozuyordu**:
beş sinyalli motor madde başına doğrulukta yalnız verici yakınlığının
altında kalıyordu (fark −0,030, %95 GA [−0,049, −0,010], p=0,004). Ayrıca
aritmetik bir kusur vardı: zincir sinyali yokken `0,20 + 0,10 = 0,30 <
0,45` — o iki sinyal tek başlarına **hiçbir kararı değiştiremiyordu**.

Ağırlıklar artık **ayar yarısında** öğreniliyor (lojistik regresyon; sklearn
yok, optimizasyon repoda ve deterministik). Öğrenilen katsayılar ablasyonu
birebir doğruluyor:

    sabit −1,993 · verici_yakınlığı +1,552 · zincir_kanıtı +1,014
                 · fonotaktik_model +0,878 · fonotaktik_ihlal +0,140
                 · ses_kanunu_ihlali +0,038 · değişimsiz_yayılım −0,018

⚠️ **Hedef ölçü seçimi sonucu belirler ve gizlenemez.** Aynı model, yalnız
eşik farklı: F hedefli 0,651/0,784 · doğruluk hedefli 0,534/0,787. İkisi
aynı anda alınamaz; hangisinin seçildiği model dosyasında saklanıyor.

⚠️ Model WOLD/Sakha'da eğitildi. Başka bir dile uygulandığında çıktı
"ALAN DIŞI" damgası taşır — sinyal dağılımı dilden dile değişir ve ölçülen
F o dilde geçerli değildir.

⚠️ Model yoksa el ağırlıklarına dönülür ama bu **ilan edilir**
(`verdict.is_trained == False`), sessizce yapılmaz.

#### İkincil ölçüt — Türkçe altın küme (TDK + Nişanyan), n=349

⚠️ **Neden gerekiyordu.** WOLD'da tek Türki dil Sakha'dır. Türkçe için
elimizdeki tek etiket kaynağı Wiktionary'ydi ve motorun zincir sinyali
**zaten o etiketi okuyor** — ona karşı ölçüm döngüseldir. Ayrıca o kümede
alıntı oranı %72,9 olduğu için F'yi en yükselten karar "hepsine alıntı
de"dir ve sistemler çöküyor.

Bu küme **Wiktionary'ye hiç değmeden** iki bağımsız Türkçe kaynaktan
kuruldu: TDK Güncel Türkçe Sözlük (`lisan` alanı) ve Nişanyan Sözlük
(`relation` alanı). 2.000 kelime sorgulandı, **699 madde** etiketlendi,
421'i iki kaynağın da onayıyla; **9 uyuşmazlık** (kaynak uyumu %98,7).

| Sistem | F | kesinlik | duyarlılık | doğruluk |
|---|---|---|---|---|
| **motor (eğitilmiş)** | **0,885** | 0,909 | 0,861 | 0,865 |
| motor (doğrusal yedek) | 0,801 | 0,793 | 0,809 | 0,759 |
| yalnız dizilim modeli | 0,800 | 0,669 | 0,995 | 0,702 |
| yalnız fonotaktik kural | 0,758 | 0,937 | 0,636 | 0,756 |
| yalnız verici yakınlığı | 0,757 | 0,846 | 0,684 | 0,736 |
| hepsi alıntı (trivial) | 0,749 | 0,599 | 1,000 | 0,599 |

⚠️ **Zincir sinyali bu kümede KAPALI** (`use_chain=False`) ve öyle kalmalı.
Ölçüldü (2026-09-24): açık olsaydı eğitilmiş motor F **0,963** alırdı. İndeksin
Wiktionary köken etiketi, etiketli maddelerin %95'inde altın etiketle aynı
(alıntı 394/404, miras 140/157); Wiktionary Türkçe kökenleri sık sık
Nişanyan'a dayandığı için bu örtüşme ortak kaynaktan da gelebilir. Tablodaki
sayılar etiketsiz ölçümdür.

⚠️ **Tanık bulma düzeltildi ve yalnız motorun kendi satırları oynadı.**
Eskiden biçim bulunamayınca bulanık aramaya düşülüp dönen ilk isabet —
anlamına bakılmadan — tanık sayılıyordu (bkz. `find_witnesses`). Tanık
kullanmayan üç satır (dizilim modeli, fonotaktik, hepsi-alıntı) **birebir
aynı kaldı**; düzeltmenin doğru yeri vurduğunun sağlaması budur.

| tanık kuralı | ort. tanık | F | `ses_kanunu` | `yayılım` |
|---|---|---|---|---|
| tüm bulanık (eski) | 2,29 | 0,8775 | 125 (%62) | 88 (%44) |
| yalnız birebir | 0,15 | **0,8873** | 32 (%16) | 9 (%4) |
| **bulanık + gloss örtüşmesi** | 0,37 | 0,8845 | **47 (%24)** | **17 (%8)** |

⚠️ **F'si en yüksek olan seçilmedi ve sebebi ölçülmüştür.** Bulanık
isabetlerin %10,1'i anlamca gerçek (`Sovyet`~`sovet`, `arzu`~`arzuw`);
`kitap` için birebir kural 1 tanık bulurken gloss süzgeci 7 buluyor
(`китап`, `кітап`, `kitob` — aynı kelimenin Kiril/Latin yazımları, birebir
eşleşmenin asla yakalayamayacağı kanıt). Birebir kural F'de ~1 madde önde
ama **tanık gerektiren iki sinyalin değerlendirilebilirliğini yarıya
indiriyor**; bu, düzeltilen hatanın ta kendisidir (sinyal sessizce devre
dışı kalır, ablasyon "katkı sağlamıyor" der). F farkı gürültü, sinyal
kaybı yapısaldır.

⚠️ Farklar n=349'da ~3 madde mertebesindedir; bu bir kazanç değil,
**denetimsiz kanıtın kaldırılmasıdır**.

✅ **İlk kez trivial taban çizgiye karşı anlamlı üstünlük:**
motor vs `always_borrowed` **+0,267**, %95 GA [+0,203, +0,330], p=0,0001.
Motor vs yalnız fonotaktik **+0,109**, %95 GA [+0,069, +0,149], p=0,0001.
(Fark **madde düzeyi doğruluk** farkıdır, F farkı değil; F cinsinden
hepsi-alıntıya karşı üstünlük +0,135'tir.)

⚠️ **Verici dil eklemek (İngilizce/Almanca/Latince) ölçülerek REDDEDİLDİ.**
Verici indeksinde 9 dil var (it 622.831 · ru 440.919 · fr 401.061 ·
el 84.722 · ar 77.118 · hy 21.748 · fa 19.694 · mn 6.480 · evn 599) ve
İngilizce/Almanca/Latince hiç yok. "Modern alıntıların çoğu tespit
edilemiyor" gerekçesiyle eklenmesi planlanmıştı; **bu altın küme için
gerekçe yanlış çıktı**:

| | madde |
|---|---|
| alıntı maddesi | 420 |
| kökeninde indekste bulunan bir verici geçen | **407 (%96,9)** |
| *yalnızca* İngilizce/Almanca/Latince kökenli | **9 (%2,1)** |

O 9 madde: `brifing, feribot, galon, geyşa, jet, lobelya, master, pikap,
trol` — `geyşa` Japonca, `lobelya` Yeni Latince. Almanca'nın toplam katkısı
**1 madde**. Buna karşılık kaikki dökümleri İngilizce 3.246 MB · Latince
1.220 MB · Almanca 1.077 MB (≈5,5 GB ham). Küme ayar/test diye ikiye
bölündüğü için (n=349) ölçüme yansıyacak kısım ~4-5 maddedir.

⚠️ Sınır: bu altın küme `alıntı` için **iki kaynağın onayını** istiyor ve
modern İngilizce alıntıları yapısal olarak az temsil ediyor olabilir.
Gerçek kullanıcı sorgularında aynı oran ölçülmedi; ret bu küme içindir.

⚠️ **Kanıt kuralı asimetriktir ve bu gizlenemez.** `alıntı` etiketi iki
kaynağın da alıntı demesini gerektirir (güçlü). `miras` etiketi Nişanyan'ın
açıkça `ses evrimi` demesi **ve** TDK'nın kaynak dil yazmamasıyla verilir
(zayıf-orta) — çünkü TDK yalnız alıntı maddelerde `lisan` yazar ve "Türkçe
kökenli" ile "etimoloji verilmemiş" ayırt edilemez. Bu asimetri yüzünden
yanlış-negatif, yanlış-pozitiften daha olasıdır.

> **⚠️ Dört ölçüm hatası, kurma sırasında bulundu ve düzeltildi.**
> **(1)** İlk sürüm TDK sessiz kaldığında etiketi **Nişanyan'dan
> türetiyordu**; "iki kaynak anlaştı" demek tautolojiydi ve uyum oranı
> yapay olarak **%100** çıkıyordu.
> **(2)** Sonraki sürüm sessizliği tümden belirsiz saydı ve küme **%100
> alıntı** oldu — ölçüt olarak işe yaramaz. Asimetrik kural ikisini de
> çözüyor.
> **(3)** Kelime listesi alfabetik sıralanınca ilk 60 madde tümüyle **ek**
> çıktı (`-abilmek`, `-acak`) ve hiçbir etiket alınamadı. Ekler eleniyor,
> örneklem deterministik aralıklı.
> **(4)** Nişanyan ilişki adı `türetme` diye yazılmıştı; gerçek değer
> `türeme`. Türemiş kelimelerin **hiçbiri** etiketlenemiyordu.

⚠️ **Lisans.** Nişanyan Sözlük geliştiriciye açık bir API lisansı ilan
etmiyor. Veri **yalnız iç doğrulama** için, atıflı ve sınırlı hacimde
çekiliyor; `data/gold/` .gitignore altında, yeniden dağıtılmıyor. TDK
Güncel Türkçe Sözlük kamuya açık bir kurum sözlüğüdür.

#### Türkçe ablasyon (Wiktionary etiketi, zincir sinyali kapalı), n=750

| Sistem | F | kesinlik | duyarlılık |
|---|---|---|---|
| hepsi alıntı (trivial) | 0,844 | 0,729 | 1,000 |
| motor (eğitilmiş) | 0,844 | 0,729 | 1,000 |
| motor (el ağırlıkları) | 0,811 | 0,738 | 0,900 |
| yalnız fonotaktik | 0,660 | 0,766 | 0,580 |

Motor vs yalnız fonotaktik: **+0,129**, %95 GA [+0,089, +0,169], p=0,0001.

⚠️ **Eğitilmiş birleştirici bu kümede ÇÖKÜYOR**: kararları
`always_borrowed` ile birebir aynı. Alıntı oranı %72,9 olduğu için F'yi en
yükselten eşik "hepsine alıntı de"dir. Bu bir başarı değil, F ölçüsünün
dengesiz sınıftaki bilinen patolojisidir — değerlendirme artık bu durumu
kendi tespit edip uyarı basıyor.

#### Neden SCA, neden anlam kısıtı

Sakha Rusça `stol`u `ostuol` yapar (öntüreme ünlü + ikizünlü); düz
Levenshtein 3/6 = 0,50 verip eşiğin üstünde kalır, SCA 0,216 verir. LingPy
yoksa sinyal **devre dışı kalır** — düz Levenshtein'a düşmek, yayınlanmış
F1 0,806'yı başka bir mesafeyle iddia etmek olurdu. (Hız için SCA öncesi
ucuz düzenlenme uzaklığıyla en yakın 40 aday seçiliyor; **karar ölçütü hâlâ
SCA**.)

Anlam kısıtı yayınlanmış kurulumun parçası. Kısıtsız arama 1,67 milyon
maddelik indekse yayılır. ⚠️ Kısıtın bedeli ölçülmüştür: sabor'da kaçan
alıntıların %45'i tam bu kısıttan gelir.

Uzunluk kapısı **konmadı**: ayrım her uzunlukta duruyor (alıntı ort. SCA
0,31, miras 0,48).

> **Düzeltme kaydı — aynı hata sınıfı iki kez.**
> **(1)** Değerlendirme kodu `witnesses` alanını hiç doldurmuyordu; dört
> sinyalden ikisi tanık gerektirdiği için **yapısal olarak devre dışıydı**.
> Sonuç fonotaktikle birebir aynı çıkıyor, biz "sinyaller katkı sağlamıyor"
> diye raporluyorduk. Doğrusu "sinyaller hiç çalıştırılmadı"ydı.
> **(2)** `_attach_witnesses` dondurulmuş kaydı alan alan yeniden kuruyordu
> ve yeni eklenen `sense` alanını **sessizce düşürüyordu**; verici yakınlığı
> anlam kısıtlı olduğu için sinyal ilk koşuda F=0,0000 verdi.
> Artık `dataclasses.replace` kullanılıyor — o hataya yapısal olarak kapalı.

### Öngörü testi (n=182, kilitli sicil)

| | |
|---|---|
| tuttu | %11,0 |
| tuttu + yakın (≤1 harf) | %29,7 |
| **şans taban çizgisi** | **%1,26** |
| şansın kaç katı | **8,7×** |

Şans kontrolü olmadan "%11'i tuttu" bir bulgu değildir: motor 108 bin kayıtlık
bir indekste arıyor ve kısa biçimler salt şansla eşleşir (Kessler 2001).

### Negatif kontroller

| Batarya | n | yanlış-pozitif | **güçlü iddia** |
|---|---|---|---|
| fonotaktik geçerli sahte | 8 | 1,000 | **0,000** |
| bariz sahte | 4 | 0,000 | **0,000** |
| sahte akraba | 4 | 0,000 | **0,000** |
| alıntı tuzağı | 5 | 1,000 | **0,000** |
| eşadlı | 3 | — | **0,000** |

En kritik sütun sonuncusu: motor hiçbir negatif kontrolde 🟢/🟡 rozet
vermiyor. Uydurma bir köke düşük güvenle aday üretmesi kabul edilebilir; onu
güçlü bir iddia olarak sunması kabul edilemez.

### Ağız kelimeleri (Faz 10, TDK Derleme, n=1631)

| | bulanık tanıklarla | **birebir tanıkla** |
|---|---|---|
| çözüldü | 1 | **0** |
| güçlü aday | 133 (%8,2) | **54 (%3,3)** |
| yetersiz kanıt | 1497 (%91,8) | **1577 (%96,7)** |
| kelime başına tanık | 3,30 | **0,36** |

⚠️ **Sağdaki sütun geçerli olandır.** Soldaki, tanık bulunamayınca bulanık
aramaya (düzenleme uzaklığı 1) düşüp dönen ilk isabeti **anlamına
bakmadan** tanık sayan sürümün sayılarıdır. Ölçüldü: tanık sayılanların
yalnız %5,8'i öngörülen biçmi gerçekten buluyordu; `çaman` için 10 "tanık"
заман (zaman), çaň (toz), yaman (kötü), qâan (kan), Саян (Akrep burcu)
idi. 662 bulanık isabetin **1'i** kelimenin Derleme anlamıyla ortak sözcük
taşıyor. Sahte tanıklar raporu değil **hipotezi** de etkiliyordu:
`inherited` 997 → 98, `unknown` 458 → 1355.

⚠️ **Katma değer ölçüldü ve pratikte sıfırdır.** Kalan 54 güçlü adayın
**51'inin kökeni sözlükte zaten kayıtlı** (döngüsellik %94,4). Geriye kalan
3 kelime — `engeç`, `görüm`, `küşüm` — üçü de `modern_coinage`, üçü de
sabit 0,40 skorunda, üçü de **tek tanıklı**; tek bir gerçek rekonstrüksiyon
yok. Yani motor 1.631 ağız kelimesinde sözlüğün üstüne ölçülebilir bir şey
koymuyor.

⚠️ "Yetersiz kanıt" bir başarısızlık değil, **dürüst sonuçtur**. Ağız
kelimelerinin çoğu tek bir ilde tanıklanmıştır ve karşılaştırmalı yöntemin
gerektirdiği bağımsız tanık yoktur. Buradaki asıl bulgu yüksek "yetersiz
kanıt" oranı değil, **onu düşürmek için sahte kanıt üretilmiş olmasıdır**.

## Kurulum

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                 # çekirdek + test araçları
pip install -e ".[dev,phon,pdf]"        # + LingPy, PanPhon, Epitran, pdfminer
```

| Ekstra | İçerik | Zorunlu mu? |
|---|---|---|
| *(çekirdek)* | `requests` | evet |
| `phon` | `lingpy`, `panphon`, `epitran` — gerçek fonetik hizalama ve IPA | hayır, ama önerilir |
| `pdf` | `pdfminer.six` — `data/books/` altındaki PDF'lerde tam metin arama | hayır |
| `semantic` | `sentence-transformers` — semantik mesafe aşaması (~2-3 GB) | hayır |
| `dev` | `pytest`, `pytest-cov`, `responses`, `ruff` | geliştirme |

Bu ekstralar kurulu değilse motor çökmez: ilgili aşama **kanıt üretmediğini
bildirir** ve skora katılmaz.

## Kullanım

```bash
python -m engine.cli search deniz              # arama
python -m engine.cli search deniz --json       # ham JSON
python -m engine.cli search deniz --ai         # + yerel LLM sentezi (Ollama)
python -m engine.cli validate göz --origin '*köŕ' --donor 'Proto-Türkçe'
python -m engine.cli list                      # kayıtlı bulgular
python -m engine.cli show göz                  # kayıtlı bulguyu göster
python -m engine.cli export göz --out cldf/    # CLDF dışa aktarım
python -m engine.cli bulk --file kelimeler.txt # toplu sorgu
python -m engine.cli search göz --verbose      # ayrıntılı log
```

### REST API ve web paneli

```bash
python -m engine.server            # http://127.0.0.1:8000
cd web && npx serve -l 3000 .      # http://localhost:3000
```

| Uç nokta | Açıklama |
|---|---|
| `GET /api/search?word=X&ai=false&save=true` | Etimoloji araması |
| `GET /api/list` | Kayıtlı bulgular |
| `GET /api/health` | Kaynak sayıları ve önbellek durumu |

Sunucu varsayılan olarak **yalnızca `127.0.0.1`** dinler ve CORS'u
yapılandırılmış origin'lerle sınırlar. Kimlik doğrulaması yoktur; dışa açmayın.

## Veri kaynakları

Motor **9 canlı kaynak** ve **9 yerel tohum (seed) veri dosyası** kullanır.
İkisi arasındaki fark her kayıtta `origin: "live" | "seed"` alanıyla,
CLI ve web panelinde ise görsel olarak belirtilir.

**Canlı kaynaklar** — TDK (Güncel Türkçe Sözlük, Tarama, Derleme), Nişanyan
Sözlük, EtimolojiTürkçe (tarihli ilk tanıklamalar), İngilizce Wiktionary
(kelime sayfası + Proto-Turkic rekonstrüksiyon sayfaları), 14 Türki dilin kendi
Wiktionary sürümü, Wiktextract/Kaikki, Internet Archive.

**Yerel veri** (`make data`, `make lexicons`) — 5 CLDF veri kümesi
(savelyevturkic, hruschkaturkic, starostinaltaic, robbeetstriangulation, WOLD)
ve 23 Türki dilin kaikki dökümü (125.879 madde). Rusça Wiktionary sürümü de
katıldığında SQLite FTS5 arama indeksi **25 dil kodu / 150.458 kayıt** taşır.
Her indirme sürüm, tarih ve SHA-256 damgası taşır (`data/SOURCES.md`).

⚠️ **Wiktionary türevi veri altın standart DEĞİLDİR.** Häuser & Stamatakis
(2025) bu verinin uzman ağaçlarıyla tutarsız olduğunu gösteriyor; ayrıca
Wiktionary'nin Proto-Türkçe rekonstrüksiyonları büyük ölçüde EDAL soyundandır
ve bu proje EDAL'ı tek kaynak olarak zaten kabul etmiyor. kaikki burada
**yalnız arama indeksi**dir; akrabalık ve ata biçim kararı uzman verisinden
(`savelyevturkic`) veya kümeleme katmanından gelir.

**Tohum veri** (`data/seed/`) — Clauson EDPT, Sevortjan ЭСТЯ, Divânu Lugâti't-Türk,
Kamûs-ı Türkî, Codex Cumanicus, Starling Altaic, Tietze, İSAM, Kubbealtı ve
donör dil kayıtlarından elle derlenmiş **toplam 59 kelimelik** çekirdek veri.
Her dosya kaynak künyesi (`_provenance`) taşır. Bu veri canlı bir servis
değildir ve öyle sunulmaz.

> Ölü uç noktalar (Glosbe API, TDK TTAS/Kişi Adları, DergiPark arama) portföyden
> **çıkarılmıştır**. Kaynak sayısını korumak için çalışmayan fetcher tutulmaz.

> Her kaynağın tam künyesi, kullanılan yöntemlerin makale referansları ve
> bilinçli olarak **kullanılmayan** kaynakların gerekçeleri için bkz.
> [Bilimsel kaynakça](#bilimsel-kaynakça).

## Mimari: hangi karar kimde

Tek bir ilke: **LLM karar vermez.**

| Katman | Ne yapar | Kim karar verir |
|---|---|---|
| Veri toplama | 9 canlı kaynak + 23 dilin yerel sözlük indeksi | — |
| Çeviri yazısı | Kiril/Arap/Runik → ortak karşılaştırma biçimi | kural |
| Çoklu hizalama | bütün tanıkları birbirine hizalar (LingPy SCA) | algoritma |
| Ata ses seçimi | tanısal denklik → tam kapsayan denklik → arkaiklik ağırlıklı oy | **sembolik** |
| Alıntı tespiti | zincir kanıtı + fonotaktik + ses kanunu + değişimsiz yayılım | **sembolik** |
| Hipotez sıralaması | rakip kökenler, red gerekçesi, karşıtsal açıklama | **sembolik** |
| Kalibrasyon | Platt ölçekleme + çekimserlik eşiği | **istatistik** |
| LLM | sözlük metnini şemaya döker, gerekçe metnini akıcılaştırır | **karar vermez** |

Gerekçe ölçüme dayanır. LLM'ler sözlük metnini şemaya dökmede güçlü
(F1 %93,6 — Jumashev ve ark. 2024), ama ses kanunu zinciri kurmada zayıf
(<%5 — PBEBench), alıntı ↔ miras ayrımında yanlı ("borrowing-blind",
F1 < 0,50 — Sousa Silva & Ahmadi 2026) ve hipotez yargıcı olarak insanla
ancak ~%66 tutarlı (ICLR 2025).

### Ata düğüm etiketi

Her rekonstrüksiyon hangi düğümü iddia ettiğini söyler:

- **`*PT`** — Proto-Türkçe. Çuvaşça (Oğur) tanığı var, rotasizm/lambdaizm
  türetilebiliyor.
- **`*PCT`** — Ana Ortak Türkçe. Oğur tanığı yok; o düğümde `*ŕ`/`*r`/`*z`
  ayrımı **zaten birleşmiştir**, dolayısıyla `*ŕ` yazmak veriden çıkmayan bir
  ayrımı iddia etmek olurdu.

Ölçüldü: 400 maddenin yalnız %28,7'sinde Oğur tanığı var.

`*PT` için gereken tanık **atteste** olmalıdır. Batı Eski Türkçe (`wot`,
Róna-Tas & Berta 2011) Oğurdur ama Macarcadaki alıntılardan geri
kurulmuştur; rekonstrüksiyondan rekonstrüksiyon türetip `*PT` yazmak
zincirleme belirsizliği tek bir iddianın arkasına saklamak olurdu. `wot`
tanısal (Lir-Şaz) kuralı ateşlemez, sıradan ağırlıklı tanık olarak sayılır
ve düğüm `*PCT`de kalır.

## Nasıl çalışır

```
kelime
  │
  ├─ 1. Morfolojik ayrıştırma  ──────────  utils/morphology, nlp/historical_morphology
  │        güzellik -> güzel +lIK ;  göz -> gö- + -z
  │
  ├─ 2. Paralel veri toplama  ───────────  18 fetcher, ThreadPoolExecutor
  │        varyantlar sınırlı (MAX_VARIANTS), her istek teşhis defterine yazılır
  │
  ├─ 3. Karşılaştırmalı rekonstrüksiyon ─  nlp/comparative_reconstruction
  │        konum duyarlı denklik kümeleri: söz başı d~t -> *t- ; söz sonu z~r -> *-ŕ
  │        güven = tanık sayısı + Türki kol çeşitliliği + sütun uyumu
  │
  ├─ 4. Alıntı keşif hattı  ─────────────  nlp/loanword_detector (4 katman)
  │        fonotaktik ihlaller -> lehçe yayılımı -> olasılık -> donör en-yakın-komşu
  │
  ├─ 5. A-HVP hakem protokolü  ──────────  nlp/hypothesis_validation_protocol
  │        4 aşama; kanıt üretemeyen aşama skordan DÜŞÜLÜR
  │
  └─ 6. Graf, CLDF, önbellek, LLM sentezi
```

### A-HVP: dört aşamalı hakem protokolü

| Aşama | Ağırlık | Ne ölçer | Kanıt yoksa |
|---|---|---|---|
| 1 · Fonetik zincir | %35 | Ata biçim ile modern biçim arasında düzenli ses denkliği | aşama düşülür |
| 2 · Kronoloji | %30 | Kaynak dil teması ilk tanıklamadan önce mi (anakronizm kilidi) | aşama düşülür |
| 3 · Semantik mesafe | %15 | Tarihsel anlam ile modern anlam arasındaki uzaklık | aşama düşülür |
| 4 · Akraba triangulation | %20 | Gerçek Türki dil karşılıklarının yayılımı ve kaynak çeşitliliği | aşama düşülür |

Skor, **yalnızca kanıt üretebilen aşamalara** normalize edilir. `evidence_coverage`
alanı kaç aşamanın konuşabildiğini bildirir; kapsam %50'nin altındaysa rozet en
fazla `⚪ YETERSİZ KANIT` olabilir.

```
göz    < *köŕ  (8 tanık, 5 kol)     -> 🟢 DOĞRULANDI — kısmi kanıt, %85 kapsam
kitap  < Arapça kitāb               -> 🟡 İNCELEME GEREKLİ
su     < Fransızca sous, 735 tanık  -> 🔴 REDDEDİLDİ (anakronizm)
zzzqx  < *zzzqx, kanıt yok          -> ⚪ YETERSİZ KANIT (%35 kapsam)
```

## Yapılandırma

Tüm ayarlar `engine/config.py` içinde toplanır ve `ETY_` önekli ortam
değişkenleriyle ezilebilir:

```bash
ETY_API_HOST=127.0.0.1 ETY_API_PORT=8000 python -m engine.server
ETY_MAX_VARIANTS=2 ETY_CACHE_ENABLED=false python -m engine.cli search göz
ETY_OLLAMA_MODEL=qwen2.5:7b python -m engine.cli search göz --ai
ETY_LOG_LEVEL=DEBUG python -m engine.cli search göz
```

## Geliştirme

```bash
make test        # ağsız test paketi (323 test, ~40 sn)
make test-live   # canlı kaynak testleri (ağ gerektirir)
make coverage    # kapsam raporu, %90 eşiği
make lint        # ruff
```

Testler **soket düzeyinde ağdan yalıtılır**: bir test yanlışlıkla canlı ağa
çıkmaya kalkarsa açık bir hata alır. Canlı kaynak testleri
`engine/tests/live/` altındadır ve yalnızca `ETY_LIVE=1` ile çalışır.

HTTP fixture'ları gerçek yanıtlardan kaydedilir:

```bash
python scripts/record_fixtures.py --live --word deniz --overwrite
```

## Harici veri kümesi içe alma

WOLD gibi CLDF veri kümelerinden alıntı kelime kayıtları alınabilir:

```bash
python -m engine.db.cldf_importer /path/to/wold-cldf --target-language tur
```

## Proje yapısı

```
engine/
  config.py                  merkezî yapılandırma (URL, timeout, eşikler)
  logging_setup.py           merkezî loglama
  search_engine.py           orkestrasyon, teşhis, önbellek
  server.py  cli.py          REST API ve komut satırı
  fetchers/                  18 veri toplayıcı + BaseFetcher sözleşmesi
  nlp/                       rekonstrüksiyon, hizalama, A-HVP, alıntı keşfi
  utils/                     fonotaktik, ortografi, morfoloji, HTTP istemcisi
  db/                        SQLite, graf, CLDF içe/dışa aktarım
  tests/                     332 test (9'u canlı ağ), fixture'lar, test ikizleri
data/
  seed/                      tohum veri (kaynak künyeli JSON)
  books/                     kullanıcı PDF'leri (tam metin taranır)
web/index.html               tek dosyalık statik panel
scripts/                     fixture kaydedici, veritabanı temizliği
```

## Bilinen sınırlar

Bunlar gizlenmiş kusurlar değil, **ölçülmüş ve raporlanmış** sınırlardır.

### Bilimsel

- **Motorun trivial taban çizgilerine üstünlüğü kanıtlanmış değil.**
  Rekonstrüksiyonda tek bölümlük dev ölçümünde (n=83) fark anlamlı çıkmıyor;
  5 katlı çapraz doğrulamada (`make eval-cv`, train+dev n=320) NED'de
  (−0,028, GA [−0,047, −0,009]) ve B-Cubed F'de (+0,031, GA [+0,012, +0,047])
  `majority_character`a karşı anlamlı, tam eşleşmede anlamlı değil
  (+0,019, GA [−0,013, +0,050]). Fark küçük ve dondurulmuş test bölümünde
  henüz doğrulanmadı; yayına gitmeden önce kapatılması gereken asıl açık budur.
- **Alıntı tespiti sözlük etiketine bağımlı.** Ablasyon ölçümü, bağımsız
  fonolojik sinyallerin ölçülebilir katkı sağlamadığını gösteriyor.
- **Öngörü sicili üçüncü tarafta kayıtlı değil.** Yerel zaman damgası ön-kayıt
  yerine geçmez; `external_doi` doldurulmadan sonuçlar "ön-kayıtlı çalışma"
  olarak sunulamaz.
- **Uzman değerlendirmesi yapılmadı.** Yeni etimoloji iddialarının en az iki
  Türkolog tarafından körlemesine puanlanması ve değerlendiriciler arası uyum
  (Cohen's κ) raporu henüz yok.
- **Ek soyma katmanının katkısı ölçülemedi** (net sıfır): CLDF biçimleri zaten
  çıplak köktür, sözlük madde başı değil.

### Veri

- **Altın standart tek okulun ürünü.** `savelyevturkic` Robbeets okulundandır
  ve kimi akrabalık kararları Clauson/Erdal/Tekin geleneğiyle çelişir.
  Altın standartlar arası uyuşmazlık oranı henüz ölçülmedi.
- **Altın kümeyi `robbeetstriangulation` ile büyütmek denendi ve geri
  alındı.** Deponun duran kararı (`scripts/download_cldf.py`: "YALNIZCA
  temas/ödünçleme analizinde kullanılır; akrabalık kanıtına asla katılmaz")
  ölçümle **doğrulandı**. Türki altküme (≥2 Türki tanık) süzülüp 236 madde
  eklenebiliyordu, ama veri altın standart olacak nitelikte değil:
  - `Root` alanına **alıntı işareti karışmış**: 190 farklı kök `… bor` ile
    bitiyor (`*adam bor`, `*ism bor`, `*kenar bor`) ve bunlar Arapça/Farsça
    alıntı. İşaret `Doubt`/`Source` sütunlarından ayırt edilemiyor.
  - **İşaretsiz alıntılar** da var: `*ḳïrmïzï`, `*šāχ` (Farsça). Motor
    `*ḳïrmïzï`yı "doğru" rekonstrüksiyon olarak tutturdu — yani alıntılar
    skoru düşürmüyor, **şişiriyor**.
  - Yazım gelenekleri karışık: aynı küme hem `ĺ/ŕ` hem `š/z` yazıyor,
    kökler `*ḳalï-` gibi biçimbirim tiresi taşıyor (139/293 madde).
  - `forms.csv`'deki `Loan` sütunu **boş** (26.224 biçimin tamamında), yani
    alıntılar veriden süzülemiyor.

  Ayrıca kazanç da gösterilemedi: yeni maddelerle mevcut maddeler arasındaki
  tam doğruluk farkı (0,398 vs 0,282) %95 GA [−0,064, +0,289] ile **anlamlı
  değil** ve bu farkı %80 güçle saptamak grup başına 263 madde isterdi —
  eklenebilecek toplam madde 236.
- **Oğur tanığı seyrek.** 400 maddenin yalnız %28,7'sinde Çuvaşça var; geri
  kalanda iddia edilebilen en derin düğüm Ana Ortak Türkçe'dir (`*PCT`).
- **Türkmence ünlü uzunluğu eksik.** Birincil uzunluk tanığı sayılan dil,
  `savelyevturkic`te yalnız 2 uzunluk işaretli biçim taşıyor.
- **Concepticon'da Türkçe kavram listesi yok.** Semantik köprü kendi
  verimizden türetildi ve Swadesh düzeyiyle (≈290 kavram) sınırlı; ayrıca
  yazılışa göre çalıştığı için eşadlılarda yanılır (`yüz` = surat/yüzmek/100).
- **Sözlük kapsamı sınırlı.** kaikki dökümlerinde Uygurca 4.215, Nogayca 484
  madde var; Kırım Tatarcası, Karaçay-Balkarca, Şorca, Eski Türkçe ve
  Osmanlıca için döküm hiç yok.
- Tohum veri yalnızca 59 kelimeyi kapsar.

### Teknik

- **Alıntı sınıflandırıcı kural tabanlıdır**, eğitilmiş bir ML modeli değildir.
- **Neo4j entegrasyonu yoktur.** `db/graph_database.py` Neo4j *şemasına uygun*
  düğüm/kenar yapısı üretir ama Cytoscape.js JSON'u olarak dışa verir.
- **Web paneli tek dosyalık statik HTML'dir**, Next.js kullanılmaz.
- **Sinirsel rekonstrüksiyon yapılmadı** — bilinçli kapsam kararı; mevcut bir
  yöntemi yeni veriye uygulamak deneydir, katkı değildir.

## Yeniden üretilebilirlik

Her sayı sıfırdan üretilebilir:

```bash
make install
make data              # 5 CLDF veri kümesi (sürüm + SHA-256 damgalı)
make lexicons          # 23 dilin kaikki dökümü (~60 MB)
make lexicon-index     # SQLite FTS5 arama indeksi
make gold              # altın standardı kur, böl, test setini MÜHÜRLE
make correspondences   # ses denkliklerini TRAIN kavramlarından öğren
make calibrate         # güven kalibratörünü TRAIN'de eğit
make semantic          # Türkçe kavram köprüsü
make chains            # alıntı zincirleri ve uyarlama kuralları

make eval-baseline     # motor vs trivial sistemler, 4 koşul, anlamlılık testi
make eval-cognates     # B-Cubed F
make eval-prediction   # ileri tahmin
make eval-calibration  # ECE + risk-coverage
make eval-borrowing    # WOLD + ablasyon
make eval-controls     # negatif kontrol bataryası
make dialect           # ağız kelimeleri toplu analizi

make test && make coverage
```

**Sızıntı önlemleri.** Test bölümü kavram bazında ayrılır, checksum'la
mühürlenir (`data/gold/SEAL.json`) ve açık onay verilmeden okunamaz. Bölme
deterministiktir — rastgele tohum yok, kavram kimliğinin SHA-256'sı kullanılır,
dolayısıyla her makinede aynıdır. Kalibratör ve denklik tabloları yalnız
`train` bölümünde öğrenilir; testler bunu denetler.

**Önbellek.** `ENGINE_VERSION` değiştiğinde eski kayıtlar otomatik olarak
ıskalama sayılır; düzeltilmiş hatalarla üretilmiş sonuçlar geri dönmez.

## Bilimsel kaynakça

Motorun kullandığı veya yol haritasında hedeflediği her veri kümesi, yöntem ve
kütüphane burada künyesiyle listelenir. **Kural: yeni bir kaynak eklendiğinde bu
bölüm aynı commit'te güncellenir.** Gerekçe — hangi kanıtın nereden geldiği
izlenebilir olmalı, ve bir yöntemin kaynağı belli olmazsa katkı ile transfer
ayırt edilemez.

Durum işaretleri: ✅ kullanımda · 🚧 yol haritasında · ⚠️ bilinçli olarak
kullanılmıyor veya kısıtlı kullanılıyor.

### Uzman değerlendirmesi protokolü (Faz E2) — kurulu, henüz koşulmadı

Altın standart yalnız **bilinen** kümeleri kapsıyor; yeni iddiaların değeri
ancak Türkologlara sorularak ölçülebilir. Protokolün ölçüm tarafı kuruldu:

- **Anket formu** `data/expert_review/review_sheet.json` — dev bölümünden
  82 gerçek iddia, her biri **gerekçesiyle**. ⚠️ Yalnız sonucu sormak
  (`göz → *köŕ`) uzmanı motorun akıl yürütmesini görmeden karar vermeye
  zorlar; ölçülen şey iddia değil uzmanın kendi bilgisi olurdu.
- **Ölçek** 0–5 ordinal ("açıkça yanlış" … "kesin").
- **Uyum katsayısı** Krippendorff **ordinal α** + bootstrap %95 GA.

⚠️ **Metrik Cohen κ DEĞİLDİR.** κ iki kodlayıcı ve **nominal** ölçek
içindir; burada 2–3 kodlayıcı ve **ordinal** ölçek var. κ iki ayrı hata
yapardı: "0 vs 5" ile "3 vs 4" uyuşmazlığını aynı sayardı, ve üçüncü
kodlayıcıyı doğrudan alamazdı (çiftler ortalanırsa güven aralığı bozulur).
Krippendorff α ikisini de çözer (Artstein & Poesio 2008).

⚠️ Nokta tahmin tek başına yetmez: n≈100 maddede α oynaktır, bootstrap
aralığı zorunludur (Zapf ve ark. 2016).

⚠️ **Uzman girdisi yoksa hiçbir sayı üretilmiyor.** `make expert-review`
derecelendirme dosyası bulamazsa hata verip çıkıyor. Uydurma
derecelendirme, hiç değerlendirme yapmamaktan kötüdür.

### Ön-kayıt (Faz E3) — taslak hazır, kayıt yapılmadı

`docs/PREREGISTRATION.md`: hipotezler, **karar eşikleri**, sızıntı
önlemleri ve "ne raporlanmayacak" listesi veriye bakılmadan yazıldı.

⚠️ Kayıt OSF hesabı gerektiriyor ve bu depodan yapılamaz. Yapıldığında DOI
hem belgeye hem `prediction_test.PredictionRegistry.external_doi` alanına
yazılacak.

Şablon: OSF **Secondary Data Preregistration** — çalışma mevcut,
yayımlanmış veriyle yapılıyor; veri toplanmadan önce kayıt yapılan
şablonlar bu senaryoya uymuyor.

⚠️ H5 (uzman kabulü) için **ön koşul** kayda yazıldı: kodlayıcılar arası
uyum α < 0,60 ise ortalama derece **yorumlanmaz**, hipotez "test edilmemiş"
sayılır.

### Yayın çıktısı — SIGTYP 2022 biçimi

Altın standardımız alanın yayınlanmış paylaşılan görev biçiminde dışa
aktarılıyor (`make sigtyp`), ki başka sistemler aynı veride ölçülebilsin ve
buradaki sayılar **bağımsız olarak yeniden üretilebilsin**:

    data/sigtyp/
        cognates.tsv           400 akraba kümesi · 32 dil · ata sütunu dahil
        training-0.10…0.50.tsv kız dil hücrelerinin %10…%50'si gizli
        test-0.10…0.50.tsv     gizlenen hücreler "?" ile işaretli
        solutions-0.10…0.50.tsv gizlenen hücrelerin gerçek değerleri
        _provenance.json       her dosyanın SHA-256'sı ve bölme tohumu

⚠️ Bölmeler **deterministiktir** (`SPLIT_SEED = 20260827`); aynı komut aynı
bölmeleri üretir. Yoksa "aynı veride ölçtük" iddiası kurulamaz.

⚠️ **Bilinen kısıtlar, veriyle birlikte gidiyor:** bölütleme harf
düzeyindedir (IPA çok-karakterli bölütleri `t͡ʃ` korunmaz); gizlenen hücreler
yalnız kız dillerden seçilir ve ata sütunu her zaman görünürdür (ST2022
görevi "eksik **refleksi** tahmin et"tir, "ata biçmi tahmin et" değil); her
satırda en az bir kız hücre görünür kalır.

⚠️ ST2022'ye dış veri PR ile eklenmemiş; organizatörler veri kümelerini
Lexibank'tan derlemiş. Bu bir **veri yayınıdır**, katkı önerisi değil.
Gerçekçi rota: Lexibank uyumlu CLDF + Zenodo DOI, ST2022 bölmelerini ondan
türetmek.

### Veri kümeleri

| Kaynak | İçerik | Durum |
|---|---|---|
| Savelyev & Robbeets 2020, *Journal of Language Evolution* — `savelyevturkic` | 8.360 biçim · 32 Türki dil · 8.360 uzman akrabalık kararı · 905 küme · 519 ata biçim · 478 uzun ünlülü biçim | 🚧 birincil altın standart |
| Hruschka ve ark. 2015, *Current Biology* — `hruschkaturkic` | 4.213 biçim · 27 dil | 🚧 bağımsız çapraz kontrol |
| Starostin, Dybo & Mudrak, *Altaic Etymological Dictionary* — `starostinaltaic` | 5.756 biçim · 55 dil | ⚠️ yalnız karşılaştırma; Vovin 2005 eleştirisiyle birlikte anılır, tek kaynak olarak kullanılmaz |
| Robbeets & Bouckaert — `robbeetstriangulation` | 26.224 biçim · 102 dil | ⚠️ yalnız **temas** çerçevesinde; akrabalık kanıtına katılmaz (Tian ve ark. 2022 eleştirileri). **Ölçümle doğrulandı**: altın kümeye alınması denendi ve geri alındı — gerekçe «Bilinen sınırlar → Veri» bölümünde |
| Róna-Tas & Berta 2011, *West Old Turkic: Turkic Loanwords in Hungarian* — `ronataswestoldturkic` ([loanwordbank](https://github.com/loanwordbank/ronataswestoldturkic), CC-BY) | 1.755 biçim · 430 kavram · 480 Oğur (Bolgar, `bolg1249`) biçimi | ⚠️ **atteste değil**, Macarcadaki alıntılardan geri kurulmuş; ayrı tanık kodu (`wot`), tek başına `*PT` taşımaz. Ölçüldü: kazanç yok, **varsayılan kapalı** |
| [kaikki.org](https://kaikki.org) — Wiktionary makine-okunur dökümleri | 23 Türki dil · 125.879 madde · ~856 MB ham (diskte 60 MB) | ✅ arama indeksi · ⚠️ **altın standart değil** (bkz. Häuser & Stamatakis 2025) |
| kaikki.org — **tarihî katman** (URL kalıbı düzeltilince erişilebildi) | Osmanlı Türkçesi 9.806 · Kırım Tatarcası 4.780 · Güney Altayca 1.914 · **Eski Türkçe 470** (Orhun runik) | ✅ arama indeksi · ⚠️ Eski Türkçe dökümü küçüktür; runik biçimler `transliterate_to_latin` ile Latin karşılaştırma biçimine çevrilir |
| kaikki.org — **Rusça Wiktionary sürümü** (10 Türki dil, ~2,8 MB) | Karayca 5.793 · Kırım Tatarcası 3.497 · Çuvaşça 2.074 · Başkurtça 4.282 · Yakutça 4.285 · Şorca 557 | ✅ tanık ve arama verisi · ⚠️ şemada `etymology_templates` **yok** (köken çıkarılamaz), anlamlar Rusça (anlam kısıtlı verici sinyali çalışmaz) |
| TDK Güncel Türkçe Sözlük + Nişanyan Sözlük | Türkçe altın alıntı kümesi | ✅ **Wiktionary'den bağımsız** ikinci alıntı ölçütü · ⚠️ kanıt kuralı asimetrik (bkz. Alıntı tespiti) |
| [NorthEuraLex](https://northeuralex.org) v4.1 (CLDF) | 8 Türki dil (Çuvaşça dahil) · ~1.000 kavram | ✅ kavram hizalı tanık (`NorthEuraLexFetcher`) · ⚠️ aynı kavram akraba değildir: aday sorguyla ya da ses denklikleriyle tahmin edilen biçimle benzerlik ≥ 0,50'den geçer · NorthEuraLex `khk` = Halha Moğolcası, motorda Hakasça |
| WOLD — World Loanword Database | uzman alıntı derlemesi | ✅ alıntı değerlendirmesinin birincil ölçütü |
| kaikki.org — **Türkçe Wiktionary sürümü** (`download_lexicons.py --tr`) | Türkçe ~210 bin madde (çekim sayfaları ayıklandı) · Osmanlıca 8.570 · Azerice, Tatarca, Özbekçe… | ✅ yalnız tanık ve arama verisi · ⚠️ köken kategorileri **okunmaz**: Nişanyan/TDK kaynaklı, Türkçe altın kümeyle döngüsellik |
| kaikki.org — Rusça sürüme 2026-09-24'te eklenen 8 dil | Kazakça, Tatarca, Azerice, Özbekçe, Kırgızca, Türkmence, Uygurca, Gagavuzca (~54 bin) | ✅ tanık ve arama verisi (anlamca doğrulanır, alt sınır 0,50) |
| kaikki.org — Eski Uygurca, Eski Anadolu Türkçesi, **Proto-Türkçe** | 446 · 593 · 1.150 madde | ✅ ilk ikisi tanık dili (yayılım paydasına girmez) · Proto-Türkçe sayfaları indekse **girmez**; kök torunları olarak okunur (`LocalProtoTurkicFetcher`) |
| [Starling](https://starlingdb.org) Türk etimolojisi (`turcet`, Dybo & Starostin 2005, `make starling`) | 2.017 Proto-Türkçe kök · 32 dil alanı · EDT/ЭСТЯ atıfları | ✅ başlık kökü (kaynak biçim yoksa) ve tarihli Eski Türkçe tanık (Orhun 732, DLT 1072) · ⚠️ ölçüldü: örüntü tablosuna ek eğitim verisi olarak **kazanç yok**; Moğolca tablosu verici indeksinde WOLD'u **bozdu**, kullanılmıyor |
| [Apertium](https://github.com/apertium) iki dilli sözlükleri (`make apertium`) | Türkçe ↔ Çuvaşça, Kırım Tatarcası, Kırgızca, Tatarca, Özbekçe, Azerice, Türkmence | ✅ tanık (`ApertiumFetcher`) · ⚠️ **çeviri karşılığıdır**, akraba değil (pencere ~ терезе): benzerlik ≥ 0,50, Çuvaşça ≥ 0,66 |
| kaikki **verici dili** dökümleri — Rusça, Moğolca, Evenkice, Arapça, Farsça, Yunanca, Ermenice, Fransızca, İtalyanca | 1.674.418 madde, ~352 MB | ✅ verici yakınlığı sinyali · ⚠️ Türki arama indeksinden **AYRI** dosyada; karışsalardı Rusça `море` Türki akraba adayı olarak dönerdi |
| DatSemShift | 10.565 anlam kayması | 🚧 semantik makullük |
| CLICS⁴ | 3.447 dilde eş-adlandırma | 🚧 semantik makullük |
| [Concepticon](https://concepticon.clld.org) · [Glottolog](https://glottolog.org) · [Lexibank](https://lexibank.clld.org) · [CLTS](https://clts.clld.org) | kavram kimliği, dil kimliği, sözvarlığı, fonetik gösterim | 🚧 standart katman |
| [SIGTYP ST2022](https://github.com/sigtyp/ST2022) (refleks tahmini) · [ST2023](https://github.com/sigtyp/ST2023) (akraba/türev tespiti) | shared task verisi ve metrikleri | 🚧 Türki verisi eklenecek (kaynak katkısı) |
| TDK Güncel Türkçe Sözlük · Tarama · **Derleme** | çağdaş, tarihî ve **ağız** sözvarlığı | ✅ canlı kaynak · ⚠️ Derleme hedef sözvarlığıdır ama motorun ölçülen katma değeri orada **pratikte sıfırdır** (bkz. «Ağız kelimeleri») |
| Etymological Wordnet (de Melo, LREC 2014) · [EtymDB-2.0](https://github.com/clefourrier/EtymDB) (Fourrier & Sagot, LREC 2020) | 1,8M sözlükbirim · 2.536 dil | ⚠️ zinciri *saklıyor*, çıkarsamıyor — karşılaştırma noktası |

**Yerel öncelik (2026-09-24).** Canlı İngilizce Wiktionary, 14 Türk dili Wiktionary'si ve Archive.org varsayılan olarak **kapalı**; aynı veri yukarıdaki yerel dökümlerde. Canlı kalanlar yalnız TDK (3 sözlük), Nişanyan ve EtimolojiTürkçe: bunların cevapları kalıcı yerel önbelleğe (`data/cache/http.db`, 90 gün) yazılır ve erişilemeyen sunucuya devre kesici 5 dakika istek atmaz. Denenip reddedilen: OTC Osmanlıca derlemi (dosya tarihleri güvenilmez: *Seyahatname* 1611 = Evliya Çelebi'nin doğum yılı).

### Standartlar ve kütüphaneler

- **CLDF** (Cross-Linguistic Data Formats) — Forkel ve ark., *Scientific Data*; iç veri formatı
  · `pycldf` · `pyconcepticon` · `pyclts` · `pysem` · `cltoolkit` · `pylexibank`
- **[LingPy](https://lingpy.org)** — SCA, LexStat, Infomap kümeleme, B-Cubed değerlendirme
- **[LingRex](https://github.com/lingpy/lingrex)** — ses karşılık örüntüleri (CoPaR), alignment trimming, refleks tahmini
- **[seabor](https://github.com/lingpy/seabor)** — aileler arası alıntı tespiti (F = 0,87)
- **[lingreg](https://codeberg.org/calc/lingreg)** — leave-one-out düzensizlik tespiti
- **[EDICTOR 3](https://aclanthology.org/2024.lchange-1.1.pdf)** — akraba kümesi düzenleme arayüzü
- **PanPhon** (6.367 IPA segmenti, 24 özellik) · **Epitran** — fonolojik özellik vektörleri ve çeviri yazısı
- **contacTrees** (BEAST2) · **DiaSim** — filogeni ve ses değişimi simülasyonu
- **[cmu-llab/dpd](https://github.com/cmu-llab/dpd)** — DPD-BiReconstructor kodu ve checkpoint'leri

### Akraba tespiti, hizalama ve rekonstrüksiyon

- List, Greenhill & Gray 2017, *PLOS ONE* — LexStat-Infomap; **B-Cubed F ≈ 0,89** taban çizgisi
- List 2019, *Computational Linguistics* — değerlendirme metrikleri; **salt edit distance reddi** (ED + NED + B-Cubed F + accuracy dörtlüsü)
- **List 2019, *Computational Linguistics* — CoPaR** ⚠️ **teşhis olarak uygulandı, karar katmanına bağlanmadı** — denklik örüntülerini asgari klik örtüsüyle veriden çıkarır. Plan bunu elle yazılmış `CORRESPONDENCES` yerine koymayı öngörüyordu; ölçüm planı değiştirdi (elle yazılmış tablo öğrenilmiş sayımı geçti). Teşhis sonucu: TRAIN'de 1.565 sütunun **%71,3'ü** düzenli örüntülere oturuyor, 762 örüntünün 450'si tekil. Bu oran kural tabanlı rekonstrüksiyonun üst sınırıdır.
- **Blum & List 2023, `lingrex.trimming`** ✅ **uygulandı** — boşluk-yönelimli hizalama budaması; yayında 10 ailenin 10'unda düzenli denklik oranını artırıyor (+0,03…+0,07). Rekonstrüksiyon **doğruluğuna** etkisi yayınlanmamıştı; bizde dev'de tam 0,361 → **0,386**, NED 0,306 → 0,302, ED 1,48 → 1,45.
- Blum & List 2026 — leave-one-out düzensizlik tespiti (%85). ⚠️ Teşhisi sınandı ve doğrulanmadı: denklik tablosunu yalnız miras kümelerden öğrenmek `ses_kanunu_ihlali` sinyalini kurtarmadı (−0,0101 → −0,0083).
- [Bouchard-Côté ve ark. 2013, *PNAS*](https://www.pnas.org/doi/10.1073/pnas.1204678110) — olasılıksal ses değişimi modeli, 637 Austronesian dili
- Meloni ve ark. 2021 · Kim ve ark. 2023 (ACL) — Transformer rekonstrüksiyon taban çizgileri (%53 Roman / %39,5 Sinitik, 8.799 eğitim örneğiyle)
- Lu, Xie & Mortensen 2024 (**ACL 2024 Best Paper**) — DPD-BiReconstructor, yarı-denetimli rekonstrüksiyon
- **[Lu, Wang & Mortensen 2024 (LREC-COLING)](https://arxiv.org/abs/2403.18769)** ⚠️ **uygulandı, KAZANÇ YOK** — refleks tahminiyle N-best yeniden sıralama (P2D); yayında dört veri setinde +0,9…+3,1 puan. Bizde konsensüsü geçemedi (P2D 0,361 vs konsensüs 0,434). Ata dil denklik tablolarına sözde dil olarak katıldı (`pt`), yine yetmedi. N-best oracle 0,506 — tavan var ama sıralayıcı ulaşamıyor. Adaylar `alternative_forms` olarak sunuluyor, seçilen biçim değişmiyor.
- Akavarapu & Bhattacharya 2023/2024 — Cognate Transformer (MSA Transformer, çapraz-aile ön-eğitim) · Cui ve ark. 2024
- [List ve ark. 2023 (LChange @ EMNLP)](https://aclanthology.org/2023.lchange-1.3/) — fonolojik rekonstrüksiyonda belirsizlik gösterimi (`*[p a|i t]`)
- ⚠️ Häuser & Stamatakis 2025 — Wiktionary/BabelNet'ten kazınan akraba kümelerinin altın standart ağaçlarla tutarsızlığı
- ⚠️ Häuser 2024 — filogenide leksikal akraba ağaçları ses denkliklerinden ~1/3 daha isabetli

### Alıntı tespiti, temas ve geçiş yolu

- **[List & Forkel 2022 — `seabor`](https://pmc.ncbi.nlm.nih.gov/articles/PMC10445856/)** ⚠️ **kısmen uygulandı, DOĞRULANAMADI** — ağaç-uyumsuz dağılımla otomatik alıntı tespiti, yayınlanmış **F = 0,87** (aileler arası). Bizde Türki-içi alıntı etiketi olmadığı için doğruluğu ölçülemedi (tek kaynak WOLD/Sakha'da 18 madde). Modül aday üreteci olarak duruyor, karar katmanına bağlı değil (`USE_AS_SIGNAL = False`); 400 kümenin 28'ini işaretliyor.
- **[Miller ve ark. 2020, *PLOS ONE* — PyBor](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0242709)** ✅ **uygulandı** — miras ve alıntı için ayrı dizilim modeli, karar log olasılık farkıyla. Yayınlanmış WOLD 41 dil ortalaması **F1 0,59–0,61**. Bizde Sakha'da tek başına **F 0,558** (bağımsız yeniden üretim); elle yazılmış fonotaktik kural aynı veride 0,215. Markov(3-gram) + Witten-Bell; LSTM sürümü yayında 0,61.
- **[Miller & List 2023, EACL — `sabor`](https://arxiv.org/pdf/2302.00189)** ✅ **uygulandı** — verici dil sözlüğüne SCA yakınlığı; yayınlanmış **F1 0,806 · kesinlik 0,931**. Bizde WOLD/Sakha'da motorun F'sini **0,385 -> 0,644** çıkardı; sinyal ablasyonundaki katkısı **+0,237**. ⚠️ Yayınlanmış kurulum **kavram kısıtlıdır** ve makale kaçan alıntıların %45'ini bu kısıta bağlıyor.
- [Neureiter ve ark. 2022, *Humanit Soc Sci Commun*](https://www.nature.com/articles/s41599-022-01211-7) — contacTrees
- Hruschka ve ark. 2015 — concerted evolution, düzenli ses değişimi tespiti
- ⚠️ [Dellert 2019, *Information-theoretic causal inference of lexical flow*](https://langsci-press.org/catalog/book/233) — leksikal akışın **yönünü** verir, **yolunu vermez**; dil düzeyinde, kelime düzeyinde değil
- [Haynie ve ark. 2014](https://www.sciencedirect.com/science/article/pii/S2215039014000022) — Wanderwörter (betimsel; otomatik yayılım-yolu algoritması yok)
- [Pantaleo ve ark. 2017](https://wikiworkshop.org/2017/papers/p1635-pantaleo.pdf) — etytree, 6M girdilik etimoloji grafı
- [Kovácsová 2025, MA tezi (Charles Univ.)](https://dspace.cuni.cz/handle/20.500.11956/203152) — aracı dil dizisi çıkarımını hedefleyip "sözlük açıklamalarının karmaşıklığı nedeniyle zorlu çıktı" diye raporluyor

### Alıntı uyarlama fonolojisi

- **Tsvetkov, Ammar & Dyer 2015 (NAACL), "Constraint-Based Models of Lexical Borrowing"** — tanıklanmış verici–alıntı çiftlerinden **OT kısıt sıralaması öğreniyor** (Arapça→Svahili). Bu alandaki kural indüksiyonunun referans noktası.
- [Mao & Hulden 2016 (COLING)](https://aclanthology.org/C16-1081/) — Japonca alıntı uyarlaması; FST gramerini **elle yazıp** değerlendiriyor
- Smith 2024, *Loanword Phonology* (Cambridge Handbook of Phonology 2) — betimsel/kuramsal genel bakış
- [al-Hashmi 2016 (Leeds doktora tezi)](https://etheses.whiterose.ac.uk/id/eprint/20807/) — Türkçedeki Arapça alıntıların fonetik/fonolojisi
- [Zaval, İhsanoğlu, Ersoy & Yıldız 2025 (ABJADNLP @ NAACL)](https://aclanthology.org/2025.abjadnlp-1.4/) — Turkish Morpholex'te Arapça kökenli kelimelerin elle anotasyonu

### Semantik değişim

- Xu ve ark. 2023 — **somut→soyut** yönü dillerin %90'ında en iyi yordayıcı
- Rzymski ve ark. — CLICS · Zalizniak ve ark. — DatSemShift
- ⚠️ Dubossarsky ve ark. — diakronik word embedding'lerin büyük ölçüde model artefaktı olması. **Bu yüzden motorda kullanılmıyor** (Türki için tarihlendirilmiş derlem de yok).

### Öngörü testi ve yanlışlanabilirlik

- Bodt & List 2019, *Papers in Historical Phonology* 4:22–44 · [Bodt & List 2022, *Diachronica* 39(1)](https://www.jbe-platform.com/content/journals/10.1075/dia.20009.bod) · [kod/veri](https://github.com/lingpy/prediction-study) — ön-kayıtlı refleks tahmini, ~%70 isabet
- [Blum, Barrientos, Ingunza & List 2024, *Scientific Reports* 14:30636](https://www.nature.com/articles/s41598-024-82515-3) — Registered Report; 206 tahmin, 41 saha doğrulaması (OSF: 10.17605/OSF.IO/FGBM7)
- [Huang ve ark. 2025 (ICML) — POPPER](https://proceedings.mlr.press/v267/huang25n.html) · [kod](https://github.com/snap-stanford/POPPER) — otomatik falsifikasyon, sıralı test ile Tip-I hata kontrolü
- [Liu ve ark. 2024 — AIGS](https://arxiv.org/abs/2411.11910) — FalsificationAgent mimarisi
- King ve ark. — Robot Scientist Adam/Eve; kapalı döngü hipotez testi
- [Apéritif (CHI 2022)](https://dl.acm.org/doi/fullHtml/10.1145/3491102.3517707) — ön-kayıt iskeletleme

### İstatistiksel geçerlilik, kalibrasyon ve açıklanabilirlik

- **Kessler 2001, *The Significance of Word Lists* (CSLI)** ✅ **uygulandı** — permütasyon testiyle **rastlantısal benzerlik** kontrolü. Verici yakınlığı sinyalinde null model olarak koşuyor: aynı havuza karşı 12 kontrol kelimesi ölçülüyor. Ham mesafe eşiği havuz büyüklüğüne gizlice bağlıdır (Sakha 448.000 madde, Türkçe 1.600.000); null modelsiz aynı eşik iki dilde aynı şeyi ölçmez. Ölçüldü: denetim motorun F'sini **0,584 -> 0,644** çıkardı.
- **List, Walworth, Greenhill, Tresoldi & Forkel 2018, *Journal of Language Evolution*** ✅ **boşluk dolduruldu** — akraba kümesi kararlarında **uzmanlar arası uyumu sistematik ölçen bir çalışma olmadığını** açıkça ilan ediyor ("no systematic study has been carried out so far"). Bu depo o ölçümü yapıyor: `savelyevturkic` × `hruschkaturkic`, 1.833 ortak öğe, 26 dil → **B-Cubed F 0,914 · ARI 0,912**. ⚠️ Kavram köprüsü kurulamadığı için (`hruschkaturkic`te Concepticon glossu yok) köprü öğe düzeyinde kuruldu; band aşağı yanlıdır. Kod: `engine/evaluation/gold_agreement.py`.
- Bagga & Baldwin 1998 — B-Cubed · Hubert & Arabie 1985 — Ayarlanmış Rand İndeksi (şansa göre düzeltilmiş bölümleme uyumu; düzeltilmemiş Rand şansla bile 0,9'un üstüne çıkar)
- Benjamini & Hochberg 1995 — FDR kontrolü (çoklu karşılaştırma)
- Efron & Tibshirani — bootstrap güven aralığı · McNemar testi
- Zadrozny & Elkan — izotonik kalibrasyon · Platt scaling · Kull ve ark. — beta kalibrasyon
- King 2026, *PLOS Computational Biology* — Bayesçi filogenilikte simülasyon tabanlı kalibrasyon (SBC)
- [Hariharan & Mortensen 2025](https://arxiv.org/abs/2512.05364) — tarihsel dilbilimde ECE raporlaması (diakronik değişim tespitinde)
- **Artstein & Poesio 2008, *Computational Linguistics* 34(4)** ✅ **uygulandı** — kodlayıcılar arası uyumda metrik seçimi; **Krippendorff ordinal α**, Cohen κ'nın iki kısıtını (iki kodlayıcı, nominal ölçek) aşar. Ordinal fark fonksiyonu "0 vs 5" uyuşmazlığını "3 vs 4"ten ağır sayar. Kod: `engine/evaluation/expert_review.py`.
- **Zapf ve ark. 2016, *BMC Medical Research Methodology*** ✅ **uygulandı** — α için **bootstrap güven aralığı**; n≈100 maddede nokta tahmin oynaktır. ⚠️ Yeniden örnekleme madde düzeyinde yapılır, derecelendirme düzeyinde değil.
- [Krarup ve ark., *JAIR*](https://arxiv.org/abs/2103.15575) — karşıtsal plan açıklaması ("neden bu değil de şu?") · Miller — contrastive explanation (fact vs. foil)
- [Lawrence & Reed 2019, *Computational Linguistics* 45(3)](https://direct.mit.edu/coli/article/45/3/603/93385/Argumentation-Mining) · [Stede & Schneider — survey](https://direct.mit.edu/coli/article/45/4/765/93362/Argument-Mining-A-Survey) — argümantasyon madenciliği

### LLM'in yeri: nerede güçlü, nerede kullanılmaz

Motorun mimari ilkesi ölçüme dayanır: **LLM = veri girişi + normalizasyon +
post-correction + gerekçe metni. Karar = sembolik/istatistiksel katman.**
Rekonstrüksiyon, akrabalık ve alıntı-mı-miras-mı kararlarını LLM **vermez**.

| Güçlü olduğu | Ölçüm |
|---|---|
| Sözlük metnini şemaya dökme | F1 %93,6 — Jumashev ve ark. 2024 (AIST), Kırgızca sözlük |
| Açık ses kanunu kuralı üretme (kara kutu yerine) | [Naik ve ark. 2025 (ACL)](https://arxiv.org/abs/2501.16524) |
| Aday üretimi (recall artırma) | [AlphaGeometry, *Nature* 2024](https://www.nature.com/articles/s41586-023-06747-5): sembolik 14/30 → +LLM **25/30** |

| Zayıf olduğu | Ölçüm |
|---|---|
| Ses kanunu zinciri kurma | [PBEBench](https://arxiv.org/html/2505.23126): zor örneklerde **<%5** |
| Alıntı ↔ miras ayrımı | [Sousa Silva & Ahmadi 2026 (LREC)](https://arxiv.org/html/2510.26254): "borrowing-blind", F1 < 0,50 · ince ayarlı XLM-R 0,851 |
| Fonolojik akıl yürütme | PhonologyBench: insana göre −%17 / −%45 |
| Hipotez yargıcı | ICLR 2025: insanla ~%66 tutarlılık, 12 yanlılık türü |

Aday üretiminin katkısını ölçme şablonu:
[recall↑/precision↓ örüntüsü](https://arxiv.org/pdf/2505.14599) ·
[Si, Yang & Hashimoto 2024](https://arxiv.org/abs/2409.04109) (LLM fikirleri daha
özgün, daha az uygulanabilir) · [Ideation-Execution Gap](https://arxiv.org/pdf/2506.20803).

### Türkoloji filolojisi

**Kaynak ağırlık sırası:** Clauson → ESTJa (Sevortjan) → Räsänen → Doerfer →
Tietze → Eren. EDAL yalnızca **destekleyici** kanıt olarak, tek kaynak olarak
değil (İngilizce Wiktionary'nin de uyguladığı politika).

- Clauson, *An Etymological Dictionary of Pre-Thirteenth-Century Turkish* (1972), 9.250 madde — yapılandırılmış veritabanı sürümü **yok**; çıkarılması yol haritasında
- Sevortjan ve ark., *Этимологический словарь тюркских языков* (ЭСТЯ)
- Räsänen · Doerfer · Tietze · Eren · Kâşgarlı Mahmud, *Divânu Lugâti't-Türk* · Şemseddin Sâmi, *Kamûs-ı Türkî* · Codex Cumanicus
- Dybo 2015 — Proto-Türkçe ünlü uzunluğu
- ⚠️ Vovin 2005 — Altay hipotezi eleştirisi · Tian ve ark. 2022 — Robbeets 2021 Transeurasian verisine eleştiriler
- Savelyev 2022 — Türki'de yoğun temasın leksikostatistiği bozması

### Yöntemsel duruş

- **Rotasizm/zetasizm** çözülmemiş bir bilimsel tartışmadır; motor taraf tutmaz, **her iki rekonstrüksiyonu da** raporlar.
- Çuvaşça (Oğur) tanığı olmadan çıkan biçim Proto-Türkçe değil **Ana Ortak Türkçe**'dir; çıktı hangi düğümü iddia ettiğini etiketler.
- "Yetersiz kanıt" bir başarısızlık değil, **dürüst sonuçtur**.

## Lisans

MIT — bkz. [LICENSE](LICENSE).
