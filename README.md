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

### Rekonstrüksiyon (dev bölümü, n=83)

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
| **motor** | **0,302** | **0,595** | **1,45** | **0,386** |
| `majority_character` (trivial) | 0,341 | 0,571 | 1,60 | 0,325 |
| `copy_anchor` (hiçbir şey yapma) | 0,392 | 0,643 | 1,88 | 0,217 |
| `copy_random_daughter` | 0,403 | 0,628 | 1,86 | 0,229 |
| `copy_longest` | 0,502 | 0,564 | 2,58 | 0,145 |

Birincil metrikte fark **−0,0387** (motor lehine), %95 GA [−0,0786,
+0,0017] → **anlamlı DEĞİL**; aralık sıfırı kılpayı içeriyor. n=83'te daha
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

#### ⚠️ Denenmiş ve KAZANÇ VERMEYEN iki şey

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

⚠️ **Çekimserlik bedava değildir.** Cevaplanmayan madde ortalamaya mümkün
olan en kötü NED'i (1,0) katar. Bir dönem yalnızca cevaplanan maddeler
ortalanıyordu; o muhasebe cevap vermemeyi kusursuz cevap vermekle bir
tutuyor ve çekimser kalmayı ödüllendiriyordu.

### Akraba tespiti (B-Cubed F, dev kavramları)

| Sistem | F | kesinlik | duyarlılık |
|---|---|---|---|
| ayarlı düzenleme uzaklığı | **0,931** | 0,943 | 0,927 |
| SCA benzeri (LingPy) | 0,859 | | |
| **motorun kümeleyicisi** | 0,813 | 0,981 | 0,727 |
| hepsi tek küme (trivial) | 0,743 | | |

Referans: LexStat-Infomap **F ≈ 0,89** (List, Greenhill & Gray 2017).
⚠️ Motorun kendi kümeleyicisi aşırı muhafazakâr: kesinlik 0,98 ama
duyarlılık 0,73.

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

⚠️ Ablasyon hükmü **ilk kez pozitif ve anlamlı**: motor vs yalnız
fonotaktik fark **+0,069**, %95 GA [+0,025, +0,112], p=0,0038.

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

### Ağız kelimeleri (Faz 10, n=100)

| | |
|---|---|
| çözüldü | %1 |
| güçlü aday | %13 |
| **yetersiz kanıt** | **%86** |

⚠️ "Yetersiz kanıt" bir başarısızlık değil, **dürüst sonuçtur**. Ağız
kelimelerinin çoğu tek bir ilde tanıklanmıştır ve karşılaştırmalı yöntemin
gerektirdiği bağımsız tanık yoktur.

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
ve 19 Türki dilin kaikki dökümü (108.708 madde, SQLite FTS5 indeksi). Her
indirme sürüm, tarih ve SHA-256 damgası taşır (`data/SOURCES.md`).

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
| Veri toplama | 9 canlı kaynak + 19 dilin yerel sözlük indeksi | — |
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
  Rekonstrüksiyonda dört koşulun hiçbirinde fark istatistiksel olarak anlamlı
  çıkmıyor. Bu, yayına gitmeden önce kapatılması gereken asıl açıktır.
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
make lexicons          # 19 dilin kaikki dökümü (~56 MB)
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

### Veri kümeleri

| Kaynak | İçerik | Durum |
|---|---|---|
| Savelyev & Robbeets 2020, *Journal of Language Evolution* — `savelyevturkic` | 8.360 biçim · 32 Türki dil · 8.360 uzman akrabalık kararı · 905 küme · 519 ata biçim · 478 uzun ünlülü biçim | 🚧 birincil altın standart |
| Hruschka ve ark. 2015, *Current Biology* — `hruschkaturkic` | 4.213 biçim · 27 dil | 🚧 bağımsız çapraz kontrol |
| Starostin, Dybo & Mudrak, *Altaic Etymological Dictionary* — `starostinaltaic` | 5.756 biçim · 55 dil | ⚠️ yalnız karşılaştırma; Vovin 2005 eleştirisiyle birlikte anılır, tek kaynak olarak kullanılmaz |
| Robbeets & Bouckaert — `robbeetstriangulation` | 26.224 biçim · 102 dil | ⚠️ yalnız **temas** çerçevesinde; akrabalık kanıtına katılmaz (Tian ve ark. 2022 eleştirileri) |
| Róna-Tas & Berta 2011, *West Old Turkic: Turkic Loanwords in Hungarian* — `ronataswestoldturkic` ([loanwordbank](https://github.com/loanwordbank/ronataswestoldturkic), CC-BY) | 1.755 biçim · 430 kavram · 480 Oğur (Bolgar, `bolg1249`) biçimi | ⚠️ **atteste değil**, Macarcadaki alıntılardan geri kurulmuş; ayrı tanık kodu (`wot`), tek başına `*PT` taşımaz. Ölçüldü: kazanç yok, **varsayılan kapalı** |
| [kaikki.org](https://kaikki.org) — Wiktionary makine-okunur dökümleri | 18 Türki dil, ~761 MB | ✅ arama indeksi · ⚠️ **altın standart değil** (bkz. Häuser & Stamatakis 2025) |
| [NorthEuraLex](https://northeuralex.org) | Türki + İrani + Slav + Ural + Moğol + Yunanca tek çatıda | 🚧 komşu aile taraması |
| WOLD — World Loanword Database | uzman alıntı derlemesi | ✅ alıntı değerlendirmesinin birincil ölçütü |
| kaikki **verici dili** dökümleri — Rusça, Moğolca, Evenkice, Arapça, Farsça, Yunanca, Ermenice, Fransızca, İtalyanca | 1.674.418 madde, ~352 MB | ✅ verici yakınlığı sinyali · ⚠️ Türki arama indeksinden **AYRI** dosyada; karışsalardı Rusça `море` Türki akraba adayı olarak dönerdi |
| DatSemShift | 10.565 anlam kayması | 🚧 semantik makullük |
| CLICS⁴ | 3.447 dilde eş-adlandırma | 🚧 semantik makullük |
| [Concepticon](https://concepticon.clld.org) · [Glottolog](https://glottolog.org) · [Lexibank](https://lexibank.clld.org) · [CLTS](https://clts.clld.org) | kavram kimliği, dil kimliği, sözvarlığı, fonetik gösterim | 🚧 standart katman |
| [SIGTYP ST2022](https://github.com/sigtyp/ST2022) (refleks tahmini) · [ST2023](https://github.com/sigtyp/ST2023) (akraba/türev tespiti) | shared task verisi ve metrikleri | 🚧 Türki verisi eklenecek (kaynak katkısı) |
| TDK Güncel Türkçe Sözlük · Tarama · **Derleme** | çağdaş, tarihî ve **ağız** sözvarlığı | ✅ canlı kaynak; Derleme asıl hedef sözvarlığı |
| Etymological Wordnet (de Melo, LREC 2014) · [EtymDB-2.0](https://github.com/clefourrier/EtymDB) (Fourrier & Sagot, LREC 2020) | 1,8M sözlükbirim · 2.536 dil | ⚠️ zinciri *saklıyor*, çıkarsamıyor — karşılaştırma noktası |

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

- [List & Forkel 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC10445856/) — otomatik alıntı tespiti, `seabor`, **F = 0,87**
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
- Benjamini & Hochberg 1995 — FDR kontrolü (çoklu karşılaştırma)
- Efron & Tibshirani — bootstrap güven aralığı · McNemar testi
- Zadrozny & Elkan — izotonik kalibrasyon · Platt scaling · Kull ve ark. — beta kalibrasyon
- King 2026, *PLOS Computational Biology* — Bayesçi filogenilikte simülasyon tabanlı kalibrasyon (SBC)
- [Hariharan & Mortensen 2025](https://arxiv.org/abs/2512.05364) — tarihsel dilbilimde ECE raporlaması (diakronik değişim tespitinde)
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
