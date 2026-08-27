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

### Rekonstrüksiyon (n=400, uzman altın standardı)

**Birincil metrikler NED ve B-Cubed F'tir.** Bu, keyfî bir tercih değil alan
standardı: SIGTYP 2022'nin resmi metrikleri ED, NED, B-Cubed F ve BLEU'dur ve
**tam doğruluk hiç yer almaz**; Bouchard-Côté ve ark. 2013 yalnız normalize
Levenshtein raporlar; Meloni ve ark. 2021'in ana metriği ortalama ED'dir.
List 2019 (*Beyond Edit Distances*) sistematik hataların ED tarafından her
örnekte yeniden cezalandırıldığını gösterip B-Cubed F önerir.

| Sistem | **NED**↓ | **BCFS**↑ | ED↓ | FER↓ | tam | kapsam |
|---|---|---|---|---|---|---|
| **motor** | **0,384** | **0,508** | 1,97 | **0,359** | 0,230 | 0,988 |
| `majority_character` (trivial) | 0,382 | 0,500 | 1,94 | 0,363 | 0,237 | 1,000 |
| `copy_anchor` (hiçbir şey yapma) | 0,417 | 0,466 | 2,13 | 0,392 | 0,203 | 1,000 |
| `copy_random_daughter` | 0,453 | 0,427 | 2,29 | 0,414 | 0,168 | 1,000 |
| `copy_longest` | 0,552 | 0,362 | 3,04 | 0,541 | 0,100 | 1,000 |

Birincil metrikte fark **+0,0018**, %95 GA [−0,0146, +0,0190] → **anlamlı
değil**. Motor trivial taban çizgisiyle istatistiksel olarak **berabere**;
B-Cubed F ve FER'de hafif önde, NED ve tam doğrulukta hafif geride.

⚠️ **Çekimserlik bedava değildir.** Cevaplanmayan madde ortalamaya mümkün
olan en kötü NED'i (1,0) katar. Bir dönem yalnızca cevaplanan maddeler
ortalanıyordu; o muhasebe cevap vermemeyi kusursuz cevap vermekle bir
tutuyor ve çekimser kalmayı ödüllendiriyordu.

Karşılaştırma noktası — **bu, kural tabanlı sistemlerin normal bandıdır**:

| Sistem sınıfı | Rom-phon tam doğruluk |
|---|---|
| rastgele kız dil | %0,06 |
| CorPaR (kural/örüntü) | %22,2 |
| SVM+PosStrIni (kural/örüntü) | %24,7 |
| **bu motor** | **%23,0** |
| RNN (denetimli) | %52,3 |
| Transformer (denetimli) | %53,8 |

Sinitic verisinde birebir aynı örüntü görülüyor: CorPaR majority'den daha iyi
PED alıyor (3,28 < 3,50) ama aynı tam doğruluğu veriyor. Sorun uygulamada
değil, **paradigmada**: %50 bandına yalnız denetimli öğrenme çıkıyor.

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
| **motor** | 0,421 | 0,416 | 0,427 | 0,646 |
| yalnız fonotaktik | 0,294 | 0,444 | 0,220 | 0,681 |
| hepsi alıntı (trivial) | 0,464 | 0,302 | 1,000 | 0,302 |

⚠️ **Ablasyon hükmü — olumsuz ve dürüstçe raporlanıyor.** Zincir sinyali
(sözlük etiketi) kapatıldığında motor, yalnız fonotaktik taban çizgisine
karşı WOLD'da **anlamlı biçimde GERİDE** (fark −0,035, %95 GA [−0,062,
−0,007], p=0,019); Wiktionary ablasyonunda ise fark anlamlı değil (+0,006,
p=0,65). Yani `ses_kanunu_ihlali` ve `değişimsiz_yayılım` sinyalleri ölçülebilir
katkı sağlamıyor, uzman ölçütünde zarar veriyor.

Bunun **kavramsal** açıklaması da var: denklikleri alıntıların da içinde
olduğu veriden öğrendik, ve Arapça alıntılar bütün Oğuz dillerinde aynı
biçimde uyarlandığı için **kendi düzenli denkliklerini yaratıyorlar** — yani
"beklenen refleks tutuyor mu" testini geçiyorlar.

> **Düzeltme kaydı.** Bu ablasyon bir kez yanlış raporlandı. Değerlendirme
> kodu `witnesses` alanını hiç doldurmuyordu ve dört sinyalden ikisi tanık
> gerektirdiği için **yapısal olarak devre dışıydı**; sonuç fonotaktikle
> birebir aynı çıkıyor, biz de "sinyaller katkı sağlamıyor" diye
> raporluyorduk. Doğrusu "sinyaller hiç çalıştırılmadı"ydı. Tanıklar
> doldurulunca (kelime başına ort. 5,1) sinyaller ateşleniyor — ve yukarıdaki
> gerçek ölçüm elde ediliyor.

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
| [kaikki.org](https://kaikki.org) — Wiktionary makine-okunur dökümleri | 18 Türki dil, ~761 MB | ✅ arama indeksi · ⚠️ **altın standart değil** (bkz. Häuser & Stamatakis 2025) |
| [NorthEuraLex](https://northeuralex.org) | Türki + İrani + Slav + Ural + Moğol + Yunanca tek çatıda | 🚧 komşu aile taraması |
| WOLD — World Loanword Database | uzman alıntı derlemesi | 🚧 alıntı değerlendirmesinin birincil ölçütü |
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
- Blum & List 2023 — alignment trimming · Blum & List 2026 — leave-one-out düzensizlik tespiti (%85)
- [Bouchard-Côté ve ark. 2013, *PNAS*](https://www.pnas.org/doi/10.1073/pnas.1204678110) — olasılıksal ses değişimi modeli, 637 Austronesian dili
- Meloni ve ark. 2021 · Kim ve ark. 2023 (ACL) — Transformer rekonstrüksiyon taban çizgileri (%53 Roman / %39,5 Sinitik, 8.799 eğitim örneğiyle)
- Lu, Xie & Mortensen 2024 (**ACL 2024 Best Paper**) — DPD-BiReconstructor, yarı-denetimli rekonstrüksiyon
- [Lu, Wang & Mortensen 2024 (LREC-COLING)](https://arxiv.org/abs/2403.18769) — refleks tahminiyle N-best yeniden sıralama (P2D)
- Akavarapu & Bhattacharya 2023/2024 — Cognate Transformer (MSA Transformer, çapraz-aile ön-eğitim) · Cui ve ark. 2024
- [List ve ark. 2023 (LChange @ EMNLP)](https://aclanthology.org/2023.lchange-1.3/) — fonolojik rekonstrüksiyonda belirsizlik gösterimi (`*[p a|i t]`)
- ⚠️ Häuser & Stamatakis 2025 — Wiktionary/BabelNet'ten kazınan akraba kümelerinin altın standart ağaçlarla tutarsızlığı
- ⚠️ Häuser 2024 — filogenide leksikal akraba ağaçları ses denkliklerinden ~1/3 daha isabetli

### Alıntı tespiti, temas ve geçiş yolu

- [List & Forkel 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC10445856/) — otomatik alıntı tespiti, `seabor`, **F = 0,87**
- [Miller ve ark. 2020, *PLOS ONE*](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0242709) — PyBor, tek dilli fonotaktik taban çizgi (**F1 ≈ 0,55**) · [Miller & List 2023](https://arxiv.org/pdf/2302.00189)
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

- **Kessler 2001, *The Significance of Word Lists* (CSLI)** — permütasyon testiyle **rastlantısal benzerlik** kontrolü. Yeni etimoloji iddiasının olmazsa olmazı.
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
