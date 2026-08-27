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

  Ana Kök / Rekonstrüksiyon : *köŕ
  Yöntem                    : karşılaştırmalı yöntem, 8 dil tanığı / 5 Türki kol
  Uygulanan denklikler      : g- ~ k- (Proto-Türkçe *k-)
                              Ortak Türkçe -z ~ Çuvaşça -r (Lir-Şaz rotasizmi)
  Hakem kararı              : 🟢 DOĞRULANDI — kısmi kanıt, %85 kapsam
  Alıntı sınıfı             : Asli Öz Türkçe (Native Turkic)
```

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

- **Alıntı sınıflandırıcı kural tabanlıdır**, eğitilmiş bir ML modeli değildir.
  Çıktıda `method: "rule_based"` olarak bildirilir. Etiketli Türkçe alıntı veri
  kümesi (WOLD ingest'i) tamamlanmadan model eğitimi planlanmamıştır.
- **Semantik aşama** yalnızca `[semantic]` ekstrası kuruluysa kanıt üretir.
- **Neo4j entegrasyonu yoktur.** `db/graph_database.py` Neo4j *şemasına uygun*
  düğüm/kenar yapısı üretir ama Cytoscape.js JSON'u olarak dışa verir.
- **Web paneli tek dosyalık statik HTML'dir**, Next.js kullanılmaz.
- Tohum veri yalnızca 59 kelimeyi kapsar; bu kelimelerin dışında sonuç
  tamamen canlı kaynaklara bağlıdır.

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
