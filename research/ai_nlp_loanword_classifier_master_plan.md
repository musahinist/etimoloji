# Gelecek Vizyonu: Otonom NLP Alıntı Kelime Sınıflandırıcısı ve Keşif Motoru (Master Plan)

Bu belge, **sözlüklerde veya literatürde henüz etimolojisi hiç yapılmamış (bilinmeyen, çözülmemiş veya ağızlarda kalmış)** kelimelerin alıntı kelime (*loanword*) olup olmadığını ve hangi kaynak dilden geçtiğini **küresel NLP kütüphaneleri, açık veri kümeleri ve makine öğrenmesi modelleriyle** tespit edecek ana plan belgesidir.

---

## 1. Dahil Edilecek Küresel Kütüphane ve Veri Kümeleri

### 1.1. Python NLP Kütüphaneleri
- **`loanpy`**: Bilgisayar destekli alıntı kelime tespiti ve fonetik adaptasyon modelleme kütüphanesi (`pip install loanpy`).
- **`csu-signal/loan-word-detection`**: COLING 2022 multilingual alıntı kelime tespiti mimarisi.
- **`Epitran` & `PanPhon`**: IPA Uluslararası Fonetik Alfabesi dönüştürücü ve ses nitelikleri (ötümlü/ötümsüz, damaksıl/dudaksıl) matris çıkarıcı.
- **`Zemberek-NLP` & `Starlang KeNet`**: Türkçe morfolojik analiz ve kelime ağı.

### 1.2. Açık Veri Kümeleri (Datasets)
- **WOLD (World Loanword Database)**: 395 dildeki alıntı kelime ilişkileri.
- **ZurichNLP / ConLoan**: Zürih Üniversitesi NLP alıntı değerlendirme veri kümesi.
- **Wiktionary CLDF (Loanword Bank)**: 1.400 dilde etimolojik borçlanma veri ağı.

---

## 2. 4 Aşamalı Otonom NLP Alıntı Keşif Hattı (Pipeline)

```
[Bilinmeyen Kelime Girdisi]
          │
          ▼
 ┌─────────────────────────────────────────────────────────┐
 ├── 1. KATMAN: Fonotaktik Yapı & Ses İhlal Analizcisi   │
 │   - Söz başı ünsüz kısıtlaması (r-, l-, m-, f-, v-)     │
 │   - Ünlü uyumu ve Arapça/Farsça vezin kalıp testi      │
 └────────────────────────┬────────────────────────────────┘
                          │
                          ▼
 ┌─────────────────────────────────────────────────────────┐
 ├── 2. KATMAN: Çapraz Türki Lehçe Dağılım Skorlaması      │
 │   - 25 Türki dil haritasındaki varlık/yokluk oranı      │
 │   - Geniş yayılım -> Öz Türkçe | Dar yayılım -> Alıntı  │
 └────────────────────────┬────────────────────────────────┘
                          │
                          ▼
 ┌─────────────────────────────────────────────────────────┐
 ├── 3. KATMAN: NLP Makine Öğrenmesi Sınıflandırıcısı     │
 │   - Epitran IPA embeddings + Char N-Gram               │
 │   - Olasılık tahmini: P(Öz Türkçe), P(Arapça), P(Batı) │
 └────────────────────────┬────────────────────────────────┘
                          │
                          ▼
 ┌─────────────────────────────────────────────────────────┐
 ├── 4. KATMAN: 10 Komşu Kaynak Dilde En Yakın Arama      │
 │   - Levenshtein Fonetik Mesafe + FastText Semantik     │
 │   - Kaynak Diller: Arapça, Farsça, Rumca, Çince, vb.  │
 └─────────────────────────────────────────────────────────┘
```

---

## 3. Detaylı Katman Mimarileri

### Katman 1: Fonotaktik Yapı & Ses İhlal Analizcisi (Phonotactic Anomaly Detector)
- **Çalışma Prensibi**: Türkçe ses mantığına aykırı hece ve harf dizilimlerini matematiksel kurallarla tespit eder.
- **İhlal Kriterleri**:
  1. *Söz Başı İhlali*: `r-, l-, m-, n-, f-, h-, j-, v-, z-` harfleriyle başlama.
  2. *Konsonant Çiftliği*: Söz başında yan yana iki ünsüz (`tr-`, `sp-`, `kr-`).
  3. *Ünlü Uyumsuzluğu*: Kalın ve ince ünlülerin aynı kelimede yer alması.
  4. *Arapça Vezin Deseni*: `müf'il`, `müfa'ale`, `təf'il`, `istif'al` kalıp uyumu.

### Katman 2: Çapraz Türki Lehçe Dağılım Skorlaması (Cross-Turkic Cognate Distribution)
- **Çalışma Prensibi**: 18 Fetcher'ımızın 25 Türki dilde ürettiği harita üzerinden kelimenin derinliğini ölçer.
- **Skorlama Formülü**:
  - `Yayılım Oranı = (Mevcut Olduğu Lehçe Sayısı / 25)`
  - `Yayılım > %70` -> Yüksek ihtimalle Asli Öz Türkçe.
  - `Yayılım < %20` (Sadece Oğuz kolunda var) -> Yüksek ihtimalle Son Dönem Alıntı.

### Katman 3: NLP Makine Öğrenmesi Sınıflandırıcısı (ML Loanword Classifier)
- **Model**: Logistic Regression / Random Forest / Transformer Classifier.
- **Girdi Vektörü**: `Epitran` IPA matrisi + TF-IDF Karakter N-Gramları.
- **Çıktı**: Kelimenin köken dil ailesi olasılık dağılımı:
  - `P(Öz Türkçe) = %4.2`
  - `P(Arapça / Farsça) = %86.5`
  - `P(Rumca / Grekçe) = %7.1`
  - `P(Fransızca / Batı) = %2.2`

### Katman 4: 10 Komşu Kaynak Dilde En Yakın Komşu Araması (Donor Nearest-Neighbor Search)
- **Arama Uzayı**:
  1. Arapça (Arabic)
  2. Farsça (Persian)
  3. Soğdca / Pehlevice (Sogdian / Middle Persian)
  4. Çince (Chinese)
  5. Moğolca (Mongolian)
  6. Rumca / Bizans Grekçesi (Greek)
  7. Ermenice (Armenian)
  8. Rusça / Slav Dilleri (Slavic)
  9. İtalyanca / Venedikçe (Italian / Venetian)
  10. Fransızca (French)
- **Eşleşme Algoritması**:
  - IPA Fonetik Mesafe (Levenshtein Edit Distance) $\le 2$
  - Vektörel Semantik Benzerlik (FastText Cosine Similarity) $\ge 0.75$

---

## 4. Zamanlama ve Entegrasyon
Bu mimari, **Data Ingestion Pipeline (Veri Toplama Katmanı)** tamamlandıktan sonra `engine/utils/loanword_detector.py` ve `engine/nlp/` klasörü altına kurulup sisteme dâhil edilecektir.
