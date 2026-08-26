# Gelecek Vizyonu: Otonom Yapay Zekâ & NLP Etimoloji Rekonstrüksiyon Mimarisi

Bu belge, **sözlüklerde veya literatürde henüz etimolojisi hiç yapılmamış (bilinmeyen, çözülmemiş veya ağızlarda kalmış)** kelimelerin etimolojisini bilimsel süreçler ve NLP / Yapay Zeka teknikleriyle otonom olarak keşfedecek gelecek mimarisinin detaylı planıdır.

---

## 1. Mimarinin Amacı ve Kapsamı
Geleneksel etimoloji sözlüklerinde bulunmayan veya kökeni tescillenmemiş kelimeler için:
- Fonetik dizilim hizalama (Phonetic Sequence Alignment),
- Otonom Rekonstrüksiyon Modelleri (Seq2Seq / HMM Proto-Language Reconstruction),
- Yapay Zeka Alıntı/Öz Dil Sınıflandırıcısı (Phonotactic Loanword ML Classifier),
- Vektörel Semantik Kayma Analizi (BERT / Word2Vec Embeddings)
kullanılarak bilimsel köken tahmini ve hipotetik rekonstrüksiyon yapılmasıdır.

---

## 2. NLP ve Bilimsel Araştırma Modülleri

### 2.1. Fonetik Vektör Gömme ve Dizilim Eşleme (Phonetic Sequence Alignment)
- Harf bazlı değil, uluslararası fonetik alfabesi (IPA) ve ses nitelikleri (ötümlü/ötümsüz, damaksıl/dudaksıl, dar/geniş) üzerinden vektörleştirme.
- **Needleman-Wunsch & Levenshtein Fonetik Algoritmaları**: 25 Türki dildeki kelimeleri ses dizilimi olarak hizalayarak gizli akrabalıkları (*cognate clusters*) tespit etme.

### 2.2. Otonom Rekonstrüksiyon Modeli (Seq2Seq / HMM Reconstructor)
- 25 Türki dildeki türevleri alıp tarihsel ses değişim kurallarını (HMM / Transformer) işleterek kelimenin muhtemel Proto-Türkçe `*kök` formunu matematiksel olarak tahmin etme.

### 2.3. Yapay Zeka Alıntı/Öz Dil Sınıflandırıcısı (Phonotactic Loanword ML Classifier)
- Ses uyumu kuralı (phonotactics), hece yapısı ve ünsüz dizilim kalıplarını öğrenmiş bir **Machine Learning Sınıflandırıcısı** ile kelimenin öz Türkçe mi yoksa alıntı mı olduğunu tespit etme.

### 2.4. Vektörel Semantik Kayma Analizi (Semantic Vector Embeddings)
- Diller arasındaki semantik kaymayı (kayma mesafesini) BERT ve Word2Vec vektörel uzayında hesaplama.

### 2.5. Morfotaktik ve Tarihsel Ek Ağacı Çözücüsü (Historical Suffix Tree Solver)
- Kelimeyi tarihsel yapım eklerine (`+gU`, `-ik`, `-gə`, `-ba`) bölerek kelimenin ham kökünü ayrıştırma.

---

## 3. Uygulama Zamanlaması
Bu mimari, **Data Ingestion Pipeline (Veri Toplama Katmanı)** internetteki tüm verileri %100 kapsayacak şekilde tamamlandıktan sonra devreye alınacaktır.

---

## Uygulama Durumu

> Bu bölüm, plandaki maddelerin koddaki karşılığını izler.
> ✅ uygulandı · 🟡 kısmen · ⏸ ertelendi (gerekçeli)

| Plan maddesi | Durum | Karşılık |
|---|---|---|
| §2.1 IPA ve ses nitelikleri üzerinden vektörleştirme | ✅ | `nlp/phonological_feature_engine.py` — gerçek **PanPhon** (6.367 IPA segmenti, 24 özellik) + **Epitran** IPA çevirisi |
| §2.1 Needleman-Wunsch & Levenshtein hizalama | ✅ | `nlp/cldf_lingpy_aligner.py` — gerçek **LingPy SCA** + yerleşik NW yedeği |
| §2.1 Akraba kümesi (*cognate cluster*) tespiti | ✅ | `nlp/cognate_clustering.py` — ikişerli benzerlik grafı + bağlantılı bileşen kümeleme |
| §2.2 Proto-Türkçe rekonstrüksiyon | ✅ | `nlp/comparative_reconstruction.py` — karşılaştırmalı yöntem, konum duyarlı denklik kümeleri. *Seq2Seq/HMM yerine klasik karşılaştırmalı yöntem seçildi: eğitim verisi gerektirmez ve çıktısı denetlenebilir.* |
| §2.3 Alıntı/öz dil sınıflandırıcısı | 🟡 | `nlp/loanword_classifier.py` + `nlp/loanword_detector.py` — 4 katmanlı kural motoru. **Eğitilmiş ML modeli ertelendi**: etiketli Türkçe veri kümesi yok. Çıktıda `method: "rule_based"` bildirilir. |
| §2.4 Vektörel semantik kayma (BERT/Word2Vec) | 🟡 | `nlp/diachronic_semantic_engine.py` — opsiyonel `sentence-transformers`. Kurulu değilse aşama **kanıt üretmediğini bildirir**; sahte skor verilmez. |
| §2.5 Tarihsel ek ağacı (`+gU`, `-Ik`, `-gA`, `-bA`) | ✅ | `nlp/historical_morphology.py` — katmanlı türetme ağacı, Eski/Orta/Modern Türkçe ek katmanları |
| §3 Veri toplama ön şartı | ✅ | 9 canlı kaynak + 9 tohum dosyası; ölü uç noktalar kaldırıldı |
