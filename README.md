# 🌍 Türki Diller Etimoloji Araştırma Motoru (Turkic Etymology Research Engine)

Bağımsız, yerel (local) çalışan ve internet üzerindeki **19 akademik, tarihi, lügat ve yerel sözlük kaynağından** kelimelerin etimolojisini, alıntı kelime yörüngelerini, kaynak dildeki orijinal imla ve anlamlarını ve tüm Türki dillerdeki (Eski Türkçe, Türkiye Türkçesi, Azerbaycan, Kazak, Özbek, Türkmen, Kırgız, Tatar, Uygur, Çuvaş, Saha/Yakut, Başkurt, Hakas, Tuva, Gagavuz, Karaçay-Balkar vb.) karşılıklarını toplayan gelişmiş araştırma motoru.

---

## 🚀 Öne Çıkan Özellikler

### 📚 19 Paralel Veri Katmanı Portföyü
- **Akademik Türkoloji Veri Bankası**: Sir Gerard Clauson (EDPT - Pre-13th Century Old Turkic Corpus) & Sevortjan ÉSTJa (SSCB Bilimler Akademisi 8 Ciltlik Türki Diller Etimoloji Sözlüğü)
- **Tarihi Türk Lehçeleri Sözlükleri**: Divanü Lugati't-Türk (Kaşgarlı Mahmud, 1074), Kamus-ı Türkî (Şemseddin Sami, 1901), Codex Cumanicus (1303 Kıpçakça), Sanglax (Nevai Çağatayca)
- **TDV İslam Ansiklopedisi (İSAM)**: M.Ö. III. Yüzyıl Doğu Hunları ve Orhun Yazıtları etimoloji maddeleri
- **Internet Archive Kitap Külliyatı**: Taranmış el yazmaları, Türkoloji kitapları ve sözlüklerde tam metin arama
- **DergiPark Akademik Dergiler**: TÜBİTAK ULAKBİM üniversite Türkoloji makale dizini
- **Osmanlıca ve Klasik Türkçe Lügatlar**: Kubbealtı Lugatı (`lugatim.com`) ve Ahmet Vefik Paşa (*Lehçe-i Osmanî*)
- **TDK Tüm Alt Portalları**: TDK Türkiye Türkçesi Ağızlar Sözlüğü (`sozluk.gov.tr/ttas`), TDK Kişi Adları Etimoloji Sözlüğü (`sozluk.gov.tr/kisi`), TDK Yazım Kılavuzu Köken İşaretlemeleri
- **Glosbe Multilingual API**: 25 Türki dil çeviri ve anlam ağı
- **Türki Cumhuriyetler Yerel İzahlı Lügat Portalları**: Obastan & Azleks (Azerbaycan), Savodxon & ZiyoNET (Özbekistan), Tilqazyna & Sozlik.kz (Kazakistan), ElSözlük (Kırgızistan), Tatarica (Tataristan)
- **Alıntı Kelimeler Kaynak Dil Orijinal İmla ve Etimoloji Katmanı**: Arapça (üçlü ünsüz kökü, vezin, orijinal imla `كتاب`), Farsça (orijinal imla `روزگار`, bileşenler `rūz` + `-gār`), Grekçe/Rumca (`αὐθέντης` -> `efendi`), Fransızca/İtalyanca/Latince ve Çince alıntı yörüngeleri
- **Tietze & Altaica Etimoloji Külliyatı**: Andreas Tietze Etimoloji Külliyatı (TÜBA), Monumenta Altaica & Starostin Altaic Database, Turuz Dijital Filoloji Kütüphanesi
- **EtimolojiTürkçe Portal**: Orhun ve Uygur Metin Tanıklamaları
- **Starling Etymological Database**: Tower of Babel Proto-Turkic Reconstructions
- **Nişanyan Etimoloji Sözlüğü & TDK GTS API**
- **TDK Tarama & Derleme Sözlükleri**: 13.-19. yy Tarihi Türkçe ve Anadolu Ağızları
- **25 Türki Dil Online Sözlük Portalları**: `az`, `kk`, `uz`, `ky`, `tt`, `ug`, `cv`, `sah`, `ba`, `tk`, `gag`, `krc`, `tyv`, `alt`, `khk`, `slq`, `chg`, `ota`, `otk`, `nog`, `kum`, `crh`, `kaa`, `cjs`

---

### 🧩 Dilbilimsel & Fonetik Zeka Katmanları
- ** Morfolojik Kök Ayrıştırıcı (`morphology.py`)**: Türemiş ve çekim ekli kelimeleri (`güzellik`, `denizcilik`, `bilimsel`) otomatik köklerine (`güzel`, `deniz`, `bilim`) ayırarak çift yönlü tarama yapar.
- ** Otomatik Alfabe Transkripsiyon Motoru (`transliteration.py`)**: Kiril ve Arap alfabesindeki çıktıları okuma kolaylığı sağlamak için Latin fonetik transkripsiyonu ile zenginleştirir (Örn: Kazakça `көзел (közеl)`, Uygurca `گۈزەل (güzəl)`).
- ** Fonetik Ses Kayması Analizcisi (`phonetic_rules.py`)**: Türkiye Türkçesi ile diğer diller arasındaki ses değişim kurallarını tespit eder (`g->k`, `z->s/ś`, `d~y~t~r` diyalekt denkliği).
- ** Tarihsel Zaman Çizelgesi (Chronological Timeline)**: Kelimenin M.Ö. 3. yüzyıl Hun/Orhun kayıtlarından günümüze geçirdiği evrimi kronolojik sunar.
- ** Kök Akraba Sözcük Ağı (`cognates.py`)**: Aynı Proto-Türkçe kökten türeyen akraba kelimeleri (`göz` -> `görmek`, `gözlem`, `gözlük`) otomatik bağlar.

---

## 💻 Kullanım (CLI)

### 🔍 Tekil Kelime Arama
```bash
python3 -m engine.cli search göz
python3 -m engine.cli search deniz
python3 -m engine.cli search efendi
python3 -m engine.cli search güzellik
```

### 📦 Toplu Arama (Bulk Search)
```bash
python3 -m engine.cli bulk --file kelimeler.txt
```

### 📄 Ham JSON Çıktısı Alma
```bash
python3 -m engine.cli search tengri --json
```

### 📂 Veritabanındaki Kayıtları Listeleme & Gösterme
```bash
python3 -m engine.cli list
python3 -m engine.cli show ayak
```

---

## 🔬 Gelecek Yapay Zeka & NLP Araştırma Planları
Henüz etimolojisi hiç yapılmamış kelimeler için geliştirilecek otonom NLP alıntı sınıflandırıcısı ve rekonstrüksiyon mimarisi planları `research/` klasöründe yer almaktadır:
- [Otonom Yaptay Zeka & NLP Etimoloji Rekonstrüksiyon Mimarisi Planı](file:///Users/mshn/Documents/etimoloji/research/ai_nlp_etymology_reconstruction_plan.md)
- [Otonom NLP Alıntı Kelime Sınıflandırıcısı Master Planı](file:///Users/mshn/Documents/etimoloji/research/ai_nlp_loanword_classifier_master_plan.md)

---

## 🧪 Birim Testlerini Çalıştırma

```bash
python3 -m unittest discover -s engine/tests
```

---

## 📜 Lisans
MIT License
