# 🌍 Türki Diller Etimoloji Araştırma Motoru (Turkic Etymology Research Engine)

Bağımsız, yerel (local) çalışan ve internet üzerindeki 17 akademik, tarihi ve yerel sözlük kaynağından kelimelerin etimolojisini ve tüm Türki dillerdeki (Eski Türkçe, Türkiye Türkçesi, Azerbaycan, Kazak, Özbek, Türkmen, Kırgız, Tatar, Uygur, Çuvaş, Saha/Yakut, Başkurt vb.) karşılıklarını toplayan araştırma motoru.

---

## 🚀 Özellikler

- **17 Veri Katmanı Entegrasyonu**:
  - Sir Gerard Clauson (EDPT - Pre-13th Century Old Turkic Corpus)
  - Sevortjan ÉSTJa (SSCB Bilimler Akademisi 8 Ciltlik Türki Diller Etimoloji Sözlüğü)
  - Divanü Lugati't-Türk (Kaşgarlı Mahmud, 1074)
  - Kamus-ı Türkî (Şemseddin Sami, 1901)
  - Codex Cumanicus (1303 Kıpçakça)
  - Sanglax (Nevai Çağatayca)
  - Andreas Tietze Etimoloji Külliyatı (TÜBA)
  - Monumenta Altaica & Starostin Altaic Database
  - Turuz Dijital Filoloji & Etimoloji Kütüphanesi
  - EtimolojiTürkçe Portal (Orhun & Uygur Metin Tanıklamaları)
  - Nişanyan Etimoloji Sözlüğü
  - TDK Güncel Türkçe Sözlük REST API
  - TDK Tarama Sözlüğü (13.-19. yy)
  - TDK Derleme Sözlüğü (Türk Ağızları)
  - TDK Bilim ve Sanat Terimleri Akademik Sözlüğü
  - Wiktionary Proto-Turkic Reconstructions
  - Türki Diller Online Vikisözlükleri (`az`, `kk`, `uz`, `ky`, `tt`, `ug`, `cv`, `sah`, `ba`, `tk`)
- **İlişkisel Veritabanı ve Standart Şema**: Bulguları SQLite veritabanına ve JSON formatına kaydeder.
- **Fonetik Ses Denkliği Matrisi**: Türki dil kollarındaki ses kaymalarını (`d~t`, `b~m~p~v`, `g~k`, `z~s~ş~r`, `e~i~ä~ə`) otomatik üreterek tam arama gerçekleştirir.

---

## 💻 Kullanım (CLI)

### Kelime Arama ve Veritabanına Kaydetme
```bash
python3 -m engine.cli search tetik
python3 -m engine.cli search su
python3 -m engine.cli search deniz
```

### Ham JSON Çıktısı Alma
```bash
python3 -m engine.cli search belge --json
```

### Veritabanındaki Kayıtları Listeleme
```bash
python3 -m engine.cli list
```

### Veritabanındaki Bir Kelimeyi Gösterme
```bash
python3 -m engine.cli show göz
```

---

## 🧪 Birim Testlerini Çalıştırma

```bash
python3 -m unittest discover -s engine/tests
```

---

## 📜 Lisans
MIT License
