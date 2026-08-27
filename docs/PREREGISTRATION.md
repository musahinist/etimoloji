# Ön-kayıt taslağı — Türki Etimoloji Motoru

⚠️ **Bu bir taslaktır; henüz kaydedilmemiştir.** OSF kaydı hesap gerektirir
ve bu depodan yapılamaz. Kayıt yapıldığında DOI aşağıya ve
`prediction_test.PredictionRegistry.external_doi` alanına yazılacaktır.

- **Şablon:** OSF *Secondary Data Preregistration* — çalışma mevcut,
  yayımlanmış veriyle yapılıyor; veri toplanmadan önce kayıt yapılan
  şablonlar (AsPredicted, Standard) bu senaryoya uymuyor.
- **Alternatif:** AsPredicted (Bodt & List bunu kullandı; daha kısa, DOI yok).
- **Onay süresi:** OSF kayıtları en geç 48 saatte otomatik onaylanır ve
  kamuya açık kayıt bir **DOI** alır.

---

## 1. Ne soruyoruz?

**S1.** Karşılaştırmalı yöntemi sembolik olarak uygulayan bir sistem,
Proto-Türkçe ata biçim rekonstrüksiyonunda trivial taban çizgilerini
**istatistiksel olarak anlamlı** biçimde geçebilir mi?

**S2.** Sözlük etiketini hiç görmeden, alıntı/miras ayrımı yapılabilir mi?

**S3.** Sistemin ürettiği **yeni** iddialar (altın standartta olmayan)
Türkologlar tarafından kabul edilebilir bulunuyor mu?

## 2. Hipotezler ve karar eşikleri

⚠️ Eşikler **veriye bakılmadan** yazılır ve sonradan değiştirilmez.

| # | Hipotez | Karar eşiği |
|---|---|---|
| H1 | Motor, `majority_character` taban çizgisini **NED**'de geçer | eşleşmiş bootstrap %95 GA sıfırı **dışlar**, motor lehine |
| H2 | Motor, `majority_character`ı **B-Cubed F**'de geçer | aynı ölçüt |
| H3 | Alıntı tespitinde motor, yalnız fonotaktik taban çizgisini geçer | permütasyon p < 0,05, FDR sonrası |
| H4 | İleri tahmin sicili: kilitli öngörülerin ≥%40'ı doğrulanır | Blum ve ark. 2024 modeli ("≥20 doğru tahmin") |
| H5 | Uzman değerlendirmesinde ortalama derece ≥ 3,0 | Krippendorff ordinal α ≥ 0,60 olmak kaydıyla |

⚠️ **H5'in ön koşulu vardır:** kodlayıcılar arası uyum düşükse ortalama
derece yorumlanamaz. α < 0,60 ise H5 **test edilmemiş** sayılır ve öyle
raporlanır; "ortalama 3,2 çıktı" denmez.

## 3. Veri

| Rol | Kaynak | Notlar |
|---|---|---|
| Birincil altın standart | `savelyevturkic` v2.1 | 400 rekonstrüksiyonlu küme |
| Çapraz kontrol | `hruschkaturkic` v1.0 | uzman uyuşmazlık bandı |
| Alıntı ölçütü | WOLD (Sakha) | Wiktionary'den bağımsız |
| Arama indeksi | kaikki.org | ⚠️ **altın standart değil** |
| Verici sözlükleri | kaikki.org (9 dil) | Türki indeksten ayrı |

**Bölme:** kavram bazlı, tuzlu karma ile deterministik
(`gold.assign_split`). train/dev/test. **Test bölümü dondurulmuştur** ve
yalnız nihai raporda bir kez açılacaktır.

## 4. Sızıntı önlemleri (önceden taahhüt)

1. Hazır Wiktionary etimolojileri değerlendirmede motordan **gizlenir**;
   kaikki yalnız arama indeksidir.
2. Alıntı ablasyonunda zincir sinyali (sözlük etiketini okuyan sinyal)
   **kapatılır**; açık bırakmak ölçümü döngüsel yapar.
3. Denetimli bileşenler yalnız TRAIN kavramlarında eğitilir. Ölçüm tüm
   veride koşulursa `report` uyarı basar ve künyeye yazar.
4. Eşikler AYAR yarısında seçilir; rapor yarısı görülmez. Eğitilmiş bir
   sinyalin eşiği, o sinyalin **görmediği** veride ayarlanır (yığın
   sızıntısı önlemi).
5. Çapa dili tanığı girdiden çıkarılır ("dürüst koşul").

## 5. Ne raporlanacak

- Her sayı **bootstrap %95 güven aralığıyla** ve trivial taban çizgilerle
  birlikte.
- Birincil metrikler: **NED, B-Cubed F, ED** (SIGTYP 2022 resmi metrikleri).
  Tam doğruluk ikincildir.
- Negatif bulgular **aynı ayrıntıda** raporlanır. Bu depoda halihazırda
  raporlanmış negatif bulgular: bağlam kodlaması (D3), P2D yeniden sıralama
  (D5), Batı Eski Türkçe entegrasyonu (B2), ses kanunu sinyalinin miras
  tablosuyla kurtarılamaması (C4).
- Çekimserlik **bedava değildir**: cevapsız madde ortalamaya mümkün olan en
  kötü NED'i (1,0) katar.

## 6. Ne raporlanmayacak (ve neden)

- "Sıfır eğitim verisi" iddiası **bırakılmıştır**; 400 uzman kümesi eğitim
  verisidir.
- Uzman değerlendirmesi yapılmadan hiçbir kabul edilebilirlik sayısı
  üretilmez.
- Test bölümü birden çok kez açılmaz; açıldığı koşu raporda belirtilir.

## 7. Sapma kaydı

Bu bölüm kayıttan **sonra** doldurulur: plandan her sapma, gerekçesiyle ve
tarihiyle buraya yazılır. Boş kalması, sapma olmadığı anlamına gelir.

| Tarih | Sapma | Gerekçe |
|---|---|---|
| — | — | — |
