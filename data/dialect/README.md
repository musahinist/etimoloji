# Ağız (diyalekt) verisi — kaynak ve kullanım koşulları

Bu dizin, **TDK Derleme Sözlüğü**nden (Türkiye'de Halk Ağzından Derleme
Sözlüğü) toplanmış ağız kayıtlarını içerir. Projenin ilan edilmiş asıl hedefi
budur: *sözlükte etimolojisi verilmemiş sözvarlığı*.

## Nereden geliyor

| | |
|---|---|
| Kaynak | `https://eski.sozluk.gov.tr/derleme?ara=<kelime>` (TDK) |
| Tohum listesi | Vikisözlük `Kategori:Türkçe halk ağzı` — **CC-BY-SA** |
| Toplayıcı | `scripts/harvest_derleme.py` |
| Künye | `_provenance.json` — kaynak, tarih, sorgu/kayıt sayısı, istek gecikmesi |

TDK'nın yeni arayüzü (`sozluk.gov.tr/derleme`) JSON döndürmüyor (SPA HTML
kabuğu). Eski uç (`eski.sozluk.gov.tr`) JSON döndürüyor ve kullanılan budur.
Listeleme uçları yoktur — yalnız tam eşleşmeli sorgu çalışır — bu yüzden
toplama, tohum listesinden başlayıp kayıtlardaki çapraz göndermeleri
(`bakin`, `asilkelim`) izleyerek yayılır.

## Lisans duruşu

⚠️ **TDK'nın ilan edilmiş açık lisansı yoktur; telif TDK'dadır.**

Bu veri burada şu koşullarla tutulur:

- **Depo private'tır.** Veri kamuya dağıtılmaz.
- **Çalışma akademiktir ve ticari değildir.**
- **Atıf açıktır:** her kayıt TDK Derleme Sözlüğü'nden geldiğini künyesiyle
  birlikte taşır (`kisaltma` alanı cilt kimliğini, `eser_ad`/`yazar_ad`
  alanları derleme künyesini verir).
- **Hacim sınırlıdır ve istekler hız sınırlıdır** (kurum sunucusu; `robots.txt`
  yok — 404).

Verinin depoda tutulma gerekçesi yeniden üretilebilirliktir: TDK içeriği
değiştiğinde ya da uç kapandığında hasat tekrarlanamaz, ve o durumda depodaki
ölçümler doğrulanamaz hâle gelir. Depo public'e açılacak olursa **bu dizin
önce çıkarılmalıdır**.

## Dosyalar

| Dosya | İçerik | Depoda |
|---|---|---|
| `derleme/records.jsonl` | ham Derleme kayıtları (bir satır = bir madde-anlam) | ✅ |
| `derleme/words.txt` | tutan madde başları — `analyse_dialect_words.py --words` girdisi | ✅ |
| `_provenance.json` | hasat künyesi | ✅ |
| `analysis.json` | türetilmiş analiz çıktısı | ❌ yeniden üretilebilir |

## Kayıt şeması

Kayıtlar TDK'nın döndürdüğü alanları olduğu gibi taşır; toplayıcı ikisini
ekler:

- `_query` — bu kaydı getiren sorgu
- `sehir_temiz` — `sehir` alanının HTML'i ayıklanmış hâli
  (`*Dinar -<b>Afyon</b><br>` → `*Dinar - Afyon`)

`sehir` alanı **yerleşim ve il düzeyinde** coğrafi etiket taşır;
`engine/utils/geo_tagger.py` bunu kullanabilir.

## Ölçüm uyarısı

`scripts/analyse_dialect_words.py` bu listeyi tam hattan geçirir ve sonuçları
kanıt gücüne göre üç kovaya ayırır. ⚠️ **Kova eşikleri bu dağılımda
doğrulanmamıştır**: kalibratör `savelyevturkic/train` üzerinde eğitildi
(n=223, CLDF çıplak kökler), buradaki girdiler ise sözlük madde başıdır.
"çözüldü" ve "güçlü aday" etiketleri ölçülmüş bir doğruluk iddiası taşımaz —
sıralama ipucudur. Script bu uyarıyı çıktısına da basar.
