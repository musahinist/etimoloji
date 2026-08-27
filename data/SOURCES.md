# Veri kaynakları künyesi

Bu dosya `scripts/download_*.py` tarafından **otomatik üretilir** — elle
düzenlemeyin. Her satır hangi veri kümesinin hangi sürümünden, ne zaman
indirildiğini ve içerik özetini gösterir.

## CLDF veri kümeleri

| Veri kümesi | Sürüm | Commit | İndirilme | Kayıt | Rol |
|---|---|---|---|---|---|
| [hruschkaturkic](https://github.com/lexibank/hruschkaturkic) | `v1.0` | `72e8095234f8` | 2026-08-26 | 4,213 biçim | bağımsız çapraz kontrol |
| [robbeetstriangulation](https://github.com/lexibank/robbeetstriangulation) | `v0.3` | `5f59f1522ed8` | 2026-08-26 | 26,224 biçim | yalnız temas çerçevesi |
| [savelyevturkic](https://github.com/lexibank/savelyevturkic) | `v2.1` | `4a540590580f` | 2026-08-26 | 8,360 biçim | birincil altın standart |
| [starostinaltaic](https://github.com/lexibank/starostinaltaic) | `main` | `638500a49eda` | 2026-08-26 | 5,756 biçim | yalnız karşılaştırma |
| [wold](https://github.com/lexibank/wold) | `master` | `1df62b9bdc72` | 2026-08-27 | 64,289 biçim | alıntı değerlendirmesinin UZMAN ölçütü |

### Künye ve uyarılar

**hruschkaturkic** — Hruschka ve ark. 2015, Current Biology
> `forms.csv` 4,213 kayıt · `cognates.csv` 4,213 kayıt · `languages.csv` 26 kayıt · `parameters.csv` 222 kayıt

**robbeetstriangulation** — Robbeets & Bouckaert, Triangulation dataset
> ⚠️ Transeurasian verisi en az EDAL kadar tartışmalıdır (Tian ve ark. 2022). YALNIZCA temas/ödünçleme analizinde kullanılır; akrabalık kanıtına asla katılmaz.
> `forms.csv` 26,224 kayıt · `cognates.csv` 26,224 kayıt · `languages.csv` 101 kayıt · `parameters.csv` 254 kayıt

**savelyevturkic** — Savelyev & Robbeets 2020, Journal of Language Evolution
> `forms.csv` 8,360 kayıt · `cognates.csv` 8,360 kayıt · `languages.csv` 32 kayıt · `parameters.csv` 254 kayıt

**starostinaltaic** — Starostin, Dybo & Mudrak, Altaic Etymological Dictionary
> ⚠️ Altay hipotezi tartışmalıdır (Vovin 2005). Tek kaynak olarak kullanılmaz; akrabalık kanıtına katılmaz.
> `forms.csv` 5,756 kayıt · `cognates.csv` 5,756 kayıt · `languages.csv` 54 kayıt · `parameters.csv` 110 kayıt

**wold** — Haspelmath & Tadmor (ed.), World Loanword Database
> ⚠️ Alıntı tespiti Wiktionary etiketine karşı ölçülemez: motorun zincir sinyali zaten o etiketi okur, ölçüm döngüsel olur. WOLD bağımsız uzman derlemesidir ve BİRİNCİL ölçüt olarak kullanılır.
> `forms.csv` 64,289 kayıt · `languages.csv` 41 kayıt · `parameters.csv` 1,814 kayıt · `borrowings.csv` 21,624 kayıt

### Doğrulama

Her dosyanın SHA-256'sı ilgili `data/cldf/<ad>/_provenance.json` içindedir.
Yeniden indirip aynı özetleri aldığınızda veri değişmemiştir.
