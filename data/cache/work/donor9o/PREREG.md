# 9o — biçim-öncelikli ikinci arama: kesinliği koruyarak verici biçimi kapsamasını artırmak (ön kayıt)

Tarih 2026-09-26. Bu dosya YENİ rapor bölümü açılmadan commit edilir. Üretim tabanı (4.3.2):
`ARABIC_VIA_RULE = "d1"`, `FRENCH_RULE = "g2"`, `WESTERN_RULE = "h1"`, `SENSE_BRIDGE = True`,
`DONOR_HONEST = "a2"`, yeni bayrak `DONOR_FORM_FIRST = "off"`. Yalnız GÖSTERİLEN BİÇİM ve kesinlik
değişir; `attribute_donor`un seçtiği dil ve biçim, a2'nin dili, alıntı GÜCÜ (`nearest_donor`,
`proximity_strength`), `donors.db`, `index.db` değişmez.

Soru: 9n dürüst etiketi biçim kesinliğini 0,451 -> 0,757'ye çıkardı ama kapsama (kesin, biçimli etiket
verilen madde payı) 0,720 -> 0,377'ye düştü. Kesin olmayan maddelerde doğru etimon neden havuza girmiyor;
havuz dışından, kesinliği bozmadan getirilebilir mi?

Ortam: paralel ajan indeksi yeniden kurabildiği için bütün ölçümler KOPYALARLA: kör indeks
`donor9o/index_blind.db` (= `xtr/index_blind.db`, sha256 `956647af…4839`), tam indeks (yalnız puanlama
referansı) `donor9o/index_full.db` (= `data/lexicons/index.db`, sha256 `f9567534…a85`). Ağ kapalı,
getirici yüklenmez.

## 1. Tanı (yalnız ayar: Türkçe TDK+Nişanyan train+dev, 9j TDK ayar, 9j TETTL ayar; `diag9o.py` -> `diag_rows.json`)

Üretim etiketi (a2) KESİN olmayan her madde için doğru etimonun (altın dilde, 9n `match_class` "biçim"
eşi) `donors.db`de aranması (Latin: karşılaştırma biçimi eşit ya da ilk iki harf + ±2 uzunluk içinde
düzenleme uzaklığı ≤ 0,20; Arap yazısı: harekesiz iskelet eşitliği) ve anlam yolu:

| sınıf | anlamı | Türkçe train+dev | 9j TDK ayar | 9j TETTL ayar | toplam | yalnız ar/fa/fr altını |
|---|---|---|---|---|---|---|
| kesin | üretimde biçimli etiket | 170 | 137 | 97 | 404 | 324 |
| ref_yok / altın "diğer" | puanlanamaz | 1 | 73 | 75 | 149 | 12 |
| **(b) sozluk_yok** | etimon `donors.db`de yok | 7 | 46 | 27 | 80 | 52 |
| **(a) kopru_yok** | etimon var; anlam Türkçe kaldı (köprü yok), ortak eşleşme sözcüğü yok | 0 | 101 | 32 | 133 | 61 |
| **(d) anlam_farkli** | etimon var; anlam İngilizce ama etimonun anlamıyla ortak eşleşme sözcüğü yok | 65 | 49 | 32 | 146 | 114 |
| **(a) sinir** | etimon FTS'yle eşleşiyor ama sırasız `LIMIT 200` havuzuna girmiyor | 28 | 16 | 9 | 53 | 43 |
| havuzda, kesin değil | etimon seçildi, kesinlik kuralı reddetti | 13 | 9 | 7 | 29 | 27 |
| havuzda, seçilmedi | başka dil/biçim kazandı | 9 | 8 | 7 | 24 | 19 |

Toplam 1.018 (293 / 439 / 286). Örnekler — (d): sultan "A monarchic title for Sunni Muslim monarchs." ~
سلطان "sultan" (ortak sözcük yok; başlığın İngilizce köprüsü "sultan" olurdu), özür "apology" ~ عذر
"excuse", rahle "lectern, bookrest" ~ رحل; (a) köprü: saykal "Maden, ayna gibi nesneleri parlatmak için
kullanılan cilâ" ~ صيقل, yave "saçma sapan söz" ~ یاوه "idle talk, nonsense", ihya ~ إحياء; (a) sınır: iltihak
"joining with" ~ التحاق, hindiba ~ هندباء, şehir "city" ~ شهر; (b): lağım, hamarat, derdest, tarumar,
endaht (ham kaikki dökümlerinde de yok — 79'un 76'sı; `diag_b.py`; biri `from_turkic` ile elenmiş: lağım).

**(c) biçim karşılaştırması sorun değil**: `comparison_for` Arap/Fars/Yunan/Ermeni yazısında kaikki
`romanization` alanını zaten kullanıyor; sözlükte bulunan etimonların mesafesi çoğunlukla küçük (d ≤ 0,10:
anlam_farkli 80/146, kopru_yok 82/133, sinir 29/53; > 0,35 yalnız 45, bir kısmı referans eşleşmesinin
gürültüsü). Yani doğru etimon **sözlükte var ve biçimce yakın ama anlam yolu onu getirmiyor** — en büyük
sınıflar (d) + (a) = 332 (ar/fa/fr altınında 218).

**Öntarama** (`sim.py`, `sim_an.py`; yalnız ayar): ünsüz iskeleti sorgununkiyle aynı bütün verici maddeleri
(anlamsız) ve en küçük mesafe: kesin olmayan maddelere eklenen biçimlerin yalnız %24–39'u doğru etimon —
çoğu **koşut alıntı** (معصوم Farsçada da var: masum ~ fa, teleskop ~ fa تلسکوپ, korna ~ el κόρνα). Anlam
şartı (geniş sözcük kümesiyle) + **dil = a2'nin önsel+ipucu dili** + Farsça eşleşmede aynı yazı iskeletli
Arapça madde (D1 gibi) eklenince eklenenlerin doğru etimon oranı Türkçe train+dev'de 0,80–0,88'e çıktı;
anlamsız ama dil kapılı arama (d ≤ 0,05) %40–60'ta kaldı (ret).

## 2. Adaylar (3; kod bu commit'te, varsayılan KAPALI: `DONOR_FORM_FIRST = "off"`)

`donor_proximity.form_first_attribution` — a2 etiketi KESİN DEĞİLSE (ya da `attribute_donor` hiç etiket
vermediyse, anlam havuzu boş): L = a2 dili (önsel · biçim ipucu argmax, ar/fa/fr). Aday = ünsüz iskeleti
sorgunun iskeletlerinden biri olan Türkçe verici kümesi maddeleri (`engine/db/donor_skeleton.py`:
`donors.db`den türetilen AYRI dizin `donor_skeleton.db`, `donors.db` yeniden kurulmaz), uzunluk farkı
≤ max(3, |sorgu|/2), etiket mesafesi ≤ T, anlamı dilbilgisi göndermesi değil (`is_form_of`), anlamıyla
**geniş anlam sözcükleri** arasında en az bir ortak içerik sözcüğü. En küçük mesafeli L dilindeki aday
gösterilir; L Arapça ve eşleşen yalnız Farsça ise aynı Arap yazısı iskeletli Arapça madde (≤ T) gösterilir
(`via="fa"`). Bulunursa etiket KESİN (biçimli) olur; **dil a2'ninkiyle aynı kalır**.

Geniş anlam sözcükleri: köprülü anlam (S1) ∪ başlığın İngilizce köprüsü (`english_sense`, anlam İngilizce
olsa da) ∪ aynı biçimli Osmanlıca maddenin anlamı ∪ özgün anlam.

- **C1** `c1` — geniş anlam, T = 0,15.
- **C2** `c2` — C1 + Türkçe anlamın içerik sözcüklerinin tek tek köprüsü (`english_sense(sözcük)`), T = 0,15.
- **C3** `c3` — C2, T = 0,20.

(Önerilen "C2 kaikki romanizasyonu" aday yapılmadı: tanıda (c) sınıfı küçük, romanizasyon zaten kullanılıyor.)

**Biçim ipucu modeli yeniden eğitildi** (9n gibi): `donor_prior.EXCLUDE_GOLDS`a bu 9o altını eklendi;
eğitim tam indeks kopyasından; ar 2.238 -> 2.098, fr 1.567 -> 1.480, fa 412 -> 382 sözcük. Bu, üretim a2
dilini biraz değiştirir (Türkçe train+dev acc_nat 0,726 -> 0,717; TDK ayar 0,573 -> 0,566; TETTL ayar
0,618 -> 0,616). Aşağıdaki "prod" bu yeni modelledir; kabul edilmezse de model yeni altını dışlamak için
kalır.

## 3. Ölçütler

- **(iii) kapsama (BİRİNCİL)** = kesin (biçimli) etiket verilen madde payı (n'ye bölünür). Test: aday vs
  prod eşleştirilmiş McNemar (kesin binom, iki yönlü), Holm (3 aday).
- **(i) biçim kesinliği** = gösterilen biçimlerden doğru etimon (9n tanımı: biçim eşi VE dil sınıfı altınla
  aynı) olanların payı, etimon referanslı maddelerde. Bilgi: eklenen biçimlerin doğru oranı.
- **(ii) doğal ağırlıklı dil doğruluğu** `acc_nat` (9m tanımı). Yapı gereği dil değişmez; yalnız anlam
  havuzu boş (etiketsiz) maddeler biçim-öncelikli aramayla dil kazanabilir -> düşemez. Test 9m işaret çevirme.

## 4. Ayar sonuçları (`ayar.log`, `res_tr.json`, `res_tdk_ayar.json`, `res_tettl_ayar.json`)

| altın | koşul | (iii) kapsama | kazanılan/kaybedilen, Holm p | (i) biçim kesinliği | gösterilen (yanlış) | eklenen doğru | (ii) acc_nat |
|---|---|---|---|---|---|---|---|
| Türkçe train+dev (n=293) | prod | 0,580 | — | 0,735 | 170 (45) | — | 0,7172 |
| | C1 | 0,652 | 21/0, 1e-6 | 0,733 | 191 (51) | 15/21 | 0,7172 |
| | C2 | 0,659 | 23/0, 5e-7 | 0,736 | 193 (51) | 17/23 | 0,7172 |
| | C3 | **0,676** | 28/0, 2e-8 | 0,737 | 198 (52) | 21/28 | 0,7172 |
| 9j TDK ayar (n=439) | prod | 0,312 | — | 0,496 | 133 (67) | — | 0,5655 |
| | C1 | 0,328 | 7/0, 0,016 | 0,500 | 138 (69) | 3/5 | 0,5655 |
| | C2 | 0,369 | 25/0, 1e-7 | 0,503 | 153 (76) | 11/20 | 0,5867 |
| | C3 | 0,387 | 33/0, 7e-10 | 0,491 | 159 (81) | 12/26 | 0,5867 |
| 9j TETTL ayar (n=286) | prod | 0,339 | — | 0,550 | 91 (41) | — | 0,6156 |
| | C1 | 0,371 | 9/0, 0,004 | 0,541 | 98 (45) | 3/7 | 0,6156 |
| | C2 | 0,406 | 19/0, 8e-6 | 0,571 | 105 (45) | 10/14 | 0,6310 |
| | C3 | 0,420 | 23/0, 7e-7 | 0,569 | 109 (47) | 12/18 | 0,6384 |

Kesin etiketlerin hiçbirinde gösterilen biçim değişmedi (`form_changed_in_prod_certain` = 0). (ii) C2/C3
TDK/TETTL ayarda yalnız etiketsiz maddelerde +3/+4 (işaret çevirme p 0,125–0,25). Eklenen yanlışların çoğu
it/el altınında (a2 İtalyanca/Yunanca seçemez: pantufla ~ fr pantoufle, fırkata ~ ar فرقاطة) ve kısa
biçimlerde (sah ~ صح, maya ~ مي); ar/fa/fr doğal oranlı rapor altınında daha az beklenir.

⚠️ Önceden görülen: ayarda (i) üç adayda da prod'a yakın (±0,01), kazanç tamamen (iii)'te.
Korumalar ölçüldü (`k2_guard.log`): bkz. §6.

## 5. Yeni rapor altını (`build_gold.py` -> `gold.json`, tuz `donor9o-tdk-v1`, n = 600)

9n kuralı (TDK GTS v12 `lisan`, tek sözcük, tek dil, kör indekste anlam, grup başına tek madde);
**dışarıda**: TDK+Nişanyan (items + disagreements), 9e, 9f, 9g, 9j-TDK (ayar+rapor), 9j-TETTL (ayar+rapor),
9l, 9m, **9n** altınlarının kelimeleri ve (verici, etimon) grupları. Doğal oranlar (`tum`, kalan sınıflar):
ar 290 / fr 248 / fa 62 (İngilizce anlamlı 151 / 90 / 28). Mevcut ar 3.649, fr 3.465, fa 325. Altın
dosyasında yalnız kelime, sınıf, kısa etimon (TDK ham metni değil).
**Mühür**: `gold.json` sha256 `a025b38c5a971ed22feb20a82060e9ce7d2c09c4eb25051592a7675f737402be`.

## Sızıntı (K2, `k2.json`, n = 600; doğruluk hesaplanmadı)
GEÇTİ. Önceki bütün altınlarla kelime kesişimi 0; ağ kapalı, kör indeks kopyası; getirici yüklenmedi;
kör indekste 600 kelimenin köken sütunları 0 dolu; anlam kör = tam 600/600; dört koşulun etiketi ve
gösterilen biçimi kör vs tam anlamla 600/600 aynı; biçim ipucu modeli 9o altınını dışlama listesinde
taşıyor ve 600 kelimenin 600'ü dışlanmış.

## Rapor (bir kez, bu commit'ten sonra)
`data/cache/work/donor9o/run_rapor.sh` (`harness.py gold 9o`).

Kabul (aday için hepsi):
1. (iii) rapor kapsama prod'dan yüksek VE McNemar Holm p < 0,05.
2. (i) rapor biçim kesinliği **≥ 0,72** (nokta tahmini; 9n rapor değeri 0,757).
3. (ii) rapor `acc_nat` Δ ≥ −0,005 VE anlamlı düşüş yok (işaret çevirme p < 0,05 ve D < 0 ise red).
4. Türkçe train+dev (ayar ölçümü): biçim kesinliği ≥ 0,72 VE `acc_nat` ≥ prod − 0,01 — ölçüldü (C1 0,733,
   C2 0,736, C3 0,737; acc_nat üçünde 0,7172 = prod).
5. Saha `eval-donor` motor ≥ 0,7136, xturkic ayar ≥ 0,7459 (Türkçe verici kümesi dışı, yapı gereği
   etkilenmez; ölçüldü, §6); `eval-borrowing` F'ler aynı (güç yolu değişmez; kabul edilirse koşulur).

Karar: kabul edilenlerden rapor kapsaması en yüksek olan `DONOR_FORM_FIRST` varsayılanı olur (eşitlikte
basit olan: C1 < C2 < C3); hiçbiri -> üretim değişmez (bayrak kapalı, kod bayrak arkasında). Kabul varsa
`make eval-tr-donor` yeniden, README verici satırı, ENGINE_VERSION bir sonraki yama.

## 6. Korumalar (ön kayıt öncesi ölçüldü)
`k2_guard.log`, `saha.json`: Saha `eval-donor` motor **0,7295** (yakınlık 0,6659) — prod, C1, C2, C3'te
AYNI (Türkçe verici kümesi dışı; biçim-öncelikli arama uygulanmaz). 9n'deki 0,7136'dan fark bu işten değil:
ölçüm tam indeks ve çalışma ağacıyla yapıldı, paralel işin (Kiril ҥ düzeltmesi, `orthography.py`,
commitlenmemiş) etkisidir; koşullar arası fark 0. (Koşu `data/eval/donor_id.json`u yeniden yazdı; bu işin
commit'ine girmez.) xturkic ayar verici **0,7459** (9n harness `xt`; `attribute_donor` dili, Türkçe verici
kümesinde 0 madde) — aynı. `eval-borrowing`: güç yolu (`nearest_donor`, `proximity_strength`) değişmedi.

---

## SONUÇ (ön kayıt commit'i 1b02f9d'den sonra, bir kez)

`rapor.log`, `res_9o.json`; mühür doğrulandı (sha256 `a025b38c…02be`). n = 600 (ar 290, fr 248, fa 62);
etimon referanslı 600, anlamlı 599, `attribute_donor` etiketi 435.

| koşul | **(iii) kapsama** | kazanç/kayıp, Holm p | **(i) biçim kesinliği** | gösterilen (yanlış) | eklenen doğru | **(ii) acc_nat** | (ii) işaret D (iyi/kötü), p |
|---|---|---|---|---|---|---|---|
| prod (off) | 0,375 | — | 0,693 (156/225) | 225 (69) | — | 0,5899 | — |
| C1 | 0,425 | 30/0, 2e-9 | 0,718 (183/255) ✗ | 255 (72) | 27/30 | 0,5899 | 0 (0/0) |
| **C2** | **0,458** | **50/0, 4e-15** | **0,720** (198/275) | 275 (77) | 42/50 | **0,6082** | +0,018 (11/0), 0,001 |
| C3 | 0,477 | 61/0, 3e-18 | 0,713 (204/286) ✗ | 286 (82) | 48/61 | 0,6149 | +0,025 (15/0), 6e-5 |

Kesin etiketlerde gösterilen biçim hiçbir koşulda değişmedi (0). C2'nin 50 eklemesinin 12'si anlam havuzu boş
(etiketsiz) maddelerde; 8 yanlışın 5'i doğru dilde başka sözcük (isim ~ عظم, şiddet ~ شديد türev, mesail ~
مسألة), 3'ü dil (daye ~ ar داية, neva/zen ~ Farsça sözcüğün Arapça yazı eşi).

**KARAR: C2 KABUL — `DONOR_FORM_FIRST = "c2"` varsayılan.** Koşullar: (iii) 50/0, Holm p = 4e-15 ✓;
(i) 0,720 ≥ 0,72 ✓ (sınırda: 198/275); (ii) +0,018, düşüş yok ✓; Türkçe train+dev kesinlik 0,736, acc_nat
aynı ✓; Saha/xturkic aynı ✓. C1 (0,718) ve C3 (0,713) (i)'deki mutlak eşiği geçemedi — ikisinde de eklenen
biçimler tabanın kesinliğinden (0,693) daha kesin (0,90 / 0,79), ama ön kayıt eşiği bu altında tabanın
kendisinin 0,72'nin altında olduğunu öngörmemişti; kural gereği red. Not: bu altında taban kesinlik 0,693,
9n raporundaki 0,757'den düşük (yeni örneklem; biçim ipucu modeli de yeniden eğitildi).

Korumalar (kabul sonrası): `eval-borrowing` (`eval_borrowing.py off|c2`, tam indeks kopyası; JSON
`evalb/<mod>/borrowing.json`, paralel işin `data/eval/borrowing.json`unu ezmemek için ayrı): iki çıktı
`trained_at` zaman damgaları dışında aynı (WOLD engine_trained F 0,6582, Türkçe 0,8894 — 9n ile aynı).
Saha 0,7295 / xturkic 0,7459 dört koşulda aynı (§6).

Kalıcı ölçüt (`make eval-tr-donor` eşdeğeri `trdonor.py`, indeks KOPYALARIYLA — paralel iş canlı kör/tam
indeksleri yeniden kurdu; `data/eval/tr_donor.json`, Türkçe TDK+Nişanyan train+dev n=293, A2 bayrakları
kapalı): `(a) etiket (gösterilen)` **0,771** [0,720–0,816], doğal ağırlıklı **0,717** (9n: 0,782 / 0,726 —
fark biçim ipucu modelinin yeniden eğitiminden; en yakın biçimin dili 0,642 / 0,602 aynı); **biçim
kesinliği 0,735 (125/170) -> 0,736 (142/193)**, **kapsama 0,580 -> 0,659**. Kör `(c)` 0,707 -> 0,703
(doğal 0,659 -> 0,656); `(b) dedektör` 0,949 -> 0,952. Önceki önbellek ve rapor `data/cache/work/trdonor_pre9o/`.

Üretim: `DONOR_FORM_FIRST = "c2"`, ENGINE_VERSION bir sonraki yama. Gösterim (CLI/web): biçim-öncelikli
etikette "ar سلطان (sultan) SCA 0.000, … (biçim-öncelikli arama: anlam havuzu dışından, dil önselle uyumlu)";
yardımcı dizin `data/lexicons/donors/donor_skeleton.db` ilk kullanımda `donors.db`den kurulur (~3 sn,
`python -m engine.db.donor_skeleton --build`).
