# Ön kayıt — Eren (1999) + Gülensoy (2007): "kök yok" başlıklarına kaynaklı kök notu (PLAN 9i, 2026-09-26)

## Durum beyanı (dürüstlük)
- Ayrıştırıcı (`engine/db/eren_gulensoy.py`) geliştirilirken 23'lük "yalnız Starling" sınıfının
  kelimeleri (`../noroot/PREREG.md`) iki kaynakta elle yoklandı ve iki geliştirme örneği
  (Eren tohum 1, 2; Gülensoy tohum 1, 2; 30'ar madde) üzerinde kurallar düzeltildi.
  Starling başvurularına (`data/eval/headline.json` → `reference`) 23'lük listede bakıldı
  (noroot tanısından zaten biliniyordu). Aşağıdaki simülasyon bu belge commit'lenmeden
  koşulmadı.
- Ayrıştırma denetimi için bakılmamış örnek: tohum 3 (her kaynaktan 30 madde), bu belgeden
  SONRA açılır; sonrasında ayrıştırıcı değişirse yeni bakılmamış örnek (tohum 4) gerekir.

## Ayrıştırma doğruluğu (kaynak başına; eşik ≥ 0,90, altındaki kaynak ölçüme ALINMAZ)
Madde doğru ⇔ (1) madde başı doğru; (2) kök: kaynak kendi sesiyle (reddettiği görüş değil)
bir kök/taban veriyorsa çıkarılan kök o olmalı (OCR bozuk kök yanlış sayılır; hiç çıkmaması da
yanlış), vermiyorsa kök boş olmalı; (3) köken: kaynak açık hüküm veriyorsa (`< Dil`, Türk
dillerinden denk listesi `~`, "Kökenini bilmiyoruz") çıkarılan hüküm o olmalı; açık hüküm
yoksa boş ya da denk listesinden "Türkçe" kabul. Eski Türkçe biçim, gönderme ve atıflar
denetim dışı (başlığa girmez). Gülensoy yalnız Türkçe kökenli sözcükleri kapsar: köken
alanı sabit "turkish".

## Değişim (yalnız başlık; skorlar, tanıklar, yeniden kurucu DEĞİŞMEZ)
Starling kapalı başlık damgası "yok — kök belirlenemedi" ise: sorgu kelimesinin TAM anahtar
eşleşmesi (`key` = küçük harf, tire atılmış; Eren fiil mastarı `-mAk` için ayrıca `verb_key`)
ile kaynak kaydı aranır; köken `loan`/`unknown` olan kayıt kök VERMEZ; gönderme maddeleri
(`X bk. Y`) izlenmez. Kök bulunursa başlık o biçim (aşağıdaki gösterim dönüşümüyle),
damga "Eren: *çap- + -gut (Türk Dilinin Etimolojik Sözlüğü, 1999)" biçiminde. Başka hiçbir
madde değişmez (Starling/indeks kökü olan maddeye dokunulmaz).

### Gösterim denkliği (Türkiye Türkolojisi çeviriyazısı → Starling/EDAL; VERİYE BAKILMADAN,
yalnız iki geleneğin yazım kurallarından)
`*` ve sondaki `-`, uzunluk `:` atılır; `ğ` → `g` (EDAL art damak sızıcısını ayrı yazmaz,
*g ile gösterir); söz sonunda ya da ünsüzden önce `ng` → `ŋ` (Türkiye yayınlarında ŋ'nin
dizgi karşılığı); `ñ` → `ŋ`; `é` → `e`; `â î û` → `a i u` (düzeltme işareti = uzunluk/
incelik, EDAL'da makron). Sonra `score_item` (tam ve `same_root_across_traditions`) aynen.

## Seçim (engine_halves A, Starling kapalı; iki serbestlik, 2×2)
- S: kaynak sırası — `E>G` (önce Eren) | `G>E`.
- M: biçim — `R` yalnız açık kök | `RO` kök yoksa sözün Eski/Orta Türkçe biçimi.
Ölçüt: A yarısı gelenek-denk; eşitlikte tam; tam eşitlikte `E>G`, `R`.

## Kabul (engine_halves B + savelyev dev), tümü:
1. Etki: Starling kapalı B gelenek-denk artışı, McNemar (kesin, iki yanlı) p < 0,05.
2. Koruma: Starling kapalı B tam düşmez (net); Starling açık B tam ve gelenek-denk düşmez;
   savelyev dev (yerel + starling_yok) tam düşmez; `eval-homonym` (bağlandıktan sonra
   gerçek koşu) (a) fark ediyor ve (b) E1 payı düşmez, tutarsız başlık artmaz.
3. Sahte kök: 50 fonotaktik sahte kelimenin (`negative_controls.GENERATED_FAKES`) kaçına kök
   yazıldığı raporlanır (tam eşleşme olduğu için 0 beklenir; >0 ise ret).

## Ek raporlar (kabul ölçütü değil)
- "Kök yok" sınıfı (Starling kapalı, iki yarı) ve 23'lük "yalnız Starling" sınıfında
  kapsam (kök verilen) ve doğruluk (tam / gelenek-denk).
- Bağımsızlık / ikinci görüş: `starling/yerel/tümü` 240 kelimesinde kaynağın kök verdiği
  kelimelerde Starling başvurusuyla gelenek-denk uyuşma oranı (kaynak başına ayrı).
  ⚠️ Eren Räsänen/Clauson'a, Starling de aynı kaynaklara atıf yapar: uyuşma kısmen ortak
  üst kaynaktandır; Eren EDAL'dan (2003) önce yayımlandığı için ayrışma bilgilendiricidir.

## Veri
Simülasyon: `git show HEAD:data/eval/headline.json` (commit ebb1c62, ENGINE 4.3.1; noroot ve
clauson ön kayıtlarıyla aynı önbellekli koşu). Çalışma ağacındaki `data/eval/headline.json`
paralel ajanın koşusu olduğu için KULLANILMAZ.

## Beklenti (ölçümden önce)
23'lük sınıfta yoklama: çaput (çap-), uçarı (uç-), savur (sav-), tansık (OCR `tang`), obuz
(ob-), bürgü (bü-), dala (tāla-), göbelek (OCR `gövtelek`) — gelenek-denk kazanç en çok ~4-5,
büyük bölümü A yarısında olabilir; B'de p<0,05 için ≥6 kazanç / 0 kayıp gerekir. Beklenen
karar: KABUL EDİLMEZ (güç yetersiz); veri + modül kalır, bağlanmaz.

## Sonuç (koşu 2026-09-26; `simulate.py` -> `analysis.json`; denetim `audit_eren.tsv`, `audit_gulensoy.tsv`)

Veri: Eren 2.654 madde (292 gönderme; köken: alıntı 1.371, Türkçe 622, bilinmiyor 219,
hükümsüz 442; açık kök 329, Eski/Orta Türkçe biçim 351). Gülensoy 7.312 madde (açık kök 4.007,
OT biçimi 1.445).

Ayrıştırma (bakılmamış tohum 3, ölçüt yukarıda, ayrıştırıcı bundan sonra DEĞİŞMEDİ):
- Eren 28/30 = 0,933 — GEÇTİ. Hatalar: `borazan` (`< boru + Far -zan`, OCR `+` -> `*`, taban
  kaçtı), `şimşek` (kök `balkı-` başka sözün, Eren şimşek için kök vermiyor).
- Gülensoy 14/30 = 0,467 — KALDI, ölçüme ALINMADI. Hatalar çoğunlukla OCR: `<` işareti
  `>`/`€`/`x` okunmuş, `+` -> `t`, söz başı `t` -> `f`; ilk ses denetimi doğru kökleri de
  eliyor (`um-` < `*üm-`, `uyar-` < `odğur-`).

Seçim (A, Starling kapalı; S artık tek değerli — yalnız Eren): R ve RO eşit (gelenek 69->70,
tam 50->51; +1 çaput) -> `R`.

Rapor (B):
1. Etki — Starling kapalı B: tam 50->50, gelenek 64->64 (+0 −0, p=1,0). KARŞILANMADI.
2. Korumalar — Starling açık A/B, savelyev dev (yerel + starling_yok): değişim yok (kök-yok
   maddesi Eren'de yok). eval-homonym: bağlanmadığı için koşulmadı.
3. Sahte kelimeler: 0/50 köke eşleşti.

Sınıflar (iki yarı): kök-yok 49 maddenin 4'ünde Eren kök veriyor (köken *kök — zaten doğruydu,
köstek *köste, obuz *ob, çaput *çap); gelenek-denk 2. 23'lük "yalnız Starling" sınıfında kapsam
2/23 (çaput doğru, obuz *ob ≠ *ōpuŕ). Sınıfın geri kalanında Eren ya madde vermiyor (tansık,
tın, savur…), ya kök önermiyor (uçarı "Yalnız Türkçede", koçkar "Bk. koç", evin yalnız OT evin),
ya da alıntı/bilinmiyor diyor (büz < Fr buse — Starling'in büz'ünden başka söz; sümek, sümter).

İkinci görüş / bağımsızlık (240 kelime, M=R): Eren 24 kelimede açık kök veriyor; Starling
başvurusuyla gelenek-denk 9/24 = 0,375; Eren kökü Starling biçiminin öneki (ya da tersi; derinlik
farkı: burun < bur-, katır < kat-) sayılırsa 16/24. Gerçek ayrışma ~8 (dil < ti-, yonca < yor-,
üzengi < üz-, kar < ka-… ve 2 OCR gürültüsü: seyrek `seöre`, sakak `sakâ`). Bilgi: Gülensoy
(denetimi geçmedi) 79 kelimede kök, gelenek-denk 21, önek-uyumlu 42.

KARAR: KABUL EDİLMEDİ. Veri (künye), ayrıştırıcı ve simülasyon kalır; arama hattına
bağlanmadı; `engine/db/eren_gulensoy.py` docstring'ine sonuç yazıldı. Beklentiyle uyumlu, ama
kapsam beklenenden de dar: Eren'in 2.654 maddesinin yarısı alıntı, açık kök yalnız 329.
