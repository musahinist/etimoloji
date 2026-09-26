# Ön kayıt — Clauson 1972 EDT yerel kaynak (PLAN 9i, 2026-09-26)

## Durum beyanı (dürüstlük)
- Ayrıştırıcı (`engine/db/clauson.py`) ve eşleştirici (`engine/fetchers/clauson_edt.py`)
  geliştirilirken 70 "kök-yok ya da `*kelime`" başlıklı kelime (Starling kapalı,
  `data/eval/headline.json`; iki yarı da) yoklandı (`probe.py`): biçim eşiği 0,75 ve
  anlam tabanı 0,50 (görevde verilen) SABİT; yoklamadan sonra yalnız ad/fiil anlam
  ayrımı ve ek (suffix) kayıtlarının dışlanması eklendi (petek/büt-, taş~taš- dersleri).
- Eval koşuları (aşağıdaki A/B) bu belge yazıldıktan SONRA başlatılır.
- Ayrıştırma doğruluğu elle: 3 × 30 rastgele madde (tohum 20260926, 7, 11). İlk iki
  örnek düzeltme için kullanıldı (28/30, 26/30); üçüncü örnek düzeltmelerden sonra
  bakılmamış örnektir (30/30). Ölçüt: madde başı + önek + eşsesli no + (D ise) taban +
  (S ise) gönderme + anlamlardan en az biri doğru.

## Değişim
1. `ClausonEDTFetcher` yerel portföye (Wilkens'ten sonra): biçim (kaba ses sınıfı,
   ≥0,75) + anlam (MiniLM ≥0,50) tutarsa `otk` tanığı; `first_attestation` Clauson
   tanıklığından (Orhun 732 / KB 1069 / DLT 1074 nokta; yoksa lehçe-yüzyıl dönemi,
   yalnız üst sınır).
2. Başlık damgası "yok — kök belirlenemedi" ise `root.old_turkic_base` =
   "X (Clauson)". PT kökü DEĞİL; `proto_turkic` değişmez.

## Ölçü
- Etkin başlık (ayrı sütun): damga "yok" ise ve Clauson tabanı varsa taban, yoksa
  motorun başlığı. Starling başvurusuyla `score_item` (tam, gelenek-denk).
- Kronoloji: `chronology_eval` Starling kapalı; yüzyıl içi (|fark|≤100) ve kapsam.

## Seçim (engine_halves A, Starling kapalı; tek serbestlik)
- G: taban modu — `head` madde başı | `base` doğrudan taban | `chain` kök zinciri.
  Ölçüt: etkin başlık gelenek-denk; eşitlikte tam; tam eşitlikte `base`.

## Kabul (engine_halves B + savelyev dev), tümü:
1. Etki: Starling kapalı B etkin başlık gelenek-denk artışı McNemar (kesin, iki
   yanlı) p<0,05 YA DA Starling kapalı kronoloji yüzyıl-içi (200 kelime, B yarısı)
   artışı McNemar p<0,05.
2. Koruma: Starling kapalı B tam ve gelenek-denk (motor başlığı) düşmez; Starling
   açık B ve savelyev dev (yerel + starling_yok) tam düşmez; Starling açık kronoloji
   yüzyıl-içi düşmez; `eval-homonym` (a) fark ediyor ve (b) E1 payı düşmez.
3. Tanık kesinliği: elle 40 Clauson tanığında (kronoloji + başlık kümesi, rastgele)
   ≥0,85.

## Beklenti (ölçümden önce)
Başlık: 23'lük "yalnız Starling" sınıfında yoklamada 3-6 eşleşme (tansık, savur, tın…),
anlam tabanı Türkçe tanım ~ İngilizce anlam karşılaştırmasında gürültülü; gelenek-denk
kazanç ≤3 → başlıkta anlamlılık beklenmez. Kronoloji: Starling kapalı yüzyıl-içi
0,08'den belirgin artış beklenir (Clauson tanıkları Starling'in MK/KB/Orkh.
etiketleriyle aynı eserler) — ⚠️ DÖNGÜSEL: Starling EDT'ye atıf yapar; artış "bağımsız
doğrulama" değil, aynı kanıta Starling'siz erişimdir.

## Sonuç (koşu 2026-09-26; aynı çalışma ağacı kodu, ETY_CLAUSON kapalı/açık iki kutu;
`analyze.py` -> `analysis.json`; tanık denetimi `witness40.tsv`, ayrıştırma `audit_parse.tsv`)

Veri: 9.107 madde (D 5.851, tabanlı 5.242, gönderme 899, tanıklı 8.032). Ayrıştırma
elle: 28/30, 26/30 (düzeltme örnekleri), 30/30 (düzeltme sonrası bakılmamış örnek);
toplam 84/90 = 0,933.

Kapsam: başlık+kronoloji kümesinin 411 kelimesinin 282'sine en az bir Clauson tanığı
(324 tanık). Kök-yok 49 kelimede (OFF koşusu, tam yanlış) eşleşen 15; 23'lük "yalnız
Starling" sınıfında 5 (bürgü, büz, savur, tansık, tın) — bunların yalnız savur (motor
başlığı *sabur, tam) ve tansık (taban *taŋ, gelenek-denk; ama motor başlığı
*taŋsuk) doğru; bürgü~burkığ, büz~opuz yanlış eşleşme, tın~tıŋ belirsiz.

Seçim (A, Starling kapalı, etkin başlık gelenek-denk): head 69->69, base 69->68,
chain 69->68 -> ölçüte göre `head`. Açık tanık motorun başlığını değiştirdiği için
"yok" damgası ON koşusunda neredeyse kalmıyor; etkin sütun motor başlığıyla aynı.

Rapor (B):
1. Etki — kronoloji Starling kapalı yüzyıl içi (B) 7->56 (+50 −1, p<1e-10); tümü
   17->119/200 (0,085->0,595); nokta kapsam 10->135. ⚠️ DÖNGÜSEL (Starling EDT'ye atıf
   yapar; başvuru etiketleri MK/KB/Orkh. Clauson'un Kaš./KB/Türkü vııı etiketleriyle
   aynı eserler). Başlık: etkin (head) gelenek-denk B 64->61 (+1 −4, p=0,375) — ARTMADI.
   Ölçüt 1 kronolojiyle KARŞILANDI.
2. Koruma — KARŞILANMADI:
   - Starling kapalı B motor başlığı tam 50->47, gelenek 64->61 (+1 −4): Clauson tanığı
     yeniden kurucuya giriyor (gür->*kur, yelme->*jeme, döl->*töl, gön->*koŋ, erte->*erte;
     kazanç savur->*sabur, çapa->*çap).
   - Starling açık kronoloji yüzyıl içi 190->182 (+1 −9, p=0,02). Kayıpların çoğu
     Clauson'un Orhun (Türkü vııı) tanığı verdiği, Starling etiketinin KB/MK dediği
     kelimeler (don~to:n, dör~tö:r, tan~taŋ, ırak~ıra:k, ye~ye:-): başvurunun eksikliği
     olabilir, ama ön kayıtlı ölçüt budur. Yanlış eşleşmeler de var (buna~buntağ,
     yara~yara:- "yaramak").
   - eval-homonym starling_yok (b) E1 payı 0,8281->0,8231, tutarsız başlık 0->1
     (yerel ve starling_yok). (a) fark ediyor 116->119 / 112->119 arttı.
   - Starling açık başlık ve savelyev dev: değişmedi.
3. Tanık kesinliği (elle 40, `witness40.tsv`): 25/40 = 0,625 < 0,85 — KARŞILANMADI.
   Hatalar: çapraz dilli MiniLM (Türkçe tanım ~ İngilizce anlam) 0,50-0,72 aralığında
   ilgisiz çiftleri geçiriyor (çakır~čağır "şıra", kemik~keyik), eşsesli yanlış madde
   (düş~tüš "konak", uç~uč- "uçmak"). Sonradan (doğrulanmamış tanı): anlam ≥0,80 olan
   15 tanığın 15'i doğru.

KARAR: KABUL EDİLMEDİ. Veri, ayrıştırıcı ve fetcher kalır; arama hattına bağlanmadı
(bağlama yaması `wiring.patch`, yalnız deney için).

Sonraki deney önerisi (ön kayıt gerekir, burada koşulmadı): "yalnız tarih" modu —
Clauson tanığı yeniden kurucuya girmez, yalnız `first_attestation` verir; anlam tabanı
çapraz dilde ≥0,80 ya da yalnız İngilizce anlamla; Starling'in nokta tarihi varsa
Clauson yılı kullanılmaz.
