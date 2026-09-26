# Ön kayıt — "kök yok" başlıkları: türetme kökü yedeği (2026-09-26)

## Durum beyanı
Sınıflama (adım 1) Starling kapalı 240 kelimenin TÜM hata listesi üzerinde
yapıldı (`classify.py`, `probe.py` -> `probe.jsonl`); hata listesi görüldü.
Kurallar sınıflamadan SONRA, benzetim koşusundan ÖNCE yazıldı. Ek envanteri
`engine/nlp/derivation.SUFFIXES` (Zemberek alt kümesi) — DEĞİŞTİRİLMEZ; hata
listesine göre ek eklenmez (`-ArI` uçarı, `+kAr` koçkar, `-sIk` tansık yok).

## Sınıflama sonucu (Starling kapalı, 45 kök-yok + 5 alıntı)
Öncelik sırası v > iii > i > ii/iv > vi (her kelime tek sınıf):
- (v) tanık var, kök yasağı/rekonstrüksiyon engeli: 0 (hiçbirinde
  `withheld_reconstruction` yok; 37 "yok" maddesinin 29'unda indekste HİÇ
  Türk dili tanığı yok, kalanında 1 tanık, yalnız `ir` 4).
- (iii) türemiş, çözümleyicinin kökünün indeks Proto-Türkçe kaydı başvuruyla
  gelenek-denk: 4 (alık, salgın, karım, tatık).
- (iii-b) türemiş ama çözümleyicinin kökü yanlış / kökün kaydı yok: 5
  (boyan, dokun, sincik, sinle, omaca).
- (i) tr/ota miras kaydı var, trk-pro bağı yok: 5 (apaçık, ası, köstek, çapa, ir).
- (ii)+(iv) etimolojik kayıt yok (yalnız etimolojisiz tr / özel ad kaydı);
  Starling açıkken 23'ünün de başlığı Starling'den ve DOĞRU: 23 (besim, bürgü,
  büz, dala, darga, duma, evin, göbelek, güce, koçkar, obuz, savur, sümek,
  sümter, tansık, tatu, toru, tumağan, tın, uçarı, yaprağı, çaput, öke).
- (vi) diğer — başlık var ama `*<kelime>`: 8 (yazım: arka, at, kala, ip;
  eşsesli: dan, kün; türetme derinliği: ordu; türemiş+yanlış kök: kalak).
- Alıntı hükmü (Starling mirası der): 5 (gala, giz, katır, maya, saka).

En büyük sınıf (ii/iv) için UCUZ düzeltme yok: yerel veride kök bilgisi
bulunmuyor; tek bilgi kaynağı Starling. Ucuz müdahale ancak (iii) sınıfına
yapılabilir; aşağıdaki ölçüm onun için.

## Değişim (yalnız başlık; skorlar değişmez)
Başlık damgası "yok — kök belirlenemedi" ise `derivation.analyze(kelime)`
çözümlerinde sırayla ilk çözümün KÖKÜ için `_index_source_proto(kök)`;
biçim bulunursa başlık o biçim, damga
"türetme: *kök + ekler (yerel indeks kök kaydı)". Başka hiçbir madde değişmez.
Kök kaydı yoksa sonraki çözüm denenir; hiçbiri yoksa madde değişmez.

## Seçim (engine_halves A; tek serbestlik)
- S: çözüm sırası — sığ (en uzun kök önce) | derin (en kısa kök önce).
Ölçüt: Starling kapalı gelenek-denk ölçüt, eşitlikte tam; tam eşitlikte sığ.

## Kabul (engine_halves B + savelyev dev)
Tümü gerekli:
1. Starling kapalı B: tam doğrulukta VE gelenek-denk ölçütte artış,
   her ikisi McNemar (kesin, iki yanlı) p<0,05.
2. Starling açık B ve savelyev dev (yerel + starling_yok): tam düşmez.
3. eval-homonym düşmez.
4. Sahte kök: `make eval-controls` bozulmaz (batarya yeniden kurucuyu ölçer,
   başlık yedeği ona dokunmaz); ek olarak 50 fonotaktik sahte kelimeye yedek
   uygulanır — kaç tanesine kök yazılacağı raporlanır (kök uydurma denetimi).

## Beklenti (ölçümden önce yazıldı)
Önceki ön kayıt (`../deriv/PREREG.md`) aynı kökleri buldu, tam eşleşme
kazanmadı (*āl↔*ăl): tam ölçütte artış yapısal olarak beklenmiyor, gelenek-denk
ölçütte en çok +4 (hepsi B'ye düşse bile McNemar p=0,125). Beklenen karar:
KABUL EDİLMEZ; yalnız tanı raporu + docstring.

## Sonuç (benzetim, önbellekli `data/eval/headline.json`, commit d1bd6af; `simulate.py`)
- Seçim (A, Starling kapalı): sığ gelenek 69->70, derin 69->71; tam ikisinde
  50->50 -> **derin** seçildi.
- Rapor (B, Starling kapalı, n=120): tam 48->47 (+0 −1 `yelme`: kendi başlığı
  *yelme Starling *jElme ile tam eşti, yedek *yẹl yazar), McNemar p=1,0;
  gelenek 64->65 (+1 −0), p=1,0. Ölçüt 1 KARŞILANMADI.
- Starling açık: "yok" maddesi hiç yok -> değişmez (A/B 95/100).
- savelyev dev (32): tam 22/21 değişmez; gelenek −1 (`atla` *atla -> *at).
- Sahte kelimeler (50): 1'ine kök yazılır (`akçık` -> *āk, ak + +CIk).
- eval-homonym / eval-controls koşulmadı: ölçüt 1 zaten geçmedi; yedek
  yeniden kurucuya dokunmaz (batarya yapısal olarak değişmez).
- KARAR: kabul edilmedi; arama hattına bağlanmadı. Tanı + docstring.
