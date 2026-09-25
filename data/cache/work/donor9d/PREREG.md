# 9d — Verici etiketinde "Farsça üzerinden Arapça" (ön kayıt)

Tarih 2026-09-26. Bu dosya DEV raporu açılmadan commit edilir.

## Sorun
ed79519: Türkçe verici etiketi (`donor_proximity.attribute_donor`, A2 kapalı)
0,427; çoğunluk ("hep Arapça") 0,505. Baskın hata Arapça -> Farsça: kazanan
Farsça aday Farsçadaki Arapça alıntının kendisi (davet~دعوت, kısmet~قسمت).

## Adaylar (en fazla 3; kod bu commit'te sabit, `ARABIC_VIA_RULE`)
Yalnız ETİKET değişir; alıntı gücü (`nearest_donor`/`proximity_strength`)
değişmez. Kural yalnız verici listesinde `ar` varken çalışır; Arapçaya
çevrilen etiket Farsça kazanandan geliyorsa `via="fa"`.

- **D1**: kazanan `fa` ve (a) Arap yazısı iskeleti (harekesiz, ة=ت, elif/hemze
  atılmış) aynı olan bir Arapça aday aynı anlam havuzunda var, ya da (b) Farsça
  madde kaikki fa dökümünde etimolojisi "from Arabic" (anahtar: madde + anlam,
  donors.db ile aynı kural) -> `ar`. Temsilci: iskeleti eşleşen en yakın Arapça
  aday, yoksa en yakın Arapça aday, o da yoksa Farsça madde.
- **D2**: kazanan `ar` değilse ve sorgunun Latin ünsüz iskeleti (sınıflar
  dtsz / bp / kgğhqx / cçj / v=w, ünlüsüz, ikizler tekleşmiş, en az 2 ünsüz;
  -Vt sonu atılmış biçim de) bir Arapça adayınkiyle aynıysa -> `ar`.
- **D3**: D1 ya da D2.

## Seçim verisi (yalnız ayar) — ÖLÇÜLDÜ, bilinen
Türkçe altın TRAIN (n=229; `harness.py tr --split train`, yakalanmış
önbellekle 229/229 aynı), etiket doğruluğu, McNemar adaya karşı `off`:

| kural | doğruluk | yalnız aday / yalnız off doğru |
|---|---|---|
| off | 0,410 (94) | — |
| D1 | 0,541 (124) | 30 / 0 |
| D2 | 0,515 (118) | 25 / 1 |
| D3 | 0,555 (127) | 34 / 1 |

X1 xturkic ayar (tune, n=1.379; kör önbellek, `engine_trained` alıntı dediği,
vericisi ru/ar/fa/mn): verici tanıma off 0,5499 (n=551), D1 0,7459, D2 0,6860,
D3 0,7441. (fa->ar 29 -> 54 D1'de: Farsça üzerinden gelen, altını "Farsça"
olan maddeler — bilinen bedel.)

**Önceden seçilen üretim adayı: D1.** Gerekçe: D3'ün TRAIN farkı +3/229 ve
Farsça->Arapça hatasını 12 -> 14 artırıyor; xturkic tune'da D1 >= D3; D1 daha
dar (yalnız Farsça kazananda).

## Rapor (bir kez, bu commit'ten sonra)
Birincil: Türkçe altın DEV bölümü (`tr-gold` assign_split = dev; test HİÇ
okunmaz), (a) etiket doğruluğu, her aday `off`a karşı iki yönlü kesin McNemar,
Holm (3 aday), α = 0,05.

Kabul (aday için hepsi):
1. DEV artış yönünde ve Holm-düzeltilmiş p < 0,05;
2. Saha `make eval-donor` doğruluğu >= 0,70 (şu an 0,714);
3. xturkic tune verici tanıma >= off (0,5499) — yukarıda ölçüldü, tutuyor.

Karar: D1 kabul -> D1 üretimde (varsayılan açık). D1 reddedilip D3 kabul ->
D3; yalnız D2 kabul -> D2; hiçbiri -> üretim `off`.

Bilgi (karar değil): xturkic R1/R2 verici tanıma (önbellekler, etiket yeniden
oynatılır), tam `make eval-tr-donor` yakalaması (üretim varsayılanıyla, A2
kapalı/açık; train+dev), Fransızca hata tanısı.

## Fransızca hata tanısı (TRAIN, D1 ile 53 Fransızca hatası) — düzeltme yok
- 16: anlam havuzu tümüyle Arapça (`by_sense` LIMIT 200, temizlik kapalıyken
  sıralama yok -> ilk 200 FTS eşleşmesi, rowid sırası, `ar` önce kurulmuş):
  aktör, istasyon, teleskop, emperyalizm… Fransızca aday havuza hiç girmiyor.
- Fransızca aday çok yakın ama başka dil kazanıyor (anomali~anomalie 0,0,
  parti, pozitron, bizon, metro): dil null'ı farkı + başka dillerdeki aynı
  uluslararası biçim (Farsça/Ermenice/Yunanca'daki Fransızca alıntılar —
  Farsça üzerinden Arapça'nın aynısı, Fransızca için).
- Uluslararası sözcüklerde İtalyanca/Yunanca biçim Türkçeye daha yakın
  (paralel~parallelo, kontrast, referans).
Ucuz düzeltme adayları (ayrı ön kayıt ister): dil başına LIMIT; D1'in
"from French" karşılığı.

---

## SONUÇ (ön kayıt commit'i 6370422'den sonra, bir kez)

Türkçe DEV (n=64), etiket doğruluğu, `harness.py tr --split dev` (önbellekle 64/64 aynı):

| kural | DEV | McNemar (yalnız aday / yalnız off) | ham p | Holm p |
|---|---|---|---|---|
| off | 0,484 (31) | — | — | — |
| D1 | 0,641 (41) | 10 / 0 | 0,00195 | 0,0039 |
| D2 | 0,609 (39) | 8 / 0 | 0,00781 | 0,0078 |
| D3 | 0,656 (42) | 11 / 0 | 0,00098 | 0,0029 |

Korumalar: Saha `make eval-donor` motor 0,7136 (değişmedi; ar/fa havuzda
yok) — tuttu. xturkic tune verici tanıma 0,5499 -> 0,7459 — tuttu.

Bilgi: xturkic R1 (proximity ateşlenen, n=300) 0,560 -> D1 0,707 / D2 0,643 /
D3 0,690; R2 (n=292) 0,548 -> 0,695 / 0,599 / 0,671.

**KARAR: D1 KABUL, üretimde varsayılan açık (`ARABIC_VIA_RULE = "d1"`).**
DEV'de D1 çoğunluk tabanını (DEV'de Arapça payı) geçip geçmediği tam
yakalama raporunda (`tr_donor_dev.json`).

Tam yakalama (üretim D1, A2 kapalı; bir kez; `make eval-tr-donor` yalnız off):

| | train+dev n=293 | DEV n=64 |
|---|---|---|
| çoğunluk | 0,505 | 0,531 |
| (a) etiket | 0,427 -> **0,563** (çoğunluğa McNemar p=0,064) | 0,641 (p=0,21) |
| (c) kör | 0,379 -> **0,515** (p=0,83) | 0,625 (p=0,31) |
| (b) dedektör | 0,932 -> 0,939 | — |

Bağımsız sistemler artık çoğunluğun altında değil, ama anlamlı üstünde de
değil. Kalan baskın hata Fransızca -> Arapça (21; anlam havuzunun Arapçayla
dolması, yukarıdaki tanı) ve Farsça -> Arapça (13). A2 açık yakalama
yapılmadı (eski önbellekler `*_pre9d.jsonl`).
