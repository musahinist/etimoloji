# 9j — İtalyanca (ve Rumca/Ermenice) verici etiketi: iki bağımsız YENİ altın (ön kayıt)

Tarih 2026-09-26. Bu dosya YENİ rapor bölümleri açılmadan commit edilir. Üretim
tabanı: `ARABIC_VIA_RULE = "d1"`, `FRENCH_RULE = "g2"`, `WESTERN_RULE = "h1"`;
`OLD_DONOR_LABELS = LABEL_FORM_FILTER = False`. Yalnız ETİKET adımı
(`donor_proximity.attribute_donor`); alıntı GÜCÜ yolu ve `donors.db` değişmez.
Araştırma: `data/cache/work/research/ITALIAN_DATA.md`; önceki: 9g SONUÇ.

## Kaynaklar (git-ignored `raw/`; telifli metin repoda YOK, yalnız künye)
| dosya | kaynak | SHA-256 |
|---|---|---|
| gts.json (gts.tgz içinden) | github.com/ogun/guncel-turkce-sozluk `sozluk/v12/v12.gts.json.tar.gz` (MIT; TDK GTS 12. baskı, 99.236 madde) | `41add9a0350b8424fb4981d7e4db9837211d802b5646fe728f2460bdb6165f36` (tgz `6935628850a8827e51db6813ce8c702ea1a65f6b56258c8d9ad97547dae5b20d`) |
| t01–t09.txt | archive.org `andreas-tietze-tarihi-ve-etimolojik-turkiye-turkcesi-lugati-cilt-01-tu-ba-yayinlari` (TÜBA; 10 cilt `_djvu.txt` OCR; hak alanı boş) | t01 `d4c02878…`, t02 `7c1d0927…`, t03 `2409334c…`, t04 `f8a45cad…`, t05 `c73f39ab…`, t06 `c67cea79…`, t07 `54daca85…`, t08 `0b36adfb…`, t09 `f2e43683…` (tam özetler `raw/` + bu commit'in `sources.sha256`) |
| t10.txt | aynı öğe, Dizin cildi | kullanılmadı (sözlük değil) |

Altın dosyalarında yalnız kelime, sınıf, kısa etimon (≤ 40 karakter), grup anahtarı.

## Altın 1 — TDK (`build_tdk_gold.py` -> `gold_tdk.json`, tuz `donor9j-tdk-v1`) — BİRİNCİL
Madde: GTS `lisan` alanı tek dil (`+`/`,` yok) ve İtalyanca / Fransızca / Rumca+Yunanca /
Arapça / Farsça / Ermenice; tek sözcük, özel ad değil; aynı başlığın kayıtları aynı dili
veriyor; TDK+Nişanyan altınında (items + disagreements) YOK; 9e/9f/9g altınlarındaki
kelimeler ve (verici, karşılaştırma biçimli etimon) grupları YOK; kör indekste Türkçe
anlamı var (ilk anlam = motorun kullandığı; `gloss_lang` en/tr katmanı). Bölme (verici,
etimon) grubuna göre, ayar payı 0,35; rapor tavanları it 200 / fr 150 / el 150 / ar 100 /
fa 100 / hy tümü.

| bölüm | it | fr | el | ar | fa | hy | toplam | İngilizce anlamlı |
|---|---|---|---|---|---|---|---|---|
| ayar | 116 (tümü) | 80 | 114 (tümü) | 60 | 60 | 9 | 439 | 157 |
| **rapor** | **200** (236 mevcut) | 150 | 150 (176) | 100 | 100 | 10 | **710** | 217 (it 29, el 35) |

Düşenler: önceki altınlarda kelime 1.459, etimon grubu 26; kör indekste anlam yok 602.
Kalite (`audit30_tdk.tsv`, 30 madde hash sırası, **LLM ön-etiketi — Claude; insan onayı
yok**): 28 doğru, 2 şüpheli (idea: Batı aracılığı olası; fanyol: etimon zayıf), 0 yanlış.

## Altın 2 — TETTL (`parse_tettl.py` -> `build_tettl_gold.py` -> `gold_tettl.json`, tuz `donor9j-tettl-v1`) — İKİNCİL
Ayrıştırıcı: boş satırla paragraf; başlık = (EOsm./AD./Osm. vb. ve `*` sonrası) ilk küçük
harfli sözcük (+ `/` varyantları; çok sözcüklü deyim başlıkları atlanır); yakın verici =
ilk 450 karakterdeki ILK köken işareti `<` (OCR bozuk ciltlerde `€`, `c`, `—` (5) ve `“` (7–9)
yalnız büyük harfli kısaltmayla); ilk işaret Türkçe/başka dile ya da türetmeye gidiyorsa
madde atlanır; `?`/`veya` etimondan hemen sonra -> şüpheli, altına girmez.
**Ayrıştırma doğruluğu** (`audit_tettl.tsv`; cilt başına 30, ölçüt başlık + yakın verici
doğru; LLM ön-etiketi): c1 30, c2 29, c3 30, c4 29, c5 30, c6 30, c7 30, c8 28, c9 30 /30 —
hepsi ≥ %90 (toplam 266/270 = %98,5); hiçbir cilt dışarıda değil (cilt 10 Dizin). Hatalar:
OCR'de bozulmuş başlık (dahl->dahi, kallık->kall), deyim başlığı (vaz geç, vekil-i harç).
⚠️ Geri çağırma 5 ve 7–9'da düşük (bozuk `<`): 792/493/361/151 madde vs c1 2.115.
Madde: verici it/fr/el/ar/fa/hy (Yun. = el; EYun./Lat./İng. dışarıda), başlık varyantlarından
kör indekste anlamı olan ilki; çelişen verici düşer; TDK+Nişanyan, 9e/9f/9g VE 9j-TDK
altınlarının kelimeleri ve grupları dışarıda. TETTL etimonları hiçbir havuza KONMADI.

| bölüm | it | fr | el | ar | fa | hy | toplam | İngilizce anlamlı |
|---|---|---|---|---|---|---|---|---|
| ayar | 22 (tümü) | 60 | 57 (tümü) | 60 | 60 | 27 (tümü) | 286 | 136 |
| **rapor** | **59** (tümü) | 150 | 112 (tümü) | 100 | 100 | 52 (tümü) | **573** | 285 |

Venedik/Ceneviz işaretli İtalyanca: rapor 11, ayar 3. Kalite (`audit30_tettl.tsv`, LLM
ön-etiketi): 27 doğru, 3 şüpheli (kaba, hop: eşsesli; baloz), 0 yanlış.

**TDK ↔ TETTL uyumu** (TETTL'de ayrıştırılan ve GTS'de tek dilli `lisan`ı olan tüm ortak
başlıklar, altın süzgecinden önce): n = 3.848, aynı dil **3.546 = %92,2**. TETTL sınıfına göre:
ar 1414/1442, fr 1178/1218, fa 499/590 (fa->ar 75: Tietze Farsça aracı, TDK Arapça köken),
**it 246/308 (%80; it->fr 32, it->el 16)**, el 201/264 (el->it 25, el->ar 14), hy 8/26.
⚠️ Yargı bağımlılığı: TDK ve Nişanyan TETTL'yi/Kahane-Tietze'yi kaynak gösterir; "iki bağımsız
kaynak uyuşuyor" denmemeli (ölçüm sızıntısı değil).

## Sızıntı (K2, `k2.json`, n = 2.008 = iki altının tümü; doğruluk hesaplanmadı) — GEÇTİ
Ölçüm AĞ KAPALI (`socket.connect` engelli), kör indeks (`ETY_LEXICON_INDEX`); koşu sonunda
yüklü getirici modülü **yok** (`engine.fetchers.*` = []; TDK getiricisi hiç çağrılmıyor —
`attribute_donor` yalnız `donors.db`, `donors_label.db`, LingPy kullanır). Kör indekste
köken sütunları 0 dolu; anlam kör = tam 2008/2008; etiket (off, i1, i2, g1p) kör vs tam
2008/2008 aynı. Havuz dilleri yalnız kaikki (donors: ar el evn fa fr hy it mn ru; label:
grc lij vec xcl). Anlamda dil adı/"from": 12 (pizza "Italian dish", kef/efrat/fukara
"Arabic …", keşkek "Armenian harissa"; kalanı rastlantı) — anlamın kendisi; motor anlam
metnindeki dil adını kullanmıyor. ⚠️ Türkçe anlamlı maddelerin (tr_edition = Türkçe
Vikisözlük) tanımları çoğu zaman TDK tanımının kopyasıdır: etimoloji içermez; motorun
üretimde kullandığı anlamla aynıdır (akıl sağlığı: yukarıdaki kör = tam eşitliği).

## İtalyanca tanı (yalnız ayar + önceden açılmış 9f raporu; `diag_*.json`, kod değişmedi)
Üretim (off) İtalyanca maddelerde, TDK/TETTL/9f etimonu donors.db `it`'de var mı, anlam
aramasına (`by_sense`, yalnız `it`, 200) giriyor mu:

| küme | doğru | (a2) aramada var, yenildi | (a1) havuzda var, aramaya girmiyor | (b) havuzda yok | etimon yok |
|---|---|---|---|---|---|
| TDK ayar it (n=116; 100 Türkçe anlamlı) | 13 | 3 | 83 (75'i Türkçe anlam) | 17 | — |
| TETTL ayar it (n=22) | 0 | 2 | 11 | 5 | 4 |
| 9f rapor it (n=120, İngilizce anlam; bilgi) | 21 | **41** | **50** | 8 | — |

- **Birinci neden kapsam da imla da değil, ANLAM ARAMASI**: TDK'ya özgü (en-Wiktionary'de
  kullanılmamış) kelimelerin çoğunun kör indeksteki tek anlamı Türkçe (Vikisözlük);
  verici havuzlarının anlamları İngilizce -> aday çıkmıyor. Ayar: Türkçe anlamlı 282
  maddede doğruluk 0,071 (182'sinde hiç etiket yok), İngilizce anlamlı 157 maddede 0,459.
  9f'de (İngilizce anlam) bile 50/120'de doğru etimon aramaya girmiyor (paylaşılan,
  sıralamasız `LIMIT 200`; G2 bu kesmeyi yalnız Fransızca için aşıyor).
- İmla ikincil: I1 normalizasyonu etimonun SCA'sını 9f'de 53/120, TDK'da 40/116 maddede
  küçültüyor (ort. 0,115 -> 0,061; kötüleşme 0), ama 41 "yenildi" maddenin çoğunda İtalyanca
  biçim zaten 0,0 (arma, balo, bomba, diva, fatura) — kaybediş `mesafe − null` seçim
  ölçütünden ve Fransızca/Arapça eş biçimden (dame, bombe, فاتورة) geliyor.

## Havuz (I2; yalnız etiket, `data/lexicons/donors_label/`, git-ignored)
`scripts/download_lexicons.py --label-donors` (Venetan + Ligurian eklendi; künyeler commit)
-> `python -m engine.db.donor_index --build-label`: **vec 4.775, lij 2.080** madde
(grc 67.809, xcl 8.472 değişmedi). donors.db'ye dokunulmadı.

## Adaylar (3; kod bu commit'te sabit, varsayılan KAPALI)
- **I1** `ITALIAN_ORTHO = True`: it/vec/lij adaylarının karşılaştırma biçimi
  `italian_phonetic` ile (çift ünsüz tekleşir; `sci/sc(e,i)`->ş, `ci/c(e,i)`->ç, `gi/g(e,i)`->c,
  `gli`->ly, `gn`->ny, `ch`->k, `gh`->g, `qu`->kv, `c`->k, `h` düşer, `-zione`->`-zyon`).
- **I2** `VENETAN_LABELS = True`: vec/lij anlam kısıtlı maddeleri İtalyanca GRUBUNA katılır
  (ayrı bilet yok, null birleşik havuzdan); etiket `it`, kaynak `kaikki-vec/lij`
  ("İtalyanca (Venedikçe biçimi)").
- **G1'** `OLD_DONOR_LABELS = True` + `OLD_DONOR_MAX = 0.35`: 9g G1 (grc/xcl ayrı grup) ama
  eski dil grubu yalnız en yakın biçimi SCA ≤ 0,35 iken seçilebilir.

### Ayar (rapordan ÖNCE; `res_*_ayar.json`) ve korumalar (`ayar.log`)
| koşul | TDK ayar n=439 | McN (aday/off) | TETTL ayar n=286 | McN | TR train+dev n=293 | xturkic | Saha | 9f rapor (bilgi) | 9e rapor (bilgi) | 9g rapor (bilgi) |
|---|---|---|---|---|---|---|---|---|---|---|
| off | **0,2096** (it 13/116, el 9/114) | — | **0,2448** (it 0/22, el 6/57) | — | 0,6416 | 0,7459 | 0,7136 | 0,4375 | 0,5875 | 0,2851 |
| I1 | 0,2073 (it 13) | 2/3 | 0,2378 | 0/2 | 0,6451 (4/3) | 0,7459 | 0,7136 | 0,4458 | 0,5857 | 0,2851 |
| I2 | 0,2027 (it 15) | 5/7 | 0,2378 (it 2) | 4/6 | **0,6007** (1/13) | 0,7459 | 0,7136 | 0,4875 (13/1) | 0,5768 | 0,2763 |
| G1' | 0,2118 (el 12) | 3/4 | 0,2343 | 6/9 | **0,6246** (4/9) | 0,7459 | 0,7136 | 0,4333 | 0,5804 | 0,4079 (30/2) |

(McNemar sütunu "aday yalnız doğru / taban yalnız doğru".)
⚠️ **Önceden görülen sonuç:** I2 (Fransızca->İtalyanca 11->22, Arapça->İtalyanca 15) ve
G1' (Arapça->Yunanca 6->14) Türkçe TDK+Nişanyan korumasını (≥ 0,6416) DÜŞÜRDÜ; I1 korumayı
geçiyor ama iki ayarda da etkisiz (uyuşmazlık 5 ve 2). Ayarda uyuşmazlıklar az çünkü
maddelerin %62'si Türkçe anlamlı ve orada hiçbir aday aday listesini değiştiremiyor.
Rapor yine BİR KEZ açılır: üretimin bağımsız tahmini ve adayların yönü için.

## Rapor (bir kez, bu commit'ten sonra)
`python data/cache/work/donor9j/harness.py gold tdk rapor` (n = 710, BİRİNCİL) ve
`gold tettl rapor` (n = 573, İKİNCİL). Birincil ölçüt: TDK rapor bölümünün TÜM sınıflarında
etiket doğruluğu (it/fr/el/ar/fa/hy; sınıf kaydırmayı yakalar); I1, I2, G1' her biri `off`a
karşı iki yönlü kesin McNemar, Holm (3 aday), α = 0,05. Bilgi: sınıf başına (it; el+hy),
İngilizce anlamlı katman, TETTL raporu (aynı test, ayrı Holm), Venedik işaretli alt küme.

Kabul (aday için hepsi):
1. TDK rapor artış yönünde ve Holm p < 0,05; TETTL raporu ters yönde anlamlı değil;
2. Türkçe tr_donor train+dev ≥ 0,6416 — ölçüldü (I1 0,6451; I2 0,6007 ✗; G1' 0,6246 ✗);
3. Saha eval-donor motor ≥ 0,7136 — ölçüldü (üçü de aynı);
4. xturkic ayar verici ≥ 0,7459 — ölçüldü (üçü de aynı);
5. eval-borrowing F'ler aynı — yapı gereği; kabul edilirse doğrulanır.

Karar: kabul koşullarını geçenlerden TDK rapor doğruluğu en yüksek olan (eşitlikte basit:
I1 < G1' < I2). Hiçbiri -> üretim değişmez (bayraklar kapalı; havuz ve kod bayrak arkasında).

Mühür: `gold_tdk.json` sha256 `8178db8cf838c892f813a7be1af0d57e553b31a5d1c5fd8682ce3b1326ac7ed1`,
`gold_tettl.json` sha256 `17533b6638f5535d6af5072bae5532f0d172fca29bb60c91185d814343d2f50a`.
