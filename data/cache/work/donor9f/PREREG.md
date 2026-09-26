# 9f — Türkçe verici etiketinde Fransızca -> İtalyanca (ön kayıt)

Tarih 2026-09-26. Bu dosya YENİ rapor bölümü açılmadan commit edilir. Üretim
tabanı: `FRENCH_RULE = "g2"` (9e, dd630e3), `ARABIC_VIA_RULE = "d1"`.

## Tanı (yalnız 9e AYAR bölümü, n=290; `diag.py` -> `diag_ayar.json`)
9e raporunda (görüldü) Fransızca -> İtalyanca 36/200; ayarda 18/100. 18 hatanın
fr/it en yakın adayları (SCA, kör indeks):

- **(b) beraberlik / yakın beraberlik — 10:** d_fr = d_it TAM eşit 5 (alizarin,
  merkantilizm, poz, santimantalizm, serenat): seçim `mesafe − null` ile, İtalyanca
  havuzun null'ı biraz büyük olduğu için İtalyanca kazanıyor. d_fr − d_it ≤ 0,05
  olan 5 daha (anksiyete, mikrobik, organizasyon, psikiyatri, stabilizasyon).
  Dönem/tarih bilgisi bu veride YOK (Wiktionary şablonunda `etydate` nadir;
  motorda tarih sinyali yok) -> dönem ölçütü uygulanamaz; yerine taban oranı
  (H2).
- **(c) sonek ipucu — 9 (b ile örtüşür):** -izm (merkantilizm, panteizm,
  santimantalizm), -syon (jenerasyon, organizasyon, stabilizasyon, tradisyon),
  -loji (fonoloji), -ik (mikrobik). Kök neden: karşılaştırma biçimi İMLADIR;
  Fransız imlası (-tion, -isme, -ique -> `ikue`) Türkçe sesçil yazıma İtalyanca
  imladan (-zione, -ismo, -ico) uzak düşer.
- **erişilemez — 3:** -é ortaçları (devalüe, kontamine, manipüle): Fransızca
  anlam havuzunda uygun aday yok (d_fr ≥ 0,69 ya da hiç).
- **(a) altın şüpheli — elle 18'in hepsi (LLM ön-etiketi, Claude; insan onayı
  yok):** 17 açıkça Fransızca (ses/ek düzeni: -e < -é, ü < u, -syon, -izm, -ik);
  1 şüpheli: serenat (İt serenata da olası). Altın büyük ölçüde doğru; hata
  motorda.

Taban oranı (H2 gerekçesi): 9e ayar havuzunda Wiktionary doğrudan verici
fr 796 / it 89 -> Batı (fr+it) alıntılarının %90'ı Fransızca.

## Adaylar (2; kod bu commit'te sabit, `donor_proximity.WESTERN_RULE`, varsayılan `off`)
Yalnız ETİKET; D1 ateşlediyse dokunulmaz; kazanan zaten fr ise dokunulmaz.

- **H1 (sonek):** Türkçe biçim uluslararası sonekle bitiyor (`FRENCH_SUFFIXES`:
  syon/zyon->ion, izm->isme, ist->iste, loji->logie, grafi->graphie,
  graf->graphe, metre->metre, metri->metrie, ör->eur, ik->ikue) VE Fransızca
  havuzda KARŞILIK sonekle biten aday `DONOR_DISTANCE_THRESHOLD` (0,35) altında
  -> en yakın o aday, `fr`. Kazanan dil fark etmez.
- **H2 (yakın beraberlik):** kazanan `it` VE en yakın Fransızca aday sorguya
  ≤ d_it + ε -> `fr`. **ε = 0,05 sabit** (`FRENCH_TIE_EPSILON`). ⚠️ Dürüstlük
  notu: ε ayar tanısındaki fark dağılımına bakılarak seçildi (0,05'te İtalyanca
  altında 1 kayıp — loca, tam eşit; 0,07'de 2).
- h12 (ikisi) yalnız bilgi.

### Ayar doğrulaması (9e ayar n=290; tabana karşı; `gold9e_ayar.json`)
| kural | ayar | McNemar (aday/taban) | TR train+dev n=293 | xturkic ayar | Saha motor |
|---|---|---|---|---|---|
| off (üretim g2) | 0,5483 | — | 0,6280 | 0,7459 | 0,7136 |
| **H1** | 0,5828 | 10/0 | 0,6416 (4/0) | 0,7459 | 0,7136 |
| **H2** | 0,5793 | 10/1 | 0,6519 (8/1) | 0,7459 | 0,7136 |
| h12 | 0,6000 | 16/1 | 0,6587 (10/1) | 0,7459 | 0,7136 |

(Aynı yollar: 9e `harness.py` yardımcıları; `harness.py tr|xt|saha`.)

## Yeni değerlendirme verisi (`build_gold.py` -> `gold.json`, tuz `donor9f-v1`)
9e havuzu birebir (6.339 madde; aynı süzgeçler, TDK+Nişanyan dışı, kör
indekste anlam) EKSİ 9e altındaki 850 maddenin kelimeleri ve (dil, etimon)
grupları; yalnız fr/it; bölme yok. Kalan: fr 2.060, it 128. Dengeli: **it 120,
fr 120 (n=240)**, hash sırası.

Kalite (`audit30.tsv`, 30 madde, **LLM ön-etiketi — Claude; insan onayı yok**):
27 doğru, 3 şüpheli (vajina: -a Latince/İt biçimi; pantolon: İt pantalone da
anılır; bezelye: Rumca aracı olası). Altının anlamı "Wiktionary doğrudan verici
şablonu".

Sızıntı (K2, `k2.json`, n=240): kör indekste köken sütunları 0 dolu; anlam kör =
tam 240/240; etiket (off ve h12) kör vs tam 240/240 aynı. Anlamda dil adı: 1
(sinyor: "courtesy title for a man of Italian origin" — anlamın kendisi,
etimoloji sütunu değil; not edildi). GEÇTİ.

## Rapor (bir kez, bu commit'ten sonra)
`python data/cache/work/donor9f/harness.py gold9f` (n=240). Birincil: etiket
doğruluğu; H1 ve H2 her biri `off`a karşı iki yönlü kesin McNemar, Holm (2
aday), α = 0,05.

Kabul (aday için hepsi):
1. Rapor artış yönünde ve Holm p < 0,05;
2. Türkçe tr_donor (a) train+dev ≥ 0,628 — ölçüldü (H1 0,642, H2 0,652);
3. Saha eval-donor motor ≥ 0,7136 — ölçüldü (değişmez);
4. xturkic ayar verici tanıma ≥ 0,7459 — ölçüldü (değişmez);
5. eval-borrowing F'ler aynı — yapı gereği; kabul sonrası doğrulanır (çıktı
   scratch'e).

Bilgi (karar ölçütü DEĞİL, görüldü): 9e rapor bölümü (n=560) yeni kurallarla
yeniden hesaplanır.

Karar: ikisi de kabul -> `h12` (iki mekanizma ayrık; ayarda ve TR'de en iyi),
ANCAK yeni raporda h12 doğruluğu ≥ max(H1, H2) ise; değilse rapor doğruluğu
yüksek olan tek aday. Yalnız biri -> o. Hiçbiri -> `off`. Kabul edilen
varsayılan açılır (`WESTERN_RULE`).

Mühür: `gold.json` sha256 `31fcb0fa0be0d13a713f11c52a9e9c904501640c932568103d96b74f32f2f31e`.

---

## SONUÇ (ön kayıt commit'i 81fe3af'tan sonra, bir kez)

Yeni rapor (n=240; `harness.py gold9f`, `rapor.log`, `gold9f_rapor.json`),
etiket doğruluğu, taban üretim (g2, `off`):

| kural | rapor | McNemar (yalnız aday / yalnız off) | ham p | Holm p |
|---|---|---|---|---|
| off (üretim) | 0,3958 (95) | — | — | — |
| **H1** | 0,4375 (105) | 10 / 0 | 0,0020 | 0,0039 |
| H2 | 0,4000 (96) | 7 / 6 | 1,0 | 1,0 |
| h12 (bilgi) | 0,4333 (104) | 15 / 6 | 0,078 | — |

Sınıf başına: Fransızca 74 -> H1 84 / H2 81 / h12 89 (120); İtalyanca 21 ->
H1 21 / H2 15 / h12 15 (120). H2 yeni veride 6 doğru İtalyanca etiketini
bozuyor: taban oranı önceliği (ayarda fr 100 / it 30) dengeli sınıflarda işe
yaramadı. Kalan baskın hata artık İtalyanca -> Fransızca (52), -> Arapça (23),
-> Yunanca (19): motorun İtalyanca kaydı düşük (21/120).

Bilgi (karar ölçütü değil; 9e rapor bölümü görülmüştü, n=560): off 0,5482 ->
H1 0,5875 (22/0), H2 0,5821 (21/2), h12 0,6054.

Korumalar (H1, önceden ölçüldü): TR train+dev 0,628 -> 0,642; Saha 0,7136;
xturkic ayar 0,7459 — tuttu. eval-borrowing: aşağıda.

**KARAR: H1 kabul, H2 red -> `WESTERN_RULE = "h1"` varsayılan açık.**

eval-borrowing (`borrowing_eval.main()` aynı süreçte iki kez, `WESTERN_RULE`
off vs h1; model kaydı devre dışı, çıktı scratch'e — `data/models`e yazılmadı):
547 sayısal alanın 0'ı değişti (yalnız `trained_at`) — tuttu.
