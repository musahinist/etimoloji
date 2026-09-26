# 9e — Türkçe verici etiketinde Fransızca hataları (ön kayıt)

Tarih 2026-09-26. Bu dosya RAPOR bölümü açılmadan commit edilir. Tanı:
623ecea / `data/cache/work/donor9d/PREREG.md` ("Fransızca hata tanısı").

## Yeni değerlendirme verisi (Türkçe DEV 9d'de açıldı)
`build_gold.py` -> `gold.json` (tohum/tuz `donor9e-v1`). Türkçe Wiktionary
(kaikki en, `data/lexicons/tr.jsonl.gz`) maddeleri: ilk verici şablonu
(bor/bor+/lbor/der; önünde en çok `inh|tr|ota`) fr/frm, ar, fa/fa-cls, it/vec,
el/gkm; önünde türetme/calque şablonu yok; aynı kelimenin kayıtları vericide
anlaşıyor; özel ad/ek değil; TDK+Nişanyan altınında (`turkish_loanwords.json`
items + disagreements, TÜM bölümler) YOK (345 atıldı); kör indekste Türkçe
anlamı var. Bölme ETİMONA göre (sha256(`donor9e-v1:<dil>:<etimon>`) % 1000 <
350 -> ayar). Dengeli örnek (sınıf tavanı, hash sırası):

| | fr | ar | fa | it | el | toplam |
|---|---|---|---|---|---|---|
| ayar | 100 | 80 | 50 | 30 | 30 | 290 |
| rapor | 200 | 160 | 100 | 50 | 50 | 560 |

Kalite (`audit40.tsv`, 40 rastgele madde, **LLM ön-etiketi — Claude; insan
onayı yok**): 39 doğru, 1 şüpheli (gazi: Wiktionary Farsça aracı, TDK Arapça).
Altının anlamı "Wiktionary doğrudan verici şablonu"dur.

Sızıntı akıl sağlığı (K2 benzeri, `k2.json`, ayar n=290): etiket adımı sözlük
indeksinden YALNIZ Türkçe anlamı okur; kör indeks kopyasında
(`index_blind.db`, xtr kör indeksinin kopyası — paralel yeniden kurulumdan
korunmak için) bu 290 kelimenin origin/donor_lang/donor_form/etymology/cognates
sütunları 0 dolu; anlam kör = tam indeks 290/290; etiket kör vs tam indeks
anlamıyla 290/290 aynı. Anlamda dil adı/"from" geçen 1 (tabak: "the dish from
which…" — sızıntı değil). GEÇTİ.

## Adaylar (2; kod bu commit'te sabit, `donor_proximity.FRENCH_RULE`)
Yalnız ETİKET; alıntı gücü (`nearest_donor`/`proximity_strength`) değişmez.
Karşılaştırma tabanı: üretim (`off` = D1 açık, paylaşılan 200'lük havuz).

- **F2** = F1 (`by_sense(per_language=True)`: HER verici dil için ayrı
  N = 200, `max_candidates`) + Fransızca aracılı.
- **G2** = G1 (paylaşılan 200'lük havuz AYNEN + yalnız Fransızcaya ayrı
  N = 200'lük havuz) + Fransızca aracılı.

"Fransızca aracılı" (D1'in karşılığı): D1 ateşlemediyse ve kazanan dil
fa/hy/el ise, (a) Fransızca havuzda kazanan biçime SCA ≤ 0,15 VE sorguya
uzaklığı ≤ kazananın uzaklığı + 0,10 olan aday varsa -> en yakını; yoksa
(b) kazanan madde kendi kaikki dökümünde etimolojisi "from French" ise
(anahtar madde+anlam) -> en yakın Fransızca aday (yoksa madde). Etiket `fr`,
`via` = kazanan dil.

### Nasıl buraya gelindi (ayar + bilinen Türkçe train+dev; rapor açılmadı)
İlk taslak (spesifikasyondaki F1/F2; `draft1/`): F2 D1'den ÖNCE çalışıyor ve
"yakın" eşiği 0,30 idi. Türkçe train+dev'de F2 = 0,563 (taban 0,563) ama
Arapça->Fransızca 28: SCA ses sınıfları kaba, 0,30'da vatan~Bhoutan,
sultan~question "yakın". Düzeltildi: D1 önce, eşik 0,15 + sorguya uzaklık
koşulu (bir kez; sonra dokunulmadı). F1 tek başına Türkçe train+dev'i 0,563 ->
0,529 düşürüyor (Arapça 119 -> 99: eskiden rakip havuzlar boş olduğu için
"doğru" çıkan şans eşleşmeleri — badire~bahira, tahmis~tammuz — artık başka
dile kayıyor) -> koruma dışı, aday değil; bu yüzden G ailesi eklendi.
G1 tek başına nötr. 2 aday ön kayda alındı.

| kural | ayar n=290 | McNemar (aday/taban) | TR train+dev n=293 | xturkic ayar verici | Saha motor |
|---|---|---|---|---|---|
| off (üretim) | 0,4655 | — | 0,5631 | 0,7459 | 0,7136 |
| F1 | 0,4586 | 17/15 | 0,5290 | 0,7568 | 0,7136 |
| **F2** | 0,5276 | 35/17, p=0,018 | 0,5939 | 0,7568 | 0,7136 |
| G1 | 0,4828 | 10/5 | 0,5631 | 0,7459 | 0,7136 |
| **G2** | 0,5483 | 30/6, p=7e-5 | 0,6280 | 0,7459 | 0,7136 |

(TR train+dev = `make eval-tr-donor` (a) etiket adımının yeniden oynatımı,
taban 165/293 = 0,563 birebir; xturkic = 9d yolu, kör önbellek, `engine_trained`;
Saha = `donor_id_eval.run()` tam indeks ortamı, "motor"; Saha etiketleri
beş kuralda birebir aynı — ru/mn/evn havuzu.)

## Rapor (bir kez, bu commit'ten sonra)
`python data/cache/work/donor9e/harness.py gold rapor` (n=560). Birincil:
etiket doğruluğu, her aday `off`a karşı iki yönlü kesin McNemar, Holm (2
aday), α = 0,05.

Kabul (aday için hepsi):
1. Rapor artış yönünde ve Holm p < 0,05;
2. Türkçe tr_donor (a) train+dev ≥ 0,563 — yukarıda ölçüldü (F2 0,594, G2 0,628);
3. Saha eval-donor motor ≥ 0,70 — ölçüldü (0,7136, değişmez);
4. xturkic ayar verici tanıma ≥ 0,7459 — ölçüldü (F2 0,7568, G2 0,7459);
5. eval-borrowing F'ler aynı — yapı gereği (etiket güce girmez); kabul
   sonrası varsayılan açıkken `make eval-borrowing` ile doğrulanır.

Karar: ikisi de kabul -> **G2** (önceden seçildi: ayarda ve TR'de daha iyi,
daha dar değişiklik — Arapça havuzuna dokunmaz). Yalnız biri -> o. Hiçbiri ->
üretim `off`. Kabul edilen varsayılan açılır (`FRENCH_RULE`).

Mühür: `gold.json` sha256 `773746a0be6475539eddd5c116104986375f9d7e4d42c35f70c428029fa9c505`.

---

## SONUÇ (ön kayıt commit'i bf1eeea'dan sonra, bir kez)

Rapor (n=560; `harness.py gold rapor`, `rapor.log`), etiket doğruluğu:

| kural | rapor | McNemar (yalnız aday / yalnız off) | ham p | Holm p |
|---|---|---|---|---|
| off (üretim) | 0,4625 (259) | — | — | — |
| F2 | 0,5500 (308) | 68 / 19 | 1,2e-7 | 1,2e-7 |
| G2 | 0,5482 (307) | 57 / 9 | 1,2e-9 | 2,4e-9 |

Bilgi (aday değil): F1 0,4821 (30/19), G1 0,4839 (20/8). Sınıf başına G2:
Fransızca 66 -> 122/200, Arapça 131 -> 124/160, Farsça 36/100, İtalyanca
9/50, Yunanca 17 -> 16/50. Kalan baskın hata Fransızca -> İtalyanca (36;
tanıdaki 3. tür: İtalyanca biçim gerçekten daha yakın) ve Farsça -> Arapça (26).

Korumalar (önceden ölçüldü): TR train+dev 0,563 -> 0,628; Saha 0,7136;
xturkic ayar 0,7459 — tuttu. eval-borrowing: aşağıda.

**KARAR: ikisi de kabul; ön kayda göre G2 üretimde (`FRENCH_RULE = "g2"`).**

eval-borrowing (G2 varsayılan açık, `make eval-borrowing` modülü): 348 sayısal
alanın 0'ı değişti (yalnız `trained_at`; geri alındı) — tuttu.
