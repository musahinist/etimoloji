# Eşik envanteri

Motorun sayısal eşik ve sabitlerinin dökümü (GAPS G12, plan D5). Bu belge
**hiçbir değeri değiştirmez**; yalnız nerede olduklarını, ne işe
yaradıklarını, nereden geldiklerini ve hangilerinin aynı kavram için
farklı değer taşıdığını kaydeder. Satır numaraları 2026-09-25 tarihli
ağaca göredir; kayarsa sabit adıyla aranır.

Kaynak sütunu:

* **ölçüldü** — yanındaki yorumda bir ölçüm (eval, örneklem, tarama) var;
* **plan/literatür** — plan belgesinden ya da yayından alınmış;
* **elle** — gerekçe yok ya da sezgisel.

Commit sütunu değerin son yazıldığı commit'tir (`git blame`); `dddf52c`
ilk büyük yeniden yapılanmadır ve "o zamandan beri dokunulmadı" demektir.

## 1. Karar eşikleri (kullanıcıya görünen ya da ölçümle ayarlanan)

| Sabit | Yer | Değer | Anlamı | Kaynak | Commit |
|---|---|---|---|---|---|
| `BADGE_THRESHOLDS["validated"]` | config.py:162 | 0.75 (env `ETY_AHVP_T_VALIDATED`) | A-HVP rozeti 🟢 | elle; rozet kalibrasyonu `make eval-badge` ile izleniyor | dddf52c |
| `BADGE_THRESHOLDS["needs_review"]` | config.py:163 | 0.50 | A-HVP rozeti 🟡 | elle | dddf52c |
| `MIN_EVIDENCE_COVERAGE` | config.py:168 | 0.50 | kanıtlanan aşama ağırlığı bunun altındaysa rozet en çok "yetersiz kanıt" | elle | dddf52c |
| (metin) | cli.py:134 | `BADGE_THRESHOLDS`'tan biçimlenir | aşama kararı eşikleri çıktı metninde (eskiden "0.75 / 0.50" gömülüydü; metin gösterilen rozetin verdict_badge olduğunu da söyler) | — | bu commit |
| `BORROWING_THRESHOLD` | nlp/borrowing_detector.py:149 | 0.45 | el skoru ≥ → "alıntı" (yalnız birleştirici modeli yoksa) | elle | 02ae367 |
| `BLOCK_THRESHOLD` | nlp/borrowing_detector.py:162 | 0.55 | el skoru ≥ → miras rekonstrüksiyonu hiç yapılmaz (model yüklüyken de) | iki ayrı eşik gerekçesi ölçüldü (26/400 yanlış engelleme); değerin kendisi elle | 02ae367 |
| birleştirici `threshold` | data/models/borrowing_combiner.json | 0.39 | eğitilmiş olasılık ≥ → "alıntı"; ≥ 0.195 "belirsiz" | ölçüldü (wold/sah/tune, F hedefi) | model dosyası |
| `UNIFORMITY_SUSPICION` | nlp/borrowing_detector.py:328 | 0.85 | Türki tanıklar arası ortalama benzerlik ≥ → `değişimsiz_yayılım` ateşlenir | elle | 02ae367 |
| `NATIVE_THRESHOLD` | nlp/loanword_classifier.py:74 | 0.55 | p_native > → "asli Öz Türkçe" | elle | dddf52c |
| `DETECTOR_NATIVE_FLOOR` / `DETECTOR_LOAN_CEILING` | nlp/loanword_classifier.py (dedektör içe aktarır) | 0.70 / 0.35 | p_native ≥ 0.70 "native", ≤ 0.35 "loanword", arası "uncertain" (yalnız `verdict` etiketi) | elle | dddf52c |
| `NATIVE_SPREAD_THRESHOLD` | nlp/cognate_alignment.py:32 | 0.70 | Türki yayılım > → asli | plan belgesi | dddf52c |
| `LOAN_SPREAD_THRESHOLD` | nlp/cognate_alignment.py:34 | 0.20 | Türki yayılım < → yeni alıntı | plan belgesi | dddf52c |
| `SPREAD_NATIVE_EVIDENCE` / `SPREAD_LOAN_EVIDENCE` | nlp/loanword_classifier.py | 0.40 / 0.12 | yayılım ≥ 0.40 asli puanı, ≤ 0.12 alıntı puanı | elle; 0,70/0,20'ye geçiş ölçüldü (§2) | dddf52c |
| (`SPREAD_NATIVE_EVIDENCE`) | nlp/hypothesis_validation_protocol.py | `spread / SPREAD_NATIVE_EVIDENCE` | A-HVP 4. aşama: yayılım 0.40'ta doyar (sınıflayıcıyla aynı sabit) | elle | dddf52c |
| `HOMONYM_SIMILARITY_FLOOR` | search_engine.py:272 (= `config.MEANING_SIMILARITY_FLOOR`) | 0.30 | tanığın anlamı sorguya bundan uzaksa eşseslinin akrabası sayılır | ölçüldü (MiniLM; eşsesliler 0,13–0,27, gerçekler 0,385+) | 3521bd5 |
| `LOCAL_WITNESS_FLOOR` | search_engine.py:280 | 0.50 | yerel çağdaş dil adayı için anlam tabanı | ölçüldü (50 kelime elle sayım) | a38a9b0 |
| `LOCAL_WITNESS_MARGIN` | search_engine.py:284 | 0.35 | en iyi adaydan bu kadar geride kalan yerel aday elenir | elle | a38a9b0 |
| `_SAME_SENSE_MARGIN` | search_engine.py:601 | 0.10 | en iyi anlamla "aynı anlam grubu" farkı | elle | a38a9b0 |
| `_OTHER_SENSE_MARGIN` | search_engine.py:634 | 0.10 | tanık başka etimolojinin anlamına bu kadar yakınsa eşsesli | elle | 251c63d |
| `_SAME_SENSE_FLOOR` | search_engine.py:637 | 0.50 | iki kendi kaydı bu benzerliğin üstünde → aynı anlam | elle ("ölçümden önce seçildi") | 251c63d |
| `MEANING_FLOOR` | fetchers/starling.py:25 (= `config.MEANING_SIMILARITY_FLOOR`) | 0.30 | Starling adayı sorgunun anlamını taşıyor mu | HOMONYM_SIMILARITY_FLOOR ölçümü (artık aynı sabit) | acbdd41 |
| `MEANING_MARGIN` | fetchers/starling.py:27 | 0.15 | eşsesli Starling kökleri arasında seçim marjı | elle | acbdd41 |
| `THETA_THRESHOLD` | nlp/diachronic_semantic_engine.py:298 | 0.70 | anlam mesafesi > → anlamlar kopuk (A-HVP 3. aşama) | ölçüldü (254 kavram kalibrasyonu) | d6478e4 |
| `COGNATE_THRESHOLD` | nlp/cognate_clustering.py:74 | 0.50 | düzenleme benzerliği > → aynı akraba kümesi (northeuralex da kullanır) | ölçüldü (`make eval-cognates`, train tarandı, dev raporlandı) | debc6c9 |
| `LANGUAGE_THRESHOLDS["cv"]` | fetchers/apertium.py:48 | 0.66 | Çuvaşça akraba benzerlik eşiği | ölçüldü (400 kelime, 47 aday elle) | 62d0eb8 |
| `SIMILARITY_THRESHOLD` | fetchers/khakas_dict.py:51 (= `COGNATE_THRESHOLD`) | 0.50 | Hakasça aday benzerlik eşiği | COGNATE_THRESHOLD ölçümü | 90c17a6 |
| `DONOR_DISTANCE_THRESHOLD` | nlp/donor_proximity.py:45 | 0.35 | SCA mesafesi < → verici yakınlığı alıntı kanıtı | ölçüldü (borrowing_eval ayar yarısı) | 1a8d73e |
| `DONOR_DISTANCE_CEILING` | nlp/donor_proximity.py:48 | 0.60 | sinyal gücü bu mesafede sıfır | elle | 1a8d73e |
| `RUSSIAN_LOAN_DISTANCE` | db/concept_donors.py:55 (= `DONOR_DISTANCE_THRESHOLD`) | 0.35 | Rusça alıntı süzgeci | DONOR_DISTANCE_THRESHOLD ölçümü | 2e24f81 |
| `DEFAULT_PLAUSIBILITY_FLOOR` | nlp/confidence.py:52 | 0.35 | ata biçim makullüğü < → çekimser | ölçüldü | 104fb57 |
| `REJECTED_LOAN_CEILING` | nlp/hypothesis_ranking.py:185 | 0.10 | birleştirici alıntıyı reddedince alıntı skoru tavanı | elle | 396affe |
| `LOAN_REJECTED_INHERITED_FLOOR` | nlp/hypothesis_ranking.py:189 | 0.20 | aynı durumda miras hipotezi tabanı | elle | 396affe |
| `LEARNED_MIN_CONFIDENCE` | nlp/proto_phonology.py:153 | 0.50 | öğrenilmiş örüntü oyu için güven | ölçüldü (dev, n=206 sütun) | ebb54bc |
| `MIN_CONFIDENCE` | nlp/proto_patterns.py:85 | 0.60 | öğrenilmiş oyun elle kurala baskın gelmesi | ölçüldü | ebb54bc |

## 2. Aynı kavram, farklı değer

İlk envanter (638fc7a) yalnız öneriydi. 2026-09-25'te her çakışma kod
okunarak ayrıldı; aynı kavram + aynı değer olanlar tek sabite bağlandı
(davranış birebir aynı, pytest), değer değiştirecek tek aday ölçüldü
(ön-kayıt ve sonuç: `data/cache/work/thresholds/PREREG.md`).

1. **Alıntı/yerli kararı** — *çoğu farklı kavram.* Birleştirici 0,39
   eğitilmiş olasılıktır, `BORROWING_THRESHOLD` 0,45 / `BLOCK_THRESHOLD`
   0,55 el skorudur (ölçek farklı; 0,45/0,55 bilinçli iki eşik: etiket vs.
   rekonstrüksiyonu engelleme). Sınıflayıcı `NATIVE_THRESHOLD` 0,55 ile
   dedektörün 0,70/0,35'i **aynı p_native** üzerindedir ama farklı karar
   türüdür: 0,55 ikili, 0,70/0,35 üçlü (belirsiz bantlı) ve 0,55 bandın
   içindedir → iki modül hiçbir kelimede zıt hüküm veremez. Dedektörün güveni
   her dalda max(p, 1−p) olduğundan bant rozet girdisini (`detect_conf`)
   etkilemez; yalnız `verdict` etiketini belirler ve hiçbir eval bunu okumaz.
   **Yapılan:** bant `DETECTOR_NATIVE_FLOOR` / `DETECTOR_LOAN_CEILING` adıyla
   `loanword_classifier`'a taşındı, dedektör içe aktarır (değer aynı).
   Değer değişikliği önerilmiyor: ölçülebilir etkisi yok.
2. **Türki yayılım oranı** — *aynı kavram, ölçüldü.* Sınıflayıcının
   0,40/0,12'si ve A-HVP'nin `spread / 0.4` doygunluğu aynı "geniş yayılım"
   eşiğidir → tek sabit `SPREAD_NATIVE_EVIDENCE` (A-HVP içe aktarır;
   birebir). `cognate_alignment`'ın 0,70/0,20'si (plan belgesi) yalnız
   `assessment` metnini seçer, hiçbir karara girmez. **Ölçüm (aday S):**
   0,40/0,12 → 0,70/0,20 (sınıflayıcı + doygunluk). Etkilediği tek ölçüt
   `eval-badge` (sınıflayıcı çıktısı yalnız rozet girdilerine, A-HVP skoru
   yalnız eski aşama kararına gider; borrowing / sıralayıcı / headline /
   homonym girdileri değişmez). Train (n=438): rozet AUC 0,8871 → 0,8816
   (Δ %95 [−0,017, 0]), aşama sırası AUC 0,449 → 0,454 ([−0,025, 0,056]),
   aşama skoru AUC 0,557 → 0,553; hiçbir rozet kodu değişmedi. **Reddedildi**
   — değerler 0,40/0,12 kalır. Rapor (bir kez): dev değişmedi (0,9844),
   savelyev_dev rozet AUC 0,648 → 0,614. Metin eşikleri (0,70/0,20)
   karar olmadığı için bırakıldı; "orta" bant metni sınıflayıcının "geniş
   yayılım" notuyla 0,40–0,70 arasında çelişik okunabilir (yalnız metin).
   *Yan bulgu (düzeltilmedi, davranış değişikliği):* `search_engine`
   sınıflayıcıya Türki tanık yokken yayılım **0,0** geçiriyor (→ "dar yayılım"
   alıntı kanıtı), dedektör aynı durumda `None` geçiriyor (kanıt yok).
3. **Anlam benzerliği tabanı** — `HOMONYM_SIMILARITY_FLOOR` ≡
   `starling.MEANING_FLOOR` (aynı MiniLM ölçütü, aynı ölçüm) → ikisi de
   `config.MEANING_SIMILARITY_FLOOR`'dan okur (birebir). 0,50'ler farklı
   kavram: `LOCAL_WITNESS_FLOOR` yerel aday tabanı (ölçüldü),
   `_SAME_SENSE_FLOOR` iki kendi kaydının aynı anlam sayılması; θ 0,70 ters
   yönlü mesafe; marjlar (0,35 / 0,15 / 0,10 / 0,10) ayrı kararlar. Belgelendi.
4. **Akraba benzerliği** — khakas_dict 0,50 ≡ `COGNATE_THRESHOLD` (aynı
   ölçü 1 − Levenshtein/uzun, NorthEuraLex de aynısını kullanır) →
   `SIMILARITY_THRESHOLD = COGNATE_THRESHOLD` (birebir). Çuvaşça 0,66 dile
   özgü, ayrıca ölçülmüş (400 kelime, 47 aday elle) daha sıkı eşik;
   birleştirilmez.
5. **Verici SCA eşiği** — `RUSSIAN_LOAN_DISTANCE = DONOR_DISTANCE_THRESHOLD`
   (aynı kavram, aynı değer; birebir).
6. **Rozet metni** — `cli.py` artık `BADGE_THRESHOLDS`'tan biçimler ve
   bunun A-HVP *aşama* kararı olduğunu, gösterilen rozetin `verdict_badge`
   (TUTARLI / ŞÜPHELİ / DEĞERLENDİRİLMEDİ) olduğunu söyler; eski metin
   "rozet bu sayıya göre verilir" diyordu (rozet v2'den beri yanlış).
7. **A-HVP docstring'i** — eski cömert varsayılanlar tarihçe olarak kaldı;
   bugünkü değerler (ağırlıklar, semantik `1 − mesafe`/θ, triangulation
   formülü ve 0,40 doygunluğu, aşama kararı ≠ gösterilen rozet) eklendi.
8. **NED kopyaları** — `iterative_hypothesis_engine` (tanık biçimi mesafesi)
   ve `northeuralex._similarity` artık `engine.utils.edit_distance.
   normalized_edit_distance` kullanır; boş dizge durumları dâhil 5000 rastgele
   çiftte birebir aynı sonuç.

## 3. Bütün modül düzeyi sayısal sabitler

AST ile çıkarıldı (`engine/` altında, testler hariç; tohumlar ve yol/dizge
ayarları dışarıda). Açıklama yorumun ilk cümlesidir. `neural_reconstruction`
satırları başka bir çalışmanın commitlenmemiş değişikliğidir.

| Sabit | Yer | Değer | Açıklama | Kaynak | Commit |
|---|---|---|---|---|---|
| `HTTP_TIMEOUT_SHORT` | config.py:66 | 3.0 (env ETY_HTTP_TIMEOUT_SHORT) | Üç kademeli zaman aşımı. | elle/belirtilmemiş | dddf52c |
| `HTTP_TIMEOUT_MEDIUM` | config.py:67 | 6.0 (env ETY_HTTP_TIMEOUT_MEDIUM) |  | elle/belirtilmemiş | dddf52c |
| `HTTP_TIMEOUT_LONG` | config.py:68 | 12.0 (env ETY_HTTP_TIMEOUT_LONG) |  | elle/belirtilmemiş | dddf52c |
| `HTTP_MAX_RETRIES` | config.py:70 | 2 (env ETY_HTTP_MAX_RETRIES) |  | elle/belirtilmemiş | dddf52c |
| `HTTP_BACKOFF_BASE` | config.py:71 | 0.3 (env ETY_HTTP_BACKOFF_BASE) |  | elle/belirtilmemiş | dddf52c |
| `CIRCUIT_FAILURES` | config.py:76 | 3 (env ETY_CIRCUIT_FAILURES) | Devre kesici: | ölçüldü | 783c5a3 |
| `CIRCUIT_COOLDOWN` | config.py:77 | 300.0 (env ETY_CIRCUIT_COOLDOWN) |  | elle/belirtilmemiş | 783c5a3 |
| `HTTP_CACHE_TTL_DAYS` | config.py:82 | 90.0 (env ETY_HTTP_CACHE_TTL_DAYS) |  | elle/belirtilmemiş | f8f2f52 |
| `MAX_WORKERS` | config.py:91 | 10 (env ETY_MAX_WORKERS) |  | elle/belirtilmemiş | dddf52c |
| `MAX_VARIANTS` | config.py:93 | 4 (env ETY_MAX_VARIANTS) | Varyant patlamasını sınırlar: | elle/belirtilmemiş | dddf52c |
| `CACHE_TTL_SECONDS` | config.py:122 | 7 * 24 * 3600 (env ETY_CACHE_TTL_SECONDS) |  | elle/belirtilmemiş | dddf52c |
| `MAX_QUERY_LENGTH` | config.py:123 | 64 (env ETY_MAX_QUERY_LENGTH) |  | elle/belirtilmemiş | dddf52c |
| `OLLAMA_TIMEOUT` | config.py:131 | 180.0 (env ETY_OLLAMA_TIMEOUT) |  | elle/belirtilmemiş | dddf52c |
| `OLLAMA_NUM_CTX` | config.py:132 | 1024 (env ETY_OLLAMA_NUM_CTX) |  | elle/belirtilmemiş | dddf52c |
| `OLLAMA_NUM_PREDICT` | config.py:133 | 250 (env ETY_OLLAMA_NUM_PREDICT) |  | elle/belirtilmemiş | dddf52c |
| `OLLAMA_TEMPERATURE` | config.py:134 | 0.15 (env ETY_OLLAMA_TEMPERATURE) |  | elle/belirtilmemiş | dddf52c |
| `MAX_UNTRUSTED_CHARS` | config.py:136 | 1500 (env ETY_MAX_UNTRUSTED_CHARS) | Kazınmış içeriğin isteme girebileceği azami karakter (prompt injection yüzeyini daraltır). | elle/belirtilmemiş | dddf52c |
| `API_PORT` | config.py:142 | 8000 (env ETY_API_PORT) |  | elle/belirtilmemiş | dddf52c |
| `MIN_EVIDENCE_COVERAGE` | config.py:168 | 0.5 (env ETY_AHVP_MIN_EVIDENCE_COVERAGE) | Kanıtlanan aşama ağırlığı bu oranın altındaysa rozet en fazla INSUFFICIENT_EVIDENCE olabilir. | elle/belirtilmemiş | dddf52c |
| `RUSSIAN_LOAN_DISTANCE` | db/concept_donors.py:55 | = `DONOR_DISTANCE_THRESHOLD` (0.35) | Rusça alıntı süzgecinin SCA eşiği. | ölçüldü (devralındı) | 2e24f81 |
| `MIN_LENGTH` | db/donor_index.py:101 | 2 | Anlamsız/çok kısa biçimler indekse alınmaz: | elle/belirtilmemiş | 1a8d73e |
| `MAX_LENGTH` | db/donor_index.py:104 | 24 | Çok uzun maddeler (deyim, çok kelimeli birim) alıntı adayı değildir. | elle/belirtilmemiş | 1a8d73e |
| `_DOUBLE_BYTE_START` | db/starling.py:61 | 1 |  | elle/belirtilmemiş | 92f9fbf |
| `_SPECIAL_NEXT` | db/starling.py:62 | 29 |  | elle/belirtilmemiş | 92f9fbf |
| `FIRST_PAGE` | db/wilkens.py:48 | 15 | Sözlüğün ilk sayfası (0 tabanlı PDF sayfa sırası; basılı s. | elle/belirtilmemiş | 7e57a35 |
| `SMALL_SIZE` | db/wilkens.py:51 | 7.0 | Anlamdaki ve madde başındaki üst simgeler bu boyuttan küçüktür (9,2 pt'ye karşı 5,4). | elle/belirtilmemiş | 7e57a35 |
| `LINE_TOLERANCE` | db/wilkens.py:54 | 2.0 | Aynı satır sayılan taban çizgisi farkı (pt). | elle/belirtilmemiş | 7e57a35 |
| `INDENT` | db/wilkens.py:56 | 6.0 | Alt madde girintisi sütun kenarından ~12 pt; bundan büyük kayma alt maddedir. | elle/belirtilmemiş | 7e57a35 |
| `TURKISH_SAMPLE` | evaluation/badge_eval.py:63 | 10000 | Türkçe altın: | elle/belirtilmemiş | d1cf978 |
| `REFERENCE_MONOLINGUAL_F1` | evaluation/borrowing_eval.py:57 | 0.55 | Miller ve ark. | plan/literatür | 02ae367 |
| `REFERENCE_CROSSFAMILY_F` | evaluation/borrowing_eval.py:60 | 0.87 | List & Forkel 2022, aileler arası ``seabor``. | plan/literatür | 02ae367 |
| `DEFAULT_BINS` | evaluation/calibration.py:80 | 10 | Varsayılan kutu sayısı. | elle/belirtilmemiş | b774cbd |
| `SAMPLE` | evaluation/chronology_eval.py:49 | 200 |  | elle/belirtilmemiş | 2ed89dc |
| `SAME_SOURCE_TOLERANCE` | evaluation/chronology_eval.py:53 | 5 | Aynı eserin iki geleneksel tarihi arasındaki pay. | elle/belirtilmemiş | 2ed89dc |
| `CENTURY` | evaluation/chronology_eval.py:54 | 100 |  | elle/belirtilmemiş | 2ed89dc |
| `REFERENCE_BCUBED_F` | evaluation/cognate_eval.py:51 | 0.89 | List, Greenhill & Gray 2017'de LexStat-Infomap'in aldığı değer. | plan/literatür | 82a6ec4 |
| `PER_CLASS` | evaluation/consistency_audit.py:38 | 30 |  | elle/belirtilmemiş | 63c0896 |
| `K` | evaluation/crossval.py:84 | 5 |  | elle/belirtilmemiş | fb303b0 |
| `STARLING_SAMPLE` | evaluation/headline_eval.py:65 | 250 | Starling TRK alanından örneklenecek kelime sayısı. | elle/belirtilmemiş | 2ed89dc |
| `MAX_ENGINE_WORDS` | evaluation/homonym_eval.py:83 | 150 | Ağır arama koşusu sınırı (kelime başına ~3 sn, iki düzen). | elle/belirtilmemiş | a9af713 |
| `WORST_CASE_EDIT_DISTANCE` | evaluation/metrics.py:49 | 5 | Cevaplanmayan bir madde ED ortalamasına ne kadar katkı yapsın? Altın biçimlerin ortalama uzunluğu ~5; boş cevabın ED'si o uzunluktur. | elle/belirtilmemiş | 104fb57 |
| `GENERATED_FAKE_COUNT` | evaluation/negative_controls.py:88 | 50 | Üretilecek sahte kök sayısı. | elle/belirtilmemiş | a49bee6 |
| `MIN_MEANINGFUL_LENGTH` | evaluation/prediction_test.py:55 | 2 | Sözlükte bulunsa bile **bulgu sayılmayan** eşleşmeler. | ölçüldü | ab2cd5a |
| `MIN_REFS` | evaluation/regularity.py:46 | 3 | CoPaR'ın bir örüntüyü "düzenli" sayması için gereken asgari sütun sayısı. | elle/belirtilmemiş | d17ad87 |
| `MAX_LOCAL_VARIANTS` | fetchers/historical_index.py:45 | 24 | Yerel taramada denenecek en çok ses varyantı. | ölçüldü | 4852a2c |
| `MAX_PER_LANGUAGE` | fetchers/historical_index.py:57 | 3 | Tek bir dil için en çok kaç tanık alınsın. | elle/belirtilmemiş | 4852a2c |
| `SIMILARITY_THRESHOLD` | fetchers/khakas_dict.py:51 | = `COGNATE_THRESHOLD` (0.5) | Aday benzerlik eşiği. | ölçüldü (devralındı) | 90c17a6 |
| `MAX_CANDIDATES` | fetchers/khakas_dict.py:50 | 40 | Aynı sorgu için en çok bu kadar aday (en benzerler); anlam süzgeci pahalı. | elle/belirtilmemiş | 90c17a6 |
| `CONTEXT_CHARS` | fetchers/local_pdf_books.py:39 | 160 | Eşleşme çevresinde döndürülecek karakter sayısı | elle/belirtilmemiş | dddf52c |
| `MAX_MATCHES_PER_BOOK` | fetchers/local_pdf_books.py:41 | 3 | Kitap başına azami eşleşme | elle/belirtilmemiş | dddf52c |
| `MEANING_FLOOR` | fetchers/starling.py:25 | = `config.MEANING_SIMILARITY_FLOOR` (0.3) | Adayın sorgunun bir anlamını "taşıdığı" en düşük benzerlik. | ölçüldü | acbdd41 |
| `MEANING_MARGIN` | fetchers/starling.py:27 | 0.15 | En iyi aday, ikinciyi bu kadar geçmiyorsa seçim yapılmaz. | elle/belirtilmemiş | acbdd41 |
| `PERIOD_START` | fetchers/wilkens_old_uyghur.py:70 | 800 | Dönemin yıl aralığı: | elle/belirtilmemiş | d43c483 |
| `FORM_THRESHOLD` | fetchers/wilkens_old_uyghur.py:74 | 0.6 | Madde başı ile sorgu (ya da tahmini Eski Türkçe biçim) arasındaki en düşük yazılış benzerliği (1 - Levenshtein / uzunluk). | elle/belirtilmemiş | 7e57a35 |
| `MAX_WITNESSES` | fetchers/wilkens_old_uyghur.py:76 | 3 | Bir sorgu için en çok tanık. | elle/belirtilmemiş | 7e57a35 |
| `L2_PENALTY` | nlp/borrowing_combiner.py:105 | 0.01 | L2 düzenlileştirme. | elle/belirtilmemiş | eee5264 |
| `LEARNING_RATE` | nlp/borrowing_combiner.py:107 | 0.5 |  | elle/belirtilmemiş | eee5264 |
| `ITERATIONS` | nlp/borrowing_combiner.py:108 | 3000 |  | elle/belirtilmemiş | eee5264 |
| `TOLERANCE` | nlp/borrowing_combiner.py:110 | 1e-07 | Gradyan adımı bu değerin altına inince eğitim durur (deterministik). | elle/belirtilmemiş | 36d9542 |
| `INNER_FOLDS` | nlp/borrowing_combiner.py:118 | 5 | İç kat sayısı. | elle/belirtilmemiş | 36d9542 |
| `SIMPLICITY_TOLERANCE` | nlp/borrowing_combiner.py:123 | 0.01 | "Daha basit model" toleransı: | elle/belirtilmemiş | 36d9542 |
| `BORROWING_THRESHOLD` | nlp/borrowing_detector.py:149 | 0.45 | Bu eşiğin üstünde kelime **alıntı olarak raporlanır**. | elle/belirtilmemiş | 02ae367 |
| `BLOCK_THRESHOLD` | nlp/borrowing_detector.py:162 | 0.55 | Bu eşiğin üstünde miras rekonstrüksiyonu **hiç yapılmaz**. | ölçüldü | 02ae367 |
| `UNIFORMITY_SUSPICION` | nlp/borrowing_detector.py:328 | 0.85 | Bu benzerlik oranının üstündeki yayılım şüphelidir. | elle/belirtilmemiş | 02ae367 |
| `MATCH_SCORE` | nlp/cldf_lingpy_aligner.py:43 | 3 | Needleman-Wunsch yedek uygulaması için puanlar | elle/belirtilmemiş | dddf52c |
| `CLASS_MATCH_SCORE` | nlp/cldf_lingpy_aligner.py:44 | 2 |  | elle/belirtilmemiş | dddf52c |
| `MISMATCH_PENALTY` | nlp/cldf_lingpy_aligner.py:45 | -1 |  | elle/belirtilmemiş | dddf52c |
| `GAP_PENALTY` | nlp/cldf_lingpy_aligner.py:46 | -2 |  | elle/belirtilmemiş | dddf52c |
| `NATIVE_SPREAD_THRESHOLD` | nlp/cognate_alignment.py:32 | 0.7 | Plan dokümanı: | plan/literatür | dddf52c |
| `LOAN_SPREAD_THRESHOLD` | nlp/cognate_alignment.py:34 | 0.2 | Plan dokümanı: | plan/literatür | dddf52c |
| `COGNATE_THRESHOLD` | nlp/cognate_clustering.py:74 | 0.5 | Bu düzenleme benzerliğinin (``1 - uzaklık / uzun biçim``) üzerindeki çiftler aynı akraba kümesine bağlanır. | ölçüldü | debc6c9 |
| `MIN_SUPPORT` | nlp/cognate_prediction.py:67 | 2 | Bir denkliğin tabloya girmesi için gereken asgari gözlem sayısı. | elle/belirtilmemiş | 658d343 |
| `CONTEXT_MIN_SUPPORT` | nlp/cognate_prediction.py:87 | 3 | ⚠️ **Hece bağlamı: | ölçüldü | df54e0e |
| `C` | nlp/column_model.py:86 | 1.0 | L2 düzenlileştirme — ön kayıtta sabitlendi, CV katlarında ayarlanmadı. | plan/literatür | e193605 |
| `TABLE_CANDIDATE_MIN` | nlp/column_model.py:88 | 0.05 | Tablo dağılımından aday sayılmak için gereken asgari olasılık. | elle/belirtilmemiş | e193605 |
| `K_INNER` | nlp/column_model.py:89 | 5 |  | elle/belirtilmemiş | e193605 |
| `DEFAULT_PLAUSIBILITY_FLOOR` | nlp/confidence.py:52 | 0.35 | **Çekimserlik biçim MAKULLÜĞÜNE bağlıdır, güven skoruna değil.**  ⚠️ Eskiden kalibre güven < 0,15 olunca çekimser kalınıyordu. | ölçüldü | 104fb57 |
| `DEFAULT_ABSTENTION_THRESHOLD` | nlp/confidence.py:56 | 0.0 | Rozet eşiğinin altındaki kalibre skor cevabı ENGELLEMEZ, yalnız rozeti düşürür. | elle/belirtilmemiş | 104fb57 |
| `DiachronicSemanticEngine.THETA_THRESHOLD` | nlp/diachronic_semantic_engine.py:298 | 0.7 | Bu eşiğin üzerindeki mesafe, anlamların birbirinden kopuk olduğunu gösterir. | elle/belirtilmemiş | d6478e4 |
| `MAX_PHONETIC_DISTANCE` | nlp/donor_lexicon.py:48 | 2 | Plan dokümanındaki eşik: | plan/literatür | dddf52c |
| `DONOR_DISTANCE_THRESHOLD` | nlp/donor_proximity.py:45 | 0.35 | Bu SCA mesafesinin altındaki en yakın verici maddesi **alıntı kanıtıdır**. | ölçüldü | 1a8d73e |
| `DONOR_DISTANCE_CEILING` | nlp/donor_proximity.py:48 | 0.6 | Sinyalin gücü bu mesafede sıfıra iner. | elle/belirtilmemiş | 1a8d73e |
| `CHANCE_CONTROL_COUNT` | nlp/donor_proximity.py:66 | 12 | Şans benzerliği denetiminde kullanılacak kontrol kelimesi sayısı. | ölçüldü | eee5264 |
| `CHANCE_MAX_PERCENTILE` | nlp/donor_proximity.py:69 | 0.1 | Gözlenen mesafe, kontrol dağılımının bu yüzdeliğinden düşük olmalı. | elle/belirtilmemiş | eee5264 |
| `SCA_SHORTLIST` | nlp/donor_proximity.py:77 | 40 | SCA hesaplanmadan önce ucuz düzenlenme uzaklığıyla kaç aday elenir?  ⚠️ SCA aday başına ~39 µs; havuz 400 maddeyken sorgu başına 16 ms eder  | elle/belirtilmemiş | eee5264 |
| `ATTRIBUTION_CONTROL_COUNT` | nlp/donor_proximity.py:466 | 12 | Etiket null'ı için kontrol sayısı. | elle/belirtilmemiş | 1facc40 |
| `MIN_STEM_LENGTH` | nlp/historical_morphology.py:66 | 2 | Bir kökün altına düşmemesi gereken asgari uzunluk. | elle/belirtilmemiş | dddf52c |
| `MAX_DEPTH` | nlp/historical_morphology.py:70 | 4 | Azami türetme derinliği (sonsuz döngü koruması). | elle/belirtilmemiş | dddf52c |
| `REJECTED_LOAN_CEILING` | nlp/hypothesis_ranking.py:185 | 0.1 | Birleştirici alıntıyı reddedince (p < eşik) alıntı skoru bu tavanın altında kalır: | elle/belirtilmemiş | 396affe |
| `LOAN_REJECTED_INHERITED_FLOOR` | nlp/hypothesis_ranking.py:189 | 0.2 | Birleştirici alıntıyı reddedince miras hipotezinin tabanı: | elle/belirtilmemiş | 396affe |
| `NATIVE_THRESHOLD` | nlp/loanword_classifier.py:74 | 0.55 | p_native bu değerin üzerindeyse "asli Öz Türkçe" sayılır (ikili karar). | elle/belirtilmemiş | dddf52c |
| `DETECTOR_NATIVE_FLOOR` | nlp/loanword_classifier.py:83 | 0.7 | Dedektörün üçlü kararında "native" alt sınırı. | elle/belirtilmemiş | dddf52c |
| `DETECTOR_LOAN_CEILING` | nlp/loanword_classifier.py:84 | 0.35 | Dedektörün üçlü kararında "loanword" üst sınırı. | elle/belirtilmemiş | dddf52c |
| `SPREAD_NATIVE_EVIDENCE` | nlp/loanword_classifier.py:92 | 0.4 | Katman 2 yayılımı ≥ → asli kanıtı; A-HVP 4. aşama doygunluğu. | elle/belirtilmemiş | dddf52c |
| `SPREAD_LOAN_EVIDENCE` | nlp/loanword_classifier.py:93 | 0.12 | Katman 2 yayılımı ≤ → alıntı kanıtı. | elle/belirtilmemiş | dddf52c |
| `TRIM_THRESHOLD` | nlp/multi_alignment.py:200 | 0.5 | Budama eşiği: | elle/belirtilmemiş | f39bbcf |
| `MAX_PER_COLUMN` | nlp/nbest_reranking.py:82 | 3 | Sütun başına en çok kaç aday taşınır?  ⚠️ Sayı kombinatoriktir: | elle/belirtilmemiş | 75c82f7 |
| `MAX_CANDIDATES` | nlp/nbest_reranking.py:85 | 64 | Toplam aday sayısı tavanı. | elle/belirtilmemiş | 75c82f7 |
| `MIN_ALTERNATIVE_SCORE` | nlp/nbest_reranking.py:98 | 0.05 | Bir alternatifin taşınması için gereken asgari göreli puan. | ölçüldü | 75c82f7 |
| `GENERATION_WEIGHT` | nlp/nbest_reranking.py:101 | 0.6 | Üretim uyumunun sıralamadaki ağırlığı (λ). | elle/belirtilmemiş | 75c82f7 |
| `PLAUSIBILITY_WEIGHT` | nlp/nbest_reranking.py:104 | 0.2 | Ata biçim makullüğünün ağırlığı. | elle/belirtilmemiş | 75c82f7 |
| `D_MODEL` | nlp/neural_reconstruction.py:61 | 128 | Hiperparametreler — ön kayıtta sabitlendi. | plan/literatür | e172029 |
| `HEADS` | nlp/neural_reconstruction.py:62 | 4 |  | elle/belirtilmemiş | e172029 |
| `LAYERS` | nlp/neural_reconstruction.py:63 | 2 |  | elle/belirtilmemiş | e172029 |
| `FF` | nlp/neural_reconstruction.py:64 | 256 |  | elle/belirtilmemiş | e172029 |
| `DROPOUT` | nlp/neural_reconstruction.py:65 | 0.25 |  | elle/belirtilmemiş | e172029 |
| `BATCH` | nlp/neural_reconstruction.py:66 | 32 |  | elle/belirtilmemiş | e172029 |
| `LR` | nlp/neural_reconstruction.py:67 | 0.001 |  | elle/belirtilmemiş | e172029 |
| `GOLD_WEIGHT` | nlp/neural_reconstruction.py:68 | 3 |  | elle/belirtilmemiş | e172029 |
| `WITNESS_DROP` | nlp/neural_reconstruction.py:69 | 0.2 |  | elle/belirtilmemiş | e172029 |
| `BEAM` | nlp/neural_reconstruction.py:70 | 5 |  | elle/belirtilmemiş | e172029 |
| `MAX_SRC` | nlp/neural_reconstruction.py:71 | 256 |  | elle/belirtilmemiş | e172029 |
| `MAX_TGT` | nlp/neural_reconstruction.py:72 | 20 |  | elle/belirtilmemiş | e172029 |
| `EPOCHS` | nlp/neural_reconstruction.py:74 | 15 | Epok sayısı — kat 0 TRAIN'inin iç ayrımında seçildi (bkz. | ölçüldü | e172029 |
| `THREADS` | nlp/neural_reconstruction.py:75 | 4 |  | elle/belirtilmemiş | e172029 |
| `SELECT_BEAM` | nlp/neural_reconstruction.py:454 | 10 | --- sinir üretir, sütun modeli seçer (ön kayıt 2) -------------------------- | plan/literatür | (commitlenmemiş) |
| `DISTANCE_SCALE` | nlp/phonological_feature_engine.py:44 | 0.3 | PanPhon Hamming özellik mesafesini [0,1] aralığına yaymak için bölen. | elle/belirtilmemiş | dddf52c |
| `MIN_EFFECTIVE_LENGTH` | nlp/phonological_feature_engine.py:47 | 4 | Kısa kelimelerde tek bir ekleme/silmenin skoru çökertmesini engelleyen taban uzunluk. | elle/belirtilmemiş | dddf52c |
| `ORDER` | nlp/phonotactic_lm.py:43 | 3 | Bağlam uzunluğu. | plan/literatür | 2be7450 |
| `MIN_SUPPORT` | nlp/proto_patterns.py:77 | 3 | Bir ``(dil, ses)`` çiftinin oy kullanabilmesi için gereken asgari gözlem. | elle/belirtilmemiş | ebb54bc |
| `MIN_CONFIDENCE` | nlp/proto_patterns.py:85 | 0.6 | Öğrenilmiş oyun elle yazılmış kurala baskın gelmesi için gereken güven. | ölçüldü | ebb54bc |
| `DEFAULT_WEIGHT` | nlp/proto_phonology.py:98 | 1.0 |  | elle/belirtilmemiş | fa6937a |
| `LEARNED_MIN_CONFIDENCE` | nlp/proto_phonology.py:153 | 0.5 | Öğrenilmiş örüntü oyunun devreye girmesi için gereken güven. | ölçüldü | ebb54bc |
| `LENGTH_THRESHOLD` | nlp/vowel_length.py:58 | 0.9 | Ata biçimde uzunluk iddia etmek için gereken asgari ağırlıklı tanık gücü. | elle/belirtilmemiş | f1212e7 |
| `HOMONYM_SIMILARITY_FLOOR` | search_engine.py:272 | = `config.MEANING_SIMILARITY_FLOOR` (0.3) | Semantik benzerlik alt sınırı: | ölçüldü | 3521bd5 |
| `LOCAL_WITNESS_FLOOR` | search_engine.py:280 | 0.5 | Yerel çağdaş dil adayları (yazılışla bulunur) için ANLAM alt sınırı. | elle/belirtilmemiş | a38a9b0 |
| `LOCAL_WITNESS_MARGIN` | search_engine.py:284 | 0.35 | Yerel çağdaş dil adayı, en iyi eşleşen adayın bu kadar altındaysa elenir (başka anlamın, yani eşseslinin kaydıdır). | elle/belirtilmemiş | a38a9b0 |
| `_SAME_SENSE_MARGIN` | search_engine.py:601 | 0.1 | En iyi anlamla "aynı anlam grubu" sayılan benzerlik farkı. | elle/belirtilmemiş | a38a9b0 |
| `_OTHER_SENSE_MARGIN` | search_engine.py:634 | 0.1 | Tanık başka bir etimolojinin anlamına başlığınkinden EN AZ bu kadar yakınsa eşsesli sayılır (göreli; mutlak eşik yok — bkz. | elle/belirtilmemiş | 251c63d |
| `_SAME_SENSE_FLOOR` | search_engine.py:637 | 0.5 | İki kendi kaydı bu benzerliğin üstündeyse aynı anlamdır ('I' ~ 'I, me', 'starch for clothes…' ~ 'laundry starch…'); ölçümden önce seçildi. | elle/belirtilmemiş | 251c63d |
| `BROKEN_CHAIN_THRESHOLD` | utils/phonetic_rules.py:12 | 0.34 | Bu değerin altındaki sıralı benzerlik, tanımlı kural da yoksa "kırık zincir" sayılır. | elle/belirtilmemiş | dddf52c |
| `RULE_BONUS` | utils/phonetic_rules.py:14 | 0.15 | Tanımlı bir ses kanunu eşleştiğinde skora eklenen pay. | elle/belirtilmemiş | dddf52c |
