.PHONY: help install test test-live lint fix coverage clean serve web \
        data gold donors patterns gold-agreement regularity sigtyp expert-review turkish-gold eval eval-baseline eval-cognates eval-borrowing eval-calibration eval-controls eval-cv audit eval-llm eval-prediction eval-headline eval-donor eval-chronology starling correspondences calibrate lexicons lexicon-index chains predict-lock predict-verify apertium wilkens semantic dialect bootstrap column-model

help:
	@echo "install     - .venv oluştur ve bağımlılıkları kur"
	@echo "test        - Ağsız test paketi (yavaş uç nokta testleri hariç, paralel)"
	@echo "test-all    - Yavaş testler dahil hepsi (commit öncesi)"
	@echo "test-live   - Canlı kaynak testleri (ağ gerektirir)"
	@echo "lint        - ruff denetimi"
	@echo "fix         - ruff otomatik düzeltme"
	@echo "coverage    - Kapsam raporu (%90 eşiği)"
	@echo ""
	@echo "  --- veri ve değerlendirme ---"
	@echo "data           - CLDF veri kümelerini indir (savelyev, hruschka, starostin, robbeets)"
	@echo "gold           - Altın standardı kur, kavram bazlı böl ve test setini mühürle"
	@echo "eval-baseline  - Taban çizgisi: motor vs trivial sistemler (dev bölümü)"
	@echo "patterns       - Ata ses örüntü tablosunu TRAIN'den öğren (denetimli katman)"
	@echo "column-model   - Sütun modelini (ata ses / ∅) TRAIN + Starling'den öğren"
	@echo "eval           - Rekonstrüksiyon ölçümü (dev bölümü)"
	@echo "eval-cognates  - Akraba tespiti B-Cubed F (LexStat-Infomap taban çizgisine karşı)"
	@echo "regularity     - CoPaR: verinin ne kadarı düzenli denkliklerle açıklanıyor (üst sınır)"
	@echo "gold-agreement - Altın standartlar arası uyum: otomatik sistemin GERÇEKÇİ tavanı"
	@echo "sigtyp         - Altın standardı SIGTYP 2022 paylaşılan görev biçiminde dışa aktar"
	@echo "expert-review  - Uzman derecelendirmelerinden Krippendorff ordinal α (uzman girdisi gerekir)"
	@echo "turkish-gold   - TDK + Nişanyan'dan Türkçe altın alıntı kümesi (Wiktionary'den bağımsız)"
	@echo "eval-borrowing - Alıntı tespiti P/R/F (verici dile göre ayrıştırılmış)"
	@echo "eval-calibration - ECE + Brier + risk-coverage"
	@echo "lexicons       - kaikki.org sözlük dökümlerini indir (19 dil, ~56 MB)"
	@echo "donors         - VERİCİ dil dökümleri + indeksi (~352 MB) — alıntı sinyali için"
	@echo "lexicon-index  - Yerel arama indeksini kur (SQLite FTS5)"
	@echo "chains         - Alıntı geçiş zincirleri ve uyarlama kuralları"
	@echo "correspondences - Ses denkliklerini TRAIN kavramlarından öğren"
	@echo "eval-prediction - İleri akraba tahmini (tr -> 31 dil)"
	@echo "semantic       - Türkçe kavram köprüsünü kur (CLICS eş-adlandırma için)"
	@echo "dialect        - Ağız kelimeleri toplu analizi (Faz 10)"
	@echo "predict-lock   - Öngörü üret ve kilitle (NAME=... gerekli)"
	@echo "predict-verify - Kilitli öngörüleri doğrula (NAME=... gerekli)"
	@echo "eval-controls  - Negatif kontrol bataryası (sahte kök, alıntı tuzağı)"
	@echo "audit          - Arama çıktısı tutarlılık denetimi (60 kelime)"
	@echo "eval-cv        - Rekonstrüksiyon 5 katlı çapraz doğrulama (train+dev, n≈320)"
	@echo "eval-llm       - Yalnız-LLM alıntı taban çizgisi (PROVIDER=ollama|claude)"
	@echo "eval-headline  - Başlık kökü × Starling ve savelyev dev (yalnız yerel kaynaklar)"
	@echo "eval-donor     - Verici dil tanıma × WOLD Sakha (Rusça/Moğolca/Tunguzca)"
	@echo "eval-chronology - A-HVP 2. aşama yılı × Starling tarihli kaynak etiketleri"
	@echo "apertium       - Apertium iki dilli Türk dili sözlüklerini indir"
	@echo "khakas         - Hakasça–Rusça ve açıklamalı sözlüğü indir (HF, CC-BY-4.0; portföyde değil)"
	@echo "wilkens        - Wilkens 2021 Eski Uygurca sözlüğünü indir ve ayrıştır (CC BY-SA 4.0; .[pdf] gerekir)"
	@echo "starling       - Starling Türk/Moğol etimoloji tablolarını indir (Dybo & Starostin 2005)"
	@echo "calibrate      - Güven kalibratörünü TRAIN bölümünde eğit"
	@echo "bootstrap      - Taze klonda tüm veriyi indir ve kur (data+lexicons+tr+index+donors+starling+apertium+gold+patterns)"
	@echo ""
	@echo "serve       - REST API sunucusu"
	@echo "web         - Web panelini yayınla (localhost:3000)"
	@echo "clean       - Önbellek ve geçici dosyaları sil"

install:
	python3 -m venv .venv || uv venv .venv
	.venv/bin/python -m pip install -e ".[dev,phon,pdf,cldf]" || \
		uv pip install --python .venv/bin/python -e ".[dev,phon,pdf,cldf]"

test:
	.venv/bin/pytest -q -n 4

# Commit öncesi: yavaş uç nokta testleri dahil hepsi.
test-all:
	.venv/bin/pytest -q -n 4 -m "slow or not slow"

test-live:
	ETY_LIVE=1 .venv/bin/pytest engine/tests/live -v

lint:
	.venv/bin/ruff check engine/ scripts/

fix:
	.venv/bin/ruff check engine/ scripts/ --fix

coverage:
	.venv/bin/pytest --cov=engine --cov-report=term-missing --cov-fail-under=90

data:
	.venv/bin/python scripts/download_cldf.py --all

gold: data
	.venv/bin/python -m engine.evaluation.gold --freeze

# ⚠️ `--split dev` ŞART. Motor denetimli bir katman taşıyor
# (`proto_patterns`, TRAIN kavramlarından öğrenilmiş); bölüm verilmezse
# eğitim maddeleri ölçüme girer ve sayı motorun performansı değil ezberi
# olur. `engine.evaluation.report` bu durumda uyarı basar.
eval-baseline: gold patterns
	.venv/bin/python -m engine.evaluation.report --split dev

eval: gold
	.venv/bin/python -m engine.evaluation.harness --split dev

# Arama çıktısı tutarlılık denetimi (60 altın kelime; canlı sözlükler kalıcı önbellekten).
audit:
	.venv/bin/python -m engine.evaluation.consistency_audit

# Rekonstrüksiyon: train+dev üzerinde 5 katlı çapraz doğrulama (n≈320).
eval-cv: gold
	.venv/bin/python -m engine.evaluation.crossval

expert-review:
	.venv/bin/python -m engine.evaluation.expert_review

sigtyp: gold
	.venv/bin/python scripts/export_sigtyp.py

regularity: gold
	.venv/bin/python -m engine.evaluation.regularity

gold-agreement: data
	.venv/bin/python -m engine.evaluation.gold_agreement

eval-cognates: data
	.venv/bin/python -m engine.evaluation.cognate_eval

eval-borrowing: data lexicon-index donors
	.venv/bin/python -m engine.evaluation.borrowing_eval

lexicons:
	.venv/bin/python scripts/download_lexicons.py --all
	# ⚠️ Rusça sürüm (~2,8 MB): İngilizce sürümde olmayan diller
	# (Karayca, Kırım Tatarcası, Şorca) ve Çuvaşça'nın 3 katı.
	.venv/bin/python scripts/download_lexicons.py --ru

turkish-gold: lexicon-index
	.venv/bin/python scripts/build_turkish_loanword_gold.py --limit 2000

donors:
	.venv/bin/python scripts/download_lexicons.py --donors \
		Russian Mongolian Evenki Arabic Persian Greek Armenian French Italian
	.venv/bin/python -m engine.db.donor_index --build

starling:
	.venv/bin/python scripts/download_starling.py

apertium:
	.venv/bin/python scripts/download_apertium.py

khakas:
	.venv/bin/python scripts/download_khakas.py

wilkens:
	.venv/bin/python scripts/download_wilkens.py

lexicon-index: lexicons
	.venv/bin/python -m engine.db.lexicon_index --build

chains: lexicons
	.venv/bin/python -m engine.nlp.borrowing_chain --save

patterns: gold
	.venv/bin/python -m engine.nlp.proto_patterns

# Sütun modeli (ata ses / ∅): savelyevturkic TRAIN + Starling turcet
# (`make starling`); dev/test Türkçe biçimleri ve dev kökleri eğitimden çıkar.
column-model: gold patterns
	.venv/bin/python -m engine.nlp.column_model --train

correspondences: gold
	.venv/bin/python -m engine.nlp.cognate_prediction

eval-prediction: correspondences
	.venv/bin/python -m engine.evaluation.prediction_eval

semantic:
	.venv/bin/python -m engine.nlp.semantic_plausibility --build-bridge

dialect: correspondences lexicon-index
	.venv/bin/python scripts/analyse_dialect_words.py --limit 200

predict-lock: correspondences lexicon-index
	.venv/bin/python -m engine.evaluation.prediction_test generate --name $(NAME)

predict-verify:
	.venv/bin/python -m engine.evaluation.prediction_test verify --name $(NAME)

eval-controls:
	.venv/bin/python -m engine.evaluation.negative_controls --verbose

PROVIDER ?= ollama
eval-llm:
	.venv/bin/python -m engine.evaluation.llm_borrowing_baseline --provider $(PROVIDER)

# Başlık ve kronoloji arama hattını kelime başına koşar (~2 sn/kelime);
# sonuçlar data/cache/eval_engine_runs altında HEAD'e bağlı önbelleğe yazılır.
eval-headline: gold
	.venv/bin/python -m engine.evaluation.headline_eval

eval-donor: data lexicon-index donors
	.venv/bin/python -m engine.evaluation.donor_id_eval

eval-chronology:
	.venv/bin/python -m engine.evaluation.chronology_eval

eval-calibration: gold
	.venv/bin/python -m engine.evaluation.calibration --split train+dev

calibrate: gold
	.venv/bin/python -m engine.nlp.confidence

serve:
	.venv/bin/python -m engine.server

web:
	cd web && npx serve -l 3000 .

# Taze klondan tüm yerel veriyi kurar (~800 MB). İndiriciler, veri dosyası
# diskte VE SHA-256'sı künyeyle aynıysa atlar; tekrar koşmak ucuzdur.
# Türkçe sürüm (--tr) indeksten ÖNCE iner ki indekse girsin. `calibrate`
# dahil değil: commit edilmiş data/calibration/model.json'u yeniden yazar.
bootstrap:
	$(MAKE) data lexicons
	.venv/bin/python scripts/download_lexicons.py --tr
	$(MAKE) lexicon-index donors starling apertium gold patterns

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
