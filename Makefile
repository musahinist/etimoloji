.PHONY: help install test test-live lint fix coverage clean serve web \
        data gold donors patterns gold-agreement eval eval-baseline eval-cognates eval-borrowing eval-calibration eval-controls eval-prediction correspondences calibrate lexicons lexicon-index chains predict-lock predict-verify semantic dialect

help:
	@echo "install     - .venv oluştur ve bağımlılıkları kur"
	@echo "test        - Ağsız test paketi"
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
	@echo "eval           - Rekonstrüksiyon ölçümü (dev bölümü)"
	@echo "eval-cognates  - Akraba tespiti B-Cubed F (LexStat-Infomap taban çizgisine karşı)"
	@echo "gold-agreement - Altın standartlar arası uyum: otomatik sistemin GERÇEKÇİ tavanı"
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
	@echo "calibrate      - Güven kalibratörünü TRAIN bölümünde eğit"
	@echo ""
	@echo "serve       - REST API sunucusu"
	@echo "web         - Web panelini yayınla (localhost:3000)"
	@echo "clean       - Önbellek ve geçici dosyaları sil"

install:
	python3 -m venv .venv || uv venv .venv
	.venv/bin/python -m pip install -e ".[dev,phon,pdf,cldf]" || \
		uv pip install --python .venv/bin/python -e ".[dev,phon,pdf,cldf]"

test:
	.venv/bin/pytest -q

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

gold-agreement: data
	.venv/bin/python -m engine.evaluation.gold_agreement

eval-cognates: data
	.venv/bin/python -m engine.evaluation.cognate_eval

eval-borrowing: data lexicon-index donors
	.venv/bin/python -m engine.evaluation.borrowing_eval

lexicons:
	.venv/bin/python scripts/download_lexicons.py --all

donors:
	.venv/bin/python scripts/download_lexicons.py --donors \
		Russian Mongolian Evenki Arabic Persian Greek Armenian French Italian
	.venv/bin/python -m engine.db.donor_index --build

lexicon-index: lexicons
	.venv/bin/python -m engine.db.lexicon_index --build

chains: lexicons
	.venv/bin/python -m engine.nlp.borrowing_chain --save

patterns: gold
	.venv/bin/python -m engine.nlp.proto_patterns

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

eval-calibration: gold
	.venv/bin/python -m engine.evaluation.calibration --split all

calibrate: gold
	.venv/bin/python -m engine.nlp.confidence

serve:
	.venv/bin/python -m engine.server

web:
	cd web && npx serve -l 3000 .

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
