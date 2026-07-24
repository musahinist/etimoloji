import os
import json
import re
import concurrent.futures
from typing import Dict, Any, List, Optional

from engine.fetchers.base import BaseFetcher, TURKIC_LANGUAGES_MAP
from engine.fetchers.academic_turkology import AcademicTurkologyFetcher
from engine.fetchers.historical_modern import HistoricalModernLexiconFetcher
from engine.fetchers.isam_ansiklopedi import IsamAnsiklopediFetcher
from engine.fetchers.archive_org import ArchiveOrgFetcher
from engine.fetchers.dergipark import DergiParkFetcher
from engine.fetchers.osmanlica_lugat import OsmanlicaLugatFetcher
from engine.fetchers.tdk_all_portals import TdkAllPortalsFetcher
from engine.fetchers.glosbe import GlosbeFetcher
from engine.fetchers.turkic_national_dictionaries import TurkicNationalDictionariesFetcher
from engine.fetchers.loanword_donor_etymology import LoanwordDonorEtymologyFetcher
from engine.fetchers.local_pdf_books import LocalPdfBooksFetcher
from engine.fetchers.tietze_altaica import TietzeAltaicaFetcher
from engine.fetchers.etimoloji_turkce import EtimolojiTurkceFetcher
from engine.fetchers.starling import StarlingFetcher
from engine.fetchers.tdk_nisanyan import NisanyanFetcher, TdkFetcher
from engine.fetchers.tdk_historical import TdkTaramaFetcher, TdkDerlemeFetcher
from engine.fetchers.wiktionary import WiktionaryFetcher
from engine.fetchers.multilang_wiktionary import MultiLangWiktionaryFetcher
from engine.fetchers.wiktextract_local import WiktextractFetcher


from engine.nlp.loanword_classifier import LoanwordClassifier
from engine.nlp.cognate_alignment import CognateAlignmentEngine
from engine.nlp.reconstruction import ProtoTurkicReconstructor
from engine.nlp.donor_search import DonorSearchEngine
from engine.nlp.iterative_hypothesis_engine import IterativeHypothesisEngine
from engine.nlp.iterative_hypothesis_prover import IterativeHypothesisProver
from engine.nlp.cldf_lingpy_aligner import CldfLingPyAligner
from engine.nlp.diachronic_semantic_engine import DiachronicSemanticEngine
from engine.nlp.sound_law_induction import SoundLawInductionEngine

from engine.utils.morphology import analyze_morphology
from engine.utils.transliteration import transliterate_to_latin
from engine.utils.phonetic_rules import analyze_phonetic_shifts
from engine.utils.geo_tagger import tag_geographical_region
from engine.utils.cognates import get_related_cognates
from engine.utils.variant_expander import generate_dynamic_phonetic_variants
from engine.utils.reference_resolver import extract_cross_references
from engine.db.database import DatabaseManager
from engine.db.graph_database import GraphDatabaseManager
from engine.llm.qwen_agent import QwenEtymologyAgent

MEANING_TRANSLATIONS = {
    "water": "su, sıvı",
    "sea": "deniz, büyük göl",
    "lake": "göl",
    "eye": "göz, görme organı",
    "foot": "ayak, bacak",
    "hand / forearm": "el, kol",
    "head": "baş, kafa, lider",
    "blood": "kan",
    "moon": "ay, uydu",
    "sun": "güneş, gün",
    "sky": "gök, gökyüzü",
    "god": "tanrı, ilah",
    "deity": "tanrı, ilahi varlık",
    "sign": "işaret, alamet",
    "mark": "damga, işaret",
    "alert": "uyanık, tetikte, keskin",
    "trigger": "tetik mekanizması",
    "realm": "ülke, il, memleket",
    "country": "ülke, memleket",
    "blessing": "kut, bereket, tanrı lütfu",
    "soul": "can, ruh, kut",
    "word": "kelime, söz"
}

def translate_meaning(meaning: str) -> str:
    m = meaning.strip()
    if not m or m.startswith("Online"):
        return m
    
    m_clean = re.sub(r'\{\{.*?\}\}', '', m).strip()
    m_clean = re.sub(r'\[\[(.*?)\]\]', r'\1', m_clean).strip()
    
    for eng, tr in MEANING_TRANSLATIONS.items():
        if eng.lower() in m_clean.lower():
            return tr
            
    return m_clean or meaning

class SearchEngine:
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        self.db = db_manager or DatabaseManager()
        self.graph_db = GraphDatabaseManager()
        self.qwen_agent = QwenEtymologyAgent()

        # NLP & İleri Hesaplamalı Modüller
        self.loanword_classifier = LoanwordClassifier()
        self.cognate_alignment_engine = CognateAlignmentEngine()
        self.reconstructor = ProtoTurkicReconstructor()
        self.donor_search_engine = DonorSearchEngine()
        self.hypothesis_engine = IterativeHypothesisEngine()
        self.hypothesis_prover = IterativeHypothesisProver()
        self.lingpy_aligner = CldfLingPyAligner()
        self.semantic_engine = DiachronicSemanticEngine()
        self.sound_law_induction = SoundLawInductionEngine()

        self.fetchers: List[BaseFetcher] = [
            AcademicTurkologyFetcher(),
            HistoricalModernLexiconFetcher(),
            IsamAnsiklopediFetcher(),
            ArchiveOrgFetcher(),
            DergiParkFetcher(),
            OsmanlicaLugatFetcher(),
            TdkAllPortalsFetcher(),
            GlosbeFetcher(),
            TurkicNationalDictionariesFetcher(),
            LoanwordDonorEtymologyFetcher(),
            LocalPdfBooksFetcher(),
            TietzeAltaicaFetcher(),
            EtimolojiTurkceFetcher(),
            StarlingFetcher(),
            NisanyanFetcher(),
            TdkFetcher(),
            TdkTaramaFetcher(),
            TdkDerlemeFetcher(),
            WiktionaryFetcher(),
            MultiLangWiktionaryFetcher(),
            WiktextractFetcher()
        ]


    def search(self, query: str, save_to_db: bool = True, use_qwen_agent: bool = False) -> Dict[str, Any]:
        word_clean = query.strip().lower()

        if save_to_db and not use_qwen_agent:
            cached = self.db.get_finding(word_clean)
            if cached:
                cached["from_cache"] = True
                return cached

        stem, suffixes = analyze_morphology(word_clean)

        search_variants = list(set([word_clean, stem] + generate_dynamic_phonetic_variants(word_clean)))

        proto_root = ""
        root_meaning = ""
        sources = []
        turkic_entries_map = {}

        def fetch_worker(fetcher: BaseFetcher):
            results = []
            for var in search_variants:
                res = fetcher.fetch(var)
                if res and (res.get("turkic_languages") or res.get("proto_turkic")):
                    results.append(res)
            return fetcher, results

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            future_to_fetcher = {executor.submit(fetch_worker, f): f for f in self.fetchers}
            for future in concurrent.futures.as_completed(future_to_fetcher):
                fetcher = future_to_fetcher[future]
                try:
                    fetcher_obj, results = future.result()
                    for res in results:
                        root_info = res.get("root", {})
                        if root_info.get("proto_turkic") and not proto_root:
                            proto_root = root_info.get("proto_turkic")
                        if root_info.get("meaning") and not root_meaning:
                            root_meaning = translate_meaning(root_info.get("meaning"))

                        for entry in res.get("turkic_languages", []):
                            entry["meaning"] = translate_meaning(entry.get("meaning", ""))
                            entry["phonetic_shift"] = analyze_phonetic_shifts(word_clean, entry.get("word", ""), entry.get("lang_name", ""))
                            key = (entry["lang_code"], entry["word"])
                            if key not in turkic_entries_map:
                                turkic_entries_map[key] = entry
                            elif turkic_entries_map[key].get("meaning") in ["", f"Online {TURKIC_LANGUAGES_MAP.get(entry['lang_code'], '')} Sözlük kaydı"]:
                                if entry.get("meaning") and not entry.get("meaning").startswith("Online"):
                                    turkic_entries_map[key] = entry

                        if res.get("turkic_languages") or root_info.get("proto_turkic"):
                            sources.append(fetcher.source_name)
                except Exception:
                    pass

        sorted_entries = sorted(
            list(turkic_entries_map.values()),
            key=lambda x: (0 if x["lang_code"] == "otk" else (0.3 if x["lang_code"] == "ai" else (0.5 if x["lang_code"] == "donor" else 1)), x["lang_name"])
        )

        # 4. KÖKEN NLP VE OTONOM İNATÇI HİPOTEZ REKONSTRÜKSİYONU
        loan_eval = self.loanword_classifier.classify(word_clean)
        cognate_eval = self.cognate_alignment_engine.evaluate_cognate_distribution(word_clean, sorted_entries)
        reconstruction_eval = self.reconstructor.reconstruct_proto_form(word_clean, sorted_entries)
        donor_eval = self.donor_search_engine.search_donor_neighbors(word_clean)

        finding_temp = {
            "root": {"proto_turkic": proto_root, "meaning": root_meaning}
        }
        proven_hypothesis_eval = self.hypothesis_engine.prove_etymological_hypothesis(word_clean, finding_temp)

        # Çözülmemiş / Etimolojisi Sözlükte Kayıtlı Olmayan Kelimeler İçin Otonom İnatçı Prover
        unattested_prover_eval = self.hypothesis_prover.prove_unattested_word(word_clean, sorted_entries)

        # Eğer bilinen sözlüklerden kök bulunamadıysa Otonom AI Hypothesis PR'ını devreye sok
        if not proto_root or proto_root == word_clean:
            hypo_pr = unattested_prover_eval["proven_hypothesis"]
            proto_root = hypo_pr.get("origin_form", f"*{word_clean}")
            proven_hypothesis_eval["proven_hypothesis"] = hypo_pr

        lingpy_eval = self.lingpy_aligner.align_sequences(proto_root or word_clean, word_clean)
        semantic_eval = self.semantic_engine.evaluate_diachronic_trajectory(root_meaning or word_clean, word_clean)
        sound_law_induced = self.sound_law_induction.induce_sound_law(proto_root or word_clean, word_clean)

        if proven_hypothesis_eval.get("proven_hypothesis", {}).get("confidence_score", 0) >= 0.95:
            hypo = proven_hypothesis_eval["proven_hypothesis"]
            proto_root = hypo.get("origin_form", proto_root)
            root_meaning = hypo.get("historical_meaning", root_meaning)
            sources.append(f"Derin Komşu Diller Etimoloji Veritabanı ({hypo.get('donor_language')})")
            sorted_entries.insert(0, {
                "lang_code": "donor",
                "lang_name": f"Kaynak Dil Etimolojisi ({hypo.get('donor_language')})",
                "word": hypo.get("origin_form"),
                "meaning": hypo.get("proof_summary"),
                "script": "Original"
            })

        timeline = []
        for entry in sorted_entries:
            lname = entry.get("lang_name", "")
            if "Divanü Lugati't-Türk" in lname or "1074" in lname or "Orhun" in lname or "Eski Türkçe" in lname or "İSAM" in lname:
                timeline.append(f"M.Ö. III. YY - 11. YY (Hun / Orhun / DLT / İSAM): {entry.get('word')} - {entry.get('meaning')[:60]}")
            elif "1303" in lname or "Codex Cumanicus" in lname:
                timeline.append(f"14. YY (Kıpçakça / Codex Cumanicus): {entry.get('word')}")
            elif "1901" in lname or "Kamus-ı Türkî" in lname or "13.-19." in lname or "Osmanlıca Lügat" in lname:
                timeline.append(f"19. YY (Osmanlıca / Lehçe-i Osmanî / Kamus-ı Türkî): {entry.get('word')}")

        morphology_info = f"Kök: {stem} + Ekler: {', '.join(suffixes)}" if suffixes else "Yalın Kök"
        related_cognates = get_related_cognates(stem) or get_related_cognates(word_clean)

        # 5. Neo4j Uyumlu Graf Veritabanı Düğüm Şeması Oluşturma
        graph_export = self.graph_db.build_etymology_graph(
            word=word_clean,
            root_form=proto_root or word_clean,
            hypothesis=proven_hypothesis_eval.get("proven_hypothesis", {}),
            attestations=timeline,
            cognates=related_cognates
        )

        finding = {
            "query_word": word_clean,
            "morphology": morphology_info,
            "turkic_languages": sorted_entries,
            "root": {
                "proto_turkic": proto_root or word_clean,
                "meaning": root_meaning or word_clean,
                "reconstruction_notes": reconstruction_eval.get("reconstruction_notes", "")
            },
            "nlp_analysis": {
                "loanword_classification": loan_eval,
                "cognate_distribution": cognate_eval,
                "reconstruction": reconstruction_eval,
                "donor_matching": donor_eval,
                "proven_hypothesis": proven_hypothesis_eval.get("proven_hypothesis"),
                "unattested_word_reconstruction": unattested_prover_eval,
                "lingpy_alignment": lingpy_eval,
                "diachronic_semantic_drift": semantic_eval,
                "induced_sound_laws": sound_law_induced
            },
            "graph_database": graph_export,
            "timeline": list(dict.fromkeys(timeline)),
            "related_cognates": related_cognates,
            "sources": sorted(list(set(sources))),
            "from_cache": False
        }

        if use_qwen_agent:
            finding = self.qwen_agent.research_and_enrich(word_clean, finding)

        if save_to_db:
            self.db.save_finding(finding)

        return finding
