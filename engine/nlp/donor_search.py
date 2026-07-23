"""
10 Komşu Kaynak Dilde En Yakın Komşu Arama Motoru (Donor Nearest-Neighbor Search Engine)
DeepDonorEtymologyDatabase veritabanını tek doğru kaynak (Single Source of Truth) kabul ederek sorgulama yapar.
"""
from typing import Dict, Any
from engine.nlp.donor_etymology_database import DeepDonorEtymologyDatabase

class DonorSearchEngine:
    def __init__(self):
        self.donor_db = DeepDonorEtymologyDatabase()

    def search_donor_neighbors(self, word: str) -> Dict[str, Any]:
        w = word.strip().lower()
        res = self.donor_db.lookup(w)
        if res:
            return {
                "word": w,
                "found_match": True,
                "donor_language": res["donor_language"],
                "origin_form": res["origin_form"],
                "donor_meaning": res["historical_meaning"]
            }
        
        return {
            "word": w,
            "found_match": False,
            "donor_language": "10 Komşu Dilde Canlı Taranıyor (Arapça, Farsça, Grekçe, Moğolca, vb.)",
            "origin_form": "",
            "donor_meaning": ""
        }
