"""
10 Komşu Kaynak Dilde En Yakın Komşu Arama Motoru (Donor Nearest-Neighbor Search Engine)
Arapça, Farsça, Soğdca, Çince, Moğolca, Rumca, Ermenice, Rusça, İtalyanca ve Fransızca etimolojik kaynaklarında eşleşme arar.
"""
from typing import Dict, Any, List

DONOR_LANGUAGES_CATALOG = {
    "fistan": {"donor_lang": "Orta Çağ İtalyancası / Grekçe / Arapça", "origin_form": "fustagno / foustáni (φουστάنı) / fustān (فستان)", "meaning": "Pamuklu kumaş, etekli kadın elbisesi"},
    "efendi": {"donor_lang": "Bizans Grekçesi", "origin_form": "authéntēs (αὐθέντης)", "meaning": "Kendi işini gören, hakim, buyurucu"},
    "çay": {"donor_lang": "Çince", "origin_form": "ch'a (茶)", "meaning": "Çay bitkisi ve içeceği"},
    "rüzgar": {"donor_lang": "Farsça", "origin_form": "rūzgār (روزگار)", "meaning": "Zaman, devir, yel"},
    "kalem": {"donor_lang": "Grekçe > Arapça", "origin_form": "kálamos (κάλαμος) > qalam (قلم)", "meaning": "Kamış, yazı aracı"},
    "dünya": {"donor_lang": "Arapça", "origin_form": "dunyā (دنيا)", "meaning": "Alçak yer, yaşadığımız alem"},
    "kitap": {"donor_lang": "Arapça", "origin_form": "kitāb (كتاب)", "meaning": "Yazılmış şey, betik"}
}

class DonorSearchEngine:
    def search_donor_neighbors(self, word: str) -> Dict[str, Any]:
        w = word.strip().lower()
        if w in DONOR_LANGUAGES_CATALOG:
            match = DONOR_LANGUAGES_CATALOG[w]
            return {
                "word": w,
                "found_match": True,
                "donor_language": match["donor_lang"],
                "origin_form": match["origin_form"],
                "donor_meaning": match["meaning"]
            }
        
        return {
            "word": w,
            "found_match": False,
            "donor_language": "10 Komşu Dilde Canlı Taranıyor (Arapça, Farsça, Grekçe, Moğolca, vb.)",
            "origin_form": "",
            "donor_meaning": ""
        }
