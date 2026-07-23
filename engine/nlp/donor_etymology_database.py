"""
Derin Komşu Diller Etimoloji Veritabanı (Deep Donor Etymology Lexicon Database)
Anadolu ağızlarında ve Türkçede yer alan kelimelerin Ermenice, Gürcüce, Bizans Grekçesi,
Farsça, Arapça, Soğdca, Moğolca, İtalyanca/Latince etimolojik kökenlerini ve tanıklamalarını içerir.
"""
from typing import Dict, Any, Optional

DEEP_DONOR_LEXICON = {
    "herkil": {
        "donor_lang": "Ermenice (Armenian)",
        "origin_form": "harkil / herkil (յարդկիլ / հերկիլ)",
        "reconstructed_etymology": "Ermenice har- (tahıl, erzak, harman) + kil (saklama kabı/ambar). Anadolu ağızlarına Doğu/İç Anadolu temasıyla geçmiştir.",
        "dialect_variant": "helkir (r~l metatez biçimi)",
        "historical_meaning": "Tahıl saklanan büyük tahta ambar, zahire sandığı"
    },
    "helkir": {
        "donor_lang": "Ermenice (Armenian)",
        "origin_form": "harkil / herkil (յարդկիլ / հերկիլ)",
        "reconstructed_etymology": "Ermenice har- (tahıl, erzak, harman) + kil (saklama kabı/ambar). Anadolu ağızlarında l~r metatez biçimi (herkil > helkir).",
        "dialect_variant": "herkil",
        "historical_meaning": "Tahıl saklanan büyük tahta ambar, zahire sandığı"
    },
    "fistan": {
        "donor_lang": "Orta Çağ İtalyancası / Grekçe / Arapça",
        "origin_form": "fustagno (İtalyanca) > foustáni (φουστάνι, Grekçe) > fustān (فستان, Arapça)",
        "reconstructed_etymology": "Mısır'ın Fustat (Kahire) kentinde dokunan pamuklu kumaş adından Akdeniz ticaret yoluyla Osmanlı Türkçesine (14. yy) geçmiştir.",
        "dialect_variant": "fistan / piston",
        "historical_meaning": "Pamuklu kumaş, etekli kadın elbisesi"
    },
    "efendi": {
        "donor_lang": "Bizans Grekçesi",
        "origin_form": "authéntēs (αὐθέντης < αὐτο- + ἔντης)",
        "reconstructed_etymology": "Eski Grekçe auto- (kendi) + entēs (yapan). Özgün anlamı: kendi işini kendi gören, hakim, buyurucu kimsedir.",
        "dialect_variant": "efendi / əfəndi",
        "historical_meaning": "Saygı ünvanı, buyurucu, bey"
    },
    "rüzgar": {
        "donor_lang": "Farsça",
        "origin_form": "rūzgār (روزگار < rūz + -gār)",
        "reconstructed_etymology": "Farsça rūz (gün) + -gār soneki. Özgün anlamı: zaman, devir. Türkçede yel/hava akımı anlamı kazanmıştır.",
        "dialect_variant": "rüzgar / rozgar",
        "historical_meaning": "Yel, esinti, zaman"
    },
    "kalem": {
        "donor_lang": "Grekçe > Arapça",
        "origin_form": "kálamos (κάλαμος) > qalam (قلم)",
        "reconstructed_etymology": "Grekçe kamış (kálamos) kelimesinden Arapçaya qalam olarak geçmiş, oradan Türkçeye aktarılmıştır.",
        "dialect_variant": "kalem / galam",
        "historical_meaning": "Yazı aracı, kamış"
    }
}

class DeepDonorEtymologyDatabase:
    def lookup(self, word: str) -> Optional[Dict[str, Any]]:
        w = word.strip().lower()
        if w in DEEP_DONOR_LEXICON:
            entry = DEEP_DONOR_LEXICON[w]
            return {
                "word": w,
                "found": True,
                "donor_language": entry["donor_lang"],
                "origin_form": entry["origin_form"],
                "etymology": entry["reconstructed_etymology"],
                "historical_meaning": entry["historical_meaning"]
            }
        return None
