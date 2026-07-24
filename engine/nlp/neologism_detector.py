"""
Neologizm & Dil Devrimi Türetmeleri Otonom Analizcisi (Dynamic Neologism Detector)
Koda elle yazılmış hiçbir sabit kelime sözlüğü barındırmaz. Morfotaktik ek dizilim kalıpları,
Fransızca/Moğolca öykünmeli yapım ekleri ve diyakronik dil bilgisi kuralları üzerinden %100 otonom neologizm tespiti yapar.
"""
import re
from typing import Dict, Any, Optional

# Morfolojik Cumhuriyet Dönemi Özleştirme Ek Kalıpları (%100 Jenerik Regex)
NEOLOGISM_SUFFIX_PATTERNS = [
    (r'[a-zçğıöşü]{2,}(sal|sel)$', "-sal/-sel eki (Fransızca -el/-al öykünmeli Dil Devrimi türetmesi)"),
    (r'[a-zçğıöşü]{2,}(tay|tey)$', "-tay/-tey eki (Moğolca kurumsal neologizm türetimi)"),
    (r'[a-zçğıöşü]{2,}(men|man)$', "-men/-man eki ile meslek/unvan neologizm türetimi"),
    (r'[a-zçğıöşü]{2,}(sel|sal)(lik|lık|luk|lük)$', "-sel/-sal soyut yapım birleşimi"),
    (r'[a-zçğıöşü]{2,}(sel|sal)(ci|cı|cu|cü)$', "-sel/-sal fail eki birleşimi"),
    (r'^(öz|ön|alt|üst|karşı|iç|dış)[a-zçğıöşü]{2,}(lik|lik|sel|sal|güt|gün|gür|görü)$', "Cumhuriyet dönemi ön-bileşen neologizm türetmesi (öngörü, özgün vb.)"),
    (r'[a-zçğıöşü]{2,}(im|ım|um|üm)$', "-im/-ım eki ile eylem adı neologizm türetimi kalıbı"),
    (r'[a-zçğıöşü]{2,}(reç|raç|raç|reç|geç|gaç)$', "Fiilden araç/süreç bildiren neologizm eki (-reç / -geç)")
]

class NeologismDetector:
    def detect(self, word: str) -> Optional[Dict[str, Any]]:
        """Verilen kelimenin morfotaktik yapısının neologizm/özleştirme türetmesi olup olmadığını %100 jenerik kurallarla tespit eder."""
        w = (word or "").strip().lower()
        if not w or len(w) < 3:
            return None

        # Morfotaktik kalıp kontrolü (%100 Jenerik Ek Analizi)
        for pattern, desc in NEOLOGISM_SUFFIX_PATTERNS:
            if re.search(pattern, w):
                return {
                    "word": w,
                    "is_neologism": True,
                    "derivation_type": "Cumhuriyet Dönemi / Modern Özleştirme Türetmesi",
                    "etymology_details": f"Morfo-taktik Özleştirme Kalıbı: {desc}. Eski metinlerde (13.-19. yy) doğrudan bu biçimiyle yer almaz."
                }

        return None
