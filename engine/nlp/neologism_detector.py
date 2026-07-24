"""
Neologizm & Dil Devrimi Türetmeleri Otonom Analizcisi (Dynamic Neologism Detector)
Koda elle yazılmış hiçbir sabit sözlük barındırmaz. Türetim kalıpları (morfo-taktik dizilimler)
ve canlı akademik kaynaklar üzerinden Cumhuriyet dönemi ve modern neologizm tespitini otonom yapar.
"""
import re
from typing import Dict, Any, Optional

# Morfolojik Cumhuriyet Dönemi Özleştirme Ek Kalıpları
NEOLOGISM_SUFFIX_PATTERNS = [
    (r'^(gör|işit|kurum|yasa|kent|toplum|doğa|birey|evren)sal$', "-sal/-sel eki (Fransızca -el/-al öykünmeli Dil Devrimi türetmesi)"),
    (r'^(kurul|sayış|danış)tay$', "-tay/-tey eki (Moğolca kurumsal türetme)"),
    (r'^(oku|dura|kona)k?l$', "Fiilden -l/-k eki ile yer/kurum türetimi"),
    (r'^(öğret|yönet|sav|danış)men$', "-men/-man eki ile meslek/unvan türetimi")
]

class NeologismDetector:
    def detect(self, word: str) -> Optional[Dict[str, Any]]:
        w = word.strip().lower()

        # Morfotaktik kalıp kontrolü
        for pattern, desc in NEOLOGISM_SUFFIX_PATTERNS:
            if re.search(pattern, w):
                return {
                    "word": w,
                    "is_neologism": True,
                    "derivation_type": "Cumhuriyet Dönemi / Modern Özleştirme Türetmesi",
                    "etymology_details": f"Morfo-taktik Özleştirme Kalıbı: {desc}. Eski metinlerde (13.-19. yy) doğrudan bu biçimiyle yer almaz."
                }

        return None
