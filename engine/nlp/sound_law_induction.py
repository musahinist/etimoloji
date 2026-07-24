"""
Otomatik Ses Kanunu İndüksiyon Motoru (Automated Sound Law Induction - SLI)
Akraba veya ata kelime çiftlerinden sıralı ses dönüşüm kurallarını (Ordered String Rewrite Rules)
otonom olarak öğrenebilen ve indükleyen hesaplamalı dilbilim motoru.
"""

from typing import Dict, Any, List, Tuple
import re

class SoundLawInductionEngine:
    """Akraba ve ata sözcük çiftlerinden ses yasası kurallarını indükleyen motor"""

    def induce_sound_law(self, source_word: str, target_word: str) -> Dict[str, Any]:
        s = re.sub(r'[^a-zçğıöşüа-я]', '', (source_word or "").lower().lstrip("*"))
        t = re.sub(r'[^a-zçğıöşüа-я]', '', (target_word or "").lower())

        if not s or not t or s == t:
            return {
                "source_word": s,
                "target_word": t,
                "has_induced_rule": False,
                "induced_rules": [],
                "rule_confidence": 1.0
            }

        induced_rules = []

        # 1. Söz Başı Dönüşümleri (Initial Consonant Shifts)
        if s[0] != t[0]:
            rule_id = f"INITIAL_{s[0].upper()}_{t[0].upper()}"
            induced_rules.append({
                "rule_id": rule_id,
                "pattern": f"#{s[0]} -> #{t[0]}",
                "position": "initial",
                "description": f"Söz başı {s[0]}- ünsüzünün {t[0]}- ünsüzüne dönüşmesi"
            })

        # 2. Söz Sonu Dönüşümleri (Final Shifts)
        if s[-1] != t[-1]:
            rule_id = f"FINAL_{s[-1].upper()}_{t[-1].upper()}"
            induced_rules.append({
                "rule_id": rule_id,
                "pattern": f"{s[-1]}# -> {t[-1]}#",
                "position": "final",
                "description": f"Söz sonu -{s[-1]} ünsüzünün -{t[-1]} sesine dönüşmesi"
            })

        # 3. İç Ses ve Ünlü Uyumu Kaymaları
        s_vowels = [c for c in s if c in 'aeıioöuü']
        t_vowels = [c for c in t if c in 'aeıioöuü']
        if s_vowels and t_vowels and s_vowels != t_vowels:
            induced_rules.append({
                "rule_id": "VOWEL_SHIFT",
                "pattern": f"{''.join(s_vowels)} -> {''.join(t_vowels)}",
                "position": "medial",
                "description": f"Kök ünlü kümesinin {''.join(s_vowels)} -> {''.join(t_vowels)} şeklinde kayması"
            })

        confidence = 0.95 if induced_rules else 0.70

        return {
            "source_word": s,
            "target_word": t,
            "has_induced_rule": len(induced_rules) > 0,
            "induced_rules": induced_rules,
            "rule_confidence": confidence
        }
