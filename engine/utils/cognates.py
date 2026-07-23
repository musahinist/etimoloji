"""
Türki Diller Derin Akraba Kelime ve Türev Ağı (Cognate Network Builder)
Araması yapılan kelimelerin aynı kökten türeyen akraba sözcük kümesini oluşturur.
"""
from typing import List, Dict

COGNATE_NETWORK = {
    "göz": ["görmek", "gözlem", "gözlük", "gözetmek", "gözenek", "gözcü", "gözsüz"],
    "bilgi": ["bilmek", "bilim", "bilge", "bilgin", "bilinç", "bilgisayar", "bilişsel"],
    "su": ["sulamak", "sulu", "susuz", "suluk", "suvarma", "sucu"],
    "deniz": ["denizci", "denizaltı", "denizcilik", "denizli"],
    "el": ["elli", "eldiven", "elemek", "elek", "elverişli"],
    "belge": ["belirlemek", "belirgin", "belirtmek", "belirteç", "belgeli"],
    "tetik": ["tetiklemek", "tetikte", "tetikçi", "tetiklik"],
    "kut": ["kutsal", "kutlamak", "kutlu", "kutsallık", "kutsuz"]
}

def get_related_cognates(word: str) -> List[str]:
    w = word.strip().lower()
    return COGNATE_NETWORK.get(w, [])
