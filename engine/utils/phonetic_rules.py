"""
Fonetik Ses Kayması ve Dilbilimsel Dönüşüm Analizcisi (Phonetic Shift & Sound Law Detector)
Türkiye Türkçesindeki kelime ile Türki dillerdeki akraba kelimeler ve kök diller arasındaki ses dönüşüm kurallarını denetler.
"""
from typing import List, Dict, Tuple, Any
import re

# Tanımlı Geçerli Ses Kanunları ve Dönüşüm Kuralları
RECOGNIZED_SOUND_LAWS = [
    {
        "id": "OGUZ_KIPCAK_INITIAL_G_K",
        "name": "Oğuz - Kıpçak/Sibirya Söz Başı Ötümlüleşme/Ötümsüzleşme (g- ~ k-)",
        "source_pattern": r"^g",
        "target_pattern": r"^[kк]",
        "valid": True,
        "description": "Baştaki ötümlü 'g-' konsonantının ötümsüz 'k-' sesine dönüşmesi"
    },
    {
        "id": "INITIAL_D_T",
        "name": "Söz Başı Sertleşme (d- ~ t-)",
        "source_pattern": r"^d",
        "target_pattern": r"^[tт]",
        "valid": True,
        "description": "Baştaki ötümlü 'd-' sesinin 't-' biçimine sertleşmesi"
    },
    {
        "id": "INITIAL_B_M",
        "name": "Söz Başı Genizsilleşme (b- ~ m-)",
        "source_pattern": r"^b",
        "target_pattern": r"^[mм]",
        "valid": True,
        "description": "Baştaki 'b-' ünsüzünün genizsilleşerek 'm-' sesine dönüşmesi (ben -> men)"
    },
    {
        "id": "FINAL_Z_S_R",
        "name": "Proto-Türkçe r-z / z-s Sızıcılaşma Denkliği (-z ~ -s ~ -r)",
        "source_pattern": r"z$",
        "target_pattern": r"[sсҫśrр]$",
        "valid": True,
        "description": "Sondaki '-z' ünsüzünün sızıcı '-s / -ś' veya Lir-Şaz kolunda '-r' sesine dönüşmesi"
    },
    {
        "id": "OGUR_INITIAL_S_SH",
        "name": "Oğur/Çuvaş Söz Başı Sızıcılaşma (s- ~ ş-)",
        "source_pattern": r"^s",
        "target_pattern": r"^[шš]",
        "valid": True,
        "description": "Oğur/Çuvaş koluna özgü baştaki 's-' ünsüzünün 'ş-' (š-) sesine kayması"
    },
    {
        "id": "INTERVOCALIC_G_G_V_W",
        "name": "Ünlü Arası ve Son Ötümlüleşme/Düşme (g ~ ğ ~ v ~ w)",
        "source_pattern": r"[gğ]",
        "target_pattern": r"[vwву]",
        "valid": True,
        "description": "Orta/son 'g/ğ' sesinin 'v/w' sesine yumuşaması (sub ~ suv ~ su)"
    },
    {
        "id": "INITIAL_Y_J_C_ZH",
        "name": "Söz Başı Akıcı Y- Değişimi (y- ~ j- ~ c- ~ zh-)",
        "source_pattern": r"^y",
        "target_pattern": r"^[jjcжdж]",
        "valid": True,
        "description": "Kazakça/Kırgızca/Altayca söz başı 'y-' ~ 'j-' ~ 'c-' diyalekt kayması"
    },
    {
        "id": "FINAL_B_V_W_DROP",
        "name": "Söz Sonu Konsonant Düşmesi (b/v/w → ∅)",
        "source_pattern": r"[bvw]$",
        "target_pattern": r"[aeiouıöüuаеіоөуүы]$",
        "valid": True,
        "description": "Söz sonundaki b/v/w ünsüzünün Türki dil kollarında düşmesi (sub > suv > su, suğ; teŋiz gibi)"
    },
    {
        "id": "FINAL_B_V_W_TO_U",
        "name": "Söz Sonu v/w/ğ → u/ü Ünlüleşmesi",
        "source_pattern": r"[vwğ]$",
        "target_pattern": r"[uü]$",
        "valid": True,
        "description": "Söz sonundaki v/w/ğ'nın ünlüleşmesi (suv > suu, suğ > su)"
    },
    {
        "id": "FRENCH_LOAN_ADAPTATION",
        "name": "Fransızca/Batı Dilleri Fonotaktik Uyarlaması (c-/qu-/ch-/ph-/küp -> k-/f-/s-)",
        "source_pattern": r"^(c|qu|ch|ph|ps|st|sp|tr|pr|kl|gr|fl)",
        "target_pattern": r"^[kçfstg]",
        "valid": True,
        "description": "Batı dillerinden geçen terimlerin (Fransızca/İtalyanca/Latince) Türkçe ses sistemine jenerik uyarlanması"
    }
]

def analyze_phonetic_shifts(source_word: str, target_word: str, lang_name: str = "") -> str:
    s = source_word.strip().lower()
    t = target_word.strip().lower()

    explanations = []
    for rule in RECOGNIZED_SOUND_LAWS:
        if re.search(rule["source_pattern"], s) and re.search(rule["target_pattern"], t):
            explanations.append(rule["name"])

    if explanations:
        return "; ".join(explanations)
    return "Standart Ses Denkliği"

def verify_phonetic_chain(source_form: str, target_form: str) -> Dict[str, Any]:
    """
    Kök ile hedef kelime arasındaki ses değişim adımlarının geçerli ses kanunlarıyla örtüşüp örtüşmediğini denetler.
    """
    s = source_form.strip().lower().lstrip("*")
    t = target_form.strip().lower()

    if not s:
        s = t

    if s == t:
        return {
            "is_valid": True,
            "score": 1.0,
            "violations": [],
            "matched_rules": ["Birebir Ses Eşleşmesi"]
        }

    matched_rules = []
    for rule in RECOGNIZED_SOUND_LAWS:
        if re.search(rule["source_pattern"], s) and re.search(rule["target_pattern"], t):
            matched_rules.append(rule["name"])

    # Özel fonetik karakterleri Latin eşdeğerlerine normalize et (ŕ→r, ŋ→n, ə→e, ä→a, ş→s vb.)
    def _normalize(text: str) -> str:
        nmap = {'ŕ': 'r', 'ŗ': 'r', 'ŋ': 'n', 'ñ': 'n', 'ə': 'e', 'ä': 'a', 'ā': 'a', 'ī': 'i',
                'ū': 'u', 'ō': 'o', 'ś': 's', 'č': 'c', 'ž': 'z', 'ǰ': 'j', 'q': 'k', 'х': 'h'}
        return ''.join(nmap.get(ch, ch) for ch in text)

    # Ünlü ve ünsüz iskelet benzerliği (karakter kümesi kesişimi)
    s_clean = re.sub(r'[^a-zçğıöşüа-яŕŗŋñəäāīūōśčžǰq]', '', _normalize(s))
    t_clean = re.sub(r'[^a-zçğıöşüа-яŕŗŋñəäāīūōśčžǰq]', '', _normalize(t))
    s_clean = re.sub(r'[^a-zçğıöşü]', '', s_clean)
    t_clean = re.sub(r'[^a-zçğıöşü]', '', t_clean)

    # Temel karakter kesişim oranı
    common_chars = set(s_clean).intersection(set(t_clean))
    char_similarity = len(common_chars) / max(len(set(s_clean)), len(set(t_clean)), 1)

    violations = []
    if not matched_rules and char_similarity < 0.3:
        violations.append(f"'{s}' ile '{t}' arasında tanımlı hiç bir fonetik evrim kuralı veya iskelet benzerliği bulunamadı (Broken Phonetic Chain).")

    is_valid = len(violations) == 0
    if matched_rules and is_valid:
        score = 0.95
    elif is_valid and char_similarity >= 0.60:
        score = 0.85  # İyi iskelet benzerliği, kural henüz tanımlanmamış
    elif is_valid and char_similarity >= 0.30:
        score = 0.75  # Kabul edilebilir iskelet benzerliği
    else:
        score = 0.20  # Fonetik halka kırık

    return {
        "is_valid": is_valid,
        "score": score,
        "violations": violations,
        "matched_rules": matched_rules if matched_rules else ["İskelet Benzerliği"]
    }
