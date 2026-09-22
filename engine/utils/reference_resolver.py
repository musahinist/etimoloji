"""
Türki Diller Çapraz Referans Çözücü (Cross-Reference Resolver)
Sözlük tanımlarında yer alan '[-> herkil]', 'bkz. herkil', 'yön. herkil' gibi yönlendirmeleri tespit eder
ve hedef kelimenin etimolojik anlamını zincirleme olarak sorgular.
"""
import re

REFERENCE_PATTERNS = [
    r'\[\s*->\s*([a-zA-ZçğıöşuüÇĞİÖŞÜ]+)\s*-\s*\d+\s*\]',
    r'\[\s*->\s*([a-zA-ZçğıöşuüÇĞİÖŞÜ]+)\s*\]',
    r'bkz\.\s*([a-zA-ZçğıöşuüÇĞİÖŞÜ]+)',
    r'bknz\.\s*([a-zA-ZçğıöşuüÇĞİÖŞÜ]+)',
    # TDK Tarama/Derleme kısaltması: "bk. derlik"
    r'\bbk\.\s*([a-zA-ZçğıöşuüÇĞİÖŞÜ]+)',
    r'->\s*([a-zA-ZçğıöşuüÇĞİÖŞÜ]+)'
]

def extract_cross_references(definition: str) -> list[str]:
    if not definition:
        return []

    found_refs = []
    for pattern in REFERENCE_PATTERNS:
        matches = re.findall(pattern, definition, re.IGNORECASE)
        for m in matches:
            ref_word = m.strip().lower()
            if ref_word and ref_word not in found_refs and len(ref_word) >= 2:
                found_refs.append(ref_word)

    return found_refs


#: Tanımın TAMAMI bir göndermeden ibaret mi: "bk. derlik", "bkz. herkil",
#: "-> herkil". Böyle bir tanım anlam taşımaz, yalnız başka maddeyi işaret eder.
_BARE_REFERENCE = re.compile(
    r'^\s*(?:\(?\s*(?:bk|bkz|bknz|bak)\.|->)\s*[\w\s,;()-]{1,60}\.?\s*$',
    re.IGNORECASE,
)


def is_cross_reference(definition: str) -> bool:
    return bool(definition) and bool(_BARE_REFERENCE.match(definition))
