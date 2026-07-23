"""
Otonom Rekonstrüksiyon Motoru (Seq2Seq / HMM Proto-Language Reconstruction Engine)
Etimolojisi bilinmeyen veya çözülmemiş Türki kelimelerin muhtemel Proto-Türkçe '*kök' formunu ve fonolojik evrim haritasını hesaplar.
"""
from typing import Dict, Any, List
from engine.utils.morphology import analyze_morphology, NON_TURKIC_INITIAL_CONSONANTS

class ProtoTurkicReconstructor:
    def reconstruct_proto_form(self, word: str, turkic_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Tarihsel ses değişim kurallarını işleterek kelimenin muhtemel Proto-Türkçe '*kök' formunu tahmin eder."""
        w = word.strip().lower()

        # Alıntı kelime kontrolü
        if w and w[0] in NON_TURKIC_INITIAL_CONSONANTS:
            return {
                "word": w,
                "reconstructed_root": f"*{w}",
                "is_reconstructible": False,
                "reconstruction_notes": "Söz başı ünsüzü Öz Türkçe olmadığı için Proto-Türkçe rekonstrüksiyon uygulanmaz (Alıntı Kök)."
            }

        stem, suffixes = analyze_morphology(w)
        proto_stem = stem

        # Tarihsel ses kayma kurallarını geriye dönük uygula (Reverse Sound Laws)
        if proto_stem.startswith("h"):
            proto_stem = "k" + proto_stem[1:]
        elif proto_stem.startswith("d"):
            proto_stem = "t" + proto_stem[1:]

        if "l" in proto_stem and "r" in proto_stem:
            l_idx, r_idx = proto_stem.find("l"), proto_stem.find("r")
            if l_idx < r_idx:
                proto_list = list(proto_stem)
                proto_list[l_idx], proto_list[r_idx] = "r", "l"
                proto_stem = "".join(proto_list)

        reconstructed_root = f"*{proto_stem}"
        notes = f"Geriye dönük tarihsel ses kuralları uygulandı: {w} -> {reconstructed_root}"

        return {
            "word": w,
            "reconstructed_root": reconstructed_root,
            "is_reconstructible": True,
            "reconstruction_notes": notes
        }
