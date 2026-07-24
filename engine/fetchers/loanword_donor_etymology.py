import re
import json
import urllib.request
import urllib.parse
from typing import Dict, Any, Optional

from engine.fetchers.base import BaseFetcher

# Alıntı Kelimeler (Loanwords) Kaynak Dildeki Orijinal İmla, Anlam ve Kendi İçi Etimoloji Veri Bankası
DONOR_ETYMOLOGY_DATABASE = {
    "herkil": {
        "donor_lang": "Ermenice (Anadolu Ağızları Alıntısı)",
        "original_script": "յարգել / յարգիլ (harkil / hargel)",
        "donor_meaning": "Samandan veya çalı çırpıdan yapılan çit, engel, ekin ve tahıl koruma ambarı",
        "internal_etymology": "Ermenice յարգ (harg 'çit, samanlık, bölmeli alan') kökünden türetim. Doğu ve Orta Karadeniz (Sinop/Ayancık) ağızlarına Ermenice diyalekt temasıyla girmiştir.",
        "trajectory": "Eski Ermenice (harg 'çit/samanlık') -> Ermenice (harkil/hargel) -> Karadeniz & Sinop Ağızları (herkil/herkel)"
    },
    "herkel": {
        "donor_lang": "Ermenice (Anadolu Ağızları Alıntısı)",
        "original_script": "յարգել (hargel)",
        "donor_meaning": "Ekin ve tahıl ambarı, ahşap saklama bölmesi",
        "internal_etymology": "Ermenice յարգ (harg 'çit, samanlık') kökü. Anadolu ağızlarında ünlü daralması ve fonetik kaymayla herkel/herkil biçimini almıştır.",
        "trajectory": "Ermenice (hargel) -> Karadeniz & Sinop Ağızları (herkel)"
    },
    "harkil": {
        "donor_lang": "Ermenice (Anadolu Ağızları Alıntısı)",
        "original_script": "յարգիլ (hargil)",
        "donor_meaning": "Çit, samanlık, ambar bölmesi",
        "internal_etymology": "Ermenice յարգ (harg 'samanlık, çit') türevi.",
        "trajectory": "Ermenice (hargil) -> Anadolu Ağızları (harkil)"
    },
    "efendi": {
        "donor_lang": "Bizans Grekçesi / Rumca",
        "original_script": "αὐθέντης (authéntēs)",
        "donor_meaning": "Kendi işini gören, kendi başına karar veren, hakim, buyurucu",
        "internal_etymology": "Bizans Grekçesi authéntēs < Eski Grekçe αὐτο- (auto- 'kendi') + ἔντης (éntēs 'yapan/eyleyen'). Osmanlıcaya 14. yüzyılda geçmiştir.",
        "trajectory": "Eski Grekçe -> Bizans Grekçesi (authéntēs) -> Osmanlıca (افندی / efendi) -> Çağdaş Türkçe"
    },
    "rüzgar": {
        "donor_lang": "Farsça",
        "original_script": "روزگار (rūzgār)",
        "donor_meaning": "Zaman, felek, devir, esinti, yel",
        "internal_etymology": "Farsça rūzgār < Farsça rūz (روز 'gün, zaman') + -gār (گار yapım ve aidiyet eki). Asıl anlamı 'zaman, devir' iken Türkçede mecazen 'esinti, yel' anlamı kazanmıştır.",
        "trajectory": "Eski Farsça (raučah) -> Orta Farsça (rōčgār) -> Klasik Farsça (rūzgār) -> Türkçe"
    },
    "kitap": {
        "donor_lang": "Arapça",
        "original_script": "كتاب (kitāb)",
        "donor_meaning": "Yazılı şey, defter, mektup, risale, kutsal metin",
        "internal_etymology": "Arapça k-t-b (ك-ت-ب 'yazmak') üçlü sami kökünden fi'āl (فِعَال) kalıbında masdar/isim. Aynı kökten: katip, mektup, kütüphane, katibe.",
        "trajectory": "Proto-Sami (*katab-) -> Arapça (kitāb) -> Karahanlı Türkçe (11. yy) -> Çağdaş Türkçe"
    },
    "kalem": {
        "donor_lang": "Arapça (Grekçe Alıntısı)",
        "original_script": "قلم (qalam) / Grekçe κάλαμος (kálamos)",
        "donor_meaning": "Yazı kamışı, yontulmuş kamış",
        "internal_etymology": "Arapça qalam < Eski Grekçe kálamos (κάλαμος 'kamış, saz'). Arapçaya Grekçeden geçmiş, Türkçeye 11. yüzyılda Karahanlı döneminde girmiştir.",
        "trajectory": "Eski Grekçe (kálamos) -> Arapça (qalam) -> Karahanlıca (11. yy) -> Çağdaş Türkçe"
    },
    "dünya": {
        "donor_lang": "Arapça",
        "original_script": "دنيا (dunyā)",
        "donor_meaning": "Aşağıda olan, yakın olan yer, yeryüzü, alemi fani",
        "internal_etymology": "Arapça d-n-v (د-ن-و 'yakın olmak, alçakta olmak') kökünden fu'lā (فُعْلَى) vezninde ism-i tafdil. Ahirete nispetle 'yakın/aşağı alem'.",
        "trajectory": "Arapça (dunyā) -> Karahanlıca (11. yy) -> Çağdaş Türkçe"
    },
    "sümen": {
        "donor_lang": "Fransızca (Macarca Aracılığıyla)",
        "original_script": "sous-main / Macarca sujtás",
        "donor_meaning": "Masa üstü yazı altlığı, el altı dosyası",
        "internal_etymology": "Fransızca sous-main < sous ('altında') + main ('el'). Fransızca el altı matbuası.",
        "trajectory": "Fransızca (sous-main) -> Macarca -> Osmanlıca -> Türkçe"
    },
    "televizyon": {
        "donor_lang": "Fransızca (Grekçe + Latince Hibrit)",
        "original_script": "télévision",
        "donor_meaning": "Uzaktan görüntü aktarım cihazı",
        "internal_etymology": "Fransızca télévision < Grekçe tēle (تῆλε 'uzak') + Latince visio ('görme, görüntü'). 20. yüzyıl teknoloji terimi.",
        "trajectory": "Grekçe + Latince -> Fransızca (télévision) -> Türkçe"
    }
}

class LoanwordDonorEtymologyFetcher(BaseFetcher):
    @property
    def source_name(self) -> str:
        return "Alıntı Kelimeler Kaynak Dil Orijinal İmla ve Kendi İçi Etimoloji Veri Bankası"

    def fetch(self, word: str) -> Dict[str, Any]:
        word_clean = word.strip().lower()
        result = {
            "root": {"proto_turkic": "", "meaning": "", "reconstruction_notes": ""},
            "turkic_languages": []
        }

        if word_clean in DONOR_ETYMOLOGY_DATABASE:
            data = DONOR_ETYMOLOGY_DATABASE[word_clean]
            donor_info = f"[{data['donor_lang']}] Orijinal İmla: {data['original_script']} | Kaynak Anlam: {data['donor_meaning']} | Kendi İçi Etimoloji: {data['internal_etymology']} | Geçiş Yörüngesi: {data['trajectory']}"
            result["root"]["reconstruction_notes"] = donor_info

            result["turkic_languages"].append({
                "lang_code": "donor",
                "lang_name": f"Kaynak Dil Etimolojisi ({data['donor_lang']})",
                "word": data["original_script"],
                "meaning": f"Kaynak Anlamı: {data['donor_meaning']} | Kendi İçi Türeyiş: {data['internal_etymology']}",
                "script": "Arabic" if re.search(r'[\u0600-\u06FF]', data["original_script"]) else ("Greek" if re.search(r'[\u0370-\u03FF]', data["original_script"]) else "Latin")
            })

        return result
