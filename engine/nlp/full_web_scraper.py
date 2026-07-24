"""
Canlı Tam Sayfa Web Kazıyıcı ve Metin Çıkarıcı (Live Deep Web Content Scraper)
Web arama sonuçlarında (DergiPark, TÜBİTAK ULAKBİM, İSAM, Wiktionary, Nişanyan vb.) bulunan URL'lerin
içine derinlemesine girerek akademik makale, paragraf, sözlük ve tam metin içeriklerini kesintisiz okur.
"""
import re
import urllib.request
import urllib.parse
from typing import Dict, Any, List

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def fetch_full_web_page_content(url: str, max_chars: int = 2500) -> str:
    """Verilen URL'nin tam HTML içeriğini indirir, akademik paragrafları süzüp zengin metin olarak döndürür."""
    if not url or not url.startswith("http"):
        return ""

    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=4) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            
            # Script ve CSS etiketlerini temizle
            clean = re.sub(r'<script[^>]*>.*?</script>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
            clean = re.sub(r'<style[^>]*>.*?</style>', ' ', clean, flags=re.DOTALL | re.IGNORECASE)
            clean = re.sub(r'<!--.*?-->', ' ', clean, flags=re.DOTALL)
            
            # Paragraf, makale metni ve blok içeriklerini ayıkla
            paragraphs = re.findall(r'<(?:p|article|blockquote|div[^>]*class="[^"]*content[^"]*")[^>]*>(.*?)</(?:p|article|blockquote|div)>', clean, flags=re.DOTALL | re.IGNORECASE)
            if paragraphs:
                extracted_texts = []
                for p in paragraphs[:10]:
                    p_clean = re.sub(r'<[^>]+>', ' ', p).strip()
                    if len(p_clean) >= 30 and not any(kw in p_clean.lower() for kw in ["cookie", "gizlilik", "javascript", "copyright"]):
                        extracted_texts.append(p_clean)
                text_content = " ".join(extracted_texts)
            else:
                text_content = re.sub(r'<[^>]+>', ' ', clean)

            clean_text = re.sub(r'\s+', ' ', text_content).strip()
            return clean_text[:max_chars]
    except Exception:
        return ""

def scrape_full_web_pages_for_results(search_results: List[Dict[str, str]], max_pages: int = 5) -> List[Dict[str, str]]:
    """Arama sonuçlarındaki ilk N akademik/sözlük sayfasının içine girerek zengin içerikleri çekersiniz."""
    enriched_results = []
    
    for item in search_results[:max_pages]:
        url = item.get("url", "")
        title = item.get("title", "")
        snippet = item.get("snippet", "")
        
        full_text = fetch_full_web_page_content(url)
        enriched_results.append({
            "title": title,
            "url": url,
            "snippet": snippet,
            "content_summary": full_text if full_text else snippet
        })
        
    return enriched_results
