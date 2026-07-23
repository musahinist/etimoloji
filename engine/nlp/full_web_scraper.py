"""
Canlı Tam Sayfa Web Kazıyıcı ve Metin Çıkarıcı (Live Full Web Content Scraper)
Web arama sonuçlarında bulunan URL'lerin içine doğrudan girerek sayfadaki tüm makale,
paragraf ve sözlük içeriklerini tam metin olarak okur.
"""
import re
import urllib.request
import urllib.parse
from typing import Dict, Any, List

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def fetch_full_web_page_content(url: str, max_chars: int = 1500) -> str:
    """Verilen URL'nin tam HTML içeriğini indirir, HTML etiketlerini temizler ve metni döndürür."""
    if not url or not url.startswith("http"):
        return ""

    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=4) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            
            # Script ve style etiketlerini temizle
            clean = re.sub(r'<script[^>]*>.*?</script>', ' ', html, flags=re.DOTALL | re.IGNORECASE)
            clean = re.sub(r'<style[^>]*>.*?</style>', ' ', clean, flags=re.DOTALL | re.IGNORECASE)
            
            # Paragraf ve ana içerik alanlarını yakala
            paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', clean, flags=re.DOTALL | re.IGNORECASE)
            if paragraphs:
                text_content = " ".join([re.sub(r'<[^>]+>', ' ', p) for p in paragraphs[:5]])
            else:
                text_content = re.sub(r'<[^>]+>', ' ', clean)

            # Fazla boşlukları temizle
            clean_text = re.sub(r'\s+', ' ', text_content).strip()
            return clean_text[:max_chars]
    except Exception:
        return ""

def scrape_full_web_pages_for_results(search_results: List[Dict[str, str]], max_pages: int = 2) -> List[Dict[str, str]]:
    """Arama sonuçlarındaki ilk N sayfanın içine girerek tam metin içeriklerini çeker."""
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
            "full_page_content": full_text if full_text else snippet
        })
        
    return enriched_results
