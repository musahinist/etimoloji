"""
Qwen2.5:14b Otonom Ajan Araştırma ve Web Keşif Araçları (Agent Research Tools)
Qwen2.5:14b modelinin otonom çağırabileceği web arama, web kazıma ve akademik arama araçları.
"""
import re
import json
import urllib.request
import urllib.parse
from typing import Dict, Any, List

def tool_web_search(query: str) -> List[Dict[str, str]]:
    """Canlı web araması yapıp yeni akademik portallar ve etimoloji siteleri keşfeder."""
    results = []
    clean_q = urllib.parse.quote(f"{query.strip()} etimolojisi")
    url = f"https://html.duckduckgo.com/html/?q={clean_q}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            links = re.findall(r'<a[^>]*class=\"result__a\"[^>]*href=\"([^\"]+)\">(.*?)</a>', html, re.DOTALL)
            snippets = re.findall(r'<a[^>]*class=\"result__snippet\"[^>]*>(.*?)</a>', html, re.DOTALL)
            for i, (link, title_text) in enumerate(links[:3]):
                clean_title = re.sub(r'<[^>]+>', '', title_text).strip()
                snippet_text = re.sub(r'<[^>]+>', '', snippets[i]).strip() if i < len(snippets) else ""
                
                # Extract real URL from DDG redirect
                real_url = link
                if "uddg=" in link:
                    m = re.search(r'uddg=([^&]+)', link)
                    if m:
                        real_url = urllib.parse.unquote(m.group(1))

                results.append({
                    "url": real_url,
                    "title": clean_title,
                    "snippet": snippet_text
                })
    except Exception:
        pass
    return results

def tool_web_scrape_url(url: str) -> str:
    """Keşfedilen bir web sayfasını ziyaret edip etimolojik içerik ve sözlük maddelerini tam metin çeker."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            text = re.sub(r'<script.*?>.*?</script>', '', html, flags=re.DOTALL)
            text = re.sub(r'<style.*?>.*?</style>', '', text, flags=re.DOTALL)
            clean_text = re.sub(r'<[^>]+>', ' ', text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            return clean_text[:1200]
    except Exception as e:
        return f"Sayfa okunamadı: {e}"

def tool_search_academic(query: str) -> str:
    """DergiPark ve akademik tez veritabanında hedeflenmiş makale araması yapar."""
    url = f"https://dergipark.org.tr/tr/search?q={urllib.parse.quote(query)}+etimoloji"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=4) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            titles = re.findall(r'<a[^>]*class=\"[^\"]*card-title[^\"]*\"[^>]*>(.*?)</a>', html, re.DOTALL)
            clean_titles = [re.sub(r'<[^>]+>', '', t).strip() for t in titles[:3] if "doğrulayınız" not in t]
            return "Akademik Makaleler: " + "; ".join(clean_titles) if clean_titles else "Ek akademik makale bulunamadı."
    except Exception as e:
        return f"Akademik arama hatası: {e}"

def tool_search_archive_books(query: str) -> str:
    """Archive.org taranmış tarihi kitaplarda ve el yazmalarında arama yapar."""
    url = f"https://archive.org/advancedsearch.php?q={urllib.parse.quote(query)}+turkic+etymology&fl[]=title,creator,year&rows=2&page=1&output=json"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=4) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            docs = data.get("response", {}).get("docs", [])
            books = [f"{d.get('title')} ({d.get('creator')}, {d.get('year')})" for d in docs if d.get('title')]
            return "Archive.org Kitapları: " + "; ".join(books) if books else "Archive.org kaydı bulunamadı."
    except Exception as e:
        return f"Archive.org hatası: {e}"
