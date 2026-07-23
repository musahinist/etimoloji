"""
Akıllı İstek ve Engellenmeme Yönetimi (Smart Network Manager)
Dinamik User-Agent rotasyonu, HTTP 429/503 durumlarında bekleme ve kesintisiz istek helper'ı.
"""
import random
import time
import urllib.request
from typing import Optional

USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "TurkicEtymologyEngine/2.0 Academic Research Crawler"
]

def fetch_url_safe(url: str, timeout: int = 4, max_retries: int = 2) -> Optional[str]:
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    req = urllib.request.Request(url, headers=headers)

    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", errors="ignore")
        except Exception as e:
            if attempt < max_retries:
                time.sleep(0.3 * (2 ** attempt)) # Exponential backoff
            else:
                return None
    return None
