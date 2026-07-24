"""
Türki Diller Etimoloji Araştırma Motoru Web REST API Sunucusu (Python HTTP Server)
Web UI (Next.js / Cytoscape.js) ile haberleşen yerel REST API sunucusudur.
Kullanıcı aramalarını canlı SearchEngine üzerinden çalıştırır ve JSON döndürür.
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import urllib.parse
import sys
import os

from engine.search_engine import SearchEngine
from engine.db.database import DatabaseManager

db = DatabaseManager()
engine = SearchEngine(db_manager=db)

class EtymologyAPIHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        params = urllib.parse.parse_qs(parsed.query)

        if path == "/api/search":
            word_list = params.get("word", [])
            if not word_list or not word_list[0].strip():
                self.send_response(400)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Kelime parametresi (word) eksik."}, ensure_ascii=False).encode('utf-8'))
                return

            word = word_list[0].strip().lower()
            use_ai = params.get("ai", ["false"])[0].lower() == "true"

            try:
                finding = engine.search(word, save_to_db=True, use_qwen_agent=use_ai)
                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(finding, ensure_ascii=False).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}, ensure_ascii=False).encode('utf-8'))

        elif path == "/api/list":
            try:
                findings = db.list_findings()
                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps(findings, ensure_ascii=False).encode('utf-8'))
            except Exception as e:
                self.send_response(500)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}, ensure_ascii=False).encode('utf-8'))

        else:
            self.send_response(404)
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(b"404 Not Found")

def run_server(port: int = 8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, EtymologyAPIHandler)
    print(f"🚀 Türki Diller Etimoloji REST API Sunucusu http://localhost:{port} adresinde aktif.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Sunucu durduruldu.")
        httpd.server_close()

if __name__ == "__main__":
    port = 8000
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        port = int(sys.argv[1])
    run_server(port)
