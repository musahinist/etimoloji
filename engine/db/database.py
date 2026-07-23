import os
import json
import sqlite3
from typing import Dict, Any, List, Optional

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "etymology.db")
SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "schema.sql")

class DatabaseManager:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def init_db(self) -> None:
        """Veritabanını ve tabloları oluşturur."""
        if os.path.exists(SCHEMA_PATH):
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                schema_sql = f.read()
            with self.get_connection() as conn:
                conn.executescript(schema_sql)
                conn.commit()

    def save_finding(self, finding: Dict[str, Any]) -> int:
        """
        Etimoloji bulgusunu standart formatta veritabanına kaydeder.
        Eğer kelime zaten varsa günceller.
        """
        query_word = finding.get("query_word", "").lower().strip()
        if not query_word:
            raise ValueError("query_word boş olamaz")

        root = finding.get("root", {})
        proto_turkic = root.get("proto_turkic", "")
        root_meaning = root.get("meaning", "")
        sources = finding.get("sources", [])
        turkic_languages = finding.get("turkic_languages", [])
        full_json = json.dumps(finding, ensure_ascii=False, indent=2)
        sources_json = json.dumps(sources, ensure_ascii=False)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Var olan kaydı kontrol et
            cursor.execute("SELECT id FROM findings WHERE query_word = ?", (query_word,))
            row = cursor.fetchone()
            
            if row:
                finding_id = row["id"]
                cursor.execute("""
                    UPDATE findings 
                    SET proto_turkic_root = ?, root_meaning = ?, sources_json = ?, full_finding_json = ?, created_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (proto_turkic, root_meaning, sources_json, full_json, finding_id))
                cursor.execute("DELETE FROM turkic_entries WHERE finding_id = ?", (finding_id,))
            else:
                cursor.execute("""
                    INSERT INTO findings (query_word, proto_turkic_root, root_meaning, sources_json, full_finding_json)
                    VALUES (?, ?, ?, ?, ?)
                """, (query_word, proto_turkic, root_meaning, sources_json, full_json))
                finding_id = cursor.lastrowid

            # Türki dillerdeki kelime karşılıklarını ekle
            for entry in turkic_languages:
                cursor.execute("""
                    INSERT INTO turkic_entries (finding_id, lang_code, lang_name, word, meaning, script)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    finding_id,
                    entry.get("lang_code", ""),
                    entry.get("lang_name", ""),
                    entry.get("word", ""),
                    entry.get("meaning", ""),
                    entry.get("script", "Latin")
                ))

            conn.commit()
            return finding_id

    def get_finding(self, query_word: str) -> Optional[Dict[str, Any]]:
        """Veritabanında kayıtlı etimoloji bulgusunu getirir."""
        query_word = query_word.lower().strip()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT full_finding_json FROM findings WHERE query_word = ?", (query_word,))
            row = cursor.fetchone()
            if row:
                return json.loads(row["full_finding_json"])
            return None

    def list_findings(self) -> List[Dict[str, Any]]:
        """Tüm kaydedilmiş etimoloji bulgularını listeler."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, query_word, proto_turkic_root, root_meaning, created_at FROM findings ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
