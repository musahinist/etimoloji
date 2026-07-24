import os
import unittest
import tempfile
from engine.db.database import DatabaseManager
from engine.search_engine import SearchEngine

class TestSearchEngine(unittest.TestCase):
    def setUp(self):
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp()
        self.db = DatabaseManager(db_path=self.temp_db_path)
        self.engine = SearchEngine(db_manager=self.db)

    def tearDown(self):
        os.close(self.temp_db_fd)
        if os.path.exists(self.temp_db_path):
            os.remove(self.temp_db_path)

    def test_search_and_cache(self):
        # İlk arama (DB'ye kaydeder)
        res1 = self.engine.search("su", save_to_db=True)
        self.assertFalse(res1.get("from_cache", True))
        self.assertEqual(res1["query_word"], "su")
        self.assertTrue(len(res1["turkic_languages"]) > 0)

        # İkinci arama (Cache kullanılmadığı için yine taze canlı arama yapılmalı)
        res2 = self.engine.search("su", save_to_db=True)
        self.assertFalse(res2.get("from_cache", True))
        self.assertEqual(res2["query_word"], "su")

if __name__ == "__main__":
    unittest.main()
