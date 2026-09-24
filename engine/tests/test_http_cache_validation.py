"""Kalıcı HTTP önbelleği yalnız geçerli gövdeyi saklar: bakım ve Cloudflare
doğrulama sayfaları da HTTP 200 ile gelebilir; önbelleğe girerse 90 gün
boyunca kelimenin sayfası diye geri verilirdi."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from engine.utils import network
from engine.utils.network import _cacheable

NISANYAN = "https://www.nisanyansozluk.com/kelime/su"
NISANYAN_PAGE = (
    "<html><head><title>su - Nişanyan Sözlük</title>"
    '<script src="/cdn-cgi/challenge-platform/scripts/jsd/main.js"></script></head>'
    "<body>text:\"su\"</body></html>"
)
CLOUDFLARE = (
    "<!DOCTYPE html><html><head><title>Just a moment...</title></head>"
    "<body><script>window._cf_chl_opt={cvId: '3'}</script></body></html>"
)


class TestCacheable(unittest.TestCase):
    def test_real_pages_are_cached(self):
        # Cloudflare her Nişanyan sayfasına challenge-platform betiği ekler;
        # bu tek başına doğrulama sayfası işareti değildir.
        self.assertTrue(_cacheable(NISANYAN, NISANYAN_PAGE))
        self.assertTrue(_cacheable("https://www.etimolojiturkce.com/kelime/su",
                                   "<html><h1>Su</h1>…</html>"))
        self.assertTrue(_cacheable("https://sozluk.gov.tr/gts?ara=su", '[{"madde": "su"}]'))
        self.assertTrue(_cacheable("https://sozluk.gov.tr/gts?ara=zzz", '{"error": "Sonuç bulunamadı"}'))

    def test_cloudflare_challenge_rejected(self):
        self.assertFalse(_cacheable(NISANYAN, CLOUDFLARE))
        self.assertFalse(_cacheable(NISANYAN, CLOUDFLARE.replace("Just a moment...", "Attention Required! | Cloudflare")))

    def test_maintenance_page_rejected(self):
        page = "<html><head><title>Bakım Çalışması</title></head><body>Nişanyan</body></html>"
        self.assertFalse(_cacheable(NISANYAN, page))
        self.assertFalse(_cacheable("https://islamansiklopedisi.org.tr/su",
                                    "<html><title>Site under maintenance</title></html>"))

    def test_gts_must_be_json(self):
        self.assertFalse(_cacheable("https://sozluk.gov.tr/gts?ara=su", "<!doctype html><html>…</html>"))

    def test_truncated_or_foreign_body_rejected(self):
        self.assertFalse(_cacheable(NISANYAN, NISANYAN_PAGE[:60]))
        self.assertFalse(_cacheable(NISANYAN, ""))
        self.assertFalse(_cacheable("https://www.etimolojiturkce.com/kelime/su", "<html>hata</html>"))

    def test_word_in_body_text_is_not_maintenance(self):
        # "ses bakımından" gibi düzyazı başlıkta değilse bakım sayılmaz.
        page = "<html><head><title>asker</title></head><h1>Asker</h1>ses bakımından problemlidir</html>"
        self.assertTrue(_cacheable("https://www.etimolojiturkce.com/kelime/asker", page))


class TestPersistentCacheSkipsInvalid(unittest.TestCase):
    def _fetch_twice(self, body):
        with tempfile.TemporaryDirectory() as tmp:
            session = mock.Mock()
            session.get.return_value = mock.Mock(status_code=200, headers={})
            with mock.patch.object(network, "HTTP_CACHE_PATH", Path(tmp) / "http.db"), \
                    mock.patch.object(network, "_persistent_enabled", True), \
                    mock.patch.object(network, "_decode_body", return_value=body), \
                    mock.patch.object(network, "get_session", return_value=session), \
                    mock.patch.object(network, "is_url_allowed", return_value=(True, "ok")):
                network.reset_circuits()
                first = network.fetch(NISANYAN)
                network.fetch(NISANYAN)
            return first, session.get.call_count

    def test_challenge_page_not_stored(self):
        first, calls = self._fetch_twice(CLOUDFLARE)
        self.assertEqual(first, CLOUDFLARE, "gövde çağırana yine döner")
        self.assertEqual(calls, 2, "ikinci istek önbellekten gelmemeli")

    def test_real_page_stored(self):
        self.assertEqual(self._fetch_twice(NISANYAN_PAGE)[1], 1)

    def test_previously_stored_invalid_body_is_not_served(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(network, "HTTP_CACHE_PATH", Path(tmp) / "http.db"):
            network._cache_put(NISANYAN, CLOUDFLARE)
            self.assertIsNone(network._cache_get(NISANYAN))
            network._cache_put(NISANYAN, NISANYAN_PAGE)
            self.assertEqual(network._cache_get(NISANYAN), NISANYAN_PAGE)


if __name__ == "__main__":
    unittest.main()
