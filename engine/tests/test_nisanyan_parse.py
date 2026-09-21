"""Nişanyan ayrıştırıcısının KAYITLI yanıtlar üzerinde DOĞRU çıktı vermesi.

Neden ayrı bir dosya
--------------------
Kayıtlı fixture'lar yalnız *yanıtı* donduruyordu, *doğru ayrıştırmayı* değil.
Mevcut test (``test_coverage_gaps.test_nisanyan_parses_payload``) tek şey
iddia ediyordu: ``assertIn("root", res)``. Bu iddia, ayrıştırıcı ``deniz``
kelimesine **Latince aequor** kökeni uydururken de geçiyordu — Nişanyan'ın
metninde Latince yalnızca *anlam benzetmesi* olarak ("Anlam bağı için karş.")
geçtiği hâlde. Gerçek köken (Eski Türkçe ``teŋiz``) ise hiç yakalanmıyordu,
çünkü eski karakter sınıfı ``[a-zçğıöşüA-ZÇĞİÖŞÜ]`` transkripsiyon
harflerini (``ŋ ġ ī ā ȫ ŕ``) içermiyordu.

Buradaki testler beklenen ÇIKTIYI sabitler; ayrıştırıcı bozulursa fixture
hâlâ oynatılabilir olsa bile test düşer.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path
from unittest import mock

from engine.fetchers import tdk_nisanyan
from engine.fetchers.tdk_nisanyan import NisanyanFetcher

FIXTURES = Path(__file__).parent / "fixtures" / "http"


def _fixture(name: str) -> str:
    path = FIXTURES / name
    if not path.exists():  # pragma: no cover - fixture eksikse açık hata ver
        raise FileNotFoundError(
            f"HTTP fixture yok: {path}. `python scripts/record_fixtures.py --live` çalıştırın."
        )
    return path.read_text(encoding="utf-8")


def _fetch_with_fixture(word: str, fixture_name: str) -> dict:
    html = _fixture(fixture_name)
    with mock.patch.object(tdk_nisanyan, "http_get", return_value=html):
        return NisanyanFetcher().fetch(word)


class TestNisanyanParse(unittest.TestCase):
    def test_deniz_eski_turkce_kokeni_bulunur(self):
        """`deniz` -> Eski Türkçe teŋiz. Transkripsiyon harfi `ŋ` kaçmamalı."""
        res = _fetch_with_fixture("deniz", "nisanyan_deniz.html")
        self.assertIn("teŋiz", res["root"]["proto_turkic"])
        self.assertEqual(res["root"]["meaning"], "büyük göl, deniz")
        otk = [e for e in res["turkic_languages"] if e["lang_code"] == "otk"]
        self.assertTrue(otk, "Eski Türkçe tanığı kaydedilmeli")
        self.assertEqual(otk[0]["word"], "teŋiz")

    def test_deniz_benzetmeyi_koken_sanmaz(self):
        """Nişanyan'ın `karş. Latince aequor` benzetmesi KÖKEN değildir."""
        res = _fetch_with_fixture("deniz", "nisanyan_deniz.html")
        blob = f"{res['root']['proto_turkic']} {res['root']['reconstruction_notes']}"
        self.assertNotIn("Latince", blob)
        self.assertNotIn("aequor", blob)

    def test_deniz_miras_olarak_hukum_verir(self):
        """`evrilmiştir` fiili miras hükmüdür; `alıntı` denmemeli."""
        res = _fetch_with_fixture("deniz", "nisanyan_deniz.html")
        self.assertIn("miras", res["root"]["reconstruction_notes"].lower())
        self.assertNotIn("alıntı kaynağı", res["root"]["reconstruction_notes"].lower())

    def test_goz_eski_turkce_kokeni_bulunur(self):
        """`göz` -> Eski Türkçe kȫz. Uzunluk işareti `ȫ` kaçmamalı."""
        res = _fetch_with_fixture("göz", "nisanyan_göz.html")
        self.assertIn("kȫz", res["root"]["proto_turkic"])
        self.assertEqual(res["root"]["meaning"], "görme organı")

    def test_goz_cuvasca_karsilastirmasini_koken_sanmaz(self):
        """`Karş. Çuvaşça kuś` bir benzetmedir, köken değil."""
        res = _fetch_with_fixture("göz", "nisanyan_göz.html")
        self.assertNotIn("Çuvaşça", res["root"]["proto_turkic"])

    def test_bos_govde_sessizce_bos_doner(self):
        with mock.patch.object(tdk_nisanyan, "http_get", return_value=None):
            res = NisanyanFetcher().fetch("deniz")
        self.assertEqual(res["root"]["proto_turkic"], "")
        self.assertEqual(res["turkic_languages"], [])


class TestNisanyanClaimRegex(unittest.TestCase):
    """Ayrıştırıcının kalıbı, sahadan toplanmış GERÇEK cümlelerle sınanır."""

    #: (cümle, beklenen kaynak dil, beklenen biçim, beklenen hüküm)
    CASES = (
        ("Eski Türkçe teŋiz “büyük göl, deniz” sözcüğünden  evrilmiştir.",
         "Eski Türkçe", "teŋiz", "miras"),
        ("Eski Türkçe kȫz “görme organı” sözcüğünden  evrilmiştir.",
         "Eski Türkçe", "kȫz", "miras"),
        ("Eski Türkçe tapuġ “hizmet, kullanım” sözcüğünden  evrilmiştir.",
         "Eski Türkçe", "tapuġ", "miras"),
        ("Eski Türkçe aynı anlama gelen kuş sözcüğünden  evrilmiştir.",
         "Eski Türkçe", "kuş", "miras"),
        ("Farsça muhr مهر  “damga” sözcüğünden  alıntıdır.",
         "Farsça", "muhr", "alıntı"),
        ("Arapça √ˁẓm kökünden gelen ˁaẓama(t) عَظَمَة  “ululuk, yücelik” sözcüğünden  alıntıdır.",
         "Arapça", "ˁaẓama(t)", "alıntı"),
        ("Arapça √ftl kökünden gelen faˁīl vezninde sıfat olan fatīl veya fatīla(t) فتيل  "
         "“burma, kandil fitili” sözcüğünden  alıntıdır.",
         "Arapça", "fatīla(t)", "alıntı"),
        ("Orta Türkçe çıkın “bohça, paket” sözcüğünden  evrilmiştir.",
         "Orta Türkçe", "çıkın", "miras"),
        ("Eski Türkçe töşe- “yatak veya sedir yaymak” fiilinden  evrilmiştir.",
         "Eski Türkçe", "töşe-", "miras"),
        # Türetme: kaynak biçim ile fiil arasına EK TARİFİ girer.
        ("Eski Türkçe bulġa- “karıştırmak, bulandırmak” fiilinden  "
         "Türkiye Türkçesi +Iş- ekiyle türetilmiştir.",
         "Eski Türkçe", "bulġa-", "türetme"),
        ("Türkiye Türkçesi çığrık “feryat” sözcüğünden  evrilmiştir.",
         "Türkiye Türkçesi", "çığrık", "miras"),
        ("Eski Türkçe (yalnız Oğuzca) tas “kötü, kaba” sözcüğünden  "
         "Türkiye Türkçesi +lA- ekiyle türetilmiş olabilir; ancak bu kesin değildir.",
         "Eski Türkçe", "tas", "türetme"),
        ("Fransızca cobalte “metalik bir element” sözcüğünden  alıntıdır.",
         "Fransızca", "cobalte", "alıntı"),
        ("Eski Türkçe taşu- “nakletmek, götürmek” fiilinden  evrilmiştir.",
         "Eski Türkçe", "taşu-", "miras"),
    )

    def test_gercek_cumleler(self):
        for sentence, lang, form, verdict in self.CASES:
            with self.subTest(sentence=sentence[:40]):
                claim = tdk_nisanyan.parse_etymology_claim(sentence)
                self.assertIsNotNone(claim, "köken cümlesi tanınmalı")
                self.assertEqual(claim["language"], lang)
                self.assertEqual(claim["form"], form)
                self.assertEqual(claim["verdict"], verdict)

    def test_benzetme_cumlesi_koken_sayilmaz(self):
        """`karş.` ile başlayan benzetmeler ayrıştırmaya girmemeli."""
        text = (
            "Eski Türkçe teŋiz “büyük göl, deniz” sözcüğünden  evrilmiştir."
            "Anlam bağı için karş. Latince aequor “deniz” sözcüğünden  alıntıdır."
        )
        claim = tdk_nisanyan.parse_etymology_claim(text)
        self.assertIsNotNone(claim)
        self.assertEqual(claim["language"], "Eski Türkçe")
        self.assertNotEqual(claim["language"], "Latince")

    def test_koken_cumlesi_yoksa_none(self):
        self.assertIsNone(tdk_nisanyan.parse_etymology_claim("Bu bir köken cümlesi değildir."))
        self.assertIsNone(tdk_nisanyan.parse_etymology_claim(""))


class TestFixtureIntegrity(unittest.TestCase):
    """Fixture'ların kendisi bozulmamış olmalı."""

    def test_nisanyan_fixtureleri_metin_tasiyor(self):
        for name in ("nisanyan_deniz.html", "nisanyan_göz.html"):
            with self.subTest(fixture=name):
                tokens = re.findall(r'text:"([^"]+)"', _fixture(name))
                self.assertGreater(len(tokens), 20, f"{name} metin taşımıyor")


if __name__ == "__main__":
    unittest.main()
