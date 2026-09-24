"""
Diyakronik Semantik Vektör Analizi (Diachronic Semantic Shift Engine)

Tarihsel anlam ile modern anlam arasındaki semantik mesafeyi ölçer ve
etimolojik olarak imkânsız anlam sıçramalarını işaretler.

Yeniden üretilebilirlik notu
----------------------------
Bu modül daha önce Python'un yerleşik ``hash()`` fonksiyonunu kullanıyordu.
CPython'da string hash'i süreç başına rastgele tohumlanır (``PYTHONHASHSEED``);
bu yüzden aynı kelime her çalıştırmada FARKLI bir vektör, farklı bir mesafe ve
farklı bir A-HVP rozeti üretiyordu. Artık ``hashlib.blake2b`` kullanılır ve
çıktı deterministiktir.

Sentence-Transformers opsiyoneldir (``pip install -e ".[semantic]"``). Kurulu
değilse modül sessizce zayıf bir karakter n-gram temsiline düşmez; semantik
aşamanın kanıt üretemediğini açıkça bildirir (``evidence_available: False``)
ve A-HVP bu aşamanın ağırlığını toplamdan düşer.
"""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections import Counter
from typing import Any

from engine.logging_setup import get_logger

logger = get_logger(__name__)

_ST_MODEL = None
_ST_TRIED = False
#: Yükleme tek sefer ve tek iş parçacığında yapılır. Kilit yokken ikinci bir
#: çağıran `_ST_TRIED` bayrağını görüp model yüklenirken `None` alabiliyordu.
_ST_LOCK = threading.Lock()
#: ⚠️ Eskiden burada `paraphrase-multilingual-MiniLM-L6-v2` yazıyordu ve
#: BÖYLE BİR MODEL YOK: HuggingFace 401/RepositoryNotFound döndürüyor, yükleme
#: sessizce başarısız oluyor ve semantik aşama paket KURULU OLSA BİLE
#: "sentence-transformers kurulu değil" diyordu. Çok dilli paraphrase
#: modelinin gerçek sürümü L12'dir (L6 yalnız İngilizce `all-MiniLM-L6-v2`
#: olarak vardır ve Türkçe için uygun değildir).
_ST_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def get_sentence_transformer():
    """Modeli tembel (lazy) yükler. Import anında ağ/disk erişimi yapılmaz."""
    global _ST_TRIED
    if _ST_TRIED:
        return _ST_MODEL
    with _ST_LOCK:
        if not _ST_TRIED:
            try:
                _load_sentence_transformer()
            finally:
                # Yükleme BİTTİKTEN sonra: kilitsiz hızlı yol bayrağı görünce
                # modelin hazır olduğunu varsayar.
                _ST_TRIED = True
        return _ST_MODEL


def prewarm_sentence_transformer() -> None:
    """Modeli arka planda yüklemeye başlar (sonucu değiştirmez, yalnız süreyi).

    Soğuk süreçte ``sentence_transformers`` içe aktarımı + ağırlıklar
    ~6–15 s sürüyor ve ilk anlam süzgecinde (``witness_filter``) bekleniyordu.
    Arama başında çağrılırsa bu süre ağ ağırlıklı ``fetch`` aşamasıyla örtüşür;
    sonraki ``get_sentence_transformer`` çağrısı kilitte yüklemenin bitmesini
    bekler ve AYNI modeli alır.
    """
    if _ST_TRIED:
        return
    threading.Thread(target=get_sentence_transformer, name="st-prewarm", daemon=True).start()


def _load_sentence_transformer():
    """``_ST_LOCK`` altında çağrılır; sonucu ``_ST_MODEL``e yazar."""
    global _ST_MODEL
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.info(
            "sentence-transformers kurulu değil; semantik aşama kanıt üretmeyecek. "
            "Etkinleştirmek için: pip install -e \".[semantic]\""
        )
        return None
    # Önce YALNIZ yerel önbellek. Model diskte olsa bile kütüphane her
    # açılışta Hub'a ~30 HEAD isteği atıp "güncel mi" diye soruyordu
    # (her aramada ~5 sn ve "unauthenticated requests" uyarısı). Önbellekte
    # yoksa (ilk kurulum) eskisi gibi indirilir.
    try:
        # Yerel diskten ağırlık okuma çubuğu ("Loading weights") her aramada
        # CLI çıktısını kirletiyordu; bilgi taşımıyor.
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
    except Exception:
        pass
    try:
        _ST_MODEL = SentenceTransformer(_ST_MODEL_NAME, local_files_only=True)
        logger.info("Semantik model yerel önbellekten yüklendi: %s", _ST_MODEL_NAME)
        return _ST_MODEL
    except Exception:
        logger.info("Semantik model önbellekte yok; indiriliyor: %s", _ST_MODEL_NAME)
    try:
        _ST_MODEL = SentenceTransformer(_ST_MODEL_NAME)
        logger.info("Semantik model yüklendi: %s", _ST_MODEL_NAME)
    except Exception:
        logger.warning("Semantik model yüklenemedi: %s", _ST_MODEL_NAME, exc_info=True)
        _ST_MODEL = None
    return _ST_MODEL


def has_semantic_model() -> bool:
    return get_sentence_transformer() is not None


class DenseSemanticVectorizer:
    """Metni sabit boyutlu bir yoğun vektöre indirger."""

    def __init__(self, vocab_size: int = 64):
        self.vocab_size = vocab_size

    def extract_ngrams(self, text: str, n_range: tuple[int, int] = (2, 4)) -> list[str]:
        t = re.sub(r"[^\w\s]", "", (text or "").lower())
        ngrams: list[str] = []
        for token in t.split():
            ngrams.append(token)
            for n in range(n_range[0], n_range[1] + 1):
                for i in range(len(token) - n + 1):
                    ngrams.append(token[i : i + n])
        return ngrams

    @staticmethod
    def _stable_bucket(gram: str, buckets: int) -> int:
        """Süreçler arası sabit hash. ``hash()`` tohumlanmış olduğu için kullanılmaz."""
        digest = hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % buckets

    def vectorise(self, text: str) -> tuple[list[float], bool]:
        """
        (vektör, transformer_kullanıldı) döndürür.

        İkinci değer ``False`` ise vektör yalnızca ortografik (karakter n-gram)
        bir temsildir; semantik kanıt sayılmaz.
        """
        # Boş metnin ANLAMI yoktur; modele sormak anlamsız bir embedding
        # üretir ve `used_model=True` diyerek bunu semantik kanıt gibi
        # gösterir. Sıfır vektör, "kanıt yok" demenin dürüst yoludur.
        if not (text or "").strip():
            return [0.0] * self.vocab_size, False

        model = get_sentence_transformer()
        if model is not None:
            try:
                # ⚠️ Burada vektör TAM boy üzerinden normalize edilip sonra
                # `emb[:64]` ile KIRPILIYORDU. Sonuç birim vektör değildi:
                # MiniLM 384 boyut üretir, 64 boyutluk dilimin normu
                # √(64/384) ≈ 0.41 olur. `cosine_distance` ise "iki birim
                # vektör" varsayıp `1 - dot` hesapladığı için ÖZDEŞ iki
                # anlamda bile dot ≈ 0.41² ≈ 0.167, mesafe ≈ 0.83 çıkıyordu.
                #
                # Ölçüldü: `göz` 0.8544, `bardak` 0.8661 — ikisi de theta
                # 0.85'in hemen üstünde. Yani model kurulu olsa bile 3. aşama
                # HER kelimeyi reddediyordu; theta'yı değiştirmek çözmezdi,
                # ulaşılabilir azami benzerlik zaten ~0.17 idi.
                #
                # Kırpma, ortografik yedek yolun `vocab_size=64` sabitinden
                # sızmış. Transformer yolunda boyut indirgemenin anlamı yok:
                # iki taraf da aynı yoldan geçtiği için boylar eşittir.
                emb = model.encode(text or "", convert_to_numpy=True, show_progress_bar=False)
                vec = [float(x) for x in emb]
                norm = math.sqrt(sum(v * v for v in vec)) or 1.0
                return [round(v / norm, 6) for v in vec], True
            except Exception:
                logger.warning("Semantik kodlama başarısız, ortografik temsile düşülüyor", exc_info=True)

        ngrams = self.extract_ngrams(text)
        if not ngrams:
            return [0.0] * self.vocab_size, False

        counts = Counter(ngrams)
        vec = [0.0] * self.vocab_size
        for gram, freq in counts.items():
            vec[self._stable_bucket(gram, self.vocab_size)] += 1.0 + math.log(freq)

        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [round(x / norm, 4) for x in vec], False


#: Anlam alanına karışan BİÇİM etiketleri. Bunlar anlam değil, veri
#: birleştirme artığıdır (`donor_etymology_database.py:87-88` bu dizeyi
#: `"Kaynak anlamı: X | Geçiş yörüngesi: Y"` diye kuruyor). Kodlanırsa
#: mesafeyi şişirirler: ölçüldü, `kitap` 0.7064 / `kalem` 0.7254 —
#: ikisi de DOĞRU etimoloji, iki gloss da "kitap"/"kalem" demek.
_GLOSS_LABEL = re.compile(
    r"(kaynak\s+anlam[ıi]|geçiş\s+yörüngesi|orijinal\s+imla|"
    r"kendi\s+içi\s+etimoloji)\s*:\s*",
    re.IGNORECASE,
)


#: **Güzergâh bölümü ANLAM DEĞİLDİR, atılır.** `donor_etymology_database`
#: alıntıların tarihî anlamını şöyle kuruyor::
#:
#:     "Kaynak anlamı: Yazı kamışı, yontulmuş kamış |
#:      Geçiş yörüngesi: Eski Grekçe (kálamos) -> Arapça (qalam) -> …"
#:
#: İlk bölüm gerçek gloss'tur; ikincisi bir YOLdur (dil adları ve oklar).
#: Yalnız etiketi silip metni bırakmak, güzergâhı anlammış gibi kodlar ve
#: mesafeyi şişirir — ölçüldü: `kalem` 0.7589, `kitap` 0.6485; ikisi de
#: DOĞRU etimoloji. Bu yüzden etiketten itibaren SONUNA KADAR kesilir.
_TRAJECTORY_CUT = re.compile(
    r"\s*\|?\s*(geçiş\s+yörüngesi|kendi\s+içi\s+etimoloji|orijinal\s+imla)\s*:.*$",
    re.IGNORECASE | re.DOTALL,
)


def _clean_gloss(text: str) -> str:
    """Anlam metnini kodlamadan önce biçim artıklarından arındırır."""
    t = _TRAJECTORY_CUT.sub("", text or "")
    t = _GLOSS_LABEL.sub(" ", t)
    t = t.replace("|", " ")
    return " ".join(t.split()).strip()


class DiachronicSemanticEngine:
    """Tarihsel ve modern anlam arasındaki semantik mesafeyi değerlendirir."""

    # Bu eşiğin üzerindeki mesafe, anlamların birbirinden kopuk olduğunu gösterir.
    #
    # ⚠️ Eskiden 0.85 idi ve HİÇ ÖLÇÜLMEMİŞTİ: model adı yanlış olduğu için
    # (bkz. _ST_MODEL_NAME notu) bu aşama hiç gerçek bir mesafe üretmemişti,
    # eşik tahminle konmuştu.
    #
    # Kalibrasyon — CLDF `savelyevturkic/parameters.csv`, 254 kavram.
    # OLUMLU çift = aynı kavramın iki yazımı ("(finger)nail (n.)" ~
    # "FINGERNAIL"); OLUMSUZ çift = farklı kavramların çaprazı (tohum
    # 20260921). Eşyazım bulaşması yok.
    #
    #   theta   olumlu geçer   ilgisiz YANLIŞ geçer   dengeli doğruluk
    #   0.45       %83,5             %2,8                  %90,4
    #   0.50       %86,6             %5,9                  %90,4
    #   0.60       %94,1            %19,7                  %87,2
    #   0.85       %100             %87,4                  %56,3   <- eski
    #
    # 0.85 pratikte LASTİK DAMGADIR: ilgisiz çiftlerin %87'sini de geçirir.
    #
    # ⚠️ BUNA RAĞMEN EŞİK 0.85'TE BIRAKILDI. Sıkılaştırma denendi (0.60) ve
    # GERİ ALINDI, çünkü asıl kusur eşikte değil GİRDİDE:
    #
    #   12 kelimede ölçüldü — aşamaya giden iki metin
    #     9/12  DEJENERE: iki taraf birebir AYNI dize (göz, bardak, su,
    #           deniz, ayak, baş, pencere, yastık, öküz) -> mesafe 0.0,
    #           bedava ✅. Çoğu tarihî tanığın `meaning` alanı modern TDK
    #           tanımının kopyası olduğu için aşama kendini kendisiyle
    #           karşılaştırıyor (search_engine.py:404'ün düzelttiğini
    #           sandığı hatanın aynısı).
    #     3/12  gerçek çift — ama üçünde de biçim gürültüsü var
    #           ("Kaynak anlamı:", "|"): kitap 0.7064, kalem 0.7254,
    #           televizyon 0.5991. Üçü de DOĞRU etimoloji.
    #
    # theta=0.60 bu üç doğru vakanın ikisini (kitap, kalem) sırf gürültü
    # yüzünden reddediyordu. Kalibrasyonu bozuk girdiye uygulamak sahte
    # kesinlik üretir; önce girdiler temizlenmeli:
    #   (a) iki taraf aynıysa `evidence_available: False` dönmeli, bedava
    #       ✅ verilmemeli;
    #   (b) "Kaynak anlamı:" / "|" gibi biçim eki kodlamadan önce ayıklanmalı.
    # O İŞ BİTTİ; eşik artık veriye dayanıyor.
    #
    # Girdi temizliği tamamlandı (hepsi teste bağlı): çöp gloss elemesi,
    # atıf öneki kırpma, Latin olmayan parantez/tırnak ayıklama, güzergâh
    # bölümünün atılması, tanığın en yakın biçime göre seçilmesi.
    # GERÇEK HAT üzerinde 20 kelime ölçüldü, 13'ünde mesafe var ve hepsi
    # doğru etimoloji:
    #
    #   öküz 0.1801  gece 0.2638  göz 0.2843  bilge 0.2891  el 0.3233
    #   deniz 0.3241 kitap 0.3267 yaz 0.4037  diz 0.4445   kalem 0.4808
    #   kamu 0.5166  baş 0.6501   bardak 0.7863
    #
    # 13 vakanın 12'si <= 0.6501. Tek istisna `bardak`: tarihî gloss
    # "testicik" seçiliyor ("su içilen kap" yerine) — gerçek anlam kayması
    # değil, biçim sıralamasının bilinen kusuru.
    #
    # 0.70 seçildi: `baş`ın (0.6501) üstünde gerçek pay bırakır, yalnız
    # bilinen kusuru eler, ve CLDF vekil tablosunda dengeli doğruluğu
    # %56,3'ten %77,4'e çıkarır. Optimum 0.45-0.50'ye İNİLMEDİ, çünkü o
    # tablodaki olumlu çift "aynı kavramın iki yazımı"dır ve yüzyıllık
    # gerçek kaymadan kolaydır; maliyet de asimetriktir (yanlış RET doğru
    # bir etimolojiyi eler, yanlış GEÇİŞ yalnız kanıt eklemez).
    #
    # ⚠️ Kapsam notu: 20 kelimenin 5'inde (pencere, masa, çete, elektrik,
    # makas) 3. aşamaya hiç tanık ulaşmıyor, 2'sinde (su, ayak) iki taraf
    # özdeş olduğu için dejenere kapısı devrede. Aşama 13/20 = %65
    # kelimede ölçülebiliyor.
    THETA_THRESHOLD = 0.70

    def __init__(self, vocab_size: int = 64):
        self.vectorizer = DenseSemanticVectorizer(vocab_size=vocab_size)

    def cosine_distance(self, v1: list[float], v2: list[float]) -> float:
        """İki birim vektör arasındaki kosinüs mesafesi (0.0 aynı, 1.0 dik)."""
        dot = sum(a * b for a, b in zip(v1, v2, strict=False))
        return round(1.0 - max(0.0, min(1.0, dot)), 4)

    def evaluate_diachronic_trajectory(
        self,
        origin_meaning: str,
        modern_meaning: str,
        timeline: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Tarihsel anlam ile modern anlam arasındaki semantik mesafeyi ölçer.

        ``evidence_available`` alanı, bu değerlendirmenin A-HVP skoruna
        katkıda bulunup bulunamayacağını belirler. Veri eksikse veya yalnızca
        ortografik temsil kullanılabildiyse ``False`` döner ve aşama
        ağırlığı toplam skordan düşülür — eskiden bu durumda otomatik 0.85
        veriliyordu.

        :param timeline: Tarihsel katman etiketleri. Şu an yalnızca çıktıda
            raporlanır; çok noktalı yörünge hesabı için ayrılmıştır.
        """
        s_m = _clean_gloss(origin_meaning)
        m_m = _clean_gloss(modern_meaning)
        layers = list(timeline or [])

        # ⚠️ DEJENERE GİRDİ: iki taraf aynıysa mesafe tanımı gereği 0 çıkar ve
        # aşama BEDAVA ✅ verir. Ölçüldü (12 kelime): 9'unda iki metin birebir
        # aynıydı (göz, bardak, su, deniz, ayak, baş, pencere, yastık, öküz),
        # çünkü çoğu tarihî tanığın `meaning` alanı modern TDK tanımının
        # kopyasıdır. Bu, ölçüm değil kendini doğrulamadır; `search_engine`
        # tarafında bir kez düzeltildiği sanılan hatanın aynısı.
        if s_m and m_m and s_m.casefold() == m_m.casefold():
            return {
                "origin_meaning": s_m,
                "modern_meaning": m_m,
                "total_shift_distance": None,
                "theta_threshold": self.THETA_THRESHOLD,
                "is_plausible": None,
                "evidence_available": False,
                "trajectory_status": "Kanıt Yok",
                "reason": (
                    "Tarihsel ve modern anlam AYNI metin; tarihî tanığın anlamı "
                    "modern tanımın kopyası. Anlam kayması ölçülemedi."
                ),
                "transformer_active": False,
                "timeline_layers": layers,
            }

        if not s_m or not m_m:
            return {
                "origin_meaning": s_m,
                "modern_meaning": m_m,
                "total_shift_distance": None,
                "theta_threshold": self.THETA_THRESHOLD,
                "is_plausible": None,
                "evidence_available": False,
                "trajectory_status": "Kanıt Yok",
                "reason": "Tarihsel veya modern anlam verisi yok; semantik aşama değerlendirilemedi.",
                "transformer_active": False,
                "timeline_layers": layers,
            }

        v_start, used_model_a = self.vectorizer.vectorise(s_m)
        v_end, used_model_b = self.vectorizer.vectorise(m_m)
        transformer_active = used_model_a and used_model_b

        distance = self.cosine_distance(v_start, v_end)

        if not transformer_active:
            # Karakter n-gram temsili anlamı değil imlayı ölçer; kanıt sayılmaz.
            return {
                "origin_meaning": s_m,
                "modern_meaning": m_m,
                "total_shift_distance": distance,
                "theta_threshold": self.THETA_THRESHOLD,
                "is_plausible": None,
                "evidence_available": False,
                "trajectory_status": "Kanıt Yok (semantik model kurulu değil)",
                "reason": (
                    "sentence-transformers kurulu olmadığı için yalnızca ortografik benzerlik "
                    "hesaplanabildi; bu semantik kanıt sayılmaz. Etkinleştirmek için: "
                    'pip install -e ".[semantic]"'
                ),
                "transformer_active": False,
                "orthographic_distance": distance,
                "timeline_layers": layers,
            }

        is_plausible = distance <= self.THETA_THRESHOLD
        reason = "Tarihsel ve modern anlam semantik uzayda birbirine yakın; anlam kayması makul."
        if not is_plausible:
            reason = (
                f"SEMANTİK KOPUKLUK: Anlamlar arası mesafe ({distance}) "
                f"theta sınırını ({self.THETA_THRESHOLD}) aştı."
            )

        return {
            "origin_meaning": s_m,
            "modern_meaning": m_m,
            "semantic_vector_origin": v_start[:8],
            "semantic_vector_modern": v_end[:8],
            "total_shift_distance": distance,
            "theta_threshold": self.THETA_THRESHOLD,
            "is_plausible": is_plausible,
            "evidence_available": True,
            "trajectory_status": "Makul Anlam Kayması" if is_plausible else "Şüpheli Etimoloji (Semantik Sıçrama)",
            "reason": reason,
            "transformer_active": True,
            "timeline_layers": layers,
        }
