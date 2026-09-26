"""
Türkçe verici dil ÖNSELİ ve biçim ipuçları — şans düzeyindeki verici etiketinin yerine (9n).

Tanı (``data/cache/work/donor9n/PREREG.md``): Türkçe verici etiketinin
(:func:`engine.nlp.donor_proximity.attribute_donor`) önemli bir kısmı doğru etimona değil ŞANS
eşleşmesine dayanıyor (sultan ~ صلى, kudret ~ قطر "Katar"): doğru etimon anlam havuzunda
yok; "en yakın biçimin dili" o zaman havuzu hangi dilin doldurduğunu gösterir. Bu modül o
maddelerde dili mesafe yerine iki bağımsız kaynakla tahmin eder:

* **doğal dağılım önseli** — TDK Güncel Türkçe Sözlük (12. baskı) ``lisan`` alanının başlık
  sayımı (:data:`NATURAL_DONOR_COUNTS`; ``data/cache/work/donor9m/natural_dist.py``);
* **biçim ipuçları** — dil başına karakter 3-gram dil modeli (:class:`CueModel`), Türkçe
  Vikisözlük'ün ``alıntı`` maddelerinden (``donor_lang``) eğitilir; bütün verici altınlarının
  kelimeleri eğitimden ÇIKARILIR (bkz. :func:`train`). Model ``data/models/donor_cue_tr.json``.

Sonsal ∝ önsel · P(biçim | dil)^τ (:data:`CUE_TEMPERATURE`). Yalnız ETİKET; alıntı gücü değişmez.

    python -m engine.nlp.donor_prior train     # modeli yeniden eğitir
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

from engine.config import PROJECT_ROOT
from engine.logging_setup import get_logger
from engine.utils.orthography import to_comparison_form

logger = get_logger(__name__)

MODEL_PATH = PROJECT_ROOT / "data" / "models" / "donor_cue_tr.json"

#: TDK GTS ``lisan`` doğal dağılımı (``tum`` sayımı; ``tr_donor_eval.NATURAL_COUNTS`` ile aynı
#: kaynak), Türkçe verici havuzunun (``TURKISH_DONORS``) dilleri.
NATURAL_DONOR_COUNTS = {"ar": 6638.2, "fr": 5676.2, "fa": 1432.0, "it": 613.7, "el": 487.5, "hy": 29.3}

#: Vikisözlük ``donor_lang`` -> sınıf kodu (Osmanlıca ``ota`` belirsiz: çıkarılır).
_TRAIN_CODE = {"ar": "ar", "arb": "ar", "fa": "fa", "fa-cls": "fa", "fr": "fr", "frm": "fr",
               "it": "it", "vec": "it", "lij": "it", "el": "el", "grc": "el", "gkm": "el",
               "hy": "hy", "xcl": "hy"}

#: Biçim olabilirliğinin üssü (τ). ``1`` tam Bayes; küçük değer önseli güçlendirir.
#: Ayarda seçildi (bkz. PREREG.md).
CUE_TEMPERATURE = 1.0

#: Dil modeli ara değerleme ağırlıkları (3-gram, 2-gram, 1-gram) ve toplama düzeltmesi.
_LAMBDAS = (0.6, 0.3, 0.1)
_ADD = 0.1


class CueModel:
    """Dil başına karakter 3-gram modeli (ara değerlemeli, toplamalı düzeltme)."""

    def __init__(self, counts: dict[str, dict[str, dict[str, int]]]):
        self.counts = counts
        self.alphabet = sorted({ch for lang in counts.values() for ctx in lang.values() for ch in ctx})
        self._totals = {lang: {ctx: sum(v.values()) for ctx, v in table.items()} for lang, table in counts.items()}

    def _prob(self, lang: str, ctx: str, ch: str) -> float:
        table, totals = self.counts[lang], self._totals[lang]
        v = max(len(self.alphabet), 1)
        out = 0.0
        for lam, c in zip(_LAMBDAS, (ctx, ctx[1:], ""), strict=True):
            n = table.get(c, {}).get(ch, 0)
            out += lam * (n + _ADD) / (totals.get(c, 0) + _ADD * v)
        return out

    def loglik(self, comparison: str, lang: str) -> float:
        word = "##" + comparison + "$"
        return sum(math.log(self._prob(lang, word[i - 2:i], word[i])) for i in range(2, len(word)))

    @property
    def languages(self) -> list[str]:
        return sorted(self.counts)


@lru_cache(maxsize=1)
def load_model(path: str = str(MODEL_PATH)) -> CueModel | None:
    p = Path(path)
    if not p.exists():
        logger.info("Verici biçim ipucu modeli yok: python -m engine.nlp.donor_prior train")
        return None
    return CueModel(json.loads(p.read_text(encoding="utf-8"))["counts"])


def posterior(comparison: str, languages: list[str] | None = None,
              temperature: float | None = None) -> dict[str, float]:
    """Sonsal P(dil | biçim) ∝ doğal önsel · P(biçim | dil)^τ. Model yoksa yalnız önsel."""
    langs = [lang for lang in (languages or list(NATURAL_DONOR_COUNTS)) if lang in NATURAL_DONOR_COUNTS]
    if not langs:
        return {}
    tau = CUE_TEMPERATURE if temperature is None else temperature
    model = load_model()
    logp = {}
    for lang in langs:
        lp = math.log(NATURAL_DONOR_COUNTS[lang])
        if model is not None and tau and comparison and lang in model.counts:
            lp += tau * model.loglik(comparison, lang)
        logp[lang] = lp
    top = max(logp.values())
    z = sum(math.exp(v - top) for v in logp.values())
    return {lang: math.exp(v - top) / z for lang, v in sorted(logp.items(), key=lambda kv: -kv[1])}


def best(comparison: str, languages: list[str] | None = None,
         temperature: float | None = None) -> tuple[str, float]:
    post = posterior(comparison, languages, temperature)
    if not post:
        return "", 0.0
    lang = max(post, key=lambda k: (post[k], k))
    return lang, post[lang]


# --- eğitim ---------------------------------------------------------------------------------

#: Eğitimden çıkarılan altınlar (kelime + karşılaştırma biçimi): verici etiketinin ölçüldüğü
#: her altın. Test bölümleri dahil ÇIKARILIR (etiketlerine bakılmaz; yalnız kelime listesi).
EXCLUDE_GOLDS = (
    "data/gold/turkish_loanwords.json",
    "data/cache/work/donor9e/gold.json", "data/cache/work/donor9f/gold.json",
    "data/cache/work/donor9g/gold.json", "data/cache/work/donor9j/gold_tdk.json",
    "data/cache/work/donor9j/gold_tettl.json", "data/cache/work/donor9l/gold.json",
    "data/cache/work/donor9m/gold.json", "data/cache/work/donor9n/gold.json",
    "data/cache/work/donor9o/gold.json",
)


def excluded_words() -> set[str]:
    out: set[str] = set()
    for rel in EXCLUDE_GOLDS:
        p = PROJECT_ROOT / rel
        if not p.exists():
            raise SystemExit(f"eğitim dışlama altını yok: {rel}")
        data = json.loads(p.read_text(encoding="utf-8"))
        rows = list(data.get("items") or []) + [r for r in data.get("disagreements") or [] if isinstance(r, dict)]
        for r in rows:
            w = str(r.get("word") or "")
            if w:
                out |= {w.casefold(), to_comparison_form(w)}
            if r.get("comparison"):
                out.add(r["comparison"])
    return out


def train(index_path: Path | None = None, out: Path = MODEL_PATH) -> dict[str, Any]:
    """Türkçe Vikisözlük ``alıntı`` maddelerinden dil başına 3-gram sayımları."""
    import sqlite3

    index_path = index_path or PROJECT_ROOT / "data" / "lexicons" / "index.db"
    skip = excluded_words()
    con = sqlite3.connect(index_path)
    seen: dict[str, set[str]] = defaultdict(set)
    for word, donor in con.execute(
            "SELECT word, donor_lang FROM entries WHERE lang_code='tr' AND origin='alıntı' AND donor_lang != ''"):
        code = _TRAIN_CODE.get(donor or "")
        comp = to_comparison_form(word or "")
        if not code or len(comp) < 2 or " " in (word or "") or (word or "")[:1].isupper():
            continue
        if (word or "").casefold() in skip or comp in skip:
            continue
        seen[comp].add(code)
    counts: dict[str, dict[str, dict[str, int]]] = {}
    n: dict[str, int] = defaultdict(int)
    for comp, codes in sorted(seen.items()):
        if len(codes) != 1:
            continue  # çelişkili verici kaydı
        code = next(iter(codes))
        n[code] += 1
        table = counts.setdefault(code, {})
        word = "##" + comp + "$"
        for i in range(2, len(word)):
            for ctx in (word[i - 2:i], word[i - 1:i], ""):
                row = table.setdefault(ctx, {})
                row[word[i]] = row.get(word[i], 0) + 1
    payload = {"_schema": "donor-cue-lm/v1", "trained_on": "Türkçe Vikisözlük alıntı donor_lang (index.db)",
               "excluded_golds": list(EXCLUDE_GOLDS), "n_excluded_keys": len(skip), "n_words": dict(n),
               "order": 3, "counts": counts}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    load_model.cache_clear()
    return {"n_words": dict(n), "path": str(out)}


if __name__ == "__main__":
    import sys

    if sys.argv[1:2] == ["train"]:
        print(json.dumps(train(), ensure_ascii=False))
